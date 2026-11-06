"""
Option contract selection and premium-level modelling for option BUYERS.

Why this module exists
----------------------
An option buyer's P&L is the option premium, not index points.  The old
risk path set stop/target on the index spot and booked P&L as
``points × lot × lots`` — i.e. delta = 1, no theta, no IV.  A flat spot then
showed ₹0 while the premium decayed, and a "1.5 R:R" on spot was often
below 1 in premium terms after theta and the bid-ask spread.

Everything here works in premium:

* :func:`time_to_expiry_years` — the single T convention (calendar time to
  15:30 IST on expiry day, floored at 15 minutes).
* :func:`bs_price_greeks` / :func:`implied_vol` — Black-Scholes on spot with
  a flat rate (adequate for the short tenors traded here).
* :func:`select_contract` — picks the strike whose *computed* delta is
  closest to the target, among strikes that are actually tradeable
  (non-trivial premium, open interest, acceptable spread).
* :func:`premium_at` — re-prices the chosen contract at another spot level
  and a later time, anchored to the *market* premium (model-relative move),
  so theta over the expected holding period is included.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from typing import Optional
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")
_EXPIRY_TIME = dtime(15, 30)
_MIN_TTE_SECONDS = 15 * 60
_SECONDS_PER_YEAR = 365.0 * 86400.0
RISK_FREE_RATE = 0.065
_MIN_PREMIUM_FLOOR = 0.05  # exchange tick


# ---------------------------------------------------------------------------
# Pricing primitives
# ---------------------------------------------------------------------------


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def time_to_expiry_years(expiry: date, now: datetime) -> float:
    """Calendar time from *now* to 15:30 IST on *expiry*, in years.

    Floored at 15 minutes so greeks stay finite in the last minutes of an
    expiry day.  Naive *now* is taken to be IST.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=_IST)
    expiry_dt = datetime.combine(expiry, _EXPIRY_TIME, tzinfo=_IST)
    seconds = (expiry_dt - now).total_seconds()
    return max(seconds, _MIN_TTE_SECONDS) / _SECONDS_PER_YEAR


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    theta_per_day: float  # ₹ per unit per calendar day (negative for longs)
    vega_per_pct: float   # ₹ per unit per 1 vol point


def bs_price_greeks(
    spot: float,
    strike: float,
    tte_years: float,
    iv: float,
    option_type: str,
    rate: float = RISK_FREE_RATE,
) -> Greeks:
    """Black-Scholes price and greeks.  *iv* is a decimal (0.15 = 15 %)."""
    if spot <= 0 or strike <= 0 or tte_years <= 0 or iv <= 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)
        return Greeks(intrinsic, 0.0, 0.0, 0.0, 0.0)
    sqrt_t = math.sqrt(tte_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * tte_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    disc = math.exp(-rate * tte_years)
    pdf = _npdf(d1)
    if option_type == "CE":
        price = spot * _ncdf(d1) - strike * disc * _ncdf(d2)
        delta = _ncdf(d1)
        theta = -(spot * pdf * iv) / (2 * sqrt_t) - rate * strike * disc * _ncdf(d2)
    else:
        price = strike * disc * _ncdf(-d2) - spot * _ncdf(-d1)
        delta = _ncdf(d1) - 1.0
        theta = -(spot * pdf * iv) / (2 * sqrt_t) + rate * strike * disc * _ncdf(-d2)
    gamma = pdf / (spot * iv * sqrt_t)
    vega = spot * pdf * sqrt_t / 100.0
    return Greeks(max(price, 0.0), delta, gamma, theta / 365.0, vega)


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    tte_years: float,
    option_type: str,
    rate: float = RISK_FREE_RATE,
) -> Optional[float]:
    """Implied volatility by bisection, or ``None`` if *price* is not
    attainable (below intrinsic, or above the 300 % vol price)."""
    if price <= 0 or spot <= 0 or strike <= 0 or tte_years <= 0:
        return None
    lo, hi = 0.005, 3.0
    if price < bs_price_greeks(spot, strike, tte_years, lo, option_type, rate).price:
        return None
    if price > bs_price_greeks(spot, strike, tte_years, hi, option_type, rate).price:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price_greeks(spot, strike, tte_years, mid, option_type, rate).price > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return 0.5 * (lo + hi)


def strike_interval(strikes: list[float]) -> Optional[float]:
    """Listed strike spacing, inferred from the chain (most common gap)."""
    ks = sorted({float(k) for k in strikes if k and k > 0})
    gaps = [round(b - a, 4) for a, b in zip(ks, ks[1:]) if b - a > 0]
    if not gaps:
        return None
    return Counter(gaps).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Contract selection
# ---------------------------------------------------------------------------


@dataclass
class ContractQuote:
    """A specific, tradeable option contract with its entry economics."""

    index_id: str
    expiry: date
    strike: float
    option_type: str        # "CE" / "PE"
    ltp: float
    bid: float
    ask: float
    mid: float              # best estimate of fair premium now
    half_spread: float      # ₹ per unit paid to cross the spread, each side
    spread_known: bool      # False when bid/ask were unavailable (assumed)
    iv: float               # decimal
    iv_source: str          # "chain" / "solved"
    oi: int
    greeks: Greeks
    priced_at: datetime     # time the greeks/mid refer to
    spot: float

    @property
    def entry_premium(self) -> float:
        """Premium actually paid per unit (mid + half spread)."""
        return self.mid + self.half_spread


@dataclass(frozen=True)
class SelectionParams:
    target_abs_delta: float = 0.45
    min_premium: float = 5.0           # ₹ — below this, tick/spread dominate
    max_spread_pct: float = 0.05       # of mid, when bid/ask are known
    assumed_half_spread_pct: float = 0.005  # of mid, when unknown
    min_oi_fraction_of_median: float = 0.10
    max_strikes_from_atm: int = 8


def _leg(strike_row, option_type: str) -> dict:
    p = "ce_" if option_type == "CE" else "pe_"
    g = lambda name: float(getattr(strike_row, p + name, 0) or 0)  # noqa: E731
    return {
        "ltp": g("ltp"), "iv": g("iv"), "oi": int(g("oi")),
        "bid": g("bid_price"), "ask": g("ask_price"),
    }


def select_contract(
    chain,
    signal_type: str,
    spot: float,
    now: datetime,
    params: SelectionParams = SelectionParams(),
) -> tuple[Optional[ContractQuote], str]:
    """Pick the tradeable strike whose delta is closest to the target.

    Returns ``(quote, "")`` or ``(None, reason)``.
    """
    if chain is None or not getattr(chain, "strikes", None):
        return None, "no options chain"
    expiry = getattr(chain, "expiry_date", None)
    if expiry is None:
        return None, "chain has no expiry"
    if spot <= 0:
        return None, "invalid spot"
    option_type = "CE" if signal_type == "BUY_CALL" else "PE"
    tte = time_to_expiry_years(expiry, now)

    rows = sorted(chain.strikes, key=lambda s: s.strike_price)
    atm_idx = min(range(len(rows)), key=lambda i: abs(rows[i].strike_price - spot))
    window = rows[max(0, atm_idx - params.max_strikes_from_atm): atm_idx + params.max_strikes_from_atm + 1]

    ois = sorted(_leg(r, option_type)["oi"] for r in window)
    median_oi = ois[len(ois) // 2] if ois else 0
    min_oi = median_oi * params.min_oi_fraction_of_median

    candidates: list[ContractQuote] = []
    rejected: Counter = Counter()
    for row in window:
        leg = _leg(row, option_type)
        ltp, bid, ask = leg["ltp"], leg["bid"], leg["ask"]
        spread_known = bid > 0 and ask > 0 and ask >= bid
        mid = (bid + ask) / 2 if spread_known else ltp
        if mid < params.min_premium:
            rejected["premium below minimum"] += 1
            continue
        if leg["oi"] <= 0 or leg["oi"] < min_oi:
            rejected["thin open interest"] += 1
            continue
        if spread_known:
            if (ask - bid) / mid > params.max_spread_pct:
                rejected["spread too wide"] += 1
                continue
            half = (ask - bid) / 2
        else:
            half = mid * params.assumed_half_spread_pct
        iv = leg["iv"] / 100.0 if leg["iv"] > 0 else 0.0
        iv_source = "chain"
        if iv <= 0:
            solved = implied_vol(mid, spot, row.strike_price, tte, option_type)
            if solved is None:
                rejected["no usable IV"] += 1
                continue
            iv, iv_source = solved, "solved"
        greeks = bs_price_greeks(spot, row.strike_price, tte, iv, option_type)
        candidates.append(ContractQuote(
            index_id=getattr(chain, "index_id", ""), expiry=expiry,
            strike=float(row.strike_price), option_type=option_type,
            ltp=ltp, bid=bid, ask=ask, mid=mid, half_spread=half,
            spread_known=spread_known, iv=iv, iv_source=iv_source,
            oi=leg["oi"], greeks=greeks, priced_at=now, spot=spot,
        ))

    if not candidates:
        detail = ", ".join(f"{n} {why}" for why, n in rejected.most_common()) or "no strikes near ATM"
        return None, f"no tradeable {option_type} strike ({detail})"
    best = min(candidates, key=lambda q: abs(abs(q.greeks.delta) - params.target_abs_delta))
    return best, ""


def premium_at(quote: ContractQuote, spot: float, at: datetime) -> float:
    """Premium of *quote*'s contract if spot moves to *spot* by time *at*.

    Anchored to the market mid: ``mid + (BS(spot, at) − BS(quote.spot, now))``
    so model/market level differences (skew, carry) cancel and only the
    modelled *change* — delta, gamma and theta — is applied.  IV held flat.
    """
    tte_now = time_to_expiry_years(quote.expiry, quote.priced_at)
    tte_then = time_to_expiry_years(quote.expiry, at)
    base = bs_price_greeks(quote.spot, quote.strike, tte_now, quote.iv, quote.option_type).price
    moved = bs_price_greeks(spot, quote.strike, tte_then, quote.iv, quote.option_type).price
    return max(quote.mid + (moved - base), _MIN_PREMIUM_FLOOR)


def estimate_premium(
    option_type: str, strike: float, expiry: date, iv: float,
    spot: float, at: datetime, anchor_mid: float, anchor_spot: float, anchor_at: datetime,
) -> float:
    """Model mark for an open position when no live option quote exists."""
    q = ContractQuote(
        index_id="", expiry=expiry, strike=strike, option_type=option_type,
        ltp=anchor_mid, bid=0.0, ask=0.0, mid=anchor_mid, half_spread=0.0,
        spread_known=False, iv=iv, iv_source="entry", oi=0,
        greeks=bs_price_greeks(anchor_spot, strike, time_to_expiry_years(expiry, anchor_at), iv, option_type),
        priced_at=anchor_at, spot=anchor_spot,
    )
    return premium_at(q, spot, at)


def minutes_until(now: datetime, hhmm: str) -> float:
    """Minutes from *now* until the given IST wall-clock time today."""
    h, m = (int(x) for x in hhmm.split(":"))
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return (target - now).total_seconds() / 60.0


__all__ = [
    "ContractQuote", "Greeks", "SelectionParams", "bs_price_greeks", "estimate_premium",
    "implied_vol", "minutes_until", "premium_at", "select_contract", "strike_interval",
    "time_to_expiry_years",
]
