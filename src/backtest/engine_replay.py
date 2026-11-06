"""
Replay the LIVE decision engine over stored 5-minute bars.

Why this exists
---------------
The old backtest (``StrategyRunner``) did not test the model that trades:
it used daily bars, booked P&L as index points and charged slippage as a
percentage of spot.  This module drives the real
:meth:`DecisionEngine.run_full_cycle` — technical analysis, regime, the
calibrated signal generator, the option-native risk manager — once per
5-minute bar, exactly as the live 5-minute job does, and books every trade
in option premium with spread and statutory costs.

What is replaced, and why (everything else is the live code path)
-----------------------------------------------------------------
* **Options chain for analysis → None.**  There is no point-in-time chain
  history (snapshots stop in June 2026), and the signal calibration was
  fitted without one.  Hero-zero / OI analytics therefore never fire.
* **Options chain for the risk step → synthetic.**  The risk manager needs a
  chain to pick a strike and price premium levels.  A Black-Scholes chain is
  built around spot at the decision time: flat IV (VIX if known at that
  time, else 1.1 × realised vol of the last 75 bars, floor 10 %), bid/ask
  ±max(0.5 % of premium, ₹0.05), flat non-zero OI (so no fake OI walls).
  Expiries follow the 2025-26 rules (NIFTY weekly Tuesday, BANKNIFTY monthly
  last Tuesday) — exchange holidays are ignored.
* **Anomaly step → None.**  ``AnomalyAggregator`` stamps, expires and reads
  its alerts on the wall clock and from its live store, so it would inject
  today's production alerts into history.  Live it can only veto/haircut.
* **Kill switch → off.**  It is an operations control whose file/DB state
  describes production *today*, not the replayed date.
* **Contract facts → rules.**  The broker contract master lists only
  *current* contracts; the replay uses fixed lot sizes (NIFTY 65,
  BANKNIFTY 30) and the expiry rules above.
* **Alerts / websocket → none.**  Signals are still recorded by the
  ``SignalTracker`` (on the DB copy) because the daily trade cap reads them.
* **Risk-manager clock → replay clock.**  The live ``RiskManager`` keys its
  daily P&L cache, loss streak and cooldown on the wall-clock date and
  stamps ``closed_at`` with the wall clock.  ``ReplayRiskManager`` keeps the
  same rules on the replay clock.

Fill model
----------
* Decision at the close of bar *t* (``as_of`` = bar start + 5 min − 1 s, so
  every as_of-bounded read sees bar *t* but not bar *t+1*).
* Entry fills at bar *t+1*'s OPEN: the selected contract is re-priced to
  that spot/time with :func:`option_contract.premium_at` and half the
  spread is paid.  Stop/target levels stay as the alert stated them.
* Every bar, for each filled position: (1) at the open — gap-through and
  time rules via ``RiskManager.update_position``; (2) intrabar — bar
  high/low against the spot SL/target triggers and the premium stop/target
  (both touched → stop first), exit at the trigger re-priced (model mark)
  minus half the spread; (3) at the close — time stop, 15:15 force exit,
  trailing stop.
* Positions are flat by 15:15; anything still open when a session's data
  ends is closed at the last bar (``SESSION_END``).

The source database is copied to a temporary file first and never opened.
"""

from __future__ import annotations

import bisect
import calendar
import csv
import json
import logging
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional, Union
from zoneinfo import ZoneInfo

from src.data.options_chain import OptionsChainData, OptionStrike
from src.database.db_manager import DatabaseManager
from src.engine.decision_engine import DecisionEngine
from src.engine.option_contract import bs_price_greeks, estimate_premium, time_to_expiry_years
from src.engine.option_costs import round_trip_cost
from src.engine.risk_manager import RiskConfig, RiskManager
from src.engine.signal_calibration import CALIBRATION_VERSION, SignalCalibration
from src.engine.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Assumptions (every one is reported in ReplayResult.meta)
# ---------------------------------------------------------------------------

BAR_MINUTES = 5
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)
OPTIONS_CLOSE = dtime(15, 30)
BARS_PER_SESSION = 75                 # 09:15-15:30 in 5-minute bars
TRADING_DAYS_PER_YEAR = 252
RV_BARS = 75
RV_MULTIPLIER = 1.1
IV_FLOOR = 0.10
IV_DEFAULT = 0.15                     # only if neither VIX nor enough bars exist
SYNTH_HALF_SPREAD_PCT = 0.005         # of premium, each side
TICK = 0.05                           # ₹, exchange tick = minimum half spread
SYNTH_STRIKES_EACH_SIDE = 20
SYNTH_OI = 100_000                    # flat: no fake OI walls
SYNTH_VOLUME = 50_000

DEFAULT_LOT_SIZES: dict[str, int] = {"NIFTY50": 65, "BANKNIFTY": 30}
STRIKE_INTERVALS: dict[str, float] = {
    "NIFTY50": 50.0, "BANKNIFTY": 100.0, "FINNIFTY": 50.0, "MIDCPNIFTY": 25.0,
    "SENSEX": 100.0, "BANKEX": 100.0,
}
_MON, _TUE, _WED, _THU, _FRI = range(5)
# 2025-26 exchange rules.  Holiday shifts are NOT modelled.
EXPIRY_RULES: dict[str, tuple[str, int]] = {
    "NIFTY50": ("weekly", _TUE),
    "BANKNIFTY": ("monthly", _TUE),
    "FINNIFTY": ("monthly", _TUE),
    "MIDCPNIFTY": ("monthly", _TUE),
    "SENSEX": ("weekly", _THU),
    "BANKEX": ("monthly", _THU),
}
_SYMBOL_ALIASES: dict[str, str] = {"NIFTY": "NIFTY50", "NIFTY_50": "NIFTY50", "NIFTYBANK": "BANKNIFTY"}

_EXIT_REASONS = {
    "EXIT_SL": "SL",
    "EXIT_TRAILING_SL": "TRAILING_SL",
    "EXIT_TARGET": "TARGET",
    "EXIT_TIME": "TIME",
    "EXIT_TIME_STOP": "TIME_STOP",
}

DIAGNOSTIC_BANNER = (
    "DIAGNOSTIC MODE — permissive calibration (every bin P(win)=0.60, validated=true). "
    "These results exercise the trade path end-to-end. They are NOT evidence of edge "
    "and must never be used to decide whether to trade."
)


def _canon(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    return _SYMBOL_ALIASES.get(s, s)


# ---------------------------------------------------------------------------
# Expiry calendar
# ---------------------------------------------------------------------------


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    last = date(year, month, calendar.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def expiry_on_or_after(index_id: str, day: date) -> date:
    """First expiry date ≥ *day* under the index's weekday rule."""
    kind, weekday = EXPIRY_RULES.get(_canon(index_id), ("monthly", _TUE))
    if kind == "weekly":
        return day + timedelta(days=(weekday - day.weekday()) % 7)
    exp = _last_weekday_of_month(day.year, day.month, weekday)
    if exp < day:
        y, m = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
        exp = _last_weekday_of_month(y, m, weekday)
    return exp


def nearest_expiry(index_id: str, now: datetime) -> date:
    """Nearest *tradeable* expiry at *now* (today's dies at 15:30)."""
    now = _as_ist(now)
    first = now.date()
    if now.time() >= OPTIONS_CLOSE:
        first += timedelta(days=1)
    return expiry_on_or_after(index_id, first)


def following_expiry(index_id: str, expiry: date) -> date:
    """The expiry after *expiry*."""
    return expiry_on_or_after(index_id, expiry + timedelta(days=1))


class ReplayContractMaster:
    """Rule-based stand-in for :class:`ContractMaster` (lot sizes + expiries).

    The live master lists only contracts trading *today*; a replay of past
    dates needs the calendar that applied then.
    """

    source = "replay-rules"
    is_loaded = True

    def __init__(self, lot_sizes: Optional[dict[str, int]] = None) -> None:
        self._lots = {_canon(k): int(v) for k, v in (lot_sizes or DEFAULT_LOT_SIZES).items()}

    def refresh(self, today: Optional[date] = None, force: bool = False) -> bool:
        return True

    @staticmethod
    def underlying_for(symbol: str) -> str:
        return _canon(symbol)

    def nearest_expiry(self, symbol: str, as_of=None) -> Optional[date]:
        if as_of is None:
            return None
        if isinstance(as_of, datetime):
            return nearest_expiry(symbol, as_of)
        return expiry_on_or_after(symbol, as_of)

    def expiries(self, symbol: str, on_or_after: Optional[date] = None, count: int = 6) -> list[date]:
        if on_or_after is None:
            return []
        out = [expiry_on_or_after(symbol, on_or_after)]
        while len(out) < count:
            out.append(following_expiry(symbol, out[-1]))
        return out

    def lot_size(self, symbol: str, expiry: Optional[date] = None, as_of=None) -> Optional[int]:
        return self._lots.get(_canon(symbol))

    def is_expiry_day(self, symbol: str, day: Optional[date] = None) -> Optional[bool]:
        if day is None:
            return None
        return expiry_on_or_after(symbol, day) == day


# ---------------------------------------------------------------------------
# Bars, volatility, synthetic chain
# ---------------------------------------------------------------------------


def _as_ist(ts: datetime) -> datetime:
    return ts.replace(tzinfo=_IST) if ts.tzinfo is None else ts.astimezone(_IST)


@dataclass(frozen=True)
class Bar:
    """A 5-minute bar labelled by its START time (as stored)."""

    start: datetime
    open: float
    high: float
    low: float
    close: float

    @property
    def day(self) -> date:
        return self.start.date()

    @property
    def end(self) -> datetime:
        """Decision instant: last second of the bar (sees this bar, not the next)."""
        return self.start + timedelta(minutes=BAR_MINUTES) - timedelta(seconds=1)

    @property
    def close_time(self) -> datetime:
        return self.start + timedelta(minutes=BAR_MINUTES)


def load_session_bars(db: DatabaseManager, index_id: str, until: Optional[date] = None) -> list[Bar]:
    """All stored 5m session bars (09:15-15:30 IST) for *index_id*, ascending."""
    rows = db.fetch_all(
        "SELECT timestamp, open, high, low, close FROM price_data "
        "WHERE index_id = ? AND timeframe = '5m' ORDER BY timestamp ASC",
        (index_id,),
    )
    out: list[Bar] = []
    seen: set[datetime] = set()
    for r in rows:
        try:
            ts = _as_ist(datetime.fromisoformat(str(r["timestamp"])))
            o, h, lo, c = (float(r[k]) for k in ("open", "high", "low", "close"))
        except (TypeError, ValueError):
            continue
        if not (SESSION_OPEN <= ts.time() < SESSION_CLOSE) or ts in seen:
            continue
        if until is not None and ts.date() > until:
            break
        if min(o, h, lo, c) <= 0:
            continue
        seen.add(ts)
        out.append(Bar(ts, o, max(h, o, c), min(lo, o, c), c))
    return out


def load_vix(db: DatabaseManager) -> list[tuple[datetime, float]]:
    rows = db.fetch_all("SELECT timestamp, vix_value FROM vix_data ORDER BY timestamp ASC", ())
    out = []
    for r in rows:
        try:
            v = float(r["vix_value"])
            ts = _as_ist(datetime.fromisoformat(str(r["timestamp"])))
        except (TypeError, ValueError):
            continue
        if v > 0:
            out.append((ts, v))
    return out


class VolatilityModel:
    """IV for the synthetic chain, using only information known at ``as_of``.

    1. India VIX: an intraday reading ≤ 1 day old, or the previous session's
       end-of-day row.  Daily rows are stamped 00:00 of their own date but
       hold that day's CLOSE, so a same-day 00:00 row is not used intraday.
    2. Else 1.1 × annualised realised vol of the last 75 5m bars (same-
       session returns only), floored at 10 %.
    """

    def __init__(self, bars: list[Bar], vix: list[tuple[datetime, float]]) -> None:
        self._bars = bars
        self._starts = [b.start for b in bars]
        self._vix = vix
        self._vix_ts = [t for t, _ in vix]
        days = sorted({b.day for b in bars})
        self._prev_session = {d: p for p, d in zip(days, days[1:])}
        self.source_counts: Counter = Counter()

    def vix_at(self, as_of: datetime) -> Optional[float]:
        as_of = _as_ist(as_of)
        i = bisect.bisect_right(self._vix_ts, as_of) - 1
        while i >= 0:
            ts, _ = self._vix[i]
            if ts.date() == as_of.date() and ts.time() == dtime(0, 0):
                i -= 1  # today's EOD value — not known yet
                continue
            break
        if i < 0:
            return None
        ts, value = self._vix[i]
        if as_of - ts <= timedelta(days=1):
            return value
        prev = self._prev_session.get(as_of.date())
        if prev is not None and ts.date() >= prev:
            return value
        return None

    def realised_vol(self, as_of: datetime) -> Optional[float]:
        idx = bisect.bisect_right(self._starts, _as_ist(as_of)) - 1
        if idx < 1:
            return None
        window = self._bars[max(0, idx - RV_BARS + 1): idx + 1]
        rets = [
            math.log(b.close / a.close)
            for a, b in zip(window, window[1:])
            if a.day == b.day
        ]
        if len(rets) < 10:
            return None
        return statistics.stdev(rets) * math.sqrt(BARS_PER_SESSION * TRADING_DAYS_PER_YEAR)

    def iv_at(self, as_of: datetime, count: bool = True) -> tuple[float, str]:
        vix = self.vix_at(as_of)
        if vix is not None:
            iv, src = vix / 100.0, "vix"
        else:
            rv = self.realised_vol(as_of)
            if rv is not None:
                iv, src = max(rv * RV_MULTIPLIER, IV_FLOOR), "realised_vol"
            else:
                iv, src = IV_DEFAULT, "default"
        if count:
            self.source_counts[src] += 1
        return iv, src


def synthetic_half_spread(mid: float, pct: float = SYNTH_HALF_SPREAD_PCT) -> float:
    """₹ per unit paid to cross half the spread: max(pct × premium, 1 tick)."""
    return max(mid * pct, TICK)


def build_synthetic_chain(
    index_id: str,
    spot: float,
    now: datetime,
    iv: float,
    expiry: date,
    available_expiries: tuple[date, ...] = (),
    strike_interval: Optional[float] = None,
    strikes_each_side: int = SYNTH_STRIKES_EACH_SIDE,
    half_spread_pct: float = SYNTH_HALF_SPREAD_PCT,
    oi: int = SYNTH_OI,
) -> OptionsChainData:
    """Black-Scholes chain around *spot*: flat IV, symmetric bid/ask, flat OI."""
    interval = strike_interval or STRIKE_INTERVALS.get(_canon(index_id), 50.0)
    now = _as_ist(now)
    tte = time_to_expiry_years(expiry, now)
    atm = round(spot / interval) * interval
    iv_pct = round(iv * 100.0, 4)
    rows: list[OptionStrike] = []
    for k in range(-strikes_each_side, strikes_each_side + 1):
        strike = atm + k * interval
        if strike <= 0:
            continue
        legs = {}
        for typ in ("CE", "PE"):
            mid = max(bs_price_greeks(spot, strike, tte, iv, typ).price, TICK)
            half = synthetic_half_spread(mid, half_spread_pct)
            legs[typ] = (mid, max(mid - half, 0.0), mid + half)
        (ce, ce_bid, ce_ask), (pe, pe_bid, pe_ask) = legs["CE"], legs["PE"]
        rows.append(OptionStrike(
            strike_price=float(strike),
            ce_oi=oi, ce_oi_change=0, ce_oi_change_pct=0.0, ce_volume=SYNTH_VOLUME,
            ce_ltp=ce, ce_ltp_change=0.0, ce_ltp_change_pct=0.0, ce_iv=iv_pct,
            ce_bid_price=ce_bid, ce_bid_qty=1000, ce_ask_price=ce_ask, ce_ask_qty=1000,
            pe_oi=oi, pe_oi_change=0, pe_oi_change_pct=0.0, pe_volume=SYNTH_VOLUME,
            pe_ltp=pe, pe_ltp_change=0.0, pe_ltp_change_pct=0.0, pe_iv=iv_pct,
            pe_bid_price=pe_bid, pe_bid_qty=1000, pe_ask_price=pe_ask, pe_ask_qty=1000,
        ))
    return OptionsChainData(
        index_id=index_id,
        spot_price=spot,
        timestamp=now,
        expiry_date=expiry,
        strikes=tuple(rows),
        available_expiries=tuple(sorted(set(available_expiries) | {expiry})),
    )


# ---------------------------------------------------------------------------
# Replay versions of the live components
# ---------------------------------------------------------------------------


class ReplaySignalGenerator(SignalGenerator):
    """Live signal generator; expiry-day check from the replay calendar."""

    def __init__(self, db, calibration: SignalCalibration, contract_master: ReplayContractMaster) -> None:
        super().__init__(db, calibration=calibration)
        self._replay_cm = contract_master

    def _is_expiry_day(self, index_id, now, technical=None) -> bool:  # type: ignore[override]
        return bool(self._replay_cm.is_expiry_day(index_id, _as_ist(now).date()))


class ReplayRiskManager(RiskManager):
    """Live risk rules on the replay clock.

    The base class reads the wall-clock date for its daily cache and stamps
    ``closed_at`` with the wall clock; the loss streak and per-index cooldown
    then never match a replayed day.  Here the same rules run on
    :meth:`set_clock` time and an in-memory ledger of replay exits.
    """

    def __init__(
        self,
        db: DatabaseManager,
        config: Optional[RiskConfig] = None,
        contract_master: Optional[ReplayContractMaster] = None,
    ) -> None:
        self._replay_cm = contract_master or ReplayContractMaster()
        self._clock: Optional[datetime] = None
        self._ledger: list[dict] = []
        super().__init__(db, config=config)

    # ── replay clock ──────────────────────────────────────────────────
    def set_clock(self, now: datetime) -> None:
        self._clock = _as_ist(now)

    @property
    def clock(self) -> Optional[datetime]:
        return self._clock

    # ── wall-clock hooks of the base class ───────────────────────────
    def _load_open_positions(self) -> None:
        """A replay starts flat — never load the live book."""

    def _cleanup_on_startup(self) -> None:
        """Nothing to clean: the replay DB copy has no live trades."""

    def _refresh_daily_pnl(self) -> None:
        day = self._clock.date() if self._clock else None
        self._daily_realised_pnl = sum(
            r["pnl"] for r in self._ledger if day is not None and r["closed_at"].date() == day
        )
        self._cache_date = day

    def _maybe_refresh_daily_cache(self) -> None:
        with self._lock:
            day = self._clock.date() if self._clock else None
            if self._cache_date != day:
                self._refresh_daily_pnl()

    def _closed_trades_today(self, now: datetime) -> Optional[list[dict]]:
        day = _as_ist(now).date()
        rows = [r for r in self._ledger if r["closed_at"].date() == day]
        rows.sort(key=lambda r: r["closed_at"])
        return [
            {"index_id": r["index_id"], "outcome": r["outcome"], "closed_at": r["closed_at"].isoformat()}
            for r in rows
        ]

    def _contract_lot_size(self, index_id: str, expiry: Optional[date]) -> Optional[int]:
        return self._replay_cm.lot_size(index_id, expiry) or self._get_lot_size(index_id)

    def _oi_sr_levels(self, chain):
        """The synthetic chain carries no positioning information."""
        return None, None

    def close_position(self, signal_id, exit_price, exit_reason, exit_premium=None):
        with self._lock:
            pos = self._open_positions.get(signal_id)
            pnl = super().close_position(signal_id, exit_price, exit_reason, exit_premium=exit_premium)
            if pnl is None or pos is None:
                return pnl
            closed_at = self._clock or pos.entry_time
            self._ledger.append({
                "index_id": pos.index_id,
                "outcome": "WIN" if pnl > 0 else "LOSS",
                "closed_at": closed_at,
                "pnl": pnl,
            })
            if pos.db_id > 0:  # base stamped the wall clock
                try:
                    self._db.execute(
                        "UPDATE trading_signals SET closed_at = ? WHERE id = ?",
                        (closed_at.isoformat(), pos.db_id),
                    )
                except Exception:
                    logger.debug("replay: could not restamp closed_at", exc_info=True)
        return pnl

    # ── replay-only lifecycle helpers ────────────────────────────────
    def apply_fill(
        self,
        signal_id: str,
        spot: float,
        at: datetime,
        half_spread_fn: Callable[[float], float] = synthetic_half_spread,
    ) -> Optional[dict]:
        """Move a just-tracked position's entry to the actual fill.

        The contract chosen at the decision is re-priced to *spot* at *at*
        (``premium_at``: delta, gamma and theta since the decision; IV flat)
        and half the spread is paid on top.  Stop/target levels are kept as
        the alert stated them.
        """
        at = _as_ist(at)
        with self._lock:
            pos = self._open_positions.get(signal_id)
            if pos is None or not pos.is_option:
                return None
            before = {"spot": pos.entry_price, "mid": pos.premium_mid, "time": pos.entry_time}
            mid = estimate_premium(
                pos.option_type, pos.strike, pos.expiry, pos.entry_iv or IV_DEFAULT,
                spot, at, pos.premium_mid or pos.entry_premium, pos.entry_price, pos.entry_time,
            )
            half = half_spread_fn(mid)
            pos.entry_price = float(spot)
            pos.entry_time = at
            pos.premium_mid = round(mid, 4)
            pos.half_spread = round(half, 4)
            pos.entry_premium = round(mid + half, 4)
            pos.last_premium = pos.premium_mid
            self._persist_position_state(pos)
            return {"decision": before, "fill_mid": pos.premium_mid, "half_spread": pos.half_spread}

    def cancel_position(self, signal_id: str) -> bool:
        """Drop an unfilled position (no P&L)."""
        with self._lock:
            pos = self._open_positions.pop(signal_id, None)
        if pos is None:
            return False
        if pos.db_id > 0:
            try:
                self._db.execute(
                    "UPDATE trading_signals SET outcome = 'CANCELLED', closed_at = ? WHERE id = ?",
                    ((self._clock or pos.entry_time).isoformat(), pos.db_id),
                )
            except Exception:
                logger.debug("replay: could not mark cancelled", exc_info=True)
        return True


class ReplayDecisionEngine(DecisionEngine):
    """The live :class:`DecisionEngine` with replay-safe inputs.

    Overrides only: kill switch, the options chain (None for analysis,
    synthetic for the risk step), the anomaly step, and alerting.  Signal
    generation and risk use the replay subclasses above.
    """

    def __init__(
        self,
        db: DatabaseManager,
        vol_model: VolatilityModel,
        calibration: Optional[SignalCalibration] = None,
        risk_config: Optional[RiskConfig] = None,
        lot_sizes: Optional[dict[str, int]] = None,
        strike_interval: Optional[float] = None,
        half_spread_pct: float = SYNTH_HALF_SPREAD_PCT,
    ) -> None:
        cm = ReplayContractMaster(lot_sizes)
        super().__init__(db, risk_config=risk_config, contract_master=cm)
        self.replay_contract_master = cm
        self.signal_generator = ReplaySignalGenerator(db, calibration or SignalCalibration(), cm)
        self.risk_manager = ReplayRiskManager(db, config=risk_config, contract_master=cm)
        self.vol_model = vol_model
        self._strike_interval = strike_interval
        self._half_spread_pct = half_spread_pct
        self.chains_built = 0

    # Operations control describing production *now* — not the replayed day.
    def _is_kill_switch_active(self) -> bool:
        return False

    # No point-in-time chain history; the calibration was fitted without one.
    def _load_options_chain(self, index_id, as_of=None, expiry=None):
        return None

    def _load_oi_summary(self, index_id, as_of=None):
        return None

    # Wall-clock alert store — would leak today's production alerts.
    def _step_anomaly(self, *args, **kwargs):
        return None

    def synthetic_chain(self, index_id: str, now: datetime, expiry: Optional[date] = None):
        bar = self._get_latest_bar(index_id, "5m", now)
        if not bar or not bar.get("close"):
            return None
        spot = float(bar["close"])
        iv, _src = self.vol_model.iv_at(now)
        exp = expiry or nearest_expiry(index_id, now)
        self.chains_built += 1
        return build_synthetic_chain(
            index_id, spot, now, iv, exp,
            available_expiries=(exp, following_expiry(index_id, exp)),
            strike_interval=self._strike_interval,
            half_spread_pct=self._half_spread_pct,
        )

    def _step_risk(self, raw_signal, options_chain, index_id=None, now=None, as_of=None):
        if (
            getattr(raw_signal, "signal_type", "NO_TRADE") in ("BUY_CALL", "BUY_PUT")
            and index_id and now is not None
        ):
            options_chain = self.synthetic_chain(index_id, now)
        return super()._step_risk(raw_signal, options_chain, index_id=index_id, now=now, as_of=as_of)

    def _load_next_expiry_chain(self, index_id, chain, now, as_of):
        return self.synthetic_chain(index_id, now, expiry=following_expiry(index_id, chain.expiry_date))

    def _step_store_and_alert(self, index_id, final_signal, is_actionable):
        # The tracker row is what the daily trade cap counts — keep it.
        try:
            self.tracker.record_signal(final_signal, is_actionable=is_actionable)
        except Exception:
            logger.exception("replay: tracker.record_signal failed for %s", index_id)
        if is_actionable:
            self.risk_manager.track_open_position(final_signal)
        return None, "NONE"


# ---------------------------------------------------------------------------
# Execution against bars
# ---------------------------------------------------------------------------


class ReplayExecutor:
    """Fills and exits for positions held by a :class:`ReplayRiskManager`."""

    def __init__(
        self,
        rm: ReplayRiskManager,
        half_spread_fn: Callable[[float], float] = synthetic_half_spread,
    ) -> None:
        self.rm = rm
        self.half_spread_fn = half_spread_fn
        self.trades: list[dict] = []
        self._info: dict[str, dict] = {}   # signal_id → decision/fill details
        self._filled: list[str] = []

    @property
    def open_ids(self) -> list[str]:
        return [s for s in self._filled if s in self.rm._open_positions]

    def fill(self, signal_id: str, bar: Bar, decision: Optional[dict] = None) -> bool:
        self.rm.set_clock(bar.start)
        fill = self.rm.apply_fill(signal_id, bar.open, bar.start, self.half_spread_fn)
        if fill is None:
            return False
        info = dict(decision or {})
        info.update({
            "decision_time": fill["decision"]["time"],
            "decision_spot": fill["decision"]["spot"],
            "decision_mid": fill["decision"]["mid"],
        })
        self._info[signal_id] = info
        self._filled.append(signal_id)
        return True

    # Phase 1 — at the bar open: gap-through and time rules.
    def on_open(self, bar: Bar) -> None:
        self.rm.set_clock(bar.start)
        for sid in self.open_ids:
            upd = self.rm.update_position(sid, bar.open, _now=bar.start)
            if upd.action != "HOLD":
                self._exit(sid, bar.open, upd.mark_premium, _EXIT_REASONS.get(upd.action, upd.action),
                           bar.start, "open")

    # Phase 2 — through the bar: high/low against every trigger.
    def on_intrabar(self, bar: Bar) -> None:
        at = bar.end
        self.rm.set_clock(at)
        for sid in self.open_ids:
            pos = self.rm._open_positions[sid]
            is_call = pos.signal_type == "BUY_CALL"
            adverse, favour = (bar.low, bar.high) if is_call else (bar.high, bar.low)

            def mark(spot: float) -> float:
                return estimate_premium(
                    pos.option_type, pos.strike, pos.expiry, pos.entry_iv or IV_DEFAULT,
                    spot, at, pos.premium_mid or pos.entry_premium, pos.entry_price, pos.entry_time,
                )

            spot_sl = adverse <= pos.current_sl if is_call else adverse >= pos.current_sl
            prem_sl = mark(adverse) <= pos.premium_stop
            if spot_sl or prem_sl:
                # On the way down the first trigger touched has the HIGHER premium.
                levels = []
                if spot_sl:
                    levels.append((mark(pos.current_sl), pos.current_sl, "intrabar_spot_sl"))
                if prem_sl:
                    levels.append((pos.premium_stop, adverse, "intrabar_premium_sl"))
                exit_mark, exit_spot, trigger = max(levels, key=lambda x: x[0])
                reason = "TRAILING_SL" if pos.premium_stop >= pos.entry_premium else "SL"
                self._exit(sid, exit_spot, exit_mark, reason, at, trigger)
                continue  # both touched → the stop is assumed first

            spot_tgt = favour >= pos.target_price if is_call else favour <= pos.target_price
            prem_tgt = mark(favour) >= pos.premium_target
            if spot_tgt or prem_tgt:
                # On the way up the first trigger touched has the LOWER premium.
                levels = []
                if spot_tgt:
                    levels.append((mark(pos.target_price), pos.target_price, "intrabar_spot_target"))
                if prem_tgt:
                    levels.append((pos.premium_target, favour, "intrabar_premium_target"))
                exit_mark, exit_spot, trigger = min(levels, key=lambda x: x[0])
                self._exit(sid, exit_spot, exit_mark, "TARGET", at, trigger)

    # Phase 3 — at the bar close: time stop, 15:15 force exit, trailing.
    def on_close(self, bar: Bar) -> None:
        self.rm.set_clock(bar.close_time)
        for sid in self.open_ids:
            upd = self.rm.update_position(sid, bar.close, _now=bar.close_time)
            if upd.action != "HOLD":
                self._exit(sid, bar.close, upd.mark_premium, _EXIT_REASONS.get(upd.action, upd.action),
                           bar.close_time, "close")

    def close_all(self, bar: Bar, reason: str) -> None:
        """Flatten at *bar*'s close (session data ended with positions open)."""
        at = bar.close_time
        self.rm.set_clock(at)
        for sid in self.open_ids:
            upd = self.rm.update_position(sid, bar.close, _now=at)
            self._exit(sid, bar.close, upd.mark_premium, reason, at, "forced")

    def _exit(self, sid: str, spot: float, mark: Optional[float], reason: str, at: datetime, trigger: str) -> None:
        pos = self.rm._open_positions.get(sid)
        if pos is None:
            return
        snap = {
            "signal_type": pos.signal_type, "index_id": pos.index_id, "strike": pos.strike,
            "expiry": pos.expiry, "option_type": pos.option_type, "lots": pos.lots,
            "qty": pos.quantity, "entry_premium": pos.entry_premium, "entry_mid": pos.premium_mid,
            "half_spread": pos.half_spread, "entry_time": pos.entry_time, "entry_spot": pos.entry_price,
            "iv": pos.entry_iv, "exchange": pos.exchange, "premium_stop": pos.premium_stop,
            "initial_premium_stop": pos.initial_premium_stop, "premium_target": pos.premium_target,
            "spot_stop": pos.current_sl, "spot_target": pos.target_price,
            "confidence": pos.confidence_level,
        }
        mark = float(mark) if mark and mark > 0 else (pos.last_premium or pos.premium_mid)
        self.rm.set_clock(at)
        net = self.rm.close_position(sid, spot, reason, exit_premium=mark)
        if net is None:
            return
        exit_px = max(mark - snap["half_spread"], 0.05)
        costs = round_trip_cost(snap["entry_premium"], exit_px, snap["qty"], snap["exchange"]).total
        info = self._info.pop(sid, {})
        self.trades.append({
            "signal_id": sid,
            "index_id": snap["index_id"],
            "direction": snap["signal_type"],
            "confidence": snap["confidence"],
            "decision_time": _iso(info.get("decision_time")),
            "entry_time": _iso(snap["entry_time"]),
            "exit_time": _iso(at),
            "strike": snap["strike"],
            "expiry": snap["expiry"].isoformat() if snap["expiry"] else None,
            "option_type": snap["option_type"],
            "lots": snap["lots"],
            "qty": snap["qty"],
            "iv": round(snap["iv"], 4),
            "iv_source": info.get("iv_source"),
            "decision_spot": info.get("decision_spot"),
            "decision_mid": info.get("decision_mid"),
            "entry_spot": snap["entry_spot"],
            "entry_mid": round(snap["entry_mid"], 4),
            "entry_premium": round(snap["entry_premium"], 4),
            "exit_spot": round(float(spot), 2),
            "exit_mark": round(mark, 4),
            "exit_premium": round(exit_px, 4),
            "half_spread": round(snap["half_spread"], 4),
            "premium_stop": snap["initial_premium_stop"],
            "premium_stop_final": snap["premium_stop"],
            "premium_target": snap["premium_target"],
            "spot_stop": snap["spot_stop"],
            "spot_target": snap["spot_target"],
            "gross_pnl": round((exit_px - snap["entry_premium"]) * snap["qty"], 2),
            "costs": round(costs, 2),
            "net_pnl": net,
            "exit_reason": reason,
            "exit_trigger": trigger,
            "premium_source": "model",
            "holding_minutes": round((at - snap["entry_time"]).total_seconds() / 60.0, 1),
        })


def _iso(ts) -> Optional[str]:
    return ts.isoformat() if isinstance(ts, (datetime, date)) else ts


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    index_id: str
    start: str
    end: str
    trades: list[dict] = field(default_factory=list)
    daily_pnl: dict[str, float] = field(default_factory=dict)
    equity_curve: list[dict] = field(default_factory=list)
    decision_counts: dict[str, int] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        nets = [t["net_pnl"] for t in self.trades]
        wins = [x for x in nets if x > 0]
        losses = [x for x in nets if x <= 0]
        capital = float(self.meta.get("capital", 0) or 0)
        peak, max_dd = capital, 0.0
        for p in self.equity_curve:
            peak = max(peak, p["equity"])
            max_dd = max(max_dd, peak - p["equity"])
        decisions = sum(self.decision_counts.values())
        return {
            "mode": self.meta.get("mode"),
            "index_id": self.index_id,
            "period": f"{self.start} → {self.end}",
            "sessions": self.meta.get("sessions"),
            "decisions": decisions,
            "trades": len(nets),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(nets), 4) if nets else None,
            "gross_pnl": round(sum(t["gross_pnl"] for t in self.trades), 2),
            "costs": round(sum(t["costs"] for t in self.trades), 2),
            "net_pnl": round(sum(nets), 2),
            "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
            "profit_factor": (
                round(sum(wins) / -sum(losses), 3) if losses and sum(losses) < 0 else None
            ),
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd / capital * 100, 3) if capital else None,
            "exit_reasons": dict(Counter(t["exit_reason"] for t in self.trades)),
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "meta": self.meta,
            "decision_counts": self.decision_counts,
            "daily_pnl": self.daily_pnl,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
        }

    def write(self, out_dir: Union[str, Path]) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": out / "replay_result.json",
            "trades": out / "trades.csv",
            "daily": out / "daily_pnl.csv",
            "decisions": out / "decision_counts.csv",
        }
        paths["json"].write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        cols = list(self.trades[0].keys()) if self.trades else [
            "entry_time", "exit_time", "index_id", "direction", "strike", "expiry", "option_type",
            "lots", "qty", "entry_premium", "exit_premium", "net_pnl", "costs", "exit_reason",
            "premium_source",
        ]
        with paths["trades"].open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(self.trades)
        with paths["daily"].open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "net_pnl"])
            w.writerows(sorted(self.daily_pnl.items()))
        with paths["decisions"].open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["outcome", "count"])
            w.writerows(sorted(self.decision_counts.items(), key=lambda kv: -kv[1]))
        return paths


# ---------------------------------------------------------------------------
# Decision classification
# ---------------------------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,:]\d+)*%?")


def reason_key(text: str) -> str:
    """Stable bucket for a NO_TRADE / rejection reason (numbers → '#')."""
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return "(no reason)"
    head = re.split(r" \(| — | -- ", lines[0].strip(), maxsplit=1)[0]
    return _NUM.sub("#", head).strip()[:90]


def classify_decision(result) -> str:
    sig = result.signal
    if result.is_actionable:
        return "TRADE"
    st = getattr(sig, "signal_type", "NO_TRADE")
    if st in ("BUY_CALL", "BUY_PUT"):
        reasons = getattr(sig, "rejection_reasons", None) or []
        return "RISK: " + (reason_key(reasons[0]) if reasons else "not actionable")
    text = getattr(sig, "reasoning", "") or ""
    if text.startswith("NO_TRADE: "):
        text = text[len("NO_TRADE: "):]
    return reason_key(text)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

_BIN_EDGES = [0.25, 0.40, 0.55, 0.70, 0.90, 1.20, 99.0]


def write_permissive_calibration(
    path: Union[str, Path],
    p_win: float = 0.60,
    indices: Iterable[str] = ("NIFTY50", "BANKNIFTY"),
) -> Path:
    """DIAGNOSTIC ONLY: a calibration that marks every bin tradeable.

    It exists so the replay can exercise contract selection, fills, exits
    and costs end-to-end.  It encodes no measured edge.
    """
    path = Path(path)
    bins = [
        {"lo": lo, "hi": hi, "n": 0, "wins": 0, "p_win": p_win}
        for lo, hi in zip(_BIN_EDGES, _BIN_EDGES[1:])
    ]
    payload = {
        "version": CALIBRATION_VERSION,
        "generated_at": datetime.now(_IST).isoformat(),
        "diagnostic": True,
        "note": DIAGNOSTIC_BANNER,
        "models": {
            ix: {"bins": bins, "validated": True, "meta": {"validation_note": "DIAGNOSTIC — not fitted"}}
            for ix in indices
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def calibration_mode(calibration_path: Optional[Union[str, Path]]) -> tuple[str, Optional[str]]:
    """``(mode, warning)`` — live / diagnostic / custom."""
    from src.engine.signal_calibration import DEFAULT_PATH

    if calibration_path is None or Path(calibration_path).resolve() == DEFAULT_PATH.resolve():
        return "live", None
    try:
        payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if payload.get("diagnostic"):
        return "diagnostic", DIAGNOSTIC_BANNER
    return "custom", (
        "Custom calibration file — results are only as good as that file's out-of-sample "
        "validation; they are not evidence of edge by themselves."
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _to_date(x: Union[str, date, datetime]) -> date:
    if isinstance(x, datetime):
        return _as_ist(x).date()
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def _copy_db(src: Path, dst_dir: Path) -> Path:
    """Byte copy of *src* (+ its WAL, if any) — the source is never opened."""
    dst = dst_dir / "replay_copy.db"
    shutil.copyfile(src, dst)
    wal = Path(str(src) + "-wal")
    if wal.exists():
        shutil.copyfile(wal, Path(str(dst) + "-wal"))
    os.chmod(dst, 0o600)
    return dst


def run_replay(
    db_path: Union[str, Path],
    index_id: str,
    start: Optional[Union[str, date, datetime]] = None,
    end: Optional[Union[str, date, datetime]] = None,
    calibration_path: Optional[Union[str, Path]] = None,
    capital: float = 500_000,
    stride: int = 1,
    out_dir: Optional[Union[str, Path]] = None,
    risk_config: Optional[RiskConfig] = None,
    lot_sizes: Optional[dict[str, int]] = None,
    progress_every: int = 0,
    keep_copy: bool = False,
    last_sessions: Optional[int] = None,
) -> ReplayResult:
    """Replay the live decision engine for *index_id* over [start, end].

    ``db_path`` is copied to a temporary file first and never written (or
    opened).  ``stride`` = decide every N bars (1 = the live 5-minute
    cycle); positions are managed on every bar regardless.  ``end`` defaults
    to the last stored session; ``start`` to ``last_sessions`` sessions
    before it (or the first stored session).
    """
    src = Path(db_path)
    if not src.is_file():
        raise FileNotFoundError(src)
    stride = max(int(stride), 1)
    d0 = _to_date(start) if start else None
    d1 = _to_date(end) if end else None
    mode, mode_warning = calibration_mode(calibration_path)
    t_wall0 = time.monotonic()

    tmpdir = Path(tempfile.mkdtemp(prefix="engine_replay_"))
    db: Optional[DatabaseManager] = None
    try:
        copy_path = _copy_db(src, tmpdir)
        db = DatabaseManager(db_path=copy_path, wal_mode=False)
        db.connect()
        # The copy's trade ledger is production's.  The daily trade cap and
        # restart recovery read it, so a replay must start from an empty one.
        db.execute("DELETE FROM trading_signals")

        all_bars = load_session_bars(db, index_id, until=d1)
        stored_days = sorted({b.day for b in all_bars})
        if not stored_days:
            raise ValueError(f"no stored 5m session bars for {index_id}")
        d1 = d1 or stored_days[-1]
        if d0 is None:
            d0 = stored_days[-last_sessions] if last_sessions and last_sessions <= len(stored_days) \
                else stored_days[0]
        window = [b for b in all_bars if d0 <= b.day <= d1]
        vol = VolatilityModel(all_bars, load_vix(db))

        cfg = risk_config or RiskConfig(total_capital=float(capital))
        calibration = SignalCalibration(Path(calibration_path)) if calibration_path else SignalCalibration()
        engine = ReplayDecisionEngine(db, vol, calibration=calibration, risk_config=cfg, lot_sizes=lot_sizes)
        rm: ReplayRiskManager = engine.risk_manager  # type: ignore[assignment]
        ex = ReplayExecutor(rm, lambda m: synthetic_half_spread(m, engine._half_spread_pct))

        counts: Counter = Counter()
        cycle_ms: list[float] = []
        pending: Optional[tuple[str, dict]] = None
        prev: Optional[Bar] = None
        n_decisions = 0

        for i, bar in enumerate(window):
            if prev is not None and bar.day != prev.day:
                ex.close_all(prev, "SESSION_END")
                if pending is not None:
                    rm.cancel_position(pending[0])
                    counts["NOT_FILLED: no next bar in session"] += 1
                    pending = None

            if pending is not None:
                if not ex.fill(pending[0], bar, pending[1]):
                    counts["NOT_FILLED: position missing at fill"] += 1
                pending = None

            ex.on_open(bar)
            ex.on_intrabar(bar)

            if i % stride == 0:
                as_of = bar.end
                rm.set_clock(as_of)
                c0 = time.monotonic()
                result = engine.run_full_cycle(index_id, as_of=as_of)
                cycle_ms.append((time.monotonic() - c0) * 1000.0)
                n_decisions += 1
                key = classify_decision(result)
                counts[key] += 1
                if result.is_actionable:
                    sig = result.signal
                    _, iv_src = vol.iv_at(as_of, count=False)
                    pending = (sig.signal_id, {
                        "iv_source": iv_src,
                        "confidence_score": getattr(sig, "confidence_score", None),
                    })
                if progress_every and n_decisions % progress_every == 0:
                    logger.warning(
                        "replay %s: %d decisions, %s, %.0f ms/cycle, %d trades",
                        index_id, n_decisions, bar.start.isoformat(),
                        statistics.mean(cycle_ms[-progress_every:]), len(ex.trades),
                    )

            ex.on_close(bar)
            prev = bar

        if prev is not None:
            ex.close_all(prev, "REPLAY_END")
        if pending is not None:
            rm.cancel_position(pending[0])
            counts["NOT_FILLED: no next bar in session"] += 1

        trades = sorted(ex.trades, key=lambda t: t["entry_time"])
        daily: dict[str, float] = {}
        for t in trades:
            day = t["exit_time"][:10]
            daily[day] = round(daily.get(day, 0.0) + t["net_pnl"], 2)
        equity = [{"time": f"{d0.isoformat()}T09:15:00+05:30", "equity": float(capital)}]
        eq = float(capital)
        for t in sorted(trades, key=lambda t: t["exit_time"]):
            eq += t["net_pnl"]
            equity.append({"time": t["exit_time"], "equity": round(eq, 2)})

        sessions = sorted({b.day for b in window})
        meta = {
            "mode": mode,
            "warning": mode_warning,
            "calibration_path": str(calibration_path) if calibration_path else "config/signal_calibration.json",
            "calibration_status": calibration.status,
            "capital": float(capital),
            "stride_bars": stride,
            "sessions": len(sessions),
            "first_session": sessions[0].isoformat() if sessions else None,
            "last_session": sessions[-1].isoformat() if sessions else None,
            "bars": len(window),
            "decisions": n_decisions,
            "cycle_ms_mean": round(statistics.mean(cycle_ms), 1) if cycle_ms else None,
            "cycle_ms_p95": (
                round(sorted(cycle_ms)[int(0.95 * (len(cycle_ms) - 1))], 1) if cycle_ms else None
            ),
            "runtime_seconds": round(time.monotonic() - t_wall0, 1),
            "iv_source_mix": {k: v for k, v in vol.source_counts.items() if v > 0},
            "synthetic_chains_built": engine.chains_built,
            "lot_sizes": dict(engine.replay_contract_master._lots),
            "assumptions": {
                "decision_time": "bar start + 5 min - 1 s (bar t complete, bar t+1 unseen)",
                "entry_fill": "next bar OPEN; contract re-priced with premium_at (IV flat) + half spread",
                "exits": "bar open (gap/time) → intrabar high/low vs spot & premium triggers "
                         "(stop first if both) → bar close (time stop, 15:15, trailing); "
                         "exit at trigger re-priced (model) minus half spread",
                "options_chain_for_analysis": "None (no point-in-time history)",
                "options_chain_for_risk": (
                    f"synthetic Black-Scholes, flat IV, ±{SYNTH_STRIKES_EACH_SIDE} strikes, "
                    f"half spread max({SYNTH_HALF_SPREAD_PCT:.1%} of premium, ₹{TICK}), flat OI {SYNTH_OI}"
                ),
                "iv": (
                    "VIX known at as_of (intraday ≤1 day old or previous session EOD), else "
                    f"{RV_MULTIPLIER}× realised vol of last {RV_BARS} 5m bars, floor {IV_FLOOR:.0%}; "
                    "same IV for every strike and for the life of the trade"
                ),
                "expiries": "NIFTY weekly Tuesday, BANKNIFTY monthly last Tuesday; holidays ignored",
                "anomaly_step": "disabled (wall-clock alert store)",
                "news": "disabled by the engine in replay mode (no point-in-time archive)",
                "costs": "src/engine/option_costs.round_trip_cost (NSE schedule)",
                "premium_source": "model (no historical option quotes)",
            },
        }
        result = ReplayResult(
            index_id=index_id, start=d0.isoformat(), end=d1.isoformat(),
            trades=trades, daily_pnl=daily, equity_curve=equity,
            decision_counts=dict(counts), meta=meta,
        )
        if keep_copy:
            meta["db_copy"] = str(copy_path)
        if out_dir is not None:
            result.write(out_dir)
        return result
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
        if not keep_copy:
            shutil.rmtree(tmpdir, ignore_errors=True)


__all__ = [
    "Bar", "DIAGNOSTIC_BANNER", "ReplayContractMaster", "ReplayDecisionEngine", "ReplayExecutor",
    "ReplayResult", "ReplayRiskManager", "ReplaySignalGenerator", "VolatilityModel",
    "build_synthetic_chain", "calibration_mode", "classify_decision", "expiry_on_or_after",
    "following_expiry", "load_session_bars", "load_vix", "nearest_expiry", "reason_key",
    "run_replay", "synthetic_half_spread", "write_permissive_calibration",
]
