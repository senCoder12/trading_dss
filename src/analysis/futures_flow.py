"""
Futures flow: positioning and order flow from the near-month index future.

The index spot carries no volume or open interest, so every volume-based
vote on it was empty.  The near-month future does: `futures_depth_1m`
(1-minute aggregates from Angel One SnapQuote) stores its close, open
interest, traded volume and best-level order-flow imbalance (OFI).

Signal over the last ``lookback`` minutes:

  price ↑  OI ↑   LONG_BUILDUP     new longs          bullish
  price ↓  OI ↑   SHORT_BUILDUP    new shorts         bearish
  price ↑  OI ↓   SHORT_COVERING   shorts exiting     bullish (weaker)
  price ↓  OI ↓   LONG_UNWINDING   longs exiting      bearish (weaker)

and OFI (aggressive buying minus selling at the touch) must not contradict
the buildup.  Moves smaller than a noise band are NEUTRAL.

The component is *unavailable* (not neutral) when the series is missing,
stale, too short, or has no traded volume — it then simply does not vote.
Its weight in the calibrated score is fixed (FLOW_WEIGHT); like every other
scored input it only becomes tradeable after calibration validates it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")

FLOW_WEIGHT = 0.30            # share of the calibrated score when available
LOOKBACK_MIN = 30
MIN_ROWS = 20                 # minutes with data inside the lookback
MAX_STALENESS_MIN = 5
PRICE_NOISE_PCT = 0.0005      # 0.05 % of price
OI_NOISE_PCT = 0.001          # 0.1 % of open interest


@dataclass(frozen=True)
class FuturesFlow:
    available: bool
    reason: str = ""
    buildup: str = "NONE"
    vote: str = "NEUTRAL"             # BULLISH / BEARISH / STRONG_* / NEUTRAL
    confidence: float = 0.0
    price_change_pct: float = 0.0
    oi_change_pct: float = 0.0
    ofi_sum: float = 0.0
    volume: float = 0.0


def _ts(value) -> Optional[datetime]:
    try:
        ts = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return ts.replace(tzinfo=_IST) if ts.tzinfo is None else ts.astimezone(_IST)


def compute_futures_flow(
    rows: Sequence[dict],
    as_of: datetime,
    lookback_min: int = LOOKBACK_MIN,
) -> FuturesFlow:
    """Flow signal from 1-minute futures rows (any order) at *as_of*.

    Only rows with timestamp in (as_of − lookback, as_of − 1 min] are used:
    a minute bar is complete once its minute has ended.
    """
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=_IST)
    start = as_of - timedelta(minutes=lookback_min)
    last_complete = as_of - timedelta(minutes=1)
    window = []
    for r in rows or ():
        ts = _ts(r.get("timestamp"))
        if ts is None or not (start < ts <= last_complete):
            continue
        try:
            close = float(r.get("fut_close") or 0)
            oi = float(r.get("open_interest") or 0)
        except (TypeError, ValueError):
            continue
        if close <= 0 or oi <= 0:
            continue
        window.append((ts, close, oi, float(r.get("volume") or 0), float(r.get("ofi") or 0)))
    window.sort(key=lambda x: x[0])

    if len(window) < MIN_ROWS:
        return FuturesFlow(False, f"only {len(window)} futures minutes in the last {lookback_min} min")
    if (as_of - window[-1][0]).total_seconds() > (MAX_STALENESS_MIN + 1) * 60:
        return FuturesFlow(False, "futures data stale")
    volume = sum(w[3] for w in window)
    if volume <= 0:
        return FuturesFlow(False, "no futures volume traded in the window")

    p0, p1 = window[0][1], window[-1][1]
    oi0, oi1 = window[0][2], window[-1][2]
    dp = (p1 - p0) / p0
    doi = (oi1 - oi0) / oi0
    ofi = sum(w[4] for w in window)

    price_up = dp > PRICE_NOISE_PCT
    price_dn = dp < -PRICE_NOISE_PCT
    oi_up = doi > OI_NOISE_PCT
    oi_dn = doi < -OI_NOISE_PCT

    if price_up and oi_up:
        buildup, direction, strength = "LONG_BUILDUP", 1, 2
    elif price_dn and oi_up:
        buildup, direction, strength = "SHORT_BUILDUP", -1, 2
    elif price_up and oi_dn:
        buildup, direction, strength = "SHORT_COVERING", 1, 1
    elif price_dn and oi_dn:
        buildup, direction, strength = "LONG_UNWINDING", -1, 1
    else:
        return FuturesFlow(True, "", "NONE", "NEUTRAL", 0.5, round(dp * 100, 4),
                           round(doi * 100, 4), ofi, volume)

    # Order flow must not contradict the buildup.
    if ofi != 0 and math.copysign(1, ofi) != direction:
        strength -= 1
    if strength <= 0:
        vote, conf = "NEUTRAL", 0.5
    elif strength == 1:
        vote, conf = ("BULLISH" if direction > 0 else "BEARISH"), 0.6
    else:
        vote, conf = ("STRONG_BULLISH" if direction > 0 else "STRONG_BEARISH"), 0.75
    return FuturesFlow(True, "", buildup, vote, conf, round(dp * 100, 4),
                       round(doi * 100, 4), ofi, volume)


def blend_score(base_score: float, flow: Optional[FuturesFlow], vote_numeric: dict) -> float:
    """Calibrated score with the flow component mixed in (when available)."""
    if flow is None or not flow.available:
        return base_score
    f = vote_numeric.get(flow.vote, 0.0) * flow.confidence
    return (1.0 - FLOW_WEIGHT) * base_score + FLOW_WEIGHT * f
