"""
Phase 4 — Step 4.1: Volume & Price Anomaly Detector.

Real-time monitor that detects volume and price anomalies as new data
arrives, compares against rolling baselines, generates severity-graded
alerts, persists them to ``anomaly_events``, and suppresses duplicates
via a cooldown system.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from src.database.db_manager import DatabaseManager
from src.database import queries as Q

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Severity cooldown windows (minutes)
# ---------------------------------------------------------------------------
_COOLDOWN_MINUTES = {
    "HIGH": 10,
    "MEDIUM": 15,
    "LOW": 30,
}

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Baselines:
    """Rolling statistical baselines for one index, recomputed daily."""

    index_id: str
    computed_at: datetime
    avg_volume_20d: float = 0.0
    std_volume_20d: float = 0.0
    avg_range_20d: float = 0.0
    avg_range_pct_20d: float = 0.0
    avg_body_20d: float = 0.0
    avg_intraday_volatility: float = 0.0
    typical_first_15min_range: float = 0.0
    typical_last_30min_volume: float = 0.0


@dataclass
class VolumeAnomaly:
    """Detected volume anomaly for a single bar."""

    index_id: str
    timestamp: datetime
    anomaly_type: str
    severity: str
    volume: float
    volume_zscore: float
    volume_ratio: float
    price_change_pct: float
    time_context: str
    details: dict = field(default_factory=dict)
    message: str = ""


@dataclass
class PriceAnomaly:
    """Detected price anomaly for a single bar."""

    index_id: str
    timestamp: datetime
    anomaly_type: str
    severity: str
    price_change_pct: float
    day_range_vs_avg: float
    details: dict = field(default_factory=dict)
    message: str = ""


@dataclass
class AnomalyEvent:
    """Unified anomaly event ready for database persistence."""

    index_id: str
    timestamp: datetime
    anomaly_type: str
    severity: str
    category: str
    details: str
    message: str
    is_active: bool = True
    cooldown_key: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _time_context(ts: datetime) -> str:
    """Classify a bar timestamp into a session bucket."""
    t = ts.time()
    from datetime import time as _t

    if t < _t(9, 30):
        return "OPEN"
    if t < _t(12, 0):
        return "MIDDAY"
    if t < _t(15, 0):
        return "PRE_CLOSE"
    return "CLOSE"


def _safe_float(val, default: float = 0.0) -> float:
    """Return *val* as float, falling back to *default* on NaN / None."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _volume_threshold_multiplier(ctx: str) -> float:
    """Time-of-day multiplier for volume z-score thresholds.

    High volume at market open (9:15-9:30) or close on expiry is normal,
    so we raise the bar.  Midday spikes are unusual — lower threshold.
    """
    if ctx == "OPEN":
        return 2.0
    if ctx == "MIDDAY":
        return 1.5
    return 1.0  # PRE_CLOSE / CLOSE — use base thresholds


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class VolumePriceDetector:
    """Real-time volume & price anomaly detector.

    Maintains per-index rolling baselines and an in-memory cooldown
    tracker so the same anomaly type for the same index is not alerted
    more frequently than the configured window.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        # {index_id: Baselines}
        self._baselines: dict[str, Baselines] = {}
        # {cooldown_key: last_alert_datetime}
        self._cooldowns: dict[str, datetime] = {}
        # {index_id: list[dict]} — recent bars cache for acceleration check
        self._recent_bars: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def get_baselines(self, index_id: str) -> Optional[Baselines]:
        """Return cached baselines for *index_id*, or ``None``."""
        return self._baselines.get(index_id)

    def update_baselines(self, index_id: str, price_df: pd.DataFrame) -> Baselines:
        """Compute and cache rolling baselines from historical OHLCV data.

        Parameters
        ----------
        index_id:
            Index identifier (e.g. ``"NIFTY50"``).
        price_df:
            DataFrame with columns ``open, high, low, close, volume`` and a
            DatetimeIndex (or ``timestamp`` column).  Must contain at least 5
            rows; ideally ≥ 20 for meaningful statistics.

        Returns
        -------
        Baselines
            Freshly computed baselines, also cached in ``self._baselines``.
        """
        df = price_df.copy()

        # Normalise index
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        n = min(20, len(df))
        recent = df.tail(n)

        vol = recent["volume"].astype(float)
        avg_volume = float(vol.mean()) if len(vol) > 0 else 0.0
        std_volume = float(vol.std(ddof=1)) if len(vol) > 1 else 0.0

        daily_range = (recent["high"] - recent["low"]).astype(float)
        avg_range = float(daily_range.mean()) if len(daily_range) > 0 else 0.0

        close = recent["close"].astype(float)
        avg_range_pct = float((daily_range / close).mean()) if len(close) > 0 else 0.0

        body = (recent["close"] - recent["open"]).abs().astype(float)
        avg_body = float(body.mean()) if len(body) > 0 else 0.0

        intraday_vol = ((recent["high"] - recent["low"]) / recent["open"]).astype(float)
        avg_intraday_volatility = float(intraday_vol.mean()) if len(intraday_vol) > 0 else 0.0

        # First-15-min range and last-30-min volume are only meaningful with
        # intraday data.  For daily bars we leave them as 0.
        typical_first_15min_range = 0.0
        typical_last_30min_volume = 0.0

        baselines = Baselines(
            index_id=index_id,
            computed_at=datetime.now(tz=_IST),
            avg_volume_20d=avg_volume,
            std_volume_20d=std_volume,
            avg_range_20d=avg_range,
            avg_range_pct_20d=avg_range_pct,
            avg_body_20d=avg_body,
            avg_intraday_volatility=avg_intraday_volatility,
            typical_first_15min_range=typical_first_15min_range,
            typical_last_30min_volume=typical_last_30min_volume,
        )
        self._baselines[index_id] = baselines
        return baselines

    # ------------------------------------------------------------------
    # Volume anomaly detection
    # ------------------------------------------------------------------

    def detect_volume_anomaly(
        self,
        index_id: str,
        current_bar: dict,
        baselines: Baselines,
    ) -> Optional[VolumeAnomaly]:
        """Check a single bar for volume anomalies.

        Returns the *most severe* anomaly found, or ``None``.
        """
        volume = _safe_float(current_bar.get("volume"))
        if volume <= 0:
            return None

        open_ = _safe_float(current_bar.get("open"))
        close = _safe_float(current_bar.get("close"))
        if open_ == 0:
            return None

        ts = current_bar.get("timestamp")
        if isinstance(ts, str):
            ts = pd.Timestamp(ts)
        if not isinstance(ts, datetime):
            ts = pd.Timestamp(ts).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_IST)

        ctx = _time_context(ts)
        price_change_pct = ((close - open_) / open_) * 100 if open_ else 0.0

        avg_vol = baselines.avg_volume_20d
        std_vol = baselines.std_volume_20d

        if avg_vol <= 0 or std_vol <= 0:
            logger.warning(
                "Baselines missing volume stats for %s — skipping volume check",
                index_id,
            )
            return None

        z_score = (volume - avg_vol) / std_vol
        volume_ratio = volume / avg_vol
        threshold_mult = _volume_threshold_multiplier(ctx)

        # --- Check 1: Volume z-score spike ---
        anomaly_type: Optional[str] = None
        severity: Optional[str] = None

        adjusted_z = z_score / threshold_mult

        if adjusted_z > 4.0:
            anomaly_type = "VOLUME_SPIKE"
            severity = "HIGH"
        elif adjusted_z > 3.0:
            anomaly_type = "VOLUME_SPIKE"
            severity = "HIGH"
        elif adjusted_z > 2.0:
            anomaly_type = "VOLUME_SPIKE"
            severity = "MEDIUM"

        # --- Check 2: Volume acceleration (current vs previous 3 bars) ---
        prev_bars = self._recent_bars.get(index_id, [])
        if len(prev_bars) >= 3:
            prev_3_avg = sum(
                _safe_float(b.get("volume")) for b in prev_bars[-3:]
            ) / 3.0
            if prev_3_avg > 0 and volume > 3 * prev_3_avg:
                # Acceleration overrides a weaker spike
                if severity is None or severity == "MEDIUM":
                    anomaly_type = "SUDDEN_ACCELERATION"
                    severity = "HIGH"

        # --- Check 3: Volume drought ---
        if z_score < -1.5 and anomaly_type is None:
            anomaly_type = "VOLUME_DROUGHT"
            severity = "LOW"

        # --- Check 4: Absorption (high volume, tiny price move) ---
        avg_body = baselines.avg_body_20d
        if (
            z_score > 2.5
            and avg_body > 0
            and abs(close - open_) < 0.3 * avg_body
        ):
            anomaly_type = "ABSORPTION"
            severity = "HIGH"

        # Cache this bar for future acceleration checks
        self._cache_bar(index_id, current_bar)

        if anomaly_type is None:
            return None

        # Build human-readable message
        ratio_str = f"{volume_ratio:.1f}x avg"
        if anomaly_type == "ABSORPTION":
            msg = (
                f"{index_id}: Volume spike ({ratio_str}, z={z_score:.1f}) at "
                f"{ts.strftime('%H:%M')} with minimal price movement "
                f"— possible institutional absorption"
            )
        elif anomaly_type == "VOLUME_DROUGHT":
            msg = (
                f"{index_id}: Unusually low volume ({ratio_str}) at "
                f"{ts.strftime('%H:%M')} — liquidity concern"
            )
        elif anomaly_type == "SUDDEN_ACCELERATION":
            msg = (
                f"{index_id}: Sudden volume acceleration ({ratio_str}, "
                f"3x recent bars) at {ts.strftime('%H:%M')}"
            )
        else:
            msg = (
                f"{index_id}: Volume spike ({ratio_str}, z={z_score:.1f}) "
                f"at {ts.strftime('%H:%M')}"
            )

        return VolumeAnomaly(
            index_id=index_id,
            timestamp=ts,
            anomaly_type=anomaly_type,
            severity=severity,
            volume=volume,
            volume_zscore=round(z_score, 2),
            volume_ratio=round(volume_ratio, 2),
            price_change_pct=round(price_change_pct, 4),
            time_context=ctx,
            details={
                "avg_volume_20d": round(avg_vol, 2),
                "std_volume_20d": round(std_vol, 2),
                "threshold_multiplier": threshold_mult,
                "adjusted_z": round(adjusted_z, 2),
            },
            message=msg,
        )

    # ------------------------------------------------------------------
    # Price anomaly detection
    # ------------------------------------------------------------------

    def detect_price_anomaly(
        self,
        index_id: str,
        current_bar: dict,
        recent_bars: list[dict],
        baselines: Baselines,
    ) -> Optional[PriceAnomaly]:
        """Check for price anomalies.  Returns the *most severe* found."""
        open_ = _safe_float(current_bar.get("open"))
        high = _safe_float(current_bar.get("high"))
        low = _safe_float(current_bar.get("low"))
        close = _safe_float(current_bar.get("close"))

        if open_ == 0 or close == 0:
            return None

        ts = current_bar.get("timestamp")
        if isinstance(ts, str):
            ts = pd.Timestamp(ts)
        if not isinstance(ts, datetime):
            ts = pd.Timestamp(ts).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_IST)

        avg_range = baselines.avg_range_20d
        if avg_range <= 0:
            logger.warning(
                "Baselines missing range stats for %s — skipping price check",
                index_id,
            )
            return None

        today_range = high - low
        day_range_vs_avg = today_range / avg_range if avg_range else 0.0

        # Collect candidates as (type, severity, pct, details, msg)
        candidates: list[tuple[str, str, float, dict, str]] = []

        # --- 1. Gap detection (first bar of day) ---
        if recent_bars:
            prev_close = _safe_float(recent_bars[-1].get("close"))
            if prev_close > 0:
                gap_pct = ((open_ - prev_close) / prev_close) * 100
                abs_gap = abs(gap_pct)
                if abs_gap >= 0.5:
                    direction = "GAP_UP" if gap_pct > 0 else "GAP_DOWN"
                    if abs_gap >= 2.0:
                        sev = "HIGH"
                    elif abs_gap >= 1.0:
                        sev = "HIGH"
                    else:
                        sev = "MEDIUM"
                    candidates.append((
                        direction,
                        sev,
                        round(gap_pct, 4),
                        {"gap_pct": round(gap_pct, 4), "prev_close": prev_close},
                        f"{index_id}: {direction.replace('_', ' ')} of {abs_gap:.2f}% "
                        f"(prev close {prev_close:.1f} → open {open_:.1f})",
                    ))

        # --- 2. Large intraday move ---
        move_from_open = abs(close - open_)
        move_pct = (move_from_open / open_) * 100
        # Check if within first 2 hours (before 11:15)
        from datetime import time as _t

        if ts.time() <= _t(11, 15):
            if move_from_open > 3 * avg_range:
                candidates.append((
                    "EXTREME_INTRADAY_MOVE",
                    "HIGH",
                    round(move_pct, 4),
                    {"move_from_open": round(move_from_open, 2), "avg_range": round(avg_range, 2)},
                    f"{index_id}: Extreme intraday move ({move_pct:.2f}%, "
                    f"{move_from_open / avg_range:.1f}x avg range) within first 2 hours",
                ))
            elif move_from_open > 2 * avg_range:
                candidates.append((
                    "LARGE_INTRADAY_MOVE",
                    "HIGH",
                    round(move_pct, 4),
                    {"move_from_open": round(move_from_open, 2), "avg_range": round(avg_range, 2)},
                    f"{index_id}: Large intraday move ({move_pct:.2f}%, "
                    f"{move_from_open / avg_range:.1f}x avg range) within first 2 hours",
                ))

        # --- 3. Intraday reversal ---
        if len(recent_bars) >= 1:
            bars_with_current = recent_bars + [current_bar]
            day_high = max(_safe_float(b.get("high")) for b in bars_with_current)
            day_low = min(
                _safe_float(b.get("low"), default=float("inf"))
                for b in bars_with_current
            )
            day_open = _safe_float(bars_with_current[0].get("open"))
            if day_open > 0:
                up_move_pct = ((day_high - day_open) / day_open) * 100
                down_from_high_pct = ((day_high - close) / day_open) * 100
                down_move_pct = ((day_open - day_low) / day_open) * 100
                up_from_low_pct = ((close - day_low) / day_open) * 100

                # Bearish reversal: went up >1% then pulled back >0.7%
                if up_move_pct > 1.0 and down_from_high_pct > 0.7:
                    candidates.append((
                        "INTRADAY_REVERSAL",
                        "MEDIUM",
                        round(-down_from_high_pct, 4),
                        {
                            "day_high": round(day_high, 2),
                            "reversal_pct": round(down_from_high_pct, 4),
                            "direction": "BEARISH",
                        },
                        f"{index_id}: Bearish intraday reversal — rallied {up_move_pct:.2f}% "
                        f"then pulled back {down_from_high_pct:.2f}%",
                    ))
                # Bullish reversal: went down >1% then recovered >0.7%
                elif down_move_pct > 1.0 and up_from_low_pct > 0.7:
                    candidates.append((
                        "INTRADAY_REVERSAL",
                        "MEDIUM",
                        round(up_from_low_pct, 4),
                        {
                            "day_low": round(day_low, 2),
                            "reversal_pct": round(up_from_low_pct, 4),
                            "direction": "BULLISH",
                        },
                        f"{index_id}: Bullish intraday reversal — dropped {down_move_pct:.2f}% "
                        f"then recovered {up_from_low_pct:.2f}%",
                    ))

        # --- 4. Range expansion ---
        if day_range_vs_avg > 1.5:
            sev = "HIGH" if day_range_vs_avg > 2.5 else "MEDIUM"
            candidates.append((
                "RANGE_EXPANSION",
                sev,
                round(((close - open_) / open_) * 100, 4),
                {"day_range": round(today_range, 2), "avg_range": round(avg_range, 2)},
                f"{index_id}: Range expansion — today's range is "
                f"{day_range_vs_avg:.1f}x the 20-day average",
            ))

        # --- 5. Compression ---
        if ts.time() >= _t(12, 0) and day_range_vs_avg < 0.3:
            candidates.append((
                "COMPRESSION",
                "LOW",
                round(((close - open_) / open_) * 100, 4),
                {"day_range": round(today_range, 2), "avg_range": round(avg_range, 2)},
                f"{index_id}: Price compression — range only "
                f"{day_range_vs_avg:.1f}x avg by midday, potential breakout building",
            ))

        if not candidates:
            return None

        # Return the most severe candidate
        _sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        candidates.sort(key=lambda c: _sev_order.get(c[1], 9))
        best = candidates[0]

        return PriceAnomaly(
            index_id=index_id,
            timestamp=ts,
            anomaly_type=best[0],
            severity=best[1],
            price_change_pct=best[2],
            day_range_vs_avg=round(day_range_vs_avg, 2),
            details=best[3],
            message=best[4],
        )

    # ------------------------------------------------------------------
    # Unified detection
    # ------------------------------------------------------------------

    def detect_all(
        self,
        index_id: str,
        current_bar: dict,
        recent_bars: list[dict],
        baselines: Baselines,
    ) -> list[AnomalyEvent]:
        """Run all volume and price checks, apply cooldowns, return events."""
        events: list[AnomalyEvent] = []

        vol_anomaly = self.detect_volume_anomaly(index_id, current_bar, baselines)
        if vol_anomaly is not None:
            ev = self._to_event(vol_anomaly, category="VOLUME")
            if not self._is_on_cooldown(ev):
                events.append(ev)
                self._set_cooldown(ev)

        price_anomaly = self.detect_price_anomaly(
            index_id, current_bar, recent_bars, baselines,
        )
        if price_anomaly is not None:
            ev = self._to_event(price_anomaly, category="PRICE")
            if not self._is_on_cooldown(ev):
                events.append(ev)
                self._set_cooldown(ev)

        return events

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_anomalies(self, anomalies: list[AnomalyEvent]) -> int:
        """Persist anomaly events to the database.

        Checks for existing active anomalies of the same type within the
        cooldown window before inserting.  Returns number of rows inserted.
        """
        inserted = 0
        for ev in anomalies:
            # DB-level dedup: check most recent event for same key
            row = self.db.fetch_one(
                Q.GET_LATEST_ANOMALY_BY_KEY,
                (ev.index_id, ev.anomaly_type),
            )
            if row is not None:
                last_ts = pd.Timestamp(row["timestamp"])
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize(_IST)
                cooldown_min = _COOLDOWN_MINUTES.get(ev.severity, 15)
                if (ev.timestamp - last_ts.to_pydatetime()) < timedelta(minutes=cooldown_min):
                    logger.debug(
                        "Skipping duplicate %s for %s (within cooldown)",
                        ev.anomaly_type,
                        ev.index_id,
                    )
                    continue

            ts_str = ev.timestamp.isoformat()
            self.db.execute(
                Q.INSERT_ANOMALY_EVENT,
                (
                    ev.index_id,
                    ts_str,
                    ev.anomaly_type,
                    ev.severity,
                    ev.category,
                    ev.details,
                    ev.message,
                    1,
                ),
            )
            inserted += 1

        return inserted

    # ------------------------------------------------------------------
    # Cooldown system
    # ------------------------------------------------------------------

    def _is_on_cooldown(self, event: AnomalyEvent) -> bool:
        key = event.cooldown_key
        last = self._cooldowns.get(key)
        if last is None:
            return False
        cooldown_min = _COOLDOWN_MINUTES.get(event.severity, 15)
        return (event.timestamp - last) < timedelta(minutes=cooldown_min)

    def _set_cooldown(self, event: AnomalyEvent) -> None:
        self._cooldowns[event.cooldown_key] = event.timestamp

    def clear_expired_cooldowns(self, now: Optional[datetime] = None) -> int:
        """Remove cooldown entries older than the maximum window (30 min)."""
        now = now or datetime.now(tz=_IST)
        max_window = timedelta(minutes=30)
        expired = [
            k for k, v in self._cooldowns.items() if (now - v) >= max_window
        ]
        for k in expired:
            del self._cooldowns[k]
        return len(expired)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_bar(self, index_id: str, bar: dict) -> None:
        """Keep the last 10 bars per index for acceleration checks."""
        bars = self._recent_bars.setdefault(index_id, [])
        bars.append(bar)
        if len(bars) > 10:
            self._recent_bars[index_id] = bars[-10:]

    def _to_event(
        self,
        anomaly: VolumeAnomaly | PriceAnomaly,
        category: str,
    ) -> AnomalyEvent:
        """Convert a typed anomaly into a unified AnomalyEvent."""
        details_dict: dict = {}
        if isinstance(anomaly, VolumeAnomaly):
            details_dict = {
                "volume": anomaly.volume,
                "volume_zscore": anomaly.volume_zscore,
                "volume_ratio": anomaly.volume_ratio,
                "price_change_pct": anomaly.price_change_pct,
                "time_context": anomaly.time_context,
                **anomaly.details,
            }
        elif isinstance(anomaly, PriceAnomaly):
            details_dict = {
                "price_change_pct": anomaly.price_change_pct,
                "day_range_vs_avg": anomaly.day_range_vs_avg,
                **anomaly.details,
            }

        return AnomalyEvent(
            index_id=anomaly.index_id,
            timestamp=anomaly.timestamp,
            anomaly_type=anomaly.anomaly_type,
            severity=anomaly.severity,
            category=category,
            details=json.dumps(details_dict),
            message=anomaly.message,
            is_active=True,
            cooldown_key=f"{anomaly.index_id}_{anomaly.anomaly_type}",
        )
