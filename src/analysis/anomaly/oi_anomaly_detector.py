"""
Phase 4 -- Step 4.2: OI & Options Anomaly Detector.

Detects anomalous patterns in open interest, PCR, max pain, and implied
volatility that deviate from historical baselines.  Runs after every
options-chain fetch (~3 min interval), builds on Phase 1 (OptionsChainFetcher)
and Phase 2 (OptionsIndicators), and emits unified ``AnomalyEvent`` objects
compatible with Step 4.1.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from zoneinfo import ZoneInfo

from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from src.data.options_chain import OptionsChainData, OptionStrike, OISummary
from src.analysis.anomaly.volume_price_detector import AnomalyEvent
from config.settings import get_anomaly_thresholds

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Severity cooldown windows (minutes) -- same as Step 4.1
# ---------------------------------------------------------------------------
_COOLDOWN_MINUTES = {
    "HIGH": 10,
    "MEDIUM": 15,
    "LOW": 30,
}

# Minimum OI at a strike to consider percentage changes meaningful
_MIN_STRIKE_OI = 1000


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OIBaselines:
    """Rolling statistical baselines for OI/options behaviour over 20 days."""

    index_id: str
    computed_at: Optional[datetime] = None
    avg_total_oi: float = 0.0
    std_total_oi: float = 0.0
    avg_oi_change_per_3min: float = 0.0
    std_oi_change_per_3min: float = 0.0
    avg_pcr: float = 0.0
    std_pcr: float = 0.0
    avg_max_pain_shift_per_day: float = 0.0
    typical_oi_at_atm: float = 0.0
    typical_iv_range: tuple[float, float] = (0.0, 0.0)


@dataclass
class OIAnomaly:
    """A single detected OI/options anomaly."""

    index_id: str
    timestamp: datetime
    anomaly_type: str
    severity: str
    total_oi_change: int = 0
    oi_change_zscore: float = 0.0
    affected_strikes: list[dict] = field(default_factory=list)
    directional_implication: str = "UNCLEAR"
    details: dict = field(default_factory=dict)
    message: str = ""


@dataclass
class OIAnomalySummary:
    """Summary of active OI anomalies for an index."""

    index_id: str
    active_anomalies: list[AnomalyEvent] = field(default_factory=list)
    dominant_signal: str = "NEUTRAL"
    anomaly_intensity: str = "NONE"
    key_strikes_to_watch: list[float] = field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _z_score(value: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    return (value - mean) / std


def _is_expiry_day(expiry_date: date, current_date: date) -> bool:
    return expiry_date == current_date


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class OIAnomalyDetector:
    """Real-time OI & options anomaly detector.

    Maintains per-index OI baselines, a sliding window of recent OI
    snapshots, and a cooldown tracker to suppress duplicate alerts.
    """

    def __init__(self, db: DatabaseManager, timeframe: str = "5m") -> None:
        self.db = db
        self.timeframe = timeframe
        self.thresholds = get_anomaly_thresholds(timeframe)
        # {index_id: OIBaselines}
        self._baselines: dict[str, OIBaselines] = {}
        # {index_id: list[OISummary]} -- last 10 snapshots
        self._snapshot_history: dict[str, list[OISummary]] = {}
        # {index_id: list[float]} -- PCR history (last 10 values)
        self._pcr_history: dict[str, list[float]] = {}
        # {cooldown_key: last_alert_datetime}
        self._cooldowns: dict[str, datetime] = {}
        # {index_id: list[AnomalyEvent]} -- active anomalies
        self._active_anomalies: dict[str, list[AnomalyEvent]] = {}

    def set_timeframe(self, timeframe: str) -> None:
        """Switch threshold profile (e.g. when backtesting on daily vs live on 5m)."""
        self.timeframe = timeframe
        self.thresholds = get_anomaly_thresholds(timeframe)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def get_baselines(self, index_id: str) -> Optional[OIBaselines]:
        cached = self._baselines.get(index_id)
        if cached is None:
            return None

        # Reject uncached fallbacks
        if cached.computed_at is None:
            if index_id in self._baselines:
                del self._baselines[index_id]
            return None

        # Check staleness — recompute after 24 hours
        now = datetime.now(tz=_IST)
        age_seconds = (now - cached.computed_at).total_seconds()
        if age_seconds > 86400:
            logger.debug(f"OI Baselines {index_id}: stale ({age_seconds/3600:.1f}h). Evicting cache.")
            del self._baselines[index_id]
            return None

        return cached

    def update_oi_baselines(self, index_id: str) -> OIBaselines:
        """Compute OI baselines from the last 20 trading days of oi_aggregated data.

        Prevents cache poisoning by validating data before storage.
        """
        MIN_BARS = 10
        end = datetime.now(tz=_IST)
        start = end - timedelta(days=30)  # fetch 30 calendar days to get ~20 trading days

        rows = self.db.fetch_all(
            Q.LIST_OI_AGGREGATED_HISTORY,
            (index_id, "%", start.isoformat(), end.isoformat()),
        )

        if not rows or len(rows) < MIN_BARS:
            logger.warning(
                f"OI Baselines {index_id}: insufficient data "
                f"({len(rows) if rows else 0} bars, need {MIN_BARS}). "
                f"NOT caching."
            )
            bl = OIBaselines(index_id=index_id, computed_at=None)
            # Return but DO NOT cache
            return bl

        df = pd.DataFrame(rows)

        # Total OI per snapshot
        total_ois = (df["total_ce_oi"].astype(float) + df["total_pe_oi"].astype(float))
        avg_total_oi = float(total_ois.mean()) if len(total_ois) > 0 else 0.0

        # Validate before caching
        if avg_total_oi <= 0:
            logger.warning(
                f"OI Baselines {index_id}: computed avg_total_oi is 0 despite {len(df)} bars. "
                "Data may be corrupt. NOT caching."
            )
            return OIBaselines(index_id=index_id, computed_at=None)

        std_total_oi = float(total_ois.std(ddof=1)) if len(total_ois) > 1 else 0.0

        # OI change between consecutive snapshots
        oi_changes = total_ois.diff().dropna().abs()
        avg_oi_change = float(oi_changes.mean()) if len(oi_changes) > 0 else 0.0
        std_oi_change = float(oi_changes.std(ddof=1)) if len(oi_changes) > 1 else 0.0

        # PCR stats
        pcr_vals = df["pcr"].astype(float).dropna()
        avg_pcr = float(pcr_vals.mean()) if len(pcr_vals) > 0 else 1.0
        std_pcr = float(pcr_vals.std(ddof=1)) if len(pcr_vals) > 1 else 0.0

        # Max pain shift per day
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily_mp = df.groupby("date")["max_pain_strike"].agg(["first", "last"])
        daily_shifts = (daily_mp["last"] - daily_mp["first"]).abs()
        avg_mp_shift = float(daily_shifts.mean()) if len(daily_shifts) > 0 else 0.0

        bl = OIBaselines(
            index_id=index_id,
            computed_at=datetime.now(tz=_IST),
            avg_total_oi=avg_total_oi,
            std_total_oi=std_total_oi,
            avg_oi_change_per_3min=avg_oi_change,
            std_oi_change_per_3min=std_oi_change,
            avg_pcr=avg_pcr,
            std_pcr=std_pcr,
            avg_max_pain_shift_per_day=avg_mp_shift,
        )
        self._baselines[index_id] = bl
        logger.debug(f"OI Baselines cached {index_id}: avg_total_oi={bl.avg_total_oi:.0f}")
        return bl

    # ------------------------------------------------------------------
    # Snapshot history management
    # ------------------------------------------------------------------

    def _push_snapshot(self, index_id: str, summary: OISummary) -> None:
        history = self._snapshot_history.setdefault(index_id, [])
        history.append(summary)
        if len(history) > 10:
            self._snapshot_history[index_id] = history[-10:]

    def _push_pcr(self, index_id: str, pcr: float) -> None:
        history = self._pcr_history.setdefault(index_id, [])
        history.append(pcr)
        if len(history) > 10:
            self._pcr_history[index_id] = history[-10:]

    # ------------------------------------------------------------------
    # OI spike detection
    # ------------------------------------------------------------------

    def detect_oi_spike(
        self,
        index_id: str,
        current_snapshot: OISummary,
        previous_snapshot: OISummary,
        baselines: OIBaselines,
        current_chain: Optional[OptionsChainData] = None,
        previous_chain: Optional[OptionsChainData] = None,
        is_expiry: bool = False,
    ) -> list[OIAnomaly]:
        """Detect unusually large OI changes between consecutive snapshots."""
        results: list[OIAnomaly] = []
        now = datetime.now(tz=_IST)
        threshold_mult = 2.0 if is_expiry else 1.0

        current_total = current_snapshot.total_ce_oi + current_snapshot.total_pe_oi
        previous_total = previous_snapshot.total_ce_oi + previous_snapshot.total_pe_oi
        change = abs(current_total - previous_total)

        # --- Check 1: Total OI change z-score ---
        z = _z_score(change, baselines.avg_oi_change_per_3min, baselines.std_oi_change_per_3min)

        if z > self.thresholds["oi_spike_zscore_high"] * threshold_mult:
            results.append(OIAnomaly(
                index_id=index_id,
                timestamp=now,
                anomaly_type="MASSIVE_OI_SPIKE",
                severity="HIGH",
                total_oi_change=change,
                oi_change_zscore=round(z, 2),
                directional_implication=_oi_direction(current_snapshot, previous_snapshot),
                details={"z_score": round(z, 2), "threshold_mult": threshold_mult},
                message=(
                    f"{index_id}: Massive OI spike — {change:,} contracts changed "
                    f"(z={z:.1f}), possible large institutional activity"
                ),
            ))
        elif z > self.thresholds["oi_spike_zscore_medium"] * threshold_mult:
            results.append(OIAnomaly(
                index_id=index_id,
                timestamp=now,
                anomaly_type="OI_SPIKE",
                severity="MEDIUM",
                total_oi_change=change,
                oi_change_zscore=round(z, 2),
                directional_implication=_oi_direction(current_snapshot, previous_snapshot),
                details={"z_score": round(z, 2), "threshold_mult": threshold_mult},
                message=(
                    f"{index_id}: OI spike — {change:,} contracts changed (z={z:.1f})"
                ),
            ))

        # --- Check 2: Concentrated OI spike at single strike ---
        if current_chain is not None and previous_chain is not None:
            prev_oi_map = _build_strike_oi_map(previous_chain)
            spot = current_chain.spot_price
            atm_range = _atm_strike_range(current_chain, n=2)

            for strike in current_chain.strikes:
                sp = strike.strike_price
                prev = prev_oi_map.get(sp, {})

                for otype, cur_oi, prev_oi_val in [
                    ("CE", strike.ce_oi, prev.get("ce_oi", 0)),
                    ("PE", strike.pe_oi, prev.get("pe_oi", 0)),
                ]:
                    if prev_oi_val < _MIN_STRIKE_OI:
                        continue
                    oi_change_abs = abs(cur_oi - prev_oi_val)
                    oi_change_pct = (oi_change_abs / prev_oi_val) * 100
                    if oi_change_pct > self.thresholds["oi_concentration_change_pct"] / threshold_mult:
                        near_atm = sp in atm_range
                        sev = "HIGH" if near_atm else "MEDIUM"
                        results.append(OIAnomaly(
                            index_id=index_id,
                            timestamp=now,
                            anomaly_type="STRIKE_OI_CONCENTRATION",
                            severity=sev,
                            total_oi_change=oi_change_abs,
                            affected_strikes=[{
                                "strike": sp,
                                "option_type": otype,
                                "oi_change": cur_oi - prev_oi_val,
                                "oi_change_pct": round(oi_change_pct, 2),
                                "near_atm": near_atm,
                            }],
                            directional_implication=(
                                "BEARISH" if otype == "CE" else "BULLISH"
                            ),
                            details={"spot": spot, "near_atm": near_atm},
                            message=(
                                f"{index_id}: Concentrated OI {'buildup' if cur_oi > prev_oi_val else 'unwinding'} "
                                f"at {sp} {otype} ({oi_change_pct:.0f}% change)"
                                f"{' — near ATM' if near_atm else ''}"
                            ),
                        ))

        # --- Check 3: One-sided OI buildup ---
        ce_change = abs(current_snapshot.total_ce_oi - previous_snapshot.total_ce_oi)
        pe_change = abs(current_snapshot.total_pe_oi - previous_snapshot.total_pe_oi)

        if ce_change > 0 and pe_change > 0:
            if ce_change > self.thresholds["oi_one_sided_ratio"] * pe_change:
                results.append(OIAnomaly(
                    index_id=index_id,
                    timestamp=now,
                    anomaly_type="ONE_SIDED_CE_BUILDUP",
                    severity="MEDIUM",
                    total_oi_change=ce_change,
                    details={
                        "ce_oi_change": ce_change,
                        "pe_oi_change": pe_change,
                        "ratio": round(ce_change / pe_change, 1),
                    },
                    directional_implication="BEARISH",
                    message=(
                        f"{index_id}: One-sided CE OI buildup — CE change {ce_change:,} vs "
                        f"PE change {pe_change:,} ({ce_change / pe_change:.1f}x). "
                        f"Heavy CE writing = bearish signal"
                    ),
                ))
            elif pe_change > self.thresholds["oi_one_sided_ratio"] * ce_change:
                results.append(OIAnomaly(
                    index_id=index_id,
                    timestamp=now,
                    anomaly_type="ONE_SIDED_PE_BUILDUP",
                    severity="MEDIUM",
                    total_oi_change=pe_change,
                    details={
                        "ce_oi_change": ce_change,
                        "pe_oi_change": pe_change,
                        "ratio": round(pe_change / ce_change, 1),
                    },
                    directional_implication="BULLISH",
                    message=(
                        f"{index_id}: One-sided PE OI buildup — PE change {pe_change:,} vs "
                        f"CE change {ce_change:,} ({pe_change / ce_change:.1f}x). "
                        f"Heavy PE writing = bullish signal"
                    ),
                ))
        elif ce_change > 0 and pe_change == 0:
            results.append(OIAnomaly(
                index_id=index_id,
                timestamp=now,
                anomaly_type="ONE_SIDED_CE_BUILDUP",
                severity="MEDIUM",
                total_oi_change=ce_change,
                details={"ce_oi_change": ce_change, "pe_oi_change": 0, "ratio": float("inf")},
                directional_implication="BEARISH",
                message=(
                    f"{index_id}: One-sided CE OI buildup — CE change {ce_change:,}, "
                    f"PE unchanged. Heavy CE writing = bearish signal"
                ),
            ))
        elif pe_change > 0 and ce_change == 0:
            results.append(OIAnomaly(
                index_id=index_id,
                timestamp=now,
                anomaly_type="ONE_SIDED_PE_BUILDUP",
                severity="MEDIUM",
                total_oi_change=pe_change,
                details={"ce_oi_change": 0, "pe_oi_change": pe_change, "ratio": float("inf")},
                directional_implication="BULLISH",
                message=(
                    f"{index_id}: One-sided PE OI buildup — PE change {pe_change:,}, "
                    f"CE unchanged. Heavy PE writing = bullish signal"
                ),
            ))

        return results

    # ------------------------------------------------------------------
    # PCR anomaly detection
    # ------------------------------------------------------------------

    def detect_pcr_anomaly(
        self,
        index_id: str,
        current_pcr: float,
        pcr_history: list[float],
        baselines: OIBaselines,
    ) -> list[OIAnomaly]:
        """Detect unusual PCR shifts."""
        results: list[OIAnomaly] = []
        now = datetime.now(tz=_IST)

        if baselines.std_pcr <= 0:
            return results

        z = _z_score(current_pcr, baselines.avg_pcr, baselines.std_pcr)

        # --- Check 1: PCR z-score extremes ---
        if z > self.thresholds["pcr_zscore_extreme"]:
            results.append(OIAnomaly(
                index_id=index_id,
                timestamp=now,
                anomaly_type="PCR_EXTREME_HIGH",
                severity="MEDIUM",
                details={
                    "pcr": round(current_pcr, 3),
                    "avg_pcr": round(baselines.avg_pcr, 3),
                    "z_score": round(z, 2),
                },
                directional_implication="BULLISH",
                message=(
                    f"{index_id}: PCR extremely high at {current_pcr:.2f} "
                    f"(z={z:.1f} vs 20d avg {baselines.avg_pcr:.2f}) — "
                    f"heavy put writing, bullish sentiment"
                ),
            ))
        elif z < -self.thresholds["pcr_zscore_extreme"]:
            results.append(OIAnomaly(
                index_id=index_id,
                timestamp=now,
                anomaly_type="PCR_EXTREME_LOW",
                severity="MEDIUM",
                details={
                    "pcr": round(current_pcr, 3),
                    "avg_pcr": round(baselines.avg_pcr, 3),
                    "z_score": round(z, 2),
                },
                directional_implication="BEARISH",
                message=(
                    f"{index_id}: PCR extremely low at {current_pcr:.2f} "
                    f"(z={z:.1f} vs 20d avg {baselines.avg_pcr:.2f}) — "
                    f"heavy call writing, bearish sentiment"
                ),
            ))

        # --- Check 2: PCR rapid shift (>0.15 within last 10 snapshots / 30 min) ---
        if len(pcr_history) >= 2:
            oldest_in_window = pcr_history[0]
            pcr_shift = abs(current_pcr - oldest_in_window)
            if pcr_shift > self.thresholds["pcr_rapid_shift_threshold"]:
                direction = "BULLISH" if current_pcr > oldest_in_window else "BEARISH"
                results.append(OIAnomaly(
                    index_id=index_id,
                    timestamp=now,
                    anomaly_type="PCR_RAPID_SHIFT",
                    severity="HIGH",
                    details={
                        "pcr": round(current_pcr, 3),
                        "pcr_30min_ago": round(oldest_in_window, 3),
                        "shift": round(pcr_shift, 3),
                    },
                    directional_implication=direction,
                    message=(
                        f"{index_id}: PCR rapid shift — moved {pcr_shift:.2f} in ~30 min "
                        f"({oldest_in_window:.2f} → {current_pcr:.2f}). "
                        f"Aggressive one-sided positioning"
                    ),
                ))

        # --- Check 3: PCR at historical extremes ---
        if current_pcr > self.thresholds["pcr_absolute_high"]:
            results.append(OIAnomaly(
                index_id=index_id,
                timestamp=now,
                anomaly_type="PCR_EXTREME",
                severity="MEDIUM",
                details={"pcr": round(current_pcr, 3), "threshold": self.thresholds["pcr_absolute_high"]},
                directional_implication="BULLISH",
                message=(
                    f"{index_id}: PCR at extreme high ({current_pcr:.2f} > {self.thresholds['pcr_absolute_high']}) — "
                    f"contrarian signal: excessive put buying may indicate bottom"
                ),
            ))
        elif current_pcr < self.thresholds["pcr_absolute_low"]:
            results.append(OIAnomaly(
                index_id=index_id,
                timestamp=now,
                anomaly_type="PCR_EXTREME",
                severity="MEDIUM",
                details={"pcr": round(current_pcr, 3), "threshold": self.thresholds["pcr_absolute_low"]},
                directional_implication="BEARISH",
                message=(
                    f"{index_id}: PCR at extreme low ({current_pcr:.2f} < {self.thresholds['pcr_absolute_low']}) — "
                    f"contrarian signal: excessive call buying may indicate top"
                ),
            ))

        return results

    # ------------------------------------------------------------------
    # Max pain anomaly detection
    # ------------------------------------------------------------------

    def detect_max_pain_anomaly(
        self,
        index_id: str,
        current_max_pain: float,
        previous_max_pain: float,
        spot_price: float,
        days_to_expiry: int = 0,
    ) -> list[OIAnomaly]:
        """Detect max pain jumps and spot-max-pain divergence."""
        results: list[OIAnomaly] = []
        now = datetime.now(tz=_IST)

        if previous_max_pain <= 0 or spot_price <= 0:
            return results

        # --- Check 1: Max pain jump (> X% shift in session) ---
        mp_shift_pct = abs(current_max_pain - previous_max_pain) / previous_max_pain * 100
        if mp_shift_pct > self.thresholds["max_pain_jump_pct"]:
            results.append(OIAnomaly(
                index_id=index_id,
                timestamp=now,
                anomaly_type="MAX_PAIN_JUMP",
                severity="HIGH",
                details={
                    "current_max_pain": current_max_pain,
                    "previous_max_pain": previous_max_pain,
                    "shift_pct": round(mp_shift_pct, 2),
                },
                directional_implication=(
                    "BULLISH" if current_max_pain > previous_max_pain else "BEARISH"
                ),
                message=(
                    f"{index_id}: Max pain jumped {mp_shift_pct:.1f}% "
                    f"({previous_max_pain:.0f} → {current_max_pain:.0f}) — "
                    f"massive OI restructuring underway"
                ),
            ))

        # --- Check 2: Spot vs max pain divergence (> X%) ---
        divergence_pct = abs(spot_price - current_max_pain) / current_max_pain * 100
        if divergence_pct > self.thresholds["spot_max_pain_diverge_pct"]:
            # Convergence probability increases closer to expiry
            if days_to_expiry <= 0:
                convergence_prob = 0.9
            elif days_to_expiry <= 2:
                convergence_prob = 0.75
            elif days_to_expiry <= 5:
                convergence_prob = 0.5
            else:
                convergence_prob = 0.3

            pull_direction = "DOWN" if spot_price > current_max_pain else "UP"
            results.append(OIAnomaly(
                index_id=index_id,
                timestamp=now,
                anomaly_type="SPOT_MAX_PAIN_DIVERGENCE",
                severity="MEDIUM" if days_to_expiry > 2 else "HIGH",
                details={
                    "spot": spot_price,
                    "max_pain": current_max_pain,
                    "divergence_pct": round(divergence_pct, 2),
                    "days_to_expiry": days_to_expiry,
                    "convergence_probability": convergence_prob,
                    "pull_direction": pull_direction,
                },
                directional_implication=(
                    "BEARISH" if pull_direction == "DOWN" else "BULLISH"
                ),
                message=(
                    f"{index_id}: Spot ({spot_price:.0f}) diverged {divergence_pct:.1f}% "
                    f"from max pain ({current_max_pain:.0f}). "
                    f"Gravitational pull {pull_direction} — "
                    f"convergence probability {convergence_prob:.0%} "
                    f"({days_to_expiry}d to expiry)"
                ),
            ))

        return results

    # ------------------------------------------------------------------
    # IV anomaly detection
    # ------------------------------------------------------------------

    def detect_iv_anomaly(
        self,
        index_id: str,
        current_chain: OptionsChainData,
        baselines: OIBaselines,
    ) -> list[OIAnomaly]:
        """Detect IV anomalies: crush, explosion, skew, term structure."""
        results: list[OIAnomaly] = []
        now = datetime.now(tz=_IST)

        atm_strike = _find_atm_strike(current_chain)
        if atm_strike is None:
            return results

        baseline_min, baseline_max = baselines.typical_iv_range
        baseline_mid = (baseline_min + baseline_max) / 2 if baseline_max > 0 else 0.0

        # Collect ATM IVs
        atm_ce_iv = _safe_float(atm_strike.ce_iv)
        atm_pe_iv = _safe_float(atm_strike.pe_iv)

        if atm_ce_iv <= 0 and atm_pe_iv <= 0:
            return results

        atm_iv = (atm_ce_iv + atm_pe_iv) / 2 if atm_ce_iv > 0 and atm_pe_iv > 0 else max(atm_ce_iv, atm_pe_iv)

        if baseline_mid > 0:
            # --- Check 1: IV crush (ATM IV dropped > X% from baseline) ---
            iv_drop_pct = (baseline_mid - atm_iv) / baseline_mid * 100
            if iv_drop_pct > self.thresholds["iv_crush_pct"]:
                results.append(OIAnomaly(
                    index_id=index_id,
                    timestamp=now,
                    anomaly_type="IV_CRUSH",
                    severity="MEDIUM",
                    details={
                        "atm_iv": round(atm_iv, 2),
                        "baseline_iv_mid": round(baseline_mid, 2),
                        "iv_drop_pct": round(iv_drop_pct, 2),
                    },
                    directional_implication="UNCLEAR",
                    message=(
                        f"{index_id}: IV crush — ATM IV at {atm_iv:.1f}% "
                        f"(dropped {iv_drop_pct:.0f}% from baseline {baseline_mid:.1f}%). "
                        f"Options becoming cheaper, event uncertainty resolved"
                    ),
                ))

            # --- Check 2: IV explosion (ATM IV spiked > X% above baseline) ---
            iv_spike_pct = (atm_iv - baseline_mid) / baseline_mid * 100
            if iv_spike_pct > self.thresholds["iv_explosion_pct"]:
                results.append(OIAnomaly(
                    index_id=index_id,
                    timestamp=now,
                    anomaly_type="IV_EXPLOSION",
                    severity="HIGH",
                    details={
                        "atm_iv": round(atm_iv, 2),
                        "baseline_iv_mid": round(baseline_mid, 2),
                        "iv_spike_pct": round(iv_spike_pct, 2),
                    },
                    directional_implication="UNCLEAR",
                    message=(
                        f"{index_id}: IV explosion — ATM IV at {atm_iv:.1f}% "
                        f"(spiked {iv_spike_pct:.0f}% above baseline {baseline_mid:.1f}%). "
                        f"Fear/uncertainty surging, options expensive"
                    ),
                ))

        # --- Check 3: IV skew anomaly (put IV >> call IV) ---
        if atm_ce_iv > 0 and atm_pe_iv > 0:
            skew_ratio = atm_pe_iv / atm_ce_iv
            # Normal skew is ~1.0-1.3; extreme is > X
            if skew_ratio > self.thresholds["iv_skew_extreme_multiplier"]:
                results.append(OIAnomaly(
                    index_id=index_id,
                    timestamp=now,
                    anomaly_type="IV_SKEW_EXTREME",
                    severity="HIGH",
                    details={
                        "ce_iv": round(atm_ce_iv, 2),
                        "pe_iv": round(atm_pe_iv, 2),
                        "skew_ratio": round(skew_ratio, 2),
                    },
                    directional_implication="BEARISH",
                    message=(
                        f"{index_id}: Extreme IV skew — put IV ({atm_pe_iv:.1f}%) is "
                        f"{skew_ratio:.1f}x call IV ({atm_ce_iv:.1f}%). "
                        f"Market aggressively buying downside protection"
                    ),
                ))

        # --- Check 4: IV term structure inversion ---
        # Compare near-expiry ATM IV to average OTM IV (proxy for far expiry)
        otm_ivs = _collect_otm_ivs(current_chain, n_strikes=5)
        if otm_ivs and atm_iv > 0:
            avg_otm_iv = sum(otm_ivs) / len(otm_ivs)
            if avg_otm_iv > 0 and atm_iv > avg_otm_iv * 1.3:
                results.append(OIAnomaly(
                    index_id=index_id,
                    timestamp=now,
                    anomaly_type="IV_TERM_INVERSION",
                    severity="MEDIUM",
                    details={
                        "atm_iv": round(atm_iv, 2),
                        "avg_otm_iv": round(avg_otm_iv, 2),
                        "ratio": round(atm_iv / avg_otm_iv, 2),
                    },
                    directional_implication="UNCLEAR",
                    message=(
                        f"{index_id}: IV term structure inversion — ATM IV ({atm_iv:.1f}%) "
                        f"significantly above OTM average ({avg_otm_iv:.1f}%). "
                        f"Near-term risk perceived higher than long-term"
                    ),
                ))

        return results

    # ------------------------------------------------------------------
    # Unified detection
    # ------------------------------------------------------------------

    def detect_all_oi_anomalies(
        self,
        index_id: str,
        current_chain: Optional[OptionsChainData],
        current_summary: OISummary,
        previous_summary: Optional[OISummary],
        baselines: OIBaselines,
        previous_chain: Optional[OptionsChainData] = None,
        expiry_date: Optional[date] = None,
        days_to_expiry: int = 0,
    ) -> list[AnomalyEvent]:
        """Run ALL OI/options checks, apply cooldowns, return unified events."""
        anomalies: list[OIAnomaly] = []
        today = datetime.now(tz=_IST).date()
        is_expiry = _is_expiry_day(expiry_date, today) if expiry_date else False

        # Update snapshot/PCR history
        self._push_snapshot(index_id, current_summary)
        self._push_pcr(index_id, current_summary.pcr)

        # 1. OI spike detection (needs previous snapshot)
        if previous_summary is not None:
            anomalies.extend(self.detect_oi_spike(
                index_id, current_summary, previous_summary, baselines,
                current_chain=current_chain,
                previous_chain=previous_chain,
                is_expiry=is_expiry,
            ))

        # 2. PCR anomaly detection
        pcr_history = self._pcr_history.get(index_id, [])
        anomalies.extend(self.detect_pcr_anomaly(
            index_id, current_summary.pcr, pcr_history, baselines,
        ))

        # 3. Max pain anomaly detection
        if previous_summary is not None:
            spot = current_chain.spot_price if current_chain else 0.0
            anomalies.extend(self.detect_max_pain_anomaly(
                index_id,
                current_summary.max_pain_strike,
                previous_summary.max_pain_strike,
                spot,
                days_to_expiry=days_to_expiry,
            ))

        # 4. IV anomaly detection
        if current_chain is not None:
            anomalies.extend(self.detect_iv_anomaly(
                index_id, current_chain, baselines,
            ))

        # Convert to unified AnomalyEvent and apply cooldowns
        events: list[AnomalyEvent] = []
        for a in anomalies:
            ev = _to_anomaly_event(a)
            if not self._is_on_cooldown(ev):
                events.append(ev)
                self._set_cooldown(ev)

        # Track active anomalies
        self._active_anomalies[index_id] = events

        return events

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_oi_anomaly_summary(self, index_id: str) -> OIAnomalySummary:
        """Summary of all active OI anomalies for an index."""
        active = self._active_anomalies.get(index_id, [])

        if not active:
            return OIAnomalySummary(
                index_id=index_id,
                message=f"{index_id}: No active OI anomalies",
            )

        # Determine dominant signal from directional implications
        bullish = 0
        bearish = 0
        key_strikes: set[float] = set()

        for ev in active:
            details = json.loads(ev.details) if isinstance(ev.details, str) else ev.details
            impl = details.get("directional_implication", "UNCLEAR")
            if impl == "BULLISH":
                bullish += 1
            elif impl == "BEARISH":
                bearish += 1

            # Collect affected strikes
            for s in details.get("affected_strikes", []):
                if "strike" in s:
                    key_strikes.add(s["strike"])

        if bullish > bearish:
            dominant = "BULLISH"
        elif bearish > bullish:
            dominant = "BEARISH"
        else:
            dominant = "NEUTRAL"

        # Intensity based on count and severity
        high_count = sum(1 for e in active if e.severity == "HIGH")
        total = len(active)

        if high_count >= 3 or total >= 6:
            intensity = "EXTREME"
        elif high_count >= 2 or total >= 4:
            intensity = "HIGH"
        elif high_count >= 1 or total >= 2:
            intensity = "MODERATE"
        elif total >= 1:
            intensity = "LOW"
        else:
            intensity = "NONE"

        types = [e.anomaly_type for e in active]
        return OIAnomalySummary(
            index_id=index_id,
            active_anomalies=active,
            dominant_signal=dominant,
            anomaly_intensity=intensity,
            key_strikes_to_watch=sorted(key_strikes),
            message=(
                f"{index_id}: {total} active OI anomalies "
                f"({', '.join(types)}). "
                f"Dominant signal: {dominant}, Intensity: {intensity}"
            ),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_anomalies(self, anomalies: list[AnomalyEvent]) -> int:
        """Persist anomaly events to the database with DB-level dedup."""
        inserted = 0
        for ev in anomalies:
            row = self.db.fetch_one(
                Q.GET_LATEST_ANOMALY_BY_KEY,
                (ev.index_id, ev.anomaly_type),
            )
            if row is not None:
                last_ts = pd.Timestamp(row["timestamp"])
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize(_IST)
                cooldown_min = self._get_cooldown_minutes(ev.severity)
                if (ev.timestamp - last_ts.to_pydatetime()) < timedelta(minutes=cooldown_min):
                    continue

            self.db.execute(
                Q.INSERT_ANOMALY_EVENT,
                (
                    ev.index_id,
                    ev.timestamp.isoformat(),
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
        cooldown_min = self._get_cooldown_minutes(event.severity)
        return (event.timestamp - last) < timedelta(minutes=cooldown_min)

    def _get_cooldown_minutes(self, severity: str) -> float:
        """Helper to get cooldown minutes from thresholds."""
        if severity == "HIGH":
            return self.thresholds.get("cooldown_high_seconds", 600) / 60
        if severity == "MEDIUM":
            return self.thresholds.get("cooldown_medium_seconds", 900) / 60
        return self.thresholds.get("cooldown_low_seconds", 1800) / 60

    def _set_cooldown(self, event: AnomalyEvent) -> None:
        self._cooldowns[event.cooldown_key] = event.timestamp

    def clear_expired_cooldowns(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.now(tz=_IST)
        max_window = timedelta(minutes=30)
        expired = [
            k for k, v in self._cooldowns.items() if (now - v) >= max_window
        ]
        for k in expired:
            del self._cooldowns[k]
        return len(expired)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _oi_direction(current: OISummary, previous: OISummary) -> str:
    """Infer directional implication from OI changes."""
    ce_change = current.total_ce_oi - previous.total_ce_oi
    pe_change = current.total_pe_oi - previous.total_pe_oi

    # Heavy CE writing (OI up) = bearish; heavy PE writing = bullish
    if abs(ce_change) > 2 * abs(pe_change) and ce_change > 0:
        return "BEARISH"
    if abs(pe_change) > 2 * abs(ce_change) and pe_change > 0:
        return "BULLISH"
    return "UNCLEAR"


def _build_strike_oi_map(chain: OptionsChainData) -> dict[float, dict]:
    """Build {strike_price: {ce_oi, pe_oi}} from a chain."""
    return {
        s.strike_price: {"ce_oi": s.ce_oi, "pe_oi": s.pe_oi}
        for s in chain.strikes
    }


def _atm_strike_range(chain: OptionsChainData, n: int = 2) -> set[float]:
    """Return the set of strike prices within *n* strikes of ATM."""
    spot = chain.spot_price
    strikes_sorted = sorted(s.strike_price for s in chain.strikes)
    if not strikes_sorted:
        return set()

    # Find nearest strike to spot
    atm_idx = min(range(len(strikes_sorted)), key=lambda i: abs(strikes_sorted[i] - spot))
    lo = max(0, atm_idx - n)
    hi = min(len(strikes_sorted), atm_idx + n + 1)
    return set(strikes_sorted[lo:hi])


def _find_atm_strike(chain: OptionsChainData) -> Optional[OptionStrike]:
    """Return the OptionStrike closest to spot."""
    if not chain.strikes:
        return None
    spot = chain.spot_price
    return min(chain.strikes, key=lambda s: abs(s.strike_price - spot))


def _collect_otm_ivs(chain: OptionsChainData, n_strikes: int = 5) -> list[float]:
    """Collect IVs from OTM strikes (far from ATM) as a proxy for term structure."""
    spot = chain.spot_price
    strikes_sorted = sorted(chain.strikes, key=lambda s: abs(s.strike_price - spot))

    # Skip ATM (first few) and collect from farther OTM strikes
    otm_ivs: list[float] = []
    for s in strikes_sorted[3:]:  # skip 3 nearest-ATM strikes
        if len(otm_ivs) >= n_strikes:
            break
        if s.strike_price > spot and s.ce_iv > 0:
            otm_ivs.append(s.ce_iv)
        elif s.strike_price < spot and s.pe_iv > 0:
            otm_ivs.append(s.pe_iv)
    return otm_ivs


def _to_anomaly_event(anomaly: OIAnomaly) -> AnomalyEvent:
    """Convert an OIAnomaly into a unified AnomalyEvent."""
    details_dict = {
        "total_oi_change": anomaly.total_oi_change,
        "oi_change_zscore": anomaly.oi_change_zscore,
        "affected_strikes": anomaly.affected_strikes,
        "directional_implication": anomaly.directional_implication,
        **anomaly.details,
    }
    return AnomalyEvent(
        index_id=anomaly.index_id,
        timestamp=anomaly.timestamp,
        anomaly_type=anomaly.anomaly_type,
        severity=anomaly.severity,
        category="OI",
        details=json.dumps(details_dict),
        message=anomaly.message,
        is_active=True,
        cooldown_key=f"{anomaly.index_id}_{anomaly.anomaly_type}",
    )
