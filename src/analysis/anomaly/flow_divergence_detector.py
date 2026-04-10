"""
Phase 4 -- Step 4.3: FII/DII Flow Anomalies & Cross-Index Divergence Detection.

Detects anomalous patterns in institutional (FII/DII) flows and identifies
divergences between correlated index pairs that may signal sector rotation,
risk-off behaviour, or structural market shifts.

Emits unified ``AnomalyEvent`` objects compatible with Steps 4.1 and 4.2.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from src.analysis.anomaly.volume_price_detector import AnomalyEvent

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Severity cooldown windows (minutes) -- consistent with Steps 4.1/4.2
# ---------------------------------------------------------------------------
_COOLDOWN_MINUTES = {
    "HIGH": 10,
    "MEDIUM": 15,
    "LOW": 30,
}

# Divergence checks run less frequently — once per 30 minutes
_DIVERGENCE_COOLDOWN_MINUTES = 30


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


def _fmt_crores(val: float) -> str:
    """Format a ₹ crore value with sign and commas."""
    sign = "+" if val >= 0 else ""
    return f"{sign}₹{val:,.0f} Cr"


# ═══════════════════════════════════════════════════════════════════════════
# FII Flow Detector
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FlowBaselines:
    """Rolling statistical baselines for FII/DII institutional flows."""

    avg_fii_net_daily: float = 0.0
    std_fii_net_daily: float = 0.0
    avg_dii_net_daily: float = 0.0
    std_dii_net_daily: float = 0.0
    fii_trend_5d: float = 0.0
    fii_trend_10d: float = 0.0
    dii_trend_5d: float = 0.0
    consecutive_fii_buy_days: int = 0
    consecutive_fii_sell_days: int = 0
    avg_fii_fo_net: float = 0.0
    std_fii_fo_net: float = 0.0


@dataclass
class FIIBias:
    """Current FII directional bias summary."""

    bias: str = "NEUTRAL"                  # STRONG_BUYING / BUYING / NEUTRAL / SELLING / STRONG_SELLING
    confidence: float = 0.0
    net_5d: float = 0.0
    net_10d: float = 0.0
    today_net: float = 0.0
    trend_direction: str = "STABLE"        # ACCELERATING / STABLE / REVERSING
    fo_signal: str = "NEUTRAL"             # BULLISH / BEARISH / NEUTRAL
    message: str = ""


class FIIFlowDetector:
    """Detects anomalous FII/DII institutional flow patterns.

    Loads the last 30 calendar days of FII/DII data from the
    ``fii_dii_activity`` table and computes rolling baselines.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self._baselines: Optional[FlowBaselines] = None
        self._daily_data: pd.DataFrame = pd.DataFrame()
        self._fo_data: pd.DataFrame = pd.DataFrame()
        self._cooldowns: dict[str, datetime] = {}
        self._load_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load last 30 days of FII/DII data and split by segment."""
        start = (date.today() - timedelta(days=30)).isoformat()
        rows = self.db.fetch_all(Q.LIST_FII_DII_RECENT, (start,))
        if not rows:
            return

        df = pd.DataFrame(rows)
        # Build a per-date aggregated cash view
        cash_df = df[df["segment"] == "CASH"].copy()
        if not cash_df.empty:
            pivot = cash_df.pivot_table(
                index="date", columns="category", values="net_value",
                aggfunc="sum",
            ).reset_index()
            pivot.columns.name = None
            pivot = pivot.rename(columns={"FII": "fii_net", "DII": "dii_net"})
            for col in ("fii_net", "dii_net"):
                if col not in pivot.columns:
                    pivot[col] = 0.0
            pivot = pivot.sort_values("date", ascending=True).reset_index(drop=True)
            self._daily_data = pivot

        # F&O segment
        fo_df = df[df["segment"] == "FO"].copy()
        if not fo_df.empty:
            fo_pivot = fo_df.pivot_table(
                index="date", columns="category", values="net_value",
                aggfunc="sum",
            ).reset_index()
            fo_pivot.columns.name = None
            fo_pivot = fo_pivot.rename(columns={"FII": "fii_fo_net"})
            if "fii_fo_net" not in fo_pivot.columns:
                fo_pivot["fii_fo_net"] = 0.0
            fo_pivot = fo_pivot.sort_values("date", ascending=True).reset_index(drop=True)
            self._fo_data = fo_pivot

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def compute_flow_baselines(self) -> FlowBaselines:
        """Compute statistical baselines from loaded historical data."""
        bl = FlowBaselines()

        df = self._daily_data
        if df.empty or len(df) < 5:
            self._baselines = bl
            return bl

        fii_nets = df["fii_net"].astype(float)
        dii_nets = df["dii_net"].astype(float)

        bl.avg_fii_net_daily = float(fii_nets.mean())
        bl.std_fii_net_daily = float(fii_nets.std(ddof=1)) if len(fii_nets) > 1 else 0.0
        bl.avg_dii_net_daily = float(dii_nets.mean())
        bl.std_dii_net_daily = float(dii_nets.std(ddof=1)) if len(dii_nets) > 1 else 0.0

        # Trends — last 5/10 days cumulative
        bl.fii_trend_5d = float(fii_nets.tail(5).sum())
        bl.fii_trend_10d = float(fii_nets.tail(10).sum())
        bl.dii_trend_5d = float(dii_nets.tail(5).sum())

        # Consecutive buy/sell days (from most recent day backwards)
        buy_days = 0
        sell_days = 0
        for val in reversed(fii_nets.tolist()):
            if val > 0 and sell_days == 0:
                buy_days += 1
            elif val < 0 and buy_days == 0:
                sell_days += 1
            else:
                break
        bl.consecutive_fii_buy_days = buy_days
        bl.consecutive_fii_sell_days = sell_days

        # F&O baselines
        if not self._fo_data.empty and "fii_fo_net" in self._fo_data.columns:
            fo_vals = self._fo_data["fii_fo_net"].astype(float)
            bl.avg_fii_fo_net = float(fo_vals.mean())
            bl.std_fii_fo_net = float(fo_vals.std(ddof=1)) if len(fo_vals) > 1 else 0.0

        self._baselines = bl
        return bl

    # ------------------------------------------------------------------
    # Cooldown management
    # ------------------------------------------------------------------

    def _is_on_cooldown(self, key: str, severity: str) -> bool:
        last = self._cooldowns.get(key)
        if last is None:
            return False
        minutes = _COOLDOWN_MINUTES.get(severity, 15)
        return datetime.now(tz=_IST) - last < timedelta(minutes=minutes)

    def _set_cooldown(self, key: str) -> None:
        self._cooldowns[key] = datetime.now(tz=_IST)

    # ------------------------------------------------------------------
    # Detection — FII anomalies
    # ------------------------------------------------------------------

    def detect_fii_anomaly(
        self,
        today_data: "FIIDIIData",
        baselines: FlowBaselines,
    ) -> list[AnomalyEvent]:
        """Run all FII flow anomaly checks against today's data.

        Parameters
        ----------
        today_data : FIIDIIData
            Today's FII/DII activity (from ``src.data.fii_dii_data``).
        baselines : FlowBaselines
            Pre-computed flow baselines.

        Returns
        -------
        list[AnomalyEvent]
            Detected anomalies in unified event format.
        """
        from src.data.fii_dii_data import FIIDIIData  # noqa: F811 — type reference

        events: list[AnomalyEvent] = []
        now = datetime.now(tz=_IST)

        if today_data is None:
            logger.info(
                "FII anomaly detection: no FII/DII data available for today. "
                "Data is published after market close (~6PM IST). "
                "FII analysis will activate once data collection begins."
            )
            return []

        if baselines is None or baselines.avg_fii_net_daily == 0:
            logger.info(
                "FII anomaly detection: insufficient historical FII data for baselines. "
                "Need 10+ trading days of FII/DII data. Currently accumulating."
            )
            return []

        # ── 1. FII flow Z-score ──────────────────────────────────────
        fii_z = _z_score(today_data.fii_net_value, baselines.avg_fii_net_daily,
                         baselines.std_fii_net_daily)
        if abs(fii_z) > 2.0:
            if abs(fii_z) > 3.0:
                severity = "HIGH"
                atype = "FII_EXTREME_FLOW"
            else:
                severity = "MEDIUM"
                atype = "FII_UNUSUAL_FLOW"

            direction = "buying" if today_data.fii_net_value > 0 else "selling"
            implication = "bullish" if today_data.fii_net_value > 0 else "bearish"
            cooldown_key = f"MARKET_{atype}"
            if not self._is_on_cooldown(cooldown_key, severity):
                events.append(AnomalyEvent(
                    index_id="MARKET",
                    timestamp=now,
                    anomaly_type=atype,
                    severity=severity,
                    category="FII_DII",
                    details=json.dumps({
                        "fii_net": today_data.fii_net_value,
                        "z_score": round(fii_z, 2),
                        "avg": round(baselines.avg_fii_net_daily, 2),
                        "std": round(baselines.std_fii_net_daily, 2),
                        "direction": direction,
                    }),
                    message=(
                        f"FII {direction} {_fmt_crores(abs(today_data.fii_net_value))} today "
                        f"(Z-score: {fii_z:.1f}). Signal: {implication} for next session."
                    ),
                    cooldown_key=cooldown_key,
                ))
                self._set_cooldown(cooldown_key)

        # ── 2. FII trend reversal ────────────────────────────────────
        if (baselines.consecutive_fii_buy_days >= 5
                and today_data.fii_net_value < 0):
            cooldown_key = "MARKET_FII_TREND_REVERSAL"
            if not self._is_on_cooldown(cooldown_key, "HIGH"):
                events.append(AnomalyEvent(
                    index_id="MARKET",
                    timestamp=now,
                    anomaly_type="FII_TREND_REVERSAL",
                    severity="HIGH",
                    category="FII_DII",
                    details=json.dumps({
                        "consecutive_buy_days": baselines.consecutive_fii_buy_days,
                        "today_net": today_data.fii_net_value,
                        "reversal_direction": "BUY_TO_SELL",
                    }),
                    message=(
                        f"FII trend reversal: {baselines.consecutive_fii_buy_days} "
                        f"consecutive buying days broken. Today: "
                        f"{_fmt_crores(today_data.fii_net_value)}."
                    ),
                    cooldown_key=cooldown_key,
                ))
                self._set_cooldown(cooldown_key)

        elif (baselines.consecutive_fii_sell_days >= 5
              and today_data.fii_net_value > 0):
            cooldown_key = "MARKET_FII_TREND_REVERSAL"
            if not self._is_on_cooldown(cooldown_key, "HIGH"):
                events.append(AnomalyEvent(
                    index_id="MARKET",
                    timestamp=now,
                    anomaly_type="FII_TREND_REVERSAL",
                    severity="HIGH",
                    category="FII_DII",
                    details=json.dumps({
                        "consecutive_sell_days": baselines.consecutive_fii_sell_days,
                        "today_net": today_data.fii_net_value,
                        "reversal_direction": "SELL_TO_BUY",
                    }),
                    message=(
                        f"FII trend reversal: {baselines.consecutive_fii_sell_days} "
                        f"consecutive selling days broken. Today: "
                        f"{_fmt_crores(today_data.fii_net_value)}."
                    ),
                    cooldown_key=cooldown_key,
                ))
                self._set_cooldown(cooldown_key)

        # ── 3. FII-DII divergence ────────────────────────────────────
        fii_net = today_data.fii_net_value
        dii_net = today_data.dii_net_value
        if ((fii_net < -2000 and dii_net > 2000)
                or (fii_net > 2000 and dii_net < -2000)):
            cooldown_key = "MARKET_FII_DII_DIVERGENCE"
            if not self._is_on_cooldown(cooldown_key, "MEDIUM"):
                if fii_net < 0 and dii_net > 0:
                    msg = (
                        f"FII-DII divergence: FII selling {_fmt_crores(abs(fii_net))} "
                        f"but DII absorbing {_fmt_crores(dii_net)}. "
                        f"Market may hold but under institutional stress."
                    )
                else:
                    msg = (
                        f"FII-DII divergence: FII buying {_fmt_crores(fii_net)} "
                        f"but DII selling {_fmt_crores(abs(dii_net))}. "
                        f"Conflicting institutional signals."
                    )
                events.append(AnomalyEvent(
                    index_id="MARKET",
                    timestamp=now,
                    anomaly_type="FII_DII_DIVERGENCE",
                    severity="MEDIUM",
                    category="FII_DII",
                    details=json.dumps({
                        "fii_net": fii_net,
                        "dii_net": dii_net,
                        "abs_divergence": abs(fii_net) + abs(dii_net),
                    }),
                    message=msg,
                    cooldown_key=cooldown_key,
                ))
                self._set_cooldown(cooldown_key)

        # ── 4. FII F&O segment anomaly ───────────────────────────────
        if today_data.fii_fo_net != 0.0 and baselines.std_fii_fo_net > 0:
            fo_z = _z_score(today_data.fii_fo_net, baselines.avg_fii_fo_net,
                            baselines.std_fii_fo_net)
            if abs(fo_z) > 2.5:
                cooldown_key = "MARKET_FII_FO_EXTREME"
                direction = "long" if today_data.fii_fo_net > 0 else "short"
                if not self._is_on_cooldown(cooldown_key, "HIGH"):
                    events.append(AnomalyEvent(
                        index_id="MARKET",
                        timestamp=now,
                        anomaly_type="FII_FO_EXTREME",
                        severity="HIGH",
                        category="FII_DII",
                        details=json.dumps({
                            "fii_fo_net": today_data.fii_fo_net,
                            "z_score": round(fo_z, 2),
                            "avg": round(baselines.avg_fii_fo_net, 2),
                            "std": round(baselines.std_fii_fo_net, 2),
                            "direction": direction,
                        }),
                        message=(
                            f"FII F&O extreme: {_fmt_crores(today_data.fii_fo_net)} "
                            f"(Z-score: {fo_z:.1f}). Large FII {direction} positions "
                            f"in futures — {'bearish' if direction == 'short' else 'bullish'} signal."
                        ),
                        cooldown_key=cooldown_key,
                    ))
                    self._set_cooldown(cooldown_key)

        # ── 5. Cumulative flow shift ─────────────────────────────────
        if len(self._daily_data) >= 10:
            fii_vals = self._daily_data["fii_net"].astype(float)
            prev_5d = float(fii_vals.iloc[-10:-5].sum())
            curr_5d = float(fii_vals.tail(5).sum()) + today_data.fii_net_value
            # Sign changed and magnitude is meaningful
            if prev_5d != 0 and curr_5d != 0 and (prev_5d * curr_5d < 0):
                cooldown_key = "MARKET_FII_CUMULATIVE_SHIFT"
                if not self._is_on_cooldown(cooldown_key, "MEDIUM"):
                    events.append(AnomalyEvent(
                        index_id="MARKET",
                        timestamp=now,
                        anomaly_type="FII_CUMULATIVE_SHIFT",
                        severity="MEDIUM",
                        category="FII_DII",
                        details=json.dumps({
                            "prev_5d_cumulative": round(prev_5d, 2),
                            "curr_5d_cumulative": round(curr_5d, 2),
                        }),
                        message=(
                            f"FII cumulative flow shift: previous 5-day "
                            f"{_fmt_crores(prev_5d)} → current 5-day "
                            f"{_fmt_crores(curr_5d)}. Medium-term positioning changed."
                        ),
                        cooldown_key=cooldown_key,
                    ))
                    self._set_cooldown(cooldown_key)

        return events

    # ------------------------------------------------------------------
    # FII bias summary
    # ------------------------------------------------------------------

    def get_fii_bias(self, today_data: Optional["FIIDIIData"] = None) -> FIIBias:
        """Compute the current FII directional bias summary.

        Parameters
        ----------
        today_data : FIIDIIData, optional
            If supplied, incorporates today's data into the bias calculation.
        """
        bl = self._baselines or self.compute_flow_baselines()

        today_net = today_data.fii_net_value if today_data else 0.0
        net_5d = bl.fii_trend_5d
        net_10d = bl.fii_trend_10d

        # Determine bias from 5-day trend
        if net_5d > 5000:
            bias = "STRONG_BUYING"
            confidence = min(0.95, 0.6 + abs(net_5d) / 20000)
        elif net_5d > 1000:
            bias = "BUYING"
            confidence = min(0.8, 0.4 + abs(net_5d) / 10000)
        elif net_5d < -5000:
            bias = "STRONG_SELLING"
            confidence = min(0.95, 0.6 + abs(net_5d) / 20000)
        elif net_5d < -1000:
            bias = "SELLING"
            confidence = min(0.8, 0.4 + abs(net_5d) / 10000)
        else:
            bias = "NEUTRAL"
            confidence = 0.3

        # Trend direction — compare recent vs prior period
        if net_10d != 0:
            prior_5d = net_10d - net_5d
            if prior_5d != 0:
                ratio = net_5d / prior_5d if prior_5d != 0 else 0
                if abs(net_5d) > abs(prior_5d) * 1.3 and net_5d * prior_5d > 0:
                    trend_direction = "ACCELERATING"
                elif net_5d * prior_5d < 0:
                    trend_direction = "REVERSING"
                else:
                    trend_direction = "STABLE"
            else:
                trend_direction = "STABLE"
        else:
            trend_direction = "STABLE"

        # F&O signal
        fo_signal = "NEUTRAL"
        if bl.std_fii_fo_net > 0 and bl.avg_fii_fo_net != 0:
            if bl.avg_fii_fo_net > bl.std_fii_fo_net:
                fo_signal = "BULLISH"
            elif bl.avg_fii_fo_net < -bl.std_fii_fo_net:
                fo_signal = "BEARISH"

        # Build message
        parts = []
        if today_net != 0:
            direction = "buying" if today_net > 0 else "selling"
            parts.append(f"FII {direction} {_fmt_crores(abs(today_net))} today")
        parts.append(f"5-day net: {_fmt_crores(net_5d)}")
        if trend_direction != "STABLE":
            parts.append(f"{trend_direction.capitalize()}")
        message = " | ".join(parts) + "."

        return FIIBias(
            bias=bias,
            confidence=round(confidence, 2),
            net_5d=net_5d,
            net_10d=net_10d,
            today_net=today_net,
            trend_direction=trend_direction,
            fo_signal=fo_signal,
            message=message,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Index Divergence Detector
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DivergenceAnomaly:
    """Detected divergence between two indices."""

    pair: str
    timestamp: datetime
    anomaly_type: str           # CORRELATION_BREAKDOWN / DIRECTIONAL_DIVERGENCE / RELATIVE_STRENGTH_SHIFT / VIX_DIVERGENCE
    severity: str
    current_correlation: float = 0.0
    expected_correlation: float = 0.0
    index1_return_pct: float = 0.0
    index2_return_pct: float = 0.0
    divergence_magnitude: float = 0.0
    implication: str = ""       # SECTOR_ROTATION / RISK_OFF / RISK_ON / UNSTABLE_RALLY / HIDDEN_WEAKNESS
    details: dict = field(default_factory=dict)
    message: str = ""


@dataclass
class SectorRotation:
    """Detected sector rotation pattern."""

    timestamp: datetime
    outperforming_sectors: list[tuple[str, float]] = field(default_factory=list)
    underperforming_sectors: list[tuple[str, float]] = field(default_factory=list)
    rotation_type: str = "BROAD_ROTATION"       # DEFENSIVE_TO_CYCLICAL / CYCLICAL_TO_DEFENSIVE / BROAD_ROTATION / NARROW_LEADERSHIP
    strength: str = "MODERATE"                  # STRONG / MODERATE / WEAK
    message: str = ""


# Default monitored pairs with expected correlations
DEFAULT_INDEX_PAIRS: dict[str, dict] = {
    "NIFTY50_vs_SENSEX": {
        "index1": "NIFTY50", "index2": "SENSEX",
        "expected_corr": 0.95, "corr_std": 0.02,
        "is_inverse": False,
    },
    "NIFTY50_vs_BANKNIFTY": {
        "index1": "NIFTY50", "index2": "BANKNIFTY",
        "expected_corr": 0.85, "corr_std": 0.05,
        "is_inverse": False,
    },
    "NIFTY50_vs_NIFTYIT": {
        "index1": "NIFTY50", "index2": "NIFTYIT",
        "expected_corr": 0.70, "corr_std": 0.10,
        "is_inverse": False,
    },
    "NIFTY50_vs_NIFTYPHARMA": {
        "index1": "NIFTY50", "index2": "NIFTYPHARMA",
        "expected_corr": 0.60, "corr_std": 0.10,
        "is_inverse": False,
    },
    "BANKNIFTY_vs_NIFTY_PSU_BANK": {
        "index1": "BANKNIFTY", "index2": "NIFTY_PSU_BANK",
        "expected_corr": 0.82, "corr_std": 0.06,
        "is_inverse": False,
    },
    "BANKNIFTY_vs_NIFTY_PRIVATE_BANK": {
        "index1": "BANKNIFTY", "index2": "NIFTY_PRIVATE_BANK",
        "expected_corr": 0.90, "corr_std": 0.04,
        "is_inverse": False,
    },
    "NIFTY50_vs_INDIA_VIX": {
        "index1": "NIFTY50", "index2": "INDIA_VIX",
        "expected_corr": -0.80, "corr_std": 0.08,
        "is_inverse": True,
    },
}

# Sector classification for rotation detection
_DEFENSIVE_SECTORS = {"NIFTYPHARMA", "NIFTY_FMCG", "NIFTYIT"}
_CYCLICAL_SECTORS = {"NIFTY_METAL", "NIFTY_REALTY", "NIFTY_AUTO", "BANKNIFTY", "NIFTY_INFRA"}


class CrossIndexDivergenceDetector:
    """Detects divergences between correlated index pairs and sector rotation.

    Monitors pre-defined pairs of indices that normally move together, and
    flags when their correlation breaks down, returns diverge, or relative
    strength shifts significantly.
    """

    def __init__(
        self,
        db: DatabaseManager,
        pair_config: Optional[dict[str, dict]] = None,
    ) -> None:
        self.db = db
        self._pairs = pair_config or DEFAULT_INDEX_PAIRS
        self._cooldowns: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Cooldown management
    # ------------------------------------------------------------------

    def _is_on_cooldown(self, key: str) -> bool:
        last = self._cooldowns.get(key)
        if last is None:
            return False
        return datetime.now(tz=_IST) - last < timedelta(minutes=_DIVERGENCE_COOLDOWN_MINUTES)

    def _set_cooldown(self, key: str) -> None:
        self._cooldowns[key] = datetime.now(tz=_IST)

    # ------------------------------------------------------------------
    # Pair divergence detection
    # ------------------------------------------------------------------

    def detect_divergence(
        self,
        pair_id: str,
        index1_df: pd.DataFrame,
        index2_df: pd.DataFrame,
        lookback: int = 20,
    ) -> Optional[DivergenceAnomaly]:
        """Detect divergence between two index return series.

        Parameters
        ----------
        pair_id : str
            Key into the pair config, e.g. ``"NIFTY50_vs_BANKNIFTY"``.
        index1_df : pd.DataFrame
            Must have columns ``date`` and ``close``.
        index2_df : pd.DataFrame
            Must have columns ``date`` and ``close``.
        lookback : int
            Number of trading days for baseline correlation.

        Returns
        -------
        DivergenceAnomaly | None
            The most significant divergence detected, or None.
        """
        pair_cfg = self._pairs.get(pair_id)
        if pair_cfg is None:
            return None

        # Merge on date, require at least `lookback` days
        merged = pd.merge(
            index1_df[["date", "close"]].rename(columns={"close": "close1"}),
            index2_df[["date", "close"]].rename(columns={"close": "close2"}),
            on="date", how="inner",
        ).sort_values("date").reset_index(drop=True)

        if len(merged) < lookback:
            return None  # insufficient history

        merged = merged.tail(lookback).reset_index(drop=True)

        # Compute daily returns
        merged["ret1"] = merged["close1"].pct_change()
        merged["ret2"] = merged["close2"].pct_change()
        merged = merged.dropna(subset=["ret1", "ret2"])

        if len(merged) < 5:
            return None

        expected_corr = pair_cfg["expected_corr"]
        corr_std = pair_cfg["corr_std"]
        is_inverse = pair_cfg.get("is_inverse", False)
        now = datetime.now(tz=_IST)

        # Latest day returns
        latest_ret1 = float(merged["ret1"].iloc[-1]) * 100  # percentage
        latest_ret2 = float(merged["ret2"].iloc[-1]) * 100

        # Rolling 5-day correlation
        if len(merged) >= 5:
            rolling_corr = merged["ret1"].rolling(5).corr(merged["ret2"])
            current_corr = float(rolling_corr.iloc[-1]) if not pd.isna(rolling_corr.iloc[-1]) else expected_corr
        else:
            current_corr = expected_corr

        # ── Check 0 (inverse pairs): VIX divergence ─────────────
        # Check VIX divergence FIRST for inverse pairs — it's the most
        # actionable signal and should not be masked by correlation breakdown.
        if is_inverse:
            # NIFTY up + VIX up → unstable rally
            if latest_ret1 > 0.3 and latest_ret2 > 0.3:
                magnitude = latest_ret1 + latest_ret2
                return DivergenceAnomaly(
                    pair=pair_id,
                    timestamp=now,
                    anomaly_type="VIX_DIVERGENCE",
                    severity="HIGH",
                    current_correlation=round(current_corr, 3),
                    expected_correlation=expected_corr,
                    index1_return_pct=round(latest_ret1, 2),
                    index2_return_pct=round(latest_ret2, 2),
                    divergence_magnitude=round(magnitude, 2),
                    implication="UNSTABLE_RALLY",
                    details={"pattern": "MARKET_UP_VIX_UP"},
                    message=(
                        f"VIX divergence: {pair_cfg['index1']} {latest_ret1:+.1f}% "
                        f"but VIX also up {latest_ret2:+.1f}%. "
                        f"Market rising with increasing fear — unstable rally warning."
                    ),
                )
            # NIFTY down + VIX down → complacent decline
            if latest_ret1 < -0.3 and latest_ret2 < -0.3:
                magnitude = abs(latest_ret1) + abs(latest_ret2)
                return DivergenceAnomaly(
                    pair=pair_id,
                    timestamp=now,
                    anomaly_type="VIX_DIVERGENCE",
                    severity="MEDIUM",
                    current_correlation=round(current_corr, 3),
                    expected_correlation=expected_corr,
                    index1_return_pct=round(latest_ret1, 2),
                    index2_return_pct=round(latest_ret2, 2),
                    divergence_magnitude=round(magnitude, 2),
                    implication="HIDDEN_WEAKNESS",
                    details={"pattern": "MARKET_DOWN_VIX_DOWN"},
                    message=(
                        f"VIX divergence: {pair_cfg['index1']} {latest_ret1:+.1f}% "
                        f"and VIX down {latest_ret2:+.1f}%. "
                        f"Market declining without fear — complacent decline."
                    ),
                )

        # ── Check 1: Correlation breakdown ────────────────────────
        corr_threshold = expected_corr - 2 * corr_std
        if not is_inverse and current_corr < corr_threshold:
            severity = "HIGH" if current_corr < (expected_corr - 3 * corr_std) else "MEDIUM"
            return DivergenceAnomaly(
                pair=pair_id,
                timestamp=now,
                anomaly_type="CORRELATION_BREAKDOWN",
                severity=severity,
                current_correlation=round(current_corr, 3),
                expected_correlation=expected_corr,
                index1_return_pct=round(latest_ret1, 2),
                index2_return_pct=round(latest_ret2, 2),
                divergence_magnitude=round(abs(expected_corr - current_corr), 3),
                implication="SECTOR_ROTATION",
                details={"corr_threshold": round(corr_threshold, 3)},
                message=(
                    f"{pair_cfg['index1']} vs {pair_cfg['index2']} correlation "
                    f"dropped to {current_corr:.2f} (normally {expected_corr:.2f}). "
                    f"Possible sector rotation or structural divergence."
                ),
            )

        # For inverse pairs, correlation breakdown means becoming less negative
        if is_inverse and current_corr > (expected_corr + 2 * corr_std):
            severity = "MEDIUM"
            return DivergenceAnomaly(
                pair=pair_id,
                timestamp=now,
                anomaly_type="CORRELATION_BREAKDOWN",
                severity=severity,
                current_correlation=round(current_corr, 3),
                expected_correlation=expected_corr,
                index1_return_pct=round(latest_ret1, 2),
                index2_return_pct=round(latest_ret2, 2),
                divergence_magnitude=round(abs(expected_corr - current_corr), 3),
                implication="HIDDEN_WEAKNESS",
                details={"corr_threshold": round(expected_corr + 2 * corr_std, 3)},
                message=(
                    f"{pair_cfg['index1']} vs {pair_cfg['index2']} inverse "
                    f"correlation weakened to {current_corr:.2f} "
                    f"(normally {expected_corr:.2f}). Unusual relationship."
                ),
            )

        # ── Check 2: Directional divergence ───────────────────────
        if (not is_inverse and expected_corr > 0.7
                and latest_ret1 > 0.5 and latest_ret2 < -0.5):
            magnitude = abs(latest_ret1 - latest_ret2)
            severity = "HIGH" if magnitude > 2.0 else "MEDIUM"
            return DivergenceAnomaly(
                pair=pair_id,
                timestamp=now,
                anomaly_type="DIRECTIONAL_DIVERGENCE",
                severity=severity,
                current_correlation=round(current_corr, 3),
                expected_correlation=expected_corr,
                index1_return_pct=round(latest_ret1, 2),
                index2_return_pct=round(latest_ret2, 2),
                divergence_magnitude=round(magnitude, 2),
                implication="SECTOR_ROTATION",
                details={"direction": f"{pair_cfg['index1']}_UP_{pair_cfg['index2']}_DOWN"},
                message=(
                    f"{pair_cfg['index1']} diverging from {pair_cfg['index2']}: "
                    f"{pair_cfg['index1']} {latest_ret1:+.1f}% but "
                    f"{pair_cfg['index2']} {latest_ret2:+.1f}%. "
                    f"Correlation {current_corr:.2f} (normally {expected_corr:.2f})."
                ),
            )

        if (not is_inverse and expected_corr > 0.7
                and latest_ret1 < -0.5 and latest_ret2 > 0.5):
            magnitude = abs(latest_ret1 - latest_ret2)
            severity = "HIGH" if magnitude > 2.0 else "MEDIUM"
            return DivergenceAnomaly(
                pair=pair_id,
                timestamp=now,
                anomaly_type="DIRECTIONAL_DIVERGENCE",
                severity=severity,
                current_correlation=round(current_corr, 3),
                expected_correlation=expected_corr,
                index1_return_pct=round(latest_ret1, 2),
                index2_return_pct=round(latest_ret2, 2),
                divergence_magnitude=round(magnitude, 2),
                implication="SECTOR_ROTATION",
                details={"direction": f"{pair_cfg['index1']}_DOWN_{pair_cfg['index2']}_UP"},
                message=(
                    f"{pair_cfg['index2']} diverging from {pair_cfg['index1']}: "
                    f"{pair_cfg['index1']} {latest_ret1:+.1f}% but "
                    f"{pair_cfg['index2']} {latest_ret2:+.1f}%. "
                    f"Correlation {current_corr:.2f} (normally {expected_corr:.2f})."
                ),
            )

        # ── Check 3: Relative strength shift (5-day) ─────────────
        if len(merged) >= 5:
            ret1_5d = merged["ret1"].tail(5)
            ret2_5d = merged["ret2"].tail(5)
            diff_5d = ret1_5d - ret2_5d
            diff_mean = float(merged["ret1"].sub(merged["ret2"]).mean())
            diff_std = float(merged["ret1"].sub(merged["ret2"]).std(ddof=1))
            if diff_std > 0:
                recent_diff = float(diff_5d.sum())
                z = abs(recent_diff) / (diff_std * math.sqrt(5))
                if z > 2.0:
                    leader = pair_cfg["index1"] if recent_diff > 0 else pair_cfg["index2"]
                    return DivergenceAnomaly(
                        pair=pair_id,
                        timestamp=now,
                        anomaly_type="RELATIVE_STRENGTH_SHIFT",
                        severity="MEDIUM",
                        current_correlation=round(current_corr, 3),
                        expected_correlation=expected_corr,
                        index1_return_pct=round(float(ret1_5d.sum()) * 100, 2),
                        index2_return_pct=round(float(ret2_5d.sum()) * 100, 2),
                        divergence_magnitude=round(z, 2),
                        implication="SECTOR_ROTATION",
                        details={
                            "outperformer": leader,
                            "z_score": round(z, 2),
                            "5d_diff": round(recent_diff * 100, 2),
                        },
                        message=(
                            f"{leader} outperforming in {pair_id.replace('_vs_', ' vs ')}: "
                            f"relative strength Z-score {z:.1f} over 5 days."
                        ),
                    )

        return None

    # ------------------------------------------------------------------
    # Sector rotation detection
    # ------------------------------------------------------------------

    def detect_sector_rotation(
        self,
        all_sector_indices_df: dict[str, pd.DataFrame],
    ) -> list[SectorRotation]:
        """Compare 5-day performance of all sector indices to detect rotation.

        Parameters
        ----------
        all_sector_indices_df : dict[str, pd.DataFrame]
            Map of ``{index_id: DataFrame}`` where each DataFrame has
            columns ``date`` and ``close``.

        Returns
        -------
        list[SectorRotation]
            Detected rotation patterns (usually 0 or 1).
        """
        now = datetime.now(tz=_IST)
        returns: dict[str, float] = {}

        # Count how many sector indices have sufficient data
        available_sectors = []
        skipped_sectors = []

        for idx_id, df in all_sector_indices_df.items():
            if df is None or len(df) < 10:
                skipped_sectors.append(f"{idx_id} ({len(df) if df is not None else 0} bars)")
                continue
            
            available_sectors.append(idx_id)
            df_sorted = df.sort_values("date").reset_index(drop=True)
            close_start = _safe_float(df_sorted["close"].iloc[-5])
            close_end = _safe_float(df_sorted["close"].iloc[-1])
            if close_start > 0:
                ret = ((close_end - close_start) / close_start) * 100
                returns[idx_id] = round(ret, 2)

        if len(available_sectors) < 4:
            logger.info(
                f"Sector rotation: insufficient data — only {len(available_sectors)} sectors available "
                f"(need 4+). Available: {available_sectors}. "
                f"Skipped: {skipped_sectors}. Analysis skipped."
            )
            return []

        if skipped_sectors:
            logger.info(
                f"Sector rotation: analyzing {len(available_sectors)} sectors, "
                f"skipped {len(skipped_sectors)}: {skipped_sectors}"
            )

        # Sort by return
        sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_returns[:3]
        bottom_3 = sorted_returns[-3:]

        # Check for broad selloff — all sectors down
        if all(r < 0 for _, r in sorted_returns):
            avg_decline = sum(r for _, r in sorted_returns) / len(sorted_returns)
            if avg_decline < -1.0:
                return [SectorRotation(
                    timestamp=now,
                    outperforming_sectors=[],
                    underperforming_sectors=sorted_returns,
                    rotation_type="BROAD_ROTATION",
                    strength="STRONG" if avg_decline < -3.0 else "MODERATE",
                    message=(
                        f"Broad market selloff: all {len(sorted_returns)} sectors "
                        f"negative (avg {avg_decline:.1f}%). No rotation — risk-off."
                    ),
                )]

        # Clear rotation: top > +2% while bottom < -2%
        top_avg = sum(r for _, r in top_3) / len(top_3)
        bottom_avg = sum(r for _, r in bottom_3) / len(bottom_3)
        spread = top_avg - bottom_avg

        if top_avg > 2.0 and bottom_avg < -2.0:
            strength = "STRONG"
        elif spread > 3.0:
            strength = "MODERATE"
        elif spread > 1.5:
            strength = "WEAK"
        else:
            return []  # No meaningful rotation

        # Classify rotation type
        top_ids = {t[0] for t in top_3}
        bottom_ids = {t[0] for t in bottom_3}
        if top_ids & _DEFENSIVE_SECTORS and bottom_ids & _CYCLICAL_SECTORS:
            rotation_type = "CYCLICAL_TO_DEFENSIVE"
        elif top_ids & _CYCLICAL_SECTORS and bottom_ids & _DEFENSIVE_SECTORS:
            rotation_type = "DEFENSIVE_TO_CYCLICAL"
        elif len(top_ids) <= 2 and top_avg > 3.0:
            rotation_type = "NARROW_LEADERSHIP"
        else:
            rotation_type = "BROAD_ROTATION"

        top_str = ", ".join(f"{t[0]} ({t[1]:+.1f}%)" for t in top_3)
        bottom_str = ", ".join(f"{t[0]} ({t[1]:+.1f}%)" for t in bottom_3)

        return [SectorRotation(
            timestamp=now,
            outperforming_sectors=top_3,
            underperforming_sectors=bottom_3,
            rotation_type=rotation_type,
            strength=strength,
            message=(
                f"Sector rotation detected: {top_str} outperforming while "
                f"{bottom_str} underperforming. {rotation_type.replace('_', ' ').title()} pattern."
            ),
        )]

    # ------------------------------------------------------------------
    # Unified anomaly detection
    # ------------------------------------------------------------------

    def detect_all_divergences(
        self,
        index_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> list[AnomalyEvent]:
        """Run all pair divergence checks and sector rotation.

        Parameters
        ----------
        index_data : dict[str, pd.DataFrame], optional
            Map of ``{index_id: DataFrame}`` with columns ``date``, ``close``.
            If None, loads from DB.

        Returns
        -------
        list[AnomalyEvent]
            Anomalies in unified format with ``category = "DIVERGENCE"``.
        """
        now = datetime.now(tz=_IST)
        results = []
        pairs_checked = 0
        pairs_skipped = 0
        skip_reasons = []  # Track why pairs were skipped

        if index_data is None:
            index_data = self._load_index_data()

        if not index_data:
            logger.info(
                "Divergence detection: no index data available. "
                "Cross-index divergence analysis is inactive. "
                "Run 'python scripts/seed_historical.py' to populate database."
            )
            return []

        # Check for broad selloff first
        broad_selloff = self._check_broad_selloff(index_data)

        # Run pair divergence checks
        for pair_id, pair_cfg in self._pairs.items():
            idx1_id = pair_cfg["index1"]
            idx2_id = pair_cfg["index2"]
            pair_name = f"{idx1_id} vs {idx2_id}"

            df1 = index_data.get(idx1_id)
            df2 = index_data.get(idx2_id)

            # Check index 1 data
            df1_len = len(df1) if df1 is not None else 0
            if df1_len < 20:
                reason = f"{pair_name}: {idx1_id} has {df1_len} bars (need 20+)"
                skip_reasons.append(reason)
                logger.info(f"Divergence skip — {reason}")
                pairs_skipped += 1
                continue

            # Check index 2 data
            df2_len = len(df2) if df2 is not None else 0
            if df2_len < 20:
                reason = f"{pair_name}: {idx2_id} has {df2_len} bars (need 20+)"
                skip_reasons.append(reason)
                logger.info(f"Divergence skip — {reason}")
                pairs_skipped += 1
                continue

            # Check overlapping date range
            try:
                if 'date' in df1.columns:
                    dates1 = set(df1['date'])
                    dates2 = set(df2['date'])
                elif 'timestamp' in df1.columns:
                    dates1 = set(df1['timestamp'].dt.date if hasattr(df1['timestamp'], 'dt') else df1['timestamp'])
                    dates2 = set(df2['timestamp'].dt.date if hasattr(df2['timestamp'], 'dt') else df2['timestamp'])
                else:
                    dates1 = set(df1.index) if hasattr(df1.index, '__iter__') else set()
                    dates2 = set(df2.index) if hasattr(df2.index, '__iter__') else set()

                overlap_count = len(dates1 & dates2)
            except Exception:
                overlap_count = min(df1_len, df2_len)

            if overlap_count < 15:
                reason = f"{pair_name}: only {overlap_count} overlapping days (need 15+)"
                skip_reasons.append(reason)
                logger.info(f"Divergence skip — {reason}")
                pairs_skipped += 1
                continue

            # Data sufficient — run detection
            cooldown_key = f"DIVERGENCE_{pair_id}"
            if self._is_on_cooldown(cooldown_key):
                pairs_checked += 1
                continue

            pairs_checked += 1
            try:
                anomaly = self.detect_divergence(pair_id, df1, df2)
                if anomaly is not None:
                    # In broad selloff, only flag if it's a VIX divergence
                    if broad_selloff and anomaly.anomaly_type != "VIX_DIVERGENCE":
                        continue

                    results.append(AnomalyEvent(
                        index_id=f"{idx1_id}|{idx2_id}",
                        timestamp=now,
                        anomaly_type=anomaly.anomaly_type,
                        severity=anomaly.severity,
                        category="DIVERGENCE",
                        details=json.dumps({
                            "pair": anomaly.pair,
                            "current_corr": anomaly.current_correlation,
                            "expected_corr": anomaly.expected_correlation,
                            "idx1_ret": anomaly.index1_return_pct,
                            "idx2_ret": anomaly.index2_return_pct,
                            "magnitude": anomaly.divergence_magnitude,
                            "implication": anomaly.implication,
                            **anomaly.details,
                        }),
                        message=anomaly.message,
                        cooldown_key=cooldown_key,
                    ))
                    self._set_cooldown(cooldown_key)
            except Exception as e:
                logger.error(f"Divergence detection error for {pair_name}: {e}")

        # === LOGGING SUMMARY ===
        total_pairs = pairs_checked + pairs_skipped

        if pairs_skipped > 0 and pairs_checked == 0:
            logger.info(
                f"Divergence detection: ALL {pairs_skipped}/{total_pairs} pairs skipped "
                f"due to insufficient data. Cross-index divergence analysis is inactive. "
                f"Need at least 2 indices with 20+ overlapping trading days. "
                f"Run 'python scripts/seed_historical.py' or wait for data collector to accumulate data."
            )
        elif pairs_skipped > 0:
            logger.info(
                f"Divergence detection: checked {pairs_checked}/{total_pairs} pairs, "
                f"skipped {pairs_skipped} (insufficient data). "
                f"Found {len(results)} anomalie(s)."
            )
        else:
            logger.debug(
                f"Divergence detection: checked {pairs_checked}/{total_pairs} pairs. "
                f"Found {len(results)} anomalie(s)."
            )

        # Sector rotation
        sector_data = {k: v for k, v in index_data.items()
                       if k not in ("NIFTY50", "SENSEX", "INDIA_VIX")}
        if sector_data and not broad_selloff:
            rotations = self.detect_sector_rotation(sector_data)
            for rot in rotations:
                cooldown_key = "DIVERGENCE_SECTOR_ROTATION"
                if self._is_on_cooldown(cooldown_key):
                    continue
                results.append(AnomalyEvent(
                    index_id="MARKET",
                    timestamp=now,
                    anomaly_type="SECTOR_ROTATION",
                    severity="MEDIUM" if rot.strength == "MODERATE" else (
                        "HIGH" if rot.strength == "STRONG" else "LOW"
                    ),
                    category="DIVERGENCE",
                    details=json.dumps({
                        "outperforming": rot.outperforming_sectors,
                        "underperforming": rot.underperforming_sectors,
                        "rotation_type": rot.rotation_type,
                        "strength": rot.strength,
                    }),
                    message=rot.message,
                    cooldown_key=cooldown_key,
                ))
                self._set_cooldown(cooldown_key)
        elif broad_selloff:
            cooldown_key = "DIVERGENCE_BROAD_SELLOFF"
            if not self._is_on_cooldown(cooldown_key):
                results.append(AnomalyEvent(
                    index_id="MARKET",
                    timestamp=now,
                    anomaly_type="BROAD_MARKET_SELLOFF",
                    severity="HIGH",
                    category="DIVERGENCE",
                    details=json.dumps({"broad_selloff": True}),
                    message="Broad market selloff detected — all indices declining. Individual divergences suppressed.",
                    cooldown_key=cooldown_key,
                ))
                self._set_cooldown(cooldown_key)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index_data(self) -> dict[str, pd.DataFrame]:
        """Load last 30 days of daily price data for all monitored indices."""
        index_ids: set[str] = set()
        for cfg in self._pairs.values():
            index_ids.add(cfg["index1"])
            index_ids.add(cfg["index2"])

        result: dict[str, pd.DataFrame] = {}
        end = datetime.now(tz=_IST)
        start = end - timedelta(days=40)  # extra buffer for weekends/holidays

        for idx_id in index_ids:
            rows = self.db.fetch_all(
                Q.LIST_PRICE_HISTORY,
                (idx_id, "1d", start.isoformat(), end.isoformat()),
            )
            if rows and len(rows) >= 20:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["timestamp"]).dt.date
                result[idx_id] = df

        return result

    def _check_broad_selloff(self, index_data: dict[str, pd.DataFrame]) -> bool:
        """Return True if all major indices are declining today."""
        declines = 0
        total = 0
        for idx_id, df in index_data.items():
            if idx_id == "INDIA_VIX" or len(df) < 2:
                continue
            total += 1
            df_sorted = df.sort_values("date")
            last_close = _safe_float(df_sorted["close"].iloc[-1])
            prev_close = _safe_float(df_sorted["close"].iloc[-2])
            if prev_close > 0 and last_close < prev_close:
                declines += 1

        if total < 3:
            return False
        return declines / total > 0.85
