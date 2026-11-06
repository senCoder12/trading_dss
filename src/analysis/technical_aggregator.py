"""
Master technical aggregator — single entry point for all technical analysis.

Combines Trend, Momentum, Volatility, Volume, Options, Quant, and Smart Money
indicators into one unified ``TechnicalAnalysisResult``.  The result includes
aggregated votes, overall signal, confidence, support/resistance levels, alerts,
and human-readable reasoning.

Thread-safe: each call creates its own indicator instances and operates on its
own data — safe to call for multiple indices in parallel.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.indicators.trend import TrendIndicators, TrendSummary
from src.analysis.indicators.momentum import MomentumIndicators, MomentumSummary
from src.analysis.indicators.volatility import VolatilityIndicators, VolatilitySummary
from src.analysis.indicators.volume import VolumeIndicators, VolumeSummary
from src.analysis.indicators.options_indicators import OptionsIndicators, OptionsSummary
from src.analysis.indicators.quant import QuantIndicators, QuantSummary
from src.analysis.indicators.smart_money import SmartMoneyIndicators, SmartMoneyScore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """A high-priority market condition detected during analysis."""

    type: str       # DIVERGENCE / BREAKOUT_TRAP / OI_SPIKE / VOLUME_CLIMAX /
                    # BB_SQUEEZE / SMART_MONEY_SIGNAL / VIX_EXTREME / REVERSAL_WARNING
    severity: str   # HIGH / MEDIUM / LOW
    message: str    # Human-readable description
    source: str     # Which indicator generated this


# ---------------------------------------------------------------------------
# TechnicalAnalysisResult
# ---------------------------------------------------------------------------

@dataclass
class TechnicalAnalysisResult:
    """Unified result from all technical analysis layers."""

    index_id: str
    timestamp: datetime
    timeframe: str

    # Individual summaries (preserved for drill-down)
    trend: TrendSummary
    momentum: MomentumSummary
    volatility: VolatilitySummary
    volume: VolumeSummary
    options: Optional[OptionsSummary]
    quant: QuantSummary
    smart_money: Optional[SmartMoneyScore]

    # ── Aggregated signals ──────────────────────────────────────────────
    votes: dict[str, str]           # {"trend": "BULLISH", "momentum": "NEUTRAL", ...}
    bullish_votes: int
    bearish_votes: int
    neutral_votes: int

    overall_signal: str             # STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
    overall_confidence: float       # 0.0 – 1.0

    # Key levels
    support_levels: list[float]
    resistance_levels: list[float]
    immediate_support: float
    immediate_resistance: float

    # Risk parameters
    suggested_stop_loss_distance: float
    suggested_target_distance: float
    position_size_modifier: float

    # Alerts
    alerts: list[Alert] = field(default_factory=list)

    # Reasoning
    reasoning: str = ""

    # Data quality
    data_completeness: float = 1.0
    warnings: list[str] = field(default_factory=list)
    # False when the bar series carries no real volume (NSE spot indices
    # publish none intraday).  Volume-derived votes are then meaningless and
    # must be treated as unavailable, not as NEUTRAL.
    volume_available: bool = True
    # Components that failed to compute (treated as unavailable, not NEUTRAL)
    unavailable_components: list[str] = field(default_factory=list)
    # Near-month futures positioning/order flow (src/analysis/futures_flow);
    # attached by the engine / calibration replay, None when not computed.
    futures_flow: Optional[object] = None

    # Multi-timeframe confluence
    timeframe_confluence_score: float = 0.0   # -1.0 to 1.0 (agreement across TFs)
    tf_agreement: str = "UNKNOWN"              # "ALIGNED_BULL" / "ALIGNED_BEAR" / "MIXED" / "UNKNOWN"
    higher_tf_signal: str = "NEUTRAL"          # Signal on 15m/1h (directional bias)
    lower_tf_signal: str = "NEUTRAL"           # Signal on 1m/5m (entry timing)

    # Opening Range Breakout data (populated if ORB available)
    orb_high: float = 0.0
    orb_low: float = 0.0
    orb_captured: bool = False


# ---------------------------------------------------------------------------
# Vote conversion helpers
# ---------------------------------------------------------------------------

_VOTE_NUMERIC: dict[str, float] = {
    "STRONG_BULLISH": 2.0,
    "STRONGLY_BULLISH": 2.0,
    "BULLISH": 1.0,
    "NEUTRAL": 0.0,
    "BEARISH": -1.0,
    "STRONGLY_BEARISH": -2.0,
    "STRONG_BEARISH": -2.0,
}

# Weights with options data available
_WEIGHTS_WITH_OPTIONS: dict[str, float] = {
    "trend": 0.20,
    "momentum": 0.15,
    "options": 0.25,
    "volume": 0.15,
    "smart_money": 0.15,
    "quant": 0.05,
    "volatility": 0.05,
}

# Weights without options data
_WEIGHTS_WITHOUT_OPTIONS: dict[str, float] = {
    "trend": 0.30,
    "momentum": 0.20,
    "volume": 0.25,
    "smart_money": 0.15,
    "quant": 0.10,
    "volatility": 0.00,
}


def _vote_to_numeric(vote: str) -> float:
    """Convert a vote string to a numeric value."""
    return _VOTE_NUMERIC.get(vote, 0.0)


def _score_to_signal(score: float) -> str:
    """Map weighted score to overall signal."""
    if score > 1.2:
        return "STRONG_BUY"
    if score >= 0.5:
        return "BUY"
    if score <= -1.2:
        return "STRONG_SELL"
    if score <= -0.5:
        return "SELL"
    return "NEUTRAL"


def _cluster_levels(levels: list[float], threshold_pct: float = 0.003) -> list[float]:
    """Cluster nearby price levels (within *threshold_pct* of each other).

    Returns the average of each cluster, sorted ascending.
    """
    if not levels:
        return []

    sorted_levels = sorted(levels)
    clusters: list[list[float]] = [[sorted_levels[0]]]

    for lvl in sorted_levels[1:]:
        cluster_avg = sum(clusters[-1]) / len(clusters[-1])
        if cluster_avg > 0 and abs(lvl - cluster_avg) / cluster_avg <= threshold_pct:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    return sorted(round(sum(c) / len(c), 2) for c in clusters)


# ---------------------------------------------------------------------------
# TechnicalAggregator
# ---------------------------------------------------------------------------

class TechnicalAggregator:
    """Master aggregator that runs ALL indicators and produces a unified result.

    This is the **single entry point** for all technical analysis.  Pass it
    price data (and optional options / benchmark data) and receive one
    ``TechnicalAnalysisResult`` covering every indicator category.

    Thread-safe — each call instantiates its own indicator objects.
    """

    def analyze(
        self,
        index_id: str,
        price_df: pd.DataFrame,
        options_chain=None,
        oi_history: Optional[list[dict]] = None,
        vix_value: Optional[float] = None,
        benchmark_df: Optional[pd.DataFrame] = None,
        timeframe: str = "1d",
        htf_df: Optional[pd.DataFrame] = None,      # Higher timeframe (15m or 1h)
        ltf_df: Optional[pd.DataFrame] = None,      # Lower timeframe (1m)
        orb_data: Optional[dict] = None,            # Opening Range {"high": x, "low": y}
        iv_history: Optional[list[float]] = None,   # 30-day ATM IV list for IV rank
        advanced_opts=None,                          # AdvancedOptionsMetrics from options_analysis
        fii_futures_net: Optional[float] = None,   # FII net futures contracts (from DB)
    ) -> TechnicalAnalysisResult:
        """Run every indicator category and aggregate into a single result.

        Parameters
        ----------
        index_id:
            Identifier for the index (e.g. ``"NIFTY50"``).
        price_df:
            OHLCV DataFrame with columns ``open, high, low, close, volume``.
        options_chain:
            An ``OptionsChainData`` object (or ``None``).
        oi_history:
            List of OI history dicts for smart money analysis.
        vix_value:
            Current India VIX value (or ``None``).
        benchmark_df:
            OHLCV DataFrame for benchmark index (for beta / correlation).
        timeframe:
            Timeframe label (default ``"1d"``).

        Returns
        -------
        TechnicalAnalysisResult
            Unified analysis with votes, signal, confidence, levels, and alerts.
        """
        now = datetime.now()
        # The result is stamped with the *data* time (last bar), not the wall
        # clock: downstream staleness checks compare against this, so a
        # wall-clock stamp would make days-old bars look fresh.
        data_ts = _last_bar_timestamp(price_df) or now
        volume_available = _has_real_volume(price_df)
        warnings: list[str] = []
        if not volume_available:
            warnings.append(
                "No real volume in price series — volume/smart-money votes disabled"
            )
        categories_available = 0
        total_categories = 7
        # Components whose computation FAILED.  A failed layer used to fall
        # back to a NEUTRAL vote that still carried its weight, silently
        # diluting the score; downstream treats these as unavailable.
        unavailable: set[str] = set()

        # ── 1. Trend ────────────────────────────────────────────────────
        trend_ind = TrendIndicators()
        try:
            trend = trend_ind.get_trend_summary(price_df, index_id=index_id, timeframe=timeframe)
            categories_available += 1
            logger.debug("Trend vote: %s (confidence %.2f)", trend.trend_vote, trend.trend_confidence)
        except Exception:
            logger.exception("Trend analysis failed for %s", index_id)
            warnings.append("Trend analysis failed — component unavailable")
            trend = _empty_trend(index_id, timeframe, now)
            unavailable.add("trend")

        # ── 2. Momentum ────────────────────────────────────────────────
        momentum_ind = MomentumIndicators()
        try:
            momentum = momentum_ind.get_momentum_summary(price_df)
            categories_available += 1
            logger.debug("Momentum vote: %s (confidence %.2f)", momentum.momentum_vote, momentum.momentum_confidence)
        except Exception:
            logger.exception("Momentum analysis failed for %s", index_id)
            warnings.append("Momentum analysis failed — component unavailable")
            momentum = _empty_momentum(now)
            unavailable.add("momentum")

        # ── 3. Volatility ──────────────────────────────────────────────
        vol_ind = VolatilityIndicators()
        try:
            volatility = vol_ind.get_volatility_summary(price_df, vix_value=vix_value)
            categories_available += 1
            logger.debug("Volatility vote: %s", volatility.volatility_vote)
        except Exception:
            logger.exception("Volatility analysis failed for %s", index_id)
            warnings.append("Volatility analysis failed — using neutral fallback")
            volatility = _empty_volatility(now)

        # ── 4. Volume ──────────────────────────────────────────────────
        volume_ind = VolumeIndicators()
        try:
            volume = volume_ind.get_volume_summary(price_df)
            categories_available += 1
            logger.debug("Volume vote: %s (confidence %.2f)", volume.volume_vote, volume.volume_confidence)
        except Exception:
            logger.exception("Volume analysis failed for %s", index_id)
            warnings.append("Volume analysis failed — component unavailable")
            volume = _empty_volume(now)
            unavailable.add("volume")

        # ── 5. Options ─────────────────────────────────────────────────
        options: Optional[OptionsSummary] = None
        if options_chain is not None:
            opts_ind = OptionsIndicators()
            try:
                options = opts_ind.get_options_summary(
                    options_chain,
                    iv_history=iv_history,
                    advanced_metrics=advanced_opts,
                )
                if options is not None:
                    categories_available += 1
                    logger.debug("Options vote: %s (confidence %.2f)", options.options_vote, options.options_confidence)
                else:
                    warnings.append("Options analysis returned None — insufficient data")
            except Exception:
                logger.exception("Options analysis failed for %s", index_id)
                warnings.append("Options analysis failed")
        else:
            warnings.append("Options data unavailable")

        # ── 6. Quant ───────────────────────────────────────────────────
        quant_ind = QuantIndicators()
        try:
            quant = quant_ind.get_quant_summary(price_df, benchmark_df=benchmark_df)
            categories_available += 1
            logger.debug("Quant vote: %s (confidence %.2f)", quant.quant_vote, quant.quant_confidence)
        except Exception:
            logger.exception("Quant analysis failed for %s", index_id)
            warnings.append("Quant analysis failed — using neutral fallback")
            quant = _empty_quant(now)

        if benchmark_df is None:
            warnings.append("Insufficient history for beta — no benchmark provided")

        # ── 7. Smart Money ─────────────────────────────────────────────
        smart_money: Optional[SmartMoneyScore] = None
        sm_ind = SmartMoneyIndicators()
        try:
            sm_support = options.oi_support if options else None
            sm_resistance = options.oi_resistance if options else None
            smart_money = sm_ind.calculate_smart_money_score(
                price_df,
                oi_data=oi_history,
                support=sm_support,
                resistance=sm_resistance,
                fii_futures_net=fii_futures_net,
            )
            if smart_money is not None and smart_money.data_completeness > 0.3:
                categories_available += 1
                logger.debug("Smart money bias: %s (score %.1f)", smart_money.smart_money_bias, smart_money.score)
            else:
                warnings.append("Smart money data completeness too low for reliable signal")
                smart_money = None
        except Exception:
            logger.exception("Smart money analysis failed for %s", index_id)
            warnings.append("Smart money analysis failed")

        # ── Collect votes ──────────────────────────────────────────────
        votes: dict[str, str] = {
            "trend": trend.trend_vote,
            "momentum": momentum.momentum_vote,
            "volatility": volatility.volatility_vote,
            "volume": volume.volume_vote,
            "quant": quant.quant_vote,
        }
        if options is not None:
            votes["options"] = options.options_vote
        if smart_money is not None:
            votes["smart_money"] = smart_money.smart_money_bias

        # Count directional votes (excluding volatility which doesn't vote on direction)
        directional_votes = {k: v for k, v in votes.items() if k != "volatility"}
        bullish_votes = sum(
            1 for v in directional_votes.values()
            if v in ("BULLISH", "STRONG_BULLISH", "STRONGLY_BULLISH")
        )
        bearish_votes = sum(
            1 for v in directional_votes.values()
            if v in ("BEARISH", "STRONG_BEARISH", "STRONGLY_BEARISH")
        )
        neutral_votes = sum(
            1 for v in directional_votes.values()
            if v in ("NEUTRAL",)
        )

        # ── Weighted score ─────────────────────────────────────────────
        has_options = options is not None
        weights = _WEIGHTS_WITH_OPTIONS if has_options else _WEIGHTS_WITHOUT_OPTIONS

        weighted_score = 0.0
        for category, vote in votes.items():
            w = weights.get(category, 0.0)
            weighted_score += _vote_to_numeric(vote) * w

        overall_signal = _score_to_signal(weighted_score)

        # ── Confidence ─────────────────────────────────────────────────
        data_completeness = categories_available / total_categories
        overall_confidence = self._calculate_confidence(
            votes=directional_votes,
            volatility=volatility,
            trend=trend,
            data_completeness=data_completeness,
        )

        # ── Support / Resistance ───────────────────────────────────────
        latest_close = float(price_df["close"].iloc[-1])
        support_levels, resistance_levels = self._aggregate_levels(
            price_df=price_df,
            trend_ind=trend_ind,
            volatility=volatility,
            volume=volume,
            options=options,
            latest_close=latest_close,
        )

        immediate_support = max(
            (s for s in support_levels if s < latest_close),
            default=latest_close * 0.99,
        )
        immediate_resistance = min(
            (r for r in resistance_levels if r > latest_close),
            default=latest_close * 1.01,
        )

        # ── Risk parameters ────────────────────────────────────────────
        suggested_sl = volatility.suggested_sl
        suggested_target = volatility.suggested_target
        position_size_mod = volatility.position_size_modifier

        # ── Alerts ─────────────────────────────────────────────────────
        alerts = self._generate_alerts(
            trend=trend,
            momentum=momentum,
            volatility=volatility,
            volume=volume,
            options=options,
            smart_money=smart_money,
            vix_value=vix_value,
        )

        # ── Multi-Timeframe Confluence ─────────────────────────────────
        tf_confluence = 0.0
        tf_agreement = "UNKNOWN"
        higher_tf_signal = "NEUTRAL"
        lower_tf_signal = "NEUTRAL"

        if htf_df is not None and len(htf_df) >= 20:
            try:
                htf_trend_ind = TrendIndicators()
                htf_trend = htf_trend_ind.get_trend_summary(htf_df, index_id=index_id, timeframe="htf")
                higher_tf_signal = htf_trend.trend_vote  # BULLISH / BEARISH / NEUTRAL
                htf_numeric = {"STRONG_BULLISH": 2, "BULLISH": 1, "NEUTRAL": 0,
                               "BEARISH": -1, "STRONG_BEARISH": -2}.get(higher_tf_signal, 0)
                # Current TF numeric
                cur_numeric = {"STRONG_BUY": 2, "BUY": 1, "NEUTRAL": 0,
                               "SELL": -1, "STRONG_SELL": -2}.get(overall_signal, 0)
                # Confluence: both point same direction = stronger signal
                if htf_numeric > 0 and cur_numeric > 0:
                    tf_agreement = "ALIGNED_BULL"
                    tf_confluence = min(1.0, (htf_numeric + cur_numeric) / 4.0)
                elif htf_numeric < 0 and cur_numeric < 0:
                    tf_agreement = "ALIGNED_BEAR"
                    tf_confluence = max(-1.0, (htf_numeric + cur_numeric) / 4.0)
                elif htf_numeric != 0 and cur_numeric != 0:
                    tf_agreement = "MIXED"
                    tf_confluence = cur_numeric * 0.3  # discount conflicted signal
                else:
                    tf_agreement = "NEUTRAL"
            except Exception as e:
                logger.debug("Higher TF confluence failed: %s", e)

        if ltf_df is not None and len(ltf_df) >= 10:
            try:
                ltf_trend_ind = TrendIndicators()
                ltf_trend = ltf_trend_ind.get_trend_summary(ltf_df, index_id=index_id, timeframe="ltf")
                lower_tf_signal = ltf_trend.trend_vote
            except Exception as e:
                logger.debug("Lower TF confluence failed: %s", e)

        # ── ORB data attachment ────────────────────────────────────────
        orb_high = 0.0
        orb_low = 0.0
        orb_captured = False
        if orb_data:
            orb_high = float(orb_data.get("high", 0))
            orb_low = float(orb_data.get("low", 0))
            orb_captured = orb_high > 0 and orb_low > 0

        # Boost confidence when higher TF aligns with current TF signal
        if tf_agreement == "ALIGNED_BULL" and overall_signal in ("BUY", "STRONG_BUY"):
            overall_confidence = min(1.0, overall_confidence + 0.08)
        elif tf_agreement == "ALIGNED_BEAR" and overall_signal in ("SELL", "STRONG_SELL"):
            overall_confidence = min(1.0, overall_confidence + 0.08)
        elif tf_agreement == "MIXED":
            # Conflicting TFs: reduce confidence
            overall_confidence = max(0.0, overall_confidence - 0.12)
            warnings.append(f"Multi-TF conflict: {higher_tf_signal} on higher TF vs {overall_signal} on current")

        # ── Build result ───────────────────────────────────────────────
        result = TechnicalAnalysisResult(
            index_id=index_id,
            timestamp=data_ts,
            timeframe=timeframe,
            trend=trend,
            momentum=momentum,
            volatility=volatility,
            volume=volume,
            options=options,
            quant=quant,
            smart_money=smart_money,
            votes=votes,
            bullish_votes=bullish_votes,
            bearish_votes=bearish_votes,
            neutral_votes=neutral_votes,
            overall_signal=overall_signal,
            overall_confidence=overall_confidence,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            immediate_support=immediate_support,
            immediate_resistance=immediate_resistance,
            suggested_stop_loss_distance=suggested_sl,
            suggested_target_distance=suggested_target,
            position_size_modifier=position_size_mod,
            alerts=alerts,
            data_completeness=data_completeness,
            warnings=warnings,
            timeframe_confluence_score=round(tf_confluence, 4),
            tf_agreement=tf_agreement,
            higher_tf_signal=higher_tf_signal,
            lower_tf_signal=lower_tf_signal,
            orb_high=orb_high,
            orb_low=orb_low,
            orb_captured=orb_captured,
            volume_available=volume_available,
            unavailable_components=sorted(unavailable),
        )

        # Generate reasoning last (needs the full result)
        result.reasoning = self.generate_reasoning(result)

        logger.info(
            "Analysis complete for %s: signal=%s confidence=%.2f (%d alerts)",
            index_id,
            overall_signal,
            overall_confidence,
            len(alerts),
        )

        return result

    # ------------------------------------------------------------------
    # Confidence calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        votes: dict[str, str],
        volatility: VolatilitySummary,
        trend: TrendSummary,
        data_completeness: float,
    ) -> float:
        """Compute overall confidence from vote agreement and modifiers.

        Base confidence from agreement: unanimous → 0.9, split → lower.
        Modifiers:
          - Extreme volatility → −0.15
          - Weak trend (ADX < 20) → −0.1
          - Missing data → dynamic reduction based on component freshness

        Completeness penalty scales with staleness:
          - Fresh data (all components recent) → minimal penalty
          - Stale data (missing or old components) → higher penalty
        Floor: 0.2, Ceiling: 0.9
        """
        if not votes:
            return 0.2

        numeric_votes = [_vote_to_numeric(v) for v in votes.values()]
        total = len(numeric_votes)

        # Agreement: ratio of majority direction
        positive = sum(1 for v in numeric_votes if v > 0)
        negative = sum(1 for v in numeric_votes if v < 0)
        majority = max(positive, negative)
        agreement_ratio = majority / total if total > 0 else 0.0

        # Base confidence: 0.5 (pure split) → 0.9 (unanimous)
        base = 0.5 + 0.4 * agreement_ratio

        # Volatility modifier
        vol_penalty = 0.15 if volatility.volatility_vote in ("EXTREME", "HIGH") else 0.0

        # ADX modifier — weak trend reduces confidence
        adx_penalty = 0.1 if trend.trend_strength == "WEAK" else 0.0

        # Data completeness modifier — dynamic based on component freshness
        # Fresh data (high completeness) → low penalty
        # Stale/missing data (low completeness) → high penalty, scaled by severity
        completeness_penalty = (1.0 - data_completeness) ** 1.5 * 0.3

        confidence = base - vol_penalty - adx_penalty - completeness_penalty
        return round(max(0.2, min(0.9, confidence)), 2)

    # ------------------------------------------------------------------
    # Support / Resistance aggregation
    # ------------------------------------------------------------------

    def _aggregate_levels(
        self,
        price_df: pd.DataFrame,
        trend_ind: TrendIndicators,
        volatility: VolatilitySummary,
        volume: VolumeSummary,
        options: Optional[OptionsSummary],
        latest_close: float,
    ) -> tuple[list[float], list[float]]:
        """Collect support/resistance from multiple sources, cluster, and return top 3 each."""
        raw_supports: list[float] = []
        raw_resistances: list[float] = []

        # EMA levels (compute numeric values)
        try:
            ema20 = trend_ind.calculate_ema(price_df, 20)
            ema50 = trend_ind.calculate_ema(price_df, 50)
            ema200 = trend_ind.calculate_ema(price_df, 200)

            for ema_series in [ema20, ema50, ema200]:
                valid = ema_series.dropna()
                if len(valid) > 0:
                    ema_val = float(valid.iloc[-1])
                    if not math.isnan(ema_val) and ema_val > 0:
                        if ema_val < latest_close:
                            raw_supports.append(ema_val)
                        else:
                            raw_resistances.append(ema_val)
        except Exception:
            logger.debug("Could not compute EMA levels for support/resistance")

        # Bollinger bands (from volatility)
        try:
            bb = VolatilityIndicators().calculate_bollinger_bands(price_df)
            upper_valid = bb.upper.dropna()
            lower_valid = bb.lower.dropna()
            if len(upper_valid) > 0:
                raw_resistances.append(float(upper_valid.iloc[-1]))
            if len(lower_valid) > 0:
                raw_supports.append(float(lower_valid.iloc[-1]))
        except Exception:
            logger.debug("Could not compute Bollinger levels for support/resistance")

        # Volume Profile
        if volume.value_area_high > 0:
            raw_resistances.append(volume.value_area_high)
        if volume.value_area_low > 0:
            raw_supports.append(volume.value_area_low)

        # Options OI-based levels
        if options is not None:
            if options.oi_support > 0:
                raw_supports.append(options.oi_support)
            if options.oi_resistance > 0:
                raw_resistances.append(options.oi_resistance)

        # Previous day high/low
        if len(price_df) >= 2:
            prev_high = float(price_df["high"].iloc[-2])
            prev_low = float(price_df["low"].iloc[-2])
            if prev_high > latest_close:
                raw_resistances.append(prev_high)
            if prev_low < latest_close:
                raw_supports.append(prev_low)

        # Cluster and take top 3
        supports = _cluster_levels(raw_supports)
        resistances = _cluster_levels(raw_resistances)

        # Return top 3 nearest to current price
        supports_below = sorted([s for s in supports if s < latest_close], reverse=True)[:3]
        supports_below.sort()  # ascending
        resistances_above = sorted([r for r in resistances if r > latest_close])[:3]

        return supports_below or [latest_close * 0.99], resistances_above or [latest_close * 1.01]

    # ------------------------------------------------------------------
    # Alert generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_alerts(
        trend: TrendSummary,
        momentum: MomentumSummary,
        volatility: VolatilitySummary,
        volume: VolumeSummary,
        options: Optional[OptionsSummary],
        smart_money: Optional[SmartMoneyScore],
        vix_value: Optional[float],
    ) -> list[Alert]:
        """Detect high-priority conditions and return alerts."""
        alerts: list[Alert] = []

        # RSI divergence
        if momentum.divergence_detected:
            div_type = momentum.rsi_divergence or "unknown"
            alerts.append(Alert(
                type="DIVERGENCE",
                severity="HIGH",
                message=f"RSI divergence detected: {div_type}",
                source="momentum",
            ))

        # OBV divergence
        if volume.obv_divergence is not None:
            alerts.append(Alert(
                type="DIVERGENCE",
                severity="MEDIUM",
                message=f"OBV divergence detected: {volume.obv_divergence}",
                source="volume",
            ))

        # Bollinger squeeze
        if volatility.bb_squeeze:
            alerts.append(Alert(
                type="BB_SQUEEZE",
                severity="HIGH",
                message="Bollinger Band squeeze detected — breakout imminent",
                source="volatility",
            ))

        # Breakout alert from volatility
        if volatility.breakout_alert:
            alerts.append(Alert(
                type="BREAKOUT_TRAP",
                severity="MEDIUM",
                message="Potential breakout detected from volatility contraction",
                source="volatility",
            ))

        # Reversal warning from momentum
        if momentum.reversal_warning is not None:
            alerts.append(Alert(
                type="REVERSAL_WARNING",
                severity="HIGH",
                message=f"Momentum reversal warning: {momentum.reversal_warning}",
                source="momentum",
            ))

        # Volume climax
        if volume.volume_ratio > 3.0:
            alerts.append(Alert(
                type="VOLUME_CLIMAX",
                severity="HIGH",
                message=f"Volume climax detected — volume ratio {volume.volume_ratio:.1f}x average",
                source="volume",
            ))

        # VIX extreme
        if vix_value is not None and vix_value > 30.0:
            alerts.append(Alert(
                type="VIX_EXTREME",
                severity="HIGH",
                message=f"India VIX at extreme level: {vix_value:.1f}",
                source="volatility",
            ))
        elif vix_value is not None and vix_value > 20.0:
            alerts.append(Alert(
                type="VIX_EXTREME",
                severity="MEDIUM",
                message=f"India VIX elevated: {vix_value:.1f}",
                source="volatility",
            ))

        # Options OI spike
        if options is not None and options.oi_change_signal != "NEUTRAL":
            alerts.append(Alert(
                type="OI_SPIKE",
                severity="MEDIUM",
                message=f"OI change signal: {options.oi_change_signal} — {options.dominant_buildup}",
                source="options",
            ))

        # Smart money signal
        if smart_money is not None and abs(smart_money.score) > 60:
            severity = "HIGH" if abs(smart_money.score) > 80 else "MEDIUM"
            alerts.append(Alert(
                type="SMART_MONEY_SIGNAL",
                severity=severity,
                message=f"Smart money {smart_money.smart_money_bias}: {smart_money.key_finding}",
                source="smart_money",
            ))

        # Death/golden cross
        if trend.death_cross:
            alerts.append(Alert(
                type="REVERSAL_WARNING",
                severity="HIGH",
                message="Death cross detected (EMA50 below EMA200)",
                source="trend",
            ))
        elif trend.golden_cross:
            alerts.append(Alert(
                type="REVERSAL_WARNING",
                severity="HIGH",
                message="Golden cross detected (EMA50 above EMA200)",
                source="trend",
            ))

        return alerts

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    @staticmethod
    def generate_reasoning(result: TechnicalAnalysisResult) -> str:
        """Build a human-readable explanation of the analysis result.

        Covers:
          - Overall signal and confidence
          - What each category voted and why (1 line each)
          - Any active alerts
          - Key levels to watch
          - Risk factors
        """
        lines: list[str] = []

        # Header
        lines.append(
            f"{result.overall_signal} signal (confidence {result.overall_confidence:.2f}):"
        )

        # Trend
        t = result.trend
        trend_detail = []
        if t.price_vs_ema20 == "BELOW":
            trend_detail.append("price below EMA20")
        if t.price_vs_ema50 == "BELOW":
            trend_detail.append("below EMA50")
        if t.price_vs_ema20 == "ABOVE":
            trend_detail.append("price above EMA20")
        if t.price_vs_ema50 == "ABOVE":
            trend_detail.append("above EMA50")
        if t.golden_cross:
            trend_detail.append("golden cross forming")
        if t.death_cross:
            trend_detail.append("death cross forming")
        detail_str = ", ".join(trend_detail) if trend_detail else t.ema_alignment
        lines.append(f"  - Trend: {t.trend_vote} ({detail_str})")

        # Momentum
        m = result.momentum
        mom_parts = [f"RSI {m.rsi_value:.0f}"]
        if m.divergence_detected:
            mom_parts.append(f"{m.rsi_divergence}")
        if m.overbought_consensus:
            mom_parts.append("overbought consensus")
        elif m.oversold_consensus:
            mom_parts.append("oversold consensus")
        lines.append(f"  - Momentum: {m.momentum_vote} ({', '.join(mom_parts)})")

        # Volume
        v = result.volume
        vol_parts = [f"VWAP {v.price_vs_vwap}", f"OBV {v.obv_trend}"]
        if v.obv_divergence:
            vol_parts.append(f"divergence: {v.obv_divergence}")
        lines.append(f"  - Volume: {v.volume_vote} ({', '.join(vol_parts)})")

        # Options
        if result.options is not None:
            o = result.options
            lines.append(
                f"  - Options: {o.options_vote} "
                f"(PCR {o.pcr:.2f}, max pain {o.max_pain:.0f}, "
                f"{o.dominant_buildup})"
            )
        else:
            lines.append("  - Options: N/A (no data)")

        # Quant
        q = result.quant
        quant_parts = [f"Z-score {q.zscore:.2f}"]
        if q.beta is not None:
            quant_parts.append(f"beta {q.beta:.2f}")
        lines.append(f"  - Quant: {q.quant_vote} ({', '.join(quant_parts)})")

        # Smart Money
        if result.smart_money is not None:
            sm = result.smart_money
            lines.append(
                f"  - Smart Money: {sm.smart_money_bias} "
                f"(score {sm.score:.0f}, {sm.key_finding})"
            )
        else:
            lines.append("  - Smart Money: N/A (insufficient data)")

        # Volatility (context, not direction)
        vol_summary = result.volatility
        lines.append(
            f"  - Volatility: {vol_summary.volatility_vote} "
            f"(ATR% {vol_summary.atr_pct:.2f}%, regime: {vol_summary.hv_regime})"
        )

        # Alerts
        if result.alerts:
            lines.append("  Alerts:")
            for alert in result.alerts:
                lines.append(f"    [{alert.severity}] {alert.message}")

        # Key levels
        lines.append(
            f"  Key levels: Support {result.immediate_support:.0f} | "
            f"Resistance {result.immediate_resistance:.0f}"
        )

        # Risk
        lines.append(
            f"  Risk: SL distance {result.suggested_stop_loss_distance:.0f}, "
            f"target distance {result.suggested_target_distance:.0f}, "
            f"position size modifier {result.position_size_modifier:.2f}"
        )

        if result.warnings:
            lines.append("  Caution: " + "; ".join(result.warnings))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data-quality helpers
# ---------------------------------------------------------------------------

# Fraction of recent bars that must carry non-zero volume for volume-based
# indicators to mean anything.
_MIN_VOLUME_BAR_FRACTION: float = 0.8
_VOLUME_LOOKBACK_BARS: int = 20


def _last_bar_timestamp(price_df: Optional[pd.DataFrame]) -> Optional[datetime]:
    """Timestamp of the newest bar as a datetime, or ``None`` if unavailable."""
    if price_df is None or price_df.empty or "timestamp" not in price_df.columns:
        return None
    try:
        ts = pd.to_datetime(price_df["timestamp"], errors="coerce").dropna()
        if ts.empty:
            return None
        return ts.max().to_pydatetime()
    except Exception:
        return None


def _has_real_volume(price_df: Optional[pd.DataFrame]) -> bool:
    """True when most of the recent bars have non-zero volume."""
    if price_df is None or price_df.empty or "volume" not in price_df.columns:
        return False
    recent = pd.to_numeric(price_df["volume"], errors="coerce").fillna(0).tail(_VOLUME_LOOKBACK_BARS)
    if recent.empty:
        return False
    return float((recent > 0).mean()) >= _MIN_VOLUME_BAR_FRACTION


# ---------------------------------------------------------------------------
# Empty / fallback summary constructors (used when an indicator fails)
# ---------------------------------------------------------------------------

def _empty_trend(index_id: str, timeframe: str, now: datetime) -> TrendSummary:
    return TrendSummary(
        index_id=index_id, timeframe=timeframe, timestamp=now,
        price_vs_ema20="BELOW", price_vs_ema50="BELOW", price_vs_ema200="BELOW",
        ema_alignment="MIXED", golden_cross=False, death_cross=False,
        macd_signal="NEUTRAL", macd_crossover=None, macd_histogram_trend="DECREASING",
        trend_strength="WEAK", trend_direction="BEARISH",
        trend_vote="NEUTRAL", trend_confidence=0.0,
    )


def _empty_momentum(now: datetime) -> MomentumSummary:
    return MomentumSummary(
        timestamp=now, rsi_value=50.0, rsi_zone="NEUTRAL", rsi_divergence=None,
        stochastic_k=50.0, stochastic_zone="NEUTRAL", stochastic_crossover=None,
        cci_value=0.0, cci_zone="NEUTRAL",
        momentum_vote="NEUTRAL", momentum_confidence=0.0,
        overbought_consensus=False, oversold_consensus=False,
        divergence_detected=False, reversal_warning=None,
    )


def _empty_volatility(now: datetime) -> VolatilitySummary:
    return VolatilitySummary(
        timestamp=now,
        bb_position="UPPER_ZONE", bb_squeeze=False, bb_bandwidth_percentile=50.0,
        atr_value=0.0, atr_pct=0.0, volatility_level="NORMAL",
        suggested_sl=0.0, suggested_target=0.0,
        hv_current=0.0, hv_regime="NORMAL", vix_regime=None,
        volatility_vote="NORMAL", volatility_confidence=0.5,
        position_size_modifier=1.0, breakout_alert=False, mean_reversion_setup=False,
    )


def _empty_volume(now: datetime) -> VolumeSummary:
    return VolumeSummary(
        timestamp=now,
        price_vs_vwap="AT_VWAP", vwap_zone="NEAR_VWAP",
        institutional_bias="NEUTRAL",
        obv_trend="FLAT", obv_divergence=None,
        accumulation_distribution="NEUTRAL",
        poc=0.0, value_area_high=0.0, value_area_low=0.0,
        in_value_area=True, volume_ratio=1.0,
        volume_confirms_price=False,
        volume_vote="NEUTRAL", volume_confidence=0.0,
    )


def _empty_quant(now: datetime) -> QuantSummary:
    return QuantSummary(
        timestamp=now,
        zscore=0.0, zscore_zone="FAIR_VALUE", mean_reversion_signal=None,
        beta=None, alpha=None, beta_interpretation=None,
        statistical_regime="NORMAL",
        quant_vote="NEUTRAL", quant_confidence=0.0,
    )
