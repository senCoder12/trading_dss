"""
Market Regime Detector — Phase 5.1 of the Decision Engine.

Answers: "What type of market are we in RIGHT NOW?"
This determines HOW the Decision Engine weighs different signals:
  - Trending market  → trust trend indicators, fade mean-reversion
  - Range-bound      → trust S/R and mean-reversion
  - Event day        → trust news, reduce position sizes
  - High volatility  → reduce positions, widen stops
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from config.settings import settings
from src.analysis.indicators.trend import TrendIndicators
from src.analysis.technical_aggregator import TechnicalAnalysisResult
from src.analysis.news.event_calendar import EventRegimeModifier
from src.analysis.anomaly.anomaly_aggregator import AnomalyDetectionResult
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_DEFAULT_MAX_TRADES = 5


@dataclass
class SignalWeights:
    """How much to trust each signal category.  All weights must sum to 1.0."""

    trend_weight: float = 0.20
    momentum_weight: float = 0.15
    options_weight: float = 0.25
    volume_weight: float = 0.15
    news_weight: float = 0.10
    smart_money_weight: float = 0.10
    anomaly_weight: float = 0.05

    def validate(self) -> bool:
        """Return True if weights sum to ~1.0 (within floating-point tolerance)."""
        total = (
            self.trend_weight
            + self.momentum_weight
            + self.options_weight
            + self.volume_weight
            + self.news_weight
            + self.smart_money_weight
            + self.anomaly_weight
        )
        return abs(total - 1.0) < 1e-6

    def normalise(self) -> None:
        """Scale all weights so they sum to exactly 1.0."""
        total = (
            self.trend_weight
            + self.momentum_weight
            + self.options_weight
            + self.volume_weight
            + self.news_weight
            + self.smart_money_weight
            + self.anomaly_weight
        )
        if total <= 0:
            return
        factor = 1.0 / total
        self.trend_weight *= factor
        self.momentum_weight *= factor
        self.options_weight *= factor
        self.volume_weight *= factor
        self.news_weight *= factor
        self.smart_money_weight *= factor
        self.anomaly_weight *= factor


@dataclass
class MarketRegime:
    """Full regime classification for a single index at a point in time."""

    index_id: str
    timestamp: datetime

    # Primary regime classification
    regime: str  # STRONG_TREND_UP / TREND_UP / RANGE_BOUND / TREND_DOWN
    #              STRONG_TREND_DOWN / VOLATILE_CHOPPY / EVENT_DRIVEN
    #              BREAKOUT / CRASH

    # Sub-classifications
    trend_regime: str       # STRONG_UP / UP / FLAT / DOWN / STRONG_DOWN
    volatility_regime: str  # LOW / NORMAL / HIGH / EXTREME
    event_regime: str       # NORMAL / PRE_EVENT / EVENT_DAY / POST_EVENT
    market_phase: str       # ACCUMULATION / MARKUP / DISTRIBUTION / MARKDOWN

    # Confidence
    regime_confidence: float      # 0–1
    regime_duration_bars: int     # how many bars this regime has persisted
    regime_changing: bool         # signs of regime transition

    # Signal weight adjustments — THE KEY OUTPUT
    weight_adjustments: SignalWeights

    # Risk adjustments
    position_size_multiplier: float  # 1.0 = normal
    stop_loss_multiplier: float      # 1.0 = normal ATR-based
    max_trades_today: int            # max concurrent signals

    # Human-readable
    description: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Classifies current market regime from technical, event, and anomaly data.

    Parameters
    ----------
    db : DatabaseManager
        Database for loading historical regime data.
    """

    # VIX thresholds (from settings or defaults)
    _VIX_LOW: float = 13.0
    _VIX_NORMAL_HIGH: float = 18.0
    _VIX_HIGH: float = 25.0
    _VIX_EXTREME: float = 35.0

    # Crash thresholds
    _CRASH_DROP_PCT: float = 3.0
    _CRASH_VOLUME_MULT: float = 2.0

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._trend_calc = TrendIndicators()
        self._regime_history: dict[str, list[dict]] = {}

        # Load thresholds from settings if available
        thresholds = settings.thresholds
        self._VIX_LOW = thresholds.vix_normal_threshold
        self._VIX_NORMAL_HIGH = thresholds.vix_elevated_threshold
        self._VIX_HIGH = thresholds.vix_panic_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_regime(
        self,
        index_id: str,
        price_df: pd.DataFrame,
        technical_result: TechnicalAnalysisResult,
        news_event_modifier: Optional[EventRegimeModifier] = None,
        anomaly_result: Optional[AnomalyDetectionResult] = None,
        vix_value: Optional[float] = None,
    ) -> MarketRegime:
        """Classify the current market regime for *index_id*.

        Parameters
        ----------
        index_id:
            Index identifier (e.g. ``"NIFTY50"``).
        price_df:
            Recent OHLCV DataFrame — at least 20 bars recommended.
        technical_result:
            Output from Phase 2 ``TechnicalAggregator``.
        news_event_modifier:
            Output from Phase 3 ``EventCalendar`` (may be ``None``).
        anomaly_result:
            Output from Phase 4 ``AnomalyAggregator`` (may be ``None``).
        vix_value:
            Latest India VIX reading (may be ``None``).
        """
        now = datetime.now(tz=_IST)
        warnings: list[str] = []

        # --- Handle insufficient data ---
        if price_df is None or len(price_df) < 20:
            warnings.append("Insufficient price data — defaulting to RANGE_BOUND")
            weights = self._weights_for_regime("RANGE_BOUND")
            return MarketRegime(
                index_id=index_id,
                timestamp=now,
                regime="RANGE_BOUND",
                trend_regime="FLAT",
                volatility_regime="NORMAL",
                event_regime="NORMAL",
                market_phase="ACCUMULATION",
                regime_confidence=0.2,
                regime_duration_bars=0,
                regime_changing=False,
                weight_adjustments=weights,
                position_size_multiplier=0.6,
                stop_loss_multiplier=1.0,
                max_trades_today=2,
                description="Insufficient data — conservative RANGE_BOUND default.",
                warnings=warnings,
            )

        # --- Check opening auction ---
        if self._is_opening_auction(now):
            warnings.append("Market just opened (<15 min) — OPENING_AUCTION, no signals yet")
            weights = SignalWeights()
            weights.normalise()
            return MarketRegime(
                index_id=index_id,
                timestamp=now,
                regime="RANGE_BOUND",
                trend_regime="FLAT",
                volatility_regime="NORMAL",
                event_regime="NORMAL",
                market_phase="ACCUMULATION",
                regime_confidence=0.1,
                regime_duration_bars=0,
                regime_changing=False,
                weight_adjustments=weights,
                position_size_multiplier=0.0,
                stop_loss_multiplier=1.0,
                max_trades_today=0,
                description="Opening auction — no signals generated yet.",
                warnings=warnings,
            )

        # ── Step 1: Trend Assessment ────────────────────────────────────
        trend_regime, adx_val, plus_di, minus_di = self._assess_trend(
            price_df, technical_result,
        )

        # ── Step 2: Volatility Assessment ───────────────────────────────
        volatility_regime = self._assess_volatility(
            vix_value, technical_result, warnings,
        )

        # ── Step 3: Event Check ─────────────────────────────────────────
        event_regime = self._assess_event(news_event_modifier)

        # ── Step 4: Special Regime Detection ────────────────────────────
        primary_regime = self._classify_primary_regime(
            trend_regime=trend_regime,
            volatility_regime=volatility_regime,
            event_regime=event_regime,
            adx_val=adx_val,
            plus_di=plus_di,
            minus_di=minus_di,
            price_df=price_df,
            technical_result=technical_result,
            anomaly_result=anomaly_result,
            vix_value=vix_value,
        )

        # ── Step 5: Wyckoff Phase ──────────────────────────────────────
        market_phase = self._detect_wyckoff_phase(
            trend_regime, adx_val, plus_di, minus_di, technical_result,
        )

        # ── Build regime warnings ───────────────────────────────────────
        if news_event_modifier and news_event_modifier.is_expiry_day:
            warnings.append("Expiry day — expect increased volatility after 2PM")
        if vix_value is not None and vix_value > self._VIX_NORMAL_HIGH:
            warnings.append(f"VIX elevated at {vix_value:.1f}")
        if anomaly_result and anomaly_result.risk_level in ("HIGH", "EXTREME"):
            warnings.append(f"Anomaly risk level: {anomaly_result.risk_level}")

        # ── Weights & risk params ───────────────────────────────────────
        weights = self._weights_for_regime(primary_regime)
        pos_mult, sl_mult, max_trades = self._risk_params_for_regime(primary_regime)

        # ── Confidence & duration ───────────────────────────────────────
        confidence = self._calculate_confidence(
            primary_regime, adx_val, volatility_regime, event_regime,
            vix_value, technical_result,
        )
        duration = self._estimate_duration(index_id, primary_regime)
        changing = self._detect_regime_change(
            index_id, primary_regime, adx_val, technical_result,
        )

        # ── Description ────────────────────────────────────────────────
        description = self._build_description(
            primary_regime, trend_regime, volatility_regime,
            event_regime, market_phase,
        )

        # ── Store in history ────────────────────────────────────────────
        self._store_regime(index_id, primary_regime, now)

        return MarketRegime(
            index_id=index_id,
            timestamp=now,
            regime=primary_regime,
            trend_regime=trend_regime,
            volatility_regime=volatility_regime,
            event_regime=event_regime,
            market_phase=market_phase,
            regime_confidence=round(confidence, 4),
            regime_duration_bars=duration,
            regime_changing=changing,
            weight_adjustments=weights,
            position_size_multiplier=round(pos_mult, 2),
            stop_loss_multiplier=round(sl_mult, 2),
            max_trades_today=max_trades,
            description=description,
            warnings=warnings,
        )

    def get_regime_history(
        self, index_id: str, days: int = 20,
    ) -> list[dict]:
        """Return cached regime classifications for recent days."""
        history = self._regime_history.get(index_id, [])
        cutoff = datetime.now(tz=_IST) - timedelta(days=days)
        return [r for r in history if r["timestamp"] >= cutoff]

    @staticmethod
    def is_safe_to_trade(regime: MarketRegime) -> tuple[bool, str]:
        """Check whether it is safe to generate new trading signals.

        Returns
        -------
        (safe, reason) : tuple[bool, str]
        """
        if regime.regime == "CRASH":
            return False, "CRASH regime — capital preservation mode, no new longs."
        if regime.regime == "VOLATILE_CHOPPY" and regime.volatility_regime in (
            "HIGH", "EXTREME",
        ):
            return (
                False,
                "VOLATILE_CHOPPY with high risk — stay out.",
            )
        if regime.position_size_multiplier <= 0.0:
            return False, "Opening auction — market not ready."
        vix_val = _extract_vix_from_warnings(regime.warnings)
        if vix_val is not None and vix_val > 35.0:
            return False, f"Extreme VIX ({vix_val:.1f}) — too dangerous to trade."
        return True, "Safe to trade with current regime adjustments."

    # ------------------------------------------------------------------
    # Step 1 — Trend Assessment
    # ------------------------------------------------------------------

    def _assess_trend(
        self,
        price_df: pd.DataFrame,
        technical_result: TechnicalAnalysisResult,
    ) -> tuple[str, float, float, float]:
        """Return (trend_regime, adx_value, plus_di, minus_di)."""
        trend = technical_result.trend

        # Map categorical trend_strength back to approximate ADX midpoints
        adx_map = {
            "WEAK": 12.0,
            "EMERGING": 22.0,
            "STRONG": 35.0,
            "VERY_STRONG": 60.0,
            "EXTREME": 80.0,
        }
        adx_val = adx_map.get(trend.trend_strength, 15.0)

        # Approximate DI values from trend direction
        if trend.trend_direction == "BULLISH":
            plus_di, minus_di = adx_val * 0.8, adx_val * 0.3
        else:
            plus_di, minus_di = adx_val * 0.3, adx_val * 0.8

        # Try to compute raw ADX from price data for more precision
        if len(price_df) >= 28:
            try:
                adx_result = self._trend_calc.calculate_adx(price_df)
                valid_adx = adx_result.adx.dropna()
                valid_plus = adx_result.plus_di.dropna()
                valid_minus = adx_result.minus_di.dropna()
                if len(valid_adx) > 0:
                    adx_val = float(valid_adx.iloc[-1])
                if len(valid_plus) > 0:
                    plus_di = float(valid_plus.iloc[-1])
                if len(valid_minus) > 0:
                    minus_di = float(valid_minus.iloc[-1])
            except Exception:
                logger.debug("Raw ADX computation failed; using TrendSummary mapping.")

        # Classify trend regime
        if adx_val >= 40 and plus_di > minus_di:
            trend_regime = "STRONG_UP"
        elif adx_val >= 25 and plus_di > minus_di:
            trend_regime = "UP"
        elif adx_val >= 40 and minus_di > plus_di:
            trend_regime = "STRONG_DOWN"
        elif adx_val >= 25 and minus_di > plus_di:
            trend_regime = "DOWN"
        else:
            trend_regime = "FLAT"

        # Confirm with EMA alignment
        ema_alignment = trend.ema_alignment  # BULLISH / BEARISH / MIXED
        if trend_regime in ("STRONG_UP", "UP") and ema_alignment == "BEARISH":
            trend_regime = "FLAT"  # contradictory — downgrade
        if trend_regime in ("STRONG_DOWN", "DOWN") and ema_alignment == "BULLISH":
            trend_regime = "FLAT"

        return trend_regime, adx_val, plus_di, minus_di

    # ------------------------------------------------------------------
    # Step 2 — Volatility Assessment
    # ------------------------------------------------------------------

    def _assess_volatility(
        self,
        vix_value: Optional[float],
        technical_result: TechnicalAnalysisResult,
        warnings: list[str],
    ) -> str:
        """Return volatility regime: LOW / NORMAL / HIGH / EXTREME."""
        vix_regime: Optional[str] = None
        atr_regime: Optional[str] = None

        # VIX-based
        if vix_value is not None:
            if vix_value < self._VIX_LOW:
                vix_regime = "LOW"
            elif vix_value <= self._VIX_NORMAL_HIGH:
                vix_regime = "NORMAL"
            elif vix_value <= self._VIX_HIGH:
                vix_regime = "HIGH"
            else:
                vix_regime = "EXTREME"

        # ATR-based from VolatilitySummary
        vol = technical_result.volatility
        vol_level = vol.volatility_level  # from VolatilitySummary
        atr_regime = _map_volatility_level(vol_level)

        # Bollinger bandwidth as additional confirmation
        if vol.bb_bandwidth_percentile >= 90:
            atr_regime = _max_vol(atr_regime or "NORMAL", "HIGH")

        # Combine
        if vix_regime is not None and atr_regime is not None:
            return _max_vol(vix_regime, atr_regime)
        if vix_regime is not None:
            return vix_regime
        if atr_regime is not None:
            warnings.append("VIX unavailable — volatility from ATR/Bollinger only.")
            return atr_regime
        warnings.append("No volatility data — assuming NORMAL.")
        return "NORMAL"

    # ------------------------------------------------------------------
    # Step 3 — Event Check
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_event(
        modifier: Optional[EventRegimeModifier],
    ) -> str:
        """Return event regime: NORMAL / PRE_EVENT / EVENT_DAY / POST_EVENT."""
        if modifier is None:
            return "NORMAL"
        if modifier.is_event_day:
            return "EVENT_DAY"
        if modifier.is_pre_event:
            return "PRE_EVENT"
        return "NORMAL"

    # ------------------------------------------------------------------
    # Step 4 — Primary Regime Classification
    # ------------------------------------------------------------------

    def _classify_primary_regime(
        self,
        *,
        trend_regime: str,
        volatility_regime: str,
        event_regime: str,
        adx_val: float,
        plus_di: float,
        minus_di: float,
        price_df: pd.DataFrame,
        technical_result: TechnicalAnalysisResult,
        anomaly_result: Optional[AnomalyDetectionResult],
        vix_value: Optional[float],
    ) -> str:
        """Determine the single primary regime string."""
        # --- CRASH detection (highest priority) ---
        if self._is_crash(price_df, vix_value, anomaly_result):
            return "CRASH"

        # --- EVENT_DRIVEN override ---
        if event_regime == "EVENT_DAY":
            return "EVENT_DRIVEN"

        # --- BREAKOUT detection ---
        if self._is_breakout(adx_val, technical_result):
            return "BREAKOUT"

        # --- VOLATILE_CHOPPY ---
        if self._is_volatile_choppy(
            adx_val, volatility_regime, anomaly_result,
        ):
            return "VOLATILE_CHOPPY"

        # --- Trend-based regimes ---
        if trend_regime == "STRONG_UP":
            return "STRONG_TREND_UP"
        if trend_regime == "UP":
            return "TREND_UP"
        if trend_regime == "STRONG_DOWN":
            return "STRONG_TREND_DOWN"
        if trend_regime == "DOWN":
            return "TREND_DOWN"

        return "RANGE_BOUND"

    def _is_crash(
        self,
        price_df: pd.DataFrame,
        vix_value: Optional[float],
        anomaly_result: Optional[AnomalyDetectionResult],
    ) -> bool:
        """Detect crash conditions."""
        if len(price_df) < 2:
            return False

        close = price_df["close"]
        prev_close = float(close.iloc[-2])
        current = float(close.iloc[-1])
        if prev_close <= 0:
            return False

        drop_pct = (prev_close - current) / prev_close * 100

        vix_high = vix_value is not None and vix_value > self._VIX_HIGH

        vol = price_df["volume"] if "volume" in price_df.columns else None
        volume_spike = False
        if vol is not None and len(vol) >= 21:
            avg_vol = float(vol.iloc[-21:-1].mean())
            if avg_vol > 0:
                volume_spike = float(vol.iloc[-1]) > avg_vol * self._CRASH_VOLUME_MULT

        bearish_anomalies = False
        if anomaly_result is not None:
            bearish_anomalies = anomaly_result.high_severity_count >= 2

        return (
            drop_pct >= self._CRASH_DROP_PCT
            and vix_high
            and volume_spike
            and bearish_anomalies
        )

    @staticmethod
    def _is_breakout(
        adx_val: float,
        technical_result: TechnicalAnalysisResult,
    ) -> bool:
        """Detect breakout from Bollinger squeeze."""
        vol = technical_result.volatility

        # Bollinger squeeze was recently active
        if not vol.breakout_alert:
            return False

        # ADX starting to rise from below 20
        if adx_val > 30:
            return False  # already well into a trend, not a fresh breakout

        # Volume confirms
        if technical_result.volume is not None:
            if technical_result.volume.volume_ratio < 1.2:
                return False  # no volume confirmation

        return True

    @staticmethod
    def _is_volatile_choppy(
        adx_val: float,
        volatility_regime: str,
        anomaly_result: Optional[AnomalyDetectionResult],
    ) -> bool:
        """Detect volatile but trendless (choppy) conditions."""
        if adx_val >= 20:
            return False  # there is some trend
        if volatility_regime not in ("HIGH", "EXTREME"):
            return False

        # Extra confirmation from anomaly data if available
        if anomaly_result is not None and anomaly_result.high_severity_count >= 1:
            return True

        # Without anomaly data, vol + no trend is enough
        return True

    # ------------------------------------------------------------------
    # Step 5 — Wyckoff Phase Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_wyckoff_phase(
        trend_regime: str,
        adx_val: float,
        plus_di: float,
        minus_di: float,
        technical_result: TechnicalAnalysisResult,
    ) -> str:
        """Approximate the Wyckoff market-cycle phase."""
        trend = technical_result.trend
        vol_summary = technical_result.volume

        obv_rising = (
            vol_summary is not None
            and vol_summary.obv_trend == "RISING"
        )
        obv_declining = (
            vol_summary is not None
            and vol_summary.obv_trend == "FALLING"
        )

        smart_money = technical_result.smart_money
        smfi_accumulation = (
            smart_money is not None
            and smart_money.smart_money_bias in ("BULLISH", "STRONGLY_BULLISH")
        )
        smfi_distribution = (
            smart_money is not None
            and smart_money.smart_money_bias in ("BEARISH", "STRONGLY_BEARISH")
        )

        price_below_ema200 = trend.price_vs_ema200 == "BELOW"
        price_above_ema200 = trend.price_vs_ema200 == "ABOVE"

        # ACCUMULATION: after decline, range-bound, smart money buying
        if price_below_ema200 and trend_regime == "FLAT" and (
            smfi_accumulation or obv_rising
        ):
            return "ACCUMULATION"

        # MARKUP: strong uptrend
        if trend_regime in ("STRONG_UP", "UP") and adx_val >= 25 and plus_di > minus_di:
            return "MARKUP"

        # DISTRIBUTION: after rally, range-bound at highs, smart money selling
        if price_above_ema200 and trend_regime == "FLAT" and (
            smfi_distribution or obv_declining
        ):
            return "DISTRIBUTION"

        # MARKDOWN: strong downtrend
        if trend_regime in ("STRONG_DOWN", "DOWN") and adx_val >= 25 and minus_di > plus_di:
            return "MARKDOWN"

        # Default: match trend
        if trend_regime in ("STRONG_UP", "UP"):
            return "MARKUP"
        if trend_regime in ("STRONG_DOWN", "DOWN"):
            return "MARKDOWN"
        return "ACCUMULATION"

    # ------------------------------------------------------------------
    # Weight & risk parameter tables
    # ------------------------------------------------------------------

    @staticmethod
    def _weights_for_regime(regime: str) -> SignalWeights:
        """Return signal weights tuned for the given regime."""
        tables: dict[str, dict[str, float]] = {
            "STRONG_TREND_UP": dict(
                trend_weight=0.30, momentum_weight=0.10, options_weight=0.20,
                volume_weight=0.15, news_weight=0.10, smart_money_weight=0.10,
                anomaly_weight=0.05,
            ),
            "STRONG_TREND_DOWN": dict(
                trend_weight=0.30, momentum_weight=0.10, options_weight=0.20,
                volume_weight=0.15, news_weight=0.10, smart_money_weight=0.10,
                anomaly_weight=0.05,
            ),
            "TREND_UP": dict(
                trend_weight=0.25, momentum_weight=0.15, options_weight=0.20,
                volume_weight=0.15, news_weight=0.10, smart_money_weight=0.10,
                anomaly_weight=0.05,
            ),
            "TREND_DOWN": dict(
                trend_weight=0.25, momentum_weight=0.15, options_weight=0.20,
                volume_weight=0.15, news_weight=0.10, smart_money_weight=0.10,
                anomaly_weight=0.05,
            ),
            "RANGE_BOUND": dict(
                trend_weight=0.10, momentum_weight=0.20, options_weight=0.25,
                volume_weight=0.15, news_weight=0.10, smart_money_weight=0.15,
                anomaly_weight=0.05,
            ),
            "EVENT_DRIVEN": dict(
                trend_weight=0.10, momentum_weight=0.05, options_weight=0.20,
                volume_weight=0.10, news_weight=0.35, smart_money_weight=0.10,
                anomaly_weight=0.10,
            ),
            "VOLATILE_CHOPPY": dict(
                trend_weight=0.15, momentum_weight=0.15, options_weight=0.20,
                volume_weight=0.15, news_weight=0.15, smart_money_weight=0.10,
                anomaly_weight=0.10,
            ),
            "CRASH": dict(
                trend_weight=0.10, momentum_weight=0.05, options_weight=0.10,
                volume_weight=0.10, news_weight=0.30, smart_money_weight=0.15,
                anomaly_weight=0.20,
            ),
            "BREAKOUT": dict(
                trend_weight=0.25, momentum_weight=0.10, options_weight=0.20,
                volume_weight=0.25, news_weight=0.05, smart_money_weight=0.10,
                anomaly_weight=0.05,
            ),
        }
        params = tables.get(regime, tables["RANGE_BOUND"])
        w = SignalWeights(**params)
        w.normalise()
        return w

    @staticmethod
    def _risk_params_for_regime(
        regime: str,
    ) -> tuple[float, float, int]:
        """Return (position_size_multiplier, stop_loss_multiplier, max_trades)."""
        table: dict[str, tuple[float, float, int]] = {
            "STRONG_TREND_UP":   (1.2, 1.0, _DEFAULT_MAX_TRADES),
            "STRONG_TREND_DOWN": (1.2, 1.0, _DEFAULT_MAX_TRADES),
            "TREND_UP":          (1.0, 1.0, _DEFAULT_MAX_TRADES),
            "TREND_DOWN":        (1.0, 1.0, _DEFAULT_MAX_TRADES),
            "RANGE_BOUND":       (0.8, 0.8, 3),
            "EVENT_DRIVEN":      (0.5, 1.5, 2),
            "VOLATILE_CHOPPY":   (0.4, 1.5, 1),
            "CRASH":             (0.3, 2.0, 1),
            "BREAKOUT":          (1.0, 1.2, _DEFAULT_MAX_TRADES),
        }
        return table.get(regime, (0.8, 1.0, 3))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_opening_auction(now: datetime) -> bool:
        """Return True if the current time is within 15 min of market open."""
        market_open_str = settings.market_hours.market_open  # "09:15"
        parts = market_open_str.split(":")
        market_open = now.replace(
            hour=int(parts[0]), minute=int(parts[1]), second=0, microsecond=0,
        )
        return market_open <= now < market_open + timedelta(minutes=15)

    def _calculate_confidence(
        self,
        regime: str,
        adx_val: float,
        volatility_regime: str,
        event_regime: str,
        vix_value: Optional[float],
        technical_result: TechnicalAnalysisResult,
    ) -> float:
        """Heuristic confidence in the regime classification (0–1)."""
        confidence = 0.5

        # Strong ADX → higher confidence in trend regimes
        if regime in ("STRONG_TREND_UP", "STRONG_TREND_DOWN") and adx_val >= 40:
            confidence = 0.85
        elif regime in ("TREND_UP", "TREND_DOWN") and adx_val >= 25:
            confidence = 0.70
        elif regime == "RANGE_BOUND":
            confidence = 0.60
        elif regime == "EVENT_DRIVEN":
            confidence = 0.75
        elif regime == "CRASH":
            confidence = 0.90
        elif regime == "BREAKOUT":
            confidence = 0.65
        elif regime == "VOLATILE_CHOPPY":
            confidence = 0.55

        # EMA alignment confirmation boosts confidence
        if technical_result.trend.ema_alignment in ("BULLISH", "BEARISH"):
            confidence = min(1.0, confidence + 0.05)

        # Data completeness affects confidence
        confidence *= technical_result.data_completeness

        return min(1.0, max(0.0, confidence))

    def _estimate_duration(self, index_id: str, regime: str) -> int:
        """Estimate how many bars the current regime has persisted."""
        history = self._regime_history.get(index_id, [])
        count = 0
        for entry in reversed(history):
            if entry["regime"] == regime:
                count += 1
            else:
                break
        return count

    def _detect_regime_change(
        self,
        index_id: str,
        current_regime: str,
        adx_val: float,
        technical_result: TechnicalAnalysisResult,
    ) -> bool:
        """Return True if there are signs of regime transition."""
        history = self._regime_history.get(index_id, [])
        if history and history[-1]["regime"] != current_regime:
            return True

        # ADX near transition thresholds
        if 18 <= adx_val <= 22:
            return True

        # EMA alignment is MIXED → transitional
        if technical_result.trend.ema_alignment == "MIXED":
            return True

        return False

    def _store_regime(
        self, index_id: str, regime: str, timestamp: datetime,
    ) -> None:
        """Append to in-memory regime history."""
        if index_id not in self._regime_history:
            self._regime_history[index_id] = []
        self._regime_history[index_id].append({
            "regime": regime,
            "timestamp": timestamp,
        })
        # Keep last 100 entries per index
        if len(self._regime_history[index_id]) > 100:
            self._regime_history[index_id] = self._regime_history[index_id][-100:]

    @staticmethod
    def _build_description(
        regime: str,
        trend_regime: str,
        volatility_regime: str,
        event_regime: str,
        market_phase: str,
    ) -> str:
        """Build a human-readable regime description."""
        parts: list[str] = []

        regime_labels = {
            "STRONG_TREND_UP": "Strong uptrend",
            "TREND_UP": "Moderate uptrend",
            "RANGE_BOUND": "Range-bound / consolidating",
            "TREND_DOWN": "Moderate downtrend",
            "STRONG_TREND_DOWN": "Strong downtrend",
            "VOLATILE_CHOPPY": "Volatile and choppy — no clear direction",
            "EVENT_DRIVEN": "Event-driven regime — news dominates",
            "BREAKOUT": "Breakout in progress",
            "CRASH": "CRASH — capital preservation mode",
        }
        parts.append(regime_labels.get(regime, regime))

        vol_labels = {
            "LOW": "low volatility",
            "NORMAL": "normal volatility",
            "HIGH": "elevated volatility",
            "EXTREME": "extreme volatility",
        }
        parts.append(f"with {vol_labels.get(volatility_regime, volatility_regime)}")

        phase_labels = {
            "ACCUMULATION": "Favor mean-reversion and accumulation entries.",
            "MARKUP": "Favor trend-following entries.",
            "DISTRIBUTION": "Watch for distribution — reduce exposure.",
            "MARKDOWN": "Favor short entries or stay defensive.",
        }
        parts.append(phase_labels.get(market_phase, ""))

        if event_regime == "EVENT_DAY":
            parts.append("Event day — prioritise news signals.")
        elif event_regime == "PRE_EVENT":
            parts.append("Pre-event caution in effect.")

        return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _map_volatility_level(level: str) -> str:
    """Map VolatilitySummary.volatility_level to our regime categories."""
    level_up = level.upper()
    if level_up in ("LOW", "NORMAL", "HIGH", "EXTREME"):
        return level_up
    if "LOW" in level_up or "CALM" in level_up:
        return "LOW"
    if "EXTREME" in level_up or "VERY" in level_up:
        return "EXTREME"
    if "HIGH" in level_up or "ELEVATED" in level_up:
        return "HIGH"
    return "NORMAL"


def _max_vol(a: str, b: str) -> str:
    """Return the higher of two volatility regimes."""
    order = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}
    return a if order.get(a, 1) >= order.get(b, 1) else b


def _extract_vix_from_warnings(warnings: list[str]) -> Optional[float]:
    """Try to parse the VIX value from warning messages."""
    for w in warnings:
        if "VIX elevated" in w or "VIX" in w:
            try:
                # "VIX elevated at 32.5"
                parts = w.split()
                for p in parts:
                    try:
                        val = float(p)
                        if 5.0 < val < 100.0:
                            return val
                    except ValueError:
                        continue
            except Exception:
                pass
    return None
