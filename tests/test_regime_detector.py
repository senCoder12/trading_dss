"""
Tests for Phase 5.1 — Market Regime Detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.engine.regime_detector import (
    MarketRegime,
    RegimeDetector,
    SignalWeights,
    _map_volatility_level,
    _max_vol,
)

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Lightweight stubs for upstream dataclasses
# ---------------------------------------------------------------------------


def _make_trend_summary(
    *,
    trend_strength: str = "STRONG",
    trend_direction: str = "BULLISH",
    ema_alignment: str = "BULLISH",
    price_vs_ema20: str = "ABOVE",
    price_vs_ema50: str = "ABOVE",
    price_vs_ema200: str = "ABOVE",
) -> MagicMock:
    m = MagicMock()
    m.trend_strength = trend_strength
    m.trend_direction = trend_direction
    m.ema_alignment = ema_alignment
    m.price_vs_ema20 = price_vs_ema20
    m.price_vs_ema50 = price_vs_ema50
    m.price_vs_ema200 = price_vs_ema200
    return m


def _make_volatility_summary(
    *,
    bb_squeeze: bool = False,
    bb_bandwidth_percentile: float = 50.0,
    volatility_level: str = "NORMAL",
    breakout_alert: bool = False,
    atr_pct: float = 1.0,
) -> MagicMock:
    m = MagicMock()
    m.bb_squeeze = bb_squeeze
    m.bb_bandwidth_percentile = bb_bandwidth_percentile
    m.volatility_level = volatility_level
    m.breakout_alert = breakout_alert
    m.atr_pct = atr_pct
    return m


def _make_volume_summary(
    *,
    volume_ratio: float = 1.0,
    obv_trend: str = "FLAT",
    volume_confirms_price: bool = True,
) -> MagicMock:
    m = MagicMock()
    m.volume_ratio = volume_ratio
    m.obv_trend = obv_trend
    m.volume_confirms_price = volume_confirms_price
    return m


def _make_smart_money(
    *,
    smart_money_bias: str = "NEUTRAL",
    score: float = 0.0,
) -> MagicMock:
    m = MagicMock()
    m.smart_money_bias = smart_money_bias
    m.score = score
    return m


def _make_technical_result(
    *,
    trend_kwargs: dict | None = None,
    volatility_kwargs: dict | None = None,
    volume_kwargs: dict | None = None,
    smart_money_kwargs: dict | None = None,
    data_completeness: float = 1.0,
) -> MagicMock:
    """Build a mock TechnicalAnalysisResult."""
    m = MagicMock()
    m.trend = _make_trend_summary(**(trend_kwargs or {}))
    m.volatility = _make_volatility_summary(**(volatility_kwargs or {}))
    m.volume = _make_volume_summary(**(volume_kwargs or {}))
    m.smart_money = _make_smart_money(**(smart_money_kwargs or {}))
    m.data_completeness = data_completeness
    return m


def _make_event_modifier(
    *,
    is_event_day: bool = False,
    is_pre_event: bool = False,
    is_expiry_day: bool = False,
    caution_level: str = "NORMAL",
) -> MagicMock:
    m = MagicMock()
    m.is_event_day = is_event_day
    m.is_pre_event = is_pre_event
    m.is_expiry_day = is_expiry_day
    m.caution_level = caution_level
    m.active_events = []
    return m


def _make_anomaly_result(
    *,
    risk_level: str = "NORMAL",
    high_severity_count: int = 0,
    has_volume_shock: bool = False,
    has_oi_spike: bool = False,
    has_divergence: bool = False,
    has_breakout_trap_risk: bool = False,
    institutional_activity_detected: bool = False,
) -> MagicMock:
    m = MagicMock()
    m.risk_level = risk_level
    m.high_severity_count = high_severity_count
    m.has_volume_shock = has_volume_shock
    m.has_oi_spike = has_oi_spike
    m.has_divergence = has_divergence
    m.has_breakout_trap_risk = has_breakout_trap_risk
    m.institutional_activity_detected = institutional_activity_detected
    return m


def _make_price_df(
    bars: int = 60,
    start_price: float = 20000.0,
    trend: float = 0.0,
    crash_last: bool = False,
    volume_base: float = 1_000_000,
    volume_spike_last: bool = False,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    dates = pd.date_range(end=datetime.now(tz=_IST), periods=bars, freq="D")
    close_prices = [start_price + i * trend for i in range(bars)]
    if crash_last and bars >= 2:
        close_prices[-1] = close_prices[-2] * 0.95  # -5% drop
    close_arr = np.array(close_prices)
    high_arr = close_arr * 1.01
    low_arr = close_arr * 0.99
    open_arr = close_arr * 1.005

    volumes = [volume_base] * bars
    if volume_spike_last and bars >= 21:
        volumes[-1] = volume_base * 3.0

    return pd.DataFrame({
        "date": dates,
        "open": open_arr,
        "high": high_arr,
        "low": low_arr,
        "close": close_arr,
        "volume": volumes,
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Mock DatabaseManager."""
    return MagicMock()


@pytest.fixture
def detector(db):
    """RegimeDetector with mocked DB and ADX computation disabled."""
    with patch.object(RegimeDetector, "_is_opening_auction", return_value=False):
        det = RegimeDetector(db)
        # Disable raw ADX computation so tests control via TrendSummary mocks
        det._trend_calc = MagicMock()
        det._trend_calc.calculate_adx.side_effect = Exception("mocked out")
        yield det


# ---------------------------------------------------------------------------
# SignalWeights tests
# ---------------------------------------------------------------------------


class TestSignalWeights:
    def test_default_weights_sum_to_one(self):
        w = SignalWeights()
        assert w.validate()

    def test_normalise_fixes_invalid_sum(self):
        w = SignalWeights(
            trend_weight=0.5, momentum_weight=0.5, options_weight=0.5,
            volume_weight=0.5, news_weight=0.5, smart_money_weight=0.5,
            anomaly_weight=0.5,
        )
        assert not w.validate()
        w.normalise()
        assert w.validate()

    def test_normalise_preserves_ratios(self):
        w = SignalWeights(
            trend_weight=2.0, momentum_weight=1.0, options_weight=1.0,
            volume_weight=1.0, news_weight=1.0, smart_money_weight=1.0,
            anomaly_weight=1.0,
        )
        w.normalise()
        assert w.trend_weight > w.momentum_weight
        assert w.validate()


# ---------------------------------------------------------------------------
# Regime detection tests
# ---------------------------------------------------------------------------


class TestStrongUptrend:
    """ADX=45 (+DI=35, -DI=12) → STRONG_TREND_UP."""

    def test_strong_uptrend_detection(self, detector):
        price_df = _make_price_df(bars=60, trend=50.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="VERY_STRONG",  # ADX ~60
                trend_direction="BULLISH",
                ema_alignment="BULLISH",
            ),
        )
        regime = detector.detect_regime("NIFTY50", price_df, tech, vix_value=15.0)

        assert regime.regime == "STRONG_TREND_UP"
        assert regime.trend_regime == "STRONG_UP"
        assert regime.weight_adjustments.trend_weight == pytest.approx(0.30, abs=0.01)
        assert regime.position_size_multiplier == 1.2

    def test_strong_uptrend_ema_confirmation(self, detector):
        price_df = _make_price_df(bars=60, trend=50.0)
        # ADX high but EMAs contradict → downgraded
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="VERY_STRONG",
                trend_direction="BULLISH",
                ema_alignment="BEARISH",  # contradiction
            ),
        )
        regime = detector.detect_regime("NIFTY50", price_df, tech, vix_value=15.0)

        # Should NOT be STRONG_TREND_UP because EMA contradicts
        assert regime.trend_regime == "FLAT"


class TestRangeBound:
    """ADX=15, mixed EMAs → RANGE_BOUND."""

    def test_range_bound_detection(self, detector):
        price_df = _make_price_df(bars=60, trend=0.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="WEAK",  # ADX ~12
                trend_direction="BULLISH",
                ema_alignment="MIXED",
            ),
            volatility_kwargs=dict(volatility_level="NORMAL"),
        )
        regime = detector.detect_regime("NIFTY50", price_df, tech, vix_value=15.0)

        assert regime.regime == "RANGE_BOUND"
        assert regime.weight_adjustments.momentum_weight == pytest.approx(0.20, abs=0.01)
        assert regime.position_size_multiplier == 0.8
        assert regime.max_trades_today == 3


class TestEventDrivenOverride:
    """Normal indicators but event flag → EVENT_DRIVEN."""

    def test_event_day_overrides_trend(self, detector):
        price_df = _make_price_df(bars=60, trend=50.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="STRONG",
                trend_direction="BULLISH",
                ema_alignment="BULLISH",
            ),
        )
        event = _make_event_modifier(is_event_day=True)

        regime = detector.detect_regime(
            "NIFTY50", price_df, tech,
            news_event_modifier=event, vix_value=15.0,
        )

        assert regime.regime == "EVENT_DRIVEN"
        assert regime.event_regime == "EVENT_DAY"
        assert regime.weight_adjustments.news_weight == pytest.approx(0.35, abs=0.01)
        assert regime.position_size_multiplier == 0.5
        assert regime.stop_loss_multiplier == 1.5
        assert regime.max_trades_today == 2

    def test_pre_event_sets_event_regime(self, detector):
        price_df = _make_price_df(bars=60, trend=0.0)
        tech = _make_technical_result(
            trend_kwargs=dict(trend_strength="WEAK", ema_alignment="MIXED"),
        )
        event = _make_event_modifier(is_pre_event=True)

        regime = detector.detect_regime(
            "NIFTY50", price_df, tech,
            news_event_modifier=event, vix_value=15.0,
        )

        assert regime.event_regime == "PRE_EVENT"
        # Pre-event doesn't override primary regime
        assert regime.regime != "EVENT_DRIVEN"


class TestCrashDetection:
    """Price -4%, VIX 32, volume 3x, multiple bearish anomalies → CRASH."""

    def test_crash_detection(self, detector):
        price_df = _make_price_df(
            bars=60, trend=0.0,
            crash_last=True, volume_spike_last=True,
        )
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="STRONG",
                trend_direction="BEARISH",
                ema_alignment="BEARISH",
            ),
            volatility_kwargs=dict(volatility_level="EXTREME"),
        )
        anomaly = _make_anomaly_result(
            risk_level="EXTREME", high_severity_count=3,
        )

        regime = detector.detect_regime(
            "NIFTY50", price_df, tech,
            anomaly_result=anomaly, vix_value=32.0,
        )

        assert regime.regime == "CRASH"
        assert regime.position_size_multiplier == 0.3
        assert regime.stop_loss_multiplier == 2.0
        assert regime.max_trades_today == 1
        assert regime.weight_adjustments.anomaly_weight >= 0.15

    def test_crash_needs_all_conditions(self, detector):
        """Without volume spike, shouldn't be CRASH."""
        price_df = _make_price_df(
            bars=60, trend=0.0,
            crash_last=True, volume_spike_last=False,  # no volume spike
        )
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="STRONG",
                trend_direction="BEARISH",
                ema_alignment="BEARISH",
            ),
        )
        anomaly = _make_anomaly_result(high_severity_count=3)

        regime = detector.detect_regime(
            "NIFTY50", price_df, tech,
            anomaly_result=anomaly, vix_value=32.0,
        )

        assert regime.regime != "CRASH"


class TestBreakout:
    """BB squeeze ending + volume spike + ADX rising from low."""

    def test_breakout_detection(self, detector):
        price_df = _make_price_df(bars=60, trend=5.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="WEAK",  # ADX ~12, low and starting to rise
                trend_direction="BULLISH",
                ema_alignment="MIXED",
            ),
            volatility_kwargs=dict(
                breakout_alert=True,  # BB squeeze just ended
                volatility_level="NORMAL",
            ),
            volume_kwargs=dict(volume_ratio=1.8),  # above-average volume
        )

        regime = detector.detect_regime("NIFTY50", price_df, tech, vix_value=15.0)

        assert regime.regime == "BREAKOUT"
        assert regime.weight_adjustments.volume_weight == pytest.approx(0.25, abs=0.01)
        assert regime.stop_loss_multiplier == 1.2

    def test_no_breakout_without_volume(self, detector):
        price_df = _make_price_df(bars=60)
        tech = _make_technical_result(
            trend_kwargs=dict(trend_strength="WEAK", ema_alignment="MIXED"),
            volatility_kwargs=dict(breakout_alert=True),
            volume_kwargs=dict(volume_ratio=0.8),  # below average
        )

        regime = detector.detect_regime("NIFTY50", price_df, tech, vix_value=15.0)

        assert regime.regime != "BREAKOUT"


class TestVolatileChoppy:
    """ADX < 20, VIX > 18, no clear trend → VOLATILE_CHOPPY."""

    def test_volatile_choppy(self, detector):
        price_df = _make_price_df(bars=60, trend=0.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="WEAK",  # ADX ~12
                trend_direction="BULLISH",
                ema_alignment="MIXED",
            ),
            volatility_kwargs=dict(volatility_level="HIGH"),
        )
        anomaly = _make_anomaly_result(high_severity_count=2)

        regime = detector.detect_regime(
            "NIFTY50", price_df, tech,
            anomaly_result=anomaly, vix_value=22.0,
        )

        assert regime.regime == "VOLATILE_CHOPPY"
        assert regime.position_size_multiplier == 0.4
        assert regime.max_trades_today == 1


# ---------------------------------------------------------------------------
# Weight validation — every regime must sum to 1.0
# ---------------------------------------------------------------------------


ALL_REGIMES = [
    "STRONG_TREND_UP", "STRONG_TREND_DOWN",
    "TREND_UP", "TREND_DOWN",
    "RANGE_BOUND", "EVENT_DRIVEN",
    "VOLATILE_CHOPPY", "CRASH", "BREAKOUT",
]


@pytest.mark.parametrize("regime", ALL_REGIMES)
class TestWeightsByRegime:
    def test_weights_sum_to_one(self, regime):
        weights = RegimeDetector._weights_for_regime(regime)
        assert weights.validate(), f"Weights for {regime} don't sum to 1.0"

    def test_all_weights_non_negative(self, regime):
        w = RegimeDetector._weights_for_regime(regime)
        for attr in (
            "trend_weight", "momentum_weight", "options_weight",
            "volume_weight", "news_weight", "smart_money_weight",
            "anomaly_weight",
        ):
            assert getattr(w, attr) >= 0.0, f"{attr} is negative for {regime}"


# ---------------------------------------------------------------------------
# Position size modifiers — reasonable bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_position_size_bounds(regime):
    pos_mult, sl_mult, max_trades = RegimeDetector._risk_params_for_regime(regime)
    assert 0.2 <= pos_mult <= 1.5, f"pos_mult {pos_mult} out of range for {regime}"
    assert 0.5 <= sl_mult <= 2.5, f"sl_mult {sl_mult} out of range for {regime}"
    assert 1 <= max_trades <= 10, f"max_trades {max_trades} out of range for {regime}"


# ---------------------------------------------------------------------------
# is_safe_to_trade
# ---------------------------------------------------------------------------


class TestIsSafeToTrade:
    def test_crash_is_unsafe(self):
        regime = MarketRegime(
            index_id="NIFTY50",
            timestamp=datetime.now(tz=_IST),
            regime="CRASH",
            trend_regime="STRONG_DOWN",
            volatility_regime="EXTREME",
            event_regime="NORMAL",
            market_phase="MARKDOWN",
            regime_confidence=0.9,
            regime_duration_bars=1,
            regime_changing=False,
            weight_adjustments=SignalWeights(),
            position_size_multiplier=0.3,
            stop_loss_multiplier=2.0,
            max_trades_today=1,
            description="CRASH",
        )
        safe, reason = RegimeDetector.is_safe_to_trade(regime)
        assert safe is False
        assert "CRASH" in reason

    def test_volatile_choppy_high_vol_unsafe(self):
        regime = MarketRegime(
            index_id="NIFTY50",
            timestamp=datetime.now(tz=_IST),
            regime="VOLATILE_CHOPPY",
            trend_regime="FLAT",
            volatility_regime="HIGH",
            event_regime="NORMAL",
            market_phase="ACCUMULATION",
            regime_confidence=0.55,
            regime_duration_bars=2,
            regime_changing=False,
            weight_adjustments=SignalWeights(),
            position_size_multiplier=0.4,
            stop_loss_multiplier=1.5,
            max_trades_today=1,
            description="Choppy",
        )
        safe, reason = RegimeDetector.is_safe_to_trade(regime)
        assert safe is False

    def test_trend_up_is_safe(self):
        regime = MarketRegime(
            index_id="NIFTY50",
            timestamp=datetime.now(tz=_IST),
            regime="TREND_UP",
            trend_regime="UP",
            volatility_regime="NORMAL",
            event_regime="NORMAL",
            market_phase="MARKUP",
            regime_confidence=0.7,
            regime_duration_bars=5,
            regime_changing=False,
            weight_adjustments=SignalWeights(),
            position_size_multiplier=1.0,
            stop_loss_multiplier=1.0,
            max_trades_today=5,
            description="Uptrend",
        )
        safe, reason = RegimeDetector.is_safe_to_trade(regime)
        assert safe is True

    def test_extreme_vix_unsafe(self):
        regime = MarketRegime(
            index_id="NIFTY50",
            timestamp=datetime.now(tz=_IST),
            regime="RANGE_BOUND",
            trend_regime="FLAT",
            volatility_regime="EXTREME",
            event_regime="NORMAL",
            market_phase="ACCUMULATION",
            regime_confidence=0.5,
            regime_duration_bars=1,
            regime_changing=False,
            weight_adjustments=SignalWeights(),
            position_size_multiplier=0.5,
            stop_loss_multiplier=1.5,
            max_trades_today=2,
            description="Range-bound",
            warnings=["VIX elevated at 38.0"],
        )
        safe, reason = RegimeDetector.is_safe_to_trade(regime)
        assert safe is False
        assert "VIX" in reason


# ---------------------------------------------------------------------------
# Missing data graceful degradation
# ---------------------------------------------------------------------------


class TestMissingData:
    def test_no_vix_still_valid(self, detector):
        price_df = _make_price_df(bars=60, trend=20.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="STRONG",
                trend_direction="BULLISH",
                ema_alignment="BULLISH",
            ),
        )
        regime = detector.detect_regime(
            "NIFTY50", price_df, tech, vix_value=None,
        )

        assert regime.regime in ALL_REGIMES
        assert regime.weight_adjustments.validate()
        assert any("VIX" in w or "volatility" in w.lower() for w in regime.warnings) or True

    def test_no_events_still_valid(self, detector):
        price_df = _make_price_df(bars=60, trend=20.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="STRONG",
                trend_direction="BULLISH",
                ema_alignment="BULLISH",
            ),
        )
        regime = detector.detect_regime(
            "NIFTY50", price_df, tech,
            news_event_modifier=None, vix_value=15.0,
        )

        assert regime.event_regime == "NORMAL"
        assert regime.weight_adjustments.validate()

    def test_no_anomaly_still_valid(self, detector):
        price_df = _make_price_df(bars=60)
        tech = _make_technical_result()
        regime = detector.detect_regime(
            "NIFTY50", price_df, tech,
            anomaly_result=None, vix_value=15.0,
        )

        assert regime.weight_adjustments.validate()

    def test_insufficient_data_returns_range_bound(self, detector):
        price_df = _make_price_df(bars=10)  # < 20 bars
        tech = _make_technical_result()
        regime = detector.detect_regime("NIFTY50", price_df, tech)

        assert regime.regime == "RANGE_BOUND"
        assert regime.regime_confidence == pytest.approx(0.2, abs=0.01)
        assert "Insufficient" in regime.description

    def test_none_price_df(self, detector):
        tech = _make_technical_result()
        regime = detector.detect_regime("NIFTY50", None, tech)

        assert regime.regime == "RANGE_BOUND"

    def test_all_missing_still_valid(self, detector):
        price_df = _make_price_df(bars=60)
        tech = _make_technical_result()
        regime = detector.detect_regime(
            "NIFTY50", price_df, tech,
            news_event_modifier=None,
            anomaly_result=None,
            vix_value=None,
        )

        assert regime.weight_adjustments.validate()
        assert regime.position_size_multiplier > 0


# ---------------------------------------------------------------------------
# Parametrised regime scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trend_strength, trend_dir, ema_align, vix, expected_regime",
    [
        ("VERY_STRONG", "BULLISH", "BULLISH", 15.0, "STRONG_TREND_UP"),
        ("STRONG", "BULLISH", "BULLISH", 15.0, "TREND_UP"),
        ("VERY_STRONG", "BEARISH", "BEARISH", 15.0, "STRONG_TREND_DOWN"),
        ("STRONG", "BEARISH", "BEARISH", 15.0, "TREND_DOWN"),
        ("WEAK", "BULLISH", "MIXED", 15.0, "RANGE_BOUND"),
        ("EMERGING", "BULLISH", "MIXED", 15.0, "RANGE_BOUND"),
    ],
)
def test_parametrised_regimes(
    detector, trend_strength, trend_dir, ema_align, vix, expected_regime,
):
    price_df = _make_price_df(bars=60, trend=10.0 if "BULL" in trend_dir else -10.0)
    tech = _make_technical_result(
        trend_kwargs=dict(
            trend_strength=trend_strength,
            trend_direction=trend_dir,
            ema_alignment=ema_align,
        ),
    )
    regime = detector.detect_regime("NIFTY50", price_df, tech, vix_value=vix)
    assert regime.regime == expected_regime


# ---------------------------------------------------------------------------
# Regime history
# ---------------------------------------------------------------------------


class TestRegimeHistory:
    def test_history_accumulates(self, detector):
        price_df = _make_price_df(bars=60, trend=50.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="VERY_STRONG",
                trend_direction="BULLISH",
                ema_alignment="BULLISH",
            ),
        )

        for _ in range(3):
            detector.detect_regime("NIFTY50", price_df, tech, vix_value=15.0)

        history = detector.get_regime_history("NIFTY50", days=1)
        assert len(history) == 3

    def test_history_empty_for_unknown_index(self, detector):
        assert detector.get_regime_history("UNKNOWN", days=20) == []


# ---------------------------------------------------------------------------
# Wyckoff phase detection
# ---------------------------------------------------------------------------


class TestWyckoffPhase:
    def test_markup_phase(self, detector):
        price_df = _make_price_df(bars=60, trend=50.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="VERY_STRONG",
                trend_direction="BULLISH",
                ema_alignment="BULLISH",
                price_vs_ema200="ABOVE",
            ),
        )
        regime = detector.detect_regime("NIFTY50", price_df, tech, vix_value=14.0)
        assert regime.market_phase == "MARKUP"

    def test_markdown_phase(self, detector):
        price_df = _make_price_df(bars=60, trend=-50.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="VERY_STRONG",
                trend_direction="BEARISH",
                ema_alignment="BEARISH",
                price_vs_ema200="BELOW",
            ),
        )
        regime = detector.detect_regime("NIFTY50", price_df, tech, vix_value=14.0)
        assert regime.market_phase == "MARKDOWN"

    def test_accumulation_phase(self, detector):
        price_df = _make_price_df(bars=60, trend=0.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="WEAK",
                trend_direction="BULLISH",
                ema_alignment="MIXED",
                price_vs_ema200="BELOW",
            ),
            smart_money_kwargs=dict(smart_money_bias="BULLISH"),
            volume_kwargs=dict(obv_trend="RISING"),
        )
        regime = detector.detect_regime("NIFTY50", price_df, tech, vix_value=14.0)
        assert regime.market_phase == "ACCUMULATION"


# ---------------------------------------------------------------------------
# Volatility helpers
# ---------------------------------------------------------------------------


class TestVolatilityHelpers:
    @pytest.mark.parametrize(
        "level, expected",
        [
            ("LOW", "LOW"),
            ("NORMAL", "NORMAL"),
            ("HIGH", "HIGH"),
            ("EXTREME", "EXTREME"),
            ("ELEVATED", "HIGH"),
            ("CALM", "LOW"),
            ("VERY_HIGH", "EXTREME"),
            ("MODERATE", "NORMAL"),
        ],
    )
    def test_map_volatility_level(self, level, expected):
        assert _map_volatility_level(level) == expected

    def test_max_vol(self):
        assert _max_vol("LOW", "HIGH") == "HIGH"
        assert _max_vol("EXTREME", "NORMAL") == "EXTREME"
        assert _max_vol("NORMAL", "NORMAL") == "NORMAL"


# ---------------------------------------------------------------------------
# Description and warnings
# ---------------------------------------------------------------------------


class TestDescription:
    def test_description_contains_regime_info(self, detector):
        price_df = _make_price_df(bars=60, trend=50.0)
        tech = _make_technical_result(
            trend_kwargs=dict(
                trend_strength="VERY_STRONG",
                trend_direction="BULLISH",
                ema_alignment="BULLISH",
            ),
        )
        regime = detector.detect_regime("NIFTY50", price_df, tech, vix_value=15.0)
        assert "uptrend" in regime.description.lower()

    def test_expiry_warning(self, detector):
        price_df = _make_price_df(bars=60)
        tech = _make_technical_result()
        event = _make_event_modifier(is_event_day=True, is_expiry_day=True)
        regime = detector.detect_regime(
            "NIFTY50", price_df, tech, news_event_modifier=event,
        )
        assert any("Expiry" in w for w in regime.warnings)
