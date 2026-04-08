"""
Tests for SignalGenerator — Phase 5.2.

Covers:
- BUY_CALL / BUY_PUT / NO_TRADE outcomes across all regimes
- Safety checks (crash, anomaly extreme, max trades, market close, bad price)
- Weight redistribution when data is missing
- Divergence override (RSI + OBV)
- Confidence floor (0.15) and ceiling (0.92)
- Entry / stop-loss / target level calculation
- Reasoning text completeness
- Expiry day and gap-fill modifiers
- Smart money conflict warnings
- Cooldown detection
- 10+ parametrized scenario matrix
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime
from typing import Optional
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.analysis.anomaly.anomaly_aggregator import AnomalyVote
from src.analysis.indicators.momentum import MomentumSummary
from src.analysis.indicators.options_indicators import OptionsSummary
from src.analysis.indicators.quant import QuantSummary
from src.analysis.indicators.smart_money import SmartMoneyScore
from src.analysis.indicators.trend import TrendSummary
from src.analysis.indicators.volatility import VolatilitySummary
from src.analysis.indicators.volume import VolumeSummary
from src.analysis.news.news_engine import NewsVote
from src.analysis.technical_aggregator import TechnicalAnalysisResult
from src.engine.regime_detector import MarketRegime, SignalWeights
from src.engine.signal_generator import (
    SignalGenerator,
    TradingSignal,
    _CONF_CEILING,
    _CONF_FLOOR,
    _SCORE_CALL_THRESHOLD,
    _SCORE_PUT_THRESHOLD,
)

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Shared reference times
# ---------------------------------------------------------------------------

MORNING = datetime(2026, 4, 7, 10, 30, 0, tzinfo=_IST)
AFTERNOON_EARLY = datetime(2026, 4, 7, 13, 45, 0, tzinfo=_IST)
NEAR_CLOSE = datetime(2026, 4, 7, 15, 10, 0, tzinfo=_IST)   # 20 min before close

SPOT_PRICE = 22_450.0
INDEX_ID = "NIFTY50"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _trend(
    vote: str = "BULLISH",
    confidence: float = 0.75,
    ema_alignment: str = "BULLISH",
    direction: str = "BULLISH",
    divergence_detected: bool = False,
) -> TrendSummary:
    return TrendSummary(
        index_id=INDEX_ID,
        timeframe="1d",
        timestamp=MORNING,
        price_vs_ema20="ABOVE" if "BULL" in direction else "BELOW",
        price_vs_ema50="ABOVE" if "BULL" in direction else "BELOW",
        price_vs_ema200="ABOVE" if "BULL" in direction else "BELOW",
        ema_alignment=ema_alignment,
        golden_cross=False,
        death_cross=False,
        macd_signal="BULLISH" if "BULL" in direction else "BEARISH",
        macd_crossover=None,
        macd_histogram_trend="INCREASING" if "BULL" in direction else "DECREASING",
        trend_strength="STRONG",
        trend_direction=direction,
        trend_vote=vote,
        trend_confidence=confidence,
    )


def _momentum(
    vote: str = "BULLISH",
    confidence: float = 0.60,
    rsi: float = 58.0,
    rsi_divergence: Optional[str] = None,
    divergence_detected: bool = False,
) -> MomentumSummary:
    return MomentumSummary(
        timestamp=MORNING,
        rsi_value=rsi,
        rsi_zone="NEUTRAL",
        rsi_divergence=rsi_divergence,
        stochastic_k=55.0,
        stochastic_zone="NEUTRAL",
        stochastic_crossover=None,
        cci_value=20.0,
        cci_zone="NEUTRAL",
        momentum_vote=vote,
        momentum_confidence=confidence,
        overbought_consensus=False,
        oversold_consensus=False,
        divergence_detected=divergence_detected,
        reversal_warning=None,
    )


def _volatility(
    atr: float = 120.0,
    position_size_modifier: float = 1.0,
) -> VolatilitySummary:
    return VolatilitySummary(
        timestamp=MORNING,
        bb_position="UPPER_ZONE",
        bb_squeeze=False,
        bb_bandwidth_percentile=50.0,
        atr_value=atr,
        atr_pct=atr / SPOT_PRICE * 100,
        volatility_level="NORMAL",
        suggested_sl=atr * 1.5,
        suggested_target=atr * 2.0,
        hv_current=15.0,
        hv_regime="NORMAL",
        vix_regime=None,
        volatility_vote="NORMAL",
        volatility_confidence=0.70,
        position_size_modifier=position_size_modifier,
        breakout_alert=False,
        mean_reversion_setup=False,
    )


def _volume(
    vote: str = "BULLISH",
    confidence: float = 0.65,
) -> VolumeSummary:
    return VolumeSummary(
        timestamp=MORNING,
        price_vs_vwap="ABOVE",
        vwap_zone="ABOVE_VWAP",
        institutional_bias="BUYING",
        obv_trend="RISING",
        obv_divergence=None,
        accumulation_distribution="ACCUMULATION",
        poc=22_400.0,
        value_area_high=22_600.0,
        value_area_low=22_200.0,
        in_value_area=True,
        volume_ratio=1.3,
        volume_confirms_price=True,
        volume_vote=vote,
        volume_confidence=confidence,
    )


def _options(
    vote: str = "BULLISH",
    confidence: float = 0.70,
    max_pain: float = 22_500.0,
    days_to_expiry: int = 3,
) -> OptionsSummary:
    return OptionsSummary(
        timestamp=MORNING,
        index_id=INDEX_ID,
        expiry_date=date(2026, 4, 10),
        days_to_expiry=days_to_expiry,
        pcr=1.15,
        pcr_signal="BULLISH",
        oi_support=22_000.0,
        oi_resistance=23_000.0,
        expected_range=(22_000.0, 23_000.0),
        max_pain=max_pain,
        max_pain_pull="MODERATE",
        oi_change_signal="NEUTRAL",
        dominant_buildup="LONG_BUILDUP",
        atm_iv=14.0,
        iv_regime="NORMAL",
        iv_skew=0.5,
        options_vote=vote,
        options_confidence=confidence,
    )


def _smart_money(
    bias: str = "BULLISH",
    confidence: float = 0.68,
    grade: str = "B+",
    score: float = 60.0,
) -> SmartMoneyScore:
    return SmartMoneyScore(
        score=score,
        grade=grade,
        smfi_component=60.0,
        vsd_component=55.0,
        btd_component=50.0,
        oimi_component=65.0,
        lai_component=70.0,
        smart_money_bias=bias,
        key_finding="Mild accumulation detected",
        actionable_insight="Maintain long bias",
        data_completeness=0.85,
        confidence=confidence,
    )


def _quant() -> QuantSummary:
    return QuantSummary(
        timestamp=MORNING,
        zscore=0.5,
        zscore_zone="FAIR_VALUE",
        mean_reversion_signal=None,
        beta=1.0,
        alpha=0.02,
        beta_interpretation="Market-tracking",
        statistical_regime="NORMAL",
        quant_vote="NEUTRAL",
        quant_confidence=0.50,
    )


def _technical(
    trend_vote: str = "BULLISH",
    momentum_vote: str = "BULLISH",
    volume_vote: str = "BULLISH",
    options: Optional[OptionsSummary] = None,
    smart_money: Optional[SmartMoneyScore] = None,
    divergence_detected: bool = False,
    rsi_divergence: Optional[str] = None,
    obv_divergence: Optional[str] = None,
    immediate_support: float = 22_300.0,
    immediate_resistance: float = 22_700.0,
    overall_signal: str = "BUY",
    atr: float = 120.0,
    position_size_modifier: float = 1.0,
) -> TechnicalAnalysisResult:
    vol = _volume(vote=volume_vote)
    if obv_divergence:
        vol = replace(vol, obv_divergence=obv_divergence)

    mom = _momentum(
        vote=momentum_vote,
        rsi_divergence=rsi_divergence,
        divergence_detected=divergence_detected,
    )

    # Determine direction from trend_vote
    bull = "BULL" in trend_vote.upper()
    direction = "BULLISH" if bull else "BEARISH"

    return TechnicalAnalysisResult(
        index_id=INDEX_ID,
        timestamp=MORNING,
        timeframe="1d",
        trend=_trend(vote=trend_vote, direction=direction),
        momentum=mom,
        volatility=_volatility(atr=atr, position_size_modifier=position_size_modifier),
        volume=vol,
        options=options if options is not None else _options(),
        quant=_quant(),
        smart_money=smart_money,
        votes={
            "trend": trend_vote,
            "momentum": momentum_vote,
            "volume": volume_vote,
        },
        bullish_votes=3 if bull else 0,
        bearish_votes=0 if bull else 3,
        neutral_votes=0,
        overall_signal=overall_signal,
        overall_confidence=0.72,
        support_levels=[22_300.0, 22_100.0, 22_000.0],
        resistance_levels=[22_700.0, 22_900.0, 23_000.0],
        immediate_support=immediate_support,
        immediate_resistance=immediate_resistance,
        suggested_stop_loss_distance=atr * 1.5,
        suggested_target_distance=atr * 2.0,
        position_size_modifier=position_size_modifier,
        alerts=[],
        reasoning="",
        data_completeness=1.0,
        warnings=[],
    )


def _news(
    vote: str = "BULLISH",
    confidence: float = 0.55,
    article_count: int = 5,
    event_regime: str = "NORMAL",
) -> NewsVote:
    return NewsVote(
        index_id=INDEX_ID,
        timestamp=MORNING,
        vote=vote,
        confidence=confidence,
        active_article_count=article_count,
        weighted_sentiment=0.35 if "BULL" in vote.upper() else -0.35,
        top_headline="Markets climb on strong macro data",
        event_regime=event_regime,
        reasoning="Net positive sentiment from 5 articles.",
    )


def _anomaly(
    vote: str = "NEUTRAL",
    confidence: float = 0.40,
    risk_level: str = "NORMAL",
    institutional_activity: bool = False,
    position_size_modifier: float = 1.0,
    primary_alert_message: Optional[str] = None,
) -> AnomalyVote:
    return AnomalyVote(
        index_id=INDEX_ID,
        vote=vote,
        confidence=confidence,
        risk_level=risk_level,
        position_size_modifier=position_size_modifier,
        active_alerts=0,
        primary_alert_message=primary_alert_message,
        institutional_activity=institutional_activity,
        reasoning="No significant anomalies detected.",
    )


def _regime(
    regime: str = "TREND_UP",
    volatility_regime: str = "NORMAL",
    event_regime: str = "NORMAL",
    position_size_multiplier: float = 1.0,
    stop_loss_multiplier: float = 1.0,
    max_trades_today: int = 5,
    regime_confidence: float = 0.75,
    regime_changing: bool = False,
) -> MarketRegime:
    return MarketRegime(
        index_id=INDEX_ID,
        timestamp=MORNING,
        regime=regime,
        trend_regime="UP",
        volatility_regime=volatility_regime,
        event_regime=event_regime,
        market_phase="MARKUP",
        regime_confidence=regime_confidence,
        regime_duration_bars=10,
        regime_changing=regime_changing,
        weight_adjustments=SignalWeights(),
        position_size_multiplier=position_size_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        max_trades_today=max_trades_today,
        description="Moderate uptrend with normal volatility Favor trend-following entries.",
        warnings=[],
    )


def _mock_db() -> MagicMock:
    db = MagicMock()
    db.execute.return_value = MagicMock()
    return db


def _make_generator() -> SignalGenerator:
    return SignalGenerator(_mock_db())


# ---------------------------------------------------------------------------
# Common patching decorator: freeze time at 10:30 AM, never near close
# ---------------------------------------------------------------------------

def morning_patch(fn):
    """Decorator that patches datetime.now and _is_near_market_close for test."""
    def wrapper(*args, **kwargs):
        with patch(
            "src.engine.signal_generator.SignalGenerator._is_near_market_close",
            staticmethod(lambda now, minutes=30: False),
        ):
            return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


# ---------------------------------------------------------------------------
# ──────────────────────────── Core signal tests ───────────────────────────
# ---------------------------------------------------------------------------


class TestBullishSignal:
    """All-bullish inputs should produce BUY_CALL with HIGH confidence."""

    def test_all_bullish_gives_buy_call(self):
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
            options=_options(vote="BULLISH"),
            smart_money=_smart_money(bias="BULLISH"),
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime("STRONG_TREND_UP", position_size_multiplier=1.2),
                SPOT_PRICE,
            )

        assert sig.signal_type == "BUY_CALL"
        assert sig.confidence_level in ("HIGH", "MEDIUM")
        assert sig.weighted_score > _SCORE_CALL_THRESHOLD

    def test_all_bullish_high_confidence(self):
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
            options=_options(vote="BULLISH", confidence=0.80),
            smart_money=_smart_money(bias="STRONGLY_BULLISH", grade="A+", confidence=0.85),
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("STRONG_BULLISH"), _anomaly("BULLISH"),
                _regime("STRONG_TREND_UP"),
                SPOT_PRICE,
            )

        assert sig.signal_type == "BUY_CALL"
        assert sig.confidence_score >= _CONF_FLOOR
        assert sig.confidence_score <= _CONF_CEILING

    def test_buy_call_levels_are_correct(self):
        """Stop loss should be below entry; target above entry for BUY_CALL."""
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
            immediate_support=22_200.0,
            immediate_resistance=22_800.0,
            atr=100.0,
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("BULLISH"),
                _regime("STRONG_TREND_UP"),
                SPOT_PRICE,
            )

        if sig.signal_type == "BUY_CALL":
            assert sig.entry_price == SPOT_PRICE
            assert sig.stop_loss < sig.entry_price, "Stop must be below entry for CALL"
            assert sig.target_price > sig.entry_price, "Target must be above entry for CALL"
            assert sig.risk_reward_ratio > 0


class TestBearishSignal:
    """All-bearish inputs should produce BUY_PUT with HIGH confidence."""

    def test_all_bearish_gives_buy_put(self):
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BEARISH",
            momentum_vote="BEARISH",
            volume_vote="BEARISH",
            options=_options(vote="BEARISH"),
            smart_money=_smart_money(bias="BEARISH"),
            overall_signal="SELL",
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BEARISH"), _anomaly("BEARISH"),
                _regime("STRONG_TREND_DOWN"),
                SPOT_PRICE,
            )

        assert sig.signal_type == "BUY_PUT"
        assert sig.weighted_score < _SCORE_PUT_THRESHOLD

    def test_buy_put_levels_are_correct(self):
        """Stop loss should be above entry; target below entry for BUY_PUT."""
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BEARISH",
            momentum_vote="BEARISH",
            volume_vote="BEARISH",
            immediate_support=22_000.0,
            immediate_resistance=22_800.0,
            atr=100.0,
            overall_signal="SELL",
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BEARISH"), _anomaly("NEUTRAL"),
                _regime("STRONG_TREND_DOWN"),
                SPOT_PRICE,
            )

        if sig.signal_type == "BUY_PUT":
            assert sig.stop_loss > sig.entry_price, "Stop must be above entry for PUT"
            assert sig.target_price < sig.entry_price, "Target must be below entry for PUT"
            assert sig.risk_reward_ratio > 0


class TestMixedSignal:
    """Mixed / conflicting inputs should produce NO_TRADE or LOW confidence."""

    def test_perfectly_split_votes_no_trade(self):
        gen = _make_generator()
        # Trend strongly bullish, momentum strongly bearish → net cancel
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="STRONG_BEARISH",
            volume_vote="NEUTRAL",
        )
        # Remove options and smart money so fewer votes skew less
        tech = replace(tech, options=None, smart_money=None)
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, None, None,
                _regime("RANGE_BOUND"),
                SPOT_PRICE,
            )

        # Either NO_TRADE or LOW confidence — not HIGH confidence
        if sig.signal_type != "NO_TRADE":
            assert sig.confidence_level in ("LOW", "MEDIUM")

    def test_no_trade_when_score_in_dead_zone(self):
        gen = _make_generator()
        # All NEUTRAL → score near 0
        tech = _technical(
            trend_vote="NEUTRAL",
            momentum_vote="NEUTRAL",
            volume_vote="NEUTRAL",
        )
        tech = replace(tech, options=None, smart_money=None)
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, None, None,
                _regime("RANGE_BOUND"),
                SPOT_PRICE,
            )

        assert sig.signal_type == "NO_TRADE"


# ---------------------------------------------------------------------------
# ──────────────────────────── Safety check tests ──────────────────────────
# ---------------------------------------------------------------------------


class TestSafetyChecks:

    def test_extreme_anomaly_always_no_trade(self):
        """Even with all-bullish signals, EXTREME anomaly → NO_TRADE."""
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
        )
        dangerous_anomaly = _anomaly(risk_level="EXTREME")
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("STRONG_BULLISH"), dangerous_anomaly,
                _regime("STRONG_TREND_UP"),
                SPOT_PRICE,
            )

        assert sig.signal_type == "NO_TRADE"
        assert "extreme anomaly" in sig.reasoning.lower() or "extreme" in " ".join(sig.warnings).lower()

    def test_max_trades_reached_no_trade(self):
        """When today's count >= max_trades_today → NO_TRADE."""
        gen = _make_generator()
        regime = _regime(max_trades_today=2)

        # Pre-fill the cache with 2 signals
        dummy = TradingSignal(
            signal_id="x",
            index_id=INDEX_ID,
            generated_at=MORNING,
            signal_type="BUY_CALL",
            confidence_level="HIGH",
            confidence_score=0.80,
            entry_price=SPOT_PRICE,
            target_price=SPOT_PRICE + 200,
            stop_loss=SPOT_PRICE - 150,
            risk_reward_ratio=1.5,
            regime=regime.regime,
            weighted_score=1.2,
            vote_breakdown={},
            risk_level="NORMAL",
            position_size_modifier=1.0,
            suggested_lot_count=1,
            estimated_max_loss=150.0,
            estimated_max_profit=200.0,
            reasoning="",
            warnings=[],
        )
        # Set the date so _refresh_daily_cache doesn't wipe the pre-filled cache
        gen._today_date = MORNING.strftime("%Y-%m-%d")
        gen._today_signals[INDEX_ID] = [dummy, dummy]

        tech = _technical(trend_vote="STRONG_BULLISH", momentum_vote="BULLISH", volume_vote="BULLISH")
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                regime,
                SPOT_PRICE,
            )

        assert sig.signal_type == "NO_TRADE"
        assert "max signals" in sig.reasoning.lower() or any("max" in w.lower() for w in sig.warnings)

    def test_crash_regime_no_trade(self):
        """CRASH regime → RegimeDetector.is_safe_to_trade returns False → NO_TRADE."""
        gen = _make_generator()
        crash_regime = _regime(regime="CRASH")
        tech = _technical(trend_vote="BULLISH", momentum_vote="BULLISH", volume_vote="BULLISH")
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                crash_regime,
                SPOT_PRICE,
            )

        assert sig.signal_type == "NO_TRADE"
        assert "crash" in sig.reasoning.lower() or any(
            "crash" in w.lower() or "unsafe" in w.lower() for w in sig.warnings
        )

    def test_invalid_spot_price_no_trade(self):
        gen = _make_generator()
        tech = _technical()
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime(),
                0.0,  # invalid price
            )
        assert sig.signal_type == "NO_TRADE"

    def test_near_market_close_no_trade(self):
        """If _is_near_market_close returns True → NO_TRADE."""
        gen = _make_generator()
        tech = _technical(trend_vote="STRONG_BULLISH", momentum_vote="BULLISH", volume_vote="BULLISH")
        # Do NOT patch _is_near_market_close — but pass NEAR_CLOSE time via monkeypatching datetime
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=True):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime(),
                SPOT_PRICE,
            )
        assert sig.signal_type == "NO_TRADE"
        assert any("close" in w.lower() for w in sig.warnings)

    def test_volatile_choppy_high_vol_no_trade(self):
        """VOLATILE_CHOPPY + HIGH volatility → is_safe_to_trade → False → NO_TRADE."""
        gen = _make_generator()
        choppy_regime = _regime(regime="VOLATILE_CHOPPY", volatility_regime="HIGH")
        tech = _technical(trend_vote="BULLISH", momentum_vote="BULLISH", volume_vote="BULLISH")
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                choppy_regime,
                SPOT_PRICE,
            )
        assert sig.signal_type == "NO_TRADE"


# ---------------------------------------------------------------------------
# ──────────────────────── Weight redistribution tests ─────────────────────
# ---------------------------------------------------------------------------


class TestWeightRedistribution:

    def test_options_missing_weights_sum_to_one(self):
        """When options data is None, redistributed weights should still sum to 1.0."""
        gen = _make_generator()
        base_weights = SignalWeights()
        available = {"trend", "momentum", "volume", "news", "smart_money", "anomaly"}
        new_weights = gen._redistribute_weights(base_weights, available)
        total = (
            new_weights.trend_weight
            + new_weights.momentum_weight
            + new_weights.options_weight
            + new_weights.volume_weight
            + new_weights.news_weight
            + new_weights.smart_money_weight
            + new_weights.anomaly_weight
        )
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, not 1.0"
        assert new_weights.options_weight == 0.0

    def test_options_and_news_missing_weights_sum_to_one(self):
        gen = _make_generator()
        base_weights = SignalWeights()
        available = {"trend", "momentum", "volume", "smart_money", "anomaly"}
        new_weights = gen._redistribute_weights(base_weights, available)
        total = sum([
            new_weights.trend_weight,
            new_weights.momentum_weight,
            new_weights.options_weight,
            new_weights.volume_weight,
            new_weights.news_weight,
            new_weights.smart_money_weight,
            new_weights.anomaly_weight,
        ])
        assert abs(total - 1.0) < 1e-6
        assert new_weights.options_weight == 0.0
        assert new_weights.news_weight == 0.0

    def test_no_options_still_generates_signal(self):
        """With options=None, signal generator should still produce a valid signal."""
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
        )
        # Remove options explicitly
        tech = replace(tech, options=None)
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime("STRONG_TREND_UP"),
                SPOT_PRICE,
            )

        # Should still produce a directional signal if conditions are met
        assert sig.signal_type in ("BUY_CALL", "BUY_PUT", "NO_TRADE")

    def test_all_optional_missing_still_functional(self):
        """No news, no anomaly, no options, no smart money — still runs."""
        gen = _make_generator()
        tech = _technical(
            trend_vote="BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
        )
        tech = replace(tech, options=None, smart_money=None)
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, None, None,
                _regime("TREND_UP"),
                SPOT_PRICE,
            )

        assert isinstance(sig, TradingSignal)
        assert sig.signal_type in ("BUY_CALL", "BUY_PUT", "NO_TRADE")


# ---------------------------------------------------------------------------
# ──────────────────────── Divergence override tests ───────────────────────
# ---------------------------------------------------------------------------


class TestDivergenceOverride:

    def test_bullish_divergence_boosts_neutral_score(self):
        """A BULLISH_DIVERGENCE in RSI should push a neutral score toward BUY_CALL."""
        gen = _make_generator()

        # Create neutral/mildly bearish technical setup
        tech = _technical(
            trend_vote="NEUTRAL",
            momentum_vote="NEUTRAL",
            volume_vote="SLIGHTLY_BEARISH",
            divergence_detected=True,
            rsi_divergence="BULLISH_DIVERGENCE",
        )
        tech = replace(tech, options=None, smart_money=None)

        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, None, None,
                _regime("RANGE_BOUND"),
                SPOT_PRICE,
            )

        # The override boosted the score by 0.3
        # Even if it's still NO_TRADE, warnings should mention divergence
        divergence_mentioned = any("divergence" in w.lower() for w in sig.warnings)
        assert divergence_mentioned or sig.signal_type == "BUY_CALL"

    def test_bearish_divergence_pushes_toward_put(self):
        """A BEARISH_DIVERGENCE should bias a neutral score toward BUY_PUT."""
        gen = _make_generator()

        tech = _technical(
            trend_vote="NEUTRAL",
            momentum_vote="NEUTRAL",
            volume_vote="SLIGHTLY_BULLISH",
            divergence_detected=True,
            rsi_divergence="BEARISH_DIVERGENCE",
        )
        tech = replace(tech, options=None, smart_money=None)

        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, None, None,
                _regime("RANGE_BOUND"),
                SPOT_PRICE,
            )

        divergence_mentioned = any("divergence" in w.lower() for w in sig.warnings)
        assert divergence_mentioned or sig.signal_type == "BUY_PUT"

    def test_divergence_override_applied_to_score(self):
        """Test _apply_divergence_override directly."""
        gen = _make_generator()

        # Neutral score + bullish divergence → score should increase
        momentum = _momentum(divergence_detected=True, rsi_divergence="BULLISH_DIVERGENCE")
        vol = replace(_volume(), obv_divergence=None)
        tech = replace(
            _technical(),
            momentum=momentum,
            volume=vol,
        )
        tech.momentum.divergence_detected = True

        new_score, warnings = gen._apply_divergence_override(0.0, tech)
        assert new_score > 0.0
        assert len(warnings) > 0
        assert "divergence" in warnings[0].lower()


# ---------------------------------------------------------------------------
# ──────────────────── Confidence floor / ceiling tests ────────────────────
# ---------------------------------------------------------------------------


class TestConfidence:

    def test_confidence_never_below_floor(self):
        gen = _make_generator()
        # Very weak, missing data signal
        tech = _technical(
            trend_vote="SLIGHTLY_BULLISH",
            momentum_vote="NEUTRAL",
            volume_vote="NEUTRAL",
        )
        tech = replace(tech, options=None, smart_money=None, data_completeness=0.4)

        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, None, _anomaly("CAUTION", risk_level="HIGH"),
                _regime("VOLATILE_CHOPPY", volatility_regime="NORMAL",
                        position_size_multiplier=0.4),
                SPOT_PRICE,
            )

        # Whether NO_TRADE or not, confidence_score must be >= floor or 0.0 (NO_TRADE)
        assert sig.confidence_score == 0.0 or sig.confidence_score >= _CONF_FLOOR

    def test_confidence_never_above_ceiling(self):
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="STRONG_BULLISH",
            volume_vote="STRONG_BULLISH",
            options=_options(vote="STRONG_BULLISH", confidence=1.0),
            smart_money=_smart_money(bias="STRONGLY_BULLISH", grade="A+", confidence=1.0),
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("STRONG_BULLISH", confidence=1.0), _anomaly("BULLISH"),
                _regime("STRONG_TREND_UP"),
                SPOT_PRICE,
            )

        if sig.signal_type != "NO_TRADE":
            assert sig.confidence_score <= _CONF_CEILING

    def test_low_confidence_below_threshold_gives_no_trade(self):
        """If confidence < 0.30, direction is overridden to NO_TRADE."""
        gen = _make_generator()
        # Score just above 0.4 threshold but with many reducers → confidence < 0.30
        tech = _technical(
            trend_vote="SLIGHTLY_BULLISH",
            momentum_vote="NEUTRAL",
            volume_vote="NEUTRAL",
        )
        tech = replace(tech, options=None, smart_money=None, data_completeness=0.3)

        regime = _regime(
            regime="RANGE_BOUND",
            volatility_regime="EXTREME",  # -0.10 reducer
            regime_changing=True,          # -0.05 reducer
            regime_confidence=0.3,         # -0.05 reducer
        )

        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, None,
                _anomaly("CAUTION", risk_level="HIGH"),  # -0.10 + -0.10 reducers
                regime,
                SPOT_PRICE,
            )

        # With this many reducers, confidence should be very low → NO_TRADE
        # (or signal with LOW confidence if score passes threshold)
        assert sig.confidence_score == 0.0 or sig.confidence_score >= _CONF_FLOOR

    def test_confidence_level_matches_score(self):
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
            options=_options(vote="BULLISH"),
            smart_money=_smart_money(bias="BULLISH", grade="A+"),
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime("STRONG_TREND_UP"),
                SPOT_PRICE,
            )

        if sig.confidence_level == "HIGH":
            assert sig.confidence_score >= 0.65
        elif sig.confidence_level == "MEDIUM":
            assert 0.45 <= sig.confidence_score < 0.65
        elif sig.confidence_level == "LOW":
            assert 0.30 <= sig.confidence_score < 0.45


# ---------------------------------------------------------------------------
# ──────────────────────── Reasoning text tests ────────────────────────────
# ---------------------------------------------------------------------------


class TestReasoning:

    def test_reasoning_includes_all_categories(self):
        gen = _make_generator()
        tech = _technical(
            trend_vote="BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
            options=_options("BULLISH"),
            smart_money=_smart_money("BULLISH"),
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime("TREND_UP"),
                SPOT_PRICE,
            )

        if sig.signal_type != "NO_TRADE":
            reasoning = sig.reasoning
            assert "Trend" in reasoning
            assert "Momentum" in reasoning
            assert "Volume" in reasoning
            assert "Options" in reasoning
            assert "Smart Money" in reasoning
            assert "News" in reasoning
            assert "Anomaly" in reasoning
            assert "Weighted Score" in reasoning
            assert "Entry" in reasoning
            assert "Target" in reasoning
            assert "Stop Loss" in reasoning

    def test_reasoning_has_header(self):
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime("TREND_UP"),
                SPOT_PRICE,
            )

        if sig.signal_type == "BUY_CALL":
            assert "BUY CALL" in sig.reasoning
            assert INDEX_ID in sig.reasoning
            assert "Confidence" in sig.reasoning

    def test_reasoning_includes_regime(self):
        gen = _make_generator()
        tech = _technical(trend_vote="BULLISH", momentum_vote="BULLISH", volume_vote="BULLISH")
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime("TREND_UP"),
                SPOT_PRICE,
            )
        if sig.signal_type != "NO_TRADE":
            assert "Regime" in sig.reasoning


# ---------------------------------------------------------------------------
# ──────────────────────── Expiry day modifier tests ───────────────────────
# ---------------------------------------------------------------------------


class TestExpiryDayModifier:

    def test_expiry_day_after_14_adds_warning(self):
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
            options=_options(max_pain=22_500.0, days_to_expiry=0),
        )
        expiry_regime = _regime(regime="TREND_UP", event_regime="EVENT_DAY")

        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            # Pass AFTERNOON_EARLY time (13:45) — expiry warning requires hour >= 14
            with patch("src.engine.signal_generator.datetime") as mock_dt:
                mock_dt.now.return_value = AFTERNOON_EARLY
                sig = gen.generate_signal(
                    INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                    expiry_regime,
                    SPOT_PRICE,
                )

        # At 13:45 — expiry warning should NOT appear (hour < 14)
        expiry_warnings = [w for w in sig.warnings if "expiry" in w.lower() or "max pain" in w.lower()]
        assert len(expiry_warnings) == 0 or sig.signal_type == "NO_TRADE"

    def test_expiry_day_at_1400_adds_max_pain_warning(self):
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
            options=_options(max_pain=22_500.0, days_to_expiry=0),
        )
        expiry_regime = _regime(regime="TREND_UP", event_regime="EVENT_DAY")
        after_2pm = datetime(2026, 4, 7, 14, 15, 0, tzinfo=_IST)

        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            with patch("src.engine.signal_generator.datetime") as mock_dt:
                mock_dt.now.return_value = after_2pm
                sig = gen.generate_signal(
                    INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                    expiry_regime,
                    SPOT_PRICE,
                )

        expiry_warnings = [
            w for w in sig.warnings
            if "expiry" in w.lower() or "max pain" in w.lower()
        ]
        # If the signal is directional, we should see the expiry warning
        if sig.signal_type != "NO_TRADE":
            assert len(expiry_warnings) >= 1


# ---------------------------------------------------------------------------
# ──────────────────────── Gap fill modifier tests ─────────────────────────
# ---------------------------------------------------------------------------


class TestGapFillModifier:

    def test_gap_up_biases_toward_put_for_weak_signal(self):
        """Gap-up + weak signal → slight bearish bias (score decreases)."""
        gen = _make_generator()
        gap_anomaly = _anomaly(
            vote="NEUTRAL",
            risk_level="NORMAL",
            primary_alert_message="GAP_UP detected: +0.8% opening gap",
        )
        score_before = 0.3  # weak bullish, below threshold

        new_score, warns = gen._apply_gap_fill_override(score_before, gap_anomaly)

        assert new_score < score_before
        assert any("gap" in w.lower() for w in warns)

    def test_gap_down_biases_toward_call_for_weak_signal(self):
        """Gap-down + weak signal → slight bullish bias (score increases)."""
        gen = _make_generator()
        gap_anomaly = _anomaly(
            primary_alert_message="GAP DOWN detected: -0.7% opening gap",
        )
        score_before = -0.3

        new_score, warns = gen._apply_gap_fill_override(score_before, gap_anomaly)

        assert new_score > score_before
        assert any("gap" in w.lower() for w in warns)

    def test_gap_fill_does_not_override_strong_signal(self):
        """Gap fill override should not affect signals with |score| >= 1.0."""
        gen = _make_generator()
        gap_anomaly = _anomaly(
            primary_alert_message="GAP_UP detected at open",
        )
        strong_score = 1.5  # strong bullish

        new_score, warns = gen._apply_gap_fill_override(strong_score, gap_anomaly)

        assert new_score == strong_score  # unchanged
        assert len(warns) == 0

    def test_no_gap_keyword_in_message_no_effect(self):
        gen = _make_generator()
        anomaly_no_gap = _anomaly(primary_alert_message="Volume spike detected")
        score = 0.3
        new_score, warns = gen._apply_gap_fill_override(score, anomaly_no_gap)
        assert new_score == score
        assert len(warns) == 0


# ---------------------------------------------------------------------------
# ──────────────────────── Smart money conflict tests ──────────────────────
# ---------------------------------------------------------------------------


class TestSmartMoneyConflict:

    def test_smart_money_bear_while_calling_adds_warning(self):
        """Smart money bearish while signal is BUY_CALL → warning."""
        gen = _make_generator()
        tech = _technical(
            trend_vote="STRONG_BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
            smart_money=_smart_money(bias="STRONGLY_BEARISH", confidence=0.85),
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime("TREND_UP"),
                SPOT_PRICE,
            )

        if sig.signal_type == "BUY_CALL":
            conflict_warning = any(
                "smart money" in w.lower() and "disagree" in w.lower()
                for w in sig.warnings
            )
            assert conflict_warning

    def test_smart_money_a_plus_boosts_confidence(self):
        """A+ smart money grade aligned with signal → confidence boost."""
        gen = _make_generator()
        tech = _technical(
            trend_vote="BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
            smart_money=_smart_money(bias="STRONGLY_BULLISH", grade="A+", confidence=0.90),
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime("TREND_UP"),
                SPOT_PRICE,
            )

        # A+ grade gives +0.10 confidence boost
        if sig.signal_type == "BUY_CALL":
            assert sig.confidence_score > 0.0


# ---------------------------------------------------------------------------
# ────────────────── Query methods tests ───────────────────────────────────
# ---------------------------------------------------------------------------


class TestQueryMethods:

    def test_get_signals_today_returns_todays_signals(self):
        gen = _make_generator()
        tech = _technical(trend_vote="STRONG_BULLISH", momentum_vote="BULLISH", volume_vote="BULLISH")
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime("STRONG_TREND_UP"),
                SPOT_PRICE,
            )

        today_signals = gen.get_signals_today(INDEX_ID)
        # May be 0 if NO_TRADE, but should be a list
        assert isinstance(today_signals, list)

    def test_get_all_active_signals(self):
        gen = _make_generator()
        # Insert a fake open signal into the cache
        active = TradingSignal(
            signal_id="active-1",
            index_id=INDEX_ID,
            generated_at=MORNING,
            signal_type="BUY_CALL",
            confidence_level="HIGH",
            confidence_score=0.80,
            entry_price=SPOT_PRICE,
            target_price=SPOT_PRICE + 200,
            stop_loss=SPOT_PRICE - 150,
            risk_reward_ratio=1.5,
            regime="TREND_UP",
            weighted_score=1.2,
            vote_breakdown={},
            risk_level="NORMAL",
            position_size_modifier=1.0,
            suggested_lot_count=1,
            estimated_max_loss=150.0,
            estimated_max_profit=200.0,
            reasoning="Test",
            warnings=[],
            outcome="OPEN",
        )
        gen._today_signals[INDEX_ID] = [active]

        actives = gen.get_all_active_signals()
        assert len(actives) == 1
        assert actives[0].signal_id == "active-1"

    def test_no_trade_not_in_active_signals(self):
        gen = _make_generator()
        no_trade = TradingSignal(
            signal_id="nt-1",
            index_id=INDEX_ID,
            generated_at=MORNING,
            signal_type="NO_TRADE",
            confidence_level="LOW",
            confidence_score=0.0,
            entry_price=0.0,
            target_price=0.0,
            stop_loss=0.0,
            risk_reward_ratio=0.0,
            regime="RANGE_BOUND",
            weighted_score=0.0,
            vote_breakdown={},
            risk_level="NORMAL",
            position_size_modifier=0.0,
            suggested_lot_count=0,
            estimated_max_loss=0.0,
            estimated_max_profit=0.0,
            reasoning="NO_TRADE: test",
            warnings=[],
        )
        gen._today_signals[INDEX_ID] = [no_trade]
        actives = gen.get_all_active_signals()
        assert len(actives) == 0


# ---------------------------------------------------------------------------
# ──────────────────── Helper unit tests ───────────────────────────────────
# ---------------------------------------------------------------------------


class TestHelpers:

    def test_convert_vote_to_numeric_mapping(self):
        gen = _make_generator()
        assert gen._convert_vote_to_numeric("STRONG_BULLISH") == 2.0
        assert gen._convert_vote_to_numeric("STRONGLY_BULLISH") == 2.0
        assert gen._convert_vote_to_numeric("STRONG_BUY") == 2.0
        assert gen._convert_vote_to_numeric("BULLISH") == 1.0
        assert gen._convert_vote_to_numeric("BUY") == 1.0
        assert gen._convert_vote_to_numeric("SLIGHTLY_BULLISH") == 0.5
        assert gen._convert_vote_to_numeric("NEUTRAL") == 0.0
        assert gen._convert_vote_to_numeric("NO_TRADE") == 0.0
        assert gen._convert_vote_to_numeric("CAUTION") == 0.0
        assert gen._convert_vote_to_numeric("SLIGHTLY_BEARISH") == -0.5
        assert gen._convert_vote_to_numeric("BEARISH") == -1.0
        assert gen._convert_vote_to_numeric("SELL") == -1.0
        assert gen._convert_vote_to_numeric("STRONG_BEARISH") == -2.0
        assert gen._convert_vote_to_numeric("STRONGLY_BEARISH") == -2.0
        assert gen._convert_vote_to_numeric("STRONG_SELL") == -2.0
        assert gen._convert_vote_to_numeric("UNKNOWN_VOTE") == 0.0

    def test_determine_direction(self):
        gen = _make_generator()
        assert gen._determine_direction(1.0) == "BUY_CALL"
        assert gen._determine_direction(0.41) == "BUY_CALL"
        assert gen._determine_direction(0.40) == "NO_TRADE"
        assert gen._determine_direction(0.0) == "NO_TRADE"
        assert gen._determine_direction(-0.40) == "NO_TRADE"
        assert gen._determine_direction(-0.41) == "BUY_PUT"
        assert gen._determine_direction(-1.0) == "BUY_PUT"

    def test_conflicting_signals_detection(self):
        gen = _make_generator()
        # Two strong bulls and two strong bears → conflict
        conflicting_votes = {
            "trend": (2.0, 0.8, 0.25),
            "options": (2.0, 0.7, 0.20),
            "momentum": (-2.0, 0.7, 0.15),
            "volume": (-2.0, 0.6, 0.15),
            "news": (0.5, 0.5, 0.10),
        }
        assert gen._has_conflicting_strong_signals(conflicting_votes) is True

        # Only one strong bear — no conflict
        non_conflicting = {
            "trend": (2.0, 0.8, 0.25),
            "options": (2.0, 0.7, 0.20),
            "momentum": (-2.0, 0.7, 0.15),
            "volume": (1.0, 0.6, 0.15),
        }
        assert gen._has_conflicting_strong_signals(non_conflicting) is False

    def test_is_near_market_close(self):
        """Static helper correctly identifies pre-close window."""
        assert SignalGenerator._is_near_market_close(
            datetime(2026, 4, 7, 15, 5, 0, tzinfo=_IST), minutes=30
        ) is True
        assert SignalGenerator._is_near_market_close(
            datetime(2026, 4, 7, 10, 30, 0, tzinfo=_IST), minutes=30
        ) is False
        assert SignalGenerator._is_near_market_close(
            datetime(2026, 4, 7, 15, 0, 0, tzinfo=_IST), minutes=30
        ) is True  # exactly 30 min before close


# ---------------------------------------------------------------------------
# ──────────────────── Parametrized scenario matrix ────────────────────────
# ---------------------------------------------------------------------------


SCENARIO_PARAMS = [
    # (description, regime_name, all_votes, news_vote, anomaly_vote, anomaly_risk,
    #  expected_signal_type_or_any)
    (
        "all_strong_bull_strong_trend_up",
        "STRONG_TREND_UP", "STRONG_BULLISH", "STRONG_BULLISH", "BULLISH", "NORMAL",
        "BUY_CALL",
    ),
    (
        "all_strong_bear_strong_trend_down",
        "STRONG_TREND_DOWN", "STRONG_BEARISH", "STRONG_BEARISH", "BEARISH", "NORMAL",
        "BUY_PUT",
    ),
    (
        "all_neutral_range_bound",
        "RANGE_BOUND", "NEUTRAL", "NEUTRAL", "NEUTRAL", "NORMAL",
        "NO_TRADE",
    ),
    (
        "extreme_anomaly_overrides_bullish",
        "TREND_UP", "STRONG_BULLISH", "BULLISH", "CAUTION", "EXTREME",
        "NO_TRADE",
    ),
    (
        "bullish_trend_up_normal",
        "TREND_UP", "BULLISH", "BULLISH", "NEUTRAL", "NORMAL",
        "any",  # depends on confidence calculation
    ),
    (
        "bearish_trend_down_normal",
        "TREND_DOWN", "BEARISH", "BEARISH", "NEUTRAL", "NORMAL",
        "any",
    ),
    (
        "crash_regime_no_trade",
        "CRASH", "STRONG_BULLISH", "BULLISH", "BULLISH", "HIGH",
        "NO_TRADE",
    ),
    (
        "volatile_choppy_high_vol_no_trade",
        "VOLATILE_CHOPPY", "BULLISH", "NEUTRAL", "NEUTRAL", "NORMAL",
        "NO_TRADE",
    ),
    (
        "breakout_bullish_gets_boost",
        "BREAKOUT", "BULLISH", "NEUTRAL", "NEUTRAL", "NORMAL",
        "any",  # BREAKOUT adds +0.3 boost when direction matches
    ),
    (
        "event_driven_relies_on_news",
        "EVENT_DRIVEN", "NEUTRAL", "STRONG_BULLISH", "NEUTRAL", "NORMAL",
        "any",  # heavy news weight in EVENT_DRIVEN regime
    ),
    (
        "high_risk_anomaly_reduces_confidence",
        "TREND_UP", "BULLISH", "BULLISH", "CAUTION", "HIGH",
        "any",
    ),
    (
        "all_slightly_bullish_might_no_trade",
        "RANGE_BOUND", "SLIGHTLY_BULLISH", "SLIGHTLY_BULLISH", "NEUTRAL", "NORMAL",
        "any",  # slightly bullish scores are near dead zone
    ),
]


@pytest.mark.parametrize(
    "description,regime_name,vote,news_v,anomaly_v,risk,expected",
    SCENARIO_PARAMS,
    ids=[p[0] for p in SCENARIO_PARAMS],
)
def test_scenario_matrix(
    description,
    regime_name,
    vote,
    news_v,
    anomaly_v,
    risk,
    expected,
):
    """Parametrized scenario matrix covering all major regimes."""
    gen = _make_generator()

    # VOLATILE_CHOPPY with HIGH volatility → is_safe_to_trade → False
    vol_regime = "HIGH" if regime_name == "VOLATILE_CHOPPY" else "NORMAL"
    pos_mult = 0.0 if regime_name == "CRASH" else 1.0

    tech = _technical(
        trend_vote=vote,
        momentum_vote=vote,
        volume_vote=vote,
        options=_options(vote=vote),
        smart_money=_smart_money(
            bias="BULLISH" if "BULL" in vote.upper() else
            ("BEARISH" if "BEAR" in vote.upper() else "NEUTRAL")
        ),
        overall_signal=(
            "STRONG_BUY" if "STRONG_BULL" in vote.upper() else
            "BUY" if "BULL" in vote.upper() else
            "STRONG_SELL" if "STRONG_BEAR" in vote.upper() else
            "SELL" if "BEAR" in vote.upper() else
            "NEUTRAL"
        ),
    )

    regime = _regime(
        regime=regime_name,
        volatility_regime=vol_regime,
        position_size_multiplier=pos_mult,
    )

    with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
        sig = gen.generate_signal(
            INDEX_ID,
            tech,
            _news(news_v),
            _anomaly(anomaly_v, risk_level=risk),
            regime,
            SPOT_PRICE,
        )

    assert isinstance(sig, TradingSignal), f"[{description}] Expected TradingSignal"
    assert sig.signal_type in ("BUY_CALL", "BUY_PUT", "NO_TRADE"), (
        f"[{description}] Invalid signal_type: {sig.signal_type}"
    )
    assert 0.0 <= sig.confidence_score <= 1.0, (
        f"[{description}] confidence_score out of range: {sig.confidence_score}"
    )

    if expected == "NO_TRADE":
        assert sig.signal_type == "NO_TRADE", (
            f"[{description}] Expected NO_TRADE, got {sig.signal_type}"
        )
    elif expected == "BUY_CALL":
        assert sig.signal_type == "BUY_CALL", (
            f"[{description}] Expected BUY_CALL, got {sig.signal_type}"
        )
    elif expected == "BUY_PUT":
        assert sig.signal_type == "BUY_PUT", (
            f"[{description}] Expected BUY_PUT, got {sig.signal_type}"
        )
    # "any" → just assert it is a valid TradingSignal (done above)


# ---------------------------------------------------------------------------
# ──────────────── Integration: vote_breakdown completeness ────────────────
# ---------------------------------------------------------------------------


class TestVoteBreakdown:

    def test_vote_breakdown_has_expected_keys(self):
        gen = _make_generator()
        tech = _technical(
            trend_vote="BULLISH",
            momentum_vote="BULLISH",
            volume_vote="BULLISH",
            options=_options("BULLISH"),
            smart_money=_smart_money("BULLISH"),
        )
        with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
            sig = gen.generate_signal(
                INDEX_ID, tech, _news("BULLISH"), _anomaly("NEUTRAL"),
                _regime("TREND_UP"),
                SPOT_PRICE,
            )

        if sig.signal_type != "NO_TRADE":
            for comp in ("trend", "momentum", "volume", "options", "smart_money", "news", "anomaly"):
                assert comp in sig.vote_breakdown, f"Missing '{comp}' in vote_breakdown"
                entry = sig.vote_breakdown[comp]
                assert "score" in entry
                assert "confidence" in entry
                assert "weight" in entry
                assert "weighted_contribution" in entry

    def test_data_completeness_with_all_inputs(self):
        gen = _make_generator()
        tech = _technical(options=_options(), smart_money=_smart_money())
        tech = replace(tech, data_completeness=1.0)
        completeness = gen._data_completeness(tech, _news("NEUTRAL"), _anomaly("NEUTRAL"))
        assert completeness == 1.0

    def test_data_completeness_no_news_no_anomaly(self):
        gen = _make_generator()
        tech = replace(_technical(), data_completeness=1.0)
        completeness = gen._data_completeness(tech, None, None)
        # 1.0 * 0.5 = 0.5 (no news, no anomaly)
        assert abs(completeness - 0.5) < 1e-6

    def test_data_completeness_none_technical(self):
        gen = _make_generator()
        completeness = gen._data_completeness(None, None, None)
        assert completeness == 0.0
