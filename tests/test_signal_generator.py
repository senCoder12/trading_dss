"""
Tests for SignalGenerator — calibrated model (Phase 2).

Covers:
- The calibrated score (model_score): only measured components score;
  options/anomaly never add weight
- Calibrated confidence: P(win) from SignalCalibration, penalties only,
  no trading without a validated calibration
- Vetoes from unmeasured components (options positioning, anomaly)
- Safety checks (crash, choppy, extreme anomaly, max trades, close, price,
  stale bars) and the fail-closed daily trade count
- Direction thresholds (same in every regime), levels, R:R gate
- Weight redistribution, conflicting signals, cooldown, dedup, reasoning
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
from src.engine.signal_calibration import CALIBRATION_VERSION, SignalCalibration
from src.engine.signal_generator import (
    SignalGenerator,
    TradingSignal,
    _CONF_HIGH,
    _CONF_MEDIUM,
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
    immediate_support: float = 22_380.0,
    immediate_resistance: float = 22_520.0,
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


def _mock_db(today_count: int = 0) -> MagicMock:
    db = MagicMock()
    db.execute.return_value = MagicMock()
    db.fetch_one.return_value = {"count": today_count}
    return db


def _calibration(tmp_path, p: float = 0.60, validated: bool = True, index_key: str = "ALL"):
    """Write a calibration file whose bins all map to *p*."""
    path = tmp_path / "cal.json"
    edges = [0.25, 0.5, 1.0, 99.0]
    path.write_text(json.dumps({
        "version": CALIBRATION_VERSION,
        "generated_at": datetime.now(tz=_IST).isoformat(),
        "models": {index_key: {
            "bins": [{"lo": lo, "hi": hi, "n": 100, "wins": int(100 * p), "p_win": p}
                     for lo, hi in zip(edges, edges[1:])],
            "validated": validated,
            "meta": {"validation_note": "test"},
        }},
    }))
    return SignalCalibration(path)


def _gen(tmp_path, p: float = 0.60, validated: bool = True, today_count: int = 0) -> SignalGenerator:
    return SignalGenerator(_mock_db(today_count), calibration=_calibration(tmp_path, p, validated))


def _signal(gen, technical=None, anomaly=None, regime=None, now=MORNING, spot=SPOT_PRICE):
    with patch(
        "src.engine.signal_generator.SignalGenerator._is_near_market_close",
        staticmethod(lambda now, minutes=30: False),
    ):
        return gen.generate_signal(
            index_id=INDEX_ID,
            technical=technical or _technical(),
            news=None,
            anomaly=anomaly,
            regime=regime or _regime(),
            current_spot_price=spot,
            timestamp=now,
        )


# ===========================================================================
# 1. The calibrated score
# ===========================================================================


class TestModelScore:
    def test_options_and_anomaly_do_not_add_weight(self):
        regime = _regime()
        bull_opts = _technical(options=_options(vote="STRONG_BULLISH", confidence=0.95))
        bear_opts = _technical(options=_options(vote="STRONG_BEARISH", confidence=0.95))
        s1, votes, comps = SignalGenerator.model_score(bull_opts, regime)
        s2, _, _ = SignalGenerator.model_score(bear_opts, regime)
        assert s1 == pytest.approx(s2)
        assert "options" not in votes and "anomaly" not in votes
        assert comps <= {"trend", "momentum", "volume", "smart_money"}

    def test_no_volume_scores_on_trend_and_momentum_only(self):
        tech = replace(_technical(smart_money=_smart_money()), volume_available=False)
        _, votes, comps = SignalGenerator.model_score(tech, _regime())
        assert comps == {"trend", "momentum"}
        assert set(votes) == {"trend", "momentum"}

    def test_failed_component_is_unavailable_not_neutral(self):
        tech = replace(_technical(), unavailable_components=["momentum"])
        _, votes, comps = SignalGenerator.model_score(tech, _regime())
        assert "momentum" not in comps and "momentum" not in votes

    def test_weights_renormalised_over_scored_components(self):
        tech = replace(_technical(), volume_available=False)
        _, votes, _ = SignalGenerator.model_score(tech, _regime())
        assert sum(w for _, _, w in votes.values()) == pytest.approx(1.0)

    def test_bearish_inputs_give_negative_score(self):
        tech = _technical(trend_vote="BEARISH", momentum_vote="BEARISH", volume_vote="BEARISH")
        score, _, _ = SignalGenerator.model_score(tech, _regime(regime="TREND_DOWN"))
        assert score < 0


class TestDirection:
    def test_thresholds(self):
        assert SignalGenerator._determine_direction(0.30) == "BUY_CALL"
        assert SignalGenerator._determine_direction(-0.30) == "BUY_PUT"
        assert SignalGenerator._determine_direction(0.10) == "NO_TRADE"

    def test_choppy_has_no_lowered_threshold(self):
        assert SignalGenerator._determine_direction(0.20, "VOLATILE_CHOPPY") == "NO_TRADE"

    def test_crash_never_calls(self):
        assert SignalGenerator._determine_direction(1.5, "CRASH") == "NO_TRADE"


# ===========================================================================
# 2. Calibrated confidence
# ===========================================================================


class TestCalibratedConfidence:
    def test_no_calibration_file_means_no_trade(self, tmp_path):
        gen = SignalGenerator(_mock_db(), calibration=SignalCalibration(tmp_path / "missing.json"))
        sig = _signal(gen)
        assert sig.signal_type == "NO_TRADE"
        assert "calibrate_signal_model" in sig.reasoning

    def test_unvalidated_calibration_means_no_trade(self, tmp_path):
        sig = _signal(_gen(tmp_path, p=0.70, validated=False))
        assert sig.signal_type == "NO_TRADE"
        assert "out-of-sample" in sig.reasoning

    def test_validated_high_probability_trades(self, tmp_path):
        sig = _signal(_gen(tmp_path, p=0.62))
        assert sig.signal_type == "BUY_CALL"
        assert sig.confidence_score == pytest.approx(0.62)
        assert sig.confidence_level == "HIGH"

    def test_medium_band(self, tmp_path):
        sig = _signal(_gen(tmp_path, p=0.53))
        assert sig.signal_type == "BUY_CALL" and sig.confidence_level == "MEDIUM"

    def test_below_medium_is_no_trade(self, tmp_path):
        sig = _signal(_gen(tmp_path, p=0.45))
        assert sig.signal_type == "NO_TRADE"
        assert "Calibrated P(win) 0.45" in sig.reasoning

    def test_haircuts_only_lower_probability(self, tmp_path):
        gen = _gen(tmp_path, p=0.60)
        sig = _signal(gen, anomaly=_anomaly(risk_level="HIGH"),
                      regime=_regime(volatility_regime="EXTREME", regime_changing=True))
        # 0.60 - 0.05 (anomaly HIGH) - 0.05 (extreme vol) - 0.02 (transition) = 0.48
        assert sig.signal_type == "NO_TRADE"
        assert "0.48" in sig.reasoning

    def test_per_index_model_preferred_over_pooled(self, tmp_path):
        cal = _calibration(tmp_path, p=0.40, index_key="ALL")
        payload = json.loads((tmp_path / "cal.json").read_text())
        payload["models"]["NIFTY50"] = {
            "bins": [{"lo": 0.25, "hi": 99.0, "n": 100, "wins": 60, "p_win": 0.60}],
            "validated": True, "meta": {},
        }
        (tmp_path / "cal.json").write_text(json.dumps(payload))
        p, _ = SignalCalibration(tmp_path / "cal.json").lookup("NIFTY50", 0.8)
        assert p == 0.60
        assert cal.lookup("BANKNIFTY", 0.8)[0] == 0.40


# ===========================================================================
# 3. Vetoes
# ===========================================================================


class TestVetoes:
    def test_options_opposing_vetoes_call(self, tmp_path):
        tech = _technical(options=_options(vote="BEARISH", confidence=0.7))
        sig = _signal(_gen(tmp_path), technical=tech)
        assert sig.signal_type == "NO_TRADE" and "Options positioning opposes" in sig.reasoning

    def test_weak_options_disagreement_does_not_veto(self, tmp_path):
        tech = _technical(options=_options(vote="BEARISH", confidence=0.4))
        assert _signal(_gen(tmp_path), technical=tech).signal_type == "BUY_CALL"

    def test_anomaly_high_risk_opposing_vetoes(self, tmp_path):
        sig = _signal(_gen(tmp_path, p=0.70), anomaly=_anomaly(vote="BEARISH", risk_level="HIGH"))
        assert sig.signal_type == "NO_TRADE" and "Anomaly engine" in sig.reasoning

    def test_agreeing_options_do_not_boost(self, tmp_path):
        base = _signal(_gen(tmp_path, p=0.55))
        with_opts = _signal(_gen(tmp_path, p=0.55),
                            technical=_technical(options=_options(vote="STRONG_BULLISH", confidence=0.95)))
        assert with_opts.confidence_score == base.confidence_score


# ===========================================================================
# 4. Safety checks
# ===========================================================================


class TestSafetyChecks:
    def test_crash_regime_no_trade(self, tmp_path):
        sig = _signal(_gen(tmp_path), regime=_regime(regime="CRASH"))
        assert sig.signal_type == "NO_TRADE" and "CRASH" in sig.reasoning

    def test_choppy_regime_blocked(self, tmp_path):
        sig = _signal(_gen(tmp_path, p=0.8), regime=_regime(regime="VOLATILE_CHOPPY"))
        assert sig.signal_type == "NO_TRADE" and "VOLATILE_CHOPPY" in sig.reasoning

    def test_extreme_anomaly_no_trade(self, tmp_path):
        sig = _signal(_gen(tmp_path), anomaly=_anomaly(risk_level="EXTREME"))
        assert sig.signal_type == "NO_TRADE"

    def test_max_trades_reached(self, tmp_path):
        sig = _signal(_gen(tmp_path, today_count=5), regime=_regime(max_trades_today=5))
        assert sig.signal_type == "NO_TRADE" and "Max signals" in sig.reasoning

    def test_trade_count_db_error_fails_closed(self, tmp_path):
        gen = _gen(tmp_path)
        gen._db.fetch_one.side_effect = RuntimeError("db down")
        sig = _signal(gen)
        assert sig.signal_type == "NO_TRADE" and "Cannot verify" in sig.reasoning

    def test_invalid_spot(self, tmp_path):
        assert _signal(_gen(tmp_path), spot=0.0).signal_type == "NO_TRADE"

    def test_near_close(self, tmp_path):
        gen = _gen(tmp_path)
        tech = replace(_technical(), timestamp=NEAR_CLOSE)
        sig = gen.generate_signal(INDEX_ID, tech, None, None, _regime(),
                                  SPOT_PRICE, timestamp=NEAR_CLOSE)
        assert sig.signal_type == "NO_TRADE" and "close" in sig.reasoning.lower()

    def test_stale_bar_rejected(self, tmp_path):
        tech = replace(_technical(), timeframe="5m", timestamp=MORNING.replace(hour=9, minute=40))
        sig = _signal(_gen(tmp_path), technical=tech)
        assert sig.signal_type == "NO_TRADE" and "stale" in sig.reasoning.lower()


# ===========================================================================
# 5. Levels, reasoning, bookkeeping
# ===========================================================================


class TestLevelsAndOutput:
    def test_call_levels_on_correct_sides(self, tmp_path):
        sig = _signal(_gen(tmp_path))
        assert sig.stop_loss < sig.entry_price < sig.target_price
        assert sig.risk_reward_ratio >= 1.3

    def test_put_levels_on_correct_sides(self, tmp_path):
        tech = _technical(trend_vote="BEARISH", momentum_vote="BEARISH", volume_vote="BEARISH",
                          options=_options(vote="NEUTRAL"))
        sig = _signal(_gen(tmp_path), technical=tech, regime=_regime(regime="TREND_DOWN"))
        assert sig.signal_type == "BUY_PUT"
        assert sig.target_price < sig.entry_price < sig.stop_loss

    def test_reasoning_and_breakdown(self, tmp_path):
        sig = _signal(_gen(tmp_path))
        assert "BUY CALL" in sig.reasoning and "Weighted Score" in sig.reasoning
        assert set(sig.vote_breakdown) >= {"trend", "momentum"}

    def test_duplicate_suppressed_within_bucket(self, tmp_path):
        gen = _gen(tmp_path)
        first = _signal(gen)
        second = _signal(gen)
        assert first.signal_type == "BUY_CALL"
        assert second.signal_type == "NO_TRADE" and "DUPLICATE_SUPPRESSED" in second.warnings

    def test_conflicting_strong_signals_helper(self):
        conflicting = {"a": (2.0, 1, 0.25), "b": (2.0, 1, 0.25), "c": (-2.0, 1, 0.25), "d": (-2.0, 1, 0.25)}
        assert SignalGenerator._has_conflicting_strong_signals(conflicting) is True
        assert SignalGenerator._has_conflicting_strong_signals({"a": (2.0, 1, 0.5)}) is False

    def test_smart_money_conflict_warning(self):
        tech = _technical(smart_money=_smart_money(bias="BEARISH"))
        warns = SignalGenerator._smart_money_warnings("BUY_CALL", tech)
        assert warns and "disagrees" in warns[0]


class TestLevelsMatchCalibration:
    def test_levels_are_calibrated_atr_multiples(self, tmp_path):
        sig = _signal(_gen(tmp_path))
        atr = 120.0
        assert sig.entry_price - sig.stop_loss == pytest.approx(1.0 * atr, abs=0.01)
        assert sig.target_price - sig.entry_price == pytest.approx(1.5 * atr, abs=0.01)
        assert sig.risk_reward_ratio == pytest.approx(1.5)

    def test_custom_label_multiples_are_used(self, tmp_path):
        _calibration(tmp_path, p=0.6)
        payload = json.loads((tmp_path / "cal.json").read_text())
        payload["label"] = {"stop_atr": 0.8, "target_atr": 1.6}
        (tmp_path / "cal.json").write_text(json.dumps(payload))
        gen = SignalGenerator(_mock_db(), calibration=SignalCalibration(tmp_path / "cal.json"))
        sig = _signal(gen)
        assert sig.entry_price - sig.stop_loss == pytest.approx(0.8 * 120.0, abs=0.01)
        assert sig.risk_reward_ratio == pytest.approx(2.0)


def test_stop_never_tighter_than_noise_floor(tmp_path):
    tech = replace(_technical(), volatility=replace(_volatility(), atr_value=10.0))
    sig = _signal(_gen(tmp_path), technical=tech)
    floor = 0.001 * SPOT_PRICE
    assert sig.entry_price - sig.stop_loss == pytest.approx(floor, abs=0.01)
    assert sig.target_price - sig.entry_price == pytest.approx(1.5 * floor, abs=0.01)
