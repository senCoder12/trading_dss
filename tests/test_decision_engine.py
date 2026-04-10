"""
Unit tests for DecisionEngine and SignalTracker — Phase 5 master orchestrator.

Covers:
- run_full_cycle produces a valid DecisionResult for any input quality
- run_all_indices processes all F&O indices and sorts correctly
- monitor_open_positions detects SL hit, target hit, and HOLD
- Dashboard data generation (get_dashboard_data)
- Alert message formatting for BUY_CALL, BUY_PUT, NO_TRADE, exit
- Graceful degradation when news / anomaly components are absent
- Timing: full cycle < 5 seconds per index (mocked sub-components)
- SignalTracker.record_signal + record_outcome round-trip
- get_performance_stats with synthetic WIN/LOSS data
- get_calibration_report classification (OVER / CALIBRATED / UNDER)
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.analysis.anomaly.anomaly_aggregator import AnomalyVote, AnomalyDetectionResult
from src.analysis.news.news_engine import NewsVote
from src.analysis.technical_aggregator import TechnicalAnalysisResult
from src.database.db_manager import DatabaseManager
from src.engine.decision_engine import (
    DecisionEngine, DecisionResult, DashboardData, IndexDashboard,
)
from src.engine.regime_detector import MarketRegime, SignalWeights
from src.engine.risk_manager import (
    RefinedSignal, RiskConfig, RiskManager, PositionUpdate,
)
from src.engine.signal_generator import TradingSignal
from src.engine.signal_tracker import (
    SignalTracker, PerformanceStats, CalibrationReport,
)

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

INDEX_ID = "NIFTY50"
SPOT = 22_450.0
LOT_SIZE = 75
MORNING = datetime(2026, 4, 8, 10, 30, 0, tzinfo=_IST)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_regime(
    index_id: str = INDEX_ID,
    regime: str = "TREND_UP",
) -> MarketRegime:
    weights = SignalWeights()
    return MarketRegime(
        index_id=index_id,
        timestamp=MORNING,
        regime=regime,
        trend_regime="UP",
        volatility_regime="NORMAL",
        event_regime="NORMAL",
        market_phase="MARKUP",
        regime_confidence=0.75,
        regime_duration_bars=15,
        regime_changing=False,
        weight_adjustments=weights,
        position_size_multiplier=1.0,
        stop_loss_multiplier=1.0,
        max_trades_today=5,
        description="Trending up",
        warnings=[],
    )


def _make_trading_signal(
    signal_type: str = "BUY_CALL",
    confidence_level: str = "HIGH",
    confidence_score: float = 0.72,
    index_id: str = INDEX_ID,
) -> TradingSignal:
    return TradingSignal(
        signal_id=str(uuid.uuid4()),
        index_id=index_id,
        generated_at=MORNING,
        signal_type=signal_type,
        confidence_level=confidence_level,
        confidence_score=confidence_score,
        entry_price=SPOT,
        target_price=SPOT + 170,
        stop_loss=SPOT - 110,
        risk_reward_ratio=1.55,
        regime="TREND_UP",
        weighted_score=0.9 if signal_type == "BUY_CALL" else (-0.9 if signal_type == "BUY_PUT" else 0.0),
        vote_breakdown={
            "trend": {"vote": "BULLISH", "score": 1.0, "weight": 0.30},
            "options": {"vote": "BULLISH", "score": 0.8, "weight": 0.20},
            "news": {"vote": "NEUTRAL", "score": 0.0, "weight": 0.10},
            "anomaly": {"vote": "NEUTRAL", "score": 0.0, "weight": 0.05},
            "smart_money": {"vote": "BULLISH", "score": 1.0, "weight": 0.10},
        },
        risk_level="NORMAL",
        position_size_modifier=1.0,
        suggested_lot_count=2,
        estimated_max_loss=8_250.0,
        estimated_max_profit=12_750.0,
        reasoning="Strong trend + options support.",
        warnings=[],
    )


def _make_refined_signal(
    signal_type: str = "BUY_CALL",
    confidence_level: str = "HIGH",
    is_valid: bool = True,
) -> RefinedSignal:
    raw = _make_trading_signal(signal_type, confidence_level)
    return RefinedSignal(
        signal_id=raw.signal_id,
        index_id=raw.index_id,
        generated_at=raw.generated_at,
        signal_type=raw.signal_type,
        confidence_level=raw.confidence_level,
        confidence_score=raw.confidence_score,
        entry_price=raw.entry_price,
        target_price=raw.target_price,
        stop_loss=raw.stop_loss,
        risk_reward_ratio=raw.risk_reward_ratio,
        regime=raw.regime,
        weighted_score=raw.weighted_score,
        vote_breakdown=raw.vote_breakdown,
        risk_level=raw.risk_level,
        position_size_modifier=raw.position_size_modifier,
        suggested_lot_count=raw.suggested_lot_count,
        estimated_max_loss=raw.estimated_max_loss,
        estimated_max_profit=raw.estimated_max_profit,
        reasoning=raw.reasoning,
        warnings=raw.warnings,
        outcome=None,
        actual_exit_price=None,
        actual_pnl=None,
        closed_at=None,
        data_completeness=1.0,
        signals_generated_today=1,
        refined_entry=SPOT,
        refined_target=SPOT + 170,
        refined_stop_loss=SPOT - 110,
        lots=2,
        total_margin_required=0.0,
        max_loss_amount=8_250.0,
        max_profit_amount=12_750.0,
        transaction_cost_total=480.0,
        breakeven_move=3.2,
        recommended_strike=22_450.0,
        recommended_expiry="10-Apr-2026",
        option_premium=185.0,
        option_greeks={"delta": 0.50, "theta": -12.0},
        risk_amount=8_250.0,
        risk_pct_of_capital=2.75,
        daily_loss_remaining=4_250.0,
        is_valid=is_valid,
        rejection_reasons=[] if is_valid else ["Test rejection"],
        adjustments_made=[],
        execution_type="LIMIT",
        limit_price=SPOT,
        validity="DAY",
    )


def _make_technical_result(index_id: str = INDEX_ID) -> TechnicalAnalysisResult:
    from src.analysis.indicators.trend import TrendSummary
    from src.analysis.indicators.momentum import MomentumSummary
    from src.analysis.indicators.volatility import VolatilitySummary
    from src.analysis.indicators.volume import VolumeSummary
    from src.analysis.indicators.quant import QuantSummary
    from src.analysis.indicators.options_indicators import OptionsSummary

    trend = TrendSummary(
        index_id=index_id,
        timeframe="1d",
        timestamp=MORNING,
        price_vs_ema20="ABOVE",
        price_vs_ema50="ABOVE",
        price_vs_ema200="ABOVE",
        ema_alignment="BULLISH",
        golden_cross=False,
        death_cross=False,
        macd_signal="BULLISH",
        macd_crossover=None,
        macd_histogram_trend="RISING",
        trend_strength="STRONG",
        trend_direction="UP",
        trend_vote="BULLISH",
        trend_confidence=0.8,
    )
    momentum = MomentumSummary(
        timestamp=MORNING,
        rsi_value=58.0,
        rsi_zone="NEUTRAL",
        rsi_divergence=None,
        stochastic_k=62.0,
        stochastic_zone="NEUTRAL",
        stochastic_crossover=None,
        cci_value=85.0,
        cci_zone="NEUTRAL",
        momentum_vote="BULLISH",
        momentum_confidence=0.65,
        overbought_consensus=False,
        oversold_consensus=False,
        divergence_detected=False,
        reversal_warning=None,
    )
    volatility = VolatilitySummary(
        timestamp=MORNING,
        bb_position="MIDDLE",
        bb_squeeze=False,
        bb_bandwidth_percentile=50.0,
        atr_value=120.0,
        atr_pct=0.53,
        volatility_level="NORMAL",
        suggested_sl=110.0,
        suggested_target=170.0,
        hv_current=0.12,
        hv_regime="NORMAL",
        vix_regime=None,
        volatility_vote="NEUTRAL",
        volatility_confidence=0.4,
        position_size_modifier=1.0,
        breakout_alert=False,
        mean_reversion_setup=False,
    )
    volume = VolumeSummary(
        timestamp=MORNING,
        price_vs_vwap="ABOVE",
        vwap_zone="ABOVE",
        institutional_bias="NEUTRAL",
        obv_trend="RISING",
        obv_divergence=None,
        accumulation_distribution="ACCUMULATION",
        poc=22_400.0,
        value_area_high=22_600.0,
        value_area_low=22_200.0,
        in_value_area=True,
        volume_ratio=1.8,
        volume_confirms_price=True,
        volume_vote="BULLISH",
        volume_confidence=0.6,
    )
    opts = OptionsSummary(
        timestamp=MORNING,
        index_id=index_id,
        expiry_date=date(2026, 4, 10),
        days_to_expiry=2,
        pcr=1.15,
        pcr_signal="BULLISH",
        oi_support=22_000.0,
        oi_resistance=22_800.0,
        expected_range=(22_200.0, 22_700.0),
        max_pain=22_400.0,
        max_pain_pull="NEUTRAL",
        oi_change_signal="BULLISH",
        dominant_buildup="PE_WRITING",
        atm_iv=14.5,
        iv_regime="NORMAL",
        iv_skew=0.02,
        options_vote="BULLISH",
        options_confidence=0.7,
    )
    quant = QuantSummary(
        timestamp=MORNING,
        zscore=0.3,
        zscore_zone="NEUTRAL",
        mean_reversion_signal=None,
        beta=0.95,
        alpha=None,
        beta_interpretation=None,
        statistical_regime="NORMAL",
        quant_vote="NEUTRAL",
        quant_confidence=0.4,
    )
    from src.analysis.indicators.smart_money import SmartMoneyScore
    smart = SmartMoneyScore(
        score=0.65,
        grade="B",
        smfi_component=0.6,
        vsd_component=0.5,
        btd_component=0.7,
        oimi_component=0.6,
        lai_component=0.5,
        smart_money_bias="ACCUMULATION",
        key_finding="FII buying detected",
        actionable_insight="Institutional accumulation supports long bias",
        data_completeness=1.0,
        confidence=0.65,
    )

    return TechnicalAnalysisResult(
        index_id=index_id,
        timestamp=MORNING,
        timeframe="1d",
        trend=trend,
        momentum=momentum,
        volatility=volatility,
        volume=volume,
        options=opts,
        quant=quant,
        smart_money=smart,
        votes={
            "trend": "BULLISH", "momentum": "BULLISH", "options": "BULLISH",
            "volume": "BULLISH", "smart_money": "BULLISH",
        },
        bullish_votes=5, bearish_votes=0, neutral_votes=2,
        overall_signal="STRONG_BUY",
        overall_confidence=0.78,
        support_levels=[22_200.0, 22_000.0],
        resistance_levels=[22_600.0, 22_800.0],
        immediate_support=22_200.0,
        immediate_resistance=22_600.0,
        suggested_stop_loss_distance=110.0,
        suggested_target_distance=170.0,
        position_size_modifier=1.0,
        alerts=[],
        reasoning="Strong bullish setup",
        data_completeness=1.0,
        warnings=[],
    )


# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path) -> DatabaseManager:
    db = DatabaseManager(db_path=tmp_path / "test_de.db")
    db.connect()
    db.initialise_schema()
    now = datetime.now(tz=_IST).isoformat()
    # Seed index_master
    for idx_id, display, lot in [
        ("NIFTY50", "NIFTY 50", 75),
        ("BANKNIFTY", "NIFTY BANK", 15),
    ]:
        db.execute(
            """
            INSERT OR IGNORE INTO index_master
                (id, display_name, nse_symbol, yahoo_symbol, exchange,
                 lot_size, has_options, option_symbol, sector_category,
                 is_active, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (idx_id, display, idx_id, f"^{idx_id}", "NSE",
             lot, 1, idx_id, "broad_market", 1, now, now),
        )

    # Seed price data so _step_gather_data finds bars
    import numpy as np
    rng = np.random.default_rng(42)
    start = datetime(2025, 6, 1, 9, 15, 0, tzinfo=_IST)
    for idx_id, base in [("NIFTY50", SPOT), ("BANKNIFTY", 48_200.0)]:
        close = base
        for i in range(60):
            ts = (start + timedelta(days=i)).isoformat()
            change = rng.normal(0, base * 0.005)
            close = max(base * 0.85, close + change)
            h = close + abs(rng.normal(0, base * 0.002))
            l = close - abs(rng.normal(0, base * 0.002))
            o = close + rng.normal(0, base * 0.001)
            db.execute(
                """
                INSERT OR IGNORE INTO price_data
                    (index_id, timestamp, open, high, low, close, volume, source, timeframe)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (idx_id, ts, round(o, 2), round(h, 2), round(l, 2),
                 round(close, 2), 2_000_000, "yfinance", "1d"),
            )
        # Current 5m bar
        db.execute(
            """
            INSERT OR IGNORE INTO price_data
                (index_id, timestamp, open, high, low, close, volume, source, timeframe)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (idx_id, now, close, close + 20, close - 10, close + 5, 500_000, "nse_live", "5m"),
        )

    # Seed VIX
    db.execute(
        "INSERT OR IGNORE INTO vix_data (timestamp, vix_value, vix_change, vix_change_pct) VALUES (?,?,?,?)",
        (now, 14.5, -0.3, -2.0),
    )
    return db


# ---------------------------------------------------------------------------
# DecisionEngine fixture with heavily mocked sub-components
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine(tmp_db: DatabaseManager, tmp_path: Path) -> DecisionEngine:
    """Return a DecisionEngine with all heavy sub-components mocked."""
    import json as _json

    # Write a minimal indices.json so the registry loads
    indices_path = tmp_path / "indices.json"
    indices_path.write_text(
        _json.dumps([
            {
                "id": "NIFTY50", "display_name": "NIFTY 50",
                "nse_symbol": "NIFTY 50", "yahoo_symbol": "^NSEI",
                "exchange": "NSE", "lot_size": 75, "has_options": True,
                "option_symbol": "NIFTY", "sector_category": "broad_market",
                "is_active": True, "description": "Test NIFTY 50",
            },
            {
                "id": "BANKNIFTY", "display_name": "NIFTY BANK",
                "nse_symbol": "NIFTY BANK", "yahoo_symbol": "^NSEBANK",
                "exchange": "NSE", "lot_size": 15, "has_options": True,
                "option_symbol": "BANKNIFTY", "sector_category": "sectoral",
                "is_active": True, "description": "Test BankNifty",
            },
        ]),
        encoding="utf-8",
    )

    with (
        patch("src.engine.decision_engine.get_registry") as mock_registry,
        patch("src.engine.decision_engine.TechnicalAggregator") as MockTA,
        patch("src.engine.decision_engine.NewsEngine") as MockNE,
        patch("src.engine.decision_engine.AnomalyAggregator") as MockAA,
        patch("src.engine.decision_engine.RegimeDetector") as MockRD,
        patch("src.engine.decision_engine.SignalGenerator") as MockSG,
        patch("src.engine.decision_engine.RiskManager") as MockRM,
    ):
        # Registry — return Index-like objects with .id attribute
        _nifty_idx = MagicMock(id="NIFTY50", display_name="NIFTY 50", has_options=True)
        _bn_idx = MagicMock(id="BANKNIFTY", display_name="NIFTY BANK", has_options=True)
        mock_reg = MagicMock()
        mock_reg.get_indices_with_options.return_value = [_nifty_idx, _bn_idx]
        mock_reg.get_index.side_effect = lambda idx: MagicMock(
            display_name={"NIFTY50": "NIFTY 50", "BANKNIFTY": "NIFTY BANK"}.get(idx, idx),
            has_options=True,
            is_active=True,
        )
        mock_registry.return_value = mock_reg

        # TechnicalAggregator
        tech_result = _make_technical_result()
        mock_ta = MagicMock()
        mock_ta.analyze.return_value = tech_result
        MockTA.return_value = mock_ta

        # NewsEngine
        mock_ne = MagicMock()
        news_vote = NewsVote(
            index_id=INDEX_ID,
            timestamp=MORNING,
            vote="BULLISH",
            confidence=0.6,
            active_article_count=3,
            weighted_sentiment=0.4,
            top_headline="Markets rally on positive data",
            event_regime="NORMAL",
            reasoning="Positive macro news",
        )
        mock_ne.get_news_vote.return_value = news_vote
        mock_ne.get_all_news_votes.return_value = {INDEX_ID: news_vote}
        mock_ne.get_news_feed.return_value = []
        mock_ne._lock = __import__("threading").Lock()
        mock_ne._last_event_regime = {}
        mock_ne._last_result = None
        MockNE.return_value = mock_ne

        # AnomalyAggregator
        anomaly_result = MagicMock(spec=AnomalyDetectionResult)
        anomaly_result.anomaly_vote = "NEUTRAL"
        anomaly_result.anomaly_confidence = 0.1
        anomaly_result.risk_level = "NORMAL"
        anomaly_result.position_size_modifier = 1.0
        anomaly_result.active_alert_count = 0
        anomaly_result.primary_alert = None
        anomaly_result.institutional_activity_detected = False
        anomaly_result.summary = "No anomalies"
        mock_aa = MagicMock()
        mock_aa.run_detection_cycle.return_value = anomaly_result
        mock_aa._open_positions = {}
        MockAA.return_value = mock_aa

        # RegimeDetector
        regime = _make_regime()
        mock_rd = MagicMock()
        mock_rd.detect_regime.return_value = regime
        MockRD.return_value = mock_rd

        # SignalGenerator
        raw_signal = _make_trading_signal()
        mock_sg = MagicMock()
        mock_sg.generate_signal.return_value = raw_signal
        MockSG.return_value = mock_sg

        # RiskManager
        refined = _make_refined_signal()
        mock_rm = MagicMock()
        mock_rm.validate_and_refine_signal.return_value = refined
        mock_rm._open_positions = {}
        port = MagicMock()
        port.open_positions = []
        port.today_pnl = 0.0
        mock_rm.get_portfolio_summary.return_value = port
        MockRM.return_value = mock_rm

        eng = DecisionEngine(tmp_db, risk_config=RiskConfig(total_capital=100_000.0))

    # Attach mocks for inspection in tests
    eng._mock_ta  = mock_ta
    eng._mock_ne  = mock_ne
    eng._mock_aa  = mock_aa
    eng._mock_rd  = mock_rd
    eng._mock_sg  = mock_sg
    eng._mock_rm  = mock_rm
    eng._mock_reg = mock_reg

    return eng


# ---------------------------------------------------------------------------
# run_full_cycle tests
# ---------------------------------------------------------------------------

class TestRunFullCycle:
    def test_returns_decision_result(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle(INDEX_ID)
        assert isinstance(result, DecisionResult)
        assert result.index_id == INDEX_ID
        assert result.timestamp is not None

    def test_is_actionable_when_refined_valid(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle(INDEX_ID)
        assert result.is_actionable is True

    def test_signal_is_refined_signal_when_actionable(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle(INDEX_ID)
        assert isinstance(result.signal, RefinedSignal)

    def test_intermediate_results_populated(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle(INDEX_ID)
        assert result.technical_result is not None
        assert result.regime is not None
        assert result.news_vote is not None
        assert result.anomaly_vote is not None

    def test_step_timings_populated(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle(INDEX_ID)
        expected_steps = [
            "1_gather_data", "2_technical", "3_news",
            "4_anomaly", "5_regime", "6_signal",
            "7_risk", "8_store_alert",
        ]
        for step in expected_steps:
            assert step in result.step_timings
            assert isinstance(result.step_timings[step], int)
            assert result.step_timings[step] >= 0

    def test_total_duration_reasonable(self, engine: DecisionEngine) -> None:
        """Full cycle (mocked) must finish well under 5 seconds."""
        t0 = time.monotonic()
        engine.run_full_cycle(INDEX_ID)
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"Cycle took {elapsed:.2f}s — too slow"

    def test_alert_message_present_when_actionable(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle(INDEX_ID)
        assert result.alert_message is not None
        assert len(result.alert_message) > 50
        assert "BUY CALL" in result.alert_message or "BUY PUT" in result.alert_message

    def test_alert_priority_critical_for_high_confidence(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle(INDEX_ID)
        # Our fixture signal is HIGH confidence
        assert result.alert_priority == "CRITICAL"

    def test_dashboard_summary_has_all_keys(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle(INDEX_ID)
        ds = result.dashboard_summary
        for key in ("index_id", "signal_type", "confidence_level", "is_actionable", "regime"):
            assert key in ds, f"Key '{key}' missing from dashboard_summary"

    def test_no_trade_when_invalid_refined(self, engine: DecisionEngine) -> None:
        """When RiskManager returns invalid refined signal, is_actionable = False."""
        invalid_refined = _make_refined_signal(is_valid=False)
        engine._mock_rm.validate_and_refine_signal.return_value = invalid_refined

        result = engine.run_full_cycle(INDEX_ID)
        assert result.is_actionable is False

    def test_result_cached_after_cycle(self, engine: DecisionEngine) -> None:
        engine.run_full_cycle(INDEX_ID)
        with engine._lock:
            assert INDEX_ID in engine._result_cache

    def test_missing_technical_result_returns_no_trade(
        self, engine: DecisionEngine
    ) -> None:
        engine._mock_ta.analyze.return_value = None
        # Also no price bar in DB — step_gather returns None current_bar
        result = engine.run_full_cycle(INDEX_ID)
        # Even with None technical, the engine should return without crashing
        assert isinstance(result, DecisionResult)
        # Signal must be NO_TRADE since technical is unavailable
        assert getattr(result.signal, "signal_type", "NO_TRADE") == "NO_TRADE"

    def test_missing_news_still_produces_signal(self, engine: DecisionEngine) -> None:
        engine._mock_ne.get_news_vote.return_value = None
        engine._mock_ne.run_news_cycle.side_effect = RuntimeError("news down")

        # Replace with explicit no-trade from signal_generator for clarity
        engine._mock_sg.generate_signal.return_value = _make_trading_signal()
        result = engine.run_full_cycle(INDEX_ID)
        assert isinstance(result, DecisionResult)

    def test_anomaly_failure_does_not_crash(self, engine: DecisionEngine) -> None:
        engine._mock_aa.run_detection_cycle.side_effect = RuntimeError("anomaly fail")
        result = engine.run_full_cycle(INDEX_ID)
        assert isinstance(result, DecisionResult)
        # anomaly_vote should be None (graceful degradation)
        assert result.anomaly_vote is None


# ---------------------------------------------------------------------------
# run_all_indices tests
# ---------------------------------------------------------------------------

class TestRunAllIndices:
    def test_returns_list_of_results(self, engine: DecisionEngine) -> None:
        results = engine.run_all_indices()
        assert isinstance(results, list)
        assert len(results) == 2  # NIFTY50 + BANKNIFTY from mock registry

    def test_actionable_signals_sorted_first(self, engine: DecisionEngine) -> None:
        # Make BANKNIFTY produce NO_TRADE
        def side_effect_sg(index_id, **kw):
            if index_id == "BANKNIFTY":
                return _make_trading_signal("NO_TRADE", "LOW", 0.0, "BANKNIFTY")
            return _make_trading_signal()

        engine._mock_sg.generate_signal.side_effect = side_effect_sg
        # RiskManager must return invalid for NO_TRADE
        def side_effect_rm(signal, **kw):
            if signal.signal_type == "NO_TRADE":
                return _make_refined_signal("NO_TRADE", "LOW", is_valid=False)
            return _make_refined_signal()

        engine._mock_rm.validate_and_refine_signal.side_effect = side_effect_rm

        results = engine.run_all_indices()
        if len(results) >= 2:
            # First result should be actionable if any
            actionable_indices = [i for i, r in enumerate(results) if r.is_actionable]
            no_trade_indices = [i for i, r in enumerate(results) if not r.is_actionable]
            if actionable_indices and no_trade_indices:
                assert max(actionable_indices) < min(no_trade_indices)

    def test_empty_registry_returns_empty_list(self, engine: DecisionEngine) -> None:
        engine._mock_reg.get_indices_with_options.return_value = []
        results = engine.run_all_indices()
        assert results == []

    def test_single_engine_crash_does_not_abort_others(
        self, engine: DecisionEngine
    ) -> None:
        call_count = [0]
        original = engine.run_full_cycle

        def patched_cycle(idx):
            call_count[0] += 1
            if idx == "NIFTY50":
                raise RuntimeError("injected crash")
            return original(idx)

        engine.run_full_cycle = patched_cycle
        results = engine.run_all_indices()
        # BANKNIFTY should still produce a result
        assert any(r.index_id == "BANKNIFTY" for r in results)


# ---------------------------------------------------------------------------
# monitor_open_positions tests
# ---------------------------------------------------------------------------

class TestMonitorOpenPositions:
    def _seed_position(self, engine: DecisionEngine, signal_id: str) -> None:
        """Inject a fake open position into RiskManager's _open_positions."""
        from src.engine.risk_manager import _OpenPosition
        pos = _OpenPosition(
            db_id=1,
            signal_id=signal_id,
            index_id=INDEX_ID,
            signal_type="BUY_CALL",
            entry_price=SPOT,
            current_sl=SPOT - 110,
            target_price=SPOT + 170,
            lots=2,
            lot_size=LOT_SIZE,
            entry_time=MORNING,
            confidence_level="HIGH",
        )
        engine.risk_manager._open_positions[signal_id] = pos

    def test_target_hit_closes_position(
        self, engine: DecisionEngine, tmp_db: DatabaseManager
    ) -> None:
        sid = str(uuid.uuid4())
        self._seed_position(engine, sid)

        # Price above target → EXIT_TARGET
        engine.risk_manager.update_position = MagicMock(
            return_value=PositionUpdate(
                signal_id=sid, action="EXIT_TARGET",
                current_pnl=12_750.0, current_pnl_pct=0.0127,
                time_in_trade_minutes=75,
            )
        )
        engine.risk_manager.close_position = MagicMock()
        engine.tracker.record_outcome = MagicMock()

        # Seed a price bar so _step_gather_data finds it
        now_str = datetime.now(tz=_IST).isoformat()
        tmp_db.execute(
            "INSERT OR IGNORE INTO price_data (index_id, timestamp, open, high, low, close, volume, source, timeframe) VALUES (?,?,?,?,?,?,?,?,?)",
            (INDEX_ID, now_str, SPOT+150, SPOT+200, SPOT, SPOT+175, 1_000_000, "nse_live", "5m"),
        )

        engine.monitor_open_positions()

        engine.risk_manager.close_position.assert_called_once()
        engine.tracker.record_outcome.assert_called_once()
        _, kwargs = engine.tracker.record_outcome.call_args
        assert engine.tracker.record_outcome.call_args[1].get("pnl", 12_750.0) > 0 or \
               engine.tracker.record_outcome.call_args[0][3] > 0

    def test_sl_hit_closes_position(
        self, engine: DecisionEngine, tmp_db: DatabaseManager
    ) -> None:
        sid = str(uuid.uuid4())
        self._seed_position(engine, sid)

        engine.risk_manager.update_position = MagicMock(
            return_value=PositionUpdate(
                signal_id=sid, action="EXIT_SL",
                current_pnl=-8_250.0, current_pnl_pct=-0.0082,
                time_in_trade_minutes=30,
            )
        )
        engine.risk_manager.close_position = MagicMock()
        engine.tracker.record_outcome = MagicMock()

        now_str = datetime.now(tz=_IST).isoformat()
        tmp_db.execute(
            "INSERT OR IGNORE INTO price_data (index_id, timestamp, open, high, low, close, volume, source, timeframe) VALUES (?,?,?,?,?,?,?,?,?)",
            (INDEX_ID, now_str, SPOT-120, SPOT-100, SPOT-150, SPOT-115, 2_000_000, "nse_live", "5m"),
        )

        engine.monitor_open_positions()
        engine.risk_manager.close_position.assert_called_once()

    def test_hold_does_not_close_position(
        self, engine: DecisionEngine, tmp_db: DatabaseManager
    ) -> None:
        sid = str(uuid.uuid4())
        self._seed_position(engine, sid)

        engine.risk_manager.update_position = MagicMock(
            return_value=PositionUpdate(
                signal_id=sid, action="HOLD",
                current_pnl=3_000.0, current_pnl_pct=0.003,
                time_in_trade_minutes=45,
            )
        )
        engine.risk_manager.close_position = MagicMock()
        engine.tracker.record_outcome = MagicMock()

        now_str = datetime.now(tz=_IST).isoformat()
        tmp_db.execute(
            "INSERT OR IGNORE INTO price_data (index_id, timestamp, open, high, low, close, volume, source, timeframe) VALUES (?,?,?,?,?,?,?,?,?)",
            (INDEX_ID, now_str, SPOT+30, SPOT+50, SPOT, SPOT+40, 500_000, "nse_live", "5m"),
        )

        engine.monitor_open_positions()
        engine.risk_manager.close_position.assert_not_called()
        engine.tracker.record_outcome.assert_not_called()

    def test_no_open_positions_is_noop(self, engine: DecisionEngine) -> None:
        engine.risk_manager._open_positions = {}
        engine.risk_manager.update_position = MagicMock()
        engine.monitor_open_positions()
        engine.risk_manager.update_position.assert_not_called()


# ---------------------------------------------------------------------------
# Alert message formatting
# ---------------------------------------------------------------------------

class TestGenerateAlertMessage:
    def _make_result(
        self,
        signal_type: str = "BUY_CALL",
        is_actionable: bool = True,
    ) -> DecisionResult:
        refined = _make_refined_signal(signal_type, "HIGH", is_valid=is_actionable)
        return DecisionResult(
            index_id=INDEX_ID,
            timestamp=MORNING,
            signal=refined,
            is_actionable=is_actionable,
            technical_result=_make_technical_result(),
            news_vote=NewsVote(
                index_id=INDEX_ID, timestamp=MORNING,
                vote="BULLISH", confidence=0.6,
                active_article_count=3, weighted_sentiment=0.4,
                top_headline="Markets rally", event_regime="NORMAL",
                reasoning="Positive news",
            ),
            regime=_make_regime(),
        )

    def test_buy_call_alert_contains_expected_fields(
        self, engine: DecisionEngine
    ) -> None:
        result = self._make_result("BUY_CALL")
        msg = engine.generate_alert_message(result)
        assert "BUY CALL" in msg
        assert "Entry:" in msg
        assert "Target:" in msg
        assert "Stop Loss:" in msg
        assert "RR:" in msg
        assert "Risk:" in msg

    def test_buy_put_alert_contains_expected_fields(
        self, engine: DecisionEngine
    ) -> None:
        result = self._make_result("BUY_PUT")
        msg = engine.generate_alert_message(result)
        assert "BUY PUT" in msg
        assert "Entry:" in msg

    def test_no_trade_returns_empty_string(self, engine: DecisionEngine) -> None:
        result = self._make_result("NO_TRADE", is_actionable=False)
        msg = engine.generate_alert_message(result)
        assert msg == ""

    def test_alert_includes_strike_info_when_available(
        self, engine: DecisionEngine
    ) -> None:
        result = self._make_result("BUY_CALL")
        msg = engine.generate_alert_message(result)
        assert "22450" in msg or "Strike" in msg

    def test_alert_includes_regime(self, engine: DecisionEngine) -> None:
        result = self._make_result("BUY_CALL")
        msg = engine.generate_alert_message(result)
        assert "Trend Up" in msg or "TREND" in msg.upper()


# ---------------------------------------------------------------------------
# Dashboard data
# ---------------------------------------------------------------------------

class TestGetDashboardData:
    def test_returns_dashboard_data(self, engine: DecisionEngine) -> None:
        dash = engine.get_dashboard_data()
        assert isinstance(dash, DashboardData)

    def test_market_status_is_valid(self, engine: DecisionEngine) -> None:
        dash = engine.get_dashboard_data()
        assert dash.market_status in ("OPEN", "CLOSED", "PRE_MARKET")

    def test_indices_populated(self, engine: DecisionEngine) -> None:
        # Run a cycle first so the cache has data
        engine.run_full_cycle(INDEX_ID)
        dash = engine.get_dashboard_data()
        assert isinstance(dash.indices, list)

    def test_does_not_raise_on_empty_db(self, engine: DecisionEngine) -> None:
        # Should gracefully handle empty DB
        dash = engine.get_dashboard_data()
        assert dash is not None


# ---------------------------------------------------------------------------
# SignalTracker unit tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def tracker(tmp_db: DatabaseManager) -> SignalTracker:
    return SignalTracker(tmp_db, capital=100_000.0)


class TestSignalTracker:
    def test_record_signal_inserts_row(
        self, tracker: SignalTracker, tmp_db: DatabaseManager
    ) -> None:
        sig = _make_trading_signal("BUY_CALL", "HIGH")
        tracker.record_signal(sig)

        rows = tmp_db.fetch_all(
            "SELECT * FROM trading_signals WHERE signal_type = 'BUY_CALL'", ()
        )
        assert len(rows) >= 1
        assert rows[0]["confidence_level"] == "HIGH"

    def test_record_no_trade_inserts_with_null_outcome(
        self, tracker: SignalTracker, tmp_db: DatabaseManager
    ) -> None:
        sig = _make_trading_signal("NO_TRADE", "LOW", 0.0)
        tracker.record_signal(sig)

        rows = tmp_db.fetch_all(
            "SELECT * FROM trading_signals WHERE signal_type = 'NO_TRADE'", ()
        )
        assert len(rows) >= 1
        assert rows[0]["outcome"] is None

    def test_record_outcome_updates_db(
        self, tracker: SignalTracker, tmp_db: DatabaseManager
    ) -> None:
        sig = _make_trading_signal("BUY_CALL", "HIGH")
        tracker.record_signal(sig)

        # Must use real DB to test round-trip
        tracker.record_outcome(
            signal_id=sig.signal_id,
            exit_price=SPOT + 170,
            exit_reason="TARGET",
            pnl=12_750.0,
        )

        rows = tmp_db.fetch_all(
            "SELECT outcome, actual_pnl FROM trading_signals WHERE signal_type='BUY_CALL'",
            (),
        )
        assert len(rows) >= 1
        # After update, outcome should be WIN (pnl > 0)
        assert rows[0]["outcome"] == "WIN"
        assert rows[0]["actual_pnl"] == 12_750.0

    def test_get_performance_stats_empty_db(self, tracker: SignalTracker) -> None:
        stats = tracker.get_performance_stats(days=30)
        assert isinstance(stats, PerformanceStats)
        assert stats.total_signals == 0
        assert stats.total_trades == 0
        assert stats.win_rate == 0.0
        assert not stats.is_profitable

    def test_get_performance_stats_with_wins_and_losses(
        self, tracker: SignalTracker, tmp_db: DatabaseManager
    ) -> None:
        """Seed 6 trades: 4 WIN, 2 LOSS → win_rate = 0.667."""
        now_str = datetime.now(tz=_IST).isoformat()
        pnl_rows = [
            (5_000.0, "WIN"),
            (3_200.0, "WIN"),
            (8_100.0, "WIN"),
            (2_400.0, "WIN"),
            (-4_500.0, "LOSS"),
            (-3_800.0, "LOSS"),
        ]
        for pnl, outcome in pnl_rows:
            tmp_db.execute(
                """
                INSERT INTO trading_signals
                    (index_id, generated_at, signal_type, confidence_level,
                     entry_price, target_price, stop_loss, risk_reward_ratio,
                     regime, technical_vote, options_vote, news_vote, anomaly_vote,
                     reasoning, outcome, actual_exit_price, actual_pnl, closed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    INDEX_ID, now_str, "BUY_CALL", "HIGH",
                    SPOT, SPOT + 170, SPOT - 110, 1.55,
                    "TRENDING", "BULLISH", "BULLISH", "NEUTRAL", "NEUTRAL",
                    '{"signal_id":"' + str(uuid.uuid4()) + '"}',
                    outcome, SPOT + (170 if pnl > 0 else -110), pnl, now_str,
                ),
            )

        stats = tracker.get_performance_stats(days=30)
        assert stats.total_trades == 6
        assert stats.wins == 4
        assert stats.losses == 2
        assert abs(stats.win_rate - 4 / 6) < 0.01
        assert stats.total_pnl == pytest.approx(sum(p for p, _ in pnl_rows), abs=0.1)
        assert stats.largest_win == pytest.approx(8_100.0)
        assert stats.largest_loss == pytest.approx(-4_500.0)
        assert stats.profit_factor > 1.0
        assert stats.is_profitable

    def test_performance_stats_expected_value(
        self, tracker: SignalTracker, tmp_db: DatabaseManager
    ) -> None:
        """EV = (WR × avg_win) - ((1-WR) × avg_loss)."""
        now_str = datetime.now(tz=_IST).isoformat()
        for pnl, outcome in [(5_000.0, "WIN"), (-3_000.0, "LOSS")]:
            tmp_db.execute(
                """
                INSERT INTO trading_signals
                    (index_id, generated_at, signal_type, confidence_level,
                     entry_price, target_price, stop_loss, risk_reward_ratio,
                     regime, technical_vote, options_vote, news_vote, anomaly_vote,
                     reasoning, outcome, actual_exit_price, actual_pnl, closed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    INDEX_ID, now_str, "BUY_CALL", "MEDIUM",
                    SPOT, SPOT + 170, SPOT - 110, 1.55,
                    "TRENDING", "BULLISH", "NEUTRAL", "NEUTRAL", "NEUTRAL",
                    '{"signal_id":"' + str(uuid.uuid4()) + '"}',
                    outcome, SPOT + (170 if pnl > 0 else -110), pnl, now_str,
                ),
            )

        stats = tracker.get_performance_stats(days=30)
        # WR=0.5, avg_win=5000, avg_loss=-3000 → EV = 0.5*5000 - 0.5*3000 = 1000
        assert stats.expected_value_per_trade == pytest.approx(1_000.0, abs=1.0)

    def test_get_signal_history_filters_by_index(
        self, tracker: SignalTracker, tmp_db: DatabaseManager
    ) -> None:
        now_str = datetime.now(tz=_IST).isoformat()
        for idx in ["NIFTY50", "BANKNIFTY", "NIFTY50"]:
            tmp_db.execute(
                """
                INSERT INTO trading_signals
                    (index_id, generated_at, signal_type, confidence_level,
                     entry_price, target_price, stop_loss, risk_reward_ratio,
                     regime, technical_vote, options_vote, news_vote, anomaly_vote,
                     reasoning)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (idx, now_str, "NO_TRADE", "LOW", 0, 0, 0, 0, "RANGE_BOUND",
                 "NEUTRAL", "NEUTRAL", "NEUTRAL", "NEUTRAL", "{}"),
            )

        history = tracker.get_signal_history(index_id="NIFTY50", days=30)
        assert all(r["index_id"] == "NIFTY50" for r in history)
        assert len(history) == 2

    def test_get_calibration_report_structure(self, tracker: SignalTracker) -> None:
        report = tracker.get_calibration_report()
        assert isinstance(report, CalibrationReport)
        assert report.high_confidence_expected_win_rate > 0
        assert report.overall_calibration in ("WELL_CALIBRATED", "NEEDS_ADJUSTMENT")
        assert isinstance(report.suggested_adjustments, list)

    def test_calibration_over_confident(
        self, tracker: SignalTracker, tmp_db: DatabaseManager
    ) -> None:
        """HIGH confidence at only 40% win rate → OVER_CONFIDENT."""
        now_str = datetime.now(tz=_IST).isoformat()
        for i in range(10):
            outcome = "WIN" if i < 4 else "LOSS"
            pnl = 5_000.0 if i < 4 else -3_000.0
            tmp_db.execute(
                """
                INSERT INTO trading_signals
                    (index_id, generated_at, signal_type, confidence_level,
                     entry_price, target_price, stop_loss, risk_reward_ratio,
                     regime, technical_vote, options_vote, news_vote, anomaly_vote,
                     reasoning, outcome, actual_exit_price, actual_pnl, closed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    INDEX_ID, now_str, "BUY_CALL", "HIGH",
                    SPOT, SPOT+170, SPOT-110, 1.55,
                    "TRENDING", "BULLISH", "NEUTRAL", "NEUTRAL", "NEUTRAL",
                    '{"signal_id":"' + str(uuid.uuid4()) + '"}',
                    outcome, SPOT + (170 if pnl > 0 else -110), pnl, now_str,
                ),
            )

        report = tracker.get_calibration_report()
        assert report.high_confidence_calibration == "OVER_CONFIDENT"


# ---------------------------------------------------------------------------
# Market status helper
# ---------------------------------------------------------------------------

class TestMarketStatus:
    @pytest.mark.parametrize("h,m,expected", [
        (9, 10, "PRE_MARKET"),
        (9, 15, "OPEN"),
        (10, 30, "OPEN"),
        (15, 30, "OPEN"),
        (15, 31, "CLOSED"),
        (17, 0, "CLOSED"),
        (8, 59, "CLOSED"),
    ])
    def test_market_status(self, h: int, m: int, expected: str) -> None:
        dt = datetime(2026, 4, 8, h, m, 0, tzinfo=_IST)
        assert DecisionEngine._get_market_status(dt) == expected


# ---------------------------------------------------------------------------
# Max drawdown and Sharpe helpers
# ---------------------------------------------------------------------------

class TestSignalTrackerHelpers:
    def test_max_drawdown_basic(self) -> None:
        # Sequence: +100, +200 (peak=300), then -400 → drawdown = 400
        pnls = [100.0, 200.0, -400.0, 50.0]
        dd, dd_pct = SignalTracker._compute_max_drawdown(pnls)
        assert dd == pytest.approx(400.0, abs=0.01)

    def test_max_drawdown_all_wins(self) -> None:
        dd, _ = SignalTracker._compute_max_drawdown([100.0, 200.0, 300.0])
        assert dd == 0.0

    def test_max_drawdown_empty(self) -> None:
        dd, dd_pct = SignalTracker._compute_max_drawdown([])
        assert dd == 0.0

    def test_sharpe_positive_series(self) -> None:
        pnls = [500.0] * 20 + [-100.0] * 5
        sharpe = SignalTracker._compute_sharpe(pnls, 30)
        assert sharpe > 0

    def test_sharpe_flat_series(self) -> None:
        pnls = [100.0] * 10
        sharpe = SignalTracker._compute_sharpe(pnls, 30)
        # Std = 0 → Sharpe = 0
        assert sharpe == 0.0

    def test_sharpe_empty(self) -> None:
        sharpe = SignalTracker._compute_sharpe([], 30)
        assert sharpe == 0.0
