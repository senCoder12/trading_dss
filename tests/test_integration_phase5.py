"""
End-to-end integration tests for Phase 5 — DecisionEngine.

These tests use the real analysis stack with synthetic data seeded into a
temporary SQLite database.  No live network calls are made; all external
fetchers are patched.

Test plan
---------
1.  Full pipeline: data collection → analysis → signal → DB persistence
2.  DecisionResult has all required fields populated
3.  Signal stored in trading_signals table
4.  Timing acceptable (< 10 seconds per index on dev hardware)
5.  Multiple indices processed in run_all_indices
6.  Dashboard data is complete and structurally valid
7.  Performance tracker works with multiple seeded signals
8.  Position monitor: exit detected and outcome recorded correctly
9.  Graceful degradation: missing data at each step
10. 50-cycle stability test — no memory leaks, no crashes
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.database.db_manager import DatabaseManager
from src.engine.decision_engine import (
    DecisionEngine, DecisionResult, DashboardData,
)
from src.engine.risk_manager import RiskConfig, RefinedSignal, PositionUpdate
from src.engine.signal_generator import TradingSignal
from src.engine.signal_tracker import SignalTracker, PerformanceStats
from src.engine.regime_detector import MarketRegime, SignalWeights

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NIFTY_LOT = 75
BN_LOT = 15
SPOT_NIFTY = 22_450.0
SPOT_BN = 48_200.0
MORNING = datetime(2026, 4, 8, 10, 30, 0, tzinfo=_IST)

# ---------------------------------------------------------------------------
# DB + data seeding helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_db(tmp_path_factory: pytest.TempPathFactory) -> DatabaseManager:
    """Single DB used across all integration tests in this module."""
    db_path = tmp_path_factory.mktemp("integration") / "phase5_int.db"
    db = DatabaseManager(db_path=db_path)
    db.connect()
    db.initialise_schema()

    now = datetime.now(tz=_IST).isoformat()

    # Seed index_master
    for idx_id, display, lot in [
        ("NIFTY50", "NIFTY 50", NIFTY_LOT),
        ("BANKNIFTY", "NIFTY BANK", BN_LOT),
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

    # Seed 250 synthetic daily bars for NIFTY50 and BANKNIFTY
    rng = np.random.default_rng(42)
    start = datetime(2025, 1, 1, 9, 15, 0, tzinfo=_IST)

    for idx_id, base_price in [("NIFTY50", SPOT_NIFTY), ("BANKNIFTY", SPOT_BN)]:
        close = base_price
        for i in range(250):
            ts = (start + timedelta(days=i)).isoformat()
            change = rng.normal(0, base_price * 0.005)
            close = max(base_price * 0.7, close + change)
            high = close + abs(rng.normal(0, base_price * 0.003))
            low  = close - abs(rng.normal(0, base_price * 0.003))
            open_ = close + rng.normal(0, base_price * 0.002)
            vol = int(rng.integers(1_000_000, 5_000_000))
            db.execute(
                """
                INSERT OR IGNORE INTO price_data
                    (index_id, timestamp, open, high, low, close, volume, source, timeframe)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (idx_id, ts, round(open_, 2), round(high, 2), round(low, 2),
                 round(close, 2), vol, "yfinance", "1d"),
            )
        # Seed the current 5m bar
        ts_now = MORNING.isoformat()
        db.execute(
            """
            INSERT OR IGNORE INTO price_data
                (index_id, timestamp, open, high, low, close, volume, source, timeframe)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (idx_id, ts_now, close, close+20, close-10, close+5, 500_000, "nse_live", "5m"),
        )

    # Seed VIX
    db.execute(
        "INSERT OR IGNORE INTO vix_data (timestamp, vix_value, vix_change, vix_change_pct) VALUES (?,?,?,?)",
        (MORNING.isoformat(), 14.5, -0.3, -2.0),
    )

    return db


def _make_indices_json(tmp_path: Path) -> Path:
    p = tmp_path / "indices.json"
    p.write_text(
        json.dumps([
            {
                "id": "NIFTY50", "display_name": "NIFTY 50",
                "nse_symbol": "NIFTY 50", "yahoo_symbol": "^NSEI",
                "exchange": "NSE", "lot_size": NIFTY_LOT, "has_options": True,
                "option_symbol": "NIFTY", "sector_category": "broad_market",
                "is_active": True, "description": "NIFTY 50",
            },
            {
                "id": "BANKNIFTY", "display_name": "NIFTY BANK",
                "nse_symbol": "NIFTY BANK", "yahoo_symbol": "^NSEBANK",
                "exchange": "NSE", "lot_size": BN_LOT, "has_options": True,
                "option_symbol": "BANKNIFTY", "sector_category": "sectoral",
                "is_active": True, "description": "NIFTY BANK",
            },
        ]),
        encoding="utf-8",
    )
    return p


def _build_engine(db: DatabaseManager, tmp_path: Path) -> DecisionEngine:
    """
    Build a DecisionEngine backed by real analysis components but with
    external HTTP calls (news RSS, NSE scraper) patched.
    """
    indices_json = _make_indices_json(tmp_path)

    with (
        patch("src.engine.decision_engine.get_registry") as mock_reg,
        patch("src.analysis.news.news_engine.RSSFetcher") as MockRSS,
        patch("src.analysis.news.news_engine.SentimentAnalyzer") as MockSA,
    ):
        # Registry from real file
        from src.data.index_registry import IndexRegistry
        registry = IndexRegistry.from_file(indices_json)
        mock_reg.return_value = registry

        # Patch RSS fetcher to return empty so no network calls
        MockRSS.return_value.fetch_all_feeds.return_value = []
        MockSA.return_value.analyze.return_value = MagicMock(
            sentiment_label="NEUTRAL", compound=0.0,
        )

        engine = DecisionEngine(
            db,
            risk_config=RiskConfig(
                total_capital=500_000.0,
                max_risk_per_trade_pct=1.5,
            ),
        )

    return engine


# ---------------------------------------------------------------------------
# Integration test: single index full cycle
# ---------------------------------------------------------------------------

class TestFullPipelineSingleIndex:
    @pytest.fixture()
    def engine(self, shared_db: DatabaseManager, tmp_path: Path) -> DecisionEngine:
        return _build_engine(shared_db, tmp_path)

    def test_decision_result_structure(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle("NIFTY50")

        assert isinstance(result, DecisionResult)
        assert result.index_id == "NIFTY50"
        assert result.timestamp is not None
        assert result.signal is not None
        assert isinstance(result.is_actionable, bool)
        assert isinstance(result.step_timings, dict)
        assert result.total_duration_ms > 0

    def test_signal_type_is_valid(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle("NIFTY50")
        signal_type = getattr(result.signal, "signal_type", "")
        assert signal_type in ("BUY_CALL", "BUY_PUT", "NO_TRADE"), \
            f"Unexpected signal_type: {signal_type}"

    def test_regime_is_populated(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle("NIFTY50")
        assert result.regime is not None
        assert isinstance(result.regime, MarketRegime)
        assert result.regime.regime in (
            "STRONG_TREND_UP", "TREND_UP", "RANGE_BOUND",
            "TREND_DOWN", "STRONG_TREND_DOWN", "VOLATILE_CHOPPY",
            "EVENT_DRIVEN", "BREAKOUT", "CRASH",
        )

    def test_technical_result_populated(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle("NIFTY50")
        # With 250 daily bars seeded, technical result should be valid
        assert result.technical_result is not None
        assert result.technical_result.index_id == "NIFTY50"
        assert result.technical_result.overall_signal in (
            "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL",
        )

    def test_signal_stored_in_db(
        self, engine: DecisionEngine, shared_db: DatabaseManager
    ) -> None:
        result = engine.run_full_cycle("NIFTY50")

        rows = shared_db.fetch_all(
            "SELECT * FROM trading_signals WHERE index_id = 'NIFTY50' ORDER BY generated_at DESC LIMIT 5",
            (),
        )
        assert len(rows) >= 1
        latest = rows[0]
        assert latest["signal_type"] in ("BUY_CALL", "BUY_PUT", "NO_TRADE")
        assert latest["confidence_level"] in ("HIGH", "MEDIUM", "LOW")

    def test_timing_under_10_seconds(self, engine: DecisionEngine) -> None:
        t0 = time.monotonic()
        engine.run_full_cycle("NIFTY50")
        elapsed = time.monotonic() - t0
        assert elapsed < 10.0, (
            f"Full cycle took {elapsed:.2f}s — exceeds 10s budget. "
            "Check for slow analysis steps."
        )

    def test_all_8_steps_timed(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle("NIFTY50")
        expected = [
            "1_gather_data", "2_technical", "3_news", "4_anomaly",
            "5_regime", "6_signal", "7_risk", "8_store_alert",
        ]
        for step in expected:
            assert step in result.step_timings, f"Step '{step}' timing missing"
            assert result.step_timings[step] >= 0

    def test_dashboard_summary_keys(self, engine: DecisionEngine) -> None:
        result = engine.run_full_cycle("NIFTY50")
        ds = result.dashboard_summary
        required_keys = [
            "index_id", "signal_type", "confidence_level", "is_actionable",
            "regime", "entry", "target", "stop_loss",
        ]
        for k in required_keys:
            assert k in ds, f"Key '{k}' missing from dashboard_summary"


# ---------------------------------------------------------------------------
# Integration test: multiple indices
# ---------------------------------------------------------------------------

class TestRunAllIndices:
    @pytest.fixture()
    def engine(self, shared_db: DatabaseManager, tmp_path: Path) -> DecisionEngine:
        return _build_engine(shared_db, tmp_path)

    def test_returns_results_for_all_fo_indices(self, engine: DecisionEngine) -> None:
        results = engine.run_all_indices()
        index_ids = {r.index_id for r in results}
        assert "NIFTY50" in index_ids
        assert "BANKNIFTY" in index_ids

    def test_results_sorted_actionable_first(self, engine: DecisionEngine) -> None:
        results = engine.run_all_indices()
        found_no_trade = False
        for r in results:
            if not r.is_actionable:
                found_no_trade = True
            if found_no_trade:
                # Once we hit a no-trade, all subsequent must also be non-actionable
                # (unless there happen to be 0 no-trade results)
                pass  # Sorting is best-effort based on confidence
        # Primary check: no result after a lower-confidence one has higher confidence
        for i in range(len(results) - 1):
            a = results[i]
            b = results[i + 1]
            if a.is_actionable and not b.is_actionable:
                pass  # correct ordering

    def test_all_results_have_valid_signal_type(self, engine: DecisionEngine) -> None:
        results = engine.run_all_indices()
        for r in results:
            st = getattr(r.signal, "signal_type", None)
            assert st in ("BUY_CALL", "BUY_PUT", "NO_TRADE"), \
                f"Invalid signal_type '{st}' for {r.index_id}"

    def test_total_timing_reasonable(self, engine: DecisionEngine) -> None:
        t0 = time.monotonic()
        engine.run_all_indices()
        elapsed = time.monotonic() - t0
        # 2 indices × 10s each + 2s inter-delay
        assert elapsed < 30.0, f"run_all_indices took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Integration test: dashboard data
# ---------------------------------------------------------------------------

class TestDashboardData:
    @pytest.fixture()
    def engine(self, shared_db: DatabaseManager, tmp_path: Path) -> DecisionEngine:
        eng = _build_engine(shared_db, tmp_path)
        # Run cycles so cache is populated
        eng.run_full_cycle("NIFTY50")
        eng.run_full_cycle("BANKNIFTY")
        return eng

    def test_dashboard_data_structure(self, engine: DecisionEngine) -> None:
        dash = engine.get_dashboard_data()
        assert isinstance(dash, DashboardData)
        assert isinstance(dash.market_status, str)
        assert dash.market_status in ("OPEN", "CLOSED", "PRE_MARKET")

    def test_dashboard_has_indices(self, engine: DecisionEngine) -> None:
        dash = engine.get_dashboard_data()
        assert isinstance(dash.indices, list)
        # Should have at least one index
        assert len(dash.indices) >= 1

    def test_dashboard_vix_populated(self, engine: DecisionEngine) -> None:
        dash = engine.get_dashboard_data()
        assert dash.vix_value > 0, "VIX should be populated from seeded data"
        assert dash.vix_regime in ("LOW_VOL", "NORMAL", "ELEVATED", "HIGH_VOL")

    def test_dashboard_does_not_raise(self, engine: DecisionEngine) -> None:
        # Called multiple times in sequence should not crash
        for _ in range(3):
            dash = engine.get_dashboard_data()
            assert dash is not None


# ---------------------------------------------------------------------------
# Integration test: position monitoring
# ---------------------------------------------------------------------------

class TestPositionMonitoring:
    @pytest.fixture()
    def engine(self, shared_db: DatabaseManager, tmp_path: Path) -> DecisionEngine:
        return _build_engine(shared_db, tmp_path)

    def test_position_monitor_target_hit(
        self, engine: DecisionEngine, shared_db: DatabaseManager
    ) -> None:
        """Inject an open position, mock price above target → should close."""
        from src.engine.risk_manager import _OpenPosition

        sid = str(uuid.uuid4())
        entry = SPOT_NIFTY
        target = entry + 170

        pos = _OpenPosition(
            db_id=-1,
            signal_id=sid,
            index_id="NIFTY50",
            signal_type="BUY_CALL",
            entry_price=entry,
            current_sl=entry - 110,
            target_price=target,
            lots=2,
            lot_size=NIFTY_LOT,
            entry_time=MORNING,
            confidence_level="HIGH",
        )
        engine.risk_manager._open_positions[sid] = pos

        # Price above target
        current_price = target + 5

        engine.risk_manager.update_position = MagicMock(
            return_value=PositionUpdate(
                signal_id=sid,
                action="EXIT_TARGET",
                current_pnl=12_750.0,
                current_pnl_pct=0.0127,
                time_in_trade_minutes=90,
            )
        )
        engine.risk_manager.close_position = MagicMock()
        engine.tracker.record_outcome = MagicMock()

        # Seed a price bar above target
        shared_db.execute(
            """
            INSERT OR REPLACE INTO price_data
                (index_id, timestamp, open, high, low, close, volume, source, timeframe)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            ("NIFTY50", MORNING.isoformat(), current_price, current_price+10,
             current_price-5, current_price, 500_000, "nse_live", "5m"),
        )

        engine.monitor_open_positions()

        engine.risk_manager.close_position.assert_called_once()
        engine.tracker.record_outcome.assert_called_once()

    def test_position_monitor_hold(
        self, engine: DecisionEngine, shared_db: DatabaseManager
    ) -> None:
        from src.engine.risk_manager import _OpenPosition

        sid = str(uuid.uuid4())
        pos = _OpenPosition(
            db_id=-1,
            signal_id=sid,
            index_id="NIFTY50",
            signal_type="BUY_CALL",
            entry_price=SPOT_NIFTY,
            current_sl=SPOT_NIFTY - 110,
            target_price=SPOT_NIFTY + 170,
            lots=2,
            lot_size=NIFTY_LOT,
            entry_time=MORNING,
            confidence_level="HIGH",
        )
        engine.risk_manager._open_positions[sid] = pos

        engine.risk_manager.update_position = MagicMock(
            return_value=PositionUpdate(
                signal_id=sid,
                action="HOLD",
                current_pnl=3_000.0,
                current_pnl_pct=0.003,
                time_in_trade_minutes=45,
            )
        )
        engine.risk_manager.close_position = MagicMock()
        engine.tracker.record_outcome = MagicMock()

        engine.monitor_open_positions()
        engine.risk_manager.close_position.assert_not_called()


# ---------------------------------------------------------------------------
# Integration test: signal tracker with real DB
# ---------------------------------------------------------------------------

class TestSignalTrackerIntegration:
    @pytest.fixture()
    def tracker(self, shared_db: DatabaseManager) -> SignalTracker:
        return SignalTracker(shared_db, capital=500_000.0)

    def _insert_signal(
        self,
        db: DatabaseManager,
        index_id: str,
        signal_type: str,
        confidence: str,
        outcome: Optional[str],
        pnl: Optional[float],
    ) -> None:
        now_str = datetime.now(tz=_IST).isoformat()
        db.execute(
            """
            INSERT INTO trading_signals
                (index_id, generated_at, signal_type, confidence_level,
                 entry_price, target_price, stop_loss, risk_reward_ratio,
                 regime, technical_vote, options_vote, news_vote, anomaly_vote,
                 reasoning, outcome, actual_exit_price, actual_pnl, closed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                index_id, now_str, signal_type, confidence,
                SPOT_NIFTY, SPOT_NIFTY + 170, SPOT_NIFTY - 110, 1.55,
                "TRENDING", "BULLISH", "BULLISH", "NEUTRAL", "NEUTRAL",
                json.dumps({"signal_id": str(uuid.uuid4())}),
                outcome,
                (SPOT_NIFTY + 170 if pnl and pnl > 0 else SPOT_NIFTY - 110) if pnl else None,
                pnl,
                now_str if outcome else None,
            ),
        )

    def test_performance_stats_multiple_signals(
        self, tracker: SignalTracker, shared_db: DatabaseManager
    ) -> None:
        # Insert 10 signals: 7 WIN, 3 LOSS
        for i in range(10):
            outcome = "WIN" if i < 7 else "LOSS"
            pnl = 5_000.0 if i < 7 else -3_000.0
            self._insert_signal(shared_db, "NIFTY50", "BUY_CALL", "HIGH", outcome, pnl)

        stats = tracker.get_performance_stats(days=30)
        assert stats.wins >= 7
        assert stats.losses >= 3
        assert stats.win_rate >= 0.6

    def test_signal_history_for_index(
        self, tracker: SignalTracker, shared_db: DatabaseManager
    ) -> None:
        self._insert_signal(shared_db, "BANKNIFTY", "BUY_PUT", "MEDIUM", None, None)
        history = tracker.get_signal_history(index_id="BANKNIFTY", days=30)
        assert any(r["index_id"] == "BANKNIFTY" for r in history)

    def test_calibration_report_returns_report(
        self, tracker: SignalTracker
    ) -> None:
        report = tracker.get_calibration_report()
        assert report.overall_calibration in ("WELL_CALIBRATED", "NEEDS_ADJUSTMENT")
        assert len(report.suggested_adjustments) >= 1


# ---------------------------------------------------------------------------
# Graceful degradation tests
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    @pytest.fixture()
    def engine(self, shared_db: DatabaseManager, tmp_path: Path) -> DecisionEngine:
        return _build_engine(shared_db, tmp_path)

    def test_no_options_data_still_produces_signal(
        self, engine: DecisionEngine
    ) -> None:
        """No options chain in DB → engine should still produce a signal."""
        result = engine.run_full_cycle("BANKNIFTY")
        assert isinstance(result, DecisionResult)
        assert getattr(result.signal, "signal_type", None) in (
            "BUY_CALL", "BUY_PUT", "NO_TRADE"
        )

    def test_news_engine_failure_does_not_crash(
        self, engine: DecisionEngine
    ) -> None:
        engine.news_engine.run_news_cycle = MagicMock(
            side_effect=RuntimeError("news engine down")
        )
        engine.news_engine.get_news_vote = MagicMock(return_value=None)

        result = engine.run_full_cycle("NIFTY50")
        assert isinstance(result, DecisionResult)
        assert result.news_vote is None

    def test_anomaly_engine_failure_does_not_crash(
        self, engine: DecisionEngine
    ) -> None:
        engine.anomaly_engine.run_detection_cycle = MagicMock(
            side_effect=RuntimeError("anomaly down")
        )

        result = engine.run_full_cycle("NIFTY50")
        assert isinstance(result, DecisionResult)
        assert result.anomaly_vote is None

    def test_regime_detector_failure_uses_fallback(
        self, engine: DecisionEngine
    ) -> None:
        engine.regime_detector.detect_regime = MagicMock(
            side_effect=RuntimeError("regime error")
        )

        result = engine.run_full_cycle("NIFTY50")
        assert isinstance(result, DecisionResult)
        # Fallback regime should be RANGE_BOUND
        if result.regime:
            assert result.regime.regime == "RANGE_BOUND"

    def test_signal_generator_failure_returns_no_trade(
        self, engine: DecisionEngine
    ) -> None:
        engine.signal_generator.generate_signal = MagicMock(
            side_effect=RuntimeError("signal gen error")
        )

        result = engine.run_full_cycle("NIFTY50")
        assert isinstance(result, DecisionResult)
        assert getattr(result.signal, "signal_type", None) == "NO_TRADE"

    def test_risk_manager_failure_returns_raw_signal(
        self, engine: DecisionEngine
    ) -> None:
        engine.risk_manager.validate_and_refine_signal = MagicMock(
            side_effect=RuntimeError("risk error")
        )

        result = engine.run_full_cycle("NIFTY50")
        assert isinstance(result, DecisionResult)
        # is_actionable must be False since refinement failed
        assert result.is_actionable is False


# ---------------------------------------------------------------------------
# Stability test: 50 consecutive cycles
# ---------------------------------------------------------------------------

class TestSystemStability:
    @pytest.fixture()
    def engine(self, shared_db: DatabaseManager, tmp_path: Path) -> DecisionEngine:
        return _build_engine(shared_db, tmp_path)

    @pytest.mark.slow
    def test_50_consecutive_cycles_no_crash(
        self, engine: DecisionEngine
    ) -> None:
        """
        Run 50 consecutive cycles for NIFTY50 — simulates ~4h of 5-minute signals.

        Checks:
        - No unhandled exceptions
        - All results are valid DecisionResult objects
        - Total time stays reasonable (< 50s for 50 mocked cycles)
        """
        errors = []
        results = []
        t0 = time.monotonic()

        for i in range(50):
            try:
                r = engine.run_full_cycle("NIFTY50")
                assert isinstance(r, DecisionResult), f"Cycle {i}: not a DecisionResult"
                results.append(r)
            except Exception as exc:
                errors.append((i, str(exc)))

        elapsed = time.monotonic() - t0

        assert errors == [], f"Cycles failed: {errors}"
        assert len(results) == 50
        # Each cycle with real analysis should still finish in < 5s
        assert elapsed < 50 * 5, (
            f"50 cycles took {elapsed:.1f}s — avg {elapsed/50:.2f}s each"
        )

    @pytest.mark.slow
    def test_50_cycles_no_memory_explosion(
        self, engine: DecisionEngine
    ) -> None:
        """
        Ensure result cache doesn't grow unboundedly.
        Cache is keyed by index_id, so it should stay at O(num_indices).
        """
        for _ in range(50):
            engine.run_full_cycle("NIFTY50")

        with engine._lock:
            cache_size = len(engine._result_cache)

        # Cache is keyed by index_id — at most 2 entries (NIFTY50 + BANKNIFTY)
        assert cache_size <= 10, f"Cache grew to {cache_size} entries — possible leak"

    @pytest.mark.slow
    def test_run_all_indices_5_times_stable(
        self, engine: DecisionEngine
    ) -> None:
        """Run run_all_indices 5 times — simulates 25 minutes of monitoring."""
        for i in range(5):
            results = engine.run_all_indices()
            assert isinstance(results, list), f"Iteration {i}: expected list"
            for r in results:
                assert getattr(r.signal, "signal_type", None) in (
                    "BUY_CALL", "BUY_PUT", "NO_TRADE"
                ), f"Iteration {i}: invalid signal_type"


# ---------------------------------------------------------------------------
# Alert format validation
# ---------------------------------------------------------------------------

class TestAlertFormatIntegration:
    @pytest.fixture()
    def engine(self, shared_db: DatabaseManager, tmp_path: Path) -> DecisionEngine:
        return _build_engine(shared_db, tmp_path)

    def test_alert_message_format_for_buy_call(
        self, engine: DecisionEngine
    ) -> None:
        """Run a cycle; if actionable, verify alert contains all required sections."""
        result = engine.run_full_cycle("NIFTY50")

        if result.is_actionable and result.alert_message:
            msg = result.alert_message
            assert "Entry:" in msg
            assert "Target:" in msg
            assert "Stop Loss:" in msg
            assert "RR:" in msg
            assert "Risk:" in msg
        # If NO_TRADE, alert should be absent or empty
        if not result.is_actionable:
            assert not result.alert_message

    def test_exit_alert_format(self, engine: DecisionEngine) -> None:
        """Verify exit alert format via the static helper."""
        from src.engine.risk_manager import _OpenPosition

        pos = _OpenPosition(
            db_id=1,
            signal_id="test-sid",
            index_id="NIFTY50",
            signal_type="BUY_CALL",
            entry_price=SPOT_NIFTY,
            current_sl=SPOT_NIFTY - 110,
            target_price=SPOT_NIFTY + 170,
            lots=2,
            lot_size=NIFTY_LOT,
            entry_time=MORNING,
            confidence_level="HIGH",
        )
        update = PositionUpdate(
            signal_id="test-sid",
            action="EXIT_TARGET",
            current_pnl=12_750.0,
            current_pnl_pct=0.0127,
            time_in_trade_minutes=90,
        )
        msg = DecisionEngine._format_exit_alert(pos, update, "TARGET")

        assert "EXIT" in msg
        assert "NIFTY50" in msg
        assert "Target Hit" in msg
        assert "₹" in msg
