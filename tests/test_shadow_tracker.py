"""
Tests for ShadowTracker (live-vs-backtest comparison).

Covers:
- Comparison logic with mocked live signals and backtest results
- Divergence scoring at each boundary (LOW / MODERATE / HIGH / CRITICAL)
- Recommendation text for each severity and edge status
- Edge-status detection (STABLE / DECAYING / NO_EDGE)
- Zero trades on one or both sides
- Report text formatting
- Report persistence to system_health
- ShadowReport.to_dict serialisation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from src.engine.shadow_tracker import ShadowTracker, ShadowReport


# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for BacktestResult / BacktestMetrics
# ---------------------------------------------------------------------------


@dataclass
class _FakeMetrics:
    total_return_amount: float = 0.0
    win_rate: float = 0.0


@dataclass
class _FakeBacktestResult:
    executed_trades: int = 0
    trade_history: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """Ephemeral test DB with schema."""
    db_path = tmp_path / "test_shadow.db"
    mgr = DatabaseManager(db_path=db_path)
    mgr.connect()
    mgr.initialise_schema()
    return mgr


@pytest.fixture()
def tracker(db: DatabaseManager) -> ShadowTracker:
    return ShadowTracker(db)


def _seed_index(db: DatabaseManager, index_id: str = "NIFTY50") -> None:
    now = datetime.now().isoformat()
    db.execute(
        Q.INSERT_INDEX_MASTER,
        (
            index_id, f"Test {index_id}", index_id, f"^{index_id}",
            "NSE", 75, 1, index_id, "broad_market", 1, now, now,
        ),
    )


def _insert_signal(
    db: DatabaseManager,
    index_id: str = "NIFTY50",
    generated_at: str = "2025-03-14T10:00:00",
    signal_type: str = "BUY_CALL",
    confidence: str = "HIGH",
    outcome: Optional[str] = "WIN",
    actual_pnl: Optional[float] = 500.0,
) -> None:
    """Insert a trading signal row."""
    db.execute(
        Q.INSERT_TRADING_SIGNAL,
        (
            index_id,
            generated_at,
            signal_type,
            confidence,
            20000.0,   # entry_price
            20200.0,   # target_price
            19900.0,   # stop_loss
            2.0,       # risk_reward_ratio
            "TRENDING",  # regime
            "BULLISH",   # technical_vote
            "NEUTRAL",   # options_vote
            "NEUTRAL",   # news_vote
            "NEUTRAL",   # anomaly_vote
            "{}",        # reasoning
            outcome,
            20200.0 if outcome == "WIN" else 19900.0 if outcome == "LOSS" else None,
            actual_pnl,
            generated_at if outcome in ("WIN", "LOSS") else None,
        ),
    )


# ---------------------------------------------------------------------------
# _compare: divergence scoring tests
# ---------------------------------------------------------------------------


class TestDivergenceScoring:
    """Test _compare logic with varying live / backtest scenarios."""

    def test_low_divergence_both_similar(self, tracker: ShadowTracker):
        """When live and backtest are similar → LOW."""
        live_signals = [
            {"actual_pnl": 500},
            {"actual_pnl": -200},
            {"actual_pnl": 300},
        ]
        bt_result = _FakeBacktestResult(executed_trades=3)
        bt_metrics = _FakeMetrics(total_return_amount=600, win_rate=66.7)

        report = tracker._compare(
            "NIFTY50", date(2025, 3, 10), date(2025, 3, 16),
            live_signals, bt_result, bt_metrics,
        )

        assert report.divergence_severity == "LOW"
        assert report.divergence_score == 0.0
        assert report.edge_status == "STABLE"
        assert len(report.issues) == 0

    def test_trade_count_divergence(self, tracker: ShadowTracker):
        """Large trade count difference → adds 0.3 to score."""
        live_signals = [{"actual_pnl": 100}]  # 1 trade
        bt_result = _FakeBacktestResult(executed_trades=5)  # 5 trades
        bt_metrics = _FakeMetrics(total_return_amount=100, win_rate=100.0)

        report = tracker._compare(
            "NIFTY50", date(2025, 3, 10), date(2025, 3, 16),
            live_signals, bt_result, bt_metrics,
        )

        assert report.trade_count_diff == 4
        assert report.divergence_score >= 0.3
        assert "Trade count divergence" in report.issues[0]

    def test_win_rate_divergence(self, tracker: ShadowTracker):
        """Win rate gap > 20 percentage points → adds 0.3."""
        live_signals = [
            {"actual_pnl": 500},
            {"actual_pnl": -200},
            {"actual_pnl": -100},
            {"actual_pnl": -50},
        ]  # 1 win / 4 = 25%
        bt_result = _FakeBacktestResult(executed_trades=4)
        bt_metrics = _FakeMetrics(total_return_amount=150, win_rate=75.0)

        report = tracker._compare(
            "NIFTY50", date(2025, 3, 10), date(2025, 3, 16),
            live_signals, bt_result, bt_metrics,
        )

        assert report.wr_diff == 50.0
        assert any("Win rate divergence" in i for i in report.issues)

    def test_pnl_divergence(self, tracker: ShadowTracker):
        """P&L gap > 50% of the larger → adds 0.2."""
        live_signals = [{"actual_pnl": 100}]
        bt_result = _FakeBacktestResult(executed_trades=1)
        bt_metrics = _FakeMetrics(total_return_amount=5000, win_rate=100.0)

        report = tracker._compare(
            "NIFTY50", date(2025, 3, 10), date(2025, 3, 16),
            live_signals, bt_result, bt_metrics,
        )

        assert report.pnl_diff == 4900.0
        assert any("P&L divergence" in i for i in report.issues)

    def test_critical_all_divergences(self, tracker: ShadowTracker):
        """Trade count + win rate + P&L + edge decay → CRITICAL."""
        live_signals = [
            {"actual_pnl": -500},
        ]  # 1 trade, losing, 0% WR
        bt_result = _FakeBacktestResult(executed_trades=5)
        bt_metrics = _FakeMetrics(total_return_amount=3000, win_rate=80.0)

        report = tracker._compare(
            "NIFTY50", date(2025, 3, 10), date(2025, 3, 16),
            live_signals, bt_result, bt_metrics,
        )

        assert report.divergence_severity == "CRITICAL"
        assert report.divergence_score > 0.6
        assert report.edge_status == "DECAYING"


# ---------------------------------------------------------------------------
# Severity thresholds
# ---------------------------------------------------------------------------


class TestSeverityThresholds:

    def test_score_zero_is_low(self, tracker: ShadowTracker):
        live = [{"actual_pnl": 100}]
        bt = _FakeBacktestResult(executed_trades=1)
        metrics = _FakeMetrics(total_return_amount=100, win_rate=100.0)
        r = tracker._compare("X", date(2025, 1, 1), date(2025, 1, 7),
                             live, bt, metrics)
        assert r.divergence_severity == "LOW"

    def test_moderate_threshold(self, tracker: ShadowTracker):
        """Score in (0.1, 0.3] → MODERATE.

        P&L divergence alone = 0.2 (both profitable, no edge decay,
        same WR, same trade count, but large P&L gap).
        """
        live = [{"actual_pnl": 100}, {"actual_pnl": 100}]  # pnl=200, WR=100%
        bt = _FakeBacktestResult(executed_trades=2)
        metrics = _FakeMetrics(total_return_amount=5000, win_rate=100.0)
        r = tracker._compare("X", date(2025, 1, 1), date(2025, 1, 7),
                             live, bt, metrics)
        assert r.divergence_severity == "MODERATE"
        assert 0.1 < r.divergence_score <= 0.3

    def test_high_threshold(self, tracker: ShadowTracker):
        """Score in (0.3, 0.6] → HIGH.

        Trade count divergence (0.3) + P&L divergence (0.2) = 0.5.
        Both sides profitable → STABLE edge, no decay penalty.
        """
        live = [{"actual_pnl": 100}]  # 1 trade, pnl=100, WR=100%
        bt = _FakeBacktestResult(executed_trades=5)  # 5 trades
        metrics = _FakeMetrics(total_return_amount=5000, win_rate=100.0)
        r = tracker._compare("X", date(2025, 1, 1), date(2025, 1, 7),
                             live, bt, metrics)
        assert r.divergence_severity == "HIGH"
        assert 0.3 < r.divergence_score <= 0.6


# ---------------------------------------------------------------------------
# Edge status detection
# ---------------------------------------------------------------------------


class TestEdgeStatus:

    def test_stable_both_positive(self, tracker: ShadowTracker):
        live = [{"actual_pnl": 200}]
        bt = _FakeBacktestResult(executed_trades=1)
        metrics = _FakeMetrics(total_return_amount=200, win_rate=100.0)
        r = tracker._compare("X", date(2025, 1, 1), date(2025, 1, 7),
                             live, bt, metrics)
        assert r.edge_status == "STABLE"

    def test_decaying_bt_positive_live_negative(self, tracker: ShadowTracker):
        live = [{"actual_pnl": -300}]
        bt = _FakeBacktestResult(executed_trades=1)
        metrics = _FakeMetrics(total_return_amount=500, win_rate=100.0)
        r = tracker._compare("X", date(2025, 1, 1), date(2025, 1, 7),
                             live, bt, metrics)
        assert r.edge_status == "DECAYING"

    def test_no_edge_both_negative(self, tracker: ShadowTracker):
        live = [{"actual_pnl": -100}]
        bt = _FakeBacktestResult(executed_trades=1)
        metrics = _FakeMetrics(total_return_amount=-200, win_rate=0.0)
        r = tracker._compare("X", date(2025, 1, 1), date(2025, 1, 7),
                             live, bt, metrics)
        assert r.edge_status == "NO_EDGE"


# ---------------------------------------------------------------------------
# Zero-trade edge cases
# ---------------------------------------------------------------------------


class TestZeroTrades:

    def test_zero_live_zero_backtest(self, tracker: ShadowTracker):
        r = tracker._compare(
            "X", date(2025, 1, 1), date(2025, 1, 7),
            [], _FakeBacktestResult(executed_trades=0),
            _FakeMetrics(),
        )
        assert r.live_trades == 0
        assert r.backtest_trades == 0
        assert r.divergence_severity == "LOW"
        assert r.live_win_rate == 0.0

    def test_zero_live_some_backtest(self, tracker: ShadowTracker):
        """Backtest found trades but live didn't → trade count divergence."""
        r = tracker._compare(
            "X", date(2025, 1, 1), date(2025, 1, 7),
            [], _FakeBacktestResult(executed_trades=4),
            _FakeMetrics(total_return_amount=1000, win_rate=75.0),
        )
        assert r.live_trades == 0
        assert r.backtest_trades == 4
        assert r.trade_count_diff == 4
        assert any("Trade count" in i for i in r.issues)

    def test_some_live_zero_backtest(self, tracker: ShadowTracker):
        """Live traded but backtest found nothing."""
        live = [{"actual_pnl": 200}, {"actual_pnl": -100}]
        r = tracker._compare(
            "X", date(2025, 1, 1), date(2025, 1, 7),
            live, _FakeBacktestResult(executed_trades=0),
            _FakeMetrics(),
        )
        assert r.live_trades == 2
        assert r.backtest_trades == 0
        assert r.trade_count_diff == 2
        assert any("Trade count" in i for i in r.issues)

    def test_none_actual_pnl_treated_as_zero(self, tracker: ShadowTracker):
        """Signals with None pnl (still open) → treated as 0."""
        live = [{"actual_pnl": None}, {"actual_pnl": None}]
        r = tracker._compare(
            "X", date(2025, 1, 1), date(2025, 1, 7),
            live, _FakeBacktestResult(executed_trades=2),
            _FakeMetrics(total_return_amount=0, win_rate=0.0),
        )
        assert r.live_pnl == 0.0
        assert r.live_win_rate == 0.0


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:

    def test_critical_recommendation(self):
        text = ShadowTracker._get_recommendation("CRITICAL", "DECAYING", [])
        assert "CRITICAL DIVERGENCE" in text
        assert "pausing live trading" in text

    def test_high_recommendation(self):
        text = ShadowTracker._get_recommendation("HIGH", "STABLE", [])
        assert "HIGH DIVERGENCE" in text
        assert "Monitor closely" in text

    def test_decaying_edge_recommendation(self):
        text = ShadowTracker._get_recommendation("MODERATE", "DECAYING", [])
        assert "EDGE DECAY" in text
        assert "overfitting" in text

    def test_no_edge_recommendation(self):
        text = ShadowTracker._get_recommendation("LOW", "NO_EDGE", [])
        assert "NO EDGE" in text
        assert "pausing" in text

    def test_stable_low_recommendation(self):
        text = ShadowTracker._get_recommendation("LOW", "STABLE", [])
        assert "within expected range" in text


# ---------------------------------------------------------------------------
# Report text formatting
# ---------------------------------------------------------------------------


class TestReportFormatting:

    def test_format_contains_key_fields(self, tracker: ShadowTracker):
        report = ShadowReport(
            index_id="NIFTY50",
            period_start=date(2025, 3, 10),
            period_end=date(2025, 3, 16),
            live_trades=5,
            live_pnl=1200.0,
            live_win_rate=60.0,
            backtest_trades=6,
            backtest_pnl=1500.0,
            backtest_win_rate=67.0,
            trade_count_diff=1,
            pnl_diff=300.0,
            wr_diff=7.0,
            divergence_score=0.0,
            divergence_severity="LOW",
            edge_status="STABLE",
            issues=[],
            recommendation="System performing within expected range. Continue monitoring.",
        )
        text = tracker.get_weekly_report_text(report)

        assert "NIFTY50" in text
        assert "2025-03-10" in text
        assert "2025-03-16" in text
        assert "LIVE" in text
        assert "BACKTEST" in text
        assert "STABLE" in text
        assert "LOW" in text
        assert "\u2705" in text  # ✅ for LOW severity

    def test_critical_emoji(self, tracker: ShadowTracker):
        report = ShadowReport(
            index_id="X", period_start=date(2025, 1, 1),
            period_end=date(2025, 1, 7),
            live_trades=0, live_pnl=0, live_win_rate=0,
            backtest_trades=0, backtest_pnl=0, backtest_win_rate=0,
            trade_count_diff=0, pnl_diff=0, wr_diff=0,
            divergence_score=0.8, divergence_severity="CRITICAL",
            edge_status="DECAYING", issues=[], recommendation="",
        )
        text = tracker.get_weekly_report_text(report)
        assert "\U0001f6a8" in text  # 🚨


# ---------------------------------------------------------------------------
# to_dict serialisation
# ---------------------------------------------------------------------------


class TestShadowReportSerialisation:

    def test_to_dict_round_trip(self):
        report = ShadowReport(
            index_id="BANKNIFTY",
            period_start=date(2025, 4, 7),
            period_end=date(2025, 4, 13),
            live_trades=3,
            live_pnl=-500.0,
            live_win_rate=33.3,
            backtest_trades=5,
            backtest_pnl=1200.0,
            backtest_win_rate=80.0,
            trade_count_diff=2,
            pnl_diff=1700.0,
            wr_diff=46.7,
            divergence_score=0.7,
            divergence_severity="CRITICAL",
            edge_status="DECAYING",
            issues=["Trade count divergence", "Win rate divergence"],
            recommendation="CRITICAL DIVERGENCE: ...",
        )
        d = report.to_dict()

        assert d["index_id"] == "BANKNIFTY"
        assert d["period_start"] == "2025-04-07"
        assert d["period_end"] == "2025-04-13"
        assert d["live_trades"] == 3
        assert d["divergence_score"] == 0.7
        assert len(d["issues"]) == 2


# ---------------------------------------------------------------------------
# _get_live_signals — DB integration
# ---------------------------------------------------------------------------


class TestGetLiveSignals:

    def test_fetches_correct_signals(self, db: DatabaseManager, tracker: ShadowTracker):
        _seed_index(db)

        # Insert signals: 2 in range, 1 out of range, 1 NO_TRADE
        _insert_signal(db, generated_at="2025-03-12T10:00:00",
                       signal_type="BUY_CALL", outcome="WIN", actual_pnl=500)
        _insert_signal(db, generated_at="2025-03-14T10:00:00",
                       signal_type="BUY_PUT", outcome="LOSS", actual_pnl=-200)
        _insert_signal(db, generated_at="2025-03-20T10:00:00",
                       signal_type="BUY_CALL", outcome="WIN", actual_pnl=300)
        _insert_signal(db, generated_at="2025-03-13T10:00:00",
                       signal_type="NO_TRADE", outcome=None, actual_pnl=None)

        signals = tracker._get_live_signals(
            "NIFTY50", date(2025, 3, 10), date(2025, 3, 16),
        )

        assert len(signals) == 2
        types = [s["signal_type"] for s in signals]
        assert "NO_TRADE" not in types

    def test_empty_when_no_signals(self, db: DatabaseManager, tracker: ShadowTracker):
        _seed_index(db)
        signals = tracker._get_live_signals(
            "NIFTY50", date(2025, 3, 10), date(2025, 3, 16),
        )
        assert signals == []


# ---------------------------------------------------------------------------
# _store_report — persistence
# ---------------------------------------------------------------------------


class TestStoreReport:

    def test_persists_to_system_health(self, db: DatabaseManager, tracker: ShadowTracker):
        report = ShadowReport(
            index_id="NIFTY50",
            period_start=date(2025, 3, 10),
            period_end=date(2025, 3, 16),
            live_trades=2, live_pnl=100, live_win_rate=50,
            backtest_trades=3, backtest_pnl=200, backtest_win_rate=67,
            trade_count_diff=1, pnl_diff=100, wr_diff=17,
            divergence_score=0.05, divergence_severity="LOW",
            edge_status="STABLE", issues=[], recommendation="OK",
        )
        tracker._store_report(report)

        row = db.fetch_one(
            "SELECT * FROM system_health WHERE component LIKE 'shadow_tracker:%'"
        )
        assert row is not None
        assert row["component"] == "shadow_tracker:NIFTY50"
        assert row["status"] == "OK"

    def test_high_severity_stores_error_status(self, db: DatabaseManager, tracker: ShadowTracker):
        report = ShadowReport(
            index_id="X", period_start=date(2025, 1, 1),
            period_end=date(2025, 1, 7),
            live_trades=0, live_pnl=0, live_win_rate=0,
            backtest_trades=0, backtest_pnl=0, backtest_win_rate=0,
            trade_count_diff=0, pnl_diff=0, wr_diff=0,
            divergence_score=0.5, divergence_severity="HIGH",
            edge_status="STABLE", issues=[], recommendation="",
        )
        tracker._store_report(report)

        row = db.fetch_one(
            "SELECT * FROM system_health WHERE component LIKE 'shadow_tracker:%'"
        )
        assert row["status"] == "ERROR"

    def test_moderate_severity_stores_warning(self, db: DatabaseManager, tracker: ShadowTracker):
        report = ShadowReport(
            index_id="X", period_start=date(2025, 1, 1),
            period_end=date(2025, 1, 7),
            live_trades=0, live_pnl=0, live_win_rate=0,
            backtest_trades=0, backtest_pnl=0, backtest_win_rate=0,
            trade_count_diff=0, pnl_diff=0, wr_diff=0,
            divergence_score=0.2, divergence_severity="MODERATE",
            edge_status="STABLE", issues=[], recommendation="",
        )
        tracker._store_report(report)

        row = db.fetch_one(
            "SELECT * FROM system_health WHERE component LIKE 'shadow_tracker:%'"
        )
        assert row["status"] == "WARNING"


# ---------------------------------------------------------------------------
# run_weekly_comparison — end-to-end with mocked backtest
# ---------------------------------------------------------------------------


class TestRunWeeklyComparison:

    @patch("src.engine.shadow_tracker.ShadowTracker._run_backtest")
    def test_end_to_end_with_mock(
        self, mock_bt, db: DatabaseManager, tracker: ShadowTracker
    ):
        _seed_index(db)

        # Insert live signals for the week
        _insert_signal(db, generated_at="2025-03-12T10:00:00",
                       outcome="WIN", actual_pnl=800)
        _insert_signal(db, generated_at="2025-03-13T11:00:00",
                       outcome="LOSS", actual_pnl=-300)
        _insert_signal(db, generated_at="2025-03-14T09:30:00",
                       outcome="WIN", actual_pnl=400)

        mock_bt.return_value = (
            _FakeBacktestResult(executed_trades=3),
            _FakeMetrics(total_return_amount=900, win_rate=66.7),
        )

        report = tracker.run_weekly_comparison("NIFTY50", date(2025, 3, 16))

        assert report.index_id == "NIFTY50"
        assert report.live_trades == 3
        assert report.live_pnl == 900.0
        assert report.backtest_trades == 3
        assert report.backtest_pnl == 900.0
        assert report.divergence_severity == "LOW"

        # Verify it was persisted
        row = db.fetch_one(
            "SELECT * FROM system_health WHERE component = 'shadow_tracker:NIFTY50'"
        )
        assert row is not None

    @patch("src.engine.shadow_tracker.ShadowTracker._run_backtest")
    def test_critical_divergence_end_to_end(
        self, mock_bt, db: DatabaseManager, tracker: ShadowTracker
    ):
        _seed_index(db)

        # Live: 1 losing trade
        _insert_signal(db, generated_at="2025-03-12T10:00:00",
                       outcome="LOSS", actual_pnl=-1000)

        # Backtest: 6 profitable trades
        mock_bt.return_value = (
            _FakeBacktestResult(executed_trades=6),
            _FakeMetrics(total_return_amount=5000, win_rate=83.0),
        )

        report = tracker.run_weekly_comparison("NIFTY50", date(2025, 3, 16))

        assert report.divergence_severity == "CRITICAL"
        assert report.edge_status == "DECAYING"
        assert "CRITICAL DIVERGENCE" in report.recommendation
