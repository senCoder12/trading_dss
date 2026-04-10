"""
End-to-end integration tests for Phase 6 (backtesting pipeline).

These tests verify the full pipeline from DataReplayEngine → StrategyRunner
→ MetricsCalculator → ReportGenerator → WalkForwardValidator.

Tests that require real database data are skipped automatically when the
database is empty (e.g., in CI environments without seeded historical data).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import date, datetime, timedelta

import pytest

from src.backtest.metrics import BacktestMetrics, MetricsCalculator
from src.backtest.report_generator import ReportGenerator
from src.backtest.strategy_runner import BacktestConfig, BacktestResult, StrategyRunner
from src.backtest.trade_simulator import ClosedTrade, EquityPoint, SimulatorConfig
from src.backtest.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.database.db_manager import DatabaseManager


# ---------------------------------------------------------------------------
# DB fixture helpers
# ---------------------------------------------------------------------------


def _has_price_data(db: DatabaseManager, index_id: str, min_bars: int = 50) -> bool:
    """Return True if the DB has at least min_bars of daily price data for index_id."""
    try:
        rows = db.fetch_all(
            "SELECT COUNT(*) as c FROM price_data WHERE index_id = ? AND timeframe = ?",
            (index_id, "1d"),
        )
        return rows and rows[0]["c"] >= min_bars
    except Exception:
        return False


@pytest.fixture(scope="module")
def db() -> DatabaseManager:
    """Module-scoped real database connection."""
    database = DatabaseManager()
    database.connect()
    yield database


@pytest.fixture(scope="module")
def db_with_nifty(db: DatabaseManager):
    """Skip the test if NIFTY50 daily data is not available."""
    if not _has_price_data(db, "NIFTY50"):
        pytest.skip("No NIFTY50 daily price data in DB — skipping integration test")
    return db


@pytest.fixture(scope="module")
def db_with_nifty_2y(db: DatabaseManager):
    """Skip the test if fewer than 500 bars of NIFTY50 data are available."""
    if not _has_price_data(db, "NIFTY50", min_bars=500):
        pytest.skip("Not enough NIFTY50 data for walk-forward (need ~500 bars)")
    return db


# ---------------------------------------------------------------------------
# Helper: create synthetic ClosedTrade / EquityPoint
# ---------------------------------------------------------------------------


def make_trade(outcome: str, net_pnl: float) -> ClosedTrade:
    ts = datetime(2024, 7, 1, 10, 0)
    return ClosedTrade(
        trade_id=str(uuid.uuid4()),
        index_id="NIFTY50",
        signal_id=str(uuid.uuid4()),
        trade_type="BUY_CALL",
        signal_entry_price=24_000.0,
        actual_entry_price=24_002.0,
        entry_timestamp=ts,
        entry_bar={"regime": "TRENDING"},
        original_stop_loss=23_900.0,
        original_target=24_200.0,
        actual_exit_price=24_200.0 if outcome == "WIN" else 23_900.0,
        exit_timestamp=ts + timedelta(hours=1),
        exit_reason="TARGET_HIT" if outcome == "WIN" else "STOP_LOSS_HIT",
        lots=1,
        lot_size=75,
        quantity=75,
        confidence_level="MEDIUM",
        gross_pnl_points=100.0,
        gross_pnl=net_pnl + 50.0,
        total_costs=50.0,
        net_pnl=net_pnl,
        net_pnl_pct=(net_pnl / 100_000) * 100,
        duration_bars=5,
        duration_minutes=25,
        max_favorable_excursion=120.0,
        max_adverse_excursion=60.0,
        outcome=outcome,
    )


def make_equity(capitals: list[float]) -> list[EquityPoint]:
    base = datetime(2024, 7, 1)
    return [
        EquityPoint(
            timestamp=base + timedelta(days=i),
            capital=c,
            cash=c,
            unrealized=0.0,
            drawdown_pct=0.0,
            open_positions=0,
        )
        for i, c in enumerate(capitals)
    ]


def make_mock_result(index_id: str = "NIFTY50") -> BacktestResult:
    """Create a synthetic BacktestResult for report tests."""
    trades = [make_trade("WIN", 3_000) for _ in range(8)] + [
        make_trade("LOSS", -1_500) for _ in range(4)
    ]
    equity = make_equity(
        [100_000, 100_500, 101_000, 100_800, 101_500, 102_000, 101_800, 102_500]
    )
    cfg = BacktestConfig(
        index_id=index_id,
        start_date=date(2024, 7, 1),
        end_date=date(2024, 12, 31),
        timeframe="1d",
        mode="TECHNICAL_ONLY",
        simulator_config=SimulatorConfig(initial_capital=100_000),
    )
    return BacktestResult(
        config=cfg,
        index_id=index_id,
        start_date=date(2024, 7, 1),
        end_date=date(2024, 12, 31),
        total_bars=126,
        trading_days=126,
        trade_history=trades,
        total_trades=len(trades),
        total_signals_generated=126,
        actionable_signals=30,
        executed_trades=len(trades),
        equity_curve=equity,
        initial_capital=100_000.0,
        final_capital=102_500.0,
        total_return_pct=2.5,
    )


# ---------------------------------------------------------------------------
# Tests that DON'T need the DB
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """Report generation from synthetic data — no DB required."""

    def setup_method(self):
        self.result = make_mock_result()
        self.metrics = MetricsCalculator.calculate_all(
            self.result.trade_history,
            self.result.equity_curve,
            100_000.0,
        )

    def test_backtest_report_contains_index_id(self):
        report = ReportGenerator.generate_backtest_report(self.result, self.metrics)
        assert "NIFTY50" in report

    def test_backtest_report_contains_grade(self):
        report = ReportGenerator.generate_backtest_report(self.result, self.metrics)
        assert "GRADE:" in report

    def test_backtest_report_contains_all_sections(self):
        report = ReportGenerator.generate_backtest_report(self.result, self.metrics)
        for section in ("RETURNS:", "TRADES:", "RISK:", "BY CONFIDENCE:", "BY EXIT REASON:"):
            assert section in report, f"Missing section: {section!r}"

    def test_backtest_report_is_string(self):
        report = ReportGenerator.generate_backtest_report(self.result, self.metrics)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_backtest_report_has_box_borders(self):
        report = ReportGenerator.generate_backtest_report(self.result, self.metrics)
        assert "\u2554" in report  # ╔
        assert "\u255a" in report  # ╚

    def test_backtest_report_zero_trades(self):
        """Should handle zero-trade case without crashing."""
        zero_metrics = MetricsCalculator.calculate_all([], [], 100_000)
        report = ReportGenerator.generate_backtest_report(self.result, zero_metrics)
        assert isinstance(report, str)

    def test_trade_log_columns(self):
        log = ReportGenerator.generate_trade_log(self.result.trade_history)
        assert "Date" in log
        assert "Type" in log
        assert "PnL" in log
        assert "Exit Reason" in log

    def test_trade_log_row_count(self):
        log = ReportGenerator.generate_trade_log(self.result.trade_history)
        # Header + separator + 12 trades
        lines = [l for l in log.splitlines() if l.strip()]
        assert len(lines) == 2 + len(self.result.trade_history)

    def test_trade_log_empty(self):
        log = ReportGenerator.generate_trade_log([])
        assert isinstance(log, str)

    def test_save_report_creates_file(self, tmp_path):
        report = ReportGenerator.generate_backtest_report(self.result, self.metrics)
        filepath = str(tmp_path / "test_report.txt")
        returned = ReportGenerator.save_report(report, filepath=filepath)
        assert returned == filepath
        assert os.path.exists(filepath)
        content = open(filepath, encoding="utf-8").read()
        assert "NIFTY50" in content

    def test_save_report_auto_path(self, tmp_path):
        """Auto-generated path should land in data/reports/."""
        report = "minimal report content"
        # Patch working dir to tmp so we don't litter the real project
        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            filepath = ReportGenerator.save_report(report, index_id="TEST")
            assert os.path.exists(filepath)
            assert "TEST" in filepath
        finally:
            os.chdir(orig)


# ---------------------------------------------------------------------------
# CLI test — no DB required for --help
# ---------------------------------------------------------------------------


class TestCLI:
    def test_help_exits_zero(self):
        """--help should print usage and exit 0."""
        script = os.path.join(
            os.path.dirname(__file__), "..", "scripts", "run_backtest.py"
        )
        result = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--index" in result.stdout

    def test_missing_required_args_exits_nonzero(self):
        """Missing --index/--start/--end should exit with error."""
        script = os.path.join(
            os.path.dirname(__file__), "..", "scripts", "run_backtest.py"
        )
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Tests that require real DB data
# ---------------------------------------------------------------------------


class TestFullBacktest:
    """Requires real NIFTY50 data in the database."""

    def test_6_month_backtest_runs_and_has_metrics(self, db_with_nifty):
        """Full 6-month daily backtest should complete and return sane metrics."""
        runner = StrategyRunner(db_with_nifty)

        # Use DataReplayEngine to find a valid date range rather than hard-coding
        from src.backtest.data_replay import DataReplayEngine
        engine = DataReplayEngine(db_with_nifty)
        dr = engine.get_date_range("NIFTY50", "1d")
        if not dr.get("first_ts") or not dr.get("last_ts"):
            pytest.skip("Cannot determine date range for NIFTY50")

        first = date.fromisoformat(dr["first_ts"][:10])
        last = date.fromisoformat(dr["last_ts"][:10])

        # Use the last ~6 months of available data
        start = max(first, last - timedelta(days=180))
        end = last

        config = BacktestConfig(
            index_id="NIFTY50",
            start_date=start,
            end_date=end,
            timeframe="1d",
            mode="TECHNICAL_ONLY",
            simulator_config=SimulatorConfig(initial_capital=100_000),
        )

        t0 = time.time()
        result = runner.run(config)
        elapsed = time.time() - t0

        assert elapsed < 60, f"Backtest took {elapsed:.1f}s — expected < 60s"
        assert isinstance(result.trade_history, list)
        assert result.initial_capital == 100_000.0
        assert isinstance(result.total_return_pct, float)

        calc = MetricsCalculator()
        metrics = calc.calculate_all(result.trade_history, result.equity_curve, 100_000)
        assert metrics is not None
        assert metrics.total_trades == len(result.trade_history)
        assert metrics.strategy_grade in ("A", "B", "C", "D", "F")

    def test_report_generates_for_real_backtest(self, db_with_nifty):
        """Generated report contains expected sections for real run."""
        runner = StrategyRunner(db_with_nifty)

        from src.backtest.data_replay import DataReplayEngine
        engine = DataReplayEngine(db_with_nifty)
        dr = engine.get_date_range("NIFTY50", "1d")
        if not dr.get("first_ts"):
            pytest.skip("Cannot determine date range")

        last = date.fromisoformat(dr["last_ts"][:10])
        start = last - timedelta(days=90)
        config = BacktestConfig(
            index_id="NIFTY50",
            start_date=start,
            end_date=last,
            timeframe="1d",
            mode="TECHNICAL_ONLY",
            simulator_config=SimulatorConfig(initial_capital=100_000),
        )
        result = runner.run(config)
        metrics = MetricsCalculator.calculate_all(
            result.trade_history, result.equity_curve, 100_000
        )
        report = ReportGenerator.generate_backtest_report(result, metrics)

        assert "NIFTY50" in report
        assert "GRADE:" in report
        assert "RETURNS:" in report


class TestWalkForwardIntegration:
    """Requires ~500 bars of NIFTY50 data."""

    def test_walk_forward_produces_windows(self, db_with_nifty_2y):
        validator = WalkForwardValidator(db_with_nifty_2y)

        from src.backtest.data_replay import DataReplayEngine
        engine = DataReplayEngine(db_with_nifty_2y)
        dr = engine.get_date_range("NIFTY50", "1d")
        if not dr.get("first_ts"):
            pytest.skip("Cannot determine date range")

        first = date.fromisoformat(dr["first_ts"][:10])
        last = date.fromisoformat(dr["last_ts"][:10])

        # Check actual trading days available
        available = validator._get_trading_days("NIFTY50", first, last)
        if len(available) < 400:
            pytest.skip(f"Not enough data ({len(available)} days) for walk-forward")

        # Use small windows to ensure at least 2 can fit
        train_days = min(180, len(available) // 3)
        test_days = min(60, len(available) // 6)
        step_days = test_days

        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=first,
            full_end_date=last,
            train_window_days=train_days,
            test_window_days=test_days,
            step_days=step_days,
        )

        result = validator.run_walk_forward(config)

        assert result.total_windows >= 2
        assert len(result.windows) == result.total_windows
        assert result.overfitting_assessment in (
            "LOW_RISK", "MODERATE_RISK", "HIGH_RISK", "SEVERE"
        )
        assert 0.0 <= result.overfitting_score <= 1.0
        assert isinstance(result.verdict, str)

    def test_walk_forward_report_generates(self, db_with_nifty_2y):
        validator = WalkForwardValidator(db_with_nifty_2y)

        from src.backtest.data_replay import DataReplayEngine
        engine = DataReplayEngine(db_with_nifty_2y)
        dr = engine.get_date_range("NIFTY50", "1d")
        if not dr.get("first_ts"):
            pytest.skip("Cannot determine date range")

        first = date.fromisoformat(dr["first_ts"][:10])
        last = date.fromisoformat(dr["last_ts"][:10])
        available = validator._get_trading_days("NIFTY50", first, last)
        if len(available) < 300:
            pytest.skip("Insufficient data for walk-forward report test")

        train_days = min(120, len(available) // 3)
        test_days = min(40, len(available) // 6)

        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=first,
            full_end_date=last,
            train_window_days=train_days,
            test_window_days=test_days,
            step_days=test_days,
        )
        result = validator.run_walk_forward(config)
        report = ReportGenerator.generate_walk_forward_report(result)

        assert "WALK-FORWARD VALIDATION REPORT" in report
        assert "NIFTY50" in report
        assert "AGGREGATE:" in report
        assert "OVERFITTING ASSESSMENT:" in report
        assert "VERDICT:" in report


class TestMultiIndex:
    """Multi-index comparison — requires data for multiple indices."""

    def test_multi_index_backtest(self, db_with_nifty):
        """Run backtest on all available indices and compare metrics."""
        db = db_with_nifty
        from src.backtest.data_replay import DataReplayEngine
        engine = DataReplayEngine(db)
        try:
            available = engine.list_available_indices()
        except Exception:
            pytest.skip("Could not list available indices")

        if not available or len(available) < 2:
            pytest.skip("Need at least 2 indices for multi-index test")

        results = []
        metrics_list = []
        labels = []

        for idx_id in list(available)[:2]:
            dr = engine.get_date_range(idx_id, "1d")
            if not dr.get("first_ts"):
                continue
            last = date.fromisoformat(dr["last_ts"][:10])
            start = last - timedelta(days=90)
            config = BacktestConfig(
                index_id=idx_id,
                start_date=start,
                end_date=last,
                timeframe="1d",
                mode="TECHNICAL_ONLY",
                simulator_config=SimulatorConfig(initial_capital=100_000),
            )
            try:
                runner = StrategyRunner(db)
                r = runner.run(config)
                m = MetricsCalculator.calculate_all(r.trade_history, r.equity_curve, 100_000)
                results.append(r)
                metrics_list.append(m)
                labels.append(idx_id)
            except Exception as exc:
                # Skip indices that don't have enough data
                continue

        if len(metrics_list) < 2:
            pytest.skip("Could not run backtests on 2 indices")

        comparison = MetricsCalculator.compare_backtests(metrics_list, labels)
        assert comparison.best_overall in labels
        assert len(comparison.ranking) == len(labels)


class TestPerformance:
    """Performance benchmark — 1-year daily backtest < 60 seconds."""

    def test_one_year_daily_under_60s(self, db_with_nifty):
        from src.backtest.data_replay import DataReplayEngine
        engine = DataReplayEngine(db_with_nifty)
        dr = engine.get_date_range("NIFTY50", "1d")
        if not dr.get("first_ts"):
            pytest.skip("Cannot determine date range")

        last = date.fromisoformat(dr["last_ts"][:10])
        start = last - timedelta(days=365)

        config = BacktestConfig(
            index_id="NIFTY50",
            start_date=start,
            end_date=last,
            timeframe="1d",
            mode="TECHNICAL_ONLY",
            simulator_config=SimulatorConfig(initial_capital=100_000),
        )
        runner = StrategyRunner(db_with_nifty)
        t0 = time.time()
        runner.run(config)
        elapsed = time.time() - t0

        assert elapsed < 60, f"1-year backtest took {elapsed:.1f}s (limit: 60s)"
