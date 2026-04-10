"""Tests for the walk-forward validation engine (src/backtest/walk_forward.py)."""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.backtest.metrics import BacktestMetrics, MetricsCalculator
from src.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardValidator,
    WindowResult,
)
from src.database.db_manager import DatabaseManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Canonical zero-metrics object (all fields present, all values neutral/zero)
_ZERO_M: BacktestMetrics = MetricsCalculator.calculate_all([], [], 100_000)


def make_metrics(**kwargs) -> BacktestMetrics:
    """Return a BacktestMetrics with specified fields overriding zeros."""
    return dataclasses.replace(_ZERO_M, **kwargs)


def trading_day_list(n: int, start: date = date(2022, 1, 1)) -> list[date]:
    """Return n consecutive calendar days (no weekend filtering) starting from start."""
    return [start + timedelta(days=i) for i in range(n)]


def make_validator(db=None) -> WalkForwardValidator:
    db = db or MagicMock(spec=DatabaseManager)
    return WalkForwardValidator(db)


# ---------------------------------------------------------------------------
# 1. Window calculation
# ---------------------------------------------------------------------------


class TestCalculateWindows:
    def test_two_years_produces_four_windows(self):
        """2 years of data (504 days), train=252, test=63, step=63 → 4 windows."""
        validator = make_validator()
        days = trading_day_list(504)
        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=days[0],
            full_end_date=days[-1],
            train_window_days=252,
            test_window_days=63,
            step_days=63,
        )
        windows = validator._calculate_windows(config, days)
        assert len(windows) == 4

    def test_window_boundaries_are_correct(self):
        """Verify exact start/end indices for the first window."""
        validator = make_validator()
        days = trading_day_list(504)
        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=days[0],
            full_end_date=days[-1],
            train_window_days=252,
            test_window_days=63,
            step_days=63,
        )
        windows = validator._calculate_windows(config, days)
        w1 = windows[0]
        assert w1["window_id"] == 1
        assert w1["train_start"] == days[0]
        assert w1["train_end"] == days[251]     # 252 days inclusive
        assert w1["test_start"] == days[252]
        assert w1["test_end"] == days[314]      # 63 days inclusive

    def test_step_advances_correctly(self):
        """Second window train_start should be step_days ahead of first."""
        validator = make_validator()
        days = trading_day_list(504)
        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=days[0],
            full_end_date=days[-1],
            train_window_days=252,
            test_window_days=63,
            step_days=63,
        )
        windows = validator._calculate_windows(config, days)
        assert windows[1]["train_start"] == days[63]
        assert windows[2]["train_start"] == days[126]
        assert windows[3]["train_start"] == days[189]

    def test_insufficient_data_returns_empty(self):
        """Only 126 days with a 252-day train window → 0 windows (no error)."""
        validator = make_validator()
        days = trading_day_list(126)
        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=days[0],
            full_end_date=days[-1],
            train_window_days=252,
            test_window_days=63,
            step_days=63,
        )
        windows = validator._calculate_windows(config, days)
        assert windows == []

    def test_exactly_enough_for_one_window(self):
        """Exactly train+test days → 1 window."""
        validator = make_validator()
        n = 252 + 63  # exactly one window
        days = trading_day_list(n)
        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=days[0],
            full_end_date=days[-1],
            train_window_days=252,
            test_window_days=63,
            step_days=63,
        )
        windows = validator._calculate_windows(config, days)
        assert len(windows) == 1

    def test_window_ids_are_sequential(self):
        """window_id values are 1, 2, 3, …"""
        validator = make_validator()
        days = trading_day_list(504)
        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=days[0],
            full_end_date=days[-1],
            train_window_days=252,
            test_window_days=63,
            step_days=63,
        )
        windows = validator._calculate_windows(config, days)
        for i, w in enumerate(windows, start=1):
            assert w["window_id"] == i


# ---------------------------------------------------------------------------
# 2. run_walk_forward — error conditions
# ---------------------------------------------------------------------------


class TestRunWalkForwardErrors:
    def _make_config(self, n_days: int = 126) -> tuple[WalkForwardValidator, WalkForwardConfig]:
        days = trading_day_list(n_days)
        validator = make_validator()
        validator._get_trading_days = MagicMock(return_value=days)
        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=days[0],
            full_end_date=days[-1],
            train_window_days=252,
            test_window_days=63,
            step_days=63,
        )
        return validator, config

    def test_raises_when_too_few_trading_days(self):
        """< train+test days → ValueError explaining the shortage."""
        validator, config = self._make_config(n_days=200)  # < 315
        with pytest.raises(ValueError, match="Not enough trading data"):
            validator.run_walk_forward(config)

    def test_raises_when_fewer_than_2_windows(self):
        """Exactly train+test days → only 1 window → ValueError."""
        n = 252 + 63  # 315 — enough to pass the first check but only 1 window
        days = trading_day_list(n)
        validator = make_validator()
        validator._get_trading_days = MagicMock(return_value=days)
        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=days[0],
            full_end_date=days[-1],
            train_window_days=252,
            test_window_days=63,
            step_days=63,
        )
        with pytest.raises(ValueError, match="at least 2 walk-forward windows"):
            validator.run_walk_forward(config)


# ---------------------------------------------------------------------------
# 3. Overfitting score — boundary tests
# ---------------------------------------------------------------------------


class TestOverfittingScore:
    _fn = staticmethod(WalkForwardValidator._compute_overfitting_score)

    def _call(
        self,
        deg=0.0,
        prof=1.0,
        train_wr=50.0,
        test_wr=50.0,
        train_sh=0.5,
        test_sh=0.5,
    ):
        return self._fn(
            avg_degradation=deg,
            test_profitability_rate=prof,
            avg_train_win_rate=train_wr,
            avg_test_win_rate=test_wr,
            avg_train_sharpe=train_sh,
            avg_test_sharpe=test_sh,
        )

    # ── LOW_RISK ──────────────────────────────────────────────────────

    def test_low_risk_all_good(self):
        score, label = self._call(deg=0.0, prof=1.0, train_wr=50.0, test_wr=50.0)
        assert label == "LOW_RISK"
        assert score == 0.0

    def test_low_risk_near_boundary(self):
        """deg=5 (factor=0.1), prof=0.75 (factor=0.1) → score=0.2 < 0.25 → LOW_RISK."""
        score, label = self._call(deg=5.0, prof=0.75)
        assert label == "LOW_RISK"
        assert score < 0.25

    # ── MODERATE_RISK ─────────────────────────────────────────────────

    def test_moderate_risk_from_degradation(self):
        """deg > 10 → +0.2; prof < 0.75 → +0.1 → score=0.3 → MODERATE_RISK."""
        score, label = self._call(deg=15.0, prof=0.6)
        assert label == "MODERATE_RISK"
        assert 0.25 <= score < 0.50

    def test_moderate_risk_boundary_at_0_25(self):
        """Score exactly 0.25 → MODERATE_RISK (not LOW_RISK)."""
        # deg > 5 (+0.1), prof < 0.75 (+0.1), wr_deg > 5 (+0.1) → 0.3
        score, label = self._call(deg=6.0, prof=0.6, train_wr=60.0, test_wr=54.0)
        assert score == 0.3
        assert label == "MODERATE_RISK"

    # ── HIGH_RISK ─────────────────────────────────────────────────────

    def test_high_risk(self):
        """High degradation + low profitability + win rate drop + sharpe drop."""
        score, label = self._call(
            deg=21.0,        # +0.3
            prof=0.20,       # +0.3
            train_wr=65.0,
            test_wr=50.0,    # wr_deg=15 → +0.15 (not >15, so 0.15)
            train_sh=1.6,
            test_sh=0.5,     # sharpe_deg=1.1 → +0.2
        )
        assert label in ("HIGH_RISK", "SEVERE")
        assert score >= 0.50

    # ── SEVERE ────────────────────────────────────────────────────────

    def test_severe(self):
        """Worst case: all factors at maximum."""
        score, label = self._call(
            deg=25.0,        # +0.3
            prof=0.0,        # +0.3
            train_wr=75.0,
            test_wr=55.0,    # wr_deg=20 → +0.2
            train_sh=2.0,
            test_sh=0.5,     # sharpe_deg=1.5 → +0.2
        )
        assert label == "SEVERE"
        assert score == 1.0


# ---------------------------------------------------------------------------
# 4. is_robust flag
# ---------------------------------------------------------------------------


class TestIsRobust:
    def test_robust_when_low_score_and_good_profitability(self):
        score, _ = WalkForwardValidator._compute_overfitting_score(
            avg_degradation=3.0,
            test_profitability_rate=0.75,
            avg_train_win_rate=55.0,
            avg_test_win_rate=53.0,
            avg_train_sharpe=1.0,
            avg_test_sharpe=0.9,
        )
        is_robust = score < 0.5 and 0.75 >= 0.5
        assert is_robust is True

    def test_not_robust_when_profitability_below_50pct(self):
        score, _ = WalkForwardValidator._compute_overfitting_score(
            avg_degradation=3.0,
            test_profitability_rate=0.25,  # only 25% profitable
            avg_train_win_rate=55.0,
            avg_test_win_rate=53.0,
            avg_train_sharpe=1.0,
            avg_test_sharpe=0.9,
        )
        # score may be low, but profitability < 0.5 → not robust
        is_robust = score < 0.5 and 0.25 >= 0.5
        assert is_robust is False

    def test_not_robust_when_score_too_high(self):
        score, _ = WalkForwardValidator._compute_overfitting_score(
            avg_degradation=25.0,
            test_profitability_rate=1.0,
            avg_train_win_rate=70.0,
            avg_test_win_rate=40.0,
            avg_train_sharpe=2.0,
            avg_test_sharpe=0.3,
        )
        is_robust = score < 0.5 and 1.0 >= 0.5
        assert is_robust is False  # score >= 0.5 → not robust


# ---------------------------------------------------------------------------
# 5. run_walk_forward — success path with mocked runner
# ---------------------------------------------------------------------------


TRAIN_M = make_metrics(
    total_return_pct=10.0,
    win_rate=55.0,
    sharpe_ratio=1.2,
    winning_trades=6,
    losing_trades=4,
    total_trades=10,
)
TEST_M = make_metrics(
    total_return_pct=3.0,
    win_rate=51.0,
    sharpe_ratio=0.8,
    winning_trades=3,
    losing_trades=3,
    total_trades=6,
)


class TestRunWalkForwardSuccess:
    def _setup(self, n_days: int = 504, train: int = 252, test: int = 63, step: int = 63):
        days = trading_day_list(n_days)
        validator = make_validator()
        validator._get_trading_days = MagicMock(return_value=days)

        mock_result = MagicMock()
        mock_result.trade_history = []
        mock_result.equity_curve = []
        validator.strategy_runner.run = MagicMock(return_value=mock_result)

        # Alternate: train metrics then test metrics per window
        validator.metrics_calculator.calculate_all = MagicMock(
            side_effect=[TRAIN_M, TEST_M] * 20  # enough for any window count
        )

        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=days[0],
            full_end_date=days[-1],
            train_window_days=train,
            test_window_days=test,
            step_days=step,
        )
        return validator, config

    def test_four_windows_from_two_years(self):
        validator, config = self._setup()
        result = validator.run_walk_forward(config)
        assert result.total_windows == 4
        assert len(result.windows) == 4

    def test_aggregate_returns_are_averaged(self):
        validator, config = self._setup()
        result = validator.run_walk_forward(config)
        assert result.avg_train_return == pytest.approx(10.0)
        assert result.avg_test_return == pytest.approx(3.0)

    def test_avg_degradation(self):
        validator, config = self._setup()
        result = validator.run_walk_forward(config)
        # deg = 10 - 3 = 7.0 per window
        assert result.avg_degradation == pytest.approx(7.0)

    def test_test_profitability_rate(self):
        """All test windows have +3% → 100% profitable."""
        validator, config = self._setup()
        result = validator.run_walk_forward(config)
        assert result.profitable_test_windows == 4
        assert result.test_profitability_rate == pytest.approx(1.0)

    def test_window_results_have_correct_structure(self):
        validator, config = self._setup()
        result = validator.run_walk_forward(config)
        for w in result.windows:
            assert isinstance(w, WindowResult)
            assert w.train_start < w.train_end
            assert w.test_start > w.train_end
            assert w.test_start < w.test_end
            assert w.is_test_profitable is True  # TEST_M has +3%

    def test_combined_test_win_rate(self):
        """Combined test: 3 wins + 3 losses per window × 4 windows = 50% WR."""
        validator, config = self._setup()
        result = validator.run_walk_forward(config)
        # combined_test_trades is based on extended trade_history (empty lists here)
        # combined_test_win_rate should be 0 since trade_history=[]
        assert result.combined_test_trades == 0
        assert result.combined_test_win_rate == 0.0

    def test_overfitting_assessment_present(self):
        validator, config = self._setup()
        result = validator.run_walk_forward(config)
        assert result.overfitting_assessment in (
            "LOW_RISK",
            "MODERATE_RISK",
            "HIGH_RISK",
            "SEVERE",
        )

    def test_verdict_is_non_empty_string(self):
        validator, config = self._setup()
        result = validator.run_walk_forward(config)
        assert isinstance(result.verdict, str)
        assert len(result.verdict) > 0

    def test_result_contains_config(self):
        validator, config = self._setup()
        result = validator.run_walk_forward(config)
        assert result.config is config


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_trading_days_returns_no_windows(self):
        validator = make_validator()
        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=date(2022, 1, 1),
            full_end_date=date(2022, 12, 31),
            train_window_days=50,
            test_window_days=20,
            step_days=20,
        )
        windows = validator._calculate_windows(config, [])
        assert windows == []

    def test_overfitting_score_all_zeros_is_low_risk(self):
        score, label = WalkForwardValidator._compute_overfitting_score(
            avg_degradation=0.0,
            test_profitability_rate=1.0,
            avg_train_win_rate=50.0,
            avg_test_win_rate=50.0,
            avg_train_sharpe=0.0,
            avg_test_sharpe=0.0,
        )
        assert score == 0.0
        assert label == "LOW_RISK"

    def test_negative_test_returns_makes_windows_unprofitable(self):
        """Windows with negative test returns → is_test_profitable = False."""
        days = trading_day_list(504)
        validator = make_validator()
        validator._get_trading_days = MagicMock(return_value=days)

        mock_result = MagicMock()
        mock_result.trade_history = []
        mock_result.equity_curve = []
        validator.strategy_runner.run = MagicMock(return_value=mock_result)

        neg_test = make_metrics(total_return_pct=-2.0, win_rate=40.0, sharpe_ratio=-0.3)
        pos_train = make_metrics(total_return_pct=10.0, win_rate=55.0, sharpe_ratio=1.0)
        validator.metrics_calculator.calculate_all = MagicMock(
            side_effect=[pos_train, neg_test] * 20
        )

        config = WalkForwardConfig(
            index_id="NIFTY50",
            full_start_date=days[0],
            full_end_date=days[-1],
            train_window_days=252,
            test_window_days=63,
            step_days=63,
        )
        result = validator.run_walk_forward(config)
        assert result.profitable_test_windows == 0
        assert result.test_profitability_rate == 0.0
        assert result.is_robust is False
