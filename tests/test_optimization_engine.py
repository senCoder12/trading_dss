"""Tests for the optimization engine (src/backtest/optimizer/optimization_engine.py)."""

from __future__ import annotations

import copy
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.backtest.metrics import BacktestMetrics, MetricsCalculator
from src.backtest.optimizer.optimization_engine import (
    EvaluationResult,
    OptimizationConfig,
    OptimizationEngine,
    OptimizationResult,
    _params_to_key,
)
from src.backtest.optimizer.param_space import (
    ParameterApplicator,
    ParameterDef,
    ParameterSpace,
    ParameterSpaceLoader,
)
from src.backtest.strategy_runner import BacktestConfig, BacktestResult
from src.backtest.trade_simulator import SimulatorConfig
from src.backtest.walk_forward import WalkForwardConfig, WalkForwardResult, WalkForwardValidator


# ---------------------------------------------------------------------------
# Helpers — fake metrics
# ---------------------------------------------------------------------------


def _make_metrics(
    total_return_pct: float = 10.0,
    sharpe_ratio: float = 1.5,
    win_rate: float = 55.0,
    profit_factor: float = 1.8,
    max_drawdown_pct: float = 8.0,
    total_trades: int = 50,
    expected_value_per_trade: float = 200.0,
) -> BacktestMetrics:
    """Build a BacktestMetrics with controllable key fields; others zeroed."""
    return BacktestMetrics(
        total_return_pct=total_return_pct,
        total_return_amount=total_return_pct * 1000,
        annualized_return_pct=total_return_pct * 2,
        monthly_returns=[], best_month_pct=0.0, worst_month_pct=0.0,
        positive_months=0, negative_months=0, monthly_win_rate=0.0,
        total_trades=total_trades,
        winning_trades=int(total_trades * win_rate / 100),
        losing_trades=total_trades - int(total_trades * win_rate / 100),
        breakeven_trades=0,
        win_rate=win_rate,
        avg_win_amount=500.0, avg_loss_amount=300.0,
        avg_win_pct=1.0, avg_loss_pct=-0.6,
        largest_win=2000.0, largest_loss=-1000.0,
        largest_win_pct=4.0, largest_loss_pct=-2.0,
        avg_trade_duration_bars=5.0,
        avg_winning_trade_duration=4.0, avg_losing_trade_duration=6.0,
        profit_factor=profit_factor,
        payoff_ratio=1.67,
        expected_value_per_trade=expected_value_per_trade,
        expected_value_pct=0.2,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_amount=max_drawdown_pct * 1000,
        max_drawdown_duration_bars=10,
        max_drawdown_start=None, max_drawdown_end=None,
        avg_drawdown_pct=max_drawdown_pct / 2,
        max_consecutive_wins=5, max_consecutive_losses=3, current_streak=1,
        max_recovery_time_bars=15,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sharpe_ratio * 1.2,
        calmar_ratio=sharpe_ratio * 0.8,
        high_confidence_trades=10, high_confidence_win_rate=60.0, high_confidence_avg_pnl=300.0,
        medium_confidence_trades=20, medium_confidence_win_rate=55.0, medium_confidence_avg_pnl=200.0,
        low_confidence_trades=20, low_confidence_win_rate=50.0, low_confidence_avg_pnl=100.0,
        call_trades=25, call_win_rate=55.0, call_total_pnl=5000.0,
        put_trades=25, put_win_rate=55.0, put_total_pnl=5000.0,
        target_hit_count=15, target_hit_pct=30.0,
        sl_hit_count=10, sl_hit_pct=20.0,
        trailing_sl_count=5, trailing_sl_pct=10.0,
        forced_eod_count=20, forced_eod_avg_pnl=50.0,
        trades_by_regime={}, best_regime="", worst_regime="",
        trades_by_day_of_week={}, trades_by_hour={},
        best_day_of_week="", worst_day_of_week="",
        is_profitable=total_return_pct > 0,
        has_edge=profit_factor > 1.0,
        strategy_grade="B",
        assessment="Test metrics",
    )


def _make_bt_result(
    config: BacktestConfig,
    metrics: Optional[BacktestMetrics] = None,
) -> BacktestResult:
    """Build a minimal BacktestResult."""
    m = metrics or _make_metrics()
    return BacktestResult(
        config=config,
        index_id=config.index_id,
        start_date=config.start_date,
        end_date=config.end_date,
        total_bars=250,
        trading_days=250,
        trade_history=[],
        total_trades=m.total_trades,
        total_signals_generated=100,
        actionable_signals=80,
        executed_trades=m.total_trades,
        metrics=m,
        equity_curve=[],
        initial_capital=100_000.0,
        final_capital=100_000 * (1 + m.total_return_pct / 100),
        total_return_pct=m.total_return_pct,
    )


def _make_wf_result() -> WalkForwardResult:
    """Build a minimal WalkForwardResult."""
    return WalkForwardResult(
        config=MagicMock(spec=WalkForwardConfig),
        windows=[],
        total_windows=4,
        profitable_test_windows=3,
        test_profitability_rate=0.75,
        avg_train_return=12.0,
        avg_test_return=8.0,
        avg_degradation=4.0,
        avg_train_win_rate=58.0,
        avg_test_win_rate=52.0,
        avg_train_sharpe=1.6,
        avg_test_sharpe=1.1,
        max_test_drawdown=10.0,
        combined_test_return=32.0,
        combined_test_trades=80,
        combined_test_win_rate=53.0,
        overfitting_score=0.3,
        overfitting_assessment="LOW_RISK",
        is_robust=True,
        verdict="Strategy appears robust.",
    )


def _tiny_space() -> ParameterSpace:
    """2 params × 2 values each = 4 grid combos."""
    return ParameterSpace(parameters=[
        ParameterDef(
            name="stop_loss_atr_multiplier",
            display_name="SL", description="stop loss",
            param_type="float", min_value=1.0, max_value=1.5,
            step=0.5, default=1.5,
        ),
        ParameterDef(
            name="target_atr_multiplier",
            display_name="Target", description="target",
            param_type="float", min_value=2.0, max_value=2.5,
            step=0.5, default=2.0,
        ),
    ])


def _base_opt_config(**overrides: Any) -> OptimizationConfig:
    """Build a basic OptimizationConfig for tests."""
    defaults = dict(
        index_id="NIFTY50",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        parameter_space=_tiny_space(),
        search_method="grid",
        show_progress=False,
        run_walk_forward=False,
        min_trades=5,          # Low threshold for test data
        max_drawdown_limit=50.0,
        min_win_rate=30.0,
        min_profit_factor=0.5,
    )
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests — Tiny parameter space (2 params × 2 values = 4 combos)
# ---------------------------------------------------------------------------

class TestTinyParameterSpace:
    """Test with a minimal 4-combo grid to verify the full pipeline."""

    def _mock_engine(self, metrics_per_combo: Optional[list[BacktestMetrics]] = None):
        """Create an engine with a mocked StrategyRunner."""
        db = MagicMock()
        engine = OptimizationEngine(db)

        # Prepare 4 metrics (one per combo) — or use defaults
        if metrics_per_combo is None:
            metrics_per_combo = [
                _make_metrics(sharpe_ratio=1.0, total_return_pct=5.0),
                _make_metrics(sharpe_ratio=2.0, total_return_pct=15.0),
                _make_metrics(sharpe_ratio=1.5, total_return_pct=10.0),
                _make_metrics(sharpe_ratio=0.5, total_return_pct=2.0),
            ]

        call_count = [0]

        def fake_run(config: BacktestConfig) -> BacktestResult:
            idx = min(call_count[0], len(metrics_per_combo) - 1)
            m = metrics_per_combo[idx]
            call_count[0] += 1
            return _make_bt_result(config, m)

        engine.strategy_runner.run = fake_run
        return engine

    def test_runs_all_4_combos(self):
        engine = self._mock_engine()
        config = _base_opt_config()
        result = engine.run(config)

        assert result.total_evaluations == 4
        assert len(result.all_results) == 4

    def test_best_is_highest_sharpe(self):
        engine = self._mock_engine()
        config = _base_opt_config(primary_objective="sharpe_ratio")
        result = engine.run(config)

        assert result.best_metrics is not None
        assert result.best_metrics.sharpe_ratio == 2.0

    def test_best_params_populated(self):
        engine = self._mock_engine()
        config = _base_opt_config()
        result = engine.run(config)

        assert result.best_params is not None
        assert "stop_loss_atr_multiplier" in result.best_params
        assert "target_atr_multiplier" in result.best_params

    def test_all_pass_constraints_with_lenient_thresholds(self):
        engine = self._mock_engine()
        config = _base_opt_config()
        result = engine.run(config)

        assert result.passed_constraints == 4
        assert result.failed_constraints == 0
        assert result.errors == 0

    def test_ranked_results_have_ranks(self):
        engine = self._mock_engine()
        config = _base_opt_config()
        result = engine.run(config)

        for i, r in enumerate(result.ranked_results):
            assert r.rank == i + 1

    def test_optimization_duration_positive(self):
        engine = self._mock_engine()
        config = _base_opt_config()
        result = engine.run(config)

        assert result.optimization_duration_seconds > 0


# ---------------------------------------------------------------------------
# Tests — Constraint filtering
# ---------------------------------------------------------------------------

class TestConstraintFiltering:
    """Test that constraint violations correctly filter results."""

    def test_too_few_trades_fails(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        # All combos produce only 10 trades
        low_trade_metrics = _make_metrics(total_trades=10)

        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg, low_trade_metrics)

        config = _base_opt_config(min_trades=30)
        result = engine.run(config)

        assert result.passed_constraints == 0
        assert result.failed_constraints == 4
        assert result.best_params is None

    def test_high_drawdown_fails(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        high_dd_metrics = _make_metrics(max_drawdown_pct=30.0)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg, high_dd_metrics)

        config = _base_opt_config(max_drawdown_limit=20.0)
        result = engine.run(config)

        assert result.passed_constraints == 0

    def test_low_win_rate_fails(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        low_wr_metrics = _make_metrics(win_rate=25.0)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg, low_wr_metrics)

        config = _base_opt_config(min_win_rate=40.0)
        result = engine.run(config)

        assert result.passed_constraints == 0

    def test_low_profit_factor_fails(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        low_pf_metrics = _make_metrics(profit_factor=0.5)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg, low_pf_metrics)

        config = _base_opt_config(min_profit_factor=0.9)
        result = engine.run(config)

        assert result.passed_constraints == 0

    def test_mixed_pass_fail(self):
        """Some combos pass, some fail — verify correct counts."""
        db = MagicMock()
        engine = OptimizationEngine(db)

        metrics_list = [
            _make_metrics(total_trades=50, win_rate=55.0),  # pass
            _make_metrics(total_trades=5, win_rate=55.0),   # fail: too few trades
            _make_metrics(total_trades=50, win_rate=55.0),  # pass
            _make_metrics(total_trades=50, win_rate=20.0),  # fail: low win rate
        ]
        call_count = [0]

        def fake_run(cfg):
            idx = min(call_count[0], len(metrics_list) - 1)
            m = metrics_list[idx]
            call_count[0] += 1
            return _make_bt_result(cfg, m)

        engine.strategy_runner.run = fake_run
        config = _base_opt_config(min_trades=30, min_win_rate=40.0)
        result = engine.run(config)

        assert result.passed_constraints == 2
        assert result.failed_constraints == 2

    def test_zero_trades_auto_fails(self):
        """Backtest producing 0 trades automatically fails min_trades."""
        db = MagicMock()
        engine = OptimizationEngine(db)

        zero_metrics = _make_metrics(total_trades=0, win_rate=0.0, profit_factor=0.0)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg, zero_metrics)

        config = _base_opt_config(min_trades=1)
        result = engine.run(config)

        assert result.passed_constraints == 0


# ---------------------------------------------------------------------------
# Tests — Ranking
# ---------------------------------------------------------------------------

class TestRanking:
    """Verify best result has the highest primary objective."""

    def test_ranking_by_sharpe(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        metrics = [
            _make_metrics(sharpe_ratio=0.5),
            _make_metrics(sharpe_ratio=2.5),
            _make_metrics(sharpe_ratio=1.0),
            _make_metrics(sharpe_ratio=1.8),
        ]
        call_count = [0]

        def fake_run(cfg):
            idx = min(call_count[0], len(metrics) - 1)
            call_count[0] += 1
            return _make_bt_result(cfg, metrics[idx])

        engine.strategy_runner.run = fake_run
        config = _base_opt_config(primary_objective="sharpe_ratio")
        result = engine.run(config)

        assert result.ranked_results[0].metrics.sharpe_ratio == 2.5
        assert result.ranked_results[0].rank == 1

    def test_ranking_by_total_return(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        metrics = [
            _make_metrics(total_return_pct=5.0),
            _make_metrics(total_return_pct=25.0),
            _make_metrics(total_return_pct=10.0),
            _make_metrics(total_return_pct=15.0),
        ]
        call_count = [0]

        def fake_run(cfg):
            idx = min(call_count[0], len(metrics) - 1)
            call_count[0] += 1
            return _make_bt_result(cfg, metrics[idx])

        engine.strategy_runner.run = fake_run
        config = _base_opt_config(primary_objective="total_return_pct")
        result = engine.run(config)

        assert result.ranked_results[0].metrics.total_return_pct == 25.0

    def test_drawdown_lower_is_better_in_secondary(self):
        """When max_drawdown_pct is a secondary, lower values rank higher."""
        db = MagicMock()
        engine = OptimizationEngine(db)

        # Same sharpe (primary ties), different drawdown (secondary differentiates)
        metrics = [
            _make_metrics(sharpe_ratio=1.5, max_drawdown_pct=15.0, win_rate=50.0),
            _make_metrics(sharpe_ratio=1.5, max_drawdown_pct=5.0, win_rate=50.0),
        ]

        # Space with only 2 combos
        space = ParameterSpace(parameters=[
            ParameterDef(
                name="stop_loss_atr_multiplier",
                display_name="SL", description="test",
                param_type="float", min_value=1.0, max_value=1.5,
                step=0.5, default=1.5,
            ),
            ParameterDef(
                name="target_atr_multiplier",
                display_name="Target", description="test",
                param_type="float", min_value=2.5, max_value=2.5,
                step=0.5, default=2.5,
            ),
        ])

        call_count = [0]

        def fake_run(cfg):
            idx = min(call_count[0], len(metrics) - 1)
            call_count[0] += 1
            return _make_bt_result(cfg, metrics[idx])

        engine.strategy_runner.run = fake_run
        config = _base_opt_config(
            parameter_space=space,
            primary_objective="sharpe_ratio",
            secondary_objectives=["max_drawdown_pct"],
        )
        result = engine.run(config)

        # The one with 5% drawdown should rank higher (lower is better)
        assert result.ranked_results[0].metrics.max_drawdown_pct == 5.0

    def test_identical_primary_differentiated_by_secondary(self):
        """When primary is identical, secondary objectives break ties."""
        db = MagicMock()
        engine = OptimizationEngine(db)

        metrics = [
            _make_metrics(sharpe_ratio=1.5, win_rate=45.0),
            _make_metrics(sharpe_ratio=1.5, win_rate=65.0),
        ]

        space = ParameterSpace(parameters=[
            ParameterDef(
                name="stop_loss_atr_multiplier",
                display_name="SL", description="test",
                param_type="float", min_value=1.0, max_value=1.5,
                step=0.5, default=1.5,
            ),
            ParameterDef(
                name="target_atr_multiplier",
                display_name="Target", description="test",
                param_type="float", min_value=2.5, max_value=2.5,
                step=0.5, default=2.5,
            ),
        ])

        call_count = [0]

        def fake_run(cfg):
            idx = min(call_count[0], len(metrics) - 1)
            call_count[0] += 1
            return _make_bt_result(cfg, metrics[idx])

        engine.strategy_runner.run = fake_run
        config = _base_opt_config(
            parameter_space=space,
            primary_objective="sharpe_ratio",
            secondary_objectives=["win_rate"],
        )
        result = engine.run(config)

        # Higher win_rate should win the tiebreak
        assert result.ranked_results[0].metrics.win_rate == 65.0


# ---------------------------------------------------------------------------
# Tests — Default params comparison (baseline)
# ---------------------------------------------------------------------------

class TestDefaultParamsComparison:
    """Ensure the baseline (default values) result is found."""

    def test_default_result_found(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        space = _tiny_space()
        defaults = space.get_default_values()

        metrics_list = [
            _make_metrics(total_return_pct=5.0),
            _make_metrics(total_return_pct=15.0),
            _make_metrics(total_return_pct=10.0),
            _make_metrics(total_return_pct=8.0),
        ]
        call_count = [0]

        def fake_run(cfg):
            idx = min(call_count[0], len(metrics_list) - 1)
            call_count[0] += 1
            return _make_bt_result(cfg, metrics_list[idx])

        engine.strategy_runner.run = fake_run
        config = _base_opt_config(parameter_space=space)
        result = engine.run(config)

        # The default combo (SL=1.5, Target=2.0) should appear somewhere in all_results
        found = False
        for r in result.all_results:
            if r.params == defaults:
                found = True
                break
        assert found, f"Default params {defaults} not found in results"

    def test_default_result_is_baseline_in_output(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        space = _tiny_space()

        # We'll make the default combo have known metrics
        combos = space.generate_grid()
        defaults = space.get_default_values()

        metrics_map = {}
        for combo in combos:
            if combo == defaults:
                metrics_map[str(combo)] = _make_metrics(total_return_pct=5.0)
            else:
                metrics_map[str(combo)] = _make_metrics(total_return_pct=15.0)

        grid_idx = [0]

        def fake_run(cfg):
            combo = combos[min(grid_idx[0], len(combos) - 1)]
            grid_idx[0] += 1
            return _make_bt_result(cfg, metrics_map[str(combo)])

        engine.strategy_runner.run = fake_run
        config = _base_opt_config(parameter_space=space)
        result = engine.run(config)

        if result.default_params_result and result.default_params_result.metrics:
            assert result.default_params_result.metrics.total_return_pct == 5.0


# ---------------------------------------------------------------------------
# Tests — Random search reproducibility
# ---------------------------------------------------------------------------

class TestRandomSearch:
    """Test random search with seed produces reproducible results."""

    def test_random_seed_reproducibility(self):
        db = MagicMock()

        metrics_counter = [0]

        def make_engine():
            e = OptimizationEngine(db)
            counter = [0]

            def fake_run(cfg):
                m = _make_metrics(sharpe_ratio=1.0 + counter[0] * 0.1)
                counter[0] += 1
                return _make_bt_result(cfg, m)

            e.strategy_runner.run = fake_run
            return e

        # Larger space for random search to be meaningful
        space = ParameterSpace(parameters=[
            ParameterDef(
                name="stop_loss_atr_multiplier",
                display_name="SL", description="test",
                param_type="float", min_value=1.0, max_value=3.0,
                step=0.25, default=1.5,
            ),
            ParameterDef(
                name="target_atr_multiplier",
                display_name="Target", description="test",
                param_type="float", min_value=2.0, max_value=5.0,
                step=0.25, default=3.0,
            ),
        ])

        config = _base_opt_config(
            parameter_space=space,
            search_method="random",
            max_evaluations=10,
            random_seed=42,
        )

        # Run twice with same seed
        engine1 = make_engine()
        result1 = engine1.run(config)

        engine2 = make_engine()
        result2 = engine2.run(config)

        # Same combos should be generated in the same order
        params1 = [r.params for r in result1.all_results]
        params2 = [r.params for r in result2.all_results]
        assert params1 == params2

    def test_random_respects_max_evaluations(self):
        db = MagicMock()
        engine = OptimizationEngine(db)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg)

        space = ParameterSpace(parameters=[
            ParameterDef(
                name="stop_loss_atr_multiplier",
                display_name="SL", description="test",
                param_type="float", min_value=1.0, max_value=3.0,
                step=0.25, default=1.5,
            ),
            ParameterDef(
                name="target_atr_multiplier",
                display_name="Target", description="test",
                param_type="float", min_value=2.0, max_value=5.0,
                step=0.25, default=3.0,
            ),
        ])

        config = _base_opt_config(
            parameter_space=space,
            search_method="random",
            max_evaluations=5,
        )
        result = engine.run(config)

        # May be less than 5 if some combos are invalid (target <= SL),
        # but total generated should be 5
        assert result.total_evaluations <= 5


# ---------------------------------------------------------------------------
# Tests — Invalid combination filtering (target <= SL)
# ---------------------------------------------------------------------------

class TestInvalidCombinationFiltering:
    """Test that target <= SL combos are filtered before backtesting."""

    def test_invalid_combos_filtered_out(self):
        """Space where some combos have target <= SL."""
        db = MagicMock()
        engine = OptimizationEngine(db)

        run_count = [0]

        def counting_run(cfg):
            run_count[0] += 1
            return _make_bt_result(cfg)

        engine.strategy_runner.run = counting_run

        # SL: [1.0, 2.0, 3.0] × Target: [1.5, 2.5, 3.5]
        # Invalid: SL=2.0,T=1.5 | SL=3.0,T=1.5 | SL=3.0,T=2.5
        space = ParameterSpace(parameters=[
            ParameterDef(
                name="stop_loss_atr_multiplier",
                display_name="SL", description="test",
                param_type="float", min_value=1.0, max_value=3.0,
                step=1.0, default=1.0,
            ),
            ParameterDef(
                name="target_atr_multiplier",
                display_name="Target", description="test",
                param_type="float", min_value=1.5, max_value=3.5,
                step=1.0, default=2.5,
            ),
        ])

        config = _base_opt_config(parameter_space=space)
        result = engine.run(config)

        # 9 total combos, 3 invalid → 6 valid backtests run
        # (SL=1.0 works with all 3 targets; SL=2.0 with T=2.5,3.5; SL=3.0 with T=3.5)
        assert run_count[0] == 6
        assert result.total_evaluations == 6


# ---------------------------------------------------------------------------
# Tests — Progress reporting
# ---------------------------------------------------------------------------

class TestProgressReporting:
    """Verify progress reporting doesn't crash."""

    def test_progress_prints_without_error(self, capsys):
        db = MagicMock()
        engine = OptimizationEngine(db)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg)

        config = _base_opt_config(
            show_progress=True,
            progress_interval=2,  # Print every 2 evaluations
        )
        result = engine.run(config)

        captured = capsys.readouterr()
        # Should have printed at least one progress line
        assert "[2/4]" in captured.out or "[4/4]" in captured.out or "Optimization:" in captured.out

    def test_no_output_when_progress_disabled(self, capsys):
        db = MagicMock()
        engine = OptimizationEngine(db)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg)

        config = _base_opt_config(show_progress=False)
        result = engine.run(config)

        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Tests — KeyboardInterrupt handling
# ---------------------------------------------------------------------------

class TestKeyboardInterrupt:
    """Test that interrupt returns partial results."""

    def test_interrupt_returns_partial(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        call_count = [0]

        def interrupting_run(cfg):
            call_count[0] += 1
            if call_count[0] >= 3:
                raise KeyboardInterrupt()
            return _make_bt_result(cfg)

        engine.strategy_runner.run = interrupting_run

        config = _base_opt_config(show_progress=False)
        result = engine.run(config)

        # Should have 2 completed results (interrupted on 3rd)
        assert result.total_evaluations == 2
        assert len(result.all_results) == 2

    def test_interrupt_still_ranks(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        call_count = [0]

        def interrupting_run(cfg):
            call_count[0] += 1
            if call_count[0] >= 3:
                raise KeyboardInterrupt()
            return _make_bt_result(cfg, _make_metrics(sharpe_ratio=call_count[0] * 0.5))

        engine.strategy_runner.run = interrupting_run

        config = _base_opt_config(show_progress=False)
        result = engine.run(config)

        # Partial results should still be ranked
        assert len(result.ranked_results) > 0
        assert result.best_params is not None


# ---------------------------------------------------------------------------
# Tests — Backtest errors
# ---------------------------------------------------------------------------

class TestBacktestErrors:
    """Test handling of backtest failures."""

    def test_error_recorded_not_crash(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        call_count = [0]

        def flaky_run(cfg):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated backtest crash")
            return _make_bt_result(cfg)

        engine.strategy_runner.run = flaky_run

        config = _base_opt_config(show_progress=False)
        result = engine.run(config)

        assert result.errors == 1
        assert result.total_evaluations == 4

        # Find the error result
        error_results = [r for r in result.all_results if r.error is not None]
        assert len(error_results) == 1
        assert "Simulated backtest crash" in error_results[0].error

    def test_all_errors_still_returns(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        engine.strategy_runner.run = MagicMock(side_effect=RuntimeError("All fail"))

        config = _base_opt_config(show_progress=False)
        result = engine.run(config)

        assert result.errors == 4
        assert result.passed_constraints == 0
        assert result.best_params is None


# ---------------------------------------------------------------------------
# Tests — Walk-forward integration
# ---------------------------------------------------------------------------

class TestWalkForward:
    """Test walk-forward validation is triggered correctly."""

    def test_walk_forward_runs_on_top_n(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        metrics = [
            _make_metrics(sharpe_ratio=2.0),
            _make_metrics(sharpe_ratio=1.5),
            _make_metrics(sharpe_ratio=1.0),
            _make_metrics(sharpe_ratio=0.5),
        ]
        call_count = [0]

        def fake_run(cfg):
            idx = min(call_count[0], len(metrics) - 1)
            call_count[0] += 1
            return _make_bt_result(cfg, metrics[idx])

        engine.strategy_runner.run = fake_run

        wf_calls = []

        def mock_wf_run(wf_config):
            wf_calls.append(wf_config)
            return _make_wf_result()

        with patch.object(WalkForwardValidator, "run_walk_forward", side_effect=mock_wf_run):
            config = _base_opt_config(
                run_walk_forward=True,
                walk_forward_top_n=2,
                show_progress=False,
            )
            result = engine.run(config)

        assert len(wf_calls) == 2
        assert len(result.walk_forward_results) == 2

    def test_walk_forward_disabled(self):
        db = MagicMock()
        engine = OptimizationEngine(db)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg)

        config = _base_opt_config(run_walk_forward=False)
        result = engine.run(config)

        assert len(result.walk_forward_results) == 0

    def test_walk_forward_failure_does_not_crash(self):
        db = MagicMock()
        engine = OptimizationEngine(db)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg)

        with patch.object(
            WalkForwardValidator, "run_walk_forward",
            side_effect=RuntimeError("WF failed"),
        ):
            config = _base_opt_config(
                run_walk_forward=True,
                walk_forward_top_n=2,
                show_progress=False,
            )
            result = engine.run(config)

        # Should complete without crash, just no WF results
        assert len(result.walk_forward_results) == 0
        assert result.best_wf_result is None


# ---------------------------------------------------------------------------
# Tests — OptimizationResult.get_summary()
# ---------------------------------------------------------------------------

class TestGetSummary:
    """Test the summary string generation."""

    def test_summary_with_results(self):
        db = MagicMock()
        engine = OptimizationEngine(db)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg)

        config = _base_opt_config(show_progress=False)
        result = engine.run(config)

        summary = result.get_summary()
        assert "OPTIMIZATION RESULTS" in summary
        assert "BEST PARAMETERS" in summary
        assert "BEST METRICS" in summary

    def test_summary_no_viable_params(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        zero_metrics = _make_metrics(total_trades=0, win_rate=0.0, profit_factor=0.0)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg, zero_metrics)

        config = _base_opt_config(min_trades=30, show_progress=False)
        result = engine.run(config)

        summary = result.get_summary()
        assert "NO VIABLE PARAMETERS FOUND" in summary

    def test_summary_with_walk_forward(self):
        db = MagicMock()
        engine = OptimizationEngine(db)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg)

        with patch.object(
            WalkForwardValidator, "run_walk_forward",
            return_value=_make_wf_result(),
        ):
            config = _base_opt_config(
                run_walk_forward=True,
                walk_forward_top_n=1,
                show_progress=False,
            )
            result = engine.run(config)

        summary = result.get_summary()
        assert "WALK-FORWARD VALIDATION" in summary


# ---------------------------------------------------------------------------
# Tests — Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_empty_parameter_space(self):
        """Empty space should produce 1 combo (the cartesian product of nothing)."""
        db = MagicMock()
        engine = OptimizationEngine(db)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg)

        empty_space = ParameterSpace(parameters=[])
        config = _base_opt_config(parameter_space=empty_space)
        result = engine.run(config)

        # itertools.product() of nothing yields one empty dict
        assert result.total_evaluations == 1

    def test_params_to_key_deterministic(self):
        p1 = {"a": 1, "b": 2.0}
        p2 = {"b": 2.0, "a": 1}
        assert _params_to_key(p1) == _params_to_key(p2)

    def test_params_to_key_different_values(self):
        p1 = {"a": 1}
        p2 = {"a": 2}
        assert _params_to_key(p1) != _params_to_key(p2)

    def test_invalid_search_method_raises(self):
        db = MagicMock()
        engine = OptimizationEngine(db)

        config = _base_opt_config(search_method="bayesian")
        with pytest.raises(ValueError, match="Unknown search_method"):
            engine.run(config)

    def test_optimization_lift_computed(self):
        """When default result exists, lift % should be computed."""
        db = MagicMock()
        engine = OptimizationEngine(db)

        space = _tiny_space()
        defaults = space.get_default_values()
        combos = space.generate_grid()

        # Assign metrics: default gets 5%, best gets 15%
        call_count = [0]

        def fake_run(cfg):
            idx = call_count[0]
            call_count[0] += 1
            combo = combos[idx] if idx < len(combos) else combos[-1]
            if combo == defaults:
                return _make_bt_result(cfg, _make_metrics(total_return_pct=5.0))
            return _make_bt_result(cfg, _make_metrics(total_return_pct=15.0))

        engine.strategy_runner.run = fake_run
        config = _base_opt_config(parameter_space=space, show_progress=False)
        result = engine.run(config)

        if result.optimization_lift_pct is not None:
            # Best (15%) vs default (5%) → lift = (15-5)/|5| * 100 = 200%
            assert result.optimization_lift_pct > 0


# ---------------------------------------------------------------------------
# Tests — Integration with real profile (conservative)
# ---------------------------------------------------------------------------

class TestWithConservativeProfile:
    """Test that the engine works with a real optimization profile."""

    @pytest.fixture
    def conservative_space(self) -> ParameterSpace:
        loader = ParameterSpaceLoader()
        return loader.load_profile("conservative")

    def test_conservative_profile_generates_combos(self, conservative_space):
        db = MagicMock()
        engine = OptimizationEngine(db)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg)

        config = _base_opt_config(
            parameter_space=conservative_space,
            show_progress=False,
        )
        result = engine.run(config)

        # Conservative has ~24-72 combos; some may be invalid
        assert result.total_evaluations > 0
        assert result.total_evaluations <= conservative_space.total_combinations

    def test_conservative_random_search(self, conservative_space):
        db = MagicMock()
        engine = OptimizationEngine(db)
        engine.strategy_runner.run = lambda cfg: _make_bt_result(cfg)

        config = _base_opt_config(
            parameter_space=conservative_space,
            search_method="random",
            max_evaluations=10,
            show_progress=False,
        )
        result = engine.run(config)

        assert result.total_evaluations <= 10


# ---------------------------------------------------------------------------
# Integration test — NIFTY50 (requires real database)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    True,  # Set to False when database with NIFTY50 data is available
    reason="Integration test — requires real database with NIFTY50 data",
)
class TestIntegrationNIFTY50:
    """Full integration test with real data (NIFTY50, 6 months, conservative)."""

    def test_real_backtest_optimization(self):
        from src.database.db_manager import DatabaseManager

        db = DatabaseManager.instance()
        db.connect()

        loader = ParameterSpaceLoader()
        space = loader.load_profile("conservative")

        engine = OptimizationEngine(db)
        config = OptimizationConfig(
            index_id="NIFTY50",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            parameter_space=space,
            search_method="grid",
            primary_objective="sharpe_ratio",
            min_trades=10,
            run_walk_forward=True,
            walk_forward_top_n=3,
            show_progress=True,
        )

        result = engine.run(config)

        assert result.total_evaluations > 0
        assert result.optimization_duration_seconds > 0
        print(result.get_summary())
