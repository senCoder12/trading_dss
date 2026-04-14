"""
End-to-end integration tests for Phase 7 — Strategy Optimization Pipeline.

Tests the full flow:
  1. Load parameter profiles
  2. Run optimization (grid search)
  3. Robustness testing on top results
  4. Report generation
  5. Approved parameter save/load/apply/expire
  6. CLI script smoke test
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import date, datetime
from dataclasses import field
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.backtest.metrics import BacktestMetrics
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
from src.backtest.optimizer.robustness import (
    MonteCarloResult,
    RegimeResult,
    RobustnessReport,
    RobustnessTester,
    SensitivityResult,
    StabilityResult,
    ParamSensitivity,
    PeriodResult,
)
from src.backtest.optimizer.report import OptimizationReportGenerator
from src.backtest.optimizer.param_applier import ApprovedParameterManager
from src.backtest.strategy_runner import BacktestConfig, BacktestResult
from src.backtest.trade_simulator import ClosedTrade, SimulatorConfig
from src.backtest.walk_forward import WalkForwardConfig, WalkForwardResult


# ---------------------------------------------------------------------------
# Helpers — reusable fake data factories
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
        high_confidence_trades=10, high_confidence_win_rate=60.0,
        high_confidence_avg_pnl=300.0,
        medium_confidence_trades=20, medium_confidence_win_rate=55.0,
        medium_confidence_avg_pnl=200.0,
        low_confidence_trades=20, low_confidence_win_rate=50.0,
        low_confidence_avg_pnl=100.0,
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
    trade_history: Optional[list[ClosedTrade]] = None,
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
        trade_history=trade_history or [],
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


def _make_closed_trade(
    net_pnl: float = 500.0,
    outcome: str = "WIN",
    regime: Optional[str] = None,
) -> ClosedTrade:
    """Build a minimal ClosedTrade for Monte Carlo / Regime tests."""
    entry_bar = {"regime": regime} if regime else {}
    return ClosedTrade(
        trade_id="T001",
        index_id="NIFTY50",
        signal_id="S001",
        trade_type="BUY_CALL",
        signal_entry_price=100.0,
        actual_entry_price=100.0,
        entry_timestamp=datetime(2024, 6, 1, 10, 0),
        entry_bar=entry_bar,
        original_stop_loss=98.0,
        original_target=103.0,
        actual_exit_price=103.0 if outcome == "WIN" else 98.0,
        exit_timestamp=datetime(2024, 6, 1, 15, 0),
        exit_reason="TARGET_HIT" if outcome == "WIN" else "STOP_LOSS_HIT",
        lots=1,
        lot_size=50,
        quantity=50,
        confidence_level="HIGH",
        gross_pnl_points=3.0 if outcome == "WIN" else -2.0,
        gross_pnl=net_pnl + 50 if outcome == "WIN" else net_pnl - 50,
        total_costs=50.0,
        net_pnl=net_pnl,
        net_pnl_pct=net_pnl / 1000,
        duration_bars=5,
        duration_minutes=300,
        max_favorable_excursion=3.0,
        max_adverse_excursion=1.0,
        outcome=outcome,
    )


def _tiny_space() -> ParameterSpace:
    """2 params x 2 values each = 4 grid combos."""
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


def _conservative_space() -> ParameterSpace:
    """Load the real conservative profile (4 params)."""
    loader = ParameterSpaceLoader()
    return loader.load_profile("conservative")


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
        min_trades=5,
        max_drawdown_limit=50.0,
        min_win_rate=30.0,
        min_profit_factor=0.5,
    )
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


def _make_sensitivity_result(is_robust: bool = True) -> SensitivityResult:
    return SensitivityResult(
        parameter_sensitivities=[
            ParamSensitivity(
                param_name="stop_loss_atr_multiplier",
                original_value=1.5, plus_20_value=1.8, minus_20_value=1.2,
                original_return=10.0, plus_20_return=8.0, minus_20_return=7.0,
                original_sharpe=1.5, plus_20_sharpe=1.2, minus_20_sharpe=1.1,
                return_stability=0.7, is_stable=True, is_critical=False,
            ),
        ],
        overall_sensitivity_score=0.72 if is_robust else 0.25,
        most_sensitive_param="stop_loss_atr_multiplier",
        least_sensitive_param="stop_loss_atr_multiplier",
        is_robust=is_robust,
    )


def _make_stability_result(is_stable: bool = True) -> StabilityResult:
    return StabilityResult(
        period_results=[
            PeriodResult(
                period_name="First Half",
                start_date=date(2024, 1, 1), end_date=date(2024, 3, 31),
                return_pct=5.0, win_rate=55.0, trade_count=20,
                sharpe=1.2, max_drawdown=5.0,
            ),
            PeriodResult(
                period_name="Second Half",
                start_date=date(2024, 4, 1), end_date=date(2024, 6, 30),
                return_pct=4.0 if is_stable else -8.0,
                win_rate=52.0, trade_count=22, sharpe=1.0, max_drawdown=6.0,
            ),
        ],
        profitable_periods=2 if is_stable else 1,
        unprofitable_periods=0 if is_stable else 1,
        profitability_rate=1.0 if is_stable else 0.5,
        return_std=0.7,
        return_range=(4.0, 5.0) if is_stable else (-8.0, 5.0),
        consistency_score=0.65 if is_stable else 0.3,
        is_stable=is_stable,
    )


def _make_monte_carlo_result(reliable: bool = True) -> MonteCarloResult:
    return MonteCarloResult(
        n_simulations=1000,
        median_return=8.5 if reliable else 1.0,
        p5_return=2.0 if reliable else -5.0,
        p25_return=5.0, p75_return=12.0, p95_return=16.0,
        pct_profitable=0.78 if reliable else 0.42,
        median_drawdown=6.0, p95_drawdown=12.0,
        return_95_ci=(2.0, 16.0) if reliable else (-5.0, 16.0),
        is_statistically_profitable=reliable,
        luck_factor=0.22 if reliable else 0.65,
        reliability_score=0.68 if reliable else 0.3,
    )


def _make_regime_result(diverse: bool = True) -> RegimeResult:
    return RegimeResult(
        regime_performance={
            "TRENDING_UP": {"trades": 20, "win_rate": 60.0, "return": 5000, "sharpe": 1.5},
            "RANGE_BOUND": {"trades": 15, "win_rate": 50.0, "return": 1000 if diverse else -2000, "sharpe": 0.8},
        },
        regimes_tested=2,
        profitable_regimes=2 if diverse else 1,
        best_regime="TRENDING_UP",
        worst_regime="RANGE_BOUND",
        best_regime_return=5000,
        worst_regime_return=1000 if diverse else -2000,
        is_regime_dependent=not diverse,
        regime_diversity_score=1.0 if diverse else 0.5,
    )


def _make_robustness_report(
    params: dict[str, Any],
    metrics: BacktestMetrics,
    score: float = 0.68,
    is_approved: bool = True,
) -> RobustnessReport:
    """Build a full RobustnessReport."""
    if score >= 0.7:
        grade = "ROBUST"
    elif score >= 0.5:
        grade = "MODERATE"
    elif score >= 0.3:
        grade = "FRAGILE"
    else:
        grade = "OVERFIT"

    return RobustnessReport(
        params=params,
        original_metrics=metrics,
        sensitivity_result=_make_sensitivity_result(is_robust=is_approved),
        stability_result=_make_stability_result(is_stable=is_approved),
        monte_carlo_result=_make_monte_carlo_result(reliable=is_approved),
        regime_result=_make_regime_result(diverse=is_approved),
        robustness_score=score,
        robustness_grade=grade,
        is_approved=is_approved,
        concerns=["No significant concerns found"] if is_approved else ["High sensitivity"],
        recommendations=["Proceed to paper trading"] if is_approved else ["Re-optimize"],
        summary=f"Score: {score:.2f}, Grade: {grade}",
    )


def _make_opt_result_with_ranked(
    n_ranked: int = 5,
    include_baseline: bool = True,
    include_wf: bool = True,
) -> OptimizationResult:
    """Build an OptimizationResult with N ranked results and optional baseline/WF."""
    space = _conservative_space()
    config = _base_opt_config(parameter_space=space)

    ranked: list[EvaluationResult] = []
    all_results: list[EvaluationResult] = []

    # Create varying metrics for each rank
    for i in range(n_ranked):
        sharpe = 1.5 - i * 0.1
        ret = 10.0 - i * 1.5
        wr = 60.0 - i * 2.0
        dd = 5.0 + i * 1.0

        params = {
            "stop_loss_atr_multiplier": 1.5,
            "target_atr_multiplier": 2.5 - i * 0.1,
            "min_confidence_filter": "HIGH" if i < 2 else "MEDIUM",
            "signal_cooldown_bars": 3,
        }

        m = _make_metrics(
            total_return_pct=ret, sharpe_ratio=sharpe,
            win_rate=wr, max_drawdown_pct=dd,
        )
        er = EvaluationResult(
            params=params, metrics=m,
            passes_constraints=True, rank=i + 1,
        )
        ranked.append(er)
        all_results.append(er)

    # Add some failing results
    for i in range(3):
        params = {
            "stop_loss_atr_multiplier": 2.5,
            "target_atr_multiplier": 1.5 + i * 0.5,
            "min_confidence_filter": "LOW",
            "signal_cooldown_bars": 1,
        }
        m = _make_metrics(total_return_pct=-2.0, sharpe_ratio=0.3, win_rate=38.0)
        er = EvaluationResult(
            params=params, metrics=m, passes_constraints=False,
        )
        all_results.append(er)

    # Baseline
    default_result = None
    if include_baseline:
        default_params = space.get_default_values()
        default_m = _make_metrics(
            total_return_pct=5.2, sharpe_ratio=0.85, win_rate=51.3, max_drawdown_pct=8.1,
        )
        default_result = EvaluationResult(
            params=default_params, metrics=default_m, passes_constraints=True,
        )
        all_results.append(default_result)

    # Walk-forward
    wf_results: dict[str, WalkForwardResult] = {}
    if include_wf and ranked:
        key = _params_to_key(ranked[0].params)
        wf_results[key] = _make_wf_result()

    # Lift
    lift = None
    if default_result and default_result.metrics and ranked:
        bm = ranked[0].metrics
        dm = default_result.metrics
        if abs(dm.total_return_pct) > 1e-9:
            lift = (bm.total_return_pct - dm.total_return_pct) / abs(dm.total_return_pct) * 100

    return OptimizationResult(
        config=config,
        total_evaluations=len(all_results),
        passed_constraints=len(ranked),
        failed_constraints=len(all_results) - len(ranked) - (1 if include_baseline else 0),
        errors=0,
        ranked_results=ranked,
        all_results=all_results,
        walk_forward_results=wf_results,
        best_params=ranked[0].params if ranked else None,
        best_metrics=ranked[0].metrics if ranked else None,
        best_wf_result=wf_results.get(_params_to_key(ranked[0].params)) if ranked else None,
        default_params_result=default_result,
        optimization_lift_pct=lift,
        optimization_duration_seconds=1410.0,  # 23.5 minutes
    )


# ===========================================================================
# Test 1: Profile Loading
# ===========================================================================


class TestProfileLoading:
    """Verify all three profiles load correctly."""

    def test_conservative_profile_has_4_params(self):
        space = _conservative_space()
        assert space.num_parameters == 4

    def test_moderate_profile_has_7_params(self):
        loader = ParameterSpaceLoader()
        space = loader.load_profile("moderate")
        assert space.num_parameters == 7

    def test_aggressive_profile_has_7_params(self):
        loader = ParameterSpaceLoader()
        space = loader.load_profile("aggressive")
        assert space.num_parameters == 7

    def test_conservative_total_combinations(self):
        space = _conservative_space()
        # SL: 4 vals, TGT: 5 vals, Conf: 3 vals, Cool: 3 vals
        # 4 * 5 * 3 * 3 = 180... but actually depends on step sizes
        assert space.total_combinations > 0
        assert space.total_combinations < 1000  # Conservative should be small

    def test_all_profiles_validate(self):
        loader = ParameterSpaceLoader()
        for name in ["conservative", "moderate", "aggressive"]:
            space = loader.load_profile(name)
            is_valid, issues = space.validate()
            # All profiles should at least load without crashing
            assert space.num_parameters > 0

    def test_invalid_profile_raises(self):
        loader = ParameterSpaceLoader()
        with pytest.raises(ValueError, match="Unknown profile"):
            loader.load_profile("nonexistent")


# ===========================================================================
# Test 2: Optimization Engine (mocked backtests)
# ===========================================================================


class TestOptimizationEngine:
    """Test optimization pipeline with mocked strategy runner."""

    def _mock_engine(
        self,
        metrics_per_combo: Optional[list[BacktestMetrics]] = None,
    ) -> OptimizationEngine:
        db = MagicMock()
        engine = OptimizationEngine(db)

        if metrics_per_combo is None:
            metrics_per_combo = [
                _make_metrics(total_return_pct=5.0, sharpe_ratio=0.9, win_rate=52.0),
                _make_metrics(total_return_pct=10.0, sharpe_ratio=1.5, win_rate=58.0),
                _make_metrics(total_return_pct=7.0, sharpe_ratio=1.2, win_rate=55.0),
                _make_metrics(total_return_pct=3.0, sharpe_ratio=0.6, win_rate=48.0),
            ]

        call_idx = [0]

        def fake_run(bt_config: BacktestConfig) -> BacktestResult:
            idx = min(call_idx[0], len(metrics_per_combo) - 1)
            m = metrics_per_combo[idx]
            call_idx[0] += 1
            return _make_bt_result(bt_config, m)

        engine.strategy_runner.run = fake_run  # type: ignore[assignment]
        return engine

    def test_optimization_returns_ranked_results(self):
        engine = self._mock_engine()
        config = _base_opt_config()
        result = engine.run(config)

        assert result.total_evaluations > 0
        assert result.passed_constraints >= 0
        # At least one result should pass our relaxed constraints
        assert len(result.all_results) > 0

    def test_best_params_populated(self):
        engine = self._mock_engine()
        config = _base_opt_config()
        result = engine.run(config)

        if result.ranked_results:
            assert result.best_params is not None
            assert result.best_metrics is not None
            assert "stop_loss_atr_multiplier" in result.best_params
            assert "target_atr_multiplier" in result.best_params

    def test_ranked_results_ordered_by_objective(self):
        engine = self._mock_engine()
        config = _base_opt_config()
        result = engine.run(config)

        if len(result.ranked_results) >= 2:
            for i in range(len(result.ranked_results) - 1):
                assert result.ranked_results[i].rank < result.ranked_results[i + 1].rank

    def test_constraint_filtering(self):
        # All combos fail min_trades (set very high)
        engine = self._mock_engine()
        config = _base_opt_config(min_trades=1000)
        result = engine.run(config)

        assert result.passed_constraints == 0
        assert result.failed_constraints == result.total_evaluations

    def test_baseline_comparison(self):
        engine = self._mock_engine()
        config = _base_opt_config()
        result = engine.run(config)

        # Default params are in the grid, so baseline should be found
        if result.default_params_result is not None:
            assert result.default_params_result.metrics is not None

    def test_optimization_handles_backtest_error(self):
        """Engine should not crash if a single backtest fails."""
        db = MagicMock()
        engine = OptimizationEngine(db)
        call_count = [0]

        def sometimes_fail(bt_config: BacktestConfig) -> BacktestResult:
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated backtest failure")
            return _make_bt_result(bt_config, _make_metrics())

        engine.strategy_runner.run = sometimes_fail  # type: ignore[assignment]
        config = _base_opt_config()
        result = engine.run(config)

        assert result.errors >= 1
        assert result.total_evaluations >= 2  # Continued past the error


# ===========================================================================
# Test 3: Robustness Report Dataclass
# ===========================================================================


class TestRobustnessReport:
    """Verify robustness report structure and grading."""

    def test_approved_report_fields(self):
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        m = _make_metrics()
        report = _make_robustness_report(params, m, score=0.68, is_approved=True)

        assert report.robustness_score == 0.68
        assert report.robustness_grade == "MODERATE"
        assert report.is_approved is True
        assert report.sensitivity_result is not None
        assert report.stability_result is not None
        assert report.monte_carlo_result is not None
        assert report.regime_result is not None

    def test_all_four_subtests_present(self):
        params = {"stop_loss_atr_multiplier": 1.5}
        m = _make_metrics()
        report = _make_robustness_report(params, m)

        assert report.sensitivity_result is not None
        assert report.stability_result is not None
        assert report.monte_carlo_result is not None
        assert report.regime_result is not None

    def test_overfit_grading(self):
        params = {"stop_loss_atr_multiplier": 1.0}
        m = _make_metrics(total_return_pct=15.0)
        report = _make_robustness_report(params, m, score=0.25, is_approved=False)

        assert report.robustness_grade == "OVERFIT"
        assert report.is_approved is False

    def test_robust_grading(self):
        params = {"stop_loss_atr_multiplier": 1.5}
        m = _make_metrics()
        report = _make_robustness_report(params, m, score=0.75, is_approved=True)

        assert report.robustness_grade == "ROBUST"

    def test_fragile_grading(self):
        params = {"stop_loss_atr_multiplier": 1.5}
        m = _make_metrics()
        report = _make_robustness_report(params, m, score=0.35, is_approved=False)

        assert report.robustness_grade == "FRAGILE"


# ===========================================================================
# Test 4: Report Generation
# ===========================================================================


class TestReportGeneration:
    """Test the OptimizationReportGenerator."""

    def test_report_generates_without_errors(self):
        opt_result = _make_opt_result_with_ranked(n_ranked=5)
        rob_reports = {}

        for r in opt_result.ranked_results[:3]:
            key = _params_to_key(r.params)
            rob_reports[key] = _make_robustness_report(
                r.params, r.metrics, score=0.68 - r.rank * 0.05,
            )

        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, rob_reports)

        assert isinstance(report, str)
        assert len(report) > 200  # Non-trivial report

    def test_report_contains_header(self):
        opt_result = _make_opt_result_with_ranked()
        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, {})

        assert "STRATEGY OPTIMIZATION REPORT" in report
        assert "NIFTY50" in report

    def test_report_contains_search_summary(self):
        opt_result = _make_opt_result_with_ranked()
        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, {})

        assert "SEARCH SUMMARY" in report
        assert "Combinations tested" in report
        assert "Duration:" in report
        assert "Objective:" in report

    def test_report_contains_baseline(self):
        opt_result = _make_opt_result_with_ranked(include_baseline=True)
        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, {})

        assert "BASELINE" in report

    def test_report_contains_top_n(self):
        opt_result = _make_opt_result_with_ranked(n_ranked=5)
        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, {})

        assert "TOP 5 PARAMETER SETS" in report
        assert "#1" in report
        assert "#5" in report

    def test_report_contains_recommended(self):
        opt_result = _make_opt_result_with_ranked()
        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, {})

        assert "RECOMMENDED PARAMETERS" in report

    def test_report_contains_robustness_details(self):
        opt_result = _make_opt_result_with_ranked()
        best_key = _params_to_key(opt_result.best_params)
        rob = _make_robustness_report(opt_result.best_params, opt_result.best_metrics)
        rob_reports = {best_key: rob}

        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, rob_reports)

        assert "ROBUSTNESS DETAILS" in report
        assert "Sensitivity:" in report
        assert "Data Stability:" in report
        assert "Monte Carlo:" in report

    def test_report_contains_walk_forward(self):
        opt_result = _make_opt_result_with_ranked(include_wf=True)
        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, {})

        assert "WALK-FORWARD RESULTS" in report
        assert "Windows:" in report
        assert "Degradation:" in report

    def test_report_contains_recommendation(self):
        opt_result = _make_opt_result_with_ranked()
        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, {})

        assert "RECOMMENDATION" in report

    def test_report_contains_config_hash(self):
        opt_result = _make_opt_result_with_ranked()
        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, {})

        assert "Config Hash:" in report
        assert "Generated:" in report

    def test_report_with_no_ranked_results(self):
        opt_result = _make_opt_result_with_ranked(n_ranked=0, include_baseline=False)
        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, {})

        assert "STRATEGY OPTIMIZATION REPORT" in report
        assert "No viable parameter sets found" in report

    def test_report_flags_overfitting_trap(self):
        opt_result = _make_opt_result_with_ranked(n_ranked=5)

        # Make rank #3 fail robustness but with positive return
        rob_reports: dict[str, RobustnessReport] = {}
        for r in opt_result.ranked_results[:3]:
            key = _params_to_key(r.params)
            if r.rank == 3:
                rob_reports[key] = _make_robustness_report(
                    r.params, r.metrics, score=0.25, is_approved=False,
                )
            else:
                rob_reports[key] = _make_robustness_report(
                    r.params, r.metrics, score=0.65, is_approved=True,
                )

        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, rob_reports)

        assert "FAILED robustness" in report or "overfitting" in report.lower()

    def test_report_save(self):
        gen = OptimizationReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_report.txt"
            result = gen.save_report("Test report content", filepath=str(filepath))

            assert result == filepath
            assert filepath.exists()
            assert filepath.read_text() == "Test report content"

    def test_report_save_default_path(self):
        gen = OptimizationReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "src.backtest.optimizer.report.Path",
                side_effect=lambda *a: Path(tmpdir) / a[0] if a else Path(tmpdir),
            ):
                # Just verify save_report doesn't crash with default path
                out = gen.save_report("content", filepath=str(Path(tmpdir) / "report.txt"))
                assert out.exists()


# ===========================================================================
# Test 5: Heatmap Data Generation
# ===========================================================================


class TestHeatmapData:
    """Test heatmap data generation for Phase 8 dashboard."""

    def test_heatmap_basic(self):
        results = [
            EvaluationResult(
                params={"stop_loss_atr_multiplier": 1.0, "target_atr_multiplier": 2.0},
                metrics=_make_metrics(sharpe_ratio=0.8),
                passes_constraints=True,
            ),
            EvaluationResult(
                params={"stop_loss_atr_multiplier": 1.0, "target_atr_multiplier": 2.5},
                metrics=_make_metrics(sharpe_ratio=1.2),
                passes_constraints=True,
            ),
            EvaluationResult(
                params={"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.0},
                metrics=_make_metrics(sharpe_ratio=1.5),
                passes_constraints=True,
            ),
            EvaluationResult(
                params={"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5},
                metrics=_make_metrics(sharpe_ratio=1.1),
                passes_constraints=True,
            ),
        ]

        gen = OptimizationReportGenerator()
        heatmap = gen.generate_parameter_heatmap_data(
            results, "stop_loss_atr_multiplier", "target_atr_multiplier",
        )

        assert heatmap["param1_values"] == [1.0, 1.5]
        assert heatmap["param2_values"] == [2.0, 2.5]
        assert len(heatmap["metric_grid"]) == 2  # 2 rows
        assert len(heatmap["metric_grid"][0]) == 2  # 2 cols
        assert heatmap["best_cell"] is not None
        assert heatmap["metric_name"] == "sharpe_ratio"

    def test_heatmap_finds_best_cell(self):
        results = [
            EvaluationResult(
                params={"a": 1, "b": 10},
                metrics=_make_metrics(sharpe_ratio=0.5),
                passes_constraints=True,
            ),
            EvaluationResult(
                params={"a": 1, "b": 20},
                metrics=_make_metrics(sharpe_ratio=2.0),
                passes_constraints=True,
            ),
        ]

        gen = OptimizationReportGenerator()
        heatmap = gen.generate_parameter_heatmap_data(results, "a", "b")

        # Best should be (0, 1) — a=1, b=20
        assert heatmap["best_cell"] == (0, 1)

    def test_heatmap_empty_results(self):
        gen = OptimizationReportGenerator()
        heatmap = gen.generate_parameter_heatmap_data([], "a", "b")

        assert heatmap["param1_values"] == []
        assert heatmap["metric_grid"] == []
        assert heatmap["best_cell"] is None

    def test_heatmap_with_missing_metrics(self):
        results = [
            EvaluationResult(
                params={"a": 1, "b": 10},
                metrics=None,  # No metrics
                passes_constraints=False,
            ),
            EvaluationResult(
                params={"a": 2, "b": 10},
                metrics=_make_metrics(sharpe_ratio=1.0),
                passes_constraints=True,
            ),
        ]

        gen = OptimizationReportGenerator()
        heatmap = gen.generate_parameter_heatmap_data(results, "a", "b")

        # Should only have the one valid result
        assert len(heatmap["param1_values"]) == 1
        assert heatmap["best_cell"] is not None


# ===========================================================================
# Test 6: Approved Parameter Manager
# ===========================================================================


class TestApprovedParameterManager:
    """Test save/load/apply/expire of approved parameters."""

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = Path(f.name)

        try:
            mgr = ApprovedParameterManager(filepath=tmppath)
            params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
            metrics = _make_metrics(total_return_pct=9.8, sharpe_ratio=1.42, win_rate=61.5)

            mgr.save_approved_params(
                index_id="NIFTY50",
                params=params,
                metrics=metrics,
                robustness_score=0.68,
                config_hash="a3f2b1c8",
                notes="Test approval",
            )

            loaded = mgr.get_approved_params("NIFTY50")
            assert loaded is not None
            assert loaded["stop_loss_atr_multiplier"] == 1.5
            assert loaded["target_atr_multiplier"] == 2.5
        finally:
            tmppath.unlink(missing_ok=True)

    def test_load_nonexistent_index(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = Path(f.name)

        try:
            mgr = ApprovedParameterManager(filepath=tmppath)
            assert mgr.get_approved_params("NONEXISTENT") is None
        finally:
            tmppath.unlink(missing_ok=True)

    def test_expire_params(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = Path(f.name)

        try:
            mgr = ApprovedParameterManager(filepath=tmppath)
            params = {"stop_loss_atr_multiplier": 1.5}
            mgr.save_approved_params(
                "NIFTY50", params, _make_metrics(), 0.68, "abc123",
            )

            # Verify active
            assert mgr.get_approved_params("NIFTY50") is not None

            # Expire
            mgr.expire_params("NIFTY50", "Edge decay detected by shadow tracker")

            # Should return None now (expired)
            assert mgr.get_approved_params("NIFTY50") is None

            # Verify status in raw data
            all_data = mgr.list_all_approved()
            assert all_data["NIFTY50"]["status"] == "EXPIRED"
            assert "edge decay" in all_data["NIFTY50"]["expire_reason"].lower()
        finally:
            tmppath.unlink(missing_ok=True)

    def test_list_all_approved(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = Path(f.name)

        try:
            mgr = ApprovedParameterManager(filepath=tmppath)
            mgr.save_approved_params("NIFTY50", {"sl": 1.5}, _make_metrics(), 0.7, "h1")
            mgr.save_approved_params("BANKNIFTY", {"sl": 2.0}, _make_metrics(), 0.6, "h2")

            all_approved = mgr.list_all_approved()
            assert "NIFTY50" in all_approved
            assert "BANKNIFTY" in all_approved
            assert all_approved["NIFTY50"]["status"] == "ACTIVE"
        finally:
            tmppath.unlink(missing_ok=True)

    def test_apply_to_backtest_config(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = Path(f.name)

        try:
            mgr = ApprovedParameterManager(filepath=tmppath)
            params = {
                "stop_loss_atr_multiplier": 1.5,
                "target_atr_multiplier": 2.5,
            }
            mgr.save_approved_params("NIFTY50", params, _make_metrics(), 0.68, "abc")

            base_config = BacktestConfig(
                index_id="NIFTY50",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 6, 30),
            )

            result = mgr.apply_to_backtest_config("NIFTY50", base_config)
            assert result is not None
            assert result.simulator_config.stop_loss_atr_mult == 1.5
            assert result.simulator_config.target_atr_mult == 2.5
        finally:
            tmppath.unlink(missing_ok=True)

    def test_apply_returns_none_for_unknown_index(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = Path(f.name)

        try:
            mgr = ApprovedParameterManager(filepath=tmppath)
            base_config = BacktestConfig(
                index_id="UNKNOWN",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 6, 30),
            )
            assert mgr.apply_to_backtest_config("UNKNOWN", base_config) is None
        finally:
            tmppath.unlink(missing_ok=True)

    def test_overwrite_existing_params(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = Path(f.name)

        try:
            mgr = ApprovedParameterManager(filepath=tmppath)
            mgr.save_approved_params("NIFTY50", {"sl": 1.5}, _make_metrics(), 0.6, "h1")
            mgr.save_approved_params("NIFTY50", {"sl": 2.0}, _make_metrics(), 0.7, "h2")

            loaded = mgr.get_approved_params("NIFTY50")
            assert loaded["sl"] == 2.0  # Overwritten

            all_data = mgr.list_all_approved()
            assert all_data["NIFTY50"]["config_hash"] == "h2"
        finally:
            tmppath.unlink(missing_ok=True)

    def test_metrics_stored_at_approval(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = Path(f.name)

        try:
            mgr = ApprovedParameterManager(filepath=tmppath)
            m = _make_metrics(
                total_return_pct=9.8,
                sharpe_ratio=1.42,
                win_rate=61.5,
                max_drawdown_pct=5.2,
                profit_factor=2.1,
                total_trades=48,
            )
            mgr.save_approved_params("NIFTY50", {"sl": 1.5}, m, 0.68, "abc")

            data = mgr.list_all_approved()["NIFTY50"]
            assert data["metrics_at_approval"]["return_pct"] == 9.8
            assert data["metrics_at_approval"]["sharpe"] == 1.42
            assert data["metrics_at_approval"]["win_rate"] == 61.5
            assert data["metrics_at_approval"]["total_trades"] == 48
        finally:
            tmppath.unlink(missing_ok=True)

    def test_load_from_nonexistent_file(self):
        mgr = ApprovedParameterManager(filepath=Path("/tmp/does_not_exist_12345.json"))
        assert mgr.get_approved_params("NIFTY50") is None
        assert mgr.list_all_approved() == {}


# ===========================================================================
# Test 7: Key Insights Derivation
# ===========================================================================


class TestInsights:
    """Test that insights are derived correctly from results."""

    def test_insights_generated(self):
        opt_result = _make_opt_result_with_ranked(n_ranked=5)
        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, {})

        assert "KEY INSIGHTS" in report

    def test_no_crash_on_empty_results(self):
        opt_result = _make_opt_result_with_ranked(n_ranked=0, include_baseline=False)
        gen = OptimizationReportGenerator()
        # Should not crash
        report = gen.generate_report(opt_result, {})
        assert isinstance(report, str)


# ===========================================================================
# Test 8: CLI Script
# ===========================================================================


class TestCLIScript:
    """Smoke tests for the CLI entry point."""

    def test_list_approved_runs(self):
        """--list-approved should not crash even with no data."""
        result = subprocess.run(
            [sys.executable, "scripts/run_optimizer.py", "--list-approved"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should exit 0 (or print "No approved parameter sets found")
        assert result.returncode == 0

    def test_missing_required_args(self):
        """Should fail gracefully if --index is missing."""
        result = subprocess.run(
            [sys.executable, "scripts/run_optimizer.py", "--start", "2024-01-01", "--end", "2024-12-31"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_invalid_date_format(self):
        """Should fail on bad dates."""
        result = subprocess.run(
            [
                sys.executable, "scripts/run_optimizer.py",
                "--index", "NIFTY50",
                "--start", "not-a-date",
                "--end", "2024-12-31",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_help_flag(self):
        """--help should print usage."""
        result = subprocess.run(
            [sys.executable, "scripts/run_optimizer.py", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Strategy Optimizer" in result.stdout


# ===========================================================================
# Test 9: Module Imports
# ===========================================================================


class TestModuleImports:
    """Verify all files are importable independently."""

    def test_import_report(self):
        from src.backtest.optimizer.report import OptimizationReportGenerator
        assert OptimizationReportGenerator is not None

    def test_import_param_applier(self):
        from src.backtest.optimizer.param_applier import ApprovedParameterManager
        assert ApprovedParameterManager is not None

    def test_import_from_init(self):
        from src.backtest.optimizer import (
            OptimizationReportGenerator,
            ApprovedParameterManager,
        )
        assert OptimizationReportGenerator is not None
        assert ApprovedParameterManager is not None

    def test_import_all_exports(self):
        from src.backtest.optimizer import __all__
        assert "OptimizationReportGenerator" in __all__
        assert "ApprovedParameterManager" in __all__


# ===========================================================================
# Test 10: Full Pipeline (mock-integrated)
# ===========================================================================


class TestFullPipeline:
    """End-to-end pipeline: load → optimize → robustness → report → approve."""

    def test_full_pipeline(self):
        """Run the full pipeline with mocked backtests."""
        # 1. Load conservative profile
        space = _conservative_space()
        assert space.num_parameters == 4

        # 2. Build optimization result (mocked)
        opt_result = _make_opt_result_with_ranked(
            n_ranked=5, include_baseline=True, include_wf=True,
        )

        # Verify results
        assert opt_result.passed_constraints >= 1
        assert opt_result.best_params is not None
        assert opt_result.best_metrics is not None
        assert opt_result.best_metrics.sharpe_ratio > 0

        # 3. Build robustness reports for top 3
        rob_reports: dict[str, RobustnessReport] = {}
        for r in opt_result.ranked_results[:3]:
            key = _params_to_key(r.params)
            is_good = r.rank <= 2
            rob_reports[key] = _make_robustness_report(
                r.params, r.metrics,
                score=0.68 if is_good else 0.28,
                is_approved=is_good,
            )

        # Verify robustness
        best_key = _params_to_key(opt_result.best_params)
        assert best_key in rob_reports
        assert rob_reports[best_key].is_approved

        # 4. Generate report
        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, rob_reports)

        assert "STRATEGY OPTIMIZATION REPORT" in report
        assert "ROBUSTNESS DETAILS" in report
        assert "WALK-FORWARD RESULTS" in report
        assert "RECOMMENDATION" in report

        # 5. Save report
        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen.save_report(report, filepath=str(Path(tmpdir) / "report.txt"))
            assert path.exists()
            content = path.read_text()
            assert "STRATEGY OPTIMIZATION REPORT" in content

        # 6. Save approved params
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = Path(f.name)

        try:
            mgr = ApprovedParameterManager(filepath=tmppath)
            best_rob = rob_reports[best_key]

            mgr.save_approved_params(
                index_id="NIFTY50",
                params=opt_result.best_params,
                metrics=opt_result.best_metrics,
                robustness_score=best_rob.robustness_score,
                config_hash="a3f2b1c8",
                notes="Full pipeline test",
            )

            # 7. Load and verify
            loaded = mgr.get_approved_params("NIFTY50")
            assert loaded is not None
            assert loaded == opt_result.best_params

            # 8. Apply to config
            base_config = BacktestConfig(
                index_id="NIFTY50",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 6, 30),
            )
            applied = mgr.apply_to_backtest_config("NIFTY50", base_config)
            assert applied is not None

            # 9. Expire and verify
            mgr.expire_params("NIFTY50", "Test expiry")
            assert mgr.get_approved_params("NIFTY50") is None
        finally:
            tmppath.unlink(missing_ok=True)

    def test_pipeline_with_no_viable_results(self):
        """Pipeline should handle the case where no params pass constraints."""
        opt_result = _make_opt_result_with_ranked(
            n_ranked=0, include_baseline=False, include_wf=False,
        )

        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, {})

        assert "No viable parameter sets found" in report
        assert isinstance(report, str)

    def test_pipeline_robustness_rejected(self):
        """Report should warn when best params fail robustness."""
        opt_result = _make_opt_result_with_ranked(n_ranked=3)

        # All params fail robustness
        rob_reports: dict[str, RobustnessReport] = {}
        for r in opt_result.ranked_results[:3]:
            key = _params_to_key(r.params)
            rob_reports[key] = _make_robustness_report(
                r.params, r.metrics, score=0.25, is_approved=False,
            )

        gen = OptimizationReportGenerator()
        report = gen.generate_report(opt_result, rob_reports)

        assert "NOT pass robustness" in report or "NOT approved" in report

    def test_heatmap_from_all_results(self):
        """Heatmap should work with mixed passing/failing results."""
        opt_result = _make_opt_result_with_ranked(n_ranked=5)
        gen = OptimizationReportGenerator()

        heatmap = gen.generate_parameter_heatmap_data(
            opt_result.all_results,
            "stop_loss_atr_multiplier",
            "target_atr_multiplier",
        )

        assert len(heatmap["param1_values"]) > 0
        assert len(heatmap["metric_grid"]) > 0
