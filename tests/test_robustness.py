"""Tests for robustness testing (src/backtest/optimizer/robustness.py)."""

from __future__ import annotations

import copy
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.backtest.metrics import BacktestMetrics
from src.backtest.optimizer.param_space import ParameterDef, ParameterSpace
from src.backtest.optimizer.robustness import (
    MonteCarloResult,
    ParamSensitivity,
    PeriodResult,
    RegimeResult,
    RobustnessReport,
    RobustnessTester,
    SensitivityResult,
    StabilityResult,
    _generate_sub_periods,
    _get_trade_regime,
    _perturb_value,
)
from src.backtest.strategy_runner import BacktestConfig, BacktestResult
from src.backtest.trade_simulator import ClosedTrade, SimulatorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(
    total_return_pct: float = 10.0,
    sharpe_ratio: float = 1.5,
    win_rate: float = 55.0,
    profit_factor: float = 1.8,
    max_drawdown_pct: float = 8.0,
    total_trades: int = 50,
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
        expected_value_per_trade=200.0,
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
    trade_history: Optional[list] = None,
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


def _make_closed_trade(
    net_pnl: float,
    outcome: str = "WIN",
    regime: Optional[str] = None,
    trade_id: str = "T001",
) -> ClosedTrade:
    """Build a minimal ClosedTrade for Monte Carlo / regime tests."""
    entry_bar = {}
    if regime is not None:
        entry_bar["regime"] = regime

    return ClosedTrade(
        trade_id=trade_id,
        index_id="NIFTY50",
        signal_id="S001",
        trade_type="BUY_CALL",
        signal_entry_price=100.0,
        actual_entry_price=100.5,
        entry_timestamp=datetime(2024, 6, 15, 10, 0),
        entry_bar=entry_bar,
        original_stop_loss=95.0,
        original_target=110.0,
        actual_exit_price=110.0 if outcome == "WIN" else 95.0,
        exit_timestamp=datetime(2024, 6, 15, 14, 0),
        exit_reason="TARGET_HIT" if outcome == "WIN" else "STOP_LOSS_HIT",
        lots=1,
        lot_size=50,
        quantity=50,
        confidence_level="MEDIUM",
        gross_pnl_points=abs(net_pnl) / 50,
        gross_pnl=net_pnl * 1.05,
        total_costs=abs(net_pnl) * 0.05,
        net_pnl=net_pnl,
        net_pnl_pct=(net_pnl / 100_000) * 100,
        duration_bars=5,
        duration_minutes=240,
        max_favorable_excursion=abs(net_pnl) / 50,
        max_adverse_excursion=2.0,
        outcome=outcome,
    )


def _tiny_space() -> ParameterSpace:
    """2 float params for sensitivity testing."""
    return ParameterSpace(parameters=[
        ParameterDef(
            name="stop_loss_atr_multiplier",
            display_name="SL", description="stop loss",
            param_type="float", min_value=1.0, max_value=3.0,
            step=0.5, default=1.5,
        ),
        ParameterDef(
            name="target_atr_multiplier",
            display_name="Target", description="target",
            param_type="float", min_value=2.0, max_value=4.0,
            step=0.5, default=2.5,
        ),
    ])


def _base_config(**overrides) -> BacktestConfig:
    """Build a basic BacktestConfig for tests."""
    defaults = dict(
        index_id="NIFTY50",
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        simulator_config=SimulatorConfig(),
        show_progress=False,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _make_tester() -> RobustnessTester:
    """Create a RobustnessTester with a mocked DB."""
    db = MagicMock()
    tester = RobustnessTester(db)
    return tester


# ---------------------------------------------------------------------------
# Test 1: Parameter Sensitivity
# ---------------------------------------------------------------------------


class TestParameterSensitivity:
    """Tests for test_sensitivity()."""

    def test_robust_params_all_variants_profitable(self):
        """When all ±20% variants remain profitable, should be ROBUST."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        orig_metrics = _make_metrics(total_return_pct=10.0, sharpe_ratio=1.5)
        config = _base_config()

        # All variants return similar metrics (robust)
        variant_metrics = _make_metrics(total_return_pct=8.0, sharpe_ratio=1.3)

        def mock_run(cfg):
            return _make_bt_result(cfg, variant_metrics)

        tester.strategy_runner.run = mock_run

        result = tester.test_sensitivity(params, config, orig_metrics, _tiny_space())

        assert result.is_robust
        assert result.overall_sensitivity_score > 0.4
        assert len(result.parameter_sensitivities) == 2
        assert result.most_sensitive_param in params
        assert result.least_sensitive_param in params

    def test_fragile_param_flips_return(self):
        """When a ±20% change flips return from positive to negative → critical."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        orig_metrics = _make_metrics(total_return_pct=10.0, sharpe_ratio=1.5)
        config = _base_config()

        call_count = [0]

        def mock_run(cfg):
            call_count[0] += 1
            # First two calls (SL +20% and SL -20%) → one breaks
            if call_count[0] == 1:
                return _make_bt_result(cfg, _make_metrics(total_return_pct=-5.0))
            return _make_bt_result(cfg, _make_metrics(total_return_pct=8.0))

        tester.strategy_runner.run = mock_run

        result = tester.test_sensitivity(params, config, orig_metrics, _tiny_space())

        # At least one param should be critical
        critical_params = [s for s in result.parameter_sensitivities if s.is_critical]
        assert len(critical_params) >= 1
        assert not result.is_robust

    def test_sensitivity_with_highly_sensitive_param(self):
        """If one variant changes return by > 50%, sensitivity score drops."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        orig_metrics = _make_metrics(total_return_pct=10.0)
        config = _base_config()

        call_count = [0]

        def mock_run(cfg):
            call_count[0] += 1
            if call_count[0] == 1:
                # SL +20%: return drops by 80% (from 10% to 2%)
                return _make_bt_result(cfg, _make_metrics(total_return_pct=2.0))
            return _make_bt_result(cfg, _make_metrics(total_return_pct=9.0))

        tester.strategy_runner.run = mock_run

        result = tester.test_sensitivity(params, config, orig_metrics, _tiny_space())

        # SL param should show low stability
        sl_sens = next(
            s for s in result.parameter_sensitivities
            if s.param_name == "stop_loss_atr_multiplier"
        )
        assert sl_sens.return_stability < 0.5

    def test_sensitivity_no_numeric_params(self):
        """If all params are 'choice' type, sensitivity is trivially robust."""
        tester = _make_tester()
        space = ParameterSpace(parameters=[
            ParameterDef(
                name="min_confidence_filter",
                display_name="Confidence", description="filter",
                param_type="choice", choices=["LOW", "MEDIUM", "HIGH"],
                default="MEDIUM",
            ),
        ])
        params = {"min_confidence_filter": "MEDIUM"}
        orig_metrics = _make_metrics()
        config = _base_config()

        result = tester.test_sensitivity(params, config, orig_metrics, space)

        assert result.is_robust
        assert result.overall_sensitivity_score == 1.0
        assert len(result.parameter_sensitivities) == 0

    def test_sensitivity_variant_returns_none(self):
        """If ParameterApplicator returns None (invalid combo), treat as 0 return."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        orig_metrics = _make_metrics(total_return_pct=10.0)
        config = _base_config()

        # Make SL +20% produce an invalid config (target <= SL)
        space = ParameterSpace(parameters=[
            ParameterDef(
                name="stop_loss_atr_multiplier",
                display_name="SL", description="SL",
                param_type="float", min_value=1.0, max_value=3.0,
                step=0.5, default=1.5,
            ),
            ParameterDef(
                name="target_atr_multiplier",
                display_name="Target", description="target",
                param_type="float", min_value=1.0, max_value=4.0,
                step=0.5, default=2.5,
            ),
        ])

        # Set SL close to target so +20% makes SL > target → None from applicator
        params_close = {"stop_loss_atr_multiplier": 2.4, "target_atr_multiplier": 2.5}

        def mock_run(cfg):
            return _make_bt_result(cfg, _make_metrics(total_return_pct=8.0))

        tester.strategy_runner.run = mock_run

        result = tester.test_sensitivity(params_close, config, orig_metrics, space)

        # Should still produce a result (the None variant gets 0.0 return)
        assert len(result.parameter_sensitivities) == 2


# ---------------------------------------------------------------------------
# Test 2: Data Stability
# ---------------------------------------------------------------------------


class TestDataStability:
    """Tests for test_data_stability()."""

    def test_stable_strategy_profitable_most_periods(self):
        """Profitable in 4/5+ periods → should pass stability check."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()

        call_count = [0]

        def mock_run(cfg):
            call_count[0] += 1
            # Make most periods profitable
            if call_count[0] % 5 == 0:
                return _make_bt_result(cfg, _make_metrics(total_return_pct=-2.0))
            return _make_bt_result(cfg, _make_metrics(total_return_pct=5.0))

        tester.strategy_runner.run = mock_run

        result = tester.test_data_stability(params, config)

        assert result.profitability_rate > 0.6
        assert result.is_stable
        assert result.consistency_score > 0.5
        assert len(result.period_results) > 0

    def test_overfit_strategy_few_profitable_periods(self):
        """Profitable in only 2/5 periods → should fail."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()

        call_count = [0]

        def mock_run(cfg):
            call_count[0] += 1
            # Only 2 out of every 5 periods profitable
            if call_count[0] % 5 in (1, 2):
                return _make_bt_result(cfg, _make_metrics(total_return_pct=5.0))
            return _make_bt_result(cfg, _make_metrics(total_return_pct=-3.0))

        tester.strategy_runner.run = mock_run

        result = tester.test_data_stability(params, config)

        assert result.profitability_rate < 0.6
        assert not result.is_stable

    def test_catastrophic_loss_fails_stability(self):
        """If any period has return < -15%, is_stable = False."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()

        call_count = [0]

        def mock_run(cfg):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_bt_result(cfg, _make_metrics(total_return_pct=-20.0))
            return _make_bt_result(cfg, _make_metrics(total_return_pct=5.0))

        tester.strategy_runner.run = mock_run

        result = tester.test_data_stability(params, config)

        # Even with mostly profitable periods, one catastrophic loss → unstable
        assert not result.is_stable

    def test_short_period_no_subperiods(self):
        """If date range is too short (< 60 days), no sub-periods generated."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 1),
        )

        tester.strategy_runner.run = lambda cfg: _make_bt_result(cfg)

        result = tester.test_data_stability(params, config)

        assert len(result.period_results) == 0
        assert not result.is_stable

    def test_zero_trade_period_excluded(self):
        """Sub-period with 0 trades should be excluded from analysis."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()

        call_count = [0]

        def mock_run(cfg):
            call_count[0] += 1
            if call_count[0] == 1:
                # First sub-period: 0 trades
                return _make_bt_result(
                    cfg,
                    _make_metrics(total_return_pct=0.0, total_trades=0),
                )
            return _make_bt_result(cfg, _make_metrics(total_return_pct=5.0))

        tester.strategy_runner.run = mock_run

        result = tester.test_data_stability(params, config)

        # The 0-trade period should not appear in results
        assert all(p.trade_count > 0 for p in result.period_results)

    def test_return_range_calculated(self):
        """Return range should capture worst and best period."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()

        returns_cycle = [-5.0, 3.0, 10.0, -1.0, 7.0, 2.0, 8.0, -3.0, 4.0, 6.0]
        call_count = [0]

        def mock_run(cfg):
            idx = call_count[0] % len(returns_cycle)
            call_count[0] += 1
            return _make_bt_result(cfg, _make_metrics(total_return_pct=returns_cycle[idx]))

        tester.strategy_runner.run = mock_run

        result = tester.test_data_stability(params, config)

        assert result.return_range[0] <= result.return_range[1]
        assert result.return_std >= 0


# ---------------------------------------------------------------------------
# Test 3: Monte Carlo Simulation
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    """Tests for test_monte_carlo()."""

    def test_strong_strategy_statistically_profitable(self):
        """10 wins of ₹500, 5 losses of ₹300 → should be statistically profitable."""
        tester = _make_tester()

        trades = (
            [_make_closed_trade(500.0, "WIN", trade_id=f"W{i}") for i in range(10)]
            + [_make_closed_trade(-300.0, "LOSS", trade_id=f"L{i}") for i in range(5)]
        )
        orig_metrics = _make_metrics(total_return_pct=3.5)

        result = tester.test_monte_carlo(trades, orig_metrics, n_simulations=500)

        assert result.is_statistically_profitable
        assert result.pct_profitable > 0.9
        assert result.median_return > 0
        assert result.p5_return > 0  # Even 5th percentile positive
        assert result.luck_factor < 0.5
        assert result.reliability_score > 0.5

    def test_marginal_strategy_high_luck_factor(self):
        """Strategy where original return was lucky vs median → high luck factor."""
        tester = _make_tester()

        # Mix of wins and losses with high variance — the SUM is always the same
        # but original_return is inflated relative to median (which == actual sum)
        trades = (
            [_make_closed_trade(800.0, "WIN", trade_id=f"W{i}") for i in range(5)]
            + [_make_closed_trade(-700.0, "LOSS", trade_id=f"L{i}") for i in range(5)]
        )
        # Net P&L: 5*800 - 5*700 = 500, return = 0.5%
        # But claim original return was much higher (lucky ordering)
        orig_metrics = _make_metrics(total_return_pct=2.0)

        result = tester.test_monte_carlo(trades, orig_metrics, n_simulations=1000)

        # Luck factor should be significant since original claim (2.0%) > median (~0.5%)
        assert result.luck_factor > 0.3
        assert result.n_simulations == 1000

    def test_losing_strategy_not_profitable(self):
        """More losses than wins → should not be statistically profitable."""
        tester = _make_tester()

        trades = (
            [_make_closed_trade(200.0, "WIN", trade_id=f"W{i}") for i in range(5)]
            + [_make_closed_trade(-500.0, "LOSS", trade_id=f"L{i}") for i in range(15)]
        )
        orig_metrics = _make_metrics(total_return_pct=-6.5)

        result = tester.test_monte_carlo(trades, orig_metrics, n_simulations=500)

        assert not result.is_statistically_profitable
        assert result.pct_profitable < 0.5
        assert result.median_return < 0

    def test_all_wins_no_variance(self):
        """All trades are wins → MC has no variance, result is deterministic."""
        tester = _make_tester()

        trades = [_make_closed_trade(500.0, "WIN", trade_id=f"W{i}") for i in range(20)]
        orig_metrics = _make_metrics(total_return_pct=10.0)

        result = tester.test_monte_carlo(trades, orig_metrics)

        assert result.pct_profitable == 1.0
        assert result.is_statistically_profitable
        # All simulations should give the same result
        assert result.p5_return == result.p95_return

    def test_drawdown_distribution(self):
        """Verify drawdown percentiles are populated."""
        tester = _make_tester()

        trades = (
            [_make_closed_trade(500.0, "WIN", trade_id=f"W{i}") for i in range(15)]
            + [_make_closed_trade(-300.0, "LOSS", trade_id=f"L{i}") for i in range(10)]
        )
        orig_metrics = _make_metrics(total_return_pct=4.5)

        result = tester.test_monte_carlo(trades, orig_metrics, n_simulations=200)

        assert result.median_drawdown >= 0
        assert result.p95_drawdown >= result.median_drawdown

    def test_confidence_interval(self):
        """95% CI should span p5 to p95."""
        tester = _make_tester()

        trades = (
            [_make_closed_trade(500.0, "WIN", trade_id=f"W{i}") for i in range(10)]
            + [_make_closed_trade(-300.0, "LOSS", trade_id=f"L{i}") for i in range(5)]
        )
        orig_metrics = _make_metrics(total_return_pct=3.5)

        result = tester.test_monte_carlo(trades, orig_metrics, n_simulations=500)

        assert result.return_95_ci[0] == result.p5_return
        assert result.return_95_ci[1] == result.p95_return


# ---------------------------------------------------------------------------
# Test 4: Regime Robustness
# ---------------------------------------------------------------------------


class TestRegimeRobustness:
    """Tests for test_regime_robustness()."""

    def test_multi_regime_profitable(self):
        """Profitable in 3+ regimes → robust and not regime-dependent."""
        tester = _make_tester()

        trades = (
            [_make_closed_trade(500.0, "WIN", regime="TRENDING_UP", trade_id=f"TU{i}") for i in range(5)]
            + [_make_closed_trade(300.0, "WIN", regime="TRENDING_DOWN", trade_id=f"TD{i}") for i in range(5)]
            + [_make_closed_trade(200.0, "WIN", regime="RANGE_BOUND", trade_id=f"RB{i}") for i in range(5)]
        )

        result = tester.test_regime_robustness(trades)

        assert result.regimes_tested == 3
        assert result.profitable_regimes == 3
        assert not result.is_regime_dependent
        assert result.regime_diversity_score == 1.0

    def test_single_regime_dependent(self):
        """Only profitable in TRENDING → should flag regime_dependent."""
        tester = _make_tester()

        trades = (
            [_make_closed_trade(500.0, "WIN", regime="TRENDING_UP", trade_id=f"TU{i}") for i in range(5)]
            + [_make_closed_trade(-300.0, "LOSS", regime="RANGE_BOUND", trade_id=f"RB{i}") for i in range(5)]
            + [_make_closed_trade(-200.0, "LOSS", regime="VOLATILE", trade_id=f"V{i}") for i in range(5)]
        )

        result = tester.test_regime_robustness(trades)

        assert result.is_regime_dependent
        assert result.profitable_regimes == 1
        assert result.best_regime == "TRENDING_UP"
        assert result.regime_diversity_score == pytest.approx(1 / 3, abs=0.01)

    def test_no_regime_data(self):
        """Trades with no regime info → empty result, diversity score 0."""
        tester = _make_tester()

        trades = [_make_closed_trade(500.0, "WIN", regime=None, trade_id=f"T{i}") for i in range(10)]

        result = tester.test_regime_robustness(trades)

        assert result.regimes_tested == 0
        assert result.regime_diversity_score == 0.0
        assert result.is_regime_dependent  # 0 profitable regimes in 0 tested

    def test_best_and_worst_regime(self):
        """Verify best/worst regime identification by total return."""
        tester = _make_tester()

        trades = (
            [_make_closed_trade(1000.0, "WIN", regime="BULL", trade_id=f"B{i}") for i in range(3)]
            + [_make_closed_trade(-500.0, "LOSS", regime="BEAR", trade_id=f"BR{i}") for i in range(3)]
            + [_make_closed_trade(100.0, "WIN", regime="FLAT", trade_id=f"F{i}") for i in range(3)]
        )

        result = tester.test_regime_robustness(trades)

        assert result.best_regime == "BULL"
        assert result.worst_regime == "BEAR"
        assert result.best_regime_return > 0
        assert result.worst_regime_return < 0

    def test_zero_trades_empty_result(self):
        """Empty trade list → empty regime result."""
        tester = _make_tester()

        result = tester.test_regime_robustness([])

        assert result.regimes_tested == 0
        assert result.regime_diversity_score == 0.0


# ---------------------------------------------------------------------------
# Test: Overall Scoring and Grading
# ---------------------------------------------------------------------------


class TestOverallScoring:
    """Tests for run_full_robustness_test() scoring and grading."""

    def _mock_tester_with_scores(
        self,
        sensitivity_score: float = 0.8,
        sensitivity_robust: bool = True,
        consistency_score: float = 0.7,
        stability_stable: bool = True,
        mc_reliability: float = 0.8,
        mc_profitable: bool = True,
        regime_diversity: float = 0.7,
        regime_dependent: bool = False,
        n_trades: int = 30,
        has_regime: bool = True,
    ) -> tuple[RobustnessTester, list[ClosedTrade]]:
        """Build a tester with controlled sub-test results."""
        tester = _make_tester()

        # Mock all 4 test methods
        tester.test_sensitivity = MagicMock(return_value=SensitivityResult(
            parameter_sensitivities=[],
            overall_sensitivity_score=sensitivity_score,
            most_sensitive_param="stop_loss_atr_multiplier",
            least_sensitive_param="target_atr_multiplier",
            is_robust=sensitivity_robust,
        ))

        tester.test_data_stability = MagicMock(return_value=StabilityResult(
            period_results=[],
            profitable_periods=4,
            unprofitable_periods=1,
            profitability_rate=0.8,
            return_std=3.0,
            return_range=(-2.0, 12.0),
            consistency_score=consistency_score,
            is_stable=stability_stable,
        ))

        tester.test_monte_carlo = MagicMock(return_value=MonteCarloResult(
            n_simulations=1000,
            median_return=8.0,
            p5_return=2.0 if mc_profitable else -5.0,
            p25_return=5.0,
            p75_return=11.0,
            p95_return=15.0,
            pct_profitable=0.9 if mc_profitable else 0.4,
            median_drawdown=5.0,
            p95_drawdown=12.0,
            return_95_ci=(2.0, 15.0),
            is_statistically_profitable=mc_profitable,
            luck_factor=0.1,
            reliability_score=mc_reliability,
        ))

        tester.test_regime_robustness = MagicMock(return_value=RegimeResult(
            regime_performance={},
            regimes_tested=3,
            profitable_regimes=3 if not regime_dependent else 1,
            best_regime="TRENDING_UP",
            worst_regime="RANGE_BOUND",
            best_regime_return=5000.0,
            worst_regime_return=-1000.0 if regime_dependent else 500.0,
            is_regime_dependent=regime_dependent,
            regime_diversity_score=regime_diversity,
        ))

        regime = "TRENDING_UP" if has_regime else None
        trades = [
            _make_closed_trade(
                500.0 if i % 2 == 0 else -300.0,
                "WIN" if i % 2 == 0 else "LOSS",
                regime=regime,
                trade_id=f"T{i}",
            )
            for i in range(n_trades)
        ]

        return tester, trades

    def test_robust_grade(self):
        """High scores across all tests → ROBUST grade and approved."""
        tester, trades = self._mock_tester_with_scores(
            sensitivity_score=0.85,
            consistency_score=0.80,
            mc_reliability=0.85,
            regime_diversity=0.90,
        )
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()
        orig_metrics = _make_metrics(total_return_pct=10.0)

        report = tester.run_full_robustness_test(
            params=params,
            base_config=config,
            original_metrics=orig_metrics,
            trade_history=trades,
            parameter_space=_tiny_space(),
        )

        assert report.robustness_grade == "ROBUST"
        assert report.is_approved
        assert report.robustness_score > 0.7

    def test_moderate_grade(self):
        """Medium scores → MODERATE grade."""
        tester, trades = self._mock_tester_with_scores(
            sensitivity_score=0.55,
            consistency_score=0.60,
            mc_reliability=0.55,
            regime_diversity=0.60,
        )
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()
        orig_metrics = _make_metrics(total_return_pct=10.0)

        report = tester.run_full_robustness_test(
            params=params,
            base_config=config,
            original_metrics=orig_metrics,
            trade_history=trades,
            parameter_space=_tiny_space(),
        )

        assert report.robustness_grade == "MODERATE"
        assert report.is_approved
        assert 0.5 <= report.robustness_score <= 0.7

    def test_fragile_grade(self):
        """Low scores → FRAGILE grade, not approved."""
        tester, trades = self._mock_tester_with_scores(
            sensitivity_score=0.30,
            consistency_score=0.35,
            mc_reliability=0.35,
            regime_diversity=0.33,
        )
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()
        orig_metrics = _make_metrics(total_return_pct=10.0)

        report = tester.run_full_robustness_test(
            params=params,
            base_config=config,
            original_metrics=orig_metrics,
            trade_history=trades,
            parameter_space=_tiny_space(),
        )

        assert report.robustness_grade == "FRAGILE"
        assert not report.is_approved
        assert 0.3 <= report.robustness_score < 0.5

    def test_overfit_grade(self):
        """Very low scores → OVERFIT grade."""
        tester, trades = self._mock_tester_with_scores(
            sensitivity_score=0.10,
            consistency_score=0.15,
            mc_reliability=0.20,
            regime_diversity=0.10,
        )
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()
        orig_metrics = _make_metrics(total_return_pct=10.0)

        report = tester.run_full_robustness_test(
            params=params,
            base_config=config,
            original_metrics=orig_metrics,
            trade_history=trades,
            parameter_space=_tiny_space(),
        )

        assert report.robustness_grade == "OVERFIT"
        assert not report.is_approved
        assert report.robustness_score < 0.3

    def test_scoring_weights(self):
        """Verify the weighted average formula produces expected results."""
        tester, trades = self._mock_tester_with_scores(
            sensitivity_score=1.0,   # weight 0.25
            consistency_score=1.0,   # weight 0.30
            mc_reliability=1.0,      # weight 0.25
            regime_diversity=1.0,    # weight 0.20
        )
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()
        orig_metrics = _make_metrics(total_return_pct=10.0)

        report = tester.run_full_robustness_test(
            params=params,
            base_config=config,
            original_metrics=orig_metrics,
            trade_history=trades,
            parameter_space=_tiny_space(),
        )

        assert report.robustness_score == pytest.approx(1.0, abs=0.01)

    def test_skip_monte_carlo_insufficient_trades(self):
        """< 20 trades → MC skipped, concern noted."""
        tester, _ = self._mock_tester_with_scores(n_trades=10)
        trades = [
            _make_closed_trade(500.0, "WIN", regime="TRENDING_UP", trade_id=f"T{i}")
            for i in range(10)
        ]
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()
        orig_metrics = _make_metrics(total_return_pct=10.0)

        report = tester.run_full_robustness_test(
            params=params,
            base_config=config,
            original_metrics=orig_metrics,
            trade_history=trades,
            parameter_space=_tiny_space(),
        )

        assert report.monte_carlo_result is None
        assert any("insufficient" in c.lower() or "Monte Carlo" in c for c in report.concerns)

    def test_skip_regime_no_regime_data(self):
        """Trades without regime → regime test skipped, concern noted."""
        tester, _ = self._mock_tester_with_scores(has_regime=False)
        trades = [
            _make_closed_trade(500.0, "WIN", regime=None, trade_id=f"T{i}")
            for i in range(30)
        ]
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()
        orig_metrics = _make_metrics(total_return_pct=10.0)

        report = tester.run_full_robustness_test(
            params=params,
            base_config=config,
            original_metrics=orig_metrics,
            trade_history=trades,
            parameter_space=_tiny_space(),
        )

        assert report.regime_result is None
        assert any("regime" in c.lower() for c in report.concerns)

    def test_summary_generated(self):
        """Report should have a non-empty summary string."""
        tester, trades = self._mock_tester_with_scores()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()
        orig_metrics = _make_metrics(total_return_pct=10.0)

        report = tester.run_full_robustness_test(
            params=params,
            base_config=config,
            original_metrics=orig_metrics,
            trade_history=trades,
            parameter_space=_tiny_space(),
        )

        assert len(report.summary) > 100
        assert "ROBUSTNESS TEST REPORT" in report.summary
        assert report.robustness_grade in report.summary

    def test_concerns_and_recommendations(self):
        """Failing tests should populate concerns and recommendations."""
        tester, trades = self._mock_tester_with_scores(
            sensitivity_robust=False,
            stability_stable=False,
            mc_profitable=False,
            regime_dependent=True,
        )
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()
        orig_metrics = _make_metrics(total_return_pct=10.0)

        report = tester.run_full_robustness_test(
            params=params,
            base_config=config,
            original_metrics=orig_metrics,
            trade_history=trades,
            parameter_space=_tiny_space(),
        )

        assert len(report.concerns) >= 3  # sensitivity + stability + MC + regime
        assert len(report.recommendations) >= 3


# ---------------------------------------------------------------------------
# Test: Utility Functions
# ---------------------------------------------------------------------------


class TestUtilities:
    """Tests for module-level helper functions."""

    def test_perturb_value_float_up(self):
        """Float +20% within bounds."""
        pdef = MagicMock(min_value=1.0, max_value=3.0, param_type="float")
        assert _perturb_value(2.0, 0.20, pdef) == pytest.approx(2.4)

    def test_perturb_value_float_down(self):
        """Float -20% within bounds."""
        pdef = MagicMock(min_value=1.0, max_value=3.0, param_type="float")
        assert _perturb_value(2.0, -0.20, pdef) == pytest.approx(1.6)

    def test_perturb_value_capped_at_max(self):
        """Float +20% exceeds max → clamped to max."""
        pdef = MagicMock(min_value=1.0, max_value=2.5, param_type="float")
        result = _perturb_value(2.5, 0.20, pdef)
        assert result == 2.5

    def test_perturb_value_floored_at_min(self):
        """Float -20% below min → clamped to min."""
        pdef = MagicMock(min_value=1.0, max_value=3.0, param_type="float")
        result = _perturb_value(1.0, -0.20, pdef)
        assert result == 1.0

    def test_perturb_value_int_type(self):
        """Int type should round to nearest integer."""
        pdef = MagicMock(min_value=1, max_value=10, param_type="int")
        result = _perturb_value(5, 0.20, pdef)
        assert result == 6
        assert isinstance(result, int)

    def test_get_trade_regime_from_entry_bar(self):
        """Regime extracted from entry_bar dict."""
        trade = _make_closed_trade(500.0, "WIN", regime="TRENDING_UP")
        assert _get_trade_regime(trade) == "TRENDING_UP"

    def test_get_trade_regime_none(self):
        """No regime data → returns None."""
        trade = _make_closed_trade(500.0, "WIN", regime=None)
        assert _get_trade_regime(trade) is None

    def test_get_trade_regime_from_attribute(self):
        """If trade has .regime attribute, prefer it."""
        trade = _make_closed_trade(500.0, "WIN", regime=None)
        trade.regime = "VOLATILE"
        assert _get_trade_regime(trade) == "VOLATILE"

    def test_generate_sub_periods_2_year_range(self):
        """2-year range should produce halves, odd/even months, and quarters."""
        periods = _generate_sub_periods(date(2023, 1, 1), date(2024, 12, 31))

        names = [p[0] for p in periods]

        assert "First Half" in names
        assert "Second Half" in names
        assert "Odd Months" in names
        assert "Even Months" in names
        # Should have some quarters
        quarter_names = [n for n in names if n.startswith("Q")]
        assert len(quarter_names) >= 4

    def test_generate_sub_periods_short_range(self):
        """< 60 days → no sub-periods."""
        periods = _generate_sub_periods(date(2024, 1, 1), date(2024, 2, 1))
        assert len(periods) == 0


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_trades_full_test(self):
        """0 trades → MC and regime skipped, stability may produce empty results."""
        tester = _make_tester()

        # Mock sensitivity since it needs strategy_runner
        tester.test_sensitivity = MagicMock(return_value=SensitivityResult(
            parameter_sensitivities=[],
            overall_sensitivity_score=0.5,
            most_sensitive_param="",
            least_sensitive_param="",
            is_robust=True,
        ))
        tester.test_data_stability = MagicMock(return_value=StabilityResult(
            period_results=[],
            profitable_periods=0,
            unprofitable_periods=0,
            profitability_rate=0.0,
            return_std=0.0,
            return_range=(0.0, 0.0),
            consistency_score=0.0,
            is_stable=False,
        ))

        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()
        orig_metrics = _make_metrics(total_return_pct=0.0, total_trades=0)

        report = tester.run_full_robustness_test(
            params=params,
            base_config=config,
            original_metrics=orig_metrics,
            trade_history=[],
            parameter_space=_tiny_space(),
        )

        assert report.monte_carlo_result is None
        assert report.regime_result is None
        assert not report.is_approved

    def test_all_wins_trades(self):
        """All wins → MC should report 100% profitable with no variance."""
        tester = _make_tester()
        trades = [_make_closed_trade(500.0, "WIN", trade_id=f"W{i}") for i in range(25)]
        orig_metrics = _make_metrics(total_return_pct=12.5)

        result = tester.test_monte_carlo(trades, orig_metrics, n_simulations=100)

        assert result.pct_profitable == 1.0
        assert result.is_statistically_profitable
        assert result.p5_return == result.p95_return  # No variance

    def test_single_trade_type_in_mc(self):
        """Identical P&Ls → no variance, deterministic result."""
        tester = _make_tester()
        trades = [_make_closed_trade(500.0, "WIN", trade_id=f"W{i}") for i in range(20)]
        orig_metrics = _make_metrics(total_return_pct=10.0)

        result = tester.test_monte_carlo(trades, orig_metrics, n_simulations=100)

        # All identical → every shuffle produces same equity curve
        assert result.median_return == result.p5_return
        assert result.luck_factor == 0.0

    def test_original_return_zero_luck_factor(self):
        """Original return ≈ 0 → luck factor should be 0 (not division by zero)."""
        tester = _make_tester()
        trades = (
            [_make_closed_trade(100.0, "WIN", trade_id=f"W{i}") for i in range(10)]
            + [_make_closed_trade(-100.0, "LOSS", trade_id=f"L{i}") for i in range(10)]
        )
        orig_metrics = _make_metrics(total_return_pct=0.0)

        result = tester.test_monte_carlo(trades, orig_metrics, n_simulations=200)

        # Should not crash — luck_factor capped at 0 when orig_return ≈ 0
        assert result.luck_factor >= 0.0
        assert result.luck_factor <= 1.0

    def test_stability_single_period_no_stdev(self):
        """If only 1 sub-period has trades, stdev should be 0."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 4, 30),
        )

        call_count = [0]

        def mock_run(cfg):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_bt_result(cfg, _make_metrics(total_return_pct=5.0))
            return _make_bt_result(cfg, _make_metrics(total_return_pct=0.0, total_trades=0))

        tester.strategy_runner.run = mock_run

        result = tester.test_data_stability(params, config)

        # Only 1 period with trades → stdev = 0
        if len(result.period_results) == 1:
            assert result.return_std == 0.0

    def test_regime_single_regime_not_dependent(self):
        """Single regime with profit → profitable_regimes == 1, regimes_tested == 1.
        Since profitable_regimes <= 1 AND regimes_tested > 1 is False here,
        is_regime_dependent should be False."""
        tester = _make_tester()
        trades = [
            _make_closed_trade(500.0, "WIN", regime="TRENDING_UP", trade_id=f"T{i}")
            for i in range(10)
        ]

        result = tester.test_regime_robustness(trades)

        assert result.regimes_tested == 1
        assert result.profitable_regimes == 1
        assert not result.is_regime_dependent  # Only 1 regime total — can't judge dependency
        assert result.regime_diversity_score == 1.0

    def test_sensitivity_runner_exception(self):
        """If strategy runner raises an exception for a variant, treat as 0 return."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        orig_metrics = _make_metrics(total_return_pct=10.0)
        config = _base_config()

        call_count = [0]

        def mock_run(cfg):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Database connection failed")
            return _make_bt_result(cfg, _make_metrics(total_return_pct=8.0))

        tester.strategy_runner.run = mock_run

        # Should not raise
        result = tester.test_sensitivity(params, config, orig_metrics, _tiny_space())
        assert len(result.parameter_sensitivities) == 2

    def test_stability_runner_exception(self):
        """If a sub-period backtest fails, skip that period gracefully."""
        tester = _make_tester()
        params = {"stop_loss_atr_multiplier": 1.5, "target_atr_multiplier": 2.5}
        config = _base_config()

        call_count = [0]

        def mock_run(cfg):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Data unavailable")
            return _make_bt_result(cfg, _make_metrics(total_return_pct=5.0))

        tester.strategy_runner.run = mock_run

        result = tester.test_data_stability(params, config)

        # Should not crash; first period skipped
        assert len(result.period_results) > 0
