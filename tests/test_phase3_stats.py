"""Phase 3B statistical-correctness tests: metrics, bootstrap MC, optimiser wiring."""

from __future__ import annotations

import dataclasses
import math
import uuid
from datetime import date, datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.backtest.metrics import (
    MetricsCalculator,
    annualised_sharpe,
    annualised_sortino,
    daily_returns_from_equity,
    daily_returns_from_trades,
    is_defined,
)
from src.backtest.optimizer.optimization_engine import OptimizationConfig, OptimizationEngine
from src.backtest.optimizer.param_space import ParameterDef, ParameterSpace
from src.backtest.optimizer.robustness import (
    RobustnessTester,
    bootstrap_returns,
    circular_block_bootstrap_indices,
    daily_pnl_from_trades,
    stationary_bootstrap_indices,
)
from src.backtest.strategy_runner import BacktestResult
from src.backtest.trade_simulator import ClosedTrade, EquityPoint
from src.backtest.walk_forward import WalkForwardValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trade(net_pnl: float, exit_ts: datetime, outcome: Optional[str] = None) -> ClosedTrade:
    outcome = outcome or ("WIN" if net_pnl > 0 else "LOSS")
    return ClosedTrade(
        trade_id=str(uuid.uuid4()), index_id="NIFTY50", signal_id="S",
        trade_type="BUY_CALL", signal_entry_price=100.0, actual_entry_price=100.0,
        entry_timestamp=exit_ts - timedelta(hours=2), entry_bar={"regime": "TRENDING"},
        original_stop_loss=95.0, original_target=110.0, actual_exit_price=105.0,
        exit_timestamp=exit_ts, exit_reason="TARGET_HIT" if outcome == "WIN" else "STOP_LOSS_HIT",
        lots=1, lot_size=50, quantity=50, confidence_level="MEDIUM",
        gross_pnl_points=1.0, gross_pnl=net_pnl, total_costs=0.0, net_pnl=net_pnl,
        net_pnl_pct=net_pnl / 1000.0, duration_bars=3, duration_minutes=120,
        max_favorable_excursion=1.0, max_adverse_excursion=1.0, outcome=outcome,
    )


def _pt(ts: datetime, cap: float) -> EquityPoint:
    return EquityPoint(ts, cap, cap, 0.0, 0.0, 0)


def _weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# 1. Metrics: daily-aggregated Sharpe, zero days, no sentinels, composite
# ---------------------------------------------------------------------------


class TestDailySharpe:
    def test_intraday_bars_aggregated_to_daily(self):
        rng = np.random.default_rng(0)
        days = _weekdays(date(2024, 1, 1), 40)
        pts, cap, closes = [], 100_000.0, []
        for d in days:
            for k in range(75):                       # 5-minute bars
                cap *= 1 + rng.normal(0.00005, 0.001)
                pts.append(_pt(datetime(d.year, d.month, d.day, 9, 15) + timedelta(minutes=5 * k), cap))
            closes.append(cap)
        m = MetricsCalculator.calculate_all([_trade(1.0, pts[-1].timestamp)], pts, 100_000.0, 0.065)

        first = pts[0].capital
        expected_daily = [closes[0] / first - 1] + [b / a - 1 for a, b in zip(closes, closes[1:])]
        assert len(m.daily_returns) == len(days)
        np.testing.assert_allclose(m.daily_returns, expected_daily, rtol=1e-12)
        assert m.sharpe_ratio == pytest.approx(annualised_sharpe(expected_daily, 0.065), abs=1e-4)

        # The old per-bar computation with sqrt(252) gives a materially different number
        bar = [b.capital / a.capital - 1 for a, b in zip(pts, pts[1:])]
        per_bar_wrong = (np.mean(bar) * 252 - 0.065) / (np.std(bar, ddof=1) * math.sqrt(252))
        assert abs(per_bar_wrong - m.sharpe_ratio) > 0.5

    def test_zero_pnl_days_included(self):
        pts = [
            _pt(datetime(2024, 1, 1, 15, 30), 100_000.0),   # Mon (start)
            _pt(datetime(2024, 1, 5, 15, 30), 101_000.0),   # Fri
            _pt(datetime(2024, 1, 8, 15, 30), 100_500.0),   # next Mon
        ]
        days, rets = daily_returns_from_equity(pts)
        assert days == [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
        assert rets[:3] == [0.0, 0.0, 0.0]
        assert rets[3] == pytest.approx(0.01)
        assert rets[4] == pytest.approx(100_500 / 101_000 - 1)

    def test_trade_based_daily_returns_fill_gaps(self):
        trades = [_trade(500.0, datetime(2024, 1, 1, 14)), _trade(-200.0, datetime(2024, 1, 10, 14))]
        days, rets = daily_returns_from_trades(trades, 100_000.0)
        assert len(days) == 8                              # Mon 1st → Wed 10th weekdays
        assert sum(1 for r in rets if r == 0.0) == 6
        m = MetricsCalculator.calculate_all(trades, [], 100_000.0)
        assert len(m.daily_returns) == 8 and m.sharpe_ratio != 0.0

    def test_risk_free_in_consistent_units(self):
        r = [0.001, 0.003, -0.002, 0.004] * 10
        s0 = annualised_sharpe(r, 0.0)
        s1 = annualised_sharpe(r, 0.0252)
        sd = np.std(r, ddof=1)
        assert s0 - s1 == pytest.approx(0.0252 / 252 / sd * math.sqrt(252))
        assert annualised_sharpe(r, 0.0, periods_per_year=52) == pytest.approx(
            np.mean(r) / sd * math.sqrt(52)
        )

    def test_sortino_targets_risk_free(self):
        r = [0.0002] * 30                                  # always below 6.5%/252
        assert annualised_sortino(r, 0.065) < 0
        assert math.isinf(annualised_sortino([0.01, 0.02], 0.0))


class TestNoSentinels:
    def test_no_999_anywhere(self):
        wins = [_trade(500.0, datetime(2024, 1, 2 + i, 14)) for i in range(4)]
        pts = [_pt(datetime(2024, 1, 1), 100_000.0)] + [
            _pt(datetime(2024, 1, 2 + i, 15), 100_000.0 + 500 * (i + 1)) for i in range(4)
        ]
        m = MetricsCalculator.calculate_all(wins, pts, 100_000.0)
        for name in ("profit_factor", "payoff_ratio", "sortino_ratio"):
            assert getattr(m, name) != 999.0
            assert math.isinf(getattr(m, name))
            assert name in m.undefined_metrics
            assert not is_defined(m, name)
        assert is_defined(m, "win_rate")

    def test_defined_when_losses_exist(self):
        trades = [_trade(500.0, datetime(2024, 1, 2, 14)), _trade(-250.0, datetime(2024, 1, 3, 14))]
        m = MetricsCalculator.calculate_all(trades, [], 100_000.0)
        assert m.profit_factor == pytest.approx(2.0)
        assert "profit_factor" not in m.undefined_metrics

    def test_undefined_profit_factor_fails_optimizer_constraints(self):
        wins = [_trade(500.0, datetime(2024, 1, 2, 14)) for _ in range(40)]
        m = MetricsCalculator.calculate_all(wins, [], 100_000.0)
        cfg = OptimizationConfig(index_id="NIFTY50", start_date=date(2024, 1, 1),
                                 end_date=date(2024, 6, 30), min_trades=5,
                                 min_profit_factor=0.9, min_win_rate=0, max_drawdown_limit=100)
        assert m.profit_factor > cfg.min_profit_factor          # inf ≥ 0.9 numerically…
        assert OptimizationEngine._check_constraints(m, cfg) is False   # …but undefined → fails


class TestCompositeScore:
    BASE = dict(sharpe_ratio=1.0, win_rate=50.0, profit_factor=1.5, max_drawdown_pct=10.0)

    def _m(self, **kw):
        base = MetricsCalculator._zero_metrics(100_000.0)
        return dataclasses.replace(base, **{**self.BASE, **kw})

    @pytest.mark.parametrize("field,values,increasing", [
        ("sharpe_ratio", [-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.9], True),
        ("win_rate", [0.0, 20.0, 40.0, 60.0, 80.0, 100.0], True),
        ("profit_factor", [0.0, 0.5, 1.0, 2.0, 2.9], True),
        ("max_drawdown_pct", [0.0, 5.0, 20.0, 50.0, 90.0], False),
    ])
    def test_monotone_in_each_metric(self, field, values, increasing):
        scores = [MetricsCalculator.composite_score(self._m(**{field: v})) for v in values]
        diffs = np.diff(scores)
        assert np.all(diffs > 0) if increasing else np.all(diffs < 0)

    def test_saturation_is_weakly_monotone_and_handles_inf(self):
        s3 = MetricsCalculator.composite_score(self._m(profit_factor=3.0))
        sinf = MetricsCalculator.composite_score(self._m(profit_factor=float("inf")))
        assert sinf >= s3
        assert -0.3 <= MetricsCalculator.composite_score(self._m(sharpe_ratio=50.0)) <= 1.0

    def test_win_rate_in_percent_does_not_dominate(self):
        high_wr = self._m(sharpe_ratio=0.0, win_rate=70.0, profit_factor=1.0, max_drawdown_pct=10.0)
        better = self._m(sharpe_ratio=2.5, win_rate=50.0, profit_factor=2.5, max_drawdown_pct=5.0)
        report = MetricsCalculator.compare_backtests([high_wr, better], ["wr", "better"])
        assert report.best_overall == "better"


# ---------------------------------------------------------------------------
# 2. Bootstrap Monte Carlo
# ---------------------------------------------------------------------------


def _noisy_trades(n_days=250, mu=150.0, sigma=1500.0, seed=1):
    rng = np.random.default_rng(seed)
    days = _weekdays(date(2023, 1, 2), n_days)
    trades = []
    for d in days:
        if rng.random() < 0.6:                              # ~40% zero-P&L days
            trades.append(_trade(float(rng.normal(mu, sigma)), datetime(d.year, d.month, d.day, 14)))
    return trades, days


def _orig(total_return_pct=0.0, capital=100_000.0):
    base = MetricsCalculator._zero_metrics(capital)
    return dataclasses.replace(base, total_return_pct=total_return_pct,
                               total_return_amount=total_return_pct / 100 * capital)


class TestBootstrapMonteCarlo:
    def test_non_degenerate_distribution(self):
        trades, days = _noisy_trades()
        tester = RobustnessTester(MagicMock())
        res = tester.test_monte_carlo(trades, _orig(5.0), n_simulations=2000,
                                      initial_capital=100_000.0,
                                      start_date=days[0], end_date=days[-1])
        assert res.resample_unit == "daily"
        assert res.n_observations == len(days)               # zero-P&L days included
        assert res.p5_return < res.median_return < res.p95_return
        assert res.final_equity_p5 < res.final_equity_p50 < res.final_equity_p95
        assert res.sharpe_p5 < res.sharpe_p50 < res.sharpe_p95
        assert res.p5_drawdown < res.median_drawdown < res.p95_drawdown
        assert res.return_95_ci[0] < res.p5_return and res.return_95_ci[1] > res.p95_return
        assert 0.0 < res.prob_loss < 1.0
        assert res.prob_loss == pytest.approx(float(np.mean(res.block.final_equity < 100_000.0)))
        assert res.iid is not None and res.block is not None
        assert res.block.method == "stationary" and res.block.block_length >= 1

    def test_loss_probability_sane(self):
        tester = RobustnessTester(MagicMock())
        strong, d1 = _noisy_trades(mu=800.0, sigma=1000.0, seed=2)
        noise, d2 = _noisy_trades(mu=0.0, sigma=1000.0, seed=3)
        losing, d3 = _noisy_trades(mu=-800.0, sigma=1000.0, seed=4)
        p_strong = tester.test_monte_carlo(strong, _orig(), initial_capital=1e5).prob_loss
        p_noise = tester.test_monte_carlo(noise, _orig(), initial_capital=1e5).prob_loss
        p_losing = tester.test_monte_carlo(losing, _orig(), initial_capital=1e5).prob_loss
        assert p_strong < 0.05
        assert 0.05 < p_noise < 0.95
        assert p_losing > 0.95

    def test_capital_is_a_parameter(self):
        trades, _ = _noisy_trades()
        tester = RobustnessTester(MagicMock())
        a = tester.test_monte_carlo(trades, _orig(), initial_capital=100_000.0)
        b = tester.test_monte_carlo(trades, _orig(), initial_capital=200_000.0)
        assert a.initial_capital == 100_000.0 and b.initial_capital == 200_000.0
        assert b.median_return == pytest.approx(a.median_return / 2, rel=1e-9)
        # inferred from original metrics when not given
        c = tester.test_monte_carlo(trades, _orig(total_return_pct=4.0, capital=250_000.0))
        assert c.initial_capital == pytest.approx(250_000.0)

    def test_full_robustness_passes_capital_and_dsr_gate(self):
        trades, days = _noisy_trades(mu=800.0, sigma=1000.0, seed=2)
        tester = RobustnessTester(MagicMock())
        tester.test_sensitivity = MagicMock(return_value=MagicMock(
            is_robust=True, overall_sensitivity_score=0.9,
            most_sensitive_param="", least_sensitive_param=""))
        tester.test_data_stability = MagicMock(return_value=MagicMock(
            is_stable=True, consistency_score=0.9, profitable_periods=4, unprofitable_periods=0))
        mc_spy = MagicMock(wraps=tester.test_monte_carlo)
        tester.test_monte_carlo = mc_spy
        base = MagicMock()
        base.simulator_config.initial_capital = 300_000.0
        base.start_date, base.end_date = days[0], days[-1]
        rep = tester.run_full_robustness_test({}, base, _orig(10.0, 300_000.0), trades,
                                              ParameterSpace(parameters=[]),
                                              deflated_sharpe=0.50, n_trials=500)
        assert mc_spy.call_args.kwargs["initial_capital"] == 300_000.0
        assert mc_spy.call_args.kwargs["start_date"] == days[0]
        assert rep.is_approved is False
        assert any("Deflated Sharpe" in c for c in rep.concerns)

    def test_bootstrap_reproducible_and_block_semantics(self):
        r = np.random.default_rng(5).normal(0.001, 0.01, 120)
        a = bootstrap_returns(r, 1e5, 500, method="stationary", block_length=5, seed=9)
        b = bootstrap_returns(r, 1e5, 500, method="stationary", block_length=5, seed=9)
        np.testing.assert_array_equal(a.final_equity, b.final_equity)
        # a circular block as long as the series is a rotation → same total
        full = bootstrap_returns(r, 1e5, 200, method="block", block_length=len(r), seed=1)
        np.testing.assert_allclose(full.final_equity, 1e5 * (1 + r.sum()))
        assert np.ptp(full.max_drawdown_pct) > 0              # but path order differs
        iid = bootstrap_returns(r, 1e5, 2000, method="iid", seed=1)
        assert iid.pct("final_equity", 5) < iid.pct("final_equity", 95)
        with pytest.raises(ValueError):
            bootstrap_returns(r, 1e5, 10, method="nope")

    def test_index_generators(self):
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(50, 300, 10.0, rng)
        assert idx.shape == (300, 50) and idx.min() >= 0 and idx.max() < 50
        cont = np.mean((idx[:, 1:] - idx[:, :-1]) % 50 == 1)
        assert 0.85 < cont < 0.95                              # ≈ 1 - 1/L
        blk = circular_block_bootstrap_indices(50, 10, 7, rng)
        assert blk.shape == (10, 50)
        assert np.all((blk[:, 1:7] - blk[:, :6]) % 50 == 1)

    def test_daily_pnl_includes_zero_days(self):
        t = [_trade(100.0, datetime(2024, 1, 1, 14)), _trade(-50.0, datetime(2024, 1, 5, 14)),
             _trade(25.0, datetime(2024, 1, 5, 15))]
        days, pnl = daily_pnl_from_trades(t)
        assert days == [date(2024, 1, d) for d in range(1, 6)]
        assert pnl == [100.0, 0.0, 0.0, 0.0, -25.0]


# ---------------------------------------------------------------------------
# 3. Optimiser: DSR / trials / holdout / full-param walk-forward
# ---------------------------------------------------------------------------


def _bt_result(cfg, metrics, trading_days=100):
    return BacktestResult(
        config=cfg, index_id=cfg.index_id, start_date=cfg.start_date, end_date=cfg.end_date,
        total_bars=trading_days, trading_days=trading_days, trade_history=[],
        total_trades=metrics.total_trades, total_signals_generated=0, actionable_signals=0,
        executed_trades=metrics.total_trades, metrics=metrics, equity_curve=[],
        initial_capital=100_000.0, final_capital=100_000.0 * (1 + metrics.total_return_pct / 100),
        total_return_pct=metrics.total_return_pct,
    )


def _metrics_from_returns(rets, **kw):
    base = MetricsCalculator._zero_metrics(100_000.0)
    sr = annualised_sharpe(list(rets), 0.0)
    fields = dict(
        total_trades=50, winning_trades=30, losing_trades=20, win_rate=60.0,
        profit_factor=1.6, max_drawdown_pct=8.0, total_return_pct=float(np.sum(rets) * 100),
        sharpe_ratio=sr, daily_returns=list(rets), undefined_metrics=[], risk_free_rate=0.0,
    )
    fields.update(kw)
    return dataclasses.replace(base, **fields)


def _space():
    return ParameterSpace(parameters=[
        ParameterDef(name="stop_loss_atr_multiplier", display_name="SL", description="",
                     param_type="float", min_value=1.0, max_value=1.5, step=0.5, default=1.5),
        ParameterDef(name="min_confidence_filter", display_name="conf", description="",
                     param_type="choice", choices=["HIGH", "MEDIUM"], default="MEDIUM"),
        ParameterDef(name="signal_cooldown_bars", display_name="cd", description="",
                     param_type="int", min_value=2, max_value=4, step=2, default=2),
    ])


def _opt_config(**kw):
    d = dict(index_id="NIFTY50", start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
             parameter_space=_space(), show_progress=False, run_walk_forward=False,
             min_trades=5, max_drawdown_limit=50.0, min_win_rate=30.0, min_profit_factor=0.5)
    d.update(kw)
    return OptimizationConfig(**d)


class TestOptimizerStatistics:
    def _engine(self, mu, sigma=0.01, seed=0):
        engine = OptimizationEngine(MagicMock())
        rng = np.random.default_rng(seed)
        calls = []

        def run(cfg):
            calls.append(cfg)
            return _bt_result(cfg, _metrics_from_returns(rng.normal(mu, sigma, 200)))

        engine.strategy_runner.run = run
        return engine, calls

    def test_selection_and_holdout_disjoint_and_holdout_once(self):
        engine, calls = self._engine(0.004)
        res = engine.run(_opt_config(holdout_fraction=0.2, embargo_days=5))
        assert res.selection_end < res.holdout_start
        assert res.holdout_end == date(2024, 12, 31)
        grid = [c for c in calls if c.end_date <= res.selection_end]
        hold = [c for c in calls if c.start_date >= res.holdout_start]
        assert len(grid) == res.total_evaluations == 8
        assert len(hold) == 1 and len(calls) == 9
        assert res.holdout_result is not None and res.holdout_passed is True

    def test_dsr_and_trial_count_reported_and_strong_config_approved(self):
        engine, _ = self._engine(0.004)
        res = engine.run(_opt_config())
        assert res.n_trials == 8
        assert res.best_dsr is not None and res.best_dsr.n_trials == 8
        assert res.best_dsr.dsr >= 0.95
        assert res.ranked_results[0].dsr == pytest.approx(res.best_dsr.dsr)
        assert res.best_approved is True and res.ranked_results[0].approved
        summary = res.get_summary()
        assert "Deflated Sharpe" in summary and "Trials evaluated (N):   8" in summary
        assert "APPROVAL: APPROVED" in summary
        assert res.pbo_result is not None and 0.0 <= res.pbo_result.pbo <= 1.0

    def test_low_dsr_never_approved(self):
        engine, _ = self._engine(0.0003, sigma=0.01, seed=4)
        res = engine.run(_opt_config(n_trials_override=5000))
        assert res.best_params is not None
        assert res.best_dsr.dsr < 0.95
        assert res.best_approved is False
        assert all(not r.approved for r in res.ranked_results if r.dsr is not None and r.dsr < 0.95)
        assert any("DSR" in n and "<" in n for n in res.approval_notes)
        assert "NOT APPROVED" in res.get_summary()

    def test_dsr_falls_with_more_trials(self):
        e1, _ = self._engine(0.002, seed=7)
        e2, _ = self._engine(0.002, seed=7)
        small = e1.run(_opt_config(n_trials_override=2)).best_dsr.dsr
        large = e2.run(_opt_config(n_trials_override=10_000)).best_dsr.dsr
        assert large < small

    def test_walk_forward_gets_all_params_and_selection_period_only(self):
        engine, _ = self._engine(0.004)
        seen = []
        with patch.object(WalkForwardValidator, "run_walk_forward",
                          side_effect=lambda wf: seen.append(wf) or MagicMock()):
            res = engine.run(_opt_config(run_walk_forward=True, walk_forward_top_n=3))
        assert len(seen) == 3
        for wf, ranked in zip(seen, res.ranked_results):
            p = ranked.params
            assert wf.full_end_date == res.selection_end            # holdout untouched
            assert wf.min_confidence == p["min_confidence_filter"]
            assert wf.backtest_template.min_confidence == p["min_confidence_filter"]
            assert wf.backtest_template.signal_cooldown_bars == p["signal_cooldown_bars"]
            assert wf.simulator_config.stop_loss_atr_mult == p["stop_loss_atr_multiplier"]

    def test_nested_walk_forward_fits_on_train_only(self):
        engine, calls = self._engine(0.004)
        res = engine.run(_opt_config(run_nested_walk_forward=True, nested_wf_folds=3,
                                     holdout_fraction=0.2, embargo_days=5))
        nwf = res.nested_wf_result
        assert nwf is not None and nwf.n_folds == 3 and nwf.holdout is None
        for fr in nwf.folds:
            f = fr.fold
            assert f.train_end < f.test_start
            assert f.test_end <= res.selection_end                   # never the holdout
            assert "sharpe_ratio" in fr.metrics
        # every nested backtest stays inside the selection period
        nested = calls[res.total_evaluations:]
        assert all(c.end_date <= res.selection_end or c.start_date >= res.holdout_start for c in nested)
        assert "NESTED WALK-FORWARD" in res.get_summary()

    def test_holdout_fraction_validation(self):
        engine, _ = self._engine(0.004)
        with pytest.raises(ValueError):
            engine.run(_opt_config(holdout_fraction=1.0))
        with pytest.raises(ValueError):
            engine.run(_opt_config(start_date=date(2024, 1, 1), end_date=date(2024, 1, 5),
                                   holdout_fraction=0.5, embargo_days=5))
