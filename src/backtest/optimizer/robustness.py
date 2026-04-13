"""
Robustness Testing — Step 7.3 of the Trading Decision Support System.

Subjects optimised parameter sets to four stress tests to determine whether
they reflect genuine market edge or are overfit artefacts:

1. **Parameter Sensitivity** — ±20% perturbation of each parameter.
2. **Data Stability** — Performance across multiple date-range sub-periods.
3. **Monte Carlo Simulation** — Trade-order reshuffling (1 000 simulations).
4. **Regime Robustness** — Performance across different market regimes.

Usage
-----
::

    tester = RobustnessTester(db)
    report = tester.run_full_robustness_test(
        params=best_params,
        base_config=config,
        original_metrics=metrics,
        trade_history=trades,
        parameter_space=space,
    )
    print(report.summary)
"""

from __future__ import annotations

import copy
import logging
import random
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

from src.backtest.metrics import BacktestMetrics, MetricsCalculator
from src.backtest.optimizer.param_space import ParameterApplicator, ParameterSpace
from src.backtest.strategy_runner import BacktestConfig, BacktestResult, StrategyRunner
from src.backtest.trade_simulator import ClosedTrade, SimulatorConfig
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses — Test 1: Parameter Sensitivity
# ---------------------------------------------------------------------------


@dataclass
class ParamSensitivity:
    """Sensitivity profile for a single parameter."""

    param_name: str
    original_value: Any
    plus_20_value: Any
    minus_20_value: Any

    original_return: float
    plus_20_return: float
    minus_20_return: float

    original_sharpe: float
    plus_20_sharpe: float
    minus_20_sharpe: float

    return_stability: float  # 1 - max(|orig-plus|, |orig-minus|) / |orig|; 1.0 = stable

    is_stable: bool  # Return changed < 50% AND still profitable
    is_critical: bool  # Return flipped from positive to negative


@dataclass
class SensitivityResult:
    """Aggregated sensitivity analysis across all parameters."""

    parameter_sensitivities: list[ParamSensitivity]
    overall_sensitivity_score: float  # 0–1, higher = more robust (less sensitive)
    most_sensitive_param: str
    least_sensitive_param: str
    is_robust: bool  # All params have stability > 0.4 AND none critical


# ---------------------------------------------------------------------------
# Dataclasses — Test 2: Data Stability
# ---------------------------------------------------------------------------


@dataclass
class PeriodResult:
    """Backtest outcome for a single sub-period."""

    period_name: str
    start_date: date
    end_date: date
    return_pct: float
    win_rate: float
    trade_count: int
    sharpe: float
    max_drawdown: float


@dataclass
class StabilityResult:
    """Aggregated stability analysis across sub-periods."""

    period_results: list[PeriodResult]

    profitable_periods: int
    unprofitable_periods: int
    profitability_rate: float  # Fraction of periods that were profitable

    return_std: float
    return_range: tuple[float, float]  # (worst, best)

    consistency_score: float  # 0–1
    is_stable: bool  # profitability_rate > 60% AND no catastrophic loss (> -15%)


# ---------------------------------------------------------------------------
# Dataclasses — Test 3: Monte Carlo
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results from trade-order reshuffling."""

    n_simulations: int

    # Return distribution
    median_return: float
    p5_return: float
    p25_return: float
    p75_return: float
    p95_return: float
    pct_profitable: float  # Fraction of simulations with positive return

    # Drawdown distribution
    median_drawdown: float
    p95_drawdown: float

    # Confidence interval
    return_95_ci: tuple[float, float]

    # Statistical significance
    is_statistically_profitable: bool  # p5_return > 0
    luck_factor: float  # (original - median) / |original|
    reliability_score: float  # 0–1


# ---------------------------------------------------------------------------
# Dataclasses — Test 4: Regime Robustness
# ---------------------------------------------------------------------------


@dataclass
class RegimeResult:
    """Performance breakdown by market regime."""

    regime_performance: dict[str, dict]  # {regime: {trades, win_rate, return, sharpe}}

    regimes_tested: int
    profitable_regimes: int

    best_regime: str
    worst_regime: str
    best_regime_return: float
    worst_regime_return: float

    is_regime_dependent: bool  # Only profitable in 1 regime
    regime_diversity_score: float  # 0–1, profitable_regimes / regimes_tested


# ---------------------------------------------------------------------------
# Dataclass — Overall Report
# ---------------------------------------------------------------------------


@dataclass
class RobustnessReport:
    """Comprehensive robustness assessment for a parameter set."""

    params: dict
    original_metrics: BacktestMetrics

    # Individual test results
    sensitivity_result: Optional[SensitivityResult]
    stability_result: Optional[StabilityResult]
    monte_carlo_result: Optional[MonteCarloResult]
    regime_result: Optional[RegimeResult]

    # Overall
    robustness_score: float  # 0–1 weighted average of all tests
    robustness_grade: str  # ROBUST / MODERATE / FRAGILE / OVERFIT
    is_approved: bool  # score >= 0.5 → approved for paper trading

    concerns: list[str]
    recommendations: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Core: RobustnessTester
# ---------------------------------------------------------------------------


class RobustnessTester:
    """Subjects optimised parameters to multiple robustness stress tests."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.strategy_runner = StrategyRunner(db)
        self.metrics_calculator = MetricsCalculator()
        self.applicator = ParameterApplicator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_robustness_test(
        self,
        params: dict,
        base_config: BacktestConfig,
        original_metrics: BacktestMetrics,
        trade_history: list[ClosedTrade],
        parameter_space: ParameterSpace,
    ) -> RobustnessReport:
        """Run all four robustness tests and produce an overall assessment."""

        concerns: list[str] = []
        recommendations: list[str] = []

        # -- Test 1: Sensitivity -----------------------------------------
        logger.info("Robustness test 1/4: Parameter sensitivity …")
        sensitivity_result = self.test_sensitivity(
            params, base_config, original_metrics, parameter_space,
        )
        if not sensitivity_result.is_robust:
            concerns.append(
                f"Parameter sensitivity: most sensitive param is "
                f"'{sensitivity_result.most_sensitive_param}'"
            )
            recommendations.append(
                "Consider wider stop-loss or less aggressive parameter values "
                "to reduce sensitivity."
            )

        # -- Test 2: Data stability --------------------------------------
        logger.info("Robustness test 2/4: Data stability …")
        stability_result = self.test_data_stability(params, base_config)
        if not stability_result.is_stable:
            concerns.append(
                f"Data stability: profitable in only "
                f"{stability_result.profitable_periods}/"
                f"{stability_result.profitable_periods + stability_result.unprofitable_periods} "
                f"sub-periods"
            )
            recommendations.append(
                "Strategy may be overfit to specific market conditions. "
                "Test on additional out-of-sample data."
            )

        # -- Test 3: Monte Carlo -----------------------------------------
        mc_result: Optional[MonteCarloResult] = None
        if len(trade_history) >= 20:
            logger.info("Robustness test 3/4: Monte Carlo (%d trades) …", len(trade_history))
            mc_result = self.test_monte_carlo(trade_history, original_metrics)
            if not mc_result.is_statistically_profitable:
                concerns.append(
                    f"Monte Carlo: only {mc_result.pct_profitable:.0%} of simulations profitable"
                )
                recommendations.append(
                    "Strategy profitability depends heavily on trade ordering / luck. "
                    "Need more trades or higher win rate."
                )
            if mc_result.luck_factor > 0.5:
                concerns.append(
                    f"Monte Carlo: luck factor is {mc_result.luck_factor:.2f} "
                    f"— over half the return may be luck"
                )
        else:
            logger.warning(
                "Skipping Monte Carlo: only %d trades (need >= 20).",
                len(trade_history),
            )
            concerns.append(
                f"Insufficient trades ({len(trade_history)}) for Monte Carlo analysis"
            )
            recommendations.append("Run backtest over a longer period to generate more trades.")

        # -- Test 4: Regime robustness -----------------------------------
        regime_result: Optional[RegimeResult] = None
        has_regime_data = any(
            _get_trade_regime(t) is not None for t in trade_history
        )
        if has_regime_data:
            logger.info("Robustness test 4/4: Regime robustness …")
            regime_result = self.test_regime_robustness(trade_history)
            if regime_result.is_regime_dependent:
                concerns.append(
                    f"Regime dependent: only profitable in '{regime_result.best_regime}'"
                )
                recommendations.append(
                    "Strategy only works in one market regime. Consider regime-specific "
                    "parameter sets or regime filters."
                )
        else:
            logger.warning("Skipping regime analysis: no regime data on trades.")
            concerns.append("No regime data available on trades for regime analysis")

        # -- Overall scoring ---------------------------------------------
        score_parts: list[tuple[float, float]] = []  # (score, weight)

        score_parts.append((sensitivity_result.overall_sensitivity_score, 0.25))

        score_parts.append((stability_result.consistency_score, 0.30))

        if mc_result is not None:
            score_parts.append((mc_result.reliability_score, 0.25))

        if regime_result is not None:
            score_parts.append((regime_result.regime_diversity_score, 0.20))

        total_weight = sum(w for _, w in score_parts)
        if total_weight > 0:
            robustness_score = sum(s * w for s, w in score_parts) / total_weight
        else:
            robustness_score = 0.0

        robustness_score = max(0.0, min(1.0, robustness_score))

        if robustness_score > 0.7:
            grade = "ROBUST"
        elif robustness_score >= 0.5:
            grade = "MODERATE"
        elif robustness_score >= 0.3:
            grade = "FRAGILE"
        else:
            grade = "OVERFIT"

        is_approved = robustness_score >= 0.5

        if not concerns:
            concerns.append("No significant concerns found")
        if not recommendations:
            recommendations.append("Parameters appear robust — proceed to paper trading")

        summary = _build_summary(
            params=params,
            original_metrics=original_metrics,
            sensitivity_result=sensitivity_result,
            stability_result=stability_result,
            monte_carlo_result=mc_result,
            regime_result=regime_result,
            robustness_score=robustness_score,
            grade=grade,
            is_approved=is_approved,
            concerns=concerns,
        )

        return RobustnessReport(
            params=params,
            original_metrics=original_metrics,
            sensitivity_result=sensitivity_result,
            stability_result=stability_result,
            monte_carlo_result=mc_result,
            regime_result=regime_result,
            robustness_score=robustness_score,
            robustness_grade=grade,
            is_approved=is_approved,
            concerns=concerns,
            recommendations=recommendations,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Test 1: Parameter Sensitivity
    # ------------------------------------------------------------------

    def test_sensitivity(
        self,
        params: dict,
        base_config: BacktestConfig,
        original_metrics: BacktestMetrics,
        parameter_space: ParameterSpace,
    ) -> SensitivityResult:
        """Perturb each parameter by ±20% and measure impact on returns."""

        param_defs = {p.name: p for p in parameter_space.parameters}
        sensitivities: list[ParamSensitivity] = []

        for pname, pvalue in params.items():
            pdef = param_defs.get(pname)
            if pdef is None or pdef.param_type == "choice":
                continue

            plus_val = _perturb_value(pvalue, +0.20, pdef)
            minus_val = _perturb_value(pvalue, -0.20, pdef)

            plus_metrics = self._run_variant(params, pname, plus_val, base_config)
            minus_metrics = self._run_variant(params, pname, minus_val, base_config)

            plus_ret = plus_metrics.total_return_pct if plus_metrics else 0.0
            minus_ret = minus_metrics.total_return_pct if minus_metrics else 0.0
            plus_sharpe = plus_metrics.sharpe_ratio if plus_metrics else 0.0
            minus_sharpe = minus_metrics.sharpe_ratio if minus_metrics else 0.0

            orig_ret = original_metrics.total_return_pct
            orig_sharpe = original_metrics.sharpe_ratio

            # return_stability: 1 - max(|orig-plus|, |orig-minus|) / |orig|
            if abs(orig_ret) > 1e-9:
                max_deviation = max(abs(orig_ret - plus_ret), abs(orig_ret - minus_ret))
                stability = max(0.0, 1.0 - max_deviation / abs(orig_ret))
            else:
                stability = 1.0 if (abs(plus_ret) < 1e-9 and abs(minus_ret) < 1e-9) else 0.0

            # is_stable: return changed < 50% AND both variants still profitable
            changed_less_than_50 = stability >= 0.5
            both_profitable = plus_ret > 0 and minus_ret > 0
            is_stable = changed_less_than_50 and both_profitable

            # is_critical: original was positive but a variant went negative
            is_critical = orig_ret > 0 and (plus_ret < 0 or minus_ret < 0)

            sensitivities.append(ParamSensitivity(
                param_name=pname,
                original_value=pvalue,
                plus_20_value=plus_val,
                minus_20_value=minus_val,
                original_return=orig_ret,
                plus_20_return=plus_ret,
                minus_20_return=minus_ret,
                original_sharpe=orig_sharpe,
                plus_20_sharpe=plus_sharpe,
                minus_20_sharpe=minus_sharpe,
                return_stability=stability,
                is_stable=is_stable,
                is_critical=is_critical,
            ))

        if not sensitivities:
            return SensitivityResult(
                parameter_sensitivities=[],
                overall_sensitivity_score=1.0,
                most_sensitive_param="",
                least_sensitive_param="",
                is_robust=True,
            )

        overall_score = statistics.mean(s.return_stability for s in sensitivities)

        sorted_by_stability = sorted(sensitivities, key=lambda s: s.return_stability)
        most_sensitive = sorted_by_stability[0].param_name
        least_sensitive = sorted_by_stability[-1].param_name

        is_robust = (
            all(s.return_stability > 0.4 for s in sensitivities)
            and not any(s.is_critical for s in sensitivities)
        )

        return SensitivityResult(
            parameter_sensitivities=sensitivities,
            overall_sensitivity_score=overall_score,
            most_sensitive_param=most_sensitive,
            least_sensitive_param=least_sensitive,
            is_robust=is_robust,
        )

    # ------------------------------------------------------------------
    # Test 2: Data Stability
    # ------------------------------------------------------------------

    def test_data_stability(
        self,
        params: dict,
        base_config: BacktestConfig,
    ) -> StabilityResult:
        """Test the same parameters across multiple date sub-periods."""

        start = base_config.start_date
        end = base_config.end_date
        sub_periods = _generate_sub_periods(start, end)

        period_results: list[PeriodResult] = []

        for name, p_start, p_end in sub_periods:
            if p_start >= p_end:
                continue

            variant_config = self.applicator.apply(base_config, params)
            if variant_config is None:
                continue

            variant_config.start_date = p_start
            variant_config.end_date = p_end
            variant_config.show_progress = False

            try:
                bt_result = self.strategy_runner.run(variant_config)
            except Exception as exc:
                logger.warning("Stability sub-period '%s' failed: %s", name, exc)
                continue

            if bt_result.total_trades == 0:
                logger.info("Stability sub-period '%s': 0 trades — skipped.", name)
                continue

            metrics = bt_result.metrics
            if metrics is None:
                continue

            period_results.append(PeriodResult(
                period_name=name,
                start_date=p_start,
                end_date=p_end,
                return_pct=metrics.total_return_pct,
                win_rate=metrics.win_rate,
                trade_count=metrics.total_trades,
                sharpe=metrics.sharpe_ratio,
                max_drawdown=metrics.max_drawdown_pct,
            ))

        if not period_results:
            return StabilityResult(
                period_results=[],
                profitable_periods=0,
                unprofitable_periods=0,
                profitability_rate=0.0,
                return_std=0.0,
                return_range=(0.0, 0.0),
                consistency_score=0.0,
                is_stable=False,
            )

        profitable = sum(1 for p in period_results if p.return_pct > 0)
        unprofitable = len(period_results) - profitable
        profitability_rate = profitable / len(period_results)

        returns = [p.return_pct for p in period_results]
        ret_std = statistics.stdev(returns) if len(returns) >= 2 else 0.0
        ret_range = (min(returns), max(returns))

        avg_ret = statistics.mean(returns)
        if abs(avg_ret) > 1e-9 and ret_std > 0:
            normalised_std = min(1.0, ret_std / (abs(avg_ret) * 3))
        else:
            normalised_std = 0.0

        consistency_score = profitability_rate * 0.6 + (1.0 - normalised_std) * 0.4
        consistency_score = max(0.0, min(1.0, consistency_score))

        catastrophic = any(p.return_pct < -15.0 for p in period_results)
        is_stable = profitability_rate > 0.6 and not catastrophic

        return StabilityResult(
            period_results=period_results,
            profitable_periods=profitable,
            unprofitable_periods=unprofitable,
            profitability_rate=profitability_rate,
            return_std=ret_std,
            return_range=ret_range,
            consistency_score=consistency_score,
            is_stable=is_stable,
        )

    # ------------------------------------------------------------------
    # Test 3: Monte Carlo Simulation
    # ------------------------------------------------------------------

    def test_monte_carlo(
        self,
        trade_history: list[ClosedTrade],
        original_metrics: BacktestMetrics,
        n_simulations: int = 1000,
        seed: int = 42,
    ) -> MonteCarloResult:
        """Reshuffle trade P&Ls to assess statistical robustness."""

        pnls = [t.net_pnl for t in trade_history]
        initial_capital = 100_000.0  # standard baseline for %

        # Edge case: all identical P&Ls → no variance
        if len(set(pnls)) <= 1:
            total_pnl = sum(pnls)
            ret = (total_pnl / initial_capital) * 100.0
            return MonteCarloResult(
                n_simulations=n_simulations,
                median_return=ret,
                p5_return=ret,
                p25_return=ret,
                p75_return=ret,
                p95_return=ret,
                pct_profitable=1.0 if ret > 0 else 0.0,
                median_drawdown=0.0,
                p95_drawdown=0.0,
                return_95_ci=(ret, ret),
                is_statistically_profitable=ret > 0,
                luck_factor=0.0,
                reliability_score=1.0 if ret > 0 else 0.0,
            )

        rng = random.Random(seed)
        sim_returns: list[float] = []
        sim_drawdowns: list[float] = []

        for _ in range(n_simulations):
            shuffled = pnls[:]
            rng.shuffle(shuffled)

            equity = initial_capital
            peak = equity
            max_dd = 0.0

            for pnl in shuffled:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = ((peak - equity) / peak) * 100.0 if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

            sim_ret = ((equity - initial_capital) / initial_capital) * 100.0
            sim_returns.append(sim_ret)
            sim_drawdowns.append(max_dd)

        sim_returns.sort()
        sim_drawdowns.sort()

        def percentile(data: list[float], pct: float) -> float:
            idx = int(len(data) * pct / 100.0)
            idx = max(0, min(idx, len(data) - 1))
            return data[idx]

        median_return = percentile(sim_returns, 50)
        p5_return = percentile(sim_returns, 5)
        p25_return = percentile(sim_returns, 25)
        p75_return = percentile(sim_returns, 75)
        p95_return = percentile(sim_returns, 95)

        pct_profitable = sum(1 for r in sim_returns if r > 0) / len(sim_returns)

        median_dd = percentile(sim_drawdowns, 50)
        p95_dd = percentile(sim_drawdowns, 95)

        return_95_ci = (p5_return, p95_return)

        is_stat_profitable = p5_return > 0

        orig_ret = original_metrics.total_return_pct
        if abs(orig_ret) > 1e-9:
            luck_factor = max(0.0, (orig_ret - median_return) / abs(orig_ret))
        else:
            luck_factor = 0.0
        luck_factor = min(1.0, luck_factor)

        # reliability_score = pct_profitable * 0.5 + (1 - luck) * 0.3 + (p5 > 0) * 0.2
        reliability_score = (
            pct_profitable * 0.5
            + (1.0 - luck_factor) * 0.3
            + (1.0 if p5_return > 0 else 0.0) * 0.2
        )
        reliability_score = max(0.0, min(1.0, reliability_score))

        return MonteCarloResult(
            n_simulations=n_simulations,
            median_return=median_return,
            p5_return=p5_return,
            p25_return=p25_return,
            p75_return=p75_return,
            p95_return=p95_return,
            pct_profitable=pct_profitable,
            median_drawdown=median_dd,
            p95_drawdown=p95_dd,
            return_95_ci=return_95_ci,
            is_statistically_profitable=is_stat_profitable,
            luck_factor=luck_factor,
            reliability_score=reliability_score,
        )

    # ------------------------------------------------------------------
    # Test 4: Regime Robustness
    # ------------------------------------------------------------------

    def test_regime_robustness(
        self,
        trade_history: list[ClosedTrade],
    ) -> RegimeResult:
        """Group trades by market regime and assess cross-regime performance."""

        regime_trades: dict[str, list[ClosedTrade]] = {}
        for t in trade_history:
            regime = _get_trade_regime(t)
            if regime is None:
                continue
            regime_trades.setdefault(regime, []).append(t)

        if not regime_trades:
            return RegimeResult(
                regime_performance={},
                regimes_tested=0,
                profitable_regimes=0,
                best_regime="",
                worst_regime="",
                best_regime_return=0.0,
                worst_regime_return=0.0,
                is_regime_dependent=True,
                regime_diversity_score=0.0,
            )

        regime_perf: dict[str, dict] = {}
        for regime, trades in regime_trades.items():
            total_pnl = sum(t.net_pnl for t in trades)
            wins = sum(1 for t in trades if t.outcome == "WIN")
            win_rate = (wins / len(trades) * 100.0) if trades else 0.0

            # Simple sharpe proxy: mean / std of trade P&Ls
            trade_pnls = [t.net_pnl for t in trades]
            mean_pnl = statistics.mean(trade_pnls) if trade_pnls else 0.0
            std_pnl = statistics.stdev(trade_pnls) if len(trade_pnls) >= 2 else 0.0
            sharpe = (mean_pnl / std_pnl) if std_pnl > 0 else 0.0

            regime_perf[regime] = {
                "trades": len(trades),
                "win_rate": win_rate,
                "return": total_pnl,
                "sharpe": sharpe,
            }

        regimes_tested = len(regime_perf)
        profitable_regimes = sum(
            1 for rp in regime_perf.values() if rp["return"] > 0
        )

        sorted_regimes = sorted(regime_perf.items(), key=lambda kv: kv[1]["return"])
        worst_regime = sorted_regimes[0][0]
        worst_return = sorted_regimes[0][1]["return"]
        best_regime = sorted_regimes[-1][0]
        best_return = sorted_regimes[-1][1]["return"]

        is_regime_dependent = profitable_regimes <= 1 and regimes_tested > 1
        diversity_score = profitable_regimes / regimes_tested if regimes_tested > 0 else 0.0

        return RegimeResult(
            regime_performance=regime_perf,
            regimes_tested=regimes_tested,
            profitable_regimes=profitable_regimes,
            best_regime=best_regime,
            worst_regime=worst_regime,
            best_regime_return=best_return,
            worst_regime_return=worst_return,
            is_regime_dependent=is_regime_dependent,
            regime_diversity_score=diversity_score,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_variant(
        self,
        params: dict,
        changed_param: str,
        new_value: Any,
        base_config: BacktestConfig,
    ) -> Optional[BacktestMetrics]:
        """Run a single backtest with one parameter changed."""
        variant_params = dict(params)
        variant_params[changed_param] = new_value

        config = self.applicator.apply(base_config, variant_params)
        if config is None:
            return None

        config.show_progress = False

        try:
            result = self.strategy_runner.run(config)
        except Exception as exc:
            logger.warning(
                "Sensitivity variant %s=%s failed: %s",
                changed_param, new_value, exc,
            )
            return None

        return result.metrics


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _perturb_value(
    value: Any,
    fraction: float,
    pdef: Any,
) -> Any:
    """Shift *value* by *fraction* (e.g. +0.20), clamped to param bounds."""
    new_val = value * (1.0 + fraction)

    if pdef.min_value is not None:
        new_val = max(new_val, pdef.min_value)
    if pdef.max_value is not None:
        new_val = min(new_val, pdef.max_value)

    if pdef.param_type == "int":
        new_val = int(round(new_val))

    return new_val


def _get_trade_regime(trade: ClosedTrade) -> Optional[str]:
    """Extract regime from a ClosedTrade's entry_bar metadata."""
    if hasattr(trade, "regime") and trade.regime is not None:
        return trade.regime
    if isinstance(trade.entry_bar, dict):
        return trade.entry_bar.get("regime")
    return None


def _generate_sub_periods(
    start: date,
    end: date,
) -> list[tuple[str, date, date]]:
    """Generate sub-periods for data-stability testing."""
    periods: list[tuple[str, date, date]] = []
    total_days = (end - start).days
    if total_days < 60:
        return periods

    mid = start + timedelta(days=total_days // 2)

    # First half / second half
    periods.append(("First Half", start, mid))
    periods.append(("Second Half", mid + timedelta(days=1), end))

    # Odd / even months
    odd_months: list[tuple[date, date]] = []
    even_months: list[tuple[date, date]] = []
    _collect_months(start, end, odd_months, even_months)

    if odd_months:
        periods.append(("Odd Months", odd_months[0][0], odd_months[-1][1]))
    if even_months:
        periods.append(("Even Months", even_months[0][0], even_months[-1][1]))

    # Quarters
    quarters = _collect_quarters(start, end)
    for q_name, q_start, q_end in quarters:
        periods.append((q_name, q_start, q_end))

    return periods


def _collect_months(
    start: date,
    end: date,
    odd_out: list[tuple[date, date]],
    even_out: list[tuple[date, date]],
) -> None:
    """Partition the date range into odd-month and even-month spans."""
    current = date(start.year, start.month, 1)
    while current <= end:
        month_start = max(current, start)
        # last day of month
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        month_end = min(next_month - timedelta(days=1), end)

        if month_start <= month_end:
            if current.month % 2 == 1:
                odd_out.append((month_start, month_end))
            else:
                even_out.append((month_start, month_end))

        current = next_month


def _collect_quarters(
    start: date,
    end: date,
) -> list[tuple[str, date, date]]:
    """Collect individual quarters within the date range."""
    quarters: list[tuple[str, date, date]] = []
    quarter_starts = [1, 4, 7, 10]

    year = start.year
    while year <= end.year:
        for qi, qm in enumerate(quarter_starts, 1):
            q_start = date(year, qm, 1)
            if qm + 3 <= 12:
                q_end = date(year, qm + 3, 1) - timedelta(days=1)
            else:
                q_end = date(year + 1, 1, 1) - timedelta(days=1)

            actual_start = max(q_start, start)
            actual_end = min(q_end, end)

            if actual_start <= actual_end and (actual_end - actual_start).days >= 14:
                quarters.append((f"Q{qi} {year}", actual_start, actual_end))

        year += 1

    return quarters


def _build_summary(
    *,
    params: dict,
    original_metrics: BacktestMetrics,
    sensitivity_result: SensitivityResult,
    stability_result: StabilityResult,
    monte_carlo_result: Optional[MonteCarloResult],
    regime_result: Optional[RegimeResult],
    robustness_score: float,
    grade: str,
    is_approved: bool,
    concerns: list[str],
) -> str:
    """Build a human-readable robustness summary."""

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("ROBUSTNESS TEST REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Overall Score : {robustness_score:.2f} / 1.00")
    lines.append(f"Grade         : {grade}")
    lines.append(f"Approved      : {'YES — safe for paper trading' if is_approved else 'NO — do not use'}")
    lines.append(f"Original Return: {original_metrics.total_return_pct:.2f}%")
    lines.append(f"Original Sharpe: {original_metrics.sharpe_ratio:.2f}")
    lines.append("")

    # Sensitivity
    lines.append("--- Parameter Sensitivity ---")
    lines.append(f"  Score: {sensitivity_result.overall_sensitivity_score:.2f}")
    lines.append(f"  Robust: {'Yes' if sensitivity_result.is_robust else 'No'}")
    if sensitivity_result.most_sensitive_param:
        lines.append(f"  Most sensitive : {sensitivity_result.most_sensitive_param}")
        lines.append(f"  Least sensitive: {sensitivity_result.least_sensitive_param}")
    lines.append("")

    # Stability
    lines.append("--- Data Stability ---")
    lines.append(f"  Score: {stability_result.consistency_score:.2f}")
    total_periods = stability_result.profitable_periods + stability_result.unprofitable_periods
    lines.append(
        f"  Profitable periods: {stability_result.profitable_periods}/{total_periods}"
    )
    lines.append(f"  Stable: {'Yes' if stability_result.is_stable else 'No'}")
    lines.append("")

    # Monte Carlo
    lines.append("--- Monte Carlo ---")
    if monte_carlo_result:
        lines.append(f"  Score: {monte_carlo_result.reliability_score:.2f}")
        lines.append(f"  Simulations: {monte_carlo_result.n_simulations}")
        lines.append(f"  Median return: {monte_carlo_result.median_return:.2f}%")
        lines.append(f"  5th pctl return: {monte_carlo_result.p5_return:.2f}%")
        lines.append(f"  % profitable: {monte_carlo_result.pct_profitable:.0%}")
        lines.append(f"  Luck factor: {monte_carlo_result.luck_factor:.2f}")
        lines.append(f"  Statistically profitable: {'Yes' if monte_carlo_result.is_statistically_profitable else 'No'}")
    else:
        lines.append("  Skipped (insufficient trades)")
    lines.append("")

    # Regime
    lines.append("--- Regime Robustness ---")
    if regime_result:
        lines.append(f"  Score: {regime_result.regime_diversity_score:.2f}")
        lines.append(
            f"  Profitable regimes: {regime_result.profitable_regimes}/"
            f"{regime_result.regimes_tested}"
        )
        lines.append(f"  Best: {regime_result.best_regime}")
        lines.append(f"  Worst: {regime_result.worst_regime}")
        lines.append(f"  Regime dependent: {'Yes' if regime_result.is_regime_dependent else 'No'}")
    else:
        lines.append("  Skipped (no regime data)")
    lines.append("")

    # Concerns
    lines.append("--- Concerns ---")
    for c in concerns:
        lines.append(f"  • {c}")
    lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
