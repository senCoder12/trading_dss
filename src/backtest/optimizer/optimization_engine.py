"""
Optimization Engine — Step 7.2 of the Trading Decision Support System.

Systematically evaluates parameter combinations via full backtest runs,
ranks results by a configurable objective, applies constraint filters,
and optionally validates top candidates with walk-forward analysis.

Usage
-----
::

    engine = OptimizationEngine(db)
    config = OptimizationConfig(
        index_id="NIFTY50",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        parameter_space=space,
        base_simulator_config=SimulatorConfig(),
    )
    result = engine.run(config)
    print(result.get_summary())
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from src.backtest.metrics import BacktestMetrics, MetricsCalculator
from src.backtest.optimizer.param_space import (
    ParameterApplicator,
    ParameterSpace,
)
from src.backtest.strategy_runner import BacktestConfig, BacktestResult, StrategyRunner
from src.backtest.trade_simulator import SimulatorConfig
from src.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardValidator,
)
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Metrics where lower is better (inverted during ranking).
_LOWER_IS_BETTER = {"max_drawdown_pct", "max_drawdown_amount", "max_drawdown_duration_bars"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _params_to_key(params: dict[str, Any]) -> str:
    """Deterministic hash for a parameter combination."""
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Outcome of a single parameter-combination backtest."""

    params: dict[str, Any]
    metrics: Optional[BacktestMetrics] = None
    passes_constraints: bool = False
    backtest_result: Optional[BacktestResult] = None
    error: Optional[str] = None
    rank: Optional[int] = None


@dataclass
class OptimizationConfig:
    """Full specification for an optimization run."""

    # What to optimize
    index_id: str
    start_date: date
    end_date: date
    timeframe: str = "1d"
    mode: str = "TECHNICAL_ONLY"

    # Base config (parameters not being optimized stay at these values)
    base_simulator_config: SimulatorConfig = field(default_factory=SimulatorConfig)

    # Search space
    parameter_space: ParameterSpace = field(default_factory=lambda: ParameterSpace(parameters=[]))

    # Search strategy
    search_method: str = "grid"
    max_evaluations: Optional[int] = None
    random_seed: int = 42

    # Objective function
    primary_objective: str = "sharpe_ratio"
    secondary_objectives: list[str] = field(
        default_factory=lambda: ["win_rate", "max_drawdown_pct"]
    )

    # Constraints
    min_trades: int = 30
    max_drawdown_limit: float = 25.0
    min_win_rate: float = 40.0
    min_profit_factor: float = 0.9

    # Walk-forward validation
    run_walk_forward: bool = True
    walk_forward_top_n: int = 5
    wf_train_days: int = 252
    wf_test_days: int = 63

    # Progress
    show_progress: bool = True
    progress_interval: int = 10


@dataclass
class OptimizationResult:
    """Aggregated output of an optimization run."""

    config: OptimizationConfig

    # Counts
    total_evaluations: int = 0
    passed_constraints: int = 0
    failed_constraints: int = 0
    errors: int = 0

    # Results
    ranked_results: list[EvaluationResult] = field(default_factory=list)
    all_results: list[EvaluationResult] = field(default_factory=list)
    walk_forward_results: dict[str, WalkForwardResult] = field(default_factory=dict)

    # Best
    best_params: Optional[dict[str, Any]] = None
    best_metrics: Optional[BacktestMetrics] = None
    best_wf_result: Optional[WalkForwardResult] = None

    # Baseline comparison
    default_params_result: Optional[EvaluationResult] = None
    optimization_lift_pct: Optional[float] = None

    # Timing
    optimization_duration_seconds: float = 0.0

    def get_summary(self) -> str:
        """Human-readable optimization summary."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("OPTIMIZATION RESULTS")
        lines.append("=" * 60)

        # Timing
        mins = self.optimization_duration_seconds / 60
        lines.append(f"Duration: {mins:.1f} minutes")
        lines.append(
            f"Evaluations: {self.total_evaluations} total | "
            f"{self.passed_constraints} passed | "
            f"{self.failed_constraints} failed | "
            f"{self.errors} errors"
        )
        lines.append("")

        if not self.ranked_results:
            lines.append("NO VIABLE PARAMETERS FOUND")
            lines.append("")
            if self.all_results:
                # Explain why nothing passed
                no_trades = sum(
                    1 for r in self.all_results
                    if r.metrics and r.metrics.total_trades < self.config.min_trades
                )
                high_dd = sum(
                    1 for r in self.all_results
                    if r.metrics and r.metrics.max_drawdown_pct > self.config.max_drawdown_limit
                )
                low_wr = sum(
                    1 for r in self.all_results
                    if r.metrics and r.metrics.win_rate < self.config.min_win_rate
                )
                low_pf = sum(
                    1 for r in self.all_results
                    if r.metrics and r.metrics.profit_factor < self.config.min_profit_factor
                )
                lines.append("Constraint violations:")
                lines.append(f"  Too few trades (<{self.config.min_trades}): {no_trades}")
                lines.append(f"  Drawdown too high (>{self.config.max_drawdown_limit}%): {high_dd}")
                lines.append(f"  Win rate too low (<{self.config.min_win_rate}%): {low_wr}")
                lines.append(f"  Profit factor too low (<{self.config.min_profit_factor}): {low_pf}")
            lines.append("=" * 60)
            return "\n".join(lines)

        # Best result
        lines.append("BEST PARAMETERS:")
        for k, v in (self.best_params or {}).items():
            lines.append(f"  {k}: {v}")
        lines.append("")

        if self.best_metrics:
            m = self.best_metrics
            lines.append("BEST METRICS:")
            lines.append(f"  Total Return:   {m.total_return_pct:+.2f}%")
            lines.append(f"  Sharpe Ratio:   {m.sharpe_ratio:.3f}")
            lines.append(f"  Win Rate:       {m.win_rate:.1f}%")
            lines.append(f"  Profit Factor:  {m.profit_factor:.2f}")
            lines.append(f"  Max Drawdown:   {m.max_drawdown_pct:.2f}%")
            lines.append(f"  Total Trades:   {m.total_trades}")
            lines.append(f"  EV/Trade:       {m.expected_value_per_trade:.2f}")
            lines.append("")

        # Baseline comparison
        if self.default_params_result and self.default_params_result.metrics:
            dm = self.default_params_result.metrics
            lines.append("DEFAULT (BASELINE) METRICS:")
            lines.append(f"  Total Return:   {dm.total_return_pct:+.2f}%")
            lines.append(f"  Sharpe Ratio:   {dm.sharpe_ratio:.3f}")
            lines.append(f"  Win Rate:       {dm.win_rate:.1f}%")
            lines.append("")
            if self.optimization_lift_pct is not None:
                lines.append(f"OPTIMIZATION LIFT: {self.optimization_lift_pct:+.1f}%")
                lines.append("")

        # Walk-forward
        if self.best_wf_result:
            wf = self.best_wf_result
            lines.append("WALK-FORWARD VALIDATION (best params):")
            lines.append(f"  Windows:            {wf.total_windows}")
            lines.append(f"  Profitable windows: {wf.profitable_test_windows}/{wf.total_windows}")
            lines.append(f"  Avg test return:    {wf.avg_test_return:+.2f}%")
            lines.append(f"  Overfitting:        {wf.overfitting_assessment}")
            lines.append(f"  Robust:             {'Yes' if wf.is_robust else 'No'}")
            lines.append(f"  Verdict:            {wf.verdict}")
            lines.append("")

        # Top 5
        top_n = min(5, len(self.ranked_results))
        lines.append(f"TOP {top_n} RESULTS:")
        lines.append(f"  {'Rank':<5} {'Return':>10} {'Sharpe':>8} {'WinRate':>8} {'PF':>6} {'DD':>6} {'Trades':>7}")
        lines.append(f"  {'-'*4:<5} {'-'*9:>10} {'-'*7:>8} {'-'*7:>8} {'-'*5:>6} {'-'*5:>6} {'-'*6:>7}")
        for r in self.ranked_results[:top_n]:
            if r.metrics:
                m = r.metrics
                lines.append(
                    f"  {r.rank or 0:<5} "
                    f"{m.total_return_pct:>+9.2f}% "
                    f"{m.sharpe_ratio:>7.3f} "
                    f"{m.win_rate:>7.1f}% "
                    f"{m.profit_factor:>5.2f} "
                    f"{m.max_drawdown_pct:>5.1f}% "
                    f"{m.total_trades:>6}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OptimizationEngine:
    """Evaluates parameter combinations via full backtest runs."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.strategy_runner = StrategyRunner(db)
        self.metrics_calculator = MetricsCalculator()
        self.applicator = ParameterApplicator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, config: OptimizationConfig) -> OptimizationResult:
        """Execute the full optimization pipeline.

        Steps: generate combos -> filter invalid -> backtest each ->
        rank -> walk-forward validate top N -> build result.
        """
        start_time = time.time()

        # --- Step 1: Generate parameter combinations -----------------------
        combos = self._generate_combinations(config)

        # --- Step 2: Filter invalid combinations ---------------------------
        base_config = BacktestConfig(
            index_id=config.index_id,
            start_date=config.start_date,
            end_date=config.end_date,
            timeframe=config.timeframe,
            mode=config.mode,
            simulator_config=config.base_simulator_config,
            show_progress=False,
        )

        valid_combos: list[tuple[dict[str, Any], BacktestConfig]] = []
        for combo in combos:
            applied = self.applicator.apply(base_config, combo)
            if applied is not None:
                valid_combos.append((combo, applied))

        logger.info(
            "Optimization: %d valid combinations (filtered %d invalid)",
            len(valid_combos),
            len(combos) - len(valid_combos),
        )

        if config.show_progress:
            print(
                f"\nOptimization: {len(valid_combos)} valid combinations "
                f"(filtered {len(combos) - len(valid_combos)} invalid)"
            )

        # --- Step 3: Run backtests -----------------------------------------
        results: list[EvaluationResult] = []
        try:
            self._run_backtests(valid_combos, config, start_time, results)
        except KeyboardInterrupt:
            logger.warning(
                "Optimization interrupted after %d/%d evaluations",
                len(results),
                len(valid_combos),
            )
            if config.show_progress:
                print(f"\n  Interrupted! Returning partial results ({len(results)} evaluations)")

        # --- Step 4: Rank results ------------------------------------------
        passing = [r for r in results if r.passes_constraints and r.metrics is not None]
        failing = [r for r in results if not r.passes_constraints]
        error_count = sum(1 for r in results if r.error is not None)

        ranked = self._rank_results(
            passing, config.primary_objective, config.secondary_objectives,
        )

        logger.info(
            "Optimization complete: %d passed constraints, %d failed, %d errors",
            len(passing),
            len(failing),
            error_count,
        )

        # --- Step 5: Walk-forward validation on top N ----------------------
        wf_results: dict[str, WalkForwardResult] = {}
        if config.run_walk_forward and ranked:
            wf_results = self._run_walk_forward(ranked, config)

        # --- Step 6: Build final result ------------------------------------
        default_result = self._find_default_result(results, config.parameter_space)

        best_params = ranked[0].params if ranked else None
        best_metrics = ranked[0].metrics if ranked else None
        best_key = _params_to_key(best_params) if best_params else None
        best_wf = wf_results.get(best_key) if best_key else None

        # Compute optimization lift
        lift: Optional[float] = None
        if (
            best_metrics
            and default_result
            and default_result.metrics
            and abs(default_result.metrics.total_return_pct) > 1e-9
        ):
            lift = (
                (best_metrics.total_return_pct - default_result.metrics.total_return_pct)
                / abs(default_result.metrics.total_return_pct)
                * 100
            )

        result = OptimizationResult(
            config=config,
            total_evaluations=len(results),
            passed_constraints=len(passing),
            failed_constraints=len(failing),
            errors=error_count,
            ranked_results=ranked,
            all_results=results,
            walk_forward_results=wf_results,
            best_params=best_params,
            best_metrics=best_metrics,
            best_wf_result=best_wf,
            default_params_result=default_result,
            optimization_lift_pct=lift,
            optimization_duration_seconds=time.time() - start_time,
        )

        if config.show_progress:
            print(f"\n{result.get_summary()}")

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_combinations(
        self, config: OptimizationConfig,
    ) -> list[dict[str, Any]]:
        """Generate parameter combinations based on search method."""
        space = config.parameter_space

        if config.search_method == "grid":
            return space.generate_grid()

        if config.search_method == "random":
            n = config.max_evaluations or min(space.total_combinations, 500)
            return space.generate_random_samples(n, config.random_seed)

        raise ValueError(
            f"Unknown search_method {config.search_method!r}. "
            f"Must be 'grid' or 'random'."
        )

    def _run_backtests(
        self,
        valid_combos: list[tuple[dict[str, Any], BacktestConfig]],
        config: OptimizationConfig,
        start_time: float,
        results: list[EvaluationResult],
    ) -> None:
        """Run a backtest for each valid parameter combination.

        Appends to *results* in-place so the caller retains partial
        results on KeyboardInterrupt.
        """
        for i, (params, bt_config) in enumerate(valid_combos):
            try:
                bt_result = self.strategy_runner.run(bt_config)

                metrics = bt_result.metrics
                if metrics is None and bt_result.trade_history:
                    metrics = MetricsCalculator.calculate_all(
                        bt_result.trade_history,
                        bt_result.equity_curve,
                        bt_config.simulator_config.initial_capital,
                    )

                passes = self._check_constraints(metrics, config) if metrics else False

                results.append(EvaluationResult(
                    params=params,
                    metrics=metrics,
                    passes_constraints=passes,
                    backtest_result=bt_result,
                ))

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error("Backtest failed for params %s: %s", params, e)
                results.append(EvaluationResult(
                    params=params,
                    metrics=None,
                    passes_constraints=False,
                    error=str(e),
                ))

            # Progress
            if config.show_progress and (i + 1) % config.progress_interval == 0:
                self._print_progress(i + 1, len(valid_combos), start_time, results, config)

    @staticmethod
    def _check_constraints(
        metrics: BacktestMetrics, config: OptimizationConfig,
    ) -> bool:
        """Return True if metrics pass all configured constraints."""
        return (
            metrics.total_trades >= config.min_trades
            and metrics.max_drawdown_pct <= config.max_drawdown_limit
            and metrics.win_rate >= config.min_win_rate
            and metrics.profit_factor >= config.min_profit_factor
        )

    @staticmethod
    def _print_progress(
        done: int,
        total: int,
        start_time: float,
        results: list[EvaluationResult],
        config: OptimizationConfig,
    ) -> None:
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0

        best = OptimizationEngine._get_best(results, config.primary_objective)
        best_val = (
            getattr(best.metrics, config.primary_objective, None)
            if best and best.metrics
            else None
        )

        print(
            f"  [{done}/{total}] "
            f"({done / total * 100:.0f}%) "
            f"ETA: {remaining / 60:.1f}min | "
            f"Best {config.primary_objective}: {best_val}"
        )

    @staticmethod
    def _get_best(
        results: list[EvaluationResult], objective: str,
    ) -> Optional[EvaluationResult]:
        """Return the best-so-far result for the given objective."""
        viable = [r for r in results if r.passes_constraints and r.metrics is not None]
        if not viable:
            return None

        reverse = objective not in _LOWER_IS_BETTER
        return max(
            viable,
            key=lambda r: getattr(r.metrics, objective, 0) * (1 if reverse else -1),
        )

    def _rank_results(
        self,
        results: list[EvaluationResult],
        primary_obj: str,
        secondary_objs: list[str],
    ) -> list[EvaluationResult]:
        """Sort results by primary objective with secondary tiebreakers.

        Ranking score = primary_normalised * 0.7 + secondary_normalised * 0.3
        """
        if not results:
            return []

        # Normalise a list of values to [0, 1].
        def _normalise(values: list[float]) -> list[float]:
            lo, hi = min(values), max(values)
            if hi - lo < 1e-12:
                return [0.5] * len(values)
            return [(v - lo) / (hi - lo) for v in values]

        def _raw(r: EvaluationResult, attr: str) -> float:
            val = getattr(r.metrics, attr, 0.0) if r.metrics else 0.0
            return -val if attr in _LOWER_IS_BETTER else val

        # Primary scores
        primary_raw = [_raw(r, primary_obj) for r in results]
        primary_norm = _normalise(primary_raw)

        # Secondary scores (average of normalised secondaries)
        sec_norms: list[list[float]] = []
        for obj in secondary_objs:
            raw = [_raw(r, obj) for r in results]
            sec_norms.append(_normalise(raw))

        secondary_avg = [0.0] * len(results)
        if sec_norms:
            for i in range(len(results)):
                secondary_avg[i] = sum(sn[i] for sn in sec_norms) / len(sec_norms)

        # Composite score
        scores = [
            p * 0.7 + s * 0.3
            for p, s in zip(primary_norm, secondary_avg)
        ]

        # Sort descending by score
        paired = list(zip(scores, results))
        paired.sort(key=lambda x: x[0], reverse=True)

        ranked = [r for _, r in paired]
        for i, r in enumerate(ranked):
            r.rank = i + 1

        return ranked

    def _run_walk_forward(
        self,
        ranked: list[EvaluationResult],
        config: OptimizationConfig,
    ) -> dict[str, WalkForwardResult]:
        """Run walk-forward validation on the top N ranked results."""
        wf_results: dict[str, WalkForwardResult] = {}
        top_n = ranked[: config.walk_forward_top_n]

        for rank_idx, eval_result in enumerate(top_n, 1):
            if config.show_progress:
                print(f"\n  Walk-forward validating rank #{rank_idx}...")

            try:
                applied_config = self.applicator.apply(
                    BacktestConfig(
                        index_id=config.index_id,
                        start_date=config.start_date,
                        end_date=config.end_date,
                        timeframe=config.timeframe,
                        mode=config.mode,
                        simulator_config=config.base_simulator_config,
                    ),
                    eval_result.params,
                )

                if applied_config is None:
                    logger.warning(
                        "Walk-forward: params for rank #%d produced invalid config, skipping",
                        rank_idx,
                    )
                    continue

                wf_config = WalkForwardConfig(
                    index_id=config.index_id,
                    full_start_date=config.start_date,
                    full_end_date=config.end_date,
                    timeframe=config.timeframe,
                    train_window_days=config.wf_train_days,
                    test_window_days=config.wf_test_days,
                    simulator_config=applied_config.simulator_config,
                    mode=config.mode,
                )

                validator = WalkForwardValidator(self.db)
                wf_result = validator.run_walk_forward(wf_config)
                wf_results[_params_to_key(eval_result.params)] = wf_result

            except Exception as e:
                logger.error(
                    "Walk-forward failed for rank #%d: %s", rank_idx, e,
                )

        return wf_results

    def _find_default_result(
        self,
        results: list[EvaluationResult],
        space: ParameterSpace,
    ) -> Optional[EvaluationResult]:
        """Find the result that used default parameter values (baseline)."""
        defaults = space.get_default_values()
        if not defaults:
            return None

        for r in results:
            if r.params == defaults:
                return r

        return None
