"""
Optimization Engine — Step 7.2 of the Trading Decision Support System.

Systematically evaluates parameter combinations via full backtest runs,
ranks results by a configurable objective, applies constraint filters,
and optionally validates top candidates with walk-forward analysis.

Statistical safeguards
----------------------
* The date range is split into a SELECTION period and a final untouched
  HOLDOUT (``holdout_fraction``, separated by ``embargo_days``).  Parameters
  are chosen on the selection period only; the chosen configuration is then
  evaluated on the holdout exactly once.
* Every ranked configuration carries its Deflated Sharpe Ratio given the
  number of trials evaluated (Bailey & López de Prado 2014).  A configuration
  is labelled *approved* only if it passes the constraints, its DSR is at least
  ``dsr_threshold`` (0.95) and — when a holdout exists — it is profitable on
  the holdout.
* ``run_nested_walk_forward`` re-runs the whole selection procedure inside
  each walk-forward fold (fit = grid search on the fold's train window), which
  is the out-of-sample estimate of the optimiser itself.
* The legacy ``run_walk_forward`` (fixed config per window) is kept, runs on
  the selection period with ALL optimised parameters, and is reported as an
  in-sample stability check.

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
import math
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import numpy as np

from src.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    BacktestMetrics,
    MetricsCalculator,
    is_defined,
    session_days,
)
from src.backtest.overfitting import (
    DeflatedSharpeResult,
    PBOResult,
    deflate_selected_sharpe,
    probability_of_backtest_overfitting,
    return_moments,
    sharpe_ratio as per_period_sharpe,
)
from src.backtest.optimizer.param_space import (
    ParameterApplicator,
    ParameterSpace,
)
from src.backtest.strategy_runner import BacktestConfig, BacktestResult, StrategyRunner
from src.backtest.trade_simulator import SimulatorConfig
from src.backtest.walk_forward import (
    NestedWalkForwardResult,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardValidator,
    nested_walk_forward,
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

    # Multiple-testing assessment (filled for ranked results)
    dsr: Optional[float] = None
    psr: Optional[float] = None
    dsr_detail: Optional[DeflatedSharpeResult] = None
    approved: bool = False


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

    # Walk-forward validation (legacy fixed-config stability check; runs on
    # the selection period only, with all optimised parameters)
    run_walk_forward: bool = True
    walk_forward_top_n: int = 5
    wf_train_days: int = 252
    wf_test_days: int = 63

    # Out-of-sample protocol
    holdout_fraction: float = 0.2      # last share of trading days kept untouched
    embargo_days: int = 5              # gap between selection period and holdout
    run_nested_walk_forward: bool = False   # OOS estimate of the whole procedure
    nested_wf_folds: int = 4
    nested_wf_train_days: Optional[int] = None  # None → expanding window

    # Multiple-testing correction
    dsr_threshold: float = 0.95
    n_trials_override: Optional[int] = None    # effective # of independent trials
    pbo_n_splits: int = 8                       # CSCV splits (0 disables PBO)

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

    # Out-of-sample protocol
    selection_start: Optional[date] = None
    selection_end: Optional[date] = None
    holdout_start: Optional[date] = None
    holdout_end: Optional[date] = None
    holdout_result: Optional[EvaluationResult] = None
    holdout_passed: Optional[bool] = None
    nested_wf_result: Optional[NestedWalkForwardResult] = None

    # Multiple-testing assessment
    n_trials: int = 0
    best_dsr: Optional[DeflatedSharpeResult] = None
    best_approved: bool = False
    approval_notes: list[str] = field(default_factory=list)
    pbo_result: Optional[PBOResult] = None

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
                    if r.metrics and (
                        not is_defined(r.metrics, "profit_factor")
                        or r.metrics.profit_factor < self.config.min_profit_factor
                    )
                )
                lines.append("Constraint violations:")
                lines.append(f"  Too few trades (<{self.config.min_trades}): {no_trades}")
                lines.append(f"  Drawdown too high (>{self.config.max_drawdown_limit}%): {high_dd}")
                lines.append(f"  Win rate too low (<{self.config.min_win_rate}%): {low_wr}")
                lines.append(
                    f"  Profit factor too low (<{self.config.min_profit_factor}) "
                    f"or undefined (no losses): {low_pf}"
                )
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

        # Out-of-sample protocol & multiple testing
        if self.selection_start and self.selection_end:
            line = f"SELECTION PERIOD: {self.selection_start} → {self.selection_end}"
            if self.holdout_start and self.holdout_end:
                line += (
                    f" | HOLDOUT: {self.holdout_start} → {self.holdout_end} "
                    f"(embargo {self.config.embargo_days}d)"
                )
            lines.append(line)
            lines.append("")

        lines.append("MULTIPLE-TESTING CORRECTION (best params):")
        lines.append(f"  Trials evaluated (N):   {self.n_trials}")
        if self.best_dsr is not None:
            d = self.best_dsr
            lines.append(f"  Sharpe (annualised):    {d.sharpe_annualised:.3f} on {d.n_obs} daily obs")
            lines.append(f"  E[max SR] of N trials:  {d.expected_max_sharpe * math.sqrt(TRADING_DAYS_PER_YEAR):.3f} (annualised)")
            lines.append(f"  PSR (vs 0, uncorrected): {d.psr:.3f}")
            lines.append(
                f"  Deflated Sharpe (DSR): {d.dsr:.3f}  [threshold {d.threshold:.2f}]"
                + ("  (normal-moment approximation)" if d.assumed_normal else "")
            )
            mintrl = d.min_track_record_length
            lines.append(
                f"  Min track record:       "
                + (f"{mintrl:.0f} obs" if math.isfinite(mintrl) else "unattainable (SR ≤ E[max SR])")
            )
        else:
            lines.append("  DSR: unavailable")
        if self.pbo_result is not None:
            lines.append(
                f"  PBO (CSCV, S={self.pbo_result.n_splits}): {self.pbo_result.pbo:.2f} "
                f"| OOS-loss prob {self.pbo_result.prob_oos_loss:.2f}"
            )
        lines.append(
            f"  APPROVAL: {'APPROVED' if self.best_approved else 'NOT APPROVED'}"
        )
        for note in self.approval_notes:
            lines.append(f"    - {note}")
        lines.append("")

        if self.holdout_result is not None and self.holdout_result.metrics is not None:
            hm = self.holdout_result.metrics
            lines.append("HOLDOUT (untouched, evaluated once):")
            lines.append(f"  Total Return:   {hm.total_return_pct:+.2f}%")
            lines.append(f"  Sharpe Ratio:   {hm.sharpe_ratio:.3f}")
            lines.append(f"  Win Rate:       {hm.win_rate:.1f}%")
            lines.append(f"  Max Drawdown:   {hm.max_drawdown_pct:.2f}%")
            lines.append(f"  Total Trades:   {hm.total_trades}")
            lines.append("")

        if self.nested_wf_result is not None:
            lines.append("NESTED WALK-FORWARD (out-of-sample estimate of the selection procedure):")
            lines.append(self.nested_wf_result.summary())
            lines.append("")

        # Walk-forward (legacy fixed-config)
        if self.best_wf_result:
            wf = self.best_wf_result
            lines.append(
                "WALK-FORWARD VALIDATION (best params, selection period; fixed "
                "config — in-sample stability check, NOT out-of-sample):"
            )
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

        # --- Step 0: Selection period / untouched holdout ------------------
        sel_start, sel_end, hold_start, hold_end = self._split_selection_holdout(config)

        # --- Step 1: Generate parameter combinations -----------------------
        combos = self._generate_combinations(config)

        # --- Step 2: Filter invalid combinations ---------------------------
        base_config = self._base_backtest_config(config, sel_start, sel_end)

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
        interrupted = False
        try:
            self._run_backtests(valid_combos, config, start_time, results)
        except KeyboardInterrupt:
            interrupted = True
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

        # --- Step 4b: Multiple-testing correction -------------------------
        n_trials = self._attach_dsr(results, ranked, config)
        pbo_result = self._compute_pbo(results, config)

        # --- Step 5: Walk-forward validation on top N ----------------------
        wf_results: dict[str, WalkForwardResult] = {}
        if config.run_walk_forward and ranked and not interrupted:
            wf_results = self._run_walk_forward(ranked, config, sel_start, sel_end)

        # --- Step 5b: Untouched holdout, evaluated ONCE for the best -------
        holdout_result: Optional[EvaluationResult] = None
        holdout_passed: Optional[bool] = None
        if ranked and hold_start is not None and not interrupted:
            holdout_result = self._evaluate_holdout(ranked[0], config, hold_start, hold_end)
            if holdout_result is not None:
                holdout_passed = holdout_result.passes_constraints

        # --- Step 5c: Nested walk-forward of the selection procedure -------
        nested_result: Optional[NestedWalkForwardResult] = None
        if config.run_nested_walk_forward and valid_combos and not interrupted:
            try:
                nested_result = self.run_nested_walk_forward(
                    config, [c for c, _ in valid_combos], sel_start, sel_end,
                )
            except Exception as e:  # noqa: BLE001 — never fail the main run
                logger.error("Nested walk-forward failed: %s", e)

        # --- Step 5d: Approval label ---------------------------------------
        best_approved, approval_notes = self._approval(
            ranked[0] if ranked else None, config,
            holdout_configured=hold_start is not None,
            holdout_result=holdout_result,
            holdout_passed=holdout_passed,
        )
        if ranked:
            ranked[0].approved = best_approved

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
            selection_start=sel_start,
            selection_end=sel_end,
            holdout_start=hold_start,
            holdout_end=hold_end,
            holdout_result=holdout_result,
            holdout_passed=holdout_passed,
            nested_wf_result=nested_result,
            n_trials=n_trials,
            best_dsr=ranked[0].dsr_detail if ranked else None,
            best_approved=best_approved,
            approval_notes=approval_notes,
            pbo_result=pbo_result,
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
        """Return True if metrics pass all configured constraints.

        An undefined profit factor (no losing trades → inf) FAILS the check:
        it is a symptom of too few trades, not of a strong edge.
        """
        if not is_defined(metrics, "profit_factor"):
            return False
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

        sign = 1 if objective not in _LOWER_IS_BETTER else -1

        def _key(r: EvaluationResult) -> float:
            if not is_defined(r.metrics, objective):
                return -math.inf
            return getattr(r.metrics, objective) * sign

        return max(viable, key=_key)

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

        # Normalise a list of values to [0, 1]; undefined (NaN) → 0 (worst).
        def _normalise(values: list[float]) -> list[float]:
            finite = [v for v in values if math.isfinite(v)]
            if not finite:
                return [0.0] * len(values)
            lo, hi = min(finite), max(finite)
            if hi - lo < 1e-12:
                return [0.5 if math.isfinite(v) else 0.0 for v in values]
            return [(v - lo) / (hi - lo) if math.isfinite(v) else 0.0 for v in values]

        def _raw(r: EvaluationResult, attr: str) -> float:
            if r.metrics is None or not is_defined(r.metrics, attr):
                return math.nan
            val = getattr(r.metrics, attr)
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
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> dict[str, WalkForwardResult]:
        """Run the (fixed-config) walk-forward on the top N ranked results.

        Runs over the SELECTION period only (never the holdout) and passes the
        fully applied BacktestConfig so that every optimised parameter —
        including BacktestConfig-level ones such as ``min_confidence`` and
        ``signal_cooldown_bars`` — is used in each window.
        """
        wf_results: dict[str, WalkForwardResult] = {}
        top_n = ranked[: config.walk_forward_top_n]
        start = start or config.start_date
        end = end or config.end_date

        for rank_idx, eval_result in enumerate(top_n, 1):
            if config.show_progress:
                print(f"\n  Walk-forward validating rank #{rank_idx}...")

            try:
                applied_config = self.applicator.apply(
                    self._base_backtest_config(config, start, end),
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
                    full_start_date=start,
                    full_end_date=end,
                    timeframe=applied_config.timeframe,
                    train_window_days=config.wf_train_days,
                    test_window_days=config.wf_test_days,
                    simulator_config=applied_config.simulator_config,
                    mode=applied_config.mode,
                    min_confidence=applied_config.min_confidence,
                    warmup_bars=applied_config.warmup_bars,
                    backtest_template=applied_config,
                )

                validator = WalkForwardValidator(self.db)
                wf_result = validator.run_walk_forward(wf_config)
                wf_results[_params_to_key(eval_result.params)] = wf_result

            except Exception as e:
                logger.error(
                    "Walk-forward failed for rank #%d: %s", rank_idx, e,
                )

        return wf_results

    # ------------------------------------------------------------------
    # Out-of-sample protocol & multiple-testing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _base_backtest_config(
        config: OptimizationConfig, start: date, end: date,
    ) -> BacktestConfig:
        return BacktestConfig(
            index_id=config.index_id,
            start_date=start,
            end_date=end,
            timeframe=config.timeframe,
            mode=config.mode,
            simulator_config=config.base_simulator_config,
            show_progress=False,
        )

    @staticmethod
    def _split_selection_holdout(
        config: OptimizationConfig,
    ) -> tuple[date, date, Optional[date], Optional[date]]:
        """Return (selection_start, selection_end, holdout_start, holdout_end).

        The holdout is the last ``ceil(holdout_fraction · n)`` trading days
        (Mon–Fri) of the range; the selection period ends ``embargo_days``
        trading days before it.  No holdout when ``holdout_fraction == 0``.
        """
        f = config.holdout_fraction
        if not 0.0 <= f < 1.0:
            raise ValueError(f"holdout_fraction must be in [0, 1), got {f}")
        if f == 0.0:
            return config.start_date, config.end_date, None, None
        days = session_days(config.start_date, config.end_date)
        n_hold = int(math.ceil(len(days) * f))
        n_sel = len(days) - n_hold - max(0, config.embargo_days)
        if n_hold < 1 or n_sel < 2:
            raise ValueError(
                f"Date range {config.start_date}→{config.end_date} ({len(days)} trading "
                f"days) is too short for holdout_fraction={f} with "
                f"embargo_days={config.embargo_days}"
            )
        return config.start_date, days[n_sel - 1], days[len(days) - n_hold], config.end_date

    @staticmethod
    def _trial_sharpe(r: EvaluationResult) -> Optional[tuple[float, int, float, float, bool]]:
        """(per-period SR, n_obs, skew, kurtosis, assumed_normal) for a trial."""
        m = r.metrics
        if m is None:
            return None
        rets = list(getattr(m, "daily_returns", None) or [])
        ppy = getattr(m, "periods_per_year", TRADING_DAYS_PER_YEAR) or TRADING_DAYS_PER_YEAR
        if len(rets) >= 2:
            rf_p = (getattr(m, "risk_free_rate", 0.0) or 0.0) / ppy
            sr = per_period_sharpe(rets, rf_per_period=rf_p)
            skew, kurt = return_moments(rets)
            return sr, len(rets), skew, kurt, False
        # Fallback (no return series): de-annualise the reported Sharpe and
        # assume normal returns over the backtest's trading days.
        n_obs = 0
        if r.backtest_result is not None:
            n_obs = int(getattr(r.backtest_result, "trading_days", 0) or 0)
        sharpe = m.sharpe_ratio if is_defined(m, "sharpe_ratio") else 0.0
        if n_obs < 2 or not isinstance(sharpe, (int, float)):
            return None
        return float(sharpe) / math.sqrt(ppy), n_obs, 0.0, 3.0, True

    def _attach_dsr(
        self,
        results: list[EvaluationResult],
        ranked: list[EvaluationResult],
        config: OptimizationConfig,
    ) -> int:
        """Attach PSR/DSR to every ranked result; return the trial count N.

        N counts every configuration that was backtested (including those that
        failed constraints) — all of them were "tried" by the search.
        """
        trials = [t for t in (self._trial_sharpe(r) for r in results) if t is not None]
        n_trials = config.n_trials_override or sum(1 for r in results if r.metrics is not None)
        trial_srs = [t[0] for t in trials]
        for r in ranked:
            t = self._trial_sharpe(r)
            if t is None or n_trials < 1:
                r.dsr = r.psr = None
                r.dsr_detail = None
                r.approved = False
                continue
            sr, n_obs, skew, kurt, assumed = t
            detail = deflate_selected_sharpe(
                sr, n_obs, trial_srs, n_trials=n_trials, skew=skew, kurtosis=kurt,
                threshold=config.dsr_threshold, assumed_normal=assumed,
            )
            r.dsr_detail = detail
            r.dsr = detail.dsr
            r.psr = detail.psr
            r.approved = bool(r.passes_constraints and detail.passes)
        return int(n_trials)

    @staticmethod
    def _compute_pbo(
        results: list[EvaluationResult], config: OptimizationConfig,
    ) -> Optional[PBOResult]:
        """CSCV PBO across trials sharing an identical daily-return calendar."""
        if config.pbo_n_splits <= 0:
            return None
        series = [
            list(r.metrics.daily_returns) for r in results
            if r.metrics is not None and getattr(r.metrics, "daily_returns", None)
        ]
        if len(series) < 2:
            return None
        lengths: dict[int, int] = {}
        for srs in series:
            lengths[len(srs)] = lengths.get(len(srs), 0) + 1
        t_len = max(lengths, key=lambda k: lengths[k])
        cols = [srs for srs in series if len(srs) == t_len]
        if len(cols) < 2 or t_len < 2 * config.pbo_n_splits:
            return None
        try:
            return probability_of_backtest_overfitting(
                np.asarray(cols, dtype=float).T, n_splits=config.pbo_n_splits,
            )
        except ValueError as e:
            logger.info("PBO skipped: %s", e)
            return None

    def _evaluate_holdout(
        self,
        best: EvaluationResult,
        config: OptimizationConfig,
        start: date,
        end: date,
    ) -> Optional[EvaluationResult]:
        """Backtest the selected params on the untouched holdout (once).

        Holdout passes if it is profitable and within the drawdown limit (the
        trade-count constraint is not applied to the shorter holdout).
        """
        applied = self.applicator.apply(self._base_backtest_config(config, start, end), best.params)
        if applied is None:
            return None
        try:
            bt = self.strategy_runner.run(applied)
            metrics = bt.metrics
            if metrics is None and bt.trade_history:
                metrics = MetricsCalculator.calculate_all(
                    bt.trade_history, bt.equity_curve, applied.simulator_config.initial_capital,
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("Holdout evaluation failed for %s: %s", best.params, e)
            return EvaluationResult(params=best.params, error=str(e))
        passed = bool(
            metrics is not None
            and metrics.total_trades > 0
            and metrics.total_return_pct > 0
            and metrics.max_drawdown_pct <= config.max_drawdown_limit
        )
        return EvaluationResult(
            params=best.params, metrics=metrics, passes_constraints=passed, backtest_result=bt,
        )

    @staticmethod
    def _approval(
        best: Optional[EvaluationResult],
        config: OptimizationConfig,
        holdout_configured: bool,
        holdout_result: Optional[EvaluationResult],
        holdout_passed: Optional[bool],
    ) -> tuple[bool, list[str]]:
        notes: list[str] = []
        if best is None:
            return False, ["No configuration passed the constraints"]
        ok = True
        if best.dsr is None:
            ok = False
            notes.append("DSR unavailable (no return series / observations)")
        elif best.dsr < config.dsr_threshold:
            ok = False
            notes.append(f"DSR {best.dsr:.3f} < {config.dsr_threshold:.2f}")
        else:
            notes.append(f"DSR {best.dsr:.3f} >= {config.dsr_threshold:.2f}")
        if holdout_configured:
            if holdout_result is None or holdout_result.metrics is None:
                ok = False
                notes.append("Holdout not evaluated")
            elif not holdout_passed:
                ok = False
                notes.append("Holdout not profitable or drawdown above limit")
            else:
                notes.append("Holdout profitable within drawdown limit")
        else:
            notes.append("No holdout reserved (holdout_fraction=0)")
        return ok and bool(best.passes_constraints), notes

    def run_nested_walk_forward(
        self,
        config: OptimizationConfig,
        combos: Optional[list[dict[str, Any]]] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> NestedWalkForwardResult:
        """Out-of-sample estimate of the WHOLE selection procedure.

        For each chronological fold the grid search is re-run on the fold's
        train window only (``fit_fn``), and the winning params are backtested
        on the fold's test window (``eval_fn``).  The optimiser's own holdout
        is excluded (``start``/``end`` default to the selection period), so the
        nested folds never see it.  Cost: folds × combos backtests.
        """
        if start is None or end is None:
            s0, s1, _, _ = self._split_selection_holdout(config)
            start, end = start or s0, end or s1
        if combos is None:
            combos = self._generate_combinations(config)
        days = session_days(start, end)

        def _backtest(params: dict[str, Any], d0: date, d1: date) -> Optional[BacktestMetrics]:
            applied = self.applicator.apply(self._base_backtest_config(config, d0, d1), params)
            if applied is None:
                return None
            bt = self.strategy_runner.run(applied)
            m = bt.metrics
            if m is None and bt.trade_history:
                m = MetricsCalculator.calculate_all(
                    bt.trade_history, bt.equity_curve, applied.simulator_config.initial_capital,
                )
            return m

        def fit_fn(train_days: list[date]) -> Optional[dict[str, Any]]:
            evals: list[EvaluationResult] = []
            for params in combos:
                try:
                    m = _backtest(params, train_days[0], train_days[-1])
                except Exception as e:  # noqa: BLE001
                    logger.debug("Nested fit backtest failed for %s: %s", params, e)
                    continue
                if m is not None and self._check_constraints(m, config):
                    evals.append(EvaluationResult(params=params, metrics=m, passes_constraints=True))
            ranked = self._rank_results(evals, config.primary_objective, config.secondary_objectives)
            return ranked[0].params if ranked else None

        def eval_fn(params: Optional[dict[str, Any]], test_days: list[date]) -> dict[str, Any]:
            if params is None:
                return {"no_viable_params": True}
            m = _backtest(params, test_days[0], test_days[-1])
            if m is None:
                return {"no_viable_params": True}
            out: dict[str, Any] = {
                "total_return_pct": m.total_return_pct,
                "sharpe_ratio": m.sharpe_ratio,
                "win_rate": m.win_rate,
                "max_drawdown_pct": m.max_drawdown_pct,
                "total_trades": m.total_trades,
            }
            if is_defined(m, "profit_factor"):
                out["profit_factor"] = m.profit_factor
            out["params"] = json.dumps(params, sort_keys=True, default=str)
            return out

        return nested_walk_forward(
            days,
            fit_fn,
            eval_fn,
            n_folds=config.nested_wf_folds,
            train_days=config.nested_wf_train_days,
            embargo_days=config.embargo_days,
            holdout_fraction=0.0,
            expanding=config.nested_wf_train_days is None,
        )

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
