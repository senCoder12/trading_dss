"""
Walk-Forward Validation Engine — Phase 6 Step 6.5.

Two tools live here:

1. :func:`nested_walk_forward` — a GENERIC, genuinely out-of-sample walk-forward.
   Chronological folds (train strictly before test, with an embargo gap),
   ``fit_fn(train_slice) -> model`` is called on training data only,
   ``eval_fn(model, test_slice) -> dict`` scores the fitted model on the test
   fold, and a final untouched HOLDOUT is evaluated exactly once with a model
   fitted on all pre-holdout data.  Use this for any out-of-sample claim.

2. :class:`WalkForwardValidator` (legacy API, kept for existing callers).
   It runs ONE fixed configuration on every train and test window — nothing is
   fitted on the train window — so its "train vs test degradation" and
   ``overfitting_score`` are a multi-period *stability* heuristic, not
   out-of-sample evidence.  Calling it emits
   :class:`FixedConfigWalkForwardWarning`.

Legacy usage
------------
::

    validator = WalkForwardValidator(db)
    config = WalkForwardConfig(
        index_id="NIFTY50",
        full_start_date=date(2023, 1, 1),
        full_end_date=date(2024, 12, 31),
        train_window_days=252,
        test_window_days=63,
    )
    result = validator.run_walk_forward(config)
    print(result.verdict)
"""

from __future__ import annotations

import dataclasses
import logging
import math
import statistics
import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Optional, Sequence

import numpy as np

from src.backtest.metrics import BacktestMetrics, MetricsCalculator, _to_date
from src.backtest.strategy_runner import BacktestConfig, StrategyRunner
from src.backtest.trade_simulator import SimulatorConfig
from src.database import queries as Q
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Parameters for a walk-forward validation run."""

    index_id: str
    full_start_date: date
    full_end_date: date
    timeframe: str = "1d"

    # Window sizing (in trading days)
    train_window_days: int = 252   # Default: 1 year
    test_window_days: int = 63     # Default: 1 quarter
    step_days: int = 63            # How far to advance each iteration

    # Strategy config applied to every window
    simulator_config: SimulatorConfig = field(default_factory=SimulatorConfig)
    mode: str = "TECHNICAL_ONLY"
    min_confidence: str = "MEDIUM"
    warmup_bars: int = 250         # Bars to warm up indicators before trading

    # Robustness checks
    parameter_sensitivity: bool = False  # If True, also test ±20% param variation

    # Gap (trading days) between each train window end and its test start.
    embargo_days: int = 0

    # Full BacktestConfig carrying every optimised parameter (min_confidence,
    # signal_cooldown_bars, …).  When given it is used for each window (dates
    # and simulator_config overridden) instead of rebuilding a BacktestConfig
    # from the few fields above — which silently dropped optimised params.
    backtest_template: Optional[BacktestConfig] = None


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WindowResult:
    """Train/test outcome for a single walk-forward window."""

    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    return_degradation_pct: float   # train_return - test_return (pp)
    win_rate_degradation: float     # train_win_rate - test_win_rate (pp)
    is_test_profitable: bool


@dataclass
class WalkForwardResult:
    """Aggregated outcome of a full walk-forward validation run."""

    config: WalkForwardConfig

    # Per-window detail
    windows: list[WindowResult]

    # Aggregate metrics
    total_windows: int
    profitable_test_windows: int
    test_profitability_rate: float       # fraction 0-1

    avg_train_return: float
    avg_test_return: float
    avg_degradation: float               # avg return drop train→test (pp)

    avg_train_win_rate: float
    avg_test_win_rate: float

    avg_train_sharpe: float
    avg_test_sharpe: float

    max_test_drawdown: float             # worst drawdown across all test windows

    # Combined test statistics (concatenated test periods)
    combined_test_return: float          # sum of per-window test returns
    combined_test_trades: int
    combined_test_win_rate: float

    # Overfitting assessment
    overfitting_score: float             # 0-1 (higher = more overfit)
    overfitting_assessment: str          # LOW_RISK / MODERATE_RISK / HIGH_RISK / SEVERE

    # Verdict
    is_robust: bool
    verdict: str

    # Methodology: False → fixed config on all windows (stability check only).
    is_out_of_sample: bool = False
    methodology_note: str = ""


class FixedConfigWalkForwardWarning(UserWarning):
    """Emitted when the legacy fixed-config walk-forward is used."""


_FIXED_CONFIG_NOTE = (
    "Fixed-config walk-forward: the SAME configuration is run on every train "
    "and test window and nothing is fitted on the train windows, so the "
    "train→test degradation and overfitting_score are a stability heuristic, "
    "NOT out-of-sample validation. Use nested_walk_forward() for OOS evidence."
)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Runs walk-forward validation across rolling train/test windows."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.strategy_runner = StrategyRunner(db)
        self.metrics_calculator = MetricsCalculator()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_walk_forward(self, config: WalkForwardConfig) -> WalkForwardResult:
        """Run the fixed-config multi-window check and return a ``WalkForwardResult``.

        .. deprecated::
            Not an out-of-sample test (see module docstring); emits
            :class:`FixedConfigWalkForwardWarning`.  Use
            :func:`nested_walk_forward` for out-of-sample validation.
        """
        warnings.warn(_FIXED_CONFIG_NOTE, FixedConfigWalkForwardWarning, stacklevel=2)
        logger.info(
            "Starting walk-forward: %s  %s → %s  train=%d test=%d step=%d",
            config.index_id,
            config.full_start_date,
            config.full_end_date,
            config.train_window_days,
            config.test_window_days,
            config.step_days,
        )

        # 1. Load actual trading dates from the database
        trading_days = self._get_trading_days(
            config.index_id,
            config.full_start_date,
            config.full_end_date,
        )

        min_needed = config.train_window_days + config.test_window_days
        if len(trading_days) < min_needed:
            raise ValueError(
                f"Not enough trading data for walk-forward validation on "
                f"{config.index_id}. Need at least {min_needed} trading days "
                f"({config.train_window_days} train + {config.test_window_days} test), "
                f"but only {len(trading_days)} available between "
                f"{config.full_start_date} and {config.full_end_date}."
            )

        # 2. Calculate window boundaries
        windows_def = self._calculate_windows(config, trading_days)

        if len(windows_def) < 2:
            total_days = len(trading_days)
            needed_for_two = (
                config.train_window_days + config.test_window_days + config.step_days
            )
            raise ValueError(
                f"Not enough data for at least 2 walk-forward windows "
                f"(got {len(windows_def)}). "
                f"Total trading days available: {total_days}. "
                f"Approx. need {needed_for_two} trading days for 2 windows. "
                f"Consider reducing train_window_days/test_window_days or "
                f"expanding the date range."
            )

        # 3. Run backtests for each window
        window_results, combined_test_trades = self._run_all_windows(
            config, windows_def
        )

        # 4. Aggregate and return
        return self._aggregate(config, window_results, combined_test_trades)

    # ------------------------------------------------------------------
    # Window calculation
    # ------------------------------------------------------------------

    def _get_trading_days(
        self,
        index_id: str,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        """Return sorted list of unique trading dates in [start_date, end_date].

        Uses the actual price_data table (via LIST_PRICE_HISTORY) so weekends
        and market holidays are automatically excluded.
        """
        rows = self.db.fetch_all(
            Q.LIST_PRICE_HISTORY,
            (
                index_id,
                "1d",
                start_date.isoformat(),
                end_date.isoformat() + "T23:59:59",
            ),
        )

        if not rows:
            # Fallback: use weekday calendar (no holidays excluded).
            # This allows unit tests to work without a real database.
            from datetime import timedelta
            result: list[date] = []
            curr = start_date
            while curr <= end_date:
                if curr.weekday() < 5:
                    result.append(curr)
                curr += timedelta(days=1)
            return result

        seen: set[date] = set()
        for row in rows:
            ts_val = row.get("timestamp", "")
            if isinstance(ts_val, str):
                d = date.fromisoformat(ts_val[:10])
            else:
                d = ts_val.date() if hasattr(ts_val, "date") else ts_val
            seen.add(d)
        return sorted(seen)

    def _calculate_windows(
        self, config: WalkForwardConfig, trading_days: list[date]
    ) -> list[dict]:
        """Build the list of train/test window definitions.

        Uses index arithmetic on the sorted trading_days list so that
        "N trading days" is exact — no calendar-day approximation.
        """
        windows: list[dict] = []
        if not trading_days:
            return windows

        current_idx = 0

        while True:
            # Train window: trading_days[current_idx … current_idx + train - 1]
            train_end_idx = current_idx + config.train_window_days - 1
            if train_end_idx >= len(trading_days):
                break

            # Test window follows after the embargo gap
            test_start_idx = train_end_idx + 1 + max(0, config.embargo_days)
            test_end_idx = test_start_idx + config.test_window_days - 1

            if test_end_idx >= len(trading_days):
                break

            test_end_date = trading_days[test_end_idx]
            if test_end_date > config.full_end_date:
                break

            windows.append(
                {
                    "window_id": len(windows) + 1,
                    "train_start": trading_days[current_idx],
                    "train_end": trading_days[train_end_idx],
                    "test_start": trading_days[test_start_idx],
                    "test_end": test_end_date,
                }
            )

            # Advance by step_days trading days
            current_idx += config.step_days

        return windows

    # ------------------------------------------------------------------
    # Backtest execution
    # ------------------------------------------------------------------

    @staticmethod
    def _make_bt_config(config: WalkForwardConfig, start: date, end: date) -> BacktestConfig:
        """BacktestConfig for one window, preserving every optimised parameter."""
        if config.backtest_template is not None:
            return dataclasses.replace(
                config.backtest_template,
                start_date=start,
                end_date=end,
                simulator_config=config.simulator_config,
                show_progress=False,
            )
        return BacktestConfig(
            index_id=config.index_id,
            start_date=start,
            end_date=end,
            timeframe=config.timeframe,
            warmup_bars=config.warmup_bars,
            simulator_config=config.simulator_config,
            mode=config.mode,
            min_confidence=config.min_confidence,
            show_progress=False,
        )

    def _run_window_backtest(
        self,
        config: WalkForwardConfig,
        start: date,
        end: date,
    ) -> BacktestMetrics:
        """Run a single-period backtest and return computed metrics."""
        bt_config = self._make_bt_config(config, start, end)
        result = self.strategy_runner.run(bt_config)
        return self.metrics_calculator.calculate_all(
            result.trade_history,
            result.equity_curve,
            config.simulator_config.initial_capital,
        )

    def _run_all_windows(
        self, config: WalkForwardConfig, windows_def: list[dict]
    ) -> tuple[list[WindowResult], list]:
        """Execute train + test backtests for every window, printing progress."""
        window_results: list[WindowResult] = []
        combined_test_trades: list = []
        total = len(windows_def)

        for w in windows_def:
            wid = w["window_id"]
            print(
                f"Window {wid}/{total}: "
                f"Train {w['train_start'].strftime('%Y-%m')} \u2192 "
                f"{w['train_end'].strftime('%Y-%m')} | "
                f"Test {w['test_start'].strftime('%Y-%m')} \u2192 "
                f"{w['test_end'].strftime('%Y-%m')}"
            )

            train_metrics = self._run_window_backtest(
                config, w["train_start"], w["train_end"]
            )

            # Run test period backtest (fresh simulator state)
            bt_config = self._make_bt_config(config, w["test_start"], w["test_end"])
            test_result = self.strategy_runner.run(bt_config)
            test_metrics = self.metrics_calculator.calculate_all(
                test_result.trade_history,
                test_result.equity_curve,
                config.simulator_config.initial_capital,
            )
            combined_test_trades.extend(test_result.trade_history)

            deg = train_metrics.total_return_pct - test_metrics.total_return_pct
            acceptable = (
                "\u2190 Within acceptable range"
                if deg <= 10
                else "\u2190 \u26a0 High degradation"
            )

            print(
                f"  Train: {train_metrics.total_trades} trades, "
                f"{train_metrics.win_rate:.1f}% win rate, "
                f"{train_metrics.total_return_pct:+.1f}%"
            )
            print(
                f"  Test:  {test_metrics.total_trades} trades, "
                f"{test_metrics.win_rate:.1f}% win rate, "
                f"{test_metrics.total_return_pct:+.1f}%"
            )
            print(f"  Degradation: {deg:.1f} percentage points {acceptable}")
            print()

            # Optional parameter sensitivity test (simplified ±20%)
            if config.parameter_sensitivity:
                self._run_sensitivity(config, w, test_metrics)

            window_results.append(
                WindowResult(
                    window_id=wid,
                    train_start=w["train_start"],
                    train_end=w["train_end"],
                    test_start=w["test_start"],
                    test_end=w["test_end"],
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    return_degradation_pct=round(deg, 4),
                    win_rate_degradation=round(
                        train_metrics.win_rate - test_metrics.win_rate, 4
                    ),
                    is_test_profitable=test_metrics.total_return_pct > 0,
                )
            )

        return window_results, combined_test_trades

    def _run_sensitivity(
        self,
        config: WalkForwardConfig,
        window: dict,
        baseline_metrics: BacktestMetrics,
    ) -> None:
        """Run test period with ±20% on key thresholds (simplified)."""
        base_sim = config.simulator_config
        for factor, label in [(0.8, "-20%"), (1.2, "+20%")]:
            try:
                tweaked = dataclasses.replace(
                    base_sim,
                    max_risk_per_trade_pct=base_sim.max_risk_per_trade_pct * factor,
                    min_risk_reward_ratio=base_sim.min_risk_reward_ratio * factor,
                )
                sens_config = dataclasses.replace(
                    config,
                    simulator_config=tweaked,
                    parameter_sensitivity=False,
                )
                sens_metrics = self._run_window_backtest(
                    sens_config, window["test_start"], window["test_end"]
                )
                delta = sens_metrics.total_return_pct - baseline_metrics.total_return_pct
                print(
                    f"    Sensitivity {label}: "
                    f"{sens_metrics.total_return_pct:+.1f}% "
                    f"(delta {delta:+.1f}pp)"
                )
            except Exception:
                logger.debug(
                    "Sensitivity test failed for window %d", window["window_id"]
                )

    # ------------------------------------------------------------------
    # Aggregation & overfitting scoring
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        config: WalkForwardConfig,
        window_results: list[WindowResult],
        combined_test_trades: list,
    ) -> WalkForwardResult:
        n = len(window_results)
        profitable_test_windows = sum(1 for w in window_results if w.is_test_profitable)
        test_profitability_rate = profitable_test_windows / n if n > 0 else 0.0

        avg_train_return = sum(w.train_metrics.total_return_pct for w in window_results) / n
        avg_test_return = sum(w.test_metrics.total_return_pct for w in window_results) / n
        avg_degradation = sum(w.return_degradation_pct for w in window_results) / n

        avg_train_win_rate = sum(w.train_metrics.win_rate for w in window_results) / n
        avg_test_win_rate = sum(w.test_metrics.win_rate for w in window_results) / n

        avg_train_sharpe = sum(w.train_metrics.sharpe_ratio for w in window_results) / n
        avg_test_sharpe = sum(w.test_metrics.sharpe_ratio for w in window_results) / n

        max_test_drawdown = max(w.test_metrics.max_drawdown_pct for w in window_results)

        # Combined test period statistics
        combined_test_count = len(combined_test_trades)
        combined_test_wins = sum(1 for t in combined_test_trades if t.outcome == "WIN")
        combined_test_win_rate = (
            combined_test_wins / combined_test_count * 100
            if combined_test_count > 0
            else 0.0
        )
        combined_test_return = sum(
            w.test_metrics.total_return_pct for w in window_results
        )

        # Overfitting assessment
        score, assessment = self._compute_overfitting_score(
            avg_degradation=avg_degradation,
            test_profitability_rate=test_profitability_rate,
            avg_train_win_rate=avg_train_win_rate,
            avg_test_win_rate=avg_test_win_rate,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
        )

        is_robust = score < 0.5 and test_profitability_rate >= 0.5

        verdict = self._build_verdict(
            is_robust=is_robust,
            assessment=assessment,
            avg_test_return=avg_test_return,
            test_profitability_rate=test_profitability_rate,
            total_windows=n,
            profitable_test_windows=profitable_test_windows,
        )

        return WalkForwardResult(
            config=config,
            windows=window_results,
            total_windows=n,
            profitable_test_windows=profitable_test_windows,
            test_profitability_rate=round(test_profitability_rate, 4),
            avg_train_return=round(avg_train_return, 4),
            avg_test_return=round(avg_test_return, 4),
            avg_degradation=round(avg_degradation, 4),
            avg_train_win_rate=round(avg_train_win_rate, 4),
            avg_test_win_rate=round(avg_test_win_rate, 4),
            avg_train_sharpe=round(avg_train_sharpe, 4),
            avg_test_sharpe=round(avg_test_sharpe, 4),
            max_test_drawdown=round(max_test_drawdown, 4),
            combined_test_return=round(combined_test_return, 4),
            combined_test_trades=combined_test_count,
            combined_test_win_rate=round(combined_test_win_rate, 4),
            overfitting_score=score,
            overfitting_assessment=assessment,
            is_robust=is_robust,
            verdict=verdict + "\n\n  NOTE: " + _FIXED_CONFIG_NOTE,
            is_out_of_sample=False,
            methodology_note=_FIXED_CONFIG_NOTE,
        )

    @staticmethod
    def _compute_overfitting_score(
        avg_degradation: float,
        test_profitability_rate: float,
        avg_train_win_rate: float,
        avg_test_win_rate: float,
        avg_train_sharpe: float,
        avg_test_sharpe: float,
    ) -> tuple[float, str]:
        """Heuristic stability score (0–1) and label.

        Only meaningful as an overfitting measure when the train metrics come
        from a configuration FITTED on the train window; for the legacy
        fixed-config walk-forward it measures period-to-period instability.
        """
        score = 0.0

        # Factor 1: Return degradation (0–0.3)
        if avg_degradation > 20:
            score += 0.3
        elif avg_degradation > 10:
            score += 0.2
        elif avg_degradation > 5:
            score += 0.1

        # Factor 2: Test period profitability (0–0.3)
        if test_profitability_rate < 0.25:
            score += 0.3
        elif test_profitability_rate < 0.50:
            score += 0.2
        elif test_profitability_rate < 0.75:
            score += 0.1

        # Factor 3: Win rate degradation (0–0.2)
        wr_deg = avg_train_win_rate - avg_test_win_rate
        if wr_deg > 15:
            score += 0.2
        elif wr_deg > 10:
            score += 0.15
        elif wr_deg > 5:
            score += 0.1

        # Factor 4: Sharpe degradation (0–0.2)
        sharpe_deg = avg_train_sharpe - avg_test_sharpe
        if sharpe_deg > 1.0:
            score += 0.2
        elif sharpe_deg > 0.5:
            score += 0.1

        score = round(score, 2)

        if score < 0.25:
            label = "LOW_RISK"
        elif score < 0.50:
            label = "MODERATE_RISK"
        elif score < 0.75:
            label = "HIGH_RISK"
        else:
            label = "SEVERE"

        return score, label

    @staticmethod
    def _build_verdict(
        is_robust: bool,
        assessment: str,
        avg_test_return: float,
        test_profitability_rate: float,
        total_windows: int,
        profitable_test_windows: int,
    ) -> str:
        pct_str = f"{profitable_test_windows}/{total_windows}"

        if is_robust:
            return (
                f"\u2705 STRATEGY IS ROBUST ENOUGH FOR PAPER TRADING\n\n"
                f"  The strategy shows consistent positive returns across "
                f"{pct_str} test windows.\n"
                f"  Expected live performance: ~{avg_test_return:.1f}% per test period "
                f"(lower than backtest suggests).\n"
                f"  Proceed with caution — monitor for 1 month of paper trading "
                f"before any live risk."
            )
        elif assessment == "MODERATE_RISK":
            return (
                f"\u26a0 STRATEGY NEEDS FURTHER REFINEMENT\n\n"
                f"  Moderate overfitting detected. Only {pct_str} test windows "
                f"are profitable.\n"
                f"  Consider simplifying the strategy or reducing the number of parameters."
            )
        elif assessment in ("HIGH_RISK", "SEVERE"):
            return (
                f"\u274c STRATEGY IS OVERFIT — DO NOT TRADE\n\n"
                f"  Only {pct_str} test windows are profitable.\n"
                f"  The strategy does not generalise out-of-sample."
            )
        else:
            return (
                f"\u2753 INCONCLUSIVE — {pct_str} test windows profitable.\n"
                f"  Avg test return: {avg_test_return:.1f}%.\n"
                f"  Gather more data before trading."
            )



# ===========================================================================
# Generic nested (purged / embargoed) walk-forward
# ===========================================================================


@dataclass
class Fold:
    """One chronological train/test split (positions refer to unique days)."""

    fold_id: int
    train_days: list[date]
    test_days: list[date]
    embargo_days: int
    is_holdout: bool = False

    @property
    def train_start(self) -> date:
        return self.train_days[0]

    @property
    def train_end(self) -> date:
        return self.train_days[-1]

    @property
    def test_start(self) -> date:
        return self.test_days[0]

    @property
    def test_end(self) -> date:
        return self.test_days[-1]


@dataclass
class FoldResult:
    """Out-of-sample evaluation of one fold (or of the holdout)."""

    fold: Fold
    metrics: dict[str, Any]
    n_train_obs: int
    n_test_obs: int
    model: Any = None


@dataclass
class NestedWalkForwardResult:
    """Per-fold and aggregate out-of-sample metrics plus the holdout."""

    folds: list[FoldResult]
    holdout: Optional[FoldResult]
    oos_summary: dict[str, dict[str, float]]   # metric → {mean, median, std, min, max, n}
    n_folds: int
    embargo_days: int
    holdout_fraction: float
    expanding: bool

    @property
    def holdout_metrics(self) -> Optional[dict[str, Any]]:
        return self.holdout.metrics if self.holdout is not None else None

    def summary(self) -> str:
        lines = [
            f"NESTED WALK-FORWARD: {self.n_folds} folds, embargo {self.embargo_days}d, "
            f"{'expanding' if self.expanding else 'rolling'} train, "
            f"holdout {self.holdout_fraction:.0%}"
        ]
        for fr in self.folds:
            f = fr.fold
            lines.append(
                f"  Fold {f.fold_id}: train {f.train_start}→{f.train_end} | "
                f"test {f.test_start}→{f.test_end} | {_fmt_metrics(fr.metrics)}"
            )
        for k, st in self.oos_summary.items():
            lines.append(
                f"  OOS {k}: mean {st['mean']:.4g} median {st['median']:.4g} "
                f"std {st['std']:.4g} [min {st['min']:.4g}, max {st['max']:.4g}] n={int(st['n'])}"
            )
        if self.holdout is not None:
            h = self.holdout.fold
            lines.append(
                f"  HOLDOUT (evaluated once) {h.test_start}→{h.test_end}: "
                f"{_fmt_metrics(self.holdout.metrics)}"
            )
        return "\n".join(lines)


def _fmt_metrics(m: dict[str, Any]) -> str:
    parts = []
    for k, v in m.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        elif isinstance(v, (int, str, bool)):
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def make_walk_forward_folds(
    days: Sequence[Any],
    n_folds: Optional[int] = 5,
    train_days: Optional[int] = None,
    test_days: Optional[int] = None,
    embargo_days: int = 1,
    holdout_fraction: float = 0.2,
    expanding: bool = True,
    min_train_days: int = 1,
) -> tuple[list[Fold], Optional[Fold]]:
    """Build chronological purged/embargoed folds over unique trading days.

    * The last ``ceil(holdout_fraction · n)`` days form the HOLDOUT; folds only
      use the days before it.
    * Fold ``k`` tests on a contiguous block of ``test_days`` days; its training
      window ends ``embargo_days`` days before the test block starts (the gap is
      neither trained on nor tested — it purges observations whose outcome
      horizon could overlap the test block).
    * ``expanding=True``: training is anchored at the first day (``train_days``
      then only sets the initial size).  ``expanding=False``: rolling window of
      ``train_days`` days.
    * If ``test_days`` is omitted it is derived from ``n_folds``; if
      ``n_folds`` is omitted as many folds as fit are produced.  Test blocks
      are aligned so the last one ends right before the holdout.
    * The holdout fold trains on all pre-holdout days except the final
      ``embargo_days``.
    """
    uniq = sorted({_to_date(d) for d in days})
    n = len(uniq)
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction}")
    if embargo_days < 0:
        raise ValueError("embargo_days must be >= 0")
    if not expanding and not train_days:
        raise ValueError("Rolling (expanding=False) folds require train_days")

    n_hold = int(math.ceil(n * holdout_fraction)) if holdout_fraction > 0 else 0
    pre = uniq[: n - n_hold]
    hold = uniq[n - n_hold:]
    m = len(pre)

    if test_days is None:
        if not n_folds:
            raise ValueError("Provide n_folds or test_days")
        initial = train_days if train_days else None
        if initial is None:
            test_days = (m - embargo_days) // (n_folds + 1)
        else:
            test_days = (m - initial - embargo_days) // n_folds
    if test_days < 1:
        raise ValueError(
            f"Not enough pre-holdout data ({m} days) for the requested folds"
        )
    if not n_folds:
        initial = train_days or test_days
        n_folds = (m - initial - embargo_days) // test_days
    if n_folds < 1:
        raise ValueError(f"Not enough pre-holdout data ({m} days) for one fold")

    first_test = m - n_folds * test_days
    folds: list[Fold] = []
    for k in range(n_folds):
        ts = first_test + k * test_days
        te = ts + test_days
        tr_end = ts - embargo_days
        tr_start = 0 if expanding else max(0, tr_end - int(train_days))
        if tr_end - tr_start < max(1, min_train_days):
            logger.warning("Fold %d skipped: only %d training days", k, max(tr_end - tr_start, 0))
            continue
        folds.append(Fold(
            fold_id=len(folds) + 1,
            train_days=pre[tr_start:tr_end],
            test_days=pre[ts:te],
            embargo_days=embargo_days,
        ))
    if not folds:
        raise ValueError("No fold has enough training data; reduce n_folds/test_days/embargo")

    holdout: Optional[Fold] = None
    if hold:
        h_train_end = m - embargo_days
        h_train_start = 0 if expanding else max(0, h_train_end - int(train_days))
        if h_train_end - h_train_start < max(1, min_train_days):
            raise ValueError("Not enough pre-holdout data to fit the holdout model")
        holdout = Fold(
            fold_id=0,
            train_days=pre[h_train_start:h_train_end],
            test_days=list(hold),
            embargo_days=embargo_days,
            is_holdout=True,
        )
    return folds, holdout


def _take(items: Any, positions: Sequence[int]) -> Any:
    """Subset ``items`` by integer positions, preserving its container type."""
    pos = np.asarray(positions, dtype=np.int64)
    if hasattr(items, "iloc"):                     # pandas Series / DataFrame
        return items.iloc[pos]
    if isinstance(items, np.ndarray):
        return items[pos]
    if hasattr(items, "take"):                     # pandas Index
        return items.take(pos)
    if isinstance(items, tuple):
        return tuple(items[i] for i in pos)
    return [items[i] for i in pos]


def _summarise_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [
            float(r[k]) for r in rows
            if k in r and isinstance(r[k], (int, float)) and not isinstance(r[k], bool)
            and math.isfinite(float(r[k]))
        ]
        if not vals:
            continue
        out[k] = {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "n": float(len(vals)),
        }
    return out


def nested_walk_forward(
    timestamps_or_dates: Sequence[Any],
    fit_fn: Callable[[Any], Any],
    eval_fn: Callable[[Any, Any], dict],
    n_folds: Optional[int] = 5,
    train_days: Optional[int] = None,
    test_days: Optional[int] = None,
    embargo_days: int = 1,
    holdout_fraction: float = 0.2,
    expanding: bool = True,
    min_train_days: int = 1,
    keep_models: bool = False,
) -> NestedWalkForwardResult:
    """Generic out-of-sample walk-forward with embargo and a final holdout.

    Args:
        timestamps_or_dates: chronological observations (dates, datetimes,
            pandas Timestamps, a DatetimeIndex, …).  Several observations may
            share a day (intraday bars); folds are cut on unique trading days so
            a day is never split between train and test.
        fit_fn: ``fit_fn(train_slice) -> model``.  Receives ONLY the training
            observations of the fold (same container type as the input).
        eval_fn: ``eval_fn(model, test_slice) -> dict`` of metrics.
        n_folds / train_days / test_days / embargo_days / holdout_fraction /
        expanding / min_train_days: see :func:`make_walk_forward_folds`.
        keep_models: store each fitted model on its :class:`FoldResult`.

    Returns:
        :class:`NestedWalkForwardResult` with per-fold OOS metrics, their
        aggregate (mean/median/std/min/max per numeric metric) and the holdout
        metrics.  The holdout is evaluated exactly once, with a model fitted on
        all pre-holdout data (less the embargo); it never influences the folds.
    """
    items = timestamps_or_dates
    if not hasattr(items, "__getitem__") or isinstance(items, (str, bytes)):
        items = list(items)
    use_index = hasattr(items, "iloc") and not _is_datelike_values(items)
    item_days = [_to_date(x) for x in (items.index if use_index else items)]
    for a, b in zip(item_days, item_days[1:]):
        if b < a:
            raise ValueError("timestamps_or_dates must be in chronological order")

    folds, holdout_fold = make_walk_forward_folds(
        item_days, n_folds=n_folds, train_days=train_days, test_days=test_days,
        embargo_days=embargo_days, holdout_fraction=holdout_fraction,
        expanding=expanding, min_train_days=min_train_days,
    )

    positions: dict[date, list[int]] = {}
    for i, d in enumerate(item_days):
        positions.setdefault(d, []).append(i)

    def _positions(days: list[date]) -> list[int]:
        out: list[int] = []
        for d in days:
            out.extend(positions[d])
        return out

    def _run(fold: Fold) -> FoldResult:
        tr = _positions(fold.train_days)
        te = _positions(fold.test_days)
        if not fold.train_end < fold.test_start:
            raise RuntimeError("fold construction error: train must end before test starts")
        model = fit_fn(_take(items, tr))
        metrics = eval_fn(model, _take(items, te))
        if not isinstance(metrics, dict):
            raise TypeError(f"eval_fn must return a dict, got {type(metrics).__name__}")
        return FoldResult(
            fold=fold, metrics=dict(metrics), n_train_obs=len(tr), n_test_obs=len(te),
            model=model if keep_models else None,
        )

    fold_results = [_run(f) for f in folds]
    holdout_result = _run(holdout_fold) if holdout_fold is not None else None

    return NestedWalkForwardResult(
        folds=fold_results,
        holdout=holdout_result,
        oos_summary=_summarise_metrics([fr.metrics for fr in fold_results]),
        n_folds=len(fold_results),
        embargo_days=embargo_days,
        holdout_fraction=holdout_fraction,
        expanding=expanding,
    )


def _is_datelike_values(obj: Any) -> bool:
    """For pandas objects: True if the VALUES (not the index) are dates."""
    try:
        first = obj.iloc[0]
        _to_date(first)
        return True
    except Exception:
        return False
