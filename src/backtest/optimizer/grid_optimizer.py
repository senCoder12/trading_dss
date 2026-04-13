"""
Parameter optimisation for strategy backtesting.

Grid-search over a parameter space and ranks results by Sharpe ratio.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.backtest.backtester import BacktestResult, Backtester, StrategyFn
from src.backtest.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

# Factory that creates a StrategyFn given a param dict
StrategyFactory = Callable[[dict[str, Any]], StrategyFn]


@dataclass
class OptimizationResult:
    """Single parameter combination result."""

    params: dict[str, Any]
    metrics: PerformanceMetrics
    final_capital: float


class GridOptimizer:
    """
    Exhaustive grid-search optimiser over a strategy parameter space.

    Parameters
    ----------
    backtester:
        Configured :class:`Backtester` instance.
    """

    def __init__(self, backtester: Backtester) -> None:
        self._bt = backtester

    def optimize(
        self,
        index_id: str,
        ohlcv: pd.DataFrame,
        strategy_factory: StrategyFactory,
        param_grid: dict[str, list[Any]],
        rank_by: str = "sharpe_ratio",
        warmup_bars: int = 50,
    ) -> list[OptimizationResult]:
        """
        Run backtest for every combination in *param_grid* and rank results.

        Parameters
        ----------
        index_id:
            Label for results.
        ohlcv:
            Historical OHLCV data.
        strategy_factory:
            Callable that accepts a param dict and returns a StrategyFn.
        param_grid:
            Mapping of parameter name → list of values to try.
        rank_by:
            Attribute of :class:`PerformanceMetrics` to rank by (default
            ``"sharpe_ratio"``).
        warmup_bars:
            Warmup bars passed to each backtest run.

        Returns
        -------
        list[OptimizationResult]:
            All results sorted by *rank_by* descending.
        """
        keys = list(param_grid.keys())
        combinations = list(itertools.product(*param_grid.values()))
        total = len(combinations)
        logger.info(
            "Grid search: %d combinations × index=%s", total, index_id
        )

        results: list[OptimizationResult] = []
        for i, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            try:
                strategy = strategy_factory(params)
                bt_result: BacktestResult = self._bt.run(
                    index_id, ohlcv, strategy, warmup_bars
                )
                results.append(OptimizationResult(
                    params=params,
                    metrics=bt_result.metrics,
                    final_capital=bt_result.final_capital,
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Params %s failed: %s", params, exc)

            if i % 50 == 0:
                logger.info("  Progress: %d/%d", i, total)

        results.sort(
            key=lambda r: getattr(r.metrics, rank_by, 0.0),
            reverse=True,
        )
        logger.info(
            "Optimisation complete — best %s=%.4f params=%s",
            rank_by,
            getattr(results[0].metrics, rank_by, 0.0) if results else 0.0,
            results[0].params if results else {},
        )
        return results
