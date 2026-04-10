"""Backtesting framework."""

from src.backtest.data_replay import (
    DataReplayEngine,
    ReplayIterator,
    ReplaySession,
    TimeSlice,
)
from src.backtest.trade_simulator import (
    ClosedTrade,
    EquityPoint,
    PortfolioState,
    SimulatorConfig,
    TradeExecution,
    TradeSimulator,
)
from src.backtest.strategy_runner import (
    BacktestConfig,
    BacktestResult,
    StrategyRunner,
)
from src.backtest.metrics import MetricsCalculator, BacktestMetrics
from src.backtest.walk_forward import WalkForwardValidator, WalkForwardConfig, WalkForwardResult, WindowResult
from src.backtest.report_generator import ReportGenerator

__all__ = [
    # Data replay
    "DataReplayEngine",
    "ReplayIterator",
    "ReplaySession",
    "TimeSlice",
    # Trade simulator
    "ClosedTrade",
    "EquityPoint",
    "PortfolioState",
    "SimulatorConfig",
    "TradeExecution",
    "TradeSimulator",
    # Strategy runner
    "BacktestConfig",
    "BacktestResult",
    "StrategyRunner",
    # Metrics
    "MetricsCalculator",
    "BacktestMetrics",
    # Walk forward
    "WalkForwardValidator",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WindowResult",
    # Report Generator
    "ReportGenerator",
]
