"""
Paper Trading — Phase 9 of the Trading Decision Support System.

Wraps the DecisionEngine to simulate live order execution against real-time
market prices, tracking P&L, slippage, and risk limits without placing
actual orders on the exchange.

Phase 9.1 — Paper Trading Engine
Phase 9.2 — Watchdog, Daily Reports, Edge Tracker
Phase 9.3 — Pre-Launch Validator
"""

from src.paper_trading.paper_engine import (
    PaperTradingConfig,
    PaperTradingEngine,
    PaperPosition,
    PaperTrade,
    PaperExecution,
    MissedSignal,
    DailyPaperSummary,
    PaperTradingStats,
    DailyRecord,
)
from src.paper_trading.watchdog import HealthCheck, SystemWatchdog
from src.paper_trading.daily_report import AutomatedReporter
from src.paper_trading.edge_tracker import EdgeAssessment, EdgeTracker
from src.paper_trading.pre_launch_validator import (
    PreLaunchValidator,
    ValidationReport,
    CheckResult,
)

__all__ = [
    # Phase 9.1 — Paper Trading Engine
    "PaperTradingConfig",
    "PaperTradingEngine",
    "PaperPosition",
    "PaperTrade",
    "PaperExecution",
    "MissedSignal",
    "DailyPaperSummary",
    "PaperTradingStats",
    "DailyRecord",
    # Phase 9.2 — Monitoring & Reporting
    "HealthCheck",
    "SystemWatchdog",
    "AutomatedReporter",
    "EdgeAssessment",
    "EdgeTracker",
    # Phase 9.3 — Pre-Launch Validation
    "PreLaunchValidator",
    "ValidationReport",
    "CheckResult",
]
