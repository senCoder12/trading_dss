"""Decision engine, risk management, and signal lifecycle tracking."""

from .decision_engine import DecisionEngine, DecisionResult, DashboardData, IndexDashboard
from .regime_detector import RegimeDetector, MarketRegime, SignalWeights
from .signal_generator import SignalGenerator, TradingSignal
from .risk_manager import RiskManager, RefinedSignal, RiskConfig, PositionUpdate, DailyPnL
from .signal_tracker import SignalTracker, PerformanceStats, CalibrationReport
from .shadow_tracker import ShadowTracker, ShadowReport

__all__ = [
    # Master orchestrator
    "DecisionEngine",
    "DecisionResult",
    "DashboardData",
    "IndexDashboard",
    # Phase 5.1
    "RegimeDetector",
    "MarketRegime",
    "SignalWeights",
    # Phase 5.2
    "SignalGenerator",
    "TradingSignal",
    # Phase 5.3
    "RiskManager",
    "RefinedSignal",
    "RiskConfig",
    "PositionUpdate",
    "DailyPnL",
    # Signal tracking
    "SignalTracker",
    "PerformanceStats",
    "CalibrationReport",
    # Shadow tracking
    "ShadowTracker",
    "ShadowReport",
]
