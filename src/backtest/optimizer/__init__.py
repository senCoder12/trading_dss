"""Strategy optimisation: parameter spaces, grid search, and applicators."""

from src.backtest.optimizer.param_space import (
    ParameterApplicator,
    ParameterDef,
    ParameterSpace,
    ParameterSpaceLoader,
)
from src.backtest.optimizer.optimization_engine import (
    EvaluationResult,
    OptimizationConfig,
    OptimizationEngine,
    OptimizationResult,
)
from src.backtest.optimizer.robustness import (
    MonteCarloResult,
    ParamSensitivity,
    PeriodResult,
    RegimeResult,
    RobustnessReport,
    RobustnessTester,
    SensitivityResult,
    StabilityResult,
)

# Legacy grid_optimizer module is available via direct import:
#   from src.backtest.optimizer.grid_optimizer import GridOptimizer
# Not auto-imported because it depends on the legacy backtester module.

__all__ = [
    "ParameterApplicator",
    "ParameterDef",
    "ParameterSpace",
    "ParameterSpaceLoader",
    "EvaluationResult",
    "OptimizationConfig",
    "OptimizationEngine",
    "OptimizationResult",
    "MonteCarloResult",
    "ParamSensitivity",
    "PeriodResult",
    "RegimeResult",
    "RobustnessReport",
    "RobustnessTester",
    "SensitivityResult",
    "StabilityResult",
]
