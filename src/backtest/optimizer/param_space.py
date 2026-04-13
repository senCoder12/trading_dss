"""
Parameter space definition and loading for strategy optimisation.

Defines the search space for the optimiser — which parameters to vary,
their ranges, and how to apply a parameter combination to a BacktestConfig.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Optional

from src.backtest.strategy_runner import BacktestConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter definition
# ---------------------------------------------------------------------------

@dataclass
class ParameterDef:
    """Definition of a single optimisable parameter."""

    name: str                          # Unique identifier: "stop_loss_atr_multiplier"
    display_name: str                  # Human-readable: "Stop Loss (ATR multiplier)"
    description: str                   # What this parameter controls

    param_type: str                    # "float" / "int" / "choice"

    # For float/int parameters
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None       # Step size for grid search
    default: Any = None                # Current/default value

    # For choice parameters
    choices: Optional[list] = None     # ["HIGH", "MEDIUM", "LOW"] or [True, False]

    # Constraints
    category: str = "general"          # "risk" / "entry" / "exit" / "filter" / "regime"
    affects: Optional[list[str]] = field(default_factory=list)

    # Overfitting protection
    sensitivity_weight: float = 1.0    # Higher = more dangerous to optimise

    def get_values(self) -> list:
        """Generate all values this parameter will be tested at."""
        if self.param_type == "choice":
            return list(self.choices) if self.choices else [self.default]

        if self.param_type == "int":
            return list(range(
                int(self.min_value),
                int(self.max_value) + 1,
                int(self.step),
            ))

        if self.param_type == "float":
            values: list[float] = []
            v = self.min_value
            while v <= self.max_value + self.step * 0.01:  # float precision guard
                values.append(round(v, 4))
                v += self.step
            return values

        return [self.default]

    @property
    def num_values(self) -> int:
        return len(self.get_values())


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

@dataclass
class ParameterSpace:
    """Complete definition of the optimisation search space."""

    parameters: list[ParameterDef]

    @property
    def total_combinations(self) -> int:
        result = 1
        for p in self.parameters:
            result *= p.num_values
        return result

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the parameter space against overfitting safety rules."""
        issues: list[str] = []

        if self.num_parameters > 7:
            issues.append(
                f"OVERFITTING RISK: {self.num_parameters} parameters exceeds "
                f"recommended max of 7. Each additional parameter exponentially "
                f"increases overfitting risk."
            )

        if self.total_combinations > 10_000:
            issues.append(
                f"SEARCH SPACE TOO LARGE: {self.total_combinations:,} combinations. "
                f"Recommended max is 5,000-10,000. Reduce parameter ranges or "
                f"increase step sizes."
            )

        if self.total_combinations < 20:
            issues.append(
                f"SEARCH SPACE TOO SMALL: {self.total_combinations} combinations. "
                f"May miss optimal parameters. Consider finer step sizes."
            )

        # Check for duplicate names
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            issues.append("Duplicate parameter names detected.")

        return (len(issues) == 0, issues)

    def generate_grid(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        param_names = [p.name for p in self.parameters]
        param_values = [p.get_values() for p in self.parameters]

        grid: list[dict[str, Any]] = []
        for combo in product(*param_values):
            grid.append(dict(zip(param_names, combo)))

        return grid

    def generate_random_samples(self, n: int, seed: int = 42) -> list[dict[str, Any]]:
        """Generate *n* random parameter combinations (for random search)."""
        import random
        rng = random.Random(seed)

        samples: list[dict[str, Any]] = []
        for _ in range(n):
            combo: dict[str, Any] = {}
            for p in self.parameters:
                values = p.get_values()
                combo[p.name] = rng.choice(values)
            samples.append(combo)

        return samples

    def get_default_values(self) -> dict[str, Any]:
        """Get current/default value for each parameter."""
        return {p.name: p.default for p in self.parameters}

    def summary(self) -> str:
        """Human-readable summary of the search space."""
        lines = [
            f"Parameter Space: {self.num_parameters} parameters, "
            f"{self.total_combinations:,} combinations\n"
        ]
        for p in self.parameters:
            vals = p.get_values()
            lines.append(
                f"  {p.display_name}: {vals[0]} -> {vals[-1]} "
                f"({p.num_values} values) [default: {p.default}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profile loader
# ---------------------------------------------------------------------------

_DEFAULT_PROFILES_PATH = Path(__file__).resolve().parents[3] / "config" / "optimization_profiles.json"


class ParameterSpaceLoader:
    """Loads optimisation profiles from config and creates ParameterSpace objects."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path else _DEFAULT_PROFILES_PATH
        with open(path, "r") as f:
            self.profiles: dict[str, Any] = json.load(f)

    def load_profile(self, profile_name: str) -> ParameterSpace:
        """Load a named profile (conservative / moderate / aggressive)."""
        if profile_name not in self.profiles:
            available = list(self.profiles.keys())
            raise ValueError(
                f"Unknown profile '{profile_name}'. Available: {available}"
            )

        profile = self.profiles[profile_name]
        params = [ParameterDef(**p) for p in profile["parameters"]]

        space = ParameterSpace(parameters=params)
        is_valid, issues = space.validate()

        if not is_valid:
            for issue in issues:
                logger.warning("Parameter space warning: %s", issue)

        return space

    def load_custom(self, parameter_defs: list[dict]) -> ParameterSpace:
        """Load a custom parameter space from a list of parameter definitions."""
        params = [ParameterDef(**p) for p in parameter_defs]
        return ParameterSpace(parameters=params)

    def list_profiles(self) -> list[dict[str, Any]]:
        """List available profiles with descriptions."""
        return [
            {
                "name": name,
                "description": p["description"],
                "param_count": len(p["parameters"]),
            }
            for name, p in self.profiles.items()
        ]


# ---------------------------------------------------------------------------
# Parameter applicator
# ---------------------------------------------------------------------------

class ParameterApplicator:
    """Applies a parameter combination to the backtest configuration.

    This is the bridge between the optimiser (which speaks in parameter dicts)
    and the backtester (which speaks in BacktestConfig and SimulatorConfig).
    """

    # Mapping from optimiser parameter name → (target_object, field_name)
    # "config" means BacktestConfig, "sim" means SimulatorConfig.
    _PARAM_MAP: dict[str, tuple[str, str]] = {
        "stop_loss_atr_multiplier": ("sim", "stop_loss_atr_mult"),
        "target_atr_multiplier":    ("sim", "target_atr_mult"),
        "min_confidence_filter":    ("config", "min_confidence"),
        "signal_cooldown_bars":     ("config", "signal_cooldown_bars"),
        "risk_per_trade_pct":       ("sim", "max_risk_per_trade_pct"),
        "trailing_sl_activation_pct": ("sim", "trailing_sl_activation"),
        "max_positions":            ("sim", "max_open_positions"),
    }

    def apply(
        self,
        base_config: BacktestConfig,
        params: dict[str, Any],
    ) -> BacktestConfig | None:
        """Create a new BacktestConfig with the given parameter values applied.

        Returns ``None`` if the combination is invalid (e.g. target <= SL).
        Does **not** modify *base_config*.
        """
        config = copy.deepcopy(base_config)

        for name, value in params.items():
            mapping = self._PARAM_MAP.get(name)
            if mapping is None:
                logger.warning("Unknown parameter '%s' — skipped", name)
                continue

            target, field_name = mapping
            if target == "sim":
                setattr(config.simulator_config, field_name, value)
            else:
                setattr(config, field_name, value)

        # Validate: target must exceed stop-loss for positive risk/reward
        sim = config.simulator_config
        if sim.target_atr_mult <= sim.stop_loss_atr_mult:
            logger.warning(
                "Target (%.2f) <= SL (%.2f). Skipping — negative RR ratio.",
                sim.target_atr_mult,
                sim.stop_loss_atr_mult,
            )
            return None

        return config
