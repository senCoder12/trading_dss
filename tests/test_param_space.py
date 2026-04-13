"""Tests for the parameter-space module (src/backtest/optimizer/param_space.py)."""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.backtest.optimizer.param_space import (
    ParameterApplicator,
    ParameterDef,
    ParameterSpace,
    ParameterSpaceLoader,
)
from src.backtest.strategy_runner import BacktestConfig
from src.backtest.trade_simulator import SimulatorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_param(name: str = "p", min_v: float = 1.0, max_v: float = 3.0,
                 step: float = 0.5, default: float = 2.0) -> ParameterDef:
    return ParameterDef(
        name=name, display_name=name, description="test",
        param_type="float", min_value=min_v, max_value=max_v,
        step=step, default=default,
    )


def _int_param(name: str = "q", min_v: int = 1, max_v: int = 5,
               step: int = 1, default: int = 3) -> ParameterDef:
    return ParameterDef(
        name=name, display_name=name, description="test",
        param_type="int", min_value=min_v, max_value=max_v,
        step=step, default=default,
    )


def _choice_param(name: str = "c", choices: list | None = None,
                  default: str = "B") -> ParameterDef:
    return ParameterDef(
        name=name, display_name=name, description="test",
        param_type="choice", choices=choices or ["A", "B", "C"],
        default=default,
    )


def _base_config() -> BacktestConfig:
    return BacktestConfig(
        index_id="NIFTY50",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
    )


# ---------------------------------------------------------------------------
# ParameterDef.get_values()
# ---------------------------------------------------------------------------

class TestParameterDefGetValues:
    """Test value generation for each parameter type."""

    def test_float_values(self):
        p = _float_param(min_v=1.0, max_v=2.5, step=0.5)
        assert p.get_values() == [1.0, 1.5, 2.0, 2.5]

    def test_float_num_values(self):
        p = _float_param(min_v=1.0, max_v=2.5, step=0.5)
        assert p.num_values == 4

    def test_float_single_value(self):
        p = _float_param(min_v=2.0, max_v=2.0, step=0.5)
        assert p.get_values() == [2.0]

    def test_int_values(self):
        p = _int_param(min_v=1, max_v=5, step=2)
        assert p.get_values() == [1, 3, 5]

    def test_int_num_values(self):
        p = _int_param(min_v=1, max_v=5, step=1)
        assert p.num_values == 5

    def test_choice_values(self):
        p = _choice_param(choices=["LOW", "MEDIUM", "HIGH"])
        assert p.get_values() == ["LOW", "MEDIUM", "HIGH"]

    def test_choice_num_values(self):
        p = _choice_param(choices=["A", "B"])
        assert p.num_values == 2

    def test_unknown_type_returns_default(self):
        p = ParameterDef(
            name="x", display_name="x", description="test",
            param_type="unknown", default=42,
        )
        assert p.get_values() == [42]


# ---------------------------------------------------------------------------
# ParameterSpace — total_combinations
# ---------------------------------------------------------------------------

class TestParameterSpaceCombinations:
    """Test combination counting."""

    def test_single_param(self):
        space = ParameterSpace(parameters=[_float_param(min_v=1.0, max_v=2.0, step=0.5)])
        # values: [1.0, 1.5, 2.0] → 3
        assert space.total_combinations == 3

    def test_two_params_multiply(self):
        space = ParameterSpace(parameters=[
            _float_param("a", 1.0, 2.0, 0.5),  # 3 values
            _int_param("b", 1, 3, 1),            # 3 values
        ])
        assert space.total_combinations == 9

    def test_three_params(self):
        space = ParameterSpace(parameters=[
            _float_param("a", 1.0, 2.0, 0.5),   # 3 values
            _choice_param("b", ["X", "Y"]),       # 2 values
            _int_param("c", 1, 5, 2),             # 3 values: [1, 3, 5]
        ])
        assert space.total_combinations == 3 * 2 * 3

    def test_num_parameters(self):
        space = ParameterSpace(parameters=[_float_param("a"), _float_param("b")])
        assert space.num_parameters == 2


# ---------------------------------------------------------------------------
# ParameterSpace — validate
# ---------------------------------------------------------------------------

class TestParameterSpaceValidate:
    """Test overfitting safety rules."""

    def test_valid_space_passes(self):
        space = ParameterSpace(parameters=[
            _float_param("a", 1.0, 3.0, 0.5),   # 5 values
            _choice_param("b"),                    # 3 values
            _int_param("c", 1, 5, 2),              # 3 values
        ])
        # 5 * 3 * 3 = 45, within [20, 10000]
        is_valid, issues = space.validate()
        assert is_valid
        assert issues == []

    def test_too_many_parameters(self):
        params = [_float_param(f"p{i}", 1.0, 2.0, 1.0) for i in range(8)]
        space = ParameterSpace(parameters=params)
        is_valid, issues = space.validate()
        assert not is_valid
        assert any("OVERFITTING RISK" in msg for msg in issues)

    def test_exactly_seven_parameters_ok(self):
        params = [_float_param(f"p{i}", 1.0, 2.0, 1.0) for i in range(7)]
        space = ParameterSpace(parameters=params)
        is_valid, issues = space.validate()
        # 7 params with 2 values each = 128 combos, within limits
        assert not any("OVERFITTING RISK" in msg for msg in issues)

    def test_too_many_combinations(self):
        # Each param has 22 values → 22^2 = 484, need > 10000
        # Use 3 params with ~22 values each → 22^3 = 10648
        params = [_float_param(f"p{i}", 0.0, 10.5, 0.5) for i in range(3)]
        space = ParameterSpace(parameters=params)
        assert space.total_combinations > 10_000
        is_valid, issues = space.validate()
        assert not is_valid
        assert any("SEARCH SPACE TOO LARGE" in msg for msg in issues)

    def test_too_few_combinations(self):
        space = ParameterSpace(parameters=[
            _float_param("a", 1.0, 1.5, 0.5),  # 2 values
            _choice_param("b", ["X", "Y"]),      # 2 values
        ])
        # 2 * 2 = 4 < 20
        is_valid, issues = space.validate()
        assert not is_valid
        assert any("SEARCH SPACE TOO SMALL" in msg for msg in issues)

    def test_duplicate_names(self):
        space = ParameterSpace(parameters=[
            _float_param("same_name", 1.0, 3.0, 0.5),
            _int_param("same_name", 1, 5, 1),
        ])
        is_valid, issues = space.validate()
        assert not is_valid
        assert any("Duplicate" in msg for msg in issues)


# ---------------------------------------------------------------------------
# ParameterSpace — generate_grid
# ---------------------------------------------------------------------------

class TestParameterSpaceGrid:
    """Test grid generation."""

    def test_grid_size_matches_combinations(self):
        space = ParameterSpace(parameters=[
            _float_param("a", 1.0, 2.0, 0.5),  # 3 values
            _choice_param("b", ["X", "Y"]),      # 2 values
        ])
        grid = space.generate_grid()
        assert len(grid) == space.total_combinations
        assert len(grid) == 6

    def test_grid_entries_are_dicts(self):
        space = ParameterSpace(parameters=[_float_param("a", 1.0, 2.0, 1.0)])
        grid = space.generate_grid()
        assert all(isinstance(entry, dict) for entry in grid)
        assert all("a" in entry for entry in grid)

    def test_grid_has_correct_keys(self):
        space = ParameterSpace(parameters=[
            _float_param("alpha"),
            _int_param("beta"),
        ])
        grid = space.generate_grid()
        for entry in grid:
            assert set(entry.keys()) == {"alpha", "beta"}

    def test_grid_covers_all_values(self):
        space = ParameterSpace(parameters=[
            _choice_param("c", ["A", "B", "C"]),
        ])
        grid = space.generate_grid()
        seen = {entry["c"] for entry in grid}
        assert seen == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# ParameterSpace — generate_random_samples
# ---------------------------------------------------------------------------

class TestParameterSpaceRandomSamples:
    """Test random sampling."""

    def test_sample_count(self):
        space = ParameterSpace(parameters=[
            _float_param("a", 1.0, 3.0, 0.5),
            _choice_param("b"),
        ])
        samples = space.generate_random_samples(10)
        assert len(samples) == 10

    def test_seed_reproducibility(self):
        space = ParameterSpace(parameters=[
            _float_param("a", 1.0, 3.0, 0.5),
            _int_param("b", 1, 10, 1),
        ])
        s1 = space.generate_random_samples(20, seed=123)
        s2 = space.generate_random_samples(20, seed=123)
        assert s1 == s2

    def test_different_seeds_differ(self):
        space = ParameterSpace(parameters=[
            _float_param("a", 1.0, 3.0, 0.5),
            _int_param("b", 1, 10, 1),
        ])
        s1 = space.generate_random_samples(50, seed=1)
        s2 = space.generate_random_samples(50, seed=999)
        assert s1 != s2

    def test_values_within_range(self):
        space = ParameterSpace(parameters=[
            _float_param("a", 1.0, 3.0, 0.5),
        ])
        valid_values = set(space.parameters[0].get_values())
        samples = space.generate_random_samples(100)
        for s in samples:
            assert s["a"] in valid_values


# ---------------------------------------------------------------------------
# ParameterSpace — defaults and summary
# ---------------------------------------------------------------------------

class TestParameterSpaceDefaults:

    def test_get_default_values(self):
        space = ParameterSpace(parameters=[
            _float_param("a", default=1.5),
            _int_param("b", default=3),
            _choice_param("c", default="HIGH"),
        ])
        defaults = space.get_default_values()
        assert defaults == {"a": 1.5, "b": 3, "c": "HIGH"}

    def test_summary_contains_param_names(self):
        space = ParameterSpace(parameters=[
            ParameterDef(
                name="sl", display_name="Stop Loss", description="d",
                param_type="float", min_value=1.0, max_value=2.0,
                step=0.5, default=1.5,
            ),
        ])
        s = space.summary()
        assert "Stop Loss" in s
        assert "1 parameters" in s


# ---------------------------------------------------------------------------
# ParameterSpaceLoader — profile loading from JSON
# ---------------------------------------------------------------------------

class TestParameterSpaceLoader:
    """Test loading profiles from JSON config."""

    @pytest.fixture
    def profiles_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "config" / "optimization_profiles.json"

    def test_load_conservative(self, profiles_path):
        loader = ParameterSpaceLoader(profiles_path)
        space = loader.load_profile("conservative")
        assert space.num_parameters == 4

    def test_load_moderate(self, profiles_path):
        loader = ParameterSpaceLoader(profiles_path)
        space = loader.load_profile("moderate")
        assert space.num_parameters == 7

    def test_load_aggressive(self, profiles_path):
        loader = ParameterSpaceLoader(profiles_path)
        space = loader.load_profile("aggressive")
        assert space.num_parameters == 7

    def test_conservative_under_500_combinations(self, profiles_path):
        loader = ParameterSpaceLoader(profiles_path)
        space = loader.load_profile("conservative")
        assert space.total_combinations < 500

    def test_moderate_valid(self, profiles_path):
        loader = ParameterSpaceLoader(profiles_path)
        space = loader.load_profile("moderate")
        is_valid, issues = space.validate()
        assert is_valid, f"moderate profile invalid: {issues}"

    def test_aggressive_same_param_count_as_moderate(self, profiles_path):
        loader = ParameterSpaceLoader(profiles_path)
        mod = loader.load_profile("moderate")
        agg = loader.load_profile("aggressive")
        assert mod.num_parameters == agg.num_parameters

    def test_aggressive_finer_steps(self, profiles_path):
        loader = ParameterSpaceLoader(profiles_path)
        mod = loader.load_profile("moderate")
        agg = loader.load_profile("aggressive")
        assert agg.total_combinations > mod.total_combinations

    def test_unknown_profile_raises(self, profiles_path):
        loader = ParameterSpaceLoader(profiles_path)
        with pytest.raises(ValueError, match="Unknown profile"):
            loader.load_profile("nonexistent")

    def test_list_profiles(self, profiles_path):
        loader = ParameterSpaceLoader(profiles_path)
        profiles = loader.list_profiles()
        names = [p["name"] for p in profiles]
        assert "conservative" in names
        assert "moderate" in names
        assert "aggressive" in names
        for p in profiles:
            assert "description" in p
            assert "param_count" in p

    def test_load_custom(self, profiles_path):
        loader = ParameterSpaceLoader(profiles_path)
        defs = [
            {
                "name": "x", "display_name": "X", "description": "t",
                "param_type": "float", "min_value": 1.0, "max_value": 2.0,
                "step": 0.5, "default": 1.5,
            },
        ]
        space = loader.load_custom(defs)
        assert space.num_parameters == 1

    def test_load_from_tmp_json(self):
        data = {
            "test_profile": {
                "description": "test",
                "parameters": [
                    {
                        "name": "x", "display_name": "X", "description": "t",
                        "param_type": "choice", "choices": ["A", "B"],
                        "default": "A",
                    },
                ],
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(data, f)
            f.flush()
            loader = ParameterSpaceLoader(f.name)
            space = loader.load_profile("test_profile")
            assert space.num_parameters == 1
            assert space.parameters[0].get_values() == ["A", "B"]


# ---------------------------------------------------------------------------
# ParameterApplicator
# ---------------------------------------------------------------------------

class TestParameterApplicator:
    """Test applying parameter dicts to BacktestConfig."""

    def test_applies_stop_loss(self):
        config = _base_config()
        applicator = ParameterApplicator()
        result = applicator.apply(config, {"stop_loss_atr_multiplier": 2.0, "target_atr_multiplier": 3.0})
        assert result is not None
        assert result.simulator_config.stop_loss_atr_mult == 2.0

    def test_applies_target(self):
        config = _base_config()
        applicator = ParameterApplicator()
        result = applicator.apply(config, {"stop_loss_atr_multiplier": 1.0, "target_atr_multiplier": 3.0})
        assert result is not None
        assert result.simulator_config.target_atr_mult == 3.0

    def test_applies_confidence_filter(self):
        config = _base_config()
        applicator = ParameterApplicator()
        result = applicator.apply(config, {"min_confidence_filter": "HIGH"})
        assert result is not None
        assert result.min_confidence == "HIGH"

    def test_applies_cooldown(self):
        config = _base_config()
        applicator = ParameterApplicator()
        result = applicator.apply(config, {"signal_cooldown_bars": 5})
        assert result is not None
        assert result.signal_cooldown_bars == 5

    def test_applies_risk_per_trade(self):
        config = _base_config()
        applicator = ParameterApplicator()
        result = applicator.apply(config, {"risk_per_trade_pct": 1.5})
        assert result is not None
        assert result.simulator_config.max_risk_per_trade_pct == 1.5

    def test_applies_trailing_sl(self):
        config = _base_config()
        applicator = ParameterApplicator()
        result = applicator.apply(config, {"trailing_sl_activation_pct": 0.6})
        assert result is not None
        assert result.simulator_config.trailing_sl_activation == 0.6

    def test_applies_max_positions(self):
        config = _base_config()
        applicator = ParameterApplicator()
        result = applicator.apply(config, {"max_positions": 5})
        assert result is not None
        assert result.simulator_config.max_open_positions == 5

    def test_returns_none_for_invalid_rr(self):
        """Target <= SL should return None."""
        config = _base_config()
        applicator = ParameterApplicator()
        result = applicator.apply(config, {
            "stop_loss_atr_multiplier": 2.5,
            "target_atr_multiplier": 2.0,
        })
        assert result is None

    def test_returns_none_for_equal_target_and_sl(self):
        config = _base_config()
        applicator = ParameterApplicator()
        result = applicator.apply(config, {
            "stop_loss_atr_multiplier": 2.0,
            "target_atr_multiplier": 2.0,
        })
        assert result is None

    def test_does_not_modify_original(self):
        config = _base_config()
        original_sl = config.simulator_config.stop_loss_atr_mult
        applicator = ParameterApplicator()
        applicator.apply(config, {"stop_loss_atr_multiplier": 99.0, "target_atr_multiplier": 100.0})
        assert config.simulator_config.stop_loss_atr_mult == original_sl

    def test_unknown_param_is_skipped(self):
        config = _base_config()
        applicator = ParameterApplicator()
        result = applicator.apply(config, {"nonexistent_param": 42})
        assert result is not None  # no crash, just warning

    def test_multiple_params_combined(self):
        config = _base_config()
        applicator = ParameterApplicator()
        result = applicator.apply(config, {
            "stop_loss_atr_multiplier": 1.0,
            "target_atr_multiplier": 2.5,
            "min_confidence_filter": "HIGH",
            "signal_cooldown_bars": 1,
            "risk_per_trade_pct": 1.0,
            "trailing_sl_activation_pct": 0.4,
            "max_positions": 2,
        })
        assert result is not None
        assert result.simulator_config.stop_loss_atr_mult == 1.0
        assert result.simulator_config.target_atr_mult == 2.5
        assert result.min_confidence == "HIGH"
        assert result.signal_cooldown_bars == 1
        assert result.simulator_config.max_risk_per_trade_pct == 1.0
        assert result.simulator_config.trailing_sl_activation == 0.4
        assert result.simulator_config.max_open_positions == 2
