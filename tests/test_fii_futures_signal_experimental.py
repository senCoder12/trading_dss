"""
Experimental FII futures signal: the replica must match the rule coded in
SmartMoneyIndicators, and nothing experimental may be wired into live paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.experimental.fii_futures_signal import (
    add_features_and_targets,
    boost_distribution,
    information_coefficient,
    proposed_boost,
)
from src.analysis.indicators.smart_money import SmartMoneyIndicators

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("net, expected", [
    (0, 0.0), (4999, 0.0), (-4999, 0.0), (5000, 10.0), (-5000, -10.0),
    (10000, 20.0), (20000, 40.0), (-137438, -40.0), (392157, 40.0),
])
def test_proposed_boost_formula(net, expected) -> None:
    assert proposed_boost(net) == pytest.approx(expected)


def test_replica_matches_live_smart_money_rule() -> None:
    """Feed identical inputs with and without fii_futures_net and compare the OIMI shift."""
    n = 60
    idx = pd.date_range("2026-10-01 09:15", periods=n, freq="5min")
    price = pd.DataFrame({
        "timestamp": idx, "open": 100.0, "high": 101.0, "low": 99.0,
        "close": np.linspace(100, 103, n), "volume": 1000.0,
    })
    oi = [{"timestamp": t.isoformat(), "total_oi": 1_000_000 + i * 1_000} for i, t in enumerate(idx)]
    sm = SmartMoneyIndicators()
    base = sm.calculate_smart_money_score(price, oi_data=oi)
    for net in (-137438, -7000, 3000, 9000):
        boosted = sm.calculate_smart_money_score(price, oi_data=oi, fii_futures_net=net)
        expected = max(-100.0, min(100.0, base.oimi_component + proposed_boost(net)))
        assert boosted.oimi_component == pytest.approx(expected)


def test_not_wired_into_live_paths() -> None:
    for path in list((ROOT / "src" / "engine").rglob("*.py")) + [
        ROOT / "src" / "analysis" / "technical_aggregator.py",
        ROOT / "src" / "data" / "data_collector.py",
    ]:
        assert "experimental" not in path.read_text(), f"{path} imports experimental code"


def test_metrics_on_synthetic_data() -> None:
    idx = pd.bdate_range("2025-01-01", periods=120)
    rng = np.random.default_rng(0)
    d = pd.DataFrame(index=idx)
    d["fii_long"] = 50000 + rng.integers(0, 1000, len(idx))
    d["fii_short"] = 190000 + rng.integers(0, 1000, len(idx))
    d["fii_net"] = d["fii_long"] - d["fii_short"]
    d["fii_long_ratio"] = d["fii_long"] / (d["fii_long"] + d["fii_short"])
    d["client_net"] = -d["fii_net"]
    d["total_idx_fut_oi"] = 400000.0
    d["nifty_close"] = 24000 * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
    d = add_features_and_targets(d)

    dist = boost_distribution(d)
    assert dist["at_-40"] == 1.0 and dist["sign_flips"] == 0   # level sits on one side → constant tilt
    r = information_coefficient(d, "fii_net_chg", 1)
    assert r.n == len(idx) - 2 and -1 <= r.ic <= 1
