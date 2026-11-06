"""Futures buildup / order-flow component."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.analysis.futures_flow import FLOW_WEIGHT, FuturesFlow, blend_score, compute_futures_flow

_IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 11, 6, 11, 0, tzinfo=_IST)


def _rows(dp=0.002, doi=0.01, ofi=100.0, n=30, volume=1000, end=NOW - timedelta(minutes=1)):
    out = []
    for i in range(n):
        f = i / (n - 1)
        out.append({
            "timestamp": (end - timedelta(minutes=n - 1 - i)).isoformat(),
            "fut_close": 25000 * (1 + dp * f), "open_interest": 1e7 * (1 + doi * f),
            "volume": volume, "ofi": ofi / n,
        })
    return out


@pytest.mark.parametrize("dp,doi,buildup,vote", [
    (0.002, 0.01, "LONG_BUILDUP", "STRONG_BULLISH"),
    (-0.002, 0.01, "SHORT_BUILDUP", "STRONG_BEARISH"),
    (0.002, -0.01, "SHORT_COVERING", "BULLISH"),
    (-0.002, -0.01, "LONG_UNWINDING", "BEARISH"),
])
def test_buildup_classification(dp, doi, buildup, vote):
    ofi = 100.0 if dp > 0 else -100.0
    f = compute_futures_flow(_rows(dp, doi, ofi), NOW)
    assert f.available and f.buildup == buildup and f.vote == vote


def test_contradicting_order_flow_weakens():
    f = compute_futures_flow(_rows(0.002, 0.01, ofi=-500.0), NOW)
    assert f.vote == "BULLISH"            # strong → normal
    f2 = compute_futures_flow(_rows(0.002, -0.01, ofi=-500.0), NOW)
    assert f2.vote == "NEUTRAL"           # weak → neutral


def test_noise_is_neutral():
    f = compute_futures_flow(_rows(0.0001, 0.0001), NOW)
    assert f.available and f.vote == "NEUTRAL"


@pytest.mark.parametrize("rows,why", [
    (_rows(n=10), "only"),
    (_rows(end=NOW - timedelta(minutes=20)), "only"),       # mostly outside lookback
    (_rows(volume=0), "no futures volume"),
    ([], "only"),
])
def test_unavailable_cases(rows, why):
    f = compute_futures_flow(rows, NOW)
    assert not f.available and why in f.reason


def test_future_rows_ignored():
    rows = _rows() + [{"timestamp": (NOW + timedelta(minutes=5)).isoformat(),
                       "fut_close": 1, "open_interest": 1, "volume": 1, "ofi": -1e9}]
    assert compute_futures_flow(rows, NOW).vote == "STRONG_BULLISH"


def test_blend_only_when_available():
    vn = {"STRONG_BULLISH": 2.0, "NEUTRAL": 0.0}
    assert blend_score(0.5, None, vn) == 0.5
    assert blend_score(0.5, FuturesFlow(False, "x"), vn) == 0.5
    f = FuturesFlow(True, "", "LONG_BUILDUP", "STRONG_BULLISH", 0.75)
    assert blend_score(0.5, f, vn) == pytest.approx((1 - FLOW_WEIGHT) * 0.5 + FLOW_WEIGHT * 1.5)


def test_model_score_uses_flow():
    from src.engine.signal_generator import SignalGenerator
    from tests.test_signal_generator import _regime, _technical

    tech = replace(_technical(), volume_available=False)
    base, votes, comps = SignalGenerator.model_score(tech, _regime())
    tech.futures_flow = FuturesFlow(True, "", "SHORT_BUILDUP", "STRONG_BEARISH", 0.75)
    blended, votes2, comps2 = SignalGenerator.model_score(tech, _regime())
    assert blended < base and "futures_flow" in votes2 and "futures_flow" in comps2
