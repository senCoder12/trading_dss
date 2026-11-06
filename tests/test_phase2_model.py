"""
Phase 2 regression tests — model logic fixes and calibration machinery.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.analysis.indicators.momentum import MomentumIndicators
from src.analysis.indicators.trend import TrendIndicators
from src.analysis.indicators.volatility import VolatilityIndicators
from src.engine.decision_engine import resample_session_bars
from src.engine.regime_detector import RegimeDetector
from src.engine.signal_calibration import (
    CALIBRATION_VERSION, SignalCalibration, _pav, brier, fit_bins,
)
from src.engine.signal_generator import SignalDeduplicator

_IST = ZoneInfo("Asia/Kolkata")


def _bars_5m(closes: list[float], start: datetime = datetime(2026, 11, 2, 9, 15, tzinfo=_IST)) -> pd.DataFrame:
    """5-minute session bars (09:15-15:25), rolling over to the next day."""
    rows, ts = [], start
    prev = closes[0]
    for c in closes:
        if ts.time() > datetime(2026, 1, 1, 15, 25).time():
            ts = (ts + timedelta(days=1)).replace(hour=9, minute=15)
        rows.append({"timestamp": ts, "open": prev, "high": max(prev, c) + 2, "low": min(prev, c) - 2,
                     "close": c, "volume": 0})
        prev = c
        ts += timedelta(minutes=5)
    return pd.DataFrame(rows)


# ── Trend: no built-in bearish bias ───────────────────────────────────────


class TestTrendVote:
    def test_uptrend_on_100_5m_bars_is_bullish_not_dragged_by_ema200(self):
        df = _bars_5m([24_000 + i * 3 for i in range(100)])
        t = TrendIndicators().get_trend_summary(df, index_id="NIFTY50", timeframe="5m")
        assert t.price_vs_ema200 == "UNKNOWN"          # not warmed up — not "BELOW"
        assert t.trend_vote in ("BULLISH", "STRONG_BULLISH")

    def test_downtrend_is_bearish(self):
        df = _bars_5m([24_000 - i * 3 for i in range(100)])
        t = TrendIndicators().get_trend_summary(df, index_id="NIFTY50", timeframe="5m")
        assert t.trend_vote in ("BEARISH", "STRONG_BEARISH")

    def test_symmetry(self):
        up = _bars_5m([24_000 + i * 3 for i in range(120)])
        down = _bars_5m([24_000 - i * 3 for i in range(120)])
        tu = TrendIndicators().get_trend_summary(up, index_id="X", timeframe="5m").trend_vote
        td = TrendIndicators().get_trend_summary(down, index_id="X", timeframe="5m").trend_vote
        assert tu.replace("BULLISH", "") == td.replace("BEARISH", "")


# ── Momentum: continuation, not contrarian ────────────────────────────────


class TestMomentumVote:
    def test_steady_rally_is_not_voted_bearish(self):
        rng = np.random.default_rng(1)
        closes = list(24_000 + np.cumsum(rng.normal(4, 6, 120)))
        m = MomentumIndicators().get_momentum_summary(_bars_5m(closes))
        assert m.momentum_vote in ("BULLISH", "STRONG_BULLISH", "NEUTRAL")
        assert m.momentum_vote not in ("BEARISH", "STRONG_BEARISH")

    def test_steady_selloff_is_not_voted_bullish(self):
        rng = np.random.default_rng(2)
        closes = list(24_000 + np.cumsum(rng.normal(-4, 6, 120)))
        m = MomentumIndicators().get_momentum_summary(_bars_5m(closes))
        assert m.momentum_vote not in ("BULLISH", "STRONG_BULLISH")

    def test_slow_stochastic_is_smoothed(self):
        rng = np.random.default_rng(3)
        df = _bars_5m(list(24_000 + np.cumsum(rng.normal(0, 10, 80))))
        res = MomentumIndicators().calculate_stochastic(df)
        raw_like = res.k_line.diff().abs().mean()
        assert res.d_line.diff().abs().mean() < raw_like


# ── Historical volatility annualisation ──────────────────────────────────


class TestHVAnnualisation:
    def test_5m_hv_matches_true_annual_vol(self):
        rng = np.random.default_rng(7)
        annual = 0.15
        per_bar = annual / math.sqrt(252 * 75)            # 75 five-minute bars per session
        rets = rng.normal(0, per_bar, 600)
        closes = list(24_000 * np.exp(np.cumsum(rets)))
        hv = VolatilityIndicators().calculate_historical_volatility(_bars_5m(closes), period=200)
        assert hv.current_hv == pytest.approx(15.0, rel=0.25)

    def test_daily_bars_still_use_252(self):
        rng = np.random.default_rng(8)
        rets = rng.normal(0, 0.15 / math.sqrt(252), 300)
        ts = pd.date_range("2025-01-01", periods=300, freq="D", tz=_IST)
        closes = 24_000 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({"timestamp": ts, "open": closes, "high": closes * 1.002,
                           "low": closes * 0.998, "close": closes, "volume": 1})
        hv = VolatilityIndicators().calculate_historical_volatility(df, period=200)
        assert hv.current_hv == pytest.approx(15.0, rel=0.25)


# ── Regime hysteresis / overrides ─────────────────────────────────────────


class TestRegimeHysteresis:
    def _rd(self, first: str = "RANGE_BOUND") -> RegimeDetector:
        rd = RegimeDetector(db=None)
        rd._store_regime("NIFTY50", first, datetime.now(tz=_IST))
        return rd

    def test_risk_on_switch_needs_confirmation(self):
        rd = self._rd()
        w: list[str] = []
        assert rd._apply_hysteresis("NIFTY50", "TREND_UP", w) == "RANGE_BOUND"
        assert any("awaiting confirmation" in x for x in w)
        assert rd._apply_hysteresis("NIFTY50", "TREND_UP", []) == "TREND_UP"

    def test_flip_flop_never_switches(self):
        rd = self._rd()
        for cand in ("TREND_UP", "RANGE_BOUND", "TREND_UP", "RANGE_BOUND"):
            out = rd._apply_hysteresis("NIFTY50", cand, [])
            rd._store_regime("NIFTY50", out, datetime.now(tz=_IST))
            assert out == "RANGE_BOUND"

    @pytest.mark.parametrize("risk_off", ["CRASH", "EVENT_DRIVEN", "VOLATILE_CHOPPY"])
    def test_risk_off_applies_immediately(self, risk_off):
        rd = self._rd("TREND_UP")
        assert rd._apply_hysteresis("NIFTY50", risk_off, []) == risk_off

    def test_negative_gex_does_not_upgrade_choppy_to_breakout(self):
        rd = RegimeDetector(db=None)
        out = rd._apply_advanced_signal_adjustments(
            regime="VOLATILE_CHOPPY", trend_regime="FLAT", volatility_regime="HIGH",
            gex_regime="NEGATIVE_GEX", gex_value=-1.0, fii_futures_bias=None,
            fii_net_position=None, ad_ratio=None, futures_basis=None, warnings=[],
        )
        assert out == "VOLATILE_CHOPPY"

    def test_negative_gex_can_upgrade_quiet_range(self):
        rd = RegimeDetector(db=None)
        out = rd._apply_advanced_signal_adjustments(
            regime="RANGE_BOUND", trend_regime="FLAT", volatility_regime="NORMAL",
            gex_regime="NEGATIVE_GEX", gex_value=-1.0, fii_futures_bias=None,
            fii_net_position=None, ad_ratio=None, futures_basis=None, warnings=[],
        )
        assert out == "BREAKOUT"


# ── Calibration machinery ─────────────────────────────────────────────────


class TestCalibrationFit:
    def test_pav_monotone(self):
        assert _pav([0.5, 0.3, 0.6], [1, 1, 1]) == pytest.approx([0.4, 0.4, 0.6])

    def test_fit_bins_monotone_and_shrunk(self):
        scores = [0.3] * 10 + [0.6] * 10 + [1.0] * 4
        wins = [0] * 7 + [1] * 3 + [1] * 6 + [0] * 4 + [1] * 4
        bins = fit_bins(scores, wins, [0.25, 0.5, 0.8, 99])
        ps = [b.p_win for b in bins]
        assert ps == sorted(ps)
        assert bins[-1].p_win < 1.0                    # 4/4 wins is not "certain"

    def test_brier(self):
        assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)

    def test_runtime_status_missing_stale_and_version(self, tmp_path):
        assert SignalCalibration(tmp_path / "nope.json").lookup("NIFTY50", 0.5)[0] is None
        old = tmp_path / "old.json"
        old.write_text(json.dumps({"version": CALIBRATION_VERSION,
                                   "generated_at": (datetime.now(tz=_IST) - timedelta(days=400)).isoformat(),
                                   "models": {}}))
        p, why = SignalCalibration(old).lookup("NIFTY50", 0.5)
        assert p is None and "days old" in why
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"version": 999, "generated_at": datetime.now(tz=_IST).isoformat()}))
        assert "version" in SignalCalibration(bad).lookup("NIFTY50", 0.5)[1]

    def test_file_change_is_picked_up(self, tmp_path):
        path = tmp_path / "c.json"
        cal = SignalCalibration(path)
        assert cal.lookup("NIFTY50", 0.5)[0] is None
        path.write_text(json.dumps({
            "version": CALIBRATION_VERSION, "generated_at": datetime.now(tz=_IST).isoformat(),
            "models": {"ALL": {"bins": [{"lo": 0.25, "hi": 99, "n": 50, "wins": 30, "p_win": 0.57}],
                               "validated": True, "meta": {}}},
        }))
        assert cal.lookup("NIFTY50", 0.5)[0] == 0.57


# ── Supporting helpers ────────────────────────────────────────────────────


class TestHelpers:
    def test_resample_never_straddles_sessions(self):
        df = _bars_5m([24_000 + i for i in range(160)])
        r = resample_session_bars(df, "15min")
        assert r is not None
        src_days = pd.to_datetime(df["timestamp"]).dt.date.value_counts()
        out_days = pd.to_datetime(r["timestamp"]).dt.date.value_counts()
        for day, n in src_days.items():
            assert out_days[day] == math.ceil(n / 3)   # 09:15-aligned, per session
        first = pd.to_datetime(df["timestamp"]).iloc[0]
        assert pd.to_datetime(r["timestamp"]).iloc[0] == first
        assert r["high"].iloc[0] == df["high"].iloc[:3].max()

    def test_dedup_uses_signal_time(self):
        d = SignalDeduplicator(window_seconds=900)
        t0 = datetime(2026, 4, 7, 10, 0, tzinfo=_IST)
        d.record("NIFTY50", "BUY_CALL", t0)
        assert d.is_duplicate("NIFTY50", "BUY_CALL", t0 + timedelta(minutes=5))
        assert not d.is_duplicate("NIFTY50", "BUY_PUT", t0 + timedelta(minutes=5))
        assert not d.is_duplicate("NIFTY50", "BUY_CALL", t0 + timedelta(minutes=16))
