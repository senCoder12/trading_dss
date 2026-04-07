"""
Tests for Phase 4 -- Step 4.3: FII/DII Flow Anomalies & Cross-Index Divergence.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.analysis.anomaly.flow_divergence_detector import (
    CrossIndexDivergenceDetector,
    DEFAULT_INDEX_PAIRS,
    DivergenceAnomaly,
    FIIBias,
    FIIFlowDetector,
    FlowBaselines,
    SectorRotation,
)
from src.analysis.anomaly.volume_price_detector import AnomalyEvent
from src.data.fii_dii_data import FIIDIIData

_IST = ZoneInfo("Asia/Kolkata")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_baselines(**overrides) -> FlowBaselines:
    defaults = dict(
        avg_fii_net_daily=-200.0,
        std_fii_net_daily=800.0,
        avg_dii_net_daily=300.0,
        std_dii_net_daily=600.0,
        fii_trend_5d=-1000.0,
        fii_trend_10d=-2000.0,
        dii_trend_5d=1500.0,
        consecutive_fii_buy_days=0,
        consecutive_fii_sell_days=2,
        avg_fii_fo_net=-100.0,
        std_fii_fo_net=500.0,
    )
    defaults.update(overrides)
    return FlowBaselines(**defaults)


def _make_fii_data(
    fii_net: float = 0.0,
    dii_net: float = 0.0,
    fii_fo_net: float = 0.0,
    trade_date: date | None = None,
) -> FIIDIIData:
    d = trade_date or date.today()
    return FIIDIIData(
        date=d,
        fii_buy_value=max(0, fii_net),
        fii_sell_value=max(0, -fii_net),
        fii_net_value=fii_net,
        dii_buy_value=max(0, dii_net),
        dii_sell_value=max(0, -dii_net),
        dii_net_value=dii_net,
        fii_fo_buy=max(0, fii_fo_net),
        fii_fo_sell=max(0, -fii_fo_net),
        fii_fo_net=fii_fo_net,
    )


def _make_price_df(
    closes: list[float],
    start_date: date | None = None,
) -> pd.DataFrame:
    """Build a minimal price DataFrame with ``date`` and ``close`` columns."""
    start = start_date or (date.today() - timedelta(days=len(closes)))
    dates = [start + timedelta(days=i) for i in range(len(closes))]
    return pd.DataFrame({"date": dates, "close": closes})


def _correlated_series(n: int = 25, base_close: float = 22000.0, corr: float = 0.9):
    """Return two DataFrames whose returns are correlated at *corr*."""
    rng = np.random.default_rng(42)
    ret1 = rng.normal(0, 0.01, n)
    noise = rng.normal(0, 0.01, n)
    ret2 = corr * ret1 + np.sqrt(1 - corr ** 2) * noise

    close1 = [base_close]
    close2 = [45000.0]
    for r1, r2 in zip(ret1, ret2):
        close1.append(close1[-1] * (1 + r1))
        close2.append(close2[-1] * (1 + r2))

    start = date.today() - timedelta(days=n)
    dates = [start + timedelta(days=i) for i in range(n + 1)]
    df1 = pd.DataFrame({"date": dates, "close": close1})
    df2 = pd.DataFrame({"date": dates, "close": close2})
    return df1, df2


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def detector(tmp_path):
    """FIIFlowDetector with empty in-memory DB."""
    from src.database.db_manager import DatabaseManager
    from src.database import queries as Q

    db = DatabaseManager(db_path=tmp_path / "test.db")
    db.connect()
    db.initialise_schema()

    now = datetime.now(tz=_IST).isoformat()
    db.execute(
        Q.INSERT_INDEX_MASTER,
        ("NIFTY50", "NIFTY 50", "NIFTY 50", "^NSEI", "NSE",
         75, 1, "NIFTY", "broad_market", 1, now, now),
    )
    return FIIFlowDetector(db)


@pytest.fixture()
def div_detector(tmp_path):
    """CrossIndexDivergenceDetector with empty in-memory DB."""
    from src.database.db_manager import DatabaseManager
    from src.database import queries as Q

    db = DatabaseManager(db_path=tmp_path / "test.db")
    db.connect()
    db.initialise_schema()

    now = datetime.now(tz=_IST).isoformat()
    for idx_id in ("NIFTY50", "BANKNIFTY", "SENSEX", "NIFTYIT", "INDIA_VIX"):
        db.execute(
            Q.INSERT_INDEX_MASTER,
            (idx_id, idx_id, idx_id, f"^{idx_id}", "NSE",
             75, 1, idx_id, "broad_market", 1, now, now),
        )
    return CrossIndexDivergenceDetector(db)


# ═══════════════════════════════════════════════════════════════════════════
# FII Flow Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFIIExtremeFlow:
    """FII extreme flow: Z-score > 3 should be detected as HIGH severity."""

    def test_extreme_selling_zscore_above_3(self, detector):
        baselines = _make_baselines(avg_fii_net_daily=-200.0, std_fii_net_daily=800.0)
        # FII net = -3000 → Z = (-3000 - -200) / 800 = -3.5
        today = _make_fii_data(fii_net=-3000.0)
        events = detector.detect_fii_anomaly(today, baselines)

        extreme = [e for e in events if e.anomaly_type == "FII_EXTREME_FLOW"]
        assert len(extreme) == 1
        assert extreme[0].severity == "HIGH"
        details = json.loads(extreme[0].details)
        assert details["z_score"] < -3.0
        assert details["direction"] == "selling"

    def test_extreme_buying_zscore_above_3(self, detector):
        baselines = _make_baselines(avg_fii_net_daily=-200.0, std_fii_net_daily=800.0)
        # FII net = +3000 → Z = (3000 - -200) / 800 = 4.0
        today = _make_fii_data(fii_net=3000.0)
        events = detector.detect_fii_anomaly(today, baselines)

        extreme = [e for e in events if e.anomaly_type == "FII_EXTREME_FLOW"]
        assert len(extreme) == 1
        assert extreme[0].severity == "HIGH"
        details = json.loads(extreme[0].details)
        assert details["direction"] == "buying"

    def test_unusual_flow_zscore_between_2_and_3(self, detector):
        baselines = _make_baselines(avg_fii_net_daily=0.0, std_fii_net_daily=1000.0)
        # FII net = -2500 → Z = -2.5
        today = _make_fii_data(fii_net=-2500.0)
        events = detector.detect_fii_anomaly(today, baselines)

        unusual = [e for e in events if e.anomaly_type == "FII_UNUSUAL_FLOW"]
        assert len(unusual) == 1
        assert unusual[0].severity == "MEDIUM"

    def test_normal_flow_no_alert(self, detector):
        baselines = _make_baselines(avg_fii_net_daily=0.0, std_fii_net_daily=1000.0)
        # FII net = -500 → Z = -0.5 (well within normal)
        today = _make_fii_data(fii_net=-500.0)
        events = detector.detect_fii_anomaly(today, baselines)

        flow_events = [e for e in events if "FLOW" in e.anomaly_type]
        assert len(flow_events) == 0


class TestFIITrendReversal:
    """5+ consecutive buying days then a sudden sell day → reversal."""

    def test_buy_streak_broken(self, detector):
        baselines = _make_baselines(consecutive_fii_buy_days=6)
        today = _make_fii_data(fii_net=-800.0)
        events = detector.detect_fii_anomaly(today, baselines)

        reversals = [e for e in events if e.anomaly_type == "FII_TREND_REVERSAL"]
        assert len(reversals) == 1
        assert reversals[0].severity == "HIGH"
        details = json.loads(reversals[0].details)
        assert details["reversal_direction"] == "BUY_TO_SELL"

    def test_sell_streak_broken(self, detector):
        baselines = _make_baselines(
            consecutive_fii_sell_days=7,
            consecutive_fii_buy_days=0,
        )
        today = _make_fii_data(fii_net=500.0)
        events = detector.detect_fii_anomaly(today, baselines)

        reversals = [e for e in events if e.anomaly_type == "FII_TREND_REVERSAL"]
        assert len(reversals) == 1
        details = json.loads(reversals[0].details)
        assert details["reversal_direction"] == "SELL_TO_BUY"

    def test_short_streak_no_reversal(self, detector):
        baselines = _make_baselines(consecutive_fii_buy_days=3)
        today = _make_fii_data(fii_net=-500.0)
        events = detector.detect_fii_anomaly(today, baselines)

        reversals = [e for e in events if e.anomaly_type == "FII_TREND_REVERSAL"]
        assert len(reversals) == 0


class TestFIIDIIDivergence:
    """FII selling heavily while DII buying heavily → divergence."""

    def test_fii_selling_dii_buying(self, detector):
        baselines = _make_baselines()
        today = _make_fii_data(fii_net=-3000.0, dii_net=2500.0)
        events = detector.detect_fii_anomaly(today, baselines)

        div = [e for e in events if e.anomaly_type == "FII_DII_DIVERGENCE"]
        assert len(div) == 1
        assert div[0].severity == "MEDIUM"
        assert "absorbing" in div[0].message

    def test_fii_buying_dii_selling(self, detector):
        baselines = _make_baselines()
        today = _make_fii_data(fii_net=2500.0, dii_net=-2500.0)
        events = detector.detect_fii_anomaly(today, baselines)

        div = [e for e in events if e.anomaly_type == "FII_DII_DIVERGENCE"]
        assert len(div) == 1
        assert "Conflicting" in div[0].message

    def test_no_divergence_below_threshold(self, detector):
        baselines = _make_baselines()
        today = _make_fii_data(fii_net=-1500.0, dii_net=1800.0)
        events = detector.detect_fii_anomaly(today, baselines)

        div = [e for e in events if e.anomaly_type == "FII_DII_DIVERGENCE"]
        assert len(div) == 0


class TestFIIFOExtreme:
    """FII F&O segment with extreme Z-score."""

    def test_fo_extreme_detected(self, detector):
        baselines = _make_baselines(avg_fii_fo_net=0.0, std_fii_fo_net=500.0)
        # FII F&O net = -2000 → Z = -4.0
        today = _make_fii_data(fii_fo_net=-2000.0)
        events = detector.detect_fii_anomaly(today, baselines)

        fo = [e for e in events if e.anomaly_type == "FII_FO_EXTREME"]
        assert len(fo) == 1
        assert fo[0].severity == "HIGH"
        details = json.loads(fo[0].details)
        assert details["direction"] == "short"

    def test_fo_normal_no_alert(self, detector):
        baselines = _make_baselines(avg_fii_fo_net=0.0, std_fii_fo_net=500.0)
        today = _make_fii_data(fii_fo_net=-200.0)
        events = detector.detect_fii_anomaly(today, baselines)

        fo = [e for e in events if e.anomaly_type == "FII_FO_EXTREME"]
        assert len(fo) == 0


class TestFIIBias:
    """FII bias calculation."""

    def test_strong_buying_bias(self, detector):
        detector._baselines = _make_baselines(
            fii_trend_5d=6000.0, fii_trend_10d=10000.0,
        )
        today = _make_fii_data(fii_net=1500.0)
        bias = detector.get_fii_bias(today)

        assert bias.bias == "STRONG_BUYING"
        assert bias.confidence > 0.6
        assert bias.today_net == 1500.0

    def test_selling_bias(self, detector):
        detector._baselines = _make_baselines(
            fii_trend_5d=-2000.0, fii_trend_10d=-3000.0,
        )
        bias = detector.get_fii_bias()

        assert bias.bias == "SELLING"

    def test_neutral_bias(self, detector):
        detector._baselines = _make_baselines(
            fii_trend_5d=200.0, fii_trend_10d=500.0,
        )
        bias = detector.get_fii_bias()

        assert bias.bias == "NEUTRAL"
        assert bias.confidence <= 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Index Divergence Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCorrelationBreakdown:
    """Two correlated series that suddenly diverge → correlation breakdown."""

    def test_breakdown_detected(self, div_detector):
        # Build two series: highly correlated for 15 days, then diverge sharply
        n = 25
        start = date.today() - timedelta(days=n)
        dates = [start + timedelta(days=i) for i in range(n + 1)]

        close1 = [22000.0]
        close2 = [45000.0]
        rng = np.random.default_rng(99)
        for i in range(n):
            r = rng.normal(0, 0.005)
            close1.append(close1[-1] * (1 + r))
            if i < 20:
                close2.append(close2[-1] * (1 + r * 0.9 + rng.normal(0, 0.001)))
            else:
                # Last 5 days: index2 goes opposite direction
                close2.append(close2[-1] * (1 - r * 1.5))

        df1 = pd.DataFrame({"date": dates, "close": close1})
        df2 = pd.DataFrame({"date": dates, "close": close2})

        result = div_detector.detect_divergence("NIFTY50_vs_BANKNIFTY", df1, df2)
        # The sharp divergence should produce some anomaly
        # (correlation breakdown, directional divergence, or relative strength shift)
        assert result is not None
        assert result.anomaly_type in (
            "CORRELATION_BREAKDOWN",
            "DIRECTIONAL_DIVERGENCE",
            "RELATIVE_STRENGTH_SHIFT",
        )

    def test_no_breakdown_when_correlated(self, div_detector):
        df1, df2 = _correlated_series(n=25, corr=0.95)
        result = div_detector.detect_divergence("NIFTY50_vs_SENSEX", df1, df2)
        # Highly correlated series should not trigger breakdown
        # (may still trigger a minor relative strength shift due to randomness)
        if result is not None:
            assert result.anomaly_type != "CORRELATION_BREAKDOWN"


class TestDirectionalDivergence:
    """One index up > 0.5%, other down > 0.5% for highly correlated pair."""

    def test_opposite_direction_detected(self, div_detector):
        n = 25
        start = date.today() - timedelta(days=n)
        dates = [start + timedelta(days=i) for i in range(n + 1)]

        # Build correlated series with divergent last day
        close1 = [22000.0]
        close2 = [45000.0]
        rng = np.random.default_rng(42)
        for i in range(n - 1):
            r = rng.normal(0, 0.005)
            close1.append(close1[-1] * (1 + r))
            close2.append(close2[-1] * (1 + r * 0.95))
        # Last day: index1 up 1%, index2 down 1%
        close1.append(close1[-1] * 1.01)
        close2.append(close2[-1] * 0.99)

        df1 = pd.DataFrame({"date": dates, "close": close1})
        df2 = pd.DataFrame({"date": dates, "close": close2})

        result = div_detector.detect_divergence("NIFTY50_vs_BANKNIFTY", df1, df2)
        assert result is not None
        assert result.anomaly_type in ("DIRECTIONAL_DIVERGENCE", "CORRELATION_BREAKDOWN", "RELATIVE_STRENGTH_SHIFT")

    def test_same_direction_no_divergence(self, div_detector):
        n = 25
        start = date.today() - timedelta(days=n)
        dates = [start + timedelta(days=i) for i in range(n + 1)]

        close1 = [22000.0]
        close2 = [45000.0]
        rng = np.random.default_rng(42)
        for i in range(n):
            r = rng.normal(0.001, 0.003)  # both trending up
            close1.append(close1[-1] * (1 + r))
            close2.append(close2[-1] * (1 + r * 0.95))

        df1 = pd.DataFrame({"date": dates, "close": close1})
        df2 = pd.DataFrame({"date": dates, "close": close2})

        result = div_detector.detect_divergence("NIFTY50_vs_BANKNIFTY", df1, df2)
        # Both moving same direction → likely no divergence
        if result is not None:
            assert result.anomaly_type not in ("DIRECTIONAL_DIVERGENCE", "VIX_DIVERGENCE")


class TestVIXDivergence:
    """NIFTY up + VIX up → unstable rally; NIFTY down + VIX down → hidden weakness."""

    def test_nifty_up_vix_up(self, div_detector):
        n = 25
        start = date.today() - timedelta(days=n)
        dates = [start + timedelta(days=i) for i in range(n + 1)]

        close1 = [22000.0]
        close2 = [15.0]  # VIX
        rng = np.random.default_rng(42)
        for i in range(n - 1):
            r = rng.normal(0, 0.005)
            close1.append(close1[-1] * (1 + r))
            close2.append(close2[-1] * (1 - r * 0.8))  # normally inverse
        # Last day: both up
        close1.append(close1[-1] * 1.008)
        close2.append(close2[-1] * 1.008)

        df1 = pd.DataFrame({"date": dates, "close": close1})
        df2 = pd.DataFrame({"date": dates, "close": close2})

        result = div_detector.detect_divergence("NIFTY50_vs_INDIA_VIX", df1, df2)
        assert result is not None
        assert result.anomaly_type == "VIX_DIVERGENCE"
        assert result.implication == "UNSTABLE_RALLY"
        assert result.severity == "HIGH"

    def test_nifty_down_vix_down(self, div_detector):
        n = 25
        start = date.today() - timedelta(days=n)
        dates = [start + timedelta(days=i) for i in range(n + 1)]

        close1 = [22000.0]
        close2 = [15.0]
        rng = np.random.default_rng(42)
        for i in range(n - 1):
            r = rng.normal(0, 0.005)
            close1.append(close1[-1] * (1 + r))
            close2.append(close2[-1] * (1 - r * 0.8))
        # Last day: both down
        close1.append(close1[-1] * 0.992)
        close2.append(close2[-1] * 0.992)

        df1 = pd.DataFrame({"date": dates, "close": close1})
        df2 = pd.DataFrame({"date": dates, "close": close2})

        result = div_detector.detect_divergence("NIFTY50_vs_INDIA_VIX", df1, df2)
        assert result is not None
        assert result.anomaly_type == "VIX_DIVERGENCE"
        assert result.implication == "HIDDEN_WEAKNESS"


class TestSectorRotation:
    """Multiple sector indices with diverging 5-day returns."""

    def test_rotation_detected(self, div_detector):
        today = date.today()
        sectors = {
            "NIFTYIT": _make_price_df(
                [1000, 1010, 1025, 1040, 1060, 1080],
                start_date=today - timedelta(days=5),
            ),
            "NIFTYPHARMA": _make_price_df(
                [500, 505, 512, 520, 530, 540],
                start_date=today - timedelta(days=5),
            ),
            "NIFTY_FMCG": _make_price_df(
                [800, 808, 815, 824, 832, 840],
                start_date=today - timedelta(days=5),
            ),
            "NIFTY_METAL": _make_price_df(
                [2000, 1980, 1950, 1920, 1900, 1880],
                start_date=today - timedelta(days=5),
            ),
            "NIFTY_REALTY": _make_price_df(
                [600, 592, 580, 570, 565, 555],
                start_date=today - timedelta(days=5),
            ),
            "NIFTY_AUTO": _make_price_df(
                [1200, 1188, 1175, 1165, 1158, 1150],
                start_date=today - timedelta(days=5),
            ),
        }

        rotations = div_detector.detect_sector_rotation(sectors)
        assert len(rotations) >= 1
        rot = rotations[0]
        assert rot.strength in ("STRONG", "MODERATE")
        assert len(rot.outperforming_sectors) > 0
        assert len(rot.underperforming_sectors) > 0

    def test_no_rotation_when_all_similar(self, div_detector):
        today = date.today()
        sectors = {}
        for idx in ("NIFTYIT", "NIFTYPHARMA", "NIFTY_METAL", "NIFTY_REALTY"):
            sectors[idx] = _make_price_df(
                [1000, 1002, 1005, 1003, 1006, 1008],
                start_date=today - timedelta(days=5),
            )

        rotations = div_detector.detect_sector_rotation(sectors)
        assert len(rotations) == 0

    def test_broad_selloff(self, div_detector):
        today = date.today()
        sectors = {}
        for idx in ("NIFTYIT", "NIFTYPHARMA", "NIFTY_METAL", "NIFTY_REALTY"):
            sectors[idx] = _make_price_df(
                [1000, 985, 970, 960, 945, 930],
                start_date=today - timedelta(days=5),
            )

        rotations = div_detector.detect_sector_rotation(sectors)
        assert len(rotations) >= 1
        assert "selloff" in rotations[0].message.lower() or "negative" in rotations[0].message.lower()


class TestInsufficientHistory:
    """Gracefully skip when not enough data."""

    def test_divergence_skips_short_history(self, div_detector):
        df1 = _make_price_df([100, 101, 102])
        df2 = _make_price_df([200, 201, 202])
        result = div_detector.detect_divergence("NIFTY50_vs_BANKNIFTY", df1, df2)
        assert result is None

    def test_sector_rotation_skips_short_series(self, div_detector):
        sectors = {
            "NIFTYIT": _make_price_df([100, 105]),
            "NIFTYPHARMA": _make_price_df([200, 190]),
        }
        rotations = div_detector.detect_sector_rotation(sectors)
        assert len(rotations) == 0

    def test_unknown_pair_returns_none(self, div_detector):
        df1 = _make_price_df([100] * 25)
        df2 = _make_price_df([200] * 25)
        result = div_detector.detect_divergence("NONEXISTENT_PAIR", df1, df2)
        assert result is None


class TestCumulativeFlowShift:
    """Cumulative 5-day FII flow sign reversal."""

    def test_shift_detected(self, detector):
        # Manually set up daily data with 10+ rows
        dates = [(date.today() - timedelta(days=i)).isoformat() for i in range(14, -1, -1)]
        fii_vals = [500, 600, 400, 500, 700,   # first 5: sum = +2700
                    300, 200, 100, -50, -100,   # next 5: sum = +450
                    -200, -300, -400, -500, -600]  # last 5: sum = -2000
        dii_vals = [0.0] * 15
        detector._daily_data = pd.DataFrame({
            "date": dates,
            "fii_net": fii_vals,
            "dii_net": dii_vals,
        })

        baselines = _make_baselines()
        # Today: slightly negative, making current 5d even more negative
        today = _make_fii_data(fii_net=-100.0)
        events = detector.detect_fii_anomaly(today, baselines)

        shifts = [e for e in events if e.anomaly_type == "FII_CUMULATIVE_SHIFT"]
        assert len(shifts) == 1
        assert shifts[0].severity == "MEDIUM"


class TestDetectAllDivergences:
    """Integration test for detect_all_divergences producing unified AnomalyEvents."""

    def test_produces_anomaly_events(self, div_detector):
        n = 25
        start = date.today() - timedelta(days=n)
        dates = [start + timedelta(days=i) for i in range(n + 1)]

        # NIFTY and VIX both going up on last day (VIX divergence)
        close_nifty = [22000.0]
        close_vix = [15.0]
        rng = np.random.default_rng(42)
        for i in range(n - 1):
            r = rng.normal(0, 0.005)
            close_nifty.append(close_nifty[-1] * (1 + r))
            close_vix.append(close_vix[-1] * (1 - r * 0.8))
        close_nifty.append(close_nifty[-1] * 1.01)
        close_vix.append(close_vix[-1] * 1.01)

        index_data = {
            "NIFTY50": pd.DataFrame({"date": dates, "close": close_nifty}),
            "INDIA_VIX": pd.DataFrame({"date": dates, "close": close_vix}),
        }

        events = div_detector.detect_all_divergences(index_data=index_data)
        # Should produce at least the VIX divergence event
        vix_events = [e for e in events if e.anomaly_type == "VIX_DIVERGENCE"]
        assert len(vix_events) >= 1
        assert vix_events[0].category == "DIVERGENCE"
        # Should have valid JSON details
        details = json.loads(vix_events[0].details)
        assert "implication" in details


class TestFlowBaselinesComputation:
    """Test compute_flow_baselines with seeded DB data."""

    def test_baselines_from_db(self, tmp_path):
        from src.database.db_manager import DatabaseManager
        from src.database import queries as Q

        db = DatabaseManager(db_path=tmp_path / "test.db")
        db.connect()
        db.initialise_schema()

        now = datetime.now(tz=_IST).isoformat()
        db.execute(
            Q.INSERT_INDEX_MASTER,
            ("NIFTY50", "NIFTY 50", "NIFTY 50", "^NSEI", "NSE",
             75, 1, "NIFTY", "broad_market", 1, now, now),
        )

        # Seed 15 days of FII/DII data
        for i in range(15):
            d = (date.today() - timedelta(days=i)).isoformat()
            fii_net = (-1) ** i * 500.0 + i * 50  # oscillating
            dii_net = -fii_net * 0.8
            db.execute(Q.INSERT_FII_DII_ACTIVITY, (d, "FII", 5000.0, 5000.0 - fii_net, fii_net, "CASH"))
            db.execute(Q.INSERT_FII_DII_ACTIVITY, (d, "DII", 4000.0, 4000.0 - dii_net, dii_net, "CASH"))

        det = FIIFlowDetector(db)
        bl = det.compute_flow_baselines()

        assert bl.avg_fii_net_daily != 0.0
        assert bl.std_fii_net_daily > 0.0
        assert bl.avg_dii_net_daily != 0.0
        assert bl.fii_trend_5d != 0.0
