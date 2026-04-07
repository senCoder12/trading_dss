"""
Tests for Phase 4 — Step 4.1: Volume & Price Anomaly Detector.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.analysis.anomaly.volume_price_detector import (
    AnomalyEvent,
    Baselines,
    PriceAnomaly,
    VolumeAnomaly,
    VolumePriceDetector,
)

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def baselines() -> Baselines:
    """Typical 20-day baselines for a NIFTY-like index."""
    return Baselines(
        index_id="NIFTY50",
        computed_at=datetime(2024, 6, 15, 15, 30, tzinfo=_IST),
        avg_volume_20d=1_000_000,
        std_volume_20d=200_000,
        avg_range_20d=150.0,
        avg_range_pct_20d=0.007,
        avg_body_20d=80.0,
        avg_intraday_volatility=0.007,
        typical_first_15min_range=60.0,
        typical_last_30min_volume=250_000,
    )


@pytest.fixture()
def detector(tmp_path) -> VolumePriceDetector:
    """Detector with an isolated in-memory DB and seeded index_master."""
    from src.database.db_manager import DatabaseManager
    from src.database import queries as Q

    db = DatabaseManager(db_path=tmp_path / "test.db")
    db.connect()
    db.initialise_schema()

    # Seed index_master so FK constraints pass on anomaly_events insert
    now = datetime.now(tz=_IST).isoformat()
    db.execute(
        Q.INSERT_INDEX_MASTER,
        ("NIFTY50", "NIFTY 50", "NIFTY 50", "^NSEI", "NSE",
         75, 1, "NIFTY", "broad_market", 1, now, now),
    )
    return VolumePriceDetector(db)


def _bar(
    ts: datetime,
    open_: float = 22000.0,
    high: float = 22100.0,
    low: float = 21900.0,
    close: float = 22050.0,
    volume: float = 1_000_000,
) -> dict:
    """Helper to build a bar dict."""
    return {
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "timeframe": "5m",
    }


def _ts(hour: int, minute: int = 0, day: int = 17) -> datetime:
    """IST timestamp helper (defaults to 2024-06-17)."""
    return datetime(2024, 6, day, hour, minute, tzinfo=_IST)


# ---------------------------------------------------------------------------
# Volume anomaly tests
# ---------------------------------------------------------------------------


class TestVolumeAnomaly:
    """Volume spike, acceleration, absorption, and drought detection."""

    def test_extreme_volume_spike_midday(self, detector, baselines):
        """Large volume spike at midday → VOLUME_SPIKE, severity HIGH."""
        # z=5.0, midday mult=1.5, adjusted_z=3.33 → HIGH
        bar = _bar(_ts(11, 45), volume=1_000_000 + 5.0 * 200_000)
        result = detector.detect_volume_anomaly("NIFTY50", bar, baselines)

        assert result is not None
        assert isinstance(result, VolumeAnomaly)
        assert result.anomaly_type == "VOLUME_SPIKE"
        assert result.severity == "HIGH"
        assert result.volume_zscore > 2.0

    def test_moderate_volume_spike(self, detector, baselines):
        """~2.5 z-score at midday → VOLUME_SPIKE MEDIUM."""
        # midday threshold mult is 1.5, so adjusted_z = 2.5/1.5 ≈ 1.67 ... not enough
        # We need adjusted_z > 2.0 → raw z > 2.0 * 1.5 = 3.0 at midday
        bar = _bar(_ts(11, 0), volume=1_000_000 + 3.1 * 200_000)
        result = detector.detect_volume_anomaly("NIFTY50", bar, baselines)

        assert result is not None
        assert result.anomaly_type == "VOLUME_SPIKE"
        assert result.severity == "MEDIUM"

    def test_volume_at_open_higher_threshold(self, detector, baselines):
        """Volume at 9:20 (OPEN context) uses 2x multiplier — moderate spike ignored."""
        # z-score = 3.0, but at OPEN threshold_mult = 2.0, adjusted = 1.5 → no alert
        bar = _bar(_ts(9, 20), volume=1_000_000 + 3.0 * 200_000)
        result = detector.detect_volume_anomaly("NIFTY50", bar, baselines)

        assert result is None

    def test_volume_at_open_extreme_still_detected(self, detector, baselines):
        """Very high volume at open should still be detected despite higher threshold."""
        # z = 5.0, adjusted = 5.0/2.0 = 2.5 → MEDIUM
        bar = _bar(_ts(9, 20), volume=1_000_000 + 5.0 * 200_000)
        result = detector.detect_volume_anomaly("NIFTY50", bar, baselines)

        assert result is not None
        assert result.time_context == "OPEN"

    def test_absorption_high_volume_tiny_move(self, detector, baselines):
        """High volume + tiny body → ABSORPTION."""
        # z > 2.5 and |close - open| < 0.3 * avg_body (0.3*80=24)
        bar = _bar(
            _ts(11, 30),
            open_=22000.0,
            close=22010.0,  # body = 10, < 24
            volume=1_000_000 + 3.0 * 200_000,
        )
        result = detector.detect_volume_anomaly("NIFTY50", bar, baselines)

        assert result is not None
        assert result.anomaly_type == "ABSORPTION"
        assert result.severity == "HIGH"
        assert "absorption" in result.message.lower()

    def test_volume_drought(self, detector, baselines):
        """Very low volume → VOLUME_DROUGHT LOW."""
        bar = _bar(_ts(13, 0), volume=1_000_000 - 1.6 * 200_000)
        result = detector.detect_volume_anomaly("NIFTY50", bar, baselines)

        assert result is not None
        assert result.anomaly_type == "VOLUME_DROUGHT"
        assert result.severity == "LOW"

    def test_normal_volume_no_anomaly(self, detector, baselines):
        """Average volume → no anomaly."""
        bar = _bar(_ts(11, 0), volume=1_050_000)
        result = detector.detect_volume_anomaly("NIFTY50", bar, baselines)

        assert result is None

    def test_sudden_acceleration(self, detector, baselines):
        """Current bar > 3x average of previous 3 bars → SUDDEN_ACCELERATION."""
        # Seed 3 low-volume bars
        for i in range(3):
            low_bar = _bar(_ts(10, 30 + i), volume=100_000)
            detector._cache_bar("NIFTY50", low_bar)

        # Current bar = 400_000 > 3 * 100_000
        bar = _bar(_ts(10, 33), volume=400_000)
        # Need z > 2.0/1.5 (midday) → that's z > 1.33, and z = (400k-1M)/200k = -3 ... negative
        # So acceleration fires but z-score check won't fire.
        # Acceleration check only overrides if severity is None or MEDIUM.
        # Since z is negative, anomaly_type is None initially, so acceleration sets it.
        # But wait — the volume is below avg so z < 0. The acceleration check
        # needs volume > 3 * prev_3_avg, which is satisfied (400k > 300k).
        # However severity would be None from z-score, so acceleration fires.
        result = detector.detect_volume_anomaly("NIFTY50", bar, baselines)

        assert result is not None
        assert result.anomaly_type == "SUDDEN_ACCELERATION"

    def test_zero_volume_skipped(self, detector, baselines):
        """Bar with zero volume → None."""
        bar = _bar(_ts(11, 0), volume=0)
        assert detector.detect_volume_anomaly("NIFTY50", bar, baselines) is None

    def test_nan_volume_skipped(self, detector, baselines):
        """Bar with NaN volume → None."""
        bar = _bar(_ts(11, 0), volume=float("nan"))
        assert detector.detect_volume_anomaly("NIFTY50", bar, baselines) is None


# ---------------------------------------------------------------------------
# Price anomaly tests
# ---------------------------------------------------------------------------


class TestPriceAnomaly:
    """Gap, intraday move, reversal, range expansion, compression."""

    def test_gap_up_detection(self, detector, baselines):
        """Today open > yesterday close by 1.5% → GAP_UP HIGH."""
        prev = _bar(_ts(15, 30, day=14), close=22000.0)
        # gap = (22330 - 22000) / 22000 = 1.5%
        current = _bar(
            _ts(9, 15), open_=22330.0, high=22400.0, low=22300.0, close=22350.0,
        )
        result = detector.detect_price_anomaly(
            "NIFTY50", current, [prev], baselines,
        )

        assert result is not None
        assert result.anomaly_type == "GAP_UP"
        assert result.severity == "HIGH"

    def test_gap_down_detection(self, detector, baselines):
        """Open below previous close by 0.6% → GAP_DOWN MEDIUM."""
        prev = _bar(_ts(15, 30, day=14), close=22000.0)
        # gap = (21868 - 22000) / 22000 = -0.6%
        current = _bar(
            _ts(9, 15), open_=21868.0, high=21900.0, low=21800.0, close=21870.0,
        )
        result = detector.detect_price_anomaly(
            "NIFTY50", current, [prev], baselines,
        )

        assert result is not None
        assert result.anomaly_type == "GAP_DOWN"
        assert result.severity == "MEDIUM"

    def test_extreme_gap_detected(self, detector, baselines):
        """Gap of 2.5% → HIGH severity."""
        prev = _bar(_ts(15, 30, day=14), close=22000.0)
        current = _bar(
            _ts(9, 15), open_=22550.0, high=22600.0, low=22500.0, close=22560.0,
        )
        result = detector.detect_price_anomaly(
            "NIFTY50", current, [prev], baselines,
        )

        assert result is not None
        assert result.anomaly_type == "GAP_UP"
        assert result.severity == "HIGH"

    def test_large_intraday_move(self, detector, baselines):
        """Move > 2x avg range within first 2 hours → LARGE_INTRADAY_MOVE."""
        # avg_range = 150, so need move > 300 points from open
        current = _bar(
            _ts(10, 30),
            open_=22000.0,
            high=22400.0,
            low=21990.0,
            close=22350.0,  # |close - open| = 350 > 2*150
        )
        result = detector.detect_price_anomaly(
            "NIFTY50", current, [], baselines,
        )

        assert result is not None
        assert result.anomaly_type == "LARGE_INTRADAY_MOVE"
        assert result.severity == "HIGH"

    def test_extreme_intraday_move(self, detector, baselines):
        """Move > 3x avg range → EXTREME_INTRADAY_MOVE."""
        current = _bar(
            _ts(10, 30),
            open_=22000.0,
            high=22500.0,
            low=21990.0,
            close=22480.0,  # |close - open| = 480 > 3*150
        )
        result = detector.detect_price_anomaly(
            "NIFTY50", current, [], baselines,
        )

        assert result is not None
        assert result.anomaly_type == "EXTREME_INTRADAY_MOVE"

    def test_intraday_bearish_reversal(self, detector, baselines):
        """Price up >1% then down >0.7% → INTRADAY_REVERSAL."""
        # Build bars: rallied then reversed
        day_open = 22000.0
        bar1 = _bar(_ts(9, 30), open_=day_open, high=22050.0, low=21990.0, close=22040.0)
        bar2 = _bar(_ts(10, 0), open_=22040.0, high=22280.0, low=22030.0, close=22270.0)
        # high at 22280 → up 1.27% from open
        bar3 = _bar(_ts(10, 30), open_=22270.0, high=22290.0, low=22090.0, close=22100.0)
        # down_from_high = (22280 - 22100)/22000 = 0.82%

        result = detector.detect_price_anomaly(
            "NIFTY50", bar3, [bar1, bar2], baselines,
        )

        assert result is not None
        assert result.anomaly_type == "INTRADAY_REVERSAL"
        assert result.details.get("direction") == "BEARISH"

    def test_intraday_bullish_reversal(self, detector, baselines):
        """Price down >1% then recovered >0.7% → INTRADAY_REVERSAL BULLISH."""
        day_open = 22000.0
        bar1 = _bar(_ts(9, 30), open_=day_open, high=22010.0, low=21770.0, close=21780.0)
        # low at 21770 → down 1.05% from open
        bar2 = _bar(_ts(10, 0), open_=21780.0, high=21950.0, low=21770.0, close=21940.0)
        # up_from_low = (21940 - 21770)/22000 = 0.77%

        result = detector.detect_price_anomaly(
            "NIFTY50", bar2, [bar1], baselines,
        )

        assert result is not None
        assert result.anomaly_type == "INTRADAY_REVERSAL"
        assert result.details.get("direction") == "BULLISH"

    def test_range_expansion(self, detector, baselines):
        """Today's range > 1.5x avg → RANGE_EXPANSION."""
        # avg_range = 150, need range > 225
        current = _bar(
            _ts(13, 0),
            open_=22000.0,
            high=22250.0,
            low=21990.0,
            close=22200.0,
        )
        # range = 260 > 1.5 * 150
        result = detector.detect_price_anomaly(
            "NIFTY50", current, [], baselines,
        )

        assert result is not None
        assert result.anomaly_type == "RANGE_EXPANSION"

    def test_compression(self, detector, baselines):
        """Tiny range by midday → COMPRESSION."""
        # avg_range = 150, need range < 0.3 * 150 = 45
        current = _bar(
            _ts(12, 30),
            open_=22000.0,
            high=22020.0,
            low=21990.0,
            close=22010.0,
        )
        # range = 30 < 45
        result = detector.detect_price_anomaly(
            "NIFTY50", current, [], baselines,
        )

        assert result is not None
        assert result.anomaly_type == "COMPRESSION"
        assert result.severity == "LOW"

    def test_no_price_anomaly(self, detector, baselines):
        """Normal bar → no anomaly."""
        current = _bar(
            _ts(11, 0),
            open_=22000.0,
            high=22080.0,
            low=21950.0,
            close=22060.0,
        )
        # range = 130, < 1.5 * 150. Move = 60, < 2 * 150. No gap (no prev bars).
        result = detector.detect_price_anomaly(
            "NIFTY50", current, [], baselines,
        )

        assert result is None


# ---------------------------------------------------------------------------
# Cooldown tests
# ---------------------------------------------------------------------------


class TestCooldown:
    """Alert deduplication via cooldown windows."""

    def test_same_anomaly_suppressed_within_window(self, detector, baselines):
        """Same anomaly type within 15 min → second suppressed."""
        bar1 = _bar(_ts(11, 0), volume=1_000_000 + 4.5 * 200_000)
        bar2 = _bar(_ts(11, 5), volume=1_000_000 + 4.5 * 200_000)

        events1 = detector.detect_all("NIFTY50", bar1, [], baselines)
        events2 = detector.detect_all("NIFTY50", bar2, [], baselines)

        assert len(events1) >= 1
        # Second call within 5 min — same cooldown key → suppressed
        vol_events2 = [e for e in events2 if e.category == "VOLUME"]
        assert len(vol_events2) == 0

    def test_anomaly_after_cooldown_expires(self, detector, baselines):
        """Same anomaly after cooldown expires → allowed."""
        bar1 = _bar(_ts(11, 0), volume=1_000_000 + 4.5 * 200_000)
        bar2 = _bar(
            datetime(2024, 6, 17, 11, 20, tzinfo=_IST),  # 20 min later
            volume=1_000_000 + 4.5 * 200_000,
        )

        events1 = detector.detect_all("NIFTY50", bar1, [], baselines)
        events2 = detector.detect_all("NIFTY50", bar2, [], baselines)

        vol_events1 = [e for e in events1 if e.category == "VOLUME"]
        vol_events2 = [e for e in events2 if e.category == "VOLUME"]
        assert len(vol_events1) >= 1
        # HIGH severity cooldown = 10 min, 20 min > 10 min → allowed
        assert len(vol_events2) >= 1

    def test_clear_expired_cooldowns(self, detector):
        """Expired cooldowns are cleared."""
        old = datetime(2024, 6, 17, 10, 0, tzinfo=_IST)
        detector._cooldowns["NIFTY50_VOLUME_SPIKE"] = old

        now = datetime(2024, 6, 17, 11, 0, tzinfo=_IST)  # 60 min later
        cleared = detector.clear_expired_cooldowns(now)

        assert cleared == 1
        assert "NIFTY50_VOLUME_SPIKE" not in detector._cooldowns

    def test_different_anomaly_types_independent(self, detector, baselines):
        """Different anomaly types have independent cooldowns."""
        # Volume spike bar
        vol_bar = _bar(_ts(11, 0), volume=1_000_000 + 4.5 * 200_000)

        # Price gap bar (with previous bar for gap calc)
        prev = _bar(_ts(15, 30, day=14), close=22000.0)
        gap_bar = _bar(
            _ts(11, 0), open_=22330.0, high=22400.0, low=22300.0, close=22350.0,
            volume=1_000_000 + 4.5 * 200_000,
        )

        events = detector.detect_all("NIFTY50", gap_bar, [prev], baselines)
        types = {e.anomaly_type for e in events}

        # Should have both volume and price anomalies
        assert len(events) >= 2
        assert "VOLUME" in {e.category for e in events}
        assert "PRICE" in {e.category for e in events}


# ---------------------------------------------------------------------------
# Baseline computation tests
# ---------------------------------------------------------------------------


class TestBaselines:
    """Baseline computation from historical data."""

    def test_baselines_from_ohlcv(self, detector):
        """Compute baselines from 30 days of data."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-06-01", periods=30, freq="B")
        close = 22000.0 + rng.standard_normal(30).cumsum() * 50
        high = close + rng.uniform(40, 120, 30)
        low = close - rng.uniform(40, 120, 30)
        open_ = low + rng.uniform(0, 1, 30) * (high - low)
        volume = rng.integers(800_000, 1_200_000, 30)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        bl = detector.update_baselines("NIFTY50", df)

        assert isinstance(bl, Baselines)
        assert bl.index_id == "NIFTY50"
        assert bl.avg_volume_20d > 0
        assert bl.std_volume_20d > 0
        assert bl.avg_range_20d > 0
        assert bl.avg_body_20d > 0
        # Cached
        assert detector.get_baselines("NIFTY50") is bl

    def test_baselines_short_data(self, detector):
        """Fewer than 20 rows → uses what's available."""
        dates = pd.date_range("2024-06-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "open": [100, 101, 102, 101, 103],
                "high": [105, 106, 107, 106, 108],
                "low": [98, 99, 100, 99, 101],
                "close": [103, 104, 105, 104, 106],
                "volume": [1000, 1100, 900, 1050, 1200],
            },
            index=dates,
        )

        bl = detector.update_baselines("TEST", df)
        assert bl.avg_volume_20d > 0
        assert bl.avg_range_20d > 0


# ---------------------------------------------------------------------------
# Graceful degradation tests
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Edge cases: missing baselines, NaN data, zero std."""

    def test_no_volume_stats_skips_check(self, detector):
        """Baselines with zero std → volume check returns None with warning."""
        bl = Baselines(
            index_id="BSE_IDX",
            computed_at=datetime.now(tz=_IST),
            avg_volume_20d=0,
            std_volume_20d=0,
        )
        bar = _bar(_ts(11, 0), volume=5_000_000)
        result = detector.detect_volume_anomaly("BSE_IDX", bar, bl)
        assert result is None

    def test_no_range_stats_skips_price_check(self, detector):
        """Baselines with zero range → price check returns None."""
        bl = Baselines(
            index_id="BSE_IDX",
            computed_at=datetime.now(tz=_IST),
            avg_range_20d=0,
        )
        bar = _bar(_ts(11, 0))
        result = detector.detect_price_anomaly("BSE_IDX", bar, [], bl)
        assert result is None

    def test_nan_in_bar_handled(self, detector, baselines):
        """NaN open → returns None gracefully."""
        bar = _bar(_ts(11, 0), open_=float("nan"))
        result = detector.detect_volume_anomaly("NIFTY50", bar, baselines)
        assert result is None

    def test_string_timestamp_parsed(self, detector, baselines):
        """Timestamp passed as ISO string → parsed correctly."""
        bar = {
            "timestamp": "2024-06-17T11:00:00+05:30",
            "open": 22000.0,
            "high": 22100.0,
            "low": 21900.0,
            "close": 22050.0,
            "volume": 1_000_000 + 4.5 * 200_000,
        }
        result = detector.detect_volume_anomaly("NIFTY50", bar, baselines)
        assert result is not None


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    """Saving anomalies to the database."""

    def test_save_anomalies(self, detector, baselines):
        """Events are inserted into anomaly_events table."""
        bar = _bar(_ts(11, 0), volume=1_000_000 + 4.5 * 200_000)
        events = detector.detect_all("NIFTY50", bar, [], baselines)
        assert len(events) >= 1

        inserted = detector.save_anomalies(events)
        assert inserted >= 1

        rows = detector.db.fetch_all(
            "SELECT * FROM anomaly_events WHERE index_id = ?", ("NIFTY50",),
        )
        assert len(rows) >= 1
        assert rows[0]["category"] in ("VOLUME", "PRICE")

    def test_db_level_cooldown_dedup(self, detector, baselines):
        """Saving same event twice within cooldown → second skipped at DB level."""
        bar = _bar(_ts(11, 0), volume=1_000_000 + 4.5 * 200_000)
        events = detector.detect_all("NIFTY50", bar, [], baselines)

        inserted1 = detector.save_anomalies(events)
        inserted2 = detector.save_anomalies(events)

        assert inserted1 >= 1
        assert inserted2 == 0


# ---------------------------------------------------------------------------
# detect_all integration
# ---------------------------------------------------------------------------


class TestDetectAll:
    """End-to-end detect_all producing AnomalyEvent list."""

    def test_both_volume_and_price_detected(self, detector, baselines):
        """A bar with volume spike AND gap → both categories in results."""
        prev = _bar(_ts(15, 30, day=14), close=22000.0)
        current = _bar(
            _ts(11, 0),
            open_=22330.0,
            high=22500.0,
            low=22300.0,
            close=22450.0,
            volume=1_000_000 + 4.5 * 200_000,
        )

        events = detector.detect_all("NIFTY50", current, [prev], baselines)

        categories = {e.category for e in events}
        assert "VOLUME" in categories
        assert "PRICE" in categories

        for ev in events:
            assert isinstance(ev, AnomalyEvent)
            assert ev.is_active is True
            assert ev.cooldown_key.startswith("NIFTY50_")

    def test_no_anomalies_returns_empty(self, detector, baselines):
        """Normal bar → empty list."""
        bar = _bar(
            _ts(11, 0),
            open_=22000.0,
            high=22080.0,
            low=21950.0,
            close=22060.0,
            volume=1_050_000,
        )
        events = detector.detect_all("NIFTY50", bar, [], baselines)
        assert events == []
