"""
Unit tests for VolumeProfileAnalyzer.

All tests use __new__ to bypass __init__ and avoid real DB/CandleStore deps.
Candle fixtures use prices that land on bucket boundaries (multiples of 10)
so volume distribution is deterministic.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, call, patch
from zoneinfo import ZoneInfo

import pytest

from src.analysis.liquidity.models import Candle, VolumeProfile, VolumeProfileLevel
from src.analysis.liquidity.volume_profile import VolumeProfileAnalyzer

_IST = ZoneInfo("Asia/Kolkata")
_DATE = date(2026, 5, 11)
_BASE_TS = datetime(2026, 5, 11, 9, 15, tzinfo=_IST)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candle(
    low: float,
    high: float,
    volume: int = 1000,
    symbol: str = "NIFTY",
    offset: int = 0,
) -> Candle:
    mid = (low + high) / 2
    return Candle(
        symbol=symbol,
        timeframe="1min",
        timestamp=_BASE_TS + timedelta(minutes=offset),
        open=mid,
        high=high,
        low=low,
        close=mid,
        volume=volume,
    )


def _make_analyzer(
    bucket_size: int = 10,
    value_area_pct: float = 0.70,
    hvn_threshold_pct: float = 1.5,
    lvn_threshold_pct: float = 0.4,
    min_candles: int = 5,
) -> VolumeProfileAnalyzer:
    """Return a VolumeProfileAnalyzer bypassing __init__ (no DB/CandleStore)."""
    analyzer = VolumeProfileAnalyzer.__new__(VolumeProfileAnalyzer)
    analyzer._tick_size = 0.05
    analyzer._bucket_size = bucket_size
    analyzer._value_area_pct = value_area_pct
    analyzer._hvn_threshold_pct = hvn_threshold_pct
    analyzer._lvn_threshold_pct = lvn_threshold_pct
    analyzer._min_candles = min_candles
    analyzer._daily_profiles = {}
    analyzer._previous_profiles = {}
    analyzer._running_candles = {}
    analyzer._last_candle_date = {}
    analyzer._candle_count = {}
    analyzer._db = MagicMock()
    analyzer._cs = MagicMock()
    return analyzer


def _simple_profile(
    poc: float = 100.0,
    vah: float = 120.0,
    val: float = 90.0,
    hvns: list[float] | None = None,
    lvns: list[float] | None = None,
    total_volume: int = 10000,
) -> VolumeProfile:
    """Build a minimal VolumeProfile for signal-method tests."""
    return VolumeProfile(
        symbol="NIFTY",
        date=_DATE,
        timeframe="1min",
        poc=poc,
        vah=vah,
        val=val,
        total_volume=total_volume,
        levels=[],
        hvns=sorted(hvns or []),
        lvns=sorted(lvns or []),
        calculated_at=datetime.now(_IST),
    )


# ---------------------------------------------------------------------------
# Profile calculation — bucket distribution
# ---------------------------------------------------------------------------

class TestProfileCalculation:
    """Verify POC, total_volume, and level assignment with known candle data."""

    def _candles_for_poc_at_100(self) -> list[Candle]:
        """
        Design:
          bucket 100 receives 10 × 1000 = 10000 vol  (POC)
          bucket 110 receives  5 × 1000 =  5000 vol
          bucket 120 receives  2 × 1000 =  2000 vol

        All candles are doji-style (low == high == exact bucket price) so
        volume assignment is unambiguous.
        """
        candles = []
        for i in range(10):
            candles.append(_candle(100, 100, volume=1000, offset=i))
        for i in range(5):
            candles.append(_candle(110, 110, volume=1000, offset=10 + i))
        for i in range(2):
            candles.append(_candle(120, 120, volume=1000, offset=15 + i))
        return candles

    def test_poc_at_highest_volume_bucket(self):
        az = _make_analyzer()
        profile = az.calculate_profile("NIFTY", self._candles_for_poc_at_100(), _DATE)
        assert profile.poc == pytest.approx(100.0)

    def test_total_volume_equals_sum_of_candle_volumes(self):
        az = _make_analyzer()
        candles = self._candles_for_poc_at_100()
        profile = az.calculate_profile("NIFTY", candles, _DATE)
        expected = sum(c.volume for c in candles)
        assert profile.total_volume == expected

    def test_levels_include_all_buckets(self):
        az = _make_analyzer()
        profile = az.calculate_profile("NIFTY", self._candles_for_poc_at_100(), _DATE)
        prices = {lvl.price for lvl in profile.levels}
        assert 100.0 in prices
        assert 110.0 in prices
        assert 120.0 in prices

    def test_poc_level_tagged_correctly(self):
        az = _make_analyzer()
        profile = az.calculate_profile("NIFTY", self._candles_for_poc_at_100(), _DATE)
        poc_levels = [lvl for lvl in profile.levels if lvl.level_type == "POC"]
        assert len(poc_levels) == 1
        assert poc_levels[0].price == pytest.approx(100.0)

    def test_profile_symbol_and_date_set(self):
        az = _make_analyzer()
        profile = az.calculate_profile("BANKNIFTY", self._candles_for_poc_at_100(), _DATE)
        assert profile.symbol == "BANKNIFTY"
        assert profile.date == _DATE

    def test_empty_candles_raises(self):
        az = _make_analyzer()
        with pytest.raises(ValueError):
            az.calculate_profile("NIFTY", [], _DATE)

    def test_single_candle_returns_profile(self):
        az = _make_analyzer()
        candles = [_candle(100, 100, volume=500)]
        profile = az.calculate_profile("NIFTY", candles, _DATE)
        assert profile.poc == pytest.approx(100.0)
        assert profile.total_volume == 500


# ---------------------------------------------------------------------------
# Value area calculation
# ---------------------------------------------------------------------------

class TestValueArea:
    """VAH/VAL must capture >= 70% of total volume centred on POC."""

    def test_value_area_vah_above_poc(self):
        az = _make_analyzer(value_area_pct=0.70)
        # Symmetric distribution: POC at 100, equal volume above and below
        candles = (
            [_candle(100, 100, volume=5000)]
            + [_candle(110, 110, volume=2000)]
            + [_candle(90, 90, volume=2000)]
        )
        profile = az.calculate_profile("NIFTY", candles, _DATE)
        assert profile.vah >= profile.poc
        assert profile.val <= profile.poc

    def test_value_area_covers_70_pct_of_volume(self):
        az = _make_analyzer(value_area_pct=0.70)
        candles = (
            [_candle(100, 100, volume=5000)]
            + [_candle(110, 110, volume=2000)]
            + [_candle(90, 90, volume=2000)]
        )
        profile = az.calculate_profile("NIFTY", candles, _DATE)
        va_levels = [lvl for lvl in profile.levels if lvl.is_value_area]
        va_volume = sum(lvl.volume for lvl in va_levels)
        assert va_volume >= profile.total_volume * 0.70

    def test_value_area_asymmetric_skews_upward(self):
        az = _make_analyzer(value_area_pct=0.70)
        # POC at 100, more volume above → VAH will extend further
        candles = (
            [_candle(100, 100, volume=5000)]   # poc
            + [_candle(110, 110, volume=3000)]  # above
            + [_candle(120, 120, volume=2000)]  # above
            + [_candle(90, 90, volume=500)]    # below (tiny)
        )
        profile = az.calculate_profile("NIFTY", candles, _DATE)
        # VAH must capture at least bucket 110 (since 5000+3000=8000 > 70% of 10500)
        assert profile.vah >= 110.0

    def test_vah_tagged_in_levels(self):
        az = _make_analyzer()
        candles = (
            [_candle(100, 100, volume=5000)]
            + [_candle(110, 110, volume=3000)]
            + [_candle(90, 90, volume=1000)]
        )
        profile = az.calculate_profile("NIFTY", candles, _DATE)
        vah_levels = [lvl for lvl in profile.levels if lvl.level_type == "VAH"]
        assert any(lvl.price == profile.vah for lvl in vah_levels)

    def test_val_tagged_in_levels(self):
        az = _make_analyzer()
        candles = (
            [_candle(100, 100, volume=5000)]
            + [_candle(110, 110, volume=1000)]
            + [_candle(90, 90, volume=3000)]
        )
        profile = az.calculate_profile("NIFTY", candles, _DATE)
        val_levels = [lvl for lvl in profile.levels if lvl.level_type == "VAL"]
        assert any(lvl.price == profile.val for lvl in val_levels)


# ---------------------------------------------------------------------------
# HVN / LVN identification
# ---------------------------------------------------------------------------

class TestHvnLvn:
    """HVN = volume > 1.5x avg; LVN = volume < 0.4x avg. Top 5 each."""

    def _profiled_candles(self) -> tuple[VolumeProfileAnalyzer, VolumeProfile]:
        az = _make_analyzer()
        # bucket 100: 10000 (HVN — avg = 17000/3 ≈ 5667, threshold = 8500)
        # bucket 110:  5000 (normal)
        # bucket 120:  2000 (LVN — threshold = 2267)
        candles = (
            [_candle(100, 100, volume=1000)] * 10   # → 10000
            + [_candle(110, 110, volume=1000)] * 5  # → 5000
            + [_candle(120, 120, volume=1000)] * 2  # → 2000
        )
        profile = az.calculate_profile("NIFTY", candles, _DATE)
        return az, profile

    def test_hvn_at_highest_volume_bucket(self):
        _, profile = self._profiled_candles()
        assert 100.0 in profile.hvns

    def test_lvn_at_lowest_volume_bucket(self):
        _, profile = self._profiled_candles()
        assert 120.0 in profile.lvns

    def test_middle_bucket_not_hvn_or_lvn(self):
        _, profile = self._profiled_candles()
        assert 110.0 not in profile.hvns
        assert 110.0 not in profile.lvns

    def test_hvn_tagged_in_levels(self):
        _, profile = self._profiled_candles()
        # Bucket 100 is POC AND HVN; level_type is "POC" (higher priority).
        # The hvns list must contain it regardless.
        assert 100.0 in profile.hvns

    def test_lvn_tagged_in_levels(self):
        _, profile = self._profiled_candles()
        lvn_prices = {lvl.price for lvl in profile.levels if lvl.level_type == "LVN"}
        assert 120.0 in lvn_prices

    def test_at_most_5_hvns(self):
        az = _make_analyzer()
        # Create 10 high-volume buckets
        candles = [_candle(float(100 + i * 10), float(100 + i * 10), volume=5000) for i in range(10)]
        profile = az.calculate_profile("NIFTY", candles, _DATE)
        assert len(profile.hvns) <= 5

    def test_at_most_5_lvns(self):
        az = _make_analyzer()
        # One big bucket (POC) and many tiny ones
        candles = [_candle(100, 100, volume=50000)]
        candles += [_candle(float(110 + i * 10), float(110 + i * 10), volume=1) for i in range(10)]
        profile = az.calculate_profile("NIFTY", candles, _DATE)
        assert len(profile.lvns) <= 5


# ---------------------------------------------------------------------------
# Price context — all location types
# ---------------------------------------------------------------------------

class TestPriceContext:

    def _az_with_profile(
        self,
        poc: float = 100.0,
        vah: float = 120.0,
        val: float = 80.0,
        hvns: list[float] | None = None,
        lvns: list[float] | None = None,
    ) -> VolumeProfileAnalyzer:
        az = _make_analyzer()
        az._daily_profiles["NIFTY"] = _simple_profile(
            poc=poc, vah=vah, val=val,
            hvns=hvns or [], lvns=lvns or [],
        )
        return az

    # AT_POC
    def test_at_poc_within_tolerance(self):
        az = self._az_with_profile()
        ctx = az.get_price_context("NIFTY", 100.0)
        assert ctx["location"] == "AT_POC"
        assert ctx["signal"] == "NEUTRAL"
        assert ctx["strength"] == pytest.approx(30.0)

    def test_at_poc_uses_01_pct_tolerance(self):
        az = self._az_with_profile()
        # 0.1% of 100 = 0.1; price 100.09 should still be AT_POC
        ctx = az.get_price_context("NIFTY", 100.09)
        assert ctx["location"] == "AT_POC"

    # ABOVE_VAH
    def test_above_vah_returns_bearish(self):
        az = self._az_with_profile()
        ctx = az.get_price_context("NIFTY", 125.0)
        assert ctx["location"] == "ABOVE_VAH"
        assert ctx["signal"] == "BEARISH"
        assert ctx["strength"] == pytest.approx(55.0)

    def test_above_vah_in_lvn_returns_bullish_breakout(self):
        az = self._az_with_profile(vah=120.0, lvns=[125.0])
        # price 125 is within bucket_half (5) of LVN at 125
        ctx = az.get_price_context("NIFTY", 125.0)
        assert ctx["location"] == "IN_LVN"
        assert ctx["signal"] == "BULLISH"
        assert ctx["strength"] == pytest.approx(70.0)

    # BELOW_VAL
    def test_below_val_returns_bullish(self):
        az = self._az_with_profile()
        ctx = az.get_price_context("NIFTY", 75.0)
        assert ctx["location"] == "BELOW_VAL"
        assert ctx["signal"] == "BULLISH"
        assert ctx["strength"] == pytest.approx(55.0)

    # AT_HVN (within value area)
    def test_at_hvn_returns_neutral(self):
        az = self._az_with_profile(hvns=[110.0])
        # bucket_half = 5; price 110 is exactly on HVN
        ctx = az.get_price_context("NIFTY", 110.0)
        assert ctx["location"] == "AT_HVN"
        assert ctx["signal"] == "NEUTRAL"
        assert ctx["strength"] == pytest.approx(40.0)

    # IN_LVN (within value area)
    def test_in_lvn_above_poc_returns_bullish(self):
        az = self._az_with_profile(poc=100.0, lvns=[110.0])
        ctx = az.get_price_context("NIFTY", 110.0)
        assert ctx["location"] == "IN_LVN"
        assert ctx["signal"] == "BULLISH"   # price(110) > poc(100)
        assert ctx["strength"] == pytest.approx(65.0)

    def test_in_lvn_below_poc_returns_bearish(self):
        az = self._az_with_profile(poc=100.0, lvns=[90.0])
        ctx = az.get_price_context("NIFTY", 90.0)
        assert ctx["location"] == "IN_LVN"
        assert ctx["signal"] == "BEARISH"   # price(90) < poc(100)

    # IN_VALUE_AREA (no HVN/LVN nearby)
    def test_in_value_area_returns_neutral(self):
        az = self._az_with_profile()
        ctx = az.get_price_context("NIFTY", 105.0)
        assert ctx["location"] == "IN_VALUE_AREA"
        assert ctx["signal"] == "NEUTRAL"

    # No profile
    def test_no_profile_returns_unknown(self):
        az = _make_analyzer()
        ctx = az.get_price_context("UNKNOWN", 100.0)
        assert ctx["location"] == "UNKNOWN"
        assert ctx["strength"] == pytest.approx(0.0)

    # Keys present
    def test_context_contains_all_required_keys(self):
        az = self._az_with_profile()
        ctx = az.get_price_context("NIFTY", 100.0)
        required = {
            "location", "poc", "vah", "val",
            "nearest_hvn_above", "nearest_hvn_below",
            "nearest_lvn_above", "nearest_lvn_below",
            "signal", "strength", "reason",
        }
        assert required.issubset(ctx.keys())


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------

class TestVpTargets:

    def _az_with_profile(self, **kw) -> VolumeProfileAnalyzer:
        az = _make_analyzer()
        az._daily_profiles["NIFTY"] = _simple_profile(**kw)
        return az

    def test_bullish_returns_hvn_above_and_vah(self):
        # Prices within 2% of 22000: 22000*0.02 = 440 pts
        az = self._az_with_profile(poc=22000.0, vah=22300.0, val=21700.0, hvns=[22150.0])
        targets = az.get_vp_targets("NIFTY", 22000.0, "BULLISH")
        assert 22150.0 in targets   # HVN 0.68% above → within 2%
        assert 22300.0 in targets   # VAH 1.36% above → within 2%

    def test_bearish_returns_hvn_below_and_val(self):
        # Prices within 2% of 22000
        az = self._az_with_profile(poc=22000.0, vah=22300.0, val=21700.0, hvns=[21850.0])
        targets = az.get_vp_targets("NIFTY", 22000.0, "BEARISH")
        assert 21850.0 in targets   # HVN 0.68% below → within 2%
        assert 21700.0 in targets   # VAL 1.36% below → within 2%

    def test_targets_filtered_to_within_2_pct(self):
        # 2% of 22000 = 440; VAH at 23000 (+4.5%) should be filtered out
        az = self._az_with_profile(poc=22000.0, vah=23000.0, val=21000.0, hvns=[22100.0, 22800.0])
        targets = az.get_vp_targets("NIFTY", 22000.0, "BULLISH")
        # 23000 (+4.5%) and 22800 (+3.6%) are beyond 2%; only 22100 (0.45%) passes
        for t in targets:
            assert abs(t - 22000.0) <= 22000.0 * 0.02

    def test_no_profile_returns_empty(self):
        az = _make_analyzer()
        assert az.get_vp_targets("NIFTY", 100.0, "BULLISH") == []

    def test_bullish_includes_prev_day_vah_if_in_range(self):
        az = _make_analyzer()
        az._daily_profiles["NIFTY"] = _simple_profile(poc=100.0, vah=105.0, val=95.0)
        az._previous_profiles["NIFTY"] = _simple_profile(poc=99.0, vah=101.0, val=97.0)
        targets = az.get_vp_targets("NIFTY", 100.0, "BULLISH")
        assert 101.0 in targets

    def test_bearish_includes_prev_day_val_if_in_range(self):
        az = _make_analyzer()
        az._daily_profiles["NIFTY"] = _simple_profile(poc=100.0, vah=105.0, val=95.0)
        az._previous_profiles["NIFTY"] = _simple_profile(poc=99.0, vah=101.0, val=99.0)
        targets = az.get_vp_targets("NIFTY", 100.0, "BEARISH")
        assert 99.0 in targets


# ---------------------------------------------------------------------------
# is_price_in_acceptance
# ---------------------------------------------------------------------------

class TestAcceptance:

    def test_price_near_poc_in_va_is_acceptance(self):
        az = _make_analyzer()
        az._daily_profiles["NIFTY"] = _simple_profile(poc=100.0, vah=120.0, val=80.0)
        assert az.is_price_in_acceptance("NIFTY", 100.1) is True

    def test_price_in_va_but_far_from_poc_not_acceptance(self):
        az = _make_analyzer()
        az._daily_profiles["NIFTY"] = _simple_profile(poc=100.0, vah=120.0, val=80.0)
        assert az.is_price_in_acceptance("NIFTY", 115.0) is False

    def test_price_outside_va_not_acceptance(self):
        az = _make_analyzer()
        az._daily_profiles["NIFTY"] = _simple_profile(poc=100.0, vah=120.0, val=80.0)
        assert az.is_price_in_acceptance("NIFTY", 125.0) is False

    def test_no_profile_returns_false(self):
        az = _make_analyzer()
        assert az.is_price_in_acceptance("NIFTY", 100.0) is False


# ---------------------------------------------------------------------------
# Previous-day profile loading from DB
# ---------------------------------------------------------------------------

class TestPreviousDayProfile:

    def _az_with_db(
        self,
        profile_row: dict | None,
        level_rows: list[dict],
    ) -> VolumeProfileAnalyzer:
        az = _make_analyzer()
        az._db = MagicMock()
        az._db.fetch_one.return_value = profile_row
        az._db.fetch_all.return_value = level_rows
        return az

    def test_returns_none_when_no_db_row(self):
        az = self._az_with_db(None, [])
        result = az.load_previous_day_profile("NIFTY")
        assert result is None

    def test_loads_profile_from_db(self):
        row = {
            "id": 1,
            "symbol": "NIFTY",
            "date": "2026-05-10",
            "poc": 22500.0,
            "vah": 22600.0,
            "val": 22400.0,
            "total_volume": 1_000_000,
            "calculated_at": "2026-05-10T15:30:00",
        }
        level_rows = [
            {"price": 22400.0, "volume": 100000, "volume_pct": 10.0, "level_type": "VAL", "is_value_area": 1},
            {"price": 22500.0, "volume": 400000, "volume_pct": 40.0, "level_type": "POC", "is_value_area": 1},
            {"price": 22600.0, "volume": 200000, "volume_pct": 20.0, "level_type": "VAH", "is_value_area": 1},
            {"price": 22550.0, "volume": 250000, "volume_pct": 25.0, "level_type": "HVN", "is_value_area": 1},
            {"price": 22450.0, "volume":  50000, "volume_pct":  5.0, "level_type": "LVN", "is_value_area": 1},
        ]
        az = self._az_with_db(row, level_rows)
        profile = az.load_previous_day_profile("NIFTY")
        assert profile is not None
        assert profile.poc == pytest.approx(22500.0)
        assert profile.vah == pytest.approx(22600.0)
        assert profile.val == pytest.approx(22400.0)
        assert 22550.0 in profile.hvns
        assert 22450.0 in profile.lvns

    def test_profile_cached_in_previous_profiles(self):
        row = {
            "id": 1,
            "symbol": "NIFTY",
            "date": "2026-05-10",
            "poc": 22500.0,
            "vah": 22600.0,
            "val": 22400.0,
            "total_volume": 500000,
            "calculated_at": "2026-05-10T15:30:00",
        }
        az = self._az_with_db(row, [])
        az.load_previous_day_profile("NIFTY")
        assert "NIFTY" in az._previous_profiles

    def test_get_previous_day_levels_after_load(self):
        row = {
            "id": 1,
            "symbol": "NIFTY",
            "date": "2026-05-10",
            "poc": 22500.0,
            "vah": 22650.0,
            "val": 22350.0,
            "total_volume": 500000,
            "calculated_at": "2026-05-10T15:30:00",
        }
        az = self._az_with_db(row, [])
        az.load_previous_day_profile("NIFTY")
        levels = az.get_previous_day_levels("NIFTY")
        assert levels["prev_poc"] == pytest.approx(22500.0)
        assert levels["prev_vah"] == pytest.approx(22650.0)
        assert levels["prev_val"] == pytest.approx(22350.0)

    def test_get_previous_day_levels_no_profile_returns_zeros(self):
        az = _make_analyzer()
        levels = az.get_previous_day_levels("NIFTY")
        assert levels == {"prev_poc": 0.0, "prev_vah": 0.0, "prev_val": 0.0}


# ---------------------------------------------------------------------------
# Callback — incremental profile update
# ---------------------------------------------------------------------------

class TestCallback:

    def test_callback_resets_on_new_day(self):
        az = _make_analyzer(min_candles=5)
        day1_ts = datetime(2026, 5, 10, 9, 15, tzinfo=_IST)
        day2_ts = datetime(2026, 5, 11, 9, 15, tzinfo=_IST)

        c1 = _candle(100, 100)
        c1_day1 = Candle(
            symbol="NIFTY", timeframe="1min", timestamp=day1_ts,
            open=100, high=100, low=100, close=100, volume=1000,
        )
        az._on_1min_candle(c1_day1)
        assert az._last_candle_date.get("NIFTY") == day1_ts.date()

        c2_day2 = Candle(
            symbol="NIFTY", timeframe="1min", timestamp=day2_ts,
            open=110, high=110, low=110, close=110, volume=1000,
        )
        az._on_1min_candle(c2_day2)
        # Running candles should have been reset for the new day
        assert az._running_candles["NIFTY"] == [c2_day2]
        assert az._candle_count["NIFTY"] == 1

    def test_callback_accumulates_candles(self):
        az = _make_analyzer(min_candles=30)
        for i in range(7):
            az._on_1min_candle(_candle(100, 100, offset=i))
        assert len(az._running_candles["NIFTY"]) == 7

    def test_profile_not_calculated_below_min_candles(self):
        az = _make_analyzer(min_candles=30)
        for i in range(5):
            az._on_1min_candle(_candle(100, 100, offset=i))
        # 5 candles → batch triggered (5 % 5 == 0) but below min_candles=30
        assert "NIFTY" not in az._daily_profiles

    def test_profile_calculated_at_5_candle_batch_when_above_min(self):
        az = _make_analyzer(min_candles=5)
        for i in range(5):
            az._on_1min_candle(_candle(100, 100, offset=i))
        assert "NIFTY" in az._daily_profiles

    def test_profile_not_recalculated_on_non_batch_candles(self):
        az = _make_analyzer(min_candles=5)
        # Trigger first batch to create profile
        for i in range(5):
            az._on_1min_candle(_candle(100, 100, offset=i))
        first_profile = az._daily_profiles.get("NIFTY")

        # Send 3 more candles (not a multiple of 5) — profile should not change
        for i in range(3):
            az._on_1min_candle(_candle(200, 200, offset=5 + i))
        assert az._daily_profiles["NIFTY"] is first_profile
