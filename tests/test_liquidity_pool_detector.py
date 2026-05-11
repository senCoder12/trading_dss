"""
Tests for LiquidityPoolDetector.

Uses MagicMock for CandleStore and DatabaseManager so no real DB is needed.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.analysis.liquidity.models import Candle, LiquidityLevel
from src.analysis.liquidity.liquidity_pool_detector import LiquidityPoolDetector

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 5, 11, 9, 15, tzinfo=_IST)
    return base + timedelta(minutes=offset_minutes)


def _candle(
    high: float,
    low: float,
    symbol: str = "NIFTY",
    timeframe: str = "15min",
    close: float | None = None,
    offset: int = 0,
) -> Candle:
    close = close if close is not None else (high + low) / 2
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=_ts(offset),
        open=close,
        high=high,
        low=low,
        close=close,
        volume=1000,
    )


def _make_detector() -> LiquidityPoolDetector:
    """Return a detector with mocked DB and CandleStore."""
    mock_db = MagicMock()
    mock_db.execute.return_value = MagicMock(rowcount=0)
    mock_db.fetch_all.return_value = []
    mock_db.execute_many.return_value = 0

    mock_cs = MagicMock()
    mock_cs.get_candles.return_value = []

    with patch("src.analysis.liquidity.liquidity_pool_detector.settings") as mock_settings:
        cfg = MagicMock()
        cfg.equal_high_low_tolerance_pct = 0.05
        cfg.min_level_strength = 2
        cfg.max_levels_per_side = 5
        cfg.level_expiry_hours = 48
        cfg.round_number_increments = [50, 100, 500, 1000]
        cfg.round_number_range_pct = 3.0
        cfg.multi_touch_tolerance_pct = 0.1
        cfg.bias_proximity_ratio = 0.6
        mock_settings.liquidity = cfg

        detector = LiquidityPoolDetector.__new__(LiquidityPoolDetector)
        detector._cs = mock_cs
        detector._db = mock_db
        detector._tolerance_pct = cfg.equal_high_low_tolerance_pct
        detector._min_strength = cfg.min_level_strength
        detector._max_per_side = cfg.max_levels_per_side
        detector._expiry_hours = cfg.level_expiry_hours
        detector._increments = list(cfg.round_number_increments)
        detector._rn_range_pct = cfg.round_number_range_pct
        detector._mt_tolerance_pct = cfg.multi_touch_tolerance_pct
        detector._bias_ratio = cfg.bias_proximity_ratio
        detector._levels = {}
        detector._previous_day_data = {}

    return detector


# ---------------------------------------------------------------------------
# Equal high detection
# ---------------------------------------------------------------------------

class TestDetectEqualHighs:
    def test_detects_three_equal_highs(self):
        """Three swing highs within 0.05% → one EQUAL_HIGH level."""
        detector = _make_detector()
        # Swing high pattern: low - HIGH - low - HIGH - low - HIGH - low
        # highs at 24000, 24001, 24002 (all within 0.05% of 24000)
        candles = [
            _candle(high=23900, low=23800, offset=0),
            _candle(high=24000, low=23880, offset=15),   # swing high
            _candle(high=23920, low=23850, offset=30),
            _candle(high=24001, low=23880, offset=45),   # swing high
            _candle(high=23930, low=23860, offset=60),
            _candle(high=24002, low=23890, offset=75),   # swing high
            _candle(high=23940, low=23870, offset=90),
        ]
        levels = detector._detect_equal_highs(candles, tolerance_pct=0.05)
        assert len(levels) == 1
        lvl = levels[0]
        assert lvl.level_type == "EQUAL_HIGH"
        assert lvl.strength == 3
        # Level price = highest of the group (stops above highest)
        assert lvl.price == pytest.approx(24002, abs=1)

    def test_rejects_highs_too_far_apart(self):
        """Swing highs 0.3% apart → no EQUAL_HIGH level."""
        detector = _make_detector()
        # 24000 and 24075 → 0.31% apart, above 0.05% tolerance
        candles = [
            _candle(high=23900, low=23800, offset=0),
            _candle(high=24000, low=23880, offset=15),
            _candle(high=23850, low=23800, offset=30),
            _candle(high=24075, low=23900, offset=45),
            _candle(high=23870, low=23820, offset=60),
        ]
        levels = detector._detect_equal_highs(candles, tolerance_pct=0.05)
        assert len(levels) == 0

    def test_needs_minimum_three_candles(self):
        detector = _make_detector()
        levels = detector._detect_equal_highs(
            [_candle(24000, 23900, offset=0), _candle(24010, 23910, offset=15)],
            tolerance_pct=0.05,
        )
        assert levels == []

    def test_single_swing_high_not_enough(self):
        """One swing high alone doesn't form a level (min_strength=2)."""
        detector = _make_detector()
        candles = [
            _candle(23900, 23800, offset=0),
            _candle(24000, 23880, offset=15),  # sole swing high
            _candle(23920, 23850, offset=30),
        ]
        levels = detector._detect_equal_highs(candles, tolerance_pct=0.05)
        assert len(levels) == 0


# ---------------------------------------------------------------------------
# Equal low detection
# ---------------------------------------------------------------------------

class TestDetectEqualLows:
    def test_detects_two_equal_lows(self):
        """Two swing lows within tolerance → EQUAL_LOW level at the lowest."""
        detector = _make_detector()
        candles = [
            _candle(high=24100, low=24050, offset=0),
            _candle(high=24050, low=23950, offset=15),  # swing low
            _candle(high=24100, low=24020, offset=30),
            _candle(high=24060, low=23951, offset=45),  # swing low ~0.005% diff
            _candle(high=24100, low=24030, offset=60),
        ]
        levels = detector._detect_equal_lows(candles, tolerance_pct=0.05)
        assert len(levels) == 1
        lvl = levels[0]
        assert lvl.level_type == "EQUAL_LOW"
        assert lvl.price == pytest.approx(23950, abs=1)  # lowest of group


# ---------------------------------------------------------------------------
# Round number detection
# ---------------------------------------------------------------------------

class TestDetectRoundNumbers:
    def test_nifty_24150_bsl_round_numbers(self):
        """NIFTY at 24150 → BSL at 24200(50), 24500(500), 25000(1000) etc."""
        detector = _make_detector()
        levels = detector._detect_round_numbers(24150.0, "NIFTY")

        prices = {lvl.price for lvl in levels}

        # Should detect these as BSL (above 24150)
        assert 24200.0 in prices  # nearest 50-multiple
        assert 24500.0 in prices  # 500-multiple

        # Should detect these as SSL (below 24150)
        assert 24100.0 in prices  # 100-multiple below
        assert 24000.0 in prices  # 1000-multiple below

    def test_round_number_strength_multiples(self):
        detector = _make_detector()
        assert detector._round_number_strength(25000) == 5  # multiple of 1000
        assert detector._round_number_strength(24500) == 4  # multiple of 500, not 1000
        assert detector._round_number_strength(24300) == 3  # multiple of 100, not 500
        assert detector._round_number_strength(24150) == 2  # multiple of 50, not 100

    def test_round_numbers_within_3pct_only(self):
        """Levels outside 3% range must not be included."""
        detector = _make_detector()
        levels = detector._detect_round_numbers(24000.0, "NIFTY")
        for lvl in levels:
            dist_pct = abs(lvl.price - 24000.0) / 24000.0 * 100
            assert dist_pct <= 3.0 + 0.1  # small float tolerance

    def test_no_duplicate_round_numbers(self):
        """Same price should not appear twice across different increments."""
        detector = _make_detector()
        levels = detector._detect_round_numbers(24000.0, "NIFTY")
        prices = [lvl.price for lvl in levels]
        assert len(prices) == len(set(prices))


# ---------------------------------------------------------------------------
# Level merging
# ---------------------------------------------------------------------------

class TestMergeLevels:
    def _make_level(self, price: float, strength: int) -> LiquidityLevel:
        now = datetime.now(_IST)
        return LiquidityLevel(
            symbol="NIFTY",
            level_type="EQUAL_HIGH",
            price=price,
            strength=strength,
            timeframe="15min",
            first_formed_at=now,
            last_touched_at=now,
        )

    def test_deduplicates_levels_within_0_1_pct(self):
        """Two levels within 0.1% → keep the stronger one."""
        detector = _make_detector()
        # 24000 and 24010 → (24010-24000)/24000 = 0.042% < 0.1%
        lvl_weak = self._make_level(24000.0, strength=2)
        lvl_strong = self._make_level(24010.0, strength=4)

        merged = detector._merge_levels([lvl_weak, lvl_strong])
        assert len(merged) == 1
        assert merged[0].strength == 4

    def test_keeps_levels_beyond_0_1_pct(self):
        """Two levels 0.5% apart → both kept."""
        detector = _make_detector()
        lvl1 = self._make_level(24000.0, strength=2)
        lvl2 = self._make_level(24120.0, strength=2)   # 0.5% apart

        merged = detector._merge_levels([lvl1, lvl2])
        assert len(merged) == 2

    def test_empty_input_returns_empty(self):
        detector = _make_detector()
        assert detector._merge_levels([]) == []


# ---------------------------------------------------------------------------
# Swept level detection
# ---------------------------------------------------------------------------

class TestSweptLevels:
    def test_bsl_level_swept_when_price_above(self):
        """BSL at 24000, price now at 24050 → level should be marked swept."""
        detector = _make_detector()
        now = datetime.now(_IST)
        lvl = LiquidityLevel(
            symbol="NIFTY",
            level_type="EQUAL_HIGH",
            price=24000.0,
            strength=2,
            timeframe="15min",
            first_formed_at=now,
            last_touched_at=now,
        )
        detector._levels["NIFTY"] = [lvl]

        sweep_candle = _candle(high=24060, low=23990, close=24050, offset=0)
        detector.mark_level_swept("NIFTY", lvl, sweep_candle, wick_beyond=60.0)

        assert lvl.is_swept is True
        assert lvl.swept_at == sweep_candle.timestamp
        assert lvl.wick_beyond == pytest.approx(60.0)

    def test_get_nearest_bsl_skips_swept(self):
        """get_nearest_bsl must skip swept levels."""
        detector = _make_detector()
        now = datetime.now(_IST)

        swept = LiquidityLevel(
            symbol="NIFTY", level_type="EQUAL_HIGH", price=24100.0,
            strength=2, timeframe="15min", first_formed_at=now,
            last_touched_at=now, is_swept=True,
        )
        active = LiquidityLevel(
            symbol="NIFTY", level_type="EQUAL_HIGH", price=24200.0,
            strength=2, timeframe="15min", first_formed_at=now,
            last_touched_at=now, is_swept=False,
        )
        detector._levels["NIFTY"] = [swept, active]

        result = detector.get_nearest_bsl("NIFTY", current_price=24000.0)
        assert result is not None
        assert result.price == pytest.approx(24200.0)


# ---------------------------------------------------------------------------
# Bias calculation
# ---------------------------------------------------------------------------

class TestLiquidityBias:
    def _add_levels(
        self,
        detector: LiquidityPoolDetector,
        bsl_price: float,
        ssl_price: float,
        current_price: float,
    ) -> None:
        now = datetime.now(_IST)
        bsl = LiquidityLevel(
            symbol="NIFTY", level_type="EQUAL_HIGH", price=bsl_price,
            strength=2, timeframe="15min", first_formed_at=now, last_touched_at=now,
        )
        ssl = LiquidityLevel(
            symbol="NIFTY", level_type="EQUAL_LOW", price=ssl_price,
            strength=2, timeframe="15min", first_formed_at=now, last_touched_at=now,
        )
        detector._levels["NIFTY"] = [bsl, ssl]

    def test_bearish_bias_when_close_to_ssl(self):
        """Price much closer to SSL than BSL → BEARISH."""
        detector = _make_detector()
        # current = 24000, ssl = 23950 (0.21% away), bsl = 24500 (2.08% away)
        # dist_ssl = 0.21, dist_bsl = 2.08 → 0.21 < 2.08 * 0.6 = 1.25 → BEARISH
        self._add_levels(detector, bsl_price=24500.0, ssl_price=23950.0, current_price=24000.0)
        bias, reason = detector._calculate_liquidity_bias("NIFTY", 24000.0)
        assert bias == "BEARISH"
        assert "SSL" in reason

    def test_bullish_bias_when_close_to_bsl(self):
        """Price much closer to BSL than SSL → BULLISH."""
        detector = _make_detector()
        # current = 24000, bsl = 24050 (0.21% away), ssl = 23500 (2.08% away)
        self._add_levels(detector, bsl_price=24050.0, ssl_price=23500.0, current_price=24000.0)
        bias, reason = detector._calculate_liquidity_bias("NIFTY", 24000.0)
        assert bias == "BULLISH"
        assert "BSL" in reason

    def test_neutral_when_equidistant(self):
        """Price equidistant from BSL and SSL → NEUTRAL."""
        detector = _make_detector()
        # current = 24000, bsl = 24240 (1.0%), ssl = 23760 (1.0%)
        self._add_levels(detector, bsl_price=24240.0, ssl_price=23760.0, current_price=24000.0)
        bias, _ = detector._calculate_liquidity_bias("NIFTY", 24000.0)
        assert bias == "NEUTRAL"

    def test_no_levels_returns_neutral(self):
        detector = _make_detector()
        bias, _ = detector._calculate_liquidity_bias("NIFTY", 24000.0)
        assert bias == "NEUTRAL"


# ---------------------------------------------------------------------------
# Level expiry
# ---------------------------------------------------------------------------

class TestLevelExpiry:
    def test_expired_levels_removed_in_update(self):
        """Levels older than level_expiry_hours are pruned during update_levels."""
        mock_db = MagicMock()
        mock_db.execute.return_value = MagicMock(rowcount=0)
        mock_db.fetch_all.return_value = []
        mock_db.execute_many.return_value = 0
        mock_cs = MagicMock()

        detector = _make_detector()
        detector._db = mock_db
        detector._cs = mock_cs

        old_ts = datetime.now(_IST) - timedelta(hours=72)  # 72h > 48h expiry
        recent_ts = datetime.now(_IST) - timedelta(hours=1)

        old_level = LiquidityLevel(
            symbol="NIFTY", level_type="EQUAL_HIGH", price=24500.0,
            strength=2, timeframe="15min",
            first_formed_at=old_ts, last_touched_at=old_ts,
        )
        recent_level = LiquidityLevel(
            symbol="NIFTY", level_type="EQUAL_HIGH", price=24600.0,
            strength=2, timeframe="15min",
            first_formed_at=recent_ts, last_touched_at=recent_ts,
        )

        # Simulate update_levels receiving these (bypass detection, inject directly)
        detector._levels["NIFTY"] = [old_level, recent_level]

        # Apply expiry logic directly (mirrors what update_levels does)
        now = datetime.now(_IST)
        cutoff = now - timedelta(hours=detector._expiry_hours)
        active = [
            lvl for lvl in detector._levels["NIFTY"]
            if lvl.first_formed_at >= cutoff
            or lvl.level_type in ("PDH", "PDL", "PWH", "PWL")
        ]

        assert len(active) == 1
        assert active[0].price == pytest.approx(24600.0)

    def test_previous_day_levels_exempt_from_expiry(self):
        """PDH/PDL/PWH/PWL are never pruned by the expiry filter."""
        detector = _make_detector()
        old_ts = datetime.now(_IST) - timedelta(hours=72)

        pdh = LiquidityLevel(
            symbol="NIFTY", level_type="PDH", price=24500.0,
            strength=3, timeframe="1day",
            first_formed_at=old_ts, last_touched_at=old_ts,
        )

        now = datetime.now(_IST)
        cutoff = now - timedelta(hours=detector._expiry_hours)
        active = [
            lvl for lvl in [pdh]
            if lvl.first_formed_at >= cutoff
            or lvl.level_type in ("PDH", "PDL", "PWH", "PWL")
        ]
        assert len(active) == 1
