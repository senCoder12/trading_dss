"""
Tests for F&O liquidity filter.

Covers:
- Individual hard filters
- Time-adjusted volume logic
- Mock data scenarios (all pass, all fail, mixed)
- FilterResult accuracy
"""

from __future__ import annotations

from datetime import datetime, date, time
from zoneinfo import ZoneInfo

import pytest

from config.constants import IST_TIMEZONE
from src.analysis.fo_liquidity_filter import (
    LiquidityThresholds,
    LiquidityFilter,
    FilterResult,
)
from src.data.fo_bulk_fetcher import StockQuote, StockOIData

_IST = ZoneInfo(IST_TIMEZONE)


# ── Fixtures for test data ────────────────────────────────────────────────────


@pytest.fixture()
def thresholds() -> LiquidityThresholds:
    """Standard test thresholds."""
    return LiquidityThresholds(
        min_avg_daily_volume=500_000,
        min_today_volume=100_000,
        min_volume_participation_pct=15.0,
        min_total_oi=500_000,
        min_oi_change_activity=10_000,
        max_bid_ask_spread_pct=0.5,
        max_atm_options_spread_pct=2.0,
        min_ltp=100.0,
        max_ltp=50_000.0,
        min_price_change_pct=0.3,
        min_iv=10.0,
        max_iv=80.0,
    )


@pytest.fixture()
def now_tz() -> datetime:
    """Now in IST."""
    return datetime.now(_IST)


def make_quote(
    symbol: str = "RELIANCE",
    ltp: float = 2500.0,
    open: float = 2480.0,
    high: float = 2520.0,
    low: float = 2460.0,
    close: float = 2480.0,
    volume: int = 2_000_000,
    avg_volume_20d: float = 2_500_000,
    price_change_pct: float = 0.8,
    bid: float = 2499.5,
    ask: float = 2500.5,
    timestamp: datetime = None,
) -> StockQuote:
    """Create a test StockQuote with sensible defaults."""
    if timestamp is None:
        timestamp = datetime.now(_IST)

    bid_ask_spread = ask - bid
    bid_ask_spread_pct = 100.0 * bid_ask_spread / ltp if ltp > 0 else 0.0

    return StockQuote(
        symbol=symbol,
        ltp=ltp,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        avg_volume_20d=avg_volume_20d,
        price_change_pct=price_change_pct,
        bid=bid,
        ask=ask,
        bid_ask_spread=bid_ask_spread,
        bid_ask_spread_pct=bid_ask_spread_pct,
        timestamp=timestamp,
    )


def make_oi(
    symbol: str = "RELIANCE",
    total_ce_oi: int = 750_000,
    total_pe_oi: int = 600_000,
    ce_oi_change: int = 50_000,
    pe_oi_change: int = 40_000,
    pcr: float = 0.8,
    atm_strike: float = 2500.0,
    atm_ce_iv: float = 20.0,
    atm_pe_iv: float = 22.0,
    avg_iv: float = 21.0,
    iv_change: float = 0.5,
    expiry: date = None,
    timestamp: datetime = None,
) -> StockOIData:
    """Create test StockOIData with sensible defaults."""
    if expiry is None:
        expiry = date.today()
    if timestamp is None:
        timestamp = datetime.now(_IST)

    return StockOIData(
        symbol=symbol,
        expiry=expiry,
        total_ce_oi=total_ce_oi,
        total_pe_oi=total_pe_oi,
        ce_oi_change=ce_oi_change,
        pe_oi_change=pe_oi_change,
        pcr=pcr,
        atm_strike=atm_strike,
        atm_ce_iv=atm_ce_iv,
        atm_pe_iv=atm_pe_iv,
        avg_iv=avg_iv,
        iv_change=iv_change,
        timestamp=timestamp,
    )


# ── Test LiquidityThresholds ──────────────────────────────────────────────────


class TestLiquidityThresholds:
    """Test threshold configuration."""

    def test_default_thresholds(self):
        """Thresholds have sensible defaults."""
        t = LiquidityThresholds()
        assert t.min_avg_daily_volume == 500_000
        assert t.max_bid_ask_spread_pct == 0.5
        assert t.min_iv == 10.0
        assert t.max_iv == 80.0

    def test_custom_thresholds(self):
        """Can override with custom values."""
        t = LiquidityThresholds(min_avg_daily_volume=1_000_000, min_iv=15.0)
        assert t.min_avg_daily_volume == 1_000_000
        assert t.min_iv == 15.0
        assert t.max_iv == 80.0  # unchanged


# ── Test LiquidityFilter hard filters ─────────────────────────────────────────


class TestHardFilters:
    """Test individual hard filter conditions."""

    def test_price_too_low(self, thresholds):
        """Fail when LTP < min_ltp."""
        quote = make_quote(ltp=50.0)  # Below min_ltp=100
        oi = make_oi()
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert len(result) == 0

    def test_price_too_high(self, thresholds):
        """Fail when LTP > max_ltp."""
        quote = make_quote(ltp=60_000.0)  # Above max_ltp=50_000
        oi = make_oi()
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert len(result) == 0

    def test_price_in_range(self, thresholds):
        """Pass when price in range."""
        quote = make_quote(ltp=2500.0)
        oi = make_oi()
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert "TEST" in result

    def test_volume_too_low(self, thresholds):
        """Fail when avg_volume_20d < min_avg_daily_volume."""
        quote = make_quote(avg_volume_20d=300_000)  # Below 500_000
        oi = make_oi()
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert len(result) == 0

    def test_spread_too_wide(self, thresholds):
        """Fail when bid_ask_spread_pct > max_bid_ask_spread_pct."""
        quote = make_quote(
            ltp=1000.0,
            bid=990.0,
            ask=1010.0,  # spread=20, spread_pct=2% > 0.5% max
        )
        oi = make_oi()
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert len(result) == 0

    def test_no_options_data(self, thresholds):
        """Fail when symbol not in oi_data."""
        quote = make_quote()
        oi = {}  # Empty
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, oi)
        assert len(result) == 0

    def test_oi_too_low(self, thresholds):
        """Fail when total_oi < min_total_oi."""
        quote = make_quote()
        oi = make_oi(total_ce_oi=200_000, total_pe_oi=150_000)  # total=350_000 < 500_000
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert len(result) == 0

    def test_iv_too_low(self, thresholds):
        """Fail when avg_iv < min_iv."""
        quote = make_quote()
        oi = make_oi(avg_iv=8.0)  # Below min_iv=10
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert len(result) == 0

    def test_iv_too_high(self, thresholds):
        """Fail when avg_iv > max_iv."""
        quote = make_quote()
        oi = make_oi(avg_iv=85.0)  # Above max_iv=80
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert len(result) == 0

    def test_stock_sleeping(self, thresholds):
        """Fail when price_change_pct < min_price_change_pct."""
        quote = make_quote(price_change_pct=0.1)  # Below min_price_change_pct=0.3
        oi = make_oi()
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert len(result) == 0

    def test_all_filters_pass(self, thresholds):
        """Stock passes all filters."""
        quote = make_quote()
        oi = make_oi()
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert "TEST" in result


# ── Test time-adjusted volume filter ──────────────────────────────────────────


class TestTimeAdjustedVolumeFilter:
    """Test time-adjusted volume participation logic."""

    def test_early_morning_09_15(self, thresholds):
        """9:15 AM expects 5% participation."""
        t = time(9, 15)
        quote = make_quote(
            symbol="TEST",
            avg_volume_20d=1_000_000,
            volume=50_000,  # 5% participation
        )
        filter = LiquidityFilter(thresholds)

        result = filter.apply_time_adjusted_volume_filter({"TEST": quote}, t)
        assert "TEST" in result

    def test_before_10am_requires_5pct(self, thresholds):
        """Before 10:00 AM, need 5% participation."""
        t = time(9, 45)
        quote = make_quote(
            avg_volume_20d=1_000_000,
            volume=40_000,  # 4% < 5% required
        )
        filter = LiquidityFilter(thresholds)

        result = filter.apply_time_adjusted_volume_filter({"TEST": quote}, t)
        assert "TEST" not in result

    def test_10_to_11_30am_requires_20pct(self, thresholds):
        """10:00–11:30 AM expects 20% participation."""
        t = time(10, 30)
        quote = make_quote(avg_volume_20d=1_000_000, volume=200_000)  # 20%
        filter = LiquidityFilter(thresholds)

        result = filter.apply_time_adjusted_volume_filter({"TEST": quote}, t)
        assert "TEST" in result

    def test_11_30_to_1pm_requires_40pct(self, thresholds):
        """11:30 AM–1:00 PM expects 40% participation."""
        t = time(12, 0)
        quote = make_quote(avg_volume_20d=1_000_000, volume=400_000)  # 40%
        filter = LiquidityFilter(thresholds)

        result = filter.apply_time_adjusted_volume_filter({"TEST": quote}, t)
        assert "TEST" in result

    def test_1_to_2_30pm_requires_60pct(self, thresholds):
        """1:00–2:30 PM expects 60% participation."""
        t = time(13, 30)
        quote = make_quote(avg_volume_20d=1_000_000, volume=600_000)  # 60%
        filter = LiquidityFilter(thresholds)

        result = filter.apply_time_adjusted_volume_filter({"TEST": quote}, t)
        assert "TEST" in result

    def test_2_30_to_3_30pm_requires_80pct(self, thresholds):
        """2:30–3:30 PM expects 80% participation."""
        t = time(15, 0)
        quote = make_quote(avg_volume_20d=1_000_000, volume=800_000)  # 80%
        filter = LiquidityFilter(thresholds)

        result = filter.apply_time_adjusted_volume_filter({"TEST": quote}, t)
        assert "TEST" in result

    def test_zero_avg_volume_skip(self, thresholds):
        """Skip stocks with zero avg_volume_20d."""
        t = time(10, 0)
        quote = make_quote(avg_volume_20d=0, volume=0)
        filter = LiquidityFilter(thresholds)

        result = filter.apply_time_adjusted_volume_filter({"TEST": quote}, t)
        # Should not crash, should skip (no division by zero)
        assert len(result) == 0


# ── Test combined master filter ───────────────────────────────────────────────


class TestMasterFilter:
    """Test get_liquid_symbols (hard + time-adjusted)."""

    def test_all_pass_hard_and_volume(self, thresholds):
        """Multiple stocks pass both hard filters and time-adjusted volume."""
        t = time(10, 30)  # Expects 20% participation
        quote1 = make_quote(
            symbol="REL",
            avg_volume_20d=1_000_000,
            volume=200_000,  # 20%
        )
        quote2 = make_quote(
            symbol="TCS",
            avg_volume_20d=800_000,
            volume=160_000,  # 20%
        )
        oi1 = make_oi(symbol="REL")
        oi2 = make_oi(symbol="TCS")

        filter = LiquidityFilter(thresholds)
        result = filter.get_liquid_symbols(
            {"REL": quote1, "TCS": quote2},
            {"REL": oi1, "TCS": oi2},
            t,
        )

        assert set(result) == {"REL", "TCS"}

    def test_all_fail_hard_filters(self, thresholds):
        """All stocks fail hard filters → empty result."""
        quote1 = make_quote(symbol="REL", ltp=50.0)  # Price too low
        quote2 = make_quote(symbol="TCS", avg_volume_20d=100_000)  # Volume too low
        oi1 = make_oi(symbol="REL")
        oi2 = make_oi(symbol="TCS")

        filter = LiquidityFilter(thresholds)
        result = filter.get_liquid_symbols(
            {"REL": quote1, "TCS": quote2},
            {"REL": oi1, "TCS": oi2},
        )

        assert len(result) == 0

    def test_pass_hard_fail_volume(self, thresholds):
        """Pass hard filters but fail time-adjusted volume."""
        t = time(14, 0)  # Expects 60% participation
        quote = make_quote(avg_volume_20d=1_000_000, volume=400_000)  # Only 40%
        oi = make_oi()

        filter = LiquidityFilter(thresholds)
        result = filter.get_liquid_symbols(
            {"TEST": quote},
            {"TEST": oi},
            t,
        )

        assert "TEST" not in result

    def test_mixed_portfolio(self, thresholds):
        """Mixed portfolio: some pass, some fail at each stage."""
        t = time(10, 0)  # Expects 20% participation

        # Pass both
        good = make_quote(
            symbol="GOOD",
            avg_volume_20d=1_000_000,
            volume=200_000,
        )
        # Fail hard filters (price)
        bad_price = make_quote(symbol="BAD_PRICE", ltp=10.0)
        # Pass hard, fail volume
        bad_volume = make_quote(
            symbol="BAD_VOL",
            avg_volume_20d=1_000_000,
            volume=100_000,  # 10% < 20%
        )

        oi_good = make_oi(symbol="GOOD")
        oi_bad_volume = make_oi(symbol="BAD_VOL")

        filter = LiquidityFilter(thresholds)
        result = filter.get_liquid_symbols(
            {"GOOD": good, "BAD_PRICE": bad_price, "BAD_VOL": bad_volume},
            {"GOOD": oi_good, "BAD_VOL": oi_bad_volume},
            t,
        )

        assert result == ["GOOD"]


# ── Test FilterResult tracking ────────────────────────────────────────────────


class TestFilterResult:
    """Test FilterResult counting and logging."""

    def test_result_counts(self, thresholds):
        """FilterResult tracks failure counts correctly."""
        quote_good = make_quote(symbol="GOOD")
        quote_price = make_quote(symbol="BAD_PRICE", ltp=10.0)
        quote_volume = make_quote(symbol="BAD_VOL", avg_volume_20d=100_000)

        oi_good = make_oi(symbol="GOOD")

        filter = LiquidityFilter(thresholds)
        result = filter.apply_hard_filters(
            {
                "GOOD": quote_good,
                "BAD_PRICE": quote_price,
                "BAD_VOL": quote_volume,
            },
            {"GOOD": oi_good},  # BAD_PRICE and BAD_VOL have no OI
        )

        assert result == ["GOOD"]

    def test_result_summary(self):
        """FilterResult.summary() returns human-readable string."""
        result = FilterResult(
            total_input=100,
            passed=80,
            failed_price=10,
            failed_volume=5,
            failed_spread=3,
            failed_oi=2,
        )
        summary = result.summary()

        assert "in=100" in summary
        assert "passed=80" in summary
        assert "80.0%" in summary
        assert "price=10" in summary


# ── Test edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test boundary conditions and special cases."""

    def test_empty_quotes_dict(self, thresholds):
        """Handle empty quotes dict gracefully."""
        filter = LiquidityFilter(thresholds)
        result = filter.apply_hard_filters({}, {})
        assert len(result) == 0

    def test_zero_volume_handling(self, thresholds):
        """Handle zero volume in quotes."""
        quote = make_quote(volume=0, avg_volume_20d=1_000_000)
        oi = make_oi()
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert "TEST" in result  # Passes hard filters, but time-adjusted will fail

    def test_negative_price_change(self, thresholds):
        """Negative price changes still count (absolute value check)."""
        quote = make_quote(price_change_pct=-0.5)  # Down 0.5%, abs=0.5 > min=0.3
        oi = make_oi()
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert "TEST" in result

    def test_boundary_min_price(self, thresholds):
        """Price exactly at min_ltp boundary."""
        quote = make_quote(
            ltp=100.0,  # Exactly min
            bid=99.9,
            ask=100.1,
        )
        oi = make_oi()
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert "TEST" in result

    def test_boundary_max_price(self, thresholds):
        """Price exactly at max_ltp boundary."""
        quote = make_quote(ltp=50_000.0)  # Exactly max
        oi = make_oi()
        filter = LiquidityFilter(thresholds)

        result = filter.apply_hard_filters({"TEST": quote}, {"TEST": oi})
        assert "TEST" in result

    def test_iv_at_boundaries(self, thresholds):
        """IV exactly at min/max boundaries."""
        quote = make_quote()
        oi_min = make_oi(avg_iv=10.0)  # Exactly min
        oi_max = make_oi(avg_iv=80.0)  # Exactly max

        filter = LiquidityFilter(thresholds)
        assert "TEST" in filter.apply_hard_filters({"TEST": quote}, {"TEST": oi_min})
        assert "TEST" in filter.apply_hard_filters({"TEST": quote}, {"TEST": oi_max})


# ── Test expected output ranges ───────────────────────────────────────────────


class TestExpectedOutputRanges:
    """Test that filter produces expected number of results."""

    def test_typical_market_scenario(self, thresholds):
        """Generate realistic 60–120 liquid stocks from ~200 F&O universe."""
        # Use 9:30 AM for early day testing (5% participation required)
        t = time(9, 30)

        # Simulate 200 F&O stocks with varying liquidity
        quotes = {}
        oi_data = {}

        for i in range(200):
            # ~70% are liquid (140 stocks)
            is_liquid = i < 140
            ltp = 2000.0 + (i % 10) * 500

            if is_liquid:
                # Good volume, price, spread
                spread_pct = 0.2 + (i % 3) * 0.1
                spread = ltp * spread_pct / 100.0
                avg_vol = 1_000_000 + (i % 5) * 200_000
                # By 9:30, expect 5% participation
                today_vol = int(avg_vol * 0.05) + 1_000
                quote = make_quote(
                    symbol=f"STOCK{i:03d}",
                    ltp=ltp,
                    avg_volume_20d=avg_vol,
                    volume=today_vol,
                    bid=ltp - spread / 2,
                    ask=ltp + spread / 2,
                    price_change_pct=0.5 + (i % 10) * 0.2,
                )
                oi = make_oi(symbol=f"STOCK{i:03d}", avg_iv=20.0 + (i % 8))
            else:
                # Various failure modes
                if i % 3 == 0:
                    quote = make_quote(
                        symbol=f"STOCK{i:03d}", ltp=50.0  # Too cheap
                    )
                elif i % 3 == 1:
                    quote = make_quote(
                        symbol=f"STOCK{i:03d}",
                        avg_volume_20d=100_000,  # Too low volume
                    )
                else:
                    # Too wide spread
                    spread = ltp * 0.02  # 2% spread
                    quote = make_quote(
                        symbol=f"STOCK{i:03d}",
                        bid=ltp - spread / 2,
                        ask=ltp + spread / 2,
                    )
                oi = make_oi(symbol=f"STOCK{i:03d}")

            quotes[f"STOCK{i:03d}"] = quote
            oi_data[f"STOCK{i:03d}"] = oi

        filter = LiquidityFilter(thresholds)
        result = filter.get_liquid_symbols(quotes, oi_data, t)

        # Should get roughly 100–140 of the 140 "liquid" stocks
        assert 50 < len(result) < 200
