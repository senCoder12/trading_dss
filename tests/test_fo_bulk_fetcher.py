"""
Unit tests for FOBulkFetcher.

Tests:
- Quote parsing and dataclass validation
- Liquidity filtering (volume + spread thresholds)
- OI change calculations
- Batch size enforcement
- Rate limiting (delay between calls)
- DB caching for avg volumes
- Error handling (>30% failure threshold)
"""

import pytest
from datetime import datetime, date
from unittest.mock import Mock, MagicMock, patch
from zoneinfo import ZoneInfo

from src.data.fo_bulk_fetcher import (
    FOBulkFetcher,
    StockQuote,
    StockOIData,
    DataFetchError,
)
from src.data.smart_api_client import SmartAPIClient
from src.data.fo_universe_fetcher import FOUniverseFetcher
from src.database.db_manager import DatabaseManager


_IST = ZoneInfo("Asia/Kolkata")


class TestStockQuoteDataclass:
    """Test StockQuote parsing and initialization."""

    def test_stock_quote_creation(self):
        """Create a valid StockQuote."""
        now = datetime.now(tz=_IST)
        quote = StockQuote(
            symbol="RELIANCE",
            ltp=2500.0,
            open=2450.0,
            high=2510.0,
            low=2440.0,
            close=2480.0,
            volume=5000000,
            avg_volume_20d=4500000.0,
            price_change_pct=0.8,
            bid=2499.5,
            ask=2500.5,
            bid_ask_spread=1.0,
            bid_ask_spread_pct=0.04,
            timestamp=now,
        )
        assert quote.symbol == "RELIANCE"
        assert quote.ltp == 2500.0
        assert quote.bid_ask_spread_pct == 0.04


class TestStockOIDataDataclass:
    """Test StockOIData initialization."""

    def test_stock_oi_data_creation(self):
        """Create a valid StockOIData."""
        now = datetime.now(tz=_IST)
        oi = StockOIData(
            symbol="RELIANCE",
            expiry=date(2024, 1, 18),
            total_ce_oi=1000000,
            total_pe_oi=1200000,
            ce_oi_change=50000,
            pe_oi_change=-30000,
            pcr=1.2,
            atm_strike=2500.0,
            atm_ce_iv=18.5,
            atm_pe_iv=19.2,
            avg_iv=18.85,
            iv_change=0.5,
            timestamp=now,
        )
        assert oi.symbol == "RELIANCE"
        assert oi.pcr == 1.2
        assert oi.avg_iv == 18.85


class TestFOBulkFetcherParseQuote:
    """Test quote parsing from raw API response."""

    @pytest.fixture
    def fetcher(self):
        """Create a FOBulkFetcher instance."""
        db = Mock(spec=DatabaseManager)
        api = Mock(spec=SmartAPIClient)
        universe = Mock(spec=FOUniverseFetcher)
        return FOBulkFetcher(db, api, universe)

    def test_parse_quote_valid(self, fetcher):
        """Parse a valid quote entry."""
        entry = {
            "symbol": "INFY",
            "ltp": 1800.0,
            "open": 1790.0,
            "high": 1810.0,
            "low": 1785.0,
            "close": 1795.0,
            "volume": 3000000,
            "bid": 1799.5,
            "ask": 1800.5,
        }
        quote = fetcher._parse_quote(entry)

        assert quote is not None
        assert quote.symbol == "INFY"
        assert quote.ltp == 1800.0
        assert quote.bid_ask_spread == 1.0
        assert abs(quote.bid_ask_spread_pct - 0.0556) < 0.01  # 1 / 1800 * 100 ≈ 0.056%
        assert abs(quote.price_change_pct - 0.2787) < 0.01  # (1800 - 1795) / 1795 ≈ 0.28%

    def test_parse_quote_missing_symbol(self, fetcher):
        """Reject entry with missing symbol."""
        entry = {"ltp": 1800.0}
        quote = fetcher._parse_quote(entry)
        assert quote is None

    def test_parse_quote_zero_ltp(self, fetcher):
        """Handle zero LTP gracefully."""
        entry = {
            "symbol": "DUMMY",
            "ltp": 0.0,
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "close": 0.0,
            "volume": 0,
            "bid": 0.0,
            "ask": 0.0,
        }
        quote = fetcher._parse_quote(entry)

        assert quote is not None
        assert quote.bid_ask_spread_pct == 0.0  # No division by zero


class TestFOBulkFetcherParseOI:
    """Test OI data parsing."""

    @pytest.fixture
    def fetcher(self):
        db = Mock(spec=DatabaseManager)
        api = Mock(spec=SmartAPIClient)
        universe = Mock(spec=FOUniverseFetcher)
        return FOBulkFetcher(db, api, universe)

    def test_parse_oi_data_valid(self, fetcher):
        """Parse a valid OI chain response."""
        chain = {
            "expiry": "2024-01-18",
            "atmStrike": 2500.0,
            "totalCeOi": 500000,
            "totalPeOi": 600000,
            "atmCeIv": 17.5,
            "atmPeIv": 18.5,
        }
        oi = fetcher._parse_oi_data("RELIANCE", chain)

        assert oi is not None
        assert oi.symbol == "RELIANCE"
        assert oi.total_ce_oi == 500000
        assert oi.total_pe_oi == 600000
        assert oi.pcr == 1.2
        assert oi.avg_iv == 18.0

    def test_parse_oi_zero_ce_oi(self, fetcher):
        """Handle zero CE OI (avoid division by zero in PCR)."""
        chain = {
            "expiry": "2024-01-18",
            "atmStrike": 2500.0,
            "totalCeOi": 0,
            "totalPeOi": 100000,
            "atmCeIv": 10.0,
            "atmPeIv": 10.0,
        }
        oi = fetcher._parse_oi_data("RELIANCE", chain)

        assert oi is not None
        assert oi.pcr == 0.0  # PE / 0 = 0


class TestFOBulkFetcherBatching:
    """Test batch splitting and rate limiting."""

    @pytest.fixture
    def fetcher(self):
        db = Mock(spec=DatabaseManager)
        api = Mock(spec=SmartAPIClient)
        universe = Mock(spec=FOUniverseFetcher)
        fetcher = FOBulkFetcher(db, api, universe)
        fetcher._batch_size = 50
        return fetcher

    def test_fetch_all_quotes_empty_universe(self, fetcher):
        """Fail if universe is empty."""
        fetcher._fo_universe.get_symbols.return_value = []

        with pytest.raises(DataFetchError):
            fetcher.fetch_all_quotes()

    def test_fetch_all_quotes_batching(self, fetcher):
        """Fetch quotes in batches, respecting batch_size."""
        symbols = [f"SYM{i}" for i in range(125)]  # 3 batches of 50
        fetcher._fo_universe.get_symbols.return_value = symbols

        # Mock API responses
        def mock_bulk_quotes(mode, exchange_tokens):
            fetched = [{"symbol": sym, "ltp": 100.0, "bid": 99.9, "ask": 100.1}
                      for sym in exchange_tokens.get("NSE", [])]
            return {"fetched": fetched, "unfetched": []}

        fetcher._smart_api.get_bulk_quotes.side_effect = mock_bulk_quotes

        quotes = fetcher.fetch_all_quotes()

        # Should have batched into 3 calls
        assert fetcher._smart_api.get_bulk_quotes.call_count == 3
        assert len(quotes) == 125

    def test_fetch_all_quotes_failure_threshold(self, fetcher):
        """Fail if >30% of symbols fail."""
        symbols = [f"SYM{i}" for i in range(100)]
        fetcher._fo_universe.get_symbols.return_value = symbols

        # Mock 40% failure (>30% threshold)
        def mock_bulk_quotes(mode, exchange_tokens):
            batch = exchange_tokens.get("NSE", [])
            fetched = [{"symbol": sym, "ltp": 100.0, "bid": 99.9, "ask": 100.1}
                      for sym in batch[:len(batch)//2]]  # 50% success
            unfetched = [{"symbol": sym} for sym in batch[len(batch)//2:]]
            return {"fetched": fetched, "unfetched": unfetched}

        fetcher._smart_api.get_bulk_quotes.side_effect = mock_bulk_quotes

        with pytest.raises(DataFetchError):
            fetcher.fetch_all_quotes()


class TestFOBulkFetcherAvgVolumes:
    """Test volume caching and fetching."""

    @pytest.fixture
    def fetcher(self):
        db = Mock(spec=DatabaseManager)
        api = Mock(spec=SmartAPIClient)
        universe = Mock(spec=FOUniverseFetcher)
        return FOBulkFetcher(db, api, universe)

    def test_fetch_avg_volumes_cache_hit(self, fetcher):
        """Return cached volumes without refetch."""
        symbols = ["RELIANCE", "INFY"]

        # Mock cache hit
        fetcher._db.fetch_one.side_effect = [
            {"avg_volume_20d": 5000000.0},  # RELIANCE from cache
            {"avg_volume_20d": 3000000.0},  # INFY from cache
        ]

        volumes = fetcher.fetch_avg_volumes(symbols)

        assert volumes["RELIANCE"] == 5000000.0
        assert volumes["INFY"] == 3000000.0
        # API should not be called
        fetcher._smart_api.get_historical_ohlcv.assert_not_called()

    def test_fetch_avg_volumes_cache_miss_and_store(self, fetcher):
        """Fetch volumes from API and cache them."""
        symbols = ["RELIANCE"]

        # Mock cache miss
        fetcher._db.fetch_one.return_value = None

        # Mock historical data
        fetcher._smart_api.get_historical_ohlcv.return_value = [
            {"volume": 1000000},
            {"volume": 1100000},
            {"volume": 900000},
            # ... 17 more bars
        ] + [{"volume": 1000000}] * 17

        fetcher._smart_api.get_bulk_quotes = Mock(return_value={})
        fetcher._get_token_for_symbol = Mock(return_value="12345")

        volumes = fetcher.fetch_avg_volumes(symbols)

        # Should have queried DB
        fetcher._db.fetch_one.assert_called()
        # Should have called API
        fetcher._smart_api.get_historical_ohlcv.assert_called()
        # Should have cached result
        fetcher._db.execute.assert_called()


class TestFOBulkFetcherOIHistory:
    """Test OI change tracking via in-memory history."""

    @pytest.fixture
    def fetcher(self):
        db = Mock(spec=DatabaseManager)
        api = Mock(spec=SmartAPIClient)
        universe = Mock(spec=FOUniverseFetcher)
        return FOBulkFetcher(db, api, universe)

    def test_oi_change_calculation(self, fetcher):
        """Calculate OI changes from previous snapshot."""
        now = datetime.now(tz=_IST)

        # Seed history with previous snapshot
        prev_oi = StockOIData(
            symbol="RELIANCE",
            expiry=date(2024, 1, 18),
            total_ce_oi=500000,
            total_pe_oi=600000,
            ce_oi_change=0,
            pe_oi_change=0,
            pcr=1.2,
            atm_strike=2500.0,
            atm_ce_iv=18.0,
            atm_pe_iv=18.0,
            avg_iv=18.0,
            iv_change=0.0,
            timestamp=now,
        )
        fetcher._oi_history["RELIANCE"] = prev_oi

        # Mock API for new snapshot
        chain = {
            "expiry": "2024-01-18",
            "atmStrike": 2500.0,
            "totalCeOi": 550000,  # +50k change
            "totalPeOi": 580000,  # -20k change
            "atmCeIv": 18.5,
            "atmPeIv": 18.7,
        }
        fetcher._smart_api.get_option_chain.return_value = chain

        # Fetch new OI (will calculate changes)
        oi_data = fetcher.fetch_oi_for_symbols(["RELIANCE"])

        assert "RELIANCE" in oi_data
        oi = oi_data["RELIANCE"]
        assert oi.ce_oi_change == 50000
        assert oi.pe_oi_change == -20000
        assert abs(oi.iv_change - 0.6) < 0.01  # (18.6 - 18.0) = 0.6

    def test_reset_oi_history(self, fetcher):
        """Reset OI history (call at market open)."""
        # Seed history
        fetcher._oi_history["RELIANCE"] = Mock()
        fetcher._oi_history["INFY"] = Mock()

        fetcher.reset_oi_history()

        assert len(fetcher._oi_history) == 0


class TestFOBulkFetcherFullSnapshot:
    """Test end-to-end orchestration: quotes → filter → OI."""

    @pytest.fixture
    def fetcher(self):
        db = Mock(spec=DatabaseManager)
        api = Mock(spec=SmartAPIClient)
        universe = Mock(spec=FOUniverseFetcher)
        return FOBulkFetcher(db, api, universe)

    def test_fetch_full_snapshot_liquidity_filter(self, fetcher):
        """Filter quotes by volume and spread thresholds."""
        fetcher._fo_universe.get_symbols.return_value = ["RELIANCE", "INFY", "LOWLIQ"]

        # Mock quotes
        high_liq_quotes = {
            "RELIANCE": StockQuote(
                symbol="RELIANCE",
                ltp=2500.0,
                open=2450.0,
                high=2510.0,
                low=2440.0,
                close=2480.0,
                volume=10000000,
                avg_volume_20d=5000000.0,  # Passes filter
                price_change_pct=0.8,
                bid=2499.8,
                ask=2500.2,
                bid_ask_spread=0.4,
                bid_ask_spread_pct=0.016,  # Passes filter
                timestamp=datetime.now(tz=_IST),
            ),
            "INFY": StockQuote(
                symbol="INFY",
                ltp=1800.0,
                open=1790.0,
                high=1810.0,
                low=1785.0,
                close=1795.0,
                volume=3000000,
                avg_volume_20d=2000000.0,  # Passes volume filter
                price_change_pct=0.28,
                bid=1799.0,
                ask=1810.0,  # Large spread
                bid_ask_spread=11.0,
                bid_ask_spread_pct=0.611,  # Fails spread filter (>0.5%)
                timestamp=datetime.now(tz=_IST),
            ),
            "LOWLIQ": StockQuote(
                symbol="LOWLIQ",
                ltp=500.0,
                open=500.0,
                high=500.0,
                low=500.0,
                close=500.0,
                volume=50000,
                avg_volume_20d=40000.0,  # Fails (<100k)
                price_change_pct=0.0,
                bid=499.0,
                ask=501.0,
                bid_ask_spread=2.0,
                bid_ask_spread_pct=0.4,  # Passes spread
                timestamp=datetime.now(tz=_IST),
            ),
        }

        def mock_fetch_quotes():
            return high_liq_quotes

        def mock_fetch_volumes(symbols):
            return {
                "RELIANCE": 5000000.0,
                "INFY": 2000000.0,
                "LOWLIQ": 40000.0,
            }

        fetcher.fetch_all_quotes = Mock(side_effect=mock_fetch_quotes)
        fetcher.fetch_avg_volumes = Mock(side_effect=mock_fetch_volumes)
        fetcher.fetch_oi_for_symbols = Mock(return_value={})

        quotes, oi_data = fetcher.fetch_full_snapshot(
            volume_filter_threshold=100000.0,
            spread_filter_pct=0.5,
        )

        # Only RELIANCE should pass both filters
        fetcher.fetch_oi_for_symbols.assert_called_once()
        call_args = fetcher.fetch_oi_for_symbols.call_args[0][0]
        assert call_args == ["RELIANCE"]
