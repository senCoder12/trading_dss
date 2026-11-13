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


def _nse_chain(ce_oi, pe_oi, ce_iv, pe_iv, strike=2500.0, expiry="18-Jan-2024",
               ce_chg=0, pe_chg=0):
    """Minimal NSE option-chain-v3 response for one ATM strike."""
    return {
        "records": {
            "underlyingValue": strike,
            "expiryDates": [expiry],
            "data": [{
                "strikePrice": strike,
                "expiryDates": expiry,
                "CE": {"openInterest": ce_oi, "impliedVolatility": ce_iv,
                       "changeinOpenInterest": ce_chg},
                "PE": {"openInterest": pe_oi, "impliedVolatility": pe_iv,
                       "changeinOpenInterest": pe_chg},
            }],
        },
        "filtered": {},
    }


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
        chain = _nse_chain(500000, 600000, 17.5, 18.5)
        oi = fetcher._parse_oi_data("RELIANCE", chain)

        assert oi is not None
        assert oi.symbol == "RELIANCE"
        assert oi.total_ce_oi == 500000
        assert oi.total_pe_oi == 600000
        assert oi.pcr == 1.2
        assert oi.avg_iv == 18.0

    def test_parse_oi_zero_ce_oi(self, fetcher):
        """Handle zero CE OI (avoid division by zero in PCR)."""
        chain = _nse_chain(0, 100000, 10.0, 10.0)
        oi = fetcher._parse_oi_data("RELIANCE", chain)

        assert oi is not None
        assert oi.pcr == 0.0  # PE / 0 = 0

    def test_parse_oi_first_look_uses_nse_day_change(self, fetcher):
        """Without a previous snapshot, OI change = NSE changeinOpenInterest."""
        chain = _nse_chain(500000, 600000, 17.5, 18.5, ce_chg=1200, pe_chg=-300)
        oi = fetcher._parse_oi_data("RELIANCE", chain)

        assert oi.ce_oi_change == 1200
        assert oi.pe_oi_change == -300


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
        # Token map normally comes from the Angel master; token == symbol here
        fetcher._token_map = {s: s for s in symbols}
        fetcher._reverse_token_map = {s: s for s in symbols}
        fetcher._master_loaded_date = datetime.now(tz=_IST).date()

        # Mock API responses (SNAP_QUOTE carries close and volume)
        def mock_bulk_quotes(mode, exchange_tokens):
            fetched = [{"symbolToken": tok, "ltp": 100.0, "close": 99.0,
                        "volume": 1000, "bid": 99.9, "ask": 100.1}
                      for tok in exchange_tokens.get("NSE", [])]
            return {"fetched": fetched, "unfetched": []}

        fetcher._smart_api.get_bulk_quotes.side_effect = mock_bulk_quotes

        quotes = fetcher.fetch_all_quotes()

        # Should have batched into 3 calls
        assert fetcher._smart_api.get_bulk_quotes.call_count == 3
        assert len(quotes) == 125

    def test_ltp_only_quotes_fall_back_to_nse(self, fetcher):
        """LTP-mode quotes (no close/volume) trigger the NSE fallback."""
        symbols = ["AAA", "BBB"]
        fetcher._fo_universe.get_symbols.return_value = symbols
        fetcher._token_map = {s: s for s in symbols}
        fetcher._reverse_token_map = {s: s for s in symbols}
        fetcher._master_loaded_date = datetime.now(tz=_IST).date()
        fetcher._smart_api.get_bulk_quotes.side_effect = lambda mode, et: {
            "fetched": [{"symbolToken": t, "ltp": 100.0} for t in et["NSE"]],
            "unfetched": [],
        }
        fetcher._nse_scraper = Mock()
        fetcher._fallback_fetch_quotes_nse = Mock(return_value={"AAA": "q", "BBB": "q"})

        assert fetcher.fetch_all_quotes() == {"AAA": "q", "BBB": "q"}
        fetcher._fallback_fetch_quotes_nse.assert_called_once_with(symbols)

    def test_partial_nse_fallback_raises(self, fetcher):
        """NSE fallback below 70% coverage is an error, not a small universe."""
        symbols = [f"S{i}" for i in range(10)]
        fetcher._fo_universe.get_symbols.return_value = symbols
        fetcher._smart_api.get_bulk_quotes.return_value = None
        fetcher._nse_scraper = Mock()
        fetcher._fallback_fetch_quotes_nse = Mock(
            return_value={s: "q" for s in symbols[:6]}
        )

        with pytest.raises(DataFetchError):
            fetcher.fetch_all_quotes()

    def test_explicit_symbols_skip_universe(self, fetcher):
        """A symbol subset is quoted instead of the whole universe."""
        fetcher._token_map = {"AAA": "1"}
        fetcher._reverse_token_map = {"1": "AAA"}
        fetcher._master_loaded_date = datetime.now(tz=_IST).date()
        fetcher._smart_api.get_bulk_quotes.return_value = {
            "fetched": [{"symbolToken": "1", "ltp": 100.0, "close": 99.0, "volume": 10}],
            "unfetched": [],
        }

        quotes = fetcher.fetch_all_quotes(["AAA"])

        assert list(quotes) == ["AAA"]
        fetcher._fo_universe.get_symbols.assert_not_called()

    def test_per_symbol_fallback_stops_after_consecutive_failures(self, fetcher):
        """Five failed NSE requests in a row end the per-symbol loop."""
        fetcher._nse_scraper = Mock()
        fetcher._nse_scraper._call.return_value = None

        quotes = fetcher._fetch_quotes_nse_per_symbol([f"S{i}" for i in range(50)])

        assert quotes == {}
        assert fetcher._nse_scraper._call.call_count == 5

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
        """OI change is NSE's day change on every fetch; IV change is vs history."""
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

        # Totals moved +50k / -20k vs the previous snapshot, but NSE's day
        # change says +1200 / -300: the day change is what is reported.
        fetcher._nse_scraper = Mock()
        fetcher._nse_scraper.get_option_chain_meta.return_value = {
            "expiryDates": ["18-Jan-2024"],
        }
        fetcher._nse_scraper.get_option_chain_v3.return_value = _nse_chain(
            550000, 580000, 18.5, 18.7, ce_chg=1200, pe_chg=-300,
        )

        oi_data = fetcher.fetch_oi_for_symbols(["RELIANCE"])

        assert "RELIANCE" in oi_data
        oi = oi_data["RELIANCE"]
        assert oi.ce_oi_change == 1200
        assert oi.pe_oi_change == -300
        assert abs(oi.iv_change - 0.6) < 0.01  # (18.6 - 18.0) = 0.6
        fetcher._nse_scraper.get_option_chain_v3.assert_called_once_with(
            "RELIANCE", type="Equities", expiry="18-Jan-2024",
        )

    def test_oi_without_nse_does_not_call_smartapi(self, fetcher):
        """SmartAPI optionGreek has no OI, so it is never used as a source."""
        assert fetcher.fetch_oi_for_symbols(["RELIANCE"]) == {}
        fetcher._smart_api.get_option_chain.assert_not_called()

    def test_stock_expiry_resolved_once_per_day(self, fetcher):
        """Expiry meta is fetched once and reused by later OI runs."""
        fetcher._nse_scraper = Mock()
        fetcher._nse_scraper.get_option_chain_meta.return_value = {
            "expiryDates": ["18-Jan-2024"],
        }
        fetcher._nse_scraper.get_option_chain_v3.return_value = None

        fetcher.fetch_oi_for_symbols(["RELIANCE"])
        fetcher.fetch_oi_for_symbols(["INFY"])

        assert fetcher._nse_scraper.get_option_chain_meta.call_count == 1

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
