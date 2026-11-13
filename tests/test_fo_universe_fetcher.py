"""Unit tests for FOUniverseFetcher."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, call
from zoneinfo import ZoneInfo

import pytest

from src.data.fo_universe_fetcher import FOUniverseFetcher, FOUniverseStock, DataQualityError
from config.constants import IST_TIMEZONE

_IST = ZoneInfo(IST_TIMEZONE)


@pytest.fixture()
def mock_db() -> MagicMock:
    """Mock DatabaseManager."""
    db = MagicMock()
    db.execute.return_value = MagicMock()
    return db


@pytest.fixture()
def mock_scraper() -> MagicMock:
    """Mock NSEScraper."""
    scraper = MagicMock()
    scraper._call.return_value = None
    return scraper


@pytest.fixture()
def fetcher(mock_db: MagicMock, mock_scraper: MagicMock) -> FOUniverseFetcher:
    """Create FOUniverseFetcher with mocked dependencies."""
    return FOUniverseFetcher(db=mock_db, nse_scraper=mock_scraper)


def test_fetch_returns_stocks(fetcher: FOUniverseFetcher, mock_scraper: MagicMock) -> None:
    """fetch_fo_universe parses and returns correct stock count."""
    # Mock primary endpoint response — need >=100 stocks to pass validation
    primary_data = [
        {
            "symbol": "RELIANCE",
            "meta": {
                "companyName": "Reliance Industries",
                "series": "EQ",
                "isin": "INE002A01018",
                "industry": "Petrochemicals",
            },
            "lastPrice": 2500.0,
        },
        {
            "symbol": "TCS",
            "meta": {
                "companyName": "Tata Consultancy Services",
                "series": "EQ",
                "isin": "INE467B01029",
                "industry": "IT",
            },
            "currentPrice": 3600.0,
        },
    ]
    # Add 98 dummy stocks to reach 100
    for i in range(98):
        primary_data.append({
            "symbol": f"FILLER{i}",
            "meta": {
                "companyName": f"Filler Company {i}",
                "series": "EQ",
                "isin": f"INE{i:06d}",
                "industry": "Tech",
            },
            "lastPrice": 100.0,
        })

    primary_response = {"data": primary_data}

    # Mock lot-size endpoint response
    lot_size_response = {
        "data": [
            {"symbol": "RELIANCE", "lotSize": "250"},
            {"symbol": "TCS", "lotSize": "1"},
        ]
    }

    mock_scraper._call.side_effect = [primary_response, lot_size_response]

    stocks = fetcher.fetch_fo_universe()

    assert len(stocks) >= 100
    reliance = next(s for s in stocks if s.symbol == "RELIANCE")
    assert reliance.company_name == "Reliance Industries"
    assert reliance.lot_size == 250
    assert reliance.underlying_value == 2500.0

    tcs = next(s for s in stocks if s.symbol == "TCS")
    assert tcs.lot_size == 1


def test_raises_on_partial_data(fetcher: FOUniverseFetcher, mock_scraper: MagicMock) -> None:
    """DataQualityError raised if count < 100; cache NOT updated."""
    # Return only 50 stocks (below 100 threshold)
    primary_response = {
        "data": [
            {
                "symbol": f"STOCK{i}",
                "meta": {
                    "companyName": f"Company {i}",
                    "series": "EQ",
                    "isin": f"INE{i:06d}",
                    "industry": "Tech",
                },
                "lastPrice": 100.0,
            }
            for i in range(50)
        ]
    }

    lot_size_response = {"data": []}
    mock_scraper._call.side_effect = [primary_response, lot_size_response]

    with pytest.raises(DataQualityError, match="validation failed"):
        fetcher.fetch_fo_universe()

    # Cache should NOT be updated
    assert len(fetcher._cache) == 0


def test_cache_returns_without_fetch(
    fetcher: FOUniverseFetcher, mock_scraper: MagicMock
) -> None:
    """Fresh cache returns without calling _call."""
    # Populate cache with recent data
    now = datetime.now(tz=_IST)
    stock = FOUniverseStock(
        symbol="RELIANCE",
        company_name="Reliance",
        series="EQ",
        isin="INE002A01018",
        lot_size=250,
        underlying_value=2500.0,
        sector="Petrochemicals",
        is_active=True,
        last_updated=now.isoformat(),
    )
    fetcher._cache = {"RELIANCE": stock}
    fetcher._last_fetched = now

    # Call get_universe
    result = fetcher.get_universe()

    # Should return cache without calling scraper
    assert len(result) == 1
    assert result[0].symbol == "RELIANCE"
    mock_scraper._call.assert_not_called()


def test_stale_cache_triggers_fetch(
    fetcher: FOUniverseFetcher, mock_scraper: MagicMock
) -> None:
    """Stale cache (age > refresh_hours) triggers new fetch."""
    # Populate cache with OLD data
    old_time = datetime.now(tz=_IST) - timedelta(hours=25)
    old_stock = FOUniverseStock(
        symbol="OLD_STOCK",
        company_name="Old",
        series="EQ",
        isin="INE000A01000",
        lot_size=1,
        underlying_value=100.0,
        sector="Old",
        is_active=True,
        last_updated=old_time.isoformat(),
    )
    fetcher._cache = {"OLD_STOCK": old_stock}
    fetcher._last_fetched = old_time

    # Mock fresh response
    primary_response = {
        "data": [
            {
                "symbol": f"NEW{i}",
                "meta": {
                    "companyName": f"New {i}",
                    "series": "EQ",
                    "isin": f"INE{i:06d}",
                    "industry": "Tech",
                },
                "lastPrice": 100.0,
            }
            for i in range(150)
        ]
    }
    lot_size_response = {"data": []}
    mock_scraper._call.side_effect = [primary_response, lot_size_response]

    # Call get_universe
    result = fetcher.get_universe()

    # Should have called scraper to refresh
    assert mock_scraper._call.call_count >= 2
    # Cache should have new stocks
    assert "NEW0" in fetcher._cache or len(fetcher._cache) > 1


def test_lot_size_lookup_happy_path(fetcher: FOUniverseFetcher, mock_scraper: MagicMock) -> None:
    """get_lot_size returns correct lot size for known symbol."""
    # Seed cache
    stock = FOUniverseStock(
        symbol="RELIANCE",
        company_name="Reliance",
        series="EQ",
        isin="INE002A01018",
        lot_size=250,
        underlying_value=2500.0,
        sector="Petrochemicals",
        is_active=True,
        last_updated=datetime.now(tz=_IST).isoformat(),
    )
    fetcher._cache = {"RELIANCE": stock}

    assert fetcher.get_lot_size("RELIANCE") == 250


def test_lot_size_lookup_unknown_symbol(fetcher: FOUniverseFetcher) -> None:
    """get_lot_size raises ValueError for unknown symbol."""
    fetcher._cache = {}

    with pytest.raises(ValueError, match="not in F&O universe"):
        fetcher.get_lot_size("UNKNOWN")


def test_history_events_added_removed(
    fetcher: FOUniverseFetcher, mock_db: MagicMock, mock_scraper: MagicMock
) -> None:
    """Second fetch with one added and one removed stock emits history events."""
    # Seed cache with one stock
    old_stock = FOUniverseStock(
        symbol="OLDSTOCK",
        company_name="Old",
        series="EQ",
        isin="INE000A00000",
        lot_size=1,
        underlying_value=100.0,
        sector="Old",
        is_active=True,
        last_updated=datetime.now(tz=_IST).isoformat(),
    )
    fetcher._cache = {"OLDSTOCK": old_stock}

    # Mock new fetch: OLDSTOCK removed, NEWSTOCK added, plus 150 others
    primary_response = {
        "data": [
            {
                "symbol": "NEWSTOCK",
                "meta": {
                    "companyName": "New Company",
                    "series": "EQ",
                    "isin": "INE001A00001",
                    "industry": "Tech",
                },
                "lastPrice": 500.0,
            },
        ]
        + [
            {
                "symbol": f"FILLER{i}",
                "meta": {
                    "companyName": f"Filler {i}",
                    "series": "EQ",
                    "isin": f"INE{i:06d}",
                    "industry": "Tech",
                },
                "lastPrice": 100.0,
            }
            for i in range(150)
        ]
    }
    lot_size_response = {"data": []}
    mock_scraper._call.side_effect = [primary_response, lot_size_response]

    fetcher.fetch_fo_universe()

    # Check that INSERT_FO_UNIVERSE_HISTORY was called for ADDED and REMOVED
    history_calls = [c for c in mock_db.execute.call_args_list
                     if "fo_universe_history" in str(c).lower()]
    assert len(history_calls) >= 2  # At least ADDED and REMOVED


def test_refresh_if_stale_returns_false_when_fresh(
    fetcher: FOUniverseFetcher, mock_scraper: MagicMock
) -> None:
    """refresh_if_stale returns False if cache is fresh."""
    # Populate fresh cache
    now = datetime.now(tz=_IST)
    stock = FOUniverseStock(
        symbol="RELIANCE",
        company_name="Reliance",
        series="EQ",
        isin="INE002A01018",
        lot_size=250,
        underlying_value=2500.0,
        sector="Petrochemicals",
        is_active=True,
        last_updated=now.isoformat(),
    )
    fetcher._cache = {"RELIANCE": stock}
    fetcher._last_fetched = now

    # Call refresh_if_stale
    result = fetcher.refresh_if_stale()

    # Should return False and not call scraper
    assert result is False
    mock_scraper._call.assert_not_called()


def test_refresh_if_stale_returns_true_when_stale(
    fetcher: FOUniverseFetcher, mock_scraper: MagicMock
) -> None:
    """refresh_if_stale returns True if cache was refreshed."""
    # Populate old cache
    old_time = datetime.now(tz=_IST) - timedelta(hours=25)
    old_stock = FOUniverseStock(
        symbol="OLDSTOCK",
        company_name="Old",
        series="EQ",
        isin="INE000A00000",
        lot_size=1,
        underlying_value=100.0,
        sector="Old",
        is_active=True,
        last_updated=old_time.isoformat(),
    )
    fetcher._cache = {"OLDSTOCK": old_stock}
    fetcher._last_fetched = old_time

    # Mock fresh response
    primary_response = {
        "data": [
            {
                "symbol": f"NEW{i}",
                "meta": {
                    "companyName": f"New {i}",
                    "series": "EQ",
                    "isin": f"INE{i:06d}",
                    "industry": "Tech",
                },
                "lastPrice": 100.0,
            }
            for i in range(150)
        ]
    }
    lot_size_response = {"data": []}
    mock_scraper._call.side_effect = [primary_response, lot_size_response]

    # Call refresh_if_stale
    result = fetcher.refresh_if_stale()

    # Should return True and call scraper
    assert result is True
    assert mock_scraper._call.called


def test_get_symbols(fetcher: FOUniverseFetcher) -> None:
    """get_symbols returns sorted symbol list."""
    stocks = [
        FOUniverseStock(
            symbol=sym,
            company_name=f"Company {sym}",
            series="EQ",
            isin="INE0000",
            lot_size=1,
            underlying_value=100.0,
            sector="Tech",
            is_active=True,
            last_updated=datetime.now(tz=_IST).isoformat(),
        )
        for sym in ["ZEEL", "RELIANCE", "INFY"]
    ]
    fetcher._cache = {s.symbol: s for s in stocks}

    symbols = fetcher.get_symbols()

    assert symbols == ["INFY", "RELIANCE", "ZEEL"]  # Sorted


def test_empty_lot_size_response_defaults_to_one(
    fetcher: FOUniverseFetcher, mock_scraper: MagicMock
) -> None:
    """Missing lot size defaults to 1."""
    primary_data = [
        {
            "symbol": "RELIANCE",
            "meta": {
                "companyName": "Reliance",
                "series": "EQ",
                "isin": "INE002A01018",
                "industry": "Petrochemicals",
            },
            "lastPrice": 2500.0,
        }
    ]
    # Add 99 dummy stocks to reach 100
    for i in range(99):
        primary_data.append({
            "symbol": f"FILLER{i}",
            "meta": {
                "companyName": f"Filler {i}",
                "series": "EQ",
                "isin": f"INE{i:06d}",
                "industry": "Tech",
            },
            "lastPrice": 100.0,
        })

    primary_response = {"data": primary_data}
    lot_size_response = {"data": []}  # Empty lot sizes

    mock_scraper._call.side_effect = [primary_response, lot_size_response]

    stocks = fetcher.fetch_fo_universe()

    # Find RELIANCE and check its lot size defaults to 1
    reliance = next(s for s in stocks if s.symbol == "RELIANCE")
    assert reliance.lot_size == 1


# ── Fallback sources / metadata preservation (NSE endpoints 404 since 2026) ──


@pytest.fixture(autouse=True)
def _no_angel_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never hit the real Angel scrip master from unit tests."""
    monkeypatch.setattr(
        FOUniverseFetcher, "_fetch_lot_sizes_angel", staticmethod(lambda: {})
    )


def _underlying_info(n: int) -> dict:
    return {"data": {"UnderlyingList": [
        {"symbol": f"S{i}", "underlying": f"Stock {i} Ltd"} for i in range(n)
    ]}}


def test_fallback_uses_underlying_information(
    fetcher: FOUniverseFetcher, mock_scraper: MagicMock
) -> None:
    """Primary 404 -> /api/underlying-information supplies symbols + names."""
    mock_scraper._call.side_effect = lambda ep, **kw: (
        _underlying_info(120) if ep == "/api/underlying-information" else None
    )
    stocks = fetcher.fetch_fo_universe()

    assert len(stocks) == 120
    assert stocks[0].company_name == "Stock 0 Ltd"


def test_fallback_master_quote_symbol_list(
    fetcher: FOUniverseFetcher, mock_scraper: MagicMock
) -> None:
    """/api/master-quote returns a bare list of symbols."""
    mock_scraper._call.side_effect = lambda ep, **kw: (
        [f"S{i}" for i in range(110)] if ep == "/api/master-quote" else None
    )
    assert len(fetcher.fetch_fo_universe()) == 110


def test_fill_metadata_keeps_known_values(
    fetcher: FOUniverseFetcher, mock_db: MagicMock, mock_scraper: MagicMock
) -> None:
    """Blank sector/ISIN/lot from a fallback source never replace DB values."""
    mock_db.fetch_all.return_value = [{
        "symbol": "RELIANCE", "company_name": "Reliance Industries Ltd",
        "isin": "INE002A01018", "sector": "Refineries & Marketing", "lot_size": 500,
    }]
    st = FOUniverseStock(
        symbol="RELIANCE", company_name="", series="EQ", isin="", lot_size=1,
        underlying_value=0.0, sector="unknown", is_active=True, last_updated="x",
    )
    out = fetcher._fill_metadata([st])[0]

    assert out.sector == "Refineries & Marketing"
    assert out.isin == "INE002A01018"
    assert out.company_name == "Reliance Industries Ltd"
    assert out.lot_size == 500
    mock_scraper._call.assert_not_called()  # nothing left to look up


def test_fill_metadata_looks_up_unknown_sector(
    fetcher: FOUniverseFetcher, mock_db: MagicMock, mock_scraper: MagicMock
) -> None:
    """New symbols with no DB row get sector from NSE GetQuoteApi."""
    mock_db.fetch_all.return_value = []
    mock_scraper._call.return_value = {"equityResponse": [{
        "metaData": {"companyName": "Radico Khaitan Limited", "isinCode": "INE944F01028"},
        "secInfo": {"basicIndustry": "Breweries & Distilleries"},
    }]}
    st = FOUniverseStock(
        symbol="RADICO", company_name="", series="EQ", isin="", lot_size=325,
        underlying_value=0.0, sector="unknown", is_active=True, last_updated="x",
    )
    out = fetcher._fill_metadata([st])[0]

    assert out.sector == "Breweries & Distilleries"
    assert out.isin == "INE944F01028"


def test_history_fresh_process_compares_against_db(
    fetcher: FOUniverseFetcher, mock_db: MagicMock
) -> None:
    """Empty cache: diff vs persisted universe, not 'everything ADDED'."""
    mock_db.fetch_all.return_value = [
        {"symbol": "A", "lot_size": 10}, {"symbol": "OLD", "lot_size": 20},
    ]
    mk = lambda s: FOUniverseStock(  # noqa: E731
        symbol=s, company_name="", series="EQ", isin="", lot_size=10,
        underlying_value=0.0, sector="unknown", is_active=True, last_updated="x",
    )
    fetcher._track_history([mk("A"), mk("NEW")])

    events = [c.args[1][:2] for c in mock_db.execute.call_args_list
              if "fo_universe_history" in c.args[0]]
    assert sorted(events) == [("NEW", "ADDED"), ("OLD", "REMOVED")]
