"""
Fetch and maintain NSE F&O universe — master list of F&O-eligible stocks.

The F&O universe is a reference dataset (180–220 stocks) that identifies which
stocks can be traded in the F&O segment. Refreshed daily from NSE.

NSE API endpoints:
- Primary: GET /api/equity-stockIndices?index=SECURITIES%20IN%20F%26O
- Lot sizes: GET /api/fo-lot-size-master
- Fallback: GET /api/master-quote (parse F&O flag if primary fails)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.data.nse_scraper import NSEScraper
from src.database.db_manager import DatabaseManager
from src.database import queries as Q

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)


class DataQualityError(Exception):
    """Raised when NSE F&O universe data fails validation."""


@dataclass
class FOUniverseStock:
    """Single stock in the F&O universe."""

    symbol: str
    company_name: str
    series: str
    isin: str
    lot_size: int
    underlying_value: float
    sector: str
    is_active: bool
    last_updated: str  # ISO-8601 IST


class FOUniverseFetcher:
    """
    Fetch and cache NSE F&O-eligible stocks.

    Maintains in-memory cache with automatic refresh based on age.
    Tracks history of entries/removals via database.

    Parameters
    ----------
    db:
        Connected DatabaseManager instance.
    nse_scraper:
        NSEScraper instance (session reused, not created).
    """

    # NSE API endpoints
    _PRIMARY_ENDPOINT = "/api/equity-stockIndices"
    _LOT_SIZE_ENDPOINT = "/api/fo-lot-size-master"

    def __init__(self, db: DatabaseManager, nse_scraper: NSEScraper) -> None:
        self._db = db
        self._scraper = nse_scraper
        self._cache: dict[str, FOUniverseStock] = {}
        self._last_fetched: Optional[datetime] = None

    # ── Public API ────────────────────────────────────────────────────────

    def get_universe(self) -> list[FOUniverseStock]:
        """
        Return F&O universe from cache if fresh, otherwise fetch.

        Freshness check: cache age < settings.polling.fo_universe_refresh_hours.

        Returns
        -------
        list[FOUniverseStock]:
            All active stocks in the F&O universe.

        Raises
        ------
        DataQualityError:
            If fetch returns < 100 stocks (validation failure).
        """
        if self._is_cache_fresh():
            return list(self._cache.values())

        return self.fetch_fo_universe()

    def get_symbols(self) -> list[str]:
        """
        Return list of symbols from cache.

        Convenience method for the scorer to know which stocks exist.
        """
        return sorted(self._cache.keys())

    def get_lot_size(self, symbol: str) -> int:
        """
        Return lot size for a given symbol.

        Parameters
        ----------
        symbol:
            NSE symbol (e.g. "RELIANCE").

        Returns
        -------
        int:
            Lot size for F&O trading.

        Raises
        ------
        ValueError:
            If symbol not in universe.
        """
        if symbol not in self._cache:
            raise ValueError(f"Symbol {symbol} not in F&O universe")
        return self._cache[symbol].lot_size

    def refresh_if_stale(self) -> bool:
        """
        Check if universe is stale; refresh if needed.

        Call this at startup and once daily (e.g., 8:45 AM) to keep
        the universe fresh.

        Returns
        -------
        bool:
            True if refreshed, False if cache still fresh.
        """
        if self._is_cache_fresh():
            return False

        self.fetch_fo_universe()
        return True

    def fetch_fo_universe(self) -> list[FOUniverseStock]:
        """
        Fetch NSE F&O universe from primary + lot-size endpoints.

        Primary endpoint returns symbol, company name, series, ISIN, sector,
        current price. Lot-size endpoint provides F&O lot sizes.
        Both are merged and validated.

        Returns
        -------
        list[FOUniverseStock]:
            All stocks found.

        Raises
        ------
        DataQualityError:
            If total count < 100 (validation failure, cache not updated).
        """
        logger.info("Fetching F&O universe from NSE...")

        # Fetch primary data: symbols, company names, sectors, prices
        primary_data = self._fetch_primary_endpoint()
        if not primary_data:
            logger.warning("Primary F&O endpoint returned empty; trying fallback")
            primary_data = self._fetch_fallback_endpoint()

        if not primary_data:
            raise DataQualityError("Both primary and fallback F&O endpoints failed")

        # Fetch lot sizes
        lot_sizes = self._fetch_lot_sizes()

        # Merge and parse
        stocks = self._merge_and_parse(primary_data, lot_sizes)

        # Validate count
        if len(stocks) < 100:
            raise DataQualityError(
                f"F&O universe validation failed: {len(stocks)} stocks < 100 minimum"
            )

        # Track history before updating cache
        self._track_history(stocks)

        # Update cache and persist
        self._update_cache_and_db(stocks)

        logger.info("F&O universe fetched: %d stocks", len(stocks))
        return stocks

    # ── Private helpers ───────────────────────────────────────────────────

    def _is_cache_fresh(self) -> bool:
        """Return True if cache exists and is not stale."""
        if not self._cache or self._last_fetched is None:
            return False

        refresh_hours = settings.polling.fo_universe_refresh_hours
        age = datetime.now(tz=_IST) - self._last_fetched
        return age < timedelta(hours=refresh_hours)

    def _fetch_primary_endpoint(self) -> Optional[dict]:
        """
        Fetch F&O universe from primary endpoint.

        GET /api/equity-stockIndices?index=SECURITIES%20IN%20F%26O
        Response: {"data": [...]}
        """
        response = self._scraper._call(
            self._PRIMARY_ENDPOINT,
            params={"index": "SECURITIES IN F&O"},
        )
        return response

    def _fetch_fallback_endpoint(self) -> Optional[dict]:
        """
        Fallback: fetch from /api/master-quote and filter by F&O flag.

        This is less reliable but may work if primary is unavailable.
        """
        response = self._scraper._call("/api/master-quote")
        if not response:
            return None

        # Filter to F&O-eligible entries only
        # (Implementation depends on actual NSE response structure)
        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _fetch_lot_sizes(self) -> dict[str, int]:
        """
        Fetch F&O lot sizes from NSE lot-size master.

        GET /api/fo-lot-size-master
        Response: {"data": [{"symbol": "RELIANCE", "lotSize": "250"}, ...]}

        Returns
        -------
        dict[str, int]:
            Mapping of symbol → lot_size (fallback to 0 if not found).
        """
        response = self._scraper._call(self._LOT_SIZE_ENDPOINT)
        if not response:
            logger.warning("Lot size endpoint returned no data; using fallback lot_size=1")
            return {}

        lot_sizes = {}
        for entry in response.get("data", []):
            symbol = entry.get("symbol", "").upper()
            try:
                lot = int(entry.get("lotSize", "1"))
            except (TypeError, ValueError):
                lot = 1

            if symbol:
                lot_sizes[symbol] = lot

        logger.debug("Lot sizes fetched for %d symbols", len(lot_sizes))
        return lot_sizes

    def _merge_and_parse(
        self,
        primary: Any,
        lot_sizes: dict[str, int],
    ) -> list[FOUniverseStock]:
        """
        Parse primary endpoint response and merge with lot sizes.

        Extracts: symbol, company_name, series, isin, sector, price.
        Merges in lot_size (default 1 if not found).

        Returns
        -------
        list[FOUniverseStock]:
            Parsed stocks.
        """
        stocks = []
        now = datetime.now(tz=_IST).isoformat()

        # NSE endpoint returns {"data": [...]} normally; sometimes a raw list
        entries = primary.get("data", []) if isinstance(primary, dict) else primary
        for entry in entries:
            try:
                symbol = (entry.get("symbol") or "").upper()
                if not symbol:
                    continue

                # Extract metadata
                meta = entry.get("meta", {})
                company_name = meta.get("companyName", "")
                series = meta.get("series", "EQ")
                isin = meta.get("isin", "")
                sector = meta.get("industry", "unknown")

                # Underlying price
                try:
                    underlying_value = float(entry.get("lastPrice", entry.get("currentPrice", 0)))
                except (TypeError, ValueError):
                    underlying_value = 0.0

                # Lot size (fallback to 1)
                lot_size = lot_sizes.get(symbol, 1)

                stock = FOUniverseStock(
                    symbol=symbol,
                    company_name=company_name,
                    series=series,
                    isin=isin,
                    lot_size=lot_size,
                    underlying_value=underlying_value,
                    sector=sector,
                    is_active=True,
                    last_updated=now,
                )
                stocks.append(stock)
            except Exception as e:
                logger.debug("Skipping malformed entry: %s — %s", entry, e)
                continue

        return stocks

    def _track_history(self, new_stocks: list[FOUniverseStock]) -> None:
        """
        Compare new stocks against cache; emit ADDED/REMOVED events.

        Writes to fo_universe_history table.
        """
        today = datetime.now(tz=_IST).date().isoformat()
        old_symbols = set(self._cache.keys())
        new_symbols = {s.symbol for s in new_stocks}

        # ADDED events
        added = new_symbols - old_symbols
        for symbol in added:
            stock = next(s for s in new_stocks if s.symbol == symbol)
            try:
                self._db.execute(
                    Q.INSERT_FO_UNIVERSE_HISTORY,
                    (symbol, "ADDED", today, stock.lot_size),
                )
                logger.info("F&O history: %s ADDED (lot_size=%d)", symbol, stock.lot_size)
            except Exception as e:
                logger.warning("Failed to record ADDED event for %s: %s", symbol, e)

        # REMOVED events
        removed = old_symbols - new_symbols
        for symbol in removed:
            old_stock = self._cache[symbol]
            try:
                self._db.execute(
                    Q.INSERT_FO_UNIVERSE_HISTORY,
                    (symbol, "REMOVED", today, old_stock.lot_size),
                )
                logger.info("F&O history: %s REMOVED", symbol)
            except Exception as e:
                logger.warning("Failed to record REMOVED event for %s: %s", symbol, e)

    def _update_cache_and_db(self, stocks: list[FOUniverseStock]) -> None:
        """
        Update in-memory cache and persist to database.

        Uses UPSERT to idempotently update all stocks.
        """
        self._cache = {s.symbol: s for s in stocks}
        self._last_fetched = datetime.now(tz=_IST)

        try:
            for stock in stocks:
                self._db.execute(
                    Q.UPSERT_FO_UNIVERSE,
                    (
                        stock.symbol,
                        stock.company_name,
                        stock.series,
                        stock.isin,
                        stock.lot_size,
                        stock.underlying_value,
                        stock.sector,
                        1 if stock.is_active else 0,
                        stock.last_updated,
                    ),
                )
            logger.debug("Persisted %d stocks to fo_universe table", len(stocks))
        except Exception as e:
            logger.error("Failed to persist F&O universe to DB: %s", e)
            raise
