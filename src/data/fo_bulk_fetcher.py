"""
Bulk F&O data fetcher — market data + OI snapshots for all F&O stocks efficiently.

Fetches live quotes for ~200 F&O stocks (batched), calculates liquidity metrics,
then fetches options chain for top 60 candidates. Respects SmartAPI rate limits
(~3 calls/second on free tier) via staggered batches.

Design:
- fetch_all_quotes() — 4 batches of 50 symbols each (1 bulk call per batch)
- fetch_avg_volumes() — 20-day average volume from historical data
- fetch_oi_for_symbols() — options chain for filtered subset only
- fetch_full_snapshot() — orchestration: quotes → liquidity filter → OI
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, date
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.data.fo_universe_fetcher import FOUniverseFetcher
from src.data.smart_api_client import SmartAPIClient
from src.database.db_manager import DatabaseManager
from src.database import queries as Q

if TYPE_CHECKING:
    from src.data.nse_scraper import NSEScraper

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)


class DataFetchError(Exception):
    """Raised when bulk data fetch fails systemically (>30% symbols lost)."""


@dataclass
class StockQuote:
    """Single stock quote snapshot — not persistent."""

    symbol: str
    ltp: float                        # Last traded price
    open: float
    high: float
    low: float
    close: float                      # Previous day close
    volume: int                       # Today's volume so far
    avg_volume_20d: float             # 20-day average volume
    price_change_pct: float           # % change from prev close
    bid: float                        # Best bid
    ask: float                        # Best ask
    bid_ask_spread: float             # ask - bid (calculated)
    bid_ask_spread_pct: float         # spread as % of LTP (calculated)
    timestamp: datetime


@dataclass
class StockOIData:
    """Options chain OI snapshot — in-memory with change tracking."""

    symbol: str
    expiry: date                      # Nearest expiry
    total_ce_oi: int
    total_pe_oi: int
    ce_oi_change: int                 # Change from previous snapshot
    pe_oi_change: int
    pcr: float                        # PE OI / CE OI
    atm_strike: float
    atm_ce_iv: float                  # ATM call IV
    atm_pe_iv: float                  # ATM put IV
    avg_iv: float                     # (atm_ce_iv + atm_pe_iv) / 2
    iv_change: float                  # Change in avg IV from previous snapshot
    timestamp: datetime


class FOBulkFetcher:
    """
    Fetch bulk market data for F&O universe efficiently.

    Coordinates quotes, volumes, and OI snapshots with rate limiting.
    Caches 20-day volumes in DB; keeps OI history in-memory with daily reset.
    """

    def __init__(
        self,
        db: DatabaseManager,
        smart_api: SmartAPIClient,
        fo_universe: FOUniverseFetcher,
        nse_scraper: Optional["NSEScraper"] = None,
    ) -> None:
        """
        Initialize bulk fetcher.

        Parameters
        ----------
        db:
            Database connection for caching volumes
        smart_api:
            SmartAPI client (Angel One authenticated)
        fo_universe:
            FOUniverseFetcher for symbol list + lot sizes
        nse_scraper:
            Optional NSEScraper for fallback when SmartAPI unavailable
        """
        self._db = db
        self._smart_api = smart_api
        self._fo_universe = fo_universe
        self._nse_scraper = nse_scraper

        # Load settings
        self._batch_size: int = getattr(settings.broker, 'bulk_fetch_batch_size', 50)
        self._batch_delay: float = getattr(settings.broker, 'bulk_fetch_delay_seconds', 0.4)
        self._oi_fetch_top_n: int = getattr(settings.broker, 'oi_fetch_top_n', 60)

        # In-memory OI history: symbol → StockOIData
        self._oi_history: dict[str, StockOIData] = {}
        self._oi_history_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_all_quotes(self) -> dict[str, StockQuote]:
        """
        Fetch live quotes for all ~200 F&O stocks.

        Batches into groups of bulk_fetch_batch_size (default 50).
        Applies bulk_fetch_delay_seconds between batches to respect rate limits.

        Returns
        -------
        dict[str, StockQuote]:
            Map of symbol → StockQuote for all successfully fetched stocks.

        Raises
        ------
        DataFetchError:
            If > 30% of symbols fail (systemic issue).
        """
        symbols = self._fo_universe.get_symbols()
        logger.info("DEBUG: fetch_all_quotes got %d symbols from universe", len(symbols))
        if not symbols:
            logger.error("F&O universe empty — cannot fetch quotes")
            raise DataFetchError("F&O universe is empty")

        logger.info("Fetching quotes for %d symbols in batches of %d", len(symbols), self._batch_size)
        start = time.time()

        # Try both NSE and EQUITY exchange codes
        exchange_code = "NSE"

        quotes: dict[str, StockQuote] = {}
        failed: list[str] = []

        # Split into batches
        for batch_idx in range(0, len(symbols), self._batch_size):
            batch = symbols[batch_idx : batch_idx + self._batch_size]

            # Use symbol names directly as exchange_tokens (SmartAPI may accept this)
            exchange_tokens = {"NSE": batch}
            logger.info("DEBUG: batch %d symbols: %s (first 3: %s)",
                       batch_idx // self._batch_size + 1, len(batch), batch[:3] if batch else [])

            try:
                logger.debug("Fetching batch %d", batch_idx // self._batch_size + 1)
                # Try different modes: LTP (simplest) -> QUOTE -> SNAP_QUOTE
                raw = self._smart_api.get_bulk_quotes("LTP", exchange_tokens)
                if not raw:
                    logger.debug("LTP returned empty, trying QUOTE mode")
                    raw = self._smart_api.get_bulk_quotes("QUOTE", exchange_tokens)
                if not raw:
                    logger.debug("QUOTE returned empty, trying SNAP_QUOTE mode")
                    raw = self._smart_api.get_bulk_quotes("SNAP_QUOTE", exchange_tokens)
                logger.info("DEBUG: batch raw response: fetched=%d, unfetched=%d, raw_keys=%s",
                            len(raw.get("fetched", [])) if raw else 0,
                            len(raw.get("unfetched", [])) if raw else 0,
                            list(raw.keys()) if raw else None)

                if raw:
                    for entry in raw.get("fetched", []):
                        try:
                            quote = self._parse_quote(entry)
                            if quote:
                                quotes[quote.symbol] = quote
                        except Exception as e:
                            logger.warning("Failed to parse quote for entry: %s", e)
                            failed.append(entry.get("symbol", "unknown"))

                    # Track errors from API
                    for entry in raw.get("unfetched", []):
                        failed.append(entry.get("symbol", "unknown"))

                # Rate limit: sleep between batches
                if batch_idx + self._batch_size < len(symbols):
                    time.sleep(self._batch_delay)

            except Exception as e:
                logger.error("Batch fetch failed: %s", e)
                failed.extend(batch)

        # Validate: fail if >30% lost
        fail_rate = len(failed) / len(symbols) if symbols else 0
        if fail_rate > 0.3:
            # Fallback to NSE scraper if SmartAPI failed
            if self._nse_scraper and len(quotes) == 0:
                logger.warning(
                    "SmartAPI quote fetch failed (%d%%). Falling back to NSE scraper.",
                    int(fail_rate * 100)
                )
                return self._fallback_fetch_quotes_nse(symbols)
            raise DataFetchError(f"Quote fetch failure rate {fail_rate:.1%} exceeds 30% threshold")

        elapsed = time.time() - start
        logger.info(
            "Quotes fetched: %d/%d in %.1fs (failed: %d, %.1f%%)",
            len(quotes),
            len(symbols),
            elapsed,
            len(failed),
            fail_rate * 100,
        )

        return quotes

    def _fallback_fetch_quotes_nse(self, symbols: list[str]) -> dict[str, StockQuote]:
        """Fallback: fetch quotes from NSE scraper when SmartAPI unavailable."""
        logger.info("Fetching %d symbols from NSE scraper (fallback)", len(symbols))
        start = time.time()

        try:
            # Get all indices from scraper
            indices = self._nse_scraper.get_all_indices()
            quotes: dict[str, StockQuote] = {}

            for idx_data in indices:
                symbol = idx_data.get("symbol", "")
                if symbol not in symbols:
                    continue

                try:
                    quote = StockQuote(
                        symbol=symbol,
                        ltp=float(idx_data.get("ltp", 0)),
                        open=float(idx_data.get("open", 0)),
                        high=float(idx_data.get("high", 0)),
                        low=float(idx_data.get("low", 0)),
                        close=float(idx_data.get("prev_close", 0)),
                        volume=int(idx_data.get("volume", 0)),
                        avg_volume_20d=0.0,  # Will be fetched separately
                        price_change_pct=float(idx_data.get("price_change_pct", 0)),
                        bid=float(idx_data.get("bid", 0)),
                        ask=float(idx_data.get("ask", 0)),
                        bid_ask_spread=float(idx_data.get("ask", 0)) - float(idx_data.get("bid", 0)),
                        bid_ask_spread_pct=(
                            (float(idx_data.get("ask", 0)) - float(idx_data.get("bid", 0)))
                            / float(idx_data.get("ltp", 1)) * 100
                            if float(idx_data.get("ltp", 0)) > 0
                            else 0
                        ),
                        timestamp=datetime.now(tz=_IST),
                    )
                    quotes[symbol] = quote
                except Exception as e:
                    logger.warning("Failed to parse quote for %s: %s", symbol, e)

            elapsed = time.time() - start
            logger.info(
                "NSE scraper fallback: fetched %d/%d symbols in %.1fs",
                len(quotes), len(symbols), elapsed,
            )
            return quotes

        except Exception as e:
            logger.error("NSE scraper fallback failed: %s", e)
            return {}

    def fetch_avg_volumes(self, symbols: list[str]) -> dict[str, float]:
        """
        Fetch 20-day average volume from SmartAPI historical data.

        Caches result in DB (fo_avg_volumes) for 24 hours to avoid intraday
        refetch (volumes don't change within a trading day).

        Parameters
        ----------
        symbols:
            List of NSE symbols (e.g., ["RELIANCE", "INFY"])

        Returns
        -------
        dict[str, float]:
            Map of symbol → 20-day average volume.
        """
        logger.info("Fetching 20-day average volumes for %d symbols", len(symbols))
        start = time.time()

        result: dict[str, float] = {}
        now = datetime.now(tz=_IST)

        for symbol in symbols:
            # Check cache first
            try:
                cached = self._db.fetch_one(Q.GET_FO_AVG_VOLUME, (symbol,))
                if cached:
                    result[symbol] = cached["avg_volume_20d"]
                    continue
            except Exception as e:
                logger.warning("Cache lookup failed for %s: %s", symbol, e)

            # Fetch historical data
            try:
                # TODO: Map symbol → token via master contract
                # For now, placeholder — will need token lookup
                token = self._get_token_for_symbol(symbol, "NSE")
                if not token:
                    logger.warning("No token found for %s", symbol)
                    continue

                bars = self._smart_api.get_historical_ohlcv("NSE", token, interval="day", count=20)
                if bars:
                    volumes = [int(bar.get("volume", 0)) for bar in bars]
                    avg_vol = sum(volumes) / len(volumes) if volumes else 0.0

                    result[symbol] = avg_vol

                    # Cache in DB
                    try:
                        self._db.execute(
                            Q.UPSERT_FO_AVG_VOLUME,
                            (symbol, avg_vol, now.isoformat()),
                        )
                    except Exception as e:
                        logger.warning("Failed to cache avg_volume for %s: %s", symbol, e)

            except Exception as e:
                logger.error("Historical fetch failed for %s: %s", symbol, e)

        elapsed = time.time() - start
        logger.info("Average volumes fetched: %d/%d in %.1fs", len(result), len(symbols), elapsed)

        return result

    def fetch_oi_for_symbols(self, symbols: list[str]) -> dict[str, StockOIData]:
        """
        Fetch options chain data for a filtered subset of symbols.

        Extracts nearest expiry, ATM calls/puts, IV, and OI. Compares against
        previous snapshot to calculate changes (ce_oi_change, pe_oi_change, iv_change).

        Parameters
        ----------
        symbols:
            Filtered list (typically ~60) of high-liquidity candidates.

        Returns
        -------
        dict[str, StockOIData]:
            Map of symbol → StockOIData with OI changes vs previous snapshot.
        """
        logger.info("Fetching OI data for %d symbols", len(symbols))
        start = time.time()

        result: dict[str, StockOIData] = {}

        for idx, symbol in enumerate(symbols):
            try:
                # TODO: Resolve nearest expiry, ATM strike from SmartAPI
                # Placeholder call
                chain = self._smart_api.get_option_chain(symbol, "18JAN2024", "0")
                if not chain:
                    logger.debug("No OI data for %s", symbol)
                    continue

                oi_data = self._parse_oi_data(symbol, chain)
                if oi_data:
                    # Calculate changes vs previous snapshot
                    with self._oi_history_lock:
                        prev = self._oi_history.get(symbol)
                        if prev:
                            oi_data.ce_oi_change = oi_data.total_ce_oi - prev.total_ce_oi
                            oi_data.pe_oi_change = oi_data.total_pe_oi - prev.total_pe_oi
                            oi_data.iv_change = oi_data.avg_iv - prev.avg_iv
                        else:
                            oi_data.ce_oi_change = 0
                            oi_data.pe_oi_change = 0
                            oi_data.iv_change = 0

                        self._oi_history[symbol] = oi_data

                    result[symbol] = oi_data

                # Rate limit between individual calls
                if idx < len(symbols) - 1:
                    time.sleep(self._batch_delay)

            except Exception as e:
                logger.error("OI fetch failed for %s: %s", symbol, e)

        elapsed = time.time() - start
        logger.info("OI data fetched: %d/%d in %.1fs", len(result), len(symbols), elapsed)

        return result

    def fetch_full_snapshot(
        self,
        volume_filter_threshold: float = 100000.0,
        spread_filter_pct: float = 0.5,
    ) -> tuple[dict[str, StockQuote], dict[str, StockOIData]]:
        """
        Orchestration: fetch quotes → apply liquidity filter → fetch OI.

        Step 1: fetch_all_quotes() for all ~200 stocks
        Step 2: Filter by volume > threshold and spread < spread_filter_pct
        Step 3: fetch_oi_for_symbols() for top oi_fetch_top_n only

        Target: complete within ~45 seconds.

        Parameters
        ----------
        volume_filter_threshold:
            Minimum 20-day avg volume to include (default 100,000)
        spread_filter_pct:
            Maximum bid-ask spread % to include (default 0.5%)

        Returns
        -------
        tuple[dict[str, StockQuote], dict[str, StockOIData]]:
            (quotes, oi_data) — both dicts by symbol.

        Raises
        ------
        DataFetchError:
            If quote fetch fails systemically.
        """
        logger.info("Starting full snapshot: quotes → liquidity filter → OI")

        # Step 1: Fetch all quotes
        quotes = self.fetch_all_quotes()

        # Step 2: Fetch volumes and apply liquidity filter
        all_symbols = [q.symbol for q in quotes.values()]
        volumes = self.fetch_avg_volumes(all_symbols)

        # Update quotes with avg_volume_20d
        for symbol, vol in volumes.items():
            if symbol in quotes:
                quotes[symbol].avg_volume_20d = vol

        # Filter candidates
        candidates = []
        for quote in quotes.values():
            if (quote.avg_volume_20d >= volume_filter_threshold and
                quote.bid_ask_spread_pct <= spread_filter_pct):
                candidates.append(quote.symbol)

        logger.info(
            "Liquidity filter: %d/%d quotes passed (volume >= %.0f, spread <= %.2f%%)",
            len(candidates),
            len(quotes),
            volume_filter_threshold,
            spread_filter_pct,
        )

        # Step 3: Fetch OI for top N candidates only
        oi_candidates = candidates[: self._oi_fetch_top_n]
        oi_data = self.fetch_oi_for_symbols(oi_candidates)

        logger.info("Full snapshot complete: %d quotes, %d OI records", len(quotes), len(oi_data))

        return quotes, oi_data

    def reset_oi_history(self) -> None:
        """Reset in-memory OI history. Call at market open (9:15 AM)."""
        with self._oi_history_lock:
            self._oi_history.clear()
        logger.info("OI history reset")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_quote(self, entry: dict) -> Optional[StockQuote]:
        """Parse raw SmartAPI quote entry into StockQuote."""
        try:
            symbol = entry.get("symbol", "").upper()
            if not symbol:
                return None

            ltp = float(entry.get("ltp", 0))
            open_ = float(entry.get("open", 0))
            high = float(entry.get("high", 0))
            low = float(entry.get("low", 0))
            close = float(entry.get("close", 0))
            volume = int(entry.get("volume", 0))
            bid = float(entry.get("bid", 0))
            ask = float(entry.get("ask", 0))

            bid_ask_spread = ask - bid if ask and bid else 0.0
            bid_ask_spread_pct = (bid_ask_spread / ltp * 100) if ltp else 0.0

            price_change_pct = ((ltp - close) / close * 100) if close else 0.0

            return StockQuote(
                symbol=symbol,
                ltp=ltp,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                avg_volume_20d=0.0,  # Will be filled later
                price_change_pct=price_change_pct,
                bid=bid,
                ask=ask,
                bid_ask_spread=bid_ask_spread,
                bid_ask_spread_pct=bid_ask_spread_pct,
                timestamp=datetime.now(tz=_IST),
            )
        except Exception as e:
            logger.warning("Failed to parse quote entry %s: %s", entry, e)
            return None

    def _parse_oi_data(self, symbol: str, chain: dict) -> Optional[StockOIData]:
        """Parse SmartAPI options chain into StockOIData."""
        try:
            # TODO: Extract nearest expiry, ATM strike, CE/PE OI, IV from chain response
            # Placeholder implementation — will need actual SmartAPI response structure
            expiry_str = chain.get("expiry", "2024-01-18")
            atm_strike = float(chain.get("atmStrike", 0))

            total_ce_oi = int(chain.get("totalCeOi", 0))
            total_pe_oi = int(chain.get("totalPeOi", 0))

            atm_ce_iv = float(chain.get("atmCeIv", 0))
            atm_pe_iv = float(chain.get("atmPeIv", 0))
            avg_iv = (atm_ce_iv + atm_pe_iv) / 2

            pcr = total_pe_oi / total_ce_oi if total_ce_oi else 0.0

            # Parse expiry date
            try:
                expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            except ValueError:
                expiry = datetime.now(tz=_IST).date()

            return StockOIData(
                symbol=symbol,
                expiry=expiry,
                total_ce_oi=total_ce_oi,
                total_pe_oi=total_pe_oi,
                ce_oi_change=0,  # Will be calculated later
                pe_oi_change=0,
                pcr=pcr,
                atm_strike=atm_strike,
                atm_ce_iv=atm_ce_iv,
                atm_pe_iv=atm_pe_iv,
                avg_iv=avg_iv,
                iv_change=0.0,  # Will be calculated later
                timestamp=datetime.now(tz=_IST),
            )
        except Exception as e:
            logger.warning("Failed to parse OI data for %s: %s", symbol, e)
            return None

    def _get_token_for_symbol(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """
        Resolve Angel One token for a symbol.

        TODO: Build master contract cache at startup to avoid repeated downloads.
        """
        try:
            master_df = self._smart_api.get_master_contract(exchange)
            if master_df is not None and not master_df.empty:
                mask = master_df["name"] == symbol
                matches = master_df[mask]
                if not matches.empty:
                    return str(matches.iloc[0]["token"])
        except Exception as e:
            logger.warning("Token lookup failed for %s: %s", symbol, e)
        return None


def fetch_eod_oi_bhav(date_str: Optional[str] = None) -> dict[str, int]:
    """
    Fetch NSE F&O bhav copy (EOD OI snapshot).

    Called once at 3:30 PM after market close to get total OI for all F&O stocks.
    Used for next-day universe scoring / OI concentration analysis.

    Parameters
    ----------
    date_str:
        "DD-MM-YYYY" format. If None, uses today.

    Returns
    -------
    dict[str, int]:
        Map of symbol → total OI from bhav copy.
    """
    import requests

    if date_str is None:
        date_str = datetime.now(tz=_IST).strftime("%d-%m-%Y")

    url = f"https://www.nseindia.com/api/fo-bhav-copy?date={date_str}"
    logger.info("Fetching NSE bhav copy for %s", date_str)

    result: dict[str, int] = {}

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        # Parse CSV-like response
        lines = resp.text.strip().split("\n")
        if len(lines) < 2:
            logger.warning("Bhav copy returned empty response")
            return result

        # Skip header, parse rows
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 8:  # Bhav copy has many columns; OI is typically around index 8
                continue

            try:
                symbol = parts[0].strip().upper()
                # Adjust column index based on actual bhav copy format
                # Placeholder: assumes OI in column 8
                oi = int(parts[8].strip())
                if symbol and oi > 0:
                    result[symbol] = oi
            except (ValueError, IndexError):
                continue

        logger.info("Bhav copy parsed: %d symbols with OI data", len(result))

    except Exception as e:
        logger.error("Bhav copy fetch failed: %s", e)

    return result
