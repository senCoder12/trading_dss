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
from datetime import datetime, date, time as dt_time
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

# NSE quote fallback must cover this share of the requested symbols, the same
# bar as the SmartAPI path's 30% failure limit; otherwise it is an error.
_MIN_FALLBACK_COVERAGE = 0.7
# Consecutive failed NSE requests after which the per-symbol loop gives up.
_MAX_CONSECUTIVE_NSE_FAILURES = 5


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
        # (day resolved, nearest stock-option expiry in NSE form)
        self._stock_expiry: Optional[tuple[date, str]] = None

        # Token mapping for SmartAPI
        self._token_map: dict[str, str] = {}
        self._reverse_token_map: dict[str, str] = {}
        self._master_loaded_date: Optional[date] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_token_map(self) -> None:
        """Load and cache symbol-to-token mapping for NSE equities."""
        today = datetime.now(tz=_IST).date()
        if self._master_loaded_date == today and self._token_map:
            return

        logger.info("Building token map from NSE master contract...")
        try:
            df = self._smart_api.get_master_contract("NSE")
            if df is not None and not df.empty:
                # For NSE equities, symbol typically matches name
                # Keep the main equity token, avoid derivatives if mixed
                df_eq = df[df["symbol"].str.endswith("-EQ") | ~df["symbol"].str.contains("-")]
                # Drop duplicates to keep primary token
                df_eq = df_eq.drop_duplicates(subset=["name"])
                
                for _, row in df_eq.iterrows():
                    symbol = str(row["name"]).upper()
                    token = str(row["token"])
                    self._token_map[symbol] = token
                    self._reverse_token_map[token] = symbol
                
                self._master_loaded_date = today
                logger.info("Loaded %d tokens into map", len(self._token_map))
            else:
                logger.error("Failed to load NSE master contract for tokens")
        except Exception as e:
            logger.error("Error building token map: %s", e)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_all_quotes(self, symbols: Optional[list[str]] = None) -> dict[str, StockQuote]:
        """
        Fetch live quotes for all ~200 F&O stocks (or just ``symbols``).

        Batches into groups of bulk_fetch_batch_size (default 50).
        Applies bulk_fetch_delay_seconds between batches to respect rate limits.

        Parameters
        ----------
        symbols:
            Optional subset to quote (e.g. the intraday top 30). Defaults to
            the whole F&O universe. Matters on the NSE per-symbol fallback,
            where every symbol costs one request from the shared NSE budget.

        Returns
        -------
        dict[str, StockQuote]:
            Map of symbol → StockQuote for all successfully fetched stocks.

        Raises
        ------
        DataFetchError:
            If > 30% of symbols fail (systemic issue), including when the
            NSE fallback covers less than 70% of the requested symbols.
        """
        if symbols is None:
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
        null_batches: int = 0

        self._ensure_token_map()

        # Split into batches
        for batch_idx in range(0, len(symbols), self._batch_size):
            batch = symbols[batch_idx : batch_idx + self._batch_size]

            # Map symbols to tokens
            tokens = []
            for s in batch:
                if s in self._token_map:
                    tokens.append(self._token_map[s])
                else:
                    logger.warning("No token found for symbol %s", s)
                    failed.append(s)

            if not tokens:
                continue

            exchange_tokens = {"NSE": tokens}
            logger.info("DEBUG: batch %d symbols: %s (mapped %d tokens)",
                       batch_idx // self._batch_size + 1, len(batch), len(tokens))

            try:
                logger.debug("Fetching batch %d", batch_idx // self._batch_size + 1)
                # Try modes in order: SNAP_QUOTE -> QUOTE -> LTP
                # Prioritize modes with volume data first
                raw = self._smart_api.get_bulk_quotes("SNAP_QUOTE", exchange_tokens)
                if not raw:
                    raw = self._smart_api.get_bulk_quotes("QUOTE", exchange_tokens)
                if not raw:
                    raw = self._smart_api.get_bulk_quotes("LTP", exchange_tokens)
                if not raw:
                    null_batches += 1
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
                        token = entry.get("symbolToken", "")
                        sym = self._reverse_token_map.get(token, "unknown")
                        failed.append(sym)

                # Rate limit: sleep between batches
                if batch_idx + self._batch_size < len(symbols):
                    time.sleep(self._batch_delay)

            except Exception as e:
                logger.error("Batch fetch failed: %s", e)
                failed.extend(batch)

        # Validate: fail if >30% lost OR if SmartAPI returned nothing at all
        # Also fallback if we only got LTP (0 volume) during market hours
        fail_rate = len(failed) / len(symbols) if symbols else 0
        total_batches = max(1, (len(symbols) + self._batch_size - 1) // self._batch_size)
        all_batches_null = null_batches >= total_batches
        
        # Check for zero volume during market hours (indicates LTP-only mode or stale data)
        avg_volume = sum(q.volume for q in quotes.values()) / len(quotes) if quotes else 0
        now = datetime.now(_IST)
        is_market_hours = now.weekday() < 5 and dt_time(9, 15) <= now.time() <= dt_time(15, 30)
        
        # LTP mode (SNAP_QUOTE/QUOTE failed) leaves close and volume at 0, so
        # price change and volume ratio would be 0 for every symbol.
        ltp_only = bool(quotes) and all(q.close == 0 and q.volume == 0 for q in quotes.values())
        if (fail_rate > 0.3 or (all_batches_null and len(quotes) == 0)
                or (is_market_hours and avg_volume == 0 and len(quotes) > 0) or ltp_only):
            # Fallback to NSE scraper if SmartAPI failed or returned incomplete data
            if self._nse_scraper:
                reason = "SmartAPI bulk quote fetch unavailable"
                if ltp_only or (is_market_hours and avg_volume == 0 and len(quotes) > 0):
                    reason = "SmartAPI returned LTP-only data (no volume)"
                
                logger.warning(
                    "%s (null_batches=%d/%d, fail_rate=%.0f%%). Falling back to NSE scraper.",
                    reason, null_batches, total_batches, fail_rate * 100
                )
                fallback = self._fallback_fetch_quotes_nse(symbols)
                if len(fallback) < _MIN_FALLBACK_COVERAGE * len(symbols):
                    raise DataFetchError(
                        f"NSE fallback covered only {len(fallback)}/{len(symbols)} symbols"
                    )
                return fallback

            raise DataFetchError(
                f"Quote fetch failed: null_batches={null_batches}/{total_batches}, "
                f"fail_rate={fail_rate:.1%}"
            )

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
            # Get all F&O stock quotes from scraper
            stocks_data = self._nse_scraper.get_all_fno_quotes()
            if not stocks_data:
                # The bulk F&O index endpoint is gone (404); quote per symbol.
                logger.warning("NSE bulk F&O quotes unavailable; fetching per symbol")
                return self._fetch_quotes_nse_per_symbol(symbols)

            quotes: dict[str, StockQuote] = {}
            for s_data in stocks_data:
                symbol = s_data.get("symbol", "")
                if symbol not in symbols:
                    continue

                try:
                    quote = StockQuote(
                        symbol=symbol,
                        ltp=float(s_data.get("ltp", 0)),
                        open=float(s_data.get("open", 0)),
                        high=float(s_data.get("high", 0)),
                        low=float(s_data.get("low", 0)),
                        close=float(s_data.get("close", 0)),
                        volume=int(s_data.get("volume", 0)),
                        avg_volume_20d=0.0,  # Will be fetched separately
                        price_change_pct=float(s_data.get("change_pct", 0)),
                        bid=0.0,
                        ask=0.0,
                        bid_ask_spread=0.0,
                        bid_ask_spread_pct=0.0,
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

    def _fetch_quotes_nse_per_symbol(self, symbols: list[str]) -> dict[str, StockQuote]:
        """
        One NSE GetQuoteApi call per symbol.

        Slow: every call takes a token from the shared NSE limiter (25 req /
        60 s), so ~200 symbols take ~8 minutes and halve the NSE budget left
        for other consumers meanwhile. Pass a small subset where possible.

        Stops early after ``_MAX_CONSECUTIVE_NSE_FAILURES`` failed requests
        in a row (NSE down or the circuit breaker open); the caller's
        coverage check then rejects the partial result.
        """
        start = time.time()
        quotes: dict[str, StockQuote] = {}
        consecutive_failures = 0
        for symbol in symbols:
            if consecutive_failures >= _MAX_CONSECUTIVE_NSE_FAILURES:
                logger.error(
                    "NSE per-symbol quotes: %d failures in a row; stopping after %d/%d",
                    consecutive_failures, len(quotes), len(symbols),
                )
                break
            try:
                resp = self._nse_scraper._call(
                    "/api/NextApi/apiClient/GetQuoteApi",
                    params={
                        "functionName": "getSymbolData",
                        "marketType": "N",
                        "series": "EQ",
                        "symbol": symbol,
                    },
                )
                if resp is None:  # request failed or circuit breaker open
                    consecutive_failures += 1
                    continue
                consecutive_failures = 0
                eq = resp.get("equityResponse") or []
                if not eq:
                    continue
                meta = eq[0].get("metaData") or {}
                trade = eq[0].get("tradeInfo") or {}
                book = eq[0].get("orderBook") or {}
                ltp = float(trade.get("lastPrice") or meta.get("closePrice") or 0)
                if ltp <= 0:
                    continue
                bid = float(book.get("buyPrice1") or 0)
                ask = float(book.get("sellPrice1") or 0)
                spread = ask - bid if bid > 0 and ask > 0 else 0.0
                quotes[symbol] = StockQuote(
                    symbol=symbol,
                    ltp=ltp,
                    open=float(meta.get("open") or 0),
                    high=float(meta.get("dayHigh") or 0),
                    low=float(meta.get("dayLow") or 0),
                    close=float(meta.get("previousClose") or meta.get("basePrice") or 0),
                    volume=int(trade.get("totalTradedVolume") or 0),
                    avg_volume_20d=0.0,  # injected later from fo_avg_volumes
                    price_change_pct=float(meta.get("pChange") or 0),
                    bid=bid,
                    ask=ask,
                    bid_ask_spread=spread,
                    bid_ask_spread_pct=(spread / ltp * 100) if spread else 0.0,
                    timestamp=datetime.now(tz=_IST),
                )
            except Exception as e:
                logger.debug("NSE quote failed for %s: %s", symbol, e)
        logger.info(
            "NSE per-symbol quotes: %d/%d in %.1fs",
            len(quotes), len(symbols), time.time() - start,
        )
        return quotes

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
            
            # Sleep to respect rate limit of 3 req/sec
            time.sleep(0.4)

        elapsed = time.time() - start
        logger.info("Average volumes fetched: %d/%d in %.1fs", len(result), len(symbols), elapsed)

        return result

    def fetch_oi_for_symbols(self, symbols: list[str]) -> dict[str, StockOIData]:
        """
        Fetch options chain data for a filtered subset of symbols.

        Source is the NSE v3 equity option chain (OI, day's OI change and IV
        per strike). SmartAPI is not a fallback: its option-chain call is
        ``optionGreek``, which returns greeks without OI.

        ``ce_oi_change`` / ``pe_oi_change`` are always NSE's change since the
        previous session's close, so they mean the same thing on the first
        fetch and on every intraday refresh. ``iv_change`` is the change since
        this process's previous fetch of the symbol (0 on the first).

        Parameters
        ----------
        symbols:
            Filtered list (typically ~40) of high-liquidity candidates.

        Returns
        -------
        dict[str, StockOIData]:
            Map of symbol → StockOIData.
        """
        if not symbols:
            return {}
        if not self._nse_scraper:
            logger.error("OI fetch needs the NSE scraper; SmartAPI has no OI source")
            return {}

        logger.info("Fetching OI data for %d symbols", len(symbols))

        # Stock options share one monthly expiry; resolve it once per day.
        nse_expiry = self._nse_stock_expiry(symbols[0])
        if not nse_expiry:
            logger.error("Could not resolve nearest expiry for OI fetch")
            return {}

        logger.info("Using expiry %s for OI fetch", nse_expiry)
        start = time.time()
        result: dict[str, StockOIData] = {}

        for symbol in symbols:
            try:
                chain = self._nse_scraper.get_option_chain_v3(
                    symbol, type="Equities", expiry=nse_expiry,
                )
                if not chain:
                    logger.debug("No OI data for %s", symbol)
                    continue

                oi_data = self._parse_oi_data(symbol, chain)
                if oi_data:
                    with self._oi_history_lock:
                        prev = self._oi_history.get(symbol)
                        oi_data.iv_change = (
                            round(oi_data.avg_iv - prev.avg_iv, 4) if prev else 0.0
                        )
                        self._oi_history[symbol] = oi_data
                    result[symbol] = oi_data

            except Exception as e:
                logger.error("OI fetch failed for %s: %s", symbol, e)

        elapsed = time.time() - start
        logger.info("OI data fetched: %d/%d in %.1fs", len(result), len(symbols), elapsed)

        return result

    def _nse_stock_expiry(self, symbol: str) -> Optional[str]:
        """
        Nearest stock-option expiry in NSE form (e.g. "23-Nov-2026").

        Cached for the day: one meta request per day, not one per OI run.
        A failed lookup is not cached, so the next run retries.
        """
        today = date.today()
        if self._stock_expiry is not None and self._stock_expiry[0] == today:
            return self._stock_expiry[1]
        try:
            meta = self._nse_scraper.get_option_chain_meta(symbol) or {}
            expiries = meta.get("expiryDates") or []
        except Exception as e:
            logger.warning("NSE expiry lookup failed for %s: %s", symbol, e)
            return None
        if not expiries:
            logger.warning("NSE expiry lookup returned no dates for %s", symbol)
            return None
        self._stock_expiry = (today, expiries[0])
        logger.info("Resolved stock-option expiry: %s", expiries[0])
        return expiries[0]

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
            token = entry.get("symbolToken", "")
            symbol = self._reverse_token_map.get(token)
            
            if not symbol:
                ts = entry.get("tradingSymbol", entry.get("symbol", ""))
                symbol = ts.split("-")[0].upper()
            
            if not symbol:
                return None

            ltp = float(entry.get("ltp", 0))
            open_ = float(entry.get("open", 0))
            high = float(entry.get("high", 0))
            low = float(entry.get("low", 0))
            close = float(entry.get("close", 0))
            volume = int(entry.get("volume", entry.get("tradedVolume", entry.get("totalTradedVolume", 0))))
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
        """Parse NSE option chain response into StockOIData."""
        try:
            records = chain.get("records", {})
            filtered = chain.get("filtered", {})

            # Get underlying spot price
            underlying = float(records.get("underlyingValue", 0))

            # Get expiry dates - use nearest
            expiry_dates = records.get("expiryDates", [])
            if not expiry_dates:
                return None
            expiry_str = expiry_dates[0]

            # Parse expiry date (format: "28-Nov-2024" or "2024-11-28")
            try:
                if len(expiry_str) == 10 and expiry_str[4] == "-":
                    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                else:
                    expiry = datetime.strptime(expiry_str, "%d-%b-%Y").date()
            except ValueError:
                expiry = datetime.now(tz=_IST).date()

            # Use totals from filtered section if available
            total_ce_oi = int(filtered.get("CE", {}).get("totOI", 0))
            total_pe_oi = int(filtered.get("PE", {}).get("totOI", 0))

            # Find ATM strike and IV from records data
            chain_data = records.get("data", [])
            nearest_expiry_data = [
                row for row in chain_data
                if (row.get("expiryDate") or row.get("expiryDates") or "") == expiry_str
            ]
            if not nearest_expiry_data and chain_data:
                # v3 responses are already limited to the requested expiry
                nearest_expiry_data = chain_data

            atm_strike = 0.0
            atm_ce_iv = 0.0
            atm_pe_iv = 0.0

            if underlying > 0 and nearest_expiry_data:
                # Find ATM strike (closest to underlying)
                strikes = [(float(row.get("strikePrice", 0)), row) for row in nearest_expiry_data]
                if strikes:
                    atm_strike, atm_row = min(strikes, key=lambda x: abs(x[0] - underlying))
                    atm_ce_iv = float((atm_row.get("CE") or {}).get("impliedVolatility", 0) or 0)
                    atm_pe_iv = float((atm_row.get("PE") or {}).get("impliedVolatility", 0) or 0)

                # If totals not in filtered section, sum from data
                if total_ce_oi == 0 and total_pe_oi == 0:
                    for row in nearest_expiry_data:
                        total_ce_oi += int((row.get("CE") or {}).get("openInterest", 0) or 0)
                        total_pe_oi += int((row.get("PE") or {}).get("openInterest", 0) or 0)

            avg_iv = (atm_ce_iv + atm_pe_iv) / 2 if (atm_ce_iv or atm_pe_iv) else 0.0
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0

            # OI change = NSE's change since the previous session's close
            # (changeinOpenInterest), summed over the expiry's strikes.
            ce_oi_change = sum(
                int((row.get("CE") or {}).get("changeinOpenInterest", 0) or 0)
                for row in nearest_expiry_data
            )
            pe_oi_change = sum(
                int((row.get("PE") or {}).get("changeinOpenInterest", 0) or 0)
                for row in nearest_expiry_data
            )

            result = StockOIData(
                symbol=symbol,
                expiry=expiry,
                total_ce_oi=total_ce_oi,
                total_pe_oi=total_pe_oi,
                ce_oi_change=ce_oi_change,
                pe_oi_change=pe_oi_change,
                pcr=round(pcr, 4),
                atm_strike=atm_strike,
                atm_ce_iv=atm_ce_iv,
                atm_pe_iv=atm_pe_iv,
                avg_iv=round(avg_iv, 4),
                iv_change=0.0,  # set by fetch_oi_for_symbols from its history
                timestamp=datetime.now(tz=_IST),
            )

            return result

        except Exception as e:
            logger.warning("Failed to parse OI data for %s: %s", symbol, e)
            return None

    def _get_token_for_symbol(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """
        Resolve Angel One token for a symbol.
        """
        if exchange == "NSE":
            self._ensure_token_map()
            return self._token_map.get(symbol)
        
        # Fallback for other exchanges
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
