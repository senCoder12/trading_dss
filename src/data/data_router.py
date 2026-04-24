"""
DataRouter — single point of entry for live market data.

Routes requests to the Angel One broker API (WebSocket) when configured,
or falls back silently to the NSE scraper.  The rest of the system never
needs to know which source is active.

Routing logic
─────────────
  USE_BROKER_API=True  AND  login succeeds  →  Angel One WebSocket
  USE_BROKER_API=False OR   login fails     →  NSE scraper (polling)

WebSocket mode: _handle_tick() aggregates ticks into 1-minute OHLCV bars and
writes them to price_data in real time.  The DataCollector's collect_index_prices
job becomes a lightweight health probe instead of a data-fetch loop.

Scraper mode: get_all_prices() delegates to NSEScraper.get_all_indices() and
returns the same dict list that DataCollector always consumed.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.database import queries as Q
from src.database.db_manager import DatabaseManager

if TYPE_CHECKING:
    from src.data.broker.websocket_feed import LiveTick
    from src.data.index_registry import IndexRegistry

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# Angel One token IDs for the four main F&O indices on NSE.
# These are stable exchange-assigned identifiers, not user-configured.
_INDEX_TOKENS: dict[str, str] = {
    "NIFTY50": "26000",
    "BANKNIFTY": "26009",
    "FINNIFTY": "99926037",
    "MIDCPNIFTY": "99926074",
    "SENSEX": "99919000", # BSE SENSEX Index Token
}
_TOKEN_TO_INDEX: dict[str, str] = {v: k for k, v in _INDEX_TOKENS.items()}

# Strike intervals per F&O index
_STRIKE_INTERVAL: dict[str, int] = {
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
}
_DEFAULT_STRIKE_INTERVAL = 50


class DataRouter:
    """Routes data requests to broker API or NSE scraper based on config.

    Parameters
    ----------
    db:
        Open DatabaseManager instance (used for WebSocket minute-bar writes).
    registry:
        IndexRegistry for iterating F&O indices during WebSocket setup.
    dry_run:
        When True, fetch data but do not write to the database.
    """

    def __init__(
        self,
        db: DatabaseManager,
        registry: Optional["IndexRegistry"] = None,
        dry_run: bool = False,
    ) -> None:
        self.db = db
        self.registry = registry
        self.dry_run = dry_run

        self.broker_client = None
        self.ws_feed: Optional[object] = None
        self._master_contract = None  # pd.DataFrame | None

        # In-progress 1-minute OHLCV bars, keyed by index_id
        self._current_bars: dict[str, dict] = {}
        self._bars_lock = threading.Lock()
        # Throttle price_tick WS broadcasts to max 1/sec per index
        self._last_tick_broadcast: dict[str, float] = {}

        if settings.broker.use_broker_api and settings.broker.angel_one_api_key:
            self._init_broker()

        # Always initialise scraper — it is the fallback even in broker mode
        # Always initialise scrapers — they are the fallbacks
        from src.data.nse_scraper import NSEScraper  # noqa: PLC0415
        from src.data.bse_scraper import BSEScraper  # noqa: PLC0415
        from src.data.rate_limiter import GlobalRateLimiter  # noqa: PLC0415

        _nse_limiter = GlobalRateLimiter.get("nseindia.com", max_requests=25, window_seconds=60)
        self.nse_scraper = NSEScraper(rate_limiter=_nse_limiter)
        self.bse_scraper = BSEScraper()

        if not self.broker_client:
            logger.info("DataRouter: using Scrapers (polling mode)")

    # ── Broker initialisation ──────────────────────────────────────────────

    def _init_broker(self) -> None:
        try:
            from src.data.broker.angel_one import AngelOneClient  # noqa: PLC0415
            from src.data.broker.websocket_feed import WebSocketFeed  # noqa: PLC0415

            client = AngelOneClient()
            if not client.login():
                logger.warning(
                    "DataRouter: Angel One login failed — falling back to NSE scraper"
                )
                return

            self.broker_client = client
            self.ws_feed = WebSocketFeed(client)
            self._start_websocket_feed()
            logger.info("DataRouter: using Angel One SmartAPI (WebSocket mode)")

        except Exception as exc:
            logger.error(
                "DataRouter: broker init failed (%s) — falling back to NSE scraper", exc
            )

    def _start_websocket_feed(self) -> None:
        """Load master contract, compute ATM option tokens, start WebSocket."""
        try:
            self._master_contract = self.broker_client.get_master_contract()

            index_tokens = list(_INDEX_TOKENS.values())
            nfo_tokens = self._get_atm_option_tokens()

            subscribe_list: list[dict] = [
                {"exchangeType": 1, "tokens": index_tokens},
            ]
            if nfo_tokens:
                subscribe_list.append({"exchangeType": 2, "tokens": nfo_tokens})

            self.ws_feed.on_tick_update = self._handle_tick
            self.ws_feed.on_oi_update = self._handle_oi_update
            self.ws_feed.start(subscribe_list)

            logger.info(
                "WebSocket: subscribed to %d index token(s) + %d option token(s)",
                len(index_tokens),
                len(nfo_tokens),
            )
        except Exception as exc:
            logger.error("WebSocket feed start failed: %s", exc)

    def _get_atm_option_tokens(self) -> list[str]:
        """Return tokens for ATM ±depth strikes for all active F&O indices."""
        if self._master_contract is None or self.registry is None:
            return []

        depth = settings.broker.websocket_subscribe_depth
        tokens: list[str] = []

        try:
            for idx in self.registry.get_active_indices():
                if not idx.has_options or not idx.option_symbol:
                    continue

                # Only subscribe index tokens we know (those in _INDEX_TOKENS)
                index_token = _INDEX_TOKENS.get(idx.id)
                if not index_token:
                    continue

                ltp = self.broker_client.get_ltp(
                    "NSE", idx.nse_symbol or idx.option_symbol, index_token
                )
                if not ltp:
                    continue

                interval = _STRIKE_INTERVAL.get(idx.id, _DEFAULT_STRIKE_INTERVAL)
                atm = round(ltp / interval) * interval

                # Find nearest weekly expiry from master contract
                mc = self._master_contract
                sym_df = mc[mc["name"] == idx.option_symbol].copy()
                if sym_df.empty:
                    continue
                sym_df = sym_df.sort_values("expiry")
                nearest_expiry = sym_df["expiry"].iloc[0]

                for offset in range(-depth, depth + 1):
                    strike = atm + offset * interval
                    for opt_type in ("CE", "PE"):
                        t = self.broker_client.get_token_for_symbol(
                            idx.option_symbol, nearest_expiry, strike, opt_type, mc
                        )
                        if t:
                            tokens.append(t)

        except Exception as exc:
            logger.error("ATM option token lookup failed: %s", exc)

        return tokens

    # ── WebSocket callbacks ────────────────────────────────────────────────

    def _handle_tick(self, tick: "LiveTick") -> None:
        """Aggregate raw ticks into 1-minute OHLCV bars; flush completed bars to DB."""
        index_id = _TOKEN_TO_INDEX.get(tick.token)
        if not index_id:
            return

        bar_minute = tick.timestamp.replace(second=0, microsecond=0)

        with self._bars_lock:
            current = self._current_bars.get(index_id)

            if current and current["minute"] == bar_minute:
                # Extend the current bar
                current["high"] = max(current["high"], tick.ltp)
                current["low"] = min(current["low"], tick.ltp)
                current["close"] = tick.ltp
                current["volume"] = tick.volume  # cumulative day volume replaces
            else:
                # New minute — flush previous bar, start fresh
                if current and not self.dry_run:
                    self._flush_bar(index_id, current)
                self._current_bars[index_id] = {
                    "minute": bar_minute,
                    "open": tick.ltp,
                    "high": tick.ltp,
                    "low": tick.ltp,
                    "close": tick.ltp,
                    "volume": tick.volume,
                }

        # Push live tick to frontend via WebSocket — throttled to 1/sec per index
        now = time.monotonic()
        if now - self._last_tick_broadcast.get(index_id, 0) >= 1.0:
            self._last_tick_broadcast[index_id] = now
            try:
                from src.api.websocket import queue_event  # noqa: PLC0415
                queue_event("price_tick", {
                    "index_id": index_id,
                    "ltp": tick.ltp,
                    "open": tick.open,
                    "high": tick.high,
                    "low": tick.low,
                    "volume": tick.volume,
                    "timestamp": tick.timestamp.isoformat(),
                })
            except Exception as exc:
                logger.debug("price_tick broadcast failed: %s", exc)

    def _flush_bar(self, index_id: str, bar: dict) -> None:
        """Write a completed 1-minute bar to price_data."""
        try:
            self.db.execute_many(
                Q.UPSERT_PRICE_DATA,
                [(
                    index_id,
                    bar["minute"].isoformat(),
                    bar["open"],
                    bar["high"],
                    bar["low"],
                    bar["close"],
                    float(bar["volume"]),
                    0.0,            # vwap not available from tick
                    "angel_one_ws",
                    "1m",
                )],
            )
        except Exception as exc:
            logger.error("Bar flush failed for %s: %s", index_id, exc)

    def _handle_oi_update(self, tick: "LiveTick") -> None:
        # OI changes for individual options are handled by the options chain
        # subsystem (OptionsChainFetcher).  No action needed here.
        pass

    # ── Public interface ───────────────────────────────────────────────────

    def get_index_price(self, index_id: str) -> Optional[dict]:
        """Return current price dict for a single index.

        In WebSocket mode returns the latest tick cached in memory.
        Falls back to scraper otherwise.
        """
        if self.ws_feed:
            token = _INDEX_TOKENS.get(index_id)
            if token:
                tick = self.ws_feed.get_latest(token)
                if tick:
                    return {
                        "index_id": index_id,
                        "index_name": index_id,
                        "ltp": tick.ltp,
                        "open": tick.open,
                        "high": tick.high,
                        "low": tick.low,
                        "volume": tick.volume,
                        "timestamp": tick.timestamp.isoformat(),
                        "bid": tick.bid_price,
                        "ask": tick.ask_price,
                    }

        # Scraper fallback — look up by symbol via registry
        if self.registry:
            idx = self.registry.get_index(index_id)
            if idx:
                if idx.exchange == "BSE":
                    return self.bse_scraper.get_index_quote(idx.id)
                if idx.nse_symbol:
                    return self.nse_scraper.get_index_quote(idx.nse_symbol)

        # Last-resort: scan all indices
        nse_data = self.nse_scraper.get_all_indices() or []
        bse_data = self.bse_scraper.get_all_indices() or []
        all_data = nse_data + bse_data
        return next(
            (item for item in all_data if item.get("index_name", "").upper() == index_id.upper()),
            None,
        )

    def get_all_prices(self) -> list[dict]:
        """Return current prices for all tracked instruments.

        In WebSocket mode, returns all ticks cached in memory.
        In scraper mode, returns the same list as NSEScraper.get_all_indices() and broadcasts to WS.
        """
        if self.ws_feed:
            all_ticks = self.ws_feed.get_all_latest()
            return [
                {
                    "index_id": index_id,
                    "index_name": index_id,
                    "ltp": tick.ltp,
                    "open": tick.open,
                    "high": tick.high,
                    "low": tick.low,
                    "volume": tick.volume,
                    "timestamp": tick.timestamp.isoformat(),
                }
                for token, tick in all_ticks.items()
                if (index_id := _TOKEN_TO_INDEX.get(token)) is not None
            ]

        nse_data = self.nse_scraper.get_all_indices() or []
        bse_data = self.bse_scraper.get_all_indices() or []
        all_data = nse_data + bse_data

        # Broadcast scraper prices to frontend via WebSocket — throttled to 1/sec per index
        now = time.monotonic()
        try:
            from src.api.websocket import queue_event
            for item in all_data:
                index_id = item.get("index_id")
                if index_id and now - self._last_tick_broadcast.get(index_id, 0) >= 1.0:
                    self._last_tick_broadcast[index_id] = now
                    queue_event("price_tick", {
                        "index_id": index_id,
                        "ltp": item.get("ltp"),
                        "open": item.get("open"),
                        "high": item.get("high"),
                        "low": item.get("low"),
                        "volume": item.get("volume"),
                        "timestamp": item.get("updated_at", ""),
                    })
        except Exception as exc:
            logger.debug("price_tick broadcast (scraper) failed: %s", exc)

        return all_data

    def is_using_broker_api(self) -> bool:
        """True when broker API is active and WebSocket feed is running."""
        return self.broker_client is not None

    def get_data_source(self) -> str:
        if self.broker_client:
            return "Angel One SmartAPI (WebSocket, real-time)"
        return "NSE Scraper (polling)"

    def stop(self) -> None:
        """Flush in-progress minute bars and close WebSocket connection."""
        if self.ws_feed:
            if not self.dry_run:
                with self._bars_lock:
                    for index_id, bar in self._current_bars.items():
                        self._flush_bar(index_id, bar)
                    self._current_bars.clear()
            self.ws_feed.stop()
