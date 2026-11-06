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

# Angel One token IDs for indices.  Stable exchange-assigned identifiers.
# Format: {index_id: (token, exchange_code)}
# exchangeType: 1=NSE, 2=NFO, 3=BSE, 4=BFO
_INDEX_TOKENS: dict[str, tuple[str, int]] = {
    "NIFTY50": ("26000", 1),         # NSE
    "BANKNIFTY": ("26009", 1),       # NSE
    "FINNIFTY": ("99926037", 1),     # NSE
    "MIDCPNIFTY": ("99926074", 1),   # NSE
    "SENSEX": ("99919000", 3),       # BSE
}
_TOKEN_TO_INDEX: dict[str, str] = {token: idx for idx, (token, _) in _INDEX_TOKENS.items()}

# Exchange type to exchange code mapping
_EXCHANGE_TYPE_MAP: dict[int, str] = {
    1: "NSE",
    2: "NFO",
    3: "BSE",
    4: "BFO",
}

# Strike intervals per F&O index
_STRIKE_INTERVAL: dict[str, int] = {
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
}
_DEFAULT_STRIKE_INTERVAL = 50


def _nearest_expiry(df: "pd.DataFrame") -> Optional[str]:
    """Return the nearest non-expired expiry string from Angel master rows.

    Angel stores expiry as text like "27NOV2025"; sorting that text is not
    chronological ("01DEC2026" < "10NOV2026"), so parse before comparing.
    """
    import pandas as pd  # noqa: PLC0415

    if df is None or df.empty:
        return None
    parsed = pd.to_datetime(df["expiry"], format="%d%b%Y", errors="coerce")
    today = pd.Timestamp(datetime.now(tz=_IST).date())
    live = parsed[parsed >= today]
    if live.empty:
        return None
    return str(df.loc[live.idxmin(), "expiry"])


def _best_level_ofi(
    prev: Optional[tuple[float, int, float, int]],
    cur: tuple[float, int, float, int],
) -> float:
    """Best-level order flow imbalance between two book snapshots.

    Cont, Kukanov & Stoikov (2014):
        e = 1{Pb >= Pb'}·qb − 1{Pb <= Pb'}·qb' − 1{Pa <= Pa'}·qa + 1{Pa >= Pa'}·qa'
    where ' marks the previous snapshot.  Tuples are (bid, bid_qty, ask, ask_qty).
    Positive = net buying pressure.  SnapQuote ticks are sampled snapshots,
    not every book event, so this is an approximation of true OFI.
    """
    if prev is None:
        return 0.0
    pb, qb, pa, qa = prev
    b, qb_n, a, qa_n = cur
    e = 0.0
    if b >= pb:
        e += qb_n
    if b <= pb:
        e -= qb
    if a <= pa:
        e -= qa_n
    if a >= pa:
        e += qa
    return e


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
        self._last_tick_times: dict[str, float] = {}  # token → last tick monotonic time
        # Near-month index futures: spot indices have no order book, so depth
        # features come from the futures contract.
        self._fut_token_to_index: dict[str, str] = {}
        self._latest_fut_depth: dict[str, "LiveTick"] = {}
        # In-progress 1-minute depth aggregates + previous best level (for OFI)
        self._depth_bars: dict[str, dict] = {}
        self._prev_best: dict[str, tuple[float, int, float, int]] = {}
        self._prev_fut_volume: dict[str, int] = {}  # last cumulative day volume seen
        self._depth_lock = threading.Lock()

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
            # Load NSE (Equity), NFO (NSE F&O), and BFO (BSE F&O) master contracts
            import pandas as pd  # noqa: PLC0415
            self._master_contract = self.broker_client.get_master_contract("NFO")
            nse_contract = self.broker_client.get_master_contract("NSE")
            bfo_contract = self.broker_client.get_master_contract("BFO")
            
            contracts = [c for c in [self._master_contract, nse_contract, bfo_contract] if c is not None]
            if contracts:
                self._master_contract = pd.concat(contracts, ignore_index=True)

            # Group index tokens by exchange type
            tokens_by_exchange: dict[int, list[str]] = {}
            for idx_id, (token, exchange_type) in _INDEX_TOKENS.items():
                if exchange_type not in tokens_by_exchange:
                    tokens_by_exchange[exchange_type] = []
                tokens_by_exchange[exchange_type].append(token)

            # ── Dynamic Stock Tokens ──────────────────────────────────────────
            # Add tokens for any active indices in the registry that are NOT
            # in the hardcoded _INDEX_TOKENS (i.e. F&O stocks).
            dynamic_token_map = {}
            if self.registry:
                for idx in self.registry.get_active_indices():
                    if idx.id not in _INDEX_TOKENS:
                        # Try to find equity token in NSE master
                        mask = (self._master_contract["name"] == idx.id) & \
                               (self._master_contract["exch_seg"] == "NSE") & \
                               (self._master_contract["symbol"].str.endswith("-EQ"))
                        res = self._master_contract[mask]
                        if not res.empty:
                            token = res["token"].iloc[0]
                            dynamic_token_map[token] = idx.id
                            if 1 not in tokens_by_exchange:
                                tokens_by_exchange[1] = []
                            tokens_by_exchange[1].append(token)
            
            # Store dynamic tokens for _handle_tick reverse lookup
            self._dynamic_token_to_index = dynamic_token_map

            option_tokens_by_exchange = self._get_atm_option_tokens()

            # Near-month index futures → order-book depth features
            fut_tokens_by_exchange, self._fut_token_to_index = self._get_near_futures_tokens()
            for exchange_type, fut_tokens in fut_tokens_by_exchange.items():
                option_tokens_by_exchange.setdefault(exchange_type, []).extend(fut_tokens)

            # Build subscription list for each exchange
            subscribe_list: list[dict] = []
            for exchange_type in sorted(tokens_by_exchange.keys()):
                subscribe_list.append({
                    "exchangeType": exchange_type,
                    "tokens": tokens_by_exchange[exchange_type],
                })
            # Add options tokens for each exchange
            for exchange_type in sorted(option_tokens_by_exchange.keys()):
                if option_tokens_by_exchange[exchange_type]:
                    subscribe_list.append({
                        "exchangeType": exchange_type,
                        "tokens": option_tokens_by_exchange[exchange_type],
                    })

            self.ws_feed.on_tick_update = self._handle_tick
            self.ws_feed.on_oi_update = self._handle_oi_update
            self.ws_feed.on_depth_update = self._handle_depth_update
            self.ws_feed.start(subscribe_list)

            total_index_tokens = sum(len(t) for t in tokens_by_exchange.values())
            total_fut_tokens = len(self._fut_token_to_index)
            total_option_tokens = (
                sum(len(t) for t in option_tokens_by_exchange.values()) - total_fut_tokens
            )
            logger.info(
                "WebSocket: subscribed to %d index token(s) + %d option token(s) "
                "+ %d index futures token(s) %s",
                total_index_tokens,
                total_option_tokens,
                total_fut_tokens,
                sorted(self._fut_token_to_index.values()),
            )
        except Exception as exc:
            logger.error("WebSocket feed start failed: %s", exc)

    def _get_atm_option_tokens(self) -> dict[int, list[str]]:
        """Return tokens for ATM ±depth strikes for all active F&O indices, grouped by exchange."""
        if self._master_contract is None or self.registry is None:
            return {}

        depth = settings.broker.websocket_subscribe_depth
        tokens_by_exchange: dict[int, list[str]] = {}

        try:
            for idx in self.registry.get_active_indices():
                if not idx.has_options or not idx.option_symbol:
                    continue

                # Only subscribe index tokens we know (those in _INDEX_TOKENS)
                token_info = _INDEX_TOKENS.get(idx.id)
                if not token_info:
                    continue

                index_token, exchange_type = token_info
                exchange = _EXCHANGE_TYPE_MAP.get(exchange_type, "NSE")

                ltp = self.broker_client.get_ltp(
                    exchange, idx.nse_symbol or idx.option_symbol, index_token
                )
                if not ltp:
                    continue

                interval = _STRIKE_INTERVAL.get(idx.id, _DEFAULT_STRIKE_INTERVAL)
                atm = round(ltp / interval) * interval

                # Find nearest weekly expiry from master contract (options rows
                # only — futures share the name but have monthly expiries)
                mc = self._master_contract
                sym_df = mc[
                    (mc["name"] == idx.option_symbol)
                    & (mc["instrumenttype"].astype(str).str.startswith("OPT"))
                ]
                nearest_expiry = _nearest_expiry(sym_df)
                if nearest_expiry is None:
                    continue

                # Determine options exchange: BFO for BSE indices, NFO for NSE indices
                opt_exchange_type = 4 if exchange_type == 3 else 2  # BFO if BSE, else NFO
                if opt_exchange_type not in tokens_by_exchange:
                    tokens_by_exchange[opt_exchange_type] = []

                for offset in range(-depth, depth + 1):
                    strike = atm + offset * interval
                    for opt_type in ("CE", "PE"):
                        t = self.broker_client.get_token_for_symbol(
                            idx.option_symbol, nearest_expiry, strike, opt_type, mc
                        )
                        if t:
                            tokens_by_exchange[opt_exchange_type].append(t)

        except Exception as exc:
            logger.error("ATM option token lookup failed: %s", exc)

        return tokens_by_exchange

    def _get_near_futures_tokens(self) -> tuple[dict[int, list[str]], dict[str, str]]:
        """Near-month futures tokens for F&O indices in _INDEX_TOKENS.

        Returns ({exchange_type: [tokens]}, {token: index_id}).
        """
        tokens_by_exchange: dict[int, list[str]] = {}
        token_to_index: dict[str, str] = {}
        if self._master_contract is None or self.registry is None:
            return tokens_by_exchange, token_to_index

        mc = self._master_contract
        try:
            for idx in self.registry.get_active_indices():
                token_info = _INDEX_TOKENS.get(idx.id)
                if not token_info or not idx.has_options or not idx.option_symbol:
                    continue

                fut_exchange_type = 4 if token_info[1] == 3 else 2  # BFO if BSE, else NFO
                fut_df = mc[
                    (mc["name"] == idx.option_symbol)
                    & (mc["exch_seg"] == _EXCHANGE_TYPE_MAP[fut_exchange_type])
                    & (mc["instrumenttype"] == "FUTIDX")
                ]
                expiry = _nearest_expiry(fut_df)
                if expiry is None:
                    logger.warning("No near-month futures contract found for %s", idx.id)
                    continue

                token = str(fut_df[fut_df["expiry"] == expiry]["token"].iloc[0])
                tokens_by_exchange.setdefault(fut_exchange_type, []).append(token)
                token_to_index[token] = idx.id
        except Exception as exc:
            logger.error("Index futures token lookup failed: %s", exc)

        return tokens_by_exchange, token_to_index

    # ── WebSocket callbacks ────────────────────────────────────────────────

    def _handle_tick(self, tick: "LiveTick") -> None:
        """Aggregate raw ticks into 1-minute OHLCV bars; flush completed bars to DB."""
        # Check static index tokens first, then dynamic stock tokens
        index_id = _TOKEN_TO_INDEX.get(tick.token) or \
                   getattr(self, "_dynamic_token_to_index", {}).get(tick.token)
        
        if not index_id:
            return

        # Track per-token freshness
        self._last_tick_times[tick.token] = time.monotonic()

        bar_minute = tick.timestamp.replace(second=0, microsecond=0)

        with self._bars_lock:
            current = self._current_bars.get(index_id)

            if current and current["minute"] == bar_minute:
                # Extend the current bar
                current["high"] = max(current["high"], tick.ltp)
                current["low"] = min(current["low"], tick.ltp)
                current["close"] = tick.ltp
                # Accumulate VWAP components using volume delta (cumulative day volume)
                vol_delta = max(0, tick.volume - current["volume"])
                current["volume"] = tick.volume
                current["cum_vol"] += vol_delta
                current["cum_tp_vol"] += tick.ltp * vol_delta
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
                    "cum_vol": tick.volume,
                    "cum_tp_vol": tick.ltp * tick.volume,
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
                    bar["cum_tp_vol"] / bar["cum_vol"] if bar.get("cum_vol", 0) > 0 else 0.0,
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

    def _handle_depth_update(self, tick: "LiveTick") -> None:
        """Cache the latest near-month futures depth and fold it into the minute bar."""
        index_id = self._fut_token_to_index.get(tick.token)
        if not index_id:
            return
        self._latest_fut_depth[index_id] = tick
        if tick.has_depth:  # need both sides for spread / OFI
            self._accumulate_depth(index_id, tick)

    def _accumulate_depth(self, index_id: str, tick: "LiveTick") -> None:
        minute = tick.timestamp.replace(second=0, microsecond=0)
        cur_best = (tick.bid_price, tick.bid_qty, tick.ask_price, tick.ask_qty)

        with self._depth_lock:
            ofi = _best_level_ofi(self._prev_best.get(index_id), cur_best)
            self._prev_best[index_id] = cur_best

            # Tick volume is cumulative for the day. The first tick after start-up
            # is only a baseline (delta 0), and a day rollover reset is clamped to 0.
            # Ticks skipped for missing depth are still counted, because their
            # volume shows up in the next delta.
            prev_vol = self._prev_fut_volume.get(index_id)
            vol_delta = max(0, tick.volume - prev_vol) if prev_vol is not None else 0
            self._prev_fut_volume[index_id] = tick.volume

            bar = self._depth_bars.get(index_id)
            if bar is None or bar["minute"] != minute:
                if bar is not None and not self.dry_run:
                    self._flush_depth_bar(index_id, bar)
                bar = {
                    "minute": minute,
                    "samples": 0,
                    "crossed": 0,
                    "spread_sum": 0.0,
                    "bid5_sum": 0,
                    "ask5_sum": 0,
                    "imb_sum": 0.0,
                    "imb_n": 0,
                    "book_imb_sum": 0.0,
                    "book_imb_n": 0,
                    "ofi": 0.0,
                    "volume": 0,
                    "last": tick,
                }
                self._depth_bars[index_id] = bar

            bar["volume"] += vol_delta
            bar["samples"] += 1
            if tick.bid_price > tick.ask_price:
                bar["crossed"] += 1
            bar["spread_sum"] += tick.spread
            bar["bid5_sum"] += tick.bid_qty_5
            bar["ask5_sum"] += tick.ask_qty_5
            if tick.depth_imbalance is not None:
                bar["imb_sum"] += tick.depth_imbalance
                bar["imb_n"] += 1
            if tick.book_imbalance is not None:
                bar["book_imb_sum"] += tick.book_imbalance
                bar["book_imb_n"] += 1
            bar["ofi"] += ofi
            bar["last"] = tick

    def _flush_depth_bar(self, index_id: str, bar: dict) -> None:
        """Write a completed 1-minute depth aggregate to futures_depth_1m."""
        n = bar["samples"]
        if n <= 0:
            return
        last = bar["last"]
        try:
            self.db.execute_many(
                Q.UPSERT_FUTURES_DEPTH_1M,
                [(
                    index_id,
                    bar["minute"].isoformat(),
                    last.token,
                    last.ltp,
                    int(last.open_interest or 0),
                    int(bar["volume"]),
                    n,
                    bar["crossed"],
                    bar["spread_sum"] / n,
                    bar["bid5_sum"] / n,
                    bar["ask5_sum"] / n,
                    bar["imb_sum"] / bar["imb_n"] if bar["imb_n"] else None,
                    last.depth_imbalance,
                    bar["book_imb_sum"] / bar["book_imb_n"] if bar["book_imb_n"] else None,
                    last.book_imbalance,
                    last.total_buy_qty,
                    last.total_sell_qty,
                    bar["ofi"],
                )],
            )
        except Exception as exc:
            logger.error("Depth bar flush failed for %s: %s", index_id, exc)

    # ── Public interface ───────────────────────────────────────────────────

    def get_index_price(self, index_id: str) -> Optional[dict]:
        """Return current price dict for a single index.

        In WebSocket mode returns the latest tick cached in memory.
        Falls back to scraper otherwise.
        """
        if self.ws_feed:
            token_info = _INDEX_TOKENS.get(index_id)
            if token_info:
                token, _ = token_info
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
                        "futures_depth": self.get_futures_depth(index_id),
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

    def get_futures_depth(self, index_id: str) -> Optional[dict]:
        """Latest near-month futures order-book snapshot for an index, or None."""
        tick = self._latest_fut_depth.get(index_id)
        if tick is None:
            return None
        return {
            "index_id": index_id,
            "token": tick.token,
            "fut_ltp": tick.ltp,
            "bid": tick.bid_price,
            "ask": tick.ask_price,
            "spread": tick.spread,
            "bid_qty": tick.bid_qty,
            "ask_qty": tick.ask_qty,
            "bid_qty_5": tick.bid_qty_5,
            "ask_qty_5": tick.ask_qty_5,
            "total_buy_qty": tick.total_buy_qty,
            "total_sell_qty": tick.total_sell_qty,
            "depth_imbalance": tick.depth_imbalance,
            "book_imbalance": tick.book_imbalance,
            "open_interest": tick.open_interest,
            "timestamp": tick.timestamp.isoformat(),
        }

    def get_futures_depth_history(self, index_id: str, start: str, end: str) -> list[dict]:
        """Stored 1-minute depth aggregates for [start, end) — ISO-8601 IST strings."""
        return self.db.fetch_all(Q.LIST_FUTURES_DEPTH_1M, (index_id, start, end))

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

    def get_stale_tokens(self, max_age_seconds: float = 60.0) -> list[str]:
        """Return list of token IDs that haven't received a tick in max_age_seconds.

        Used for per-token stale detection — identifies individual dead feeds
        that would be missed by a global stale check.
        """
        now = time.monotonic()
        stale = []
        for token, last_time in self._last_tick_times.items():
            if now - last_time > max_age_seconds:
                index_id = _TOKEN_TO_INDEX.get(token, token)
                stale.append(index_id)
        return stale

    def check_feed_health(self) -> dict:
        """Return per-token feed health summary.

        Returns dict with: all_healthy (bool), stale_feeds (list), oldest_feed_age_sec.
        """
        if not self.ws_feed:
            return {"all_healthy": True, "stale_feeds": [], "oldest_feed_age_sec": 0}

        now = time.monotonic()
        stale = self.get_stale_tokens(max_age_seconds=60.0)
        ages = [(now - t) for t in self._last_tick_times.values()]
        oldest = max(ages) if ages else 0.0

        return {
            "all_healthy": len(stale) == 0,
            "stale_feeds": stale,
            "oldest_feed_age_sec": round(oldest, 1),
            "active_tokens": len(self._last_tick_times),
            "ws_connected": getattr(self.ws_feed, "is_connected", False),
            "ws_reconnects": getattr(self.ws_feed, "reconnect_count", 0),
        }

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
                with self._depth_lock:
                    for index_id, bar in self._depth_bars.items():
                        self._flush_depth_bar(index_id, bar)
                    self._depth_bars.clear()
            self.ws_feed.stop()
