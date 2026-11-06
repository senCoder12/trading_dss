"""
Angel One SmartWebSocketV2 feed handler.

Pushes live market data to registered callbacks the instant a subscribed
instrument changes.  Runs in a daemon thread; auto-reconnects on disconnect.

This eliminates the 45-second price-polling loop and 180-second options-chain
loop when USE_BROKER_API=True — data arrives in real time instead.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# Reconnect policy (WebSocketFeed owns reconnection; the SDK's retry is disabled)
_MAX_RECONNECT_DELAY = 60.0       # cap for exponential backoff, seconds
_STABLE_CONNECTION_SECS = 60.0    # connection alive this long → backoff resets
_RELOGIN_AFTER_FAILURES = 3       # consecutive short-lived connects → fresh login


@dataclass
class LiveTick:
    """Normalised tick received from the Angel One WebSocket feed."""

    token: str
    exchange_type: int          # 1=NSE, 2=NFO, 3=BSE, 4=BFO
    ltp: float                  # Last traded price (₹)
    volume: int                 # Volume for the day
    open_interest: int          # 0 for equity indices, non-zero for options
    oi_change_pct: float
    bid_price: Optional[float]  # Best bid (None when depth not available)
    ask_price: Optional[float]  # Best ask (None when depth not available)
    open: float
    high: float
    low: float
    close: float                # Previous day close
    timestamp: datetime
    # ── Depth (SnapQuote only; zero when the instrument has no order book,
    #    e.g. spot indices) ─────────────────────────────────────────────────
    bid_qty: int = 0            # Quantity at best bid
    ask_qty: int = 0            # Quantity at best ask
    bid_qty_5: int = 0          # Sum of quantity across top-5 bid levels
    ask_qty_5: int = 0          # Sum of quantity across top-5 ask levels
    total_buy_qty: float = 0.0  # Whole-book pending buy quantity
    total_sell_qty: float = 0.0 # Whole-book pending sell quantity

    @property
    def has_depth(self) -> bool:
        return self.bid_price is not None and self.ask_price is not None

    @property
    def depth_imbalance(self) -> Optional[float]:
        """Top-5 order book imbalance in [-1, 1]; +ve = bid-heavy."""
        total = self.bid_qty_5 + self.ask_qty_5
        if total <= 0:
            return None
        return (self.bid_qty_5 - self.ask_qty_5) / total

    @property
    def book_imbalance(self) -> Optional[float]:
        """Whole-book imbalance in [-1, 1] from total buy/sell quantity."""
        total = self.total_buy_qty + self.total_sell_qty
        if total <= 0:
            return None
        return (self.total_buy_qty - self.total_sell_qty) / total

    @property
    def spread(self) -> Optional[float]:
        if not self.has_depth:
            return None
        return self.ask_price - self.bid_price


def _parse_depth_side(levels: Optional[list], descending: bool) -> list[tuple[float, int]]:
    """Convert SmartWebSocketV2 depth packets to [(price_rupees, qty)], best first.

    Empty levels (price or quantity 0) are dropped.  Prices arrive in paise.
    """
    if not levels:
        return []
    out = [
        (lvl.get("price", 0) / 100, int(lvl.get("quantity", 0)))
        for lvl in levels
        if lvl.get("price", 0) > 0 and lvl.get("quantity", 0) > 0
    ]
    out.sort(key=lambda pq: pq[0], reverse=descending)
    return out


@lru_cache(maxsize=None)
def _make_ws_class():
    """SmartWebSocketV2 subclass patched for websocket-client >= 1.0.

    smartapi-python 1.5.5 has two bugs that leave the feed dead after the
    first disconnect:

    * ``_on_close(self, wsapp)`` — websocket-client >= 1.0 calls on_close with
      ``(ws, close_status_code, close_msg)``, so the SDK handler raises
      TypeError and the user's on_close never fires.
    * ``_on_error`` reconnects by calling ``connect()`` from inside the
      callback (nested ``run_forever``), retries once by default and never
      resets its retry counter.

    This subclass forwards both callbacks without reconnecting.
    ``run_forever`` returns on disconnect and WebSocketFeed's supervisor loop
    opens a fresh connection.
    """
    try:
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "smartapi-python not installed — run: pip install smartapi-python pyotp"
        ) from exc

    class _FeedSocket(SmartWebSocketV2):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.wsapp = None  # SDK only sets this in connect(); close_connection() needs it

        def _on_close(self, wsapp, *args) -> None:
            self.on_close(wsapp, *args)

        def _on_error(self, wsapp, error) -> None:
            self.on_error(wsapp, error)

    return _FeedSocket


class WebSocketFeed:
    """Real-time WebSocket data feed from Angel One SmartAPI.

    Usage
    -----
    feed = WebSocketFeed(client)
    feed.on_tick_update = my_handler
    feed.start([{"exchangeType": 1, "tokens": ["3045", "26000"]}])
    ...
    feed.stop()

    Subscription modes (passed to SmartWebSocketV2.subscribe):
      1 = LTP only
      2 = Quote (OHLC + volume)
      3 = SnapQuote (Quote + depth + OI) ← used here
    """

    def __init__(self, client) -> None:
        self.client = client
        self.ws = None
        self.is_running = False
        self._tokens_to_subscribe: list[dict] = []
        self._thread: Optional[threading.Thread] = None

        # Callbacks — register before calling start()
        self.on_tick_update: Optional[Callable[[LiveTick], None]] = None
        self.on_depth_update: Optional[Callable[[LiveTick], None]] = None
        self.on_oi_update: Optional[Callable[[LiveTick], None]] = None
        self.on_error: Optional[Callable] = None

        # Thread-safe latest-value cache
        self._latest_data: dict[str, LiveTick] = {}
        self._lock = threading.Lock()

        # Depth sanity counters (crossed book = best bid > best ask)
        self.depth_ticks = 0
        self.crossed_depth_ticks = 0

        # Connection supervision
        self._stop_event = threading.Event()
        self.is_connected = False
        self.reconnect_count = 0
        self._connected_at: Optional[float] = None
        self.reconnect_base_delay = float(settings.broker.websocket_reconnect_delay)
        self.reconnect_max_delay = _MAX_RECONNECT_DELAY

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self, tokens_to_subscribe: list[dict]) -> None:
        """Open WebSocket and subscribe to instruments.

        tokens_to_subscribe format:
            [
                {"exchangeType": 1, "tokens": ["3045"]},    # NSE
                {"exchangeType": 2, "tokens": ["35003"]},   # NFO
            ]
        exchangeType: 1=NSE, 2=NFO, 3=BSE, 4=BFO

        Returns immediately; a supervisor thread keeps exactly one connection
        open and reconnects with exponential backoff until stop().
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("WebSocket feed already running — start() ignored")
            return

        self._ws_class = _make_ws_class()  # raise here, not in the thread, if SDK missing
        self._tokens_to_subscribe = tokens_to_subscribe
        self._stop_event.clear()
        self.is_running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="WebSocketFeed")
        self._thread.start()
        logger.info(
            "WebSocket feed started, subscribing to %d instrument group(s)",
            len(tokens_to_subscribe),
        )

    def stop(self) -> None:
        """Disconnect and prevent reconnection."""
        self.is_running = False
        self._stop_event.set()
        ws = self.ws
        if ws:
            try:
                ws.close_connection()
            except Exception:
                pass

    def _run(self) -> None:
        """Supervisor loop: connect, block until disconnect, back off, repeat."""
        delay = self.reconnect_base_delay
        short_lived = 0  # consecutive connections that failed or died quickly

        while not self._stop_event.is_set():
            self._connected_at = None
            try:
                if short_lived >= _RELOGIN_AFTER_FAILURES:
                    logger.warning(
                        "WebSocket: %d consecutive failed connections — forcing fresh login",
                        short_lived,
                    )
                    self.client.login()
                    short_lived = 0
                else:
                    self.client.ensure_authenticated()

                ws = self._ws_class(
                    auth_token=self.client.auth_token,
                    api_key=self.client.api_key,
                    client_code=self.client.client_id,
                    feed_token=self.client.feed_token,
                )
                ws.on_open = self._on_open
                ws.on_data = self._on_data
                ws.on_error = self._on_error
                ws.on_close = self._on_close
                self.ws = ws
                if self._stop_event.is_set():
                    break
                ws.connect()  # blocks until this connection ends
            except Exception as exc:
                logger.error("WebSocket: connection attempt failed: %s", exc)
            finally:
                self.is_connected = False

            if self._stop_event.is_set():
                break

            opened = self._connected_at is not None
            uptime = (time.monotonic() - self._connected_at) if opened else 0.0
            if opened and uptime >= _STABLE_CONNECTION_SECS:
                delay = self.reconnect_base_delay
                short_lived = 0
            else:
                short_lived += 1

            self.reconnect_count += 1
            logger.warning(
                "WebSocket: disconnected after %.0fs — reconnecting in %.1fs (reconnect #%d)",
                uptime, delay, self.reconnect_count,
            )
            if self._stop_event.wait(delay):
                break
            delay = min(delay * 2, self.reconnect_max_delay)

        logger.info("WebSocket: supervisor stopped")

    # ── WebSocket event handlers ───────────────────────────────────────────

    def _on_open(self, ws) -> None:
        self.is_connected = True
        self._connected_at = time.monotonic()
        if self._stop_event.is_set():
            self.ws.close_connection()
            return
        logger.info("WebSocket: connection opened")
        # Mode 3 = SnapQuote: full depth, OI, IV for options
        self.ws.subscribe("trading_dss_feed", 3, self._tokens_to_subscribe)
        logger.info("WebSocket: subscribed (SnapQuote mode)")

    def _on_data(self, ws, message) -> None:
        """Fired on every price/OI/depth change for any subscribed instrument."""
        try:
            token = str(message.get("token", ""))
            # SmartWebSocketV2 emits "best_5_buy_data" / "best_5_sell_data"
            # (already side-corrected by the SDK) in SnapQuote mode only.
            bids = _parse_depth_side(message.get("best_5_buy_data"), descending=True)
            asks = _parse_depth_side(message.get("best_5_sell_data"), descending=False)
            tick = LiveTick(
                token=token,
                exchange_type=message.get("exchange_type", 0),
                ltp=message.get("last_traded_price", 0) / 100,
                volume=message.get("volume_trade_for_the_day", 0),
                open_interest=message.get("open_interest", 0),
                oi_change_pct=float(message.get("open_interest_change_percentage", 0.0)),
                bid_price=bids[0][0] if bids else None,
                ask_price=asks[0][0] if asks else None,
                open=message.get("open_price_of_the_day", 0) / 100,
                high=message.get("high_price_of_the_day", 0) / 100,
                low=message.get("low_price_of_the_day", 0) / 100,
                close=message.get("closed_price", 0) / 100,
                timestamp=datetime.now(tz=_IST),
                bid_qty=bids[0][1] if bids else 0,
                ask_qty=asks[0][1] if asks else 0,
                bid_qty_5=sum(q for _, q in bids),
                ask_qty_5=sum(q for _, q in asks),
                total_buy_qty=float(message.get("total_buy_quantity", 0) or 0),
                total_sell_qty=float(message.get("total_sell_quantity", 0) or 0),
            )
            if tick.has_depth:
                self._check_depth_sanity(tick)

            with self._lock:
                self._latest_data[token] = tick

            if self.on_tick_update:
                self.on_tick_update(tick)
            if tick.open_interest > 0 and self.on_oi_update:
                self.on_oi_update(tick)
            if tick.bid_price is not None and self.on_depth_update:
                self.on_depth_update(tick)

        except Exception as exc:
            logger.error("WebSocket data processing error: %s", exc)

    def _check_depth_sanity(self, tick: LiveTick) -> None:
        """Count crossed books (best bid > best ask).

        In continuous trading a crossed snapshot means bid/ask sides are
        mislabelled.  Pre-open auction can legitimately cross, so this only
        logs — it never mutates the tick.
        """
        self.depth_ticks += 1
        if tick.bid_price > tick.ask_price:
            self.crossed_depth_ticks += 1
            if self.crossed_depth_ticks == 1 or self.crossed_depth_ticks % 500 == 0:
                logger.warning(
                    "WebSocket depth crossed: token=%s bid=%.2f ask=%.2f "
                    "(%d/%d depth ticks crossed) — check bid/ask side mapping",
                    tick.token, tick.bid_price, tick.ask_price,
                    self.crossed_depth_ticks, self.depth_ticks,
                )

    def _on_error(self, ws, error) -> None:
        # Only reports. Real disconnects make run_forever return, and _run reconnects.
        logger.error("WebSocket error: %s", error)
        if self.on_error:
            self.on_error(error)

    def _on_close(self, ws, close_status_code=None, close_msg=None) -> None:
        self.is_connected = False
        logger.warning("WebSocket closed: %s %s", close_status_code, close_msg)

    # ── Data access ────────────────────────────────────────────────────────

    def get_latest(self, token: str) -> Optional[LiveTick]:
        with self._lock:
            return self._latest_data.get(token)

    def get_all_latest(self) -> dict[str, LiveTick]:
        with self._lock:
            return dict(self._latest_data)
