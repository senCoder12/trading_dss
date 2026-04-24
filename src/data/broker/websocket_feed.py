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
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)


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

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self, tokens_to_subscribe: list[dict]) -> None:
        """Open WebSocket and subscribe to instruments.

        tokens_to_subscribe format:
            [
                {"exchangeType": 1, "tokens": ["3045"]},    # NSE
                {"exchangeType": 2, "tokens": ["35003"]},   # NFO
            ]
        exchangeType: 1=NSE, 2=NFO, 3=BSE, 4=BFO
        """
        self.client.ensure_authenticated()
        self._tokens_to_subscribe = tokens_to_subscribe

        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "smartapi-python not installed — run: pip install smartapi-python pyotp"
            ) from exc

        self.ws = SmartWebSocketV2(
            auth_token=self.client.auth_token,
            api_key=self.client.api_key,
            client_code=self.client.client_id,
            feed_token=self.client.feed_token,
        )
        self.ws.on_open = self._on_open
        self.ws.on_data = self._on_data
        self.ws.on_error = self._on_error
        self.ws.on_close = self._on_close

        self.is_running = True
        self._thread = threading.Thread(
            target=self.ws.connect,
            daemon=True,
            name="WebSocketFeed",
        )
        self._thread.start()
        logger.info(
            "WebSocket feed started, subscribing to %d instrument group(s)",
            len(tokens_to_subscribe),
        )

    def stop(self) -> None:
        """Disconnect and prevent reconnection."""
        self.is_running = False
        if self.ws:
            try:
                self.ws.close_connection()
            except Exception:
                pass

    # ── WebSocket event handlers ───────────────────────────────────────────

    def _on_open(self, ws) -> None:
        logger.info("WebSocket: connection opened")
        # Mode 3 = SnapQuote: full depth, OI, IV for options
        self.ws.subscribe("trading_dss_feed", 3, self._tokens_to_subscribe)
        logger.info("WebSocket: subscribed (SnapQuote mode)")

    def _on_data(self, ws, message) -> None:
        """Fired on every price/OI/depth change for any subscribed instrument."""
        try:
            token = str(message.get("token", ""))
            tick = LiveTick(
                token=token,
                exchange_type=message.get("exchange_type", 0),
                ltp=message.get("last_traded_price", 0) / 100,
                volume=message.get("volume_trade_for_the_day", 0),
                open_interest=message.get("open_interest", 0),
                oi_change_pct=float(message.get("open_interest_change_percentage", 0.0)),
                bid_price=(
                    message["best_5_buy"][0]["price"] / 100
                    if message.get("best_5_buy")
                    else None
                ),
                ask_price=(
                    message["best_5_sell"][0]["price"] / 100
                    if message.get("best_5_sell")
                    else None
                ),
                open=message.get("open_price_of_the_day", 0) / 100,
                high=message.get("high_price_of_the_day", 0) / 100,
                low=message.get("low_price_of_the_day", 0) / 100,
                close=message.get("closed_price", 0) / 100,
                timestamp=datetime.now(tz=_IST),
            )

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

    def _on_error(self, ws, error) -> None:
        logger.error("WebSocket error: %s", error)
        if self.on_error:
            self.on_error(error)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        logger.warning("WebSocket closed: %s %s", close_status_code, close_msg)
        if self.is_running:
            delay = settings.broker.websocket_reconnect_delay
            logger.info("WebSocket: reconnecting in %ds…", delay)
            time.sleep(delay)
            self.start(self._tokens_to_subscribe)

    # ── Data access ────────────────────────────────────────────────────────

    def get_latest(self, token: str) -> Optional[LiveTick]:
        with self._lock:
            return self._latest_data.get(token)

    def get_all_latest(self) -> dict[str, LiveTick]:
        with self._lock:
            return dict(self._latest_data)
