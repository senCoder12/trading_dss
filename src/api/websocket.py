"""
WebSocket event broadcaster for real-time signal push.

Provides a global ``broadcaster`` singleton that pushes events to all
connected WebSocket clients, and a thread-safe ``queue_event()`` function
that the synchronous decision-engine thread can call without importing
asyncio.

Event types
-----------
- ``signal``        — New actionable trading signal (CRITICAL priority)
- ``position_exit`` — An open position was closed (target / SL / trailing)
- ``anomaly``       — High-severity anomaly detected
- ``system_alert``  — Kill-switch, health changes, etc.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect

from src.utils.date_utils import get_ist_now

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EventBroadcaster — async, used inside the FastAPI event loop
# ---------------------------------------------------------------------------


class EventBroadcaster:
    """Broadcasts JSON events to every connected WebSocket client."""

    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("WebSocket client connected. Total: %d", len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        logger.info("WebSocket client disconnected. Total: %d", len(self._connections))

    async def broadcast(self, event_type: str, data: dict) -> None:
        """Send *event_type* with *data* to all connected clients."""
        message = json.dumps({
            "type": event_type,
            "data": data,
            "timestamp": get_ist_now().isoformat(),
        })

        dead: set[WebSocket] = set()
        async with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.add(ws)
            self._connections -= dead

    @property
    def client_count(self) -> int:
        return len(self._connections)


# Global singleton
broadcaster = EventBroadcaster()

# ---------------------------------------------------------------------------
# Thread-safe event queue (sync → async bridge)
# ---------------------------------------------------------------------------
# The decision engine runs in a synchronous thread.  It cannot directly
# ``await broadcaster.broadcast()``.  Instead it pushes events into this
# list; a background coroutine inside the FastAPI process drains it every
# 500 ms and broadcasts to clients.
# ---------------------------------------------------------------------------

_event_queue: list[tuple[str, dict]] = []
_queue_lock = threading.Lock()


def queue_event(event_type: str, data: dict) -> None:
    """Thread-safe: enqueue an event from any sync thread."""
    with _queue_lock:
        _event_queue.append((event_type, data))
    logger.debug("Event queued: %s (queue size: %d)", event_type, len(_event_queue))


async def process_event_queue() -> None:
    """Async background task — drains the queue and broadcasts.

    Started once at application startup; runs until the process exits.
    """
    logger.info("WebSocket event-queue processor started")
    while True:
        events: list[tuple[str, dict]] = []
        with _queue_lock:
            if _event_queue:
                events = list(_event_queue)
                _event_queue.clear()

        for event_type, data in events:
            try:
                await broadcaster.broadcast(event_type, data)
            except Exception:
                logger.exception("Failed to broadcast queued event: %s", event_type)

        await asyncio.sleep(0.5)
