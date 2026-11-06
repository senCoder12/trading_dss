"""
Tests for WebSocketFeed connection supervision: reconnect after disconnect,
re-subscribe on every new connection, exponential backoff, forced re-login
after repeated failures, and the SmartWebSocketV2 compatibility subclass.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

import src.data.broker.websocket_feed as wf
from src.data.broker.websocket_feed import WebSocketFeed

TOKENS = [{"exchangeType": 2, "tokens": ["50000"]}]


def _wait_until(cond, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def _fake_socket_class(script: list[str]):
    """Fake SmartWebSocketV2. Each connect() consumes one scripted behaviour:

    "drop"  — open, then disconnect immediately
    "fail"  — handshake failure (raises)
    "stay"  — open and block until close_connection()
    Script exhausted → behaves like "stay".
    """
    instances: list = []

    class FakeSocket:
        def __init__(self, auth_token, api_key, client_code, feed_token):
            self.subscribed: list = []
            self._closed = threading.Event()
            instances.append(self)

        def subscribe(self, correlation_id, mode, token_list):
            self.subscribed.append((mode, token_list))

        def close_connection(self):
            self._closed.set()

        def connect(self):
            behaviour = script.pop(0) if script else "stay"
            if behaviour == "fail":
                raise Exception("Handshake status 429 Too Many Requests")
            self.on_open("wsapp")
            if behaviour == "drop":
                self.on_close("wsapp", 1006, "connection dropped")
                return
            self._closed.wait(5)
            self.on_close("wsapp", 1000, "closed by client")

    return FakeSocket, instances


@pytest.fixture()
def make_feed(monkeypatch):
    feeds: list[WebSocketFeed] = []

    def _make(script: list[str], base_delay: float = 0.01, max_delay: float = 0.05):
        cls, instances = _fake_socket_class(script)
        monkeypatch.setattr(wf, "_make_ws_class", lambda: cls)
        feed = WebSocketFeed(client=MagicMock())
        feed.reconnect_base_delay = base_delay
        feed.reconnect_max_delay = max_delay
        feeds.append(feed)
        return feed, instances

    yield _make
    for feed in feeds:
        feed.stop()
        if feed._thread:
            feed._thread.join(2)


def test_reconnects_and_resubscribes_after_drop(make_feed) -> None:
    feed, sockets = make_feed(["drop", "drop", "stay"])
    feed.start(TOKENS)

    assert _wait_until(lambda: len(sockets) == 3 and sockets[2].subscribed)
    assert all(s.subscribed == [(3, TOKENS)] for s in sockets)
    assert feed.reconnect_count == 2
    assert _wait_until(lambda: feed.is_connected)

    feed.stop()
    feed._thread.join(2)
    assert not feed._thread.is_alive()
    assert not feed.is_connected


def test_forces_login_after_repeated_failures(make_feed) -> None:
    feed, sockets = make_feed(["fail", "fail", "fail", "stay"])
    feed.start(TOKENS)

    assert _wait_until(lambda: len(sockets) == 4 and sockets[3].subscribed)
    feed.client.login.assert_called_once()
    assert feed.client.ensure_authenticated.call_count == 3


def test_backoff_doubles_and_caps(make_feed) -> None:
    feed, _ = make_feed(["fail"] * 10, base_delay=1.0, max_delay=5.0)
    delays: list[float] = []

    class RecordingEvent(threading.Event):
        def wait(self, timeout=None):
            delays.append(timeout)
            if len(delays) >= 5:
                self.set()
                return True
            return False

    feed._stop_event = RecordingEvent()
    feed.start(TOKENS)
    feed._thread.join(2)

    assert delays == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_stable_connection_resets_backoff(make_feed, monkeypatch) -> None:
    feed, _ = make_feed(["fail", "fail", "drop", "fail"], base_delay=1.0, max_delay=60.0)
    monkeypatch.setattr(wf, "_STABLE_CONNECTION_SECS", 0.0)  # every opened connection counts as stable
    delays: list[float] = []

    class RecordingEvent(threading.Event):
        def wait(self, timeout=None):
            delays.append(timeout)
            if len(delays) >= 4:
                self.set()
                return True
            return False

    feed._stop_event = RecordingEvent()
    feed.start(TOKENS)
    feed._thread.join(2)

    # fail → 1, fail → 2, stable drop → reset to 1, fail → 2
    assert delays == [1.0, 2.0, 1.0, 2.0]


def test_stop_interrupts_backoff_wait(make_feed) -> None:
    feed, sockets = make_feed(["fail"] * 5, base_delay=30.0, max_delay=30.0)
    feed.start(TOKENS)
    assert _wait_until(lambda: len(sockets) == 1)

    started = time.monotonic()
    feed.stop()
    feed._thread.join(2)
    assert not feed._thread.is_alive()
    assert time.monotonic() - started < 1.0


def test_second_start_is_ignored(make_feed) -> None:
    feed, sockets = make_feed(["stay"])
    feed.start(TOKENS)
    assert _wait_until(lambda: len(sockets) == 1 and sockets[0].subscribed)
    first_thread = feed._thread

    feed.start(TOKENS)

    assert feed._thread is first_thread
    assert len(sockets) == 1


# ── Real SDK subclass compatibility ───────────────────────────────────────────


def test_sdk_subclass_close_and_error_compat(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # SDK __init__ creates logs/<date>/app.log in CWD
    cls = wf._make_ws_class()
    sock = cls("auth", "key", "client", "feed")

    closes: list = []
    errors: list = []
    sock.on_close = lambda *args: closes.append(args)
    sock.on_error = lambda *args: errors.append(args)
    sock.connect = MagicMock()

    # websocket-client >= 1.0 calls on_close with 3 args — must not raise
    sock._on_close("wsapp", 1000, "bye")
    assert closes == [("wsapp", 1000, "bye")]

    # Errors are forwarded; the SDK's nested reconnect must not run
    err = Exception("boom")
    sock._on_error("wsapp", err)
    assert errors == [("wsapp", err)]
    sock.connect.assert_not_called()

    # close before connect() must not raise
    sock.close_connection()
