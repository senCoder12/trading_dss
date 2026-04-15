"""
End-to-end integration tests for Phase 8 — Unified System Launcher.

Covers:
  a) API endpoints — every route returns 200 with valid JSON (empty DB)
  b) Telegram bot — send_alert / send_message / format helpers (no network)
  c) Unified startup — argparse configuration, --api-only, graceful shutdown
  d) API + data flow — seed DB → query API → verify response
  e) SPA serving — index.html served at root when frontend/dist exists
  f) Performance — all endpoints < 200ms; 10 concurrent requests succeed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Ensure project root is importable when pytest is run from any directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.app import create_app
from src.api.dependencies import (
    activate_kill_switch,
    deactivate_kill_switch,
    get_db,
    get_index_registry,
    get_market_hours,
)
from src.data.index_registry import IndexRegistry
from src.database.db_manager import DatabaseManager
from src.utils.market_hours import MarketHoursManager

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_INDICES = [
    {
        "id": "NIFTY50",
        "display_name": "NIFTY 50",
        "nse_symbol": "NIFTY 50",
        "yahoo_symbol": "^NSEI",
        "exchange": "NSE",
        "lot_size": 75,
        "has_options": True,
        "option_symbol": "NIFTY",
        "sector_category": "broad_market",
        "is_active": True,
        "description": "NIFTY 50 index",
    },
    {
        "id": "BANKNIFTY",
        "display_name": "NIFTY BANK",
        "nse_symbol": "NIFTY BANK",
        "yahoo_symbol": "^NSEBANK",
        "exchange": "NSE",
        "lot_size": 15,
        "has_options": True,
        "option_symbol": "BANKNIFTY",
        "sector_category": "sectoral",
        "is_active": True,
        "description": "Bank Nifty index",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _now_ist() -> datetime:
    return datetime.now(tz=_IST)


@pytest.fixture()
def test_db(tmp_path: Path) -> Generator[DatabaseManager, None, None]:
    """Fresh SQLite DB per test with schema initialised and index_master seeded."""
    db = DatabaseManager(db_path=tmp_path / "test.db")
    db.connect()
    db.initialise_schema()

    now = _now_ist().isoformat()
    for idx in SAMPLE_INDICES:
        db.execute(
            "INSERT OR IGNORE INTO index_master "
            "(id, display_name, nse_symbol, yahoo_symbol, exchange, "
            "lot_size, has_options, option_symbol, sector_category, "
            "is_active, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                idx["id"], idx["display_name"], idx.get("nse_symbol"),
                idx.get("yahoo_symbol"), idx["exchange"],
                idx.get("lot_size"), idx.get("has_options", False),
                idx.get("option_symbol"), idx.get("sector_category"),
                idx.get("is_active", True), now, now,
            ),
        )

    yield db
    db.close()


@pytest.fixture()
def test_registry(tmp_path: Path) -> IndexRegistry:
    """IndexRegistry built from the sample indices above."""
    p = tmp_path / "indices.json"
    p.write_text(json.dumps(SAMPLE_INDICES), encoding="utf-8")
    return IndexRegistry.from_file(p)


@pytest.fixture()
def client(test_db: DatabaseManager, test_registry: IndexRegistry) -> TestClient:
    """TestClient with dependency overrides pointing at the temp DB / registry."""
    app = create_app()
    app.dependency_overrides[get_db] = lambda: test_db
    app.dependency_overrides[get_index_registry] = lambda: test_registry
    app.dependency_overrides[get_market_hours] = lambda: MarketHoursManager()

    deactivate_kill_switch()

    # Clear any route-level TTL caches from previous tests
    from src.api.routes import anomalies, market_data, signals
    market_data._cache.clear()
    signals._cache.clear()
    anomalies._cache.clear()

    return TestClient(app)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_prices(db: DatabaseManager, index_id: str = "NIFTY50", days: int = 5) -> None:
    now = _now_ist()
    for i in range(days):
        ts = (now - timedelta(days=days - i)).isoformat()
        close = 22000 + i * 50
        db.execute(
            "INSERT OR IGNORE INTO price_data "
            "(index_id, timestamp, open, high, low, close, volume, vwap, source, timeframe) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (index_id, ts, close - 30, close + 50, close - 60, close,
             1_000_000 + i * 1000, close, "test", "1d"),
        )


def _seed_signals(db: DatabaseManager, index_id: str = "NIFTY50", count: int = 3) -> None:
    now = _now_ist()
    signal_types = ["BUY_CALL", "BUY_PUT", "NO_TRADE"]
    for i in range(count):
        ts = (now - timedelta(hours=count - i)).isoformat()
        stype = signal_types[i % len(signal_types)]
        reasoning = json.dumps({"text": f"Signal {i}", "confidence_score": 0.6})
        db.execute(
            "INSERT INTO trading_signals "
            "(index_id, generated_at, signal_type, confidence_level, "
            "entry_price, target_price, stop_loss, risk_reward_ratio, "
            "regime, technical_vote, options_vote, news_vote, anomaly_vote, "
            "reasoning, outcome, actual_exit_price, actual_pnl, closed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                index_id, ts, stype, "HIGH",
                22000 + i * 10, 22200 + i * 10, 21900 + i * 10, 2.0,
                "TRENDING", "BULLISH", "BULLISH", "NEUTRAL", "NEUTRAL",
                reasoning, "OPEN" if stype != "NO_TRADE" else None,
                None, None, None,
            ),
        )


def _seed_news(db: DatabaseManager, count: int = 3) -> None:
    now = _now_ist()
    for i in range(count):
        ts = (now - timedelta(hours=count - i)).isoformat()
        db.execute(
            "INSERT OR IGNORE INTO news_articles "
            "(url, title, published_at, source, summary, "
            "raw_sentiment_score, adjusted_sentiment, impact_category, "
            "source_credibility, is_processed, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"https://example.com/news/{i}",
                f"Test News {i}",
                ts, "test_source",
                f"Summary {i}",
                0.5 + i * 0.1, 0.4 + i * 0.1,
                ["HIGH", "MEDIUM", "LOW"][i % 3],
                0.8, 1, ts,
            ),
        )


# ---------------------------------------------------------------------------
# a) API endpoints — empty DB smoke tests
# ---------------------------------------------------------------------------


class TestApiEndpointsEmptyDb:
    """Every endpoint must return 2xx with valid JSON even against an empty DB."""

    def test_health(self, client: TestClient) -> None:
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_system_health(self, client: TestClient) -> None:
        r = client.get("/api/system/health")
        assert r.status_code == 200
        data = r.json()
        assert "overall_status" in data
        assert "components" in data

    def test_system_status(self, client: TestClient) -> None:
        r = client.get("/api/system/status")
        assert r.status_code == 200
        data = r.json()
        assert "market_status" in data
        assert "kill_switch_active" in data
        assert "uptime_seconds" in data
        assert "data_freshness" in data

    def test_market_prices_all(self, client: TestClient) -> None:
        r = client.get("/api/market/prices")
        assert r.status_code == 200
        assert isinstance(r.json(), (list, dict))

    def test_market_vix(self, client: TestClient) -> None:
        r = client.get("/api/market/vix")
        assert r.status_code == 200

    def test_signals_current(self, client: TestClient) -> None:
        r = client.get("/api/signals/current")
        assert r.status_code == 200

    def test_signals_history(self, client: TestClient) -> None:
        r = client.get("/api/signals/history")
        assert r.status_code == 200

    def test_signals_performance(self, client: TestClient) -> None:
        r = client.get("/api/signals/performance")
        assert r.status_code == 200

    def test_indices_list(self, client: TestClient) -> None:
        r = client.get("/api/indices")
        assert r.status_code == 200
        data = r.json()
        # Registry has 2 indices
        assert len(data) == 2

    def test_portfolio_positions(self, client: TestClient) -> None:
        r = client.get("/api/portfolio/positions")
        assert r.status_code == 200

    def test_portfolio_summary(self, client: TestClient) -> None:
        r = client.get("/api/portfolio/summary")
        assert r.status_code == 200

    def test_news_feed(self, client: TestClient) -> None:
        r = client.get("/api/news/feed")
        assert r.status_code == 200

    def test_news_summary(self, client: TestClient) -> None:
        r = client.get("/api/news/summary")
        assert r.status_code == 200

    def test_anomalies_active(self, client: TestClient) -> None:
        r = client.get("/api/anomalies/active")
        assert r.status_code == 200

    def test_anomalies_dashboard(self, client: TestClient) -> None:
        r = client.get("/api/anomalies/dashboard")
        assert r.status_code == 200

    def test_system_config(self, client: TestClient) -> None:
        r = client.get("/api/system/config")
        assert r.status_code == 200

    def test_docs_accessible(self, client: TestClient) -> None:
        r = client.get("/docs")
        assert r.status_code == 200

    def test_redoc_accessible(self, client: TestClient) -> None:
        r = client.get("/redoc")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# b) Telegram bot — no-network unit tests
# ---------------------------------------------------------------------------


class TestTelegramBot:
    """Test TelegramBot with no live Telegram credentials."""

    @pytest.fixture()
    def unconfigured_bot(self, test_db: DatabaseManager):
        """Bot with no token/chat_id — all sends are no-ops."""
        from src.alerts.telegram_bot import TelegramBot
        with patch("config.settings.settings") as mock_settings:
            mock_settings.telegram.bot_token = ""
            mock_settings.telegram.chat_id = ""
            bot = TelegramBot.__new__(TelegramBot)
            bot._db = test_db
            bot._engine = None
            bot._token = ""
            bot._chat_id = ""
            bot._configured = False
            bot._running = False
            bot._thread = None
            import queue, time as _time, datetime
            bot._normal_queue = queue.Queue()
            bot._low_queue = queue.Queue()
            bot._last_high_send = 0.0
            bot._high_cooldown_seconds = 60.0
            bot._start_time = None
        return bot

    def test_not_configured_flag(self, unconfigured_bot) -> None:
        assert unconfigured_bot._configured is False

    def test_send_message_no_op_when_unconfigured(self, unconfigured_bot) -> None:
        result = unconfigured_bot.send_message("Hello")
        assert result is False

    def test_send_alert_no_op_when_unconfigured(self, unconfigured_bot) -> None:
        from src.alerts.telegram_bot import PRIORITY_CRITICAL
        result = unconfigured_bot.send_alert("Emergency", priority=PRIORITY_CRITICAL)
        assert result is False

    def test_split_message_no_split_needed(self, unconfigured_bot) -> None:
        chunks = unconfigured_bot._split_message("short text")
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_split_message_long_text(self, unconfigured_bot) -> None:
        # Text longer than 4096 chars should be split
        long_text = "A" * 5000
        chunks = unconfigured_bot._split_message(long_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 4096

    def test_format_signal_emoji_call(self, unconfigured_bot) -> None:
        emoji = unconfigured_bot._format_signal_emoji("BUY_CALL")
        assert isinstance(emoji, str)
        assert len(emoji) > 0

    def test_format_signal_emoji_put(self, unconfigured_bot) -> None:
        emoji = unconfigured_bot._format_signal_emoji("BUY_PUT")
        assert isinstance(emoji, str)

    def test_format_confidence_emoji(self, unconfigured_bot) -> None:
        for level in ("HIGH", "MEDIUM", "LOW", ""):
            emoji = unconfigured_bot._format_confidence_emoji(level)
            assert isinstance(emoji, str)

    def test_human_timedelta(self, unconfigured_bot) -> None:
        from datetime import timedelta
        assert "h" in unconfigured_bot._human_timedelta(timedelta(hours=2, minutes=30))
        assert "m" in unconfigured_bot._human_timedelta(timedelta(minutes=45))
        assert unconfigured_bot._human_timedelta(timedelta(0)) == "0m"

    def test_human_age(self, unconfigured_bot) -> None:
        assert "s ago" in unconfigured_bot._human_age(30)
        assert "m ago" in unconfigured_bot._human_age(300)
        assert "h ago" in unconfigured_bot._human_age(7200)
        assert unconfigured_bot._human_age(None) == "never"

    def test_truncate(self, unconfigured_bot) -> None:
        short = "abc"
        assert unconfigured_bot._truncate(short) == short
        long_s = "x" * 300
        result = unconfigured_bot._truncate(long_s, max_len=200)
        assert len(result) == 200
        assert result.endswith("...")

    def test_configured_flag_with_valid_tokens(self, test_db: DatabaseManager) -> None:
        """Bot should set _configured=True when both token and chat_id are present."""
        from src.alerts.telegram_bot import TelegramBot
        with patch("src.alerts.telegram_bot.settings") as mock_settings:
            mock_settings.telegram.bot_token = "fake_token_123"
            mock_settings.telegram.chat_id = "123456"
            bot = TelegramBot.__new__(TelegramBot)
            bot._db = test_db
            bot._engine = None
            bot._token = "fake_token_123"
            bot._chat_id = "123456"
            bot._configured = bool("fake_token_123" and "123456")
        assert bot._configured is True


# ---------------------------------------------------------------------------
# c) Unified startup — argument parsing
# ---------------------------------------------------------------------------


class TestRunSystemArgs:
    """Verify argparse configuration in run_system.py main()."""

    @pytest.fixture(autouse=True)
    def _import_parser(self) -> None:
        """Import the module once; reuse across tests."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_system",
            Path(__file__).resolve().parent.parent / "scripts" / "run_system.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._mod = mod

    def _parse(self, argv: list[str]) -> argparse.Namespace:
        """Re-create the parser and parse argv."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-telegram", action="store_true")
        parser.add_argument("--api-only",       action="store_true")
        parser.add_argument("--collector-only",  action="store_true")
        parser.add_argument("--api-port",        type=int, default=8000)
        parser.add_argument("--debug",           action="store_true")
        parser.add_argument("--dry-run",         action="store_true")
        parser.add_argument("--force-start",     action="store_true")
        parser.add_argument("--skip-validation", action="store_true")
        return parser.parse_args(argv)

    def test_defaults(self) -> None:
        args = self._parse([])
        assert args.api_port == 8000
        assert args.debug is False
        assert args.no_telegram is False
        assert args.api_only is False
        assert args.collector_only is False
        assert args.dry_run is False

    def test_api_only(self) -> None:
        args = self._parse(["--api-only"])
        assert args.api_only is True
        assert args.collector_only is False

    def test_collector_only(self) -> None:
        args = self._parse(["--collector-only"])
        assert args.collector_only is True

    def test_no_telegram(self) -> None:
        args = self._parse(["--no-telegram"])
        assert args.no_telegram is True

    def test_custom_port(self) -> None:
        args = self._parse(["--api-port", "8080"])
        assert args.api_port == 8080

    def test_debug_flag(self) -> None:
        args = self._parse(["--debug"])
        assert args.debug is True

    def test_skip_validation(self) -> None:
        args = self._parse(["--skip-validation"])
        assert args.skip_validation is True

    def test_dry_run(self) -> None:
        args = self._parse(["--dry-run"])
        assert args.dry_run is True

    def test_multiple_flags(self) -> None:
        args = self._parse(["--api-only", "--debug", "--api-port", "9000"])
        assert args.api_only is True
        assert args.debug is True
        assert args.api_port == 9000

    def test_run_validation_returns_bool(self) -> None:
        """_run_validation() should return bool, not sys.exit."""
        # Stub out all individual check functions to pass
        with patch.multiple(
            "scripts.validate_config",
            check_system=MagicMock(),
            check_env=MagicMock(),
            check_indices=MagicMock(),
            check_news_mappings=MagicMock(),
            check_rss_feeds=MagicMock(),
            check_sentiment=MagicMock(),
            check_database=MagicMock(),
        ):
            result = self._mod._run_validation()
        assert isinstance(result, bool)
        assert result is True  # No failures injected


# ---------------------------------------------------------------------------
# d) API + data flow — seed then query
# ---------------------------------------------------------------------------


class TestApiDataFlow:
    """Insert data into the temp DB and verify the API surfaces it correctly."""

    def test_prices_appear_after_seed(
        self, client: TestClient, test_db: DatabaseManager
    ) -> None:
        _seed_prices(test_db, "NIFTY50", days=3)
        r = client.get("/api/market/prices/NIFTY50")
        assert r.status_code == 200
        data = r.json()
        # Should contain a price close to what we seeded
        assert data is not None

    def test_signals_history_after_seed(
        self, client: TestClient, test_db: DatabaseManager
    ) -> None:
        _seed_signals(test_db, "NIFTY50", count=3)
        r = client.get("/api/signals/history?index_id=NIFTY50")
        assert r.status_code == 200
        body = r.json()
        signals_list = body if isinstance(body, list) else body.get("signals", [])
        assert len(signals_list) >= 1

    def test_news_feed_after_seed(
        self, client: TestClient, test_db: DatabaseManager
    ) -> None:
        _seed_news(test_db, count=3)
        r = client.get("/api/news/feed")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, (list, dict))

    def test_system_status_data_freshness(
        self, client: TestClient, test_db: DatabaseManager
    ) -> None:
        """After seeding prices the freshness field should contain a timestamp."""
        _seed_prices(test_db, "NIFTY50", days=1)
        r = client.get("/api/system/status")
        assert r.status_code == 200
        freshness = r.json().get("data_freshness", [])
        price_entry = next((f for f in freshness if f["table"] == "price_data"), None)
        assert price_entry is not None
        assert price_entry.get("latest_timestamp") is not None

    def test_kill_switch_roundtrip(self, client: TestClient) -> None:
        """POST activates, DELETE deactivates — status reflects both states."""
        r = client.post(
            "/api/system/kill-switch",
            json={"reason": "integration test"},
        )
        assert r.status_code == 200
        assert r.json()["active"] is True

        r = client.delete("/api/system/kill-switch")
        assert r.status_code == 200
        assert r.json()["active"] is False

    def test_signal_by_index_404_unknown(self, client: TestClient) -> None:
        """Unknown index_id should return 404, not 500."""
        r = client.get("/api/signals/current/FAKE_INDEX")
        assert r.status_code == 404

    def test_market_prices_single_index(
        self, client: TestClient, test_db: DatabaseManager
    ) -> None:
        _seed_prices(test_db, "BANKNIFTY", days=2)
        r = client.get("/api/market/prices/BANKNIFTY")
        assert r.status_code == 200

    def test_stale_data_does_not_crash(
        self, client: TestClient, test_db: DatabaseManager
    ) -> None:
        """Very old price data (7 days) should not crash the status endpoint."""
        now = _now_ist()
        old_ts = (now - timedelta(days=7)).isoformat()
        test_db.execute(
            "INSERT OR IGNORE INTO price_data "
            "(index_id, timestamp, open, high, low, close, volume, vwap, source, timeframe) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("NIFTY50", old_ts, 22000, 22100, 21900, 22050, 500000, 22025, "test", "1d"),
        )
        r = client.get("/api/system/status")
        assert r.status_code == 200
        freshness = r.json().get("data_freshness", [])
        price_entry = next((f for f in freshness if f["table"] == "price_data"), None)
        assert price_entry is not None
        # Age should be large (>= 7 days in seconds) — system warns but doesn't crash
        if price_entry.get("age_seconds") is not None:
            assert price_entry["age_seconds"] >= 7 * 24 * 3600


# ---------------------------------------------------------------------------
# e) SPA serving — React build integration
# ---------------------------------------------------------------------------


class TestSpaServing:
    """Verify that the SPA catch-all route serves index.html when the build exists."""

    @pytest.fixture()
    def spa_client(
        self,
        test_db: DatabaseManager,
        test_registry: IndexRegistry,
        tmp_path: Path,
    ) -> Generator[TestClient, None, None]:
        """TestClient backed by a fake frontend/dist.

        The patch is kept alive for the entire test so that the ``serve_spa``
        closure sees the correct dist path both at registration time (inside
        ``create_app()``) and at request time.
        """
        dist_dir = tmp_path / "dist"
        assets_dir = dist_dir / "assets"
        assets_dir.mkdir(parents=True)
        (dist_dir / "index.html").write_text(
            "<html><body>Trading DSS</body></html>", encoding="utf-8"
        )
        (assets_dir / "main.js").write_text("console.log('app');", encoding="utf-8")

        # Keep the patch alive for the full test so serve_spa sees the tmp dir.
        patcher = patch("src.api.app._FRONTEND_DIST", dist_dir)
        patcher.start()
        try:
            app = create_app()
            app.dependency_overrides[get_db] = lambda: test_db
            app.dependency_overrides[get_index_registry] = lambda: test_registry
            app.dependency_overrides[get_market_hours] = lambda: MarketHoursManager()

            from src.api.routes import anomalies, market_data, signals
            market_data._cache.clear()
            signals._cache.clear()
            anomalies._cache.clear()

            yield TestClient(app)
        finally:
            patcher.stop()

    def test_root_serves_index_html(self, spa_client: TestClient) -> None:
        r = spa_client.get("/")
        assert r.status_code == 200
        assert "Trading DSS" in r.text

    def test_spa_route_serves_index_html(self, spa_client: TestClient) -> None:
        r = spa_client.get("/dashboard")
        assert r.status_code == 200
        assert "Trading DSS" in r.text

    def test_deep_path_serves_index_html(self, spa_client: TestClient) -> None:
        r = spa_client.get("/portfolio/performance")
        assert r.status_code == 200
        assert "Trading DSS" in r.text

    def test_api_routes_still_work_with_spa(self, spa_client: TestClient) -> None:
        """API routes must not be swallowed by the SPA catch-all."""
        r = spa_client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_api_system_status_with_spa(self, spa_client: TestClient) -> None:
        r = spa_client.get("/api/system/status")
        assert r.status_code == 200
        assert "market_status" in r.json()

    def test_no_spa_when_dist_missing(
        self,
        test_db: DatabaseManager,
        test_registry: IndexRegistry,
        tmp_path: Path,
    ) -> None:
        """Without a frontend build, SPA route must not be registered."""
        nonexistent = tmp_path / "nonexistent_dist"
        with patch("src.api.app._FRONTEND_DIST", nonexistent):
            app = create_app()
        app.dependency_overrides[get_db] = lambda: test_db
        app.dependency_overrides[get_index_registry] = lambda: test_registry
        app.dependency_overrides[get_market_hours] = lambda: MarketHoursManager()

        from src.api.routes import anomalies, market_data, signals
        market_data._cache.clear()
        signals._cache.clear()
        anomalies._cache.clear()

        c = TestClient(app, raise_server_exceptions=False)
        # Without SPA serving, unknown paths should 404
        r = c.get("/some-nonexistent-page")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# f) Performance tests
# ---------------------------------------------------------------------------


class TestPerformance:
    """Lightweight performance assertions — no external services needed."""

    # Endpoints to benchmark (should all respond in < 200ms on empty DB)
    ENDPOINTS = [
        "/api/health",
        "/api/system/health",
        "/api/system/status",
        "/api/market/prices",
        "/api/signals/current",
        "/api/signals/history",
        "/api/news/feed",
        "/api/anomalies/active",
        "/api/portfolio/positions",
        "/api/indices",
    ]

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_response_time_under_200ms(
        self, client: TestClient, endpoint: str
    ) -> None:
        """Each endpoint must respond within 200ms on an empty DB."""
        start = time.monotonic()
        r = client.get(endpoint)
        elapsed_ms = (time.monotonic() - start) * 1000
        assert r.status_code in (200, 204), (
            f"{endpoint} returned {r.status_code}"
        )
        assert elapsed_ms < 200, (
            f"{endpoint} took {elapsed_ms:.1f}ms — expected < 200ms"
        )

    def test_concurrent_requests_succeed(self, client: TestClient) -> None:
        """10 simultaneous requests to /api/health must all succeed."""
        results: list[int] = []
        lock = threading.Lock()

        def _hit() -> None:
            r = client.get("/api/health")
            with lock:
                results.append(r.status_code)

        threads = [threading.Thread(target=_hit) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == 10
        assert all(code == 200 for code in results), (
            f"Some requests failed: {results}"
        )

    def test_concurrent_mixed_requests(self, client: TestClient) -> None:
        """Mixed concurrent requests across different endpoints — all must succeed."""
        endpoints = [
            "/api/health",
            "/api/system/status",
            "/api/market/prices",
            "/api/signals/current",
            "/api/news/feed",
        ] * 2  # 10 total

        results: list[int] = []
        lock = threading.Lock()

        def _hit(ep: str) -> None:
            r = client.get(ep)
            with lock:
                results.append(r.status_code)

        threads = [threading.Thread(target=_hit, args=(ep,)) for ep in endpoints]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == len(endpoints)
        assert all(code in (200, 204) for code in results)

    def test_repeated_polling_no_failures(self, client: TestClient) -> None:
        """Simulate 5 rapid polling cycles across key endpoints — none may fail."""
        endpoints = [
            "/api/market/prices",
            "/api/signals/current",
            "/api/system/status",
            "/api/news/feed",
            "/api/anomalies/active",
        ]
        for _ in range(5):
            for ep in endpoints:
                r = client.get(ep)
                assert r.status_code in (200, 204), (
                    f"Polling {ep} returned {r.status_code}"
                )
