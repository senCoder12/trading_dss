"""
Comprehensive tests for the FastAPI REST API.

Uses FastAPI TestClient (no server needed).
Tests with both empty DB and seeded data.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

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

# ── Fixtures ─────────────────────────────────────────────────────────────────

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
    {
        "id": "NIFTY_IT",
        "display_name": "NIFTY IT",
        "nse_symbol": "NIFTY IT",
        "yahoo_symbol": "^CNXIT",
        "exchange": "NSE",
        "lot_size": None,
        "has_options": False,
        "option_symbol": None,
        "sector_category": "sectoral",
        "is_active": True,
        "description": "NIFTY IT index",
    },
]


@pytest.fixture()
def test_db(tmp_path: Path) -> Generator[DatabaseManager, None, None]:
    """Create a temporary SQLite DB with schema initialised and index_master seeded."""
    db_path = tmp_path / "test_trading.db"
    db = DatabaseManager(db_path=db_path)
    db.connect()
    db.initialise_schema()

    # Seed index_master — required by foreign keys on all data tables
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
    """Create a test IndexRegistry from sample data."""
    p = tmp_path / "indices.json"
    p.write_text(json.dumps(SAMPLE_INDICES), encoding="utf-8")
    return IndexRegistry.from_file(p)


@pytest.fixture()
def client(test_db: DatabaseManager, test_registry: IndexRegistry) -> TestClient:
    """Create a TestClient with overridden dependencies."""
    app = create_app()
    app.dependency_overrides[get_db] = lambda: test_db
    app.dependency_overrides[get_index_registry] = lambda: test_registry
    app.dependency_overrides[get_market_hours] = lambda: MarketHoursManager()

    # Ensure kill switch is off at start of each test
    deactivate_kill_switch()

    # Clear route-level caches to avoid cross-test contamination
    from src.api.routes import market_data, signals, anomalies
    market_data._cache.clear()
    signals._cache.clear()
    anomalies._cache.clear()

    return TestClient(app)


def _now_ist() -> datetime:
    return datetime.now(tz=_IST)


def _seed_prices(db: DatabaseManager, index_id: str = "NIFTY50", days: int = 5):
    """Insert sample price data."""
    now = _now_ist()
    for i in range(days):
        ts = (now - timedelta(days=days - i)).isoformat()
        close = 22000 + (i * 50)
        db.execute(
            "INSERT OR IGNORE INTO price_data "
            "(index_id, timestamp, open, high, low, close, volume, vwap, source, timeframe) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (index_id, ts, close - 30, close + 50, close - 60, close, 1000000, close, "test", "1d"),
        )


def _seed_signals(db: DatabaseManager, index_id: str = "NIFTY50", count: int = 5):
    """Insert sample trading signals."""
    now = _now_ist()
    for i in range(count):
        ts = (now - timedelta(hours=count - i)).isoformat()
        signal_type = "BUY_CALL" if i % 3 != 0 else "NO_TRADE"
        outcome = None
        pnl = None
        closed_at = None

        if signal_type != "NO_TRADE":
            if i % 2 == 0:
                outcome = "WIN"
                pnl = 150.0 + i * 10
                closed_at = (now - timedelta(hours=count - i - 1)).isoformat()
            else:
                outcome = "OPEN"

        reasoning = json.dumps({"text": f"Test signal {i}", "confidence_score": 0.5 + i * 0.05})

        db.execute(
            "INSERT INTO trading_signals "
            "(index_id, generated_at, signal_type, confidence_level, "
            "entry_price, target_price, stop_loss, risk_reward_ratio, "
            "regime, technical_vote, options_vote, news_vote, anomaly_vote, "
            "reasoning, outcome, actual_exit_price, actual_pnl, closed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                index_id, ts, signal_type, "HIGH" if i > 2 else "MEDIUM",
                22000 + i * 10, 22200 + i * 10, 21900 + i * 10, 1.5,
                "TRENDING", "BULLISH", "BULLISH", "NEUTRAL", "NEUTRAL",
                reasoning, outcome,
                (22200 + i * 10) if outcome == "WIN" else None,
                pnl, closed_at,
            ),
        )


def _seed_news(db: DatabaseManager, count: int = 5):
    """Insert sample news articles and index impacts."""
    now = _now_ist()
    for i in range(count):
        ts = (now - timedelta(hours=count - i)).isoformat()
        severity = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NOISE"][i % 5]
        url = f"https://example.com/news/{i}"
        db.execute(
            "INSERT OR IGNORE INTO news_articles "
            "(title, summary, source, url, published_at, fetched_at, "
            "raw_sentiment_score, adjusted_sentiment, impact_category, "
            "source_credibility, is_processed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"Test headline {i}", f"Summary {i}", "TestSource", url,
                ts, ts, 0.5 - i * 0.2, 0.3 - i * 0.15, severity, 0.8, 1,
            ),
        )
        # Map to NIFTY50
        row = db.fetch_one("SELECT id FROM news_articles WHERE url = ?", (url,))
        if row:
            db.execute(
                "INSERT OR IGNORE INTO news_index_impact "
                "(news_id, index_id, relevance_score, mapped_via) "
                "VALUES (?, ?, ?, ?)",
                (row["id"], "NIFTY50", 0.9, "keyword"),
            )


def _seed_anomalies(db: DatabaseManager, count: int = 4):
    """Insert sample anomaly events."""
    now = _now_ist()
    types = ["VOLUME_SPIKE", "OI_SPIKE", "GAP_UP", "ABSORPTION"]
    severities = ["HIGH", "MEDIUM", "LOW", "MEDIUM"]
    categories = ["VOLUME", "OI", "PRICE", "VOLUME"]
    for i in range(count):
        ts = (now - timedelta(hours=count - i)).isoformat()
        db.execute(
            "INSERT INTO anomaly_events "
            "(index_id, timestamp, anomaly_type, severity, category, "
            "details, message, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "NIFTY50", ts, types[i], severities[i], categories[i],
                json.dumps({"value": 100 + i}), f"Alert: {types[i]}", 1 if i < 3 else 0,
            ),
        )


def _seed_vix(db: DatabaseManager):
    """Insert sample VIX data."""
    now = _now_ist()
    db.execute(
        "INSERT INTO vix_data (timestamp, vix_value, vix_change, vix_change_pct) "
        "VALUES (?, ?, ?, ?)",
        (now.isoformat(), 15.2, 0.45, 3.05),
    )


def _seed_system_health(db: DatabaseManager):
    """Insert sample system health records."""
    now = _now_ist()
    for comp, status in [("data_collector", "OK"), ("signal_engine", "OK"), ("news_engine", "WARNING")]:
        db.execute(
            "INSERT INTO system_health (timestamp, component, status, message, response_time_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (now.isoformat(), comp, status, f"{comp} is {status}", 42),
        )


# ============================================================================
# Health check
# ============================================================================

class TestHealthCheck:
    def test_health_ok(self, client: TestClient):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


# ============================================================================
# Indices
# ============================================================================

class TestIndices:
    def test_list_all_indices(self, client: TestClient):
        resp = client.get("/api/indices/")
        assert resp.status_code == 200
        data = resp.json()
        assert "indices" in data
        assert data["count"] == 3  # 3 active indices
        ids = [i["id"] for i in data["indices"]]
        assert "NIFTY50" in ids

    def test_filter_by_exchange(self, client: TestClient):
        resp = client.get("/api/indices/?exchange=NSE")
        assert resp.status_code == 200
        data = resp.json()
        assert all(i["exchange"] == "NSE" for i in data["indices"])

    def test_filter_has_options(self, client: TestClient):
        resp = client.get("/api/indices/?has_options=true")
        assert resp.status_code == 200
        data = resp.json()
        assert all(i["has_options"] for i in data["indices"])
        assert data["count"] == 2  # NIFTY50 and BANKNIFTY

    def test_get_single_index(self, client: TestClient, test_db: DatabaseManager):
        resp = client.get("/api/indices/NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "NIFTY50"
        assert data["has_options"] is True

    def test_index_not_found(self, client: TestClient):
        resp = client.get("/api/indices/INVALID")
        assert resp.status_code == 404

    def test_index_with_price_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_prices(test_db)
        resp = client.get("/api/indices/NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ltp"] is not None


# ============================================================================
# Market Data
# ============================================================================

class TestMarketData:
    def test_prices_empty_db(self, client: TestClient):
        resp = client.get("/api/market/prices")
        assert resp.status_code == 200
        data = resp.json()
        assert "indices" in data
        assert "timestamp" in data
        # All indices present but no price data
        for idx in data["indices"]:
            assert "index_id" in idx

    def test_prices_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_prices(test_db, "NIFTY50")
        _seed_prices(test_db, "BANKNIFTY")
        resp = client.get("/api/market/prices")
        assert resp.status_code == 200
        data = resp.json()
        nifty = next((i for i in data["indices"] if i["index_id"] == "NIFTY50"), None)
        assert nifty is not None
        assert nifty["ltp"] is not None

    def test_price_detail(self, client: TestClient, test_db: DatabaseManager):
        _seed_prices(test_db)
        resp = client.get("/api/market/prices/NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index_id"] == "NIFTY50"
        assert data["ltp"] is not None

    def test_price_detail_empty(self, client: TestClient):
        resp = client.get("/api/market/prices/NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index_id"] == "NIFTY50"
        assert data["ltp"] is None

    def test_price_detail_not_found(self, client: TestClient):
        resp = client.get("/api/market/prices/INVALID")
        assert resp.status_code == 404

    def test_price_history(self, client: TestClient, test_db: DatabaseManager):
        _seed_prices(test_db)
        resp = client.get("/api/market/prices/NIFTY50/history?days=30&timeframe=1d")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index_id"] == "NIFTY50"
        assert data["timeframe"] == "1d"
        assert isinstance(data["bars"], list)
        assert data["count"] == len(data["bars"])

    def test_price_history_empty(self, client: TestClient):
        resp = client.get("/api/market/prices/NIFTY50/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bars"] == []
        assert data["count"] == 0

    def test_price_history_pagination(self, client: TestClient, test_db: DatabaseManager):
        _seed_prices(test_db, days=10)
        resp = client.get("/api/market/prices/NIFTY50/history?limit=3&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] <= 3

    def test_price_history_not_found(self, client: TestClient):
        resp = client.get("/api/market/prices/INVALID/history")
        assert resp.status_code == 404

    def test_options_no_data(self, client: TestClient):
        resp = client.get("/api/market/options/NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index_id"] == "NIFTY50"
        assert data["pcr"] is None

    def test_options_not_fo(self, client: TestClient):
        resp = client.get("/api/market/options/NIFTY_IT")
        assert resp.status_code == 400

    def test_options_not_found(self, client: TestClient):
        resp = client.get("/api/market/options/INVALID")
        assert resp.status_code == 404

    def test_vix_empty(self, client: TestClient):
        resp = client.get("/api/market/vix")
        assert resp.status_code == 200
        data = resp.json()
        assert data["regime"] == "UNKNOWN"

    def test_vix_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_vix(test_db)
        resp = client.get("/api/market/vix")
        assert resp.status_code == 200
        data = resp.json()
        assert data["value"] == 15.2
        assert data["regime"] == "NORMAL"


# ============================================================================
# Signals
# ============================================================================

class TestSignals:
    def test_current_signals_empty(self, client: TestClient):
        resp = client.get("/api/signals/current")
        assert resp.status_code == 200
        data = resp.json()
        assert "signals" in data
        # F&O indices should be present (NIFTY50, BANKNIFTY)
        ids = [s["index_id"] for s in data["signals"]]
        assert "NIFTY50" in ids

    def test_current_signals_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db)
        resp = client.get("/api/signals/current")
        assert resp.status_code == 200
        data = resp.json()
        nifty_sig = next((s for s in data["signals"] if s["index_id"] == "NIFTY50"), None)
        assert nifty_sig is not None
        assert nifty_sig["signal_type"] in ("BUY_CALL", "BUY_PUT", "NO_TRADE")

    def test_current_signal_detail(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db)
        resp = client.get("/api/signals/current/NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index_id"] == "NIFTY50"
        assert "reasoning" in data

    def test_current_signal_detail_not_found(self, client: TestClient):
        resp = client.get("/api/signals/current/INVALID")
        assert resp.status_code == 404

    def test_signal_history(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db)
        resp = client.get("/api/signals/history?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert "signals" in data
        assert data["count"] >= 0
        assert "total" in data

    def test_signal_history_filter_by_index(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db)
        resp = client.get("/api/signals/history?index_id=NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        for s in data["signals"]:
            assert s["index_id"] == "NIFTY50"

    def test_signal_history_filter_by_outcome(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db)
        resp = client.get("/api/signals/history?outcome=WIN")
        assert resp.status_code == 200
        data = resp.json()
        for s in data["signals"]:
            assert s["outcome"] == "WIN"

    def test_signal_history_pagination(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db, count=10)
        resp = client.get("/api/signals/history?limit=3&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] <= 3
        assert data["total"] >= data["count"]

    def test_signal_history_empty(self, client: TestClient):
        resp = client.get("/api/signals/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["signals"] == []
        assert data["count"] == 0

    def test_performance_empty(self, client: TestClient):
        resp = client.get("/api/signals/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_signals"] == 0
        assert data["total_trades"] == 0

    def test_performance_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db, count=10)
        resp = client.get("/api/signals/performance?days=30")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period_days"] == 30
        assert "win_rate" in data
        assert "is_profitable" in data


# ============================================================================
# Portfolio
# ============================================================================

class TestPortfolio:
    def test_summary_empty(self, client: TestClient):
        resp = client.get("/api/portfolio/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["initial_capital"] == 100000
        assert data["capital"] == 100000
        assert data["open_positions"] == []

    def test_summary_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db)
        _seed_prices(test_db)
        resp = client.get("/api/portfolio/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["capital"] >= 0
        assert "total_return_pct" in data
        assert isinstance(data["open_positions"], list)

    def test_positions_empty(self, client: TestClient):
        resp = client.get("/api/portfolio/positions")
        assert resp.status_code == 200
        data = resp.json()
        assert data == []

    def test_positions_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db)
        _seed_prices(test_db)
        resp = client.get("/api/portfolio/positions")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_history_empty(self, client: TestClient):
        resp = client.get("/api/portfolio/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["history"] == []
        assert data["count"] == 0

    def test_history_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db, count=10)
        resp = client.get("/api/portfolio/history?days=30")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["history"], list)

    def test_trades_empty(self, client: TestClient):
        resp = client.get("/api/portfolio/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trades"] == []
        assert data["count"] == 0

    def test_trades_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db)
        resp = client.get("/api/portfolio/trades?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["trades"], list)
        assert "total" in data

    def test_trades_pagination(self, client: TestClient, test_db: DatabaseManager):
        _seed_signals(test_db, count=10)
        resp = client.get("/api/portfolio/trades?limit=2&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] <= 2


# ============================================================================
# News
# ============================================================================

class TestNews:
    def test_feed_empty(self, client: TestClient):
        resp = client.get("/api/news/feed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["articles"] == []
        assert data["count"] == 0

    def test_feed_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_news(test_db)
        resp = client.get("/api/news/feed?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["articles"]) > 0
        assert "title" in data["articles"][0]

    def test_feed_filter_by_index(self, client: TestClient, test_db: DatabaseManager):
        _seed_news(test_db)
        resp = client.get("/api/news/feed?index_id=NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["articles"]) > 0

    def test_feed_filter_by_severity(self, client: TestClient, test_db: DatabaseManager):
        _seed_news(test_db)
        resp = client.get("/api/news/feed?min_severity=HIGH")
        assert resp.status_code == 200
        data = resp.json()
        for article in data["articles"]:
            assert article["impact_category"] in ("CRITICAL", "HIGH")

    def test_feed_pagination(self, client: TestClient, test_db: DatabaseManager):
        _seed_news(test_db, count=10)
        resp = client.get("/api/news/feed?limit=3&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] <= 3

    def test_summary_empty(self, client: TestClient):
        resp = client.get("/api/news/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_articles"] == 0

    def test_summary_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_news(test_db)
        resp = client.get("/api/news/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_articles"] > 0
        assert "overall_sentiment_label" in data

    def test_sentiment_not_found(self, client: TestClient):
        resp = client.get("/api/news/sentiment/INVALID")
        assert resp.status_code == 404

    def test_sentiment_empty(self, client: TestClient):
        resp = client.get("/api/news/sentiment/NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index_id"] == "NIFTY50"
        assert data["article_count"] == 0

    def test_sentiment_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_news(test_db)
        resp = client.get("/api/news/sentiment/NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["article_count"] > 0
        assert "sentiment_label" in data


# ============================================================================
# Anomalies
# ============================================================================

class TestAnomalies:
    def test_active_empty(self, client: TestClient):
        resp = client.get("/api/anomalies/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["anomalies"] == []
        assert data["count"] == 0

    def test_active_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_anomalies(test_db)
        resp = client.get("/api/anomalies/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        for a in data["anomalies"]:
            assert a["is_active"] is True

    def test_active_filter_by_index(self, client: TestClient, test_db: DatabaseManager):
        _seed_anomalies(test_db)
        resp = client.get("/api/anomalies/active?index_id=NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        for a in data["anomalies"]:
            assert a["index_id"] == "NIFTY50"

    def test_active_filter_by_severity(self, client: TestClient, test_db: DatabaseManager):
        _seed_anomalies(test_db)
        resp = client.get("/api/anomalies/active?min_severity=HIGH")
        assert resp.status_code == 200
        data = resp.json()
        for a in data["anomalies"]:
            assert a["severity"] == "HIGH"

    def test_history_empty(self, client: TestClient):
        resp = client.get("/api/anomalies/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["anomalies"] == []

    def test_history_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_anomalies(test_db)
        resp = client.get("/api/anomalies/history?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0

    def test_history_pagination(self, client: TestClient, test_db: DatabaseManager):
        _seed_anomalies(test_db)
        resp = client.get("/api/anomalies/history?limit=2&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] <= 2

    def test_dashboard_empty(self, client: TestClient):
        resp = client.get("/api/anomalies/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_active"] == 0

    def test_dashboard_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_anomalies(test_db)
        resp = client.get("/api/anomalies/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_active"] > 0
        assert "by_category" in data
        assert "by_index" in data
        assert isinstance(data["recent_alerts"], list)


# ============================================================================
# System
# ============================================================================

class TestSystem:
    def test_health_empty(self, client: TestClient):
        resp = client.get("/api/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert "overall_status" in data
        assert "db_size" in data

    def test_health_with_data(self, client: TestClient, test_db: DatabaseManager):
        _seed_system_health(test_db)
        resp = client.get("/api/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["components"]) >= 3
        # At least one WARNING component
        assert data["overall_status"] == "WARNING"

    def test_status(self, client: TestClient):
        resp = client.get("/api/system/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "market_status" in data
        assert "uptime_seconds" in data
        assert data["kill_switch_active"] is False
        assert isinstance(data["data_freshness"], list)

    def test_kill_switch_activate(self, client: TestClient):
        resp = client.post("/api/system/kill-switch", json={"reason": "Test activation"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["reason"] == "Test activation"

        # Verify it's active in status
        status_resp = client.get("/api/system/status")
        assert status_resp.json()["kill_switch_active"] is True

    def test_kill_switch_deactivate(self, client: TestClient):
        # First activate
        client.post("/api/system/kill-switch", json={"reason": "Test"})
        # Then deactivate
        resp = client.delete("/api/system/kill-switch")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is False

        # Verify it's off in status
        status_resp = client.get("/api/system/status")
        assert status_resp.json()["kill_switch_active"] is False

    def test_config(self, client: TestClient):
        resp = client.get("/api/system/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "approved_params" in data


# ============================================================================
# Backtest
# ============================================================================

class TestBacktest:
    def test_latest_not_found(self, client: TestClient):
        resp = client.get("/api/backtest/latest?index_id=NIFTY50")
        assert resp.status_code == 404

    def test_optimization_empty(self, client: TestClient):
        resp = client.get("/api/backtest/optimization")
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []

    def test_shadow_report_not_found(self, client: TestClient):
        resp = client.get("/api/backtest/shadow-report?index_id=NIFTY50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False


# ============================================================================
# CORS
# ============================================================================

class TestCORS:
    def test_cors_allowed_origin(self, client: TestClient):
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_cors_vite_origin(self, client: TestClient):
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
