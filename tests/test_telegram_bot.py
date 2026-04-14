"""
Tests for the interactive Telegram bot (src/alerts/telegram_bot.py).

Covers:
  - Command handler output formatting
  - Message formatting helpers
  - Priority routing (CRITICAL/HIGH/NORMAL/LOW)
  - Kill switch activation via /kill
  - Graceful degradation when bot token is not configured
  - Message splitting for long messages
  - Unknown command handling
"""

from __future__ import annotations

import queue
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

# ---------------------------------------------------------------------------
# Helpers — lightweight fakes for DatabaseManager and DecisionEngine
# ---------------------------------------------------------------------------

_IST = ZoneInfo("Asia/Kolkata")


class FakeDB:
    """In-memory SQLite that behaves like DatabaseManager."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._setup_tables()

    def _setup_tables(self):
        stmts = [
            """CREATE TABLE IF NOT EXISTS vix_data (
                id INTEGER PRIMARY KEY, timestamp TEXT, vix_value REAL,
                vix_change REAL DEFAULT 0, vix_change_pct REAL DEFAULT 0)""",
            """CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT, index_id TEXT,
                generated_at TEXT, signal_type TEXT, confidence_level TEXT,
                entry_price REAL DEFAULT 0, target_price REAL DEFAULT 0,
                stop_loss REAL DEFAULT 0, risk_reward_ratio REAL DEFAULT 0,
                regime TEXT DEFAULT 'RANGE_BOUND',
                technical_vote TEXT DEFAULT 'NEUTRAL', options_vote TEXT DEFAULT 'NEUTRAL',
                news_vote TEXT DEFAULT 'NEUTRAL', anomaly_vote TEXT DEFAULT 'NEUTRAL',
                reasoning TEXT DEFAULT '{}', outcome TEXT, actual_exit_price REAL,
                actual_pnl REAL, closed_at TEXT)""",
            """CREATE TABLE IF NOT EXISTS anomaly_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, index_id TEXT,
                timestamp TEXT, anomaly_type TEXT, severity TEXT DEFAULT 'MEDIUM',
                category TEXT DEFAULT 'OTHER', details TEXT DEFAULT '{}',
                message TEXT DEFAULT '', cooldown_key TEXT DEFAULT '',
                is_active INTEGER DEFAULT 1)""",
            """CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, summary TEXT,
                source TEXT, url TEXT UNIQUE, published_at TEXT, fetched_at TEXT,
                raw_sentiment_score REAL DEFAULT 0, adjusted_sentiment REAL DEFAULT 0,
                impact_category TEXT DEFAULT 'MEDIUM', source_credibility REAL DEFAULT 0.7,
                is_processed INTEGER DEFAULT 0)""",
            """CREATE TABLE IF NOT EXISTS fii_dii_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, category TEXT,
                buy_value REAL DEFAULT 0, sell_value REAL DEFAULT 0,
                net_value REAL DEFAULT 0, segment TEXT DEFAULT 'CASH')""",
            """CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT, index_id TEXT,
                timestamp TEXT, timeframe TEXT, vwap REAL, ema_20 REAL,
                ema_50 REAL, rsi_14 REAL, support_1 REAL, resistance_1 REAL,
                support_2 REAL, resistance_2 REAL, avg_volume_20 REAL,
                technical_signal TEXT DEFAULT 'NEUTRAL', signal_strength REAL DEFAULT 0)""",
            """CREATE TABLE IF NOT EXISTS oi_aggregated (
                id INTEGER PRIMARY KEY AUTOINCREMENT, index_id TEXT,
                timestamp TEXT, expiry_date TEXT, total_ce_oi INTEGER DEFAULT 0,
                total_pe_oi INTEGER DEFAULT 0, total_ce_oi_change INTEGER DEFAULT 0,
                total_pe_oi_change INTEGER DEFAULT 0, pcr REAL DEFAULT 0,
                max_pain_strike REAL DEFAULT 0, highest_ce_oi_strike REAL DEFAULT 0,
                highest_pe_oi_strike REAL DEFAULT 0)""",
            """CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT, index_id TEXT,
                timestamp TEXT, open REAL, high REAL, low REAL, close REAL,
                volume REAL DEFAULT 0, vwap REAL, source TEXT DEFAULT 'nse_live',
                timeframe TEXT DEFAULT '1d')""",
            """CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
                component TEXT, status TEXT DEFAULT 'OK', message TEXT,
                response_time_ms INTEGER)""",
        ]
        cur = self._conn.cursor()
        for s in stmts:
            cur.execute(s)
        self._conn.commit()

    def fetch_one(self, query, params=()):
        cur = self._conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)

    def fetch_all(self, query, params=()):
        cur = self._conn.cursor()
        cur.execute(query, params)
        return [dict(r) for r in cur.fetchall()]

    def execute(self, query, params=()):
        cur = self._conn.cursor()
        cur.execute(query, params)
        self._conn.commit()
        return cur

    def get_db_size(self):
        return "1.2 MB"

    def insert(self, table, **kwargs):
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join("?" for _ in kwargs)
        self.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
            tuple(kwargs.values()),
        )


@dataclass
class FakeRiskConfig:
    total_capital: float = 100_000.0
    max_risk_per_day_pct: float = 5.0


@dataclass
class FakePortfolioSummary:
    total_capital: float = 100_000.0
    current_exposure: float = 15_000.0
    available_capital: float = 85_000.0
    open_positions: list = field(default_factory=list)
    today_pnl: float = 1250.0
    this_week_pnl: float = 3800.0
    this_month_pnl: float = 5500.0
    total_trades_today: int = 3
    daily_limit_remaining_pct: float = 55.0


@dataclass
class FakePerformanceStats:
    period_days: int = 7
    total_signals: int = 15
    actionable_signals: int = 12
    total_trades: int = 12
    wins: int = 7
    losses: int = 5
    win_rate: float = 58.3
    total_pnl: float = 3800.0
    avg_pnl_per_trade: float = 316.7
    largest_win: float = 900.0
    largest_loss: float = -500.0
    avg_win: float = 520.0
    avg_loss: float = -380.0
    profit_factor: float = 1.42
    max_drawdown: float = -2100.0
    max_drawdown_pct: float = -2.1
    sharpe_ratio: float = 1.1
    avg_risk_reward: float = 1.37
    high_confidence_win_rate: float = 80.0
    medium_confidence_win_rate: float = 42.9
    low_confidence_win_rate: float = 33.3
    win_rate_by_regime: dict = field(default_factory=dict)
    win_rate_by_index: dict = field(default_factory=dict)
    pnl_by_index: dict = field(default_factory=dict)
    avg_confidence_of_wins: float = 0.72
    avg_confidence_of_losses: float = 0.55
    no_trade_accuracy: float = 0.0
    expected_value_per_trade: float = 150.0
    is_profitable: bool = True
    edge_comment: str = "Profitable edge detected"


@dataclass
class FakeSignal:
    signal_id: str = "sig-001"
    index_id: str = "NIFTY50"
    generated_at: str = "2026-04-14T10:00:00"
    signal_type: str = "BUY_CALL"
    confidence_level: str = "HIGH"
    confidence_score: float = 0.72
    entry_price: float = 22450.0
    target_price: float = 22620.0
    stop_loss: float = 22340.0
    risk_reward_ratio: float = 1.55
    regime: str = "Strong Uptrend"
    weighted_score: float = 0.72
    vote_breakdown: dict = field(default_factory=dict)
    risk_level: str = "NORMAL"
    position_size_modifier: float = 1.0
    suggested_lot_count: int = 2
    estimated_max_loss: float = 2750.0
    estimated_max_profit: float = 4250.0
    reasoning: str = "Trend: BULLISH above EMA20/50\nOptions: PCR 1.15\nRSI: 58 neutral"
    warnings: list = field(default_factory=list)
    outcome: Optional[str] = None
    actual_exit_price: Optional[float] = None
    actual_pnl: Optional[float] = None
    closed_at: Optional[datetime] = None
    data_completeness: float = 1.0
    signals_generated_today: int = 3
    refined_entry: float = 22450.0
    refined_target: float = 22620.0
    refined_stop_loss: float = 22340.0
    lots: int = 2
    recommended_strike: float = 22450.0
    recommended_expiry: str = "2026-04-17"
    option_premium: float = 185.0
    max_loss_amount: float = 2750.0
    risk_pct_of_capital: float = 2.75


@dataclass
class FakeDecisionResult:
    index_id: str = "NIFTY50"
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=_IST))
    signal: FakeSignal = field(default_factory=FakeSignal)
    is_actionable: bool = True
    alert_message: str = ""
    alert_priority: str = "CRITICAL"


class FakeRiskManager:
    config = FakeRiskConfig()

    def get_portfolio_summary(self):
        return FakePortfolioSummary()


class FakeTracker:
    def get_performance_stats(self, days=7):
        return FakePerformanceStats()


class FakeEngine:
    """Minimal DecisionEngine stand-in."""

    def __init__(self):
        self.risk_manager = FakeRiskManager()
        self.tracker = FakeTracker()
        self._result_cache = {
            "NIFTY50": FakeDecisionResult(),
            "BANKNIFTY": FakeDecisionResult(
                index_id="BANKNIFTY",
                signal=FakeSignal(signal_type="NO_TRADE", confidence_level="LOW", confidence_score=0.0),
                is_actionable=False,
            ),
        }
        self._kill_switch_active = False

    def _is_kill_switch_active(self):
        return self._kill_switch_active

    def activate_kill_switch(self, reason="test"):
        self._kill_switch_active = True

    def deactivate_kill_switch(self):
        self._kill_switch_active = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_db():
    return FakeDB()


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def bot(fake_db, fake_engine):
    """TelegramBot with token NOT configured (all sends are no-ops)."""
    with patch("src.alerts.telegram_bot.settings") as mock_settings:
        mock_settings.telegram.bot_token = ""
        mock_settings.telegram.chat_id = ""
        mock_settings.thresholds.vix_panic_threshold = 25.0
        mock_settings.thresholds.vix_elevated_threshold = 18.0
        mock_settings.thresholds.vix_normal_threshold = 13.0
        from src.alerts.telegram_bot import TelegramBot
        b = TelegramBot(db=fake_db, decision_engine=fake_engine)
    return b


@pytest.fixture
def configured_bot(fake_db, fake_engine):
    """TelegramBot with token configured (sends are mocked)."""
    with patch("src.alerts.telegram_bot.settings") as mock_settings:
        mock_settings.telegram.bot_token = "FAKE_TOKEN"
        mock_settings.telegram.chat_id = "123456"
        mock_settings.thresholds.vix_panic_threshold = 25.0
        mock_settings.thresholds.vix_elevated_threshold = 18.0
        mock_settings.thresholds.vix_normal_threshold = 13.0
        from src.alerts.telegram_bot import TelegramBot
        b = TelegramBot(db=fake_db, decision_engine=fake_engine)
    return b


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class TestFormatHelpers:
    def test_format_currency_positive(self, bot):
        assert bot._format_currency(1250) == "\u20b91,250"

    def test_format_currency_negative(self, bot):
        result = bot._format_currency(-800)
        assert "\u20b9" in result
        assert "-800" in result

    def test_format_currency_zero(self, bot):
        assert bot._format_currency(0) == "\u20b90"

    def test_format_pct_positive(self, bot):
        assert bot._format_pct(2.5) == "+2.5%"

    def test_format_pct_negative(self, bot):
        assert bot._format_pct(-1.3) == "-1.3%"

    def test_format_signal_emoji_call(self, bot):
        emoji = bot._format_signal_emoji("BUY_CALL")
        assert emoji == "\U0001f7e2"

    def test_format_signal_emoji_put(self, bot):
        emoji = bot._format_signal_emoji("BUY_PUT")
        assert emoji == "\U0001f534"

    def test_format_signal_emoji_no_trade(self, bot):
        emoji = bot._format_signal_emoji("NO_TRADE")
        assert emoji == "\u26aa"

    def test_format_confidence_emoji(self, bot):
        assert bot._format_confidence_emoji("HIGH") == "\U0001f7e2"
        assert bot._format_confidence_emoji("MEDIUM") == "\U0001f7e1"
        assert bot._format_confidence_emoji("LOW") == "\U0001f535"

    def test_format_severity_emoji(self, bot):
        assert bot._format_severity_emoji("HIGH") == "\U0001f534"
        assert bot._format_severity_emoji("CRITICAL") == "\U0001f534"
        assert bot._format_severity_emoji("MEDIUM") == "\U0001f7e1"
        assert bot._format_severity_emoji("LOW") == "\u26aa"

    def test_truncate_short(self, bot):
        assert bot._truncate("hello", 10) == "hello"

    def test_truncate_long(self, bot):
        text = "a" * 50
        result = bot._truncate(text, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_human_timedelta(self, bot):
        assert bot._human_timedelta(timedelta(hours=2, minutes=15)) == "2h 15m"
        assert bot._human_timedelta(timedelta(minutes=45)) == "45m"
        assert bot._human_timedelta(timedelta(seconds=30)) == "0m"

    def test_human_age(self, bot):
        assert bot._human_age(12) == "12s ago"
        assert bot._human_age(90) == "1m ago"
        assert bot._human_age(7200) == "2h ago"
        assert bot._human_age(None) == "never"


# ---------------------------------------------------------------------------
# Message splitting
# ---------------------------------------------------------------------------


class TestMessageSplitting:
    def test_short_message_no_split(self, bot):
        chunks = bot._split_message("Hello")
        assert chunks == ["Hello"]

    def test_long_message_splits(self, bot):
        # Build a message larger than 4096
        text = "\n".join(f"Line {i}" for i in range(1000))
        assert len(text) > 4096
        chunks = bot._split_message(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 4096

    def test_no_data_lost_after_split(self, bot):
        lines = [f"Line {i}" for i in range(500)]
        text = "\n".join(lines)
        chunks = bot._split_message(text)
        rejoined = "\n".join(chunks)
        # All original lines should be present
        for line in lines:
            assert line in rejoined


# ---------------------------------------------------------------------------
# Command handlers — /start, /help
# ---------------------------------------------------------------------------


class TestStartCommand:
    def test_start_returns_help_text(self, bot):
        result = bot.handle_command("start")
        assert "Trading DSS Bot Active" in result
        assert "/status" in result
        assert "/signal" in result
        assert "/kill" in result
        assert "/resume" in result

    def test_help_is_alias_for_start(self, bot):
        assert bot.handle_command("help") == bot.handle_command("start")


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


class TestStatusCommand:
    def test_status_basic(self, bot, fake_db):
        # Insert VIX data
        fake_db.insert("vix_data", timestamp="2026-04-14T10:00:00", vix_value=14.8,
                        vix_change=0.3, vix_change_pct=2.1)

        with patch("src.utils.market_hours.MarketHoursManager") as MockMH:
            mh = MockMH.return_value
            mh.get_market_status.return_value = {"is_trading_day": True}
            mh.is_market_open.return_value = True
            mh.time_to_market_close.return_value = timedelta(hours=2, minutes=15)
            mh.get_session.return_value = MagicMock(name="OPEN")

            with patch("src.data.rate_limiter.freshness_tracker") as mock_ft:
                mock_ft.get_all_status.return_value = {
                    "index_prices": {"age_seconds": 12, "is_stale": False},
                    "options_chain": {"age_seconds": 45, "is_stale": False},
                    "vix": {"age_seconds": 60, "is_stale": False},
                    "fii_dii": {"age_seconds": 64800, "is_stale": False},
                }
                result = bot._cmd_status()

        assert "System Status" in result
        assert "14.8" in result
        assert "OPEN" in result or "Market" in result

    def test_status_market_closed(self, bot, fake_db):
        fake_db.insert("vix_data", timestamp="2026-04-14T16:00:00", vix_value=15.0,
                        vix_change=0.0, vix_change_pct=0.0)

        with patch("src.utils.market_hours.MarketHoursManager") as MockMH:
            mh = MockMH.return_value
            mh.get_market_status.return_value = {"is_trading_day": False}
            mh.is_market_open.return_value = False
            mh.get_session.return_value = MagicMock(name="CLOSED")

            with patch("src.data.rate_limiter.freshness_tracker") as mock_ft:
                mock_ft.get_all_status.return_value = {}
                result = bot._cmd_status()

        assert "System Status" in result


# ---------------------------------------------------------------------------
# /signal
# ---------------------------------------------------------------------------


class TestSignalCommand:
    def test_signal_compact_all_indices(self, bot):
        result = bot.handle_command("signal")
        assert "Current Signals" in result
        assert "NIFTY50" in result
        assert "BANKNIFTY" in result
        assert "BUY CALL" in result
        assert "NO TRADE" in result

    def test_signal_detailed_single_index(self, bot):
        result = bot.handle_command("signal", "NIFTY50")
        assert "BUY CALL" in result
        assert "NIFTY50" in result
        assert "HIGH" in result
        assert "22,450" in result or "22450" in result
        assert "Levels" in result
        assert "Why" in result

    def test_signal_unknown_index(self, bot):
        result = bot.handle_command("signal", "UNKNOWN")
        assert "No recent signal" in result

    def test_signal_no_engine(self, bot):
        bot._engine = None
        result = bot.handle_command("signal")
        assert "starting up" in result

    def test_signal_no_trade_detail(self, bot):
        result = bot.handle_command("signal", "BANKNIFTY")
        assert "NO TRADE" in result


# ---------------------------------------------------------------------------
# /portfolio
# ---------------------------------------------------------------------------


class TestPortfolioCommand:
    def test_portfolio_basic(self, bot):
        result = bot.handle_command("portfolio")
        assert "Portfolio" in result
        assert "Capital" in result
        assert "100,000" in result or "1,00,000" in result
        assert "P&L" in result

    def test_portfolio_with_positions(self, bot, fake_engine):
        fake_engine.risk_manager.get_portfolio_summary = lambda: FakePortfolioSummary(
            open_positions=[{
                "signal_id": "sig-1",
                "index": "NIFTY50",
                "signal_type": "BUY_CALL",
                "entry": 22450,
                "current_sl": 22400,
                "target": 22620,
                "lots": 2,
                "lot_size": 25,
                "entry_time": "2026-04-14T10:00:00",
            }],
        )
        result = bot.handle_command("portfolio")
        assert "NIFTY50" in result
        assert "CALL" in result

    def test_portfolio_no_engine(self, bot):
        bot._engine = None
        result = bot.handle_command("portfolio")
        assert "starting up" in result


# ---------------------------------------------------------------------------
# /performance
# ---------------------------------------------------------------------------


class TestPerformanceCommand:
    def test_performance_basic(self, bot, fake_db):
        # Insert some closed trades for streak calculation
        now = datetime.now(tz=_IST).isoformat()
        for i in range(3):
            fake_db.insert(
                "trading_signals",
                index_id="NIFTY50", generated_at=now, signal_type="BUY_CALL",
                confidence_level="HIGH", outcome="WIN", actual_pnl=500.0,
                closed_at=now,
            )

        result = bot.handle_command("performance")
        assert "7-Day Performance" in result
        assert "Win Rate" in result
        assert "Profit Factor" in result
        assert "By Confidence" in result

    def test_performance_no_engine(self, bot):
        bot._engine = None
        result = bot.handle_command("performance")
        assert "starting up" in result


# ---------------------------------------------------------------------------
# /news
# ---------------------------------------------------------------------------


class TestNewsCommand:
    def test_news_with_articles(self, bot, fake_db):
        for i in range(3):
            fake_db.insert(
                "news_articles",
                title=f"Test article {i}",
                summary="Summary",
                source="Reuters",
                url=f"https://example.com/{i}",
                published_at="2026-04-14T09:00:00",
                fetched_at="2026-04-14T09:01:00",
                raw_sentiment_score=0.3,
                adjusted_sentiment=0.3,
                impact_category="HIGH",
                is_processed=1,
            )

        result = bot.handle_command("news")
        assert "Top Market News" in result
        assert "Test article" in result
        assert "Bullish" in result

    def test_news_no_articles(self, bot):
        result = bot.handle_command("news")
        assert "No recent news" in result


# ---------------------------------------------------------------------------
# /alerts
# ---------------------------------------------------------------------------


class TestAlertsCommand:
    def test_alerts_active(self, bot, fake_db):
        now = datetime.now(tz=_IST).isoformat()
        fake_db.insert(
            "anomaly_events",
            index_id="NIFTY50", timestamp=now, anomaly_type="OI_SPIKE",
            severity="HIGH", category="OI", message="OI spike at 22500 CE +18%",
            cooldown_key="oi_spike_nifty", is_active=1,
        )

        result = bot.handle_command("alerts")
        assert "Active Alerts" in result
        assert "NIFTY50" in result
        assert "OI spike" in result

    def test_alerts_none(self, bot):
        result = bot.handle_command("alerts")
        assert "No active alerts" in result


# ---------------------------------------------------------------------------
# /vix
# ---------------------------------------------------------------------------


class TestVixCommand:
    def test_vix_normal(self, bot, fake_db):
        fake_db.insert("vix_data", timestamp="2026-04-14T10:00:00",
                        vix_value=14.82, vix_change=0.45, vix_change_pct=3.1)
        result = bot.handle_command("vix")
        assert "14.82" in result
        assert "NORMAL" in result
        assert "Standard conditions" in result

    def test_vix_high(self, bot, fake_db):
        fake_db.insert("vix_data", timestamp="2026-04-14T10:00:00",
                        vix_value=28.5, vix_change=3.0, vix_change_pct=12.0)
        result = bot.handle_command("vix")
        assert "28.5" in result
        assert "HIGH" in result
        assert "Extreme caution" in result

    def test_vix_no_data(self, bot):
        result = bot.handle_command("vix")
        assert "not available" in result


# ---------------------------------------------------------------------------
# /fii
# ---------------------------------------------------------------------------


class TestFiiCommand:
    def test_fii_with_data(self, bot, fake_db):
        fake_db.insert("fii_dii_activity", date="2026-04-13",
                        category="FII", buy_value=5000.0, sell_value=7100.0,
                        net_value=-2100.0, segment="CASH")
        fake_db.insert("fii_dii_activity", date="2026-04-13",
                        category="DII", buy_value=6800.0, sell_value=5000.0,
                        net_value=1800.0, segment="CASH")

        result = bot.handle_command("fii")
        assert "FII/DII Activity" in result
        assert "2026-04-13" in result

    def test_fii_no_data(self, bot):
        result = bot.handle_command("fii")
        assert "not available" in result


# ---------------------------------------------------------------------------
# /levels
# ---------------------------------------------------------------------------


class TestLevelsCommand:
    def test_levels_with_data(self, bot, fake_db):
        now = datetime.now(tz=_IST).isoformat()
        fake_db.insert(
            "technical_indicators",
            index_id="NIFTY50", timestamp=now, timeframe="5m",
            ema_20=22500.0, ema_50=22100.0, rsi_14=58.0,
            support_1=22400.0, resistance_1=22550.0,
            technical_signal="BULLISH", signal_strength=0.7,
        )
        fake_db.insert(
            "oi_aggregated",
            index_id="NIFTY50", timestamp=now, expiry_date="2026-04-17",
            total_ce_oi=50000, total_pe_oi=57500,
            pcr=1.15, max_pain_strike=22500.0,
            highest_ce_oi_strike=22680.0, highest_pe_oi_strike=22250.0,
        )
        fake_db.insert(
            "price_data",
            index_id="NIFTY50", timestamp=now, open=22400, high=22520,
            low=22380, close=22485, timeframe="5m",
        )

        result = bot.handle_command("levels", "NIFTY50")
        assert "Key Levels" in result
        assert "NIFTY50" in result
        assert "22,485" in result or "22485" in result
        assert "Resistance" in result
        assert "Support" in result
        assert "Max Pain" in result
        assert "PCR" in result

    def test_levels_no_index(self, bot):
        result = bot.handle_command("levels")
        assert "Usage" in result

    def test_levels_no_data(self, bot):
        result = bot.handle_command("levels", "UNKNOWN")
        assert "No technical data" in result


# ---------------------------------------------------------------------------
# /params
# ---------------------------------------------------------------------------


class TestParamsCommand:
    def test_params_with_data(self, bot):
        mock_all = {
            "NIFTY50": {
                "params": {"sl_multiplier": 1.5, "target_multiplier": 2.5},
                "approved_at": "2026-04-01T10:00:00+05:30",
                "status": "ACTIVE",
                "robustness_score": 0.68,
            },
        }
        with patch("src.backtest.optimizer.param_applier.ApprovedParameterManager") as MockAPM:
            MockAPM.return_value.list_all_approved.return_value = mock_all
            result = bot._cmd_params()

        assert "Strategy Parameters" in result
        assert "NIFTY50" in result
        assert "1.5" in result

    def test_params_empty(self, bot):
        with patch("src.backtest.optimizer.param_applier.ApprovedParameterManager") as MockAPM:
            MockAPM.return_value.list_all_approved.return_value = {}
            result = bot._cmd_params()
        assert "No approved parameters" in result


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthCommand:
    def test_health_basic(self, bot, fake_db):
        now = datetime.now(tz=_IST).isoformat()
        fake_db.insert("system_health", timestamp=now, component="nse_scraper",
                        status="OK", message="", response_time_ms=180)
        fake_db.insert("system_health", timestamp=now, component="news_engine",
                        status="WARNING", message="2 failures in last hour",
                        response_time_ms=None)
        bot._start_time = datetime.now(tz=_IST) - timedelta(hours=5, minutes=42)

        result = bot.handle_command("health")
        assert "System Health" in result
        assert "FULL" in result
        assert "5h 42m" in result
        assert "nse_scraper" in result

    def test_health_kill_active(self, bot, fake_engine, fake_db):
        fake_engine._kill_switch_active = True
        bot._start_time = datetime.now(tz=_IST)
        result = bot.handle_command("health")
        assert "KILLED" in result


# ---------------------------------------------------------------------------
# /kill and /resume
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def test_kill_activates(self, bot, fake_engine):
        assert not fake_engine._kill_switch_active
        result = bot.handle_command("kill")
        assert "KILL SWITCH ACTIVATED" in result
        assert fake_engine._kill_switch_active

    def test_resume_deactivates(self, bot, fake_engine):
        fake_engine.activate_kill_switch()
        assert fake_engine._kill_switch_active
        result = bot.handle_command("resume")
        assert "resumed" in result
        assert not fake_engine._kill_switch_active

    def test_kill_no_engine(self, bot):
        bot._engine = None
        result = bot.handle_command("kill")
        assert "starting up" in result

    def test_resume_no_engine(self, bot):
        bot._engine = None
        result = bot.handle_command("resume")
        assert "starting up" in result


# ---------------------------------------------------------------------------
# Unknown command handling
# ---------------------------------------------------------------------------


class TestUnknownCommand:
    def test_unknown_command(self, bot):
        result = bot.handle_command("foo")
        assert "Unknown command" in result
        assert "/help" in result

    def test_close_match_suggestion(self, bot):
        result = bot.handle_command("staus")  # typo for 'status'
        assert "Unknown command" in result
        assert "status" in result

    def test_slash_prefix_stripped(self, bot):
        result = bot.handle_command("/start")
        assert "Trading DSS Bot Active" in result


# ---------------------------------------------------------------------------
# Priority routing
# ---------------------------------------------------------------------------


class TestPriorityRouting:
    def test_critical_sends_immediately(self, configured_bot):
        with patch.object(configured_bot, "send_message", return_value=True) as mock_send:
            result = configured_bot.send_alert("CRITICAL ALERT", priority="CRITICAL")
        assert result is True
        mock_send.assert_called_once_with("CRITICAL ALERT")

    def test_high_sends_immediately_first_time(self, configured_bot):
        # Set last send far in the past so cooldown has expired
        configured_bot._last_high_send = time.monotonic() - 120
        with patch.object(configured_bot, "send_message", return_value=True) as mock_send:
            result = configured_bot.send_alert("HIGH ALERT", priority="HIGH")
        assert result is True
        mock_send.assert_called_once_with("HIGH ALERT")

    def test_high_rate_limited_to_normal_queue(self, configured_bot):
        # Simulate recent HIGH send
        configured_bot._last_high_send = time.monotonic()
        with patch.object(configured_bot, "send_message", return_value=True) as mock_send:
            result = configured_bot.send_alert("HIGH ALERT 2", priority="HIGH")
        assert result is True
        mock_send.assert_not_called()
        assert not configured_bot._normal_queue.empty()

    def test_normal_queued(self, configured_bot):
        result = configured_bot.send_alert("NORMAL MSG", priority="NORMAL")
        assert result is True
        assert not configured_bot._normal_queue.empty()
        assert configured_bot._normal_queue.get_nowait() == "NORMAL MSG"

    def test_low_queued(self, configured_bot):
        result = configured_bot.send_alert("LOW MSG", priority="LOW")
        assert result is True
        assert not configured_bot._low_queue.empty()
        assert configured_bot._low_queue.get_nowait() == "LOW MSG"

    def test_not_configured_returns_false(self, bot):
        result = bot.send_alert("test", priority="CRITICAL")
        assert result is False


# ---------------------------------------------------------------------------
# Digest flushing
# ---------------------------------------------------------------------------


class TestDigestFlushing:
    def test_flush_normal_digest(self, configured_bot):
        configured_bot._normal_queue.put("Alert 1")
        configured_bot._normal_queue.put("Alert 2")

        with patch.object(configured_bot, "send_message", return_value=True) as mock_send:
            result = configured_bot.flush_normal_digest()
        assert result is True
        sent_text = mock_send.call_args[0][0]
        assert "Hourly Digest" in sent_text
        assert "Alert 1" in sent_text
        assert "Alert 2" in sent_text

    def test_flush_empty_digest(self, configured_bot):
        result = configured_bot.flush_normal_digest()
        assert result is False

    def test_flush_eod_summary(self, configured_bot):
        configured_bot._low_queue.put("EOD item 1")
        configured_bot._normal_queue.put("Leftover normal")

        with patch.object(configured_bot, "send_message", return_value=True) as mock_send:
            result = configured_bot.flush_eod_summary()
        assert result is True
        sent_text = mock_send.call_args[0][0]
        assert "EOD Summary" in sent_text
        assert "EOD item 1" in sent_text
        assert "Leftover normal" in sent_text


# ---------------------------------------------------------------------------
# No-op when not configured
# ---------------------------------------------------------------------------


class TestNotConfigured:
    def test_send_message_noop(self, bot):
        assert bot.send_message("hello") is False

    def test_send_alert_noop(self, bot):
        assert bot.send_alert("hello", "CRITICAL") is False

    def test_start_bot_noop(self, bot):
        bot.start_bot()  # should not crash
        assert not bot.is_running()

    def test_all_commands_work_without_token(self, bot, fake_db):
        """Commands that read from DB should still return text even without token."""
        fake_db.insert("vix_data", timestamp="2026-04-14T10:00:00",
                        vix_value=15.0, vix_change=0.0, vix_change_pct=0.0)
        result = bot.handle_command("vix")
        assert "15.0" in result


# ---------------------------------------------------------------------------
# Signal formatting for proactive alerts
# ---------------------------------------------------------------------------


class TestSignalFormatting:
    def test_format_signal_alert(self, bot):
        result_obj = FakeDecisionResult()
        text = bot.format_signal_alert(result_obj)
        assert "NEW SIGNAL" in text
        assert "NIFTY50" in text
        assert "BUY CALL" in text
        assert "HIGH" in text
        assert "/signal NIFTY50" in text

    def test_format_exit_alert_win(self, bot):
        text = bot.format_exit_alert("NIFTY50", "TARGET_HIT", 2500.0)
        assert "POSITION EXIT" in text
        assert "WIN" in text
        assert "NIFTY50" in text

    def test_format_exit_alert_loss(self, bot):
        text = bot.format_exit_alert("BANKNIFTY", "SL_HIT", -1200.0)
        assert "POSITION EXIT" in text
        assert "LOSS" in text


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_is_running_default_false(self, bot):
        assert not bot.is_running()

    def test_stop_bot_when_not_running(self, bot):
        bot.stop_bot()  # should not crash
        assert not bot.is_running()
