"""
Phase 0 regression tests — data correctness in the live decision path.

Each test pins one bug found in the model review:
  * the options chain was loaded from the oldest (long-expired) expiry
  * the position monitor never received a price (2 params for a 3-param query)
  * DESC-ordered bar lists were sliced as if ascending
  * the stale-data guard compared against the wall clock, not the bar time
  * risk-rejected signals were stored as OPEN trades
  * months-old VIX was served as "current"
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.analysis.technical_aggregator import _has_real_volume, _last_bar_timestamp
from src.database import queries as Q
from src.database.db_manager import DatabaseManager
from src.engine.decision_engine import DecisionEngine
from src.engine.signal_generator import SignalGenerator, TradingSignal, _max_bar_age_seconds
from src.engine.signal_tracker import SignalTracker

_IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 11, 6, 10, 30, tzinfo=_IST)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    d = DatabaseManager(db_path=tmp_path / "phase0.db")
    d.connect()
    d.initialise_schema()
    ts = NOW.isoformat()
    d.execute(
        """
        INSERT OR IGNORE INTO index_master
            (id, display_name, nse_symbol, yahoo_symbol, exchange, lot_size,
             has_options, option_symbol, sector_category, is_active,
             created_at, updated_at)
        VALUES ('NIFTY50','NIFTY 50','NIFTY 50','^NSEI','NSE',65,1,'NIFTY',
                'broad_market',1,?,?)
        """,
        (ts, ts),
    )
    return d


def _bare_engine(db: DatabaseManager) -> DecisionEngine:
    """DecisionEngine without its heavy sub-components (helpers only)."""
    eng = DecisionEngine.__new__(DecisionEngine)
    eng._db = db
    eng._lock = threading.Lock()
    eng._missing_price_cycles = {}
    eng.registry = MagicMock()
    eng.registry.get_index.return_value = SimpleNamespace(option_symbol="NIFTY")
    eng.contract_master = MagicMock()
    eng.contract_master.is_expiry_day.return_value = None
    eng.signal_generator = SimpleNamespace(_min_conf_medium=0.45, _min_conf_high=0.65)
    eng.risk_manager = MagicMock()
    eng.tracker = MagicMock()
    return eng


def _insert_chain(db, expiry: str, ts: datetime, strike: float = 24500.0) -> None:
    for opt in ("CE", "PE"):
        db.execute(
            """
            INSERT INTO options_chain_snapshot
                (index_id, timestamp, expiry_date, strike_price, option_type,
                 open_interest, ltp, iv)
            VALUES ('NIFTY50', ?, ?, ?, ?, 1000, 100.0, 15.0)
            """,
            (ts.isoformat(), expiry, strike, opt),
        )


def _insert_bar(db, ts: datetime, close: float, tf: str = "5m") -> None:
    db.execute(
        "INSERT OR IGNORE INTO price_data "
        "(index_id, timestamp, open, high, low, close, volume, source, timeframe) "
        "VALUES ('NIFTY50', ?, ?, ?, ?, ?, 0, 'test', ?)",
        (ts.isoformat(), close, close + 5, close - 5, close, tf),
    )


# ── Options chain: expiry selection + freshness ───────────────────────────


class TestActiveExpiry:
    def test_skips_long_expired_contract(self, db):
        _insert_chain(db, "2026-04-21", datetime(2026, 4, 20, 15, 0, tzinfo=_IST))
        _insert_chain(db, "2026-11-10", NOW - timedelta(minutes=2))
        eng = _bare_engine(db)
        assert eng._nearest_active_expiry("NIFTY50", NOW) == "2026-11-10"

    def test_expiry_day_contract_dead_after_close(self, db):
        _insert_chain(db, "2026-11-10", NOW)
        _insert_chain(db, "2026-11-17", NOW)
        eng = _bare_engine(db)
        before = datetime(2026, 11, 10, 15, 0, tzinfo=_IST)
        after = datetime(2026, 11, 10, 15, 30, tzinfo=_IST)
        assert eng._nearest_active_expiry("NIFTY50", before) == "2026-11-10"
        assert eng._nearest_active_expiry("NIFTY50", after) == "2026-11-17"

    def test_load_options_chain_uses_active_expiry(self, db):
        _insert_chain(db, "2026-04-21", datetime(2026, 4, 20, 15, 0, tzinfo=_IST))
        _insert_chain(db, "2026-11-10", NOW - timedelta(minutes=2))
        eng = _bare_engine(db)
        chain = eng._load_options_chain("NIFTY50", as_of=NOW)
        assert chain is not None
        assert chain.expiry_date == date(2026, 11, 10)
        assert date(2026, 4, 21) not in chain.available_expiries

    def test_no_active_expiry_returns_none(self, db):
        _insert_chain(db, "2026-04-21", datetime(2026, 4, 20, 15, 0, tzinfo=_IST))
        eng = _bare_engine(db)
        assert eng._load_options_chain("NIFTY50", as_of=NOW) is None


class TestChainValidation:
    def _chain(self, expiry: date, ts: datetime):
        return SimpleNamespace(expiry_date=expiry, timestamp=ts)

    def test_fresh_chain_kept(self, db):
        eng = _bare_engine(db)
        chain = self._chain(date(2026, 11, 10), NOW - timedelta(minutes=3))
        kept, oi, warn = eng._validate_options_chain("NIFTY50", chain, "oi", NOW)
        assert kept is chain and oi == "oi" and warn is None

    def test_old_snapshot_dropped(self, db):
        eng = _bare_engine(db)
        chain = self._chain(date(2026, 11, 10), NOW - timedelta(minutes=40))
        kept, oi, warn = eng._validate_options_chain("NIFTY50", chain, "oi", NOW)
        assert kept is None and oi is None
        assert "min old" in warn

    def test_expired_chain_dropped(self, db):
        eng = _bare_engine(db)
        chain = self._chain(date(2026, 4, 21), NOW - timedelta(minutes=1))
        kept, _, warn = eng._validate_options_chain("NIFTY50", chain, None, NOW)
        assert kept is None
        assert "in the past" in warn


# ── Position monitor ──────────────────────────────────────────────────────


class TestMonitorPrice:
    def test_get_latest_bar_binds_all_params_and_respects_as_of(self, db):
        _insert_bar(db, NOW - timedelta(minutes=5), 24500.0)
        _insert_bar(db, NOW + timedelta(minutes=5), 99999.0)  # "future" bar
        eng = _bare_engine(db)
        bar = eng._get_latest_bar("NIFTY50", "5m", as_of=NOW)
        assert bar is not None and bar["close"] == 24500.0

    def test_fresh_price_used(self, db):
        _insert_bar(db, NOW - timedelta(minutes=1), 24510.0, tf="1m")
        _insert_bar(db, NOW - timedelta(minutes=5), 24500.0, tf="5m")
        eng = _bare_engine(db)
        price, problem = eng._monitor_price("NIFTY50", NOW)
        assert problem is None and price == 24510.0

    def test_daily_bar_is_not_a_fallback(self, db):
        _insert_bar(db, NOW.replace(hour=0, minute=0), 24000.0, tf="1d")
        eng = _bare_engine(db)
        price, problem = eng._monitor_price("NIFTY50", NOW)
        assert problem == "no intraday price in DB"

    def test_stale_price_not_acted_on_and_alerts(self, db):
        _insert_bar(db, datetime.now(tz=_IST) - timedelta(minutes=30), 24500.0)
        eng = _bare_engine(db)
        pos = SimpleNamespace(signal_id="sig-123456789", index_id="NIFTY50", signal_type="BUY_CALL")
        eng.risk_manager._open_positions = {pos.signal_id: pos}
        with patch("src.engine.decision_engine.queue_event") as qe:
            eng.monitor_open_positions()
            eng.monitor_open_positions()
        eng.risk_manager.update_position.assert_not_called()
        alerts = [c for c in qe.call_args_list if c.args[0] == "system_alert"]
        assert len(alerts) == 1
        assert alerts[0].args[1]["alert_type"] == "POSITION_UNMONITORED"


# ── Bar ordering + point-in-time reads ────────────────────────────────────


class TestGatherData:
    def test_rows_are_chronological(self, db):
        for i in range(5):
            _insert_bar(db, NOW - timedelta(days=5 - i), 24000.0 + i, tf="1d")
            _insert_bar(db, NOW - timedelta(minutes=5 * (5 - i)), 24400.0 + i, tf="5m")
        eng = _bare_engine(db)
        eng._mtf_cache, eng._iv_history_cache, eng._fii_futures_cache = {}, {}, {}
        eng.registry.get_index.return_value = SimpleNamespace(has_options=False)
        _, daily, intraday, *_ = eng._step_gather_data("NIFTY50", as_of=NOW)
        assert [r["close"] for r in daily] == sorted(r["close"] for r in daily)
        assert intraday[-1]["close"] == 24404.0  # newest bar last

    def test_old_vix_not_served_as_current(self, db):
        db.execute(
            "INSERT INTO vix_data (timestamp, vix_value) VALUES (?, ?)",
            ((NOW - timedelta(days=150)).isoformat(), 13.0),
        )
        eng = _bare_engine(db)
        eng._mtf_cache, eng._iv_history_cache, eng._fii_futures_cache = {}, {}, {}
        eng.registry.get_index.return_value = SimpleNamespace(has_options=False)
        out = eng._step_gather_data("NIFTY50", as_of=NOW)
        assert out[5] is None  # vix_value

    def test_fresh_vix_used(self, db):
        db.execute(
            "INSERT INTO vix_data (timestamp, vix_value) VALUES (?, ?)",
            ((NOW - timedelta(minutes=2)).isoformat(), 14.5),
        )
        eng = _bare_engine(db)
        eng._mtf_cache, eng._iv_history_cache, eng._fii_futures_cache = {}, {}, {}
        eng.registry.get_index.return_value = SimpleNamespace(has_options=False)
        assert eng._step_gather_data("NIFTY50", as_of=NOW)[5] == 14.5


# ── Stale penalty ─────────────────────────────────────────────────────────


def _signal(conf: float, level: str = "HIGH") -> TradingSignal:
    return TradingSignal(
        signal_id="s1", index_id="NIFTY50", generated_at=NOW,
        signal_type="BUY_CALL", confidence_level=level, confidence_score=conf,
        entry_price=24500, target_price=24600, stop_loss=24450, risk_reward_ratio=2.0,
        regime="TREND_UP", weighted_score=1.2, vote_breakdown={}, risk_level="NORMAL",
        position_size_modifier=1.0, suggested_lot_count=1, estimated_max_loss=50,
        estimated_max_profit=100, reasoning="test",
    )


class TestStalePenalty:
    def test_penalty_below_medium_blocks_trade(self, db):
        eng = _bare_engine(db)
        regime = DecisionEngine._fallback_regime("NIFTY50", "test")
        out = eng._apply_stale_penalty("NIFTY50", _signal(0.72), regime, 0.3)
        assert out.signal_type == "NO_TRADE"
        assert "Stale data" in out.reasoning

    def test_penalty_downgrades_level(self, db):
        eng = _bare_engine(db)
        regime = DecisionEngine._fallback_regime("NIFTY50", "test")
        out = eng._apply_stale_penalty("NIFTY50", _signal(0.70), regime, 0.1)
        assert out.signal_type == "BUY_CALL"
        assert out.confidence_score == pytest.approx(0.60)
        assert out.confidence_level == "MEDIUM"


# ── Tracker: only taken trades are OPEN ───────────────────────────────────


class TestTrackerActionability:
    def test_rejected_signal_not_open_and_not_counted(self, db):
        tracker = SignalTracker(db, capital=100_000.0)
        tracker.record_signal(_signal(0.72), is_actionable=False)
        row = db.fetch_one("SELECT outcome FROM trading_signals", ())
        assert row["outcome"] is None
        count = db.fetch_one(Q.COUNT_SIGNALS_TODAY, ("NIFTY50", NOW.isoformat()))
        assert count["count"] == 0

    def test_taken_signal_open_with_refined_levels_and_signal_time(self, db):
        tracker = SignalTracker(db, capital=100_000.0)
        sig = _signal(0.72)
        sig.refined_entry, sig.refined_stop_loss, sig.refined_target = 24501.0, 24430.0, 24650.0
        tracker.record_signal(sig, is_actionable=True)
        row = db.fetch_one(
            "SELECT outcome, entry_price, stop_loss, target_price, generated_at "
            "FROM trading_signals", (),
        )
        assert row["outcome"] == "OPEN"
        assert (row["entry_price"], row["stop_loss"], row["target_price"]) == (24501.0, 24430.0, 24650.0)
        assert row["generated_at"] == NOW.isoformat()

    def test_invalid_refined_signal_inferred_not_actionable(self, db):
        tracker = SignalTracker(db, capital=100_000.0)
        sig = _signal(0.72)
        sig.is_valid = False
        tracker.record_signal(sig)
        assert db.fetch_one("SELECT outcome FROM trading_signals", ())["outcome"] is None


# ── Signal generator data-quality gates ───────────────────────────────────


class TestSignalGeneratorGates:
    def test_bar_age_limits(self):
        assert _max_bar_age_seconds("5m") == 600
        assert _max_bar_age_seconds("15m") == 1800
        assert _max_bar_age_seconds("1d") == 600  # daily bar always stale intraday

    def test_expiry_day_from_chain_not_weekday(self):
        tech = SimpleNamespace(options=SimpleNamespace(expiry_date=date(2026, 11, 9)))
        monday = datetime(2026, 11, 9, 11, 0, tzinfo=_IST)
        thursday = datetime(2026, 11, 12, 11, 0, tzinfo=_IST)
        assert SignalGenerator._is_expiry_day("NIFTY50", monday, tech) is True
        assert SignalGenerator._is_expiry_day("NIFTY50", thursday, tech) is False

    def test_no_volume_disables_volume_and_smart_money(self):
        tech = SimpleNamespace(options=None, smart_money=object(), volume_available=False)
        avail = SignalGenerator._available_components(tech, None, None)
        assert "volume" not in avail and "smart_money" not in avail
        assert {"trend", "momentum"} <= avail

    def test_volume_present_keeps_components(self):
        tech = SimpleNamespace(options=None, smart_money=object(), volume_available=True)
        avail = SignalGenerator._available_components(tech, None, None)
        assert {"volume", "smart_money"} <= avail


class TestAggregatorHelpers:
    def test_has_real_volume(self):
        assert _has_real_volume(pd.DataFrame({"volume": [0] * 20})) is False
        assert _has_real_volume(pd.DataFrame({"volume": [100] * 20})) is True
        assert _has_real_volume(pd.DataFrame({"close": [1, 2]})) is False

    def test_last_bar_timestamp_is_data_time(self):
        df = pd.DataFrame({"timestamp": ["2026-11-06T10:00:00+05:30", "2026-11-06T10:05:00+05:30"]})
        ts = _last_bar_timestamp(df)
        assert ts == datetime(2026, 11, 6, 10, 5, tzinfo=_IST)
        assert _last_bar_timestamp(pd.DataFrame({"close": [1]})) is None
