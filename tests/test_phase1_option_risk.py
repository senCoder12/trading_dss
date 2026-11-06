"""
Phase 1 regression tests — option-native risk wired into the decision path.

Risk-manager internals are covered in test_risk_manager.py; this file pins
the engine-side wiring and the IV-percentile fixes.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.analysis.options_analysis import MIN_IV_HISTORY_OBSERVATIONS, calculate_iv_percentile
from src.database.db_manager import DatabaseManager
from src.engine.decision_engine import DecisionEngine
from src.engine.risk_manager import PositionUpdate

_IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 11, 5, 10, 30, tzinfo=_IST)


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    d = DatabaseManager(db_path=tmp_path / "p1.db")
    d.connect()
    d.initialise_schema()
    ts = NOW.isoformat()
    d.execute(
        """INSERT OR IGNORE INTO index_master
           (id, display_name, nse_symbol, yahoo_symbol, exchange, lot_size, has_options,
            option_symbol, sector_category, is_active, created_at, updated_at)
           VALUES ('NIFTY50','NIFTY 50','NIFTY 50','^NSEI','NSE',65,1,'NIFTY','broad_market',1,?,?)""",
        (ts, ts),
    )
    return d


def _bare_engine(db) -> DecisionEngine:
    eng = DecisionEngine.__new__(DecisionEngine)
    eng._db = db
    eng._lock = threading.Lock()
    eng._missing_price_cycles = {}
    eng.registry = MagicMock()
    eng.contract_master = MagicMock()
    eng.risk_manager = MagicMock()
    eng.tracker = MagicMock()
    return eng


# ── IV percentile ─────────────────────────────────────────────────────────


class TestIVPercentile:
    def test_short_history_is_neutral_not_cheap(self):
        # 10 observations: the old code produced a bias from them
        assert calculate_iv_percentile(15.0, [20.0] * 10) == (50.0, 50.0, "NEUTRAL")

    def test_enough_history_gives_a_bias(self):
        hist = [20.0] * MIN_IV_HISTORY_OBSERVATIONS
        ivp, _, bias = calculate_iv_percentile(10.0, hist)
        assert ivp == 0.0 and bias == "BUY_OPTIONS"

    def test_daily_series_takes_one_median_per_day(self):
        rows = [
            {"timestamp": "2026-11-03T09:20:00+05:30", "atm_iv": 14.0},
            {"timestamp": "2026-11-03T10:20:00+05:30", "atm_iv": 16.0},
            {"timestamp": "2026-11-03T11:20:00+05:30", "atm_iv": 30.0},
            {"timestamp": "2026-11-04T09:20:00+05:30", "atm_iv": 15.0},
            {"timestamp": "2026-11-04T09:25:00+05:30", "atm_iv": 0.0},   # invalid
        ]
        assert DecisionEngine._daily_iv_series(rows) == [16.0, 15.0]

    def test_insufficient_rank_history_does_not_bias_options_vote(self):
        from src.analysis.indicators.options_indicators import OptionsIndicators

        oi = OptionsIndicators()
        result = oi.calculate_iv_rank(15.0, [20.0] * 5)
        assert result.iv_regime == "insufficient_history"  # contract unchanged
        assert result.iv_percentile == 0.0                  # … which used to read as "cheap"


# ── Risk step wiring ──────────────────────────────────────────────────────


class TestRiskStepWiring:
    def _signal(self):
        return SimpleNamespace(signal_type="BUY_CALL")

    def test_passes_decision_time_and_no_next_chain_on_normal_day(self, db):
        eng = _bare_engine(db)
        chain = SimpleNamespace(expiry_date=date(2026, 11, 10), available_expiries=())
        eng.risk_manager.validate_and_refine_signal.return_value = SimpleNamespace(
            is_valid=True, signal_type="BUY_CALL", rejection_reasons=[],
        )
        _, actionable = eng._step_risk(self._signal(), chain, index_id="NIFTY50", now=NOW)
        kwargs = eng.risk_manager.validate_and_refine_signal.call_args.kwargs
        assert actionable is True
        assert kwargs["_now"] == NOW and kwargs["next_chain"] is None

    def test_expiry_day_loads_next_expiry_chain(self, db):
        eng = _bare_engine(db)
        today = NOW.date()
        chain = SimpleNamespace(expiry_date=today, available_expiries=(today, date(2026, 11, 12)))
        nxt = SimpleNamespace(expiry_date=date(2026, 11, 12), timestamp=NOW, strikes=(1,))
        eng._load_options_chain = MagicMock(return_value=nxt)
        eng.risk_manager.validate_and_refine_signal.return_value = SimpleNamespace(
            is_valid=False, signal_type="BUY_CALL", rejection_reasons=["x"],
        )
        eng._step_risk(self._signal(), chain, index_id="NIFTY50", now=NOW)
        eng._load_options_chain.assert_called_once_with("NIFTY50", as_of=None, expiry="2026-11-12")
        assert eng.risk_manager.validate_and_refine_signal.call_args.kwargs["next_chain"] is nxt


# ── Monitor: option mark + net P&L ────────────────────────────────────────


class TestMonitorOptionMark:
    def _pos(self):
        return SimpleNamespace(
            signal_id="sig-abcdef12", index_id="NIFTY50", signal_type="BUY_CALL",
            option_type="CE", strike=24550.0, expiry=date(2026, 11, 10),
            half_spread=0.5, entry_premium=150.0, entry_price=24500.0,
        )

    def _chain_row(self, db, ts, ltp, bid=0.0, ask=0.0):
        db.execute(
            """INSERT INTO options_chain_snapshot (index_id, timestamp, expiry_date, strike_price,
               option_type, open_interest, ltp, bid_price, ask_price)
               VALUES ('NIFTY50', ?, '2026-11-10', 24550, 'CE', 1000, ?, ?, ?)""",
            (ts.isoformat(), ltp, bid, ask),
        )

    def test_option_mark_prefers_mid(self, db):
        self._chain_row(db, NOW - timedelta(minutes=2), ltp=161.0, bid=159.0, ask=160.0)
        assert _bare_engine(db)._option_mark(self._pos(), NOW) == pytest.approx(159.5)

    def test_option_mark_ignores_stale_quote(self, db):
        self._chain_row(db, NOW - timedelta(minutes=30), ltp=161.0)
        assert _bare_engine(db)._option_mark(self._pos(), NOW) is None

    def test_exit_records_net_pnl_and_option_exit_price(self, db):
        eng = _bare_engine(db)
        pos = self._pos()
        eng.risk_manager._open_positions = {pos.signal_id: pos}
        eng._monitor_price = MagicMock(return_value=(24600.0, None))
        eng._option_mark = MagicMock(return_value=180.0)
        eng.risk_manager.update_position.return_value = PositionUpdate(
            signal_id=pos.signal_id, action="EXIT_TARGET", current_pnl=5000.0,
            current_pnl_pct=1.0, time_in_trade_minutes=30, mark_premium=180.0,
            premium_source="quote",
        )
        eng.risk_manager.close_position.return_value = 4321.0  # net of costs
        with patch("src.engine.decision_engine.queue_event"):
            eng.monitor_open_positions()
        assert eng.risk_manager.update_position.call_args.kwargs["option_ltp"] == 180.0
        assert eng.risk_manager.close_position.call_args.kwargs["exit_premium"] == 180.0
        kw = eng.tracker.record_outcome.call_args.kwargs
        assert kw["pnl"] == 4321.0 and kw["exit_price"] == pytest.approx(179.5)


# ── Hero-zero default stop ────────────────────────────────────────────────


def test_hero_zero_put_default_stop_is_above_spot():
    from src.engine.signal_generator import SignalGenerator

    gen = SignalGenerator(MagicMock())
    gen._get_signals_today_count = lambda *a: 0  # type: ignore[method-assign]
    hz = SimpleNamespace(signal_direction="BUY_PUT", reasoning="hz")
    regime = DecisionEngine._fallback_regime("NIFTY50", "t")
    sig = gen._build_hero_zero_signal("NIFTY50", NOW, 24500.0, hz, regime, None, None, None)
    assert sig.stop_loss > 24500.0 and sig.target_price < 24500.0
