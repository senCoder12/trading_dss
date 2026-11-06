"""
Tests for the premium-native Paper Trading Engine and its adapter.

The old suite tested index-point fills (index LTP ± "slippage points"),
which was the bug.  It was rewritten for premium semantics:

- Adapter: only actionable, option-native RefinedSignals become orders
- Fills: option ASK of the first quote at/after decision + latency;
  LTP + half spread without a book; no quote within the window → NO FILL
  (reason recorded); cash debited premium × qty + entry-leg costs; no re-size
- Marks: fresh quote (mid) else model re-price flagged "model"
- Exits: premium stop / spot stop / target / trailing / 60-min time stop /
  15:15 forced exit / hero-zero 14:55, fills at BID; parity with RiskManager
- P&L: (bid − ask) × qty − round_trip_cost; WIN iff net > 0
- Caps: same defaults as the live RiskConfig
- Sharpe/Sortino from DAILY returns on equity incl. zero-P&L days
- Persistence: restart restores positions/pending orders/cash/trades;
  a position left open overnight is settled at its entry day's 15:15 mark
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.database.db_manager import DatabaseManager
from src.engine.option_contract import estimate_premium
from src.engine.option_costs import round_trip_cost, schedule_for
from src.engine.risk_manager import RefinedSignal, RiskConfig, RiskManager
from src.paper_trading import paper_engine as pe
from src.paper_trading.adapter import PaperTradingAdapter
from src.paper_trading.edge_tracker import annualised_sharpe, daily_equity_returns
from src.paper_trading.paper_engine import (
    RM_ACTION_TO_EXIT_REASON,
    PaperExecution,
    PaperTrade,
    PaperTradingConfig,
    PaperTradingEngine,
)
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 11, 5, 10, 30, tzinfo=_IST)   # Thursday, decision time
EXPIRY = "2026-11-12"
STRIKE = 24_550.0
QTY = 130                                           # 2 lots × 65


def _at(h: int, m: int = 0, s: int = 0, day: int = 5) -> datetime:
    return datetime(2026, 11, day, h, m, s, tzinfo=_IST)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture()
def clock(monkeypatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(pe, "get_ist_now", c)
    return c


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    d = DatabaseManager(db_path=tmp_path / "paper.db")
    d.connect()
    d.initialise_schema()
    ts = NOW.isoformat()
    for idx, sym in (("NIFTY50", "NIFTY"), ("BANKNIFTY", "BANKNIFTY"), ("FINNIFTY", "FINNIFTY")):
        d.execute(
            """INSERT OR IGNORE INTO index_master
               (id, display_name, nse_symbol, yahoo_symbol, exchange, lot_size, has_options,
                option_symbol, sector_category, is_active, created_at, updated_at)
               VALUES (?,?,?,?, 'NSE', 65, 1, ?, 'broad_market', 1, ?, ?)""",
            (idx, idx, idx, f"^{idx}", sym, ts, ts),
        )
    return d


def _engine(db, **cfg) -> PaperTradingEngine:
    base = dict(initial_capital=100_000.0, active_indices=["NIFTY50", "BANKNIFTY", "FINNIFTY"])
    base.update(cfg)
    return PaperTradingEngine(db, PaperTradingConfig(**base))


def _refined(**kw) -> RefinedSignal:
    call = kw.get("signal_type", "BUY_CALL") == "BUY_CALL"
    base = dict(
        signal_id=str(uuid.uuid4()), index_id="NIFTY50", generated_at=NOW,
        signal_type="BUY_CALL", confidence_level="HIGH", confidence_score=0.7,
        entry_price=24_500.0, target_price=24_700.0 if call else 24_300.0,
        stop_loss=24_400.0 if call else 24_600.0, risk_reward_ratio=2.0, regime="TRENDING",
        weighted_score=1.0, vote_breakdown={}, risk_level="NORMAL", position_size_modifier=1.0,
        suggested_lot_count=2, estimated_max_loss=0.0, estimated_max_profit=0.0,
        reasoning="test", warnings=[], outcome=None, actual_exit_price=None, actual_pnl=None,
        closed_at=None, data_completeness=1.0, signals_generated_today=1,
        refined_entry=24_500.0,
        refined_stop_loss=24_400.0 if call else 24_600.0,
        refined_target=24_700.0 if call else 24_300.0,
        lots=2, lot_size=65, quantity=QTY,
        recommended_strike=STRIKE, recommended_expiry=EXPIRY,
        option_type="CE" if call else "PE",
        entry_premium=150.5, premium_mid=150.0, half_spread=0.5, spread_known=True,
        premium_stop=110.0, premium_target=210.0, option_iv=0.14,
        max_loss_amount=5_500.0, is_valid=True,
    )
    base.update(kw)
    return RefinedSignal(**base)


def _quote(db, ts, ltp=0.0, bid=0.0, ask=0.0, index="NIFTY50", strike=STRIKE,
           opt="CE", expiry=EXPIRY) -> None:
    db.execute(
        """INSERT INTO options_chain_snapshot (index_id, timestamp, expiry_date, strike_price,
           option_type, open_interest, ltp, bid_price, ask_price)
           VALUES (?, ?, ?, ?, ?, 1000, ?, ?, ?)""",
        (index, ts.isoformat(), expiry, strike, opt, ltp, bid, ask),
    )


def _book(db, ts, mid, half=0.5, **kw) -> None:
    _quote(db, ts, ltp=mid, bid=mid - half, ask=mid + half, **kw)


def _spot(db, ts, close, index="NIFTY50") -> None:
    db.execute(
        """INSERT INTO price_data (index_id, timestamp, open, high, low, close, volume, timeframe)
           VALUES (?, ?, ?, ?, ?, ?, 0, '1m')""",
        (index, ts.isoformat(), close, close, close, close),
    )


def _entry_leg(premium, qty=QTY):
    s = schedule_for("NSE")
    return round(round_trip_cost(premium, 0.0, qty).total - s.brokerage_per_order * (1 + s.gst_pct), 2)


def _open(db, clock, eng=None, sig=None, ask=151.0, fill_at=None, **cfg):
    """Submit a signal at NOW and fill it on a quote at 10:31."""
    eng = eng or _engine(db, **cfg)
    sig = sig or _refined()
    fill_at = fill_at or _at(10, 31)
    opt = sig.option_type
    _quote(db, fill_at, ltp=ask - 0.5, bid=ask - 1.0, ask=ask, index=sig.index_id,
           strike=sig.recommended_strike, opt=opt)
    clock.now = fill_at
    order = eng.submit_order(sig, decision_time=sig.generated_at)
    assert order is not None and order.status == "FILLED", order and order.reason
    return eng, sig, eng.open_positions[order.position_id]


def _result(signal, actionable=True, ts=NOW):
    return SimpleNamespace(index_id=signal.index_id, timestamp=ts, signal=signal, is_actionable=actionable)


# ===========================================================================
# 1. Adapter
# ===========================================================================


class TestAdapter:
    def test_non_actionable_result_creates_nothing(self, db, clock):
        eng = _engine(db)
        assert PaperTradingAdapter(eng).on_decision(_result(_refined(), actionable=False)) is None
        assert not eng.pending_orders and not eng.open_positions

    def test_actionable_without_option_contract_is_skipped(self, db, clock):
        eng = _engine(db)
        spot_only = SimpleNamespace(signal_type="BUY_CALL", index_id="NIFTY50", option_type=None)
        assert PaperTradingAdapter(eng).on_decision(_result(spot_only)) is None
        assert not eng.pending_orders

    def test_no_trade_signal_is_skipped(self, db, clock):
        eng = _engine(db)
        sig = SimpleNamespace(signal_type="NO_TRADE", index_id="NIFTY50", option_type=None)
        assert PaperTradingAdapter(eng).on_decision(_result(sig)) is None

    def test_actionable_signal_pending_then_filled_on_tick(self, db, clock):
        eng = _engine(db)
        adapter = PaperTradingAdapter(eng)
        sig = _refined()
        order = adapter.on_decision(_result(sig))
        assert order is not None and order.status == "PENDING"
        assert order.decision_time == NOW
        _book(db, _at(10, 32), 151.0)
        clock.now = _at(10, 33)
        assert adapter.on_monitor_tick() == []
        assert order.status == "FILLED"
        pos = eng.open_positions[order.position_id]
        assert pos.execution_price == pytest.approx(151.5)    # ask
        assert pos.signal_id == sig.signal_id

    def test_on_decisions_filters_list(self, db, clock):
        eng = _engine(db)
        a, b = _refined(), _refined(index_id="BANKNIFTY")
        orders = PaperTradingAdapter(eng).on_decisions([_result(a), _result(b, actionable=False), None])
        assert [o.signal_id for o in orders] == [a.signal_id]

    def test_adapter_never_raises(self, db, clock):
        eng = MagicMock()
        eng.on_tick.side_effect = RuntimeError("boom")
        eng.submit_order.side_effect = RuntimeError("boom")
        adapter = PaperTradingAdapter(eng)
        assert adapter.on_monitor_tick() == []
        assert adapter.on_decision(_result(_refined())) is None


# ===========================================================================
# 2. Fills
# ===========================================================================


class TestFills:
    def test_fills_at_ask_of_first_quote_after_latency(self, db, clock):
        eng = _engine(db, fill_latency_seconds=30)
        _book(db, NOW - timedelta(minutes=1), 140.0)            # before the decision
        _book(db, NOW + timedelta(seconds=10), 145.0)           # inside the latency
        _book(db, NOW + timedelta(seconds=90), 152.0)           # first eligible
        _book(db, NOW + timedelta(seconds=150), 160.0)
        clock.now = NOW + timedelta(minutes=3)
        order = eng.submit_order(_refined(), decision_time=NOW)
        assert order.status == "FILLED"
        pos = eng.open_positions[order.position_id]
        assert pos.execution_price == pytest.approx(152.5)
        assert pos.entry_fill_source == "ask"
        assert pos.entry_timestamp == NOW + timedelta(seconds=90)
        assert pos.decision_time == NOW

    def test_no_look_ahead_order_waits_for_future_quote(self, db, clock):
        eng = _engine(db)
        _book(db, NOW + timedelta(seconds=60), 152.0)
        clock.now = NOW                                          # quote not yet "arrived"
        order = eng.submit_order(_refined(), decision_time=NOW)
        assert order.status == "PENDING"
        clock.now = NOW + timedelta(seconds=61)
        eng.on_tick()
        assert order.status == "FILLED"

    def test_no_book_fills_at_ltp_plus_half_spread(self, db, clock):
        eng = _engine(db)
        _quote(db, NOW + timedelta(seconds=60), ltp=150.0)       # no bid/ask
        clock.now = NOW + timedelta(minutes=2)
        order = eng.submit_order(_refined(half_spread=0.75), decision_time=NOW)
        pos = eng.open_positions[order.position_id]
        assert pos.execution_price == pytest.approx(150.75)
        assert pos.entry_fill_source == "ltp+half_spread"

    def test_no_quote_within_window_is_no_fill_with_reason(self, db, clock):
        eng = _engine(db, fill_window_minutes=5)
        sig = _refined()
        clock.now = NOW
        order = eng.submit_order(sig, decision_time=NOW)
        assert order.status == "PENDING"
        clock.now = NOW + timedelta(minutes=3)
        eng.on_tick()
        assert order.status == "PENDING"                       # window still open
        _book(db, NOW + timedelta(minutes=6), 150.0)             # arrives too late
        clock.now = NOW + timedelta(minutes=7)
        eng.on_tick()
        assert order.status == "NO_FILL"
        assert order.reason.startswith("no_quote")
        assert not eng.open_positions and not eng.pending_orders
        assert eng.unfilled_orders == [order]
        assert eng.missed_signals[-1].reason_missed == "no_fill:no_quote"
        assert eng.missed_signals[-1].signal_id == sig.signal_id
        assert eng.daily_ledger[NOW.date()].signals_missed == 1
        assert eng.cash == pytest.approx(100_000.0)

    def test_cash_debited_premium_times_qty_plus_entry_costs(self, db, clock):
        eng, sig, pos = _open(db, clock, ask=151.0)
        expected_cost = _entry_leg(151.0)
        assert pos.entry_cost == pytest.approx(expected_cost)
        assert eng.cash == pytest.approx(100_000.0 - 151.0 * QTY - expected_cost)
        # realised equity is unchanged by opening (cash moved into the position)
        assert eng.current_capital == pytest.approx(100_000.0)

    def test_approved_size_is_not_resized(self, db, clock):
        eng, sig, pos = _open(db, clock, sig=_refined(lots=3, quantity=195))
        assert (pos.lots, pos.lot_size, pos.quantity) == (3, 65, 195)

    def test_insufficient_cash_is_no_fill(self, db, clock):
        eng = _engine(db, initial_capital=10_000.0)
        _book(db, NOW + timedelta(seconds=60), 150.0)
        clock.now = NOW + timedelta(minutes=2)
        order = eng.submit_order(_refined(), decision_time=NOW)
        assert order.status == "NO_FILL" and order.reason.startswith("insufficient_cash")
        assert eng.cash == pytest.approx(10_000.0)

    def test_slippage_cap_rejects_runaway_ask(self, db, clock):
        eng = _engine(db, max_entry_slippage_pct=0.02)
        _book(db, NOW + timedelta(seconds=60), 160.0)            # ask 160.5 vs planned 150.5
        clock.now = NOW + timedelta(minutes=2)
        order = eng.submit_order(_refined(), decision_time=NOW)
        assert order.status == "NO_FILL" and order.reason.startswith("slippage_cap")

    def test_no_fill_at_or_after_force_exit(self, db, clock):
        eng = _engine(db)
        t = _at(15, 14, 50)
        _book(db, _at(15, 15, 30), 150.0)
        clock.now = _at(15, 16)
        order = eng.submit_order(_refined(generated_at=t), decision_time=t)
        assert order.status == "NO_FILL" and order.reason.startswith("past_force_exit")

    def test_on_signal_returns_execution_when_filled(self, db, clock):
        eng = _engine(db)
        _book(db, NOW + timedelta(seconds=60), 151.0)
        clock.now = NOW + timedelta(minutes=2)
        ex = eng.on_signal(_refined(), {"timestamp": NOW})
        assert isinstance(ex, PaperExecution)
        assert ex.slippage_applied == pytest.approx(151.5 - 150.5)

    def test_duplicate_signal_is_ignored(self, db, clock):
        eng, sig, _ = _open(db, clock)
        assert eng.submit_order(sig, decision_time=NOW) is None
        assert len(eng.open_positions) == 1


# ===========================================================================
# 3. Caps and filters (same defaults as live RiskConfig)
# ===========================================================================


class TestCapsAndFilters:
    def test_default_caps_match_live_risk_config(self):
        cfg, rc = PaperTradingConfig(), RiskConfig()
        assert cfg.max_open_positions == rc.max_open_positions
        assert cfg.max_positions_per_index == rc.max_positions_per_index
        assert cfg.max_same_direction_positions == rc.max_same_direction_positions
        assert cfg.max_risk_per_day_pct == rc.max_risk_per_day_pct
        assert cfg.force_exit_time == rc.force_exit_time
        assert (cfg.time_stop_minutes, cfg.time_stop_min_r) == (rc.time_stop_minutes, rc.time_stop_min_r)

    def test_per_index_cap_counts_pending_orders(self, db, clock):
        eng = _engine(db)
        assert eng.submit_order(_refined(), decision_time=NOW).status == "PENDING"
        assert eng.submit_order(_refined(signal_type="BUY_PUT"), decision_time=NOW) is None
        assert eng.missed_signals[-1].reason_missed == "existing_position_in_index"

    def test_same_direction_cap(self, db, clock):
        eng = _engine(db)
        eng.submit_order(_refined(index_id="NIFTY50"), decision_time=NOW)
        eng.submit_order(_refined(index_id="BANKNIFTY"), decision_time=NOW)
        assert eng.submit_order(_refined(index_id="FINNIFTY"), decision_time=NOW) is None
        assert eng.missed_signals[-1].reason_missed == "max_same_direction"

    def test_max_open_positions(self, db, clock):
        eng = _engine(db, max_open_positions=1)
        eng.submit_order(_refined(index_id="NIFTY50"), decision_time=NOW)
        assert eng.submit_order(_refined(index_id="BANKNIFTY"), decision_time=NOW) is None
        assert eng.missed_signals[-1].reason_missed == "max_positions"

    def test_daily_loss_limit(self, db, clock):
        eng = _engine(db)
        eng._ensure_daily_record(NOW.date())
        eng.daily_ledger[NOW.date()].realized_pnl = -4_001.0
        assert eng.submit_order(_refined(), decision_time=NOW) is None
        assert eng.missed_signals[-1].reason_missed == "daily_loss_limit"

    def test_confidence_filter(self, db, clock):
        eng = _engine(db, min_confidence="HIGH")
        assert eng.submit_order(_refined(confidence_level="MEDIUM"), decision_time=NOW) is None
        assert eng.missed_signals[-1].reason_missed == "below_min_confidence"

    def test_kill_switch_and_watchdog_pause_block_entries(self, db, clock):
        eng = _engine(db)
        eng.activate_kill_switch("test")
        assert eng.submit_order(_refined(), decision_time=NOW) is None
        eng.deactivate_kill_switch()
        eng.pause_new_trades = True
        assert eng.submit_order(_refined(), decision_time=NOW) is None
        eng.pause_new_trades = False
        assert eng.submit_order(_refined(), decision_time=NOW) is not None

    def test_inactive_index_filtered(self, db, clock):
        eng = _engine(db, active_indices=["NIFTY50"])
        assert eng.submit_order(_refined(index_id="BANKNIFTY"), decision_time=NOW) is None


# ===========================================================================
# 4. Marks
# ===========================================================================


class TestMarks:
    def test_fresh_quote_mark_is_mid(self, db, clock):
        eng, sig, pos = _open(db, clock)
        _quote(db, _at(10, 40), ltp=161.0, bid=159.0, ask=160.0)
        clock.now = _at(10, 41)
        eng.on_tick()
        assert pos.current_price == pytest.approx(159.5)
        assert pos.mark_source == "quote"
        fill = 159.0
        costs = round_trip_cost(151.0, fill, QTY).total
        assert pos.unrealized_pnl == pytest.approx((fill - 151.0) * QTY - costs, abs=0.01)

    def test_model_mark_fallback_is_flagged(self, db, clock):
        eng, sig, pos = _open(db, clock)
        now = _at(10, 45)                                       # 10:31 quote is > 10 min old
        _spot(db, _at(10, 44), 24_560.0)
        clock.now = now
        eng.on_tick()
        expected = estimate_premium("CE", STRIKE, date(2026, 11, 12), 0.14, 24_560.0, now,
                                    150.0, 24_500.0, NOW)
        assert pos.mark_source == "model"
        assert pos.current_price == pytest.approx(expected, abs=0.01)

    def test_model_exit_is_flagged_on_trade(self, db, clock):
        eng, sig, pos = _open(db, clock)
        _spot(db, _at(10, 49), 24_390.0)                         # below the spot stop
        clock.now = _at(10, 50)
        trades = eng.on_tick()
        assert len(trades) == 1
        t = trades[0]
        assert t.exit_mark_source == "model" and t.exit_fill_source == "model-half_spread"
        assert t.exit_reason == "STOP_LOSS_HIT"
        assert t.actual_exit_price == pytest.approx(t.exit_price - 0.5, abs=0.01)

    def test_no_quote_no_spot_position_not_evaluated(self, db, clock):
        eng, sig, pos = _open(db, clock)
        clock.now = _at(11, 0)
        assert eng.on_tick() == []
        assert pos.position_id in eng.open_positions


# ===========================================================================
# 5. Exits (rules mirror RiskManager; fills at BID)
# ===========================================================================


class TestExits:
    def _tick(self, db, clock, eng, when, mid=None, spot=None, half=0.5):
        if mid is not None:
            _book(db, when - timedelta(seconds=30), mid, half=half)
        if spot is not None:
            _spot(db, when - timedelta(seconds=30), spot)
        clock.now = when
        return eng.on_tick()

    def test_premium_stop_exits_at_bid(self, db, clock):
        eng, sig, pos = _open(db, clock)
        trades = self._tick(db, clock, eng, _at(10, 50), mid=109.0)
        assert [t.exit_reason for t in trades] == ["STOP_LOSS_HIT"]
        assert trades[0].actual_exit_price == pytest.approx(108.5)   # bid
        assert trades[0].exit_fill_source == "bid"
        assert trades[0].outcome == "LOSS"

    def test_spot_stop_trigger_exits_even_if_premium_above_stop(self, db, clock):
        eng, sig, pos = _open(db, clock)
        trades = self._tick(db, clock, eng, _at(10, 50), mid=140.0, spot=24_395.0)
        assert [t.exit_reason for t in trades] == ["STOP_LOSS_HIT"]

    def test_put_spot_stop_is_above(self, db, clock):
        sig = _refined(signal_type="BUY_PUT")
        eng, sig, pos = _open(db, clock, sig=sig)
        _quote(db, _at(10, 49, 30), ltp=140.0, bid=139.5, ask=140.5, opt="PE")
        _spot(db, _at(10, 49, 30), 24_605.0)
        clock.now = _at(10, 50)
        assert [t.exit_reason for t in eng.on_tick()] == ["STOP_LOSS_HIT"]

    def test_premium_target_exits_at_bid_as_win(self, db, clock):
        eng, sig, pos = _open(db, clock)
        trades = self._tick(db, clock, eng, _at(11, 0), mid=211.0)
        t = trades[0]
        assert t.exit_reason == "TARGET_HIT" and t.actual_exit_price == pytest.approx(210.5)
        assert t.outcome == "WIN"

    def test_spot_target_trigger(self, db, clock):
        eng, sig, pos = _open(db, clock)
        trades = self._tick(db, clock, eng, _at(11, 0), mid=180.0, spot=24_705.0)
        assert [t.exit_reason for t in trades] == ["TARGET_HIT"]

    def test_time_stop_after_60_min_below_0_3r(self, db, clock):
        eng, sig, pos = _open(db, clock)
        # 59 min after the decision: hold
        assert self._tick(db, clock, eng, _at(11, 29), mid=155.0) == []
        # 60 min: (155 − 150.5) / (150.5 − 110) = 0.11 R < 0.3 R → exit
        trades = self._tick(db, clock, eng, _at(11, 30), mid=155.0)
        assert [t.exit_reason for t in trades] == ["TIME_STOP"]

    def test_no_time_stop_when_working(self, db, clock):
        eng, sig, pos = _open(db, clock)
        assert self._tick(db, clock, eng, _at(11, 45), mid=165.0) == []   # 0.36 R

    def test_forced_exit_at_1515(self, db, clock):
        eng, sig, pos = _open(db, clock)
        assert self._tick(db, clock, eng, _at(15, 14), mid=170.0) == []
        trades = self._tick(db, clock, eng, _at(15, 15), mid=170.0)
        assert [t.exit_reason for t in trades] == ["FORCED_EOD"]
        assert trades[0].actual_exit_price == pytest.approx(169.5)

    def test_forced_exit_without_any_mark_uses_last_known(self, db, clock):
        eng, sig, pos = _open(db, clock)
        clock.now = _at(15, 16)
        trades = eng.on_tick()
        assert [t.exit_reason for t in trades] == ["FORCED_EOD"]
        assert trades[0].exit_mark_source == "last_known"

    def test_hero_zero_exits_at_1455(self, db, clock):
        sig = _refined(is_hero_zero=True)
        eng, sig, pos = _open(db, clock, sig=sig)
        assert self._tick(db, clock, eng, _at(14, 54), mid=160.0) == []
        trades = self._tick(db, clock, eng, _at(14, 55), mid=160.0)
        assert [t.exit_reason for t in trades] == ["HERO_ZERO_CUTOFF"]

    def test_trailing_stop_ratchets_then_exits(self, db, clock):
        eng, sig, pos = _open(db, clock)
        # 80 % of the way from planned entry (150.5) to target (210) → lock 30 %
        assert self._tick(db, clock, eng, _at(11, 0), mid=198.0) == []
        assert pos.stop_loss == pytest.approx(150.5 + 0.3 * (210.0 - 150.5), abs=0.01)
        assert pos.original_stop_loss == pytest.approx(110.0)
        trades = self._tick(db, clock, eng, _at(11, 10), mid=165.0)
        assert [t.exit_reason for t in trades] == ["TRAILING_SL_HIT"]
        assert trades[0].outcome == "WIN"

    def test_manual_close_uses_mark_minus_half_spread(self, db, clock):
        eng, sig, pos = _open(db, clock)
        clock.now = _at(12, 0)
        t = eng.close_position(pos.position_id, 170.0, "MANUAL")
        assert t.exit_reason == "MANUAL" and t.actual_exit_price == pytest.approx(169.5)
        assert eng.close_position("nope", 170.0, "MANUAL") is None


class TestParityWithLiveRiskManager:
    """Paper exit decisions equal RiskManager.update_position on the same inputs."""

    SCENARIOS = [
        # (signal kwargs, mark, spot, now)
        ({}, 150.0, 24_500.0, _at(10, 45)),
        ({}, 109.0, 24_500.0, _at(10, 45)),
        ({}, 140.0, 24_390.0, _at(10, 45)),
        ({}, 211.0, 24_500.0, _at(10, 45)),
        ({}, 160.0, 24_710.0, _at(10, 45)),
        ({}, 155.0, 24_520.0, _at(11, 31)),
        ({}, 170.0, 24_560.0, _at(11, 31)),
        ({}, 150.0, 24_500.0, _at(15, 15)),
        ({"is_hero_zero": True}, 150.0, 24_500.0, _at(14, 56)),
        ({"is_hero_zero": True}, 140.0, 24_500.0, _at(12, 0)),
        ({"signal_type": "BUY_PUT"}, 140.0, 24_610.0, _at(10, 45)),
        ({"signal_type": "BUY_PUT"}, 150.0, 24_290.0, _at(10, 45)),
    ]

    @pytest.mark.parametrize("kw,mark,spot,now", SCENARIOS)
    def test_same_decision(self, db, clock, kw, mark, spot, now):
        sig = _refined(**kw)
        rm = RiskManager(db, config=RiskConfig())
        rm.track_open_position(sig)
        live = rm.update_position(sig.signal_id, spot, _now=now, option_ltp=mark).action

        eng, _, pos = _open(db, clock, sig=sig)
        paper = eng._exit_reason(pos, mark, spot, now)

        if live == "HOLD":
            assert paper is None
        elif live == "EXIT_TIME":
            assert paper in ("FORCED_EOD", "HERO_ZERO_CUTOFF")
        else:
            assert paper == RM_ACTION_TO_EXIT_REASON[live]


# ===========================================================================
# 6. P&L
# ===========================================================================


class TestPnL:
    def test_net_pnl_uses_round_trip_cost_and_cash_reconciles(self, db, clock):
        eng, sig, pos = _open(db, clock, ask=151.0)
        _book(db, _at(10, 59), 211.0)
        clock.now = _at(11, 0)
        t = eng.on_tick()[0]
        costs = round_trip_cost(151.0, 210.5, QTY)
        assert t.gross_pnl == pytest.approx((210.5 - 151.0) * QTY, abs=0.01)
        assert t.total_costs == pytest.approx(costs.total, abs=0.01)
        assert t.net_pnl == pytest.approx(t.gross_pnl - costs.total, abs=0.01)
        assert t.entry_cost + t.exit_cost == pytest.approx(t.total_costs, abs=0.01)
        assert eng.cash == pytest.approx(100_000.0 + t.net_pnl, abs=0.01)
        assert eng.current_capital == pytest.approx(100_000.0 + t.net_pnl, abs=0.01)
        rec = eng.daily_ledger[NOW.date()]
        assert rec.realized_pnl == pytest.approx(t.net_pnl) and rec.trades_won == 1

    def test_win_only_if_net_positive(self, db, clock):
        # Bid above the ask paid, but not by enough to cover the costs → LOSS
        eng, sig, pos = _open(db, clock, ask=151.0)
        clock.now = _at(11, 0)
        t = eng.close_position(pos.position_id, 151.6, "MANUAL")    # fill 151.1
        assert t.gross_pnl > 0 and t.net_pnl < 0
        assert t.outcome == "LOSS"
        assert eng.daily_ledger[NOW.date()].trades_lost == 1

    def test_outcomes_are_only_win_or_loss(self, db, clock):
        eng, sig, pos = _open(db, clock, ask=151.0)
        clock.now = _at(11, 0)
        t = eng.close_position(pos.position_id, 151.5, "MANUAL")
        assert t.outcome in ("WIN", "LOSS")


# ===========================================================================
# 7. Sharpe / Sortino from daily returns
# ===========================================================================


class TestDailySharpe:
    MON, TUE, WED, THU, FRI = (date(2026, 11, d) for d in (2, 3, 4, 5, 6))

    def test_daily_returns_include_zero_pnl_days(self):
        r = daily_equity_returns({self.MON: 1_000.0, self.WED: -500.0}, self.MON, self.FRI, 100_000.0)
        assert r == pytest.approx([0.01, 0.0, -500.0 / 101_000.0, 0.0, 0.0])

    def test_weekend_skipped_but_days_with_pnl_kept(self):
        sat = date(2026, 11, 7)
        r = daily_equity_returns({sat: 100.0}, self.FRI, date(2026, 11, 9), 100_000.0)
        assert len(r) == 3   # Fri, Sat (has P&L), Mon — Sunday skipped

    def test_risk_free_rate_in_same_units_as_returns(self):
        rf_d = 0.065 / 252
        rets = [rf_d + 0.001, rf_d - 0.001] * 10       # mean excess exactly 0
        assert annualised_sharpe(rets) == pytest.approx(0.0, abs=1e-9)

    def _trade(self, day: date, pnl: float) -> PaperTrade:
        ts = datetime(day.year, day.month, day.day, 14, 0, tzinfo=_IST)
        return PaperTrade(
            position_id=str(uuid.uuid4()), signal_id=str(uuid.uuid4()), index_id="NIFTY50",
            trade_type="BUY_CALL", signal_entry_price=150.5, market_price_at_signal=24_500.0,
            execution_price=151.0, entry_timestamp=ts - timedelta(hours=1),
            original_stop_loss=110.0, original_target=210.0, strike=STRIKE, expiry=EXPIRY,
            option_premium_at_entry=150.0, lots=2, lot_size=65, quantity=QTY,
            exit_price=160.0, actual_exit_price=159.5, exit_timestamp=ts, exit_reason="MANUAL",
            gross_pnl=pnl + 80, entry_cost=40.0, exit_cost=40.0, total_costs=80.0, net_pnl=pnl,
            net_pnl_pct=pnl / 1_000.0, max_favorable_excursion=10.0, max_adverse_excursion=5.0,
            capture_ratio=0.5, duration_seconds=3_600, duration_bars=60, confidence="HIGH",
            regime="TRENDING", risk_amount=5_000.0, entry_cost_at_open=40.0,
            outcome="WIN" if pnl > 0 else "LOSS",
        )

    def test_engine_sharpe_uses_all_days_in_window(self, db, clock):
        eng = _engine(db)
        for d in (self.MON, self.TUE, self.WED, self.THU, self.FRI):
            eng._ensure_daily_record(d)
        eng.trade_history = [self._trade(self.MON, 1_000.0), self._trade(self.WED, -500.0),
                             self._trade(self.THU, 700.0)]
        clock.now = datetime(2026, 11, 6, 15, 45, tzinfo=_IST)
        stats = eng.get_cumulative_stats(days=30)

        expected = annualised_sharpe(daily_equity_returns(
            {self.MON: 1_000.0, self.WED: -500.0, self.THU: 700.0}, self.MON, self.FRI, 100_000.0))
        trade_days_only = annualised_sharpe([0.01, -500 / 101_000, 700 / 100_500])
        assert stats.trading_days == 5
        assert stats.sharpe_ratio == pytest.approx(expected, abs=1e-4)
        assert stats.sharpe_ratio != pytest.approx(trade_days_only, abs=1e-3)
        assert (stats.profitable_days, stats.unprofitable_days) == (2, 1)


# ===========================================================================
# 8. Persistence / restart
# ===========================================================================


class TestPersistence:
    def test_restart_restores_positions_pending_cash_and_trades(self, db, clock):
        eng, sig, pos = _open(db, clock)
        _book(db, _at(10, 59), 198.0)
        clock.now = _at(11, 0)
        eng.on_tick()                                            # ratchets the stop
        trailed = pos.stop_loss
        pending = eng.submit_order(_refined(index_id="BANKNIFTY", signal_type="BUY_PUT"),
                                   decision_time=_at(11, 0))
        assert pending.status == "PENDING"
        eng.trade_history.append(TestDailySharpe()._trade(date(2026, 11, 4), 250.0))
        eng.save_state()
        cash = eng.cash

        clock.now = _at(11, 2)
        eng2 = _engine(db)
        assert eng2.cash == pytest.approx(cash)
        restored = eng2.open_positions[pos.position_id]
        assert (restored.option_type, restored.strike, restored.expiry) == ("CE", STRIKE, EXPIRY)
        assert restored.stop_loss == pytest.approx(trailed)
        assert restored.original_stop_loss == pytest.approx(110.0)
        assert restored.decision_time == NOW
        assert restored.execution_price == pytest.approx(pos.execution_price)
        assert pending.order_id in eng2.pending_orders
        assert len(eng2.trade_history) == 1
        assert eng2.current_capital == pytest.approx(eng.current_capital)

        # and it keeps working after the restart
        _book(db, _at(11, 4), 211.0)
        clock.now = _at(11, 5)
        assert [t.exit_reason for t in eng2.on_tick()] == ["TARGET_HIT"]

    def test_next_day_restart_settles_at_entry_day_force_exit_quote(self, db, clock):
        eng, sig, pos = _open(db, clock, ask=151.0)
        eng.save_state()
        _quote(db, _at(15, 10), ltp=160.5, bid=160.0, ask=161.0)
        _quote(db, _at(15, 20), ltp=300.0, bid=299.0, ask=301.0)   # after 15:15 — ignored

        clock.now = _at(9, 20, day=6)
        eng2 = _engine(db)
        assert not eng2.open_positions
        t = eng2.trade_history[-1]
        assert t.exit_reason == "FORCED_EOD" and t.actual_exit_price == pytest.approx(160.0)
        assert t.exit_timestamp == _at(15, 10)
        assert eng2.cash == pytest.approx(100_000.0 + t.net_pnl, abs=0.01)
        assert eng2.daily_ledger[NOW.date()].realized_pnl == pytest.approx(t.net_pnl)
        # persisted: a third start sees the settled state
        eng3 = _engine(db)
        assert not eng3.open_positions and len(eng3.trade_history) == 1

    def test_legacy_index_point_state_is_discarded_but_ledger_kept(self, db, clock):
        PaperTradingEngine(db, PaperTradingConfig())     # creates the state table
        db.execute(
            """INSERT OR REPLACE INTO paper_trading_state
               (id, current_capital, open_positions_json, daily_ledger_json, updated_at)
               VALUES (1, ?, ?, ?, ?)""",
            (98_000.0, json.dumps([{"position_id": "x", "execution_price": 22_002.0}]),
             json.dumps({"2026-11-04": {"starting_capital": 100_000.0, "realized_pnl": -2_000.0,
                                         "trades_taken": 1}}),
             NOW.isoformat()),
        )
        eng = _engine(db)
        assert not eng.open_positions
        assert eng.cash == pytest.approx(98_000.0)
        assert eng.daily_ledger[date(2026, 11, 4)].realized_pnl == pytest.approx(-2_000.0)

    def test_daily_ledger_round_trip(self, db, clock):
        eng = _engine(db)
        eng._ensure_daily_record(NOW.date())
        eng.daily_ledger[NOW.date()].signals_generated = 7
        eng.save_state()
        assert _engine(db).daily_ledger[NOW.date()].signals_generated == 7

    def test_paper_never_writes_trading_signals(self, db, clock):
        eng, sig, pos = _open(db, clock)
        clock.now = _at(12, 0)
        eng.close_position(pos.position_id, 170.0, "MANUAL")
        assert db.fetch_all("SELECT * FROM trading_signals WHERE outcome IS NOT NULL", ()) == []

    def test_reset(self, db, clock):
        eng, sig, pos = _open(db, clock)
        with pytest.raises(ValueError, match="confirm=True"):
            eng.reset()
        eng.reset(confirm=True)
        assert not eng.open_positions and eng.cash == pytest.approx(100_000.0)
        assert _engine(db).cash == pytest.approx(100_000.0)


# ===========================================================================
# 9. Reporting
# ===========================================================================


class TestReporting:
    def test_daily_summary_after_round_trip(self, db, clock):
        eng, sig, pos = _open(db, clock)
        _book(db, _at(10, 59), 211.0)
        clock.now = _at(11, 0)
        t = eng.on_tick()[0]
        s = eng.get_daily_summary(NOW.date())
        assert s.trades_taken == 1 and s.trades_won == 1
        assert s.signals_generated == 1 and s.signals_executed == 1
        assert s.total_pnl == pytest.approx(t.net_pnl)
        assert s.ending_capital == pytest.approx(100_000.0 + t.net_pnl)

    def test_cumulative_stats_profit_factor(self, db, clock):
        eng = _engine(db)
        mk = TestDailySharpe()._trade
        eng.trade_history = [mk(NOW.date(), 1_000.0)] * 3 + [mk(NOW.date(), -500.0)]
        stats = eng.get_cumulative_stats(days=30)
        assert stats.total_trades == 4
        assert stats.win_rate == pytest.approx(0.75)
        assert stats.profit_factor == pytest.approx(6.0)
        assert stats.edge_status == "INTACT"

    def test_mtm_equity_tracks_open_position(self, db, clock):
        eng, sig, pos = _open(db, clock)
        _book(db, _at(10, 59), 171.0)
        clock.now = _at(11, 0)
        eng.on_tick()
        assert eng.equity() == pytest.approx(eng.current_capital + pos.unrealized_pnl, abs=0.01)
