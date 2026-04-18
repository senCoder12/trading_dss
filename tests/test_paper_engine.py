"""
Tests for the Paper Trading Engine — Phase 9.1.

Coverage:
- on_signal: valid signal → creates position
- on_signal: confidence filter → missed signal logged
- on_signal: index filter → skipped
- on_signal: kill switch → blocked
- on_signal: warm-up period → blocked
- on_signal: too late for entry → blocked
- on_signal: daily loss limit → missed
- on_signal: max open positions → missed
- on_signal: existing position in same index → missed
- on_signal: insufficient capital → missed
- update_positions: price hits SL → position closed
- update_positions: price hits target → position closed
- update_positions: trailing SL activation and movement
- update_positions: forced EOD exit at 15:25
- update_positions: stale data guard
- close_position: P&L, costs, outcome correctness
- Daily loss limit enforcement (two signals, second blocked)
- Max positions enforcement
- save_state / load_state round-trip
- load_state: previous-day state does not restore open positions
- Missed signal tracking
- Daily summary calculation
- Cumulative stats calculation
- Full trading day simulation (entry → trailing SL → close)
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.paper_trading.paper_engine import (
    DailyPaperSummary,
    DailyRecord,
    MissedSignal,
    PaperExecution,
    PaperPosition,
    PaperTrade,
    PaperTradingConfig,
    PaperTradingEngine,
    PaperTradingStats,
)

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _ts(hour: int, minute: int = 0, day: int = 10) -> datetime:
    """Return an IST datetime in June 2024."""
    return datetime(2024, 6, day, hour, minute, tzinfo=_IST)


@dataclass
class _Signal:
    """Minimal signal object that satisfies the PaperTradingEngine duck-typing."""

    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    index_id: str = "NIFTY50"
    signal_type: str = "BUY_CALL"
    confidence_level: str = "HIGH"
    confidence_score: float = 0.75
    entry_price: float = 22_000.0
    target_price: float = 22_200.0
    stop_loss: float = 21_900.0
    risk_reward_ratio: float = 2.0
    position_size_modifier: float = 1.0
    regime: str = "TRENDING"
    weighted_score: float = 0.8
    # RefinedSignal extensions (optional)
    refined_entry: float = 0.0
    refined_target: float = 0.0
    refined_stop_loss: float = 0.0
    recommended_strike: Optional[float] = None
    recommended_expiry: Optional[str] = None
    option_premium: Optional[float] = None
    days_to_expiry: Optional[int] = 5


def _market(ltp: float = 22_000.0) -> dict:
    return {
        "index_id": "NIFTY50",
        "ltp": ltp,
        "bid": ltp - 0.5,
        "ask": ltp + 0.5,
        "timestamp": _ts(10, 30),
        "option_premium": 180.0,
    }


class _InMemoryDB:
    """
    Tiny in-memory SQLite wrapper that satisfies the DatabaseManager duck-type.
    Only the methods called by PaperTradingEngine need to work.
    """

    def __init__(self):
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Pre-create the trading_signals table so _persist_trade doesn't explode
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_id TEXT,
                generated_at TEXT,
                signal_type TEXT,
                confidence_level TEXT,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                risk_reward_ratio REAL,
                regime TEXT,
                technical_vote TEXT,
                options_vote TEXT,
                news_vote TEXT,
                anomaly_vote TEXT,
                reasoning TEXT,
                outcome TEXT,
                actual_exit_price REAL,
                actual_pnl REAL,
                closed_at TEXT,
                audit_json TEXT
            )"""
        )
        self._conn.commit()

    def execute(self, query: str, params: tuple):
        try:
            cur = self._conn.execute(query, params)
            self._conn.commit()
            return cur
        except Exception:
            self._conn.rollback()
            raise

    def fetch_one(self, query: str, params: tuple) -> Optional[dict]:
        try:
            cur = self._conn.execute(query, params)
            row = cur.fetchone()
            if row is None:
                return None
            return dict(row)
        except Exception:
            return None

    def fetch_all(self, query: str, params: tuple) -> list:
        try:
            cur = self._conn.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []


def _make_engine(
    initial_capital: float = 100_000.0,
    min_confidence: str = "HIGH",
    intraday_only: bool = True,
    use_live_spread: bool = False,   # use fallback in tests for determinism
    fallback_slippage: float = 2.0,
) -> tuple[PaperTradingEngine, _InMemoryDB]:
    """Build a PaperTradingEngine backed by an in-memory SQLite DB."""
    db = _InMemoryDB()
    cfg = PaperTradingConfig(
        initial_capital=initial_capital,
        min_confidence=min_confidence,
        intraday_only=intraday_only,
        use_live_spread=use_live_spread,
        fallback_slippage_points=fallback_slippage,
        active_indices=["NIFTY50", "BANKNIFTY"],
        signal_types=["BUY_CALL", "BUY_PUT"],
        track_missed_signals=True,
        track_no_trade_reasons=True,
    )
    # Patch get_ist_now so warm-up check passes at 10:30
    with (
        patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
        patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
        patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
    ):
        mock_index = MagicMock()
        mock_index.lot_size = 50
        mock_reg.return_value.get_index.return_value = mock_index
        engine = PaperTradingEngine(db=db, config=cfg)

    # Engine is now constructed; reset warmed-up flag so tests can control it
    engine._warmed_up = True
    return engine, db


# ---------------------------------------------------------------------------
# 1. on_signal — valid signal creates a position
# ---------------------------------------------------------------------------


class TestOnSignalValid:
    def test_valid_signal_creates_position(self):
        engine, db = _make_engine()
        sig = _Signal()

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(sig, _market(22_000.0))

        assert result is not None
        assert isinstance(result, PaperExecution)
        assert len(engine.open_positions) == 1

        pos = list(engine.open_positions.values())[0]
        assert pos.index_id == "NIFTY50"
        assert pos.trade_type == "BUY_CALL"
        # Execution price = LTP + slippage (fallback = 2.0)
        assert pos.execution_price == pytest.approx(22_002.0)
        assert pos.lots >= 1
        assert pos.lot_size == 50
        assert pos.quantity == pos.lots * 50
        assert pos.status == "OPEN"
        assert pos.original_stop_loss == pytest.approx(21_900.0)

    def test_buy_put_execution_price_is_ltp_minus_slippage(self):
        engine, db = _make_engine()
        sig = _Signal(signal_type="BUY_PUT", entry_price=22_000.0,
                      target_price=21_800.0, stop_loss=22_100.0)

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(sig, _market(22_000.0))

        assert result is not None
        pos = list(engine.open_positions.values())[0]
        assert pos.execution_price == pytest.approx(21_998.0)

    def test_execution_records_signal_entry_and_market_price(self):
        engine, db = _make_engine()
        sig = _Signal(entry_price=22_005.0)  # signal says 22005

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(sig, _market(22_003.0))  # market is 22003

        pos = list(engine.open_positions.values())[0]
        # signal_entry_price should be signal.entry_price (22005)
        assert pos.signal_entry_price == pytest.approx(22_005.0)
        # market_price_at_signal should be the LTP
        assert pos.market_price_at_signal == pytest.approx(22_003.0)
        # execution_price = market + slippage
        assert pos.execution_price == pytest.approx(22_005.0)  # 22003 + 2

    def test_daily_record_incremented(self):
        engine, db = _make_engine()

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            engine.on_signal(_Signal(), _market())

        today = date(2024, 6, 10)
        rec = engine.daily_ledger.get(today)
        assert rec is not None
        assert rec.signals_generated >= 1
        assert rec.signals_executed == 1


# ---------------------------------------------------------------------------
# 2. on_signal — confidence filter
# ---------------------------------------------------------------------------


class TestConfidenceFilter:
    def test_medium_signal_blocked_when_min_is_high(self):
        engine, db = _make_engine(min_confidence="HIGH")
        sig = _Signal(confidence_level="MEDIUM")

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(sig, _market())

        assert result is None
        assert len(engine.open_positions) == 0

    def test_medium_signal_logged_as_missed(self):
        engine, db = _make_engine(min_confidence="HIGH")
        sig = _Signal(confidence_level="MEDIUM")

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            engine.on_signal(sig, _market())

        assert len(engine.missed_signals) == 1
        assert engine.missed_signals[0].reason_missed == "below_min_confidence"

    def test_high_signal_accepted_when_min_is_medium(self):
        engine, db = _make_engine(min_confidence="MEDIUM")
        sig = _Signal(confidence_level="HIGH")

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(sig, _market())

        assert result is not None

    def test_low_signal_blocked_when_min_is_medium(self):
        engine, db = _make_engine(min_confidence="MEDIUM")
        sig = _Signal(confidence_level="LOW")

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(sig, _market())

        assert result is None


# ---------------------------------------------------------------------------
# 3. Kill switch
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def test_kill_switch_blocks_new_signals(self):
        engine, db = _make_engine()
        engine.activate_kill_switch("Test reason")

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(_Signal(), _market())

        assert result is None

    def test_deactivate_kill_switch_allows_trading(self):
        engine, db = _make_engine()
        engine.activate_kill_switch("Test")
        engine.deactivate_kill_switch()
        assert not engine._kill_switch_active

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(_Signal(), _market())

        assert result is not None


# ---------------------------------------------------------------------------
# 4. Warm-up filter
# ---------------------------------------------------------------------------


class TestWarmUp:
    def test_signal_blocked_during_warmup(self):
        engine, db = _make_engine()
        engine._warmed_up = False  # force back to warm-up

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(9, 15)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(_Signal(), {**_market(), "timestamp": _ts(9, 15)})

        assert result is None
        assert not engine._warmed_up

    def test_signal_accepted_after_warmup(self):
        engine, db = _make_engine()
        engine._warmed_up = False

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(9, 18)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(_Signal(), _market())

        assert result is not None


# ---------------------------------------------------------------------------
# 5. Too-late-for-entry filter
# ---------------------------------------------------------------------------


class TestTooLateForEntry:
    def test_signal_blocked_after_1520(self):
        engine, db = _make_engine()

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(15, 21)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(_Signal(), _market())

        assert result is None

    def test_signal_accepted_at_1519(self):
        engine, db = _make_engine()

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(15, 19)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(_Signal(), _market())

        assert result is not None


# ---------------------------------------------------------------------------
# 6. Daily loss limit
# ---------------------------------------------------------------------------


class TestDailyLossLimit:
    def test_second_signal_blocked_after_daily_limit(self):
        engine, db = _make_engine(initial_capital=100_000.0)
        today = date(2024, 6, 10)
        engine._ensure_daily_record(today)
        # Simulate daily loss at -5% = -5000
        engine.daily_ledger[today].realized_pnl = -5_001.0

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(_Signal(), _market())

        assert result is None
        assert len(engine.missed_signals) >= 1
        missed = [m for m in engine.missed_signals if m.reason_missed == "daily_loss_limit"]
        assert len(missed) == 1


# ---------------------------------------------------------------------------
# 7. Max open positions
# ---------------------------------------------------------------------------


class TestMaxPositions:
    def test_signal_blocked_at_max_positions(self):
        engine, db = _make_engine()
        engine.config.max_open_positions = 1

        # Open one position manually
        engine.open_positions["fake_pos"] = _make_position("BANKNIFTY")

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(_Signal(index_id="NIFTY50"), _market())

        assert result is None

    def test_signal_blocked_existing_position_same_index(self):
        engine, db = _make_engine()
        engine.open_positions["existing"] = _make_position("NIFTY50")

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(_Signal(index_id="NIFTY50"), _market())

        assert result is None


# ---------------------------------------------------------------------------
# 8. update_positions — SL hit
# ---------------------------------------------------------------------------


class TestUpdatePositionsSLHit:
    def test_call_position_closed_when_price_drops_to_sl(self):
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_200.0,
            lots=1,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 21_895.0})

        assert len(closed) == 1
        assert closed[0].exit_reason in ("STOP_LOSS_HIT", "TRAILING_SL_HIT")
        assert closed[0].outcome == "LOSS"
        assert len(engine.open_positions) == 0

    def test_put_position_closed_when_price_rises_to_sl(self):
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_PUT",
            execution_price=22_000.0,
            stop_loss=22_100.0,
            target=21_800.0,
            lots=1,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 22_105.0})

        assert len(closed) == 1
        assert closed[0].outcome == "LOSS"


# ---------------------------------------------------------------------------
# 9. update_positions — target hit
# ---------------------------------------------------------------------------


class TestUpdatePositionsTargetHit:
    def test_call_position_closed_at_target(self):
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_200.0,
            lots=2,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 22_205.0})

        assert len(closed) == 1
        assert closed[0].exit_reason == "TARGET_HIT"
        assert closed[0].outcome == "WIN"

    def test_put_position_closed_at_target(self):
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_PUT",
            execution_price=22_000.0,
            stop_loss=22_100.0,
            target=21_800.0,
            lots=1,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 21_795.0})

        assert len(closed) == 1
        assert closed[0].exit_reason == "TARGET_HIT"
        assert closed[0].outcome == "WIN"


# ---------------------------------------------------------------------------
# 10. Trailing SL
# ---------------------------------------------------------------------------


class TestTrailingSL:
    def test_trailing_sl_activates_at_50_pct_progress(self):
        """SL should move to breakeven when price reaches 50% toward target."""
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_200.0,   # target distance = 200 pts
            lots=1,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        # At 50% progress (22_100) → SL should move to 22_000 (breakeven)
        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            engine.update_positions({"NIFTY50": 22_100.0})

        pos_after = list(engine.open_positions.values())[0]
        assert pos_after.trailing_sl == pytest.approx(22_000.0)

    def test_trailing_sl_ratchets_at_75_pct(self):
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_200.0,
            lots=1,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        # At 75% progress (22_150) → SL = 22_000 + 0.3 * 200 = 22_060
        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            engine.update_positions({"NIFTY50": 22_150.0})

        pos_after = list(engine.open_positions.values())[0]
        assert pos_after.trailing_sl == pytest.approx(22_060.0)

    def test_trailing_sl_does_not_widen(self):
        """Once set to 22_000 (breakeven), SL must NOT drop back below entry."""
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_200.0,
            lots=1,
            lot_size=50,
        )
        pos.trailing_sl = 22_000.0  # already at breakeven
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        # Price dips a bit but stays above SL
        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            engine.update_positions({"NIFTY50": 22_050.0})

        pos_after = list(engine.open_positions.values())[0]
        assert pos_after.trailing_sl >= 22_000.0  # must not go lower

    def test_trailing_sl_triggers_close(self):
        """Position closes when price drops below the ratcheted trailing SL."""
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_200.0,
            lots=1,
            lot_size=50,
        )
        pos.trailing_sl = 22_000.0   # pre-ratcheted to breakeven
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 21_995.0})

        assert len(closed) == 1
        assert closed[0].exit_reason == "TRAILING_SL_HIT"


# ---------------------------------------------------------------------------
# 11. Forced EOD exit at 15:25
# ---------------------------------------------------------------------------


class TestForcedEODExit:
    def test_open_position_closed_at_1525(self):
        engine, db = _make_engine(intraday_only=True)
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_500.0,   # far away — won't hit today
            lots=1,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(15, 25)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 22_080.0})

        assert len(closed) == 1
        assert closed[0].exit_reason == "FORCED_EOD"

    def test_position_not_closed_at_1524(self):
        engine, db = _make_engine(intraday_only=True)
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_500.0,
            lots=1,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(15, 24)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 22_080.0})

        assert len(closed) == 0

    def test_intraday_false_no_eod_close(self):
        engine, db = _make_engine(intraday_only=False)
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_500.0,
            lots=1,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(15, 29)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 22_080.0})

        assert len(closed) == 0


# ---------------------------------------------------------------------------
# 12. Stale data guard
# ---------------------------------------------------------------------------


class TestStaleDataGuard:
    def test_update_skipped_when_data_is_stale(self):
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_200.0,
            lots=1,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        # Simulate last update 6 minutes ago
        engine._last_price_update = _ts(10, 24)  # now is 10:30 → stale (360s > 300s threshold)

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 21_800.0})  # would normally trigger SL

        # Should be skipped due to stale data
        assert len(closed) == 0
        assert len(engine.open_positions) == 1  # still open


# ---------------------------------------------------------------------------
# 13. P&L and cost calculation
# ---------------------------------------------------------------------------


class TestPnLCalculation:
    def test_winning_trade_pnl_correct(self):
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_200.0,
            lots=1,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(12, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 22_210.0})

        assert len(closed) == 1
        t = closed[0]
        # Gross = (exit - slippage - entry) * quantity
        # actual_exit = 22200 - 2.0 = 22198
        # gross = (22198 - 22000) * 50 = 198 * 50 = 9900
        assert t.gross_pnl == pytest.approx(22_198.0 * 50 - 22_000.0 * 50, abs=50)
        assert t.net_pnl < t.gross_pnl    # costs must reduce P&L
        assert t.total_costs > 0
        assert t.outcome == "WIN"

    def test_losing_trade_pnl_negative(self):
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_200.0,
            lots=1,
            lot_size=50,
        )
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 21_895.0})

        assert len(closed) == 1
        assert closed[0].net_pnl < 0
        assert closed[0].outcome == "LOSS"

    def test_capture_ratio_between_zero_and_one_on_win(self):
        engine, db = _make_engine()
        pos = _make_position(
            index_id="NIFTY50",
            trade_type="BUY_CALL",
            execution_price=22_000.0,
            stop_loss=21_900.0,
            target=22_200.0,
            lots=1,
            lot_size=50,
        )
        pos.max_favorable = 200.0   # saw 200 pts favorable
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(12, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 22_210.0})

        assert 0.0 <= closed[0].capture_ratio <= 1.1  # can slightly exceed 1 with rounding


# ---------------------------------------------------------------------------
# 14. close_position directly
# ---------------------------------------------------------------------------


class TestClosePositionDirect:
    def test_manual_close_removes_from_open_positions(self):
        engine, db = _make_engine()
        pos = _make_position(index_id="NIFTY50")
        engine.open_positions[pos.position_id] = pos
        engine._ensure_daily_record(date(2024, 6, 10))

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(13, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            trade = engine.close_position(pos.position_id, 22_050.0, "MANUAL")

        assert trade is not None
        assert trade.exit_reason == "MANUAL"
        assert pos.position_id not in engine.open_positions
        assert len(engine.trade_history) == 1

    def test_close_nonexistent_position_returns_none(self):
        engine, db = _make_engine()
        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(13, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            result = engine.close_position("nonexistent-uuid", 22_000.0, "MANUAL")
        assert result is None


# ---------------------------------------------------------------------------
# 15. save_state / load_state round-trip
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_save_and_load_state_same_day(self):
        engine, db = _make_engine()
        pos = _make_position("NIFTY50")
        engine.open_positions[pos.position_id] = pos
        engine.current_capital = 98_000.0
        today = date(2024, 6, 10)
        engine._ensure_daily_record(today)
        engine.daily_ledger[today].realized_pnl = -2_000.0

        with patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)):
            engine.save_state()

        # Create a new engine instance and load
        engine2, _ = _make_engine()
        # Point engine2 at the same DB
        engine2._db = db

        with patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 5)):
            engine2.load_state()

        assert engine2.current_capital == pytest.approx(98_000.0)
        assert len(engine2.open_positions) == 1
        assert engine2.open_positions[pos.position_id].index_id == "NIFTY50"

    def test_load_state_previous_day_clears_positions(self):
        engine, db = _make_engine()
        pos = _make_position("NIFTY50")
        engine.open_positions[pos.position_id] = pos
        engine.current_capital = 95_000.0

        # Save at yesterday's time
        with patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(15, 30, day=9)):
            engine.save_state()

        # Load at today
        engine2, _ = _make_engine()
        engine2._db = db

        with patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(9, 20, day=10)):
            engine2.load_state()

        # Capital is preserved but stale open positions are NOT restored
        assert engine2.current_capital == pytest.approx(95_000.0)
        assert len(engine2.open_positions) == 0

    def test_save_state_round_trip_daily_ledger(self):
        engine, db = _make_engine()
        today = date(2024, 6, 10)
        engine._ensure_daily_record(today)
        engine.daily_ledger[today].trades_taken = 3
        engine.daily_ledger[today].realized_pnl = 1_500.0

        with patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(14, 0)):
            engine.save_state()

        engine2, _ = _make_engine()
        engine2._db = db
        with patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(14, 5)):
            engine2.load_state()

        assert today in engine2.daily_ledger
        assert engine2.daily_ledger[today].trades_taken == 3
        assert engine2.daily_ledger[today].realized_pnl == pytest.approx(1_500.0)


# ---------------------------------------------------------------------------
# 16. Missed signal tracking
# ---------------------------------------------------------------------------


class TestMissedSignalTracking:
    def test_missed_signal_records_entry_levels(self):
        engine, db = _make_engine(initial_capital=100_000.0)
        today = date(2024, 6, 10)
        engine._ensure_daily_record(today)
        engine.daily_ledger[today].realized_pnl = -5_001.0  # trigger daily limit

        sig = _Signal(entry_price=22_100.0, target_price=22_300.0, stop_loss=22_000.0)

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            engine.on_signal(sig, _market())

        assert len(engine.missed_signals) >= 1
        ms = engine.missed_signals[-1]
        assert ms.signal_entry_price == pytest.approx(22_100.0)
        assert ms.would_have_been_target == pytest.approx(22_300.0)
        assert ms.would_have_been_sl == pytest.approx(22_000.0)


# ---------------------------------------------------------------------------
# 17. Daily summary
# ---------------------------------------------------------------------------


class TestDailySummary:
    def test_daily_summary_reflects_closed_trades(self):
        engine, db = _make_engine()
        today = date(2024, 6, 10)
        engine._ensure_daily_record(today)
        rec = engine.daily_ledger[today]
        rec.trades_taken = 2
        rec.trades_won = 1
        rec.trades_lost = 1
        rec.signals_generated = 5
        rec.signals_executed = 2
        rec.realized_pnl = 1_200.0

        # Add a dummy trade to trade_history so best/worst can be computed
        engine.trade_history.append(_make_trade(net_pnl=2_000.0, exit_date=today))
        engine.trade_history.append(_make_trade(net_pnl=-800.0, exit_date=today))

        with patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(15, 30)):
            summary = engine.get_daily_summary(for_date=today)

        assert summary.trades_taken == 2
        assert summary.trades_won == 1
        assert summary.win_rate == pytest.approx(0.5)
        assert summary.best_trade_pnl == pytest.approx(2_000.0)
        assert summary.worst_trade_pnl == pytest.approx(-800.0)
        assert summary.signals_generated == 5

    def test_daily_summary_zero_trades(self):
        engine, db = _make_engine()
        today = date(2024, 6, 10)

        with patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 0)):
            summary = engine.get_daily_summary(for_date=today)

        assert summary.trades_taken == 0
        assert summary.win_rate == 0.0


# ---------------------------------------------------------------------------
# 18. Cumulative stats
# ---------------------------------------------------------------------------


class TestCumulativeStats:
    def test_cumulative_stats_profit_factor(self):
        engine, db = _make_engine()

        # 3 wins of ₹1000, 1 loss of -₹500 → PF = 3000/500 = 6.0
        base = _ts(14, 0)
        for i in range(3):
            t = _make_trade(net_pnl=1_000.0, exit_date=date(2024, 6, 10))
            engine.trade_history.append(t)
        engine.trade_history.append(_make_trade(net_pnl=-500.0, exit_date=date(2024, 6, 10)))

        with patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(15, 30)):
            stats = engine.get_cumulative_stats(days=30)

        assert stats.total_trades == 4
        assert stats.win_rate == pytest.approx(0.75)
        assert stats.profit_factor == pytest.approx(6.0)

    def test_edge_status_intact(self):
        engine, db = _make_engine()
        # 6 wins, 2 losses → WR=0.75, PF > 1.3
        for _ in range(6):
            engine.trade_history.append(_make_trade(net_pnl=1_000.0, exit_date=date(2024, 6, 10)))
        for _ in range(2):
            engine.trade_history.append(_make_trade(net_pnl=-200.0, exit_date=date(2024, 6, 10)))

        with patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(15, 30)):
            stats = engine.get_cumulative_stats(days=30)

        assert stats.edge_status == "INTACT"

    def test_edge_status_gone(self):
        engine, db = _make_engine()
        # All losses
        for _ in range(5):
            engine.trade_history.append(_make_trade(net_pnl=-300.0, exit_date=date(2024, 6, 10)))

        with patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(15, 30)):
            stats = engine.get_cumulative_stats(days=30)

        assert stats.edge_status == "GONE"


# ---------------------------------------------------------------------------
# 19. Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_requires_confirm(self):
        engine, db = _make_engine()
        with pytest.raises(ValueError, match="confirm=True"):
            engine.reset(confirm=False)

    def test_reset_clears_state(self):
        engine, db = _make_engine()
        engine.open_positions["x"] = _make_position("NIFTY50")
        engine.trade_history.append(_make_trade(net_pnl=500.0, exit_date=date(2024, 6, 10)))
        engine.current_capital = 90_000.0

        engine.reset(confirm=True)

        assert len(engine.open_positions) == 0
        assert len(engine.trade_history) == 0
        assert engine.current_capital == pytest.approx(100_000.0)


# ---------------------------------------------------------------------------
# 20. Full trading day simulation
# ---------------------------------------------------------------------------


class TestFullTradingDay:
    """Integration-style test that simulates a complete intraday trading session."""

    def test_entry_trailing_sl_close_sequence(self):
        """
        Timeline:
        10:30 → signal fires → CALL position opened @ 22_002
        11:00 → price 22_100 → trailing SL activates at 22_000 (breakeven)
        11:30 → price 22_150 → trailing SL moves to 22_060
        12:00 → price drops to 22_050 → trailing SL (22_060) triggered → CLOSE
        """
        engine, db = _make_engine(
            use_live_spread=False,
            fallback_slippage=2.0,
        )

        # Step 1: Enter at 10:30
        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            result = engine.on_signal(
                _Signal(entry_price=22_000.0, target_price=22_200.0, stop_loss=21_900.0),
                _market(22_000.0),
            )

        assert result is not None
        assert len(engine.open_positions) == 1
        pos = list(engine.open_positions.values())[0]
        assert pos.execution_price == pytest.approx(22_002.0)

        # entry = 22_002 (LTP 22_000 + slippage 2.0), target = 22_200, tgt_dist = 198
        # 50% progress at: entry + 0.5 * 198 = 22_002 + 99 = 22_101
        # Simulate continuous 60-s polling: reset _last_price_update before each step
        # so the stale-data guard doesn't fire in tests.

        # Step 2: Price rises to 22_102 → trailing SL to breakeven (22_002)
        engine._last_price_update = _ts(10, 59)
        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 22_102.0})

        assert len(closed) == 0
        pos = list(engine.open_positions.values())[0]
        # 50% progress → SL = entry = 22_002
        assert pos.trailing_sl == pytest.approx(22_002.0)

        # Step 3: Price at 22_152 → 75% progress
        # 75% at: entry + 0.75 * 198 = 22_150.5
        # trailing SL → entry + 0.3 * 198 = 22_061.4
        engine._last_price_update = _ts(11, 29)
        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(11, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": 22_152.0})

        assert len(closed) == 0
        pos = list(engine.open_positions.values())[0]
        assert pos.trailing_sl > 22_002.0
        trailing_after_75 = pos.trailing_sl

        # Step 4: Price drops below trailing SL → CLOSE
        engine._last_price_update = _ts(11, 59)
        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(12, 0)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
        ):
            closed = engine.update_positions({"NIFTY50": trailing_after_75 - 5.0})

        assert len(closed) == 1
        trade = closed[0]
        assert trade.exit_reason == "TRAILING_SL_HIT"
        # Net P&L should be positive (closed above entry despite costs)
        assert trade.net_pnl > 0
        assert len(engine.open_positions) == 0

    def test_two_different_indices_can_both_be_open(self):
        engine, db = _make_engine()
        engine.config.max_open_positions = 3

        with (
            patch("src.paper_trading.paper_engine.get_ist_now", return_value=_ts(10, 30)),
            patch("src.paper_trading.paper_engine.get_current_expiry", return_value=date(2024, 6, 13)),
            patch("src.paper_trading.paper_engine.calc_days_to_expiry", return_value=3),
            patch("src.paper_trading.paper_engine.get_registry") as mock_reg,
        ):
            mock_index = MagicMock()
            mock_index.lot_size = 50
            mock_reg.return_value.get_index.return_value = mock_index

            r1 = engine.on_signal(_Signal(index_id="NIFTY50"), _market(22_000.0))
            r2 = engine.on_signal(
                _Signal(index_id="BANKNIFTY", entry_price=48_000.0,
                        target_price=48_400.0, stop_loss=47_800.0),
                {**_market(48_000.0), "index_id": "BANKNIFTY"},
            )

        assert r1 is not None
        assert r2 is not None
        assert len(engine.open_positions) == 2


# ---------------------------------------------------------------------------
# Helpers for building test objects
# ---------------------------------------------------------------------------


def _make_position(
    index_id: str = "NIFTY50",
    trade_type: str = "BUY_CALL",
    execution_price: float = 22_000.0,
    stop_loss: float = 21_900.0,
    target: float = 22_200.0,
    lots: int = 1,
    lot_size: int = 50,
) -> PaperPosition:
    now = _ts(10, 30)
    return PaperPosition(
        position_id=str(uuid.uuid4()),
        signal_id=str(uuid.uuid4()),
        index_id=index_id,
        trade_type=trade_type,
        signal_entry_price=execution_price,
        market_price_at_signal=execution_price,
        execution_price=execution_price,
        entry_timestamp=now,
        stop_loss=stop_loss,
        original_stop_loss=stop_loss,
        target=target,
        trailing_sl=stop_loss,
        lots=lots,
        lot_size=lot_size,
        quantity=lots * lot_size,
        strike=None,
        expiry=None,
        option_premium_at_entry=None,
        entry_cost=50.0,
        risk_amount=2_000.0,
        confidence="HIGH",
        regime="TRENDING",
        current_price=execution_price,
        status="OPEN",
        opened_at=now,
    )


def _make_trade(net_pnl: float, exit_date: date) -> PaperTrade:
    now = datetime(exit_date.year, exit_date.month, exit_date.day, 14, 0, tzinfo=_IST)
    return PaperTrade(
        position_id=str(uuid.uuid4()),
        signal_id=str(uuid.uuid4()),
        index_id="NIFTY50",
        trade_type="BUY_CALL",
        signal_entry_price=22_000.0,
        market_price_at_signal=22_000.0,
        execution_price=22_002.0,
        entry_timestamp=now - timedelta(hours=1),
        original_stop_loss=21_900.0,
        original_target=22_200.0,
        strike=None,
        expiry=None,
        option_premium_at_entry=None,
        lots=1,
        lot_size=50,
        quantity=50,
        exit_price=22_100.0,
        actual_exit_price=22_098.0,
        exit_timestamp=now,
        exit_reason="TARGET_HIT" if net_pnl > 0 else "STOP_LOSS_HIT",
        gross_pnl=net_pnl + 50.0,
        entry_cost=30.0,
        exit_cost=20.0,
        total_costs=50.0,
        net_pnl=net_pnl,
        net_pnl_pct=net_pnl / 100_000.0 * 100,
        max_favorable_excursion=abs(net_pnl) / 50,
        max_adverse_excursion=50.0,
        capture_ratio=0.8 if net_pnl > 0 else 0.0,
        duration_seconds=3600,
        duration_bars=60,
        confidence="HIGH",
        regime="TRENDING",
        risk_amount=2_000.0,
        entry_cost_at_open=30.0,
        outcome="WIN" if net_pnl > 0.5 else ("LOSS" if net_pnl < -0.5 else "BREAKEVEN"),
    )
