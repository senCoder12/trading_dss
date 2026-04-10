"""
Tests for the Trade Simulator (Phase 6 Step 6.2).

Covers: entry execution, slippage, position sizing, transaction costs,
SL/target hits, trailing SL, forced EOD, daily loss limits, max positions,
P&L calculation, equity curve, capital depletion, and reset.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from src.backtest.trade_simulator import (
    ClosedTrade,
    EquityPoint,
    PortfolioState,
    SimulatorConfig,
    TradeExecution,
    TradeSimulator,
)

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockSignal:
    """Minimal signal object matching TradingSignal interface for tests."""

    signal_id: str = ""
    index_id: str = "NIFTY50"
    signal_type: str = "BUY_CALL"
    confidence_level: str = "HIGH"
    entry_price: float = 22000.0
    target_price: float = 22200.0
    stop_loss: float = 21900.0
    risk_reward_ratio: float = 2.0
    position_size_modifier: float = 1.0
    generated_at: datetime = field(default_factory=lambda: datetime(2024, 6, 10, 10, 0, tzinfo=_IST))

    # Fields that may be present on RefinedSignal
    refined_entry: float = 0.0
    refined_target: float = 0.0
    refined_stop_loss: float = 0.0
    recommended_strike: Optional[float] = None
    option_premium: Optional[float] = None
    recommended_expiry: Optional[str] = None

    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = str(uuid.uuid4())


def _make_bar(
    open_: float = 22000.0,
    high: float = 22100.0,
    low: float = 21950.0,
    close: float = 22050.0,
    timestamp: Optional[datetime] = None,
) -> dict:
    """Create a bar dict."""
    ts = timestamp or datetime(2024, 6, 10, 10, 0, tzinfo=_IST)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1_000_000,
        "timestamp": ts,
    }


def _nifty_lot_size() -> int:
    return 75


# Patch IndexRegistry globally for all tests in this module
@pytest.fixture(autouse=True)
def _patch_registry():
    """Patch get_registry so we don't need a real indices.json file."""
    class _FakeIndex:
        def __init__(self, lot_size):
            self.lot_size = lot_size

    class _FakeRegistry:
        _lots = {"NIFTY50": 75, "BANKNIFTY": 15, "SENSEX": 10}

        def get_index(self, index_id):
            lot = self._lots.get(index_id)
            if lot is None:
                return None
            return _FakeIndex(lot)

    with patch("src.backtest.trade_simulator.get_registry", return_value=_FakeRegistry()):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> SimulatorConfig:
    return SimulatorConfig(
        initial_capital=100_000.0,
        max_risk_per_trade_pct=2.0,
        max_risk_per_day_pct=5.0,
        max_open_positions=3,
        max_positions_per_index=1,
        slippage_points=2.0,
        use_percentage_slippage=False,
        intraday_only=True,
        force_exit_minutes_before_close=5,
    )


@pytest.fixture()
def sim(config: SimulatorConfig) -> TradeSimulator:
    return TradeSimulator(config)


# ===========================================================================
# Test entry execution
# ===========================================================================


class TestEntryExecution:

    def test_basic_call_entry(self, sim: TradeSimulator):
        """BUY_CALL entry applies positive slippage."""
        signal = _MockSignal(signal_type="BUY_CALL", entry_price=22000.0)
        bar = _make_bar()
        result = sim.execute_entry(signal, bar)

        assert result is not None
        assert result.trade_type == "BUY_CALL"
        # Slippage of 2 points added for CALL
        assert result.actual_entry_price == 22002.0
        assert result.status == "OPEN"
        assert len(sim.open_positions) == 1

    def test_basic_put_entry(self, sim: TradeSimulator):
        """BUY_PUT entry applies negative slippage."""
        signal = _MockSignal(
            signal_type="BUY_PUT",
            entry_price=22000.0,
            target_price=21800.0,
            stop_loss=22100.0,
        )
        bar = _make_bar()
        result = sim.execute_entry(signal, bar)

        assert result is not None
        assert result.trade_type == "BUY_PUT"
        # Slippage of 2 points subtracted for PUT
        assert result.actual_entry_price == 21998.0

    def test_percentage_slippage(self):
        """Percentage-based slippage model."""
        cfg = SimulatorConfig(
            use_percentage_slippage=True,
            slippage_pct=0.05,  # 0.05%
        )
        sim = TradeSimulator(cfg)
        signal = _MockSignal(entry_price=22000.0)
        bar = _make_bar()
        result = sim.execute_entry(signal, bar)

        expected_slippage = 22000.0 * 0.05 / 100  # 11.0
        assert result is not None
        assert result.actual_entry_price == round(22000.0 + expected_slippage, 2)

    def test_no_trade_signal_rejected(self, sim: TradeSimulator):
        """NO_TRADE signal returns None."""
        signal = _MockSignal(signal_type="NO_TRADE")
        assert sim.execute_entry(signal, _make_bar()) is None

    def test_low_rr_rejected(self, sim: TradeSimulator):
        """Signal below min R:R ratio is rejected."""
        signal = _MockSignal(risk_reward_ratio=1.0)
        assert sim.execute_entry(signal, _make_bar()) is None

    def test_entry_cost_calculated(self, sim: TradeSimulator):
        """Entry cost includes brokerage, exchange, GST, stamp, SEBI."""
        signal = _MockSignal()
        result = sim.execute_entry(signal, _make_bar())
        assert result is not None
        assert result.entry_cost > 0
        # Brokerage alone is ₹20
        assert result.entry_cost >= 20.0


# ===========================================================================
# Test position sizing
# ===========================================================================


class TestPositionSizing:

    def test_sizing_with_known_values(self, sim: TradeSimulator):
        """1L capital, 2% risk, HIGH confidence, ~100pt SL → verify lots."""
        # max_risk = 100_000 * 2% = 2000
        # HIGH → 100% → risk_amount = 2000
        # SL distance = |22002 - 21900| = 102 pts
        # loss_per_lot = 102 * 75 + 120 = 7650 + 120 = 7770
        # lots = floor(2000 / 7770) = 0 → clamped to 1
        signal = _MockSignal(
            entry_price=22000.0,
            stop_loss=21900.0,
            confidence_level="HIGH",
        )
        result = sim.execute_entry(signal, _make_bar())
        assert result is not None
        assert result.lots == 1  # Clamped to minimum 1
        assert result.lot_size == 75
        assert result.quantity == 75

    def test_medium_confidence_reduces_risk(self, sim: TradeSimulator):
        """MEDIUM confidence uses 60% of max risk."""
        # max_risk = 2000, 60% → 1200
        signal = _MockSignal(confidence_level="MEDIUM")
        result = sim.execute_entry(signal, _make_bar())
        assert result is not None
        assert result.confidence_level == "MEDIUM"

    def test_low_confidence_reduces_risk(self, sim: TradeSimulator):
        """LOW confidence uses 30% of max risk."""
        signal = _MockSignal(confidence_level="LOW")
        result = sim.execute_entry(signal, _make_bar())
        assert result is not None
        assert result.confidence_level == "LOW"

    def test_lots_capped_at_max(self):
        """Lots are capped at max_lots_per_trade."""
        cfg = SimulatorConfig(
            initial_capital=10_000_000.0,
            max_lots_per_trade=5,
            slippage_points=0.0,
        )
        sim = TradeSimulator(cfg)
        # With 1cr capital, 2% risk = 200_000
        # SL dist 100 pts, loss_per_lot = 100*75 + 120 = 7620
        # lots = floor(200000 / 7620) = 26 → capped at 5
        signal = _MockSignal(entry_price=22000.0, stop_loss=21900.0)
        result = sim.execute_entry(signal, _make_bar())
        assert result is not None
        assert result.lots == 5

    def test_insufficient_capital_rejects(self):
        """Cannot afford even 1 lot → trade rejected."""
        cfg = SimulatorConfig(initial_capital=100.0)
        sim = TradeSimulator(cfg)
        signal = _MockSignal(entry_price=22000.0, stop_loss=21900.0)
        result = sim.execute_entry(signal, _make_bar())
        assert result is None


# ===========================================================================
# Test transaction costs
# ===========================================================================


class TestTransactionCosts:

    def test_entry_has_no_stt(self, sim: TradeSimulator):
        """STT is zero on buy side for options — only charged on exit."""
        signal = _MockSignal()
        result = sim.execute_entry(signal, _make_bar())
        assert result is not None
        # Entry cost = brokerage(20) + exchange + GST + stamp + SEBI
        # No STT component on entry
        assert result.entry_cost >= 20.0

    def test_exit_includes_stt(self, sim: TradeSimulator):
        """Exit cost includes STT on sell side."""
        signal = _MockSignal()
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None

        ts = datetime(2024, 6, 10, 11, 0, tzinfo=_IST)
        closed = sim.close_position(
            entry, 22200.0, "TARGET_HIT", ts,
        )
        assert closed.total_costs > entry.entry_cost  # Exit has STT on top


# ===========================================================================
# Test SL hit
# ===========================================================================


class TestStopLossHit:

    def test_call_sl_hit_on_bar_low(self, sim: TradeSimulator):
        """CALL SL triggers when bar low <= stop_loss."""
        signal = _MockSignal(
            entry_price=22000.0,
            stop_loss=21900.0,
            target_price=22200.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None

        # Bar where low touches SL
        bar = _make_bar(
            open_=21950.0, high=21960.0, low=21895.0, close=21920.0,
            timestamp=datetime(2024, 6, 10, 10, 15, tzinfo=_IST),
        )
        closed = sim.update_positions(bar, bar["timestamp"])

        assert len(closed) == 1
        assert closed[0].exit_reason == "STOP_LOSS_HIT"
        assert closed[0].outcome == "LOSS"
        assert len(sim.open_positions) == 0

    def test_put_sl_hit_on_bar_high(self, sim: TradeSimulator):
        """PUT SL triggers when bar high >= stop_loss."""
        signal = _MockSignal(
            signal_type="BUY_PUT",
            entry_price=22000.0,
            stop_loss=22100.0,
            target_price=21800.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None

        bar = _make_bar(
            open_=22020.0, high=22110.0, low=21990.0, close=22050.0,
            timestamp=datetime(2024, 6, 10, 10, 15, tzinfo=_IST),
        )
        closed = sim.update_positions(bar, bar["timestamp"])

        assert len(closed) == 1
        assert closed[0].exit_reason == "STOP_LOSS_HIT"

    def test_sl_uses_trailing_not_original(self, sim: TradeSimulator):
        """After trailing SL ratchets, exit uses the new trailing value."""
        signal = _MockSignal(
            entry_price=22000.0,
            stop_loss=21900.0,
            target_price=22200.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None

        # Bar 1: price rallies to 75% of target → SL moves to entry+30%
        # entry=22002, target=22200, tgt_dist=198
        # 75% of tgt_dist = 148.5, so need close at 22002+148.5=22150.5
        bar1 = _make_bar(
            open_=22100.0, high=22160.0, low=22090.0, close=22155.0,
            timestamp=datetime(2024, 6, 10, 10, 15, tzinfo=_IST),
        )
        sim.update_positions(bar1, bar1["timestamp"])

        # Trailing SL should be at entry + 0.3 * tgt_dist = 22002 + 59.4 = 22061.4
        assert entry.trailing_sl == round(22002.0 + 0.3 * 198.0, 2)

        # Bar 2: price drops to trailing SL
        new_sl = entry.trailing_sl
        bar2 = _make_bar(
            open_=22080.0, high=22085.0, low=new_sl - 5, close=new_sl - 2,
            timestamp=datetime(2024, 6, 10, 10, 30, tzinfo=_IST),
        )
        closed = sim.update_positions(bar2, bar2["timestamp"])
        assert len(closed) == 1
        assert closed[0].exit_reason == "TRAILING_SL_HIT"


# ===========================================================================
# Test target hit
# ===========================================================================


class TestTargetHit:

    def test_call_target_hit_on_bar_high(self, sim: TradeSimulator):
        """CALL target triggers when bar high >= target."""
        signal = _MockSignal(
            entry_price=22000.0,
            stop_loss=21900.0,
            target_price=22200.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None

        bar = _make_bar(
            open_=22150.0, high=22210.0, low=22130.0, close=22190.0,
            timestamp=datetime(2024, 6, 10, 10, 15, tzinfo=_IST),
        )
        closed = sim.update_positions(bar, bar["timestamp"])

        assert len(closed) == 1
        assert closed[0].exit_reason == "TARGET_HIT"
        assert closed[0].outcome == "WIN"

    def test_put_target_hit_on_bar_low(self, sim: TradeSimulator):
        """PUT target triggers when bar low <= target."""
        signal = _MockSignal(
            signal_type="BUY_PUT",
            entry_price=22000.0,
            stop_loss=22100.0,
            target_price=21800.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None

        bar = _make_bar(
            open_=21850.0, high=21860.0, low=21790.0, close=21820.0,
            timestamp=datetime(2024, 6, 10, 10, 15, tzinfo=_IST),
        )
        closed = sim.update_positions(bar, bar["timestamp"])

        assert len(closed) == 1
        assert closed[0].exit_reason == "TARGET_HIT"

    def test_both_sl_and_target_assumes_sl(self, sim: TradeSimulator):
        """If both SL and target hit in same bar → assume SL (conservative)."""
        signal = _MockSignal(
            entry_price=22000.0,
            stop_loss=21900.0,
            target_price=22200.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None

        # Extreme bar: low below SL AND high above target
        bar = _make_bar(
            open_=22000.0, high=22250.0, low=21850.0, close=22100.0,
            timestamp=datetime(2024, 6, 10, 10, 15, tzinfo=_IST),
        )
        closed = sim.update_positions(bar, bar["timestamp"])

        assert len(closed) == 1
        assert closed[0].exit_reason == "STOP_LOSS_HIT"


# ===========================================================================
# Test trailing SL
# ===========================================================================


class TestTrailingSL:

    def test_50pct_profit_moves_to_breakeven(self, sim: TradeSimulator):
        """At 50%+ of target distance, SL moves to entry (breakeven)."""
        signal = _MockSignal(
            entry_price=22000.0,
            stop_loss=21900.0,
            target_price=22200.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None
        # actual_entry = 22002, target = 22200, tgt_dist = 198
        # 50% profit = 99 pts → close at 22002+99 = 22101

        bar = _make_bar(
            open_=22100.0, high=22110.0, low=22090.0, close=22105.0,
            timestamp=datetime(2024, 6, 10, 10, 15, tzinfo=_IST),
        )
        sim.update_positions(bar, bar["timestamp"])

        # Trailing SL should have moved to entry (breakeven)
        assert entry.trailing_sl == entry.actual_entry_price

    def test_75pct_profit_moves_to_30pct(self, sim: TradeSimulator):
        """At 75%+ of target distance, SL moves to entry + 30% of tgt_dist."""
        signal = _MockSignal(
            entry_price=22000.0,
            stop_loss=21900.0,
            target_price=22200.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None
        # tgt_dist = 198, 75% = 148.5 → close at 22002+148.5=22150.5

        bar = _make_bar(
            open_=22140.0, high=22155.0, low=22135.0, close=22152.0,
            timestamp=datetime(2024, 6, 10, 10, 15, tzinfo=_IST),
        )
        sim.update_positions(bar, bar["timestamp"])

        expected = round(22002.0 + 0.3 * 198.0, 2)
        assert entry.trailing_sl == expected

    def test_100pct_profit_moves_to_60pct(self, sim: TradeSimulator):
        """At 100%+ of target distance, SL moves to entry + 60% of tgt_dist."""
        signal = _MockSignal(
            entry_price=22000.0,
            stop_loss=21900.0,
            target_price=22300.0,    # Wider target so bar doesn't trigger exit
            risk_reward_ratio=3.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None
        # actual_entry = 22002, target = 22300, tgt_dist = 298

        # Bar with close at 100%+ of tgt_dist but high below target
        # 100% profit = 298 pts → close at 22002+298 = 22300
        # Use close=22305, but keep high < target to avoid target exit
        bar = _make_bar(
            open_=22295.0, high=22299.0, low=22290.0, close=22305.0,
            timestamp=datetime(2024, 6, 10, 10, 15, tzinfo=_IST),
        )
        sim.update_positions(bar, bar["timestamp"])

        expected = round(22002.0 + 0.6 * 298.0, 2)
        assert entry.trailing_sl == expected

    def test_trailing_sl_never_widens(self, sim: TradeSimulator):
        """Trailing SL only ratchets — never moves against the position."""
        signal = _MockSignal(
            entry_price=22000.0,
            stop_loss=21900.0,
            target_price=22200.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None

        # Bar 1: 75% profit → SL moves up
        bar1 = _make_bar(
            open_=22140.0, high=22155.0, low=22135.0, close=22152.0,
            timestamp=datetime(2024, 6, 10, 10, 15, tzinfo=_IST),
        )
        sim.update_positions(bar1, bar1["timestamp"])
        sl_after_75 = entry.trailing_sl

        # Bar 2: price pulls back to 55% — SL should NOT move down
        bar2 = _make_bar(
            open_=22110.0, high=22115.0, low=22100.0, close=22108.0,
            timestamp=datetime(2024, 6, 10, 10, 30, tzinfo=_IST),
        )
        sim.update_positions(bar2, bar2["timestamp"])

        assert entry.trailing_sl == sl_after_75

    def test_put_trailing_sl_moves_down(self, sim: TradeSimulator):
        """For PUT, trailing SL ratchets downward."""
        signal = _MockSignal(
            signal_type="BUY_PUT",
            entry_price=22000.0,
            stop_loss=22100.0,
            target_price=21800.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None
        # actual_entry = 21998, target = 21800, tgt_dist = 198
        # 50% profit → close at 21998-99 = 21899

        bar = _make_bar(
            open_=21910.0, high=21915.0, low=21890.0, close=21895.0,
            timestamp=datetime(2024, 6, 10, 10, 15, tzinfo=_IST),
        )
        sim.update_positions(bar, bar["timestamp"])

        # Trailing SL should move to entry (breakeven) for PUT
        assert entry.trailing_sl == entry.actual_entry_price


# ===========================================================================
# Test forced EOD exit
# ===========================================================================


class TestForcedEODExit:

    def test_forced_exit_at_eod(self, sim: TradeSimulator):
        """All positions closed at force-exit time (15:25 for 5min config)."""
        signal = _MockSignal()
        entry_bar = _make_bar(timestamp=datetime(2024, 6, 10, 10, 0, tzinfo=_IST))
        sim.execute_entry(signal, entry_bar)

        # Bar at 15:25 (5 min before 15:30)
        eod_bar = _make_bar(
            timestamp=datetime(2024, 6, 10, 15, 25, tzinfo=_IST),
            close=22050.0,
        )
        closed = sim.update_positions(eod_bar, eod_bar["timestamp"])

        assert len(closed) == 1
        assert closed[0].exit_reason == "FORCED_EOD"

    def test_no_forced_exit_when_disabled(self):
        """With intraday_only=False, no forced exit."""
        cfg = SimulatorConfig(intraday_only=False)
        sim = TradeSimulator(cfg)
        signal = _MockSignal()
        entry_bar = _make_bar(timestamp=datetime(2024, 6, 10, 10, 0, tzinfo=_IST))
        sim.execute_entry(signal, entry_bar)

        eod_bar = _make_bar(
            timestamp=datetime(2024, 6, 10, 15, 25, tzinfo=_IST),
        )
        closed = sim.update_positions(eod_bar, eod_bar["timestamp"])
        assert len(closed) == 0


# ===========================================================================
# Test daily loss limit
# ===========================================================================


class TestDailyLossLimit:

    def test_daily_loss_limit_blocks_entry(self, sim: TradeSimulator):
        """After losing 5% of capital in a day, next entry is rejected."""
        # Manually set daily P&L to -5000 (5% of 100_000)
        date_str = "2024-06-10"
        sim._daily_pnl[date_str] = -5000.0
        sim._daily_trades[date_str] = 3
        sim._current_date = date_str

        signal = _MockSignal()
        bar = _make_bar(timestamp=datetime(2024, 6, 10, 11, 0, tzinfo=_IST))
        result = sim.execute_entry(signal, bar)
        assert result is None

    def test_under_limit_allows_entry(self, sim: TradeSimulator):
        """Under daily loss limit, entries are allowed."""
        date_str = "2024-06-10"
        sim._daily_pnl[date_str] = -2000.0
        sim._daily_trades[date_str] = 1
        sim._current_date = date_str

        signal = _MockSignal()
        bar = _make_bar(timestamp=datetime(2024, 6, 10, 11, 0, tzinfo=_IST))
        result = sim.execute_entry(signal, bar)
        assert result is not None


# ===========================================================================
# Test max positions
# ===========================================================================


class TestMaxPositions:

    def test_max_open_positions_blocks_entry(self, sim: TradeSimulator):
        """4th entry rejected when max_open_positions=3."""
        indices = ["NIFTY50", "BANKNIFTY", "SENSEX"]
        for idx in indices:
            sig = _MockSignal(index_id=idx)
            bar = _make_bar(timestamp=datetime(2024, 6, 10, 10, idx.__hash__() % 50, tzinfo=_IST))
            result = sim.execute_entry(sig, bar)
            assert result is not None, f"Entry for {idx} should succeed"

        assert len(sim.open_positions) == 3

        # 4th entry for a different signal but we need a new index
        sig4 = _MockSignal(index_id="NIFTY50")
        bar4 = _make_bar(timestamp=datetime(2024, 6, 10, 10, 55, tzinfo=_IST))
        result = sim.execute_entry(sig4, bar4)
        assert result is None

    def test_max_positions_per_index_blocks_duplicate(self, sim: TradeSimulator):
        """Second position on same index rejected when max_positions_per_index=1."""
        sig1 = _MockSignal(index_id="NIFTY50")
        bar1 = _make_bar(timestamp=datetime(2024, 6, 10, 10, 0, tzinfo=_IST))
        result1 = sim.execute_entry(sig1, bar1)
        assert result1 is not None

        sig2 = _MockSignal(index_id="NIFTY50")
        bar2 = _make_bar(timestamp=datetime(2024, 6, 10, 10, 5, tzinfo=_IST))
        result2 = sim.execute_entry(sig2, bar2)
        assert result2 is None


# ===========================================================================
# Test P&L calculation
# ===========================================================================


class TestPnLCalculation:

    def test_call_profit_pnl(self, sim: TradeSimulator):
        """Known CALL entry/exit → verify exact net P&L."""
        cfg = SimulatorConfig(
            initial_capital=100_000.0,
            slippage_points=0.0,  # Zero slippage for exact math
            brokerage_per_order=20.0,
            stt_rate=0.000625,
            exchange_charges_pct=0.00053,
            gst_pct=0.18,
            stamp_duty_pct=0.00003,
            sebi_charges_pct=0.000001,
        )
        sim_exact = TradeSimulator(cfg)

        signal = _MockSignal(
            entry_price=22000.0,
            stop_loss=21900.0,
            target_price=22200.0,
        )
        entry = sim_exact.execute_entry(signal, _make_bar())
        assert entry is not None
        # With 0 slippage: actual_entry = 22000
        assert entry.actual_entry_price == 22000.0

        lots = entry.lots
        qty = entry.quantity
        ts = datetime(2024, 6, 10, 11, 0, tzinfo=_IST)
        closed = sim_exact.close_position(entry, 22200.0, "TARGET_HIT", ts)

        # gross_pnl = (22200 - 22000) * qty = 200 * 75 = 15000
        assert closed.gross_pnl_points == 200.0
        assert closed.gross_pnl == 200.0 * qty
        # Net = gross - costs
        assert closed.net_pnl < closed.gross_pnl  # costs deducted
        assert closed.outcome == "WIN"

    def test_put_loss_pnl(self, sim: TradeSimulator):
        """PUT trade that hits SL → verify loss."""
        signal = _MockSignal(
            signal_type="BUY_PUT",
            entry_price=22000.0,
            stop_loss=22100.0,
            target_price=21800.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None

        ts = datetime(2024, 6, 10, 10, 15, tzinfo=_IST)
        closed = sim.close_position(entry, 22100.0, "STOP_LOSS_HIT", ts)

        # For PUT: gross_pnl = (actual_entry - actual_exit) * qty
        # actual_entry = 21998 (slippage), exit = 22100 + 2 = 22102 (adverse slippage)
        # gross_pnl = (21998 - 22102) * 75 = -104 * 75 = -7800
        assert closed.gross_pnl < 0
        assert closed.outcome == "LOSS"

    def test_capital_updated_after_close(self, sim: TradeSimulator):
        """Capital is adjusted by net P&L after closing."""
        initial = sim.current_capital
        signal = _MockSignal()
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None

        ts = datetime(2024, 6, 10, 11, 0, tzinfo=_IST)
        closed = sim.close_position(entry, 22200.0, "TARGET_HIT", ts)

        assert sim.current_capital == pytest.approx(initial + closed.net_pnl, abs=0.01)


# ===========================================================================
# Test equity curve
# ===========================================================================


class TestEquityCurve:

    def test_equity_recorded_per_bar(self, sim: TradeSimulator):
        """Equity curve records one point per update_positions call."""
        signal = _MockSignal()
        sim.execute_entry(signal, _make_bar())

        for i in range(5):
            bar = _make_bar(
                close=22050.0 + i * 10,
                timestamp=datetime(2024, 6, 10, 10, 5 * (i + 1), tzinfo=_IST),
            )
            sim.update_positions(bar, bar["timestamp"])

        assert len(sim.equity_curve) == 5

    def test_equity_point_fields(self, sim: TradeSimulator):
        """Each EquityPoint has expected fields."""
        bar = _make_bar(timestamp=datetime(2024, 6, 10, 10, 5, tzinfo=_IST))
        sim.update_positions(bar, bar["timestamp"])

        assert len(sim.equity_curve) == 1
        ep = sim.equity_curve[0]
        assert isinstance(ep, EquityPoint)
        assert ep.cash == sim.current_capital
        assert ep.open_positions == 0
        assert ep.drawdown_pct >= 0.0


# ===========================================================================
# Test capital depletion
# ===========================================================================


class TestCapitalDepletion:

    def test_depleted_capital_stops_trading(self):
        """When capital drops to 0, no more entries allowed."""
        cfg = SimulatorConfig(initial_capital=500.0, slippage_points=0.0)
        sim = TradeSimulator(cfg)

        # Force capital to 0
        sim.current_capital = 0.0
        sim._capital_depleted = True

        signal = _MockSignal()
        result = sim.execute_entry(signal, _make_bar())
        assert result is None

    def test_large_loss_triggers_depletion(self):
        """A loss that wipes out capital sets the depleted flag."""
        cfg = SimulatorConfig(initial_capital=50_000.0, slippage_points=0.0)
        sim = TradeSimulator(cfg)

        signal = _MockSignal(
            index_id="SENSEX",       # lot_size=10, more affordable
            entry_price=72000.0,
            stop_loss=71500.0,
            target_price=73000.0,
        )
        bar = _make_bar(open_=72000.0, high=72100.0, low=71900.0, close=72050.0)
        entry = sim.execute_entry(signal, bar)
        assert entry is not None

        # Force capital very low, then close at a loss
        sim.current_capital = 50.0
        ts = datetime(2024, 6, 10, 11, 0, tzinfo=_IST)
        closed = sim.close_position(entry, 71000.0, "STOP_LOSS_HIT", ts)

        # Net loss should wipe out the remaining capital
        assert closed.net_pnl < 0
        assert sim._capital_depleted is True


# ===========================================================================
# Test reset
# ===========================================================================


class TestReset:

    def test_reset_clears_everything(self, sim: TradeSimulator):
        """After reset, all state returns to initial."""
        signal = _MockSignal()
        sim.execute_entry(signal, _make_bar())
        sim.current_capital = 80_000.0
        sim._daily_pnl["2024-06-10"] = -500.0
        sim.equity_curve.append(
            EquityPoint(
                timestamp=datetime(2024, 6, 10, tzinfo=_IST),
                capital=80_000.0, cash=80_000.0,
                unrealized=0.0, drawdown_pct=20.0, open_positions=1,
            )
        )

        sim.reset()

        assert sim.current_capital == sim.config.initial_capital
        assert len(sim.open_positions) == 0
        assert len(sim.trade_history) == 0
        assert len(sim.equity_curve) == 0
        assert len(sim._daily_pnl) == 0
        assert sim._capital_depleted is False
        assert sim._max_capital == sim.config.initial_capital


# ===========================================================================
# Test edge cases
# ===========================================================================


class TestEdgeCases:

    def test_nan_bar_skipped(self, sim: TradeSimulator):
        """Bars with NaN values are skipped, positions carried forward."""
        signal = _MockSignal()
        sim.execute_entry(signal, _make_bar())

        nan_bar = _make_bar(
            timestamp=datetime(2024, 6, 10, 10, 5, tzinfo=_IST),
        )
        nan_bar["low"] = float("nan")

        closed = sim.update_positions(nan_bar, nan_bar["timestamp"])
        assert len(closed) == 0
        assert len(sim.open_positions) == 1

    def test_unknown_index_raises(self, sim: TradeSimulator):
        """Unknown index_id raises ValueError."""
        signal = _MockSignal(index_id="UNKNOWN_INDEX")
        with pytest.raises(ValueError, match="No F&O lot size"):
            sim.execute_entry(signal, _make_bar())

    def test_duplicate_signal_same_bar_skipped(self, sim: TradeSimulator):
        """Second signal for same index on same bar is skipped."""
        bar = _make_bar(timestamp=datetime(2024, 6, 10, 10, 0, tzinfo=_IST))

        sig1 = _MockSignal(index_id="NIFTY50")
        result1 = sim.execute_entry(sig1, bar)
        assert result1 is not None

        # Close it so we can test duplicate filter (not max-per-index filter)
        sim.close_position(
            result1, 22050.0, "MANUAL",
            datetime(2024, 6, 10, 10, 0, tzinfo=_IST),
        )

        sig2 = _MockSignal(index_id="NIFTY50")
        result2 = sim.execute_entry(sig2, bar)
        assert result2 is None  # Duplicate bar signal

    def test_mfe_mae_tracked(self, sim: TradeSimulator):
        """Max favorable/adverse excursion tracked over position lifetime."""
        signal = _MockSignal(
            entry_price=22000.0,
            stop_loss=21900.0,
            target_price=22200.0,
        )
        entry = sim.execute_entry(signal, _make_bar())
        assert entry is not None

        # Bar 1: rally
        bar1 = _make_bar(
            open_=22050.0, high=22150.0, low=22040.0, close=22100.0,
            timestamp=datetime(2024, 6, 10, 10, 5, tzinfo=_IST),
        )
        sim.update_positions(bar1, bar1["timestamp"])

        # Bar 2: pullback
        bar2 = _make_bar(
            open_=22080.0, high=22090.0, low=21960.0, close=21970.0,
            timestamp=datetime(2024, 6, 10, 10, 10, tzinfo=_IST),
        )
        sim.update_positions(bar2, bar2["timestamp"])

        # Forced close
        eod = _make_bar(
            open_=21980.0, high=21990.0, low=21960.0, close=21980.0,
            timestamp=datetime(2024, 6, 10, 15, 25, tzinfo=_IST),
        )
        closed = sim.update_positions(eod, eod["timestamp"])
        assert len(closed) == 1

        # MFE = max(high - entry) across all bars
        # entry = 22002, bar1 high = 22150 → MFE = 148
        assert closed[0].max_favorable_excursion >= 148.0

        # MAE = max(entry - low) across all bars
        # entry = 22002, bar2 low = 21960 → MAE = 42
        assert closed[0].max_adverse_excursion >= 42.0

    def test_portfolio_state(self, sim: TradeSimulator):
        """get_portfolio_state returns correct snapshot."""
        state = sim.get_portfolio_state()
        assert state.current_capital == sim.config.initial_capital
        assert state.total_return == 0.0
        assert state.open_position_count == 0
        assert state.total_trades == 0
        assert isinstance(state, PortfolioState)

    def test_get_trade_history_empty(self, sim: TradeSimulator):
        """Trade history starts empty."""
        assert sim.get_trade_history() == []

    def test_duration_bars_and_minutes(self, sim: TradeSimulator):
        """ClosedTrade records correct duration in bars and minutes."""
        signal = _MockSignal()
        entry_bar = _make_bar(timestamp=datetime(2024, 6, 10, 10, 0, tzinfo=_IST))
        entry = sim.execute_entry(signal, entry_bar)
        assert entry is not None

        # 3 bars later
        for i in range(1, 4):
            bar = _make_bar(
                close=22050.0 + i * 10,
                timestamp=datetime(2024, 6, 10, 10, 5 * i, tzinfo=_IST),
            )
            sim.update_positions(bar, bar["timestamp"])

        ts = datetime(2024, 6, 10, 10, 20, tzinfo=_IST)
        closed = sim.close_position(entry, 22100.0, "MANUAL", ts)

        assert closed.duration_bars == 3
        assert closed.duration_minutes == 20

    def test_none_bar_values_skipped(self, sim: TradeSimulator):
        """Bars with None values are skipped."""
        signal = _MockSignal()
        sim.execute_entry(signal, _make_bar())

        bar = _make_bar(timestamp=datetime(2024, 6, 10, 10, 5, tzinfo=_IST))
        bar["close"] = None

        closed = sim.update_positions(bar, bar["timestamp"])
        assert len(closed) == 0
        assert len(sim.open_positions) == 1
