"""
Tests for EdgeTracker — Phase 9.2.

Coverage:
- 30 winning trades → edge STRONG
- Alternating wins/losses at 50% WR (no history) → edge INTACT
- Declining win rate over 3 consecutive windows → edge WEAKENING
- Profit factor < 1.0 (consistent losses) → edge GONE
- Insufficient data (< 20 trades) → INSUFFICIENT_DATA
- Confidence scaling: 0 below threshold, ≥ 0.5 at threshold, 1.0 at 50+ trades
- Trend detection: IMPROVING / STABLE / DECLINING
- assess_edge returns correct should_pause / should_reduce_size / should_reoptimize flags
- get_rolling_edge_history with empty DB returns []
"""
from __future__ import annotations

import math
import sqlite3
import uuid
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pytest

from src.paper_trading.edge_tracker import EdgeAssessment, EdgeTracker
from src.paper_trading.paper_engine import PaperTrade

_IST = ZoneInfo("Asia/Kolkata")
_ENTRY_TS = datetime(2024, 6, 10, 10, 30, tzinfo=_IST)
_EXIT_TS = datetime(2024, 6, 10, 14, 0, tzinfo=_IST)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    net_pnl: float,
    confidence: str = "HIGH",
    outcome: Optional[str] = None,
) -> PaperTrade:
    """Return a fully-populated PaperTrade with the given net_pnl."""
    if outcome is None:
        outcome = "WIN" if net_pnl > 0 else ("LOSS" if net_pnl < 0 else "BREAKEVEN")

    gross_pnl = net_pnl + 120.0  # approximate gross (cost ~ ₹120)
    return PaperTrade(
        position_id=str(uuid.uuid4()),
        signal_id=str(uuid.uuid4()),
        index_id="NIFTY50",
        trade_type="BUY_CALL",
        signal_entry_price=22_000.0,
        market_price_at_signal=22_000.0,
        execution_price=22_003.0,
        entry_timestamp=_ENTRY_TS,
        original_stop_loss=21_900.0,
        original_target=22_200.0,
        strike=22_000.0,
        expiry="2024-06-13",
        option_premium_at_entry=180.0,
        lots=1,
        lot_size=50,
        quantity=50,
        exit_price=22_200.0,
        actual_exit_price=22_198.0,
        exit_timestamp=_EXIT_TS,
        exit_reason="TARGET_HIT" if outcome == "WIN" else "STOP_LOSS_HIT",
        gross_pnl=gross_pnl,
        entry_cost=60.0,
        exit_cost=60.0,
        total_costs=120.0,
        net_pnl=net_pnl,
        net_pnl_pct=net_pnl / 100_000 * 100,
        max_favorable_excursion=200.0,
        max_adverse_excursion=50.0,
        capture_ratio=0.8,
        duration_seconds=12_600,
        duration_bars=210,
        confidence=confidence,
        regime="TRENDING",
        risk_amount=2_000.0,
        entry_cost_at_open=60.0,
        outcome=outcome,
    )


def _wins(n: int, pnl: float = 1_000.0) -> list[PaperTrade]:
    return [_make_trade(pnl, outcome="WIN") for _ in range(n)]


def _losses(n: int, pnl: float = -800.0) -> list[PaperTrade]:
    return [_make_trade(pnl, outcome="LOSS") for _ in range(n)]


class _InMemoryDB:
    """Minimal DB stub satisfying the DatabaseManager duck-type."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS trading_signals (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                outcome          TEXT,
                actual_pnl       REAL,
                closed_at        TEXT,
                confidence_level TEXT
            )"""
        )
        self._conn.commit()

    def execute(self, query: str, params: tuple):
        cur = self._conn.execute(query, params)
        self._conn.commit()
        return cur

    def fetch_one(self, query: str, params: tuple) -> Optional[dict]:
        cur = self._conn.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def fetch_all(self, query: str, params: tuple) -> list:
        cur = self._conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]


def _make_tracker() -> EdgeTracker:
    return EdgeTracker(_InMemoryDB())


# ---------------------------------------------------------------------------
# 1. STRONG edge — 30 consecutive winners
# ---------------------------------------------------------------------------


class TestStrongEdge:
    def test_30_winners_returns_strong(self):
        tracker = _make_tracker()
        trades = _wins(30)
        result = tracker.assess_edge(trades)

        assert result.edge_status == "STRONG"
        assert result.win_rate == pytest.approx(100.0)
        assert result.profit_factor == math.inf
        assert result.expected_value > 0
        assert result.should_pause is False
        assert result.should_reduce_size is False
        assert result.should_reoptimize is False

    def test_strong_metrics_60pct_wr(self):
        """60% WR, PF > 1.3, EV > 0 → STRONG."""
        tracker = _make_tracker()
        trades = _wins(30, pnl=1_000.0) + _losses(20, pnl=-400.0)
        result = tracker.assess_edge(trades)

        assert result.edge_status == "STRONG"
        assert result.win_rate == pytest.approx(60.0)
        assert result.profit_factor > 1.3


# ---------------------------------------------------------------------------
# 2. INTACT edge — 50 % win rate, positive EV, no declining trend
# ---------------------------------------------------------------------------


class TestIntactEdge:
    def test_50pct_wr_no_history_returns_intact(self):
        """50% WR, PF slightly above 1.1, no prior history → INTACT."""
        tracker = _make_tracker()
        # 25 wins at +500, 25 losses at -450 → PF ≈ 1.11, EV = +25
        trades = _wins(25, pnl=500.0) + _losses(25, pnl=-450.0)
        result = tracker.assess_edge(trades)

        assert result.edge_status == "INTACT"
        assert result.win_rate == pytest.approx(50.0)
        assert result.profit_factor > 1.0
        assert result.expected_value > 0
        assert result.should_pause is False
        assert result.should_reduce_size is False

    def test_46pct_wr_just_above_thresholds_intact(self):
        """46% WR, PF = 1.14, EV > 0 → INTACT (all metrics above thresholds)."""
        tracker = _make_tracker()
        # 23 wins at +700, 27 losses at -600 → PF = 16100/16200 ≈ 0.99... not good
        # Use: 23 wins at +800, 27 losses at -600 → PF = 18400/16200 ≈ 1.135
        trades = _wins(23, pnl=800.0) + _losses(27, pnl=-600.0)
        result = tracker.assess_edge(trades)

        assert result.edge_status == "INTACT"
        assert result.win_rate == pytest.approx(46.0)
        assert result.profit_factor >= 1.1


# ---------------------------------------------------------------------------
# 3. WEAKENING edge — declining win rate over 3 consecutive windows
# ---------------------------------------------------------------------------


class TestWeakeningEdge:
    def test_3_window_win_rate_decline_triggers_weakening(self):
        """
        Three successive calls with decreasing WR should trigger WEAKENING on
        the third call even though the absolute metrics still pass INTACT thresholds.

        Window 1: 30/50 = 60% WR → STRONG (stored in history)
        Window 2: 27/50 = 54% WR → INTACT  (stored in history)
        Window 3: 24/50 = 48% WR → WEAKENING (history[-2]=60 > history[-1]=54 > 48)
        """
        tracker = _make_tracker()

        # Window 1 — 60% WR
        w1 = _wins(30, pnl=1_000.0) + _losses(20, pnl=-400.0)
        r1 = tracker.assess_edge(w1)
        assert r1.edge_status == "STRONG"

        # Window 2 — 54% WR, PF > 1.1
        w2 = _wins(27, pnl=1_000.0) + _losses(23, pnl=-600.0)
        r2 = tracker.assess_edge(w2)
        assert r2.edge_status in ("STRONG", "INTACT")  # still ok

        # Window 3 — 48% WR: decline confirmed over 3 windows
        w3 = _wins(24, pnl=1_000.0) + _losses(26, pnl=-700.0)
        r3 = tracker.assess_edge(w3)

        assert r3.edge_status == "WEAKENING"
        assert r3.win_rate_trend == "DECLINING"
        assert r3.should_reduce_size is True
        assert r3.should_reoptimize is True
        assert r3.should_pause is False  # WEAKENING, not GONE

    def test_pf_below_1_1_triggers_weakening(self):
        """PF between 1.0 and 1.1 (no trend needed) → WEAKENING.

        Need WR >= 45% (not GONE by WR) and PF in [1.0, 1.1).
        23 wins at +800, 27 losses at -680:
          gross_wins=18400, gross_losses=18360 → PF≈1.002, WR=46%, EV=+0.8
        """
        tracker = _make_tracker()
        trades = _wins(23, pnl=800.0) + _losses(27, pnl=-680.0)
        result = tracker.assess_edge(trades)

        assert result.edge_status == "WEAKENING"
        assert 1.0 <= result.profit_factor < 1.1
        assert result.win_rate >= 45


# ---------------------------------------------------------------------------
# 4. GONE edge — profit factor < 1.0
# ---------------------------------------------------------------------------


class TestGoneEdge:
    def test_pf_below_1_returns_gone(self):
        """10 wins, 30 losses with large losses → PF < 1.0 → GONE."""
        tracker = _make_tracker()
        trades = _wins(10, pnl=500.0) + _losses(30, pnl=-800.0)
        result = tracker.assess_edge(trades)

        assert result.edge_status == "GONE"
        assert result.profit_factor < 1.0
        assert result.should_reoptimize is True

    def test_win_rate_below_45_returns_gone(self):
        """Win rate below 45% → GONE regardless of PF."""
        tracker = _make_tracker()
        # 8 wins, 22 losses → 36% WR < 45%
        trades = _wins(8, pnl=1_000.0) + _losses(22, pnl=-100.0)
        result = tracker.assess_edge(trades)

        assert result.edge_status == "GONE"
        assert result.win_rate < 45

    def test_negative_ev_returns_gone(self):
        """Negative expected value → GONE."""
        tracker = _make_tracker()
        # 10 wins at +100, 10 losses at -200 → EV = -50, PF = 0.5
        trades = _wins(10, pnl=100.0) + _losses(10, pnl=-200.0)
        result = tracker.assess_edge(trades)

        assert result.edge_status == "GONE"
        assert result.expected_value < 0

    def test_gone_with_high_confidence_sets_should_pause(self):
        """GONE + >30 trades → should_pause = True."""
        tracker = _make_tracker()
        trades = _wins(10, pnl=200.0) + _losses(40, pnl=-600.0)  # 50 trades
        result = tracker.assess_edge(trades)

        assert result.edge_status == "GONE"
        assert result.trade_count > 30
        assert result.should_pause is True

    def test_gone_with_few_trades_does_not_pause(self):
        """GONE but only 30 trades → should_pause = False (may be bad luck)."""
        tracker = _make_tracker()
        trades = _wins(8, pnl=200.0) + _losses(22, pnl=-600.0)  # 30 trades
        result = tracker.assess_edge(trades)

        assert result.edge_status == "GONE"
        assert result.trade_count == 30
        assert result.should_pause is False


# ---------------------------------------------------------------------------
# 5. Insufficient data
# ---------------------------------------------------------------------------


class TestInsufficientData:
    def test_fewer_than_20_trades_returns_insufficient(self):
        tracker = _make_tracker()
        trades = _wins(10, pnl=1_000.0) + _losses(5, pnl=-400.0)  # 15 trades
        result = tracker.assess_edge(trades)

        assert result.edge_status == "INSUFFICIENT_DATA"
        assert result.confidence_in_assessment == 0.0
        assert result.should_pause is False
        assert result.should_reduce_size is False

    def test_exactly_19_trades_is_insufficient(self):
        tracker = _make_tracker()
        trades = _wins(15) + _losses(4)  # 19 trades
        result = tracker.assess_edge(trades)
        assert result.edge_status == "INSUFFICIENT_DATA"

    def test_exactly_20_trades_is_sufficient(self):
        tracker = _make_tracker()
        trades = _wins(12, pnl=1_000.0) + _losses(8, pnl=-500.0)  # 20 trades
        result = tracker.assess_edge(trades)
        # 60% WR, PF = 3.0 → STRONG (just at minimum data threshold)
        assert result.edge_status != "INSUFFICIENT_DATA"

    def test_zero_trades_returns_insufficient(self):
        tracker = _make_tracker()
        result = tracker.assess_edge([])
        assert result.edge_status == "INSUFFICIENT_DATA"
        assert result.trade_count == 0
        assert result.confidence_in_assessment == 0.0


# ---------------------------------------------------------------------------
# 6. Confidence scaling
# ---------------------------------------------------------------------------


class TestConfidenceScaling:
    def test_confidence_zero_below_threshold(self):
        tracker = _make_tracker()
        trades = _wins(10)  # 10 < 20
        result = tracker.assess_edge(trades)
        assert result.confidence_in_assessment == 0.0

    def test_confidence_half_at_threshold(self):
        """At exactly min_trades (20) confidence should be 0.5."""
        tracker = _make_tracker()
        trades = _wins(12, pnl=1_000.0) + _losses(8, pnl=-400.0)  # 20 trades
        result = tracker.assess_edge(trades)
        assert result.confidence_in_assessment == pytest.approx(0.5, abs=0.01)

    def test_confidence_max_at_50_trades(self):
        tracker = _make_tracker()
        trades = _wins(30, pnl=1_000.0) + _losses(20, pnl=-400.0)  # 50 trades
        result = tracker.assess_edge(trades)
        assert result.confidence_in_assessment == pytest.approx(1.0, abs=0.01)

    def test_confidence_capped_at_1(self):
        tracker = _make_tracker()
        trades = _wins(60, pnl=1_000.0) + _losses(40, pnl=-400.0)  # 100 trades
        result = tracker.assess_edge(trades)
        assert result.confidence_in_assessment == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. Trend detection
# ---------------------------------------------------------------------------


class TestTrendDetection:
    def test_win_rate_improving_trend(self):
        tracker = _make_tracker()
        # First window: 50% WR
        tracker.assess_edge(_wins(25, 500.0) + _losses(25, -450.0))
        # Second window: 65% WR (>5% relative improvement from 50)
        result = tracker.assess_edge(_wins(33, 500.0) + _losses(17, -450.0))
        assert result.win_rate_trend == "IMPROVING"

    def test_win_rate_stable_trend(self):
        tracker = _make_tracker()
        # First window: 50% WR
        tracker.assess_edge(_wins(25, 500.0) + _losses(25, -450.0))
        # Second window: 51% WR (within ±5% band)
        result = tracker.assess_edge(_wins(26, 500.0) + _losses(24, -450.0))
        assert result.win_rate_trend == "STABLE"

    def test_win_rate_declining_trend(self):
        tracker = _make_tracker()
        # First window: 60% WR
        tracker.assess_edge(_wins(30, 1_000.0) + _losses(20, -400.0))
        # Second window: 48% WR (significant drop)
        result = tracker.assess_edge(_wins(24, 1_000.0) + _losses(26, -600.0))
        assert result.win_rate_trend == "DECLINING"

    def test_no_history_trend_is_stable(self):
        """First ever assessment has no prior window to compare — STABLE."""
        tracker = _make_tracker()
        result = tracker.assess_edge(_wins(25, 500.0) + _losses(25, -450.0))
        assert result.win_rate_trend == "STABLE"


# ---------------------------------------------------------------------------
# 8. Action flags
# ---------------------------------------------------------------------------


class TestActionFlags:
    def test_strong_no_action_needed(self):
        tracker = _make_tracker()
        result = tracker.assess_edge(_wins(30, 1_000.0) + _losses(20, -300.0))
        assert result.should_pause is False
        assert result.should_reduce_size is False
        assert result.should_reoptimize is False

    def test_intact_no_action_needed(self):
        tracker = _make_tracker()
        result = tracker.assess_edge(_wins(25, 500.0) + _losses(25, -450.0))
        assert result.should_pause is False
        assert result.should_reduce_size is False
        assert result.should_reoptimize is False

    def test_weakening_reduce_and_reoptimize(self):
        tracker = _make_tracker()
        # Three declining windows
        tracker.assess_edge(_wins(30, 1_000.0) + _losses(20, -400.0))
        tracker.assess_edge(_wins(27, 1_000.0) + _losses(23, -600.0))
        result = tracker.assess_edge(_wins(24, 1_000.0) + _losses(26, -700.0))
        assert result.edge_status == "WEAKENING"
        assert result.should_reduce_size is True
        assert result.should_reoptimize is True
        assert result.should_pause is False

    def test_gone_reoptimize_set(self):
        tracker = _make_tracker()
        result = tracker.assess_edge(_wins(10, 300.0) + _losses(30, -800.0))
        assert result.should_reoptimize is True


# ---------------------------------------------------------------------------
# 9. State resets between tracker instances
# ---------------------------------------------------------------------------


class TestTrackerStateIsolation:
    def test_each_tracker_has_independent_history(self):
        """Separate EdgeTracker instances do not share metric history."""
        t1 = _make_tracker()
        t2 = _make_tracker()

        # Prime t1 with a declining window
        t1.assess_edge(_wins(30, 1_000.0) + _losses(20, -400.0))
        t1.assess_edge(_wins(27, 1_000.0) + _losses(23, -600.0))

        # t2 has fresh history — same 48% WR window should not be WEAKENING
        r_t2 = t2.assess_edge(_wins(24, 1_000.0) + _losses(26, -700.0))
        assert r_t2.edge_status not in ("WEAKENING",)  # no historical context

        # t1 should be WEAKENING for the same window
        r_t1 = t1.assess_edge(_wins(24, 1_000.0) + _losses(26, -700.0))
        assert r_t1.edge_status == "WEAKENING"


# ---------------------------------------------------------------------------
# 10. get_rolling_edge_history with empty DB
# ---------------------------------------------------------------------------


class TestRollingHistory:
    def test_empty_db_returns_empty_list(self):
        tracker = _make_tracker()
        history = tracker.get_rolling_edge_history(days=10)
        assert history == []

    def test_empty_db_default_days_returns_empty_list(self):
        tracker = _make_tracker()
        assert tracker.get_rolling_edge_history() == []

    def test_history_with_db_data(self):
        """Seed DB with trades and verify history returns correct count.

        Trades are inserted 5 days ago so they fall within the 14-day rolling
        window for ALL 5 daily assessments (window_end ranges today-4..today).
        """
        db = _InMemoryDB()
        tracker = EdgeTracker(db)

        today = date.today()
        # 5 days ago is within the 14-day window for every assessment
        from datetime import timedelta
        trade_date = (today - timedelta(days=5)).isoformat()

        for _ in range(30):
            db.execute(
                "INSERT INTO trading_signals (outcome, actual_pnl, closed_at, confidence_level)"
                " VALUES (?, ?, ?, ?)",
                ("WIN", 1_000.0, f"{trade_date} 10:30:00", "HIGH"),
            )

        history = tracker.get_rolling_edge_history(days=5)

        assert len(history) == 5
        # Every daily window (ending today-4 through today) covers the trades
        for assessment in history:
            assert assessment.trade_count == 30
            assert assessment.win_rate == pytest.approx(100.0)
            assert assessment.edge_status == "STRONG"

    def test_history_returns_oldest_first(self):
        """Oldest assessment should have the earliest timestamp."""
        db = _InMemoryDB()
        tracker = EdgeTracker(db)

        today = date.today()
        db.execute(
            "INSERT INTO trading_signals (outcome, actual_pnl, closed_at, confidence_level)"
            " VALUES (?, ?, ?, ?)",
            ("WIN", 500.0, f"{today.isoformat()} 10:00:00", "HIGH"),
        )

        history = tracker.get_rolling_edge_history(days=3)
        # history is oldest → newest
        assert len(history) == 3
        timestamps = [h.timestamp for h in history]
        assert timestamps == sorted(timestamps)
