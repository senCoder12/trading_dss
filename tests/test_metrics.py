import math
import statistics
import uuid
from datetime import datetime, timedelta

import pytest

from src.backtest.metrics import BacktestMetrics, ComparisonReport, MetricsCalculator
from src.backtest.trade_simulator import ClosedTrade, EquityPoint

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_trade(
    outcome: str,
    net_pnl: float,
    net_pnl_pct: float = None,
    confidence: str = "MEDIUM",
    trade_type: str = "BUY_CALL",
    duration_bars: int = 3,
    exit_reason: str = None,
    regime: str = "TRENDING",
    entry_time: datetime = None,
) -> ClosedTrade:
    if entry_time is None:
        entry_time = datetime(2024, 7, 1, 10, 0)
    if net_pnl_pct is None:
        net_pnl_pct = (net_pnl / 100_000) * 100
    if exit_reason is None:
        exit_reason = "TARGET_HIT" if outcome == "WIN" else "STOP_LOSS_HIT"
    return ClosedTrade(
        trade_id=str(uuid.uuid4()),
        index_id="NIFTY50",
        signal_id=str(uuid.uuid4()),
        trade_type=trade_type,
        signal_entry_price=20_000.0,
        actual_entry_price=20_000.0,
        entry_timestamp=entry_time,
        entry_bar={"regime": regime},
        original_stop_loss=19_900.0,
        original_target=20_200.0,
        actual_exit_price=20_200.0 if outcome == "WIN" else 19_900.0,
        exit_timestamp=entry_time + timedelta(minutes=duration_bars * 5),
        exit_reason=exit_reason,
        lots=1,
        lot_size=50,
        quantity=50,
        confidence_level=confidence,
        gross_pnl_points=100.0,
        gross_pnl=net_pnl + 50.0,
        total_costs=50.0,
        net_pnl=net_pnl,
        net_pnl_pct=net_pnl_pct,
        duration_bars=duration_bars,
        duration_minutes=duration_bars * 5,
        max_favorable_excursion=100.0,
        max_adverse_excursion=50.0,
        outcome=outcome,
    )


def make_equity(capitals: list[float], base: datetime = None) -> list[EquityPoint]:
    if base is None:
        base = datetime(2024, 1, 1)
    return [
        EquityPoint(
            timestamp=base + timedelta(days=i),
            capital=c,
            cash=c,
            unrealized=0.0,
            drawdown_pct=0.0,
            open_positions=0,
        )
        for i, c in enumerate(capitals)
    ]


# ─────────────────────────────────────────────
# 1. Known-trade arithmetic
# ─────────────────────────────────────────────

class TestKnownTradeStats:
    """10 wins of ₹500, 5 losses of ₹300 → exact verified values."""

    def setup_method(self):
        wins = [make_trade("WIN", 500.0) for _ in range(10)]
        losses = [make_trade("LOSS", -300.0) for _ in range(5)]
        self.m = MetricsCalculator.calculate_all(wins + losses, [], 100_000.0)

    def test_trade_counts(self):
        assert self.m.total_trades == 15
        assert self.m.winning_trades == 10
        assert self.m.losing_trades == 5
        assert self.m.breakeven_trades == 0

    def test_win_rate(self):
        assert abs(self.m.win_rate - 66.67) < 0.01

    def test_profit_factor(self):
        # gross_wins=5000, gross_losses=1500 → 5000/1500 ≈ 3.3333
        assert abs(self.m.profit_factor - 3.3333) < 0.001

    def test_payoff_ratio(self):
        # avg_win=500, avg_loss=300 → 500/300 ≈ 1.6667
        assert abs(self.m.payoff_ratio - 1.6667) < 0.001

    def test_expected_value(self):
        # EV = 0.6667 * 500 - 0.3333 * 300 = 333.33 - 100 = 233.33
        assert abs(self.m.expected_value_per_trade - 233.33) < 0.1

    def test_avg_win_and_loss(self):
        assert abs(self.m.avg_win_amount - 500.0) < 0.01
        assert abs(self.m.avg_loss_amount - 300.0) < 0.01

    def test_largest_win_loss(self):
        assert abs(self.m.largest_win - 500.0) < 0.01
        assert abs(self.m.largest_loss - 300.0) < 0.01

    def test_has_edge(self):
        assert self.m.has_edge is True
        assert self.m.is_profitable is False  # no equity curve, return_amount = 0


# ─────────────────────────────────────────────
# 2. Sharpe ratio
# ─────────────────────────────────────────────

class TestSharpeRatio:
    """Verify sign and rough magnitude with deterministic daily returns."""

    def _metrics_from_returns(self, daily_returns: list[float], rf: float = 0.065):
        """Build equity curve from known daily returns and compute metrics."""
        cap = 100_000.0
        points = [EquityPoint(datetime(2024, 1, 1), cap, cap, 0.0, 0.0, 0)]
        for i, r in enumerate(daily_returns):
            cap = cap * (1 + r)
            points.append(
                EquityPoint(datetime(2024, 1, 1) + timedelta(days=i + 1), cap, cap, 0.0, 0.0, 0)
            )
        dummy = make_trade("WIN", 0)
        return MetricsCalculator.calculate_all([dummy], points, 100_000.0, rf)

    def test_sharpe_positive_for_strong_uptrend(self):
        # All returns 0.3% per day: annualized ~75%, but std=0 → sharpe=0
        # Use alternating to get nonzero std with positive mean
        returns = [0.003 if i % 2 == 0 else 0.001 for i in range(20)]
        m = self._metrics_from_returns(returns)
        assert m.sharpe_ratio > 0

    def test_sharpe_negative_for_losing_strategy(self):
        returns = [-0.003 if i % 2 == 0 else -0.001 for i in range(20)]
        m = self._metrics_from_returns(returns)
        assert m.sharpe_ratio < 0

    def test_sharpe_zero_when_flat(self):
        # Flat equity → std=0 → sharpe=0
        returns = [0.0] * 10
        m = self._metrics_from_returns(returns)
        assert m.sharpe_ratio == 0.0

    def test_sharpe_known_value(self):
        # 20 alternating returns: +0.002, 0.000 → mean=0.001, std=stdev([0.002,0,...])
        returns = [0.002 if i % 2 == 0 else 0.0 for i in range(20)]
        avg = sum(returns) / len(returns)  # = 0.001
        std = statistics.stdev(returns)

        ann_ret = avg * 252
        ann_std = std * math.sqrt(252)
        expected_sharpe = (ann_ret - 0.065) / ann_std

        m = self._metrics_from_returns(returns)
        assert abs(m.sharpe_ratio - expected_sharpe) < 0.01

    def test_single_equity_point_gives_zero_sharpe(self):
        # Only 1 point → no daily returns → sharpe=0
        points = [EquityPoint(datetime(2024, 1, 1), 100_000.0, 100_000.0, 0.0, 0.0, 0)]
        dummy = make_trade("WIN", 0)
        m = MetricsCalculator.calculate_all([dummy], points, 100_000.0)
        assert m.sharpe_ratio == 0.0


# ─────────────────────────────────────────────
# 3. Max drawdown
# ─────────────────────────────────────────────

class TestMaxDrawdown:
    def test_15_percent_drawdown(self):
        # peak=100 at bar 1, trough=85 at bar 3 → 15% drawdown
        points = make_equity([100.0, 100.0, 90.0, 85.0, 95.0])
        dummy = make_trade("WIN", 0)
        m = MetricsCalculator.calculate_all([dummy], points, 100.0)

        assert abs(m.max_drawdown_pct - 15.0) < 0.01
        assert abs(m.max_drawdown_amount - 15.0) < 0.01

    def test_no_drawdown(self):
        points = make_equity([100.0, 105.0, 110.0, 115.0])
        dummy = make_trade("WIN", 0)
        m = MetricsCalculator.calculate_all([dummy], points, 100.0)

        assert m.max_drawdown_pct == 0.0
        assert m.max_drawdown_amount == 0.0

    def test_drawdown_start_timestamp_recorded(self):
        base = datetime(2024, 6, 1)
        points = make_equity([100.0, 100.0, 90.0, 85.0, 95.0], base)
        dummy = make_trade("WIN", 0)
        m = MetricsCalculator.calculate_all([dummy], points, 100.0)

        # Drawdown starts on the first bar below the peak (bar index 2)
        assert m.max_drawdown_start == base + timedelta(days=2)

    def test_second_deeper_drawdown_wins(self):
        # First small dip then bigger dip
        points = make_equity([100.0, 98.0, 99.0, 100.0, 100.0, 80.0, 85.0])
        dummy = make_trade("WIN", 0)
        m = MetricsCalculator.calculate_all([dummy], points, 100.0)

        assert abs(m.max_drawdown_pct - 20.0) < 0.01

    def test_calmar_ratio(self):
        # Build equity with known annualized return and drawdown
        # Equity: 100k → 110k over 252 points (1 year), with a 10% drawdown midway
        capitals = [100_000.0 + i * (10_000 / 252) for i in range(253)]
        # Inject a dip: lower a stretch by 10k
        for j in range(120, 150):
            capitals[j] -= 10_000.0
        points = make_equity(capitals, datetime(2024, 1, 1))
        dummy = make_trade("WIN", 0)
        m = MetricsCalculator.calculate_all([dummy], points, 100_000.0)

        assert m.calmar_ratio > 0
        assert m.max_drawdown_pct > 0


# ─────────────────────────────────────────────
# 4. Monthly returns
# ─────────────────────────────────────────────

class TestMonthlyReturns:
    def test_two_months(self):
        points = [
            EquityPoint(datetime(2024, 1, 1), 100.0, 100.0, 0, 0, 0),
            EquityPoint(datetime(2024, 1, 31), 110.0, 110.0, 0, 0, 0),
            EquityPoint(datetime(2024, 2, 1), 110.0, 110.0, 0, 0, 0),
            EquityPoint(datetime(2024, 2, 28), 104.5, 104.5, 0, 0, 0),
        ]
        dummy = make_trade("WIN", 0)
        m = MetricsCalculator.calculate_all([dummy], points, 100.0)

        assert len(m.monthly_returns) == 2
        assert m.monthly_returns[0]["month"] == "2024-01"
        assert abs(m.monthly_returns[0]["return_pct"] - 10.0) < 0.01
        assert m.monthly_returns[1]["month"] == "2024-02"
        assert abs(m.monthly_returns[1]["return_pct"] - (-5.0)) < 0.01

    def test_positive_and_negative_months(self):
        points = [
            EquityPoint(datetime(2024, 1, 1), 100.0, 100.0, 0, 0, 0),
            EquityPoint(datetime(2024, 1, 31), 110.0, 110.0, 0, 0, 0),
            EquityPoint(datetime(2024, 2, 1), 110.0, 110.0, 0, 0, 0),
            EquityPoint(datetime(2024, 2, 28), 104.5, 104.5, 0, 0, 0),
        ]
        dummy = make_trade("WIN", 0)
        m = MetricsCalculator.calculate_all([dummy], points, 100.0)

        assert m.positive_months == 1
        assert m.negative_months == 1
        assert abs(m.monthly_win_rate - 50.0) < 0.01

    def test_best_and_worst_month(self):
        points = [
            EquityPoint(datetime(2024, 1, 1), 100.0, 100.0, 0, 0, 0),
            EquityPoint(datetime(2024, 1, 31), 105.0, 105.0, 0, 0, 0),
            EquityPoint(datetime(2024, 2, 1), 105.0, 105.0, 0, 0, 0),
            EquityPoint(datetime(2024, 2, 29), 98.0, 98.0, 0, 0, 0),
            EquityPoint(datetime(2024, 3, 1), 98.0, 98.0, 0, 0, 0),
            EquityPoint(datetime(2024, 3, 31), 112.0, 112.0, 0, 0, 0),
        ]
        dummy = make_trade("WIN", 0)
        m = MetricsCalculator.calculate_all([dummy], points, 100.0)

        assert m.best_month_pct > m.worst_month_pct
        assert m.best_month_pct > 0
        assert m.worst_month_pct < 0

    def test_no_equity_curve_gives_empty_months(self):
        m = MetricsCalculator.calculate_all([make_trade("WIN", 100.0)], [], 100_000.0)
        assert m.monthly_returns == []
        assert m.positive_months == 0
        assert m.negative_months == 0


# ─────────────────────────────────────────────
# 5. Strategy grading (each boundary)
# ─────────────────────────────────────────────

class TestStrategyGrading:
    """Tests _grade_strategy directly so we can hit every boundary cleanly."""

    def test_grade_A(self):
        # win>55, pf>1.5, sharpe>1.5, dd<15
        assert MetricsCalculator._grade_strategy(56.0, 1.6, 1.6, 10.0) == "A"

    def test_grade_A_exact_boundary_excluded(self):
        # win=55 (not >55) → should not be A
        result = MetricsCalculator._grade_strategy(55.0, 2.0, 2.0, 5.0)
        assert result != "A"

    def test_grade_B(self):
        # win>50, pf>1.2, sharpe>1.0, dd<20; but not A thresholds
        assert MetricsCalculator._grade_strategy(51.0, 1.3, 1.1, 18.0) == "B"

    def test_grade_B_fails_when_dd_too_high(self):
        # Would be B but dd >= 20 → not B
        result = MetricsCalculator._grade_strategy(51.0, 1.3, 1.1, 20.0)
        assert result not in ("A", "B")

    def test_grade_C(self):
        # win>45, pf>1.0, sharpe>0.5; but not A/B thresholds
        assert MetricsCalculator._grade_strategy(46.0, 1.05, 0.6, 25.0) == "C"

    def test_grade_D(self):
        # pf>0.8 but not meeting C criteria
        assert MetricsCalculator._grade_strategy(40.0, 0.9, 0.3, 30.0) == "D"

    def test_grade_F(self):
        # pf <= 0.8
        assert MetricsCalculator._grade_strategy(30.0, 0.5, -0.5, 40.0) == "F"

    def test_grade_F_zero_pf(self):
        assert MetricsCalculator._grade_strategy(0.0, 0.0, -1.0, 50.0) == "F"

    def test_grade_via_calculate_all(self):
        # 20 wins, 2 losses → win_rate=90.9%, pf very high
        # Need Sharpe > 1.5 and dd < 15% → provide strong equity curve
        wins = [make_trade("WIN", 500.0) for _ in range(20)]
        losses = [make_trade("LOSS", -200.0) for _ in range(2)]
        # Equity curve: steadily rising (low std, low DD)
        caps = [100_000.0 + i * 200 for i in range(50)]
        points = make_equity(caps)
        m = MetricsCalculator.calculate_all(wins + losses, points, 100_000.0)
        assert m.strategy_grade in ("A", "B", "C")  # definitely not F


# ─────────────────────────────────────────────
# 6. Zero trades
# ─────────────────────────────────────────────

class TestZeroTrades:
    def setup_method(self):
        self.m = MetricsCalculator.calculate_all([], [], 100_000.0)

    def test_grade_is_F(self):
        assert self.m.strategy_grade == "F"

    def test_assessment_message(self):
        assert "No trades generated" in self.m.assessment

    def test_all_numeric_fields_zero(self):
        assert self.m.total_trades == 0
        assert self.m.win_rate == 0.0
        assert self.m.profit_factor == 0.0
        assert self.m.sharpe_ratio == 0.0
        assert self.m.max_drawdown_pct == 0.0

    def test_is_not_profitable_no_edge(self):
        assert self.m.is_profitable is False
        assert self.m.has_edge is False


# ─────────────────────────────────────────────
# 7. One trade
# ─────────────────────────────────────────────

class TestOneTrade:
    def test_single_win(self):
        m = MetricsCalculator.calculate_all([make_trade("WIN", 500.0)], [], 100_000.0)
        assert m.total_trades == 1
        assert m.winning_trades == 1
        assert m.losing_trades == 0
        assert m.win_rate == 100.0
        assert m.avg_win_amount == 500.0
        assert m.avg_loss_amount == 0.0
        assert m.profit_factor == 999.0   # all wins, capped at 999
        assert m.largest_win == 500.0
        assert m.largest_loss == 0.0

    def test_single_loss(self):
        m = MetricsCalculator.calculate_all([make_trade("LOSS", -300.0)], [], 100_000.0)
        assert m.total_trades == 1
        assert m.losing_trades == 1
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0
        assert m.largest_loss == 300.0

    def test_single_win_consecutives(self):
        m = MetricsCalculator.calculate_all([make_trade("WIN", 500.0)], [], 100_000.0)
        assert m.max_consecutive_wins == 1
        assert m.max_consecutive_losses == 0
        assert m.current_streak == 1


# ─────────────────────────────────────────────
# 8. All wins / all losses edge cases
# ─────────────────────────────────────────────

class TestEdgeCases:
    def test_all_wins_profit_factor_capped(self):
        trades = [make_trade("WIN", 500.0) for _ in range(5)]
        m = MetricsCalculator.calculate_all(trades, [], 100_000.0)
        assert m.profit_factor == 999.0

    def test_all_losses_expected_value_negative(self):
        trades = [make_trade("LOSS", -300.0) for _ in range(5)]
        m = MetricsCalculator.calculate_all(trades, [], 100_000.0)
        assert m.expected_value_per_trade < 0
        assert m.profit_factor == 0.0
        assert m.has_edge is False

    def test_sortino_999_when_no_downside(self):
        # Equity curve with no negative daily returns
        caps = [100_000.0 + i * 100 for i in range(10)]
        points = make_equity(caps)
        dummy = make_trade("WIN", 0)
        m = MetricsCalculator.calculate_all([dummy], points, 100_000.0)
        assert m.sortino_ratio == 999.0


# ─────────────────────────────────────────────
# 9. Consecutive win/loss streaks
# ─────────────────────────────────────────────

class TestConsecutiveStreaks:
    def test_max_consecutive_wins(self):
        trades = (
            [make_trade("WIN", 100.0)] * 3
            + [make_trade("LOSS", -100.0)] * 1
            + [make_trade("WIN", 100.0)] * 5
        )
        m = MetricsCalculator.calculate_all(trades, [], 100_000.0)
        assert m.max_consecutive_wins == 5
        assert m.max_consecutive_losses == 1

    def test_max_consecutive_losses(self):
        trades = (
            [make_trade("WIN", 100.0)] * 2
            + [make_trade("LOSS", -100.0)] * 4
            + [make_trade("WIN", 100.0)] * 1
        )
        m = MetricsCalculator.calculate_all(trades, [], 100_000.0)
        assert m.max_consecutive_losses == 4

    def test_current_streak_win(self):
        trades = (
            [make_trade("LOSS", -100.0)] * 2
            + [make_trade("WIN", 100.0)] * 3
        )
        m = MetricsCalculator.calculate_all(trades, [], 100_000.0)
        assert m.current_streak == 3   # positive = win streak

    def test_current_streak_loss(self):
        trades = (
            [make_trade("WIN", 100.0)] * 2
            + [make_trade("LOSS", -100.0)] * 4
        )
        m = MetricsCalculator.calculate_all(trades, [], 100_000.0)
        assert m.current_streak == -4  # negative = loss streak


# ─────────────────────────────────────────────
# 10. Confidence level analysis
# ─────────────────────────────────────────────

class TestConfidenceAnalysis:
    def setup_method(self):
        high = [make_trade("WIN", 600.0, confidence="HIGH") for _ in range(6)] + \
               [make_trade("LOSS", -200.0, confidence="HIGH") for _ in range(2)]
        med  = [make_trade("WIN", 300.0, confidence="MEDIUM") for _ in range(4)] + \
               [make_trade("LOSS", -300.0, confidence="MEDIUM") for _ in range(4)]
        low  = [make_trade("LOSS", -400.0, confidence="LOW") for _ in range(5)]
        self.m = MetricsCalculator.calculate_all(high + med + low, [], 100_000.0)

    def test_counts(self):
        assert self.m.high_confidence_trades == 8
        assert self.m.medium_confidence_trades == 8
        assert self.m.low_confidence_trades == 5

    def test_win_rates(self):
        assert abs(self.m.high_confidence_win_rate - 75.0) < 0.01
        assert abs(self.m.medium_confidence_win_rate - 50.0) < 0.01
        assert self.m.low_confidence_win_rate == 0.0

    def test_avg_pnl_ordering(self):
        assert self.m.high_confidence_avg_pnl > self.m.medium_confidence_avg_pnl
        assert self.m.low_confidence_avg_pnl < 0


# ─────────────────────────────────────────────
# 11. Exit analysis
# ─────────────────────────────────────────────

class TestExitAnalysis:
    def setup_method(self):
        trades = (
            [make_trade("WIN", 500.0, exit_reason="TARGET_HIT")] * 4 +
            [make_trade("LOSS", -300.0, exit_reason="STOP_LOSS_HIT")] * 3 +
            [make_trade("WIN", 100.0, exit_reason="TRAILING_SL_HIT")] * 2 +
            [make_trade("LOSS", -50.0, exit_reason="FORCED_EOD")] * 1
        )
        self.m = MetricsCalculator.calculate_all(trades, [], 100_000.0)

    def test_counts(self):
        assert self.m.target_hit_count == 4
        assert self.m.sl_hit_count == 3
        assert self.m.trailing_sl_count == 2
        assert self.m.forced_eod_count == 1

    def test_percentages(self):
        assert abs(self.m.target_hit_pct - 40.0) < 0.01
        assert abs(self.m.sl_hit_pct - 30.0) < 0.01
        assert abs(self.m.trailing_sl_pct - 20.0) < 0.01

    def test_forced_eod_avg_pnl(self):
        assert abs(self.m.forced_eod_avg_pnl - (-50.0)) < 0.01


# ─────────────────────────────────────────────
# 12. Signal type (CALL / PUT)
# ─────────────────────────────────────────────

class TestSignalTypeAnalysis:
    def setup_method(self):
        calls = (
            [make_trade("WIN", 500.0, trade_type="BUY_CALL")] * 6 +
            [make_trade("LOSS", -200.0, trade_type="BUY_CALL")] * 4
        )
        puts = (
            [make_trade("WIN", 300.0, trade_type="BUY_PUT")] * 3 +
            [make_trade("LOSS", -300.0, trade_type="BUY_PUT")] * 7
        )
        self.m = MetricsCalculator.calculate_all(calls + puts, [], 100_000.0)

    def test_call_counts_and_win_rate(self):
        assert self.m.call_trades == 10
        assert abs(self.m.call_win_rate - 60.0) < 0.01

    def test_put_counts_and_win_rate(self):
        assert self.m.put_trades == 10
        assert abs(self.m.put_win_rate - 30.0) < 0.01

    def test_call_pnl_higher_than_put(self):
        assert self.m.call_total_pnl > self.m.put_total_pnl


# ─────────────────────────────────────────────
# 13. Regime analysis
# ─────────────────────────────────────────────

class TestRegimeAnalysis:
    def setup_method(self):
        trending = [make_trade("WIN", 500.0, regime="TRENDING")] * 5 + \
                   [make_trade("LOSS", -100.0, regime="TRENDING")] * 1
        choppy = [make_trade("WIN", 100.0, regime="CHOPPY")] * 2 + \
                 [make_trade("LOSS", -400.0, regime="CHOPPY")] * 4
        self.m = MetricsCalculator.calculate_all(trending + choppy, [], 100_000.0)

    def test_regime_keys_present(self):
        assert "TRENDING" in self.m.trades_by_regime
        assert "CHOPPY" in self.m.trades_by_regime

    def test_best_and_worst_regime(self):
        assert self.m.best_regime == "TRENDING"
        assert self.m.worst_regime == "CHOPPY"

    def test_regime_stats_structure(self):
        for stats in self.m.trades_by_regime.values():
            assert "count" in stats
            assert "win_rate" in stats
            assert "avg_pnl" in stats


# ─────────────────────────────────────────────
# 14. Timing analysis (day of week)
# ─────────────────────────────────────────────

class TestTimingAnalysis:
    def test_day_of_week_populated(self):
        # Monday=2024-01-01, Tuesday=2024-01-02, ...
        mon = make_trade("WIN", 500.0, entry_time=datetime(2024, 1, 1, 10, 0))
        tue = make_trade("LOSS", -200.0, entry_time=datetime(2024, 1, 2, 10, 0))
        wed = make_trade("WIN", 300.0, entry_time=datetime(2024, 1, 3, 10, 0))
        m = MetricsCalculator.calculate_all([mon, tue, wed], [], 100_000.0)

        assert len(m.trades_by_day_of_week) == 3
        assert m.best_day_of_week != ""
        assert m.worst_day_of_week != ""

    def test_hour_breakdown(self):
        t1 = make_trade("WIN", 400.0, entry_time=datetime(2024, 1, 1, 9, 30))
        t2 = make_trade("WIN", 200.0, entry_time=datetime(2024, 1, 1, 10, 15))
        t3 = make_trade("LOSS", -300.0, entry_time=datetime(2024, 1, 1, 14, 45))
        m = MetricsCalculator.calculate_all([t1, t2, t3], [], 100_000.0)

        assert 9 in m.trades_by_hour
        assert 10 in m.trades_by_hour
        assert 14 in m.trades_by_hour


# ─────────────────────────────────────────────
# 15. Comparison of two backtest results
# ─────────────────────────────────────────────

class TestCompareBacktests:
    def test_ranking_respects_composite_score(self):
        # Run 1: mediocre
        m1 = MetricsCalculator.calculate_all(
            [make_trade("WIN", 100.0)] * 5 + [make_trade("LOSS", -200.0)] * 5,
            [],
            100_000.0,
        )
        # Run 2: much better — higher win rate
        m2 = MetricsCalculator.calculate_all(
            [make_trade("WIN", 500.0)] * 9 + [make_trade("LOSS", -100.0)] * 1,
            [],
            100_000.0,
        )
        # Force sharpe so the score ordering is unambiguous
        m1.sharpe_ratio = 0.5
        m2.sharpe_ratio = 2.0

        report = MetricsCalculator.compare_backtests([m1, m2], ["Run 1", "Run 2"])

        assert report.best_overall == "Run 2"
        assert report.ranking[0][0] == "Run 2"
        assert report.ranking[1][0] == "Run 1"

    def test_comparison_report_structure(self):
        m1 = MetricsCalculator.calculate_all([make_trade("WIN", 100.0)], [], 100_000.0)
        m2 = MetricsCalculator.calculate_all([make_trade("WIN", 200.0)], [], 100_000.0)

        report = MetricsCalculator.compare_backtests([m1, m2], ["A", "B"])

        assert report.headers == ["A", "B"]
        assert len(report.ranking) == 2
        for row in report.metrics_table:
            assert "metric_name" in row
            assert "values" in row
            assert len(row["values"]) == 2

    def test_auto_labels_when_none_given(self):
        m1 = MetricsCalculator.calculate_all([make_trade("WIN", 100.0)], [], 100_000.0)
        m2 = MetricsCalculator.calculate_all([make_trade("WIN", 200.0)], [], 100_000.0)

        report = MetricsCalculator.compare_backtests([m1, m2])

        assert report.headers == ["Run 1", "Run 2"]

    def test_single_result_comparison(self):
        m = MetricsCalculator.calculate_all([make_trade("WIN", 100.0)], [], 100_000.0)
        report = MetricsCalculator.compare_backtests([m], ["Only"])

        assert report.best_overall == "Only"
        assert len(report.ranking) == 1


# ─────────────────────────────────────────────
# 16. Assessment text generation
# ─────────────────────────────────────────────

class TestAssessmentText:
    def setup_method(self):
        wins = [make_trade("WIN", 500.0, confidence="HIGH") for _ in range(8)]
        losses = [make_trade("LOSS", -200.0, confidence="LOW") for _ in range(4)]
        self.m = MetricsCalculator.calculate_all(wins + losses, [], 100_000.0)
        self.text = MetricsCalculator.generate_assessment(self.m)

    def test_contains_grade_section(self):
        assert "GRADE:" in self.text

    def test_contains_returns_section(self):
        assert "RETURNS:" in self.text

    def test_contains_trade_quality_section(self):
        assert "TRADE QUALITY:" in self.text

    def test_contains_risk_section(self):
        assert "RISK:" in self.text

    def test_contains_confidence_section(self):
        assert "CONFIDENCE BREAKDOWN:" in self.text

    def test_contains_signal_type_section(self):
        assert "SIGNAL TYPE:" in self.text

    def test_contains_recommendations(self):
        assert "RECOMMENDATIONS:" in self.text

    def test_contains_separator(self):
        assert "━" in self.text

    def test_zero_trade_assessment(self):
        assert MetricsCalculator.generate_assessment(MetricsCalculator._zero_metrics(100_000.0)) == "No trades generated"

    def test_numeric_values_appear_in_text(self):
        # Trade count should appear
        assert str(self.m.total_trades) in self.text
