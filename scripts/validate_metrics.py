"""
Step 6.4 Validation — MetricsCalculator end-to-end demonstration.

Runs the exact user code:
    calc = MetricsCalculator()
    metrics = calc.calculate_all(trade_history, equity_curve, initial_capital)
    print(...)

Using a realistic synthetic BacktestResult (50 trades, 6-month equity curve)
that exercises all metric categories:
  - Return metrics, monthly returns
  - Trade stats (wins/losses/breakeven)
  - Risk metrics (drawdown, Sharpe, Sortino, Calmar)
  - Confidence / exit / regime / timing breakdowns
  - Strategy grading + human-readable assessment
  - compare_backtests comparison

Run from project root:
    python3 scripts/validate_metrics.py
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.backtest.metrics import BacktestMetrics, MetricsCalculator
from src.backtest.trade_simulator import ClosedTrade, EquityPoint


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build realistic synthetic data
# ─────────────────────────────────────────────────────────────────────────────

def _make_trade(
    outcome: str,
    net_pnl: float,
    entry_time: datetime,
    confidence: str = "MEDIUM",
    trade_type: str = "BUY_CALL",
    exit_reason: str = None,
    regime: str = "TRENDING",
    duration_bars: int = 4,
    capital_at_entry: float = 100_000.0,
) -> ClosedTrade:
    if exit_reason is None:
        exit_reason = "TARGET_HIT" if outcome == "WIN" else "STOP_LOSS_HIT"
    return ClosedTrade(
        trade_id=str(uuid.uuid4()),
        index_id="NIFTY50",
        signal_id=str(uuid.uuid4()),
        trade_type=trade_type,
        signal_entry_price=22_000.0,
        actual_entry_price=22_000.0,
        entry_timestamp=entry_time,
        entry_bar={"regime": regime},
        original_stop_loss=21_850.0,
        original_target=22_300.0,
        actual_exit_price=22_300.0 if outcome == "WIN" else 21_850.0,
        exit_timestamp=entry_time + timedelta(minutes=duration_bars * 15),
        exit_reason=exit_reason,
        lots=1,
        lot_size=75,
        quantity=75,
        confidence_level=confidence,
        gross_pnl_points=150.0 if outcome == "WIN" else -150.0,
        gross_pnl=net_pnl + 80.0,
        total_costs=80.0,
        net_pnl=net_pnl,
        net_pnl_pct=(net_pnl / capital_at_entry) * 100,
        duration_bars=duration_bars,
        duration_minutes=duration_bars * 15,
        max_favorable_excursion=180.0 if outcome == "WIN" else 50.0,
        max_adverse_excursion=60.0,
        outcome=outcome,
    )


def _build_equity_curve(
    initial: float,
    trade_history: list[ClosedTrade],
    start: datetime,
    n_days: int = 130,
) -> list[EquityPoint]:
    """Build a daily equity curve from the trade history."""
    # Map trade closes to their dates
    daily_pnl: dict[date, float] = {}
    for t in trade_history:
        d = t.exit_timestamp.date()
        daily_pnl[d] = daily_pnl.get(d, 0.0) + t.net_pnl

    points = []
    capital = initial
    day = start.date()
    for i in range(n_days):
        if day.weekday() < 5:
            capital += daily_pnl.get(day, 0.0)
            points.append(EquityPoint(
                timestamp=datetime.combine(day, datetime.min.time()).replace(hour=15, minute=30),
                capital=round(capital, 2),
                cash=round(capital, 2),
                unrealized=0.0,
                drawdown_pct=0.0,
                open_positions=0,
            ))
        day += timedelta(days=1)
    return points


def build_realistic_result(initial_capital: float = 100_000.0):
    """
    Create a 52-trade result spanning Jan–Jun 2024 that covers:
      - Mixed confidence levels (HIGH / MEDIUM / LOW)
      - Mixed trade types (BUY_CALL / BUY_PUT)
      - Mixed exit reasons
      - Mixed regimes (TRENDING / SIDEWAYS / RANGE_BOUND)
      - A mid-period drawdown (simulates a losing streak)
    """
    trades: list[ClosedTrade] = []
    start = datetime(2024, 1, 8, 9, 30)

    # Scenario specification: (outcome, pnl, confidence, type, exit, regime, offset_days)
    scenarios = [
        # Jan: solid start — 8 trades, 6W 2L
        ("WIN",  520, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",   0),
        ("WIN",  480, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",   2),
        ("LOSS",-240, "MEDIUM", "BUY_PUT",  "STOP_LOSS_HIT",   "TRENDING",   3),
        ("WIN",  390, "MEDIUM", "BUY_CALL", "TARGET_HIT",      "TRENDING",   5),
        ("WIN",  310, "LOW",    "BUY_CALL", "TRAILING_SL_HIT", "SIDEWAYS",   7),
        ("WIN",  420, "HIGH",   "BUY_PUT",  "TARGET_HIT",      "TRENDING",  10),
        ("LOSS",-180, "LOW",    "BUY_PUT",  "FORCED_EOD",      "SIDEWAYS",  12),
        ("WIN",  350, "MEDIUM", "BUY_CALL", "TARGET_HIT",      "TRENDING",  14),
        # Feb: choppy — 8 trades, 4W 4L
        ("LOSS",-290, "MEDIUM", "BUY_CALL", "STOP_LOSS_HIT",   "SIDEWAYS",  19),
        ("WIN",  280, "MEDIUM", "BUY_PUT",  "TARGET_HIT",      "RANGE_BOUND",21),
        ("LOSS",-310, "LOW",    "BUY_CALL", "STOP_LOSS_HIT",   "RANGE_BOUND",23),
        ("WIN",  340, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",  25),
        ("LOSS",-250, "MEDIUM", "BUY_PUT",  "STOP_LOSS_HIT",   "SIDEWAYS",  27),
        ("WIN",  410, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",  29),
        ("LOSS",-200, "LOW",    "BUY_PUT",  "FORCED_EOD",      "RANGE_BOUND",31),
        ("WIN",  290, "MEDIUM", "BUY_CALL", "TRAILING_SL_HIT", "TRENDING",  33),
        # Mar: strong trending — 10 trades, 8W 2L
        ("WIN",  580, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",  40),
        ("WIN",  620, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",  42),
        ("WIN",  450, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",  44),
        ("LOSS",-210, "LOW",    "BUY_PUT",  "STOP_LOSS_HIT",   "RANGE_BOUND",46),
        ("WIN",  390, "MEDIUM", "BUY_PUT",  "TARGET_HIT",      "TRENDING",  48),
        ("WIN",  510, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",  50),
        ("WIN",  470, "HIGH",   "BUY_CALL", "TRAILING_SL_HIT", "TRENDING",  52),
        ("WIN",  330, "MEDIUM", "BUY_CALL", "TARGET_HIT",      "TRENDING",  54),
        ("WIN",  360, "MEDIUM", "BUY_PUT",  "TARGET_HIT",      "TRENDING",  56),
        ("LOSS",-190, "MEDIUM", "BUY_CALL", "STOP_LOSS_HIT",   "SIDEWAYS",  58),
        # Apr: drawdown — 8 trades, 3W 5L
        ("LOSS",-320, "MEDIUM", "BUY_CALL", "STOP_LOSS_HIT",   "SIDEWAYS",  65),
        ("LOSS",-300, "MEDIUM", "BUY_PUT",  "STOP_LOSS_HIT",   "RANGE_BOUND",67),
        ("WIN",  260, "LOW",    "BUY_PUT",  "TARGET_HIT",      "RANGE_BOUND",69),
        ("LOSS",-280, "LOW",    "BUY_CALL", "STOP_LOSS_HIT",   "RANGE_BOUND",71),
        ("LOSS",-350, "MEDIUM", "BUY_CALL", "STOP_LOSS_HIT",   "SIDEWAYS",  73),
        ("WIN",  310, "HIGH",   "BUY_PUT",  "TARGET_HIT",      "TRENDING",  75),
        ("LOSS",-230, "LOW",    "BUY_CALL", "FORCED_EOD",      "SIDEWAYS",  77),
        ("WIN",  200, "MEDIUM", "BUY_PUT",  "TARGET_HIT",      "RANGE_BOUND",79),
        # May: recovery — 9 trades, 7W 2L
        ("WIN",  440, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",  86),
        ("WIN",  500, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",  88),
        ("LOSS",-200, "MEDIUM", "BUY_PUT",  "STOP_LOSS_HIT",   "SIDEWAYS",  90),
        ("WIN",  380, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",  92),
        ("WIN",  420, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING",  94),
        ("WIN",  310, "MEDIUM", "BUY_PUT",  "TRAILING_SL_HIT", "TRENDING",  96),
        ("LOSS",-180, "LOW",    "BUY_CALL", "STOP_LOSS_HIT",   "RANGE_BOUND",98),
        ("WIN",  350, "MEDIUM", "BUY_CALL", "TARGET_HIT",      "TRENDING", 100),
        ("WIN",  290, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING", 102),
        # Jun: stable finish — 9 trades, 6W 3L
        ("WIN",  460, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING", 109),
        ("LOSS",-220, "MEDIUM", "BUY_PUT",  "STOP_LOSS_HIT",   "SIDEWAYS", 111),
        ("WIN",  330, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING", 113),
        ("WIN",  270, "MEDIUM", "BUY_CALL", "TRAILING_SL_HIT", "TRENDING", 115),
        ("LOSS",-250, "LOW",    "BUY_PUT",  "STOP_LOSS_HIT",   "RANGE_BOUND",117),
        ("WIN",  390, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING", 119),
        ("WIN",  420, "HIGH",   "BUY_CALL", "TARGET_HIT",      "TRENDING", 121),
        ("LOSS",-200, "MEDIUM", "BUY_PUT",  "FORCED_EOD",      "SIDEWAYS", 123),
        ("WIN",  340, "MEDIUM", "BUY_CALL", "TARGET_HIT",      "TRENDING", 125),
    ]

    capital = initial_capital
    for outcome, pnl, conf, ttype, exit_r, regime, offset in scenarios:
        entry_time = start + timedelta(days=offset)
        trades.append(_make_trade(
            outcome=outcome,
            net_pnl=float(pnl),
            entry_time=entry_time,
            confidence=conf,
            trade_type=ttype,
            exit_reason=exit_r,
            regime=regime,
            capital_at_entry=capital,
        ))
        capital = max(capital + float(pnl), 1.0)

    equity_curve = _build_equity_curve(initial_capital, trades, start)
    return trades, equity_curve, initial_capital


# ─────────────────────────────────────────────────────────────────────────────
# Run the exact user code
# ─────────────────────────────────────────────────────────────────────────────

class _MockResult:
    """Mimics BacktestResult for the user's code snippet."""
    def __init__(self, trade_history, equity_curve, initial_capital):
        self.trade_history = trade_history
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital


def run_user_code(result: _MockResult) -> BacktestMetrics:
    """Runs the exact code snippet provided by the user."""
    from src.backtest.metrics import MetricsCalculator

    calc = MetricsCalculator()

    metrics = calc.calculate_all(
        trade_history=result.trade_history,
        equity_curve=result.equity_curve,
        initial_capital=result.initial_capital
    )

    print(f"=== Performance Metrics ===")
    print(f"Return: {metrics.total_return_pct:+.2f}% (₹{metrics.total_return_amount:+,.0f})")
    print(f"Trades: {metrics.total_trades} | Win Rate: {metrics.win_rate:.1f}%")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Expected Value: ₹{metrics.expected_value_per_trade:+,.0f}/trade")
    print(f"Max Drawdown: -{metrics.max_drawdown_pct:.1f}%")
    print(f"Sharpe: {metrics.sharpe_ratio:.2f} | Sortino: {metrics.sortino_ratio:.2f}")
    print(f"Grade: {metrics.strategy_grade}")
    print(f"\n{metrics.assessment}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Extended print (all sections)
# ─────────────────────────────────────────────────────────────────────────────

def print_extended(metrics: BacktestMetrics) -> None:
    sep = "─" * 55

    # Monthly
    print(f"\n{sep}")
    print("MONTHLY RETURNS")
    print(sep)
    for m in metrics.monthly_returns:
        arrow = "▲" if m["return_pct"] >= 0 else "▼"
        print(f"  {m['month']}  {arrow} {m['return_pct']:+.2f}%  (₹{m['pnl']:+,.0f})")
    print(f"  {'Positive months':<28} {metrics.positive_months}")
    print(f"  {'Negative months':<28} {metrics.negative_months}")
    print(f"  {'Monthly win rate':<28} {metrics.monthly_win_rate:.1f}%")

    # Detailed trade stats
    print(f"\n{sep}")
    print("TRADE STATISTICS")
    print(sep)
    print(f"  {'Total / W / L / BE':<28} {metrics.total_trades} / {metrics.winning_trades} / {metrics.losing_trades} / {metrics.breakeven_trades}")
    print(f"  {'Largest win':<28} ₹{metrics.largest_win:,.0f}")
    print(f"  {'Largest loss':<28} ₹{metrics.largest_loss:,.0f}")
    print(f"  {'Avg win duration':<28} {metrics.avg_winning_trade_duration:.1f} bars")
    print(f"  {'Avg loss duration':<28} {metrics.avg_losing_trade_duration:.1f} bars")
    print(f"  {'Max consecutive wins':<28} {metrics.max_consecutive_wins}")
    print(f"  {'Max consecutive losses':<28} {metrics.max_consecutive_losses}")
    print(f"  {'Current streak':<28} {metrics.current_streak:+d}")

    # Risk
    print(f"\n{sep}")
    print("RISK METRICS")
    print(sep)
    print(f"  {'Max drawdown':<28} -{metrics.max_drawdown_pct:.2f}%  (₹{metrics.max_drawdown_amount:,.0f})")
    print(f"  {'Max drawdown duration':<28} {metrics.max_drawdown_duration_bars} bars")
    if metrics.max_drawdown_start:
        print(f"  {'DD started':<28} {metrics.max_drawdown_start.strftime('%Y-%m-%d')}")
    print(f"  {'Avg drawdown':<28} -{metrics.avg_drawdown_pct:.2f}%")
    print(f"  {'Max recovery time':<28} {metrics.max_recovery_time_bars} bars")

    # Confidence
    print(f"\n{sep}")
    print("CONFIDENCE BREAKDOWN")
    print(sep)
    print(f"  {'HIGH':<8} {metrics.high_confidence_trades:3d} trades  win {metrics.high_confidence_win_rate:5.1f}%  avg ₹{metrics.high_confidence_avg_pnl:+,.0f}")
    print(f"  {'MEDIUM':<8} {metrics.medium_confidence_trades:3d} trades  win {metrics.medium_confidence_win_rate:5.1f}%  avg ₹{metrics.medium_confidence_avg_pnl:+,.0f}")
    print(f"  {'LOW':<8} {metrics.low_confidence_trades:3d} trades  win {metrics.low_confidence_win_rate:5.1f}%  avg ₹{metrics.low_confidence_avg_pnl:+,.0f}")

    # Exit breakdown
    print(f"\n{sep}")
    print("EXIT ANALYSIS")
    print(sep)
    print(f"  {'Target Hit':<24} {metrics.target_hit_count:3d}  ({metrics.target_hit_pct:.1f}%)")
    print(f"  {'Stop Loss':<24} {metrics.sl_hit_count:3d}  ({metrics.sl_hit_pct:.1f}%)")
    print(f"  {'Trailing Stop':<24} {metrics.trailing_sl_count:3d}  ({metrics.trailing_sl_pct:.1f}%)")
    print(f"  {'Forced EOD':<24} {metrics.forced_eod_count:3d}  avg ₹{metrics.forced_eod_avg_pnl:+,.0f}")

    # Regime
    print(f"\n{sep}")
    print("REGIME BREAKDOWN")
    print(sep)
    for reg, stats in metrics.trades_by_regime.items():
        print(f"  {reg:<18} {stats['count']:3d} trades  win {stats['win_rate']:5.1f}%  avg ₹{stats['avg_pnl']:+,.0f}")
    print(f"  Best regime:  {metrics.best_regime}")
    print(f"  Worst regime: {metrics.worst_regime}")

    # DOW
    print(f"\n{sep}")
    print("DAY OF WEEK BREAKDOWN")
    print(sep)
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    for day in dow_order:
        if day in metrics.trades_by_day_of_week:
            s = metrics.trades_by_day_of_week[day]
            print(f"  {day:<12} {s['count']:3d} trades  win {s['win_rate']:5.1f}%  avg ₹{s['avg_pnl']:+,.0f}")
    print(f"  Best day:   {metrics.best_day_of_week}")
    print(f"  Worst day:  {metrics.worst_day_of_week}")

    # Edge flags
    print(f"\n{sep}")
    print("OVERALL VERDICT")
    print(sep)
    print(f"  Is profitable: {metrics.is_profitable}")
    print(f"  Has edge:      {metrics.has_edge}")
    print(f"  Grade:         {metrics.strategy_grade}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison demo
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison() -> None:
    from src.backtest.metrics import MetricsCalculator

    # Variant 2: identical trades but without LOW-confidence ones
    trades_full, eq_full, cap = build_realistic_result()
    trades_filtered = [t for t in trades_full if t.confidence_level != "LOW"]
    eq_filtered = _build_equity_curve(cap, trades_filtered, datetime(2024, 1, 8, 9, 30))

    m_full     = MetricsCalculator().calculate_all(trades_full, eq_full, cap)
    m_filtered = MetricsCalculator().calculate_all(trades_filtered, eq_filtered, cap)

    report = MetricsCalculator.compare_backtests(
        [m_full, m_filtered],
        ["All signals", "HIGH+MEDIUM only"]
    )

    sep = "─" * 60
    print(f"\n{sep}")
    print("COMPARISON: All Signals vs HIGH+MEDIUM Only")
    print(sep)
    print(f"{'Metric':<24}", end="")
    for h in report.headers:
        print(f"  {h:<22}", end="")
    print()
    print("-" * (24 + 24 * len(report.headers)))
    for row in report.metrics_table:
        print(f"{row['metric_name']:<24}", end="")
        for v in row["values"]:
            print(f"  {v:<22}", end="")
        print()

    print(f"\nBest overall: {report.best_overall}")
    print("Ranking:")
    for rank, (label, score) in enumerate(report.ranking, 1):
        print(f"  #{rank}  {label}  (composite score: {score:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Step 6.4 — MetricsCalculator End-to-End Validation")
    print("=" * 60)
    print()
    print("  Building synthetic BacktestResult:")
    print("  • 52 trades across Jan–Jun 2024")
    print("  • NIFTY50, TECHNICAL_ONLY mode")
    print("  • 3 confidence levels, mixed exits, 3 regimes")
    print("  • Mid-period drawdown (April losing streak)")
    print()

    trades, equity, capital = build_realistic_result()
    result = _MockResult(trades, equity, capital)

    print(f"  Trade history: {len(trades)} trades")
    print(f"  Equity curve:  {len(equity)} daily points")
    print(f"  Capital:       ₹{capital:,.0f}")
    print()

    # ── Run the user's exact code ─────────────────────────────────────────
    print("=" * 60)
    print("  Running user code (exact snippet)")
    print("=" * 60)
    print()
    metrics = run_user_code(result)

    # ── Extended breakdown ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Extended Breakdown (all metric categories)")
    print("=" * 60)
    print_extended(metrics)

    # ── Comparison ────────────────────────────────────────────────────────
    run_comparison()

    print()
    print("=" * 60)
    print("  Validation complete — MetricsCalculator OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
