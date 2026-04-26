#!/usr/bin/env python3
"""
Phase 2 Walk-Forward Validation Script.

Validates strategy robustness using walk-forward testing:
- 5-fold or rolling windows to avoid in-sample overfitting
- 1% entry + 0.5% exit slippage
- Confidence-bucketed results (HIGH/MEDIUM/LOW)
- Checks profitability across train/test periods

Usage:
    python scripts/run_phase2_walk_forward.py [--index NIFTY50] [--months 24]
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.trade_simulator import SimulatorConfig
from src.backtest.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.database.db_manager import DatabaseManager


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Walk-Forward Validation")
    parser.add_argument("--index", default="NIFTY50", help="Index to test (default: NIFTY50)")
    parser.add_argument("--months", type=int, default=24, help="Months of history to use (default: 24)")
    parser.add_argument("--train-days", type=int, default=252, help="Train window size in trading days")
    parser.add_argument("--test-days", type=int, default=63, help="Test window size in trading days")
    parser.add_argument("--step-days", type=int, default=63, help="Step size between windows in trading days")
    parser.add_argument("--warmup-bars", type=int, default=50, help="Bars to warm up indicators (default: 50)")
    args = parser.parse_args()

    db = DatabaseManager()
    try:
        db.connect()
    except Exception as e:
        print(f"Failed to connect to database: {e}", file=sys.stderr)
        return 1

    # Calculate date range (approximate, using calendar days)
    end_date = date.today()
    start_date = end_date - timedelta(days=args.months * 30)

    print(f"\n{'='*70}")
    print(f"PHASE 2 WALK-FORWARD VALIDATION")
    print(f"{'='*70}")
    print(f"Index:        {args.index}")
    print(f"Date Range:   {start_date} → {end_date}")
    print(f"Train Window: {args.train_days} trading days")
    print(f"Test Window:  {args.test_days} trading days")
    print(f"Step Size:    {args.step_days} trading days")
    print(f"Warmup:       {args.warmup_bars} bars")
    print(f"Entry/Exit Slippage: 1% / 0.5%")
    print(f"{'='*70}\n")

    # Create walk-forward config with Phase 2 settings
    simulator_config = SimulatorConfig(
        initial_capital=500_000,
        max_risk_per_trade_pct=2.0,
        max_risk_per_day_pct=5.0,
        max_open_positions=3,
        use_percentage_slippage=True,       # Use percentage-based slippage
        use_variable_spread=False,           # Not dynamic spreads
        entry_slippage_pct=1.0,             # Phase 2: 1% entry
        exit_slippage_pct=0.5,              # Phase 2: 0.5% exit
    )

    config = WalkForwardConfig(
        index_id=args.index,
        full_start_date=start_date,
        full_end_date=end_date,
        timeframe="1d",
        train_window_days=args.train_days,
        test_window_days=args.test_days,
        step_days=args.step_days,
        simulator_config=simulator_config,
        mode="TECHNICAL_ONLY",
        min_confidence="LOW",
        warmup_bars=args.warmup_bars,
        parameter_sensitivity=False,
    )

    validator = WalkForwardValidator(db)
    try:
        result = validator.run_walk_forward(config)
    except Exception as e:
        print(f"Walk-forward validation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD RESULTS SUMMARY")
    print(f"{'='*70}\n")

    print(f"Total Windows:                   {result.total_windows}")
    print(f"Profitable Test Windows:         {result.profitable_test_windows}/{result.total_windows}")
    print(f"Test Profitability Rate:         {result.test_profitability_rate:.1%}")
    print()

    print(f"Avg Train Return:                {result.avg_train_return:+.2f}%")
    print(f"Avg Test Return:                 {result.avg_test_return:+.2f}%")
    print(f"Avg Degradation (Train→Test):    {result.avg_degradation:.2f}pp")
    print()

    print(f"Avg Train Win Rate:              {result.avg_train_win_rate:.1f}%")
    print(f"Avg Test Win Rate:               {result.avg_test_win_rate:.1f}%")
    print()

    print(f"Avg Train Sharpe:                {result.avg_train_sharpe:.2f}")
    print(f"Avg Test Sharpe:                 {result.avg_test_sharpe:.2f}")
    print(f"Max Test Drawdown:               {result.max_test_drawdown:.2f}%")
    print()

    print(f"Combined Test Trades:            {result.combined_test_trades}")
    print(f"Combined Test Return:            {result.combined_test_return:+.2f}%")
    print(f"Combined Test Win Rate:          {result.combined_test_win_rate:.1f}%")
    print()

    print(f"Overfitting Score:               {result.overfitting_score:.2f}/1.0")
    print(f"Overfitting Assessment:          {result.overfitting_assessment}")
    print(f"Is Robust for Paper Trading:     {result.is_robust}")
    print()

    print(f"\nVERDICT:")
    print(result.verdict)
    print()

    # Print per-window details
    print(f"\n{'='*70}")
    print(f"PER-WINDOW DETAILS")
    print(f"{'='*70}\n")

    for w in result.windows:
        print(f"Window {w.window_id}: {w.train_start} → {w.test_start}")
        print(f"  Train: {w.train_metrics.total_trades:3d} trades, {w.train_metrics.win_rate:5.1f}% WR, {w.train_metrics.total_return_pct:+7.2f}%")
        print(f"  Test:  {w.test_metrics.total_trades:3d} trades, {w.test_metrics.win_rate:5.1f}% WR, {w.test_metrics.total_return_pct:+7.2f}%")
        print(f"  Degradation: {w.return_degradation_pct:+6.2f}pp (Train→Test return)")

        # Confidence breakdown
        print(f"  Confidence Split (Test):")
        print(f"    HIGH:   {w.test_metrics.high_confidence_trades:3d} trades, {w.test_metrics.high_confidence_win_rate:5.1f}% WR, {w.test_metrics.high_confidence_avg_pnl:+8.0f}₹")
        print(f"    MEDIUM: {w.test_metrics.medium_confidence_trades:3d} trades, {w.test_metrics.medium_confidence_win_rate:5.1f}% WR, {w.test_metrics.medium_confidence_avg_pnl:+8.0f}₹")
        print(f"    LOW:    {w.test_metrics.low_confidence_trades:3d} trades, {w.test_metrics.low_confidence_win_rate:5.1f}% WR, {w.test_metrics.low_confidence_avg_pnl:+8.0f}₹")
        print()

    print(f"\n{'='*70}")
    print("PHASE 2 VALIDATION COMPLETE")
    print(f"{'='*70}\n")

    return 0 if result.is_robust else 1


if __name__ == "__main__":
    sys.exit(main())
