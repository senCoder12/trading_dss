#!/usr/bin/env python3
"""
CLI tool to optimize F&O stock scoring weights via backtesting.

Usage:
    python scripts/tune_fo_weights.py --days 60
    python scripts/tune_fo_weights.py --days 60 --output config/fo_scoring_weights.json --report reports/
"""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure src is in path (run from project root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.fo_universe_backtester import FOUniverseBacktester, BacktestConfig
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


def main():
    """Run weight optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize F&O universe scoring weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/tune_fo_weights.py --days 60
  python scripts/tune_fo_weights.py --days 120 --output config/fo_scoring_weights.json
        """,
    )

    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of recent days to backtest (default: 60)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for optimized weights JSON (default: config/fo_scoring_weights.json)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output directory for markdown report (default: reports/)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to SQLite database (default: data/trading_dss.db)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Universe size to optimize for (default: 50)",
    )
    parser.add_argument(
        "--priority-n",
        type=int,
        default=15,
        help="Priority list size (default: 15)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Database path
    db_path = args.db or Path("data/trading_dss.db")
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    logger.info(f"Loading database: {db_path}")

    try:
        db = DatabaseManager(str(db_path))
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    # Setup backtester
    backtester = FOUniverseBacktester(db)

    # Load historical data
    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    logger.info(f"Backtesting period: {start_date} to {end_date}")

    if not backtester.load_historical_data(start_date, end_date):
        logger.error("Failed to load historical data")
        sys.exit(1)

    # Run grid search
    logger.info(f"Starting grid search (top_n={args.top_n}, priority_n={args.priority_n})")

    results = backtester.run_grid_search(
        top_n=args.top_n,
        priority_n=args.priority_n,
    )

    if not results:
        logger.error("Grid search returned no results")
        sys.exit(1)

    # Display top 5
    print("\n" + "=" * 80)
    print("TOP 5 WEIGHT COMBINATIONS")
    print("=" * 80)

    for i, result in enumerate(results[:5], 1):
        w = result.weights_tested
        print(f"\n#{i}")
        print(f"  Hit Rate:        {result.hit_rate:.1%}")
        print(f"  Priority Hit:    {result.priority_hit_rate:.1%}")
        print(f"  Avg Daily Move:  {result.avg_daily_move:.2%}")
        print(f"  Weights:")
        print(f"    OI:            {w.oi_weight:.3f}")
        print(f"    OI Change:     {w.oi_change_weight:.3f}")
        print(f"    Volume:        {w.volume_weight:.3f}")
        print(f"    Price Move:    {w.price_move_weight:.3f}")
        print(f"    IV Change:     {w.iv_change_weight:.3f}")
        print(f"    Spread:        {w.spread_tightness_weight:.3f}")

    # Save best weights
    best = results[0]
    output_path = args.output or (Path("config") / "fo_scoring_weights.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    weights_dict = {
        'oi_weight': best.weights_tested.oi_weight,
        'oi_change_weight': best.weights_tested.oi_change_weight,
        'volume_weight': best.weights_tested.volume_weight,
        'price_move_weight': best.weights_tested.price_move_weight,
        'iv_change_weight': best.weights_tested.iv_change_weight,
        'spread_tightness_weight': best.weights_tested.spread_tightness_weight,
    }

    with open(output_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)

    print(f"\n✓ Best weights saved to: {output_path}")

    # Generate report
    report_dir = args.report or Path("reports")
    report_path = backtester.generate_report(results, report_dir)
    print(f"✓ Report saved to: {report_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
