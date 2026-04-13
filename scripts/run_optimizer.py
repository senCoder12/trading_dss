import argparse
import os
import sys
from datetime import datetime

# Add the project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.db_manager import DatabaseManager
from src.backtest.trade_simulator import SimulatorConfig
from src.backtest.optimizer.param_space import ParameterSpaceLoader
from src.backtest.optimizer.optimization_engine import OptimizationConfig, OptimizationEngine

def main():
    parser = argparse.ArgumentParser(description="Trading DSS — Strategy Optimizer")
    parser.add_argument("--index", required=True, help="Index ID (e.g. NIFTY50)")
    parser.add_argument("--profile", required=True, choices=["conservative", "moderate", "aggressive"], help="Optimization profile")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2023-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--timeframe", default="1d", help="Bar timeframe")
    parser.add_argument("--mode", default="TECHNICAL_ONLY", choices=["TECHNICAL_ONLY", "TECHNICAL_OPTIONS", "FULL"])
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--no-wf", action="store_true", help="Disable walk-forward validation")

    args = parser.parse_args()

    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        sys.exit(1)

    db = DatabaseManager()
    db.connect()

    loader = ParameterSpaceLoader()
    print(f"Loading optimization profile '{args.profile}'...")
    try:
        space = loader.load_profile(args.profile)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n" + space.summary() + "\n")

    config = OptimizationConfig(
        index_id=args.index,
        start_date=start_date,
        end_date=end_date,
        timeframe=args.timeframe,
        mode=args.mode,
        base_simulator_config=SimulatorConfig(initial_capital=args.capital),
        parameter_space=space,
        run_walk_forward=not args.no_wf,
    )

    print(f"Starting optimization for {args.index} over {args.start} to {args.end}...")
    engine = OptimizationEngine(db)
    result = engine.run(config)

    # Note: result summary is already printed by engine.run() if show_progress is True (the default).

if __name__ == "__main__":
    main()
