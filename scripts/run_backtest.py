import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.db_manager import DatabaseManager
from src.backtest.strategy_runner import StrategyRunner, BacktestConfig
from src.backtest.metrics import MetricsCalculator
from src.backtest.trade_simulator import SimulatorConfig
from src.backtest.walk_forward import WalkForwardValidator, WalkForwardConfig
from src.backtest.report_generator import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="Trading DSS — Backtesting Engine v1.0")
    parser.add_argument("--index", nargs="+", required=True, help="One or more index IDs")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--timeframe", default="1d", help="Bar timeframe, default '1d'")
    parser.add_argument("--mode", default="TECHNICAL_ONLY", choices=["TECHNICAL_ONLY", "TECHNICAL_OPTIONS", "FULL"])
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital ₹")
    parser.add_argument("--risk-per-trade", type=float, default=2.0, help="Max risk %%")
    parser.add_argument("--min-confidence", default="MEDIUM", choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--walk-forward", action="store_true", help="Enable walk-forward validation")
    parser.add_argument("--train-days", type=int, default=252, help="Walk-forward train window")
    parser.add_argument("--test-days", type=int, default=63, help="Walk-forward test window")
    parser.add_argument("--save-report", action="store_true", help="Save report to data/reports/")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        sys.exit(1)
        
    db = DatabaseManager()
    db.connect()
    
    simulator_config = SimulatorConfig(
        initial_capital=args.capital,
        max_risk_per_trade_pct=args.risk_per_trade / 100.0
    )
    
    for index_id in args.index:
        print("=" * 44)
        print("Trading DSS \u2014 Backtesting Engine v1.0")
        print(f"Index: {index_id}")
        print(f"Period: {args.start} \u2192 {args.end}")
        print(f"Mode: {args.mode}")
        print(f"Capital: \u20b9{args.capital:,.0f}")
        print("=" * 44)
        
        if args.walk_forward:
            validator = WalkForwardValidator(db)
            wf_config = WalkForwardConfig(
                index_id=index_id,
                full_start_date=start_date,
                full_end_date=end_date,
                timeframe=args.timeframe,
                train_window_days=args.train_days,
                test_window_days=args.test_days,
                step_days=args.test_days,
                simulator_config=simulator_config,
                mode=args.mode,
                min_confidence=args.min_confidence
            )
            wf_result = validator.run_walk_forward(wf_config)
            report = ReportGenerator.generate_walk_forward_report(wf_result)
            print(report)
            
            if args.save_report:
                filepath = ReportGenerator.save_report(report, index_id=f"{index_id}_WF")
                print(f"Report saved to: {filepath}")
                
        else:
            runner = StrategyRunner(db)
            config = BacktestConfig(
                index_id=index_id,
                start_date=start_date,
                end_date=end_date,
                timeframe=args.timeframe,
                mode=args.mode,
                simulator_config=simulator_config,
                min_confidence=args.min_confidence
            )
            result = runner.run(config)
            
            calc = MetricsCalculator()
            metrics = calc.calculate_all(result.trade_history, result.equity_curve, simulator_config.initial_capital)
            
            report = ReportGenerator.generate_backtest_report(result, metrics)
            print(report)
            
            if args.save_report:
                filepath = ReportGenerator.save_report(report, index_id=index_id)
                print(f"Report saved to: {filepath}")
        print()

if __name__ == "__main__":
    main()
