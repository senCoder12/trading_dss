"""
Run robustness tests on the best parameters from an optimization run.

Executes a full optimization (Step 7.2) then subjects the best parameters
to four robustness stress tests (Step 7.3):
  1. Parameter Sensitivity (±20% perturbation)
  2. Data Stability (sub-period analysis)
  3. Monte Carlo Simulation (trade-order reshuffling)
  4. Regime Robustness (market-regime breakdown)
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.db_manager import DatabaseManager
from src.backtest.trade_simulator import SimulatorConfig
from src.backtest.strategy_runner import BacktestConfig, StrategyRunner
from src.backtest.optimizer.param_space import ParameterSpaceLoader, ParameterApplicator
from src.backtest.optimizer.optimization_engine import OptimizationConfig, OptimizationEngine
from src.backtest.optimizer.robustness import RobustnessTester


def main():
    parser = argparse.ArgumentParser(description="Trading DSS — Robustness Tester")
    parser.add_argument("--index", default="NIFTY50", help="Index ID (default: NIFTY50)")
    parser.add_argument("--profile", default="conservative",
                        choices=["conservative", "moderate", "aggressive"],
                        help="Optimization profile (default: conservative)")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2023-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--timeframe", default="1d", help="Bar timeframe")
    parser.add_argument("--mode", default="TECHNICAL_ONLY",
                        choices=["TECHNICAL_ONLY", "TECHNICAL_OPTIONS", "FULL"])
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--no-wf", action="store_true", help="Disable walk-forward validation")
    parser.add_argument("--min-trades", type=int, default=30, help="Minimum trades constraint")
    parser.add_argument("--max-dd", type=float, default=25.0, help="Max drawdown %% limit")
    parser.add_argument("--min-wr", type=float, default=40.0, help="Min win rate %%")
    parser.add_argument("--min-pf", type=float, default=0.9, help="Min profit factor")
    args = parser.parse_args()

    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        sys.exit(1)

    # --- 1. Initialize ---
    db = DatabaseManager()
    db.connect()

    sim_config = SimulatorConfig(initial_capital=args.capital)

    # --- 2. Load parameter space ---
    loader = ParameterSpaceLoader()
    print(f"Loading optimization profile '{args.profile}'...")
    space = loader.load_profile(args.profile)
    print(space.summary() + "\n")

    # --- 3. Run optimization (Step 7.2) ---
    opt_config = OptimizationConfig(
        index_id=args.index,
        start_date=start_date,
        end_date=end_date,
        timeframe=args.timeframe,
        mode=args.mode,
        base_simulator_config=sim_config,
        parameter_space=space,
        run_walk_forward=not args.no_wf,
        min_trades=args.min_trades,
        max_drawdown_limit=args.max_dd,
        min_win_rate=args.min_wr,
        min_profit_factor=args.min_pf,
    )

    print(f"Starting optimization for {args.index} over {args.start} to {args.end}...")
    engine = OptimizationEngine(db)
    result = engine.run(opt_config)

    if result.best_params is None or result.best_metrics is None:
        print("\nOptimization produced no viable parameters. Cannot run robustness tests.")
        sys.exit(1)

    print(result.get_summary())

    best_params = result.best_params
    best_metrics = result.best_metrics

    # --- 4. Get trade history for Monte Carlo / Regime tests ---
    # Re-run the best parameter set to get full trade history.
    # (The optimization engine stores backtest_result in ranked_results.)
    trade_history = []
    if result.ranked_results and result.ranked_results[0].backtest_result:
        trade_history = result.ranked_results[0].backtest_result.trade_history
        print(f"\nUsing {len(trade_history)} trades from optimization's best result.")
    else:
        # Fall back: re-run the best params to get trade history
        print("\nRe-running best params to collect trade history...")
        applicator = ParameterApplicator()
        base_config = BacktestConfig(
            index_id=args.index,
            start_date=start_date,
            end_date=end_date,
            timeframe=args.timeframe,
            mode=args.mode,
            simulator_config=sim_config,
            show_progress=False,
        )
        applied_config = applicator.apply(best_params, base_config)
        runner = StrategyRunner(db)
        bt_result = runner.run(applied_config)
        trade_history = bt_result.trade_history
        print(f"  Collected {len(trade_history)} trades.")

    # --- 5. Build base config for robustness tests ---
    base_config = BacktestConfig(
        index_id=args.index,
        start_date=start_date,
        end_date=end_date,
        timeframe=args.timeframe,
        mode=args.mode,
        simulator_config=sim_config,
        show_progress=False,
    )

    # --- 6. Run robustness tests (Step 7.3) ---
    print("\n" + "=" * 60)
    print("ROBUSTNESS TESTING")
    print("=" * 60)

    tester = RobustnessTester(db)
    report = tester.run_full_robustness_test(
        params=best_params,
        base_config=base_config,
        original_metrics=best_metrics,
        trade_history=trade_history,
        parameter_space=space,
    )

    # --- 7. Print results ---
    print(f"\n=== Robustness Report ===")
    print(f"Grade: {report.robustness_grade} (score: {report.robustness_score:.2f})")
    print(f"Approved for paper trading: {report.is_approved}")

    if report.sensitivity_result:
        print(f"\nSensitivity: {report.sensitivity_result.overall_sensitivity_score:.2f}")
        print(f"  Most sensitive: {report.sensitivity_result.most_sensitive_param}")
        print(f"  Least sensitive: {report.sensitivity_result.least_sensitive_param}")
        print(f"  Robust: {report.sensitivity_result.is_robust}")

    if report.stability_result:
        total_periods = (report.stability_result.profitable_periods
                         + report.stability_result.unprofitable_periods)
        print(f"\nStability: {report.stability_result.consistency_score:.2f}")
        print(f"  Profitable periods: {report.stability_result.profitable_periods}/{total_periods}")
        print(f"  Return range: [{report.stability_result.return_range[0]:.1f}%, "
              f"{report.stability_result.return_range[1]:.1f}%]")
        print(f"  Stable: {report.stability_result.is_stable}")

    if report.monte_carlo_result:
        print(f"\nMonte Carlo: {report.monte_carlo_result.reliability_score:.2f}")
        print(f"  Simulations: {report.monte_carlo_result.n_simulations}")
        print(f"  Profitable sims: {report.monte_carlo_result.pct_profitable:.0f}%")
        print(f"  95% CI: [{report.monte_carlo_result.return_95_ci[0]:.1f}%, "
              f"{report.monte_carlo_result.return_95_ci[1]:.1f}%]")
        print(f"  Luck factor: {report.monte_carlo_result.luck_factor:.2f}")
    else:
        print(f"\nMonte Carlo: SKIPPED (need >= 20 trades, got {len(trade_history)})")

    if report.regime_result:
        print(f"\nRegime Robustness: {report.regime_result.regime_diversity_score:.2f}")
        print(f"  Regimes tested: {report.regime_result.regimes_tested}")
        print(f"  Profitable regimes: {report.regime_result.profitable_regimes}")
        print(f"  Best: {report.regime_result.best_regime} ({report.regime_result.best_regime_return:.1f}%)")
        print(f"  Worst: {report.regime_result.worst_regime} ({report.regime_result.worst_regime_return:.1f}%)")
        print(f"  Regime dependent: {report.regime_result.is_regime_dependent}")
    else:
        print(f"\nRegime Robustness: SKIPPED (no regime data in trades)")

    if report.concerns:
        print(f"\nConcerns:")
        for c in report.concerns:
            print(f"  - {c}")

    if report.recommendations:
        print(f"\nRecommendations:")
        for r in report.recommendations:
            print(f"  - {r}")

    print(f"\n{'=' * 60}")
    print(report.summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
