"""
Trading DSS — Strategy Optimizer CLI (v2.0)

Unified CLI that runs optimization, robustness testing, report generation,
and optional auto-approval of parameters.

Usage examples::

    # Basic optimization with conservative profile
    python scripts/run_optimizer.py --index NIFTY50 --start 2024-01-01 --end 2024-12-31

    # With specific profile
    python scripts/run_optimizer.py --index NIFTY50 --start 2023-01-01 --end 2024-12-31 --profile moderate

    # Random search (faster for large spaces)
    python scripts/run_optimizer.py --index NIFTY50 --start 2024-01-01 --end 2024-12-31 \\
        --profile moderate --search random --max-eval 200

    # Optimize and auto-approve
    python scripts/run_optimizer.py --index NIFTY50 --start 2024-01-01 --end 2024-12-31 --auto-approve

    # Multiple indices
    python scripts/run_optimizer.py --index NIFTY50 BANKNIFTY --start 2024-01-01 --end 2024-12-31

    # Save report
    python scripts/run_optimizer.py --index NIFTY50 --start 2024-01-01 --end 2024-12-31 --save-report

    # List currently approved params
    python scripts/run_optimizer.py --list-approved
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime

# Add the project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.db_manager import DatabaseManager
from src.backtest.trade_simulator import SimulatorConfig
from src.backtest.strategy_runner import BacktestConfig, StrategyRunner
from src.backtest.optimizer.param_space import (
    ParameterApplicator,
    ParameterSpaceLoader,
)
from src.backtest.optimizer.optimization_engine import (
    OptimizationConfig,
    OptimizationEngine,
    _params_to_key,
)
from src.backtest.optimizer.robustness import RobustnessTester
from src.backtest.optimizer.report import OptimizationReportGenerator
from src.backtest.optimizer.param_applier import ApprovedParameterManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = """\
============================================
Trading DSS — Strategy Optimizer v2.0
{details}
============================================"""


def _print_banner(args: argparse.Namespace, profile_desc: str, combos: int) -> None:
    details_lines = [
        f"Index:     {', '.join(args.index)}",
        f"Period:    {args.start} -> {args.end}",
        f"Profile:   {args.profile.title()} ({combos} combinations)",
        f"Method:    {'Grid Search' if args.search == 'grid' else 'Random Search'}",
        f"Objective: {_pretty_objective(args.objective)}",
    ]
    print(_BANNER.format(details="\n".join(details_lines)))


def _pretty_objective(obj: str) -> str:
    return {
        "sharpe_ratio": "Sharpe Ratio",
        "total_return_pct": "Total Return",
        "profit_factor": "Profit Factor",
    }.get(obj, obj)


# ---------------------------------------------------------------------------
# List approved
# ---------------------------------------------------------------------------


def _list_approved() -> None:
    mgr = ApprovedParameterManager()
    all_approved = mgr.list_all_approved()

    if not all_approved:
        print("No approved parameter sets found.")
        return

    print("\nApproved Parameter Sets:")
    print("=" * 70)

    for index_id, entry in all_approved.items():
        status = entry.get("status", "UNKNOWN")
        approved_at = entry.get("approved_at", "?")
        score = entry.get("robustness_score", 0)
        params = entry.get("params", {})
        metrics = entry.get("metrics_at_approval", {})

        icon = {"ACTIVE": "\u2705", "EXPIRED": "\u274c", "SUSPENDED": "\u26a0\ufe0f"}.get(
            status, "\u2753"
        )

        print(f"\n{icon} {index_id} [{status}]")
        print(f"  Approved: {approved_at}")
        print(f"  Robustness: {score:.2f}")
        print(f"  Params: {json.dumps(params, default=str)}")

        ret = metrics.get("return_pct")
        sharpe = metrics.get("sharpe")
        wr = metrics.get("win_rate")
        if ret is not None:
            print(f"  Metrics: Return {ret:+.1f}% | Sharpe {sharpe:.2f} | Win Rate {wr:.1f}%")

        if entry.get("expire_reason"):
            print(f"  Expired: {entry['expire_reason']}")
        if entry.get("notes"):
            print(f"  Notes: {entry['notes']}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Config hash helper
# ---------------------------------------------------------------------------


def _config_hash(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Run optimisation for a single index
# ---------------------------------------------------------------------------


def _run_for_index(
    index_id: str,
    args: argparse.Namespace,
    db: DatabaseManager,
    loader: ParameterSpaceLoader,
) -> None:
    """Execute the full optimization pipeline for one index."""

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    # 1. Load profile
    space = loader.load_profile(args.profile)

    if args.verbose:
        print(space.summary())
        print()

    _print_banner(args, args.profile, space.total_combinations)

    # 2. Build optimization config
    sim_config = SimulatorConfig(initial_capital=args.capital)

    opt_config = OptimizationConfig(
        index_id=index_id,
        start_date=start_date,
        end_date=end_date,
        timeframe="1d",
        mode="TECHNICAL_ONLY",
        base_simulator_config=sim_config,
        parameter_space=space,
        search_method=args.search,
        max_evaluations=args.max_eval if args.search == "random" else None,
        primary_objective=args.objective,
        run_walk_forward=args.robustness,
        show_progress=True,
    )

    # 3. Run optimization
    print(f"\nRunning optimization for {index_id}...")
    engine = OptimizationEngine(db)
    opt_result = engine.run(opt_config)

    if opt_result.best_params is None or opt_result.best_metrics is None:
        print(f"\nNo viable parameters found for {index_id}.")
        print(opt_result.get_summary())
        return

    # 4. Run robustness testing on top 3
    robustness_reports: dict[str, object] = {}
    if args.robustness:
        print("\n" + "=" * 60)
        print("ROBUSTNESS TESTING")
        print("=" * 60)

        tester = RobustnessTester(db)
        top_n = min(3, len(opt_result.ranked_results))

        for i, eval_r in enumerate(opt_result.ranked_results[:top_n]):
            print(f"\n  Testing Rank #{eval_r.rank or i + 1}...")

            # Get trade history
            trade_history = []
            if eval_r.backtest_result:
                trade_history = eval_r.backtest_result.trade_history

            if not trade_history:
                # Re-run to get trade history
                applicator = ParameterApplicator()
                base_config = BacktestConfig(
                    index_id=index_id,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe="1d",
                    mode="TECHNICAL_ONLY",
                    simulator_config=sim_config,
                    show_progress=False,
                )
                applied = applicator.apply(base_config, eval_r.params)
                if applied is not None:
                    try:
                        runner = StrategyRunner(db)
                        bt_result = runner.run(applied)
                        trade_history = bt_result.trade_history
                    except Exception as exc:
                        logger.error("Re-run for robustness failed: %s", exc)

            base_config = BacktestConfig(
                index_id=index_id,
                start_date=start_date,
                end_date=end_date,
                timeframe="1d",
                mode="TECHNICAL_ONLY",
                simulator_config=sim_config,
                show_progress=False,
            )

            try:
                rob_report = tester.run_full_robustness_test(
                    params=eval_r.params,
                    base_config=base_config,
                    original_metrics=eval_r.metrics,
                    trade_history=trade_history,
                    parameter_space=space,
                )
                key = _params_to_key(eval_r.params)
                robustness_reports[key] = rob_report

                icon = "\u2705" if rob_report.is_approved else "\u274c"
                print(
                    f"  Rank #{eval_r.rank or i + 1}: "
                    f"{rob_report.robustness_grade} ({rob_report.robustness_score:.2f}) "
                    f"{icon}"
                )
            except Exception as exc:
                logger.error(
                    "Robustness test failed for Rank #%d: %s",
                    eval_r.rank or i + 1,
                    exc,
                )
                print(f"  Rank #{eval_r.rank or i + 1}: FAILED ({exc})")

    # 5. Generate and print report
    print("\n")
    report_gen = OptimizationReportGenerator()
    report_text = report_gen.generate_report(opt_result, robustness_reports)
    print(report_text)

    # 6. Auto-approve if requested and best result passes robustness
    if args.auto_approve:
        best_key = _params_to_key(opt_result.best_params)
        best_rob = robustness_reports.get(best_key)

        if best_rob and best_rob.is_approved:
            mgr = ApprovedParameterManager()
            mgr.save_approved_params(
                index_id=index_id,
                params=opt_result.best_params,
                metrics=opt_result.best_metrics,
                robustness_score=best_rob.robustness_score,
                config_hash=_config_hash(opt_result.best_params),
                notes=f"Auto-approved via CLI. Profile: {args.profile}",
            )
            print(f"\n\u2705 Parameters auto-approved for {index_id}.")
        elif best_rob and not best_rob.is_approved:
            print(
                f"\n\u274c Auto-approve skipped for {index_id}: "
                f"robustness score {best_rob.robustness_score:.2f} "
                f"({best_rob.robustness_grade}) — below threshold."
            )
        else:
            print(
                f"\n\u26a0\ufe0f Auto-approve skipped for {index_id}: "
                f"no robustness data available."
            )

    # 7. Save report if requested
    if args.save_report:
        path = report_gen.save_report(report_text, index_id=index_id)
        print(f"\nReport saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trading DSS — Strategy Optimizer v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --index NIFTY50 --start 2024-01-01 --end 2024-12-31\n"
            "  %(prog)s --index NIFTY50 BANKNIFTY --start 2024-01-01 --end 2024-12-31 --profile moderate\n"
            "  %(prog)s --index NIFTY50 --start 2024-01-01 --end 2024-12-31 --auto-approve --save-report\n"
            "  %(prog)s --list-approved\n"
        ),
    )

    parser.add_argument(
        "--index", nargs="+",
        help="One or more index IDs (e.g. NIFTY50 BANKNIFTY)",
    )
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--profile", default="conservative",
        choices=["conservative", "moderate", "aggressive"],
        help="Optimization profile (default: conservative)",
    )
    parser.add_argument(
        "--search", default="grid", choices=["grid", "random"],
        help="Search method (default: grid)",
    )
    parser.add_argument(
        "--max-eval", type=int, default=500,
        help="Max evaluations for random search (default: 500)",
    )
    parser.add_argument(
        "--objective", default="sharpe_ratio",
        choices=["sharpe_ratio", "total_return_pct", "profit_factor"],
        help="Primary objective (default: sharpe_ratio)",
    )
    parser.add_argument(
        "--robustness", default=True,
        action=argparse.BooleanOptionalAction,
        help="Run robustness testing on top results (default: True)",
    )
    parser.add_argument(
        "--auto-approve", action="store_true",
        help="Auto-save approved params if robustness passes",
    )
    parser.add_argument(
        "--save-report", action="store_true",
        help="Save full report to data/reports/",
    )
    parser.add_argument(
        "--list-approved", action="store_true",
        help="List all currently approved parameter sets and exit",
    )
    parser.add_argument(
        "--capital", type=float, default=100000.0,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Detailed output",
    )

    args = parser.parse_args()

    # --list-approved is standalone
    if args.list_approved:
        _list_approved()
        return

    # Validate required args
    if not args.index:
        parser.error("--index is required (unless using --list-approved)")
    if not args.start or not args.end:
        parser.error("--start and --end are required")

    # Validate dates
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        sys.exit(1)

    # Setup logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )

    # Initialize
    db = DatabaseManager()
    db.connect()
    loader = ParameterSpaceLoader()

    try:
        for index_id in args.index:
            _run_for_index(index_id, args, db, loader)
            if len(args.index) > 1:
                print("\n" + "=" * 70 + "\n")
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Optimization failed: %s", exc)
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
