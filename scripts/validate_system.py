"""
Pre-launch system validator for the Trading Decision Support System.

Runs a comprehensive check of every component (config, database, data layer,
analysis, decision engine, paper trading, alerts, API, backtesting, system)
and prints a formatted report.

Usage
-----
::

    python scripts/validate_system.py           # full validation
    python scripts/validate_system.py --quiet   # report only on failure
    python scripts/validate_system.py --json    # output raw JSON

Exit code: 0 if all critical checks pass, 1 if any critical check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trading DSS — Pre-Launch System Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output when all checks pass",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Print results as JSON instead of the formatted report",
    )
    args = parser.parse_args()

    from src.paper_trading.pre_launch_validator import PreLaunchValidator

    validator = PreLaunchValidator()
    report = validator.run_full_validation()

    if args.json_output:
        data = {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "warned": report.warned,
            "skipped": report.skipped,
            "is_ready": report.is_ready,
            "checks": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "is_critical": r.is_critical,
                    "section": r.details.get("section", "") if r.details else "",
                }
                for r in report.results
            ],
        }
        print(json.dumps(data, indent=2))
    elif not args.quiet or not report.is_ready:
        print(report.format_report())

    sys.exit(0 if report.is_ready else 1)


if __name__ == "__main__":
    main()
