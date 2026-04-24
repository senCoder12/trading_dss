"""
Entry point for the Trading Decision Support System — Data Collector.

Initialises the database, loads the index registry, configures logging,
and runs the APScheduler-backed DataCollector until interrupted.

Usage
-----
    python scripts/run_data_collector.py [--debug] [--dry-run] [--force-start]

Flags
-----
--debug         Verbose DEBUG-level logging to console (file always gets DEBUG).
--dry-run       Fetch data but do not write anything to the database.
--force-start   Bypass market-hours gate — run all interval jobs immediately.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure project root is on sys.path when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import IST_TIMEZONE
from config.logging_config import setup_logging
from config.settings import settings
from src.data.data_collector import DataCollector
from src.data.index_registry import IndexRegistry
from src.database.db_manager import get_db_manager
from src.database.migrations import MigrationRunner

_IST = ZoneInfo(IST_TIMEZONE)
_VERSION = "1.0.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trading DSS Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug",       action="store_true", help="Verbose console logging")
    parser.add_argument("--dry-run",     action="store_true", help="Fetch without storing to DB")
    parser.add_argument("--force-start", action="store_true", help="Ignore market-hours gate")
    return parser.parse_args()


def _print_banner(
    registry: IndexRegistry,
    collector: DataCollector,
    dry_run: bool,
    force_start: bool,
) -> None:
    """Print the startup banner after the scheduler is running."""
    market_status = collector.market_hours.get_market_status()
    active_indices = registry.get_active_indices()
    fo_indices     = registry.get_indices_with_options()

    status_line = market_status.get("status", "UNKNOWN")
    now_str = datetime.now(tz=_IST).strftime("%Y-%m-%d %H:%M:%S IST")

    print()
    print("=" * 64)
    print(f"  Trading Decision Support System  v{_VERSION}")
    print(f"  Started: {now_str}")
    print("=" * 64)
    print(f"  Active indices loaded : {len(active_indices)}")
    print(f"  F&O indices           : {len(fo_indices)}")
    print(f"  Market status         : {status_line}")
    print(f"  Mode                  : {'DRY RUN (no DB writes)' if dry_run else 'PRODUCTION'}")
    print(f"  Force-start           : {force_start}")
    print("-" * 64)
    print("  Scheduled jobs (next run times):")
    for job in collector.get_status()["jobs"]:
        nrt = job["next_run_time"] or "—"
        print(f"    [{job['id']:<25s}] {nrt}")
    print("=" * 64)
    print("  Press Ctrl+C to stop gracefully.")
    print()


def main() -> None:
    args = _parse_args()

    # ── Logging ────────────────────────────────────────────────────────────
    console_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(
        log_dir=settings.logging.log_dir,
        console_level=console_level,
        file_level=logging.DEBUG,
    )
    logger = logging.getLogger("dss.main")
    logger.info("Trading DSS Data Collector starting (v%s)", _VERSION)

    # ── PID Lock ─────────────────────────────────────────────────────────────
    pid_path = Path("data/trading_dss.pid")
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
            os.kill(old_pid, 0)
            print(f"\n  ❌ ERROR: Another instance is already running (PID {old_pid})")
            sys.exit(1)
        except (OSError, ValueError):
            pass
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))

    # ── Database ───────────────────────────────────────────────────────────
    try:
        db = get_db_manager()
        db.connect()
        db.initialise_schema()
        n_migrations = MigrationRunner(db).run_pending()
        if n_migrations:
            logger.info("Applied %d pending migration(s)", n_migrations)
    except Exception as exc:
        logger.critical("Database initialisation failed: %s — aborting", exc)
        sys.exit(1)

    # ── Index registry ─────────────────────────────────────────────────────
    try:
        registry = IndexRegistry.from_file(settings.indices_config_path)
        registry.sync_to_db(db)
        logger.info(
            "Registry loaded: %d active indices, %d F&O",
            len(registry.get_active_indices()),
            len(registry.get_indices_with_options()),
        )
    except Exception as exc:
        logger.critical("Registry initialisation failed: %s — aborting", exc)
        sys.exit(1)

    # ── Collector ─────────────────────────────────────────────────────────
    collector = DataCollector(
        db=db,
        registry=registry,
        dry_run=args.dry_run,
        force_start=args.force_start,
    )

    # ── Signal handlers ────────────────────────────────────────────────────
    def _shutdown(sig: object, frame: object) -> None:
        print()
        logger.info("Shutdown signal received — waiting for running jobs to finish…")
        collector.stop()
        db.close()
        if pid_path.exists():
            pid_path.unlink()
        logger.info("Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Start ──────────────────────────────────────────────────────────────
    collector.start()
    _print_banner(registry, collector, dry_run=args.dry_run, force_start=args.force_start)

    # Keep the main thread alive; APScheduler runs jobs on its own threads.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
