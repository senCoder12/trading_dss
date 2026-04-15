"""
Unified system launcher for the Trading Decision Support System.

Starts all components — data collector, decision engine, Telegram bot, and
FastAPI server — in co-ordinated daemon threads with graceful shutdown on
SIGINT / SIGTERM.

Usage
-----
::

    python scripts/run_system.py                     # everything
    python scripts/run_system.py --no-telegram        # skip Telegram bot
    python scripts/run_system.py --api-only           # dashboard / API only
    python scripts/run_system.py --collector-only     # data collection only
    python scripts/run_system.py --api-port 8080      # custom port
    python scripts/run_system.py --debug              # verbose logging
    python scripts/run_system.py --dry-run            # collect data, no DB writes
    python scripts/run_system.py --force-start        # ignore market-hours gate
    python scripts/run_system.py --skip-validation    # bypass config checks
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(*, debug: bool = False) -> None:
    """Configure unified logging for all threads."""
    from config.logging_config import setup_logging
    from config.settings import settings

    setup_logging(
        log_dir=settings.logging.log_dir,
        console_level=logging.DEBUG if debug else logging.INFO,
        file_level=logging.DEBUG,
    )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def _run_validation() -> bool:
    """
    Run all configuration checks without calling ``sys.exit``.

    Returns
    -------
    bool
        ``True`` if no fatal failures were found, ``False`` otherwise.
    """
    from scripts.validate_config import (
        Reporter,
        check_database,
        check_env,
        check_indices,
        check_news_mappings,
        check_rss_feeds,
        check_sentiment,
        check_system,
    )

    rpt = Reporter()
    check_system(rpt)
    check_env(rpt)
    check_indices(rpt)
    check_news_mappings(rpt)
    check_rss_feeds(rpt)
    check_sentiment(rpt)
    check_database(rpt)
    # Intentionally omit check_network — slow and non-fatal at startup.

    if rpt.warnings:
        print(f"  ⚠️  {rpt.warnings} warning(s) — check logs for details")

    if rpt.failed:
        print(f"\n  {rpt.failed} validation failure(s):")
        for err in rpt.errors:
            print(f"    - {err}")
        return False

    return True


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_banner(args: argparse.Namespace) -> None:
    """Print the startup banner with live market status."""
    from src.utils.market_hours import MarketHoursManager

    mh = MarketHoursManager()
    market_status = mh.get_market_status()
    status_str = market_status.get("status", "unknown").upper()
    is_open = market_status.get("status") == "open"
    status_emoji = "🟢" if is_open else "🔴"

    mode_parts: list[str] = []
    if not args.api_only:
        mode_parts.append("Collector")
    if not args.collector_only:
        mode_parts.append("API")
    if not args.no_telegram and not args.api_only:
        mode_parts.append("Telegram")
    mode_str = " + ".join(mode_parts) if mode_parts else "None"

    # Fixed-width box: inner width = 47 chars
    print("""
    ╔═══════════════════════════════════════════════╗
    ║    Trading Decision Support System v1.0        ║
    ║    Indian Markets — Zero Cost Edition          ║
    ╠═══════════════════════════════════════════════╣""")
    market_line = f"{status_emoji} {status_str}"
    print(f"    ║  Market : {market_line:<37}║")
    print(f"    ║  Mode   : {mode_str:<37}║")
    print(f"    ╚═══════════════════════════════════════════════╝")


def _print_status_summary(components: dict[str, Any], args: argparse.Namespace) -> None:
    """Print the list of active services after startup."""
    print("\n    Active Services:")
    lines: list[str] = []
    if "collector" in components:
        lines += [
            "    │  📡 Data Collector   (prices, options, VIX)",
            "    │  🧠 Decision Engine  (signals, regime)",
            "    │  📰 News Engine      (sentiment, impact)",
            "    │  ⚠️  Anomaly Detector (volume, OI, divergence)",
        ]
    if "telegram" in components:
        lines.append("    │  🤖 Telegram Bot    (alerts, commands)")
    if "api" in components:
        lines.append(f"    │  🌐 API + Dashboard  (port {args.api_port})")

    for i, line in enumerate(lines):
        # Replace last │ with └ for tree aesthetics
        print(line.replace("│", "└", 1) if i == len(lines) - 1 else line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, initialise components, and run forever."""

    # ── Argument parsing ─────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Trading Decision Support System — Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-telegram", action="store_true",
        help="Skip Telegram bot even if configured",
    )
    parser.add_argument(
        "--api-only", action="store_true",
        help="Start only the FastAPI server (no data collection)",
    )
    parser.add_argument(
        "--collector-only", action="store_true",
        help="Start only data collector (no API / dashboard)",
    )
    parser.add_argument(
        "--api-port", type=int, default=8000, metavar="PORT",
        help="Port for the FastAPI server (default: 8000)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level console logging",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Collect data but do not write signals to the database",
    )
    parser.add_argument(
        "--force-start", action="store_true",
        help="Bypass market-hours gate and run all jobs immediately",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip configuration validation (not recommended for production)",
    )
    args = parser.parse_args()

    # ── Logging ──────────────────────────────────────────────────────────────
    _setup_logging(debug=args.debug)

    # ── Banner ───────────────────────────────────────────────────────────────
    _print_banner(args)

    # ── Step 1: Validate configuration ───────────────────────────────────────
    if not args.skip_validation:
        print("\n[1/5] Validating configuration...")
        if not _run_validation():
            print(
                "\n  Configuration validation failed. Fix the issues above and retry.\n"
                "  Use --skip-validation to bypass (not recommended)."
            )
            sys.exit(1)
        print("  ✅ Configuration valid")
    else:
        print("\n[1/5] Skipping configuration validation (--skip-validation)")

    # ── Step 2: Initialise database ──────────────────────────────────────────
    print("\n[2/5] Initialising database...")
    db = None
    try:
        from src.database.db_manager import get_db_manager
        from src.database.migrations import MigrationRunner

        db = get_db_manager()
        db.connect()
        db.initialise_schema()
        applied = MigrationRunner(db).run_pending()
        if applied:
            print(f"  Applied {applied} pending migration(s)")
        print(f"  ✅ Database ready ({db.get_db_size()})")
    except Exception as exc:
        logger.critical("Database initialisation failed: %s — aborting", exc)
        sys.exit(1)

    # ── Step 3: Initialise components ────────────────────────────────────────
    print("\n[3/5] Initialising components...")
    components: dict[str, Any] = {}

    if not args.api_only:
        # Index registry — required by DataCollector
        registry = None
        try:
            from config.settings import settings
            from src.data.index_registry import IndexRegistry

            registry = IndexRegistry.from_file(settings.indices_config_path)
            registry.sync_to_db(db)
            logger.info(
                "Registry loaded: %d active, %d F&O",
                len(registry.get_active_indices()),
                len(registry.get_indices_with_options()),
            )
        except Exception as exc:
            logger.critical("Index registry initialisation failed: %s — aborting", exc)
            sys.exit(1)

        # Data Collector
        try:
            from src.data.data_collector import DataCollector

            collector = DataCollector(
                db=db,
                registry=registry,
                dry_run=args.dry_run,
                force_start=args.force_start,
            )
            components["collector"] = collector
            print("  ✅ Data Collector initialised")
        except Exception as exc:
            logger.error("Data Collector initialisation failed: %s", exc)
            # Non-fatal — continue without collector

        # Decision Engine
        engine = None
        try:
            from src.engine.decision_engine import DecisionEngine

            engine = DecisionEngine(db)
            components["engine"] = engine
            print("  ✅ Decision Engine initialised")
        except Exception as exc:
            logger.error("Decision Engine initialisation failed: %s", exc)

        # Wire engine into API dependency injection so /api/signals/* routes
        # can call it without a separate import cycle.
        if engine is not None:
            try:
                from src.api.dependencies import set_decision_engine
                set_decision_engine(engine)
            except Exception as exc:
                logger.warning("Could not wire Decision Engine into API: %s", exc)

        # Telegram Bot
        if not args.no_telegram:
            try:
                from src.alerts.telegram_bot import TelegramBot

                bot = TelegramBot(db, engine)
                if bot._configured:  # noqa: SLF001  (private but intentional)
                    components["telegram"] = bot
                    print("  ✅ Telegram Bot initialised")
                else:
                    print(
                        "  ⚠️  Telegram not configured "
                        "(set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)"
                    )
            except Exception as exc:
                logger.error("Telegram Bot initialisation failed: %s", exc)

    if not args.collector_only:
        components["api"] = True
        print(f"  ✅ API Server will start on port {args.api_port}")

    # ── Step 4: Start services ────────────────────────────────────────────────
    print("\n[4/5] Starting services...")
    threads: list[threading.Thread] = []

    # Data collector — APScheduler manages its own threads internally; the
    # wrapper thread just keeps the scheduler alive and logs crashes.
    if "collector" in components:
        def _run_collector() -> None:
            try:
                components["collector"].start()
                logger.info("Data Collector scheduler running")
                while True:
                    time.sleep(5)
            except Exception as exc:
                logger.error("Data Collector crashed: %s", exc, exc_info=True)

        t = threading.Thread(target=_run_collector, name="DataCollector", daemon=True)
        t.start()
        threads.append(t)
        print("  📡 Data Collector started")

    # Telegram Bot — runs its own polling event loop in a thread.
    if "telegram" in components:
        def _run_telegram() -> None:
            try:
                components["telegram"].start_bot()
                logger.info("Telegram Bot polling started")
            except Exception as exc:
                logger.error("Telegram Bot crashed: %s", exc, exc_info=True)

        t = threading.Thread(target=_run_telegram, name="TelegramBot", daemon=True)
        t.start()
        threads.append(t)
        print("  🤖 Telegram Bot started")

    # API Server — uvicorn blocks; run in a daemon thread so Ctrl+C still
    # reaches the main thread's signal handler.
    if "api" in components:
        try:
            import uvicorn
        except ImportError:
            logger.critical("uvicorn is not installed. Run: pip install uvicorn")
            sys.exit(1)

        def _run_api() -> None:
            try:
                uvicorn.run(
                    "src.api.app:app",
                    host="0.0.0.0",
                    port=args.api_port,
                    log_level="info" if args.debug else "warning",
                    access_log=args.debug,
                )
            except Exception as exc:
                logger.error("API Server crashed: %s", exc, exc_info=True)

        t = threading.Thread(target=_run_api, name="APIServer", daemon=True)
        t.start()
        threads.append(t)
        print(f"  🌐 API Server started   → http://localhost:{args.api_port}")
        print(f"  📊 Dashboard            → http://localhost:{args.api_port}")
        print(f"  📋 API Docs             → http://localhost:{args.api_port}/docs")

    # ── Step 5: Running ───────────────────────────────────────────────────────
    print("\n[5/5] System running!")
    _print_status_summary(components, args)
    print("\n  Press Ctrl+C to stop all services.\n")

    # ── Graceful shutdown ────────────────────────────────────────────────────
    def _shutdown(signum: object, frame: object) -> None:
        print("\n\nShutting down gracefully...")

        if "collector" in components:
            print("  Stopping Data Collector...")
            try:
                components["collector"].stop()
            except Exception as exc:
                logger.warning("Error stopping Data Collector: %s", exc)

        if "telegram" in components:
            print("  Stopping Telegram Bot...")
            try:
                components["telegram"].stop_bot()
            except Exception as exc:
                logger.warning("Error stopping Telegram Bot: %s", exc)

        if db is not None:
            try:
                db.close()
            except Exception:
                pass

        print("  ✅ All services stopped gracefully")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep the main thread alive; daemon threads die automatically when this
    # returns.  Periodically log a warning if any thread died unexpectedly.
    try:
        while True:
            time.sleep(10)
            for t in threads:
                if not t.is_alive():
                    logger.warning(
                        "Thread '%s' has stopped unexpectedly — "
                        "check logs for the root cause",
                        t.name,
                    )
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
