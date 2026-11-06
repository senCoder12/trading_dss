"""
Main orchestrator for all data collection operations.

Coordinates scrapers, fetchers, database interactions, and executes scheduled
data collection jobs using APScheduler.  Designed to run as a long-lived
background process during and around Indian market hours.
"""

# ================================================================
# TABLE OWNERSHIP: This class is the SOLE writer for:
#   - price_data
#   - options_chain_snapshot
#   - oi_aggregated
#   - vix_data
#   - fii_dii_activity
#   - system_health
#   - technical_indicators    (via IndicatorStore.save_analysis)
# Analysis components only READ from these tables.
# ================================================================

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.data.bse_scraper import BSEScraper
from src.data.data_router import DataRouter
from src.data.data_validator import validate_price_data
from src.data.fii_dii_data import FIIDIIFetcher
from src.data.historical_data import HistoricalDataManager
from src.data.index_registry import IndexRegistry
from src.data.nse_scraper import NSEScraper
from src.data.options_chain import OptionsChainFetcher
from src.data.rate_limiter import GlobalCircuitBreaker, GlobalRateLimiter, RateLimiter, create_nse_limiter
from src.data.vix_data import VIXTracker
from src.database import queries as Q
from src.database.db_manager import DatabaseManager
from src.utils.market_hours import MarketHoursManager
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# Threshold (% OI change) above which a spike is classified as HIGH severity
_SPIKE_HIGH_PCT = 50.0
_SPIKE_MEDIUM_PCT = 25.0


class DataCollector:
    """
    Coordinates and schedules all data fetching tasks for the Trading DSS.

    Parameters
    ----------
    db:
        Open :class:`~src.database.db_manager.DatabaseManager` instance.
    registry:
        Loaded :class:`~src.data.index_registry.IndexRegistry`.
    dry_run:
        If ``True``, fetch data but do not write to the database.
    force_start:
        If ``True``, run market-hours-gated jobs regardless of current time.
    """

    def __init__(
        self,
        db: DatabaseManager,
        registry: IndexRegistry,
        dry_run: bool = False,
        force_start: bool = False,
        fo_manager: Optional[Any] = None,
    ) -> None:
        self.db = db
        self.registry = registry
        self.dry_run = dry_run
        self.force_start = force_start

        self.market_hours = MarketHoursManager()

        # Shared global rate limiter — all NSE components share one instance
        self._nse_limiter = GlobalRateLimiter.get("nseindia.com", max_requests=25, window_seconds=60)
        self.nse_scraper = NSEScraper(rate_limiter=self._nse_limiter)
        self.bse_scraper = BSEScraper()

        # DataRouter: uses Angel One WebSocket when configured, otherwise wraps nse_scraper
        self.data_router = DataRouter(db=self.db, registry=self.registry, dry_run=self.dry_run)
        logger.info("DataRouter source: %s", self.data_router.get_data_source())
        self.options_fetcher = OptionsChainFetcher(
            scraper=self.nse_scraper,
            bse_scraper=self.bse_scraper,
            broker_client=self.data_router.broker_client
        )
        self.historical_manager = HistoricalDataManager(registry=self.registry, db=self.db)
        self.fii_dii_fetcher = FIIDIIFetcher(scraper=self.nse_scraper)
        self.vix_tracker = VIXTracker(scraper=self.nse_scraper)
        from src.analysis.news.news_engine import NewsEngine
        self.news_engine = NewsEngine(self.db)

        # Wired lazily via set_overnight_engine() and set_decision_engine()
        self.overnight_engine: Any = None
        self.decision_engine: Any = None
        self.telegram_bot: Any = None
        self.fo_manager: Optional[Any] = fo_manager

        self.scheduler = BackgroundScheduler(timezone=_IST)
        self._setup_jobs()

        # Create new tables introduced after initial schema migration
        try:
            self.db.execute(Q.CREATE_FUTURES_BASIS)
            self.db.execute(Q.CREATE_IV_HISTORY)
            self.db.execute(Q.CREATE_MARKET_BREADTH)
            self.db.execute(Q.CREATE_GIFT_NIFTY)
        except Exception as e:
            logger.warning("New table creation failed (may already exist): %s", e)

        # If manager provided, do initial refresh
        if self.fo_manager:
            self.refresh_registry_with_stocks()

        # Health / failure tracking ─────────────────────────────────────────
        self._failures: Dict[str, int] = {
            "nse": 0, "bse": 0, "options": 0, "vix": 0,
        }
        # Monotonic timestamp of last auto-recovery attempt per component
        self._last_recovery: Dict[str, float] = {}

        # Signal generation heartbeat monitoring
        self._last_signal_time: float = time.monotonic()
        self._last_heartbeat_log: float = time.monotonic()

        self.health_status: Dict[str, Any] = {
            "last_successful_price_fetch": None,
            "last_vix_regime": None,
        }

    # ── Scheduler setup ────────────────────────────────────────────────────

    def _setup_jobs(self) -> None:
        """Register all APScheduler jobs with production-correct triggers."""
        p = settings.polling

        # ── Interval jobs (market hours only) ─────────────────────────────
        self.scheduler.add_job(
            self.collect_index_prices,
            trigger=IntervalTrigger(seconds=p.price_interval, timezone=_IST),
            id="collect_prices",
            name="NSE Index Prices",
            max_instances=1,
            coalesce=True,
            next_run_time=datetime.now(),
        )
        self.scheduler.add_job(
            self.collect_options_chain,
            trigger=IntervalTrigger(seconds=p.options_interval, timezone=_IST),
            id="collect_options",
            name="NSE Options Chain",
            max_instances=1,
            coalesce=True,
            next_run_time=datetime.now(),
        )
        self.scheduler.add_job(
            self.collect_vix,
            trigger=IntervalTrigger(seconds=p.vix_interval, timezone=_IST),
            id="collect_vix",
            name="Collect VIX Data",
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._collect_news,
            trigger=IntervalTrigger(seconds=p.news_interval, timezone=_IST),
            id="collect_news",
            name="Collect and Analyze News",
            max_instances=1,
            coalesce=True,
            next_run_time=datetime.now(),
        )
        self.scheduler.add_job(
            self.collect_technical_indicators,
            trigger=IntervalTrigger(seconds=p.technical_interval, timezone=_IST),
            id="collect_technical_indicators",
            name="Compute Technical Indicators",
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self.health_check,
            trigger=IntervalTrigger(seconds=300, timezone=_IST),
            id="health_check",
            name="System Health Check",
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._run_decision_cycle,
            trigger=IntervalTrigger(seconds=p.decision_interval if hasattr(p, "decision_interval") else 300, timezone=_IST),
            id="decision_cycle",
            name="Run Decision Engine Cycle",
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._monitor_positions,
            trigger=IntervalTrigger(seconds=60, timezone=_IST),
            id="monitor_positions",
            name="Monitor Open Positions",
            max_instances=1,
            coalesce=True,
        )

        # ── Cron jobs (run at fixed times) ────────────────────────────────
        self.scheduler.add_job(
            self.warm_up_sessions,
            trigger=CronTrigger(hour=9, minute=0, timezone=_IST),
            id="warm_up",
            name="Pre-market Session Warm-up",
        )
        self.scheduler.add_job(
            self.collect_fii_dii,
            trigger=CronTrigger(hour=18, minute=0, timezone=_IST),
            id="collect_fii_dii",
            name="Collect FII/DII Data",
        )
        self.scheduler.add_job(
            self.update_historical,
            trigger=CronTrigger(hour=18, minute=30, timezone=_IST),
            id="update_historical",
            name="Update Historical Data",
        )
        self.scheduler.add_job(
            self.cleanup,
            trigger=CronTrigger(hour=20, minute=0, timezone=_IST),
            id="cleanup",
            name="Daily Database Cleanup",
        )

        # ── F&O Stock Registry Sync ───────────────────────────────────────
        # Keeps the IndexRegistry in sync with the Top 15 Priority Stocks
        # so they can be processed by the DecisionEngine.
        self.scheduler.add_job(
            self.refresh_registry_with_stocks,
            trigger=CronTrigger(hour=9, minute=1, timezone=_IST),
            id="refresh_stock_registry",
            name="Refresh Registry with Priority Stocks",
        )
        self.scheduler.add_job(
            self.refresh_registry_with_stocks,
            trigger=IntervalTrigger(minutes=15, timezone=_IST),
            id="refresh_stock_registry_intraday",
            name="Intraday Stock Registry Sync",
        )

        # ── Overnight news pipeline (all times IST) ────────────────────────
        # These jobs run only when set_overnight_engine() has wired in an
        # OvernightNewsEngine instance. Until then, they are no-ops.
        self.scheduler.add_job(
            self._overnight_post_market,
            trigger=CronTrigger(hour=16, minute=0, day_of_week="mon-fri", timezone=_IST),
            id="overnight_post_market",
            name="Overnight: Post-Market Collection",
        )
        self.scheduler.add_job(
            self._overnight_evening_data,
            trigger=CronTrigger(hour=18, minute=30, day_of_week="mon-fri", timezone=_IST),
            id="overnight_evening_data",
            name="Overnight: Evening Data Collection",
        )
        self.scheduler.add_job(
            self._overnight_evening_retry,
            trigger=CronTrigger(hour=19, minute=30, day_of_week="mon-fri", timezone=_IST),
            id="overnight_evening_retry",
            name="Overnight: FII/DII Retry",
        )
        self.scheduler.add_job(
            self._overnight_us_market_close,
            trigger=CronTrigger(hour=1, minute=45, day_of_week="tue-sat", timezone=_IST),
            id="overnight_us_market_close",
            name="Overnight: US Market Close",
        )
        self.scheduler.add_job(
            self._overnight_asian_open,
            trigger=CronTrigger(hour=6, minute=45, day_of_week="mon-fri", timezone=_IST),
            id="overnight_asian_open",
            name="Overnight: Asian Market Open",
        )
        self.scheduler.add_job(
            self._overnight_pre_market_final,
            trigger=CronTrigger(hour=8, minute=45, day_of_week="mon-fri", timezone=_IST),
            id="overnight_pre_market_final",
            name="Overnight: Pre-Market Compile",
        )
        self.scheduler.add_job(
            self._overnight_send_report,
            trigger=CronTrigger(hour=8, minute=50, day_of_week="mon-fri", timezone=_IST),
            id="overnight_send_report",
            name="Overnight: Pre-Market Report Dispatch",
        )

        # ── New market data collection jobs ───────────────────────────────
        # Futures basis collection — every 5 min during market hours
        self.scheduler.add_job(
            self.collect_futures_basis,
            trigger=IntervalTrigger(seconds=300, timezone=_IST),
            id="collect_futures_basis",
            name="Collect NIFTY Futures Basis",
            max_instances=1,
            coalesce=True,
        )
        # IV history snapshot — every 5 min during market hours
        self.scheduler.add_job(
            self.collect_iv_history,
            trigger=IntervalTrigger(seconds=300, timezone=_IST),
            id="collect_iv_history",
            name="Collect ATM IV History",
            max_instances=1,
            coalesce=True,
        )
        # FII futures positions (EOD archive) — gap-fill check every 30 min
        self.scheduler.add_job(
            self.collect_fii_futures_intraday,
            trigger=IntervalTrigger(minutes=30, timezone=_IST),
            id="collect_fii_futures",
            name="Collect FII Futures Positions",
            max_instances=1,
            coalesce=True,
        )
        # FII cash + futures freshness → Telegram alert when stale/missing
        self.scheduler.add_job(
            self.check_data_freshness,
            trigger=IntervalTrigger(minutes=30, timezone=_IST),
            id="check_data_freshness",
            name="Check FII Data Freshness",
            max_instances=1,
            coalesce=True,
        )
        # Market breadth (AD ratio) — every 5 min during market hours
        self.scheduler.add_job(
            self.collect_market_breadth,
            trigger=IntervalTrigger(seconds=300, timezone=_IST),
            id="collect_market_breadth",
            name="Collect Market Breadth AD Ratio",
            max_instances=1,
            coalesce=True,
        )
        # GIFT NIFTY pre-market — runs at 9:00 AM and 9:10 AM
        self.scheduler.add_job(
            self.collect_gift_nifty_premarket,
            trigger=CronTrigger(hour=9, minute=0, timezone=_IST),
            id="collect_gift_nifty_0900",
            name="GIFT NIFTY Pre-market 9:00",
        )
        self.scheduler.add_job(
            self.collect_gift_nifty_premarket,
            trigger=CronTrigger(hour=9, minute=10, timezone=_IST),
            id="collect_gift_nifty_0910",
            name="GIFT NIFTY Pre-market 9:10",
        )

    # ── Overnight engine hooks ─────────────────────────────────────────────

    def set_overnight_engine(self, engine: Any) -> None:
        """Wire in an :class:`OvernightNewsEngine` instance.

        The scheduled overnight jobs become active as soon as this is set.
        """
        self.overnight_engine = engine
        logger.info("DataCollector: overnight engine wired in")

    def set_decision_engine(self, engine: Any) -> None:
        """Wire in a :class:`DecisionEngine` instance.

        The decision cycle jobs become active as soon as this is set.
        """
        self.decision_engine = engine
        logger.info("DataCollector: decision engine wired in")

    def set_telegram_bot(self, bot: Any) -> None:
        """Wire in the Telegram bot used for pre-market report dispatch."""
        self.telegram_bot = bot

    def _overnight_post_market(self) -> None:
        if self.overnight_engine is None:
            return
        status = self.market_hours.get_market_status()
        if status.get("is_holiday"):
            logger.info("Skipping post-market overnight job (market holiday)")
            return
        try:
            self.overnight_engine.collect_post_market()
        except Exception:
            logger.exception("Overnight: post_market job failed")

    def _overnight_evening_data(self) -> None:
        if self.overnight_engine is None:
            return
        status = self.market_hours.get_market_status()
        if status.get("is_holiday"):
            logger.info("Skipping evening data overnight job (market holiday)")
            return
        try:
            self.overnight_engine.collect_evening_data()
        except Exception:
            logger.exception("Overnight: evening_data job failed")

    def _overnight_evening_retry(self) -> None:
        if self.overnight_engine is None:
            return
        try:
            self.overnight_engine.collect_evening_retry()
        except Exception:
            logger.exception("Overnight: evening_retry job failed")

    def _overnight_us_market_close(self) -> None:
        if self.overnight_engine is None:
            return
        try:
            self.overnight_engine.collect_us_market_close()
        except Exception:
            logger.exception("Overnight: us_market_close job failed")

    def _overnight_asian_open(self) -> None:
        if self.overnight_engine is None:
            return
        try:
            self.overnight_engine.collect_asian_open()
        except Exception:
            logger.exception("Overnight: asian_open job failed")

    def _overnight_pre_market_final(self) -> None:
        if self.overnight_engine is None:
            return
        # Skip compile on holidays — no market tomorrow means no report needed.
        today = self.market_hours.now_ist().date() if hasattr(
            self.market_hours, "now_ist"
        ) else datetime.now(tz=_IST).date()
        try:
            from src.utils.market_hours import is_trading_day
            if not is_trading_day(today):
                logger.info("Skipping pre-market compile (non-trading day: %s)", today)
                return
        except Exception:
            pass
        try:
            self.overnight_engine.collect_pre_market_final()
        except Exception:
            logger.exception("Overnight: pre_market_final job failed")

    def _overnight_send_report(self) -> None:
        """Dispatch the pre-market intelligence report via Telegram (08:50 IST)."""
        if self.overnight_engine is None:
            return
        try:
            from src.utils.market_hours import is_trading_day
            today = datetime.now(tz=_IST).date()
            if not is_trading_day(today):
                return
        except Exception:
            pass

        try:
            from src.analysis.news.pre_market_report import PreMarketReportGenerator
            from src.analysis.news.overnight_engine import load_latest_overnight_data

            data = load_latest_overnight_data(self.db)
            report = PreMarketReportGenerator().generate_report(data)
            logger.info("Pre-market report generated (%d chars)", len(report))

            if self.telegram_bot is not None:
                try:
                    self.telegram_bot.send_alert(report, priority="HIGH")
                except Exception:
                    logger.exception("Pre-market report: Telegram dispatch failed")
            else:
                logger.info("Pre-market report (Telegram not wired):\n%s", report)
        except Exception:
            logger.exception("Overnight: send_report job failed")

    def _collect_news(self) -> None:
        """Fetch, parse, and analyze news articles from all active RSS feeds."""
        logger.info("Starting news collection cycle...")
        if not self._should_run():
            return
        try:
            result = self.news_engine.run_news_cycle()
            logger.info(
                "News cycle: %d new articles, %d alerts",
                result.articles_new, len(result.alerts),
            )
            # Dispatch critical alerts to Telegram
            if self.telegram_bot is not None:
                for alert in result.alerts:
                    if alert.severity in ("CRITICAL", "HIGH"):
                        self.telegram_bot.send_alert(alert.message)
        except Exception as exc:
            logger.error("News collection failed: %s", exc)

    def _run_decision_cycle(self) -> None:
        """Scheduled task: Run full decision cycle for all indices.

        Includes heartbeat monitoring to detect signal generation stalls:
        - Log heartbeat every 30 seconds if no new signals
        - Alert (HIGH severity) if > 60 seconds without actionable signals
        """
        if self.decision_engine is None or not self._should_run():
            return

        logger.info("Starting scheduled decision cycle...")
        try:
            results = self.decision_engine.run_all_indices()
            actionable = [r for r in results if r.is_actionable]

            # Update signal generation timestamp if actionable signals found
            if actionable:
                self._last_signal_time = time.monotonic()

            # Heartbeat monitoring: log every 30s if stale, alert if > 60s
            now = time.monotonic()
            elapsed_since_signal = now - self._last_signal_time
            elapsed_since_log = now - self._last_heartbeat_log

            if elapsed_since_signal > 60:
                # Critical: no signals for > 60 seconds
                logger.critical(
                    "Signal generation stalled: no actionable signals in %.1f seconds",
                    elapsed_since_signal,
                )
                if self.telegram_bot:
                    self.telegram_bot.send_alert(
                        f"CRITICAL: No trading signals generated in {int(elapsed_since_signal)}s. "
                        "Check data sources and decision engine.",
                        priority="CRITICAL",
                    )
                self._last_heartbeat_log = now
            elif elapsed_since_log > 30:
                # Regular heartbeat: log every 30s if no signals
                if elapsed_since_signal > 0:
                    logger.info(
                        "Heartbeat: %d seconds since last actionable signal",
                        int(elapsed_since_signal),
                    )
                    self._last_heartbeat_log = now

            # Send alerts for actionable signals
            if actionable and self.telegram_bot:
                for res in actionable:
                    self.telegram_bot.send_alert(res.alert_message, priority="HIGH")

            logger.info("Decision cycle complete: %d signals, %d actionable", len(results), len(actionable))
        except Exception as exc:
            logger.error("Decision cycle failed: %s", exc, exc_info=True)

    def _monitor_positions(self) -> None:
        """Scheduled task: Monitor open positions for SL/Targets."""
        if self.decision_engine is None or not self._should_run():
            return

        try:
            self.decision_engine.monitor_open_positions()
        except Exception as exc:
            logger.error("Position monitoring failed: %s", exc)

    # ── Failure / recovery helpers ─────────────────────────────────────────

    def _handle_failure(self, component: str, exc: Exception) -> None:
        """
        Increment failure counter, log, and attempt auto-recovery when the
        threshold is reached — without blocking the scheduler thread.
        """
        self._failures[component] = self._failures.get(component, 0) + 1
        n = self._failures[component]
        logger.error("Component '%s' failed (%d consecutive): %s", component, n, exc)

        if n >= 10:
            now = time.monotonic()
            last = self._last_recovery.get(component, 0.0)
            if now - last >= 60.0:
                self._last_recovery[component] = now
                logger.critical(
                    "Auto-recovering '%s' after %d consecutive failures", component, n
                )
                if component == "nse":
                    self.nse_scraper = NSEScraper(rate_limiter=self._nse_limiter)
                    self.options_fetcher = OptionsChainFetcher(scraper=self.nse_scraper, bse_scraper=self.bse_scraper)
                    self.fii_dii_fetcher = FIIDIIFetcher(scraper=self.nse_scraper)
                    self.vix_tracker = VIXTracker(scraper=self.nse_scraper)
                elif component == "bse":
                    self.bse_scraper = BSEScraper()
                    self.options_fetcher = OptionsChainFetcher(scraper=self.nse_scraper, bse_scraper=self.bse_scraper)

    def _reset_failure(self, component: str) -> None:
        self._failures[component] = 0

    def refresh_registry_with_stocks(self) -> None:
        """Fetch priority stocks from FOUniverseManager and register them in IndexRegistry.
        
        This makes the Top 15 liquid F&O stocks visible to the collection jobs
        (price, options) and the decision engine.
        """
        if not self.fo_manager:
            return
            
        try:
            symbols = self.fo_manager.get_priority_symbols()
            if not symbols:
                logger.debug("No priority symbols found to refresh registry")
                return
                
            # Fetch stock details from fo_universe table
            stocks = []
            for sym in symbols:
                s = self.db.fetch_one(Q.GET_FO_UNIVERSE_STOCK, (sym,))
                if s:
                    stocks.append(s)
                    
            if stocks:
                # Remove old dynamic indices first to keep it clean
                removed = self.registry.remove_dynamic_indices()
                count = self.registry.register_stocks_as_indices(stocks)
                
                logger.info(
                    "DataCollector: registry refreshed (%d removed, %d registered)",
                    removed, count
                )
                
                # Update DataRouter with the new registry view
                self.data_router.registry = self.registry
                
                # If using broker API, we need to re-subscribe to include new stock tokens
                if self.data_router.is_using_broker_api():
                    logger.info("DataCollector: re-initializing WebSocket feed for new stocks")
                    self.data_router._start_websocket_feed()
        except Exception as exc:
            logger.error("DataCollector: failed to refresh registry with stocks: %s", exc)

    # ── Market-hours guard ─────────────────────────────────────────────────

    def _should_run(self) -> bool:
        """Return True when market-hours-gated jobs are allowed to execute."""
        if self.force_start:
            return True
        status = self.market_hours.get_market_status()
        return bool(status.get("status") in ("open", "pre_market"))

    # ── Job definitions ────────────────────────────────────────────────────

    def warm_up_sessions(self) -> None:
        """9:00 AM — seed NSE/BSE cookies before market opens at 9:15."""
        status = self.market_hours.get_market_status()
        if status.get("is_holiday") or status.get("is_weekend"):
            logger.info("Skipping pre-market warm-up (holiday / weekend)")
            return
        GlobalCircuitBreaker.reset_all()
        logger.info("Pre-market warm-up: seeding NSE/BSE sessions")
        try:
            self.nse_scraper.get_all_indices()
        except Exception as exc:
            logger.warning("NSE warm-up failed: %s", exc)
        try:
            self.bse_scraper.get_all_indices()
        except Exception as exc:
            logger.warning("BSE warm-up failed: %s", exc)

    def collect_index_prices(self) -> None:
        """
        Every ``price_interval`` seconds — fetch all NSE indices in one call,
        validate, and batch-upsert into ``price_data``.

        When the broker WebSocket is active, ticks are written to DB in real time
        by DataRouter._handle_tick().  This job then becomes a lightweight health
        probe that verifies the feed is flowing.

        Falls back to BSE scraper if NSE returns nothing in polling mode.
        """
        if not self._should_run():
            return

        # ── WebSocket / broker mode ────────────────────────────────────────
        if self.data_router.is_using_broker_api():
            try:
                now = datetime.now(tz=_IST)
                live = self.data_router.get_all_prices()
                logger.debug(
                    "WebSocket feed active: %d instruments live (%s)",
                    len(live),
                    self.data_router.get_data_source(),
                )
                if live:
                    self.health_status["last_successful_price_fetch"] = now
                    self._reset_failure("nse")
            except Exception as exc:
                self._handle_failure("nse", exc)
            return

        # ── Scraper / polling mode (original logic) ────────────────────────
        try:
            now = datetime.now(tz=_IST)
            try:
                nse_data: list[dict] = self.nse_scraper.get_all_indices() or []
            except Exception as e:
                logger.warning("NSE get_all_indices failed: %s — trying BSE fallback", e)
                self._handle_failure("nse", e)
                nse_data = []

            bse_data: list[dict] = []
            if not nse_data:
                bse_data = self.bse_scraper.get_all_indices() or []

            # Build O(1) lookup tables keyed by normalised index name
            # NSE normalizer stores the NSE API symbol in "index_name"
            # e.g. "NIFTY 50", "NIFTY BANK" — matches Index.nse_symbol
            nse_lookup: dict[str, dict] = {
                item["index_name"].upper(): item for item in nse_data
            }
            # BSE normalizer stores names like "SENSEX", "BSE 100"
            bse_lookup: dict[str, dict] = {
                item["index_name"].upper(): item for item in bse_data
            }

            inserts: list[tuple] = []
            collected = 0

            for idx in self.registry.get_active_indices():
                # Try NSE first (by nse_symbol), then BSE (by id)
                item = (
                    nse_lookup.get((idx.nse_symbol or "").upper())
                    or bse_lookup.get(idx.id.upper())
                )
                if item is None:
                    continue

                ltp = item.get("ltp", 0.0)
                price_dict = {
                    "ltp":    ltp,
                    "open":   item.get("open", 0.0),
                    "high":   item.get("high", 0.0),
                    "low":    item.get("low", 0.0),
                    # NSE returns "close" as previous day's close, not today's LTP.
                    # For live 1m tick data the current close IS the LTP.
                    "close":  ltp,
                    "volume": item.get("volume", 0.0),
                }
                result = validate_price_data(price_dict)
                if not result.is_valid or not result.cleaned_data:
                    logger.debug(
                        "Skipping invalid price for %s: %s", idx.id, result.errors
                    )
                    continue

                cd = result.cleaned_data
                source = "BSE" if item in bse_data else "NSE"
                inserts.append((
                    idx.id, now.isoformat(),
                    cd["open"], cd["high"], cd["low"], cd["close"],
                    float(price_dict["volume"]), 0.0,  # vwap not available from tick
                    source, "1m",
                ))
                collected += 1

            if not self.dry_run and inserts:
                self.db.execute_many(Q.UPSERT_PRICE_DATA, inserts)

            logger.info("Collected prices for %d indices", collected)
            self.health_status["last_successful_price_fetch"] = now
            if nse_data:
                self._reset_failure("nse")

        except Exception as exc:
            self._handle_failure("nse", exc)

    def collect_options_chain(self) -> None:
        """
        Every ``options_interval`` seconds — fetch chain for each F&O index,
        compute OI summary, detect buildups/spikes, and persist.
        """
        if not self._should_run():
            return

        try:
            fo_indices = self.registry.get_indices_with_options()
            for idx in fo_indices:
                sym = idx.option_symbol
                if not sym:
                    continue
                try:
                    chain = self.options_fetcher.get_options_chain(sym)
                    if not chain:
                        continue

                    # Compute OI summary from the already-fetched chain — no extra HTTP call
                    summary = self.options_fetcher._compute_oi_summary(chain)
                    if summary:
                        logger.info(
                            "Options chain updated for %s — PCR: %.2f, Max Pain: %.0f",
                            idx.id, summary.pcr, summary.max_pain_strike,
                        )

                    if not self.dry_run:
                        self.options_fetcher.save_to_db(chain, self.db)

                    from src.api.websocket import queue_event
                    queue_event("options_update", {
                        "index_id": idx.id,
                        "expiry_date": chain.expiry_date,
                        "timestamp": chain.timestamp,
                    })


                    # ── OI buildup / spike detection ──────────────────────
                    # NOTE: Spike detection runs here but persistence is
                    # handled by AlertManager (sole writer for anomaly_events).
                    # DataCollector logs spikes; the anomaly subsystem
                    # picks them up via OIAnomalyDetector → AlertManager.
                    snaps = self.options_fetcher.get_memory_snapshots(
                        sym, chain.expiry_date
                    )
                    if len(snaps) >= 2:
                        prev_chain = snaps[0]
                        spikes = self.options_fetcher.detect_oi_spikes(chain, prev_chain)
                        for spike in spikes:
                            pct = abs(spike.change_pct)
                            severity = (
                                "HIGH" if pct >= _SPIKE_HIGH_PCT
                                else "MEDIUM" if pct >= _SPIKE_MEDIUM_PCT
                                else "LOW"
                            )
                            logger.info(
                                "OI spike detected: %s %s strike=%.0f "
                                "change=%.1f%% severity=%s",
                                idx.id, spike.option_type,
                                spike.strike_price, spike.change_pct,
                                severity,
                            )

                except Exception as exc:
                    logger.error("Options chain failed for %s: %s", idx.id, exc)

                time.sleep(5)  # stagger requests; respects NSE rate limits

            self._reset_failure("options")

        except Exception as exc:
            self._handle_failure("options", exc)

    def collect_vix(self) -> None:
        """Every ``vix_interval`` seconds — fetch India VIX and log regime shifts."""
        if not self._should_run():
            return

        try:
            vix_data = self.vix_tracker.get_current_vix()
            if not vix_data:
                return

            # Compute regime directly from fetched value — avoids a redundant scraper call
            t = settings.thresholds
            v = vix_data.value
            if v < t.vix_normal_threshold:
                regime = "LOW_VOL"
            elif v < t.vix_elevated_threshold:
                regime = "NORMAL"
            elif v < t.vix_panic_threshold:
                regime = "ELEVATED"
            else:
                regime = "HIGH_VOL"

            if regime != self.health_status.get("last_vix_regime"):
                logger.info(
                    "VIX regime shift: %s → %s  (VIX = %.2f)",
                    self.health_status.get("last_vix_regime", "—"), regime, v,
                )
                self.health_status["last_vix_regime"] = regime

            if not self.dry_run:
                self.vix_tracker.save_to_db(vix_data, self.db)

            self._reset_failure("vix")

        except Exception as exc:
            self._handle_failure("vix", exc)

    def collect_technical_indicators(self) -> None:
        """
        Recompute technical indicators for every F&O index and persist them
        to ``technical_indicators``.

        Reads the most recent daily bars out of ``price_data``, runs the
        full ``TechnicalAggregator`` pipeline, and upserts via
        ``IndicatorStore``. Runs on a fixed interval (``technical_interval``)
        — cheap because all inputs are local DB reads, no network calls.
        """
        if not self._should_run():
            return

        try:
            import pandas as pd
            from src.analysis.indicator_store import IndicatorStore
            from src.analysis.technical_aggregator import TechnicalAggregator

            aggregator = TechnicalAggregator()
            fo_indices = self.registry.get_indices_with_options()
            if not fo_indices:
                return

            # Latest VIX for the aggregator's vol context — a single read.
            vix_row = self.db.fetch_one(
                "SELECT vix_value FROM vix_data ORDER BY timestamp DESC LIMIT 1"
            )
            vix_value = float(vix_row["vix_value"]) if vix_row else None

            persisted = 0
            for idx in fo_indices:
                try:
                    rows = self.db.fetch_all(
                        "SELECT timestamp, open, high, low, close, volume "
                        "FROM price_data "
                        "WHERE index_id = ? AND timeframe = '1d' "
                        "ORDER BY timestamp DESC LIMIT 200",
                        (idx.id,),
                    )
                    # Need at least a few bars for any aggregator output. Below
                    # this we skip — but log visibly so data gaps don't hide.
                    if len(rows) < 5:
                        logger.warning(
                            "Skipping indicators for %s — only %d daily bars "
                            "(yfinance history limited for this ticker)",
                            idx.id, len(rows),
                        )
                        continue

                    df = pd.DataFrame([dict(r) for r in reversed(rows)])
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")

                    result = aggregator.analyze(
                        index_id=idx.id,
                        price_df=df,
                        vix_value=vix_value,
                        timeframe="1d",
                    )
                    if not self.dry_run:
                        IndicatorStore.save_analysis(result, self.db)
                    persisted += 1
                except Exception:
                    logger.exception(
                        "Technical indicators failed for %s", idx.id,
                    )

            if persisted:
                logger.info("Technical indicators computed for %d indices", persisted)

        except Exception as exc:
            logger.error("Technical indicators job failed: %s", exc)

    def collect_fii_dii(self) -> None:
        """18:00 IST — fetch today's FII/DII data after market close."""
        try:
            fii_data = self.fii_dii_fetcher.fetch_today_fii_dii()
            if not fii_data:
                logger.warning("FII/DII: no data returned from NSE")
                return

            trend = self.fii_dii_fetcher.get_fii_trend(days=5)
            consecutive = trend["consecutive_sell_days"] or trend["consecutive_buy_days"]
            logger.info(
                "FII net: %.0f Cr (%s trend, %d consecutive days)",
                fii_data.fii_net_value,
                trend["trend"].lower(),
                consecutive,
            )

            if not self.dry_run:
                self.fii_dii_fetcher.save_to_db(fii_data, self.db)

        except Exception as exc:
            logger.error("FII/DII collection failed: %s", exc)

    def collect_futures_basis(self) -> None:
        """Collect NIFTY futures LTP and compute basis vs spot price.

        Basis = futures LTP - spot. Positive = contango (bullish).
        Negative/collapsing = bearish. Stored every 5 minutes.
        """
        if not self._should_run():
            return
        try:
            now = datetime.now(tz=_IST)
            # NIFTY spot
            spot_data = self.nse_scraper.get_index_quote("NIFTY 50")
            if not spot_data:
                return
            spot = float(spot_data.get("ltp", 0))
            if spot <= 0:
                return

            # NIFTY nearest futures (approximate from options chain meta)
            # Get the nearest expiry futures price from NSE
            # Try to get NIFTY FUT from equity_fno endpoint
            fno_data = self.nse_scraper.get_all_fno_quotes()
            futures_ltp = 0.0
            expiry_str = ""

            if fno_data:
                for item in fno_data:
                    sym = str(item.get("symbol", "")).upper()
                    if "NIFTY" in sym and "FUT" in sym and "BANK" not in sym:
                        futures_ltp = float(item.get("ltp", 0))
                        expiry_str = item.get("expiryDate", "")
                        if futures_ltp > 0:
                            break

            # Fallback: estimate futures from options chain (near-ATM CE - PE + spot)
            if futures_ltp <= 0:
                futures_ltp = spot  # neutral fallback

            basis = futures_ltp - spot
            basis_pct = basis / spot * 100 if spot > 0 else 0.0

            # Fair value basis: spot × (1 + risk_free_rate × days_to_expiry/365)
            # Approximate: risk_free_rate = 7%, find days to expiry
            days_to_expiry = 7  # default weekly
            fair_value_basis = spot * 0.07 * days_to_expiry / 365

            # Signal: basis expanding = bullish, collapsing/negative = bearish
            if basis > fair_value_basis * 1.5:
                basis_signal = "STRONG_BULLISH"
            elif basis > 0:
                basis_signal = "BULLISH"
            elif basis < -10:
                basis_signal = "STRONG_BEARISH"
            elif basis < 0:
                basis_signal = "BEARISH"
            else:
                basis_signal = "NEUTRAL"

            if not self.dry_run:
                self.db.execute(Q.UPSERT_FUTURES_BASIS, (
                    now.isoformat(), "NIFTY50",
                    round(spot, 2), round(futures_ltp, 2),
                    round(basis, 2), round(basis_pct, 4),
                    round(fair_value_basis, 2), expiry_str,
                    days_to_expiry, basis_signal,
                ))
            logger.debug("Futures basis: %.1f pts (%s) for NIFTY", basis, basis_signal)
        except Exception as exc:
            logger.warning("collect_futures_basis failed: %s", exc)

    def collect_iv_history(self) -> None:
        """Snapshot ATM IV from the latest options chain for IV percentile tracking.

        Stored every 5 minutes. Used by options_analysis.calculate_iv_percentile().
        """
        if not self._should_run():
            return
        try:
            now = datetime.now(tz=_IST)
            for index_id in ("NIFTY50", "BANKNIFTY"):
                try:
                    # Nearest live expiry. The snapshot table keeps every expiry
                    # ever stored, so an unfiltered "ORDER BY expiry LIMIT 1"
                    # picked a long-expired contract. Today's expiry is dead
                    # after 15:30 IST.
                    today = now.date().isoformat()
                    expiries = [
                        r["expiry_date"]
                        for r in self.db.fetch_all(Q.LIST_ACTIVE_EXPIRY_DATES, (index_id, today))
                    ]
                    if expiries and expiries[0] == today and (now.hour, now.minute) >= (15, 30):
                        expiries = expiries[1:]
                    if not expiries:
                        continue
                    expiry = expiries[0]

                    # Get ATM strikes
                    spot_row = self.db.fetch_one(
                        "SELECT close FROM price_data WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1",
                        (index_id,)
                    )
                    if not spot_row:
                        continue
                    spot = float(spot_row["close"])
                    interval = 100 if "BANK" in index_id else 50
                    atm = round(spot / interval) * interval

                    # Get ATM CE and PE IV
                    rows = self.db.fetch_all(
                        """SELECT option_type, iv, ltp, open_interest FROM options_chain_snapshot
                           WHERE index_id = ? AND expiry_date = ? AND strike_price = ?
                           ORDER BY timestamp DESC LIMIT 2""",
                        (index_id, expiry, atm)
                    )

                    ce_iv = pe_iv = 0.0
                    ce_ltp = pe_ltp = 0.0
                    total_ce_oi = total_pe_oi = 1.0

                    for row in rows:
                        if row["option_type"] == "CE":
                            ce_iv = float(row["iv"] or 0)
                            ce_ltp = float(row["ltp"] or 0)
                        else:
                            pe_iv = float(row["iv"] or 0)
                            pe_ltp = float(row["ltp"] or 0)

                    atm_iv = (ce_iv + pe_iv) / 2 if (ce_iv or pe_iv) else 0.0
                    if atm_iv <= 0:
                        continue

                    # Get total OI for PCR
                    oi_row = self.db.fetch_one(
                        "SELECT total_ce_oi, total_pe_oi FROM oi_aggregated WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1",
                        (index_id,)
                    )
                    if oi_row:
                        total_ce_oi = float(oi_row["total_ce_oi"] or 1)
                        total_pe_oi = float(oi_row["total_pe_oi"] or 1)

                    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0
                    skew = pe_iv - ce_iv
                    straddle = ce_ltp + pe_ltp

                    if not self.dry_run:
                        self.db.execute(Q.UPSERT_IV_HISTORY, (
                            now.isoformat(), index_id,
                            round(atm_iv, 4), round(ce_iv, 4), round(pe_iv, 4),
                            round(pcr, 4), round(skew, 4), round(straddle, 2),
                        ))
                    logger.debug("IV snapshot: %s ATM IV=%.1f PCR=%.2f", index_id, atm_iv, pcr)
                except Exception as e:
                    logger.debug("IV history snapshot failed for %s: %s", index_id, e)
        except Exception as exc:
            logger.warning("collect_iv_history failed: %s", exc)

    def collect_fii_futures_intraday(self) -> None:
        """Sync participant-wise futures positions into fii_futures_positions.

        NSE publishes these once per day after the close, as an archive CSV.
        The old intraday endpoint (/api/fiiStatsData) now returns 404. Each run
        fills any trading day missing in the last week, so restarts and
        downtime self-heal. Once a week is complete, a run makes no requests.
        Not gated to market hours: the file appears in the evening.
        """
        if self.dry_run:
            return
        try:
            self.fii_dii_fetcher.sync_futures_positions(self.db, lookback_days=7)
        except Exception as exc:
            logger.warning("collect_fii_futures_intraday failed: %s", exc)

    def check_data_freshness(self) -> None:
        """Alert (Telegram) when FII cash or futures data falls behind the latest published session."""
        try:
            if getattr(self, "_freshness_monitor", None) is None:
                from src.monitoring.data_freshness import FreshnessMonitor  # noqa: PLC0415

                self._freshness_monitor = FreshnessMonitor(self.db, self._send_ops_alert)
            self._freshness_monitor.run()
        except Exception as exc:
            logger.warning("check_data_freshness failed: %s", exc)

    def _send_ops_alert(self, message: str, priority: str) -> None:
        bot = getattr(self, "telegram_bot", None)
        if bot is None:
            logger.warning("Ops alert (Telegram not configured): %s", message)
            return
        bot.send_alert(message, priority=priority)

    def collect_market_breadth(self) -> None:
        """Collect Advance-Decline ratio from NSE allIndices response.

        The advances/declines data is already in the NSE allIndices API.
        This stores it separately for signal analysis.
        """
        if not self._should_run():
            return
        try:
            now = datetime.now(tz=_IST)
            raw = self.nse_scraper.get_all_indices()
            if not raw:
                return

            # Find NIFTY 50 specifically for breadth (most representative)
            for entry in raw:
                sym = str(entry.get("index_name", "")).upper()
                if sym in ("NIFTY 50", "NIFTY50"):
                    advances = int(entry.get("advances", 0))
                    declines = int(entry.get("declines", 0))
                    unchanged = int(entry.get("unchanged", 0))

                    if advances + declines == 0:
                        continue

                    ad_ratio = advances / declines if declines > 0 else float(advances)

                    if ad_ratio > 2.5:
                        breadth_signal = "STRONGLY_BULLISH"
                    elif ad_ratio > 1.5:
                        breadth_signal = "BULLISH"
                    elif ad_ratio < 0.5:
                        breadth_signal = "STRONGLY_BEARISH"
                    elif ad_ratio < 0.8:
                        breadth_signal = "BEARISH"
                    else:
                        breadth_signal = "NEUTRAL"

                    if not self.dry_run:
                        self.db.execute(Q.UPSERT_MARKET_BREADTH, (
                            now.isoformat(), "NIFTY50",
                            advances, declines, unchanged,
                            round(ad_ratio, 4), breadth_signal,
                        ))
                    logger.debug("Market breadth: A/D=%.2f (%s)", ad_ratio, breadth_signal)
                    break
        except Exception as exc:
            logger.warning("collect_market_breadth failed: %s", exc)

    def collect_gift_nifty_premarket(self) -> None:
        """Fetch GIFT NIFTY pre-market price and compute implied gap."""
        try:
            now = datetime.now(tz=_IST)
            gift_data = self.nse_scraper.get_gift_nifty()
            if not gift_data:
                logger.info("GIFT NIFTY data not available pre-market")
                return

            gift_ltp = float(gift_data.get("ltp", 0))
            if gift_ltp <= 0:
                return

            # Get previous NIFTY close
            prev_close_row = self.db.fetch_one(
                "SELECT close FROM price_data WHERE index_id = 'NIFTY50' AND timeframe = '1d' ORDER BY timestamp DESC LIMIT 1",
                ()
            )
            prev_close = float(prev_close_row["close"]) if prev_close_row else 0.0

            # Compute implied gap
            implied_gap = gift_ltp - prev_close if prev_close > 0 else 0.0
            implied_gap_pct = implied_gap / prev_close * 100 if prev_close > 0 else 0.0
            gap_abs = abs(implied_gap)

            if gap_abs < 30:
                gap_class = "FLAT_OPEN"
                wait_15min = 0
            elif gap_abs < 80:
                gap_class = "MODERATE_GAP"
                wait_15min = 0
            elif gap_abs < 100:
                gap_class = "LARGE_GAP"
                wait_15min = 1
            else:
                gap_class = "EXTREME_GAP"
                wait_15min = 1

            if not self.dry_run:
                self.db.execute(Q.UPSERT_GIFT_NIFTY, (
                    now.isoformat(), gift_ltp,
                    round(gift_data.get("change", 0), 2),
                    round(gift_data.get("change_pct", 0), 4),
                    round(prev_close, 2),
                    round(implied_gap, 2),
                    round(implied_gap_pct, 4),
                    gap_class, wait_15min,
                ))
            logger.info("GIFT NIFTY: %.0f (gap %+.0f pts, %s)", gift_ltp, implied_gap, gap_class)
        except Exception as exc:
            logger.warning("collect_gift_nifty_premarket failed: %s", exc)

    def update_historical(self) -> None:
        """18:30 IST — fill any gaps in daily candle data."""
        try:
            logger.info("Updating daily historical data for all active indices")
            if not self.dry_run:
                self.historical_manager.update_daily_data()
        except Exception as exc:
            logger.error("Historical update failed: %s", exc)

    def health_check(self) -> None:
        """
        Every 5 minutes — verify all components are healthy and record status.

        Warns if any price data is stale (> 5 min old) during market hours.
        Uses :meth:`~src.database.db_manager.DatabaseManager.write_health`
        so the record goes into the canonical ``system_health`` table.
        """
        try:
            now = datetime.now(tz=_IST)
            messages: list[str] = []

            # Price staleness check
            last_price: Optional[datetime] = self.health_status["last_successful_price_fetch"]
            if self.market_hours.is_market_open() and last_price is not None:
                age = (now - last_price).total_seconds()
                if age > 300:
                    msg = f"Prices stale — last successful fetch {age:.0f}s ago"
                    logger.warning(msg)
                    messages.append(msg)

            # Consecutive-failure summary
            for comp, n in self._failures.items():
                if n > 0:
                    msg = f"{comp} has {n} consecutive failure(s)"
                    logger.warning(msg)
                    messages.append(msg)

            status = "WARNING" if messages else "OK"
            detail = "; ".join(messages) if messages else "All systems nominal"

            if not self.dry_run:
                self.db.write_health(
                    component="data_collector",
                    status=status,
                    message=detail,
                )

            logger.info("Health check: %s — %s", status, detail)

        except Exception as exc:
            logger.error("Health check failed: %s", exc)

    def cleanup(self) -> None:
        """
        20:00 IST — purge old intraday data, stale options snapshots, and VACUUM.

        Retention policy
        ----------------
        - 1-minute price bars older than 90 days → deleted
        - options_chain_snapshot older than 30 days → deleted
        - Daily price bars and oi_aggregated → kept indefinitely
        """
        try:
            logger.info("Starting daily database cleanup")

            before = self.db.get_db_size()

            if not self.dry_run:
                now = datetime.now(tz=_IST)

                ninety_days_ago = (now - timedelta(days=90)).isoformat()
                self.db.execute(
                    "DELETE FROM price_data WHERE timeframe = '1m' AND timestamp < ?",
                    (ninety_days_ago,),
                )

                thirty_days_ago = (now - timedelta(days=30)).isoformat()
                self.db.execute(
                    "DELETE FROM options_chain_snapshot WHERE timestamp < ?",
                    (thirty_days_ago,),
                )

                self.db.vacuum()

            after = self.db.get_db_size()
            logger.info("Cleanup complete. DB size: %s → %s", before, after)

        except Exception as exc:
            logger.error("Cleanup failed: %s", exc)

    # ── Control interface ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the APScheduler background scheduler."""
        logger.info("Starting DataCollector scheduler")
        self.scheduler.start()

        # FII/DII cash data is published once a day after the close, and NSE
        # serves only the latest day, so a missed 18:00 run loses that day for
        # good. Fetch once on every start. This is scheduled only now that the
        # scheduler is running: a next_run_time set at add_job() time is
        # dropped as a misfire (default grace 1 s) when start() comes later.
        # The --force-start path below already runs it synchronously.
        if not self.force_start:
            self.scheduler.modify_job("collect_fii_dii", next_run_time=datetime.now(tz=_IST))

        # When launched outside market hours with --force-start, cron-gated
        # jobs (FII/DII at 18:00, historical update at 18:30) won't fire
        # until the next day. Trigger them once on startup so the DB is
        # current — matches what an operator expects from --force-start.
        if self.force_start:
            for job_id in ("collect_fii_dii", "update_historical", "collect_technical_indicators", "decision_cycle"):
                try:
                    self.force_run(job_id)
                except Exception:
                    logger.exception("force_start: '%s' backfill failed", job_id)

    def stop(self) -> None:
        """Graceful shutdown — wait for any running jobs to complete."""
        logger.info("Shutting down DataCollector…")
        self.scheduler.shutdown(wait=True)
        self.data_router.stop()
        logger.info("Data collector shut down gracefully")

    def pause(self) -> None:
        """Pause all jobs (e.g. during maintenance windows)."""
        self.scheduler.pause()
        logger.info("Scheduler paused")

    def resume(self) -> None:
        """Resume all previously paused jobs."""
        self.scheduler.resume()
        logger.info("Scheduler resumed")

    def force_run(self, job_id: str) -> None:
        """
        Synchronously execute a job by its ID, bypassing its trigger schedule.

        Parameters
        ----------
        job_id:
            One of ``collect_index_prices``, ``collect_options_chain``,
            ``collect_vix``, ``collect_fii_dii``, ``update_historical``,
            ``health_check``, ``cleanup``, ``warm_up``.
        """
        job = self.scheduler.get_job(job_id)
        if job is None:
            logger.warning("force_run: unknown job id %r", job_id)
            return
        logger.info("force_run: executing '%s'", job_id)
        job.func()

    def get_status(self) -> Dict[str, Any]:
        """
        Return a snapshot of all job schedules and component health.

        Returns
        -------
        dict:
            ``health`` — current health metrics.
            ``failures`` — consecutive failure counts per component.
            ``jobs`` — list of ``{id, name, next_run_time}`` dicts.
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            nrt = getattr(job, "next_run_time", None)
            jobs.append({
                "id":            job.id,
                "name":          job.name,
                "next_run_time": nrt.isoformat() if nrt else None,
            })
        return {
            "health":   self.health_status,
            "failures": dict(self._failures),
            "jobs":     jobs,
        }
