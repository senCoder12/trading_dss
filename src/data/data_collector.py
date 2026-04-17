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
from src.data.data_validator import validate_price_data
from src.data.fii_dii_data import FIIDIIFetcher
from src.data.historical_data import HistoricalDataManager
from src.data.index_registry import IndexRegistry
from src.data.nse_scraper import NSEScraper
from src.data.options_chain import OptionsChainFetcher
from src.data.rate_limiter import GlobalRateLimiter, RateLimiter, create_nse_limiter
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
        self.options_fetcher = OptionsChainFetcher(scraper=self.nse_scraper)
        self.historical_manager = HistoricalDataManager(registry=self.registry, db=self.db)
        self.fii_dii_fetcher = FIIDIIFetcher(scraper=self.nse_scraper)
        self.vix_tracker = VIXTracker(scraper=self.nse_scraper)

        self.scheduler = BackgroundScheduler(timezone=_IST)
        self._setup_jobs()

        # Health / failure tracking ─────────────────────────────────────────
        self._failures: Dict[str, int] = {
            "nse": 0, "bse": 0, "options": 0, "vix": 0,
        }
        # Monotonic timestamp of last auto-recovery attempt per component
        self._last_recovery: Dict[str, float] = {}

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
            id="collect_index_prices",
            name="Collect Index Prices",
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self.collect_options_chain,
            trigger=IntervalTrigger(seconds=p.options_interval, timezone=_IST),
            id="collect_options_chain",
            name="Collect Options Chain",
            max_instances=1,
            coalesce=True,
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
                    self.options_fetcher = OptionsChainFetcher(scraper=self.nse_scraper)
                    self.fii_dii_fetcher = FIIDIIFetcher(scraper=self.nse_scraper)
                    self.vix_tracker = VIXTracker(scraper=self.nse_scraper)
                elif component == "bse":
                    self.bse_scraper = BSEScraper()

    def _reset_failure(self, component: str) -> None:
        self._failures[component] = 0

    # ── Market-hours guard ─────────────────────────────────────────────────

    def _should_run(self) -> bool:
        """Return True when market-hours-gated jobs are allowed to execute."""
        if self.force_start:
            return True
        status = self.market_hours.get_market_status()
        return bool(status.get("status") == "open")

    # ── Job definitions ────────────────────────────────────────────────────

    def warm_up_sessions(self) -> None:
        """9:00 AM — seed NSE/BSE cookies before market opens at 9:15."""
        status = self.market_hours.get_market_status()
        if status.get("is_holiday") or status.get("is_weekend"):
            logger.info("Skipping pre-market warm-up (holiday / weekend)")
            return
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

        Falls back to BSE if NSE returns nothing.
        """
        if not self._should_run():
            return

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

        # When launched outside market hours with --force-start, cron-gated
        # jobs (FII/DII at 18:00, historical update at 18:30) won't fire
        # until the next day. Trigger them once on startup so the DB is
        # current — matches what an operator expects from --force-start.
        if self.force_start:
            for job_id in ("collect_fii_dii", "update_historical", "collect_technical_indicators"):
                try:
                    self.force_run(job_id)
                except Exception:
                    logger.exception("force_start: '%s' backfill failed", job_id)

    def stop(self) -> None:
        """Graceful shutdown — wait for any running jobs to complete."""
        logger.info("Shutting down DataCollector…")
        self.scheduler.shutdown(wait=True)
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
