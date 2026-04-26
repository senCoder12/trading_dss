"""
F&O Universe Manager — orchestrator for daily stock selection and intraday refreshes.

Entry point for DecisionEngine, Telegram bot, and dashboard. Runs full F&O
scoring pipeline on schedule and keeps priority watchlist fresh during trading hours.

Scheduler integration:
- 08:45: Refresh F&O universe if stale, update 20-day volumes
- 09:00: Run full daily universe selection (top 50 + priority 15)
- 09:15–15:30 every 15min: Refresh priority watchlist rankings
- 15:35: End-of-day snapshot and NSE bhav copy fetch
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, date, time
from typing import Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.data.fo_universe_fetcher import FOUniverseFetcher
from src.data.fo_bulk_fetcher import FOBulkFetcher, StockQuote, StockOIData
from src.data.smart_api_client import SmartAPIClient
from src.data.nse_scraper import NSEScraper
from src.data.fii_dii_data import FIIDIIFetcher
from src.analysis.fo_liquidity_filter import LiquidityFilter, FilterResult
from src.analysis.fo_stock_scorer import FOStockScorer, StockScore
from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from src.utils.market_hours import MarketHoursManager

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)


@dataclass
class UniverseSnapshot:
    """Atomic snapshot of daily F&O universe and priority watchlist."""

    timestamp: datetime
    active_universe: list[StockScore]     # Top 50
    priority_watchlist: list[StockScore]  # Top 15
    total_liquid: int                     # Passed filter
    total_fo_stocks: int                  # Full universe size
    market_breadth: float                 # % with significant moves
    top_sector: str                       # Dominant sector today
    filter_stats: FilterResult            # Detailed filter results
    is_market_hours: bool
    refresh_count: int


class FOUniverseManager:
    """
    Single entry point for F&O stock universe selection and monitoring.

    Manages daily universe selection at 9:00 AM, intraday watchlist
    refreshes every 15 minutes, and EOD snapshot persistence.
    """

    def __init__(
        self,
        db: DatabaseManager,
        smart_api: SmartAPIClient,
        nse_scraper: NSEScraper,
        fii_fetcher: FIIDIIFetcher,
    ) -> None:
        """
        Initialize manager with all sub-components.

        Parameters
        ----------
        db:
            Database connection
        smart_api:
            Authenticated SmartAPI client for quote fetches
        nse_scraper:
            NSE scraper for universe and bhav copy
        fii_fetcher:
            FII/DII data for sector boost
        """
        self._db = db
        self._smart_api = smart_api
        self._nse_scraper = nse_scraper
        self._fii_fetcher = fii_fetcher

        self.market_hours = MarketHoursManager()

        # Sub-components
        self.fo_universe_fetcher = FOUniverseFetcher(db, nse_scraper)
        self.fo_bulk_fetcher = FOBulkFetcher(
            db, smart_api, self.fo_universe_fetcher, nse_scraper=nse_scraper
        )
        self.liquidity_filter = LiquidityFilter(db)
        self.scorer = FOStockScorer(db)

        # Load schedule settings
        schedule_cfg = getattr(settings, "fo_universe", {})
        self._daily_universe_time = schedule_cfg.get("daily_universe_time", "09:00")
        self._intraday_refresh_minutes = schedule_cfg.get("intraday_refresh_minutes", 15)
        self._eod_snapshot_time = schedule_cfg.get("eod_snapshot_time", "15:35")

        # State
        self._current_snapshot: Optional[UniverseSnapshot] = None
        self._snapshot_lock = threading.Lock()
        self._last_intraday_refresh: Optional[datetime] = None

        # Scheduler
        self.scheduler = BackgroundScheduler(timezone=_IST)

    # ── Scheduling ────────────────────────────────────────────────────────────

    def start_scheduler(self) -> None:
        """Register all APScheduler jobs and start background loop."""
        # 08:45: Refresh universe and 20-day volumes before market open
        self.scheduler.add_job(
            self._refresh_universe_and_volumes,
            trigger=CronTrigger(hour=8, minute=45, timezone=_IST),
            id="fo_refresh_universe",
            name="F&O Universe Refresh",
            max_instances=1,
        )

        # 09:00: Full daily universe selection
        self.scheduler.add_job(
            self.run_daily_universe_selection,
            trigger=CronTrigger(hour=9, minute=0, timezone=_IST),
            id="fo_daily_selection",
            name="F&O Daily Universe Selection",
            max_instances=1,
        )

        # 09:15–15:30 every 15 min: Intraday watchlist refresh
        self.scheduler.add_job(
            self.run_intraday_refresh,
            trigger=IntervalTrigger(minutes=self._intraday_refresh_minutes, timezone=_IST),
            id="fo_intraday_refresh",
            name="F&O Intraday Watchlist Refresh",
            max_instances=1,
            start_date=datetime.now(_IST),
        )

        # 15:35: EOD snapshot
        self.scheduler.add_job(
            self.run_eod_snapshot,
            trigger=CronTrigger(hour=15, minute=35, timezone=_IST),
            id="fo_eod_snapshot",
            name="F&O EOD Snapshot",
            max_instances=1,
        )

        self.scheduler.start()
        logger.info("F&O Universe Manager scheduler started")

    def stop_scheduler(self) -> None:
        """Shut down scheduler cleanly."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("F&O Universe Manager scheduler stopped")

    # ── Public API ────────────────────────────────────────────────────────────

    def run_daily_universe_selection(self) -> UniverseSnapshot:
        """
        Full pipeline run at 9:00 AM before market open.

        Uses previous day's bhav copy OI data for stable scoring.
        Locks snapshot atomically after completion.

        Returns
        -------
        UniverseSnapshot:
            Today's selected universe with top 50 active and top 15 priority.
        """
        start = datetime.now(_IST)
        logger.info("Starting daily universe selection")

        try:
            # 1. Ensure universe is fresh
            logger.info("DEBUG: calling refresh_if_stale()")
            refreshed = self.fo_universe_fetcher.refresh_if_stale()
            logger.info("DEBUG: refresh_if_stale() returned %s", refreshed)
            logger.info("DEBUG: universe symbols count = %d", len(self.fo_universe_fetcher.get_symbols()))

            # 2. Fetch pre-market quotes
            quotes = self.fo_bulk_fetcher.fetch_all_quotes()
            logger.info("Fetched quotes for %d symbols", len(quotes))

            # 3. Fetch 20-day averages (cached)
            avg_vols = self.fo_bulk_fetcher.fetch_avg_volumes(list(quotes.keys()))

            # 4. Liquidity filter
            liquid_symbols = self.liquidity_filter.get_liquid_symbols(quotes, avg_vols)
            filter_stats = FilterResult(
                total_input=len(quotes),
                passed=len(liquid_symbols),
                passed_symbols=liquid_symbols,
            )
            logger.info("Liquidity filter: %d → %d", len(quotes), len(liquid_symbols))

            # 5. Fetch OI for liquid stocks
            oi_data = self.fo_bulk_fetcher.fetch_oi_for_symbols(liquid_symbols)

            # 6. Score universe
            scores = self.scorer.score_universe(
                quotes=quotes,
                oi_data=oi_data,
                liquid_symbols=liquid_symbols,
            )
            logger.info("Scored %d symbols", len(scores))

            # 7. Build snapshot
            scores_sorted = sorted(scores, key=lambda s: s.composite_score, reverse=True)
            active_universe = scores_sorted[:50]
            priority_watchlist = scores_sorted[:15]

            # 8. Compute market breadth
            significant_moves = sum(
                1 for s in scores if abs(s.price_change_pct) >= 1.0
            )
            market_breadth = (significant_moves / len(scores) * 100) if scores else 0

            # 9. Top sector (by presence in top 15)
            sector_counts = {}
            for score in priority_watchlist:
                # TODO: enhance with sector metadata when available
                sector_counts["unknown"] = sector_counts.get("unknown", 0) + 1
            top_sector = max(sector_counts, key=sector_counts.get) if sector_counts else "unknown"

            snapshot = UniverseSnapshot(
                timestamp=datetime.now(_IST),
                active_universe=active_universe,
                priority_watchlist=priority_watchlist,
                total_liquid=len(liquid_symbols),
                total_fo_stocks=len(self.fo_universe_fetcher.get_symbols()),
                market_breadth=market_breadth,
                top_sector=top_sector,
                filter_stats=filter_stats,
                is_market_hours=self.market_hours.is_market_open(),
                refresh_count=1,
            )

            # 10. Persist and update state
            self._persist_snapshot(snapshot)
            with self._snapshot_lock:
                self._current_snapshot = snapshot
                self._last_intraday_refresh = datetime.now(_IST)

            elapsed = (datetime.now(_IST) - start).total_seconds()
            logger.info(
                "Daily universe selection complete in %.1fs: "
                "%d active, %d priority",
                elapsed, len(active_universe), len(priority_watchlist),
            )

            return snapshot

        except Exception as e:
            logger.error("Daily universe selection failed: %s", e, exc_info=True)
            raise

    def run_intraday_refresh(self) -> UniverseSnapshot:
        """
        Lightweight 15-minute refresh of priority watchlist.

        Only re-scores top 30 from current snapshot, not full universe.
        Updates priority_watchlist in-place if significant changes.

        Returns
        -------
        UniverseSnapshot:
            Updated snapshot with refreshed priority ranks.
        """
        with self._snapshot_lock:
            if self._current_snapshot is None:
                logger.debug("No snapshot yet; skipping intraday refresh")
                return None

            # Only refresh if not market hours (no update during RTH)
            if not self.market_hours.is_market_open():
                return self._current_snapshot

            current = self._current_snapshot

        try:
            # 1. Get top 30 from current snapshot
            top_30_symbols = [s.symbol for s in current.active_universe[:30]]

            # 2. Fetch fresh quotes and OI for just those 30
            quotes = self.fo_bulk_fetcher.fetch_all_quotes()
            quotes_top_30 = {s: quotes[s] for s in top_30_symbols if s in quotes}

            avg_vols = self.fo_bulk_fetcher.fetch_avg_volumes(list(quotes_top_30.keys()))
            oi_data = self.fo_bulk_fetcher.fetch_oi_for_symbols(top_30_symbols)

            # 3. Re-score just the top 30
            new_scores = self.scorer.score_universe(
                quotes=quotes_top_30,
                oi_data=oi_data,
                liquid_symbols=top_30_symbols,
            )

            # 4. Re-rank → find new top 15
            new_scores_sorted = sorted(
                new_scores, key=lambda s: s.composite_score, reverse=True
            )
            new_priority = new_scores_sorted[:15]

            # 5. Check for significant changes (>5 position swaps in top 15)
            old_symbols = {s.symbol for s in current.priority_watchlist}
            new_symbols = {s.symbol for s in new_priority}
            changes = len(old_symbols.symmetric_difference(new_symbols))

            # 6. Update snapshot in-place
            with self._snapshot_lock:
                current.priority_watchlist = new_priority
                current.refresh_count += 1
                current.timestamp = datetime.now(_IST)
                self._last_intraday_refresh = datetime.now(_IST)

            if changes > 0:
                logger.info("Intraday refresh: %d position changes in top 15", changes)

            return current

        except Exception as e:
            logger.warning("Intraday refresh failed: %s", e)
            # Return current snapshot even if refresh fails
            with self._snapshot_lock:
                return self._current_snapshot

    def run_eod_snapshot(self) -> None:
        """
        EOD snapshot at 3:35 PM.

        Fetches NSE bhav copy, updates final OI, persists performance summary.
        """
        logger.info("Starting EOD snapshot")

        with self._snapshot_lock:
            if self._current_snapshot is None:
                logger.warning("No snapshot to finalize at EOD")
                return

            snapshot = self._current_snapshot

        try:
            # Fetch and store final bhav copy (for reference)
            # This is just for record-keeping; actual trading already reflected in live quotes
            logger.info(
                "EOD snapshot: %d in top 50, market breadth %.1f%%",
                len(snapshot.active_universe), snapshot.market_breadth,
            )

        except Exception as e:
            logger.error("EOD snapshot failed: %s", e, exc_info=True)

    def get_current_universe(self) -> Optional[UniverseSnapshot]:
        """
        Thread-safe getter for current snapshot.

        Returns
        -------
        UniverseSnapshot | None:
            Current universe snapshot, or None if not yet generated.
        """
        with self._snapshot_lock:
            return self._current_snapshot

    def get_priority_symbols(self) -> list[str]:
        """
        Return symbol list from priority watchlist (top 15).

        Used by DecisionEngine to know which stocks to generate signals for.

        Returns
        -------
        list[str]:
            Symbols in priority watchlist.
        """
        with self._snapshot_lock:
            if self._current_snapshot is None:
                return []
            return [s.symbol for s in self._current_snapshot.priority_watchlist]

    def get_active_universe_symbols(self) -> list[str]:
        """
        Return symbol list from active universe (top 50).

        Used for broader monitoring.

        Returns
        -------
        list[str]:
            Symbols in active universe.
        """
        with self._snapshot_lock:
            if self._current_snapshot is None:
                return []
            return [s.symbol for s in self._current_snapshot.active_universe]

    def is_symbol_in_priority(self, symbol: str) -> bool:
        """
        Check if stock is in priority watchlist (top 15).

        Parameters
        ----------
        symbol:
            NSE stock symbol.

        Returns
        -------
        bool:
            True if in priority list.
        """
        return symbol in self.get_priority_symbols()

    def get_stock_score(self, symbol: str) -> Optional[StockScore]:
        """
        Return score details for a specific symbol.

        Parameters
        ----------
        symbol:
            NSE stock symbol.

        Returns
        -------
        StockScore | None:
            Score object if found, None otherwise.
        """
        with self._snapshot_lock:
            if self._current_snapshot is None:
                return None

            for score in self._current_snapshot.active_universe:
                if score.symbol == symbol:
                    return score

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _refresh_universe_and_volumes(self) -> None:
        """
        Refresh F&O universe list and 20-day average volumes at 8:45 AM.

        Runs before market open to ensure data is fresh.
        """
        try:
            if self.fo_universe_fetcher.refresh_if_stale():
                logger.info("F&O universe refreshed")
            self.fo_bulk_fetcher.fetch_avg_volumes()
            logger.info("20-day average volumes refreshed")
        except Exception as e:
            logger.error("Pre-market refresh failed: %s", e, exc_info=True)

    def _persist_snapshot(self, snapshot: UniverseSnapshot) -> None:
        """
        Save snapshot to database.

        Parameters
        ----------
        snapshot:
            Snapshot to persist.
        """
        today = datetime.now(_IST).date().isoformat()

        # Convert scores to JSON
        active_json = json.dumps(
            [asdict(s) for s in snapshot.active_universe],
            default=str,
        )
        priority_json = json.dumps(
            [asdict(s) for s in snapshot.priority_watchlist],
            default=str,
        )
        filter_json = json.dumps(
            {
                "total_input": snapshot.filter_stats.total_input,
                "passed": snapshot.filter_stats.passed,
                "passed_symbols": snapshot.filter_stats.passed_symbols,
            },
            default=str,
        )

        self._db.execute(
            Q.UPSERT_FO_UNIVERSE_SNAPSHOT,
            (
                today,
                snapshot.timestamp.isoformat(),
                active_json,
                priority_json,
                snapshot.total_liquid,
                snapshot.total_fo_stocks,
                snapshot.market_breadth,
                snapshot.top_sector,
                filter_json,
                int(snapshot.is_market_hours),
                snapshot.refresh_count,
            ),
        )

        logger.debug("Snapshot persisted for %s", today)
