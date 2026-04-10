"""
Master news engine — single entry point for all Phase 3 news analysis.

The Data Collector calls ``NewsEngine.run_news_cycle()`` every 120 seconds
during market hours.  Every other component (Decision Engine, Telegram bot,
dashboard) reads results via ``get_news_vote()`` / ``get_all_news_votes()``.

Data Collector integration
--------------------------
Add the following to ``src/data/data_collector.py``:

.. code-block:: python

    # In DataCollector.__init__:
    #   self.news_engine = NewsEngine(self.db)
    #
    # New scheduled job — runs every 120 seconds during market hours:
    #
    # def _collect_news(self):
    #     result = self.news_engine.run_news_cycle()
    #     logger.info(
    #         "News cycle: %d new articles, %d alerts",
    #         result.articles_new, len(result.alerts),
    #     )
    #     for alert in self.news_engine.get_critical_alerts():
    #         self.telegram.send_alert(alert.message)   # Phase 9
    #
    # Register in _setup_jobs():
    #   self.scheduler.add_job(
    #       self._collect_news,
    #       "interval", seconds=120,
    #       id="collect_news",
    #       next_run_time=datetime.now(),
    #   )
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.database.db_manager import DatabaseManager
from src.analysis.news.rss_fetcher import RSSFetcher
from src.analysis.news.article_parser import ArticleParser, ParsedArticle
from src.analysis.news.deduplicator import ArticleDeduplicator
from src.analysis.news.sentiment_analyzer import SentimentAnalyzer
from src.analysis.news.impact_mapper import (
    NewsImpactMapper,
    MappedArticle,
    MarketNewsSummary,
)
from src.analysis.news.time_decay import (
    TimeDecayEngine,
    EffectiveNewsScore,
    DecayedArticle,
)
from src.analysis.news.event_calendar import EventCalendar, EventRegimeModifier
from src.analysis.news.news_store import NewsStore

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")

# Severity ordering (lower = more severe)
_SEVERITY_ORDER: dict[str, int] = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
    "NOISE": 4,
}

# Prune URL cache when it exceeds this size
_MAX_URL_CACHE = 5_000

# Rolling window kept in the in-memory article cache
_CACHE_WINDOW_HOURS = 6


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NewsAlert:
    """
    High-priority alert for a single article.

    Generated for CRITICAL and HIGH severity articles each cycle.
    Suitable for forwarding directly to Telegram or a dashboard widget.
    """

    title: str
    severity: str           # CRITICAL / HIGH
    sentiment: str          # sentiment_label from SentimentResult
    affected_indices: list[str]
    is_actionable: bool
    message: str            # Human-readable one-liner for Telegram / dashboard


@dataclass
class NewsVote:
    """
    Directional vote for a single index, derived from active news articles.

    Consumed by the Decision Engine (Phase 4) to incorporate news signals
    into the final trading signal.
    """

    index_id: str
    timestamp: datetime
    vote: str               # STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG_BEARISH
    confidence: float       # 0–1
    active_article_count: int
    weighted_sentiment: float
    top_headline: Optional[str]
    event_regime: str       # NORMAL / ELEVATED / HIGH / EXTREME
    reasoning: str          # Human-readable summary


@dataclass
class NewsCycleResult:
    """
    Output of a single ``run_news_cycle()`` call.

    The Decision Engine consumes ``index_news_scores`` and ``event_regime``.
    The Telegram bot / dashboard consumes ``alerts`` and ``market_summary``.
    """

    timestamp: datetime

    # Fetch stats
    articles_fetched: int = 0
    articles_new: int = 0           # after deduplication
    articles_duplicate: int = 0
    feeds_successful: int = 0
    feeds_failed: int = 0

    # Processing stats
    articles_processed: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    by_event_type: dict[str, int] = field(default_factory=dict)

    # Per-index effective news scores — primary output for Decision Engine
    index_news_scores: dict[str, EffectiveNewsScore] = field(default_factory=dict)

    # Event calendar context — used for position-size modifiers
    event_regime: dict[str, EventRegimeModifier] = field(default_factory=dict)

    # Aggregated market view
    market_summary: Optional[MarketNewsSummary] = None

    # High-priority alerts (CRITICAL + HIGH only)
    alerts: list[NewsAlert] = field(default_factory=list)

    # Wall-clock duration of the cycle
    cycle_duration_ms: int = 0


# ---------------------------------------------------------------------------
# News Engine
# ---------------------------------------------------------------------------


class NewsEngine:
    """
    Orchestrates the complete news analysis pipeline.

    A single instance is created at startup and shared across threads.
    The in-memory state (article cache, URL cache, last-cycle results) is
    protected by ``_lock`` so concurrent reads by the Decision Engine never
    see a half-updated snapshot.

    Parameters
    ----------
    db:
        Connected DatabaseManager (the singleton from the Data Collector).
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

        # Pipeline components ─ each handles its own file-loading gracefully
        self.fetcher = RSSFetcher()
        self.parser = ArticleParser()
        self.deduplicator = ArticleDeduplicator()
        self.sentiment = SentimentAnalyzer()
        self.impact_mapper = NewsImpactMapper()
        self.decay_engine = TimeDecayEngine()
        self.calendar = EventCalendar()
        self.store = NewsStore()

        # Thread safety
        self._lock = threading.Lock()

        # --- Shared mutable state (always accessed under _lock) ---
        # Rolling window of MappedArticle objects (last CACHE_WINDOW_HOURS hours)
        self._recent_mapped: list[MappedArticle] = []
        # Fast URL lookup so we don't hit the DB for every article
        self._recent_urls: set[str] = set()
        # Snapshot of the last completed cycle
        self._last_result: Optional[NewsCycleResult] = None
        # Per-index EffectiveNewsScore from the last cycle
        self._last_index_scores: dict[str, EffectiveNewsScore] = {}
        # Event regime modifiers from the last cycle
        self._last_event_regime: dict[str, EventRegimeModifier] = {}
        # DecayedArticle list from the last cycle (for get_news_feed)
        self._last_decayed: list[DecayedArticle] = []

        logger.info("NewsEngine initialised")
        self._cleanup_on_startup()

    # ------------------------------------------------------------------
    # Startup cleanup
    # ------------------------------------------------------------------

    def _cleanup_on_startup(self) -> None:
        """Mark stale unprocessed articles from previous sessions as processed.

        Articles fetched more than 6 hours ago that were never processed are too
        old to meaningfully analyze — their sentiment impact has decayed to near zero.
        """
        try:
            from src.utils.date_utils import get_ist_now
            from datetime import timedelta as _td

            now = get_ist_now()
            cutoff = (now - _td(hours=6)).isoformat()

            self.db.execute(
                """UPDATE news_articles
                   SET is_processed = TRUE
                   WHERE is_processed = FALSE AND fetched_at < ?""",
                (cutoff,)
            )

            stale = self.db.fetch_one(
                """SELECT COUNT(*) as c FROM news_articles
                   WHERE is_processed = TRUE AND fetched_at < ?
                   AND fetched_at > ?""",
                (cutoff, (now - _td(hours=12)).isoformat())
            )
            count = stale['c'] if stale else 0

            if count > 0:
                logger.info(f"News cleanup: marked {count} stale unprocessed article(s) as processed")

        except Exception as e:
            logger.error(f"News startup cleanup failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_news_cycle(self) -> NewsCycleResult:
        """
        Execute one complete news pipeline cycle.

        This method is the heartbeat of the news subsystem.  It is designed
        to **never raise** — all exceptions are caught, logged, and result in
        a partial (but valid) ``NewsCycleResult`` being returned.

        Pipeline
        --------
        1. Fetch raw articles from all active RSS feeds.
        2. Parse each article (clean text, extract companies/sectors/indices,
           classify event type).
        3. Deduplicate: within-batch first, then against DB (URL + title
           Jaccard similarity).
        4. Sentiment analysis on new articles only (VADER + keyword boost).
        5. Map each article to affected indices; classify impact severity.
        6. Save new articles + index impacts to the database.
        7. Apply time decay to ALL active articles (cache + new).
        8. Compute effective news score per affected index.
        9. Get event calendar regime modifiers for all relevant indices.
        10. Build market summary, generate high-priority alerts, return result.

        Returns
        -------
        NewsCycleResult
        """
        cycle_start = time.monotonic()
        now = datetime.now(tz=_IST)
        result = NewsCycleResult(timestamp=now)

        try:
            # Step 1 — fetch
            raw_articles = self._fetch(result)

            # Step 2 — parse
            parsed = self._parse(raw_articles)

            # Step 3 — deduplicate
            new_articles = self._dedup(parsed)
            result.articles_fetched = len(raw_articles)
            result.articles_new = len(new_articles)
            result.articles_duplicate = len(parsed) - len(new_articles)

            # Steps 4–5 — sentiment + mapping
            mapped_new = self._analyze_and_map(new_articles)
            result.articles_processed = len(mapped_new)
            result.by_severity = _count_by(mapped_new, lambda m: m.impact_severity)
            result.by_event_type = _count_by(
                mapped_new,
                lambda m: m.article.event_type or "UNKNOWN",
            )

            # Step 6 — persist
            self._save(mapped_new)

            # Steps 7–8 — decay + scores
            decayed, index_scores = self._decay_and_score(mapped_new, now)
            result.index_news_scores = index_scores

            # Step 9 — event calendar
            event_regime = self._event_regime(set(index_scores.keys()))
            result.event_regime = event_regime

            # Step 10 — summary + alerts
            with self._lock:
                all_recent = list(self._recent_mapped)

            result.market_summary = self.impact_mapper.get_market_news_summary(
                all_recent
            )
            result.alerts = self._make_alerts(mapped_new)

            # Cache for subsequent reads by other threads
            with self._lock:
                self._last_result = result
                self._last_index_scores = index_scores
                self._last_event_regime = event_regime
                self._last_decayed = decayed

        except Exception:
            logger.exception("Unexpected error in run_news_cycle — partial result returned")

        result.cycle_duration_ms = int((time.monotonic() - cycle_start) * 1000)
        logger.info(
            "News cycle complete — fetched=%d new=%d processed=%d "
            "alerts=%d duration=%dms",
            result.articles_fetched,
            result.articles_new,
            result.articles_processed,
            len(result.alerts),
            result.cycle_duration_ms,
        )
        return result

    def get_news_vote(self, index_id: str) -> NewsVote:
        """
        Return the current news-based directional vote for *index_id*.

        Uses results from the most recent completed cycle (cached in memory).
        Returns a NEUTRAL vote with zero confidence if no cycle has run yet,
        or if *index_id* has no relevant articles.

        Parameters
        ----------
        index_id:
            Registry ID of the index (e.g. ``"NIFTY50"``).

        Returns
        -------
        NewsVote
        """
        with self._lock:
            score = self._last_index_scores.get(index_id)
            regime = self._last_event_regime.get(index_id)
            timestamp = (
                self._last_result.timestamp
                if self._last_result
                else datetime.now(tz=_IST)
            )

        if score is None:
            return NewsVote(
                index_id=index_id,
                timestamp=timestamp,
                vote="NEUTRAL",
                confidence=0.0,
                active_article_count=0,
                weighted_sentiment=0.0,
                top_headline=None,
                event_regime="NORMAL",
                reasoning=f"No recent news data for {index_id}",
            )

        caution = regime.caution_level if regime else "NORMAL"
        return NewsVote(
            index_id=index_id,
            timestamp=score.timestamp,
            vote=score.news_vote,
            confidence=score.news_confidence,
            active_article_count=score.article_count,
            weighted_sentiment=score.weighted_sentiment,
            top_headline=score.top_article_title,
            event_regime=caution,
            reasoning=_build_reasoning(score, regime),
        )

    def get_all_news_votes(self) -> dict[str, NewsVote]:
        """
        Return news votes for all active F&O indices plus any index that has
        at least one active news article.

        Returns
        -------
        dict[str, NewsVote]:
            ``{index_id: NewsVote}`` sorted by index_id.
        """
        with self._lock:
            scored_ids = set(self._last_index_scores.keys())

        fo_ids = set(self._fo_index_ids())
        all_ids = scored_ids | fo_ids

        return {idx: self.get_news_vote(idx) for idx in sorted(all_ids)}

    def get_news_feed(
        self,
        index_id: Optional[str] = None,
        min_severity: str = "LOW",
        limit: int = 20,
    ) -> list[dict]:
        """
        Return a formatted list of recent news items for dashboard display.

        Articles are taken from the last cycle's decayed snapshot.  Stale
        articles (decay < 0.05) are excluded automatically.

        Parameters
        ----------
        index_id:
            If given, return only articles that affect this index.
        min_severity:
            Include articles at or above this severity (e.g. ``"LOW"``
            includes LOW, MEDIUM, HIGH, CRITICAL).
        limit:
            Maximum number of articles to return.

        Returns
        -------
        list[dict]:
            Each dict contains: ``title``, ``source``, ``published_at``,
            ``sentiment_label``, ``severity``, ``affected_indices``,
            ``age_minutes``, ``decay_factor``.
            Sorted by severity DESC (CRITICAL first), then age ASC (newest first).
        """
        min_order = _SEVERITY_ORDER.get(min_severity.upper(), 3)
        now = datetime.now(tz=_IST)

        with self._lock:
            snapshot = list(self._last_decayed)

        feed: list[dict] = []
        for da in snapshot:
            if da.is_stale:
                continue

            mapped = da.article
            sev = mapped.impact_severity
            if _SEVERITY_ORDER.get(sev, 4) > min_order:
                continue

            affected = [imp.index_id for imp in mapped.index_impacts]
            if index_id is not None and index_id not in affected:
                continue

            pub = mapped.article.published_at
            try:
                pub_aware = pub if pub.tzinfo else pub.replace(tzinfo=_IST)
                age_min = (now - pub_aware).total_seconds() / 60.0
            except Exception:
                age_min = da.age_minutes

            feed.append(
                {
                    "title": mapped.article.title,
                    "source": mapped.article.source,
                    "published_at": pub.isoformat(),
                    "sentiment_label": mapped.sentiment.sentiment_label,
                    "severity": sev,
                    "affected_indices": affected,
                    "age_minutes": round(age_min, 1),
                    "decay_factor": round(da.decay_factor, 3),
                }
            )

        # Sort: severity ASC (CRITICAL=0 first), then age ASC (newest first)
        feed.sort(
            key=lambda x: (_SEVERITY_ORDER.get(x["severity"], 4), x["age_minutes"])
        )
        return feed[:limit]

    def get_critical_alerts(self) -> list[NewsAlert]:
        """
        Return CRITICAL and HIGH severity alerts from the most recent cycle.

        Suitable for direct forwarding to the Telegram notification service.

        Returns
        -------
        list[NewsAlert]:
            Empty list if no cycle has run or no high-priority alerts exist.
        """
        with self._lock:
            if self._last_result is None:
                return []
            return [
                a
                for a in self._last_result.alerts
                if a.severity in ("CRITICAL", "HIGH")
            ]

    # ------------------------------------------------------------------
    # Private helpers — pipeline steps
    # ------------------------------------------------------------------

    def _fetch(self, result: NewsCycleResult) -> list:
        """Step 1: Fetch raw articles; update feed stats on result."""
        try:
            stats_before = self.fetcher.get_fetch_stats()
        except Exception:
            stats_before = {}

        raw: list = []
        try:
            raw = self.fetcher.fetch_all_feeds()
        except Exception:
            logger.exception("RSSFetcher.fetch_all_feeds() failed — skipping fetch")

        try:
            stats_after = self.fetcher.get_fetch_stats()
        except Exception:
            stats_after = {}

        failed_delta = max(
            0,
            stats_after.get("failed_requests", 0)
            - stats_before.get("failed_requests", 0),
        )
        active_feeds = stats_after.get("active_feeds", 0)
        result.feeds_failed = failed_delta
        result.feeds_successful = max(0, active_feeds - failed_delta)
        return raw

    def _parse(self, raw_articles: list) -> list[ParsedArticle]:
        """Step 2: Parse RawArticle → ParsedArticle; skip failures silently."""
        parsed: list[ParsedArticle] = []
        for raw in raw_articles:
            try:
                p = self.parser.parse_article(raw)
                if p is not None:
                    parsed.append(p)
            except Exception:
                logger.debug(
                    "parse_article failed for url=%r",
                    getattr(raw, "url", "?"),
                    exc_info=True,
                )
        return parsed

    def _dedup(self, parsed: list[ParsedArticle]) -> list[ParsedArticle]:
        """
        Step 3: Remove duplicates.

        Two-stage dedup:
        a) Batch dedup (within this fetch cycle, by title Jaccard similarity).
        b) DB + in-memory-URL-cache dedup (cross-cycle).
        """
        # a) Within-batch
        batch_deduped = self.deduplicator.deduplicate_batch(parsed)

        new_articles: list[ParsedArticle] = []
        for article in batch_deduped:
            # Fast path: URL already seen in this process
            with self._lock:
                already_seen = article.url in self._recent_urls

            if already_seen:
                continue

            # Slower path: DB check (URL exact + title similarity)
            is_dup = False
            try:
                is_dup = self.deduplicator.is_duplicate_in_db(article, self.db)
            except Exception:
                logger.debug(
                    "is_duplicate_in_db failed for url=%r — treating as new",
                    article.url,
                    exc_info=True,
                )

            with self._lock:
                self._recent_urls.add(article.url)

            if is_dup:
                continue

            new_articles.append(article)

        # Prune the URL cache if it gets too large (keep the latter half)
        with self._lock:
            if len(self._recent_urls) > _MAX_URL_CACHE:
                urls = list(self._recent_urls)
                self._recent_urls = set(urls[len(urls) // 2 :])

        return new_articles

    def _analyze_and_map(
        self, new_articles: list[ParsedArticle]
    ) -> list[MappedArticle]:
        """Steps 4–5: Sentiment analysis and index impact mapping."""
        mapped: list[MappedArticle] = []
        for article in new_articles:
            try:
                sentiment = self.sentiment.analyze_sentiment(article)
                m = self.impact_mapper.map_and_classify(article, sentiment)
                mapped.append(m)
            except Exception:
                logger.debug(
                    "analyze_and_map failed for url=%r",
                    article.url,
                    exc_info=True,
                )
        return mapped

    def _save(self, mapped_new: list[MappedArticle]) -> None:
        """Step 6: Persist new articles and their index impacts to the DB."""
        if not mapped_new:
            return
        try:
            article_ids = self.store.save_articles(mapped_new, self.db)
            for mapped, article_id in zip(mapped_new, article_ids):
                if article_id is not None:
                    self.store.save_index_impacts(
                        article_id, mapped.index_impacts, self.db
                    )
        except Exception:
            logger.exception("_save() failed — DB writes rolled back or partial")

    def _decay_and_score(
        self,
        new_mapped: list[MappedArticle],
        now: datetime,
    ) -> tuple[list[DecayedArticle], dict[str, EffectiveNewsScore]]:
        """
        Steps 7–8: Update in-memory article cache, apply time decay, compute
        per-index effective news scores.
        """
        cutoff = now - timedelta(hours=_CACHE_WINDOW_HOURS)

        with self._lock:
            # Prune stale articles from rolling cache
            self._recent_mapped = [
                a
                for a in self._recent_mapped
                if _article_time(a) >= cutoff
            ]
            self._recent_mapped.extend(new_mapped)
            all_active = list(self._recent_mapped)

        # Apply time decay to everything in the cache
        decayed: list[DecayedArticle] = []
        try:
            decayed = self.decay_engine.apply_decay_to_articles(all_active, now)
        except Exception:
            logger.exception("apply_decay_to_articles() failed")

        # Collect all indices touched by non-stale articles
        affected: set[str] = set()
        for da in decayed:
            if not da.is_stale:
                affected.update(da.effective_impacts.keys())

        # Compute EffectiveNewsScore per index
        scores: dict[str, EffectiveNewsScore] = {}
        for idx_id in affected:
            try:
                scores[idx_id] = self.decay_engine.get_effective_news_score(
                    decayed, idx_id
                )
            except Exception:
                logger.debug(
                    "get_effective_news_score failed for index=%r",
                    idx_id,
                    exc_info=True,
                )

        return decayed, scores

    def _event_regime(
        self, scored_index_ids: set[str]
    ) -> dict[str, EventRegimeModifier]:
        """Step 9: Get event calendar regime modifiers for all relevant indices."""
        all_ids = scored_index_ids | set(self._fo_index_ids())
        result: dict[str, EventRegimeModifier] = {}
        for idx_id in all_ids:
            try:
                result[idx_id] = self.calendar.get_regime_modifier(idx_id)
            except Exception:
                logger.debug(
                    "get_regime_modifier failed for index=%r",
                    idx_id,
                    exc_info=True,
                )
        return result

    def _make_alerts(self, mapped_new: list[MappedArticle]) -> list[NewsAlert]:
        """Step 10a: Generate NewsAlert objects for CRITICAL and HIGH articles."""
        alerts: list[NewsAlert] = []
        for mapped in mapped_new:
            if mapped.impact_severity not in ("CRITICAL", "HIGH"):
                continue

            article = mapped.article
            sentiment = mapped.sentiment
            affected = [imp.index_id for imp in mapped.index_impacts]

            direction = "BULLISH" if sentiment.adjusted_score > 0 else "BEARISH"
            indices_str = ", ".join(affected[:3])
            if len(affected) > 3:
                indices_str += f" +{len(affected) - 3} more"

            message = (
                f"[{mapped.impact_severity}] {article.title} | "
                f"{direction} ({sentiment.sentiment_label}) | "
                f"Affects: {indices_str or 'broad market'} | "
                f"Source: {article.source}"
            )

            alerts.append(
                NewsAlert(
                    title=article.title,
                    severity=mapped.impact_severity,
                    sentiment=sentiment.sentiment_label,
                    affected_indices=affected,
                    is_actionable=mapped.is_actionable,
                    message=message,
                )
            )

        # Most critical first
        alerts.sort(key=lambda a: _SEVERITY_ORDER.get(a.severity, 4))
        return alerts

    # ------------------------------------------------------------------
    # Private helpers — utilities
    # ------------------------------------------------------------------

    def _fo_index_ids(self) -> list[str]:
        """Return IDs of all F&O-enabled active indices from the registry."""
        try:
            from src.data.index_registry import get_registry

            registry = get_registry()
            return [idx.id for idx in registry.get_indices_with_options()]
        except Exception:
            logger.debug("Could not load F&O index IDs from registry", exc_info=True)
            return []


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _article_time(mapped: MappedArticle) -> datetime:
    """Return a timezone-aware ``published_at`` for the given article."""
    pub = mapped.article.published_at
    return pub if pub.tzinfo is not None else pub.replace(tzinfo=_IST)


def _count_by(items: list[MappedArticle], key_fn) -> dict[str, int]:
    """Count MappedArticle objects grouped by ``key_fn``."""
    counts: dict[str, int] = {}
    for item in items:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _build_reasoning(
    score: EffectiveNewsScore,
    regime: Optional[EventRegimeModifier],
) -> str:
    """
    Build a human-readable reasoning string for a NewsVote.

    Example output::

        "BEARISH (conf 0.65): 3 articles (1 bullish, 2 bearish) |
         Top: 'FII selling pressure spikes ahead of RBI meeting' |
         Regime: HIGH (RBI Policy Decision)"
    """
    parts: list[str] = [f"{score.news_vote} (conf {score.news_confidence:.2f}):"]

    if score.article_count == 0:
        parts.append("no active news articles")
    else:
        parts.append(f"{score.article_count} article{'s' if score.article_count > 1 else ''}")

        breakdown_parts: list[str] = []
        if score.bullish_articles:
            breakdown_parts.append(f"{score.bullish_articles} bullish")
        if score.bearish_articles:
            breakdown_parts.append(f"{score.bearish_articles} bearish")
        if score.neutral_articles:
            breakdown_parts.append(f"{score.neutral_articles} neutral")
        if breakdown_parts:
            parts.append(f"({', '.join(breakdown_parts)})")

        if score.top_article_title:
            headline = score.top_article_title
            if len(headline) > 80:
                headline = headline[:77] + "..."
            parts.append(f"| Top: {headline!r}")

    if regime and regime.caution_level != "NORMAL":
        events = ", ".join(regime.active_events[:2]) if regime.active_events else "events"
        parts.append(f"| Regime: {regime.caution_level} ({events})")

    return " ".join(parts)
