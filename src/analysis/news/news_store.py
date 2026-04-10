"""
Database storage layer for the news analysis pipeline.

All SQL reads and writes for the news subsystem go through this class.
NewsEngine is the only caller; nothing else should import NewsStore directly.
"""

# ================================================================
# TABLE OWNERSHIP: This class is the SOLE writer for `news_articles`
# and `news_index_impact` tables.
# No other module should INSERT, UPDATE, or DELETE from these tables.
# ================================================================

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from src.analysis.news.impact_mapper import IndexImpact, MappedArticle

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")

# Maps IndexImpact.mapping_method → news_index_impact.mapped_via (CHECK constraint)
_METHOD_TO_MAPPED_VIA: dict[str, str] = {
    "DIRECT_MENTION": "direct",
    "COMPANY_MAPPING": "direct",   # company IS a constituent of the index
    "KEYWORD_MAPPING": "keyword",
    "MULTI_METHOD": "direct",
}

_SEVERITY_ORDER: dict[str, int] = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
    "NOISE": 4,
}

# Retention policy: delete articles of this severity older than N days
_RETENTION_DAYS: dict[str, int] = {
    "NOISE": 7,
    "LOW": 14,
    "MEDIUM": 30,
    "HIGH": 30,
    "CRITICAL": 90,
}


class NewsStore:
    """
    Handles all DB read/write operations for the news pipeline.

    Methods are stateless — the caller passes a DatabaseManager on each call
    so NewsStore can be used safely across threads.
    """

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def save_articles(
        self,
        articles: list[MappedArticle],
        db: DatabaseManager,
    ) -> list[Optional[int]]:
        """
        Insert new articles into ``news_articles`` (ON CONFLICT DO NOTHING).

        Parameters
        ----------
        articles:
            MappedArticle objects produced by the impact mapper.
        db:
            Connected DatabaseManager for this call.

        Returns
        -------
        list[Optional[int]]:
            Article DB IDs in the same order as *articles*.
            ``None`` means the row failed to insert or could not be retrieved.
        """
        ids: list[Optional[int]] = []
        for mapped in articles:
            article = mapped.article
            sentiment = mapped.sentiment
            try:
                db.execute(
                    Q.INSERT_NEWS_ARTICLE,
                    (
                        article.title,
                        article.clean_text[:500] if article.clean_text else None,
                        article.source,
                        article.url,
                        article.published_at.isoformat(),
                        article.fetched_at.isoformat(),
                        sentiment.raw_vader_score,
                        sentiment.adjusted_score,
                        mapped.impact_severity,
                        article.source_credibility,
                        0,  # is_processed = False initially
                    ),
                )
                row = db.fetch_one(Q.GET_NEWS_BY_URL, (article.url,))
                ids.append(row["id"] if row else None)
            except Exception:
                logger.exception("Failed to save article url=%r", article.url)
                ids.append(None)
        return ids

    def save_index_impacts(
        self,
        article_id: int,
        impacts: list[IndexImpact],
        db: DatabaseManager,
    ) -> int:
        """
        Insert one row per (article, index) pair into ``news_index_impact``.

        Parameters
        ----------
        article_id:
            PK from ``news_articles``.
        impacts:
            IndexImpact objects from the impact mapper.
        db:
            Connected DatabaseManager.

        Returns
        -------
        int:
            Number of rows inserted (0 if none or on error).
        """
        if not impacts:
            return 0

        rows: list[tuple] = []
        for impact in impacts:
            mapped_via = _METHOD_TO_MAPPED_VIA.get(impact.mapping_method, "keyword")
            rows.append(
                (article_id, impact.index_id, impact.relevance_score, mapped_via)
            )

        try:
            return db.execute_many(Q.INSERT_NEWS_INDEX_IMPACT, rows)
        except Exception:
            logger.exception(
                "Failed to save index impacts for article_id=%d", article_id
            )
            return 0

    def mark_article_processed(self, article_id: int, db: DatabaseManager) -> None:
        """Set ``is_processed = 1`` on a news article."""
        try:
            db.execute(Q.UPDATE_NEWS_PROCESSED, (article_id,))
        except Exception:
            logger.exception(
                "Failed to mark article %d as processed", article_id
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_recent_articles(
        self,
        db: DatabaseManager,
        hours: int = 6,
        min_severity: str = "NOISE",
    ) -> list[dict]:
        """
        Fetch articles published in the last *hours* hours that are at or
        above *min_severity*.

        Used by the dashboard / analytics — **not** for time-decay
        calculations (those use the in-memory cache in NewsEngine).

        Parameters
        ----------
        db:
            Connected DatabaseManager.
        hours:
            Look-back window.
        min_severity:
            Minimum impact category to include (inclusive, default = all).

        Returns
        -------
        list[dict]:
            Rows from ``news_articles``, sorted by ``published_at DESC``.
        """
        since = (datetime.now(tz=_IST) - timedelta(hours=hours)).isoformat()
        min_order = _SEVERITY_ORDER.get(min_severity.upper(), 4)
        try:
            rows = db.fetch_all(Q.LIST_RECENT_NEWS, (since, 1000))
            return [
                r
                for r in rows
                if _SEVERITY_ORDER.get(
                    str(r.get("impact_category", "NOISE")).upper(), 4
                )
                <= min_order
            ]
        except Exception:
            logger.exception("Failed to fetch recent articles")
            return []

    def get_article_count_by_day(
        self,
        db: DatabaseManager,
        days: int = 30,
    ) -> list[dict]:
        """
        Return per-day article counts for the last *days* days.

        Returns
        -------
        list[dict]:
            ``[{date, article_count, critical_count, high_count, avg_sentiment}]``
            sorted newest-first.
        """
        since = (datetime.now(tz=_IST) - timedelta(days=days)).isoformat()
        try:
            return db.fetch_all(
                """
                SELECT
                    DATE(published_at)   AS date,
                    COUNT(*)             AS article_count,
                    SUM(CASE WHEN impact_category = 'CRITICAL' THEN 1 ELSE 0 END)
                                         AS critical_count,
                    SUM(CASE WHEN impact_category = 'HIGH'     THEN 1 ELSE 0 END)
                                         AS high_count,
                    AVG(adjusted_sentiment) AS avg_sentiment
                FROM news_articles
                WHERE published_at >= ?
                GROUP BY DATE(published_at)
                ORDER BY date DESC
                """,
                (since,),
            )
        except Exception:
            logger.exception("Failed to fetch article counts by day")
            return []

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_old_articles(self, db: DatabaseManager) -> dict[str, int]:
        """
        Remove articles that have exceeded their retention period.

        Retention policy
        ----------------
        - NOISE    : 7 days
        - LOW      : 14 days
        - MEDIUM   : 30 days
        - HIGH     : 30 days
        - CRITICAL : 90 days

        Orphaned ``news_index_impact`` rows are removed automatically via
        ``ON DELETE CASCADE`` in the schema.  A final sweep cleans up any
        rows that slipped through (e.g. from manual DB edits).

        Returns
        -------
        dict[str, int]:
            ``{severity: rows_deleted}``
        """
        now = datetime.now(tz=_IST)
        deleted: dict[str, int] = {}

        for category, days in _RETENTION_DAYS.items():
            cutoff = (now - timedelta(days=days)).isoformat()
            try:
                cursor = db.execute(
                    "DELETE FROM news_articles "
                    "WHERE impact_category = ? AND published_at < ?",
                    (category, cutoff),
                )
                deleted[category] = cursor.rowcount
            except Exception:
                logger.exception("Failed to cleanup %s articles", category)
                deleted[category] = 0

        # Belt-and-suspenders: remove orphaned impact rows
        try:
            db.execute(
                """
                DELETE FROM news_index_impact
                WHERE news_id NOT IN (SELECT id FROM news_articles)
                """
            )
        except Exception:
            logger.exception("Failed to cleanup orphaned news_index_impact rows")

        total = sum(deleted.values())
        if total:
            logger.info(
                "cleanup_old_articles: removed %d articles total — %s",
                total,
                deleted,
            )
        return deleted
