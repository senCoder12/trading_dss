"""
News feed API endpoints.

GET /api/news/feed                      — news articles with filtering
GET /api/news/summary                   — market news summary
GET /api/news/sentiment/{index_id}      — news sentiment for one index
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_db, get_index_registry
from src.data.index_registry import IndexRegistry
from src.database import queries as Q
from src.database.db_manager import DatabaseManager
from src.utils.date_utils import get_ist_now

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Response models ──────────────────────────────────────────────────────────

class NewsArticle(BaseModel):
    id: Optional[int] = None
    title: str
    summary: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[str] = None
    sentiment: Optional[float] = None
    impact_category: Optional[str] = None
    source_credibility: Optional[float] = None
    related_indices: list[str] = Field(default_factory=list)


class NewsFeedResponse(BaseModel):
    articles: list[NewsArticle]
    count: int
    total: int


class NewsSummaryResponse(BaseModel):
    total_articles: int = 0
    by_severity: dict = Field(default_factory=dict)
    overall_sentiment: float = 0.0
    overall_sentiment_label: str = "NEUTRAL"
    most_impacted_indices: list[dict] = Field(default_factory=list)
    critical_count: int = 0


class NewsSentimentResponse(BaseModel):
    index_id: str
    article_count: int = 0
    weighted_sentiment: float = 0.0
    sentiment_label: str = "NEUTRAL"
    bullish_count: int = 0
    bearish_count: int = 0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sentiment_label(score: float) -> str:
    if score > 0.2:
        return "BULLISH"
    elif score > 0.05:
        return "SLIGHTLY_BULLISH"
    elif score < -0.2:
        return "BEARISH"
    elif score < -0.05:
        return "SLIGHTLY_BEARISH"
    return "NEUTRAL"


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/feed", response_model=NewsFeedResponse, summary="News articles with filtering")
async def get_news_feed(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    min_severity: Optional[str] = Query(
        default=None, description="CRITICAL, HIGH, MEDIUM, LOW, NOISE",
    ),
    index_id: Optional[str] = Query(default=None),
    days: int = Query(default=3, ge=1, le=30),
    db: DatabaseManager = Depends(get_db),
):
    now = get_ist_now()
    since = (now - timedelta(days=days)).isoformat()

    if index_id:
        # Filter by index via join
        idx = index_id.upper()
        clauses = ["na.published_at >= ?"]
        params: list = [since]

        severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NOISE"]
        if min_severity and min_severity.upper() in severity_order:
            allowed = severity_order[: severity_order.index(min_severity.upper()) + 1]
            placeholders = ", ".join("?" for _ in allowed)
            clauses.append(f"na.impact_category IN ({placeholders})")
            params.extend(allowed)

        where = " AND ".join(clauses)

        count_row = db.fetch_one(
            f"SELECT COUNT(*) AS cnt FROM news_articles na "
            f"JOIN news_index_impact nii ON na.id = nii.news_id "
            f"WHERE nii.index_id = ? AND {where}",
            (idx, *params),
        )
        total = count_row["cnt"] if count_row else 0

        rows = db.fetch_all(
            f"SELECT na.*, nii.relevance_score FROM news_articles na "
            f"JOIN news_index_impact nii ON na.id = nii.news_id "
            f"WHERE nii.index_id = ? AND {where} "
            f"ORDER BY na.published_at DESC LIMIT ? OFFSET ?",
            (idx, *params, limit, offset),
        )
    else:
        clauses = ["published_at >= ?"]
        params = [since]

        severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NOISE"]
        if min_severity and min_severity.upper() in severity_order:
            allowed = severity_order[: severity_order.index(min_severity.upper()) + 1]
            placeholders = ", ".join("?" for _ in allowed)
            clauses.append(f"impact_category IN ({placeholders})")
            params.extend(allowed)

        where = " AND ".join(clauses)

        count_row = db.fetch_one(
            f"SELECT COUNT(*) AS cnt FROM news_articles WHERE {where}",
            tuple(params),
        )
        total = count_row["cnt"] if count_row else 0

        rows = db.fetch_all(
            f"SELECT * FROM news_articles WHERE {where} "
            f"ORDER BY published_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )

    articles: list[NewsArticle] = []
    for r in rows:
        # Get related indices for this article
        related = db.fetch_all(
            Q.LIST_INDICES_FOR_NEWS, (r["id"],),
        )
        articles.append(NewsArticle(
            id=r.get("id"),
            title=r["title"],
            summary=r.get("summary"),
            source=r.get("source"),
            url=r.get("url"),
            published_at=r.get("published_at"),
            sentiment=r.get("adjusted_sentiment"),
            impact_category=r.get("impact_category"),
            source_credibility=r.get("source_credibility"),
            related_indices=[ri["index_id"] for ri in related],
        ))

    return NewsFeedResponse(articles=articles, count=len(articles), total=total)


@router.get("/summary", response_model=NewsSummaryResponse, summary="Market news summary")
async def get_news_summary(
    days: int = Query(default=1, ge=1, le=7),
    db: DatabaseManager = Depends(get_db),
):
    now = get_ist_now()
    since = (now - timedelta(days=days)).isoformat()

    # Count by severity
    severity_rows = db.fetch_all(
        "SELECT impact_category, COUNT(*) AS cnt FROM news_articles "
        "WHERE published_at >= ? GROUP BY impact_category",
        (since,),
    )
    by_severity = {r["impact_category"]: r["cnt"] for r in severity_rows if r.get("impact_category")}
    total = sum(by_severity.values())

    # Overall sentiment
    sentiment_row = db.fetch_one(
        "SELECT AVG(adjusted_sentiment) AS avg_sent FROM news_articles "
        "WHERE published_at >= ? AND adjusted_sentiment IS NOT NULL",
        (since,),
    )
    avg_sent = sentiment_row["avg_sent"] if sentiment_row and sentiment_row.get("avg_sent") else 0.0

    # Most impacted indices
    impact_rows = db.fetch_all(
        "SELECT nii.index_id, COUNT(*) AS article_count, "
        "AVG(na.adjusted_sentiment) AS avg_sentiment "
        "FROM news_index_impact nii "
        "JOIN news_articles na ON na.id = nii.news_id "
        "WHERE na.published_at >= ? "
        "GROUP BY nii.index_id "
        "ORDER BY article_count DESC LIMIT 10",
        (since,),
    )
    most_impacted = [
        {
            "index_id": r["index_id"],
            "article_count": r["article_count"],
            "avg_sentiment": round(r["avg_sentiment"], 4) if r.get("avg_sentiment") else 0.0,
        }
        for r in impact_rows
    ]

    return NewsSummaryResponse(
        total_articles=total,
        by_severity=by_severity,
        overall_sentiment=round(avg_sent, 4),
        overall_sentiment_label=_sentiment_label(avg_sent),
        most_impacted_indices=most_impacted,
        critical_count=by_severity.get("CRITICAL", 0),
    )


@router.get(
    "/sentiment/{index_id}",
    response_model=NewsSentimentResponse,
    summary="News sentiment for one index",
)
async def get_news_sentiment(
    index_id: str,
    days: int = Query(default=1, ge=1, le=7),
    db: DatabaseManager = Depends(get_db),
    registry: IndexRegistry = Depends(get_index_registry),
):
    index_id = index_id.upper()
    if registry.get_or_none(index_id) is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_id}")

    now = get_ist_now()
    since = (now - timedelta(days=days)).isoformat()

    row = db.fetch_one(Q.AGG_NEWS_SENTIMENT_FOR_INDEX, (index_id, since))
    if row is None or not row.get("article_count"):
        return NewsSentimentResponse(index_id=index_id)

    ws = row.get("weighted_sentiment") or 0.0
    return NewsSentimentResponse(
        index_id=index_id,
        article_count=row["article_count"],
        weighted_sentiment=round(ws, 4),
        sentiment_label=_sentiment_label(ws),
        bullish_count=row.get("bullish_count") or 0,
        bearish_count=row.get("bearish_count") or 0,
    )
