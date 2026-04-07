"""
Phase 3 — News Analysis subsystem.

Public API
----------
NewsEngine          Master orchestrator; call run_news_cycle() every 2-3 min.
NewsCycleResult     Typed result returned by run_news_cycle().
NewsVote            Directional vote for a single index (consumed by Phase 4).
NewsAlert           High-priority alert (forwarded to Telegram in Phase 9).
NewsStore           DB read/write layer (used internally by NewsEngine).

RSSFetcher          Raw article fetcher (RSS/Atom feeds).
RawArticle          Raw feed item before parsing.
ArticleParser       Cleans text, extracts companies/sectors/indices, classifies events.
ParsedArticle       Structured article after parsing.
ArticleDeduplicator Removes near-duplicate articles within a batch and vs. DB.
SentimentAnalyzer   VADER + keyword-boosted sentiment scorer.
SentimentResult     Sentiment output with label, score, and confidence.
NewsImpactMapper    Maps articles to indices; classifies impact severity.
MappedArticle       Article enriched with index impacts and severity.
IndexImpact         Per-index impact record (relevance, method, severity).
MarketNewsSummary   Aggregated market-wide sentiment snapshot.
TimeDecayEngine     Applies exponential time decay to article impact scores.
EffectiveNewsScore  Decayed, aggregated score for a single index.
DecayedArticle      Single article after decay calculation.
EventCalendar       Tracks upcoming events (RBI, earnings, expiry, etc.).
EventRegimeModifier Position-size and volatility modifier for an event period.
UpcomingEvent       An individual calendar event with timing and severity.
"""

from .news_engine import NewsEngine, NewsCycleResult, NewsVote, NewsAlert
from .news_store import NewsStore
from .rss_fetcher import RSSFetcher, RawArticle
from .article_parser import ArticleParser, ParsedArticle
from .deduplicator import ArticleDeduplicator
from .sentiment_analyzer import SentimentAnalyzer, SentimentResult
from .impact_mapper import (
    NewsImpactMapper,
    MappedArticle,
    IndexImpact,
    MarketNewsSummary,
)
from .time_decay import TimeDecayEngine, EffectiveNewsScore, DecayedArticle
from .event_calendar import EventCalendar, EventRegimeModifier, UpcomingEvent

__all__ = [
    # Engine + results
    "NewsEngine",
    "NewsCycleResult",
    "NewsVote",
    "NewsAlert",
    "NewsStore",
    # Fetcher
    "RSSFetcher",
    "RawArticle",
    # Parser
    "ArticleParser",
    "ParsedArticle",
    # Deduplicator
    "ArticleDeduplicator",
    # Sentiment
    "SentimentAnalyzer",
    "SentimentResult",
    # Impact mapper
    "NewsImpactMapper",
    "MappedArticle",
    "IndexImpact",
    "MarketNewsSummary",
    # Time decay
    "TimeDecayEngine",
    "EffectiveNewsScore",
    "DecayedArticle",
    # Event calendar
    "EventCalendar",
    "EventRegimeModifier",
    "UpcomingEvent",
]
