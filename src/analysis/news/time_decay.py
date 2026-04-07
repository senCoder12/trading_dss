"""
Time Decay Engine — models how news impact diminishes over time.

News impact follows exponential decay with event-type-specific half-lives.
A rate cut headline at 9:30 AM has massive impact at 9:35 AM but reduced
impact by 2:00 PM.  By next day it is mostly priced in.

Special rules handle pre-market news (mostly priced in at open), previous-day
news, and market-closed freeze (decay pauses outside trading hours).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional

from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.analysis.news.impact_mapper import MappedArticle
from src.utils.market_hours import (
    MarketSession,
    get_market_session,
    is_trading_day,
)

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# ---------------------------------------------------------------------------
# Severity weights (shared with impact_mapper but kept local to avoid import)
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHTS: dict[str, int] = {
    "CRITICAL": 5,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "NOISE": 0,
}

# ---------------------------------------------------------------------------
# Half-life tables (minutes)
# ---------------------------------------------------------------------------

_HALF_LIVES: dict[str, dict[str, float]] = {
    "CRITICAL": {
        "POLICY": 240.0,
        "GLOBAL": 360.0,
        "_DEFAULT": 180.0,
    },
    "HIGH": {
        "EARNINGS": 120.0,
        "MACRO": 180.0,
        "_DEFAULT": 90.0,
    },
    "MEDIUM": {
        "_DEFAULT": 60.0,
    },
    "LOW": {
        "_DEFAULT": 30.0,
    },
    "NOISE": {
        "_DEFAULT": 10.0,
    },
}

# Stale threshold — below this decay factor we consider the article dead
_STALE_THRESHOLD = 0.05

# Multipliers for pre-market / previous-day news
_PRE_MARKET_MULTIPLIER = 0.3
_PREVIOUS_DAY_MULTIPLIER = 0.1


def _parse_hhmm(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


def _market_open_time() -> time:
    return _parse_hhmm(settings.market_hours.market_open)


def _market_close_time() -> time:
    return _parse_hhmm(settings.market_hours.market_close)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DecayedArticle:
    """A mapped article with time-decay applied."""

    article: MappedArticle
    decay_factor: float
    effective_sentiment: float
    effective_impacts: dict[str, float] = field(default_factory=dict)
    is_stale: bool = False
    age_minutes: float = 0.0
    half_life_used: float = 0.0


@dataclass
class EffectiveNewsScore:
    """Aggregated, decay-weighted news score for a single index."""

    index_id: str
    timestamp: datetime

    # Aggregated score
    weighted_sentiment: float = 0.0
    article_count: int = 0

    # Directional
    bullish_pressure: float = 0.0
    bearish_pressure: float = 0.0
    net_pressure: float = 0.0

    # Sentiment breakdown
    bullish_articles: int = 0
    bearish_articles: int = 0
    neutral_articles: int = 0

    # News vote
    news_vote: str = "NEUTRAL"
    news_confidence: float = 0.0

    # Most influential article
    top_article_title: str | None = None
    top_article_impact: float = 0.0


# ---------------------------------------------------------------------------
# TimeDecayEngine
# ---------------------------------------------------------------------------


class TimeDecayEngine:
    """Compute time decay for news articles and aggregate effective scores."""

    # ------------------------------------------------------------------
    # Core decay calculation
    # ------------------------------------------------------------------

    def calculate_decay(
        self,
        article_time: datetime,
        current_time: datetime,
        impact_severity: str,
        event_type: str | None = None,
    ) -> float:
        """Return a decay factor between 0.0 (fully decayed) and 1.0 (fresh).

        Uses exponential decay: ``exp(-lambda * elapsed_minutes)`` where
        ``lambda = ln(2) / half_life``.
        """
        # Ensure IST-aware datetimes
        article_time = self._ensure_ist(article_time)
        current_time = self._ensure_ist(current_time)

        if current_time <= article_time:
            return 1.0

        half_life = self._get_half_life(impact_severity, event_type)
        lam = math.log(2) / half_life

        # Determine effective elapsed minutes (may freeze during market close)
        elapsed = self._effective_elapsed_minutes(article_time, current_time)

        decay = math.exp(-lam * elapsed)

        # --- Special multipliers ---

        mkt_open = _market_open_time()
        today = current_time.date()
        article_date = article_time.date()

        session = get_market_session(current_time)
        during_market = session == MarketSession.OPEN

        # Previous trading day news: only discount during an open market session.
        # When the market is closed, elapsed time is already frozen at the prior
        # market close — applying the multiplier again would double-penalise.
        if article_date < today and is_trading_day(article_date) and during_market:
            decay *= _PREVIOUS_DAY_MULTIPLIER

        # Pre-market news (article before today's open, now during market hours)
        elif article_date == today and during_market:
            article_t = article_time.time()
            if article_t < mkt_open:
                decay *= _PRE_MARKET_MULTIPLIER

        # Clamp
        if decay < 1e-6:
            decay = 0.0

        return decay

    # ------------------------------------------------------------------
    # Batch application
    # ------------------------------------------------------------------

    def apply_decay_to_articles(
        self,
        articles: list[MappedArticle],
        current_time: datetime,
    ) -> list[DecayedArticle]:
        """Apply time decay to a batch of mapped articles.

        Articles with ``decay_factor < 0.05`` are marked stale and filtered
        from the returned list.
        """
        current_time = self._ensure_ist(current_time)
        result: list[DecayedArticle] = []

        for ma in articles:
            pub_time = self._ensure_ist(ma.article.published_at)
            age_min = max(0.0, (current_time - pub_time).total_seconds() / 60.0)

            half_life = self._get_half_life(
                ma.impact_severity, ma.article.event_type
            )
            decay = self.calculate_decay(
                pub_time, current_time, ma.impact_severity, ma.article.event_type
            )

            eff_sentiment = ma.sentiment.adjusted_score * decay

            # Per-index effective impacts
            eff_impacts: dict[str, float] = {}
            for impact in ma.index_impacts:
                eff_impacts[impact.index_id] = impact.directional_impact * decay

            stale = decay < _STALE_THRESHOLD

            result.append(
                DecayedArticle(
                    article=ma,
                    decay_factor=decay,
                    effective_sentiment=eff_sentiment,
                    effective_impacts=eff_impacts,
                    is_stale=stale,
                    age_minutes=age_min,
                    half_life_used=half_life,
                )
            )

        # Filter stale
        active = [d for d in result if not d.is_stale]
        stale_count = len(result) - len(active)
        if stale_count:
            logger.debug("Filtered %d stale articles (decay < %.2f)", stale_count, _STALE_THRESHOLD)

        return active

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_effective_news_score(
        self,
        decayed_articles: list[DecayedArticle],
        index_id: str,
    ) -> EffectiveNewsScore:
        """Aggregate all active decayed articles for *index_id*."""
        now = datetime.now(tz=_IST)

        # Filter to articles that actually affect this index
        relevant: list[DecayedArticle] = []
        for da in decayed_articles:
            if index_id in da.effective_impacts:
                relevant.append(da)

        score = EffectiveNewsScore(index_id=index_id, timestamp=now)
        score.article_count = len(relevant)

        if not relevant:
            return score

        # --- Weighted sentiment ---
        weight_sum = 0.0
        sentiment_sum = 0.0
        top_impact = 0.0
        top_title: str | None = None

        for da in relevant:
            sev_w = _SEVERITY_WEIGHTS.get(da.article.impact_severity, 1)
            # Find relevance for this index
            relevance = 1.0
            for imp in da.article.index_impacts:
                if imp.index_id == index_id:
                    relevance = imp.relevance_score
                    break

            w = relevance * sev_w
            sentiment_sum += da.effective_sentiment * w
            weight_sum += w

            eff_imp = da.effective_impacts.get(index_id, 0.0)

            # Directional buckets
            if eff_imp > 0:
                score.bullish_pressure += eff_imp
                score.bullish_articles += 1
            elif eff_imp < 0:
                score.bearish_pressure += abs(eff_imp)
                score.bearish_articles += 1
            else:
                score.neutral_articles += 1

            # Track top article
            if abs(eff_imp) > top_impact:
                top_impact = abs(eff_imp)
                top_title = da.article.article.title

        score.weighted_sentiment = sentiment_sum / weight_sum if weight_sum else 0.0
        score.net_pressure = score.bullish_pressure - score.bearish_pressure
        score.top_article_title = top_title
        score.top_article_impact = top_impact

        # --- Vote logic ---
        score.news_vote, score.news_confidence = self._compute_vote(score)

        return score

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_ist(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=_IST)
        return dt.astimezone(_IST)

    @staticmethod
    def _get_half_life(severity: str, event_type: str | None) -> float:
        severity_table = _HALF_LIVES.get(severity, _HALF_LIVES["MEDIUM"])
        if event_type and event_type.upper() in severity_table:
            return severity_table[event_type.upper()]
        return severity_table["_DEFAULT"]

    def _effective_elapsed_minutes(
        self, article_time: datetime, current_time: datetime
    ) -> float:
        """Compute elapsed minutes, freezing decay during non-trading hours.

        If the market is currently closed the decay is frozen at the value
        it had at the previous market close.
        """
        session = get_market_session(current_time)
        mkt_close = _market_close_time()
        mkt_open = _market_open_time()

        # If market is currently open, use real elapsed time
        if session == MarketSession.OPEN:
            return (current_time - article_time).total_seconds() / 60.0

        # Market is closed — freeze decay at market close time
        today = current_time.date()

        # Find the most recent market close before current_time
        if is_trading_day(today) and current_time.time() >= mkt_close:
            # Today was a trading day, market already closed
            freeze_dt = datetime(
                today.year, today.month, today.day,
                mkt_close.hour, mkt_close.minute, tzinfo=_IST,
            )
        else:
            # Walk backward to find last trading day
            d = today - timedelta(days=1)
            while not is_trading_day(d):
                d -= timedelta(days=1)
            freeze_dt = datetime(
                d.year, d.month, d.day,
                mkt_close.hour, mkt_close.minute, tzinfo=_IST,
            )

        # If article was published after the freeze point, use real elapsed
        # time since publication — weekend/holiday news should still decay.
        if article_time >= freeze_dt:
            return (current_time - article_time).total_seconds() / 60.0

        # Article published before freeze: count only up to market close
        # (decay is frozen during the non-trading gap).
        return (freeze_dt - article_time).total_seconds() / 60.0

    @staticmethod
    def _compute_vote(score: EffectiveNewsScore) -> tuple[str, float]:
        """Derive news_vote and news_confidence from aggregated pressures."""
        net = score.net_pressure
        n = score.article_count
        bull = score.bullish_articles
        bear = score.bearish_articles

        # Determine vote
        vote = "NEUTRAL"
        if net > 0.6 and n >= 3:
            vote = "STRONG_BULLISH"
        elif net > 0.3 and bull > bear * 1.5:
            vote = "BULLISH"
        elif net < -0.6 and n >= 3:
            vote = "STRONG_BEARISH"
        elif net < -0.3 and bear > bull * 1.5:
            vote = "BEARISH"

        # Confidence
        if n == 0:
            return vote, 0.0

        # Base confidence from article count
        count_factor = min(n / 8.0, 1.0)  # saturates at 8 articles

        # Agreement ratio
        total_directional = bull + bear
        if total_directional > 0:
            majority = max(bull, bear)
            agreement = majority / total_directional
        else:
            agreement = 0.5

        # Severity boost (more high-severity articles → more confident)
        confidence = 0.3 * count_factor + 0.4 * agreement + 0.3 * min(abs(net), 1.0)

        # Cap at 0.4 for single article
        if n == 1:
            confidence = min(confidence, 0.4)

        # Multiple agreeing articles can push to 0.8
        if n >= 5 and agreement >= 0.8:
            confidence = max(confidence, 0.7)

        confidence = min(confidence, 0.9)

        return vote, round(confidence, 3)
