"""
Tests for TimeDecayEngine — time decay, batch application, and news score aggregation.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from src.analysis.news.article_parser import ParsedArticle
from src.analysis.news.impact_mapper import IndexImpact, MappedArticle
from src.analysis.news.sentiment_analyzer import SentimentResult
from src.analysis.news.time_decay import (
    DecayedArticle,
    EffectiveNewsScore,
    TimeDecayEngine,
    _HALF_LIVES,
    _STALE_THRESHOLD,
)

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article(
    title: str = "Test article",
    event_type: str | None = None,
    published_at: datetime | None = None,
) -> ParsedArticle:
    return ParsedArticle(
        title=title,
        clean_text=title,
        url="https://example.com/test",
        source="test_source",
        source_credibility=0.8,
        category="market",
        published_at=published_at or datetime.now(tz=_IST),
        fetched_at=datetime.now(tz=_IST),
        mentioned_companies=[],
        mentioned_sectors=[],
        mentioned_indices=[],
        event_type=event_type,
        is_market_hours_relevant=True,
        language="en",
        word_count=5,
    )


def _make_sentiment(
    adjusted_score: float = 0.5,
    label: str = "BULLISH",
) -> SentimentResult:
    return SentimentResult(
        raw_vader_score=adjusted_score * 0.8,
        keyword_boost=0.1,
        source_weight=0.9,
        adjusted_score=adjusted_score,
        sentiment_label=label,
        confidence=0.7,
    )


def _make_mapped(
    title: str = "Test",
    sentiment_score: float = 0.5,
    sentiment_label: str = "BULLISH",
    severity: str = "MEDIUM",
    event_type: str | None = None,
    published_at: datetime | None = None,
    index_impacts: list[IndexImpact] | None = None,
) -> MappedArticle:
    art = _make_article(title=title, event_type=event_type, published_at=published_at)
    sent = _make_sentiment(adjusted_score=sentiment_score, label=sentiment_label)
    impacts = index_impacts or [
        IndexImpact(
            index_id="NIFTY50",
            relevance_score=0.8,
            mapping_method="KEYWORD_MAPPING",
            sentiment_score=sentiment_score,
            directional_impact=sentiment_score * 0.8,
        )
    ]
    return MappedArticle(
        article=art,
        sentiment=sent,
        index_impacts=impacts,
        impact_severity=severity,
        primary_index="NIFTY50",
        is_actionable=True,
    )


# ---------------------------------------------------------------------------
# Tests: Core decay calculation
# ---------------------------------------------------------------------------


class TestCalculateDecay:
    """Test exponential decay factor at various time intervals."""

    def setup_method(self) -> None:
        self.engine = TimeDecayEngine()

    def test_zero_elapsed_returns_one(self) -> None:
        t = datetime(2026, 4, 6, 10, 0, tzinfo=_IST)
        assert self.engine.calculate_decay(t, t, "MEDIUM") == 1.0

    def test_future_article_returns_one(self) -> None:
        article = datetime(2026, 4, 6, 11, 0, tzinfo=_IST)
        now = datetime(2026, 4, 6, 10, 0, tzinfo=_IST)
        assert self.engine.calculate_decay(article, now, "MEDIUM") == 1.0

    def test_at_half_life_returns_half(self) -> None:
        """At exactly one half-life, decay should be ~0.5."""
        # MEDIUM default half-life = 60 min
        # Use a time during market hours on a trading day (Monday)
        article = datetime(2026, 4, 6, 10, 0, tzinfo=_IST)  # Monday 10:00
        current = datetime(2026, 4, 6, 11, 0, tzinfo=_IST)  # Monday 11:00
        decay = self.engine.calculate_decay(article, current, "MEDIUM")
        assert abs(decay - 0.5) < 0.01

    def test_at_two_half_lives_returns_quarter(self) -> None:
        """At 2x half-life, decay should be ~0.25."""
        article = datetime(2026, 4, 6, 10, 0, tzinfo=_IST)
        current = datetime(2026, 4, 6, 12, 0, tzinfo=_IST)  # +120 min
        decay = self.engine.calculate_decay(article, current, "MEDIUM")
        assert abs(decay - 0.25) < 0.01

    def test_critical_policy_half_life(self) -> None:
        """CRITICAL + POLICY half-life = 240 min."""
        article = datetime(2026, 4, 6, 10, 0, tzinfo=_IST)
        current = datetime(2026, 4, 6, 14, 0, tzinfo=_IST)  # +240 min
        decay = self.engine.calculate_decay(article, current, "CRITICAL", "POLICY")
        assert abs(decay - 0.5) < 0.01

    def test_critical_global_half_life(self) -> None:
        """CRITICAL + GLOBAL half-life = 360 min."""
        article = datetime(2026, 4, 6, 9, 30, tzinfo=_IST)
        current = datetime(2026, 4, 6, 15, 30, tzinfo=_IST)  # +360 min = market close
        decay = self.engine.calculate_decay(article, current, "CRITICAL", "GLOBAL")
        assert abs(decay - 0.5) < 0.02

    def test_noise_decays_fast(self) -> None:
        """NOISE half-life = 10 min — after 40 min during market hours, very small."""
        article = datetime(2026, 4, 6, 10, 0, tzinfo=_IST)
        current = datetime(2026, 4, 6, 10, 40, tzinfo=_IST)  # 4 half-lives
        decay = self.engine.calculate_decay(article, current, "NOISE")
        assert decay < 0.07  # 0.5^4 = 0.0625

    def test_high_earnings_half_life(self) -> None:
        """HIGH + EARNINGS = 120 min half-life."""
        article = datetime(2026, 4, 6, 10, 0, tzinfo=_IST)
        current = datetime(2026, 4, 6, 12, 0, tzinfo=_IST)
        decay = self.engine.calculate_decay(article, current, "HIGH", "EARNINGS")
        assert abs(decay - 0.5) < 0.01

    def test_low_default_half_life(self) -> None:
        """LOW default = 30 min half-life."""
        article = datetime(2026, 4, 6, 10, 0, tzinfo=_IST)
        current = datetime(2026, 4, 6, 10, 30, tzinfo=_IST)
        decay = self.engine.calculate_decay(article, current, "LOW")
        assert abs(decay - 0.5) < 0.01


class TestPreMarketDecay:
    """Pre-market news should be penalized once market opens."""

    def setup_method(self) -> None:
        self.engine = TimeDecayEngine()

    def test_pre_market_article_during_market(self) -> None:
        """Article published at 08:00, now 10:00 during market hours → 0.3x multiplier."""
        # Monday 2026-04-06
        article = datetime(2026, 4, 6, 8, 0, tzinfo=_IST)
        current = datetime(2026, 4, 6, 10, 0, tzinfo=_IST)
        decay = self.engine.calculate_decay(article, current, "HIGH")
        # Without pre-market penalty: exp(-ln2/90 * 120) ≈ 0.397
        # With 0.3x: ≈ 0.119
        assert decay < 0.2
        assert decay > 0.0


class TestPreviousDayDecay:
    """Previous trading day news should be nearly fully priced in."""

    def setup_method(self) -> None:
        self.engine = TimeDecayEngine()

    def test_previous_day_article(self) -> None:
        """Article from Friday during Monday market → 0.1x multiplier."""
        # Friday 2026-04-03 at 14:00, Monday 2026-04-06 at 10:00
        article = datetime(2026, 4, 3, 14, 0, tzinfo=_IST)
        current = datetime(2026, 4, 6, 10, 0, tzinfo=_IST)
        decay = self.engine.calculate_decay(article, current, "CRITICAL")
        assert decay < 0.1  # 0.1x multiplier on already-decayed value


class TestMarketClosedFreeze:
    """Decay freezes at market close during non-trading hours."""

    def setup_method(self) -> None:
        self.engine = TimeDecayEngine()

    def test_decay_frozen_after_close(self) -> None:
        """Article at 14:00, checked at 18:00 same day → same as at 15:30."""
        # Monday
        article = datetime(2026, 4, 6, 14, 0, tzinfo=_IST)
        at_close = datetime(2026, 4, 6, 15, 30, tzinfo=_IST)
        at_evening = datetime(2026, 4, 6, 18, 0, tzinfo=_IST)

        decay_close = self.engine.calculate_decay(article, at_close, "MEDIUM")
        decay_evening = self.engine.calculate_decay(article, at_evening, "MEDIUM")

        assert abs(decay_close - decay_evening) < 0.01

    def test_decay_frozen_on_weekend(self) -> None:
        """Article from Friday market, checked Saturday → frozen at Friday close."""
        # Use 2026-04-10 (regular Friday); 2026-04-03 is Good Friday (market holiday)
        article = datetime(2026, 4, 10, 14, 0, tzinfo=_IST)  # Friday
        friday_close = datetime(2026, 4, 10, 15, 30, tzinfo=_IST)
        saturday = datetime(2026, 4, 11, 12, 0, tzinfo=_IST)

        decay_close = self.engine.calculate_decay(article, friday_close, "MEDIUM")
        decay_sat = self.engine.calculate_decay(article, saturday, "MEDIUM")

        assert abs(decay_close - decay_sat) < 0.01


# ---------------------------------------------------------------------------
# Tests: Batch application
# ---------------------------------------------------------------------------


class TestApplyDecayToArticles:

    def setup_method(self) -> None:
        self.engine = TimeDecayEngine()

    def test_fresh_articles_not_stale(self) -> None:
        now = datetime(2026, 4, 6, 10, 30, tzinfo=_IST)
        articles = [
            _make_mapped(
                title="Fresh news",
                published_at=datetime(2026, 4, 6, 10, 25, tzinfo=_IST),
            )
        ]
        result = self.engine.apply_decay_to_articles(articles, now)
        assert len(result) == 1
        assert result[0].decay_factor > 0.9
        assert not result[0].is_stale

    def test_stale_articles_filtered(self) -> None:
        now = datetime(2026, 4, 6, 14, 0, tzinfo=_IST)
        # NOISE article from 4 hours ago → many half-lives
        articles = [
            _make_mapped(
                title="Old noise",
                severity="NOISE",
                published_at=datetime(2026, 4, 6, 10, 0, tzinfo=_IST),
            )
        ]
        result = self.engine.apply_decay_to_articles(articles, now)
        assert len(result) == 0  # filtered as stale

    def test_effective_sentiment_scaled(self) -> None:
        now = datetime(2026, 4, 6, 11, 0, tzinfo=_IST)
        articles = [
            _make_mapped(
                sentiment_score=0.6,
                severity="MEDIUM",
                published_at=datetime(2026, 4, 6, 10, 0, tzinfo=_IST),
            )
        ]
        result = self.engine.apply_decay_to_articles(articles, now)
        assert len(result) == 1
        # At half-life: effective_sentiment ≈ 0.6 * 0.5 = 0.3
        assert abs(result[0].effective_sentiment - 0.3) < 0.05

    def test_effective_impacts_per_index(self) -> None:
        now = datetime(2026, 4, 6, 10, 5, tzinfo=_IST)
        articles = [
            _make_mapped(
                published_at=datetime(2026, 4, 6, 10, 0, tzinfo=_IST),
                index_impacts=[
                    IndexImpact(
                        index_id="NIFTY50",
                        relevance_score=0.9,
                        mapping_method="DIRECT_MENTION",
                        directional_impact=0.4,
                    ),
                    IndexImpact(
                        index_id="BANKNIFTY",
                        relevance_score=0.5,
                        mapping_method="KEYWORD_MAPPING",
                        directional_impact=0.2,
                    ),
                ],
            )
        ]
        result = self.engine.apply_decay_to_articles(articles, now)
        assert "NIFTY50" in result[0].effective_impacts
        assert "BANKNIFTY" in result[0].effective_impacts
        # Very fresh → impacts close to original
        assert result[0].effective_impacts["NIFTY50"] > 0.35


# ---------------------------------------------------------------------------
# Tests: Effective news score aggregation
# ---------------------------------------------------------------------------


class TestGetEffectiveNewsScore:

    def setup_method(self) -> None:
        self.engine = TimeDecayEngine()

    def _make_decayed(
        self,
        title: str,
        eff_sentiment: float,
        eff_impact: float,
        severity: str = "MEDIUM",
        index_id: str = "NIFTY50",
        relevance: float = 0.8,
    ) -> DecayedArticle:
        ma = _make_mapped(
            title=title,
            sentiment_score=eff_sentiment,
            severity=severity,
            index_impacts=[
                IndexImpact(
                    index_id=index_id,
                    relevance_score=relevance,
                    mapping_method="DIRECT_MENTION",
                    directional_impact=eff_impact,
                )
            ],
        )
        return DecayedArticle(
            article=ma,
            decay_factor=1.0,
            effective_sentiment=eff_sentiment,
            effective_impacts={index_id: eff_impact},
            is_stale=False,
            age_minutes=5.0,
            half_life_used=60.0,
        )

    def test_empty_articles(self) -> None:
        score = self.engine.get_effective_news_score([], "NIFTY50")
        assert score.article_count == 0
        assert score.news_vote == "NEUTRAL"

    def test_single_bullish_article(self) -> None:
        articles = [self._make_decayed("RBI cuts rate", 0.7, 0.5)]
        score = self.engine.get_effective_news_score(articles, "NIFTY50")
        assert score.article_count == 1
        assert score.bullish_articles == 1
        assert score.bullish_pressure > 0
        assert score.news_confidence <= 0.4  # single article cap

    def test_multiple_bullish_strong(self) -> None:
        articles = [
            self._make_decayed("Good news 1", 0.6, 0.25, severity="HIGH"),
            self._make_decayed("Good news 2", 0.7, 0.25, severity="HIGH"),
            self._make_decayed("Good news 3", 0.5, 0.15, severity="MEDIUM"),
        ]
        score = self.engine.get_effective_news_score(articles, "NIFTY50")
        assert score.article_count == 3
        assert score.net_pressure > 0.3
        assert score.news_vote in ("BULLISH", "STRONG_BULLISH")

    def test_bearish_vote(self) -> None:
        articles = [
            self._make_decayed("Bad news 1", -0.6, -0.25),
            self._make_decayed("Bad news 2", -0.5, -0.2),
        ]
        score = self.engine.get_effective_news_score(articles, "NIFTY50")
        assert score.bearish_articles == 2
        assert score.net_pressure < 0
        assert score.news_vote in ("BEARISH", "STRONG_BEARISH")

    def test_mixed_sentiment_neutral(self) -> None:
        articles = [
            self._make_decayed("Good", 0.5, 0.2),
            self._make_decayed("Bad", -0.5, -0.2),
        ]
        score = self.engine.get_effective_news_score(articles, "NIFTY50")
        assert abs(score.net_pressure) < 0.05
        assert score.news_vote == "NEUTRAL"

    def test_irrelevant_index_excluded(self) -> None:
        articles = [
            self._make_decayed("Bank news", 0.5, 0.3, index_id="BANKNIFTY")
        ]
        score = self.engine.get_effective_news_score(articles, "NIFTY50")
        assert score.article_count == 0

    def test_top_article_tracked(self) -> None:
        articles = [
            self._make_decayed("Small move", 0.3, 0.1),
            self._make_decayed("Big mover", 0.8, 0.6, severity="CRITICAL"),
        ]
        score = self.engine.get_effective_news_score(articles, "NIFTY50")
        assert score.top_article_title == "Big mover"
        assert score.top_article_impact == 0.6

    def test_high_agreement_confidence(self) -> None:
        """5+ agreeing articles → confidence can reach 0.7+."""
        articles = [
            self._make_decayed(f"Bull {i}", 0.5, 0.15, severity="HIGH")
            for i in range(6)
        ]
        score = self.engine.get_effective_news_score(articles, "NIFTY50")
        assert score.news_confidence >= 0.7
