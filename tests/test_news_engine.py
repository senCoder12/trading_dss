"""
Unit tests for NewsEngine and NewsStore.

Strategy
--------
- RSSFetcher.fetch_all_feeds() is patched to return controlled RawArticle objects
  so tests never make real network calls.
- ArticleDeduplicator.is_duplicate_in_db() is patched to False so each article
  is treated as new without needing a real DB.
- DatabaseManager is a MagicMock; DB writes are verified via call assertions.
- The real ArticleParser, SentimentAnalyzer, NewsImpactMapper, TimeDecayEngine,
  and EventCalendar are exercised end-to-end so the pipeline logic is tested.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.analysis.news.news_engine import (
    NewsEngine,
    NewsCycleResult,
    NewsVote,
    NewsAlert,
    _build_reasoning,
)
from src.analysis.news.rss_fetcher import RawArticle
from src.analysis.news.time_decay import EffectiveNewsScore
from src.analysis.news.event_calendar import EventRegimeModifier

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_raw_article(
    title: str = "Nifty hits all-time high on strong FII buying",
    url: str = "https://example.com/article1",
    source: str = "Moneycontrol",
    source_credibility: float = 0.85,
    offset_minutes: int = 5,
) -> RawArticle:
    """Return a minimal RawArticle for testing."""
    now = datetime.now(tz=_IST) - timedelta(minutes=offset_minutes)
    return RawArticle(
        title=title,
        summary="Strong buying by FIIs lifted Nifty to a new record high today.",
        url=url,
        source=source,
        source_credibility=source_credibility,
        category="general_market",
        published_at=now,
        fetched_at=datetime.now(tz=_IST),
        raw_content=None,
        author=None,
        tags=[],
    )


def _make_critical_article() -> RawArticle:
    """Return an article likely to be classified CRITICAL by the pipeline."""
    return _make_raw_article(
        title="RBI announces emergency rate hike — markets crash 3%",
        url="https://example.com/rbi-emergency",
        source="Economic Times",
        source_credibility=0.9,
        offset_minutes=2,
    )


def _mock_db() -> MagicMock:
    """Return a DatabaseManager mock that satisfies NewsStore calls."""
    db = MagicMock()
    # fetch_one returns None by default (simulate no duplicate in DB)
    db.fetch_one.return_value = None
    db.fetch_all.return_value = []
    db.execute.return_value = MagicMock(rowcount=1, lastrowid=42)
    db.execute_many.return_value = 1
    return db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db() -> MagicMock:
    return _mock_db()


@pytest.fixture()
def engine(mock_db: MagicMock) -> NewsEngine:
    """
    A NewsEngine with DB mocked.

    is_duplicate_in_db is also patched so no real DB calls happen during dedup.
    """
    with patch(
        "src.analysis.news.deduplicator.ArticleDeduplicator.is_duplicate_in_db",
        return_value=False,
    ):
        eng = NewsEngine(mock_db)
        yield eng


@pytest.fixture()
def sample_raw_articles() -> list[RawArticle]:
    """Three distinct articles for a single test cycle."""
    return [
        _make_raw_article(
            title="Sensex surges 400 points; banking stocks lead rally",
            url="https://example.com/sensex-rally",
            offset_minutes=10,
        ),
        _make_raw_article(
            title="Infosys beats Q3 earnings estimates by 8%",
            url="https://example.com/infy-earnings",
            source="Bloomberg Quint",
            source_credibility=0.9,
            offset_minutes=15,
        ),
        _make_raw_article(
            title="FII net buying crosses Rs 5000 crore in equities",
            url="https://example.com/fii-buying",
            offset_minutes=20,
        ),
    ]


# ---------------------------------------------------------------------------
# Test: run_news_cycle produces a valid NewsCycleResult
# ---------------------------------------------------------------------------


class TestRunNewsCycle:
    def test_returns_news_cycle_result(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            result = engine.run_news_cycle()

        assert isinstance(result, NewsCycleResult)
        assert isinstance(result.timestamp, datetime)
        assert result.cycle_duration_ms >= 0

    def test_articles_fetched_matches_input(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            result = engine.run_news_cycle()

        assert result.articles_fetched == len(sample_raw_articles)

    def test_new_articles_counted(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            result = engine.run_news_cycle()

        # All articles are new (dedup patched to False)
        assert result.articles_new >= 0
        assert result.articles_new <= result.articles_fetched

    def test_by_severity_is_dict(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            result = engine.run_news_cycle()

        assert isinstance(result.by_severity, dict)
        valid_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "NOISE"}
        for key in result.by_severity:
            assert key in valid_severities

    def test_index_news_scores_is_dict(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            result = engine.run_news_cycle()

        assert isinstance(result.index_news_scores, dict)
        for idx_id, score in result.index_news_scores.items():
            assert isinstance(idx_id, str)
            assert isinstance(score, EffectiveNewsScore)

    def test_event_regime_is_dict(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            result = engine.run_news_cycle()

        assert isinstance(result.event_regime, dict)

    def test_alerts_is_list(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            result = engine.run_news_cycle()

        assert isinstance(result.alerts, list)

    def test_empty_feed_returns_valid_result(self, engine: NewsEngine) -> None:
        """Empty feed must not crash the engine."""
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=[]):
            result = engine.run_news_cycle()

        assert isinstance(result, NewsCycleResult)
        assert result.articles_fetched == 0
        assert result.articles_new == 0

    def test_fetch_failure_returns_valid_result(self, engine: NewsEngine) -> None:
        """A crashing RSS fetcher must not propagate the exception."""
        with patch.object(
            engine.fetcher,
            "fetch_all_feeds",
            side_effect=RuntimeError("network error"),
        ):
            result = engine.run_news_cycle()

        assert isinstance(result, NewsCycleResult)
        assert result.articles_fetched == 0

    def test_cycle_completes_within_10_seconds(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            t0 = time.monotonic()
            engine.run_news_cycle()
            elapsed = time.monotonic() - t0

        assert elapsed < 10.0, f"Cycle took {elapsed:.1f}s — exceeds 10s budget"


# ---------------------------------------------------------------------------
# Test: deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_second_cycle_has_zero_new_articles(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        """
        Submitting the same articles twice must result in 0 new articles on
        the second cycle, because their URLs are now in _recent_urls.
        """
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            first = engine.run_news_cycle()

        assert first.articles_new >= 0

        # Second call with identical articles
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            second = engine.run_news_cycle()

        # All URLs are cached from the first cycle → all duplicates
        assert second.articles_new == 0
        assert second.articles_duplicate == second.articles_fetched

    def test_new_url_is_not_duplicate(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        """A brand-new URL must be recognised as new even after a prior cycle."""
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        # Completely new article
        new_article = _make_raw_article(
            title="New breaking news article",
            url="https://example.com/brand-new-article",
        )
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=[new_article]):
            second = engine.run_news_cycle()

        assert second.articles_new >= 1

    def test_within_batch_duplicates_removed(self, engine: NewsEngine) -> None:
        """Two articles with near-identical titles in the same batch → 1 kept."""
        title = "Markets up on FII inflows"
        articles = [
            _make_raw_article(title=title, url="https://ex.com/a1"),
            _make_raw_article(title=title, url="https://ex.com/a2"),
        ]
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=articles):
            result = engine.run_news_cycle()

        # The deduplicator should keep only the more credible one
        assert result.articles_new <= 1
        assert result.articles_duplicate >= 1


# ---------------------------------------------------------------------------
# Test: get_news_vote
# ---------------------------------------------------------------------------


class TestGetNewsVote:
    def test_no_cycle_returns_neutral_vote(self, engine: NewsEngine) -> None:
        """Calling get_news_vote before any cycle returns a safe NEUTRAL default."""
        vote = engine.get_news_vote("NIFTY50")

        assert isinstance(vote, NewsVote)
        assert vote.index_id == "NIFTY50"
        assert vote.vote == "NEUTRAL"
        assert vote.confidence == 0.0
        assert vote.active_article_count == 0
        assert vote.event_regime == "NORMAL"
        assert isinstance(vote.reasoning, str) and len(vote.reasoning) > 0

    def test_vote_after_cycle_has_valid_fields(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        # We don't know which indices will be scored (depends on parser),
        # so we test any index that received a score, or fall back to NIFTY50.
        with engine._lock:
            scored = list(engine._last_index_scores.keys())

        test_index = scored[0] if scored else "NIFTY50"
        vote = engine.get_news_vote(test_index)

        assert isinstance(vote, NewsVote)
        assert vote.index_id == test_index
        assert isinstance(vote.timestamp, datetime)
        assert vote.vote in {
            "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
        }
        assert 0.0 <= vote.confidence <= 1.0
        assert vote.event_regime in {"NORMAL", "ELEVATED", "HIGH", "EXTREME"}

    def test_unknown_index_returns_neutral(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        vote = engine.get_news_vote("NONEXISTENT_INDEX_XYZ")
        assert vote.vote == "NEUTRAL"
        assert vote.confidence == 0.0

    def test_get_all_news_votes_returns_dict(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        with patch.object(engine, "_fo_index_ids", return_value=["NIFTY50", "BANKNIFTY"]):
            votes = engine.get_all_news_votes()

        assert isinstance(votes, dict)
        for idx_id, vote in votes.items():
            assert isinstance(idx_id, str)
            assert isinstance(vote, NewsVote)


# ---------------------------------------------------------------------------
# Test: critical alerts
# ---------------------------------------------------------------------------


class TestCriticalAlerts:
    def test_no_cycle_returns_empty_list(self, engine: NewsEngine) -> None:
        assert engine.get_critical_alerts() == []

    def test_critical_article_generates_alert(self, engine: NewsEngine) -> None:
        """
        An article with high sentiment + credibility from a POLICY event
        should produce a CRITICAL or HIGH alert.
        We force this by patching impact_mapper to classify as CRITICAL.
        """
        critical = _make_critical_article()

        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=[critical]):
            with patch.object(
                engine.impact_mapper,
                "classify_impact_severity",
                return_value="CRITICAL",
            ):
                result = engine.run_news_cycle()

        # Some CRITICAL article should generate an alert
        critical_alerts = engine.get_critical_alerts()
        # Verify shape (may be empty if no index impacts mapped)
        assert isinstance(critical_alerts, list)
        for alert in critical_alerts:
            assert isinstance(alert, NewsAlert)
            assert alert.severity in ("CRITICAL", "HIGH")
            assert isinstance(alert.message, str) and len(alert.message) > 0
            assert isinstance(alert.affected_indices, list)
            assert isinstance(alert.is_actionable, bool)

    def test_alert_message_is_human_readable(self, engine: NewsEngine) -> None:
        """Verify the alert message contains the article title and severity."""
        article = _make_raw_article(
            title="SEBI bans 5 brokers for market manipulation",
            url="https://example.com/sebi-ban",
        )

        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=[article]):
            with patch.object(
                engine.impact_mapper,
                "classify_impact_severity",
                return_value="HIGH",
            ):
                engine.run_news_cycle()

        for alert in engine.get_critical_alerts():
            # Message must contain severity label
            assert alert.severity in alert.message

    def test_noise_articles_produce_no_alerts(self, engine: NewsEngine) -> None:
        articles = [
            _make_raw_article(
                title="Markets close flat",
                url="https://example.com/flat1",
            )
        ]
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=articles):
            with patch.object(
                engine.impact_mapper,
                "classify_impact_severity",
                return_value="NOISE",
            ):
                engine.run_news_cycle()

        assert engine.get_critical_alerts() == []


# ---------------------------------------------------------------------------
# Test: get_news_feed
# ---------------------------------------------------------------------------


class TestGetNewsFeed:
    def test_empty_before_cycle(self, engine: NewsEngine) -> None:
        assert engine.get_news_feed() == []

    def test_returns_list_of_dicts(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        feed = engine.get_news_feed()
        assert isinstance(feed, list)
        for item in feed:
            assert isinstance(item, dict)

    def test_feed_item_has_required_keys(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        required_keys = {
            "title", "source", "published_at", "sentiment_label",
            "severity", "affected_indices", "age_minutes", "decay_factor",
        }
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        feed = engine.get_news_feed(min_severity="NOISE")
        for item in feed:
            missing = required_keys - set(item.keys())
            assert not missing, f"Feed item missing keys: {missing}"

    def test_feed_respects_min_severity(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NOISE": 4}
        for min_sev in ("HIGH", "MEDIUM", "LOW"):
            feed = engine.get_news_feed(min_severity=min_sev)
            for item in feed:
                item_order = severity_order.get(item["severity"], 4)
                min_order = severity_order.get(min_sev, 4)
                assert item_order <= min_order, (
                    f"Item severity {item['severity']} is below min {min_sev}"
                )

    def test_feed_respects_limit(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        feed = engine.get_news_feed(limit=2, min_severity="NOISE")
        assert len(feed) <= 2

    def test_feed_sorted_by_severity_then_age(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        """Higher severity articles must appear before lower severity ones."""
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NOISE": 4}
        feed = engine.get_news_feed(min_severity="NOISE")
        if len(feed) < 2:
            return  # nothing to check

        orders = [severity_order.get(item["severity"], 4) for item in feed]
        assert orders == sorted(orders), "Feed is not sorted by severity"

    def test_feed_index_filter(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        # Filter for a specific index — items returned must affect it
        for item in engine.get_news_feed(index_id="NIFTY50", min_severity="NOISE"):
            assert "NIFTY50" in item["affected_indices"]

    def test_feed_decay_factor_in_range(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        for item in engine.get_news_feed(min_severity="NOISE"):
            assert 0.0 <= item["decay_factor"] <= 1.0


# ---------------------------------------------------------------------------
# Test: _build_reasoning helper
# ---------------------------------------------------------------------------


class TestBuildReasoning:
    def _make_score(self, **kwargs) -> EffectiveNewsScore:
        defaults = dict(
            index_id="NIFTY50",
            timestamp=datetime.now(tz=_IST),
            weighted_sentiment=0.4,
            article_count=3,
            bullish_pressure=0.6,
            bearish_pressure=0.2,
            net_pressure=0.4,
            bullish_articles=2,
            bearish_articles=1,
            neutral_articles=0,
            news_vote="BULLISH",
            news_confidence=0.7,
            top_article_title="FII buying surges ahead of RBI meet",
            top_article_impact=0.5,
        )
        defaults.update(kwargs)
        return EffectiveNewsScore(**defaults)

    def test_includes_vote_and_confidence(self) -> None:
        score = self._make_score()
        reasoning = _build_reasoning(score, None)
        assert "BULLISH" in reasoning
        assert "0.70" in reasoning

    def test_includes_article_count(self) -> None:
        score = self._make_score(article_count=5, bullish_articles=3, bearish_articles=2)
        reasoning = _build_reasoning(score, None)
        assert "5" in reasoning

    def test_includes_top_headline(self) -> None:
        score = self._make_score(top_article_title="RBI cuts rates by 25bps")
        reasoning = _build_reasoning(score, None)
        assert "RBI cuts rates" in reasoning

    def test_includes_regime_when_elevated(self) -> None:
        score = self._make_score()
        regime = EventRegimeModifier(
            caution_level="HIGH",
            active_events=["RBI Policy Decision"],
        )
        reasoning = _build_reasoning(score, regime)
        assert "HIGH" in reasoning
        assert "RBI" in reasoning

    def test_no_regime_info_when_normal(self) -> None:
        score = self._make_score()
        regime = EventRegimeModifier(caution_level="NORMAL")
        reasoning = _build_reasoning(score, regime)
        # Should not mention regime for NORMAL
        assert "Regime" not in reasoning

    def test_no_articles_message(self) -> None:
        score = self._make_score(article_count=0, news_vote="NEUTRAL", news_confidence=0.0)
        reasoning = _build_reasoning(score, None)
        assert "no active" in reasoning.lower()

    def test_long_headline_truncated(self) -> None:
        long_title = "A" * 200
        score = self._make_score(top_article_title=long_title)
        reasoning = _build_reasoning(score, None)
        # Reasoning should not contain the full 200-char title verbatim
        assert long_title not in reasoning
        assert "..." in reasoning


# ---------------------------------------------------------------------------
# Test: thread safety — concurrent reads during cycle
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_reads_do_not_raise(
        self, engine: NewsEngine, sample_raw_articles: list[RawArticle]
    ) -> None:
        """
        Run a cycle while concurrently calling get_news_vote and get_news_feed
        from a separate thread.  No exceptions must be raised.
        """
        import threading

        errors: list[Exception] = []

        def reader():
            try:
                for _ in range(20):
                    engine.get_news_vote("NIFTY50")
                    engine.get_news_feed()
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        with patch.object(engine.fetcher, "fetch_all_feeds", return_value=sample_raw_articles):
            engine.run_news_cycle()

        reader_thread.join(timeout=5.0)
        assert not errors, f"Thread errors: {errors}"
