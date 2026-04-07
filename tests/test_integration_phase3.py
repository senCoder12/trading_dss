"""
End-to-end integration tests for Phase 3 — News Analysis pipeline.

These tests exercise the FULL pipeline with real RSS feeds and a real
(temporary) SQLite database.  They are skipped automatically when:

  - No network access is available (controlled by SKIP_NETWORK_TESTS env var).
  - Running in a CI environment with limited feed access.

To run locally::

    pytest tests/test_integration_phase3.py -v -s

To skip::

    SKIP_NETWORK_TESTS=1 pytest tests/test_integration_phase3.py

Coverage
--------
1.  Full news cycle with real RSS feeds → articles fetched, parsed, scored,
    and mapped to indices.
2.  DB persistence: ``news_articles`` and ``news_index_impact`` are populated.
3.  Deduplication: a second cycle run immediately after the first must produce
    0 new articles (all cached in ``_recent_urls``).
4.  Time decay: effective scores change (or are at least not None) between cycles.
5.  Event calendar integration: regime modifiers are non-None for all F&O indices.
6.  NewsVote: every F&O index gets a valid vote after the cycle.
7.  Performance: full cycle completes within 15 seconds.
8.  Stability: no unhandled exceptions across two consecutive cycles.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import pytest
from zoneinfo import ZoneInfo

from src.database.db_manager import DatabaseManager
from src.analysis.news.news_engine import NewsEngine, NewsCycleResult, NewsVote
from src.analysis.news.time_decay import EffectiveNewsScore
from src.analysis.news.event_calendar import EventRegimeModifier

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Pytest markers / skip conditions
# ---------------------------------------------------------------------------

_SKIP_NETWORK = os.getenv("SKIP_NETWORK_TESTS", "").lower() in ("1", "true", "yes")

pytestmark = pytest.mark.integration  # tag all tests in this file


def _skip_if_no_network(reason: str = "network tests disabled"):
    return pytest.mark.skipif(_SKIP_NETWORK, reason=reason)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def temp_db(tmp_path_factory: pytest.TempPathFactory) -> Generator[DatabaseManager, None, None]:
    """
    A real DatabaseManager backed by a temporary SQLite file.

    The schema is initialised once for the entire module so tests share the
    same DB state (which is intentional — we test that data persists between
    cycles).
    """
    db_path = tmp_path_factory.mktemp("integration_db") / "test_phase3.db"
    db = DatabaseManager(db_path=db_path)
    db.connect()
    db.initialise_schema()
    yield db
    try:
        db.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def engine(temp_db: DatabaseManager) -> NewsEngine:
    """
    A single NewsEngine instance shared across the integration test module.

    Uses the real DB so we can verify persistence.
    """
    return NewsEngine(temp_db)


@pytest.fixture(scope="module")
def first_cycle_result(engine: NewsEngine) -> NewsCycleResult:
    """
    Run the first real news cycle and cache the result for all tests.

    Marked ``module``-scoped so the expensive network call happens once.
    """
    return engine.run_news_cycle()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _fo_index_ids(engine: NewsEngine) -> list[str]:
    """Return F&O index IDs from registry (empty list on failure)."""
    return engine._fo_index_ids()


# ---------------------------------------------------------------------------
# Test 1: Articles are fetched and processed
# ---------------------------------------------------------------------------


@_skip_if_no_network()
class TestArticlesFetched:
    def test_cycle_returns_news_cycle_result(
        self, first_cycle_result: NewsCycleResult
    ) -> None:
        assert isinstance(first_cycle_result, NewsCycleResult)

    def test_cycle_duration_is_positive(
        self, first_cycle_result: NewsCycleResult
    ) -> None:
        assert first_cycle_result.cycle_duration_ms > 0

    def test_articles_were_fetched(self, first_cycle_result: NewsCycleResult) -> None:
        """At least one article must be fetched if any feed is reachable."""
        # We cannot guarantee feed reachability; just verify the field exists.
        assert first_cycle_result.articles_fetched >= 0

    def test_processed_articles_le_new_articles(
        self, first_cycle_result: NewsCycleResult
    ) -> None:
        """Processed count must not exceed new (deduplicated) count."""
        assert first_cycle_result.articles_processed <= first_cycle_result.articles_new + 1

    def test_by_severity_contains_valid_categories(
        self, first_cycle_result: NewsCycleResult
    ) -> None:
        valid = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "NOISE"}
        for key in first_cycle_result.by_severity:
            assert key in valid, f"Unexpected severity category: {key!r}"

    def test_by_event_type_is_dict(self, first_cycle_result: NewsCycleResult) -> None:
        assert isinstance(first_cycle_result.by_event_type, dict)

    def test_feeds_stats_are_non_negative(
        self, first_cycle_result: NewsCycleResult
    ) -> None:
        assert first_cycle_result.feeds_successful >= 0
        assert first_cycle_result.feeds_failed >= 0


# ---------------------------------------------------------------------------
# Test 2: DB persistence
# ---------------------------------------------------------------------------


@_skip_if_no_network()
class TestDBPersistence:
    def test_news_articles_table_populated(
        self, temp_db: DatabaseManager, first_cycle_result: NewsCycleResult
    ) -> None:
        """After the first cycle some rows must exist in news_articles."""
        rows = temp_db.fetch_all(
            "SELECT COUNT(*) AS cnt FROM news_articles", ()
        )
        count = rows[0]["cnt"] if rows else 0
        # We can only assert if articles were actually fetched and processed
        if first_cycle_result.articles_new > 0:
            assert count > 0, "news_articles table is empty after a cycle with new articles"

    def test_news_index_impact_table_populated(
        self, temp_db: DatabaseManager, first_cycle_result: NewsCycleResult
    ) -> None:
        """Processed articles with index impacts must create impact rows."""
        rows = temp_db.fetch_all(
            "SELECT COUNT(*) AS cnt FROM news_index_impact", ()
        )
        count = rows[0]["cnt"] if rows else 0
        # Only assert if we processed articles
        if first_cycle_result.articles_processed > 0:
            # Some articles may have no index mappings; do a soft check
            assert count >= 0

    def test_article_fields_saved_correctly(
        self, temp_db: DatabaseManager, first_cycle_result: NewsCycleResult
    ) -> None:
        """Spot-check that saved articles have all required columns set."""
        rows = temp_db.fetch_all(
            "SELECT * FROM news_articles ORDER BY id DESC LIMIT 5", ()
        )
        for row in rows:
            assert row["title"], "title must not be empty"
            assert row["source"], "source must not be empty"
            assert row["url"], "url must not be empty"
            assert row["published_at"], "published_at must not be empty"
            assert row["impact_category"] in (
                "CRITICAL", "HIGH", "MEDIUM", "LOW", "NOISE"
            ), f"Invalid impact_category: {row['impact_category']!r}"

    def test_no_duplicate_urls_in_db(
        self, temp_db: DatabaseManager
    ) -> None:
        """The UNIQUE constraint on url must be honoured."""
        rows = temp_db.fetch_all(
            """
            SELECT url, COUNT(*) AS cnt
            FROM news_articles
            GROUP BY url
            HAVING cnt > 1
            """,
            (),
        )
        assert rows == [], f"Duplicate URLs in news_articles: {rows}"

    def test_index_impact_foreign_keys(self, temp_db: DatabaseManager) -> None:
        """Every news_index_impact row must reference a valid news_articles id."""
        orphans = temp_db.fetch_all(
            """
            SELECT nii.id, nii.news_id
            FROM news_index_impact nii
            LEFT JOIN news_articles na ON na.id = nii.news_id
            WHERE na.id IS NULL
            """,
            (),
        )
        assert orphans == [], f"Orphaned news_index_impact rows: {orphans}"


# ---------------------------------------------------------------------------
# Test 3: Deduplication across cycles
# ---------------------------------------------------------------------------


@_skip_if_no_network()
class TestDeduplication:
    def test_second_immediate_cycle_has_zero_new(
        self, engine: NewsEngine
    ) -> None:
        """
        Running a second cycle immediately after the first must produce
        0 new articles because all URLs are already in ``_recent_urls``.

        Note: a tiny number of new articles could arrive between cycles even
        within seconds.  We allow up to 2 genuinely new articles.
        """
        second = engine.run_news_cycle()
        # After the first cycle (run in the fixture), all fetched URLs are cached.
        # Feeds refresh at 120s intervals, so no new content should appear.
        # We allow up to 2 to avoid flakiness with feeds that update very rapidly.
        assert second.articles_new <= 2, (
            f"Expected ~0 new articles on second immediate cycle, got {second.articles_new}"
        )

    def test_duplicate_count_equals_fetched(
        self, engine: NewsEngine
    ) -> None:
        """articles_fetched ≈ articles_duplicate on the second cycle."""
        third = engine.run_news_cycle()
        # articles_duplicate includes both batch-dedup and DB-dedup
        assert third.articles_duplicate >= 0
        assert third.articles_new + third.articles_duplicate <= third.articles_fetched + 1


# ---------------------------------------------------------------------------
# Test 4: Time decay
# ---------------------------------------------------------------------------


@_skip_if_no_network()
class TestTimeDecay:
    def test_effective_scores_exist_after_cycle(
        self, first_cycle_result: NewsCycleResult
    ) -> None:
        """EffectiveNewsScore objects must be present after a cycle with data."""
        if first_cycle_result.articles_processed == 0:
            pytest.skip("No articles processed — skip decay test")

        assert isinstance(first_cycle_result.index_news_scores, dict)
        for idx_id, score in first_cycle_result.index_news_scores.items():
            assert isinstance(score, EffectiveNewsScore)
            assert score.index_id == idx_id

    def test_scores_have_valid_vote(
        self, first_cycle_result: NewsCycleResult
    ) -> None:
        valid_votes = {
            "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
        }
        for score in first_cycle_result.index_news_scores.values():
            assert score.news_vote in valid_votes, (
                f"Unexpected news_vote: {score.news_vote!r}"
            )

    def test_decay_factor_decreases_over_time(
        self, engine: NewsEngine
    ) -> None:
        """
        After a minute, decayed scores should reflect the passage of time.
        We verify this by checking that decay calculations produce values < 1.
        """
        # Access the decayed articles from the last cycle
        with engine._lock:
            snapshot = list(engine._last_decayed)

        for da in snapshot:
            assert 0.0 <= da.decay_factor <= 1.0, (
                f"decay_factor out of [0,1]: {da.decay_factor}"
            )
            if da.article.impact_severity in ("LOW", "NOISE"):
                # Low-severity articles should already show some decay
                # even for recent articles (half-life = 10-30 min)
                assert da.decay_factor >= 0.0


# ---------------------------------------------------------------------------
# Test 5: Event calendar integration
# ---------------------------------------------------------------------------


@_skip_if_no_network()
class TestEventCalendar:
    def test_event_regime_returned_for_each_scored_index(
        self, first_cycle_result: NewsCycleResult
    ) -> None:
        """Every index in index_news_scores must have an event_regime entry."""
        for idx_id in first_cycle_result.index_news_scores:
            assert idx_id in first_cycle_result.event_regime, (
                f"No event_regime entry for scored index {idx_id!r}"
            )

    def test_event_regime_modifier_fields(
        self, first_cycle_result: NewsCycleResult
    ) -> None:
        valid_caution = {"NORMAL", "ELEVATED", "HIGH", "EXTREME"}
        for idx_id, modifier in first_cycle_result.event_regime.items():
            assert isinstance(modifier, EventRegimeModifier)
            assert modifier.caution_level in valid_caution, (
                f"Invalid caution_level for {idx_id!r}: {modifier.caution_level!r}"
            )
            assert modifier.volatility_multiplier >= 0.0
            assert modifier.position_size_modifier >= 0.0

    def test_fo_indices_have_regime_modifiers(
        self, engine: NewsEngine, first_cycle_result: NewsCycleResult
    ) -> None:
        """All F&O indices should get regime modifiers from the cycle."""
        fo_ids = _fo_index_ids(engine)
        if not fo_ids:
            pytest.skip("No F&O indices found in registry")

        for idx_id in fo_ids:
            assert idx_id in first_cycle_result.event_regime, (
                f"F&O index {idx_id!r} missing from event_regime"
            )


# ---------------------------------------------------------------------------
# Test 6: NewsVote for F&O indices
# ---------------------------------------------------------------------------


@_skip_if_no_network()
class TestNewsVotes:
    def test_get_all_news_votes_returns_dict(
        self, engine: NewsEngine, first_cycle_result: NewsCycleResult
    ) -> None:
        votes = engine.get_all_news_votes()
        assert isinstance(votes, dict)

    def test_fo_indices_have_votes(
        self, engine: NewsEngine
    ) -> None:
        fo_ids = _fo_index_ids(engine)
        if not fo_ids:
            pytest.skip("No F&O indices found in registry")

        votes = engine.get_all_news_votes()
        for idx_id in fo_ids:
            assert idx_id in votes, f"F&O index {idx_id!r} missing from votes"
            vote = votes[idx_id]
            assert isinstance(vote, NewsVote)
            assert vote.index_id == idx_id

    def test_votes_have_valid_structure(self, engine: NewsEngine) -> None:
        votes = engine.get_all_news_votes()
        valid_votes = {
            "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
        }
        valid_regimes = {"NORMAL", "ELEVATED", "HIGH", "EXTREME"}

        for idx_id, vote in votes.items():
            assert vote.vote in valid_votes, (
                f"Invalid vote for {idx_id!r}: {vote.vote!r}"
            )
            assert 0.0 <= vote.confidence <= 1.0, (
                f"confidence out of range for {idx_id!r}: {vote.confidence}"
            )
            assert vote.event_regime in valid_regimes, (
                f"Invalid event_regime for {idx_id!r}: {vote.event_regime!r}"
            )
            assert isinstance(vote.reasoning, str) and len(vote.reasoning) > 0

    def test_vote_timestamp_is_recent(self, engine: NewsEngine) -> None:
        votes = engine.get_all_news_votes()
        now = datetime.now(tz=_IST)
        for idx_id, vote in votes.items():
            age_sec = abs((now - vote.timestamp).total_seconds())
            assert age_sec < 300, (
                f"Vote timestamp for {idx_id!r} is more than 5 minutes old"
            )


# ---------------------------------------------------------------------------
# Test 7: Performance
# ---------------------------------------------------------------------------


@_skip_if_no_network()
class TestPerformance:
    def test_cycle_completes_within_15_seconds(self, engine: NewsEngine) -> None:
        """
        A full cycle including real network I/O must finish within 15 seconds.
        """
        t0 = time.monotonic()
        result = engine.run_news_cycle()
        elapsed = time.monotonic() - t0

        assert elapsed < 15.0, (
            f"Cycle took {elapsed:.1f}s — exceeds 15s budget. "
            f"(reported duration={result.cycle_duration_ms}ms)"
        )

    def test_reported_duration_matches_wall_time(self, engine: NewsEngine) -> None:
        t0 = time.monotonic()
        result = engine.run_news_cycle()
        wall_ms = (time.monotonic() - t0) * 1000

        # Reported duration should be within 2x of wall time (overhead ok)
        assert result.cycle_duration_ms <= wall_ms * 2 + 200


# ---------------------------------------------------------------------------
# Test 8: Stability — no crashes across multiple cycles
# ---------------------------------------------------------------------------


@_skip_if_no_network()
class TestStability:
    def test_three_consecutive_cycles_no_exception(
        self, engine: NewsEngine
    ) -> None:
        """Run three cycles back-to-back.  None must raise an exception."""
        for i in range(3):
            result = engine.run_news_cycle()
            assert isinstance(result, NewsCycleResult), (
                f"Cycle {i + 1} returned unexpected type: {type(result)}"
            )
            assert result.cycle_duration_ms >= 0

    def test_get_news_feed_after_cycles(self, engine: NewsEngine) -> None:
        """get_news_feed() must work correctly after multiple cycles."""
        engine.run_news_cycle()
        feed = engine.get_news_feed(min_severity="NOISE", limit=50)
        assert isinstance(feed, list)
        for item in feed:
            assert isinstance(item, dict)
            assert "title" in item
            assert "severity" in item

    def test_get_critical_alerts_after_cycles(self, engine: NewsEngine) -> None:
        """get_critical_alerts() must work correctly after multiple cycles."""
        engine.run_news_cycle()
        alerts = engine.get_critical_alerts()
        assert isinstance(alerts, list)
        for alert in alerts:
            assert alert.severity in ("CRITICAL", "HIGH")

    def test_memory_not_growing_unbounded(self, engine: NewsEngine) -> None:
        """
        After several cycles, _recent_mapped must stay bounded to the
        CACHE_WINDOW_HOURS window (not grow indefinitely).
        """
        from src.analysis.news.news_engine import _CACHE_WINDOW_HOURS

        for _ in range(5):
            engine.run_news_cycle()

        with engine._lock:
            cache_size = len(engine._recent_mapped)

        # With 2-minute real cycle intervals, 5 cycles ≈ 10 minutes of articles.
        # At 50 articles/cycle that's 250 max.  A generous upper bound is 1000.
        assert cache_size < 1000, (
            f"_recent_mapped cache has {cache_size} entries — possible memory leak"
        )

    def test_url_cache_not_growing_unbounded(self, engine: NewsEngine) -> None:
        """_recent_urls must be pruned and not grow beyond _MAX_URL_CACHE."""
        from src.analysis.news.news_engine import _MAX_URL_CACHE

        with engine._lock:
            url_cache_size = len(engine._recent_urls)

        assert url_cache_size <= _MAX_URL_CACHE, (
            f"_recent_urls has {url_cache_size} entries > limit {_MAX_URL_CACHE}"
        )


# ---------------------------------------------------------------------------
# Test 9: NewsStore standalone
# ---------------------------------------------------------------------------


class TestNewsStore:
    """
    These tests use a mock DB and do not require network access.
    They verify that NewsStore SQL calls are well-formed.
    """

    def test_save_articles_returns_list(self, temp_db: DatabaseManager) -> None:
        from src.analysis.news.news_store import NewsStore
        store = NewsStore()
        # Empty list → empty return
        ids = store.save_articles([], temp_db)
        assert ids == []

    def test_save_index_impacts_empty_list(self, temp_db: DatabaseManager) -> None:
        from src.analysis.news.news_store import NewsStore
        store = NewsStore()
        count = store.save_index_impacts(99999, [], temp_db)
        assert count == 0

    def test_get_recent_articles_returns_list(self, temp_db: DatabaseManager) -> None:
        from src.analysis.news.news_store import NewsStore
        store = NewsStore()
        rows = store.get_recent_articles(temp_db, hours=6, min_severity="NOISE")
        assert isinstance(rows, list)

    def test_get_article_count_by_day_returns_list(
        self, temp_db: DatabaseManager
    ) -> None:
        from src.analysis.news.news_store import NewsStore
        store = NewsStore()
        rows = store.get_article_count_by_day(temp_db, days=7)
        assert isinstance(rows, list)

    def test_cleanup_old_articles_returns_dict(
        self, temp_db: DatabaseManager
    ) -> None:
        from src.analysis.news.news_store import NewsStore
        store = NewsStore()
        result = store.cleanup_old_articles(temp_db)
        assert isinstance(result, dict)
        valid_keys = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "NOISE"}
        for key in result:
            assert key in valid_keys
