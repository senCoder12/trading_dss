"""
RSS feed fetcher with rate limiting and retry logic.

Fetches articles from all configured RSS feed sources, respecting
per-feed refresh intervals and a global rate limit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import feedparser
from curl_cffi import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# feedparser default UA is sometimes blocked by Indian news sites
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RawArticle:
    """A single article as fetched from an RSS feed, before any cleaning."""

    title: str
    summary: str | None
    url: str
    source: str
    source_credibility: float
    category: str
    published_at: datetime
    fetched_at: datetime
    raw_content: str | None = None
    author: str | None = None
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RSSFetcher
# ---------------------------------------------------------------------------


class RSSFetcher:
    """
    Fetches RSS feeds sequentially with per-feed refresh gating and
    a global rate limiter (max *rate_limit* requests per minute).

    Parameters
    ----------
    feeds_path:
        Path to ``rss_feeds.json``.  Defaults to the project config.
    rate_limit:
        Maximum number of HTTP requests per minute across all feeds.
    """

    def __init__(
        self,
        feeds_path: Optional[Path] = None,
        rate_limit: int = 10,
    ) -> None:
        import json

        self._feeds_path = feeds_path or settings.rss_feeds_path
        self._feeds: list[dict] = json.loads(
            self._feeds_path.read_text(encoding="utf-8")
        )

        self._session = requests.Session(impersonate="chrome131")

        # Rate limiter state
        self._rate_limit = rate_limit
        self._request_times: list[float] = []

        # Per-feed last-fetch tracking  {feed_id: epoch}
        self._last_fetch: dict[str, float] = {}

        # Statistics
        self._stats: dict = {
            "total_fetches": 0,
            "successful": 0,
            "failed": 0,
            "articles_fetched": 0,
            "last_fetch_time": None,
            "feeds_status": {},
        }

    # ── Rate limiting ────────────────────────────────────────────────────────

    def _wait_for_rate_limit(self) -> None:
        """Block until we are within the per-minute request budget."""
        now = time.monotonic()
        # Purge timestamps older than 60 s
        self._request_times = [
            t for t in self._request_times if now - t < 60
        ]
        if len(self._request_times) >= self._rate_limit:
            sleep_for = 60 - (now - self._request_times[0])
            if sleep_for > 0:
                logger.debug("Rate limit reached, sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
        self._request_times.append(time.monotonic())

    # ── Refresh gating ───────────────────────────────────────────────────────

    def _should_fetch(self, feed_cfg: dict) -> bool:
        """Return True if enough time has passed since the last fetch."""
        feed_id = feed_cfg["id"]
        interval = feed_cfg.get("refresh_interval_seconds", 120)
        last = self._last_fetch.get(feed_id)
        if last is None:
            return True
        return (time.time() - last) >= interval

    # ── Single feed fetch ────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5), reraise=True)
    def _parse_feed(self, url: str) -> feedparser.FeedParserDict:
        """Parse an RSS URL with retry. Uses curl_cffi for fetching."""
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            
            # If the response is HTML, it's likely a redirect or block page
            if resp.text.strip().startswith("<!DOCTYPE html") or "<html" in resp.text[:200].lower():
                 raise RuntimeError("Received HTML instead of XML (likely a block or landing page)")

            result = feedparser.parse(resp.content)
            
            if result.bozo and not result.entries:
                raise RuntimeError(
                    f"Feed parsing error: {result.bozo_exception}"
                )
            return result
        except Exception as exc:
             # curl_cffi errors are usually curl_cffi.CurlError or subclassed under requests.errors
             raise RuntimeError(f"HTTP fetch failed: {exc}") from exc

    def fetch_feed(self, feed_config: dict) -> list[RawArticle]:
        """
        Fetch and parse a single RSS feed into :class:`RawArticle` objects.

        Parameters
        ----------
        feed_config:
            One element from ``rss_feeds.json``.

        Returns
        -------
        list[RawArticle]
        """
        url = feed_config.get("url", "")
        source = feed_config.get("source_name", "Unknown")
        credibility = float(feed_config.get("credibility_score", 0.7))
        category = feed_config.get("category", "general")
        now = datetime.now(tz=_IST)

        self._wait_for_rate_limit()
        start = time.monotonic()

        parsed = self._parse_feed(url)

        articles: list[RawArticle] = []
        for entry in parsed.entries:
            published = self._parse_pubdate(entry)
            articles.append(
                RawArticle(
                    title=entry.get("title", "").strip(),
                    summary=entry.get("summary") or entry.get("description"),
                    url=entry.get("link", ""),
                    source=source,
                    source_credibility=credibility,
                    category=category,
                    published_at=published,
                    fetched_at=now,
                    raw_content=self._extract_content(entry),
                    author=entry.get("author"),
                    tags=[
                        t.get("term", "") for t in entry.get("tags", []) if t.get("term")
                    ],
                )
            )

        elapsed = time.monotonic() - start
        logger.info(
            "Fetched %d articles from %s (took %.1fs)", len(articles), source, elapsed
        )
        return articles

    # ── Fetch all feeds ──────────────────────────────────────────────────────

    def fetch_all_feeds(self) -> list[RawArticle]:
        """
        Iterate through all active feeds, respecting refresh intervals.

        Returns combined list sorted by ``published_at`` descending.
        """
        active_feeds = [f for f in self._feeds if f.get("is_active", True)]
        all_articles: list[RawArticle] = []

        feeds_attempted = 0
        feeds_successful = 0
        feeds_failed = 0

        for feed_cfg in active_feeds:
            if not self._should_fetch(feed_cfg):
                logger.debug("Skipping %s (refresh interval not reached)", feed_cfg["id"])
                continue

            feeds_attempted += 1
            try:
                articles = self.fetch_feed(feed_cfg)
                self._last_fetch[feed_cfg["id"]] = time.time()
                all_articles.extend(articles)
                feeds_successful += 1
                self._stats["feeds_status"][feed_cfg["id"]] = time.time()
            except Exception as exc:  # noqa: BLE001
                feeds_failed += 1
                logger.error("Failed to fetch %s: %s", feed_cfg["id"], exc)

        # Update cumulative stats
        self._stats["total_fetches"] += feeds_attempted
        self._stats["successful"] += feeds_successful
        self._stats["failed"] += feeds_failed
        self._stats["articles_fetched"] += len(all_articles)
        self._stats["last_fetch_time"] = datetime.now(tz=_IST).isoformat()

        all_articles.sort(key=lambda a: a.published_at, reverse=True)

        logger.info(
            "Fetch complete: %d attempted, %d succeeded, %d failed, %d articles",
            feeds_attempted,
            feeds_successful,
            feeds_failed,
            len(all_articles),
        )
        return all_articles

    # ── Fetch single source ──────────────────────────────────────────────────

    def fetch_single_source(self, source_name: str) -> list[RawArticle]:
        """Fetch from one source only (matched by ``source_name``)."""
        matching = [
            f for f in self._feeds
            if f.get("source_name", "").lower() == source_name.lower()
            and f.get("is_active", True)
        ]
        if not matching:
            logger.warning("No active feed found for source %r", source_name)
            return []

        all_articles: list[RawArticle] = []
        for feed_cfg in matching:
            try:
                articles = self.fetch_feed(feed_cfg)
                self._last_fetch[feed_cfg["id"]] = time.time()
                all_articles.extend(articles)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to fetch %s: %s", feed_cfg["id"], exc)

        all_articles.sort(key=lambda a: a.published_at, reverse=True)
        return all_articles

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_fetch_stats(self) -> dict:
        """Return cumulative fetch statistics."""
        return dict(self._stats)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_pubdate(entry: dict) -> datetime:
        """Convert RSS published_parsed to IST datetime."""
        parsed_time = entry.get("published_parsed")
        if parsed_time:
            from calendar import timegm

            utc_ts = timegm(parsed_time)
            return datetime.fromtimestamp(utc_ts, tz=_IST)
        # Fallback: use current time
        return datetime.now(tz=_IST)

    @staticmethod
    def _extract_content(entry: dict) -> str | None:
        """Pull content:encoded or first content block from feed entry."""
        content_list = entry.get("content")
        if content_list and isinstance(content_list, list):
            return content_list[0].get("value")
        return None
