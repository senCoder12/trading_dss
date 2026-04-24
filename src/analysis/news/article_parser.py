"""
Article parser — cleans and enriches raw RSS articles.

Strips HTML, extracts mentioned companies / sectors / indices, classifies
event types, and determines market-hours relevance.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.analysis.news.language_handler import LanguageHandler
from src.analysis.news.rss_fetcher import RawArticle
from src.data.index_registry import IndexRegistry, get_registry

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

_MAX_TEXT_LENGTH = 2000

# ── Sector keywords ──────────────────────────────────────────────────────────

_SECTOR_KEYWORDS: list[str] = [
    "banking", "finance", "it", "technology", "pharma", "healthcare",
    "auto", "automobile", "energy", "oil", "gas", "metal", "mining",
    "fmcg", "consumer", "real estate", "realty", "infrastructure",
    "telecom", "media", "psu", "government", "defence", "chemical",
    "textile",
]

# ── Event-type classification rules ─────────────────────────────────────────

_EVENT_RULES: list[tuple[str, list[str]]] = [
    ("EARNINGS", [
        "earnings", "quarterly results", "quarterly result",
        "profit", "revenue", "q1", "q2", "q3", "q4",
        "net income", "results beat", "results miss",
    ]),
    ("POLICY", [
        "rbi", "monetary policy", "rate cut", "rate hike",
        "repo rate", "fiscal", "interest rate", "policy rate",
    ]),
    ("MACRO", [
        "gdp", "inflation", "cpi", "iip", "trade deficit",
        "current account", "pmi", "industrial production",
    ]),
    ("GLOBAL", [
        "fed", "us market", "global", "china", "crude oil",
        "geopolitical", "tariff", "war", "us fed", "wall street",
        "dow jones", "nasdaq", "s&p 500",
    ]),
    ("CORPORATE", [
        "merger", "acquisition", "deal", "stake", "buyback",
        "dividend", "split", "takeover", "delisting",
    ]),
    ("REGULATORY", [
        "sebi", "regulation", "compliance", "ban",
        "investigation", "penalty", "fine", "circular",
    ]),
    ("MARKET_MOVE", [
        "rally", "crash", "surge", "plunge", "record high",
        "all time high", "all-time high", "all time low",
        "all-time low", "circuit breaker",
    ]),
    ("FII_DII", [
        "fii", "dii", "foreign investor", "institutional",
        "foreign portfolio", "fpi",
    ]),
    ("SECTOR", [
        "sector rally", "sector decline", "sectoral",
    ]),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ParsedArticle:
    """A cleaned and enriched article ready for deduplication and storage."""

    title: str
    clean_text: str
    url: str
    source: str
    source_credibility: float
    category: str
    published_at: datetime
    fetched_at: datetime

    # Extracted metadata
    mentioned_companies: list[str] = field(default_factory=list)
    mentioned_sectors: list[str] = field(default_factory=list)
    mentioned_indices: list[str] = field(default_factory=list)
    event_type: str | None = None
    is_market_hours_relevant: bool = False
    word_count: int = 0

    # Language metadata (populated by LanguageHandler)
    original_language: str = "en"
    was_translated: bool = False
    translation_confidence: float = 1.0
    is_hinglish: bool = False


# ---------------------------------------------------------------------------
# ArticleParser
# ---------------------------------------------------------------------------


class ArticleParser:
    """
    Cleans raw articles, extracts entities, and classifies events.

    Parameters
    ----------
    companies_path:
        Path to ``companies.json``.  Defaults to the project config dir.
    registry:
        An :class:`IndexRegistry` instance for index-name matching.
        Defaults to the process singleton.
    """

    def __init__(
        self,
        companies_path: Optional[Path] = None,
        registry: Optional[IndexRegistry] = None,
    ) -> None:
        companies_path = companies_path or (
            Path(settings.config_dir) / "companies.json"
        )
        self._companies: list[dict] = self._load_json(companies_path)
        self._registry = registry
        self.language_handler = LanguageHandler()

        # Pre-build index lookup: {lowercase_token: display_name}
        self._index_lookup: dict[str, str] = {}
        self._build_index_lookup()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_json(path: Path) -> list | dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load %s: %s", path, exc)
            return []

    def _get_registry(self) -> IndexRegistry | None:
        """Lazy-load registry to avoid import-time side effects."""
        if self._registry is not None:
            return self._registry
        try:
            self._registry = get_registry()
        except Exception:  # noqa: BLE001
            logger.debug("IndexRegistry not available, index extraction limited")
        return self._registry

    def _build_index_lookup(self) -> None:
        """Build a lowercase-token → display_name map from the registry."""
        # Always seed with common aliases (covers word-order variants like
        # "bank nifty" vs "nifty bank" that display_name alone may miss)
        _COMMON_ALIASES: list[tuple[str, str]] = [
            ("nifty 50", "NIFTY 50"), ("nifty50", "NIFTY 50"),
            ("nifty", "NIFTY 50"), ("sensex", "BSE SENSEX"),
            ("bank nifty", "NIFTY BANK"), ("banknifty", "NIFTY BANK"),
            ("nifty bank", "NIFTY BANK"), ("nifty it", "NIFTY IT"),
            ("nifty pharma", "NIFTY PHARMA"), ("nifty auto", "NIFTY AUTO"),
            ("nifty metal", "NIFTY METAL"), ("nifty fmcg", "NIFTY FMCG"),
            ("nifty realty", "NIFTY REALTY"), ("nifty energy", "NIFTY ENERGY"),
            ("nifty midcap", "NIFTY MIDCAP"), ("nifty smallcap", "NIFTY SMALLCAP"),
            ("finnifty", "NIFTY FINANCIAL SERVICES"),
            ("nifty fin service", "NIFTY FINANCIAL SERVICES"),
        ]
        for token, name in _COMMON_ALIASES:
            self._index_lookup[token] = name

        registry = self._get_registry()
        if registry is None:
            return

        for idx in registry.get_all_indices():
            display = idx.display_name
            # Add the display name itself
            self._index_lookup[display.lower()] = display
            # Add the NSE symbol if different
            if idx.nse_symbol:
                self._index_lookup[idx.nse_symbol.lower()] = display
            # Add the option symbol
            if idx.option_symbol:
                self._index_lookup[idx.option_symbol.lower()] = display

    # ── Public interface ──────────────────────────────────────────────────────

    def parse_article(self, raw: RawArticle) -> ParsedArticle | None:
        """
        Clean a :class:`RawArticle` and extract metadata.

        Returns ``None`` if the article has no usable text.
        """
        # Language detection + translation on raw text (before HTML stripping
        # so Devanagari / other non-ASCII characters are preserved for detection).
        raw_combined = (raw.title or "") + " " + (raw.summary or "")
        processed = self.language_handler.process(raw_combined)

        # Clean the (possibly translated) English text for downstream analysis
        clean_english = self._clean_text(processed.english_text)
        content = self._clean_text(raw.raw_content or "")

        parts = [p for p in (clean_english, content) if p]
        if not parts:
            return None
        clean = " ".join(parts)[:_MAX_TEXT_LENGTH]

        # Preserve original (untranslated) title for display purposes
        title = self._clean_text(raw.title or "")

        event_type = self._classify_event_type(clean)

        return ParsedArticle(
            title=title,
            clean_text=clean,
            url=raw.url,
            source=raw.source,
            source_credibility=raw.source_credibility,
            category=raw.category,
            published_at=raw.published_at,
            fetched_at=raw.fetched_at,
            mentioned_companies=self._extract_companies(clean),
            mentioned_sectors=self._extract_sectors(clean),
            mentioned_indices=self._extract_indices(clean),
            event_type=event_type,
            is_market_hours_relevant=self._is_market_hours_relevant(
                raw.published_at, event_type
            ),
            word_count=len(clean.split()),
            original_language=processed.original_lang,
            was_translated=processed.was_translated,
            translation_confidence=processed.translation_confidence,
            is_hinglish=processed.is_hinglish,
        )

    # ── Text cleaning ────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip HTML, URLs, and normalise whitespace."""
        if not text:
            return ""
        # Strip HTML
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        # Decode residual HTML entities (BS4 handles most, belt-and-suspenders)
        import html as html_mod

        text = html_mod.unescape(text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove special chars but keep basic punctuation
        text = re.sub(r"[^\w\s.,;:!?'\"\-%&><]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text[:_MAX_TEXT_LENGTH]

    # ── Entity extraction ────────────────────────────────────────────────────

    def _extract_companies(self, text: str) -> list[str]:
        """Return company names found in *text* (case-insensitive)."""
        lower = text.lower()
        found: list[str] = []
        for company in self._companies:
            name = company["name"]
            aliases = company.get("aliases", [])
            for token in [name.lower()] + [a.lower() for a in aliases]:
                if token in lower:
                    found.append(name)
                    break
        return found

    def _extract_sectors(self, text: str) -> list[str]:
        """Return sector keywords found in *text*."""
        lower = text.lower()
        return [s for s in _SECTOR_KEYWORDS if s in lower]

    def _extract_indices(self, text: str) -> list[str]:
        """Return index display names found in *text*."""
        lower = text.lower()
        found: set[str] = set()
        # Sort keys longest-first so "nifty bank" matches before "nifty"
        for token in sorted(self._index_lookup, key=len, reverse=True):
            if token in lower:
                found.add(self._index_lookup[token])
        return sorted(found)

    # ── Event classification ─────────────────────────────────────────────────

    @staticmethod
    def _classify_event_type(text: str) -> str | None:
        """Classify the article into a predefined event type."""
        lower = text.lower()
        for event_type, keywords in _EVENT_RULES:
            for kw in keywords:
                if kw in lower:
                    return event_type
        return None

    # ── Market-hours relevance ───────────────────────────────────────────────

    @staticmethod
    def _is_market_hours_relevant(
        published_at: datetime, event_type: str | None
    ) -> bool:
        """
        Determine if the article is relevant to the current or next session.

        Rules:
        - During market hours (09:15–15:30 IST weekdays) → True
        - After close but POLICY/GLOBAL/MACRO event → True (next session)
        - Weekend but GLOBAL/MACRO event → True (affects Monday)
        - Otherwise → False
        """
        ist_time = published_at.astimezone(_IST)
        weekday = ist_time.weekday()  # 0=Mon … 6=Sun
        hour_min = ist_time.hour * 60 + ist_time.minute
        market_open = 9 * 60 + 15   # 09:15
        market_close = 15 * 60 + 30  # 15:30

        is_weekday = weekday < 5
        during_market = is_weekday and market_open <= hour_min <= market_close

        if during_market:
            return True

        next_session_events = {"POLICY", "GLOBAL", "MACRO"}
        if event_type in next_session_events:
            return True

        return False
