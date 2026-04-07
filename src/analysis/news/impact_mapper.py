"""
News-to-Index Impact Mapper — connects parsed news articles to specific indices.

Uses three mapping strategies (direct mention, company-to-index, keyword-to-index)
to determine which indices a news article affects, with what confidence, and
classifies the overall impact severity.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.analysis.news.article_parser import ParsedArticle
from src.analysis.news.sentiment_analyzer import SentimentResult
from src.data.index_registry import IndexRegistry, get_registry

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# Severity ordering for comparisons and sorting
_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NOISE": 4}
_SEVERITY_WEIGHTS = {"CRITICAL": 5, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NOISE": 0}

# Crisis keywords for CRITICAL global events
_CRISIS_KEYWORDS = {
    "war", "crash", "crisis", "meltdown", "collapse", "default",
    "recession", "panic", "contagion", "black swan",
}

# Maximum confidence cap
_MAX_CONFIDENCE = 0.95

# Confidence reduction when article maps to too many indices
_BROAD_NEWS_THRESHOLD = 10
_BROAD_NEWS_PENALTY = 0.2

# Article age thresholds
_OLD_ARTICLE_HOURS = 24
_STALE_ARTICLE_HOURS = 6


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class IndexImpact:
    """Impact of a news article on a specific index."""

    index_id: str
    relevance_score: float
    mapping_method: str  # DIRECT_MENTION / COMPANY_MAPPING / KEYWORD_MAPPING / MULTI_METHOD
    affected_by: list[str] = field(default_factory=list)
    confidence: float = 0.0

    # Directional impact (combining relevance with sentiment)
    sentiment_score: float = 0.0
    directional_impact: float = 0.0

    # Company-specific (if mapped via company)
    company_name: str | None = None
    estimated_index_weight: float | None = None


@dataclass
class MappedArticle:
    """A fully mapped article with index impacts and severity classification."""

    article: ParsedArticle
    sentiment: SentimentResult
    index_impacts: list[IndexImpact] = field(default_factory=list)
    impact_severity: str = "NOISE"
    primary_index: str | None = None
    is_actionable: bool = False


@dataclass
class MarketNewsSummary:
    """Aggregate view across all recent mapped news."""

    total_articles: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    by_event_type: dict[str, int] = field(default_factory=dict)

    overall_market_sentiment: float = 0.0
    overall_sentiment_label: str = "NEUTRAL"

    most_impacted_indices: list[tuple[str, float]] = field(default_factory=list)
    critical_alerts: list[str] = field(default_factory=list)

    sentiment_distribution: dict[str, int] = field(default_factory=dict)
    dominant_event_type: str = ""


# ---------------------------------------------------------------------------
# Estimated index weights for major companies (approximate, in %)
# ---------------------------------------------------------------------------

_COMPANY_INDEX_WEIGHTS: dict[str, dict[str, float]] = {
    "Reliance Industries": {"NIFTY50": 10.5, "SENSEX": 12.0},
    "TCS": {"NIFTY50": 4.0, "NIFTY_IT": 25.0, "SENSEX": 5.0},
    "HDFC Bank": {"NIFTY50": 8.5, "BANKNIFTY": 28.0, "SENSEX": 10.0},
    "Infosys": {"NIFTY50": 6.0, "NIFTY_IT": 22.0, "SENSEX": 7.0},
    "ICICI Bank": {"NIFTY50": 6.5, "BANKNIFTY": 22.0, "SENSEX": 7.5},
    "Hindustan Unilever": {"NIFTY50": 2.5, "SENSEX": 3.0},
    "ITC": {"NIFTY50": 3.5, "SENSEX": 4.0},
    "State Bank of India": {"NIFTY50": 3.0, "BANKNIFTY": 8.0, "SENSEX": 3.5},
    "Bharti Airtel": {"NIFTY50": 3.5, "SENSEX": 4.0},
    "Kotak Mahindra Bank": {"NIFTY50": 3.0, "BANKNIFTY": 10.0, "SENSEX": 3.5},
    "Larsen & Toubro": {"NIFTY50": 3.5, "SENSEX": 4.0},
    "Axis Bank": {"NIFTY50": 2.5, "BANKNIFTY": 8.0, "SENSEX": 3.0},
    "Bajaj Finance": {"NIFTY50": 2.0, "SENSEX": 2.5},
    "Maruti Suzuki": {"NIFTY50": 1.5, "NIFTY_AUTO": 15.0, "SENSEX": 2.0},
    "Sun Pharmaceutical": {"NIFTY50": 1.5, "NIFTY_PHARMA": 18.0, "SENSEX": 2.0},
    "Tata Steel": {"NIFTY50": 1.0, "NIFTY_METAL": 20.0, "SENSEX": 1.5},
    "Wipro": {"NIFTY50": 1.0, "NIFTY_IT": 10.0, "SENSEX": 1.5},
    "HCL Technologies": {"NIFTY50": 2.0, "NIFTY_IT": 15.0, "SENSEX": 2.5},
    "Tech Mahindra": {"NIFTY50": 1.0, "NIFTY_IT": 8.0, "SENSEX": 1.5},
    "Tata Motors": {"NIFTY50": 1.5, "NIFTY_AUTO": 12.0, "SENSEX": 2.0},
    "IndusInd Bank": {"NIFTY50": 1.0, "BANKNIFTY": 4.0},
    "Adani Enterprises": {"NIFTY50": 1.5, "SENSEX": 1.5},
}


# ---------------------------------------------------------------------------
# NewsImpactMapper
# ---------------------------------------------------------------------------


class NewsImpactMapper:
    """Maps parsed news articles to affected indices and classifies impact."""

    def __init__(
        self,
        news_mappings_path: Optional[Path] = None,
        companies_path: Optional[Path] = None,
        registry: Optional[IndexRegistry] = None,
    ) -> None:
        news_mappings_path = news_mappings_path or settings.news_mappings_path
        companies_path = companies_path or (settings.config_dir / "companies.json")

        self._news_mappings: dict = self._load_json(news_mappings_path)
        self._companies: list[dict] = self._load_json(companies_path)
        self._registry = registry

        # Build lookup structures
        self._display_to_id: dict[str, str] = {}
        self._id_to_display: dict[str, str] = {}
        self._build_display_id_lookup()

        # company name (lower) → company dict
        self._company_lookup: dict[str, dict] = {}
        self._build_company_lookup()

        # index_id → set of keywords that affect it (reverse lookup)
        self._index_keywords: dict[str, set[str]] = {}
        self._build_keyword_reverse_lookup()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_json(path: Path) -> list | dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load %s: %s", path, exc)
            return {}

    def _get_registry(self) -> IndexRegistry | None:
        if self._registry is not None:
            return self._registry
        try:
            self._registry = get_registry()
        except Exception:  # noqa: BLE001
            logger.debug("IndexRegistry not available")
        return self._registry

    def _build_display_id_lookup(self) -> None:
        """Map display names ↔ index IDs so we can translate mentioned_indices."""
        # Hard-coded common aliases (same as article_parser)
        _COMMON = {
            "NIFTY 50": "NIFTY50",
            "NIFTY BANK": "BANKNIFTY",
            "BSE SENSEX": "SENSEX",
            "NIFTY IT": "NIFTY_IT",
            "NIFTY PHARMA": "NIFTY_PHARMA",
            "NIFTY AUTO": "NIFTY_AUTO",
            "NIFTY METAL": "NIFTY_METAL",
            "NIFTY FMCG": "NIFTY_FMCG",
            "NIFTY REALTY": "NIFTY_REALTY",
            "NIFTY ENERGY": "NIFTY_ENERGY",
            "NIFTY MIDCAP": "NIFTY_MIDCAP100",
            "NIFTY SMALLCAP": "NIFTY_SMALLCAP100",
            "NIFTY FINANCIAL SERVICES": "FINNIFTY",
            "INDIA VIX": "INDIA_VIX",
        }
        for display, idx_id in _COMMON.items():
            self._display_to_id[display] = idx_id
            self._id_to_display[idx_id] = display

        registry = self._get_registry()
        if registry is None:
            return
        for idx in registry.get_all_indices():
            self._display_to_id[idx.display_name] = idx.id
            self._id_to_display[idx.id] = idx.display_name

    def _resolve_to_id(self, name: str) -> str | None:
        """Convert a display name or ID to a canonical index_id."""
        # Already an ID?
        if name in self._id_to_display:
            return name
        # Display name?
        if name in self._display_to_id:
            return self._display_to_id[name]
        # Try case-insensitive
        upper = name.upper()
        for display, idx_id in self._display_to_id.items():
            if display.upper() == upper:
                return idx_id
        for idx_id in self._id_to_display:
            if idx_id.upper() == upper:
                return idx_id
        return None

    def _build_company_lookup(self) -> None:
        """Build a fast company name/alias → company dict lookup."""
        for company in self._companies:
            name_lower = company["name"].lower()
            self._company_lookup[name_lower] = company
            for alias in company.get("aliases", []):
                self._company_lookup[alias.lower()] = company

    def _build_keyword_reverse_lookup(self) -> None:
        """Build index_id → set of keywords from news_mappings.json."""
        for category, mapping in self._news_mappings.items():
            if category.startswith("_"):
                continue
            keywords = mapping.get("keywords", [])
            affected = mapping.get("affected_indices", [])
            for idx_id in affected:
                if idx_id not in self._index_keywords:
                    self._index_keywords[idx_id] = set()
                self._index_keywords[idx_id].update(keywords)

    # ── Strategy 1: Direct index mention ──────────────────────────────────────

    def _map_direct_mentions(
        self, article: ParsedArticle
    ) -> dict[str, dict]:
        """Map indices directly mentioned in the article. Returns {index_id: info}."""
        results: dict[str, dict] = {}
        for display_name in article.mentioned_indices:
            idx_id = self._resolve_to_id(display_name)
            if idx_id is None:
                logger.debug("Could not resolve index: %s", display_name)
                continue
            results[idx_id] = {
                "confidence": 0.9,
                "affected_by": [display_name],
                "method": "DIRECT_MENTION",
            }
        return results

    # ── Strategy 2: Company-to-index mapping ──────────────────────────────────

    def _map_via_companies(
        self, article: ParsedArticle
    ) -> dict[str, dict]:
        """Map indices via mentioned companies. Returns {index_id: info}."""
        results: dict[str, dict] = {}
        for company_name in article.mentioned_companies:
            company = self._company_lookup.get(company_name.lower())
            if company is None:
                logger.debug("Company not in lookup: %s", company_name)
                continue

            indices = company.get("indices", [])
            if not indices:
                continue

            # Get index weight info for this company
            weights = _COMPANY_INDEX_WEIGHTS.get(company["name"], {})

            for idx_id in indices:
                weight = weights.get(idx_id)
                # Higher confidence for heavier-weight companies
                if weight and weight >= 5.0:
                    conf = 0.8
                else:
                    conf = 0.75

                if idx_id in results:
                    # Keep higher confidence and accumulate affected_by
                    existing = results[idx_id]
                    if conf > existing["confidence"]:
                        existing["confidence"] = conf
                    if company["name"] not in existing["affected_by"]:
                        existing["affected_by"].append(company["name"])
                    # Accumulate weight
                    if weight and existing.get("estimated_weight") is not None:
                        existing["estimated_weight"] += weight
                    elif weight:
                        existing["estimated_weight"] = weight
                else:
                    results[idx_id] = {
                        "confidence": conf,
                        "affected_by": [company["name"]],
                        "method": "COMPANY_MAPPING",
                        "company_name": company["name"],
                        "estimated_weight": weight,
                    }
        return results

    # ── Strategy 3: Keyword-to-index mapping ──────────────────────────────────

    def _map_via_keywords(
        self, article: ParsedArticle
    ) -> dict[str, dict]:
        """Map indices via keyword matches in article text. Returns {index_id: info}."""
        results: dict[str, dict] = {}
        text_lower = article.clean_text.lower()

        for category, mapping in self._news_mappings.items():
            if category.startswith("_"):
                continue

            keywords = mapping.get("keywords", [])
            affected = mapping.get("affected_indices", [])
            category_weight = mapping.get("weight", 0.7)

            matched_keywords = [kw for kw in keywords if kw in text_lower]
            if not matched_keywords:
                continue

            # Base confidence scaled by category weight and number of matches
            # More keyword matches → slightly higher confidence
            match_boost = min(len(matched_keywords) * 0.02, 0.1)
            base_conf = 0.5 + match_boost

            for idx_id in affected:
                conf = min(base_conf * category_weight, 0.6)
                trigger = f"{category} keyword"

                if idx_id in results:
                    existing = results[idx_id]
                    if conf > existing["confidence"]:
                        existing["confidence"] = conf
                    existing["affected_by"].extend(
                        kw for kw in matched_keywords
                        if kw not in existing["affected_by"]
                    )
                else:
                    results[idx_id] = {
                        "confidence": conf,
                        "affected_by": matched_keywords[:3] + [trigger],
                        "method": "KEYWORD_MAPPING",
                    }
        return results

    # ── Core mapping method ───────────────────────────────────────────────────

    def map_to_indices(
        self, article: ParsedArticle, sentiment: SentimentResult
    ) -> list[IndexImpact]:
        """
        Determine which indices an article affects and how.

        Applies three strategies in priority order, combines overlapping
        results, and computes directional impact.
        """
        direct = self._map_direct_mentions(article)
        company = self._map_via_companies(article)
        keyword = self._map_via_keywords(article)

        # Merge: collect all index_ids across strategies
        all_ids = set(direct) | set(company) | set(keyword)

        impacts: list[IndexImpact] = []
        for idx_id in all_ids:
            sources = []
            if idx_id in direct:
                sources.append(direct[idx_id])
            if idx_id in company:
                sources.append(company[idx_id])
            if idx_id in keyword:
                sources.append(keyword[idx_id])

            # Take highest confidence across strategies
            best = max(sources, key=lambda s: s["confidence"])
            confidence = best["confidence"]

            # Multi-method boost: +0.1 if 2+ strategies agree
            method_count = len(sources)
            if method_count >= 2:
                confidence = min(confidence + 0.1, _MAX_CONFIDENCE)
                method = "MULTI_METHOD"
            else:
                method = best["method"]

            # Collect all affected_by triggers
            affected_by: list[str] = []
            for src in sources:
                for trigger in src.get("affected_by", []):
                    if trigger not in affected_by:
                        affected_by.append(trigger)

            # Company info (prefer company strategy if present)
            company_name = None
            estimated_weight = None
            if idx_id in company:
                company_info = company[idx_id]
                company_name = company_info.get("company_name")
                estimated_weight = company_info.get("estimated_weight")

            # Relevance = confidence (they're conceptually the same here)
            relevance = confidence

            # Directional impact = relevance * sentiment
            directional = relevance * sentiment.adjusted_score

            impacts.append(IndexImpact(
                index_id=idx_id,
                relevance_score=relevance,
                mapping_method=method,
                affected_by=affected_by,
                confidence=confidence,
                sentiment_score=sentiment.adjusted_score,
                directional_impact=round(directional, 4),
                company_name=company_name,
                estimated_index_weight=estimated_weight,
            ))

        # Broad news penalty: if too many indices, reduce per-index confidence
        if len(impacts) > _BROAD_NEWS_THRESHOLD:
            for impact in impacts:
                impact.confidence = max(0.1, impact.confidence - _BROAD_NEWS_PENALTY)
                impact.relevance_score = impact.confidence
                impact.directional_impact = round(
                    impact.relevance_score * impact.sentiment_score, 4
                )

        # Sort by confidence descending
        impacts.sort(key=lambda x: x.confidence, reverse=True)
        return impacts

    # ── Severity classification ───────────────────────────────────────────────

    def classify_impact_severity(
        self,
        article: ParsedArticle,
        sentiment: SentimentResult,
        index_impacts: list[IndexImpact],
    ) -> str:
        """Classify the article's impact into one of five severity buckets."""
        event = article.event_type
        cred = article.source_credibility
        sent = abs(sentiment.adjusted_score)
        text_lower = article.clean_text.lower()

        # Very old articles → auto LOW
        now = datetime.now(tz=_IST)
        pub = article.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=_IST)
        age_hours = (now - pub).total_seconds() / 3600
        if age_hours > _OLD_ARTICLE_HOURS:
            return "LOW"

        # No index mapping → NOISE
        if not index_impacts:
            return "NOISE"

        # Extremely low sentiment → NOISE
        if sent < 0.05:
            return "NOISE"

        # Per-index combined weight: sum weights of all mentioned companies
        # in the same index. Only meaningful when multiple companies contribute.
        max_per_index_weight = 0.0
        multi_company_count = 0
        index_weight_map: dict[str, float] = {}
        for imp in index_impacts:
            if imp.estimated_index_weight:
                w = index_weight_map.get(imp.index_id, 0.0)
                index_weight_map[imp.index_id] = w + imp.estimated_index_weight
        if index_weight_map:
            max_per_index_weight = max(index_weight_map.values())
        # Count how many distinct companies are involved
        company_names = {
            imp.company_name for imp in index_impacts if imp.company_name
        }
        multi_company_count = len(company_names)

        has_crisis = any(kw in text_lower for kw in _CRISIS_KEYWORDS)

        # ── CRITICAL ──────────────────────────────────────────────────────
        if event == "POLICY" and cred > 0.7 and sent > 0.5:
            return "CRITICAL"
        if event == "GLOBAL" and has_crisis and sent > 0.7:
            return "CRITICAL"
        if multi_company_count >= 2 and max_per_index_weight > 10.0:
            return "CRITICAL"

        # ── HIGH ──────────────────────────────────────────────────────────
        if event in ("EARNINGS", "POLICY", "REGULATORY") and sent > 0.4:
            return "HIGH"
        # Major company (weight > 5%)
        if any(
            (imp.estimated_index_weight or 0) > 5.0 for imp in index_impacts
        ) and sent > 0.3:
            return "HIGH"
        # FII/DII with strong sentiment
        if event == "FII_DII" and sent > 0.3:
            return "HIGH"
        if event == "MACRO" and sent > 0.5:
            return "HIGH"

        # ── LOW (check before MEDIUM so low-cred/stale don't get MEDIUM) ─
        if sent < 0.2:
            return "LOW"
        if cred < 0.5:
            return "LOW"
        # Stale during market hours
        if age_hours > _STALE_ARTICLE_HOURS:
            return "LOW"

        # ── MEDIUM ────────────────────────────────────────────────────────
        if event in ("CORPORATE", "SECTOR") and sent > 0.2:
            return "MEDIUM"
        if event in ("EARNINGS", "POLICY", "REGULATORY", "MACRO", "GLOBAL"):
            return "MEDIUM"
        if any(imp.company_name for imp in index_impacts) and sent > 0.3:
            return "MEDIUM"
        # General market commentary with moderate sentiment
        if sent > 0.2:
            return "MEDIUM"

        return "LOW"

    # ── Combined pipeline ─────────────────────────────────────────────────────

    def map_and_classify(
        self, article: ParsedArticle, sentiment: SentimentResult
    ) -> MappedArticle:
        """Map article to indices and classify severity in one call."""
        impacts = self.map_to_indices(article, sentiment)
        severity = self.classify_impact_severity(article, sentiment, impacts)

        primary = impacts[0].index_id if impacts else None
        is_actionable = severity in ("CRITICAL", "HIGH") and len(impacts) > 0

        return MappedArticle(
            article=article,
            sentiment=sentiment,
            index_impacts=impacts,
            impact_severity=severity,
            primary_index=primary,
            is_actionable=is_actionable,
        )

    # ── Index-specific feed ───────────────────────────────────────────────────

    @staticmethod
    def get_index_news_feed(
        index_id: str,
        mapped_articles: list[MappedArticle],
        min_severity: str = "LOW",
    ) -> list[MappedArticle]:
        """
        Filter mapped articles for a specific index.

        Returns articles sorted by severity (CRITICAL first), then by
        absolute directional impact descending.
        """
        min_order = _SEVERITY_ORDER.get(min_severity, 3)

        filtered: list[MappedArticle] = []
        for ma in mapped_articles:
            order = _SEVERITY_ORDER.get(ma.impact_severity, 4)
            if order > min_order:
                continue
            if any(imp.index_id == index_id for imp in ma.index_impacts):
                filtered.append(ma)

        def sort_key(ma: MappedArticle) -> tuple[int, float]:
            sev = _SEVERITY_ORDER.get(ma.impact_severity, 4)
            # Max absolute directional impact for this index
            max_dir = max(
                (abs(imp.directional_impact) for imp in ma.index_impacts
                 if imp.index_id == index_id),
                default=0.0,
            )
            return (sev, -max_dir)

        filtered.sort(key=sort_key)
        return filtered

    # ── Market-wide summary ───────────────────────────────────────────────────

    @staticmethod
    def get_market_news_summary(
        mapped_articles: list[MappedArticle],
    ) -> MarketNewsSummary:
        """Aggregate view across all recent mapped news."""
        if not mapped_articles:
            return MarketNewsSummary()

        by_severity: dict[str, int] = {}
        by_event_type: dict[str, int] = {}
        bullish = 0
        bearish = 0
        neutral = 0

        weighted_sentiment_sum = 0.0
        weight_sum = 0.0
        index_impact_totals: dict[str, float] = {}
        critical_alerts: list[str] = []

        for ma in mapped_articles:
            # Severity counts
            by_severity[ma.impact_severity] = by_severity.get(ma.impact_severity, 0) + 1

            # Event type counts
            evt = ma.article.event_type or "UNKNOWN"
            by_event_type[evt] = by_event_type.get(evt, 0) + 1

            # Sentiment distribution
            adj = ma.sentiment.adjusted_score
            if adj > 0.05:
                bullish += 1
            elif adj < -0.05:
                bearish += 1
            else:
                neutral += 1

            # Weighted sentiment: severity_weight * credibility
            sev_w = _SEVERITY_WEIGHTS.get(ma.impact_severity, 0)
            cred = ma.article.source_credibility
            w = sev_w * cred
            weighted_sentiment_sum += adj * w
            weight_sum += w

            # Index impact accumulation
            for imp in ma.index_impacts:
                idx = imp.index_id
                index_impact_totals[idx] = (
                    index_impact_totals.get(idx, 0.0) + abs(imp.directional_impact)
                )

            if ma.impact_severity == "CRITICAL":
                critical_alerts.append(ma.article.title)

        overall = weighted_sentiment_sum / weight_sum if weight_sum > 0 else 0.0
        if overall > 0.1:
            label = "BULLISH"
        elif overall < -0.1:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        # Top 5 most impacted indices
        top_indices = sorted(
            index_impact_totals.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Dominant event type
        dominant = max(by_event_type, key=by_event_type.get) if by_event_type else ""

        return MarketNewsSummary(
            total_articles=len(mapped_articles),
            by_severity=by_severity,
            by_event_type=by_event_type,
            overall_market_sentiment=round(overall, 4),
            overall_sentiment_label=label,
            most_impacted_indices=top_indices,
            critical_alerts=critical_alerts,
            sentiment_distribution={
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
            },
            dominant_event_type=dominant,
        )
