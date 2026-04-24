"""
Sentiment Analysis Engine for financial news articles.

Uses VADER as a base sentiment scorer, boosted with domain-specific financial
keywords and weighted by source credibility.  Includes a calibration component
that learns from actual market outcomes over time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import settings
from src.analysis.news.article_parser import ParsedArticle
from src.analysis.news.hinglish_processor import HinglishProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SentimentResult:
    """Output of a single article sentiment analysis."""

    raw_vader_score: float
    keyword_boost: float
    source_weight: float
    adjusted_score: float

    sentiment_label: str
    confidence: float

    bullish_keywords_found: list[str] = field(default_factory=list)
    bearish_keywords_found: list[str] = field(default_factory=list)
    uncertainty_keywords_found: list[str] = field(default_factory=list)
    keyword_count: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class CalibrationStats:
    """Summary statistics from the sentiment calibrator."""

    sample_count: int
    avg_bullish_accuracy: float
    avg_bearish_accuracy: float
    sentiment_bias: float
    suggested_offset: float


@dataclass
class _Outcome:
    """A single recorded prediction-vs-reality pair."""

    sentiment: float
    market_move_pct: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LABEL_THRESHOLDS: list[tuple[str, float, float]] = [
    ("VERY_BULLISH", 0.6, float("inf")),
    ("BULLISH", 0.2, 0.6),
    ("SLIGHTLY_BULLISH", 0.05, 0.2),
    ("NEUTRAL", -0.05, 0.05),
    ("SLIGHTLY_BEARISH", -0.2, -0.05),
    ("BEARISH", -0.6, -0.2),
    ("VERY_BEARISH", float("-inf"), -0.6),
]


def _score_to_label(score: float) -> str:
    for label, lo, hi in _LABEL_THRESHOLDS:
        if lo <= score < hi:
            return label
    return "VERY_BULLISH" if score >= 0.6 else "VERY_BEARISH"


def _is_likely_non_english(text: str) -> bool:
    """Heuristic: if >40 % of characters are non-ASCII, treat as non-English."""
    if not text:
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return (non_ascii / len(text)) > 0.40


# ---------------------------------------------------------------------------
# SentimentCalibrator
# ---------------------------------------------------------------------------


class SentimentCalibrator:
    """
    Track sentiment predictions vs actual market moves and derive a bias
    correction offset over time.
    """

    def __init__(self) -> None:
        self._outcomes: list[_Outcome] = []
        self._stats: Optional[CalibrationStats] = None

    def record_outcome(
        self,
        article_sentiment: float,
        actual_market_move_pct: float,
        timestamp: datetime,
    ) -> None:
        self._outcomes.append(
            _Outcome(article_sentiment, actual_market_move_pct, timestamp),
        )
        # Invalidate cached stats so they are recomputed on next access
        self._stats = None

    def get_calibration_stats(
        self,
        min_samples: int = 50,
    ) -> Optional[CalibrationStats]:
        if len(self._outcomes) < min_samples:
            return None

        bullish_correct = 0
        bullish_total = 0
        bearish_correct = 0
        bearish_total = 0
        bias_sum = 0.0

        for o in self._outcomes:
            # Bias: positive means we predicted higher than reality
            bias_sum += o.sentiment - (o.market_move_pct / 100.0)

            if o.sentiment > 0.05:
                bullish_total += 1
                if o.market_move_pct > 0:
                    bullish_correct += 1
            elif o.sentiment < -0.05:
                bearish_total += 1
                if o.market_move_pct < 0:
                    bearish_correct += 1

        n = len(self._outcomes)
        avg_bias = bias_sum / n

        self._stats = CalibrationStats(
            sample_count=n,
            avg_bullish_accuracy=(
                bullish_correct / bullish_total if bullish_total else 0.0
            ),
            avg_bearish_accuracy=(
                bearish_correct / bearish_total if bearish_total else 0.0
            ),
            sentiment_bias=round(avg_bias, 6),
            suggested_offset=round(-avg_bias * 0.5, 6),
        )
        return self._stats

    def apply_calibration(self, raw_score: float) -> float:
        stats = self._stats or self.get_calibration_stats()
        if stats is None:
            return raw_score
        return raw_score + stats.suggested_offset


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------


class SentimentAnalyzer:
    """
    Analyse financial news sentiment using VADER + domain keyword boosting.

    Parameters
    ----------
    keywords_path:
        Path to ``sentiment_keywords.json``.  Falls back to the project
        default from ``config.settings``.
    calibrator:
        Optional :class:`SentimentCalibrator` for bias correction.
    """

    def __init__(
        self,
        keywords_path: Optional[Path] = None,
        calibrator: Optional[SentimentCalibrator] = None,
    ) -> None:
        self._vader = SentimentIntensityAnalyzer()
        self._calibrator = calibrator
        self.hinglish_processor = HinglishProcessor()

        kw_path = keywords_path or settings.sentiment_keywords_path
        self._bullish, self._bearish, self._uncertainty = self._load_keywords(
            kw_path,
        )

    # ── keyword loading ──────────────────────────────────────────────────

    @staticmethod
    def _load_keywords(
        path: Path,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        bullish: dict[str, float] = {}
        bearish: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("bullish", []):
                bullish[entry["keyword"].lower()] = float(entry["intensity"])
            for entry in data.get("bearish", []):
                bearish[entry["keyword"].lower()] = float(entry["intensity"])
            for entry in data.get("neutral_uncertainty", []):
                uncertainty[entry["keyword"].lower()] = float(entry["intensity"])
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to load sentiment keywords from %s: %s", path, exc)
        return bullish, bearish, uncertainty

    # ── core analysis ────────────────────────────────────────────────────

    def analyze_sentiment(self, article: ParsedArticle) -> SentimentResult:
        """Return a :class:`SentimentResult` for *article*."""

        text = article.clean_text
        credibility = article.source_credibility

        # Edge case: empty text
        if not text or not text.strip():
            return SentimentResult(
                raw_vader_score=0.0,
                keyword_boost=0.0,
                source_weight=0.5 + 0.5 * credibility,
                adjusted_score=0.0,
                sentiment_label="NEUTRAL",
                confidence=0.0,
            )

        # Edge case: likely non-English (safety net — article_parser should have
        # translated already, but guard against articles that bypassed the parser)
        if _is_likely_non_english(text) and not getattr(article, "was_translated", False):
            logger.warning(
                "Non-English text detected for '%s'; returning NEUTRAL",
                article.title[:60],
            )
            return SentimentResult(
                raw_vader_score=0.0,
                keyword_boost=0.0,
                source_weight=0.5 + 0.5 * credibility,
                adjusted_score=0.0,
                sentiment_label="NEUTRAL",
                confidence=0.05,
            )

        # Translation confidence modifier — slightly reduces confidence for
        # translated articles to reflect potential translation errors
        if getattr(article, "was_translated", False):
            confidence_modifier = getattr(article, "translation_confidence", 1.0) * 0.9
        else:
            confidence_modifier = 1.0

        # Normalise all-caps text for VADER
        normalized = text.lower() if text.isupper() else text

        # Step 1: base VADER score
        base_score = self._vader.polarity_scores(normalized)["compound"]

        # Apply Hinglish-specific keyword boosts using the original (pre-translation)
        # title, which retains Hinglish terms that may have been normalised away
        if getattr(article, "is_hinglish", False):
            base_score = self.hinglish_processor.boost_hinglish_sentiment(
                article.title, base_score
            )

        # Step 2: keyword boosting
        lower_text = text.lower()
        bullish_found, bearish_found, uncertainty_found = self._scan_keywords(
            lower_text,
        )
        total_boost = self._compute_keyword_boost(
            bullish_found, bearish_found, uncertainty_found, base_score,
        )
        keyword_count = (
            len(bullish_found) + len(bearish_found) + len(uncertainty_found)
        )

        # Step 3: source credibility weighting
        source_weight = 0.5 + 0.5 * credibility
        adjusted = (base_score + total_boost) * source_weight

        # Optional calibration
        if self._calibrator:
            adjusted = self._calibrator.apply_calibration(adjusted)

        # Step 4: clamp
        adjusted = max(-1.0, min(1.0, adjusted))

        # Confidence
        confidence = min(
            1.0,
            abs(adjusted) * 0.8
            + credibility * 0.15
            + min(keyword_count * 0.05, 0.2),
        )

        # Edge case: very short text → reduce confidence
        if len(text.split()) < 10:
            confidence = max(0.0, confidence - 0.3)

        # Apply translation confidence modifier
        confidence = round(confidence * confidence_modifier, 4)

        label = _score_to_label(adjusted)

        # Attach a warning if translation quality is low
        result_warnings: list[str] = []
        if (
            getattr(article, "was_translated", False)
            and getattr(article, "translation_confidence", 1.0) < 0.70
        ):
            result_warnings.append(
                f"Translated from {getattr(article, 'original_language', 'unknown')}"
                " — accuracy may vary"
            )

        return SentimentResult(
            raw_vader_score=base_score,
            keyword_boost=total_boost,
            source_weight=source_weight,
            adjusted_score=round(adjusted, 6),
            sentiment_label=label,
            confidence=confidence,
            bullish_keywords_found=bullish_found,
            bearish_keywords_found=bearish_found,
            uncertainty_keywords_found=uncertainty_found,
            keyword_count=keyword_count,
            warnings=result_warnings,
        )

    # ── batch analysis ───────────────────────────────────────────────────

    def analyze_batch(
        self,
        articles: list[ParsedArticle],
    ) -> list[tuple[ParsedArticle, SentimentResult]]:
        """Analyse a batch of articles, sorted by strongest sentiment first."""

        results: list[tuple[ParsedArticle, SentimentResult]] = []
        for article in articles:
            results.append((article, self.analyze_sentiment(article)))

        results.sort(key=lambda pair: abs(pair[1].adjusted_score), reverse=True)

        bullish = sum(1 for _, r in results if r.adjusted_score > 0.05)
        bearish = sum(1 for _, r in results if r.adjusted_score < -0.05)
        neutral = len(results) - bullish - bearish
        logger.info(
            "Analyzed %d articles: %d bullish, %d bearish, %d neutral",
            len(results),
            bullish,
            bearish,
            neutral,
        )
        return results

    # ── headline-only sentiment ──────────────────────────────────────────

    def get_headline_sentiment(self, title: str) -> float:
        """Quick sentiment score for a headline string.  Returns [-1, +1]."""

        if not title or not title.strip():
            return 0.0

        normalized = title.lower() if title.isupper() else title
        base = self._vader.polarity_scores(normalized)["compound"]

        lower = title.lower()
        bullish_found, bearish_found, uncertainty_found = self._scan_keywords(lower)
        boost = self._compute_keyword_boost(
            bullish_found, bearish_found, uncertainty_found, base,
        )
        return max(-1.0, min(1.0, base + boost))

    # ── internal helpers ─────────────────────────────────────────────────

    def _scan_keywords(
        self,
        lower_text: str,
    ) -> tuple[list[str], list[str], list[str]]:
        bullish = [kw for kw in self._bullish if kw in lower_text]
        bearish = [kw for kw in self._bearish if kw in lower_text]
        uncertainty = [kw for kw in self._uncertainty if kw in lower_text]
        return bullish, bearish, uncertainty

    def _compute_keyword_boost(
        self,
        bullish_found: list[str],
        bearish_found: list[str],
        uncertainty_found: list[str],
        base_score: float,
    ) -> float:
        individual_boosts: list[float] = []

        for kw in bullish_found:
            individual_boosts.append(self._bullish[kw] * 0.15)
        for kw in bearish_found:
            individual_boosts.append(-self._bearish[kw] * 0.15)

        num_matches = len(individual_boosts) + len(uncertainty_found)
        if num_matches == 0:
            return 0.0

        raw_boost = sum(individual_boosts)
        # Diminishing returns
        raw_boost *= 1 / (1 + 0.1 * num_matches)

        # Uncertainty keywords reduce the absolute magnitude
        uncertainty_penalty = sum(
            self._uncertainty[kw] * 0.1 for kw in uncertainty_found
        )
        effective = base_score + raw_boost
        if effective > 0:
            raw_boost -= uncertainty_penalty
        elif effective < 0:
            raw_boost += uncertainty_penalty

        return raw_boost
