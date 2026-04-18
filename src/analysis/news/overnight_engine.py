"""
Overnight news engine — collects market-relevant information during off-market
hours (3:30 PM → 9:15 AM IST) and compiles a pre-market intelligence snapshot
ready for the Decision Engine at market open.

Unlike :class:`~src.analysis.news.news_engine.NewsEngine` (polls every 2 min
during market hours), this engine runs at a small number of SCHEDULED,
STRATEGIC times when important information becomes available:

    16:00  post_market        — closing commentary, intraday recap
    18:30  evening_data       — FII/DII + post-market corporate filings
    01:45  us_market_close    — S&P/NDX/Dow/VIX/Crude/DXY + US news
    06:45  asian_open         — Nikkei, HSI, SGX Nifty, STI
    08:45  pre_market_final   — compile everything, persist snapshot, generate report

Pipeline integration
--------------------
Scheduling lives in ``src/data/data_collector.py``.  The compiled snapshot
is persisted to the ``overnight_data`` table (one row per trading day), so
the Decision Engine can load it synchronously at market open without
coordinating with this background process.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.analysis.news.impact_mapper import MappedArticle
from src.analysis.news.news_engine import NewsEngine
from src.database.db_manager import DatabaseManager
from src.utils.date_utils import get_ist_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------


_DEFAULT_CONFIG: dict[str, Any] = {
    "us_indices": {
        "S&P 500":   "^GSPC",
        "NASDAQ":    "^IXIC",
        "Dow Jones": "^DJI",
        "VIX":       "^VIX",
    },
    "global_macro": {
        "Crude Oil":    "CL=F",
        "Dollar Index": "DX-Y.NYB",
    },
    "asian_indices": {
        "Nikkei 225":    "^N225",
        "Hang Seng":     "^HSI",
        "Straits Times": "^STI",
        "SGX Nifty":     "^NSEI",
    },
    "impact_scoring": {
        "sp500_strong_threshold_pct":  1.0,
        "sp500_mild_threshold_pct":    0.3,
        "vix_elevated_threshold":      25,
        "vix_complacent_threshold":    13,
        "crude_strong_threshold_pct":  3.0,
        "asian_strong_threshold_pct":  0.3,
    },
    "gap_prediction_weights": {
        "us_market":      0.35,
        "asian_markets":  0.25,
        "fii":            0.20,
        "news_sentiment": 0.20,
    },
    "fii_bias_thresholds": {
        "strong_buying_cr":   2000,
        "buying_cr":           500,
        "selling_cr":         -500,
        "strong_selling_cr": -2000,
    },
}


def _load_overnight_config() -> dict[str, Any]:
    """Load overnight_sources.json from the project config dir, with defaults."""
    try:
        path = Path(settings.config_dir) / "overnight_sources.json"
    except Exception:
        path = Path(__file__).resolve().parents[3] / "config" / "overnight_sources.json"

    if not path.exists():
        logger.warning("overnight_sources.json not found at %s — using defaults", path)
        return dict(_DEFAULT_CONFIG)

    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Merge with defaults so missing sections still work
        merged = dict(_DEFAULT_CONFIG)
        for k, v in cfg.items():
            merged[k] = v
        return merged
    except Exception as exc:
        logger.error("Failed to load overnight_sources.json (%s) — using defaults", exc)
        return dict(_DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class OvernightData:
    """Accumulated overnight intelligence data for one trading day."""

    # Post-market (4:00 PM)
    post_market_articles: Optional[list[MappedArticle]] = None
    post_market_sentiment: Optional[float] = None

    # Evening (6:30 PM)
    fii_data: Any = None
    fii_trend: Optional[dict] = None
    fii_bias: Optional[str] = None
    corporate_announcements: Optional[list[MappedArticle]] = None

    # US market close (1:45 AM)
    us_market: Optional[dict] = None
    us_impact: Optional[str] = None
    global_news: Optional[list[MappedArticle]] = None

    # Asian open (6:45 AM)
    asian_markets: Optional[dict] = None

    # Pre-market (8:45 AM)
    pre_market_articles: Optional[list[MappedArticle]] = None

    # Compiled analysis
    overall_sentiment: Optional[float] = None    # -1 to +1
    gap_prediction: Optional[dict] = None
    regime_expectation: Optional[str] = None

    is_complete: bool = False
    compiled_at: Optional[datetime] = None

    # ── Serialization helpers ────────────────────────────────────────────

    def to_payload(self) -> dict:
        """Render to a JSON-serializable dict for DB persistence / API output."""
        return {
            "post_market_sentiment": self.post_market_sentiment,
            "fii_data": _fii_to_dict(self.fii_data),
            "fii_trend": self.fii_trend,
            "fii_bias": self.fii_bias,
            "corporate_announcements": [
                _mapped_article_to_dict(a) for a in (self.corporate_announcements or [])
            ],
            "us_market": self.us_market,
            "us_impact": self.us_impact,
            "global_news": [
                _mapped_article_to_dict(a) for a in (self.global_news or [])
            ],
            "asian_markets": self.asian_markets,
            "pre_market_articles": [
                _mapped_article_to_dict(a) for a in (self.pre_market_articles or [])
            ],
            "overall_sentiment": self.overall_sentiment,
            "gap_prediction": self.gap_prediction,
            "regime_expectation": self.regime_expectation,
            "is_complete": self.is_complete,
            "compiled_at": self.compiled_at.isoformat() if self.compiled_at else None,
        }


def _fii_to_dict(fii: Any) -> Optional[dict]:
    if fii is None:
        return None
    try:
        if is_dataclass(fii):
            d = asdict(fii)
        else:
            d = dict(fii.__dict__)
        if "date" in d and hasattr(d["date"], "isoformat"):
            d["date"] = d["date"].isoformat()
        return d
    except Exception:
        return None


def _mapped_article_to_dict(mapped: MappedArticle) -> dict:
    """Slim a MappedArticle down to the fields that survive JSON round-tripping."""
    article = mapped.article
    pub = article.published_at
    return {
        "title": article.title,
        "source": article.source,
        "url": article.url,
        "published_at": pub.isoformat() if pub else None,
        "event_type": article.event_type,
        "impact_severity": mapped.impact_severity,
        "sentiment_label": mapped.sentiment.sentiment_label,
        "sentiment_score": mapped.sentiment.adjusted_score,
        "affected_indices": [imp.index_id for imp in mapped.index_impacts],
    }


# ---------------------------------------------------------------------------
# OvernightNewsEngine
# ---------------------------------------------------------------------------


class OvernightNewsEngine:
    """Collects and analyzes market-relevant information during off-market hours.

    Runs at SPECIFIC strategic times when important information becomes
    available (not hourly polling). The compiled snapshot is persisted to
    the ``overnight_data`` table so the Decision Engine can consume it at
    market open.

    Parameters
    ----------
    db:
        Shared :class:`DatabaseManager` instance.
    news_engine:
        Existing :class:`NewsEngine` — its parser, sentiment analyzer,
        impact mapper and RSS fetcher are reused.
    """

    def __init__(self, db: DatabaseManager, news_engine: NewsEngine) -> None:
        self.db = db
        self.news_engine = news_engine
        self.config = _load_overnight_config()
        self.overnight_data = OvernightData()

    # ==================================================================
    # Scheduled collection windows
    # ==================================================================

    def collect_post_market(self) -> None:
        """Run at 16:00 IST — closing commentary / intraday recap."""
        logger.info("OvernightEngine: post-market collection starting")
        parsed = self._fetch_and_map()

        self.overnight_data.post_market_articles = parsed
        self.overnight_data.post_market_sentiment = self._aggregate_sentiment(parsed)

        logger.info(
            "Post-market collection: %d articles, sentiment: %.2f",
            len(parsed), self.overnight_data.post_market_sentiment or 0.0,
        )

    def collect_evening_data(self) -> None:
        """Run at 18:30 IST — FII/DII data + post-market corporate filings."""
        logger.info("OvernightEngine: evening data collection starting")

        # FII/DII
        try:
            from src.data.fii_dii_data import FIIDIIFetcher

            fetcher = FIIDIIFetcher()
            fii_data = fetcher.fetch_today_fii_dii()
            if fii_data:
                try:
                    fetcher.save_to_db(fii_data, self.db)
                except Exception:
                    logger.exception("Failed to persist FII/DII data (non-fatal)")

                fii_trend = fetcher.get_fii_trend(days=5)
                self.overnight_data.fii_data = fii_data
                self.overnight_data.fii_trend = fii_trend
                self.overnight_data.fii_bias = self._classify_fii_bias(
                    fii_data, fii_trend,
                )
                logger.info(
                    "FII data: net \u20b9%+,.0f Cr | 5-day trend: %s | bias: %s",
                    fii_data.fii_net_value,
                    (fii_trend or {}).get("trend", "UNKNOWN"),
                    self.overnight_data.fii_bias,
                )
            else:
                logger.info("FII/DII data unavailable at 18:30 — will retry at 19:30")
        except Exception:
            logger.exception("Evening FII/DII fetch failed (non-fatal)")

        # Corporate announcements
        parsed = self._fetch_and_map()
        corporate = [
            a for a in parsed
            if a.article.event_type in ("EARNINGS", "CORPORATE", "REGULATORY")
        ]
        self.overnight_data.corporate_announcements = corporate

        if corporate:
            logger.info("Corporate announcements: %d items", len(corporate))
            for c in corporate[:3]:
                logger.info(
                    "  [%s] %s", c.impact_severity, c.article.title[:80],
                )

    def collect_evening_retry(self) -> None:
        """Run at 19:30 IST — retry FII/DII if 18:30 fetch missed."""
        if self.overnight_data.fii_data is not None:
            return
        logger.info("OvernightEngine: retrying FII/DII fetch at 19:30")
        try:
            from src.data.fii_dii_data import FIIDIIFetcher

            fetcher = FIIDIIFetcher()
            fii_data = fetcher.fetch_today_fii_dii()
            if fii_data:
                try:
                    fetcher.save_to_db(fii_data, self.db)
                except Exception:
                    logger.exception("FII/DII retry: persistence failed (non-fatal)")

                fii_trend = fetcher.get_fii_trend(days=5)
                self.overnight_data.fii_data = fii_data
                self.overnight_data.fii_trend = fii_trend
                self.overnight_data.fii_bias = self._classify_fii_bias(
                    fii_data, fii_trend,
                )
                logger.info(
                    "FII data (retry): net \u20b9%+,.0f Cr | bias: %s",
                    fii_data.fii_net_value, self.overnight_data.fii_bias,
                )
        except Exception:
            logger.exception("Evening FII/DII retry failed")

    def collect_us_market_close(self) -> None:
        """Run at 01:45 IST — US market close data + global news."""
        logger.info("OvernightEngine: US market close collection starting")

        us_data: dict[str, dict] = {}

        # US indices
        for name, symbol in self.config.get("us_indices", {}).items():
            vals = self._fetch_ticker_change(symbol)
            if vals:
                us_data[name] = vals

        # Crude / dollar index / gold
        for name, symbol in self.config.get("global_macro", {}).items():
            vals = self._fetch_ticker_change(symbol)
            if vals:
                us_data[name] = vals

        self.overnight_data.us_market = us_data
        self.overnight_data.us_impact = self._assess_us_impact(us_data)

        # US / global news
        parsed = self._fetch_and_map()
        global_articles = [
            a for a in parsed
            if a.article.event_type in ("GLOBAL", "MACRO")
        ]
        self.overnight_data.global_news = global_articles

        sp500_pct = us_data.get("S&P 500", {}).get("change_pct")
        logger.info(
            "US market close: S&P %s%% | Impact: %s | Global news: %d",
            f"{sp500_pct:+.2f}" if sp500_pct is not None else "N/A",
            self.overnight_data.us_impact,
            len(global_articles),
        )

    def collect_asian_open(self) -> None:
        """Run at 06:45 IST — Asian market opens and SGX Nifty."""
        logger.info("OvernightEngine: Asian market open collection starting")

        asian_data: dict[str, dict] = {}
        for name, symbol in self.config.get("asian_indices", {}).items():
            vals = self._fetch_ticker_change(symbol)
            if vals:
                asian_data[name] = vals

        self.overnight_data.asian_markets = asian_data

        if asian_data:
            summary = ", ".join(
                f"{k}: {v.get('change_pct', 0):+.1f}%" for k, v in asian_data.items()
            )
            logger.info("Asian markets: %s", summary)
        else:
            logger.info("Asian markets: no data fetched (markets may not be open)")

    def collect_pre_market_final(self) -> None:
        """Run at 08:45 IST — final news fetch, compile snapshot, persist."""
        logger.info("OvernightEngine: pre-market final compilation starting")

        morning_articles = self._fetch_and_map()
        self.overnight_data.pre_market_articles = morning_articles

        all_articles: list[MappedArticle] = []
        all_articles.extend(self.overnight_data.post_market_articles or [])
        all_articles.extend(self.overnight_data.corporate_announcements or [])
        all_articles.extend(self.overnight_data.global_news or [])
        all_articles.extend(morning_articles)

        self.overnight_data.overall_sentiment = self._aggregate_sentiment(all_articles)
        self.overnight_data.gap_prediction = self._predict_gap(self.overnight_data)
        self.overnight_data.regime_expectation = self._predict_regime(
            self.overnight_data,
        )
        self.overnight_data.is_complete = True
        self.overnight_data.compiled_at = get_ist_now()

        self._save_overnight_data(self.overnight_data)

        gap = self.overnight_data.gap_prediction or {}
        logger.info(
            "Pre-market compilation complete. Sentiment: %.2f | Gap: %s (%.0f%%) | Regime: %s",
            self.overnight_data.overall_sentiment or 0.0,
            gap.get("direction", "?"),
            (gap.get("confidence", 0.0) or 0.0) * 100,
            self.overnight_data.regime_expectation or "NORMAL",
        )

    # ==================================================================
    # Analysis methods
    # ==================================================================

    def _assess_us_impact(self, us_data: dict) -> str:
        """Classify how the US session will impact the Indian open."""
        if not us_data:
            return "UNAVAILABLE"

        sc = self.config.get("impact_scoring", {})
        sp500 = us_data.get("S&P 500", {}).get("change_pct", 0) or 0
        vix = us_data.get("VIX", {}).get("close", 15) or 15
        crude = us_data.get("Crude Oil", {}).get("change_pct", 0) or 0

        score = 0.0
        if sp500 > sc.get("sp500_strong_threshold_pct", 1.0):
            score += 2
        elif sp500 > sc.get("sp500_mild_threshold_pct", 0.3):
            score += 1
        elif sp500 < -sc.get("sp500_strong_threshold_pct", 1.0):
            score -= 2
        elif sp500 < -sc.get("sp500_mild_threshold_pct", 0.3):
            score -= 1

        if vix > sc.get("vix_elevated_threshold", 25):
            score -= 1
        elif vix < sc.get("vix_complacent_threshold", 13):
            score += 0.5

        if crude > sc.get("crude_strong_threshold_pct", 3.0):
            score -= 1
        elif crude < -sc.get("crude_strong_threshold_pct", 3.0):
            score += 0.5

        if score >= 2:
            return "STRONG_POSITIVE"
        if score >= 1:
            return "MILDLY_POSITIVE"
        if score <= -2:
            return "STRONG_NEGATIVE"
        if score <= -1:
            return "MILDLY_NEGATIVE"
        return "NEUTRAL"

    def _predict_gap(self, data: OvernightData) -> dict:
        """Predict whether Indian market will gap up / down at open."""
        w = self.config.get("gap_prediction_weights", {})
        w_us = w.get("us_market", 0.35)
        w_asian = w.get("asian_markets", 0.25)
        w_fii = w.get("fii", 0.20)
        w_news = w.get("news_sentiment", 0.20)

        signals: list[tuple[str, str, float]] = []

        us_impact = data.us_impact or "NEUTRAL"
        if "POSITIVE" in us_impact:
            signals.append(("US_MARKET", "GAP_UP", w_us))
        elif "NEGATIVE" in us_impact:
            signals.append(("US_MARKET", "GAP_DOWN", w_us))

        if data.asian_markets:
            asian_avg = sum(
                (v.get("change_pct", 0) or 0) for v in data.asian_markets.values()
            ) / max(len(data.asian_markets), 1)
            sc = self.config.get("impact_scoring", {})
            thr = sc.get("asian_strong_threshold_pct", 0.3)
            if asian_avg > thr:
                signals.append(("ASIAN_MARKETS", "GAP_UP", w_asian))
            elif asian_avg < -thr:
                signals.append(("ASIAN_MARKETS", "GAP_DOWN", w_asian))

        if data.fii_bias in ("STRONG_BUYING", "BUYING"):
            signals.append(("FII", "GAP_UP", w_fii))
        elif data.fii_bias in ("STRONG_SELLING", "SELLING"):
            signals.append(("FII", "GAP_DOWN", w_fii))

        if data.overall_sentiment is not None:
            if data.overall_sentiment > 0.3:
                signals.append(("NEWS", "GAP_UP", w_news))
            elif data.overall_sentiment < -0.3:
                signals.append(("NEWS", "GAP_DOWN", w_news))

        up_weight = sum(w for _, d, w in signals if d == "GAP_UP")
        down_weight = sum(w for _, d, w in signals if d == "GAP_DOWN")

        if up_weight > down_weight + 0.15:
            direction = "GAP_UP"
            confidence = min(up_weight, 0.85)
        elif down_weight > up_weight + 0.15:
            direction = "GAP_DOWN"
            confidence = min(down_weight, 0.85)
        else:
            direction = "FLAT_OPEN"
            confidence = 0.4

        # Reduce confidence when we're missing major inputs
        missing = 0
        if not data.us_market:
            missing += 1
        if not data.asian_markets:
            missing += 1
        if data.fii_bias is None:
            missing += 1
        if missing > 0:
            confidence = max(0.2, confidence - 0.1 * missing)

        return {
            "direction": direction,
            "confidence": round(confidence, 2),
            "signals": [
                {"source": s, "direction": d, "weight": round(w, 2)}
                for s, d, w in signals
            ],
            "reasoning": self._build_gap_reasoning(signals, direction),
        }

    def _predict_regime(self, data: OvernightData) -> str:
        """Predict expected market regime for the next session."""
        critical = [
            a for a in ((data.global_news or []) + (data.corporate_announcements or []))
            if a.impact_severity == "CRITICAL"
        ]
        if critical:
            return "EVENT_DRIVEN"

        sp500_change = (data.us_market or {}).get("S&P 500", {}).get("change_pct", 0) or 0
        if sp500_change < -2:
            return "VOLATILE_CHOPPY"

        vix = (data.us_market or {}).get("VIX", {}).get("close")
        if vix is not None and vix > 30:
            return "VOLATILE_CHOPPY"

        return "NORMAL"

    @staticmethod
    def _build_gap_reasoning(signals: list[tuple], direction: str) -> str:
        """Render a human-readable reasoning string for the gap prediction."""
        if not signals:
            return "No strong directional signals overnight."
        lines: list[str] = []
        for source, d, weight in signals:
            arrow = "\u2191" if d == "GAP_UP" else "\u2193"
            lines.append(f"{source.replace('_', ' ').title()} {arrow} (w={weight:.2f})")
        return f"{direction.replace('_', ' ').title()} based on: " + "; ".join(lines)

    def _classify_fii_bias(self, fii_data: Any, fii_trend: Optional[dict]) -> str:
        """Classify FII bias using today's net + 5-day trend."""
        thr = self.config.get("fii_bias_thresholds", {})
        net = float(getattr(fii_data, "fii_net_value", 0) or 0)
        trend = (fii_trend or {}).get("trend", "NEUTRAL")

        if net >= thr.get("strong_buying_cr", 2000) or (
            net >= thr.get("buying_cr", 500) and trend == "BUYING"
        ):
            return "STRONG_BUYING"
        if net >= thr.get("buying_cr", 500):
            return "BUYING"
        if net <= thr.get("strong_selling_cr", -2000) or (
            net <= thr.get("selling_cr", -500) and trend == "SELLING"
        ):
            return "STRONG_SELLING"
        if net <= thr.get("selling_cr", -500):
            return "SELLING"
        return "NEUTRAL"

    @staticmethod
    def _aggregate_sentiment(articles: list[MappedArticle]) -> float:
        """Aggregate article sentiment weighted by source credibility + severity."""
        if not articles:
            return 0.0
        _SEV_WEIGHT = {"CRITICAL": 2.0, "HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.5, "NOISE": 0.2}

        numerator = 0.0
        denominator = 0.0
        for a in articles:
            cred = getattr(a.article, "source_credibility", 1.0) or 1.0
            sev_w = _SEV_WEIGHT.get(a.impact_severity, 0.5)
            w = cred * sev_w
            numerator += (a.sentiment.adjusted_score or 0.0) * w
            denominator += w

        if denominator == 0:
            return 0.0
        return round(max(-1.0, min(1.0, numerator / denominator)), 3)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _fetch_and_map(self) -> list[MappedArticle]:
        """Fetch, parse, sentiment + impact map one round of RSS articles.

        Returns newly mapped articles (duplicates against the in-memory /
        DB cache are skipped via the underlying NewsEngine pipeline).
        """
        try:
            raw = self.news_engine.fetcher.fetch_all_feeds()
        except Exception:
            logger.exception("Overnight _fetch_and_map: RSS fetch failed")
            return []

        parsed = self.news_engine._parse(raw)
        new_articles = self.news_engine._dedup(parsed)
        mapped = self.news_engine._analyze_and_map(new_articles)

        # Persist — cross-process visibility if the main NewsEngine reads
        # from DB on startup.
        try:
            self.news_engine._save(mapped)
        except Exception:
            logger.debug("Overnight _save failed (non-fatal)", exc_info=True)

        return mapped

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=False,
    )
    def _fetch_ticker_change(self, symbol: str) -> Optional[dict]:
        """Fetch last-close + percent-change for *symbol* via yfinance.

        Returns ``None`` on failure (yfinance raises frequently for
        temporary rate limiting / network hiccups).
        """
        try:
            import yfinance as yf  # noqa: PLC0415 — lazy import
        except ImportError:
            logger.warning("yfinance not installed — cannot fetch %s", symbol)
            return None

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", auto_adjust=False)
            if hist is None or len(hist) < 2:
                return None
            prev_close = float(hist.iloc[-2]["Close"])
            last_close = float(hist.iloc[-1]["Close"])
            if prev_close <= 0:
                return None
            change_pct = (last_close - prev_close) / prev_close * 100.0
            return {
                "close": round(last_close, 2),
                "change_pct": round(change_pct, 2),
                "direction": "UP" if change_pct > 0.1 else ("DOWN" if change_pct < -0.1 else "FLAT"),
            }
        except Exception as exc:
            logger.debug("yfinance fetch failed for %s: %s", symbol, exc)
            return None

    # ==================================================================
    # Persistence
    # ==================================================================

    def _save_overnight_data(self, data: OvernightData) -> None:
        """Persist the compiled snapshot to the ``overnight_data`` table."""
        try:
            compiled_at = data.compiled_at or get_ist_now()
            trade_date = compiled_at.date().isoformat()
            payload = json.dumps(data.to_payload(), default=str)

            gap = data.gap_prediction or {}
            self.db.execute(
                """
                INSERT INTO overnight_data
                    (trade_date, compiled_at, is_complete, overall_sentiment,
                     gap_direction, gap_confidence, regime_expectation,
                     us_impact, fii_bias, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_date) DO UPDATE SET
                    compiled_at        = excluded.compiled_at,
                    is_complete        = excluded.is_complete,
                    overall_sentiment  = excluded.overall_sentiment,
                    gap_direction      = excluded.gap_direction,
                    gap_confidence     = excluded.gap_confidence,
                    regime_expectation = excluded.regime_expectation,
                    us_impact          = excluded.us_impact,
                    fii_bias           = excluded.fii_bias,
                    payload_json       = excluded.payload_json
                """,
                (
                    trade_date,
                    compiled_at.isoformat(),
                    1 if data.is_complete else 0,
                    data.overall_sentiment,
                    gap.get("direction"),
                    gap.get("confidence"),
                    data.regime_expectation,
                    data.us_impact,
                    data.fii_bias,
                    payload,
                ),
            )
            logger.info("Overnight snapshot persisted for %s", trade_date)
        except Exception:
            logger.exception("Failed to persist overnight snapshot")


# ---------------------------------------------------------------------------
# Read-side helper used by DecisionEngine / API / Telegram
# ---------------------------------------------------------------------------


def load_latest_overnight_data(
    db: DatabaseManager,
    trade_date: Optional[str] = None,
) -> Optional[dict]:
    """Return the most recent overnight snapshot as a dict.

    Parameters
    ----------
    db:
        Connected DatabaseManager.
    trade_date:
        ISO date string. When given, returns the row matching that date
        exactly (or ``None`` if not found). When omitted, returns the
        newest row.

    Returns
    -------
    dict | None:
        Dict with the top-level snapshot columns plus a decoded
        ``payload`` sub-dict. ``None`` if no snapshot exists.
    """
    try:
        if trade_date:
            row = db.fetch_one(
                "SELECT * FROM overnight_data WHERE trade_date = ?",
                (trade_date,),
            )
        else:
            row = db.fetch_one(
                "SELECT * FROM overnight_data ORDER BY trade_date DESC LIMIT 1", (),
            )
        if row is None:
            return None

        result = dict(row)
        try:
            result["payload"] = json.loads(result.pop("payload_json", "{}") or "{}")
        except Exception:
            result["payload"] = {}
        result["is_complete"] = bool(result.get("is_complete"))
        return result
    except Exception:
        logger.exception("load_latest_overnight_data failed")
        return None
