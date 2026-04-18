"""
Unit tests for OvernightNewsEngine and PreMarketReportGenerator.

Strategy
--------
- Pure analysis methods (_assess_us_impact, _predict_gap, _aggregate_sentiment,
  _classify_fii_bias) are tested directly with controlled inputs — no network,
  no real DB.
- OvernightData dataclass instances are constructed inline to exercise
  _predict_gap / _predict_regime end-to-end.
- PreMarketReportGenerator is tested for report structure and graceful
  degradation when data is missing.
- DecisionEngine._overnight_decay_weight is tested at boundary times to verify
  the correct time-decay schedule.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.analysis.news.overnight_engine import OvernightData, OvernightNewsEngine
from src.analysis.news.pre_market_report import PreMarketReportGenerator

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine() -> OvernightNewsEngine:
    """Return an OvernightNewsEngine with fully mocked dependencies."""
    db = MagicMock()
    news_engine = MagicMock()
    return OvernightNewsEngine(db=db, news_engine=news_engine)


def _make_mapped_article(
    *,
    impact_severity: str = "HIGH",
    sentiment_score: float = 0.5,
    event_type: str = "MACRO",
    source_credibility: float = 0.8,
) -> MagicMock:
    ma = MagicMock()
    ma.article.event_type = event_type
    ma.article.source_credibility = source_credibility
    ma.impact_severity = impact_severity
    ma.sentiment.adjusted_score = sentiment_score
    ma.index_impacts = []
    return ma


def _make_fii(net: float = 1500.0, dii_net: float = 800.0) -> MagicMock:
    fii = MagicMock()
    fii.fii_net_value = net
    fii.dii_net_value = dii_net
    return fii


# ---------------------------------------------------------------------------
# _assess_us_impact
# ---------------------------------------------------------------------------


class TestAssessUsImpact:
    def setup_method(self):
        self.engine = _make_engine()

    def test_strong_negative_sp500_crash_with_high_vix(self):
        """S&P -2%, VIX 28 → STRONG_NEGATIVE."""
        us_data = {
            "S&P 500": {"close": 4800.0, "change_pct": -2.0, "direction": "DOWN"},
            "VIX":     {"close": 28.0,   "change_pct": 15.0,  "direction": "UP"},
        }
        result = self.engine._assess_us_impact(us_data)
        assert result == "STRONG_NEGATIVE"

    def test_strong_positive_sp500_surge(self):
        """S&P +1.5%, low VIX → STRONG_POSITIVE."""
        us_data = {
            "S&P 500": {"close": 5100.0, "change_pct": 1.5, "direction": "UP"},
            "VIX":     {"close": 12.0,   "change_pct": -5.0, "direction": "DOWN"},
        }
        result = self.engine._assess_us_impact(us_data)
        assert result == "STRONG_POSITIVE"

    def test_mildly_positive(self):
        """S&P +0.5%, normal VIX → MILDLY_POSITIVE."""
        us_data = {
            "S&P 500": {"close": 5000.0, "change_pct": 0.5, "direction": "UP"},
            "VIX":     {"close": 17.0,   "change_pct": 0.0,  "direction": "FLAT"},
        }
        result = self.engine._assess_us_impact(us_data)
        assert result == "MILDLY_POSITIVE"

    def test_mildly_negative(self):
        """S&P -0.5%, normal VIX → MILDLY_NEGATIVE."""
        us_data = {
            "S&P 500": {"close": 4900.0, "change_pct": -0.5, "direction": "DOWN"},
            "VIX":     {"close": 16.0,   "change_pct": 0.0,   "direction": "FLAT"},
        }
        result = self.engine._assess_us_impact(us_data)
        assert result == "MILDLY_NEGATIVE"

    def test_neutral_flat_market(self):
        """S&P ±0.1%, normal VIX → NEUTRAL."""
        us_data = {
            "S&P 500": {"close": 5000.0, "change_pct": 0.05, "direction": "FLAT"},
            "VIX":     {"close": 15.0,   "change_pct": 0.0,   "direction": "FLAT"},
        }
        result = self.engine._assess_us_impact(us_data)
        assert result == "NEUTRAL"

    def test_empty_us_data_returns_unavailable(self):
        result = self.engine._assess_us_impact({})
        assert result == "UNAVAILABLE"

    def test_high_crude_penalises_score(self):
        """S&P mildly positive but crude +5% should drop to NEUTRAL."""
        us_data = {
            "S&P 500":   {"close": 5000.0, "change_pct": 0.5,  "direction": "UP"},
            "VIX":       {"close": 15.0,   "change_pct": 0.0,   "direction": "FLAT"},
            "Crude Oil": {"close": 85.0,   "change_pct": 5.0,   "direction": "UP"},
        }
        result = self.engine._assess_us_impact(us_data)
        # score: +1 (S&P mild) -1 (crude) = 0 → NEUTRAL
        assert result == "NEUTRAL"

    def test_sp500_crash_alone_is_strong_negative(self):
        """S&P -2% alone (no VIX data) should still be STRONG_NEGATIVE."""
        us_data = {"S&P 500": {"close": 4500.0, "change_pct": -2.0, "direction": "DOWN"}}
        result = self.engine._assess_us_impact(us_data)
        assert result == "STRONG_NEGATIVE"


# ---------------------------------------------------------------------------
# _predict_gap
# ---------------------------------------------------------------------------


class TestPredictGap:
    def setup_method(self):
        self.engine = _make_engine()

    def test_all_bearish_signals_predict_gap_down(self):
        """All four signals bearish (no missing data) → GAP_DOWN with confidence ≥ 0.8."""
        data = OvernightData(
            us_market={"S&P 500": {"close": 4500.0, "change_pct": -2.0}},
            us_impact="STRONG_NEGATIVE",
            asian_markets={
                "Nikkei 225":  {"change_pct": -1.2},
                "Hang Seng":   {"change_pct": -0.9},
            },
            fii_data=_make_fii(net=-1200.0),
            fii_bias="STRONG_SELLING",
            overall_sentiment=-0.6,
        )
        result = self.engine._predict_gap(data)

        assert result["direction"] == "GAP_DOWN"
        assert result["confidence"] >= 0.80
        assert len(result["signals"]) == 4
        assert all(s["direction"] == "GAP_DOWN" for s in result["signals"])

    def test_all_bullish_signals_predict_gap_up(self):
        """All four signals bullish (no missing data) → GAP_UP with confidence ≥ 0.8."""
        data = OvernightData(
            us_market={"S&P 500": {"close": 5200.0, "change_pct": 1.5}},
            us_impact="STRONG_POSITIVE",
            asian_markets={
                "Nikkei 225":  {"change_pct": 1.5},
                "Hang Seng":   {"change_pct": 0.8},
            },
            fii_data=_make_fii(net=2200.0),
            fii_bias="STRONG_BUYING",
            overall_sentiment=0.7,
        )
        result = self.engine._predict_gap(data)

        assert result["direction"] == "GAP_UP"
        assert result["confidence"] >= 0.80

    def test_mixed_signals_predict_flat_open(self):
        """US mildly positive vs FII selling → FLAT_OPEN (difference ≤ 0.15)."""
        data = OvernightData(
            us_impact="MILDLY_POSITIVE",   # weight 0.35 → GAP_UP
            asian_markets=None,            # no signal
            fii_bias="SELLING",            # weight 0.20 → GAP_DOWN
            overall_sentiment=0.0,         # no signal
        )
        result = self.engine._predict_gap(data)

        # up_weight=0.35, down_weight=0.20 → difference exactly 0.15 → not strictly >
        assert result["direction"] == "FLAT_OPEN"

    def test_missing_data_reduces_confidence(self):
        """Two missing inputs should reduce confidence by 0.2."""
        data = OvernightData(
            us_impact="STRONG_NEGATIVE",   # us present
            asian_markets=None,            # missing (-0.1)
            fii_bias=None,                 # missing (-0.1)
            overall_sentiment=-0.5,
        )
        result_full = self.engine._predict_gap(OvernightData(
            us_impact="STRONG_NEGATIVE",
            asian_markets={"N225": {"change_pct": -1.0}},
            fii_bias="SELLING",
            overall_sentiment=-0.5,
        ))
        result_missing = self.engine._predict_gap(data)

        # The full version should have higher or equal confidence
        assert result_full["confidence"] >= result_missing["confidence"]

    def test_gap_prediction_contains_reasoning(self):
        """Gap prediction dict must always include a non-empty reasoning string."""
        data = OvernightData(us_impact="MILDLY_NEGATIVE")
        result = self.engine._predict_gap(data)
        assert "reasoning" in result
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0

    def test_confidence_capped_at_85_percent(self):
        """Confidence must never exceed 0.85 regardless of signal strength."""
        data = OvernightData(
            us_impact="STRONG_NEGATIVE",
            asian_markets={
                "N225": {"change_pct": -2.0},
                "HSI":  {"change_pct": -1.5},
            },
            fii_bias="STRONG_SELLING",
            overall_sentiment=-0.9,
        )
        result = self.engine._predict_gap(data)
        assert result["confidence"] <= 0.85


# ---------------------------------------------------------------------------
# _predict_regime
# ---------------------------------------------------------------------------


class TestPredictRegime:
    def setup_method(self):
        self.engine = _make_engine()

    def test_critical_news_triggers_event_driven(self):
        article = _make_mapped_article(impact_severity="CRITICAL", event_type="GLOBAL")
        data = OvernightData(global_news=[article])
        assert self.engine._predict_regime(data) == "EVENT_DRIVEN"

    def test_sp500_crash_triggers_volatile_choppy(self):
        data = OvernightData(
            us_market={"S&P 500": {"close": 4200.0, "change_pct": -2.5}},
        )
        assert self.engine._predict_regime(data) == "VOLATILE_CHOPPY"

    def test_high_vix_triggers_volatile_choppy(self):
        data = OvernightData(
            us_market={"VIX": {"close": 32.0, "change_pct": 20.0}},
        )
        assert self.engine._predict_regime(data) == "VOLATILE_CHOPPY"

    def test_normal_conditions_return_normal(self):
        data = OvernightData(
            us_market={"S&P 500": {"close": 5000.0, "change_pct": 0.3}},
        )
        assert self.engine._predict_regime(data) == "NORMAL"

    def test_empty_data_returns_normal(self):
        assert self.engine._predict_regime(OvernightData()) == "NORMAL"


# ---------------------------------------------------------------------------
# _aggregate_sentiment
# ---------------------------------------------------------------------------


class TestAggregateSentiment:
    def setup_method(self):
        self.engine = _make_engine()

    def test_empty_articles_returns_zero(self):
        assert self.engine._aggregate_sentiment([]) == 0.0

    def test_all_bullish_articles(self):
        articles = [
            _make_mapped_article(sentiment_score=0.8, impact_severity="HIGH"),
            _make_mapped_article(sentiment_score=0.6, impact_severity="MEDIUM"),
        ]
        result = self.engine._aggregate_sentiment(articles)
        assert result > 0.0

    def test_all_bearish_articles(self):
        articles = [
            _make_mapped_article(sentiment_score=-0.7, impact_severity="HIGH"),
            _make_mapped_article(sentiment_score=-0.5, impact_severity="MEDIUM"),
        ]
        result = self.engine._aggregate_sentiment(articles)
        assert result < 0.0

    def test_result_clamped_to_minus_one_plus_one(self):
        articles = [
            _make_mapped_article(sentiment_score=-2.0, impact_severity="CRITICAL"),
        ]
        result = self.engine._aggregate_sentiment(articles)
        assert -1.0 <= result <= 1.0

    def test_critical_article_weighted_more_than_noise(self):
        """A single CRITICAL article should pull sentiment more than NOISE."""
        articles_critical = [
            _make_mapped_article(sentiment_score=0.9, impact_severity="CRITICAL"),
            _make_mapped_article(sentiment_score=-0.5, impact_severity="NOISE"),
        ]
        articles_equal = [
            _make_mapped_article(sentiment_score=0.9, impact_severity="NOISE"),
            _make_mapped_article(sentiment_score=-0.5, impact_severity="NOISE"),
        ]
        result_critical = self.engine._aggregate_sentiment(articles_critical)
        result_equal = self.engine._aggregate_sentiment(articles_equal)
        # The CRITICAL-weighted version should be more positive
        assert result_critical > result_equal

    def test_mixed_sentiment_near_neutral(self):
        """Equal bullish and bearish MEDIUM articles should net near zero."""
        articles = [
            _make_mapped_article(sentiment_score=0.5, impact_severity="MEDIUM"),
            _make_mapped_article(sentiment_score=-0.5, impact_severity="MEDIUM"),
        ]
        result = self.engine._aggregate_sentiment(articles)
        assert abs(result) < 0.05


# ---------------------------------------------------------------------------
# _classify_fii_bias
# ---------------------------------------------------------------------------


class TestClassifyFiiBias:
    def setup_method(self):
        self.engine = _make_engine()

    def test_strong_buying_threshold(self):
        fii = _make_fii(net=2500.0)
        assert self.engine._classify_fii_bias(fii, {"trend": "NEUTRAL"}) == "STRONG_BUYING"

    def test_buying_threshold(self):
        fii = _make_fii(net=800.0)
        assert self.engine._classify_fii_bias(fii, {"trend": "NEUTRAL"}) == "BUYING"

    def test_buying_trend_upgrades_to_strong_buying(self):
        """Net ≥ 500 Cr + BUYING trend → STRONG_BUYING."""
        fii = _make_fii(net=600.0)
        assert self.engine._classify_fii_bias(fii, {"trend": "BUYING"}) == "STRONG_BUYING"

    def test_strong_selling_threshold(self):
        fii = _make_fii(net=-2500.0)
        assert self.engine._classify_fii_bias(fii, {"trend": "NEUTRAL"}) == "STRONG_SELLING"

    def test_selling_threshold(self):
        fii = _make_fii(net=-800.0)
        assert self.engine._classify_fii_bias(fii, {"trend": "NEUTRAL"}) == "SELLING"

    def test_neutral_small_flow(self):
        fii = _make_fii(net=100.0)
        assert self.engine._classify_fii_bias(fii, {"trend": "NEUTRAL"}) == "NEUTRAL"

    def test_selling_trend_upgrades_to_strong_selling(self):
        """Net ≤ -500 Cr + SELLING trend → STRONG_SELLING."""
        fii = _make_fii(net=-600.0)
        assert self.engine._classify_fii_bias(fii, {"trend": "SELLING"}) == "STRONG_SELLING"


# ---------------------------------------------------------------------------
# PreMarketReportGenerator
# ---------------------------------------------------------------------------


class TestPreMarketReportGenerator:
    def setup_method(self):
        self.gen = PreMarketReportGenerator()

    def test_none_data_returns_empty_report_banner(self):
        report = self.gen.generate_report(None)
        assert "No overnight data" in report
        assert "Pre-Market Intelligence" in report

    def test_complete_data_contains_all_sections(self):
        fii = _make_fii(net=1200.0, dii_net=600.0)
        data = OvernightData(
            us_market={
                "S&P 500":   {"close": 5100.0, "change_pct": 0.8, "direction": "UP"},
                "VIX":       {"close": 14.0,   "change_pct": -3.0, "direction": "DOWN"},
                "Crude Oil": {"close": 78.0,   "change_pct": -0.5, "direction": "DOWN"},
            },
            us_impact="MILDLY_POSITIVE",
            asian_markets={
                "Nikkei 225": {"change_pct": 0.6},
                "Hang Seng":  {"change_pct": 0.4},
            },
            fii_data=fii,
            fii_bias="BUYING",
            overall_sentiment=0.45,
            gap_prediction={
                "direction":  "GAP_UP",
                "confidence": 0.65,
                "signals":    [],
                "reasoning":  "Gap Up based on: US Market (w=0.35); Asian Markets (w=0.25)",
            },
            regime_expectation="NORMAL",
            is_complete=True,
            compiled_at=datetime(2026, 4, 18, 8, 45, tzinfo=_IST),
        )
        report = self.gen.generate_report(data)

        assert "Pre-Market Intelligence" in report
        assert "GAP_UP" in report
        assert "US Markets" in report
        assert "S&P 500" in report
        assert "Asian Markets" in report
        assert "Nikkei" in report
        assert "FII" in report
        assert "BULLISH" in report
        assert "Recommendation" in report

    def test_partial_data_no_us_no_fii_still_generates(self):
        """Report must render even when US and FII data are both absent."""
        data = OvernightData(
            asian_markets={"Nikkei 225": {"change_pct": 0.3}},
            overall_sentiment=0.1,
        )
        report = self.gen.generate_report(data)

        assert "Pre-Market Intelligence" in report
        assert "unavailable" in report.lower() or "not yet" in report.lower()

    def test_partial_data_no_asian_markets(self):
        data = OvernightData(
            us_market={"S&P 500": {"close": 5000.0, "change_pct": 0.5}},
            us_impact="MILDLY_POSITIVE",
        )
        report = self.gen.generate_report(data)
        assert "Asian Markets" in report
        assert "Pre-Market Intelligence" in report

    def test_gap_up_high_confidence_recommendation(self):
        data = OvernightData(
            gap_prediction={"direction": "GAP_UP", "confidence": 0.75, "signals": [], "reasoning": ""},
            regime_expectation="NORMAL",
        )
        report = self.gen.generate_report(data)
        assert "higher" in report.lower() or "gap" in report.lower()

    def test_gap_down_high_confidence_recommendation(self):
        data = OvernightData(
            gap_prediction={"direction": "GAP_DOWN", "confidence": 0.72, "signals": [], "reasoning": ""},
        )
        report = self.gen.generate_report(data)
        assert "lower" in report.lower() or "put" in report.lower() or "gap" in report.lower()

    def test_event_driven_regime_recommendation(self):
        data = OvernightData(regime_expectation="EVENT_DRIVEN")
        report = self.gen.generate_report(data)
        assert "event" in report.lower() or "Event" in report

    def test_volatile_choppy_regime_recommendation(self):
        data = OvernightData(regime_expectation="VOLATILE_CHOPPY")
        report = self.gen.generate_report(data)
        assert "volatil" in report.lower() or "high confidence" in report.lower()

    def test_db_row_dict_format_accepted(self):
        """The generator must also handle dicts returned from load_latest_overnight_data."""
        db_row = {
            "trade_date": "2026-04-18",
            "compiled_at": "2026-04-18T08:45:00+05:30",
            "is_complete": 1,
            "overall_sentiment": 0.3,
            "gap_direction": "GAP_UP",
            "gap_confidence": 0.6,
            "regime_expectation": "NORMAL",
            "us_impact": "MILDLY_POSITIVE",
            "fii_bias": "BUYING",
            "payload": {
                "us_market": {"S&P 500": {"change_pct": 0.6}},
                "asian_markets": {"Nikkei 225": {"change_pct": 0.4}},
                "fii_data": {"fii_net_value": 700.0, "dii_net_value": 300.0},
            },
        }
        report = self.gen.generate_report(db_row)
        assert "Pre-Market Intelligence" in report
        assert "GAP_UP" in report

    def test_corporate_announcements_shown(self):
        announcement = MagicMock()
        announcement.impact_severity = "HIGH"
        # Explicitly set .title to None so the report generator falls through
        # to the .article.title attribute (MagicMock auto-attrs are truthy).
        announcement.title = None
        announcement.article.title = "Reliance Q4 profit beats estimates by 12%"

        data = OvernightData(
            corporate_announcements=[announcement],
        )
        report = self.gen.generate_report(data)
        assert "Announcements" in report
        assert "Reliance" in report


# ---------------------------------------------------------------------------
# OvernightData.to_payload  (serialisation round-trip)
# ---------------------------------------------------------------------------


class TestOvernightDataPayload:
    def test_to_payload_is_json_serialisable(self):
        import json

        fii = _make_fii(net=500.0)
        # Give the mock a dataclass-like interface that _fii_to_dict handles
        fii.__dict__ = {"fii_net_value": 500.0, "dii_net_value": 200.0}

        data = OvernightData(
            post_market_sentiment=0.2,
            us_market={"S&P 500": {"close": 5000.0, "change_pct": 0.5}},
            us_impact="MILDLY_POSITIVE",
            overall_sentiment=0.3,
            gap_prediction={"direction": "GAP_UP", "confidence": 0.6, "signals": [], "reasoning": ""},
            regime_expectation="NORMAL",
            is_complete=True,
            compiled_at=datetime(2026, 4, 18, 8, 45, tzinfo=_IST),
        )
        payload = data.to_payload()
        # Must serialise without raising
        serialised = json.dumps(payload, default=str)
        assert len(serialised) > 0
        parsed = json.loads(serialised)
        assert parsed["us_impact"] == "MILDLY_POSITIVE"
        assert parsed["is_complete"] is True

    def test_to_payload_handles_none_fields(self):
        """to_payload must not crash when most fields are None."""
        import json

        data = OvernightData()
        payload = data.to_payload()
        json.dumps(payload, default=str)  # must not raise


# ---------------------------------------------------------------------------
# DecisionEngine._overnight_decay_weight  (time-decay schedule)
# ---------------------------------------------------------------------------


class TestOvernightDecayWeight:
    """Verify the intraday time-decay schedule for overnight data importance."""

    @staticmethod
    def _at(hour: int, minute: int) -> datetime:
        return datetime(2026, 4, 18, hour, minute, tzinfo=_IST)

    def _weight(self, hour: int, minute: int) -> float:
        from src.engine.decision_engine import DecisionEngine

        return DecisionEngine._overnight_decay_weight(self._at(hour, minute))

    def test_at_market_open_weight_is_one(self):
        """9:15 AM → weight = 1.0 (full overnight data relevance)."""
        assert self._weight(9, 15) == pytest.approx(1.0)

    def test_at_10am_weight_is_approximately_half(self):
        """10:00 AM (45 min after open) → weight ≈ 0.5."""
        assert self._weight(10, 0) == pytest.approx(0.5, abs=0.01)

    def test_at_11am_weight_is_approximately_0_2(self):
        """11:00 AM (105 min after open) → weight ≈ 0.2."""
        assert self._weight(11, 0) == pytest.approx(0.2, abs=0.01)

    def test_at_noon_weight_is_zero(self):
        """12:00 PM (165 min after open) → weight = 0.0."""
        assert self._weight(12, 0) == pytest.approx(0.0, abs=0.01)

    def test_before_market_open_weight_is_zero(self):
        """8:00 AM (before open) → weight = 0.0."""
        assert self._weight(8, 0) == 0.0

    def test_after_noon_weight_stays_zero(self):
        """2:00 PM → weight = 0.0."""
        assert self._weight(14, 0) == 0.0

    def test_weight_is_monotonically_decreasing(self):
        """Weight must decrease (or stay equal) as time advances through the session."""
        times = [(9, 15), (9, 30), (10, 0), (10, 30), (11, 0), (11, 30), (12, 0)]
        weights = [self._weight(h, m) for h, m in times]
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1], (
                f"Weight increased from {weights[i]} at {times[i]} to {weights[i+1]} at {times[i+1]}"
            )
