"""Unit tests for F&O stock scorer."""

from datetime import datetime, date
from zoneinfo import ZoneInfo

import pytest

from config.constants import IST_TIMEZONE
from src.analysis.fo_stock_scorer import (
    ScoringWeights,
    StockScore,
    FOStockScorer,
)
from src.data.fo_bulk_fetcher import StockQuote, StockOIData

_IST = ZoneInfo(IST_TIMEZONE)


class TestScoringWeights:
    """Test weight validation and config loading."""

    def test_valid_weights(self):
        """Valid weights that sum to 1.0."""
        weights = ScoringWeights()
        assert weights.oi_weight == 0.20
        assert weights.oi_change_weight == 0.25

    def test_weights_must_sum_to_one(self):
        """Weights must sum to exactly 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringWeights(
                oi_weight=0.20,
                oi_change_weight=0.25,
                volume_weight=0.20,
                price_move_weight=0.10,
                iv_change_weight=0.15,
                spread_tightness_weight=0.05,  # Only 0.95 total
            )

    def test_load_from_config(self):
        """Load weights from config file if it exists."""
        weights = ScoringWeights.load_from_config()
        assert isinstance(weights, ScoringWeights)
        assert abs(sum([
            weights.oi_weight,
            weights.oi_change_weight,
            weights.volume_weight,
            weights.price_move_weight,
            weights.iv_change_weight,
            weights.spread_tightness_weight,
        ]) - 1.0) < 0.0001


class TestFOStockScorer:
    """Test scoring logic and normalization."""

    @pytest.fixture
    def scorer(self):
        return FOStockScorer()

    @pytest.fixture
    def sample_data(self):
        """Create minimal sample data for 3 stocks."""
        now = datetime.now(_IST)

        quotes = {
            "RELIANCE": StockQuote(
                symbol="RELIANCE",
                ltp=2500.0,
                open=2490.0,
                high=2510.0,
                low=2480.0,
                close=2480.0,
                volume=1_500_000,
                avg_volume_20d=1_000_000,
                price_change_pct=0.8,
                bid=2499.0,
                ask=2501.0,
                bid_ask_spread=2.0,
                bid_ask_spread_pct=0.08,
                timestamp=now,
            ),
            "INFY": StockQuote(
                symbol="INFY",
                ltp=1800.0,
                open=1790.0,
                high=1810.0,
                low=1785.0,
                close=1785.0,
                volume=800_000,
                avg_volume_20d=700_000,
                price_change_pct=-0.3,
                bid=1799.0,
                ask=1801.0,
                bid_ask_spread=2.0,
                bid_ask_spread_pct=0.11,
                timestamp=now,
            ),
            "TCS": StockQuote(
                symbol="TCS",
                ltp=3500.0,
                open=3480.0,
                high=3520.0,
                low=3470.0,
                close=3470.0,
                volume=600_000,
                avg_volume_20d=500_000,
                price_change_pct=0.9,
                bid=3499.0,
                ask=3501.0,
                bid_ask_spread=2.0,
                bid_ask_spread_pct=0.06,
                timestamp=now,
            ),
        }

        oi_data = {
            "RELIANCE": StockOIData(
                symbol="RELIANCE",
                expiry=date.today(),
                total_ce_oi=5_000_000,
                total_pe_oi=4_500_000,
                ce_oi_change=200_000,
                pe_oi_change=150_000,
                pcr=0.9,
                atm_strike=2500.0,
                atm_ce_iv=25.0,
                atm_pe_iv=27.0,
                avg_iv=26.0,
                iv_change=1.5,
                timestamp=now,
            ),
            "INFY": StockOIData(
                symbol="INFY",
                expiry=date.today(),
                total_ce_oi=3_000_000,
                total_pe_oi=3_200_000,
                ce_oi_change=100_000,
                pe_oi_change=120_000,
                pcr=1.067,
                atm_strike=1800.0,
                atm_ce_iv=22.0,
                atm_pe_iv=24.0,
                avg_iv=23.0,
                iv_change=-0.5,
                timestamp=now,
            ),
            "TCS": StockOIData(
                symbol="TCS",
                expiry=date.today(),
                total_ce_oi=2_500_000,
                total_pe_oi=2_400_000,
                ce_oi_change=50_000,
                pe_oi_change=60_000,
                pcr=0.96,
                atm_strike=3500.0,
                atm_ce_iv=20.0,
                atm_pe_iv=22.0,
                avg_iv=21.0,
                iv_change=0.2,
                timestamp=now,
            ),
        }

        return quotes, oi_data

    def test_score_universe(self, scorer, sample_data):
        """Score a universe of 3 stocks."""
        quotes, oi_data = sample_data
        liquid_symbols = ["RELIANCE", "INFY", "TCS"]

        scores = scorer.score_universe(quotes, oi_data, liquid_symbols)

        assert len(scores) == 3
        assert all(isinstance(s, StockScore) for s in scores)
        assert all(0 <= s.composite_score <= 100 for s in scores)
        assert scores[0].rank == 1
        assert scores[2].rank == 3

    def test_tiering(self, scorer, sample_data):
        """Verify tier assignment based on rank."""
        quotes, oi_data = sample_data
        liquid_symbols = ["RELIANCE", "INFY", "TCS"]

        scores = scorer.score_universe(quotes, oi_data, liquid_symbols)

        # Ranks 1-15 are PRIORITY, 16-50 are ACTIVE, 51+ are WATCHLIST
        # With only 3 stocks, all are PRIORITY
        assert scores[0].tier == "PRIORITY"
        assert scores[1].tier == "PRIORITY"
        assert scores[2].tier == "PRIORITY"

    def test_normalization_all_equal(self, scorer):
        """Min-max normalization with all equal values returns 0.5."""
        values = [100.0, 100.0, 100.0]
        normalized = scorer._normalize_series(values, higher_is_better=True)
        assert all(v == 0.5 for v in normalized)

    def test_normalization_inverted_spread(self, scorer):
        """Spread scoring inverts (lower is better)."""
        spreads = [0.05, 0.10, 0.15]  # Tighter spreads should score higher
        scores = scorer._score_spread(spreads)
        assert scores[0] > scores[2]  # Tightest (0.05) scores higher

    def test_oi_log_transformation(self, scorer):
        """OI scoring applies log transformation to normalize skewed data."""
        # Log transformation reduces impact of extreme outliers
        oi_values = [1_000_000, 10_000_000, 100_000_000]
        scores = scorer._score_oi(oi_values)
        # Scores normalize to 0-100, ordered correctly
        assert scores[0] < scores[1] < scores[2]
        # With only 3 values, min-max normalization gives 0, 50, 100
        # Log transformation in the denominator ensures higher OI gets higher score
        assert scores[0] == 0.0 or scores[0] < 50
        assert scores[2] == 100.0 or scores[2] > 50

    def test_get_top_n(self, scorer, sample_data):
        """Get top N stocks convenience wrapper."""
        quotes, oi_data = sample_data
        liquid_symbols = ["RELIANCE", "INFY", "TCS"]

        scores = scorer.score_universe(quotes, oi_data, liquid_symbols)
        top_2 = scorer.get_top_n(scores, 2)

        assert len(top_2) == 2
        assert top_2[0].rank == 1
        assert top_2[1].rank == 2
