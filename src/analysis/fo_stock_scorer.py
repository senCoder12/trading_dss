"""
Stock scoring engine for F&O liquidity universe.

Assigns a composite score (0–100) to each liquid stock based on weighted
signals: OI, OI change, volume, price movement, IV change, and spread.
Includes optional sector boost from FII/DII flows and daily ranking/tiering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from config.constants import IST_TIMEZONE
from src.data.fo_bulk_fetcher import StockQuote, StockOIData

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Approximate sector mapping for top F&O stocks
_SECTOR_MAP: dict[str, str] = {
    # Banking
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "AXISBANK": "BANKING",
    "KOTAKBANK": "BANKING", "SBIN": "BANKING", "BANKBARODA": "BANKING",
    "FEDERALBNK": "BANKING", "INDUSINDBK": "BANKING", "PNB": "BANKING",
    # IT
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "LTIM": "IT", "MPHASIS": "IT", "COFORGE": "IT",
    # Energy/Oil
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY",
    "IOC": "ENERGY", "GAIL": "ENERGY", "HINDPETRO": "ENERGY",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    # Auto
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "M&M": "AUTO",
    "BAJAJ-AUTO": "AUTO", "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO",
    # Pharma
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "DIVISLAB": "PHARMA", "APOLLOHOSP": "PHARMA", "AUROPHARMA": "PHARMA",
    # Metals
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "HINDALCO": "METALS",
    "VEDL": "METALS", "COALINDIA": "METALS", "NMDC": "METALS",
    # Infra/Capital Goods
    "LT": "INFRA", "ADANIPORTS": "INFRA", "POWERGRID": "INFRA",
    "NTPC": "INFRA", "SIEMENS": "INFRA",
}


@dataclass
class ScoringWeights:
    """Configurable weights for composite scoring. All must sum to 1.0."""

    oi_weight: float = 0.20
    oi_change_weight: float = 0.25
    volume_weight: float = 0.20
    price_move_weight: float = 0.10
    iv_change_weight: float = 0.15
    spread_tightness_weight: float = 0.10

    def __post_init__(self):
        total = (
            self.oi_weight
            + self.oi_change_weight
            + self.volume_weight
            + self.price_move_weight
            + self.iv_change_weight
            + self.spread_tightness_weight
        )
        if not (0.9999 < total < 1.0001):  # Allow floating point tolerance
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    @staticmethod
    def load_from_config() -> ScoringWeights:
        """Load weights from config/fo_scoring_weights.json or return defaults."""
        config_path = PROJECT_ROOT / "config" / "fo_scoring_weights.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
                    return ScoringWeights(**data)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning(f"Failed to load scoring weights from {config_path}: {e}. Using defaults.")
                return ScoringWeights()
        return ScoringWeights()


@dataclass
class StockScore:
    """Ranked composite score for one stock with per-component transparency."""

    # Identity
    symbol: str
    timestamp: datetime

    # Composite score (0–100)
    composite_score: float

    # Component scores (each 0–100)
    oi_score: float
    oi_change_score: float
    volume_score: float
    price_move_score: float
    iv_change_score: float
    spread_score: float

    # Raw values (for display and debugging)
    total_oi: int
    oi_change: int
    volume_ratio: float
    price_change_pct: float
    iv_change: float
    bid_ask_spread_pct: float

    # Metadata
    rank: int
    tier: str  # 'PRIORITY' (1–15) | 'ACTIVE' (16–50) | 'WATCHLIST' (51+)


class FOStockScorer:
    """
    Multi-signal scoring engine for F&O stock tradability.

    Normalizes each signal independently across the liquid universe,
    weights them, and produces a final 0–100 rank with tier assignment.
    """

    def __init__(self, weights: ScoringWeights | None = None):
        """Initialize with weights (defaults loaded from config)."""
        self.weights = weights or ScoringWeights.load_from_config()
        logger.info(f"Initialized FOStockScorer with weights: {self.weights}")

    def _normalize_series(
        self,
        values: list[float],
        higher_is_better: bool = True,
    ) -> list[float]:
        """
        Min-max normalization to [0.0, 1.0] across universe.

        Args:
            values: Input values (one per stock)
            higher_is_better: If False, invert the scale (e.g., spread)

        Returns:
            Normalized [0.0, 1.0] for each value
        """
        arr = np.array(values, dtype=np.float64)

        # Edge case: all equal → return 0.5 for all
        if np.allclose(arr, arr[0]):
            return [0.5] * len(values)

        min_val = np.min(arr)
        max_val = np.max(arr)

        if min_val == max_val:
            return [0.5] * len(values)

        # Normalize to [0, 1]
        normalized = (arr - min_val) / (max_val - min_val)

        # Invert if lower is better
        if not higher_is_better:
            normalized = 1.0 - normalized

        return normalized.tolist()

    def _score_oi(self, oi_values: list[int]) -> list[float]:
        """Normalize absolute OI with log transformation (heavily skewed)."""
        # Apply log transformation, add 1 to handle zero
        log_oi = [np.log1p(v) for v in oi_values]
        normalized = self._normalize_series(log_oi, higher_is_better=True)
        return [v * 100 for v in normalized]

    def _score_oi_change(self, oi_change_values: list[int]) -> list[float]:
        """Use absolute OI change (buildup or unwinding both tradeable)."""
        # Absolute value: magnitude of activity matters, not direction
        abs_changes = [abs(v) for v in oi_change_values]
        log_changes = [np.log1p(v) for v in abs_changes]
        normalized = self._normalize_series(log_changes, higher_is_better=True)
        return [v * 100 for v in normalized]

    def _score_volume(self, volume_ratios: list[float]) -> list[float]:
        """Score volume ratio (today vs 20-day avg), capped at 5.0 to prevent outlier skew."""
        capped = [min(v, 5.0) for v in volume_ratios]
        normalized = self._normalize_series(capped, higher_is_better=True)
        return [v * 100 for v in normalized]

    def _score_price_move(self, price_change_pcts: list[float]) -> list[float]:
        """Score absolute price change magnitude (direction irrelevant)."""
        abs_changes = [abs(v) for v in price_change_pcts]
        # Cap at 10% to prevent outlier skew
        capped = [min(v, 10.0) for v in abs_changes]
        normalized = self._normalize_series(capped, higher_is_better=True)
        return [v * 100 for v in normalized]

    def _score_iv_change(self, iv_changes: list[float]) -> list[float]:
        """Score IV magnitude of change (both rising and falling are tradeable)."""
        abs_changes = [abs(v) for v in iv_changes]
        normalized = self._normalize_series(abs_changes, higher_is_better=True)
        return [v * 100 for v in normalized]

    def _score_spread(self, spread_pcts: list[float]) -> list[float]:
        """Score spread tightness (lower spread = higher score, so inverted)."""
        normalized = self._normalize_series(spread_pcts, higher_is_better=False)
        return [v * 100 for v in normalized]

    def score_universe(
        self,
        quotes: dict[str, StockQuote],
        oi_data: dict[str, StockOIData],
        liquid_symbols: list[str],
    ) -> list[StockScore]:
        """
        Score all liquid symbols and return ranked list with tier assignment.

        Args:
            quotes: Symbol → StockQuote (price data)
            oi_data: Symbol → StockOIData (options chain data)
            liquid_symbols: Filtered list of tradeable symbols

        Returns:
            Sorted list[StockScore] by composite_score descending
        """
        # Collect raw values for scoring (only liquid universe)
        symbols = []
        oi_values = []
        oi_changes = []
        volume_ratios = []
        price_changes = []
        iv_changes = []
        spreads = []

        for sym in liquid_symbols:
            quote = quotes.get(sym)
            oi = oi_data.get(sym)

            if not quote:
                logger.warning(f"Missing quote for {sym}")
                continue

            symbols.append(sym)
            
            # OI data is optional
            if oi:
                oi_values.append(oi.total_ce_oi + oi.total_pe_oi)
                oi_changes.append(oi.ce_oi_change + oi.pe_oi_change)
                iv_changes.append(oi.iv_change)
            else:
                oi_values.append(0)
                oi_changes.append(0)
                iv_changes.append(0)

            volume_ratios.append(quote.volume / quote.avg_volume_20d if quote.avg_volume_20d > 0 else 0)
            price_changes.append(quote.price_change_pct)
            spreads.append(quote.bid_ask_spread_pct)

        if not symbols:
            logger.warning("No liquid stocks found for scoring")
            return []

        # Score each component
        oi_scores = self._score_oi(oi_values)
        oi_change_scores = self._score_oi_change(oi_changes)
        volume_scores = self._score_volume(volume_ratios)
        price_move_scores = self._score_price_move(price_changes)
        iv_change_scores = self._score_iv_change(iv_changes)
        spread_scores = self._score_spread(spreads)

        # Compute composite scores and build StockScore objects
        stock_scores = []
        for i, sym in enumerate(symbols):
            composite = (
                self.weights.oi_weight * oi_scores[i]
                + self.weights.oi_change_weight * oi_change_scores[i]
                + self.weights.volume_weight * volume_scores[i]
                + self.weights.price_move_weight * price_move_scores[i]
                + self.weights.iv_change_weight * iv_change_scores[i]
                + self.weights.spread_tightness_weight * spread_scores[i]
            )

            score_obj = StockScore(
                symbol=sym,
                timestamp=datetime.now(_IST),
                composite_score=composite,
                oi_score=oi_scores[i],
                oi_change_score=oi_change_scores[i],
                volume_score=volume_scores[i],
                price_move_score=price_move_scores[i],
                iv_change_score=iv_change_scores[i],
                spread_score=spread_scores[i],
                total_oi=oi_values[i],
                oi_change=oi_changes[i],
                volume_ratio=volume_ratios[i],
                price_change_pct=price_changes[i],
                iv_change=iv_changes[i],
                bid_ask_spread_pct=spreads[i],
                rank=0,  # Will be set after sorting
                tier="",  # Will be set after ranking
            )
            stock_scores.append(score_obj)

        # Sort descending by composite score
        stock_scores.sort(key=lambda x: x.composite_score, reverse=True)

        # Assign rank and tier
        for rank, score in enumerate(stock_scores, start=1):
            score.rank = rank
            if rank <= 15:
                score.tier = "PRIORITY"
            elif rank <= 50:
                score.tier = "ACTIVE"
            else:
                score.tier = "WATCHLIST"

        return stock_scores

    def get_top_n(self, scores: list[StockScore], n: int) -> list[StockScore]:
        """Return top N stocks by composite score."""
        return scores[:n]

    def apply_sector_boost(
        self,
        scores: list[StockScore],
        fii_sector_data: dict[str, float],
    ) -> list[StockScore]:
        """
        Apply sector boost from FII/DII activity (optional enhancement).

        Groups stocks by sector, computes per-sector average composite score as
        a momentum proxy, boosts top sectors by 1.10x and penalises bottom
        sectors by 0.92x. Falls back gracefully when sector data is unavailable.

        Args:
            scores: Scored universe
            fii_sector_data: Sector → FII score (0–1, higher = buying pressure)

        Returns:
            Modified scores with sector boost applied (same ordering)
        """
        if not scores:
            return scores

        try:
            # Group composite scores by sector
            sector_scores: dict[str, list[float]] = {}
            for s in scores:
                sector = _SECTOR_MAP.get(s.symbol)
                if sector:
                    sector_scores.setdefault(sector, []).append(s.composite_score)

            if not sector_scores:
                # No mapped symbols — return unchanged
                logger.debug("Sector boost: no mapped symbols, skipping")
                return scores

            # Per-sector average composite score (momentum proxy)
            sector_avg: dict[str, float] = {
                sec: sum(vals) / len(vals)
                for sec, vals in sector_scores.items()
            }

            overall_avg = sum(sector_avg.values()) / len(sector_avg)

            # Top 2 sectors (above-average composite score)
            ranked_sectors = sorted(sector_avg, key=sector_avg.__getitem__, reverse=True)
            top_sectors = set(ranked_sectors[:2])
            bottom_sectors = set(ranked_sectors[-2:]) if len(ranked_sectors) > 2 else set()

            # Apply multipliers
            boosted: list[StockScore] = []
            for s in scores:
                sector = _SECTOR_MAP.get(s.symbol)
                if sector in top_sectors:
                    multiplier = 1.10
                elif sector in bottom_sectors:
                    multiplier = 0.92
                else:
                    multiplier = 1.0

                if multiplier != 1.0:
                    s.composite_score = round(
                        min(s.composite_score * multiplier, 100.0), 4
                    )

                boosted.append(s)

            logger.info(
                "Sector boost applied: top=%s (1.10x), bottom=%s (0.92x)",
                top_sectors,
                bottom_sectors,
            )
            return boosted

        except Exception as e:
            logger.warning("Sector boost failed, returning original scores: %s", e)
            return scores

    def save_scores_to_db(
        self,
        scores: list[StockScore],
        db_manager,
    ) -> bool:
        """
        Persist scores to fo_daily_scores table.

        Args:
            scores: List of computed StockScore objects
            db_manager: Database manager with execute method

        Returns:
            True if successful
        """
        if not scores:
            logger.warning("No scores to persist")
            return False

        today = date.today().isoformat()
        insert_count = 0

        for score in scores:
            sql = """
            INSERT INTO fo_daily_scores
                (date, symbol, rank, tier, composite_score,
                 oi_score, oi_change_score, volume_score,
                 price_move_score, iv_change_score, spread_score,
                 total_oi, volume_ratio, timestamp)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, symbol) DO UPDATE SET
                rank = excluded.rank,
                tier = excluded.tier,
                composite_score = excluded.composite_score,
                oi_score = excluded.oi_score,
                oi_change_score = excluded.oi_change_score,
                volume_score = excluded.volume_score,
                price_move_score = excluded.price_move_score,
                iv_change_score = excluded.iv_change_score,
                spread_score = excluded.spread_score,
                total_oi = excluded.total_oi,
                volume_ratio = excluded.volume_ratio,
                timestamp = excluded.timestamp
            """

            try:
                db_manager.execute(
                    sql,
                    (
                        today,
                        score.symbol,
                        score.rank,
                        score.tier,
                        score.composite_score,
                        score.oi_score,
                        score.oi_change_score,
                        score.volume_score,
                        score.price_move_score,
                        score.iv_change_score,
                        score.spread_score,
                        score.total_oi,
                        score.volume_ratio,
                        score.timestamp.isoformat(),
                    ),
                )
                insert_count += 1
            except Exception as e:
                logger.error(f"Failed to insert score for {score.symbol}: {e}")

        logger.info(f"Persisted {insert_count}/{len(scores)} scores to fo_daily_scores")
        return insert_count == len(scores)
