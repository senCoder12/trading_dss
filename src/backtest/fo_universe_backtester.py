"""
F&O universe scoring weight optimizer.

Tests different weight combinations against historical data to find
which weighting produces the most useful universe (highest correlation
between top-N selected stocks and actual daily movers).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

from config.constants import IST_TIMEZONE
from src.analysis.fo_stock_scorer import ScoringWeights
from src.database.db_manager import DatabaseManager
from src.database import queries as Q

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class BacktestConfig:
    """Configuration for a universe backtest run."""

    start_date: date
    end_date: date
    top_n: int = 50                    # Universe size to evaluate
    priority_n: int = 15               # Priority list size
    metric: str = 'hit_rate'           # Optimization metric: 'hit_rate' | 'avg_move' | 'sharpe'


@dataclass
class UniverseBacktestResult:
    """Result of backtesting one weight combination."""

    weights_tested: ScoringWeights
    hit_rate: float                    # % of days where top-mover was in top_n
    avg_daily_move: float              # Avg price move of selected stocks
    priority_hit_rate: float           # % of days where top-mover was in top_priority_n
    best_weights: Optional[ScoringWeights] = None
    daily_results: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary of result."""
        return (
            f"hit_rate={self.hit_rate:.1%} priority={self.priority_hit_rate:.1%} "
            f"avg_move={self.avg_daily_move:.2%}"
        )


class FOUniverseBacktester:
    """
    Test weight combinations to find optimal F&O universe scoring.

    Loads historical daily scores from DB, simulates scoring with
    different weights, compares selected universe to actual movers.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self._historical_scores: dict[str, list[dict]] = {}

    def load_historical_data(self, start_date: date, end_date: date) -> bool:
        """
        Load all daily scores for date range.

        Parameters
        ----------
        start_date, end_date:
            Date range to load (inclusive).

        Returns
        -------
        bool:
            True if data loaded successfully, False if no data found.
        """
        logger.info(f"Loading historical scores: {start_date} to {end_date}")

        query = """
        SELECT
            date, symbol, rank, tier, composite_score,
            oi_score, oi_change_score, volume_score,
            price_move_score, iv_change_score, spread_score,
            total_oi, volume_ratio
        FROM fo_daily_scores
        WHERE date BETWEEN ? AND ?
        ORDER BY date ASC, rank ASC
        """

        rows = self.db.fetch_all(query, (start_date.isoformat(), end_date.isoformat()))

        if not rows:
            logger.warning(f"No historical scores found for {start_date} to {end_date}")
            return False

        # Group by date. Need to fetch price_change_pct separately from price_data
        self._historical_scores.clear()
        for row in rows:
            date_str = row['date']
            if date_str not in self._historical_scores:
                self._historical_scores[date_str] = []

            self._historical_scores[date_str].append({
                'symbol': row['symbol'],
                'rank': row['rank'],
                'tier': row['tier'],
                'composite_score': row['composite_score'],
                'oi_score': row['oi_score'],
                'oi_change_score': row['oi_change_score'],
                'volume_score': row['volume_score'],
                'price_move_score': row['price_move_score'],
                'iv_change_score': row['iv_change_score'],
                'spread_score': row['spread_score'],
                'total_oi': row['total_oi'],
                'volume_ratio': row['volume_ratio'],
            })

        logger.info(f"Loaded scores for {len(self._historical_scores)} days")
        return True

    def simulate_day(
        self,
        date_str: str,
        weights: ScoringWeights,
        top_n: int,
        priority_n: int,
    ) -> dict:
        """
        Simulate one day with given weights.

        Recomputes composite scores using provided weights, gets top_n,
        then checks if actual top price movers were included.

        Parameters
        ----------
        date_str:
            Date in 'YYYY-MM-DD' format.
        weights:
            Scoring weights to test.
        top_n:
            Universe size.
        priority_n:
            Priority list size.

        Returns
        -------
        dict:
            {
                'date': str,
                'top_n_selected': list[str] (symbols in top_n),
                'priority_selected': list[str],
                'top_movers': list[str] (by price_move_score),
                'mover_in_top_n': bool,
                'mover_in_priority': bool,
                'hit_count': int,
                'priority_hit_count': int,
                'avg_move_selected': float,
            }
        """
        if date_str not in self._historical_scores:
            return None

        scores = self._historical_scores[date_str]

        if not scores:
            return None

        # Recompute composite scores with new weights
        for score in scores:
            score['reweighted'] = (
                weights.oi_weight * score['oi_score']
                + weights.oi_change_weight * score['oi_change_score']
                + weights.volume_weight * score['volume_score']
                + weights.price_move_weight * score['price_move_score']
                + weights.iv_change_weight * score['iv_change_score']
                + weights.spread_tightness_weight * score['spread_score']
            )

        # Sort by reweighted score
        sorted_scores = sorted(scores, key=lambda x: x['reweighted'], reverse=True)

        # Get top_n and priority_n
        top_n_selected = [s['symbol'] for s in sorted_scores[:top_n]]
        priority_selected = [s['symbol'] for s in sorted_scores[:priority_n]]

        # Find actual top movers by price_move_score (normalized 0-100)
        sorted_movers = sorted(scores, key=lambda x: x['price_move_score'], reverse=True)
        top_movers = [s['symbol'] for s in sorted_movers[:10]]  # Top 10 movers

        # Check hits
        mover_in_top_n = len(set(top_movers) & set(top_n_selected)) > 0
        mover_in_priority = len(set(top_movers) & set(priority_selected)) > 0
        hit_count = len(set(top_movers) & set(top_n_selected))
        priority_hit_count = len(set(top_movers) & set(priority_selected))

        # Avg price move score of selected stocks
        selected_moves = [s['price_move_score'] for s in sorted_scores[:top_n]]
        avg_move = sum(selected_moves) / len(selected_moves) / 100.0 if selected_moves else 0

        return {
            'date': date_str,
            'top_n_selected': top_n_selected,
            'priority_selected': priority_selected,
            'top_movers': top_movers,
            'mover_in_top_n': mover_in_top_n,
            'mover_in_priority': mover_in_priority,
            'hit_count': hit_count,
            'priority_hit_count': priority_hit_count,
            'avg_move_selected': avg_move,
        }

    def run_grid_search(
        self,
        weight_combinations: Optional[list[ScoringWeights]] = None,
        top_n: int = 50,
        priority_n: int = 15,
    ) -> list[UniverseBacktestResult]:
        """
        Test multiple weight combinations and rank by performance.

        Parameters
        ----------
        weight_combinations:
            List of ScoringWeights to test. If None, generates 27 default
            combinations (3 levels × 6 weights).
        top_n:
            Universe size.
        priority_n:
            Priority list size.

        Returns
        -------
        list[UniverseBacktestResult]:
            Results sorted by hit_rate descending.
        """
        if not self._historical_scores:
            raise ValueError("No historical data loaded. Call load_historical_data() first.")

        if weight_combinations is None:
            weight_combinations = self._generate_default_combinations()

        logger.info(f"Testing {len(weight_combinations)} weight combinations")

        results = []

        for i, weights in enumerate(weight_combinations):
            daily_results = []
            hit_count = 0
            priority_hit_count = 0
            total_move = 0.0
            valid_days = 0

            for date_str in sorted(self._historical_scores.keys()):
                day_result = self.simulate_day(date_str, weights, top_n, priority_n)
                if day_result:
                    daily_results.append(day_result)
                    if day_result['mover_in_top_n']:
                        hit_count += 1
                    if day_result['mover_in_priority']:
                        priority_hit_count += 1
                    total_move += day_result['avg_move_selected']
                    valid_days += 1

            if valid_days > 0:
                hit_rate = hit_count / valid_days
                priority_hit_rate = priority_hit_count / valid_days
                avg_daily_move = total_move / valid_days
            else:
                hit_rate = 0.0
                priority_hit_rate = 0.0
                avg_daily_move = 0.0

            result = UniverseBacktestResult(
                weights_tested=weights,
                hit_rate=hit_rate,
                avg_daily_move=avg_daily_move,
                priority_hit_rate=priority_hit_rate,
                daily_results=daily_results,
            )
            results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"Tested {i + 1}/{len(weight_combinations)}: {result.summary()}")

        # Sort by hit_rate descending
        results.sort(key=lambda x: x.hit_rate, reverse=True)

        logger.info(f"Grid search complete. Best: {results[0].summary()}")

        return results

    def _generate_default_combinations(self) -> list[ScoringWeights]:
        """
        Generate 27 default weight combinations.

        Tests each of 6 weights at 3 levels (low, medium, high emphasis).
        For each emphasized weight, redistribute remaining weight equally.
        """
        combinations = []
        weight_names = [
            'oi_weight',
            'oi_change_weight',
            'volume_weight',
            'price_move_weight',
            'iv_change_weight',
            'spread_tightness_weight',
        ]

        # Test each weight at 3 emphasis levels
        emphasis_levels = [0.10, 0.20, 0.35]

        for weight_name in weight_names:
            for emphasis in emphasis_levels:
                # Distribute emphasis to target, rest equally to others
                other_weight = (1.0 - emphasis) / 5

                weights_dict = {name: other_weight for name in weight_names}
                weights_dict[weight_name] = emphasis

                try:
                    w = ScoringWeights(**weights_dict)
                    combinations.append(w)
                except ValueError as e:
                    logger.debug(f"Skipped invalid combination: {e}")

        logger.info(f"Generated {len(combinations)} default weight combinations")
        return combinations

    def generate_report(
        self,
        results: list[UniverseBacktestResult],
        output_dir: Optional[Path] = None,
    ) -> str:
        """
        Generate markdown report of backtest results.

        Parameters
        ----------
        results:
            Sorted list of results from run_grid_search.
        output_dir:
            Directory to save report. If None, uses reports/.

        Returns
        -------
        str:
            Path to generated report file.
        """
        if output_dir is None:
            output_dir = PROJECT_ROOT / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"fo_universe_backtest_{timestamp}.md"

        best = results[0] if results else None

        lines = [
            "# F&O Universe Scoring Weight Optimization",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Executive Summary",
            "",
        ]

        if best:
            lines.extend([
                f"**Best Weight Combination:**",
                "",
                f"- Hit Rate: {best.hit_rate:.1%}",
                f"- Priority Hit Rate: {best.priority_hit_rate:.1%}",
                f"- Avg Daily Move (Selected): {best.avg_daily_move:.2%}",
                f"- Days Tested: {len(best.daily_results)}",
                "",
                "**Weights:**",
                "",
                f"- OI Weight: {best.weights_tested.oi_weight:.2f}",
                f"- OI Change Weight: {best.weights_tested.oi_change_weight:.2f}",
                f"- Volume Weight: {best.weights_tested.volume_weight:.2f}",
                f"- Price Move Weight: {best.weights_tested.price_move_weight:.2f}",
                f"- IV Change Weight: {best.weights_tested.iv_change_weight:.2f}",
                f"- Spread Tightness Weight: {best.weights_tested.spread_tightness_weight:.2f}",
                "",
            ])

        lines.extend([
            "## Top 10 Results",
            "",
            "| Rank | Hit Rate | Priority HR | Avg Move | OI | OI Chg | Vol | Price | IV | Spread |",
            "|------|----------|-------------|----------|----|----|-----|-------|-----|--------|",
        ])

        for i, result in enumerate(results[:10], 1):
            w = result.weights_tested
            lines.append(
                f"| {i} | {result.hit_rate:.1%} | {result.priority_hit_rate:.1%} | "
                f"{result.avg_daily_move:.2%} | {w.oi_weight:.2f} | {w.oi_change_weight:.2f} | "
                f"{w.volume_weight:.2f} | {w.price_move_weight:.2f} | {w.iv_change_weight:.2f} | "
                f"{w.spread_tightness_weight:.2f} |"
            )

        lines.extend([
            "",
            "## Recommendation",
            "",
            "Use the best weight combination above. If hit rates are all below 40%, "
            "consider:",
            "",
            "1. Expanding the backtest window (more data = more signal)",
            "2. Adjusting top_n or priority_n thresholds",
            "3. Examining daily_results for systematic patterns in misses",
            "",
        ])

        report_text = "\n".join(lines)

        with open(report_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Report saved to {report_path}")

        return str(report_path)


def main():
    """CLI entry point (used by scripts/tune_fo_weights.py)."""
    import argparse
    from datetime import timedelta

    parser = argparse.ArgumentParser(description="Optimize F&O scoring weights")
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of recent days to backtest (default 60)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for weights JSON (default config/fo_scoring_weights.json)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output directory for markdown report",
    )

    args = parser.parse_args()

    db = DatabaseManager(":memory:")  # In real usage, this should be the actual DB
    backtester = FOUniverseBacktester(db)

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    if not backtester.load_historical_data(start_date, end_date):
        logger.error("Failed to load historical data")
        return

    results = backtester.run_grid_search(top_n=50, priority_n=15)

    if results:
        best = results[0]
        logger.info(f"Best weights: {best.summary()}")

        output_path = args.output or (PROJECT_ROOT / "config" / "fo_scoring_weights.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        weights_dict = {
            'oi_weight': best.weights_tested.oi_weight,
            'oi_change_weight': best.weights_tested.oi_change_weight,
            'volume_weight': best.weights_tested.volume_weight,
            'price_move_weight': best.weights_tested.price_move_weight,
            'iv_change_weight': best.weights_tested.iv_change_weight,
            'spread_tightness_weight': best.weights_tested.spread_tightness_weight,
        }

        with open(output_path, 'w') as f:
            json.dump(weights_dict, f, indent=2)

        logger.info(f"Weights saved to {output_path}")

        if args.report:
            backtester.generate_report(results, args.report)
