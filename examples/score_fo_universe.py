#!/usr/bin/env python
"""
Example: Score the F&O liquid universe and rank stocks by tradability.

Usage:
    python examples/score_fo_universe.py

This script demonstrates the full pipeline:
1. Fetch F&O universe and volume data
2. Get live quotes and options data
3. Filter to liquid stocks only
4. Score and rank by tradability
5. Display top 10 by tier
"""

import asyncio
import logging
from datetime import datetime

from config.constants import IST_TIMEZONE
from src.data.fo_bulk_fetcher import FOBulkFetcher
from src.analysis.fo_liquidity_filter import FOLiquidityFilter
from src.analysis.fo_stock_scorer import FOStockScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    fetcher = FOBulkFetcher()
    filter_ = FOLiquidityFilter()
    scorer = FOStockScorer()

    logger.info("Fetching F&O universe and quotes...")
    try:
        quotes, oi_data = await fetcher.fetch_universe()
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    # Filter to liquid stocks
    logger.info(f"Screening {len(quotes)} stocks for liquidity...")
    liquid_symbols = filter_.filter_stocks(quotes, oi_data)
    logger.info(f"✓ Liquid universe: {len(liquid_symbols)} stocks")

    # Score the liquid universe
    logger.info("Scoring tradability...")
    scores = scorer.score_universe(quotes, oi_data, liquid_symbols)
    logger.info(f"✓ Scored {len(scores)} stocks")

    # Display results by tier
    priority = [s for s in scores if s.tier == "PRIORITY"]
    active = [s for s in scores if s.tier == "ACTIVE"]
    watchlist = [s for s in scores if s.tier == "WATCHLIST"]

    print("\n" + "=" * 100)
    print(f"F&O STOCK RANKING — {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    print("=" * 100)

    # Top 10 PRIORITY stocks
    print(f"\n🔥 PRIORITY ({len(priority)} stocks) — Rank 1–15")
    print("-" * 100)
    for score in priority[:10]:
        print(
            f"  {score.rank:2d}. {score.symbol:10s}  "
            f"Score: {score.composite_score:6.1f}  "
            f"OI: {score.oi_score:5.1f}  "
            f"OI Δ: {score.oi_change_score:5.1f}  "
            f"Vol: {score.volume_score:5.1f}  "
            f"Spread: {score.spread_score:5.1f}"
        )

    # Top 10 ACTIVE stocks
    print(f"\n🔔 ACTIVE ({len(active)} stocks) — Rank 16–50")
    print("-" * 100)
    for score in active[:10]:
        print(
            f"  {score.rank:2d}. {score.symbol:10s}  "
            f"Score: {score.composite_score:6.1f}  "
            f"OI: {score.oi_score:5.1f}  "
            f"OI Δ: {score.oi_change_score:5.1f}  "
            f"Vol: {score.volume_score:5.1f}  "
            f"Spread: {score.spread_score:5.1f}"
        )

    # Raw component scores for #1
    if scores:
        s = scores[0]
        print(f"\n📊 Component Breakdown — #{s.symbol}")
        print("-" * 100)
        print(
            f"  Composite Score:  {s.composite_score:6.1f} / 100\n"
            f"  OI Level          {s.oi_score:6.1f} / 100  (weight: 20%)\n"
            f"  OI Change         {s.oi_change_score:6.1f} / 100  (weight: 25%)\n"
            f"  Volume            {s.volume_score:6.1f} / 100  (weight: 20%)\n"
            f"  Price Move        {s.price_move_score:6.1f} / 100  (weight: 10%)\n"
            f"  IV Change         {s.iv_change_score:6.1f} / 100  (weight: 15%)\n"
            f"  Spread Tightness  {s.spread_score:6.1f} / 100  (weight: 10%)\n"
        )
        print(f"  Raw Data for {s.symbol}:")
        print(f"    Total OI:              {s.total_oi:,}")
        print(f"    OI Change:             {s.oi_change:,}")
        print(f"    Volume Ratio (vs avg): {s.volume_ratio:.2f}x")
        print(f"    Price Change:          {s.price_change_pct:+.2f}%")
        print(f"    IV Change:             {s.iv_change:+.2f}pp")
        print(f"    Bid-Ask Spread:        {s.bid_ask_spread_pct:.3f}%")


if __name__ == "__main__":
    asyncio.run(main())
