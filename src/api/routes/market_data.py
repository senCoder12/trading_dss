"""
Market data API endpoints.

GET /api/v1/market/quote/{index_id}         — live quote
GET /api/v1/market/historical/{index_id}    — OHLCV history
GET /api/v1/market/vix                      — India VIX
GET /api/v1/market/fii-dii                  — FII/DII activity
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_index_registry
from src.data.index_registry import IndexRegistry
from src.data.nse_scraper import NSEScraper, NSEScraperError

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/quote/{index_id}", summary="Live quote for an index")
async def get_quote(
    index_id: str,
    registry: IndexRegistry = Depends(get_index_registry),
) -> dict:
    """
    Return the latest live quote for *index_id*.

    Only works for NSE indices during market hours.
    """
    defn = registry.get_or_none(index_id.upper())
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_id!r}")
    if defn.exchange != "NSE" or defn.nse_symbol is None:
        raise HTTPException(
            status_code=400,
            detail=f"{index_id} is not an NSE index — live quotes only available for NSE",
        )

    try:
        with NSEScraper() as scraper:
            quote = scraper.get_index_quote(defn.nse_symbol)
            if quote is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Quote not found in NSE response for {index_id}",
                )
            return {
                "index_id": index_id,
                "display_name": defn.display_name,
                "last": quote.get("last") or quote.get("ltp"),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "previous_close": quote.get("previousClose") or quote.get("close"),
                "change": quote.get("change"),
                "pct_change": quote.get("change_pct"),
                "advances": quote.get("advances"),
                "declines": quote.get("declines"),
            }
    except NSEScraperError as exc:
        logger.error("NSE scraper error for %s: %s", index_id, exc)
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/vix", summary="India VIX current reading")
async def get_vix() -> dict:
    """Return the latest India VIX value and regime."""
    from src.data.vix_data import VIXTracker
    try:
        with NSEScraper() as scraper:
            tracker = VIXTracker(scraper)
            vix_data = tracker.get_current_vix()
            if vix_data is None:
                raise HTTPException(status_code=503, detail="VIX data unavailable")
            regime = tracker.get_vix_regime()
            return {
                "value": vix_data.value,
                "change": vix_data.change,
                "change_pct": vix_data.change_pct,
                "regime": regime,
            }
    except NSEScraperError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/fii-dii", summary="Latest FII/DII activity")
async def get_fii_dii() -> dict:
    """Return the most recent FII and DII trading activity."""
    from src.data.fii_dii_data import FIIDIIFetcher
    try:
        with NSEScraper() as scraper:
            fetcher = FIIDIIFetcher(scraper)
            activity = fetcher.fetch_today_fii_dii()
            if activity is None:
                raise HTTPException(status_code=503, detail="FII/DII data unavailable")
            return {
                "trade_date": str(activity.date),
                "fii_buy": activity.fii_buy_value,
                "fii_sell": activity.fii_sell_value,
                "fii_net": activity.fii_net_value,
                "dii_buy": activity.dii_buy_value,
                "dii_sell": activity.dii_sell_value,
                "dii_net": activity.dii_net_value,
                "total_net": activity.fii_net_value + activity.dii_net_value,
            }
    except NSEScraperError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
