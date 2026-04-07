"""
Index listing and info API endpoints.

GET /api/v1/indices/               — list all active indices
GET /api/v1/indices/{index_id}     — get metadata for a single index
GET /api/v1/indices/filter         — filter by exchange, category, has_options
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_index_registry
from src.data.index_registry import Index, IndexRegistry

router = APIRouter()


def _to_dict(defn: Index) -> dict:
    return {
        "id": defn.id,
        "display_name": defn.display_name,
        "nse_symbol": defn.nse_symbol,
        "yahoo_symbol": defn.yahoo_symbol,
        "exchange": defn.exchange,
        "lot_size": defn.lot_size,
        "has_options": defn.has_options,
        "option_symbol": defn.option_symbol,
        "sector_category": defn.sector_category,
        "is_active": defn.is_active,
        "description": defn.description,
    }


@router.get("/", summary="List all active indices")
async def list_indices(
    registry: IndexRegistry = Depends(get_index_registry),
) -> list[dict]:
    """Return metadata for all active indices."""
    return [_to_dict(i) for i in registry.get_all_indices(active_only=True)]


@router.get("/{index_id}", summary="Get a single index by ID")
async def get_index(
    index_id: str,
    registry: IndexRegistry = Depends(get_index_registry),
) -> dict:
    """Return metadata for *index_id*."""
    defn = registry.get_or_none(index_id.upper())
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_id!r}")
    return _to_dict(defn)


@router.get("/filter/search", summary="Filter indices by criteria")
async def filter_indices(
    exchange: Optional[str] = Query(default=None, description="NSE or BSE"),
    has_options: Optional[bool] = Query(default=None),
    sector_category: Optional[str] = Query(default=None),
    registry: IndexRegistry = Depends(get_index_registry),
) -> list[dict]:
    """
    Filter the index list by exchange, options availability, and sector.
    """
    results = registry.filter(
        exchange=exchange.upper() if exchange else None,
        has_options=has_options,
        sector_category=sector_category,
        active_only=True,
    )
    return [_to_dict(i) for i in results]
