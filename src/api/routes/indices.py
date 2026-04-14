"""
Index listing and info API endpoints.

GET /api/indices/                       — list all indices with metadata
GET /api/indices/{index_id}             — full detail for one index
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_db, get_index_registry
from src.data.index_registry import Index, IndexRegistry
from src.database import queries as Q
from src.database.db_manager import DatabaseManager

router = APIRouter()


# ── Response models ──────────────────────────────────────────────────────────

class IndexInfo(BaseModel):
    id: str
    display_name: str
    nse_symbol: Optional[str] = None
    yahoo_symbol: Optional[str] = None
    exchange: str
    lot_size: Optional[int] = None
    has_options: bool = False
    option_symbol: Optional[str] = None
    sector_category: Optional[str] = None
    is_active: bool = True
    description: Optional[str] = None


class IndexDetail(IndexInfo):
    ltp: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    latest_signal: Optional[str] = None
    signal_confidence: Optional[str] = None
    regime: Optional[str] = None


class IndicesListResponse(BaseModel):
    indices: list[IndexInfo]
    count: int


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_info(defn: Index) -> IndexInfo:
    return IndexInfo(
        id=defn.id,
        display_name=defn.display_name,
        nse_symbol=defn.nse_symbol,
        yahoo_symbol=defn.yahoo_symbol,
        exchange=defn.exchange,
        lot_size=defn.lot_size,
        has_options=defn.has_options,
        option_symbol=defn.option_symbol,
        sector_category=defn.sector_category,
        is_active=defn.is_active,
        description=defn.description,
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/", response_model=IndicesListResponse, summary="List all indices with metadata")
async def list_indices(
    exchange: Optional[str] = Query(default=None, description="NSE or BSE"),
    has_options: Optional[bool] = Query(default=None),
    sector: Optional[str] = Query(default=None, alias="sector_category"),
    registry: IndexRegistry = Depends(get_index_registry),
):
    results = registry.filter(
        exchange=exchange.upper() if exchange else None,
        has_options=has_options,
        sector_category=sector,
        active_only=True,
    )
    items = [_to_info(i) for i in results]
    return IndicesListResponse(indices=items, count=len(items))


@router.get("/{index_id}", response_model=IndexDetail, summary="Full detail for one index")
async def get_index(
    index_id: str,
    db: DatabaseManager = Depends(get_db),
    registry: IndexRegistry = Depends(get_index_registry),
):
    index_id = index_id.upper()
    defn = registry.get_or_none(index_id)
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_id}")

    # Base info
    detail = IndexDetail(**_to_info(defn).model_dump())

    # Latest price
    price_row = db.fetch_one(Q.GET_LATEST_PRICE_ANY_TF, (index_id,))
    if price_row:
        detail.ltp = price_row.get("close")
        prev = db.fetch_one(
            "SELECT close FROM price_data WHERE index_id = ? AND timeframe = '1d' "
            "ORDER BY timestamp DESC LIMIT 1",
            (index_id,),
        )
        if prev and prev["close"] and detail.ltp:
            detail.change = round(detail.ltp - prev["close"], 2)
            detail.change_pct = round((detail.change / prev["close"]) * 100, 2)

    # Latest signal
    signal_row = db.fetch_one(Q.GET_LATEST_SIGNAL, (index_id,))
    if signal_row:
        detail.latest_signal = signal_row.get("signal_type")
        detail.signal_confidence = signal_row.get("confidence_level")
        detail.regime = signal_row.get("regime")

    return detail
