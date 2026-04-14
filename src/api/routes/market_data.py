"""
Market data API endpoints (DB-backed).

GET /api/market/prices                         — latest prices for all active indices
GET /api/market/prices/{index_id}              — detailed price for one index
GET /api/market/prices/{index_id}/history      — historical OHLCV bars
GET /api/market/options/{index_id}             — options chain summary
GET /api/market/vix                            — current VIX reading
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_db, get_index_registry
from src.data.index_registry import IndexRegistry
from src.database import queries as Q
from src.database.db_manager import DatabaseManager
from src.utils.date_utils import get_ist_now

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Simple TTL cache ─────────────────────────────────────────────────────────
_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: int = 30):
    """Return cached value if still valid, else None."""
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < ttl:
        return entry[1]
    return None


def _set_cache(key: str, value: Any) -> Any:
    _cache[key] = (time.monotonic(), value)
    return value


# ── Response models ──────────────────────────────────────────────────────────

class IndexPrice(BaseModel):
    index_id: str
    ltp: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[int] = None
    updated_at: Optional[str] = None


class PricesResponse(BaseModel):
    timestamp: str
    indices: list[IndexPrice]


class PriceDetail(BaseModel):
    index_id: str
    ltp: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    vwap: Optional[float] = None
    previous_close: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    timeframe: Optional[str] = None
    updated_at: Optional[str] = None


class OHLCVBar(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None
    vwap: Optional[float] = None


class PriceHistoryResponse(BaseModel):
    index_id: str
    timeframe: str
    bars: list[OHLCVBar]
    count: int


class StrikeOI(BaseModel):
    strike: float
    oi: int


class OptionsResponse(BaseModel):
    index_id: str
    spot: Optional[float] = None
    expiry: Optional[str] = None
    pcr: Optional[float] = None
    max_pain: Optional[float] = None
    oi_support: Optional[float] = None
    oi_resistance: Optional[float] = None
    total_ce_oi: Optional[int] = None
    total_pe_oi: Optional[int] = None
    top_ce_strikes: list[StrikeOI] = Field(default_factory=list)
    top_pe_strikes: list[StrikeOI] = Field(default_factory=list)
    iv_atm: Optional[float] = None
    updated_at: Optional[str] = None


class VIXResponse(BaseModel):
    value: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    regime: str = "UNKNOWN"
    updated_at: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/prices", response_model=PricesResponse, summary="Latest prices for all active indices")
async def get_prices(
    db: DatabaseManager = Depends(get_db),
    registry: IndexRegistry = Depends(get_index_registry),
):
    cached = _cached("all_prices", ttl=15)
    if cached:
        return cached

    now = get_ist_now()
    indices: list[IndexPrice] = []

    for defn in registry.get_active_indices():
        row = db.fetch_one(Q.GET_LATEST_PRICE_ANY_TF, (defn.id,))
        if row is None:
            indices.append(IndexPrice(index_id=defn.id))
            continue

        # Get previous day close for change calculation
        prev = db.fetch_one(
            "SELECT close FROM price_data WHERE index_id = ? AND timeframe = '1d' "
            "ORDER BY timestamp DESC LIMIT 1",
            (defn.id,),
        )
        prev_close = prev["close"] if prev else None
        ltp = row["close"]
        change = round(ltp - prev_close, 2) if prev_close and ltp else None
        change_pct = round((change / prev_close) * 100, 2) if prev_close and change else None

        indices.append(IndexPrice(
            index_id=defn.id,
            ltp=ltp,
            change=change,
            change_pct=change_pct,
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            volume=int(row["volume"]) if row.get("volume") else None,
            updated_at=row.get("timestamp"),
        ))

    result = PricesResponse(timestamp=now.isoformat(), indices=indices)
    return _set_cache("all_prices", result)


@router.get("/prices/{index_id}", response_model=PriceDetail, summary="Detailed price for one index")
async def get_price_detail(
    index_id: str,
    db: DatabaseManager = Depends(get_db),
    registry: IndexRegistry = Depends(get_index_registry),
):
    index_id = index_id.upper()
    if registry.get_or_none(index_id) is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_id}")

    row = db.fetch_one(Q.GET_LATEST_PRICE_ANY_TF, (index_id,))
    if row is None:
        return PriceDetail(index_id=index_id)

    prev = db.fetch_one(
        "SELECT close FROM price_data WHERE index_id = ? AND timeframe = '1d' "
        "AND timestamp < ? ORDER BY timestamp DESC LIMIT 1",
        (index_id, row["timestamp"]),
    )
    prev_close = prev["close"] if prev else None
    ltp = row["close"]
    change = round(ltp - prev_close, 2) if prev_close and ltp else None
    change_pct = round((change / prev_close) * 100, 2) if prev_close and change else None

    return PriceDetail(
        index_id=index_id,
        ltp=ltp,
        open=row.get("open"),
        high=row.get("high"),
        low=row.get("low"),
        close=row.get("close"),
        volume=int(row["volume"]) if row.get("volume") else None,
        vwap=row.get("vwap"),
        previous_close=prev_close,
        change=change,
        change_pct=change_pct,
        timeframe=row.get("timeframe"),
        updated_at=row.get("timestamp"),
    )


@router.get(
    "/prices/{index_id}/history",
    response_model=PriceHistoryResponse,
    summary="Historical OHLCV bars",
)
async def get_price_history(
    index_id: str,
    days: int = Query(default=30, ge=1, le=365),
    timeframe: str = Query(default="1d", description="1m, 5m, 15m, 1h, 1d"),
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    db: DatabaseManager = Depends(get_db),
    registry: IndexRegistry = Depends(get_index_registry),
):
    index_id = index_id.upper()
    if registry.get_or_none(index_id) is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_id}")

    now = get_ist_now()
    since = (now - timedelta(days=days)).isoformat()

    rows = db.fetch_all(
        Q.LIST_PRICE_HISTORY,
        (index_id, timeframe, since, now.isoformat()),
    )

    # Apply offset and limit
    rows = rows[offset: offset + limit]

    bars = [
        OHLCVBar(
            timestamp=r["timestamp"],
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            volume=int(r["volume"]) if r.get("volume") else None,
            vwap=r.get("vwap"),
        )
        for r in rows
    ]

    return PriceHistoryResponse(
        index_id=index_id,
        timeframe=timeframe,
        bars=bars,
        count=len(bars),
    )


@router.get("/options/{index_id}", response_model=OptionsResponse, summary="Options chain summary")
async def get_options_summary(
    index_id: str,
    db: DatabaseManager = Depends(get_db),
    registry: IndexRegistry = Depends(get_index_registry),
):
    index_id = index_id.upper()
    defn = registry.get_or_none(index_id)
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_id}")
    if not defn.has_options:
        raise HTTPException(status_code=400, detail=f"{index_id} does not have options")

    cache_key = f"options_{index_id}"
    cached = _cached(cache_key, ttl=60)
    if cached:
        return cached

    # Get the nearest expiry date
    expiries = db.fetch_all(Q.LIST_EXPIRY_DATES, (index_id,))
    if not expiries:
        return _set_cache(cache_key, OptionsResponse(index_id=index_id))

    now = get_ist_now()
    today_str = now.strftime("%Y-%m-%d")
    nearest_expiry = None
    for e in expiries:
        if e["expiry_date"] >= today_str:
            nearest_expiry = e["expiry_date"]
            break
    if nearest_expiry is None:
        nearest_expiry = expiries[-1]["expiry_date"]

    # Get aggregated OI data
    oi_row = db.fetch_one(Q.GET_LATEST_OI_AGGREGATED, (index_id, nearest_expiry))

    # Get spot price
    price_row = db.fetch_one(Q.GET_LATEST_PRICE_ANY_TF, (index_id,))
    spot = price_row["close"] if price_row else None

    # Get top strikes by OI from options chain
    top_ce = db.fetch_all(
        "SELECT strike_price AS strike, open_interest AS oi "
        "FROM options_chain_snapshot "
        "WHERE index_id = ? AND expiry_date = ? AND option_type = 'CE' "
        "AND timestamp = (SELECT MAX(timestamp) FROM options_chain_snapshot "
        "WHERE index_id = ? AND expiry_date = ?) "
        "ORDER BY open_interest DESC LIMIT 5",
        (index_id, nearest_expiry, index_id, nearest_expiry),
    )
    top_pe = db.fetch_all(
        "SELECT strike_price AS strike, open_interest AS oi "
        "FROM options_chain_snapshot "
        "WHERE index_id = ? AND expiry_date = ? AND option_type = 'PE' "
        "AND timestamp = (SELECT MAX(timestamp) FROM options_chain_snapshot "
        "WHERE index_id = ? AND expiry_date = ?) "
        "ORDER BY open_interest DESC LIMIT 5",
        (index_id, nearest_expiry, index_id, nearest_expiry),
    )

    # ATM IV: find the strike closest to spot
    iv_atm = None
    if spot:
        iv_row = db.fetch_one(
            "SELECT iv FROM options_chain_snapshot "
            "WHERE index_id = ? AND expiry_date = ? AND option_type = 'CE' "
            "AND timestamp = (SELECT MAX(timestamp) FROM options_chain_snapshot "
            "WHERE index_id = ? AND expiry_date = ?) "
            "ORDER BY ABS(strike_price - ?) LIMIT 1",
            (index_id, nearest_expiry, index_id, nearest_expiry, spot),
        )
        if iv_row and iv_row.get("iv"):
            iv_atm = round(iv_row["iv"], 2)

    result = OptionsResponse(
        index_id=index_id,
        spot=spot,
        expiry=nearest_expiry,
        pcr=round(oi_row["pcr"], 2) if oi_row and oi_row.get("pcr") else None,
        max_pain=oi_row.get("max_pain_strike") if oi_row else None,
        oi_support=oi_row.get("highest_pe_oi_strike") if oi_row else None,
        oi_resistance=oi_row.get("highest_ce_oi_strike") if oi_row else None,
        total_ce_oi=int(oi_row["total_ce_oi"]) if oi_row and oi_row.get("total_ce_oi") else None,
        total_pe_oi=int(oi_row["total_pe_oi"]) if oi_row and oi_row.get("total_pe_oi") else None,
        top_ce_strikes=[StrikeOI(strike=r["strike"], oi=int(r["oi"])) for r in top_ce if r.get("oi")],
        top_pe_strikes=[StrikeOI(strike=r["strike"], oi=int(r["oi"])) for r in top_pe if r.get("oi")],
        iv_atm=iv_atm,
        updated_at=oi_row.get("timestamp") if oi_row else None,
    )
    return _set_cache(cache_key, result)


@router.get("/vix", response_model=VIXResponse, summary="Current India VIX")
async def get_vix(db: DatabaseManager = Depends(get_db)):
    cached = _cached("vix", ttl=30)
    if cached:
        return cached

    row = db.fetch_one(Q.GET_LATEST_VIX)
    if row is None:
        return VIXResponse()

    vix_val = row.get("vix_value")
    if vix_val is None:
        return VIXResponse(updated_at=row.get("timestamp"))

    # Determine regime
    if vix_val < 13:
        regime = "LOW"
    elif vix_val < 20:
        regime = "NORMAL"
    elif vix_val < 25:
        regime = "ELEVATED"
    elif vix_val < 35:
        regime = "HIGH"
    else:
        regime = "EXTREME"

    result = VIXResponse(
        value=round(vix_val, 2),
        change=round(row["vix_change"], 2) if row.get("vix_change") is not None else None,
        change_pct=round(row["vix_change_pct"], 2) if row.get("vix_change_pct") is not None else None,
        regime=regime,
        updated_at=row.get("timestamp"),
    )
    return _set_cache("vix", result)
