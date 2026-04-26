"""
F&O Universe Scanner API endpoints.

GET /api/universe/snapshot          — latest universe snapshot with scores and tiers
GET /api/universe/options-chain/{symbol}  — options chain for one symbol
"""

from __future__ import annotations

import json
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

_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: int = 15):
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < ttl:
        return entry[1]
    return None


def _set_cache(key: str, value: Any) -> Any:
    _cache[key] = (time.monotonic(), value)
    return value


# ── Response models ──────────────────────────────────────────────────────────

class UniverseStock(BaseModel):
    """Single stock in F&O universe."""
    symbol: str
    rank: int
    tier: str  # "PRIORITY" | "ACTIVE" | "WATCHLIST"
    composite_score: float

    # Component scores (0-100)
    oi_score: float
    oi_change_score: float
    volume_score: float
    price_move_score: float
    iv_change_score: float
    spread_score: float

    # Price and volume
    ltp: float
    price_change_pct: float
    total_oi: int
    volume_ratio: float

    # Metadata
    sector: str
    lot_size: int
    company_name: str


class UniverseSnapshotResponse(BaseModel):
    """Latest universe snapshot."""
    timestamp: str
    active_universe: list[UniverseStock]
    total_fo_stocks: int
    total_liquid: int
    top_sector: str
    refresh_count: int


class OptionStrike(BaseModel):
    """Single strike in options chain."""
    strike: float

    # CE side
    ce_oi: int
    ce_oi_change: int
    ce_volume: int
    ce_iv: float
    ce_ltp: float

    # PE side
    pe_oi: int
    pe_oi_change: int
    pe_volume: int
    pe_iv: float
    pe_ltp: float


class OptionsChainResponse(BaseModel):
    """Options chain for one symbol."""
    symbol: str
    spot_price: float
    lot_size: int
    expiry_date: str
    available_expiries: list[str]

    atm_strike: float
    max_pain_strike: float
    pcr: float

    total_ce_oi: int
    total_pe_oi: int
    ce_oi_change_pct: float
    pe_oi_change_pct: float

    strikes: list[OptionStrike]


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/snapshot", response_model=UniverseSnapshotResponse)
async def get_universe_snapshot(db: DatabaseManager = Depends(get_db)):
    """
    Get latest F&O universe snapshot.

    Returns top 50 active stocks with scores, tiers, and summary metrics.
    """
    cache_key = "universe_snapshot"
    cached = _cached(cache_key, ttl=15)
    if cached:
        return cached

    try:
        # Query latest snapshot from DB
        row = db.fetch_one(
            "SELECT * FROM fo_universe_snapshots ORDER BY date DESC LIMIT 1"
        )

        if not row:
            # No snapshot yet, return empty
            result = UniverseSnapshotResponse(
                timestamp=get_ist_now().isoformat(),
                active_universe=[],
                total_fo_stocks=0,
                total_liquid=0,
                top_sector="—",
                refresh_count=0,
            )
            _set_cache(cache_key, result)
            return result

        # Parse JSON data
        active_universe_raw = json.loads(row.get("active_universe", "[]"))

        # Convert to response model
        active_universe = [
            UniverseStock(
                symbol=s.get("symbol"),
                rank=s.get("rank"),
                tier=s.get("tier"),
                composite_score=s.get("composite_score", 0),
                oi_score=s.get("oi_score", 0),
                oi_change_score=s.get("oi_change_score", 0),
                volume_score=s.get("volume_score", 0),
                price_move_score=s.get("price_move_score", 0),
                iv_change_score=s.get("iv_change_score", 0),
                spread_score=s.get("spread_score", 0),
                ltp=s.get("ltp", 0),
                price_change_pct=s.get("price_change_pct", 0),
                total_oi=s.get("total_oi", 0),
                volume_ratio=s.get("volume_ratio", 0),
                sector=s.get("sector", "—"),
                lot_size=s.get("lot_size", 0),
                company_name=s.get("company_name", ""),
            )
            for s in active_universe_raw
        ]

        result = UniverseSnapshotResponse(
            timestamp=row.get("timestamp", get_ist_now().isoformat()),
            active_universe=active_universe,
            total_fo_stocks=row.get("total_fo_stocks", 0),
            total_liquid=row.get("total_liquid", 0),
            top_sector=row.get("top_sector", "—"),
            refresh_count=row.get("refresh_count", 0),
        )

        _set_cache(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Error fetching universe snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/options-chain/{symbol}", response_model=OptionsChainResponse)
async def get_options_chain(
    symbol: str,
    expiry: Optional[str] = Query(None),
    db: DatabaseManager = Depends(get_db),
):
    """
    Get options chain for a symbol.

    If expiry is not provided, returns current week's chain.
    Shows 11 strikes (5 above and below ATM) by default.
    """
    try:
        symbol = symbol.upper()

        # Query latest price for the symbol
        price_row = db.fetch_one(
            "SELECT close FROM price_data WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1",
            (symbol,)
        )

        if not price_row:
            raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

        spot_price = price_row.get("close", 0)

        # Get available expiries
        expiry_rows = db.fetch_all(
            "SELECT DISTINCT expiry_date FROM options_chain_snapshot WHERE index_id = ? ORDER BY expiry_date ASC",
            (symbol,)
        )
        available_expiries = [r["expiry_date"] for r in expiry_rows]

        if not available_expiries:
            raise HTTPException(status_code=404, detail=f"No options chain for {symbol}")

        # Use provided expiry or first available
        selected_expiry = expiry or available_expiries[0]

        if selected_expiry not in available_expiries:
            raise HTTPException(status_code=400, detail=f"Expiry {selected_expiry} not available")

        # Get options chain for this expiry
        chain_rows = db.fetch_all(
            """
            SELECT strike_price, option_type, open_interest, oi_change, volume, iv, ltp
            FROM options_chain_snapshot
            WHERE index_id = ? AND expiry_date = ?
            ORDER BY strike_price ASC
            """,
            (symbol, selected_expiry)
        )

        if not chain_rows:
            raise HTTPException(status_code=404, detail=f"No chain data for {symbol} on {selected_expiry}")

        # Organize by strike
        strikes_map = {}
        for row in chain_rows:
            strike = row["strike_price"]
            if strike not in strikes_map:
                strikes_map[strike] = {"strike": strike}

            opt_type = row["option_type"]
            prefix = "ce" if opt_type == "CE" else "pe"
            strikes_map[strike][f"{prefix}_oi"] = row.get("open_interest", 0)
            strikes_map[strike][f"{prefix}_oi_change"] = row.get("oi_change", 0)
            strikes_map[strike][f"{prefix}_volume"] = row.get("volume", 0)
            strikes_map[strike][f"{prefix}_iv"] = row.get("iv", 0.0) or 0.0
            strikes_map[strike][f"{prefix}_ltp"] = row.get("ltp", 0.0) or 0.0

        # Convert to list
        strikes = [
            OptionStrike(
                strike=s["strike"],
                ce_oi=s.get("ce_oi", 0),
                ce_oi_change=s.get("ce_oi_change", 0),
                ce_volume=s.get("ce_volume", 0),
                ce_iv=s.get("ce_iv", 0.0),
                ce_ltp=s.get("ce_ltp", 0.0),
                pe_oi=s.get("pe_oi", 0),
                pe_oi_change=s.get("pe_oi_change", 0),
                pe_volume=s.get("pe_volume", 0),
                pe_iv=s.get("pe_iv", 0.0),
                pe_ltp=s.get("pe_ltp", 0.0),
            )
            for s in strikes_map.values()
        ]

        # Calculate summary metrics
        total_ce_oi = sum(s.ce_oi for s in strikes)
        total_pe_oi = sum(s.pe_oi for s in strikes)
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

        # Find ATM and max pain
        atm_strike = min(strikes, key=lambda s: abs(s.strike - spot_price)).strike
        max_pain_strike = atm_strike  # Simplified; could calculate properly

        result = OptionsChainResponse(
            symbol=symbol,
            spot_price=spot_price,
            lot_size=250,  # Default, should query from index_master
            expiry_date=selected_expiry,
            available_expiries=available_expiries,
            atm_strike=atm_strike,
            max_pain_strike=max_pain_strike,
            pcr=round(pcr, 2),
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            ce_oi_change_pct=0.0,  # Would calculate from previous snapshot
            pe_oi_change_pct=0.0,
            strikes=strikes,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching options chain for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh", summary="Manually trigger universe refresh")
async def trigger_universe_refresh():
    """
    Manually trigger F&O universe selection without waiting for scheduler.

    Useful for dev/testing. Requires FOUniverseManager to be initialized.
    """
    try:
        from src.api.dependencies import get_fo_universe_manager

        manager = get_fo_universe_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="F&O Universe Manager not initialized. Start via run_system.py",
            )

        logger.info("Manual universe refresh triggered via API")
        snapshot = manager.run_daily_universe_selection()

        return {
            "status": "success",
            "timestamp": snapshot.timestamp.isoformat(),
            "active_universe_count": len(snapshot.active_universe),
            "priority_count": len(snapshot.priority_watchlist),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual universe refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
