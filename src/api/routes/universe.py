"""
F&O Universe Scanner API endpoints.

GET /api/universe/snapshot          — latest universe snapshot with scores and tiers
GET /api/universe/options-chain/{symbol}  — options chain for one symbol
"""

from __future__ import annotations

import asyncio
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


_nse_scraper = None


def _get_nse_scraper():
    """Lazily created NSE scraper shared by live chain requests."""
    global _nse_scraper
    if _nse_scraper is None:
        from src.data.nse_scraper import NSEScraper  # noqa: PLC0415
        _nse_scraper = NSEScraper()
    return _nse_scraper


def _live_chain(symbol: str, kind: str, expiry: Optional[str]):
    """
    Fetch an option chain from NSE v3 (cached 60s).

    ``kind`` is "Equities" or "Indices"; ``symbol`` is the NSE option
    symbol (e.g. RELIANCE, NIFTY).

    Returns (spot, expiries, selected_expiry, rows) with rows shaped like
    options_chain_snapshot rows, or None when NSE has no chain.
    """
    cache_key = f"live_chain:{kind}:{symbol}:{expiry or ''}"
    cached = _cached(cache_key, ttl=60)
    if cached is not None:
        return cached

    scraper = _get_nse_scraper()
    meta = scraper.get_option_chain_meta(symbol) or {}
    expiries = meta.get("expiryDates") or []
    if not expiries:
        return None
    selected = expiry or expiries[0]
    if selected not in expiries:
        raise HTTPException(status_code=400, detail=f"Expiry {selected} not available")

    resp = scraper.get_option_chain_v3(symbol, type=kind, expiry=selected) or {}
    records = resp.get("records") or {}
    data = records.get("data") or []
    if not data:
        return None

    spot = float(records.get("underlyingValue") or 0)
    rows = []
    for entry in data:
        for opt in ("CE", "PE"):
            leg = entry.get(opt)
            if not leg:
                continue
            rows.append({
                "strike_price": float(leg.get("strikePrice") or entry.get("strikePrice") or 0),
                "option_type": opt,
                "open_interest": int(leg.get("openInterest") or 0),
                "oi_change": int(leg.get("changeinOpenInterest") or 0),
                "volume": int(leg.get("totalTradedVolume") or 0),
                "iv": float(leg.get("impliedVolatility") or 0.0),
                "ltp": float(leg.get("lastPrice") or 0.0),
            })
    rows.sort(key=lambda r: r["strike_price"])
    return _set_cache(cache_key, (spot, expiries, selected, rows))


def _max_pain(strikes: list[OptionStrike]) -> Optional[float]:
    """
    Strike at which option writers' total payout at expiry is smallest.

    payout(K) = sum over strikes s of
        ce_oi(s) * max(K - s, 0) + pe_oi(s) * max(s - K, 0)
    """
    if not strikes or not any(s.ce_oi or s.pe_oi for s in strikes):
        return None
    best_strike, best_pain = None, None
    for k in strikes:
        pain = sum(
            s.ce_oi * max(k.strike - s.strike, 0) + s.pe_oi * max(s.strike - k.strike, 0)
            for s in strikes
        )
        if best_pain is None or pain < best_pain:
            best_strike, best_pain = k.strike, pain
    return best_strike


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

        # Index chains are recorded in options_chain_snapshot; stock chains
        # are not, so stocks fall back to a live NSE v3 fetch.
        index_row = db.fetch_one(
            "SELECT option_symbol, lot_size FROM index_master WHERE id = ? AND has_options = 1",
            (symbol,)
        )
        today_iso = get_ist_now().date().isoformat()
        expiry_rows = db.fetch_all(
            "SELECT DISTINCT expiry_date FROM options_chain_snapshot "
            "WHERE index_id = ? AND expiry_date >= ? ORDER BY expiry_date ASC",
            (symbol, today_iso)
        )
        available_expiries = [r["expiry_date"] for r in expiry_rows]

        if available_expiries:
            price_row = db.fetch_one(
                "SELECT close FROM price_data WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1",
                (symbol,)
            )
            if not price_row:
                raise HTTPException(status_code=404, detail=f"No price data for {symbol}")
            spot_price = price_row.get("close", 0)

            selected_expiry = expiry or available_expiries[0]
            if selected_expiry not in available_expiries:
                raise HTTPException(status_code=400, detail=f"Expiry {selected_expiry} not available")

            chain_rows = db.fetch_all(
                """
                SELECT strike_price, option_type, open_interest, oi_change, volume, iv, ltp
                FROM options_chain_snapshot
                WHERE index_id = ? AND expiry_date = ?
                  AND timestamp = (
                      SELECT MAX(timestamp) FROM options_chain_snapshot
                      WHERE index_id = ? AND expiry_date = ?
                  )
                ORDER BY strike_price ASC
                """,
                (symbol, selected_expiry, symbol, selected_expiry)
            )
        else:
            nse_symbol = (index_row or {}).get("option_symbol") or symbol
            kind = "Indices" if index_row else "Equities"
            live = await asyncio.to_thread(_live_chain, nse_symbol, kind, expiry)
            if live is None:
                raise HTTPException(status_code=404, detail=f"No options chain for {symbol}")
            spot_price, available_expiries, selected_expiry, chain_rows = live

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

        # ATM and max pain
        atm_strike = min(strikes, key=lambda s: abs(s.strike - spot_price)).strike
        max_pain_strike = _max_pain(strikes) or atm_strike

        # OI change % vs the implied previous OI (current - change)
        ce_change = sum(s.ce_oi_change for s in strikes)
        pe_change = sum(s.pe_oi_change for s in strikes)
        ce_prev = total_ce_oi - ce_change
        pe_prev = total_pe_oi - pe_change
        ce_oi_change_pct = round(ce_change / ce_prev * 100, 2) if ce_prev > 0 else 0.0
        pe_oi_change_pct = round(pe_change / pe_prev * 100, 2) if pe_prev > 0 else 0.0

        if index_row:
            lot_row = index_row
        else:
            lot_row = db.fetch_one(
                "SELECT lot_size FROM fo_universe WHERE symbol = ?", (symbol,)
            )
        lot_size = int(lot_row["lot_size"]) if lot_row and lot_row.get("lot_size") else 0

        result = OptionsChainResponse(
            symbol=symbol,
            spot_price=spot_price,
            lot_size=lot_size,
            expiry_date=selected_expiry,
            available_expiries=available_expiries,
            atm_strike=atm_strike,
            max_pain_strike=max_pain_strike,
            pcr=round(pcr, 2),
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            ce_oi_change_pct=ce_oi_change_pct,
            pe_oi_change_pct=pe_oi_change_pct,
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
