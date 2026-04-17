"""
Portfolio and P&L API endpoints.

GET /api/portfolio/summary              — portfolio overview
GET /api/portfolio/positions            — open positions with live P&L
GET /api/portfolio/history              — daily P&L history for equity curve
GET /api/portfolio/trades               — recent closed trades
GET /api/portfolio/trades/export        — download CSV of trade history
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_db
from src.database.db_manager import DatabaseManager
from src.utils.date_utils import get_ist_now

router = APIRouter()
logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000.0


# ── Response models ──────────────────────────────────────────────────────────

class OpenPosition(BaseModel):
    id: int
    index_id: str
    signal_type: str
    confidence_level: str
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    generated_at: Optional[str] = None


class PortfolioSummary(BaseModel):
    capital: float = INITIAL_CAPITAL
    initial_capital: float = INITIAL_CAPITAL
    total_return_pct: float = 0.0
    today_pnl: float = 0.0
    today_pnl_pct: float = 0.0
    today_trades: int = 0
    today_win_rate: float = 0.0
    open_positions: list[OpenPosition] = Field(default_factory=list)
    total_closed_trades: int = 0
    overall_win_rate: float = 0.0


class DailyPnL(BaseModel):
    date: str
    capital: float
    pnl: float
    trades: int


class PnLHistoryResponse(BaseModel):
    history: list[DailyPnL]
    count: int


class ClosedTrade(BaseModel):
    id: int
    index_id: str
    signal_type: str
    confidence_level: str
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    outcome: Optional[str] = None
    generated_at: Optional[str] = None
    closed_at: Optional[str] = None


class TradesResponse(BaseModel):
    trades: list[ClosedTrade]
    count: int
    total: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/summary", response_model=PortfolioSummary, summary="Portfolio overview")
async def get_portfolio_summary(db: DatabaseManager = Depends(get_db)):
    now = get_ist_now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    # All closed trades
    all_closed = db.fetch_all(
        "SELECT actual_pnl, outcome FROM trading_signals "
        "WHERE outcome IN ('WIN', 'LOSS') AND actual_pnl IS NOT NULL",
    )
    total_pnl = sum(r["actual_pnl"] for r in all_closed)
    capital = INITIAL_CAPITAL + total_pnl

    total_closed = len(all_closed)
    total_wins = sum(1 for r in all_closed if r["outcome"] == "WIN")
    overall_wr = (total_wins / total_closed * 100) if total_closed else 0.0

    # Today's closed trades
    today_closed = db.fetch_all(
        "SELECT actual_pnl, outcome FROM trading_signals "
        "WHERE outcome IN ('WIN', 'LOSS') AND closed_at >= ?",
        (today_start,),
    )
    today_pnl = sum(r["actual_pnl"] for r in today_closed if r["actual_pnl"])
    today_trades = len(today_closed)
    today_wins = sum(1 for r in today_closed if r["outcome"] == "WIN")
    today_wr = (today_wins / today_trades * 100) if today_trades else 0.0

    # Open positions
    open_rows = db.fetch_all(
        "SELECT id, index_id, signal_type, confidence_level, "
        "entry_price, target_price, stop_loss, generated_at "
        "FROM trading_signals WHERE outcome = 'OPEN' "
        "ORDER BY generated_at DESC",
    )
    positions: list[OpenPosition] = []
    for r in open_rows:
        # Get current price for unrealized P&L
        price_row = db.fetch_one(
            "SELECT close FROM price_data WHERE index_id = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (r["index_id"],),
        )
        current = price_row["close"] if price_row else None
        entry = r["entry_price"] or 0
        upnl = None
        upnl_pct = None
        if current and entry:
            if r["signal_type"] == "BUY_CALL":
                upnl = round(current - entry, 2)
            else:
                upnl = round(entry - current, 2)
            upnl_pct = round((upnl / entry) * 100, 2) if entry else None

        positions.append(OpenPosition(
            id=r["id"],
            index_id=r["index_id"],
            signal_type=r["signal_type"],
            confidence_level=r["confidence_level"],
            entry_price=entry,
            target_price=r.get("target_price"),
            stop_loss=r.get("stop_loss"),
            current_price=current,
            unrealized_pnl=upnl,
            unrealized_pnl_pct=upnl_pct,
            generated_at=r.get("generated_at"),
        ))

    return PortfolioSummary(
        capital=round(capital, 2),
        initial_capital=INITIAL_CAPITAL,
        total_return_pct=round((total_pnl / INITIAL_CAPITAL) * 100, 2),
        today_pnl=round(today_pnl, 2),
        today_pnl_pct=round((today_pnl / capital) * 100, 2) if capital else 0.0,
        today_trades=today_trades,
        today_win_rate=round(today_wr, 1),
        open_positions=positions,
        total_closed_trades=total_closed,
        overall_win_rate=round(overall_wr, 1),
    )


@router.get("/positions", response_model=list[OpenPosition], summary="Open positions with live P&L")
async def get_positions(db: DatabaseManager = Depends(get_db)):
    open_rows = db.fetch_all(
        "SELECT id, index_id, signal_type, confidence_level, "
        "entry_price, target_price, stop_loss, generated_at "
        "FROM trading_signals WHERE outcome = 'OPEN' "
        "ORDER BY generated_at DESC",
    )
    positions: list[OpenPosition] = []
    for r in open_rows:
        price_row = db.fetch_one(
            "SELECT close FROM price_data WHERE index_id = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (r["index_id"],),
        )
        current = price_row["close"] if price_row else None
        entry = r["entry_price"] or 0
        upnl = None
        upnl_pct = None
        if current and entry:
            upnl = round(current - entry, 2) if r["signal_type"] == "BUY_CALL" else round(entry - current, 2)
            upnl_pct = round((upnl / entry) * 100, 2) if entry else None

        positions.append(OpenPosition(
            id=r["id"],
            index_id=r["index_id"],
            signal_type=r["signal_type"],
            confidence_level=r["confidence_level"],
            entry_price=entry,
            target_price=r.get("target_price"),
            stop_loss=r.get("stop_loss"),
            current_price=current,
            unrealized_pnl=upnl,
            unrealized_pnl_pct=upnl_pct,
            generated_at=r.get("generated_at"),
        ))
    return positions


@router.get("/history", response_model=PnLHistoryResponse, summary="Daily P&L history for equity curve")
async def get_pnl_history(
    days: int = Query(default=30, ge=1, le=365),
    db: DatabaseManager = Depends(get_db),
):
    now = get_ist_now()
    since = (now - timedelta(days=days)).isoformat()

    rows = db.fetch_all(
        "SELECT DATE(closed_at) AS date, "
        "SUM(actual_pnl) AS pnl, "
        "COUNT(*) AS trades "
        "FROM trading_signals "
        "WHERE outcome IN ('WIN', 'LOSS') AND closed_at >= ? "
        "GROUP BY DATE(closed_at) "
        "ORDER BY date ASC",
        (since,),
    )

    # Build cumulative equity curve
    running_capital = INITIAL_CAPITAL
    history: list[DailyPnL] = []
    for r in rows:
        pnl = r["pnl"] or 0
        running_capital += pnl
        history.append(DailyPnL(
            date=r["date"],
            capital=round(running_capital, 2),
            pnl=round(pnl, 2),
            trades=r["trades"],
        ))

    return PnLHistoryResponse(history=history, count=len(history))


@router.get("/trades", response_model=TradesResponse, summary="Recent closed trades")
async def get_trades(
    days: int = Query(default=7, ge=1, le=365),
    index_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: DatabaseManager = Depends(get_db),
):
    now = get_ist_now()
    since = (now - timedelta(days=days)).isoformat()

    clauses = ["outcome IN ('WIN', 'LOSS')", "closed_at >= ?"]
    params: list = [since]

    if index_id:
        clauses.append("index_id = ?")
        params.append(index_id.upper())

    where = " AND ".join(clauses)

    count_row = db.fetch_one(
        f"SELECT COUNT(*) AS cnt FROM trading_signals WHERE {where}",
        tuple(params),
    )
    total = count_row["cnt"] if count_row else 0

    rows = db.fetch_all(
        f"SELECT id, index_id, signal_type, confidence_level, "
        f"entry_price, actual_exit_price, actual_pnl, outcome, "
        f"generated_at, closed_at "
        f"FROM trading_signals WHERE {where} "
        f"ORDER BY closed_at DESC LIMIT ? OFFSET ?",
        (*params, limit, offset),
    )

    trades = [
        ClosedTrade(
            id=r["id"],
            index_id=r["index_id"],
            signal_type=r["signal_type"],
            confidence_level=r["confidence_level"],
            entry_price=r["entry_price"],
            exit_price=r.get("actual_exit_price"),
            pnl=r.get("actual_pnl"),
            outcome=r.get("outcome"),
            generated_at=r.get("generated_at"),
            closed_at=r.get("closed_at"),
        )
        for r in rows
    ]
    return TradesResponse(trades=trades, count=len(trades), total=total)


@router.get("/trades/export", summary="Export trade history as CSV")
async def export_trades(
    days: int = Query(default=30, ge=1, le=365),
    index_id: Optional[str] = Query(default=None),
    db: DatabaseManager = Depends(get_db),
):
    now = get_ist_now()
    since = (now - timedelta(days=days)).isoformat()

    clauses = ["outcome IN ('WIN', 'LOSS')", "closed_at >= ?"]
    params: list = [since]

    if index_id:
        clauses.append("index_id = ?")
        params.append(index_id.upper())

    where = " AND ".join(clauses)

    rows = db.fetch_all(
        f"SELECT index_id, signal_type, confidence_level, "
        f"entry_price, actual_exit_price, stop_loss, target_price, "
        f"actual_pnl, outcome, generated_at, closed_at "
        f"FROM trading_signals WHERE {where} "
        f"ORDER BY closed_at DESC",
        tuple(params),
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Date", "Index", "Type", "Confidence", "Entry", "Exit",
        "SL", "Target", "P&L (INR)", "Result", "Opened", "Closed",
    ])
    for r in rows:
        writer.writerow([
            r.get("closed_at", ""),
            r["index_id"],
            r["signal_type"],
            r["confidence_level"],
            r.get("entry_price", ""),
            r.get("actual_exit_price", ""),
            r.get("stop_loss", ""),
            r.get("target_price", ""),
            round(r["actual_pnl"], 2) if r.get("actual_pnl") is not None else "",
            r.get("outcome", ""),
            r.get("generated_at", ""),
            r.get("closed_at", ""),
        ])

    output.seek(0)
    filename = f"trades_{date.today().isoformat()}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
