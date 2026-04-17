"""
Trading signals API endpoints.

GET /api/signals/current                — latest signal for each F&O index
GET /api/signals/current/{index_id}     — full signal detail for one index
GET /api/signals/history                — signal history with filtering
GET /api/signals/performance            — performance statistics
"""

from __future__ import annotations

import json
import logging
import time
from datetime import timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_db, get_index_registry, get_signal_tracker
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

class OptionTrade(BaseModel):
    strike: Optional[float] = None
    expiry: Optional[str] = None
    option_type: Optional[str] = None  # "CE" or "PE"
    premium: Optional[float] = None
    lots: Optional[int] = None
    max_loss_amount: Optional[float] = None
    risk_pct_of_capital: Optional[float] = None


class SignalSummary(BaseModel):
    index_id: str
    signal_type: str = "NO_TRADE"
    confidence_level: str = "LOW"
    confidence_score: Optional[float] = None
    entry: Optional[float] = None
    target: Optional[float] = None
    sl: Optional[float] = None
    rr_ratio: Optional[float] = None
    regime: Optional[str] = None
    generated_at: Optional[str] = None
    reasoning_summary: Optional[str] = None
    option_trade: Optional[OptionTrade] = None


class CurrentSignalsResponse(BaseModel):
    signals: list[SignalSummary]


class SignalDetail(BaseModel):
    id: Optional[int] = None
    index_id: str
    signal_type: str = "NO_TRADE"
    confidence_level: str = "LOW"
    confidence_score: Optional[float] = None
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    regime: Optional[str] = None
    technical_vote: Optional[str] = None
    options_vote: Optional[str] = None
    news_vote: Optional[str] = None
    anomaly_vote: Optional[str] = None
    reasoning: Optional[dict] = None
    option_trade: Optional[OptionTrade] = None
    outcome: Optional[str] = None
    actual_exit_price: Optional[float] = None
    actual_pnl: Optional[float] = None
    generated_at: Optional[str] = None
    closed_at: Optional[str] = None


class SignalHistoryResponse(BaseModel):
    signals: list[SignalDetail]
    count: int
    total: int


class PerformanceStatsResponse(BaseModel):
    period_days: int = 0
    total_signals: int = 0
    actionable_signals: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    high_confidence_win_rate: float = 0.0
    medium_confidence_win_rate: float = 0.0
    low_confidence_win_rate: float = 0.0
    win_rate_by_index: dict = Field(default_factory=dict)
    pnl_by_index: dict = Field(default_factory=dict)
    is_profitable: bool = False
    edge_comment: str = "Insufficient data"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_reasoning(raw: Optional[str]) -> tuple[Optional[dict], Optional[float], Optional[str]]:
    """Parse reasoning JSON, extract confidence_score and summary."""
    if not raw:
        return None, None, None
    try:
        d = json.loads(raw) if isinstance(raw, str) else raw
        score = d.get("confidence_score")
        text = d.get("text") or d.get("reasoning") or d.get("summary")
        return d, float(score) if score is not None else None, str(text) if text else None
    except (json.JSONDecodeError, TypeError):
        return None, None, str(raw)[:200] if raw else None


def _extract_option_trade(reasoning_dict: Optional[dict]) -> Optional[OptionTrade]:
    if not reasoning_dict:
        return None
    raw = reasoning_dict.get("option_trade")
    if not isinstance(raw, dict):
        return None
    try:
        return OptionTrade(**{k: raw.get(k) for k in OptionTrade.model_fields})
    except (TypeError, ValueError):
        return None


def _row_to_detail(row: dict) -> SignalDetail:
    reasoning_dict, conf_score, _ = _parse_reasoning(row.get("reasoning"))
    return SignalDetail(
        id=row.get("id"),
        index_id=row["index_id"],
        signal_type=row.get("signal_type", "NO_TRADE"),
        confidence_level=row.get("confidence_level", "LOW"),
        confidence_score=conf_score,
        entry_price=row.get("entry_price"),
        target_price=row.get("target_price"),
        stop_loss=row.get("stop_loss"),
        risk_reward_ratio=row.get("risk_reward_ratio"),
        regime=row.get("regime"),
        technical_vote=row.get("technical_vote"),
        options_vote=row.get("options_vote"),
        news_vote=row.get("news_vote"),
        anomaly_vote=row.get("anomaly_vote"),
        reasoning=reasoning_dict,
        option_trade=_extract_option_trade(reasoning_dict),
        outcome=row.get("outcome"),
        actual_exit_price=row.get("actual_exit_price"),
        actual_pnl=row.get("actual_pnl"),
        generated_at=row.get("generated_at"),
        closed_at=row.get("closed_at"),
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/current", response_model=CurrentSignalsResponse, summary="Latest signal per F&O index")
async def get_current_signals(
    db: DatabaseManager = Depends(get_db),
    registry: IndexRegistry = Depends(get_index_registry),
):
    cached = _cached("current_signals", ttl=15)
    if cached:
        return cached

    fo_indices = registry.filter(has_options=True, active_only=True)
    signals: list[SignalSummary] = []

    for defn in fo_indices:
        row = db.fetch_one(Q.GET_LATEST_SIGNAL, (defn.id,))
        if row is None:
            signals.append(SignalSummary(index_id=defn.id))
            continue

        reasoning_dict, conf_score, summary = _parse_reasoning(row.get("reasoning"))

        signals.append(SignalSummary(
            index_id=defn.id,
            signal_type=row.get("signal_type", "NO_TRADE"),
            confidence_level=row.get("confidence_level", "LOW"),
            confidence_score=conf_score,
            entry=row.get("entry_price"),
            target=row.get("target_price"),
            sl=row.get("stop_loss"),
            rr_ratio=row.get("risk_reward_ratio"),
            regime=row.get("regime"),
            generated_at=row.get("generated_at"),
            reasoning_summary=summary,
            option_trade=_extract_option_trade(reasoning_dict),
        ))

    result = CurrentSignalsResponse(signals=signals)
    return _set_cache("current_signals", result)


@router.get("/current/{index_id}", response_model=SignalDetail, summary="Full signal detail for one index")
async def get_current_signal_detail(
    index_id: str,
    db: DatabaseManager = Depends(get_db),
    registry: IndexRegistry = Depends(get_index_registry),
):
    index_id = index_id.upper()
    if registry.get_or_none(index_id) is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_id}")

    row = db.fetch_one(Q.GET_LATEST_SIGNAL, (index_id,))
    if row is None:
        return SignalDetail(index_id=index_id)

    return _row_to_detail(row)


@router.get("/history", response_model=SignalHistoryResponse, summary="Signal history with filtering")
async def get_signal_history(
    days: int = Query(default=7, ge=1, le=365),
    index_id: Optional[str] = Query(default=None),
    outcome: Optional[str] = Query(default=None, description="WIN, LOSS, OPEN, EXPIRED"),
    signal_type: Optional[str] = Query(default=None, description="BUY_CALL, BUY_PUT, NO_TRADE"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: DatabaseManager = Depends(get_db),
):
    now = get_ist_now()
    since = (now - timedelta(days=days)).isoformat()

    clauses = ["generated_at >= ?"]
    params: list[Any] = [since]

    if index_id:
        clauses.append("index_id = ?")
        params.append(index_id.upper())
    if outcome:
        clauses.append("outcome = ?")
        params.append(outcome.upper())
    if signal_type:
        clauses.append("signal_type = ?")
        params.append(signal_type.upper())

    where = " AND ".join(clauses)

    # Count total matching rows
    count_row = db.fetch_one(
        f"SELECT COUNT(*) AS cnt FROM trading_signals WHERE {where}",
        tuple(params),
    )
    total = count_row["cnt"] if count_row else 0

    # Fetch page
    rows = db.fetch_all(
        f"SELECT * FROM trading_signals WHERE {where} "
        f"ORDER BY generated_at DESC LIMIT ? OFFSET ?",
        (*params, limit, offset),
    )

    return SignalHistoryResponse(
        signals=[_row_to_detail(r) for r in rows],
        count=len(rows),
        total=total,
    )


@router.get("/performance", response_model=PerformanceStatsResponse, summary="Performance statistics")
async def get_performance(
    days: int = Query(default=30, ge=1, le=365),
    db: DatabaseManager = Depends(get_db),
):
    cache_key = f"perf_{days}"
    cached = _cached(cache_key, ttl=60)
    if cached:
        return cached

    tracker = get_signal_tracker()
    if tracker is not None:
        stats = tracker.get_performance_stats(days=days)
        result = PerformanceStatsResponse(
            period_days=stats.period_days,
            total_signals=stats.total_signals,
            actionable_signals=stats.actionable_signals,
            total_trades=stats.total_trades,
            wins=stats.wins,
            losses=stats.losses,
            win_rate=stats.win_rate,
            total_pnl=stats.total_pnl,
            avg_pnl_per_trade=stats.avg_pnl_per_trade,
            largest_win=stats.largest_win,
            largest_loss=stats.largest_loss,
            profit_factor=stats.profit_factor,
            max_drawdown=stats.max_drawdown,
            sharpe_ratio=stats.sharpe_ratio,
            high_confidence_win_rate=stats.high_confidence_win_rate,
            medium_confidence_win_rate=stats.medium_confidence_win_rate,
            low_confidence_win_rate=stats.low_confidence_win_rate,
            win_rate_by_index=stats.win_rate_by_index,
            pnl_by_index=stats.pnl_by_index,
            is_profitable=stats.is_profitable,
            edge_comment=stats.edge_comment,
        )
        return _set_cache(cache_key, result)

    # Fallback: compute from DB directly
    from src.engine.signal_tracker import SignalTracker
    tracker = SignalTracker(db)
    stats = tracker.get_performance_stats(days=days)
    result = PerformanceStatsResponse(
        period_days=stats.period_days,
        total_signals=stats.total_signals,
        actionable_signals=stats.actionable_signals,
        total_trades=stats.total_trades,
        wins=stats.wins,
        losses=stats.losses,
        win_rate=stats.win_rate,
        total_pnl=stats.total_pnl,
        avg_pnl_per_trade=stats.avg_pnl_per_trade,
        largest_win=stats.largest_win,
        largest_loss=stats.largest_loss,
        profit_factor=stats.profit_factor,
        max_drawdown=stats.max_drawdown,
        sharpe_ratio=stats.sharpe_ratio,
        high_confidence_win_rate=stats.high_confidence_win_rate,
        medium_confidence_win_rate=stats.medium_confidence_win_rate,
        low_confidence_win_rate=stats.low_confidence_win_rate,
        win_rate_by_index=stats.win_rate_by_index,
        pnl_by_index=stats.pnl_by_index,
        is_profitable=stats.is_profitable,
        edge_comment=stats.edge_comment,
    )
    return _set_cache(cache_key, result)
