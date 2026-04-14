"""
Anomaly alert API endpoints.

GET /api/anomalies/active               — active anomaly alerts
GET /api/anomalies/history              — anomaly alert history
GET /api/anomalies/dashboard            — anomaly overview across all indices
"""

from __future__ import annotations

import json
import logging
import time
from datetime import timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_db
from src.database import queries as Q
from src.database.db_manager import DatabaseManager
from src.utils.date_utils import get_ist_now

router = APIRouter()
logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: int = 30):
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < ttl:
        return entry[1]
    return None


def _set_cache(key: str, value: Any) -> Any:
    _cache[key] = (time.monotonic(), value)
    return value


# ── Response models ──────────────────────────────────────────────────────────

class AnomalyEvent(BaseModel):
    id: Optional[int] = None
    index_id: str
    timestamp: Optional[str] = None
    anomaly_type: str
    severity: str
    category: str
    message: Optional[str] = None
    details: Optional[dict] = None
    is_active: bool = True


class ActiveAnomaliesResponse(BaseModel):
    anomalies: list[AnomalyEvent]
    count: int


class AnomalyHistoryResponse(BaseModel):
    anomalies: list[AnomalyEvent]
    count: int
    total: int


class AnomalyDashboard(BaseModel):
    total_active: int = 0
    high_severity: int = 0
    medium_severity: int = 0
    low_severity: int = 0
    by_category: dict = Field(default_factory=dict)
    by_index: dict = Field(default_factory=dict)
    recent_alerts: list[AnomalyEvent] = Field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_details(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None


def _row_to_event(row: dict) -> AnomalyEvent:
    return AnomalyEvent(
        id=row.get("id"),
        index_id=row["index_id"],
        timestamp=row.get("timestamp"),
        anomaly_type=row.get("anomaly_type", "UNKNOWN"),
        severity=row.get("severity", "LOW"),
        category=row.get("category", "OTHER"),
        message=row.get("message"),
        details=_parse_details(row.get("details")),
        is_active=bool(row.get("is_active", True)),
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/active", response_model=ActiveAnomaliesResponse, summary="Active anomaly alerts")
async def get_active_anomalies(
    index_id: Optional[str] = Query(default=None),
    min_severity: Optional[str] = Query(default=None, description="HIGH, MEDIUM, LOW"),
    category: Optional[str] = Query(default=None, description="VOLUME, PRICE, OI, FII_DII, DIVERGENCE"),
    db: DatabaseManager = Depends(get_db),
):
    rows = db.fetch_all(Q.LIST_ACTIVE_ANOMALIES_FILTERED)

    # Apply filters in Python (the query already orders by severity)
    if index_id:
        idx = index_id.upper()
        rows = [r for r in rows if r["index_id"] == idx]

    if min_severity:
        sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        threshold = sev_order.get(min_severity.upper(), 2)
        rows = [r for r in rows if sev_order.get(r.get("severity"), 2) <= threshold]

    if category:
        cat = category.upper()
        rows = [r for r in rows if r.get("category") == cat]

    events = [_row_to_event(r) for r in rows]
    return ActiveAnomaliesResponse(anomalies=events, count=len(events))


@router.get("/history", response_model=AnomalyHistoryResponse, summary="Anomaly alert history")
async def get_anomaly_history(
    days: int = Query(default=7, ge=1, le=90),
    index_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: DatabaseManager = Depends(get_db),
):
    now = get_ist_now()
    since = (now - timedelta(days=days)).isoformat()

    clauses = ["timestamp >= ?"]
    params: list = [since]

    if index_id:
        clauses.append("index_id = ?")
        params.append(index_id.upper())

    where = " AND ".join(clauses)

    count_row = db.fetch_one(
        f"SELECT COUNT(*) AS cnt FROM anomaly_events WHERE {where}",
        tuple(params),
    )
    total = count_row["cnt"] if count_row else 0

    rows = db.fetch_all(
        f"SELECT * FROM anomaly_events WHERE {where} "
        f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (*params, limit, offset),
    )

    events = [_row_to_event(r) for r in rows]
    return AnomalyHistoryResponse(anomalies=events, count=len(events), total=total)


@router.get("/dashboard", response_model=AnomalyDashboard, summary="Anomaly overview across all indices")
async def get_anomaly_dashboard(db: DatabaseManager = Depends(get_db)):
    cached = _cached("anomaly_dashboard", ttl=30)
    if cached:
        return cached

    now = get_ist_now()
    since_24h = (now - timedelta(hours=24)).isoformat()

    # All active anomalies
    active = db.fetch_all(Q.LIST_ACTIVE_ANOMALIES)

    total_active = len(active)
    high = sum(1 for r in active if r.get("severity") == "HIGH")
    medium = sum(1 for r in active if r.get("severity") == "MEDIUM")
    low = sum(1 for r in active if r.get("severity") == "LOW")

    # By category
    by_category: dict[str, int] = {}
    for r in active:
        cat = r.get("category", "OTHER")
        by_category[cat] = by_category.get(cat, 0) + 1

    # By index
    by_index: dict[str, int] = {}
    for r in active:
        idx = r["index_id"]
        by_index[idx] = by_index.get(idx, 0) + 1

    # Recent alerts (last 24h, up to 10)
    recent_rows = db.fetch_all(
        "SELECT * FROM anomaly_events WHERE timestamp >= ? "
        "ORDER BY timestamp DESC LIMIT 10",
        (since_24h,),
    )
    recent = [_row_to_event(r) for r in recent_rows]

    result = AnomalyDashboard(
        total_active=total_active,
        high_severity=high,
        medium_severity=medium,
        low_severity=low,
        by_category=by_category,
        by_index=by_index,
        recent_alerts=recent,
    )
    return _set_cache("anomaly_dashboard", result)
