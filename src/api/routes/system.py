"""
System health and control API endpoints.

GET    /api/system/health               — system component health
GET    /api/system/status               — overall system status
POST   /api/system/kill-switch          — activate kill switch
DELETE /api/system/kill-switch          — deactivate kill switch
GET    /api/system/config               — approved parameters and settings
GET    /api/system/config/live          — live-tunable settings
POST   /api/system/config/live          — update live settings (no restart)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.dependencies import (
    activate_kill_switch,
    deactivate_kill_switch,
    get_db,
    get_market_hours,
    is_kill_switch_active,
)
from src.database import queries as Q
from src.database.db_manager import DatabaseManager
from src.utils.date_utils import get_ist_now
from src.utils.market_hours import MarketHoursManager

router = APIRouter()
logger = logging.getLogger(__name__)

_start_time = time.monotonic()


# ── Response models ──────────────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    component: str
    status: str
    last_seen: Optional[str] = None
    message: Optional[str] = None


class SystemHealthResponse(BaseModel):
    components: list[ComponentHealth]
    overall_status: str = "OK"
    db_size: Optional[str] = None


class DataFreshness(BaseModel):
    table: str
    latest_timestamp: Optional[str] = None
    age_seconds: Optional[int] = None


class SystemStatusResponse(BaseModel):
    market_status: str = "CLOSED"
    market_session: Optional[str] = None
    next_event: Optional[str] = None
    time_remaining: Optional[str] = None
    uptime_seconds: int = 0
    kill_switch_active: bool = False
    kill_switch_reason: Optional[str] = None
    data_freshness: list[DataFreshness] = Field(default_factory=list)
    timestamp: str = ""


class KillSwitchRequest(BaseModel):
    reason: str = "Manual activation"


class KillSwitchResponse(BaseModel):
    active: bool
    reason: Optional[str] = None
    message: str


class ConfigResponse(BaseModel):
    approved_params: dict = Field(default_factory=dict)


class LiveConfig(BaseModel):
    riskPerTrade: float = 2.0
    minConfidence: str = "MEDIUM"
    maxPositions: int = 3
    activeIndices: list[str] = Field(
        default_factory=lambda: ["NIFTY50", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
    )
    soundEnabled: bool = True
    notificationThreshold: str = "HIGH"


# File-backed live config store (no restart required)
_LIVE_CONFIG_PATH = Path(__file__).resolve().parents[3] / "data" / "live_config.json"

_DEFAULT_LIVE: dict = LiveConfig().model_dump()


def _read_live_config() -> dict:
    if _LIVE_CONFIG_PATH.exists():
        try:
            return {**_DEFAULT_LIVE, **json.loads(_LIVE_CONFIG_PATH.read_text())}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt live_config.json, using defaults: %s", exc)
    return dict(_DEFAULT_LIVE)


def _write_live_config(data: dict) -> None:
    _LIVE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LIVE_CONFIG_PATH.write_text(json.dumps(data, indent=2))


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/health", response_model=SystemHealthResponse, summary="System component health")
async def get_health(db: DatabaseManager = Depends(get_db)):
    rows = db.fetch_all(Q.GET_LATEST_HEALTH_BY_COMPONENT)

    components = [
        ComponentHealth(
            component=r["component"],
            status=r.get("status", "UNKNOWN"),
            last_seen=r.get("last_seen"),
            message=r.get("message"),
        )
        for r in rows
    ]

    # Determine overall status
    statuses = [c.status for c in components]
    if any(s == "ERROR" for s in statuses):
        overall = "ERROR"
    elif any(s == "WARNING" for s in statuses):
        overall = "WARNING"
    elif components:
        overall = "OK"
    else:
        overall = "UNKNOWN"

    return SystemHealthResponse(
        components=components,
        overall_status=overall,
        db_size=db.get_db_size(),
    )


@router.get("/status", response_model=SystemStatusResponse, summary="Overall system status")
async def get_status(
    db: DatabaseManager = Depends(get_db),
    mh: MarketHoursManager = Depends(get_market_hours),
):
    now = get_ist_now()
    market = mh.get_market_status(now)

    ks_active, ks_reason = is_kill_switch_active()
    uptime = int(time.monotonic() - _start_time)

    # Data freshness check for key tables
    freshness: list[DataFreshness] = []
    for table, col in [
        ("price_data", "timestamp"),
        ("options_chain_snapshot", "timestamp"),
        ("news_articles", "published_at"),
        ("anomaly_events", "timestamp"),
        ("vix_data", "timestamp"),
    ]:
        try:
            row = db.fetch_one(
                f"SELECT MAX({col}) AS latest FROM {table}",
            )
            latest = row["latest"] if row and row.get("latest") else None
            age = None
            if latest:
                from datetime import datetime
                from zoneinfo import ZoneInfo
                try:
                    dt = datetime.fromisoformat(latest)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
                    age = int((now - dt).total_seconds())
                except (ValueError, TypeError):
                    pass
            freshness.append(DataFreshness(
                table=table, latest_timestamp=latest, age_seconds=age,
            ))
        except Exception:
            freshness.append(DataFreshness(table=table))

    # Map market session to friendly status
    session = market.get("status", "closed")
    if session == "open":
        market_status = "OPEN"
    elif session == "pre_market":
        market_status = "PRE_MARKET"
    elif session == "post_market":
        market_status = "POST_MARKET"
    elif session in ("holiday", "weekend"):
        market_status = session.upper()
    else:
        market_status = "CLOSED"

    return SystemStatusResponse(
        market_status=market_status,
        market_session=session,
        next_event=market.get("next_event"),
        time_remaining=market.get("time_remaining"),
        uptime_seconds=uptime,
        kill_switch_active=ks_active,
        kill_switch_reason=ks_reason,
        data_freshness=freshness,
        timestamp=now.isoformat(),
    )


@router.post("/kill-switch", response_model=KillSwitchResponse, summary="Activate kill switch")
async def post_kill_switch(body: KillSwitchRequest):
    activate_kill_switch(body.reason)
    logger.warning("KILL SWITCH ACTIVATED: %s", body.reason)
    return KillSwitchResponse(
        active=True,
        reason=body.reason,
        message="Kill switch activated — all trading halted",
    )


@router.delete("/kill-switch", response_model=KillSwitchResponse, summary="Deactivate kill switch")
async def delete_kill_switch():
    deactivate_kill_switch()
    logger.info("Kill switch deactivated")
    return KillSwitchResponse(
        active=False,
        reason=None,
        message="Kill switch deactivated — trading resumed",
    )


@router.get("/config", response_model=ConfigResponse, summary="Approved parameters and settings")
async def get_config():
    try:
        from src.backtest.optimizer.param_applier import ApprovedParameterManager
        mgr = ApprovedParameterManager()
        params = mgr.list_all_approved()
    except Exception as exc:
        logger.warning("Could not load approved params: %s", exc)
        params = {}

    return ConfigResponse(approved_params=params)


@router.get("/config/live", response_model=LiveConfig, summary="Live-tunable settings")
async def get_live_config():
    return LiveConfig(**_read_live_config())


@router.post("/config/live", response_model=LiveConfig, summary="Update live settings (no restart)")
async def update_live_config(body: LiveConfig):
    data = body.model_dump()
    _write_live_config(data)
    logger.info("Live config updated: %s", data)
    return LiveConfig(**data)
