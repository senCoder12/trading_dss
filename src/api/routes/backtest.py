"""
Backtest results and history API endpoints.

POST /api/backtest/run                  — run a backtest (existing)
GET  /api/backtest/latest               — most recent backtest metrics
GET  /api/backtest/optimization         — most recent optimization results
GET  /api/backtest/shadow-report        — latest shadow tracker comparison
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_db, get_index_registry
from src.data.index_registry import IndexRegistry
from src.database.db_manager import DatabaseManager

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Response models ──────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    index_id: str = Field(..., description="Index registry ID")
    start_date: str = Field(..., description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="End date YYYY-MM-DD")
    initial_capital: float = Field(default=100_000.0, ge=1_000.0)
    risk_per_trade_pct: float = Field(default=2.0, ge=0.5, le=10.0)
    mode: str = Field(default="TECHNICAL_ONLY", description="TECHNICAL_ONLY / FULL")
    min_confidence: str = Field(default="MEDIUM", description="LOW / MEDIUM / HIGH")


class BacktestMetricsSummary(BaseModel):
    index_id: str
    total_return_pct: Optional[float] = None
    annualized_return_pct: Optional[float] = None
    total_trades: int = 0
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    strategy_grade: Optional[str] = None
    is_profitable: bool = False
    has_edge: bool = False
    assessment: Optional[str] = None


class OptimizationResult(BaseModel):
    index_id: str
    params: dict = Field(default_factory=dict)
    metrics_at_approval: dict = Field(default_factory=dict)
    robustness_score: Optional[float] = None
    config_hash: Optional[str] = None
    approved_at: Optional[str] = None
    status: str = "UNKNOWN"
    notes: Optional[str] = None


class OptimizationResponse(BaseModel):
    results: list[OptimizationResult]


class ShadowReportResponse(BaseModel):
    index_id: str
    available: bool = False
    message: str = "Shadow report not available"
    data: dict = Field(default_factory=dict)


# ── POST /run (existing functionality) ───────────────────────────────────────

@router.post("/run", summary="Run a backtest")
async def run_backtest(
    request: BacktestRequest,
    registry: IndexRegistry = Depends(get_index_registry),
    db=Depends(get_db),
) -> dict:
    """Run a full backtest for the requested index."""
    defn = registry.get_or_none(request.index_id.upper())
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {request.index_id!r}")

    try:
        start = date.fromisoformat(request.start_date)
        end = date.fromisoformat(request.end_date) if request.end_date else date.today()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date format: {exc}")

    if start >= end:
        raise HTTPException(status_code=422, detail="start_date must be before end_date")

    try:
        from src.backtest.strategy_runner import BacktestConfig, StrategyRunner
        from src.backtest.trade_simulator import SimulatorConfig

        sim_cfg = SimulatorConfig(
            initial_capital=request.initial_capital,
            max_risk_per_trade_pct=request.risk_per_trade_pct,
        )
        config = BacktestConfig(
            index_id=request.index_id.upper(),
            start_date=start,
            end_date=end,
            mode=request.mode,
            min_confidence=request.min_confidence,
            simulator_config=sim_cfg,
            show_progress=False,
        )

        runner = StrategyRunner(db)
        result = runner.run(config)
        m = result.metrics

        return {
            "index_id": result.index_id,
            "start_date": str(result.start_date),
            "end_date": str(result.end_date),
            "total_bars": result.total_bars,
            "trading_days": result.trading_days,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return_pct": result.total_return_pct,
            "total_trades": result.total_trades,
            "backtest_duration_seconds": result.backtest_duration_seconds,
            "warnings": result.warnings,
            "metrics": {
                "total_return_pct": m.total_return_pct,
                "total_return_amount": m.total_return_amount,
                "annualized_return_pct": m.annualized_return_pct,
                "best_month_pct": m.best_month_pct,
                "worst_month_pct": m.worst_month_pct,
                "monthly_win_rate": m.monthly_win_rate,
                "total_trades": m.total_trades,
                "win_rate": m.win_rate,
                "profit_factor": m.profit_factor,
                "payoff_ratio": m.payoff_ratio,
                "expected_value_per_trade": m.expected_value_per_trade,
                "avg_win_amount": m.avg_win_amount,
                "avg_loss_amount": m.avg_loss_amount,
                "max_drawdown_pct": m.max_drawdown_pct,
                "max_drawdown_amount": m.max_drawdown_amount,
                "max_consecutive_losses": m.max_consecutive_losses,
                "sharpe_ratio": m.sharpe_ratio,
                "sortino_ratio": m.sortino_ratio,
                "calmar_ratio": m.calmar_ratio,
                "strategy_grade": m.strategy_grade,
                "is_profitable": m.is_profitable,
                "has_edge": m.has_edge,
                "assessment": m.assessment,
                "monthly_returns": m.monthly_returns,
                "trades_by_regime": m.trades_by_regime,
                "trades_by_day_of_week": m.trades_by_day_of_week,
                "confidence_breakdown": {
                    "high": {
                        "trades": m.high_confidence_trades,
                        "win_rate": m.high_confidence_win_rate,
                        "avg_pnl": m.high_confidence_avg_pnl,
                    },
                    "medium": {
                        "trades": m.medium_confidence_trades,
                        "win_rate": m.medium_confidence_win_rate,
                        "avg_pnl": m.medium_confidence_avg_pnl,
                    },
                    "low": {
                        "trades": m.low_confidence_trades,
                        "win_rate": m.low_confidence_win_rate,
                        "avg_pnl": m.low_confidence_avg_pnl,
                    },
                },
                "exit_breakdown": {
                    "target_hit": {"count": m.target_hit_count, "pct": m.target_hit_pct},
                    "stop_loss": {"count": m.sl_hit_count, "pct": m.sl_hit_pct},
                    "trailing_sl": {"count": m.trailing_sl_count, "pct": m.trailing_sl_pct},
                    "forced_eod": {"count": m.forced_eod_count, "avg_pnl": m.forced_eod_avg_pnl},
                },
            },
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Backtest failed for %s: %s", request.index_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET endpoints ────────────────────────────────────────────────────────────

@router.get(
    "/latest",
    response_model=BacktestMetricsSummary,
    summary="Most recent backtest metrics from approved params",
)
async def get_latest_backtest(
    index_id: str = Query(..., description="Index ID"),
):
    """Return the metrics stored when the last parameter set was approved."""
    index_id = index_id.upper()
    try:
        from src.backtest.optimizer.param_applier import ApprovedParameterManager
        mgr = ApprovedParameterManager()
        all_params = mgr.list_all_approved()
    except Exception:
        all_params = {}

    entry = all_params.get(index_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No backtest results found for {index_id}",
        )

    m = entry.get("metrics_at_approval", {})
    return BacktestMetricsSummary(
        index_id=index_id,
        total_return_pct=m.get("return_pct"),
        annualized_return_pct=None,
        total_trades=m.get("total_trades", 0),
        win_rate=m.get("win_rate"),
        profit_factor=m.get("profit_factor"),
        sharpe_ratio=m.get("sharpe"),
        sortino_ratio=None,
        max_drawdown_pct=m.get("max_drawdown"),
        strategy_grade=None,
        is_profitable=(m.get("return_pct") or 0) > 0,
        has_edge=(m.get("profit_factor") or 0) > 1.0,
        assessment=entry.get("notes"),
    )


@router.get(
    "/optimization",
    response_model=OptimizationResponse,
    summary="Most recent optimization results",
)
async def get_optimization_results(
    index_id: Optional[str] = Query(default=None, description="Filter by index"),
):
    try:
        from src.backtest.optimizer.param_applier import ApprovedParameterManager
        mgr = ApprovedParameterManager()
        all_params = mgr.list_all_approved()
    except Exception:
        all_params = {}

    results: list[OptimizationResult] = []
    for idx, entry in all_params.items():
        if index_id and idx != index_id.upper():
            continue
        results.append(OptimizationResult(
            index_id=idx,
            params=entry.get("params", {}),
            metrics_at_approval=entry.get("metrics_at_approval", {}),
            robustness_score=entry.get("robustness_score"),
            config_hash=entry.get("config_hash"),
            approved_at=entry.get("approved_at"),
            status=entry.get("status", "UNKNOWN"),
            notes=entry.get("notes"),
        ))

    return OptimizationResponse(results=results)


@router.get(
    "/shadow-report",
    response_model=ShadowReportResponse,
    summary="Latest shadow tracker comparison",
)
async def get_shadow_report(
    index_id: str = Query(..., description="Index ID"),
):
    index_id = index_id.upper()

    # Try to load shadow report from saved file
    report_path = Path(f"data/reports/shadow_{index_id}.json")
    if not report_path.exists():
        return ShadowReportResponse(
            index_id=index_id,
            available=False,
            message=f"No shadow report found for {index_id}",
        )

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return ShadowReportResponse(
            index_id=index_id,
            available=True,
            message="Shadow report loaded",
            data=data,
        )
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load shadow report for %s: %s", index_id, exc)
        return ShadowReportResponse(
            index_id=index_id,
            available=False,
            message=f"Error reading shadow report: {exc}",
        )
