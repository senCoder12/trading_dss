"""
Backtest API endpoints.

POST /api/v1/backtest/run    — run a backtest for an index
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_db, get_index_registry
from src.data.index_registry import IndexRegistry

router = APIRouter()
logger = logging.getLogger(__name__)


class BacktestRequest(BaseModel):
    """Request body for a backtest run."""

    index_id: str = Field(..., description="Index registry ID")
    start_date: str = Field(..., description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="End date YYYY-MM-DD")
    initial_capital: float = Field(default=100_000.0, ge=1_000.0)
    risk_per_trade_pct: float = Field(default=2.0, ge=0.5, le=10.0)
    mode: str = Field(default="TECHNICAL_ONLY", description="TECHNICAL_ONLY / FULL")
    min_confidence: str = Field(default="MEDIUM", description="LOW / MEDIUM / HIGH")


@router.post("/run", summary="Run a backtest")
async def run_backtest(
    request: BacktestRequest,
    registry: IndexRegistry = Depends(get_index_registry),
    db=Depends(get_db),
) -> dict:
    """
    Run a full backtest for the requested index using StrategyRunner
    (technical analysis + regime detection + signal generation).

    Returns trade history summary and all performance metrics
    computed by MetricsCalculator.
    """
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
        m = result.metrics  # always populated by _build_result

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
                # Returns
                "total_return_pct": m.total_return_pct,
                "total_return_amount": m.total_return_amount,
                "annualized_return_pct": m.annualized_return_pct,
                "best_month_pct": m.best_month_pct,
                "worst_month_pct": m.worst_month_pct,
                "monthly_win_rate": m.monthly_win_rate,
                # Trades
                "total_trades": m.total_trades,
                "win_rate": m.win_rate,
                "profit_factor": m.profit_factor,
                "payoff_ratio": m.payoff_ratio,
                "expected_value_per_trade": m.expected_value_per_trade,
                "avg_win_amount": m.avg_win_amount,
                "avg_loss_amount": m.avg_loss_amount,
                # Risk
                "max_drawdown_pct": m.max_drawdown_pct,
                "max_drawdown_amount": m.max_drawdown_amount,
                "max_consecutive_losses": m.max_consecutive_losses,
                # Risk-adjusted
                "sharpe_ratio": m.sharpe_ratio,
                "sortino_ratio": m.sortino_ratio,
                "calmar_ratio": m.calmar_ratio,
                # Grade
                "strategy_grade": m.strategy_grade,
                "is_profitable": m.is_profitable,
                "has_edge": m.has_edge,
                "assessment": m.assessment,
                # Breakdowns
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
