"""
Edge Tracker — Phase 9.2 of the Trading Decision Support System.

Monitors whether the strategy's statistical edge is intact, weakening, or gone
by analysing a rolling window of recent paper trades.

Usage (called at market close each day)::

    tracker = EdgeTracker(db)
    assessment = tracker.assess_edge(paper_engine.trade_history[-200:])

    if assessment.edge_status in ("WEAKENING", "GONE"):
        telegram.send_alert("Edge alert: " + assessment.recommendation, priority="HIGH")

    # For dashboard / weekly report
    history = tracker.get_rolling_edge_history(days=30)
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from src.utils.date_utils import get_ist_now

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)


# ---------------------------------------------------------------------------
# EdgeAssessment dataclass
# ---------------------------------------------------------------------------


@dataclass
class EdgeAssessment:
    """Result of one edge assessment over a rolling window of trades."""

    timestamp: datetime
    window_days: int
    trade_count: int

    # Current-window metrics
    win_rate: float           # 0–100
    profit_factor: float      # gross_wins / abs(gross_losses); inf if no losses
    expected_value: float     # mean net P&L per trade (₹)
    sharpe_ratio: float       # annualised per-trade Sharpe

    # Trend vs previous window
    win_rate_trend: str       # "IMPROVING" / "STABLE" / "DECLINING"
    pf_trend: str             # "IMPROVING" / "STABLE" / "DECLINING"

    # Overall assessment
    edge_status: str          # "STRONG" / "INTACT" / "WEAKENING" / "GONE" / "INSUFFICIENT_DATA"
    confidence_in_assessment: float  # 0–1 (scales with trade count)

    # Action guidance
    recommendation: str
    should_pause: bool        # True when edge is confidently GONE
    should_reduce_size: bool  # True when edge is WEAKENING
    should_reoptimize: bool   # True when parameters may be stale


# ---------------------------------------------------------------------------
# EdgeTracker
# ---------------------------------------------------------------------------


class EdgeTracker:
    """Monitors whether the strategy's statistical edge is intact or decaying.

    Uses a rolling window of recent trades to track key metrics over time.
    If metrics deteriorate below thresholds, issues warnings.

    The instance maintains a lightweight in-memory history of past metrics so
    that trend detection works across successive daily calls.

    Parameters
    ----------
    db:
        DatabaseManager instance (used by ``get_rolling_edge_history``).
    """

    # ── Thresholds ──────────────────────────────────────────────────────

    min_win_rate: float = 45.0            # Below → edge questionable
    min_profit_factor: float = 1.0        # Below → losing money
    min_expected_value: float = 0.0       # Below → negative EV
    min_trades_for_assessment: int = 20   # Need at least 20 trades
    rolling_window_days: int = 14         # 2-week rolling window

    # Trend sensitivity: ±5% relative change = STABLE band
    _TREND_STABLE_BAND: float = 0.05

    def __init__(self, db) -> None:
        self.db = db

        # Running history of raw metrics from each assess_edge() call.
        # Each entry: {"win_rate": float, "profit_factor": float, "ev": float}
        self._metric_history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_edge(self, recent_trades: list) -> EdgeAssessment:
        """
        Assess the strategy edge from a list of closed PaperTrade objects.

        Parameters
        ----------
        recent_trades:
            Closed trades for the current rolling window (typically the last
            ``rolling_window_days`` days pulled from ``paper_engine.trade_history``).

        Returns
        -------
        EdgeAssessment
            Full assessment including status, trends, and action guidance.
        """
        metrics = self._compute_metrics(recent_trades)
        trade_count = metrics["trade_count"]

        # Trend vs previous window (requires at least one prior assessment)
        win_rate_trend = self._determine_trend(
            metrics["win_rate"], "win_rate"
        )
        pf_trend = self._determine_trend(
            metrics["profit_factor"], "profit_factor"
        )

        edge_status = self._determine_status(metrics, win_rate_trend)
        confidence = self._confidence(trade_count)

        assessment = EdgeAssessment(
            timestamp=get_ist_now(),
            window_days=self.rolling_window_days,
            trade_count=trade_count,
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            expected_value=metrics["expected_value"],
            sharpe_ratio=metrics["sharpe_ratio"],
            win_rate_trend=win_rate_trend,
            pf_trend=pf_trend,
            edge_status=edge_status,
            confidence_in_assessment=confidence,
            recommendation=self._generate_recommendation(edge_status, metrics),
            should_pause=edge_status == "GONE" and trade_count > 30,
            should_reduce_size=edge_status == "WEAKENING",
            should_reoptimize=edge_status in ("WEAKENING", "GONE"),
        )

        # Persist metrics for future trend detection
        self._metric_history.append({
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "ev": metrics["expected_value"],
        })

        logger.info(
            "EdgeTracker: status=%s win_rate=%.1f%% PF=%.2f EV=₹%.0f"
            " trades=%d confidence=%.2f",
            edge_status,
            metrics["win_rate"],
            metrics["profit_factor"],
            metrics["expected_value"],
            trade_count,
            confidence,
        )

        return assessment

    def get_rolling_edge_history(self, days: int = 30) -> list[EdgeAssessment]:
        """
        Compute a daily edge assessment for each of the last ``days`` days.

        For each calendar day D, the assessment is based on all trades closed
        in the trailing ``rolling_window_days``-day window ending on D.

        Queries the ``trading_signals`` table (populated by ``_persist_trade``).
        Returns oldest-first so it is ready for charting.
        """
        today = get_ist_now().date()
        fetch_from = today - timedelta(days=days + self.rolling_window_days)

        rows = self._fetch_closed_trades_from_db(fetch_from)
        if not rows:
            logger.debug("EdgeTracker: no closed trades in DB for rolling history")
            return []

        history: list[EdgeAssessment] = []
        rolling_metrics: list[dict] = []  # Used for trend detection within the loop

        # Iterate oldest → newest so trend detection is correct
        for offset in range(days - 1, -1, -1):
            window_end = today - timedelta(days=offset)
            window_start = window_end - timedelta(days=self.rolling_window_days)

            window_rows = [
                r for r in rows
                if window_start.isoformat()
                <= (r.get("closed_at") or "")[:10]
                <= window_end.isoformat()
            ]

            metrics = self._compute_metrics_from_dicts(window_rows)
            trade_count = metrics["trade_count"]

            win_rate_trend = self._trend_from_history(
                metrics["win_rate"], rolling_metrics, "win_rate"
            )
            pf_trend = self._trend_from_history(
                metrics["profit_factor"], rolling_metrics, "profit_factor"
            )

            edge_status = self._determine_status_from_history(
                metrics, rolling_metrics, win_rate_trend
            )
            confidence = self._confidence(trade_count)

            ts = datetime.combine(window_end, datetime.min.time()).replace(tzinfo=_IST)
            assessment = EdgeAssessment(
                timestamp=ts,
                window_days=self.rolling_window_days,
                trade_count=trade_count,
                win_rate=metrics["win_rate"],
                profit_factor=metrics["profit_factor"],
                expected_value=metrics["expected_value"],
                sharpe_ratio=metrics["sharpe_ratio"],
                win_rate_trend=win_rate_trend,
                pf_trend=pf_trend,
                edge_status=edge_status,
                confidence_in_assessment=confidence,
                recommendation=self._generate_recommendation(edge_status, metrics),
                should_pause=edge_status == "GONE" and trade_count > 30,
                should_reduce_size=edge_status == "WEAKENING",
                should_reoptimize=edge_status in ("WEAKENING", "GONE"),
            )
            history.append(assessment)
            rolling_metrics.append({
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "ev": metrics["expected_value"],
            })

        return history  # oldest → newest

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(self, trades: list) -> dict:
        """Compute core performance metrics from a list of PaperTrade objects."""
        trade_count = len(trades)
        if trade_count == 0:
            return self._zero_metrics()

        pnls = [float(getattr(t, "net_pnl", 0.0)) for t in trades]
        outcomes = [getattr(t, "outcome", "LOSS") for t in trades]

        return self._metrics_from_pnls_outcomes(pnls, outcomes, trade_count)

    def _compute_metrics_from_dicts(self, rows: list[dict]) -> dict:
        """Compute metrics from DB rows (must have 'outcome' and 'actual_pnl')."""
        trade_count = len(rows)
        if trade_count == 0:
            return self._zero_metrics()

        pnls = [float(r.get("actual_pnl") or 0.0) for r in rows]
        outcomes = [r.get("outcome") or "LOSS" for r in rows]

        return self._metrics_from_pnls_outcomes(pnls, outcomes, trade_count)

    def _metrics_from_pnls_outcomes(
        self, pnls: list[float], outcomes: list[str], trade_count: int
    ) -> dict:
        wins = [p for p, o in zip(pnls, outcomes) if o == "WIN"]
        losses = [p for p, o in zip(pnls, outcomes) if o == "LOSS"]

        win_count = len(wins)
        win_rate = win_count / trade_count * 100.0

        gross_wins = sum(wins) if wins else 0.0
        gross_losses = abs(sum(losses)) if losses else 0.0
        profit_factor = (
            gross_wins / gross_losses if gross_losses > 0 else (math.inf if gross_wins > 0 else 0.0)
        )

        expected_value = sum(pnls) / trade_count if pnls else 0.0

        sharpe = self._compute_sharpe(pnls)

        return {
            "trade_count": trade_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expected_value": expected_value,
            "sharpe_ratio": sharpe,
        }

    @staticmethod
    def _zero_metrics() -> dict:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expected_value": 0.0,
            "sharpe_ratio": 0.0,
        }

    @staticmethod
    def _compute_sharpe(pnls: list[float]) -> float:
        """Per-trade annualised Sharpe ratio (252 trading-day convention)."""
        if len(pnls) < 2:
            return 0.0
        try:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            if std_pnl == 0:
                return 0.0
            # Annualise assuming ~252 trades per year is a rough proxy
            return (mean_pnl / std_pnl) * math.sqrt(252)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Status determination
    # ------------------------------------------------------------------

    def _determine_status(self, metrics: dict, win_rate_trend: str) -> str:
        """Determine edge status using the instance's live metric history."""
        return self._determine_status_from_history(
            metrics, self._metric_history, win_rate_trend
        )

    def _determine_status_from_history(
        self,
        metrics: dict,
        history: list[dict],
        win_rate_trend: str,
    ) -> str:
        trade_count = metrics["trade_count"]
        win_rate = metrics["win_rate"]
        profit_factor = metrics["profit_factor"]
        ev = metrics["expected_value"]

        if trade_count < self.min_trades_for_assessment:
            return "INSUFFICIENT_DATA"

        # GONE: any fundamental failure
        if (
            profit_factor < self.min_profit_factor
            or ev < self.min_expected_value
            or win_rate < self.min_win_rate
        ):
            return "GONE"

        # STRONG: all metrics comfortably above thresholds
        if win_rate > 55 and profit_factor > 1.3 and ev > 0:
            return "STRONG"

        # WEAKENING: 3-window consecutive decline OR PF approaching breakeven
        if self._is_declining_3_windows(metrics["win_rate"], history, "win_rate"):
            return "WEAKENING"
        if profit_factor < 1.1:
            return "WEAKENING"

        return "INTACT"

    # ------------------------------------------------------------------
    # Trend detection
    # ------------------------------------------------------------------

    def _determine_trend(self, current_value: float, metric_key: str) -> str:
        """Compare current value against the last entry in _metric_history."""
        return self._trend_from_history(current_value, self._metric_history, metric_key)

    def _trend_from_history(
        self, current_value: float, history: list[dict], metric_key: str
    ) -> str:
        if not history:
            return "STABLE"

        prev = history[-1].get(metric_key, 0.0)
        if prev == 0:
            return "STABLE"

        relative_change = (current_value - prev) / abs(prev)

        if relative_change > self._TREND_STABLE_BAND:
            return "IMPROVING"
        if relative_change < -self._TREND_STABLE_BAND:
            return "DECLINING"
        return "STABLE"

    def _is_declining_3_windows(
        self, current_value: float, history: list[dict], metric_key: str
    ) -> bool:
        """Return True if the metric has declined over 3 consecutive windows.

        Requires at least 2 historical entries + the current value:
            history[-2] > history[-1] > current_value
        """
        if len(history) < 2:
            return False
        prev1 = history[-1].get(metric_key, 0.0)  # most recent historical
        prev2 = history[-2].get(metric_key, 0.0)  # second most recent
        return prev2 > prev1 > current_value

    # ------------------------------------------------------------------
    # Confidence scaling
    # ------------------------------------------------------------------

    def _confidence(self, trade_count: int) -> float:
        """Scale 0 → 1 as trade_count grows from min_trades to 50."""
        if trade_count < self.min_trades_for_assessment:
            return 0.0
        # 0.5 at min_trades (20), 1.0 at 50+ trades
        scale = (trade_count - self.min_trades_for_assessment) / max(
            50 - self.min_trades_for_assessment, 1
        )
        return round(min(1.0, 0.5 + scale * 0.5), 4)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_recommendation(edge_status: str, metrics: dict) -> str:
        win_rate = metrics.get("win_rate", 0.0)
        pf = metrics.get("profit_factor", 0.0)

        if edge_status == "INSUFFICIENT_DATA":
            return (
                f"Only {metrics.get('trade_count', 0)} trades — need "
                f"{EdgeTracker.min_trades_for_assessment} for a reliable assessment."
            )
        if edge_status == "STRONG":
            return (
                f"Edge is strong (WR={win_rate:.0f}%, PF={pf:.2f}). "
                "Continue at full sizing."
            )
        if edge_status == "INTACT":
            return (
                f"Edge is intact (WR={win_rate:.0f}%, PF={pf:.2f}). "
                "Continue monitoring."
            )
        if edge_status == "WEAKENING":
            return (
                f"Edge weakening (WR={win_rate:.0f}%, PF={pf:.2f}). "
                "Reduce position size and re-run optimizer."
            )
        if edge_status == "GONE":
            return (
                f"Edge gone (WR={win_rate:.0f}%, PF={pf:.2f}). "
                "Pause trading and re-optimize with recent data."
            )
        return "Edge status unknown."

    # ------------------------------------------------------------------
    # DB helper
    # ------------------------------------------------------------------

    def _fetch_closed_trades_from_db(self, since: date) -> list[dict]:
        """
        Query trading_signals for rows with a recorded outcome (closed paper trades).

        Returns list of dicts with at least 'outcome', 'actual_pnl', 'closed_at'.
        """
        try:
            return self.db.fetch_all(
                """
                SELECT outcome, actual_pnl, closed_at, confidence_level
                FROM   trading_signals
                WHERE  closed_at IS NOT NULL
                  AND  outcome   IS NOT NULL
                  AND  closed_at >= ?
                ORDER  BY closed_at ASC
                """,
                (since.isoformat(),),
            )
        except Exception as exc:
            logger.warning("EdgeTracker: DB query failed: %s", exc)
            return []
