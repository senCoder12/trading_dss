"""
Shadow Tracker — compares live signal generation against retrospective backtesting.

Runs weekly: takes the past week's data, backtests it, and compares
the backtest results against what the live system actually produced.

This detects:
- **Overfitting**: backtest looks great but live doesn't match
- **Bugs**: live system produces different signals than backtest on same data
- **Regime shifts**: strategy edge is decaying

Schedule ``run_weekly_comparison()`` every Saturday at 10:00 AM IST.
If divergence severity is HIGH or CRITICAL, fire an alert.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class ShadowReport:
    """Weekly shadow-comparison report: live vs retrospective backtest."""

    index_id: str
    period_start: date
    period_end: date

    # Live performance
    live_trades: int
    live_pnl: float
    live_win_rate: float

    # Backtest performance
    backtest_trades: int
    backtest_pnl: float
    backtest_win_rate: float

    # Divergence metrics
    trade_count_diff: int
    pnl_diff: float
    wr_diff: float

    divergence_score: float          # 0.0 – 1.0
    divergence_severity: str         # LOW | MODERATE | HIGH | CRITICAL
    edge_status: str                 # STABLE | DECAYING | NO_EDGE
    issues: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        """Serialise for DB / JSON storage."""
        return {
            "index_id": self.index_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "live_trades": self.live_trades,
            "live_pnl": self.live_pnl,
            "live_win_rate": self.live_win_rate,
            "backtest_trades": self.backtest_trades,
            "backtest_pnl": self.backtest_pnl,
            "backtest_win_rate": self.backtest_win_rate,
            "trade_count_diff": self.trade_count_diff,
            "pnl_diff": self.pnl_diff,
            "wr_diff": self.wr_diff,
            "divergence_score": self.divergence_score,
            "divergence_severity": self.divergence_severity,
            "edge_status": self.edge_status,
            "issues": self.issues,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Shadow Tracker
# ---------------------------------------------------------------------------


class ShadowTracker:
    """Compares live signal generation against retrospective backtesting.

    Runs weekly: takes the past week's data, backtests it, and compares
    the backtest results against what the live system actually produced.

    This detects:
    - Overfitting: backtest looks great but live doesn't match
    - Bugs: live system produces different signals than backtest on same data
    - Regime shifts: strategy edge is decaying
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_weekly_comparison(
        self,
        index_id: str,
        week_end_date: date,
    ) -> ShadowReport:
        """Run shadow comparison for the past week.

        Call this every Friday after market close (or Saturday morning).

        Parameters
        ----------
        index_id:
            Index to compare (e.g. ``"NIFTY50"``).
        week_end_date:
            Last day of the comparison window (typically Friday).
            The window covers ``week_end_date - 6`` → ``week_end_date``.
        """
        week_start = week_end_date - timedelta(days=6)

        # Step 1: fetch what the live system actually signalled
        live_signals = self._get_live_signals(index_id, week_start, week_end_date)

        # Step 2: run a retrospective backtest over the same period
        bt_result, bt_metrics = self._run_backtest(index_id, week_start, week_end_date)

        # Step 3: compare
        report = self._compare(
            index_id, week_start, week_end_date,
            live_signals, bt_result, bt_metrics,
        )

        # Step 4: persist
        self._store_report(report)

        logger.info(
            "Shadow report for %s (%s → %s): severity=%s score=%.2f",
            index_id, week_start, week_end_date,
            report.divergence_severity, report.divergence_score,
        )
        return report

    def get_weekly_report_text(self, report: ShadowReport) -> str:
        """Format a shadow report for Telegram / dashboard display."""
        emoji = {
            "LOW": "\u2705",        # ✅
            "MODERATE": "\u26a0\ufe0f",  # ⚠️
            "HIGH": "\U0001f536",    # 🔶
            "CRITICAL": "\U0001f6a8",  # 🚨
        }
        e = emoji.get(report.divergence_severity, "\u2753")

        lines = [
            f"{e} Shadow Report \u2014 {report.index_id}",
            f"Week: {report.period_start} \u2192 {report.period_end}",
            "\u2501" * 26,
            "         LIVE        BACKTEST",
            f"Trades:  {report.live_trades:<12}{report.backtest_trades}",
            f"P&L:     \u20b9{report.live_pnl:<+10,.0f}  \u20b9{report.backtest_pnl:+,.0f}",
            f"Win %:   {report.live_win_rate:<10.0f}%  {report.backtest_win_rate:.0f}%",
            "",
            f"Divergence: {report.divergence_severity} ({report.divergence_score:.2f})",
            f"Edge: {report.edge_status}",
            "",
            report.recommendation,
            "\u2501" * 26,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal — data retrieval
    # ------------------------------------------------------------------

    def _get_live_signals(
        self,
        index_id: str,
        start: date,
        end: date,
    ) -> list[dict]:
        """Fetch signals the live system actually generated."""
        rows = self.db.fetch_all(
            """SELECT signal_type, confidence_level,
                      entry_price, target_price, stop_loss,
                      outcome, actual_pnl, generated_at
               FROM trading_signals
               WHERE index_id = ?
                 AND DATE(generated_at) BETWEEN ? AND ?
                 AND signal_type != 'NO_TRADE'
               ORDER BY generated_at""",
            (index_id, start.isoformat(), end.isoformat()),
        )
        return rows or []

    def _run_backtest(self, index_id: str, start: date, end: date):
        """Run a retrospective backtest over the given period."""
        from src.backtest.strategy_runner import BacktestConfig, StrategyRunner
        from src.backtest.trade_simulator import SimulatorConfig
        from src.backtest.metrics import MetricsCalculator

        runner = StrategyRunner(self.db)
        config = BacktestConfig(
            index_id=index_id,
            start_date=start,
            end_date=end,
            timeframe="1d",
            mode="TECHNICAL_ONLY",
            simulator_config=SimulatorConfig(initial_capital=100_000),
            min_confidence="MEDIUM",
            show_progress=False,
        )

        bt_result = runner.run(config)
        bt_metrics = MetricsCalculator.calculate_all(
            bt_result.trade_history,
            bt_result.equity_curve,
            100_000,
        )
        return bt_result, bt_metrics

    # ------------------------------------------------------------------
    # Internal — comparison
    # ------------------------------------------------------------------

    def _compare(
        self,
        index_id: str,
        start: date,
        end: date,
        live_signals: list[dict],
        bt_result,
        bt_metrics,
    ) -> ShadowReport:
        """Compare live vs backtest results and score the divergence."""
        live_trade_count = len(live_signals)
        bt_trade_count = bt_result.executed_trades

        live_pnl = sum(s.get("actual_pnl", 0) or 0 for s in live_signals)
        bt_pnl = bt_metrics.total_return_amount if bt_metrics else 0.0

        live_wins = sum(1 for s in live_signals if (s.get("actual_pnl") or 0) > 0)
        live_wr = (live_wins / live_trade_count * 100) if live_trade_count > 0 else 0.0
        bt_wr = bt_metrics.win_rate if bt_metrics else 0.0

        # ----- divergence scoring -----
        trade_count_diff = abs(live_trade_count - bt_trade_count)
        pnl_diff = abs(live_pnl - bt_pnl)
        wr_diff = abs(live_wr - bt_wr)

        divergence_score = 0.0
        issues: list[str] = []

        # Trade-count divergence
        max_trades = max(live_trade_count, bt_trade_count, 1)
        if trade_count_diff > max_trades * 0.5:
            divergence_score += 0.3
            issues.append(
                f"Trade count divergence: live={live_trade_count}, "
                f"backtest={bt_trade_count}"
            )

        # Win-rate and P&L divergence (only meaningful when both sides trade)
        if live_trade_count > 0 and bt_trade_count > 0:
            if wr_diff > 20:
                divergence_score += 0.3
                issues.append(
                    f"Win rate divergence: live={live_wr:.0f}%, "
                    f"backtest={bt_wr:.0f}%"
                )

            max_pnl = abs(max(live_pnl, bt_pnl, 1))
            if pnl_diff > max_pnl * 0.5:
                divergence_score += 0.2
                issues.append(
                    f"P&L divergence: live=\u20b9{live_pnl:+,.0f}, "
                    f"backtest=\u20b9{bt_pnl:+,.0f}"
                )

        # Edge-decay assessment
        edge_status = "STABLE"
        if bt_pnl > 0 and live_pnl < 0:
            edge_status = "DECAYING"
            divergence_score += 0.2
            issues.append(
                "Backtest profitable but live is losing "
                "\u2014 possible overfitting or regime shift"
            )
        elif bt_pnl < 0 and live_pnl < 0:
            edge_status = "NO_EDGE"
            issues.append(
                "Both backtest and live are losing "
                "\u2014 strategy may need review"
            )

        # Severity classification
        if divergence_score > 0.6:
            severity = "CRITICAL"
        elif divergence_score > 0.3:
            severity = "HIGH"
        elif divergence_score > 0.1:
            severity = "MODERATE"
        else:
            severity = "LOW"

        recommendation = self._get_recommendation(severity, edge_status, issues)

        return ShadowReport(
            index_id=index_id,
            period_start=start,
            period_end=end,
            live_trades=live_trade_count,
            live_pnl=live_pnl,
            live_win_rate=live_wr,
            backtest_trades=bt_trade_count,
            backtest_pnl=bt_pnl,
            backtest_win_rate=bt_wr,
            trade_count_diff=trade_count_diff,
            pnl_diff=pnl_diff,
            wr_diff=wr_diff,
            divergence_score=divergence_score,
            divergence_severity=severity,
            edge_status=edge_status,
            issues=issues,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Internal — recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _get_recommendation(
        severity: str,
        edge_status: str,
        issues: list[str],
    ) -> str:
        if severity == "CRITICAL":
            return (
                "CRITICAL DIVERGENCE: Live performance is significantly different "
                "from backtest. Possible causes: look-ahead bias in backtest, bug "
                "in live system, or major regime shift. ACTION: Review system logs, "
                "compare exact signals, consider pausing live trading."
            )
        if severity == "HIGH":
            return (
                "HIGH DIVERGENCE: Notable gap between live and backtest. "
                "Monitor closely for the next week. If divergence persists, "
                "review strategy parameters."
            )
        if edge_status == "DECAYING":
            return (
                "EDGE DECAY DETECTED: Strategy is losing money live while backtest "
                "shows profit. This often indicates overfitting. Consider re-running "
                "walk-forward validation with recent data."
            )
        if edge_status == "NO_EDGE":
            return (
                "NO EDGE: Both live and backtest are negative. Strategy may not "
                "work in current market conditions. Consider pausing and "
                "re-evaluating."
            )
        return "System performing within expected range. Continue monitoring."

    # ------------------------------------------------------------------
    # Internal — persistence
    # ------------------------------------------------------------------

    def _store_report(self, report: ShadowReport) -> None:
        """Persist shadow report as a system_health row with JSON payload."""
        status = "OK"
        if report.divergence_severity in ("HIGH", "CRITICAL"):
            status = "ERROR"
        elif report.divergence_severity == "MODERATE":
            status = "WARNING"

        self.db.execute(
            """INSERT INTO system_health (timestamp, component, status, message)
               VALUES (?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                f"shadow_tracker:{report.index_id}",
                status,
                json.dumps(report.to_dict()),
            ),
        )
