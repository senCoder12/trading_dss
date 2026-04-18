"""
Automated Reporting — Phase 9.2 of the Trading Decision Support System.

Generates and sends daily / weekly performance reports.

- Daily report: called at 15:45 IST by the scheduler
- Weekly report: called Saturday morning

Both are sent via Telegram (if configured) and saved to data/reports/.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Optional

from config.settings import STRATEGY_VERSION
from src.utils.date_utils import get_ist_now

logger = logging.getLogger(__name__)

# Severity / edge-status ranking helpers used when aggregating shadow reports
_SEVERITY_RANK: dict[str, int] = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
_EDGE_RANK: dict[str, int] = {"STABLE": 0, "DECAYING": 1, "NO_EDGE": 2}


class AutomatedReporter:
    """Generates and sends daily / weekly performance reports automatically."""

    def __init__(self, db, paper_engine, telegram_bot=None) -> None:
        """
        Parameters
        ----------
        db:
            DatabaseManager instance.
        paper_engine:
            PaperTradingEngine instance.
        telegram_bot:
            Optional TelegramBot instance.  When None, reports are only logged
            and saved to disk.
        """
        self.db = db
        self.paper_engine = paper_engine
        self.telegram = telegram_bot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_daily_report(self) -> str:
        """
        Build the end-of-day report string.

        Called at 15:45 IST by the scheduler.  Pulls today's summary plus
        rolling 7-day stats from the paper engine.
        """
        summary = self.paper_engine.get_daily_summary()
        stats = self.paper_engine.get_cumulative_stats(days=7)

        pnl_emoji = "✅" if summary.total_pnl >= 0 else "❌"
        edge_line = (
            f"⚠️ Edge Status: {stats.edge_status}"
            if stats.edge_status != "INTACT"
            else "✅ Edge: Intact"
        )

        # Benchmark comparison (cumulative since paper trading started)
        benchmark_return = self.paper_engine.get_benchmark_return_pct()
        strategy_cumulative_pct = (
            (self.paper_engine.current_capital - self.paper_engine.config.initial_capital)
            / self.paper_engine.config.initial_capital * 100
            if self.paper_engine.config.initial_capital > 0 else 0.0
        )
        if benchmark_return is not None:
            alpha = strategy_cumulative_pct - benchmark_return
            alpha_icon = "✅" if alpha > 0 else "❌"
            bm_index = getattr(self.paper_engine, '_benchmark_index', 'NIFTY50')
            benchmark_line = (
                f"📊 vs {bm_index} (since start): You {strategy_cumulative_pct:+.2f}%"
                f" | Buy-hold {benchmark_return:+.2f}%"
                f" | Alpha {alpha:+.2f}% {alpha_icon}\n"
            )
        else:
            benchmark_line = ""

        report = (
            f"\n📊 Daily Report — {summary.date.strftime('%d %b %Y')} (v{STRATEGY_VERSION})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"\n"
            f"{pnl_emoji} Today's P&L: ₹{summary.total_pnl:+,.0f}"
            f" ({summary.total_pnl_pct:+.2f}%)\n"
            f"Capital: ₹{summary.ending_capital:,.0f}\n"
            f"{benchmark_line}"
            f"\n"
            f"📈 Trades: {summary.trades_taken}\n"
            f"   Won: {summary.trades_won} | Lost: {summary.trades_lost}\n"
            f"   Win Rate: {summary.win_rate:.0f}%\n"
            f"   Best: ₹{summary.best_trade_pnl:+,.0f}"
            f" | Worst: ₹{summary.worst_trade_pnl:+,.0f}\n"
            f"\n"
            f"📡 Signals: {summary.signals_generated} generated\n"
            f"   Filtered: {summary.signals_filtered}"
            f" | Executed: {summary.signals_executed}\n"
            f"   Missed: {summary.signals_missed}\n"
            f"\n"
            f"📊 7-Day Rolling:\n"
            f"   Return: {stats.total_return_pct:+.2f}%"
            f" | Win Rate: {stats.win_rate:.0f}%\n"
            f"   Sharpe: {stats.sharpe_ratio:.2f}"
            f" | Max DD: -{stats.max_drawdown_pct:.1f}%\n"
            f"\n"
            f"🔧 System: {summary.system_mode}\n"
            f"   Data OK: {'✅' if summary.data_freshness_ok else '⚠️'}\n"
            f"   Anomalies: {summary.anomaly_alerts_count}\n"
            f"\n"
            f"{edge_line}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        return report

    def generate_weekly_report(self) -> str:
        """
        Build the weekly summary report string.

        Called Saturday morning by the scheduler.  Includes shadow-tracker
        comparison (live vs retrospective backtest) for all active indices.
        """
        stats = self.paper_engine.get_cumulative_stats(days=7)
        shadow = self._get_weekly_shadow()
        recs = self._generate_weekly_recommendations(stats, shadow)

        # Version comparison section
        version_section = ""
        try:
            from src.engine.signal_tracker import SignalTracker
            tracker = SignalTracker(self.db)
            by_version = tracker.get_performance_by_version(days=90)
            if len(by_version) >= 2:
                version_section = (
                    f"\n📊 Version Performance (90 days):\n"
                    + "\n".join(
                        f"   v{v}: {s.total_trades} trades, {s.win_rate:.0f}% WR,"
                        f" PF {s.profit_factor:.2f}, {s.total_return_pct:+.1f}%"
                        + (" ← Active" if v == STRATEGY_VERSION else "")
                        for v, s in sorted(by_version.items())
                    )
                    + "\n"
                )
        except Exception:
            pass

        report = (
            f"\n📊 Weekly Report — Week of {date.today().strftime('%d %b %Y')} (v{STRATEGY_VERSION})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"\n"
            f"💰 Performance:\n"
            f"   Return: {stats.total_return_pct:+.2f}%"
            f" (₹{stats.current_capital - stats.initial_capital:+,.0f})\n"
            f"   Trades: {stats.total_trades} | Win Rate: {stats.win_rate:.1f}%\n"
            f"   Profit Factor: {stats.profit_factor:.2f}\n"
            f"   Expected Value: ₹{stats.expected_value:+,.0f}/trade\n"
            f"\n"
            f"📉 Risk:\n"
            f"   Max Drawdown: -{stats.max_drawdown_pct:.1f}%\n"
            f"   Sharpe: {stats.sharpe_ratio:.2f} | Sortino: {stats.sortino_ratio:.2f}\n"
            f"   Losing Days: "
            f"{stats.trading_days - stats.profitable_days}/{stats.trading_days}\n"
            f"\n"
            f"🎯 By Confidence:\n"
            f"   HIGH:   {stats.high_conf_trades} trades,"
            f" {stats.high_conf_win_rate:.0f}% WR\n"
            f"   MEDIUM: {stats.medium_conf_trades} trades,"
            f" {stats.medium_conf_win_rate:.0f}% WR\n"
            f"\n"
            f"🔍 Shadow Check (Live vs Backtest):\n"
            f"   Live: ₹{shadow['live_pnl']:+,.0f}"
            f" | Backtest: ₹{shadow['backtest_pnl']:+,.0f}\n"
            f"   Divergence: {shadow['divergence_severity']}\n"
            f"   Edge: {shadow['edge_status']}\n"
            f"\n"
            f"📋 Recommendations:\n"
            f"{recs}\n"
            f"{version_section}"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        return report

    def send_daily_report(self) -> None:
        """Generate, send via Telegram, and persist the daily report."""
        report = self.generate_daily_report()
        if self.telegram:
            try:
                self.telegram.send_alert(report, priority="LOW")
            except Exception as exc:
                logger.error("Failed to send daily report via Telegram: %s", exc)
        logger.info("Daily report:\n%s", report)
        self._save_report(report, f"daily_{date.today()}.txt")

    def send_weekly_report(self) -> None:
        """Generate, send via Telegram, and persist the weekly report."""
        report = self.generate_weekly_report()
        if self.telegram:
            try:
                self.telegram.send_alert(report, priority="NORMAL")
            except Exception as exc:
                logger.error("Failed to send weekly report via Telegram: %s", exc)
        logger.info("Weekly report:\n%s", report)
        self._save_report(report, f"weekly_{date.today()}.txt")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_weekly_shadow(self) -> dict:
        """
        Run ShadowTracker for all active indices and return an aggregated result.

        If the shadow tracker raises (e.g. insufficient data), the function logs
        a warning and returns safe defaults rather than crashing the report.
        """
        from src.engine.shadow_tracker import ShadowTracker

        tracker = ShadowTracker(self.db)
        today = date.today()

        active_indices: list[str] = (
            getattr(getattr(self.paper_engine, "config", None), "active_indices", None)
            or ["NIFTY50", "BANKNIFTY"]
        )

        live_total = 0.0
        backtest_total = 0.0
        severities: list[str] = []
        edge_statuses: list[str] = []

        for index_id in active_indices:
            try:
                report = tracker.run_weekly_comparison(
                    index_id=index_id,
                    week_end_date=today,
                )
                live_total += report.live_pnl
                backtest_total += report.backtest_pnl
                severities.append(report.divergence_severity)
                edge_statuses.append(report.edge_status)
            except Exception as exc:
                logger.warning(
                    "Shadow tracker failed for %s: %s", index_id, exc
                )

        # Aggregate: surface the worst severity / edge status seen
        divergence_severity = max(
            severities,
            key=lambda s: _SEVERITY_RANK.get(s, 0),
            default="LOW",
        )
        edge_status = max(
            edge_statuses,
            key=lambda s: _EDGE_RANK.get(s, 0),
            default="STABLE",
        )

        return {
            "live_pnl": live_total,
            "backtest_pnl": backtest_total,
            "divergence_severity": divergence_severity,
            "edge_status": edge_status,
        }

    def _generate_weekly_recommendations(self, stats, shadow: dict) -> str:
        recs: list[str] = []

        if stats.win_rate < 45:
            recs.append(
                "   ⚠️ Win rate below 45% — consider pausing and re-optimizing"
            )

        if stats.max_drawdown_pct > 10:
            recs.append(
                "   ⚠️ Drawdown exceeding 10% — reduce position sizes"
            )

        if (
            stats.high_conf_win_rate > 65
            and stats.medium_conf_trades > 0
            and stats.medium_conf_win_rate < 45
        ):
            recs.append(
                "   💡 HIGH conf signals outperforming. Consider filtering to HIGH only."
            )

        if shadow.get("edge_status") == "DECAYING":
            recs.append(
                "   🔴 Edge may be decaying — re-run optimizer with recent data"
            )

        if stats.trading_days > 0:
            day_win_rate = stats.profitable_days / stats.trading_days
            if day_win_rate > 0.6:
                recs.append("   ✅ Consistent performance — system operating well")

        if not recs:
            recs.append("   ✅ All metrics within acceptable range")

        return "\n".join(recs)

    def _save_report(self, report: str, filename: str) -> None:
        """Persist the report text to data/reports/<filename>."""
        os.makedirs("data/reports", exist_ok=True)
        path = os.path.join("data", "reports", filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)
        logger.info("Report saved to %s", path)
