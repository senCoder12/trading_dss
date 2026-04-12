"""
Master Decision Engine — Phase 5 orchestrator for the Trading DSS.

Ties together every upstream component (Phase 1 data, Phase 2 technical
analysis, Phase 3 news, Phase 4 anomaly detection, Phase 5.1 regime
detection, Phase 5.2 signal generation, Phase 5.3 risk management) into
a single, production-ready decision loop.

Integration with DataCollector
------------------------------
Add these jobs to ``src/data/data_collector.py``:

    # In DataCollector.__init__:
    #   self.decision_engine = DecisionEngine(self.db)
    #
    # Job: run_decision_cycle — every 5 minutes during market hours
    # def _run_decision_cycle(self):
    #     results = self.decision_engine.run_all_indices()
    #     for result in results:
    #         if result.is_actionable:
    #             self.telegram.send_signal(result.alert_message)
    #             logger.info(
    #                 "SIGNAL: %s %s conf=%s",
    #                 result.signal.signal_type, result.index_id,
    #                 result.signal.confidence_level,
    #             )
    #
    # Job: monitor_positions — every 60 seconds during market hours
    # def _monitor_positions(self):
    #     self.decision_engine.monitor_open_positions()
    #
    # Job: end_of_day_report — at 15:45 IST
    # def _end_of_day_report(self):
    #     pnl  = self.decision_engine.risk_manager.get_daily_pnl_summary()
    #     stats = self.decision_engine.tracker.get_performance_stats(days=1)
    #     self.telegram.send_daily_report(pnl, stats)
"""

from __future__ import annotations

import logging
import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd

from config.constants import IST_TIMEZONE
from config.settings import KILL_SWITCH_FILE
from src.data.index_registry import get_registry
from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from src.analysis.technical_aggregator import TechnicalAggregator, TechnicalAnalysisResult
from src.analysis.news.news_engine import NewsEngine, NewsVote
from src.analysis.anomaly.anomaly_aggregator import (
    AnomalyAggregator, AnomalyDetectionResult, AnomalyVote,
)
from src.engine.regime_detector import RegimeDetector, MarketRegime, SignalWeights
from src.engine.signal_generator import SignalGenerator, TradingSignal
from src.engine.risk_manager import RiskManager, RefinedSignal, RiskConfig
from src.data.rate_limiter import freshness_tracker
from src.engine.signal_tracker import SignalTracker

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Delay between consecutive index analyses in run_all_indices (seconds)
_INTER_INDEX_DELAY: float = 2.0
# Trigger a new news cycle if the last one is older than this
_NEWS_STALE_SECONDS: int = 300   # 5 minutes
# Minimum daily bars needed before analysis is meaningful
_MIN_DAILY_BARS: int = 50
_MIN_BARS_FALLBACK: int = 5     # absolute floor — below this, return NO_TRADE


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DecisionResult:
    """
    Complete output of a single ``run_full_cycle()`` call.

    Contains the final signal, every intermediate result, per-step timing,
    and a pre-formatted alert message ready for Telegram / dashboard display.
    """

    index_id: str
    timestamp: datetime

    # ── The final answer ──────────────────────────────────────────────
    signal: Union[RefinedSignal, TradingSignal]
    is_actionable: bool   # True iff BUY_CALL/BUY_PUT and passed risk checks

    # ── Intermediate results ──────────────────────────────────────────
    technical_result: Optional[TechnicalAnalysisResult] = None
    news_vote: Optional[NewsVote] = None
    anomaly_vote: Optional[AnomalyVote] = None
    regime: Optional[MarketRegime] = None

    # ── Timing ────────────────────────────────────────────────────────
    step_timings: dict = field(default_factory=dict)   # {step_name: ms}
    total_duration_ms: int = 0

    # ── Alert ─────────────────────────────────────────────────────────
    alert_message: Optional[str] = None
    alert_priority: str = "NONE"   # CRITICAL / HIGH / NORMAL / NONE

    # ── Condensed dashboard summary ───────────────────────────────────
    dashboard_summary: dict = field(default_factory=dict)

    # ── Data-quality warnings (e.g. stale feeds) ─────────────────────
    warnings: list[str] = field(default_factory=list)


@dataclass
class IndexDashboard:
    """Per-index row on the frontend dashboard."""

    index_id: str
    current_price: float
    change_pct: float
    regime: str
    signal: str          # BUY_CALL / BUY_PUT / NO_TRADE
    confidence: str      # HIGH / MEDIUM / LOW / -
    key_levels: dict     # {support: float, resistance: float}
    active_alerts: int
    risk_level: str      # NORMAL / ELEVATED / HIGH / EXTREME


@dataclass
class DashboardData:
    """Comprehensive data package for the frontend dashboard."""

    timestamp: datetime
    market_status: str   # OPEN / CLOSED / PRE_MARKET

    indices: list[IndexDashboard] = field(default_factory=list)
    active_signals: list[dict] = field(default_factory=list)

    today_pnl: float = 0.0
    today_trades: int = 0
    today_win_rate: float = 0.0

    market_sentiment: str = "NEUTRAL"
    vix_value: float = 0.0
    vix_regime: str = "NORMAL"
    fii_bias: str = "NEUTRAL"

    recent_alerts: list[dict] = field(default_factory=list)
    top_news: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------


class DecisionEngine:
    """
    Master orchestrator for the Trading Decision Support System.

    Initialises all Phase 1-5 sub-components exactly once and exposes a
    ``run_full_cycle()`` method that executes the complete 8-step analysis
    pipeline for one index.  Each step is isolated in try/except so a
    single component failure never crashes the full pipeline.

    Parameters
    ----------
    db:
        Shared :class:`DatabaseManager` instance.
    risk_config:
        Optional custom :class:`RiskConfig`.  Conservative defaults are used
        when omitted.
    """

    def __init__(
        self,
        db: DatabaseManager,
        risk_config: Optional[RiskConfig] = None,
    ) -> None:
        self._db = db
        self._lock = threading.Lock()

        # ── Sub-components ────────────────────────────────────────────
        self.registry = get_registry()

        self.tech_aggregator = TechnicalAggregator()
        self.news_engine = NewsEngine(db)
        self.anomaly_engine = AnomalyAggregator(db)
        self.regime_detector = RegimeDetector(db)
        self.signal_generator = SignalGenerator(db)
        self.risk_manager = RiskManager(db, config=risk_config)

        _cap = (risk_config or RiskConfig()).total_capital
        self.tracker = SignalTracker(db, capital=_cap)

        # ── Staleness tracking ────────────────────────────────────────
        self._last_news_cycle: Optional[datetime] = None

        # ── Per-index result cache ────────────────────────────────────
        # Results older than _CACHE_TTL_SECONDS are not served from cache.
        self._result_cache: dict[str, DecisionResult] = {}
        self._CACHE_TTL_SECONDS: int = 60

        logger.info("DecisionEngine initialised — all sub-components ready")

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def _is_kill_switch_active(self) -> bool:
        """Check if emergency kill switch is active.

        Kill switch can be activated by:
        1. Creating file: data/KILL_SWITCH
        2. Inserting kill_switch=ACTIVE in system_health table

        When active, NO new signals are generated.  Open positions are NOT
        automatically closed (that requires manual decision).
        """
        # File-based check (fastest, works even if DB is down)
        if os.path.exists(KILL_SWITCH_FILE):
            return True

        # DB-based check (for remote activation)
        try:
            row = self._db.fetch_one(
                "SELECT status FROM system_health "
                "WHERE component = 'kill_switch' ORDER BY timestamp DESC LIMIT 1",
                (),
            )
            if row and row["status"] == "ACTIVE":
                return True
        except Exception:
            pass  # If DB check fails, rely on file check only

        return False

    def activate_kill_switch(self, reason: str = "Manual activation") -> None:
        """Activate emergency kill switch."""
        from src.utils.date_utils import get_ist_now

        with open(KILL_SWITCH_FILE, "w") as f:
            f.write(f"Activated at {get_ist_now().isoformat()}\nReason: {reason}\n")

        try:
            self._db.execute(
                "INSERT INTO system_health (timestamp, component, status, message) "
                "VALUES (?, 'kill_switch', 'ACTIVE', ?)",
                (get_ist_now().isoformat(), reason),
            )
        except Exception:
            logger.warning("Kill switch file created but DB record failed")

        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def deactivate_kill_switch(self) -> None:
        """Deactivate emergency kill switch."""
        from src.utils.date_utils import get_ist_now

        if os.path.exists(KILL_SWITCH_FILE):
            os.remove(KILL_SWITCH_FILE)

        try:
            self._db.execute(
                "INSERT INTO system_health (timestamp, component, status, message) "
                "VALUES (?, 'kill_switch', 'INACTIVE', 'Deactivated')",
                (get_ist_now().isoformat(),),
            )
        except Exception:
            logger.warning("Kill switch file removed but DB record failed")

        logger.info("Kill switch deactivated — signal generation resumed")

    def _kill_switch_no_trade(self, index_id: str) -> DecisionResult:
        """Return a safe NO_TRADE DecisionResult for kill-switch scenarios."""
        now = datetime.now(tz=_IST)
        regime = self._fallback_regime(index_id, "Kill switch active")
        signal = self._make_no_trade_signal(
            index_id, regime, "EMERGENCY: Kill switch active. All signal generation halted.",
        )
        signal.warnings.append("KILL_SWITCH_ACTIVE")
        return DecisionResult(
            index_id=index_id,
            timestamp=now,
            signal=signal,
            is_actionable=False,
            alert_message="KILL SWITCH ACTIVE — System halted. Remove data/KILL_SWITCH to resume.",
            alert_priority="CRITICAL",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_cycle(self, index_id: str) -> DecisionResult:
        """
        Execute the complete 8-step analysis pipeline for *index_id*.

        Execution order:
          1. Gather data (price, options, VIX)
          2. Technical analysis
          3. News assessment
          4. Anomaly detection
          5. Regime detection
          6. Signal generation
          7. Risk management
          8. Persist + alert

        Parameters
        ----------
        index_id:
            Registry identifier, e.g. ``"NIFTY50"``.

        Returns
        -------
        DecisionResult
            Always returns — never raises.
        """
        # EMERGENCY CHECK — before any computation
        if self._is_kill_switch_active():
            logger.critical("KILL SWITCH ACTIVE — all signal generation halted")
            return self._kill_switch_no_trade(index_id)

        cycle_start = time.monotonic()
        now = datetime.now(tz=_IST)
        step_timings: dict[str, int] = {}

        logger.info("DecisionEngine: starting cycle for %s", index_id)

        # ── Step 1: Gather Data ────────────────────────────────────────
        t0 = time.monotonic()
        (
            current_bar,
            daily_rows,
            intraday_rows,
            options_chain,
            oi_summary,
            vix_value,
            benchmark_rows,
        ) = self._step_gather_data(index_id)
        step_timings["1_gather_data"] = int((time.monotonic() - t0) * 1000)

        price_df = self._build_price_df(daily_rows, intraday_rows)
        current_price = float(current_bar["close"]) if current_bar else 0.0

        # ── Stale-data check (non-blocking — adds warnings) ──────────
        stale_warnings: list[str] = []
        _stale_confidence_penalty = 0.0
        for dtype, max_age in (("index_prices", 300), ("options_chain", 600), ("vix", 480)):
            if freshness_tracker.is_stale(dtype, max_age_seconds=max_age):
                age = freshness_tracker.get_age_seconds(dtype)
                age_str = f"{age:.0f}s" if age is not None else "never fetched"
                msg = f"Stale data: {dtype} ({age_str} old, max {max_age}s)"
                stale_warnings.append(msg)
                logger.warning("DecisionEngine [%s]: %s", index_id, msg)
                _stale_confidence_penalty += 0.1  # 0.1 per stale source

        # ── Step 2: Technical Analysis ─────────────────────────────────
        t0 = time.monotonic()
        technical_result = self._step_technical(
            index_id, price_df, options_chain, vix_value, benchmark_rows,
        )
        step_timings["2_technical"] = int((time.monotonic() - t0) * 1000)

        # ── Step 3: News Assessment ────────────────────────────────────
        t0 = time.monotonic()
        news_vote, event_modifier = self._step_news(index_id)
        step_timings["3_news"] = int((time.monotonic() - t0) * 1000)

        # ── Step 4: Anomaly Detection ──────────────────────────────────
        t0 = time.monotonic()
        anomaly_result = self._step_anomaly(
            index_id, current_bar, daily_rows, options_chain, oi_summary,
        )
        step_timings["4_anomaly"] = int((time.monotonic() - t0) * 1000)

        # ── Step 5: Regime Detection ───────────────────────────────────
        t0 = time.monotonic()
        regime = self._step_regime(
            index_id, price_df, technical_result, event_modifier, anomaly_result, vix_value,
        )
        step_timings["5_regime"] = int((time.monotonic() - t0) * 1000)

        # ── Step 6: Signal Generation ──────────────────────────────────
        t0 = time.monotonic()
        anomaly_vote = self._anomaly_result_to_vote(anomaly_result, index_id)
        raw_signal = self._step_signal(
            index_id, technical_result, news_vote, anomaly_vote, regime, current_price,
        )
        step_timings["6_signal"] = int((time.monotonic() - t0) * 1000)

        # ── Step 7: Risk Management ────────────────────────────────────
        t0 = time.monotonic()
        final_signal, is_actionable = self._step_risk(raw_signal, options_chain)
        step_timings["7_risk"] = int((time.monotonic() - t0) * 1000)

        # ── Step 8: Store & Alert ──────────────────────────────────────
        t0 = time.monotonic()
        alert_message, alert_priority = self._step_store_and_alert(
            index_id, final_signal, is_actionable,
        )
        step_timings["8_store_alert"] = int((time.monotonic() - t0) * 1000)

        total_ms = int((time.monotonic() - cycle_start) * 1000)

        # ── Apply stale-data confidence penalty ──────────────────────
        if _stale_confidence_penalty > 0 and hasattr(final_signal, "confidence_score"):
            original = final_signal.confidence_score or 0.0
            penalized = max(0.0, original - _stale_confidence_penalty)
            object.__setattr__(final_signal, "confidence_score", penalized)
            logger.info(
                "DecisionEngine [%s]: confidence reduced %.2f → %.2f (stale data)",
                index_id, original, penalized,
            )

        result = DecisionResult(
            index_id=index_id,
            timestamp=now,
            signal=final_signal,
            is_actionable=is_actionable,
            technical_result=technical_result,
            news_vote=news_vote,
            anomaly_vote=anomaly_vote,
            regime=regime,
            step_timings=step_timings,
            total_duration_ms=total_ms,
            alert_message=alert_message,
            alert_priority=alert_priority,
            warnings=stale_warnings,
        )
        result.dashboard_summary = self._build_dashboard_summary(result, current_price)

        with self._lock:
            self._result_cache[index_id] = result

        logger.info(
            "DecisionEngine: %s → %s (%s) | %dms",
            index_id,
            getattr(final_signal, "signal_type", "N/A"),
            getattr(final_signal, "confidence_level", "-"),
            total_ms,
        )
        return result

    def run_all_indices(self) -> list[DecisionResult]:
        """
        Run ``run_full_cycle()`` sequentially for every F&O-enabled index.

        A 2-second delay between indices avoids overwhelming upstream data
        sources.  Results are sorted: actionable signals first, then by
        confidence score descending.

        Returns
        -------
        list[DecisionResult]
        """
        if self._is_kill_switch_active():
            logger.critical("KILL SWITCH ACTIVE — skipping all indices")
            return []

        fo_indices = self.registry.get_indices_with_options()
        if not fo_indices:
            logger.warning("DecisionEngine.run_all_indices: no F&O indices in registry")
            return []

        index_ids = [idx.id for idx in fo_indices]
        results: list[DecisionResult] = []
        for i, index_id in enumerate(index_ids):
            try:
                results.append(self.run_full_cycle(index_id))
            except Exception:
                logger.exception("run_all_indices: unhandled error for %s", index_id)

            if i < len(index_ids) - 1:
                time.sleep(_INTER_INDEX_DELAY)

        # Sort: actionable first, then highest confidence
        results.sort(key=lambda r: (
            0 if r.is_actionable else 1,
            -(getattr(r.signal, "confidence_score", 0.0) or 0.0),
        ))

        # Summary log
        buy_calls = [r for r in results if getattr(r.signal, "signal_type", "") == "BUY_CALL"]
        buy_puts  = [r for r in results if getattr(r.signal, "signal_type", "") == "BUY_PUT"]
        no_trades = [r for r in results if getattr(r.signal, "signal_type", "") == "NO_TRADE"]
        parts = (
            [f"BUY_CALL ({r.index_id}, {getattr(r.signal, 'confidence_level', '?')})" for r in buy_calls]
            + [f"BUY_PUT ({r.index_id}, {getattr(r.signal, 'confidence_level', '?')})" for r in buy_puts]
            + ([f"{len(no_trades)} NO_TRADE"] if no_trades else [])
        )
        logger.info(
            "DecisionEngine: analysed %d indices — %s",
            len(results),
            " | ".join(parts) if parts else "all NO_TRADE",
        )
        return results

    def monitor_open_positions(self) -> None:
        """
        Check all tracked open positions against current prices.

        Intended to be called every 60 seconds by the data collector.
        - Checks SL / target / trailing-SL / time-exit conditions
        - Closes any triggered position and records the outcome
        - Generates an exit alert string (logged at INFO)
        """
        open_pos = list(self.risk_manager._open_positions.values())
        if not open_pos:
            return

        logger.debug("DecisionEngine: monitoring %d open position(s)", len(open_pos))

        for pos in open_pos:
            try:
                bar = (
                    self._get_latest_bar(pos.index_id, "5m")
                    or self._get_latest_bar(pos.index_id, "1d")
                )
                if not bar:
                    logger.warning(
                        "monitor_open_positions: no price for %s — skipping", pos.index_id
                    )
                    continue

                current_price = float(bar.get("close", 0))
                if current_price <= 0:
                    continue

                update = self.risk_manager.update_position(pos.signal_id, current_price)

                if update.action != "HOLD":
                    exit_reason = update.action.replace("EXIT_", "")
                    self.risk_manager.close_position(
                        pos.signal_id, current_price, exit_reason,
                    )
                    self.tracker.record_outcome(
                        pos.signal_id,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        pnl=update.current_pnl,
                    )
                    logger.info(
                        "EXIT ALERT:\n%s",
                        self._format_exit_alert(pos, update, exit_reason),
                    )
                else:
                    logger.debug(
                        "HOLD  %s (%s)  PnL=₹%.0f (%.2f%%)",
                        pos.index_id, pos.signal_id[:8],
                        update.current_pnl, update.current_pnl_pct * 100,
                    )

            except Exception:
                logger.exception(
                    "monitor_open_positions: error for position %s",
                    getattr(pos, "signal_id", "?"),
                )

    def generate_alert_message(self, result: DecisionResult) -> str:
        """
        Format a ``DecisionResult`` into a Telegram-ready alert string.

        Returns an empty string for NO_TRADE (no alert sent).
        """
        sig = result.signal
        signal_type = getattr(sig, "signal_type", "NO_TRADE")

        if signal_type == "NO_TRADE" or not result.is_actionable:
            return ""

        idx = self.registry.get_index(result.index_id)
        display_name = idx.display_name if idx else result.index_id
        direction = "CALL" if signal_type == "BUY_CALL" else "PUT"

        conf_score = getattr(sig, "confidence_score", 0.0) or 0.0
        conf_level = getattr(sig, "confidence_level", "LOW")
        regime_str = getattr(sig, "regime", "")
        if result.regime and not regime_str:
            regime_str = result.regime.regime

        entry   = getattr(sig, "refined_entry",     None) or getattr(sig, "entry_price",  0.0) or 0.0
        target  = getattr(sig, "refined_target",    None) or getattr(sig, "target_price", 0.0) or 0.0
        sl      = getattr(sig, "refined_stop_loss", None) or getattr(sig, "stop_loss",    0.0) or 0.0
        rr      = getattr(sig, "risk_reward_ratio", 0.0) or 0.0

        target_pts = (target - entry) if signal_type == "BUY_CALL" else (entry - target)
        sl_pts     = (entry - sl)     if signal_type == "BUY_CALL" else (sl - entry)

        strike  = getattr(sig, "recommended_strike",  None)
        expiry  = getattr(sig, "recommended_expiry",  None)
        premium = getattr(sig, "option_premium",      None)
        lots    = getattr(sig, "lots", None) or getattr(sig, "suggested_lot_count", 1) or 1
        max_loss_amt = getattr(sig, "max_loss_amount",      None) or getattr(sig, "estimated_max_loss", 0.0) or 0.0
        risk_pct     = getattr(sig, "risk_pct_of_capital", 0.0) or 0.0

        # Why — from vote_breakdown
        vb: dict = getattr(sig, "vote_breakdown", {}) or {}
        why_lines: list[str] = []
        trend_vote = (vb.get("trend") or {}).get("vote", "")
        if trend_vote:
            why_lines.append(f"• Trend: {trend_vote.replace('_', ' ').title()}")
        opt_vote = (vb.get("options") or {}).get("vote", "")
        if opt_vote and result.technical_result and result.technical_result.options:
            opts_summary = result.technical_result.options
            pcr = getattr(opts_summary, "pcr", None)
            pcr_str = f", PCR {pcr:.2f}" if pcr else ""
            why_lines.append(f"• Options: {opt_vote.replace('_', ' ').title()}{pcr_str}")
        sm_vote = (vb.get("smart_money") or {}).get("vote", "")
        if sm_vote and sm_vote not in ("NEUTRAL", ""):
            why_lines.append(f"• Smart Money: {sm_vote.replace('_', ' ').title()}")
        if result.news_vote and result.news_vote.vote not in ("NEUTRAL", ""):
            why_lines.append(f"• News: {result.news_vote.vote.replace('_', ' ').title()}")

        warnings: list[str] = getattr(sig, "warnings", []) or []

        SEP = "━" * 20
        lines = [
            f"BUY {direction} — {display_name}",
            SEP,
            f"Confidence: {conf_level} ({conf_score:.2f})",
            f"Regime: {regime_str.replace('_', ' ').title()}",
            "",
            "📊 Levels:",
            f"Entry:     {entry:,.0f}",
            f"Target:    {target:,.0f} (+{target_pts:.0f} pts)",
            f"Stop Loss: {sl:,.0f} (-{sl_pts:.0f} pts)",
            f"RR:        1:{rr:.2f}",
        ]

        if strike and expiry:
            opt_type = "CE" if signal_type == "BUY_CALL" else "PE"
            lines += [
                "",
                f"📋 Strike:  {result.index_id} {strike:.0f} {opt_type} ({expiry})",
            ]
            if premium:
                lines.append(f"Premium:   ~₹{premium:.0f}")
        lines += [
            f"Lots:      {lots}",
            f"Risk:      ₹{max_loss_amt:,.0f} ({risk_pct:.2f}% of capital)",
        ]

        if why_lines:
            lines += ["", "📈 Why:"] + why_lines

        if warnings:
            lines += ["", "⚠️ Caution:"]
            lines += [f"• {w}" for w in warnings[:3]]

        lines.append(SEP)
        return "\n".join(lines)

    def get_dashboard_data(self) -> DashboardData:
        """
        Build a complete data package for the frontend dashboard.

        Pulls from DB, cached results, and risk manager state.
        Never raises — returns whatever data is available.
        """
        now = datetime.now(tz=_IST)
        dashboard = DashboardData(
            timestamp=now,
            market_status=self._get_market_status(now),
        )

        # Per-index rows
        for idx in self.registry.get_indices_with_options():
            try:
                dashboard.indices.append(self._build_index_dashboard(idx.id))
            except Exception:
                logger.exception("get_dashboard_data: IndexDashboard failed for %s", idx.id)

        # Portfolio / today stats
        try:
            port = self.risk_manager.get_portfolio_summary()
            dashboard.active_signals = port.open_positions
            dashboard.today_pnl = port.today_pnl
        except Exception:
            logger.exception("get_dashboard_data: portfolio summary failed")

        try:
            today_str = now.date().isoformat()
            rows = self._db.fetch_all(
                """
                SELECT outcome FROM trading_signals
                WHERE date(generated_at) = ?
                  AND signal_type != 'NO_TRADE'
                  AND outcome IN ('WIN','LOSS')
                """,
                (today_str,),
            )
            dashboard.today_trades = len(rows)
            wins = sum(1 for r in rows if r["outcome"] == "WIN")
            dashboard.today_win_rate = wins / len(rows) if rows else 0.0
        except Exception:
            logger.exception("get_dashboard_data: today stats query failed")

        # VIX
        try:
            row = self._db.fetch_one(
                "SELECT vix_value FROM vix_data ORDER BY timestamp DESC LIMIT 1", ()
            )
            if row:
                dashboard.vix_value = float(row["vix_value"])
                dashboard.vix_regime = self._vix_regime_label(dashboard.vix_value)
        except Exception:
            logger.exception("get_dashboard_data: VIX query failed")

        # Market-wide news sentiment
        try:
            all_votes = self.news_engine.get_all_news_votes()
            dashboard.market_sentiment = self._aggregate_sentiment(
                [v.vote for v in all_votes.values()]
            )
        except Exception:
            logger.exception("get_dashboard_data: news sentiment aggregation failed")

        # FII bias
        try:
            row = self._db.fetch_one(
                """
                SELECT SUM(net_value) AS net FROM fii_dii_activity
                WHERE category = 'FII' AND segment = 'CASH'
                  AND date = (SELECT MAX(date) FROM fii_dii_activity)
                """,
                (),
            )
            if row and row["net"] is not None:
                net = float(row["net"])
                dashboard.fii_bias = "BULLISH" if net > 500 else ("BEARISH" if net < -500 else "NEUTRAL")
        except Exception:
            logger.exception("get_dashboard_data: FII query failed")

        # Recent anomaly alerts
        try:
            alert_rows = self._db.fetch_all(
                """
                SELECT index_id, anomaly_type, severity, message, timestamp
                FROM anomaly_events WHERE is_active = 1
                ORDER BY
                    CASE severity WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END,
                    timestamp DESC
                LIMIT 10
                """,
                (),
            )
            dashboard.recent_alerts = [dict(r) for r in alert_rows]
        except Exception:
            logger.exception("get_dashboard_data: anomaly alerts query failed")

        # Top news
        try:
            dashboard.top_news = self.news_engine.get_news_feed(limit=5, min_severity="MEDIUM")
        except Exception:
            logger.exception("get_dashboard_data: news feed failed")

        return dashboard

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _step_gather_data(self, index_id: str) -> tuple:
        """Step 1 — Load data from DB. Each sub-fetch is isolated."""
        current_bar: Optional[dict] = None
        daily_rows: list[dict] = []
        intraday_rows: list[dict] = []
        options_chain = None
        oi_summary: Optional[dict] = None
        vix_value: Optional[float] = None
        benchmark_rows: list[dict] = []

        try:
            current_bar = (
                self._db.fetch_one(Q.GET_LATEST_PRICE, (index_id, "5m"))
                or self._db.fetch_one(Q.GET_LATEST_PRICE, (index_id, "1d"))
            )
        except Exception:
            logger.warning("Step 1: latest bar unavailable for %s", index_id)

        try:
            daily_rows = self._db.fetch_all(
                Q.LIST_PRICE_HISTORY_LIMIT, (index_id, "1d", 250)
            )
        except Exception:
            logger.warning("Step 1: daily history unavailable for %s", index_id)

        try:
            intraday_rows = self._db.fetch_all(
                Q.LIST_PRICE_HISTORY_LIMIT, (index_id, "5m", 100)
            )
        except Exception:
            logger.debug("Step 1: intraday unavailable for %s", index_id)

        index_def = self.registry.get_index(index_id)
        if index_def and index_def.has_options:
            try:
                options_chain = self._load_options_chain(index_id)
            except Exception:
                logger.warning("Step 1: options chain unavailable for %s", index_id)
            try:
                oi_summary = self._load_oi_summary(index_id)
            except Exception:
                logger.warning("Step 1: OI summary unavailable for %s", index_id)

        try:
            row = self._db.fetch_one(
                "SELECT vix_value FROM vix_data ORDER BY timestamp DESC LIMIT 1", ()
            )
            if row:
                vix_value = float(row["vix_value"])
        except Exception:
            logger.debug("Step 1: VIX unavailable")

        try:
            if index_id != "NIFTY50":
                benchmark_rows = self._db.fetch_all(
                    Q.LIST_PRICE_HISTORY_LIMIT, ("NIFTY50", "1d", 250)
                )
        except Exception:
            logger.debug("Step 1: benchmark data unavailable")

        return (
            current_bar, daily_rows, intraday_rows,
            options_chain, oi_summary, vix_value, benchmark_rows,
        )

    def _step_technical(
        self,
        index_id: str,
        price_df: pd.DataFrame,
        options_chain,
        vix_value: Optional[float],
        benchmark_rows: list[dict],
    ) -> Optional[TechnicalAnalysisResult]:
        """Step 2 — TechnicalAggregator."""
        try:
            if price_df is None or len(price_df) < _MIN_BARS_FALLBACK:
                logger.warning(
                    "Step 2: insufficient price data for %s (%d bars)",
                    index_id,
                    len(price_df) if price_df is not None else 0,
                )
                return None

            bm_df = self._rows_to_df(benchmark_rows) if benchmark_rows else None
            return self.tech_aggregator.analyze(
                index_id=index_id,
                price_df=price_df,
                options_chain=options_chain,
                oi_history=None,
                vix_value=vix_value,
                benchmark_df=bm_df,
                timeframe="1d",
            )
        except Exception:
            logger.exception("Step 2: TechnicalAggregator failed for %s", index_id)
            return None

    def _step_news(self, index_id: str) -> tuple:
        """
        Step 3 — News assessment.

        Triggers a fresh cycle when cached data is stale (> 5 minutes old).
        Returns (NewsVote | None, EventRegimeModifier | None).
        """
        try:
            with self._lock:
                last = self._last_news_cycle

            if last is None or (datetime.now(tz=_IST) - last).total_seconds() > _NEWS_STALE_SECONDS:
                logger.debug("Step 3: triggering news cycle for %s", index_id)
                try:
                    self.news_engine.run_news_cycle()
                    with self._lock:
                        self._last_news_cycle = datetime.now(tz=_IST)
                except Exception:
                    logger.warning("Step 3: news cycle failed — serving cached results")

            news_vote = self.news_engine.get_news_vote(index_id)

            event_modifier = None
            try:
                with self.news_engine._lock:
                    event_modifier = self.news_engine._last_event_regime.get(index_id)
            except Exception:
                pass

            return news_vote, event_modifier

        except Exception:
            logger.exception("Step 3: news assessment failed for %s", index_id)
            return None, None

    def _step_anomaly(
        self,
        index_id: str,
        current_bar: Optional[dict],
        recent_bars: list[dict],
        options_chain,
        oi_summary,
    ) -> Optional[AnomalyDetectionResult]:
        """Step 4 — Anomaly detection."""
        try:
            if current_bar is None:
                logger.warning("Step 4: no current bar for %s — skipping anomaly", index_id)
                return None

            return self.anomaly_engine.run_detection_cycle(
                index_id=index_id,
                current_price_bar=current_bar,
                recent_price_bars=recent_bars[-20:] if recent_bars else [],
                options_chain=options_chain,
                options_summary=oi_summary,
            )
        except Exception:
            logger.exception("Step 4: anomaly detection failed for %s", index_id)
            return None

    def _step_regime(
        self,
        index_id: str,
        price_df: pd.DataFrame,
        technical_result: Optional[TechnicalAnalysisResult],
        event_modifier,
        anomaly_result: Optional[AnomalyDetectionResult],
        vix_value: Optional[float],
    ) -> MarketRegime:
        """Step 5 — Regime detection. Always returns a valid MarketRegime."""
        try:
            if technical_result is None:
                return self._fallback_regime(index_id, "No technical result")

            return self.regime_detector.detect_regime(
                index_id=index_id,
                price_df=price_df,
                technical_result=technical_result,
                news_event_modifier=event_modifier,
                anomaly_result=anomaly_result,
                vix_value=vix_value,
            )
        except Exception:
            logger.exception("Step 5: regime detection failed for %s", index_id)
            return self._fallback_regime(index_id, "Regime detection error")

    def _step_signal(
        self,
        index_id: str,
        technical_result: Optional[TechnicalAnalysisResult],
        news_vote: Optional[NewsVote],
        anomaly_vote: Optional[AnomalyVote],
        regime: MarketRegime,
        current_price: float,
    ) -> TradingSignal:
        """Step 6 — Signal generation."""
        try:
            if technical_result is None or current_price <= 0:
                logger.warning(
                    "Step 6: missing technical result or price for %s — NO_TRADE", index_id
                )
                return self._make_no_trade_signal(index_id, regime, "Missing technical data")

            return self.signal_generator.generate_signal(
                index_id=index_id,
                technical=technical_result,
                news=news_vote,
                anomaly=anomaly_vote,
                regime=regime,
                current_spot_price=current_price,
            )
        except Exception:
            logger.exception("Step 6: signal generation failed for %s", index_id)
            return self._make_no_trade_signal(index_id, regime, "Signal generation error")

    def _step_risk(
        self,
        raw_signal: TradingSignal,
        options_chain,
    ) -> tuple:
        """
        Step 7 — Risk management.

        Returns
        -------
        (final_signal, is_actionable)
        """
        try:
            if raw_signal.signal_type == "NO_TRADE":
                return raw_signal, False

            refined = self.risk_manager.validate_and_refine_signal(
                signal=raw_signal,
                current_chain=options_chain,
            )
            is_actionable = (
                refined.is_valid
                and refined.signal_type in ("BUY_CALL", "BUY_PUT")
            )
            return refined, is_actionable

        except Exception:
            logger.exception("Step 7: risk management failed")
            return raw_signal, False

    def _step_store_and_alert(
        self,
        index_id: str,
        final_signal,
        is_actionable: bool,
    ) -> tuple:
        """
        Step 8 — Persist signal; prepare alert text and priority.

        Returns
        -------
        (alert_message | None, alert_priority)
        """
        alert_message: Optional[str] = None
        alert_priority = "NONE"

        try:
            self.tracker.record_signal(final_signal)
        except Exception:
            logger.exception("Step 8: tracker.record_signal failed for %s", index_id)

        if is_actionable:
            try:
                self.risk_manager.track_open_position(final_signal)
            except Exception:
                logger.exception("Step 8: track_open_position failed for %s", index_id)

            try:
                stub = DecisionResult(
                    index_id=index_id,
                    timestamp=datetime.now(tz=_IST),
                    signal=final_signal,
                    is_actionable=True,
                )
                alert_message = self.generate_alert_message(stub)
                conf = getattr(final_signal, "confidence_level", "LOW")
                alert_priority = (
                    "CRITICAL" if conf == "HIGH"
                    else "HIGH" if conf == "MEDIUM"
                    else "NORMAL"
                )
            except Exception:
                logger.exception("Step 8: alert generation failed for %s", index_id)

        return alert_message, alert_priority

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_latest_bar(self, index_id: str, timeframe: str = "1d") -> Optional[dict]:
        try:
            return self._db.fetch_one(Q.GET_LATEST_PRICE, (index_id, timeframe))
        except Exception:
            return None

    def _load_options_chain(self, index_id: str):
        """Reconstruct OptionsChainData from the latest DB snapshot."""
        try:
            from src.data.options_chain import OptionsChainData, OptionStrike
            from datetime import date as date_

            expiry_rows = self._db.fetch_all(Q.LIST_EXPIRY_DATES, (index_id,))
            if not expiry_rows:
                return None

            expiry_str = expiry_rows[0]["expiry_date"]
            rows = self._db.fetch_all(
                Q.LIST_OPTIONS_CHAIN_FOR_EXPIRY,
                (index_id, expiry_str, index_id, expiry_str),
            )
            if not rows:
                return None

            strikes_map: dict[float, dict] = {}
            for row in rows:
                sp = float(row["strike_price"])
                strikes_map.setdefault(sp, {})
                strikes_map[sp][row["option_type"]] = row

            strikes_list = []
            for sp in sorted(strikes_map):
                d = strikes_map[sp]
                ce = d.get("CE") or {}
                pe = d.get("PE") or {}
                strikes_list.append(OptionStrike(
                    strike_price=sp,
                    ce_oi=int(ce.get("open_interest", 0)),
                    ce_oi_change=int(ce.get("oi_change", 0)),
                    ce_volume=int(ce.get("volume", 0)),
                    ce_ltp=float(ce.get("ltp", 0)),
                    ce_iv=float(ce.get("iv") or 0),
                    pe_oi=int(pe.get("open_interest", 0) if pe else 0),
                    pe_oi_change=int(pe.get("oi_change", 0) if pe else 0),
                    pe_volume=int(pe.get("volume", 0) if pe else 0),
                    pe_ltp=float(pe.get("ltp", 0) if pe else 0),
                    pe_iv=float(pe.get("iv") or 0 if pe else 0),
                ))

            latest_price_row = self._db.fetch_one(
                "SELECT close FROM price_data WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1",
                (index_id,),
            )
            spot = float(latest_price_row["close"]) if latest_price_row else 0.0

            ts_raw = rows[0].get("timestamp", datetime.now(tz=_IST).isoformat())
            try:
                ts = datetime.fromisoformat(ts_raw).replace(tzinfo=_IST)
            except (ValueError, TypeError):
                ts = datetime.now(tz=_IST)

            all_expiries = tuple(date_.fromisoformat(r["expiry_date"]) for r in expiry_rows)

            return OptionsChainData(
                index_id=index_id,
                spot_price=spot,
                timestamp=ts,
                expiry_date=date_.fromisoformat(expiry_str),
                strikes=tuple(strikes_list),
                available_expiries=all_expiries,
            )
        except Exception:
            logger.exception("_load_options_chain failed for %s", index_id)
            return None

    def _load_oi_summary(self, index_id: str) -> Optional[dict]:
        try:
            rows = self._db.fetch_all(Q.LIST_EXPIRY_DATES, (index_id,))
            if not rows:
                return None
            return self._db.fetch_one(
                Q.GET_LATEST_OI_AGGREGATED, (index_id, rows[0]["expiry_date"])
            )
        except Exception:
            return None

    @staticmethod
    def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _build_price_df(
        self,
        daily_rows: list[dict],
        intraday_rows: list[dict],
    ) -> pd.DataFrame:
        """Prefer daily bars; fall back to combining both if daily is thin."""
        if daily_rows and len(daily_rows) >= _MIN_DAILY_BARS:
            return self._rows_to_df(daily_rows)
        all_rows = list(daily_rows or []) + list(intraday_rows or [])
        return self._rows_to_df(all_rows) if all_rows else pd.DataFrame()

    @staticmethod
    def _anomaly_result_to_vote(
        result: Optional[AnomalyDetectionResult],
        index_id: str,
    ) -> Optional[AnomalyVote]:
        if result is None:
            return None
        return AnomalyVote(
            index_id=index_id,
            vote=result.anomaly_vote,
            confidence=result.anomaly_confidence,
            risk_level=result.risk_level,
            position_size_modifier=result.position_size_modifier,
            active_alerts=result.active_alert_count,
            primary_alert_message=(
                result.primary_alert.message if result.primary_alert else None
            ),
            institutional_activity=result.institutional_activity_detected,
            reasoning=result.summary,
        )

    @staticmethod
    def _make_no_trade_signal(
        index_id: str, regime: MarketRegime, reason: str,
    ) -> TradingSignal:
        import uuid
        return TradingSignal(
            signal_id=str(uuid.uuid4()),
            index_id=index_id,
            generated_at=datetime.now(tz=_IST),
            signal_type="NO_TRADE",
            confidence_level="LOW",
            confidence_score=0.0,
            entry_price=0.0,
            target_price=0.0,
            stop_loss=0.0,
            risk_reward_ratio=0.0,
            regime=regime.regime if regime else "RANGE_BOUND",
            weighted_score=0.0,
            vote_breakdown={},
            risk_level="NORMAL",
            position_size_modifier=0.5,
            suggested_lot_count=0,
            estimated_max_loss=0.0,
            estimated_max_profit=0.0,
            reasoning=reason,
            warnings=[reason],
        )

    @staticmethod
    def _fallback_regime(index_id: str, reason: str) -> MarketRegime:
        weights = SignalWeights()
        return MarketRegime(
            index_id=index_id,
            timestamp=datetime.now(tz=_IST),
            regime="RANGE_BOUND",
            trend_regime="FLAT",
            volatility_regime="NORMAL",
            event_regime="NORMAL",
            market_phase="ACCUMULATION",
            regime_confidence=0.1,
            regime_duration_bars=0,
            regime_changing=False,
            weight_adjustments=weights,
            position_size_multiplier=0.5,
            stop_loss_multiplier=1.2,
            max_trades_today=1,
            description=f"Fallback regime: {reason}",
            warnings=[reason],
        )

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def _build_index_dashboard(self, index_id: str) -> IndexDashboard:
        bar = self._get_latest_bar(index_id, "5m") or self._get_latest_bar(index_id, "1d")
        current_price = float(bar["close"]) if bar else 0.0

        prev_close = current_price
        try:
            rows = self._db.fetch_all(Q.LIST_PRICE_HISTORY_LIMIT, (index_id, "1d", 2))
            if len(rows) >= 2:
                prev_close = float(rows[1]["close"])
        except Exception:
            pass

        change_pct = (
            (current_price - prev_close) / prev_close * 100 if prev_close else 0.0
        )

        with self._lock:
            cached = self._result_cache.get(index_id)

        regime_str = "UNKNOWN"
        signal_str = "NO_TRADE"
        conf_str = "-"
        key_levels: dict = {}
        risk_level = "NORMAL"

        if cached:
            if cached.regime:
                regime_str = cached.regime.regime
            signal_str = getattr(cached.signal, "signal_type", "NO_TRADE")
            conf_str = getattr(cached.signal, "confidence_level", "-")
            risk_level = getattr(cached.signal, "risk_level", "NORMAL") or "NORMAL"
            if cached.technical_result:
                key_levels = {
                    "support": cached.technical_result.immediate_support,
                    "resistance": cached.technical_result.immediate_resistance,
                }

        active_alerts = 0
        try:
            row = self._db.fetch_one(
                "SELECT COUNT(*) AS cnt FROM anomaly_events WHERE index_id=? AND is_active=1",
                (index_id,),
            )
            if row:
                active_alerts = int(row["cnt"])
        except Exception:
            pass

        return IndexDashboard(
            index_id=index_id,
            current_price=current_price,
            change_pct=round(change_pct, 2),
            regime=regime_str,
            signal=signal_str,
            confidence=conf_str,
            key_levels=key_levels,
            active_alerts=active_alerts,
            risk_level=risk_level,
        )

    @staticmethod
    def _build_dashboard_summary(result: DecisionResult, current_price: float) -> dict:
        sig = result.signal
        return {
            "index_id": result.index_id,
            "current_price": current_price,
            "signal_type": getattr(sig, "signal_type", "NO_TRADE"),
            "confidence_level": getattr(sig, "confidence_level", "-"),
            "confidence_score": round(getattr(sig, "confidence_score", 0.0) or 0.0, 4),
            "is_actionable": result.is_actionable,
            "regime": result.regime.regime if result.regime else "UNKNOWN",
            "regime_confidence": round(
                result.regime.regime_confidence, 4
            ) if result.regime else 0.0,
            "entry":    getattr(sig, "refined_entry",     None) or getattr(sig, "entry_price",  0.0),
            "target":   getattr(sig, "refined_target",    None) or getattr(sig, "target_price", 0.0),
            "stop_loss":getattr(sig, "refined_stop_loss", None) or getattr(sig, "stop_loss",    0.0),
            "risk_reward": round(getattr(sig, "risk_reward_ratio", 0.0) or 0.0, 2),
            "lots": getattr(sig, "lots", None) or getattr(sig, "suggested_lot_count", 0),
            "risk_level": getattr(sig, "risk_level", "NORMAL"),
            "alert_priority": result.alert_priority,
            "total_duration_ms": result.total_duration_ms,
            "step_timings": result.step_timings,
            "news_vote": result.news_vote.vote if result.news_vote else "NEUTRAL",
            "anomaly_vote": result.anomaly_vote.vote if result.anomaly_vote else "NEUTRAL",
            "warnings": getattr(sig, "warnings", []),
        }

    @staticmethod
    def _format_exit_alert(pos, update, exit_reason: str) -> str:
        direction = "CALL" if pos.signal_type == "BUY_CALL" else "PUT"
        reason_map = {
            "TARGET": "Target Hit ✅",
            "SL": "Stop Loss Hit ❌",
            "TRAILING_SL": "Trailing SL Hit 🔄",
            "TIME": "Time Exit (EOD) ⏰",
        }
        reason_str = reason_map.get(exit_reason, exit_reason)
        pnl_sign = "+" if update.current_pnl >= 0 else ""
        SEP = "━" * 20
        return "\n".join([
            f"🔴 EXIT — {pos.index_id} {direction}",
            SEP,
            f"Reason: {reason_str}",
            f"Entry: {pos.entry_price:,.0f}",
            f"PnL: ₹{pnl_sign}{update.current_pnl:,.0f}",
            f"Duration: {update.time_in_trade_minutes}m",
            SEP,
        ])

    @staticmethod
    def _get_market_status(now: datetime) -> str:
        h, m = now.hour, now.minute
        if h == 9 and m < 15:
            return "PRE_MARKET"
        if (h > 9 or (h == 9 and m >= 15)) and (h < 15 or (h == 15 and m <= 30)):
            return "OPEN"
        return "CLOSED"

    @staticmethod
    def _vix_regime_label(vix: float) -> str:
        if vix < 13:
            return "LOW_VOL"
        if vix < 18:
            return "NORMAL"
        if vix < 25:
            return "ELEVATED"
        return "HIGH_VOL"

    @staticmethod
    def _aggregate_sentiment(votes: list[str]) -> str:
        if not votes:
            return "NEUTRAL"
        _MAP = {
            "STRONG_BULLISH": 2, "BULLISH": 1, "NEUTRAL": 0,
            "BEARISH": -1, "STRONG_BEARISH": -2,
        }
        avg = sum(_MAP.get(v, 0) for v in votes) / len(votes)
        if avg > 0.8:
            return "BULLISH"
        if avg < -0.8:
            return "BEARISH"
        return "NEUTRAL"
