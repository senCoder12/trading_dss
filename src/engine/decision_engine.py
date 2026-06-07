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
from src.api.websocket import queue_event

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Delay between consecutive index analyses in run_all_indices (seconds)
_INTER_INDEX_DELAY: float = 2.0
# Trigger a new news cycle if the last one is older than this
_NEWS_STALE_SECONDS: int = 180   # 3 minutes
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
        fo_manager: Optional[Any] = None,
        candle_store=None,
        smart_api=None,
        options_analyzer=None,
    ) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._fo_manager = fo_manager

        # ── Sub-components ────────────────────────────────────────────
        self.registry = get_registry()

        self.tech_aggregator = TechnicalAggregator()
        self.news_engine = NewsEngine(db)
        self.anomaly_engine = AnomalyAggregator(db)
        self.regime_detector = RegimeDetector(db)
        from src.engine.hero_zero_detector import HeroZeroDetector
        self._hz_detector = HeroZeroDetector(db)
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

        # ── Overnight data cache (loaded once per cycle batch) ────────
        # Keyed by trade date so we avoid hitting the DB for every index.
        self._overnight_cache: Optional[dict] = None
        self._overnight_cache_date: Optional[str] = None

        # Opening Range (9:15-9:30) cache — keyed by {index_id}_{date}
        self._orb_cache: dict[str, dict] = {}

        # Multi-timeframe cache — keyed by index_id, holds 15m and 1h bars
        self._mtf_cache: dict[str, dict] = {}

        # IV history cache — keyed by index_id, holds 30-day ATM IV list
        self._iv_history_cache: dict[str, list[float]] = {}

        # FII futures net position cache — keyed by index_id
        self._fii_futures_cache: dict[str, Optional[float]] = {}

        # Health/status dict for storing computed metrics accessible by the API
        self.health_status: dict = {}

        # ── Liquidity Engine (optional — requires candle_store + smart_api) ──
        self.liquidity_engine = None
        if candle_store is not None and smart_api is not None:
            import weakref
            from src.analysis.liquidity.liquidity_engine import LiquidityEngine
            self.liquidity_engine = LiquidityEngine(
                candle_store=candle_store,
                db=db,
                smart_api=smart_api,
                options_analyzer=options_analyzer,
                decision_engine_ref=weakref.ref(self),
            )

        # ── GEX Regime Detector (optional — requires smart_api) ──────────
        self.gex_calculator = None
        self.gex_regime_detector = None
        if smart_api is not None:
            try:
                from src.analysis.gex import GEXCalculator
                from src.analysis.gex.gex_regime_detector import GEXRegimeDetector
                self.gex_calculator = GEXCalculator(smart_api=smart_api, db=db)
                self.gex_regime_detector = GEXRegimeDetector(
                    gex_calculator=self.gex_calculator, db=db
                )
            except Exception as _gex_exc:
                logger.warning("GEX components failed to init (non-fatal): %s", _gex_exc)

        # GEX state tracking
        self._vix_readings: list[tuple] = []        # [(timestamp, float)]
        self._price_trend: dict[str, str] = {}      # symbol → 'UP'/'DOWN'/'SIDEWAYS'
        self._daily_plans: dict[str, dict] = {}     # symbol → morning trading plan
        self._gex_brief_date: Optional[str] = None  # date string for dedup

        logger.info("DecisionEngine initialised — all sub-components ready")

    def set_fo_universe_manager(self, manager: Any) -> None:
        """Wire the F&O Universe Manager for priority stock signal generation."""
        self._fo_manager = manager
        logger.info("DecisionEngine: FOUniverseManager wired")

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

        try:
            queue_event("system_alert", {
                "alert_type": "KILL_SWITCH_ON",
                "message": f"Kill switch activated: {reason}",
                "severity": "CRITICAL",
            })
        except Exception:
            pass

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

        try:
            queue_event("system_alert", {
                "alert_type": "KILL_SWITCH_OFF",
                "message": "Kill switch deactivated — signal generation resumed",
                "severity": "HIGH",
            })
        except Exception:
            pass

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

    def run_full_cycle(self, index_id: str, as_of: Optional[datetime] = None) -> DecisionResult:
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
            intraday_tf,
        ) = self._step_gather_data(index_id, as_of=as_of)
        step_timings["1_gather_data"] = int((time.monotonic() - t0) * 1000)

        price_df, timeframe_label = self._build_price_df(daily_rows, intraday_rows, intraday_tf)
        current_price = float(current_bar["close"]) if current_bar else 0.0

        # Update Opening Range cache (9:15-9:30 window)
        self._update_orb_cache(index_id, now, current_bar)

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

        # ── Step 2b: Advanced Options Analytics ───────────────────────
        # Run before Step 2 so advanced_opts is available for TechnicalAggregator
        advanced_opts = None
        if options_chain is not None and current_price > 0:
            try:
                from src.analysis.options_analysis import run_advanced_options_analytics
                chain_rows = []
                for s in getattr(options_chain, 'strikes', []):
                    chain_rows.append({
                        "strike": s.strike_price,
                        "ce_oi": s.ce_oi,
                        "pe_oi": s.pe_oi,
                        "ce_ltp": s.ce_ltp,
                        "pe_ltp": s.pe_ltp,
                        "ce_iv": s.ce_iv,
                        "pe_iv": s.pe_iv,
                        "ce_volume": s.ce_volume,
                        "pe_volume": s.pe_volume,
                    })
                if chain_rows:
                    chain_df = pd.DataFrame(chain_rows)
                    idx_def = self.registry.get_index(index_id)
                    lot_size = getattr(idx_def, 'lot_size', 50) or 50
                    advanced_opts = run_advanced_options_analytics(
                        chain_df=chain_df,
                        spot_price=current_price,
                        lot_size=lot_size,
                    )
                    logger.debug(
                        "Advanced opts for %s: GEX=%.2f (%s) straddle=%.0f skew=%.2f",
                        index_id, advanced_opts.gex, advanced_opts.gex_regime,
                        advanced_opts.atm_straddle_price, advanced_opts.put_call_skew,
                    )
            except Exception as exc:
                logger.debug("Advanced options analytics failed (non-fatal): %s", exc)
        step_timings["2b_advanced_opts"] = 0  # negligible

        # ── Step 2: Technical Analysis ─────────────────────────────────
        t0 = time.monotonic()
        # Load multi-TF data from cache (populated by _step_gather_data)
        mtf = self._mtf_cache.get(index_id, {})
        htf_rows = mtf.get("1h") or mtf.get("15m") or []
        ltf_rows = mtf.get("1m") or []
        htf_df = self._rows_to_df(htf_rows) if htf_rows else None
        ltf_df = self._rows_to_df(ltf_rows) if ltf_rows else None
        orb_data = self._get_orb(index_id, now)

        _iv_hist = self._iv_history_cache.get(index_id, [])
        _fii_net = self._fii_futures_cache.get(index_id)
        technical_result = self._step_technical(
            index_id, price_df, options_chain, vix_value, benchmark_rows, timeframe_label,
            htf_df=htf_df, ltf_df=ltf_df, orb_data=orb_data,
            iv_history=_iv_hist,
            advanced_opts=advanced_opts,
            fii_futures_net=_fii_net,
        )
        step_timings["2_technical"] = int((time.monotonic() - t0) * 1000)

        # ── Step 3: News Assessment ────────────────────────────────────
        t0 = time.monotonic()
        news_vote, event_modifier = self._step_news(index_id)
        step_timings["3_news"] = int((time.monotonic() - t0) * 1000)

        # ── Step 3b: Overnight data overlay ────────────────────────────
        overnight = self._load_overnight_snapshot()
        overnight_weight = self._overnight_decay_weight(now)
        if overnight and overnight_weight > 0:
            news_vote = self._apply_overnight_to_news_vote(
                news_vote, overnight, overnight_weight, index_id,
            )

        # Apply GIFT NIFTY pre-market gap bias (if available and within first 45 min)
        if overnight_weight > 0.4:  # Only effective before 10:00 AM
            try:
                gift_row = self._db.fetch_one(
                    "SELECT implied_gap_pts, gap_classification FROM gift_nifty ORDER BY timestamp DESC LIMIT 1",
                    ()
                )
                if gift_row and gift_row.get("implied_gap_pts"):
                    gap_pts = float(gift_row["implied_gap_pts"])
                    if abs(gap_pts) > 30:
                        logger.debug("GIFT NIFTY gap %.0f pts will bias gap fill detection", gap_pts)
                        # Store for anomaly engine to use as gap signal
                        self.health_status["gift_nifty_gap_pts"] = round(gap_pts, 1)
                        self.health_status["gift_nifty_gap_class"] = gift_row.get("gap_classification", "")
            except Exception:
                pass

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
            advanced_opts=advanced_opts,
        )
        step_timings["5_regime"] = int((time.monotonic() - t0) * 1000)

        # ── Step 5c: BankNIFTY/NIFTY Divergence → inject into regime ──
        try:
            div_result = self._compute_banknifty_nifty_divergence()
            if div_result and div_result.get("signal") not in ("ALIGNED", None):
                regime.warnings.append(div_result["signal"])
                div_pct = div_result.get("divergence_pct", 0)
                logger.debug(
                    "BankNIFTY divergence [%s]: %s (%.3f%%)",
                    index_id, div_result["signal"], div_pct,
                )
        except Exception:
            pass  # Non-fatal

        # ── Step 5b: Hero Zero Detection ───────────────────────────────
        t0 = time.monotonic()
        hz_result = self._step_hero_zero(index_id, options_chain, intraday_rows, as_of)
        step_timings["5b_hero_zero"] = int((time.monotonic() - t0) * 1000)

        # ── Step 5d: GEX Regime Overlay ───────────────────────────────
        t0 = time.monotonic()
        gex_block_reason = self._step_gex_regime(
            index_id=index_id,
            options_chain=options_chain,
            current_price=current_price,
            vix_value=vix_value,
            price_df=price_df,
            event_modifier=event_modifier,
        )
        if vix_value is not None:
            self._vix_readings.append((now, vix_value))
            self._vix_readings = self._vix_readings[-20:]
        step_timings["5d_gex"] = int((time.monotonic() - t0) * 1000)

        # ── Step 6: Signal Generation ──────────────────────────────────
        t0 = time.monotonic()
        anomaly_vote = self._anomaly_result_to_vote(anomaly_result, index_id)
        raw_signal = self._step_signal(
            index_id, technical_result, anomaly_vote, regime, current_price,
            hz_result=hz_result, as_of=as_of
        )
        step_timings["6_signal"] = int((time.monotonic() - t0) * 1000)

        # Apply GEX block (after signal generation so signal_type is known)
        if (
            gex_block_reason
            and getattr(raw_signal, "signal_type", "NO_TRADE") in ("BUY_CALL", "BUY_PUT")
        ):
            logger.info(
                "DecisionEngine [%s]: GEX blocks directional signal — %s",
                index_id, gex_block_reason,
            )
            raw_signal = self._make_no_trade_signal(index_id, regime, gex_block_reason)

        # ── Step 6b: Attach Recommended Strike to Signal ───────────────
        if (
            getattr(raw_signal, "signal_type", "NO_TRADE") in ("BUY_CALL", "BUY_PUT")
            and options_chain is not None
            and current_price > 0
        ):
            try:
                idx_def = self.registry.get_index(index_id)
                interval = 100.0 if (idx_def and "BANK" in (idx_def.id or "").upper()) else 50.0
                rec_strike, rec_delta = self._select_optimal_strike(
                    raw_signal.signal_type,
                    current_price,
                    options_chain,
                    strike_interval=interval,
                )
                # Attach to signal using object.__setattr__ (works on both dataclasses and regular objects)
                try:
                    object.__setattr__(raw_signal, "recommended_strike", rec_strike)
                    object.__setattr__(raw_signal, "recommended_delta", round(rec_delta, 4))
                except (AttributeError, TypeError):
                    # If dataclass is frozen or setattr not supported, skip
                    pass
                logger.debug(
                    "Recommended strike for %s %s: %.0f (Δ≈%.2f)",
                    index_id, raw_signal.signal_type, rec_strike, rec_delta,
                )
            except Exception as _e:
                logger.debug("Strike selection failed (non-fatal): %s", _e)

        # ── Step 6c: Liquidity Engine overlay ─────────────────────────
        t0 = time.monotonic()
        raw_signal = self._step_liquidity(
            index_id, raw_signal, current_price, regime,
            options_chain, stale_warnings,
        )
        step_timings["6c_liquidity"] = int((time.monotonic() - t0) * 1000)

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

        # Surface the overnight gap prediction as a warning during the
        # first hour so traders are aware of the pre-market bias.
        overnight_warnings: list[str] = []
        if overnight and overnight_weight > 0:
            gap = (overnight.get("payload", {}) or {}).get("gap_prediction") or {}
            gap_conf = gap.get("confidence") or 0
            gap_dir = gap.get("direction")
            if gap_dir and gap_dir != "FLAT_OPEN" and gap_conf > 0.5:
                overnight_warnings.append(
                    f"Overnight gap: {gap_dir} ({gap_conf:.0%}) — "
                    f"weight {overnight_weight:.0%}"
                )
            regime_hint = overnight.get("regime_expectation")
            if regime_hint and regime_hint not in ("NORMAL", None):
                overnight_warnings.append(
                    f"Overnight regime hint: {regime_hint}"
                )

        rollover_warnings: list[str] = []
        if self._is_rollover_week(now):
            rollover_warnings.append(
                "ROLLOVER WEEK: Near monthly expiry — liquidity may drop, spreads wider. "
                "Use stricter entry filters."
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
            warnings=stale_warnings + overnight_warnings + rollover_warnings,
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

        from config.settings import settings
        fo_indices = self.registry.get_trading_universe(
            major_only=settings.trading.restrict_to_major_indices
        )
        if not fo_indices:
            logger.warning("DecisionEngine.run_all_indices: no F&O indices in registry")
            return []

        index_ids = [idx.id for idx in fo_indices]

        # ── Priority Stock Expansion (Top 15) ──────────────────────────
        # If F&O Universe manager is wired, we include the top 15 priority
        # stocks in the full analysis cycle.
        if self._fo_manager:
            try:
                priority_symbols = self._fo_manager.get_priority_symbols()
                for symbol in priority_symbols:
                    if symbol not in index_ids:
                        index_ids.append(symbol)
                        # Ensure the signal generator allows these stocks
                        self.signal_generator.add_allowed_symbol(symbol)
            except Exception:
                logger.exception("Failed to fetch priority symbols for signal generation")

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

                    # Push position-exit event to WebSocket clients
                    try:
                        outcome = "WIN" if update.current_pnl >= 0 else "LOSS"
                        queue_event("position_exit", {
                            "index_id": pos.index_id,
                            "trade_type": getattr(pos, "signal_type", ""),
                            "exit_reason": exit_reason,
                            "pnl": round(update.current_pnl, 2),
                            "pnl_pct": round(update.current_pnl_pct * 100, 2),
                            "outcome": outcome,
                            "entry_price": getattr(pos, "entry_price", 0.0),
                            "exit_price": current_price,
                        })
                    except Exception as e:
                        logger.debug("WebSocket exit broadcast failed (non-fatal): %s", e)
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

        # VIX-based strategy mode recommendation
        try:
            if dashboard.vix_value > 20:
                vix_strategy = "SELL_PREMIUM"
            elif dashboard.vix_value < 14:
                vix_strategy = "BUY_OPTIONS"
            else:
                vix_strategy = "NEUTRAL"
            # Store in health status for API access
            self.health_status["vix_strategy_mode"] = vix_strategy
        except Exception:
            pass

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

    def _step_gather_data(self, index_id: str, as_of: Optional[datetime] = None) -> tuple:
        """Step 1 — Load data from DB. Each sub-fetch is isolated."""
        as_of_ts = as_of.isoformat() if as_of else "9999-12-31T23:59:59"
        
        current_bar: Optional[dict] = None
        daily_rows: list[dict] = []
        intraday_rows: list[dict] = []
        options_chain = None
        oi_summary: Optional[dict] = None
        vix_value: Optional[float] = None
        benchmark_rows: list[dict] = []

        try:
            current_bar = (
                self._db.fetch_one(Q.GET_LATEST_PRICE, (index_id, "5m", as_of_ts))
                or self._db.fetch_one(Q.GET_LATEST_PRICE, (index_id, "1d", as_of_ts))
            )
        except Exception:
            logger.warning("Step 1: latest bar unavailable for %s", index_id)

        try:
            daily_rows = self._db.fetch_all(
                Q.LIST_PRICE_HISTORY_LIMIT, (index_id, "1d", as_of_ts, 250)
            )
        except Exception:
            logger.warning("Step 1: daily history unavailable for %s", index_id)

        _intraday_tf = "5m"
        try:
            intraday_rows = self._db.fetch_all(
                Q.LIST_PRICE_HISTORY_LIMIT, (index_id, "5m", as_of_ts, 100)
            )
            if not intraday_rows:
                intraday_rows = self._db.fetch_all(
                    Q.LIST_PRICE_HISTORY_LIMIT, (index_id, "1m", as_of_ts, 500)
                )
                if intraday_rows:
                    _intraday_tf = "1m"
                    logger.debug("Step 1: using 1m intraday fallback for %s (%d bars)", index_id, len(intraday_rows))
        except Exception:
            logger.debug("Step 1: intraday unavailable for %s", index_id)

        index_def = self.registry.get_index(index_id)
        if index_def and index_def.has_options:
            try:
                options_chain = self._load_options_chain(index_id, as_of=as_of)
            except Exception:
                logger.warning("Step 1: options chain unavailable for %s", index_id)
            try:
                oi_summary = self._load_oi_summary(index_id, as_of=as_of)
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
                    Q.LIST_PRICE_HISTORY_LIMIT, ("NIFTY50", "1d", as_of_ts, 250)
                )
        except Exception:
            logger.debug("Step 1: benchmark data unavailable")

        # Load 15m and 1h for multi-timeframe analysis
        intraday_15m: list[dict] = []
        intraday_1h: list[dict] = []
        try:
            intraday_15m = self._db.fetch_all(
                Q.LIST_PRICE_HISTORY_LIMIT, (index_id, "15m", as_of_ts, 50)
            )
        except Exception:
            logger.debug("Step 1: 15m bars unavailable for %s", index_id)
        try:
            intraday_1h = self._db.fetch_all(
                Q.LIST_PRICE_HISTORY_LIMIT, (index_id, "1h", as_of_ts, 30)
            )
        except Exception:
            logger.debug("Step 1: 1h bars unavailable for %s", index_id)

        self._mtf_cache[index_id] = {
            "15m": intraday_15m,
            "1h": intraday_1h,
        }

        # Load 30-day IV history for IV percentile calculation
        try:
            iv_rows = self._db.fetch_all(
                Q.LIST_IV_HISTORY_30D, (index_id,)
            )
            iv_list = [float(r["atm_iv"]) for r in iv_rows if r.get("atm_iv", 0) > 0]
            self._iv_history_cache[index_id] = iv_list
        except Exception:
            logger.debug("Step 1: IV history unavailable for %s", index_id)
            self._iv_history_cache.setdefault(index_id, [])

        # Load FII futures net position for SmartMoney OIMI wiring
        fii_futures_net: Optional[float] = None
        try:
            fii_row = self._db.fetch_one(
                "SELECT net_value FROM fii_dii_activity "
                "WHERE segment = 'F&O_FUTURES' ORDER BY date DESC LIMIT 1",
                ()
            )
            if fii_row and fii_row["net_value"] is not None:
                fii_futures_net = float(fii_row["net_value"])
        except Exception:
            logger.debug("Step 1: FII futures position unavailable for %s", index_id)
        self._fii_futures_cache[index_id] = fii_futures_net

        return (
            current_bar, daily_rows, intraday_rows,
            options_chain, oi_summary, vix_value, benchmark_rows,
            _intraday_tf,
        )

    def _step_technical(
        self,
        index_id: str,
        price_df: pd.DataFrame,
        options_chain,
        vix_value: Optional[float],
        benchmark_rows: list[dict],
        timeframe: str = "1d",
        htf_df: Optional[pd.DataFrame] = None,
        ltf_df: Optional[pd.DataFrame] = None,
        orb_data: Optional[dict] = None,
        iv_history: Optional[list[float]] = None,
        advanced_opts=None,
        fii_futures_net: Optional[float] = None,
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
                timeframe=timeframe,
                htf_df=htf_df,
                ltf_df=ltf_df,
                orb_data=orb_data,
                iv_history=iv_history,
                advanced_opts=advanced_opts,
                fii_futures_net=fii_futures_net,
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
        advanced_opts=None,
    ) -> MarketRegime:
        """Step 5 — Regime detection. Always returns a valid MarketRegime."""
        try:
            if technical_result is None:
                return self._fallback_regime(index_id, "No technical result")

            # Load FII futures positioning from DB
            fii_futures_bias = None
            fii_net_position = None
            try:
                row = self._db.fetch_one(
                    "SELECT net_value FROM fii_dii_activity WHERE segment = 'F&O_FUTURES' ORDER BY date DESC LIMIT 1",
                    ()
                )
                if row:
                    fii_net_position = float(row["net_value"] or 0)
                    fii_futures_bias = "BULLISH" if fii_net_position > 5000 else ("BEARISH" if fii_net_position < -5000 else "NEUTRAL")
            except Exception:
                pass

            # Load AD ratio from DB
            ad_ratio = None
            try:
                row = self._db.fetch_one(
                    "SELECT ad_ratio FROM market_breadth ORDER BY timestamp DESC LIMIT 1",
                    ()
                )
                if row:
                    ad_ratio = float(row["ad_ratio"] or 0)
            except Exception:
                pass

            # Load futures basis from DB
            futures_basis = None
            try:
                row = self._db.fetch_one(
                    "SELECT basis FROM futures_basis WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (index_id if index_id == "NIFTY50" else "NIFTY50",)
                )
                if row:
                    futures_basis = float(row["basis"] or 0)
            except Exception:
                pass

            # Extract GEX from advanced options analytics
            gex_regime = None
            gex_value = None
            if advanced_opts is not None:
                gex_regime = getattr(advanced_opts, "gex_regime", None)
                gex_value = getattr(advanced_opts, "gex", None)

            return self.regime_detector.detect_regime(
                index_id=index_id,
                price_df=price_df,
                technical_result=technical_result,
                news_event_modifier=event_modifier,
                anomaly_result=anomaly_result,
                vix_value=vix_value,
                fii_futures_bias=fii_futures_bias,
                fii_net_position=fii_net_position,
                gex_regime=gex_regime,
                gex_value=gex_value,
                ad_ratio=ad_ratio,
                futures_basis=futures_basis,
            )
        except Exception:
            logger.exception("Step 5: regime detection failed for %s", index_id)
            return self._fallback_regime(index_id, "Regime detection error")

    def _step_hero_zero(
        self,
        index_id: str,
        chain: Optional[object],
        intraday_rows: list[dict],
        now: Optional[datetime],
    ) -> Optional[object]:
        """Step 5b — Hero Zero gamma squeeze detection.

        Returns HeroZeroResult or None if not in window / chain missing.
        Never raises — failure is non-fatal.
        """
        try:
            if chain is None:
                return None

            idx = self.registry.get_index(index_id)
            lot_size = (idx.lot_size or 25) if idx else 25
            vol_ratio = self._compute_volume_ratio(intraday_rows)
            current_now = now or datetime.now(tz=_IST)

            return self._hz_detector.detect(
                index_id=index_id,
                chain=chain,
                price_bars=intraday_rows[-5:] if len(intraday_rows) >= 5 else intraday_rows,
                volume_ratio=vol_ratio,
                now=current_now,
                lot_size=lot_size,
            )
        except Exception:
            logger.exception("Step 5b: Hero Zero detection failed for %s (non-fatal)", index_id)
            return None

    def _compute_volume_ratio(self, price_bars: list[dict]) -> float:
        """Compute current volume vs 20-period average.

        Returns 0.0 if insufficient data.
        """
        if len(price_bars) < 5:
            return 0.0
        vols = [float(b.get("volume", 0) or 0) for b in price_bars]
        if not vols:
            return 0.0
        current = vols[-1]
        avg = sum(vols[-20:]) / min(len(vols), 20) if vols else 0.0
        return current / avg if avg > 0 else 0.0

    def _step_signal(
        self,
        index_id: str,
        technical_result: Optional[TechnicalAnalysisResult],
        anomaly_vote: Optional[AnomalyVote],
        regime: MarketRegime,
        current_price: float,
        hz_result: Optional[object] = None,
        as_of: Optional[datetime] = None,
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
                news=None,
                anomaly=anomaly_vote,
                regime=regime,
                hero_zero_result=hz_result,
                current_spot_price=current_price,
                timestamp=as_of,
            )
        except Exception:
            logger.exception("Step 6: signal generation failed for %s", index_id)
            return self._make_no_trade_signal(index_id, regime, "Signal generation error")

    def _step_liquidity(
        self,
        index_id: str,
        raw_signal: TradingSignal,
        current_price: float,
        regime: MarketRegime,
        options_chain,
        stale_warnings: list[str],
    ) -> TradingSignal:
        """Step 6c — Liquidity engine overlay (non-fatal)."""
        if self.liquidity_engine is None:
            return raw_signal

        proposed = getattr(raw_signal, "signal_type", "NO_TRADE")
        if proposed == "NO_TRADE":
            return raw_signal

        try:
            # Gather inputs
            futures_ltp = float(getattr(raw_signal, "entry_price", current_price) or current_price)
            basis_history: list[float] = []
            try:
                rows = self._db.fetch_all(
                    "SELECT basis FROM futures_basis WHERE index_id = ? "
                    "ORDER BY timestamp DESC LIMIT 20",
                    (index_id,),
                )
                basis_history = [float(r["basis"]) for r in rows if r.get("basis") is not None]
            except Exception:
                pass

            atm_ce_oi_change = 0
            atm_pe_oi_change = 0
            if options_chain is not None:
                try:
                    # Find ATM strike closest to current_price
                    strikes = getattr(options_chain, "strikes", [])
                    if strikes:
                        atm = min(strikes, key=lambda s: abs(s.strike_price - current_price))
                        atm_ce_oi_change = int(getattr(atm, "ce_oi_change", 0) or 0)
                        atm_pe_oi_change = int(getattr(atm, "pe_oi_change", 0) or 0)
                except Exception:
                    pass

            liq_input = self.liquidity_engine.get_decision_engine_input(
                symbol=index_id,
                proposed_signal=proposed,
                current_price=current_price,
                futures_ltp=futures_ltp,
                futures_basis_history=basis_history,
                atm_ce_oi_change=atm_ce_oi_change,
                atm_pe_oi_change=atm_pe_oi_change,
            )

            action = liq_input.get("signal_action", "ALLOW")
            liq_reason = liq_input.get("reason", "")

            if action == "BLOCK":
                logger.info(
                    "DecisionEngine [%s]: signal BLOCKED by liquidity engine — %s",
                    index_id, liq_reason,
                )
                return self._make_no_trade_signal(
                    index_id, regime, f"Liquidity: {liq_reason}"
                )

            if action == "REVERSE":
                flipped = "BUY_PUT" if proposed == "BUY_CALL" else "BUY_CALL"
                logger.warning(
                    "DecisionEngine [%s]: signal REVERSED by liquidity engine (%s → %s) — %s",
                    index_id, proposed, flipped, liq_reason,
                )
                try:
                    object.__setattr__(raw_signal, "signal_type", flipped)
                    object.__setattr__(raw_signal, "reasoning",
                                       f"[LIQUIDITY REVERSED] {liq_reason}")
                except (AttributeError, TypeError):
                    pass

            # Apply confidence adjustment
            conf_adj = float(liq_input.get("confidence_adjustment", 0.0))
            if conf_adj != 0.0:
                try:
                    current_conf = float(getattr(raw_signal, "confidence_score", 0.0) or 0.0)
                    object.__setattr__(raw_signal, "confidence_score",
                                       max(0.0, current_conf + conf_adj))
                except (AttributeError, TypeError):
                    pass

            # Attach snapshot for dashboard display
            try:
                object.__setattr__(raw_signal, "liquidity_snapshot", liq_input.get("snapshot"))
            except (AttributeError, TypeError):
                pass

            # Use liquidity price targets if available and signal is actionable
            liq_targets = liq_input.get("price_targets", [])
            if liq_targets:
                try:
                    object.__setattr__(raw_signal, "target_price", liq_targets[0])
                except (AttributeError, TypeError):
                    pass

        except Exception:
            logger.exception("Step 6c: liquidity overlay failed for %s (non-fatal)", index_id)

        return raw_signal

    # ------------------------------------------------------------------
    # Step 5d — GEX Regime Overlay
    # ------------------------------------------------------------------

    def _step_gex_regime(
        self,
        index_id: str,
        options_chain,
        current_price: float,
        vix_value: Optional[float],
        price_df: Optional[pd.DataFrame],
        event_modifier,
    ) -> Optional[str]:
        """
        Calculate or read latest GEX profile, run regime detector, return
        block reason (non-None → caller should suppress directional signal).

        Also triggers morning GEX brief on first call of the trading day.
        """
        if self.gex_calculator is None or self.gex_regime_detector is None:
            return None

        try:
            # Calculate GEX profile inline if no fresh one exists
            profile = self.gex_calculator._latest_profiles.get(index_id)
            if profile is None and options_chain is not None and current_price > 0:
                profile = self._update_gex_profile(
                    index_id, options_chain, current_price
                )

            if profile is None:
                return None

            vix = vix_value or 15.0
            vix_trend = self._compute_vix_trend()
            price_trend = self._compute_price_trend_15min(index_id, price_df)
            is_expiry = bool(
                event_modifier and getattr(event_modifier, "is_expiry_day", False)
            ) or self._default_is_expiry_day()
            mins_to_close = self._minutes_to_close()

            gex_regime = self.gex_regime_detector.classify_regime(
                symbol=index_id,
                gex_profile=profile,
                vix_level=vix,
                vix_trend=vix_trend,
                current_price=current_price,
                price_trend_15min=price_trend,
                is_expiry_day=is_expiry,
                minutes_to_close=mins_to_close,
            )

            # Morning brief (once per trading day per symbol)
            today_str = datetime.now(tz=_IST).strftime("%Y-%m-%d")
            brief_key = f"{index_id}_{today_str}"
            if brief_key != self._gex_brief_date:
                try:
                    daily_plan = self.gex_regime_detector.get_day_trading_plan(
                        symbol=index_id,
                        gex_profile=profile,
                        vix_level=vix,
                        vix_trend=vix_trend,
                        is_expiry_day=is_expiry,
                    )
                    self._daily_plans[index_id] = daily_plan
                    self._gex_brief_date = brief_key
                    logger.info(
                        "GEX morning plan [%s]: %s | pos_size=%.1fx",
                        index_id,
                        daily_plan.get("day_type", ""),
                        daily_plan.get("position_size_multiplier", 1.0),
                    )
                except Exception:
                    logger.debug("GEX morning plan failed for %s (non-fatal)", index_id)

            # Apply confidence multiplier via GEX signal
            gex_signal = self.gex_calculator.get_gex_signal_for_engine(
                index_id, current_price
            )

            # Block AVOID_DIRECTIONAL when regime is strong
            if (
                gex_regime.get("strategy_type") == "AVOID_DIRECTIONAL"
                and gex_signal.get("regime_strength", 0) > 60
            ):
                return f"GEX blocks directional: {gex_regime.get('reason', 'POSITIVE_GEX regime')}"

            return None

        except Exception:
            logger.debug("_step_gex_regime failed for %s (non-fatal)", index_id)
            return None

    def _update_gex_profile(self, index_id: str, options_chain, current_price: float):
        """Convert options_chain object and calculate GEX profile inline."""
        try:
            from collections import defaultdict
            idx_def = self.registry.get_index(index_id)
            lot_size = getattr(idx_def, "lot_size", 50) or 50

            strikes_raw = getattr(options_chain, "strikes", [])
            if not strikes_raw:
                return None

            chain: dict = defaultdict(lambda: defaultdict(dict))
            for s in strikes_raw:
                expiry = getattr(s, "expiry_date", None)
                if expiry is None:
                    continue
                expiry_key = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry)
                strike = float(getattr(s, "strike_price", 0) or 0)
                if strike <= 0:
                    continue
                chain[strike][expiry_key]["CE"] = {
                    "openInterest": int(getattr(s, "ce_oi", 0) or 0),
                    "lastPrice": float(getattr(s, "ce_ltp", 0) or 0),
                    "impliedVolatility": float(getattr(s, "ce_iv", 0) or 0),
                }
                chain[strike][expiry_key]["PE"] = {
                    "openInterest": int(getattr(s, "pe_oi", 0) or 0),
                    "lastPrice": float(getattr(s, "pe_ltp", 0) or 0),
                    "impliedVolatility": float(getattr(s, "pe_iv", 0) or 0),
                }

            if not chain:
                return None

            return self.gex_calculator.calculate_gex_profile(
                symbol=index_id,
                spot_price=current_price,
                options_chain=dict(chain),
                lot_size=lot_size,
                current_time=datetime.now(tz=_IST),
            )
        except Exception:
            logger.debug("_update_gex_profile failed for %s (non-fatal)", index_id)
            return None

    def _compute_vix_trend(self) -> str:
        """Compute VIX trend from recent readings: 'RISING' | 'FALLING' | 'FLAT'."""
        if len(self._vix_readings) < 3:
            return "FLAT"
        recent = [v for _, v in self._vix_readings[-6:]]
        change_pct = (recent[-1] - recent[0]) / recent[0] * 100 if recent[0] > 0 else 0
        if change_pct > 5:
            return "RISING"
        if change_pct < -5:
            return "FALLING"
        return "FLAT"

    def _compute_price_trend_15min(
        self, index_id: str, price_df: Optional[pd.DataFrame]
    ) -> str:
        """Compute 15-min price trend: 'UP' | 'DOWN' | 'SIDEWAYS'."""
        try:
            if price_df is None or len(price_df) < 3:
                return "SIDEWAYS"
            closes = price_df["close"].iloc[-3:].tolist()
            if closes[0] <= 0:
                return "SIDEWAYS"
            chg_pct = (closes[-1] - closes[0]) / closes[0] * 100
            if chg_pct >= 0.3:
                return "UP"
            if chg_pct <= -0.3:
                return "DOWN"
            return "SIDEWAYS"
        except Exception:
            return "SIDEWAYS"

    @staticmethod
    def _minutes_to_close() -> int:
        """Return minutes until 3:30 PM IST."""
        now = datetime.now(tz=_IST)
        close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        delta = (close - now).total_seconds() / 60
        return max(0, int(delta))

    @staticmethod
    def _default_is_expiry_day() -> bool:
        """Fallback expiry check: Thursday = weekly F&O expiry day."""
        return datetime.now(tz=_IST).weekday() == 3  # Thursday

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

            # Push real-time event to WebSocket clients
            try:
                queue_event("signal", {
                    "index_id": index_id,
                    "signal_type": getattr(final_signal, "signal_type", ""),
                    "confidence": getattr(final_signal, "confidence_level", "LOW"),
                    "confidence_score": getattr(final_signal, "confidence_score", 0.0) or 0.0,
                    "entry": getattr(final_signal, "refined_entry", None)
                        or getattr(final_signal, "entry_price", 0.0) or 0.0,
                    "target": getattr(final_signal, "refined_target", None)
                        or getattr(final_signal, "target_price", 0.0) or 0.0,
                    "stop_loss": getattr(final_signal, "refined_stop_loss", None)
                        or getattr(final_signal, "stop_loss", 0.0) or 0.0,
                    "reasoning_summary": (getattr(final_signal, "reasoning", "") or "")[:200],
                    "alert_message": alert_message or "",
                    "alert_priority": alert_priority,
                })
            except Exception as e:
                logger.debug("WebSocket broadcast failed (non-fatal): %s", e)

        return alert_message, alert_priority

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_latest_bar(self, index_id: str, timeframe: str = "1d") -> Optional[dict]:
        try:
            return self._db.fetch_one(Q.GET_LATEST_PRICE, (index_id, timeframe))
        except Exception:
            return None

    def _load_options_chain(self, index_id: str, as_of: Optional[datetime] = None) -> Optional[OptionsChainData]:
        """Reconstruct OptionsChainData from the latest DB snapshot."""
        try:
            from src.data.options_chain import OptionsChainData, OptionStrike
            from datetime import date as date_
            
            as_of_ts = as_of.isoformat() if as_of else "9999-12-31T23:59:59"

            expiry_rows = self._db.fetch_all(Q.LIST_EXPIRY_DATES, (index_id,))
            if not expiry_rows:
                return None

            expiry_str = expiry_rows[0]["expiry_date"]
            rows = self._db.fetch_all(
                Q.LIST_OPTIONS_CHAIN_FOR_EXPIRY,
                (index_id, expiry_str, as_of_ts, index_id, expiry_str, as_of_ts),
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
                    ce_oi_change_pct=float(ce.get("oi_change_pct", 0.0)),
                    ce_volume=int(ce.get("volume", 0)),
                    ce_ltp=float(ce.get("ltp", 0.0)),
                    ce_ltp_change=float(ce.get("ltp_change", 0.0)),
                    ce_ltp_change_pct=float(ce.get("ltp_change_pct", 0.0)),
                    ce_iv=float(ce.get("iv") or 0.0),
                    ce_bid_price=float(ce.get("bid_price", 0.0)),
                    ce_bid_qty=int(ce.get("bid_qty", 0)),
                    ce_ask_price=float(ce.get("ask_price", 0.0)),
                    ce_ask_qty=int(ce.get("ask_qty", 0)),
                    
                    pe_oi=int(pe.get("open_interest", 0)),
                    pe_oi_change=int(pe.get("oi_change", 0)),
                    pe_oi_change_pct=float(pe.get("oi_change_pct", 0.0)),
                    pe_volume=int(pe.get("volume", 0)),
                    pe_ltp=float(pe.get("ltp", 0.0)),
                    pe_ltp_change=float(pe.get("ltp_change", 0.0)),
                    pe_ltp_change_pct=float(pe.get("ltp_change_pct", 0.0)),
                    pe_iv=float(pe.get("iv") or 0.0),
                    pe_bid_price=float(pe.get("bid_price", 0.0)),
                    pe_bid_qty=int(pe.get("bid_qty", 0)),
                    pe_ask_price=float(pe.get("ask_price", 0.0)),
                    pe_ask_qty=int(pe.get("ask_qty", 0)),
                ))

            latest_price_row = self._db.fetch_one(
                "SELECT close FROM price_data WHERE index_id = ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
                (index_id, as_of_ts),
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

    def _load_oi_summary(self, index_id: str, as_of: Optional[datetime] = None) -> Optional[OISummary]:
        try:
            from src.data.options_chain import OISummary
            
            as_of_ts = as_of.isoformat() if as_of else "9999-12-31T23:59:59"
            
            rows = self._db.fetch_all(Q.LIST_EXPIRY_DATES, (index_id,))
            if not rows:
                return None
            
            expiry = rows[0]["expiry_date"]
            row = self._db.fetch_one(Q.GET_LATEST_OI_AGGREGATED, (index_id, expiry, as_of_ts))
            if not row:
                return None
            
            return OISummary(
                total_ce_oi=int(row["total_ce_oi"]),
                total_pe_oi=int(row["total_pe_oi"]),
                total_ce_oi_change=int(row["total_ce_oi_change"]),
                total_pe_oi_change=int(row["total_pe_oi_change"]),
                total_ce_volume=0,  # Not in DB
                total_pe_volume=0,  # Not in DB
                pcr=float(row["pcr"]),
                pcr_change=0.0,     # Not in DB
                max_pain_strike=float(row["max_pain_strike"]),
                highest_ce_oi_strike=float(row["highest_ce_oi_strike"]),
                highest_pe_oi_strike=float(row["highest_pe_oi_strike"]),
                top_5_ce_oi_strikes=(), # Not in DB
                top_5_pe_oi_strikes=(), # Not in DB
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
        intraday_tf: str = "5m",
    ) -> tuple[pd.DataFrame, str]:
        """Prioritize intraday for intraday; fall back to 1d."""
        if intraday_rows and len(intraday_rows) >= _MIN_BARS_FALLBACK:
            return self._rows_to_df(intraday_rows), intraday_tf
        if daily_rows:
            return self._rows_to_df(daily_rows), "1d"
        return pd.DataFrame(), "1d"

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

    def _build_dashboard_summary(self, result: DecisionResult, current_price: float) -> dict:
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
            "vix_strategy_mode": self.health_status.get("vix_strategy_mode", "NEUTRAL"),
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
    def _select_optimal_strike(
        direction: str,
        spot_price: float,
        chain,
        strike_interval: float = 50.0,
        target_delta_min: float = 0.35,
        target_delta_max: float = 0.45,
    ) -> tuple[float, float]:
        """Select optimal strike based on delta target (0.35-0.45).

        ATM is not always optimal. 0.35-0.45 delta strikes give the best
        premium-to-probability ratio for directional trades.

        Returns (optimal_strike, approximate_delta).
        Falls back to ATM if no suitable strike found.
        """
        if chain is None or spot_price <= 0:
            atm = round(spot_price / strike_interval) * strike_interval
            return atm, 0.40

        try:
            # ATM baseline
            atm = round(spot_price / strike_interval) * strike_interval

            # For BUY_CALL: look at strikes from ATM to 2-3 strikes OTM
            # For BUY_PUT: look at strikes from ATM to 2-3 strikes OTM (put side)
            candidates = []
            for strike in getattr(chain, 'strikes', []):
                s = getattr(strike, 'strike_price', 0)
                if direction == "BUY_CALL":
                    # Calls: at or slightly OTM (strike >= spot - 1 interval)
                    if spot_price - strike_interval <= s <= spot_price + 3 * strike_interval:
                        # Approximate delta using moneyness
                        moneyness = (spot_price - s) / spot_price
                        approx_delta = 0.5 + moneyness * 5  # crude linear approx
                        approx_delta = max(0.05, min(0.95, approx_delta))
                        if target_delta_min <= approx_delta <= target_delta_max:
                            candidates.append((s, approx_delta))
                else:  # BUY_PUT
                    # Puts: at or slightly OTM (strike <= spot + 1 interval)
                    if spot_price - 3 * strike_interval <= s <= spot_price + strike_interval:
                        moneyness = (s - spot_price) / spot_price
                        approx_delta = 0.5 + moneyness * 5
                        approx_delta = max(0.05, min(0.95, approx_delta))
                        if target_delta_min <= approx_delta <= target_delta_max:
                            candidates.append((s, approx_delta))

            if candidates:
                # Pick candidate closest to mid-delta (0.40)
                best = min(candidates, key=lambda x: abs(x[1] - 0.40))
                return best[0], best[1]

            return atm, 0.40

        except Exception:
            return round(spot_price / strike_interval) * strike_interval, 0.40

    @staticmethod
    def _is_rollover_week(now: datetime) -> bool:
        """Detect if we're in the last week of monthly F&O expiry.

        During rollover week: OI interpretation flips to next month,
        spreads widen, liquidity drops. Apply stricter filters.

        Returns True if within 7 days of month's last Thursday.
        """
        import calendar
        import datetime as dt_mod
        today = now.date()
        year, month = today.year, today.month

        # Find last Thursday of current month
        last_day = calendar.monthrange(year, month)[1]
        last_thursday = None
        for day in range(last_day, 0, -1):
            d = dt_mod.date(year, month, day)
            if d.weekday() == 3:  # Thursday
                last_thursday = d
                break

        if last_thursday is None:
            return False

        days_to_expiry = (last_thursday - today).days
        return 0 <= days_to_expiry <= 7

    @staticmethod
    def _get_gap_analysis(current_price: float, prev_close: float) -> dict:
        """Classify opening gap and compute fill probability.

        Returns dict with: gap_points, gap_type, fill_probability, wait_15min.
        """
        if prev_close <= 0:
            return {"gap_points": 0.0, "gap_type": "FLAT", "fill_probability": 0.5, "wait_15min": False}

        gap = current_price - prev_close
        gap_abs = abs(gap)

        if gap_abs < 30:
            gap_type = "FLAT_OPEN"
            fill_prob = 0.70
            wait = False
        elif gap_abs < 80:
            gap_type = "MODERATE_GAP"
            fill_prob = 0.55
            wait = False
        elif gap_abs < 100:
            gap_type = "LARGE_GAP"
            fill_prob = 0.45
            wait = True
        else:
            gap_type = "EXTREME_GAP"
            fill_prob = 0.35
            wait = True

        return {
            "gap_points": round(gap, 1),
            "gap_direction": "UP" if gap > 0 else "DOWN",
            "gap_type": gap_type,
            "fill_probability": fill_prob,
            "wait_15min": wait,
        }

    def _update_orb_cache(self, index_id: str, now: datetime, current_bar: Optional[dict]) -> None:
        """Track Opening Range (9:15-9:30 AM) high/low for ORB detection."""
        if current_bar is None:
            return
        today = now.date().isoformat()
        cache_key = f"{index_id}_{today}"

        # Only capture during 9:15-9:30 window
        if now.hour == 9 and 15 <= now.minute < 30:
            if cache_key not in self._orb_cache:
                self._orb_cache[cache_key] = {
                    "high": float(current_bar.get("high", 0)),
                    "low": float(current_bar.get("low", float("inf"))),
                    "captured_at": now.isoformat(),
                }
            else:
                orb = self._orb_cache[cache_key]
                orb["high"] = max(orb["high"], float(current_bar.get("high", 0)))
                orb["low"] = min(orb["low"], float(current_bar.get("low", float("inf"))))

    def _get_orb(self, index_id: str, now: datetime) -> Optional[dict]:
        """Get Opening Range for today if captured."""
        today = now.date().isoformat()
        cache_key = f"{index_id}_{today}"
        orb = self._orb_cache.get(cache_key)
        if orb and orb["high"] > 0 and orb["low"] < float("inf"):
            return orb
        return None

    def _compute_banknifty_nifty_divergence(self) -> Optional[dict]:
        """Check if BankNIFTY is diverging significantly from NIFTY50.

        Divergence > 0.5% for 30 min = sector rotation signal.
        BankNIFTY leading NIFTY = financial sector rotation in/out.
        """
        try:
            nifty_row = self._db.fetch_one(
                "SELECT close, open FROM price_data WHERE index_id = 'NIFTY50' ORDER BY timestamp DESC LIMIT 1",
                ()
            )
            bank_row = self._db.fetch_one(
                "SELECT close, open FROM price_data WHERE index_id = 'BANKNIFTY' ORDER BY timestamp DESC LIMIT 1",
                ()
            )
            if not nifty_row or not bank_row:
                return None

            nifty_chg = (float(nifty_row["close"]) - float(nifty_row["open"])) / float(nifty_row["open"]) * 100
            bank_chg = (float(bank_row["close"]) - float(bank_row["open"])) / float(bank_row["open"]) * 100

            divergence = bank_chg - nifty_chg

            if abs(divergence) < 0.3:
                signal = "ALIGNED"
            elif divergence > 0.5:
                signal = "BANKNIFTY_LEADING_BULLISH"
            elif divergence < -0.5:
                signal = "BANKNIFTY_LAGGING_BEARISH"
            else:
                signal = "MILD_DIVERGENCE"

            return {
                "nifty_change_pct": round(nifty_chg, 3),
                "banknifty_change_pct": round(bank_chg, 3),
                "divergence_pct": round(divergence, 3),
                "signal": signal,
            }
        except Exception as e:
            logger.debug("BankNIFTY/NIFTY divergence computation failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Overnight data overlay
    # ------------------------------------------------------------------

    def _load_overnight_snapshot(self) -> Optional[dict]:
        """Load today's overnight snapshot (cached per trade date)."""
        today_iso = datetime.now(tz=_IST).date().isoformat()
        with self._lock:
            if self._overnight_cache_date == today_iso:
                return self._overnight_cache

        try:
            from src.analysis.news.overnight_engine import load_latest_overnight_data

            data = load_latest_overnight_data(self._db, trade_date=today_iso)
            with self._lock:
                self._overnight_cache = data
                self._overnight_cache_date = today_iso
            return data
        except Exception:
            logger.debug("Overnight snapshot load failed (non-fatal)", exc_info=True)
            return None

    @staticmethod
    def _overnight_decay_weight(now: datetime) -> float:
        """Time-decay weight for overnight data importance.

        9:15 AM = 100%, 10:00 AM = 50%, 11:00 AM = 20%, >=12:00 PM = 0%.
        Outside the 9:15-12:00 window returns 0.
        """
        minutes_since_open = (now.hour - 9) * 60 + (now.minute - 15)
        if minutes_since_open < 0 or minutes_since_open >= 165:
            return 0.0
        if minutes_since_open <= 45:
            return 1.0 - (minutes_since_open / 45.0) * 0.5   # 1.0 → 0.5
        if minutes_since_open <= 105:
            return 0.5 - ((minutes_since_open - 45) / 60.0) * 0.3  # 0.5 → 0.2
        return max(0.0, 0.2 - ((minutes_since_open - 105) / 60.0) * 0.2)  # 0.2 → 0.0

    def _apply_overnight_to_news_vote(
        self,
        news_vote: Optional[NewsVote],
        overnight: dict,
        weight: float,
        index_id: str,
    ) -> Optional[NewsVote]:
        """Blend overnight sentiment into the news vote when intraday news is thin.

        Only takes effect when the current vote is NEUTRAL or missing — we
        never override a strong live signal with yesterday-evening sentiment.
        """
        if news_vote is not None and news_vote.vote not in ("NEUTRAL", ""):
            return news_vote

        sentiment = overnight.get("overall_sentiment")
        if sentiment is None:
            return news_vote

        if sentiment > 0.5:
            vote = "STRONG_BULLISH"
        elif sentiment > 0.2:
            vote = "BULLISH"
        elif sentiment < -0.5:
            vote = "STRONG_BEARISH"
        elif sentiment < -0.2:
            vote = "BEARISH"
        else:
            vote = "NEUTRAL"

        confidence = min(0.6, abs(sentiment) * weight)
        reasoning = (
            f"Overnight sentiment ({sentiment:+.2f}, weight {weight:.0%}) | "
            f"US impact: {overnight.get('us_impact', 'N/A')}, "
            f"FII: {overnight.get('fii_bias', 'N/A')}"
        )

        return NewsVote(
            index_id=index_id,
            timestamp=datetime.now(tz=_IST),
            vote=vote,
            confidence=round(confidence, 2),
            active_article_count=0,
            weighted_sentiment=float(sentiment),
            top_headline=None,
            event_regime="NORMAL",
            reasoning=reasoning,
        )

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
