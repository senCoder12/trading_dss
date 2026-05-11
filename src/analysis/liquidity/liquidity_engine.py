"""
LiquidityEngine — master orchestrator for all SMC liquidity analysis.

Single entry point for the DecisionEngine. Runs all sub-analyzers every
cycle and returns a unified LiquiditySnapshot.  The DecisionEngine calls
get_decision_engine_input() which returns a structured action dict used to
ALLOW / BLOCK / REVERSE / BOOST the preliminary signal.
"""

from __future__ import annotations

import logging
import threading
import weakref
from datetime import datetime
from typing import Optional

from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from src.analysis.liquidity.candle_store import CandleStore
from src.analysis.liquidity.imbalance_detector import ImbalanceDetector
from src.analysis.liquidity.liquidity_pool_detector import LiquidityPoolDetector
from src.analysis.liquidity.models import LiquidityLevel, LiquiditySnapshot
from src.analysis.liquidity.stop_hunt_detector import StopHuntDetector
from src.analysis.liquidity.sweep_detector import SweepDetector
from src.analysis.liquidity.volume_profile import VolumeProfileAnalyzer
from src.database.db_manager import DatabaseManager
from src.data.smart_api_client import SmartAPIClient

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# Composite score weights — must sum to 100
_W_BIAS = 30
_W_SWEEP = 25
_W_IMBALANCE = 25
_W_VP = 20


class LiquidityEngine:
    """
    Master orchestrator that combines all liquidity sub-analyzers and
    exposes a single clean interface to the DecisionEngine.
    """

    def __init__(
        self,
        candle_store: CandleStore,
        db: DatabaseManager,
        smart_api: SmartAPIClient,
        options_analyzer,
        decision_engine_ref,
    ) -> None:
        self._db = db
        self._de_ref = decision_engine_ref

        # Sub-components
        self.pool_detector = LiquidityPoolDetector(candle_store, db)
        self.sweep_detector = SweepDetector(candle_store, self.pool_detector, db)
        self.imbalance_detector = ImbalanceDetector(candle_store, db)
        self.volume_profile = VolumeProfileAnalyzer(candle_store, db)
        self.stop_hunt_detector = StopHuntDetector(
            candle_store, self.sweep_detector, self.pool_detector, options_analyzer, db
        )

        # Config
        self.liquidity_signal_weight: float = 0.20
        self.min_liquidity_confidence: float = 40.0

        self._snapshots: dict[str, LiquiditySnapshot] = {}
        self._snapshot_lock = threading.Lock()

        logger.info("LiquidityEngine initialised — all sub-components ready")

    # ── Core orchestration ─────────────────────────────────────────────────────

    def analyze(
        self,
        symbol: str,
        current_price: float,
        futures_ltp: float,
        futures_basis_history: list[float],
        atm_ce_oi_change: int,
        atm_pe_oi_change: int,
        gex_profile=None,
    ) -> LiquiditySnapshot:
        """
        Run full liquidity analysis pipeline for *symbol*.

        Returns a LiquiditySnapshot and caches it under self._snapshots[symbol].
        gex_profile: optional GEXProfile — when provided, adjusts sweep signal
        quality and adds GEX walls as synthetic liquidity levels.
        """
        try:
            return self._run_pipeline(
                symbol, current_price, futures_ltp,
                futures_basis_history, atm_ce_oi_change, atm_pe_oi_change,
                gex_profile=gex_profile,
            )
        except Exception:
            logger.exception("LiquidityEngine.analyze failed for %s — returning empty snapshot", symbol)
            empty = LiquiditySnapshot(
                symbol=symbol,
                timestamp=datetime.now(tz=_IST),
                timeframe="5min",
            )
            with self._snapshot_lock:
                self._snapshots[symbol] = empty
            return empty

    def _run_pipeline(
        self,
        symbol: str,
        current_price: float,
        futures_ltp: float,
        futures_basis_history: list[float],
        atm_ce_oi_change: int,
        atm_pe_oi_change: int,
        gex_profile=None,
    ) -> LiquiditySnapshot:
        # 1. Active liquidity levels
        bsl_levels, ssl_levels = self.pool_detector.get_all_active_levels(symbol, current_price)

        # Inject GEX walls as synthetic max-strength liquidity levels
        if gex_profile is not None:
            now_ts = datetime.now(tz=_IST)
            if gex_profile.call_wall:
                bsl_levels = list(bsl_levels) + [LiquidityLevel(
                    symbol=symbol,
                    level_type="CALL_WALL",
                    price=gex_profile.call_wall,
                    strength=5,
                    timeframe="5min",
                    first_formed_at=now_ts,
                    last_touched_at=now_ts,
                )]
            if gex_profile.put_wall:
                ssl_levels = list(ssl_levels) + [LiquidityLevel(
                    symbol=symbol,
                    level_type="PUT_WALL",
                    price=gex_profile.put_wall,
                    strength=5,
                    timeframe="5min",
                    first_formed_at=now_ts,
                    last_touched_at=now_ts,
                )]

        # 2. Recent sweeps (last 30 min)
        recent_sweeps = self.sweep_detector.get_recent_sweeps(symbol, 30)

        # 3. Unfilled FVGs and active OBs (read internal stores directly)
        unfilled_fvgs = [
            fvg for fvg in self.imbalance_detector._active_fvgs.get(symbol, [])
            if not fvg.is_filled and not fvg.is_mitigated
        ]
        active_obs = [
            ob for ob in self.imbalance_detector._active_obs.get(symbol, [])
            if not ob.is_mitigated
        ]

        # 4. Volume profile
        vp = self.volume_profile.load_previous_day_profile(symbol)
        vp_context = self.volume_profile.get_price_context(symbol, current_price)

        # 5. Stop hunt on latest candle
        latest_candle = self.pool_detector._cs.get_latest_candle(symbol, "5min")
        stop_hunt: Optional[dict] = None
        if latest_candle is not None:
            try:
                stop_hunt = self.stop_hunt_detector.detect_stop_hunt(
                    symbol, latest_candle, current_price,
                    futures_ltp, futures_basis_history,
                    atm_ce_oi_change, atm_pe_oi_change,
                )
            except Exception:
                logger.debug("stop_hunt detection failed for %s (non-fatal)", symbol)

        # 6. Liquidity bias
        bias, bias_reason = self.pool_detector._calculate_liquidity_bias(symbol, current_price)

        # 7. Imbalance signal
        imbalance_signal: dict = {"signal": "NEUTRAL", "strength": 0.0, "reason": ""}
        if latest_candle is not None:
            try:
                imbalance_signal = self.imbalance_detector.get_imbalance_signal(
                    symbol, current_price, latest_candle
                )
            except Exception:
                logger.debug("imbalance_signal failed for %s (non-fatal)", symbol)

        # 8. Composite score
        score = self._compute_score(
            bias, recent_sweeps, imbalance_signal, vp_context, stop_hunt,
            gex_profile=gex_profile,
        )

        # 9. Active setups
        active_setups = self._collect_setups(
            recent_sweeps, imbalance_signal, stop_hunt, vp_context, current_price
        )

        # 10. Build snapshot
        snapshot = LiquiditySnapshot(
            symbol=symbol,
            timestamp=datetime.now(tz=_IST),
            timeframe="5min",
            active_bsl_levels=bsl_levels,
            active_ssl_levels=ssl_levels,
            nearest_bsl=self.pool_detector.get_nearest_bsl(symbol, current_price),
            nearest_ssl=self.pool_detector.get_nearest_ssl(symbol, current_price),
            distance_to_bsl_pct=(
                abs(bsl_levels[0].price - current_price) / current_price * 100
                if bsl_levels else 0.0
            ),
            distance_to_ssl_pct=(
                abs(current_price - ssl_levels[0].price) / current_price * 100
                if ssl_levels else 0.0
            ),
            liquidity_bias=bias,
            bias_reason=bias_reason,
            unfilled_fvgs=unfilled_fvgs,
            active_order_blocks=active_obs,
            volume_profile=vp,
            recent_sweeps=recent_sweeps,
            liquidity_score=score,
            active_setups=active_setups,
        )

        with self._snapshot_lock:
            self._snapshots[symbol] = snapshot

        logger.debug(
            "LiquidityEngine [%s]: score=%.1f bias=%s setups=%s",
            symbol, score, bias, active_setups,
        )
        return snapshot

    def _compute_score(
        self,
        bias: str,
        recent_sweeps: list,
        imbalance_signal: dict,
        vp_context: dict,
        stop_hunt: Optional[dict],
        gex_profile=None,
    ) -> float:
        score = 0.0

        # Bias (30%)
        if bias == "BULLISH":
            score += _W_BIAS
        elif bias == "BEARISH":
            score -= _W_BIAS

        # GEX regime sweep quality multiplier
        # NEGATIVE_GEX: sweeps are more violent → higher quality boost
        # POSITIVE_GEX: sweeps signal mean reversion → even higher reliability
        gex_regime = getattr(gex_profile, "regime", None) if gex_profile else None

        # Sweep (25%)
        recent_ssl_sweeps = [s for s in recent_sweeps if s.sweep_type == "SSL_SWEEP"]
        recent_bsl_sweeps = [s for s in recent_sweeps if s.sweep_type == "BSL_SWEEP"]
        if recent_ssl_sweeps:
            base_quality = _W_SWEEP if recent_ssl_sweeps[0].signal_quality == "HIGH" else 15
            if gex_regime == "NEGATIVE_GEX":
                boost = 15 if recent_ssl_sweeps[0].signal_quality == "HIGH" else 5
            elif gex_regime == "POSITIVE_GEX":
                boost = 20 if recent_ssl_sweeps[0].signal_quality == "HIGH" else 10
            else:
                boost = 0
            score += base_quality + boost
        if recent_bsl_sweeps:
            base_quality = _W_SWEEP if recent_bsl_sweeps[0].signal_quality == "HIGH" else 15
            if gex_regime == "NEGATIVE_GEX":
                boost = 15 if recent_bsl_sweeps[0].signal_quality == "HIGH" else 5
            elif gex_regime == "POSITIVE_GEX":
                boost = 20 if recent_bsl_sweeps[0].signal_quality == "HIGH" else 10
            else:
                boost = 0
            score -= (base_quality + boost)

        # Imbalance (25%)
        imb_sig = imbalance_signal.get("signal", "NEUTRAL")
        imb_strength = float(imbalance_signal.get("strength", 0.0) or 0.0)
        if imb_sig == "BULLISH":
            score += imb_strength * (_W_IMBALANCE / 100.0)
        elif imb_sig == "BEARISH":
            score -= imb_strength * (_W_IMBALANCE / 100.0)

        # Volume profile (20%)
        vp_sig = vp_context.get("signal", "NEUTRAL") if vp_context else "NEUTRAL"
        if vp_sig == "BULLISH":
            score += _W_VP
        elif vp_sig == "BEARISH":
            score -= _W_VP

        # Stop hunt override (high-conviction)
        if stop_hunt and stop_hunt.get("confidence", 0) >= 70:
            hunt_type = stop_hunt.get("type", "")
            if hunt_type == "SSL_HUNT":
                score = max(score, 60.0)
            elif hunt_type == "BSL_HUNT":
                score = min(score, -60.0)

        return max(-100.0, min(100.0, score))

    def _collect_setups(
        self,
        recent_sweeps: list,
        imbalance_signal: dict,
        stop_hunt: Optional[dict],
        vp_context: dict,
        current_price: float,
    ) -> list[str]:
        setups: list[str] = []

        if recent_sweeps:
            sweep = recent_sweeps[0]
            direction = "BSL" if sweep.sweep_type == "BSL_SWEEP" else "SSL"
            setups.append(f"{direction} sweep confirmed")

        if imbalance_signal.get("signal", "NEUTRAL") != "NEUTRAL":
            reason = imbalance_signal.get("reason", "")
            if reason:
                setups.append(reason)

        if stop_hunt:
            hunt_type = stop_hunt.get("type", "HUNT")
            confidence = float(stop_hunt.get("confidence", 0.0))
            setups.append(f"{hunt_type} detected ({confidence:.0f}%)")

        if vp_context:
            location = vp_context.get("location", "")
            if location == "LVN":
                setups.append("Price in thin volume zone — expect acceleration")
            elif location in ("ABOVE_VAH", "BELOW_VAL"):
                setups.append("Price outside value area — mean reversion probable")

        return setups

    # ── Primary integration method ─────────────────────────────────────────────

    def get_decision_engine_input(
        self,
        symbol: str,
        proposed_signal: str,
        current_price: float,
        futures_ltp: float,
        futures_basis_history: list[float],
        atm_ce_oi_change: int,
        atm_pe_oi_change: int,
    ) -> dict:
        """
        Primary integration point — DecisionEngine calls this before final scoring.

        Returns a structured dict:
          liquidity_score, signal_action, confidence_adjustment,
          active_setups, price_targets, stop_suggestion, reason, snapshot.
        """
        snapshot = self.analyze(
            symbol, current_price, futures_ltp,
            futures_basis_history, atm_ce_oi_change, atm_pe_oi_change,
        )

        score = snapshot.liquidity_score

        # Default
        action = "ALLOW"
        confidence_adjustment = 0.0
        reason = "Liquidity: neutral"

        # Re-fetch recent sweeps and hunts to pass into override methods
        recent_sweeps = snapshot.recent_sweeps
        recent_hunts = self.stop_hunt_detector.get_recent_hunts(symbol, 20)

        # Priority 1: sweep override
        sweep_override = self.sweep_detector.get_signal_override(
            symbol, proposed_signal, current_price, recent_sweeps
        )
        if sweep_override.get("action") in ("BLOCK", "REVERSE"):
            action = sweep_override["action"]
            confidence_adjustment = float(sweep_override.get("confidence_adjustment", 0.0))
            reason = sweep_override.get("reason", "Sweep override")
        else:
            # Priority 2: stop hunt override
            hunt_override = self.stop_hunt_detector.get_stop_hunt_signal_for_engine(
                symbol, proposed_signal, recent_hunts
            )
            if hunt_override.get("action") in ("BLOCK", "REVERSE"):
                action = hunt_override["action"]
                confidence_adjustment = float(hunt_override.get("confidence_adjustment", 0.0))
                reason = hunt_override.get("reason", "Stop hunt override")
            else:
                # Priority 3: score-based logic
                action, confidence_adjustment, reason = self._score_action(
                    score, proposed_signal,
                    sweep_override, hunt_override,
                )

        # Price targets from volume profile
        vp_direction = "up" if proposed_signal == "BUY_CALL" else "down"
        try:
            price_targets = self.volume_profile.get_vp_targets(symbol, current_price, vp_direction)
        except Exception:
            price_targets = []

        # Stop suggestion: nearest opposing liquidity level
        stop_suggestion = self._compute_stop(symbol, current_price, proposed_signal)

        return {
            "liquidity_score": score,
            "signal_action": action,
            "confidence_adjustment": confidence_adjustment,
            "active_setups": snapshot.active_setups,
            "price_targets": price_targets,
            "stop_suggestion": stop_suggestion,
            "reason": reason,
            "snapshot": snapshot,
        }

    def _score_action(
        self,
        score: float,
        proposed_signal: str,
        sweep_override: dict,
        hunt_override: dict,
    ) -> tuple[str, float, str]:
        """Map composite score → action/confidence_adjustment/reason."""
        sweep_adj = float(sweep_override.get("confidence_adjustment", 0.0))
        hunt_adj = float(hunt_override.get("confidence_adjustment", 0.0))
        passive_adj = sweep_adj + hunt_adj

        if score > 50 and proposed_signal == "BUY_CALL":
            return "BOOST", 10.0 + passive_adj, f"Liquidity bullish (score={score:.0f}) — aligned with BUY_CALL"
        if score > 50 and proposed_signal == "BUY_PUT":
            return "ALLOW", -10.0 + passive_adj, f"Liquidity bullish (score={score:.0f}) — opposes BUY_PUT"
        if score < -50 and proposed_signal == "BUY_PUT":
            return "BOOST", 10.0 + passive_adj, f"Liquidity bearish (score={score:.0f}) — aligned with BUY_PUT"
        if score < -50 and proposed_signal == "BUY_CALL":
            return "ALLOW", -10.0 + passive_adj, f"Liquidity bearish (score={score:.0f}) — opposes BUY_CALL"
        if abs(score) < 30:
            return "ALLOW", passive_adj, f"Liquidity neutral (score={score:.0f})"
        return "ALLOW", passive_adj, f"Liquidity inconclusive (score={score:.0f})"

    def _compute_stop(self, symbol: str, current_price: float, proposed_signal: str) -> float:
        """Return nearest opposing liquidity level as stop suggestion, or 0.0."""
        try:
            if proposed_signal == "BUY_CALL":
                ssl = self.pool_detector.get_nearest_ssl(symbol, current_price)
                return ssl.price if ssl else 0.0
            else:
                bsl = self.pool_detector.get_nearest_bsl(symbol, current_price)
                return bsl.price if bsl else 0.0
        except Exception:
            return 0.0

    # ── Pre-market startup ─────────────────────────────────────────────────────

    def startup(self, symbols: list[str]) -> None:
        """
        Called at 9:00 AM before market opens.

        Loads historical candles, calculates previous day volume profiles,
        detects initial liquidity levels, and pre-populates snapshots.
        """
        logger.info("LiquidityEngine.startup: initialising %d symbols", len(symbols))
        for symbol in symbols:
            try:
                self._startup_symbol(symbol)
            except Exception:
                logger.exception("LiquidityEngine.startup failed for %s", symbol)
        logger.info("LiquidityEngine.startup complete")

    def _startup_symbol(self, symbol: str) -> None:
        # Load previous day volume profile
        vp = self.volume_profile.load_previous_day_profile(symbol)
        if vp:
            logger.debug("LiquidityEngine startup [%s]: loaded VP poc=%.2f", symbol, vp.poc)

        # Pre-populate an empty snapshot so first live cycle has data
        empty = LiquiditySnapshot(
            symbol=symbol,
            timestamp=datetime.now(tz=_IST),
            timeframe="5min",
            volume_profile=vp,
        )
        with self._snapshot_lock:
            self._snapshots[symbol] = empty

    # ── Snapshot access ────────────────────────────────────────────────────────

    def get_snapshot(self, symbol: str) -> Optional[LiquiditySnapshot]:
        with self._snapshot_lock:
            return self._snapshots.get(symbol)
