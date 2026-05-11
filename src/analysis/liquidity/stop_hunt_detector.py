"""
StopHuntDetector — identifies institutional stop raids.

Distinct from SweepDetector (price structure only). A stop hunt requires
all three layers of confirmation:
  1. Price structure — wick through a widely-watched level, body closes back
  2. Options OI shift — institutions writing calls/puts at the hunted strike
  3. Futures basis change — institutional directional positioning in futures

Integration: call detect_stop_hunt() on every 1-min candle close;
call get_stop_hunt_signal_for_engine() before DecisionEngine emits any signal.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from statistics import mean
from typing import Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.analysis.liquidity.candle_store import CandleStore
from src.analysis.liquidity.liquidity_pool_detector import LiquidityPoolDetector
from src.analysis.liquidity.models import Candle, LiquidityLevel
from src.analysis.liquidity.sweep_detector import SweepDetector
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)


class StopHuntDetector:
    """
    Detects institutional stop hunts by combining price structure,
    options OI shifts, and futures basis changes.

    A stop hunt differs from a sweep in that it requires market-data
    confirmation from the derivatives side — price-structure alone is
    insufficient.
    """

    def __init__(
        self,
        candle_store: CandleStore,
        sweep_detector: SweepDetector,
        pool_detector: LiquidityPoolDetector,
        options_analyzer,
        db: DatabaseManager,
    ) -> None:
        self._cs = candle_store
        self._sweep = sweep_detector
        self._pool = pool_detector
        self._options = options_analyzer
        self._db = db

        cfg = settings.stop_hunt
        self._window_candles: int = cfg.stop_hunt_window_candles
        self._min_reversal_pct: float = cfg.min_reversal_strength_pct
        self._min_level_strength: int = cfg.widely_watched_level_min_strength
        self._basis_change_threshold: float = cfg.futures_basis_change_threshold

        # symbol → list of confirmed hunt dicts
        self._confirmed_hunts: dict[str, list[dict]] = {}

        self._ensure_db_tables()

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_stop_hunt(
        self,
        symbol: str,
        candle: Candle,
        current_price: float,
        futures_ltp: float,
        futures_basis_history: list[float],
        atm_ce_oi_change: int,
        atm_pe_oi_change: int,
    ) -> dict | None:
        """
        Analyse *candle* for a stop hunt pattern.

        Returns a stop-hunt dict on detection, None otherwise.
        """
        bsl_levels, ssl_levels = self._pool.get_all_active_levels(symbol, current_price)

        # ── Step 1: find breached widely-watched level ─────────────────────────
        hunted_level: Optional[LiquidityLevel] = None
        is_bsl_hunt: bool = False

        for lvl in bsl_levels:
            if candle.high > lvl.price and lvl.strength >= self._min_level_strength:
                hunted_level = lvl
                is_bsl_hunt = True
                break

        if hunted_level is None:
            for lvl in ssl_levels:
                if candle.low < lvl.price and lvl.strength >= self._min_level_strength:
                    hunted_level = lvl
                    is_bsl_hunt = False
                    break

        if hunted_level is None:
            return None

        # ── Step 2: wick check — body must close back inside the level ─────────
        if candle.total_range <= 0:
            return None

        if is_bsl_hunt:
            body_closed_back = candle.close < hunted_level.price
            wick_dominant = candle.upper_wick / candle.total_range > 0.5
        else:
            body_closed_back = candle.close > hunted_level.price
            wick_dominant = candle.lower_wick / candle.total_range > 0.5

        if not (body_closed_back and wick_dominant):
            return None

        # ── Scoring ────────────────────────────────────────────────────────────
        score: int = 0
        confirmation: dict = {}

        # ── Step 3: options OI confirmation (+30) ─────────────────────────────
        oi_confirms = False
        if is_bsl_hunt:
            if atm_ce_oi_change > 0 and atm_ce_oi_change > abs(atm_pe_oi_change):
                oi_confirms = True
                score += 30
                confirmation["oi"] = f"CE OI +{atm_ce_oi_change} (call writing = bearish intent)"
            else:
                confirmation["oi"] = "No CE OI confirmation"
        else:
            if atm_pe_oi_change > 0 and atm_pe_oi_change > abs(atm_ce_oi_change):
                oi_confirms = True
                score += 30
                confirmation["oi"] = f"PE OI +{atm_pe_oi_change} (put writing = bullish intent)"
            else:
                confirmation["oi"] = "No PE OI confirmation"

        # ── Step 4: futures basis confirmation (+25) ──────────────────────────
        futures_confirms = False
        if len(futures_basis_history) >= 3 and futures_ltp > 0:
            current_basis = futures_ltp - current_price
            prev_basis = mean(futures_basis_history[-3:])
            basis_change = current_basis - prev_basis
            confirmation["basis_change_pts"] = round(basis_change, 2)

            if is_bsl_hunt and basis_change < -self._basis_change_threshold:
                futures_confirms = True
                score += 25
                confirmation["futures"] = f"Basis collapsed {basis_change:.2f} pts (bearish)"
            elif not is_bsl_hunt and basis_change > self._basis_change_threshold:
                futures_confirms = True
                score += 25
                confirmation["futures"] = f"Basis expanded +{basis_change:.2f} pts (bullish)"
            else:
                confirmation["futures"] = f"Basis change {basis_change:.2f} pts below threshold"
        else:
            confirmation["futures"] = "Insufficient basis history"

        # ── Step 5: volume spike (+20) ─────────────────────────────────────────
        recent_20 = self._cs.get_candles(symbol, candle.timeframe, 20)
        avg_vol = _avg_volume(recent_20)
        if avg_vol > 0 and candle.volume > 2.0 * avg_vol:
            score += 20
            confirmation["volume"] = f"Spike {candle.volume / avg_vol:.1f}x avg"
        else:
            confirmation["volume"] = "No volume spike"

        # ── Step 6: reversal speed (+15) ──────────────────────────────────────
        recent_3 = self._cs.get_candles(symbol, candle.timeframe, 3)
        next_candle: Optional[Candle] = None
        for c in recent_3:
            if c.timestamp > candle.timestamp:
                next_candle = c
                break

        if next_candle is not None:
            if is_bsl_hunt and next_candle.close < hunted_level.price:
                score += 15
                confirmation["reversal_speed"] = "Fast — next candle closed back below level"
            elif not is_bsl_hunt and next_candle.close > hunted_level.price:
                score += 15
                confirmation["reversal_speed"] = "Fast — next candle closed back above level"
            else:
                confirmation["reversal_speed"] = "Subsequent candle not yet reversed"
        else:
            confirmation["reversal_speed"] = "No subsequent candle yet"

        # ── Step 7: threshold ──────────────────────────────────────────────────
        if score < 50:
            return None

        confidence = float(min(score, 100))
        if confidence >= 80:
            tier = "HIGH"
        elif confidence >= 60:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        # ── Step 8: entry parameters ───────────────────────────────────────────
        if is_bsl_hunt:
            hunt_type = "BSL_HUNT"
            reversal_direction = "BEARISH"
            suggested_signal = "BUY_PUT"
            hunt_price = candle.high
            entry_zone = (hunted_level.price - 5.0, hunted_level.price + 2.0)
            stop = candle.high + 5.0
            target_level = self._pool.get_nearest_ssl(symbol, current_price)
            target = (
                target_level.price
                if target_level is not None
                else current_price - abs(hunted_level.price - current_price)
            )
        else:
            hunt_type = "SSL_HUNT"
            reversal_direction = "BULLISH"
            suggested_signal = "BUY_CALL"
            hunt_price = candle.low
            entry_zone = (hunted_level.price - 2.0, hunted_level.price + 5.0)
            stop = candle.low - 5.0
            target_level = self._pool.get_nearest_bsl(symbol, current_price)
            target = (
                target_level.price
                if target_level is not None
                else current_price + abs(current_price - hunted_level.price)
            )

        result: dict = {
            "type": hunt_type,
            "hunted_level": hunted_level,
            "hunt_price": hunt_price,
            "reversal_direction": reversal_direction,
            "confidence": confidence,
            "confidence_tier": tier,
            "confirmation_score": confirmation,
            "suggested_signal": suggested_signal,
            "entry_zone": entry_zone,
            "target": target,
            "stop": stop,
            "symbol": symbol,
            "timestamp": candle.timestamp,
            "oi_confirms": oi_confirms,
            "futures_confirms": futures_confirms,
        }

        self._store_confirmed_hunt(symbol, result)
        self._persist_hunt(result)
        logger.info(
            "%s: %s detected at level %.2f — confidence=%.0f tier=%s oi=%s futures=%s",
            symbol, hunt_type, hunted_level.price,
            confidence, tier, oi_confirms, futures_confirms,
        )
        return result

    def get_stop_hunt_signal_for_engine(
        self,
        symbol: str,
        proposed_signal: str,
        recent_hunts: list[dict],
    ) -> dict:
        """
        Integration with DecisionEngine.

        Evaluates recent_hunts against proposed_signal and returns an
        action dict: { action, confidence_adjustment, reason }.
        """
        if not recent_hunts:
            return {
                "action": "ALLOW",
                "confidence_adjustment": 0.0,
                "reason": "No recent stop hunts",
            }

        now = datetime.now(_IST)

        high_bsl_15m = [
            h for h in recent_hunts
            if h["type"] == "BSL_HUNT"
            and h["confidence_tier"] == "HIGH"
            and (now - h["timestamp"]).total_seconds() <= 900
        ]
        high_ssl_15m = [
            h for h in recent_hunts
            if h["type"] == "SSL_HUNT"
            and h["confidence_tier"] == "HIGH"
            and (now - h["timestamp"]).total_seconds() <= 900
        ]
        medium_hunts = [h for h in recent_hunts if h["confidence_tier"] == "MEDIUM"]

        if proposed_signal == "BUY_CALL":
            if high_bsl_15m:
                return {
                    "action": "BLOCK",
                    "confidence_adjustment": 0.0,
                    "reason": (
                        f"HIGH confidence BSL hunt at {high_bsl_15m[0]['hunted_level'].price:.2f} "
                        "within last 15 min — expect bearish reversal"
                    ),
                }
            if high_ssl_15m:
                return {
                    "action": "BOOST",
                    "confidence_adjustment": +20.0,
                    "reason": (
                        f"HIGH confidence SSL hunt at {high_ssl_15m[0]['hunted_level'].price:.2f} "
                        "confirmed — bullish reversal aligned with BUY_CALL"
                    ),
                }
            aligned = [h for h in medium_hunts if h["reversal_direction"] == "BULLISH"]
            opposing = [h for h in medium_hunts if h["reversal_direction"] == "BEARISH"]
            if aligned:
                return {
                    "action": "ALLOW",
                    "confidence_adjustment": +10.0,
                    "reason": "MEDIUM confidence SSL hunt — aligned bullish reversal",
                }
            if opposing:
                return {
                    "action": "ALLOW",
                    "confidence_adjustment": -15.0,
                    "reason": "MEDIUM confidence BSL hunt — bearish reversal opposes signal",
                }

        elif proposed_signal == "BUY_PUT":
            if high_ssl_15m:
                return {
                    "action": "BLOCK",
                    "confidence_adjustment": 0.0,
                    "reason": (
                        f"HIGH confidence SSL hunt at {high_ssl_15m[0]['hunted_level'].price:.2f} "
                        "within last 15 min — expect bullish reversal"
                    ),
                }
            if high_bsl_15m:
                return {
                    "action": "BOOST",
                    "confidence_adjustment": +20.0,
                    "reason": (
                        f"HIGH confidence BSL hunt at {high_bsl_15m[0]['hunted_level'].price:.2f} "
                        "confirmed — bearish reversal aligned with BUY_PUT"
                    ),
                }
            aligned = [h for h in medium_hunts if h["reversal_direction"] == "BEARISH"]
            opposing = [h for h in medium_hunts if h["reversal_direction"] == "BULLISH"]
            if aligned:
                return {
                    "action": "ALLOW",
                    "confidence_adjustment": +10.0,
                    "reason": "MEDIUM confidence BSL hunt — aligned bearish reversal",
                }
            if opposing:
                return {
                    "action": "ALLOW",
                    "confidence_adjustment": -15.0,
                    "reason": "MEDIUM confidence SSL hunt — bullish reversal opposes signal",
                }

        return {
            "action": "ALLOW",
            "confidence_adjustment": 0.0,
            "reason": "No stop hunt conditions triggered",
        }

    def get_recent_hunts(self, symbol: str, minutes: int = 20) -> list[dict]:
        """Return confirmed hunts for *symbol* within the last *minutes* minutes."""
        cutoff = datetime.now(_IST) - timedelta(minutes=minutes)
        return [
            h for h in self._confirmed_hunts.get(symbol, [])
            if h["timestamp"] >= cutoff
        ]

    # ── In-memory cache ────────────────────────────────────────────────────────

    def _store_confirmed_hunt(self, symbol: str, hunt: dict) -> None:
        if symbol not in self._confirmed_hunts:
            self._confirmed_hunts[symbol] = []
        self._confirmed_hunts[symbol].append(hunt)
        cutoff = datetime.now(_IST) - timedelta(hours=2)
        self._confirmed_hunts[symbol] = [
            h for h in self._confirmed_hunts[symbol] if h["timestamp"] >= cutoff
        ]

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _ensure_db_tables(self) -> None:
        try:
            from src.database import queries as Q
            self._db.execute(Q.CREATE_STOP_HUNTS)
            self._db.execute(Q.CREATE_STOP_HUNTS_IDX)
        except Exception as exc:
            logger.error("StopHuntDetector: failed to ensure DB tables: %s", exc)

    def _persist_hunt(self, hunt: dict) -> None:
        try:
            from src.database import queries as Q
            level: LiquidityLevel = hunt["hunted_level"]
            today = datetime.now(_IST).date().isoformat()
            self._db.execute(
                Q.INSERT_STOP_HUNT,
                (
                    hunt["symbol"],
                    hunt["type"],
                    level.level_type,
                    level.price,
                    hunt["hunt_price"],
                    hunt["timestamp"].isoformat(),
                    hunt["reversal_direction"],
                    hunt["confidence"],
                    hunt["confidence_tier"],
                    int(hunt["oi_confirms"]),
                    int(hunt["futures_confirms"]),
                    hunt["suggested_signal"],
                    hunt["entry_zone"][0],
                    hunt["entry_zone"][1],
                    hunt["target"],
                    hunt["stop"],
                    today,
                ),
            )
        except Exception as exc:
            logger.error("StopHuntDetector: DB persist failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _avg_volume(candles: list[Candle]) -> float:
    if not candles:
        return 0.0
    return sum(c.volume for c in candles) / len(candles)
