"""
SweepDetector — distinguishes liquidity sweeps from genuine breakouts.

A liquidity sweep: price wicks through a level (taking stops), then
reverses.  A genuine breakout: price closes beyond the level and
continues.  Conflating the two causes the decision engine to signal
in the wrong direction on sweep setups.

Integration point: call get_signal_override() before the DecisionEngine
emits any signal.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.analysis.liquidity.candle_store import CandleStore
from src.analysis.liquidity.liquidity_pool_detector import LiquidityPoolDetector
from src.analysis.liquidity.models import Candle, LiquidityLevel, LiquiditySweeep
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# Level types that sit above price (buy-side liquidity)
_BSL_TYPES = {"BSL", "EQUAL_HIGH", "PDH", "PWH", "ROUND_NUMBER"}
# Level types that sit below price (sell-side liquidity)
_SSL_TYPES = {"SSL", "EQUAL_LOW", "PDL", "PWL"}

# Tolerance for "close near level" comparisons (% of price)
_LEVEL_TOLERANCE_PCT = 0.05


class SweepDetector:
    """
    Detects and classifies liquidity sweep events.

    Registers callbacks on CandleStore for 1-min and 5-min closes.
    On every close it tests each active LiquidityLevel for interaction,
    classifies the interaction, and maintains a pending-sweep buffer
    while waiting for reversal confirmation.
    """

    def __init__(
        self,
        candle_store: CandleStore,
        pool_detector: LiquidityPoolDetector,
        db: DatabaseManager,
    ) -> None:
        self._cs = candle_store
        self._pool = pool_detector
        self._db = db

        cfg = settings.sweep
        self._confirmation_candles: int = cfg.sweep_confirmation_candles
        self._min_wick_pct: float = cfg.min_wick_beyond_pct
        self._max_wick_pct: float = cfg.max_wick_beyond_pct
        self._reversion_threshold_pct: float = cfg.reversion_close_threshold_pct
        self._volume_spike_ratio: float = cfg.volume_spike_ratio
        self._min_reversal_body_pct: float = cfg.min_reversal_body_pct

        # symbol → list of pending-sweep dicts waiting for confirmation
        self._pending_sweeps: dict[str, list[dict]] = {}

        # Confirmed sweeps kept in memory for signal-override queries
        # symbol → list[LiquiditySweeep]
        self._confirmed_sweeps: dict[str, list[LiquiditySweeep]] = {}

        # Listeners for the sweep_detected event (5-min quality)
        self._sweep_listeners: list[Callable[[LiquiditySweeep], None]] = []

        self._ensure_db_tables()
        candle_store.register_candle_callback("1min", self._on_1min_candle)
        candle_store.register_candle_callback("5min", self._on_5min_candle)

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_sweep_listener(
        self, callback: Callable[[LiquiditySweeep], None]
    ) -> None:
        """Register a listener fired when a 5-min sweep is confirmed."""
        self._sweep_listeners.append(callback)

    def get_recent_sweeps(self, symbol: str, minutes: int = 30) -> list[LiquiditySweeep]:
        """Return confirmed sweeps for *symbol* within the last *minutes* minutes."""
        cutoff = datetime.now(_IST) - timedelta(minutes=minutes)
        return [
            s for s in self._confirmed_sweeps.get(symbol, [])
            if s.timestamp >= cutoff
        ]

    # ── Classification ─────────────────────────────────────────────────────────

    def classify_level_interaction(
        self,
        candle: Candle,
        level: LiquidityLevel,
        recent_candles: list[Candle],
        current_price: float,
    ) -> str:
        """
        Classify how *candle* interacted with *level*.

        Returns one of: 'SWEEP' | 'BREAKOUT' | 'TEST' | 'BOUNCE' | 'NONE'
        """
        tolerance = level.price * _LEVEL_TOLERANCE_PCT / 100.0
        is_bsl = level.level_type in _BSL_TYPES

        # Step 1 — did this candle reach the level?
        if is_bsl:
            if candle.high < level.price - tolerance:
                return "NONE"
        else:
            if candle.low > level.price + tolerance:
                return "NONE"

        # Step 2 — how far did price penetrate?
        if is_bsl:
            wick_beyond = candle.high - level.price
        else:
            wick_beyond = level.price - candle.low

        wick_beyond_pct = (wick_beyond / level.price) * 100.0 if level.price else 0.0

        # Step 3 — did price CLOSE beyond the level?
        if is_bsl:
            close_beyond = candle.close > level.price
        else:
            close_beyond = candle.close < level.price

        avg_vol = _avg_volume(recent_candles)
        vol_ok = avg_vol > 0 and candle.volume >= self._volume_spike_ratio * avg_vol

        # Step 4 — classify

        if wick_beyond_pct < self._min_wick_pct and not close_beyond:
            return "BOUNCE"

        if (
            wick_beyond_pct >= self._min_wick_pct
            and wick_beyond_pct <= self._max_wick_pct
            and not close_beyond
            and candle.total_range > 0
            and _wick_fraction(candle, is_bsl) >= 0.60
            and vol_ok
        ):
            self._add_pending_sweep(candle, level, wick_beyond_pct, is_bsl)
            return "SWEEP"

        if (
            close_beyond
            and wick_beyond_pct > 0
            and candle.total_range > 0
            and candle.body_size / candle.total_range >= 0.40
            and vol_ok
        ):
            return "BREAKOUT"

        return "TEST"

    def validate_breakout(
        self,
        symbol: str,
        level: LiquidityLevel,
        breakout_candle: Candle,
        candles_after: list[Candle],
        options_oi_confirms: bool = False,
    ) -> dict:
        """
        Validate whether a breakout is genuine, failed, or pending.

        Returns dict with keys: is_valid, breakout_type, confidence, reason.
        """
        is_bsl = level.level_type in _BSL_TYPES

        # Breakout candle must close beyond level
        if is_bsl:
            body_closes_beyond = breakout_candle.close > level.price
        else:
            body_closes_beyond = breakout_candle.close < level.price

        if not body_closes_beyond:
            return {
                "is_valid": False,
                "breakout_type": "FAILED",
                "confidence": 0.0,
                "reason": "Breakout candle body did not close beyond level",
            }

        # Check volume
        recent = candles_after[:20] if len(candles_after) >= 20 else candles_after
        avg_vol = _avg_volume(recent) if recent else 0.0
        vol_multiplier = (
            breakout_candle.volume / avg_vol if avg_vol > 0 else 0.0
        )
        strong_volume = vol_multiplier >= 2.0

        # Check if price returned back within 5 candles — failed breakout wins
        # over pending; check this before the candle-count gate.
        returned = False
        for c in candles_after[:5]:
            if is_bsl:
                if c.close < level.price:
                    returned = True
                    break
            else:
                if c.close > level.price:
                    returned = True
                    break

        if returned:
            return {
                "is_valid": False,
                "breakout_type": "FAILED",
                "confidence": 20.0,
                "reason": "Price returned to previous side within 5 candles — failed breakout",
            }

        # Not enough follow-through candles yet
        if len(candles_after) < self._confirmation_candles:
            confidence = 40.0 + (20.0 if strong_volume else 0.0)
            if options_oi_confirms:
                confidence += 15.0
            return {
                "is_valid": False,
                "breakout_type": "PENDING",
                "confidence": min(confidence, 100.0),
                "reason": f"Waiting for {self._confirmation_candles} confirmation candles",
            }

        # Count how many of the next candles also close beyond level
        continuations = 0
        for c in candles_after[: self._confirmation_candles]:
            if is_bsl:
                if c.close > level.price:
                    continuations += 1
            else:
                if c.close < level.price:
                    continuations += 1

        if continuations >= 2:
            confidence = 55.0
            confidence += 20.0 if strong_volume else 0.0
            confidence += 10.0 if continuations == 3 else 0.0
            if options_oi_confirms:
                confidence += 15.0
            return {
                "is_valid": True,
                "breakout_type": "GENUINE",
                "confidence": min(confidence, 100.0),
                "reason": (
                    f"{continuations}/{self._confirmation_candles} candles confirm beyond level"
                    + ("; OI confirms" if options_oi_confirms else "")
                ),
            }

        return {
            "is_valid": False,
            "breakout_type": "PENDING",
            "confidence": 35.0,
            "reason": f"Only {continuations}/{self._confirmation_candles} continuation candles",
        }

    def get_signal_override(
        self,
        symbol: str,
        proposed_signal: str,
        current_price: float,
        recent_sweeps: list[LiquiditySweeep],
    ) -> dict:
        """
        Integration point with DecisionEngine.

        Returns dict with keys: action ('ALLOW'|'BLOCK'|'REVERSE'),
        reason, confidence_adjustment.
        """
        now = datetime.now(_IST)

        # Bucket sweeps by age and quality
        bsl_sweeps_10m = [
            s for s in recent_sweeps
            if s.sweep_type == "BSL_SWEEP"
            and (now - s.timestamp).total_seconds() <= 600
        ]
        bsl_sweeps_15m = [
            s for s in recent_sweeps
            if s.sweep_type == "BSL_SWEEP"
            and (now - s.timestamp).total_seconds() <= 900
        ]
        ssl_sweeps_20m = [
            s for s in recent_sweeps
            if s.sweep_type == "SSL_SWEEP"
            and (now - s.timestamp).total_seconds() <= 1200
        ]
        any_sweeps_30m = recent_sweeps  # caller already filtered to 30 min

        high_bsl_15m = [s for s in bsl_sweeps_15m if s.signal_quality == "HIGH"]
        high_bsl_10m = [s for s in bsl_sweeps_10m if s.signal_quality == "HIGH"]
        high_ssl_20m = [s for s in ssl_sweeps_20m if s.signal_quality == "HIGH"]

        if proposed_signal == "BUY_CALL":
            # Rule 2 — reverse after confirmed HIGH-quality BSL sweep within 10 min
            if high_bsl_10m and any(s.reversion_confirmed for s in high_bsl_10m):
                return {
                    "action": "REVERSE",
                    "reason": (
                        f"HIGH quality BSL sweep at {high_bsl_10m[0].level.price:.2f} "
                        "confirmed within last 10 min — expect bearish reversal"
                    ),
                    "confidence_adjustment": +10.0,
                }

            # Rule 1 — block continuation after HIGH BSL sweep within 15 min
            if high_bsl_15m:
                return {
                    "action": "BLOCK",
                    "reason": (
                        f"HIGH quality BSL sweep at {high_bsl_15m[0].level.price:.2f} "
                        "occurred within last 15 min"
                    ),
                    "confidence_adjustment": 0.0,
                }

            # Rule 3 — boost after confirmed SSL sweep (stop-hunt below → bullish)
            if high_ssl_20m and any(s.reversion_confirmed for s in high_ssl_20m):
                return {
                    "action": "ALLOW",
                    "reason": (
                        f"HIGH quality SSL sweep at {high_ssl_20m[0].level.price:.2f} "
                        "confirmed — aligned bullish reversal"
                    ),
                    "confidence_adjustment": +15.0,
                }

        elif proposed_signal == "BUY_PUT":
            # Mirror rules for put signals
            high_ssl_10m = [
                s for s in ssl_sweeps_20m
                if s.signal_quality == "HIGH"
                and (now - s.timestamp).total_seconds() <= 600
            ]
            high_bsl_20m = [
                s for s in recent_sweeps
                if s.sweep_type == "BSL_SWEEP"
                and s.signal_quality == "HIGH"
                and (now - s.timestamp).total_seconds() <= 1200
            ]

            if high_ssl_10m and any(s.reversion_confirmed for s in high_ssl_10m):
                return {
                    "action": "REVERSE",
                    "reason": (
                        f"HIGH quality SSL sweep at {high_ssl_10m[0].level.price:.2f} "
                        "confirmed within last 10 min — expect bullish reversal"
                    ),
                    "confidence_adjustment": +10.0,
                }

            if [s for s in ssl_sweeps_20m if s.signal_quality == "HIGH"
                    and (now - s.timestamp).total_seconds() <= 900]:
                return {
                    "action": "BLOCK",
                    "reason": "HIGH quality SSL sweep occurred within last 15 min",
                    "confidence_adjustment": 0.0,
                }

            if high_bsl_20m and any(s.reversion_confirmed for s in high_bsl_20m):
                return {
                    "action": "ALLOW",
                    "reason": "HIGH quality BSL sweep confirmed — aligned bearish reversal",
                    "confidence_adjustment": +15.0,
                }

        # Rule 4 — no relevant sweeps
        if not any_sweeps_30m:
            return {
                "action": "ALLOW",
                "reason": "No sweeps in last 30 min",
                "confidence_adjustment": 0.0,
            }

        return {
            "action": "ALLOW",
            "reason": "No overriding sweep condition",
            "confidence_adjustment": 0.0,
        }

    # ── Pending sweep management ───────────────────────────────────────────────

    def _add_pending_sweep(
        self,
        candle: Candle,
        level: LiquidityLevel,
        wick_beyond_pct: float,
        is_bsl: bool,
    ) -> None:
        symbol = candle.symbol
        if symbol not in self._pending_sweeps:
            self._pending_sweeps[symbol] = []

        entry = {
            "sweep_candle": candle,
            "level": level,
            "wick_beyond_pct": wick_beyond_pct,
            "is_bsl": is_bsl,
            "candles_seen": 0,
            "sweep_type": "BSL_SWEEP" if is_bsl else "SSL_SWEEP",
        }
        self._pending_sweeps[symbol].append(entry)
        logger.debug(
            "%s: pending %s sweep at %.2f (wick=%.3f%%)",
            symbol, entry["sweep_type"], level.price, wick_beyond_pct,
        )

    def _try_confirm_pending_sweeps(
        self, symbol: str, candle: Candle, is_5min: bool
    ) -> list[LiquiditySweeep]:
        pending = self._pending_sweeps.get(symbol, [])
        if not pending:
            return []

        confirmed: list[LiquiditySweeep] = []
        still_pending: list[dict] = []

        for entry in pending:
            entry["candles_seen"] += 1
            sweep = self._confirm_sweep(symbol, entry, candle)
            if sweep is not None:
                confirmed.append(sweep)
                self._pool.mark_level_swept(
                    symbol,
                    entry["level"],
                    entry["sweep_candle"],
                    entry["level"].price * entry["wick_beyond_pct"] / 100.0,
                )
                self._store_confirmed_sweep(symbol, sweep)
                self._persist_sweep(sweep)
                logger.info(
                    "%s: %s CONFIRMED at %.2f quality=%s",
                    symbol, sweep.sweep_type, sweep.level.price, sweep.signal_quality,
                )
            elif entry["candles_seen"] >= self._confirmation_candles:
                logger.debug(
                    "%s: pending sweep at %.2f expired without confirmation",
                    symbol, entry["level"].price,
                )
            else:
                still_pending.append(entry)

        self._pending_sweeps[symbol] = still_pending
        return confirmed

    def _confirm_sweep(
        self, symbol: str, pending: dict, confirmation_candle: Candle
    ) -> Optional[LiquiditySweeep]:
        sweep_candle: Candle = pending["sweep_candle"]
        level: LiquidityLevel = pending["level"]
        is_bsl: bool = pending["is_bsl"]
        wick_beyond_pct: float = pending["wick_beyond_pct"]

        # Condition 1 — candle closes in OPPOSITE direction of sweep
        if is_bsl:
            # BSL sweep → bullish wick; confirmation must be BEARISH close below level
            direction_ok = (
                not confirmation_candle.is_bullish
                and confirmation_candle.close < level.price
            )
        else:
            # SSL sweep → bearish wick; confirmation must be BULLISH close above level
            direction_ok = (
                confirmation_candle.is_bullish
                and confirmation_candle.close > level.price
            )

        if not direction_ok:
            return None

        # Condition 2 — closed back inside the level by >= threshold
        if is_bsl:
            reversion_pct = (level.price - confirmation_candle.close) / level.price * 100.0
        else:
            reversion_pct = (confirmation_candle.close - level.price) / level.price * 100.0

        if reversion_pct < self._reversion_threshold_pct:
            return None

        # Condition 3 — real body (not a doji)
        if confirmation_candle.total_range > 0:
            body_ratio = confirmation_candle.body_size / confirmation_candle.total_range
        else:
            body_ratio = 0.0

        if body_ratio < self._min_reversal_body_pct:
            return None

        # All conditions met — build recent candles list for quality scoring
        recent = self._cs.get_candles(symbol, sweep_candle.timeframe, 20)
        quality = self._calculate_signal_quality(
            sweep_candle, confirmation_candle, level, recent
        )

        return LiquiditySweeep(
            symbol=symbol,
            timeframe=sweep_candle.timeframe,
            level=level,
            sweep_candle=sweep_candle,
            sweep_type=pending["sweep_type"],
            sweep_high=sweep_candle.high,
            sweep_low=sweep_candle.low,
            reversion_confirmed=True,
            timestamp=sweep_candle.timestamp,
            reversion_candle=confirmation_candle,
            signal_quality=quality,
        )

    def _calculate_signal_quality(
        self,
        sweep_candle: Candle,
        reversal_candle: Candle,
        level: LiquidityLevel,
        recent_candles: list[Candle],
    ) -> str:
        avg_vol = _avg_volume(recent_candles)
        vol_ratio = sweep_candle.volume / avg_vol if avg_vol > 0 else 0.0

        # Volume score (40 pts max)
        if vol_ratio >= 3.0:
            vol_score = 40
        elif vol_ratio >= 2.0:
            vol_score = 25
        elif vol_ratio >= 1.5:
            vol_score = 10
        else:
            vol_score = 0

        # Wick precision score (30 pts max)
        wick_beyond_pct = (
            abs(sweep_candle.high - level.price)
            if level.level_type in _BSL_TYPES
            else abs(level.price - sweep_candle.low)
        ) / level.price * 100.0 if level.price else 0.0

        if wick_beyond_pct < 0.1:
            wick_score = 30
        elif wick_beyond_pct < 0.2:
            wick_score = 20
        else:
            wick_score = 5

        # Reversal candle quality (30 pts max)
        if reversal_candle.total_range > 0:
            body_ratio = reversal_candle.body_size / reversal_candle.total_range
        else:
            body_ratio = 0.0

        if body_ratio >= 0.70:
            reversal_score = 30
        elif body_ratio >= 0.50:
            reversal_score = 20
        else:
            reversal_score = 5

        total = vol_score + wick_score + reversal_score

        if total >= 75:
            return "HIGH"
        if total >= 45:
            return "MEDIUM"
        return "LOW"

    # ── CandleStore callbacks ──────────────────────────────────────────────────

    def _on_1min_candle(self, candle: Candle) -> None:
        symbol = candle.symbol
        try:
            current_price = candle.close
            bsl_levels, ssl_levels = self._pool.get_all_active_levels(symbol, current_price)
            recent = self._cs.get_candles(symbol, "1min", 20)

            for level in bsl_levels + ssl_levels:
                interaction = self.classify_level_interaction(
                    candle, level, recent, current_price
                )
                if interaction in ("SWEEP", "BREAKOUT"):
                    logger.debug(
                        "%s 1min: %s on %s at %.2f",
                        symbol, interaction, level.level_type, level.price,
                    )

            self._try_confirm_pending_sweeps(symbol, candle, is_5min=False)
        except Exception as exc:
            logger.error("SweepDetector._on_1min_candle %s: %s", symbol, exc)

    def _on_5min_candle(self, candle: Candle) -> None:
        symbol = candle.symbol
        try:
            current_price = candle.close
            bsl_levels, ssl_levels = self._pool.get_all_active_levels(symbol, current_price)
            recent = self._cs.get_candles(symbol, "5min", 20)

            for level in bsl_levels + ssl_levels:
                interaction = self.classify_level_interaction(
                    candle, level, recent, current_price
                )
                if interaction in ("SWEEP", "BREAKOUT"):
                    logger.info(
                        "%s 5min: %s on %s at %.2f",
                        symbol, interaction, level.level_type, level.price,
                    )

            confirmed = self._try_confirm_pending_sweeps(symbol, candle, is_5min=True)
            for sweep in confirmed:
                self._emit_sweep_event(sweep)
        except Exception as exc:
            logger.error("SweepDetector._on_5min_candle %s: %s", symbol, exc)

    def _emit_sweep_event(self, sweep: LiquiditySweeep) -> None:
        for listener in self._sweep_listeners:
            try:
                listener(sweep)
            except Exception as exc:
                logger.error("SweepDetector: listener error: %s", exc)

    # ── In-memory cache ────────────────────────────────────────────────────────

    def _store_confirmed_sweep(self, symbol: str, sweep: LiquiditySweeep) -> None:
        if symbol not in self._confirmed_sweeps:
            self._confirmed_sweeps[symbol] = []
        self._confirmed_sweeps[symbol].append(sweep)
        # Prune sweeps older than 2 hours to keep memory bounded
        cutoff = datetime.now(_IST) - timedelta(hours=2)
        self._confirmed_sweeps[symbol] = [
            s for s in self._confirmed_sweeps[symbol] if s.timestamp >= cutoff
        ]

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _ensure_db_tables(self) -> None:
        try:
            from src.database import queries as Q
            self._db.execute(Q.CREATE_LIQUIDITY_SWEEPS)
            self._db.execute(Q.CREATE_LIQUIDITY_SWEEPS_IDX)
        except Exception as exc:
            logger.error("SweepDetector: failed to ensure DB tables: %s", exc)

    def _persist_sweep(self, sweep: LiquiditySweeep) -> None:
        try:
            from src.database import queries as Q
            today = datetime.now(_IST).date().isoformat()
            rev_time = (
                sweep.reversion_candle.timestamp.isoformat()
                if sweep.reversion_candle else None
            )
            self._db.execute(
                Q.INSERT_LIQUIDITY_SWEEP,
                (
                    sweep.symbol,
                    sweep.timeframe,
                    sweep.sweep_type,
                    sweep.level.level_type,
                    sweep.level.price,
                    sweep.sweep_candle.timestamp.isoformat(),
                    sweep.sweep_high,
                    sweep.sweep_low,
                    sweep.level.wick_beyond / sweep.level.price * 100.0
                    if sweep.level.price else 0.0,
                    sweep.signal_quality,
                    int(sweep.reversion_confirmed),
                    rev_time,
                    int(sweep.oi_confirms),
                    int(sweep.futures_confirms),
                    today,
                ),
            )
        except Exception as exc:
            logger.error("SweepDetector: DB persist failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _avg_volume(candles: list[Candle]) -> float:
    """Return average volume of *candles*, or 0 if empty."""
    if not candles:
        return 0.0
    return sum(c.volume for c in candles) / len(candles)


def _wick_fraction(candle: Candle, is_bsl: bool) -> float:
    """Fraction of candle total range that is the relevant wick."""
    if candle.total_range == 0:
        return 0.0
    if is_bsl:
        return candle.upper_wick / candle.total_range
    return candle.lower_wick / candle.total_range
