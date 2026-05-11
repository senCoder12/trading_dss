"""
GEX Regime Detector — translates raw GEX data into actionable market regime
classification with VIX context, time-of-day, expiry, and price adjustments.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.analysis.gex.models import GEXProfile

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")


class GEXRegimeDetector:
    """
    Translates raw GEX data into actionable market regime classification.

    Adds intelligence on top of raw GEX values by incorporating:
      - VIX level (high VIX changes the meaning of GEX)
      - Time of day (GEX regime matters more in afternoon than morning)
      - Recent price action (GEX + strong trend != GEX + chop)
      - Expiry proximity (GEX spikes near expiry — adjust interpretation)
    """

    def __init__(self, gex_calculator=None, db=None) -> None:
        self.gex_calculator = gex_calculator
        self.db = db

        try:
            from config.settings import settings  # type: ignore[import]
            self._negative_gex_threshold: float = -500_000_000
            self._positive_gex_threshold: float = 500_000_000
            self._high_vix_threshold: float = getattr(
                settings.thresholds, "vix_elevated_threshold", 18.0
            )
            self._regime_change_confirmation_minutes: int = 15
        except Exception:
            self._negative_gex_threshold = -500_000_000
            self._positive_gex_threshold = 500_000_000
            self._high_vix_threshold = 18.0
            self._regime_change_confirmation_minutes = 15

        # Per-symbol state
        self._current_regimes: dict[str, str] = {}
        self._regime_history: dict[str, deque] = {}
        self._pending_regime_change: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    # Primary classification
    # ─────────────────────────────────────────────────────────────────────────

    def classify_regime(
        self,
        symbol: str,
        gex_profile: GEXProfile,
        vix_level: float,
        vix_trend: str,           # 'RISING' | 'FALLING' | 'FLAT'
        current_price: float,
        price_trend_15min: str,   # 'UP' | 'DOWN' | 'SIDEWAYS'
        is_expiry_day: bool,
        minutes_to_close: int,
    ) -> dict:
        """
        Classify GEX regime with full context modifiers.

        Returns a dict with keys: regime, sub_regime, strategy_type,
        confidence, option_buying_quality, expected_behavior,
        key_levels, adjustments, reason.
        """
        now = datetime.now(tz=_IST)
        adjustments: list[str] = []

        base_regime = gex_profile.regime   # POSITIVE_GEX / NEGATIVE_GEX / NEUTRAL_GEX
        base_confidence: float = float(gex_profile.regime_strength)

        flip_point = gex_profile.gex_flip_point
        call_wall = gex_profile.call_wall
        put_wall = gex_profile.put_wall

        # ── Modifier 1: VIX context ───────────────────────────────────────────
        high_vix = vix_level >= self._high_vix_threshold

        if base_regime == "NEGATIVE_GEX":
            if high_vix:
                sub_regime = "HIGH_VOLATILITY_TRENDING"
                option_buying_quality = "GOOD"
                expected_behavior = "Large, fast directional moves — trade with trend"
            else:
                sub_regime = "COMPLACENT_TRENDING"
                option_buying_quality = "EXCELLENT"
                expected_behavior = "Trend days with cheap options — best day to trade"
        elif base_regime == "POSITIVE_GEX":
            if high_vix:
                sub_regime = "FEAR_WITH_STABILIZATION"
                option_buying_quality = "POOR"
                expected_behavior = "Volatile but mean-reverting, options overpriced"
            else:
                sub_regime = "CALM_RANGEBOUND"
                option_buying_quality = "AVOID"
                expected_behavior = "Slow, grinding, options expire worthless"
        else:  # NEUTRAL_GEX
            sub_regime = "NEUTRAL"
            option_buying_quality = "POOR"
            expected_behavior = "Mixed signals — wait for clearer GEX structure"

        # ── Modifier 2: VIX trend ─────────────────────────────────────────────
        if vix_trend == "RISING":
            if base_regime == "POSITIVE_GEX":
                base_confidence -= 10
            adjustments.append("VIX rising sharply — volatility expanding")
        elif vix_trend == "FALLING":
            if base_regime == "NEGATIVE_GEX":
                adjustments.append("VIX falling — fear receding, watch for regime shift")

        # ── Modifier 3: Expiry day ────────────────────────────────────────────
        after_2pm = now.hour >= 14

        if is_expiry_day:
            if after_2pm:
                adjustments.append("Post 2 PM expiry day — GEX unreliable, pinning dominates")
                confidence = max(20, min(95, base_confidence))
                return {
                    "regime": base_regime,
                    "sub_regime": "EXPIRY_PINNING",
                    "strategy_type": "EXPIRY_SPECIFIC",
                    "confidence": confidence,
                    "option_buying_quality": "AVOID",
                    "expected_behavior": "Max pain gravitational pull dominates",
                    "key_levels": self._build_key_levels(call_wall, put_wall, flip_point),
                    "adjustments": adjustments,
                    "reason": "Expiry day after 2PM — max pain dominates",
                }
            else:
                base_confidence *= 0.5
                adjustments.append("Expiry day — GEX less reliable post 2 PM")

        # ── Modifier 4: Time of day ───────────────────────────────────────────
        minutes_since_midnight = now.hour * 60 + now.minute
        market_open_minutes = 9 * 60 + 15   # 9:15 AM
        current_market_minutes = minutes_since_midnight - market_open_minutes

        if current_market_minutes < 45:       # 9:15–10:00 AM
            base_confidence -= 30
            adjustments.append("Early session — GEX regime forming")
        elif current_market_minutes < 165:    # 10:00–12:00 PM — full window
            pass
        elif current_market_minutes < 255:    # 12:00–1:30 PM
            base_confidence -= 20
        elif current_market_minutes < 345:    # 1:30–3:00 PM — most reliable
            pass
        else:                                 # 3:00–3:30 PM
            base_confidence -= 40
            adjustments.append("Near close — closing flows override GEX")

        # ── Modifier 5: Price vs Flip Point ──────────────────────────────────
        strategy_type_override: Optional[str] = None

        if flip_point and flip_point > 0 and current_price > 0:
            dist_pct = abs(current_price - flip_point) / current_price * 100

            if dist_pct <= 0.3:
                base_confidence -= 20
                sub_regime = "FLIP_ZONE"
                adjustments.append("Price near GEX flip point — unstable regime")
            elif current_price > flip_point and base_regime == "POSITIVE_GEX":
                base_confidence += 15
            elif current_price < flip_point and base_regime == "NEGATIVE_GEX":
                base_confidence += 15

        # ── Modifier 6: Wall proximity ────────────────────────────────────────
        key_levels: dict = self._build_key_levels(call_wall, put_wall, flip_point)
        expected_behavior_override: Optional[str] = None

        if call_wall and current_price > 0:
            call_wall_dist_pct = abs(call_wall - current_price) / current_price * 100
            if call_wall_dist_pct <= 0.5:
                strategy_type_override = "WALL_TRADE"
                expected_behavior_override = "Approaching call wall — strong resistance zone"
                key_levels["call_wall"] = call_wall

        if put_wall and current_price > 0:
            put_wall_dist_pct = abs(put_wall - current_price) / current_price * 100
            if put_wall_dist_pct <= 0.5:
                strategy_type_override = "WALL_TRADE"
                expected_behavior_override = "Approaching put wall — strong support zone"
                key_levels["put_wall"] = put_wall

        if expected_behavior_override:
            expected_behavior = expected_behavior_override

        # ── Final assembly ────────────────────────────────────────────────────
        confidence = max(20.0, min(95.0, base_confidence))

        if strategy_type_override:
            strategy_type = strategy_type_override
        else:
            strategy_type = self._determine_strategy_type(
                base_regime, sub_regime, option_buying_quality
            )

        reason = self._build_reason(base_regime, sub_regime, vix_level, adjustments)

        self._handle_regime_change(symbol, base_regime, confidence)

        result = {
            "regime": base_regime,
            "sub_regime": sub_regime,
            "strategy_type": strategy_type,
            "confidence": confidence,
            "option_buying_quality": option_buying_quality,
            "expected_behavior": expected_behavior,
            "key_levels": key_levels,
            "adjustments": adjustments,
            "reason": reason,
        }

        with self._lock:
            if symbol not in self._regime_history:
                self._regime_history[symbol] = deque(maxlen=200)
            self._regime_history[symbol].append({
                "timestamp": now,
                "regime": base_regime,
                "sub_regime": sub_regime,
                "confidence": confidence,
            })

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Regime change confirmation
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_regime_change(
        self, symbol: str, new_regime: str, confidence: float
    ) -> bool:
        """
        Confirm regime change only after regime_change_confirmation_minutes.

        Returns True if a confirmed change occurred.
        """
        with self._lock:
            current = self._current_regimes.get(symbol)
            if current == new_regime:
                self._pending_regime_change.pop(symbol, None)
                return False

            pending = self._pending_regime_change.get(symbol)

            if pending is None or pending.get("regime") != new_regime:
                self._pending_regime_change[symbol] = {
                    "regime": new_regime,
                    "first_detected": datetime.now(tz=_IST),
                    "confidence": confidence,
                }
                return False

            elapsed = datetime.now(tz=_IST) - pending["first_detected"]
            if elapsed >= timedelta(minutes=self._regime_change_confirmation_minutes):
                old_regime = current
                self._current_regimes[symbol] = new_regime
                self._pending_regime_change.pop(symbol, None)
                logger.info(
                    "REGIME CHANGE CONFIRMED: %s %s → %s (conf=%.0f%%)",
                    symbol, old_regime, new_regime, confidence,
                )
                return True

            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Day trading plan (called at 9:15 AM)
    # ─────────────────────────────────────────────────────────────────────────

    def get_day_trading_plan(
        self,
        symbol: str,
        gex_profile: GEXProfile,
        vix_level: float,
        vix_trend: str,
        is_expiry_day: bool,
    ) -> dict:
        """
        Generate the daily trading approach for *symbol*.

        Called at 9:15 AM to set the day's strategy framework.
        """
        regime = gex_profile.regime
        net_gex_cr = gex_profile.net_gex / 1e7  # convert to crore
        high_vix = vix_level >= self._high_vix_threshold

        if is_expiry_day:
            day_type = "EXPIRY"
        elif regime == "NEGATIVE_GEX" and not high_vix:
            day_type = "TRENDING"
        elif regime == "NEGATIVE_GEX" and high_vix:
            day_type = "VOLATILE"
        elif regime == "POSITIVE_GEX" and not high_vix:
            day_type = "RANGEBOUND"
        elif regime == "POSITIVE_GEX" and high_vix:
            day_type = "VOLATILE"
        else:
            day_type = "TRANSITIONAL"

        strategy_map = {
            "TRENDING":     "Directional option buying — buy ATM on pullbacks to VWAP",
            "RANGEBOUND":   "Range scalping — sell edges, avoid naked directional",
            "VOLATILE":     "Reduce size, trade with trend, avoid OTM options",
            "EXPIRY":       "Max pain play — avoid new positions after 2 PM",
            "TRANSITIONAL": "Wait for regime clarity after 10 AM",
        }
        recommended_strategy = strategy_map.get(day_type, "Standard approach")

        size_map = {
            "TRENDING": 1.2, "RANGEBOUND": 0.7, "VOLATILE": 0.5,
            "EXPIRY": 0.6, "TRANSITIONAL": 0.8,
        }
        position_size_multiplier = size_map.get(day_type, 0.8)

        if day_type == "TRENDING" and not high_vix:
            preferred_strikes = "ATM"
        elif high_vix:
            preferred_strikes = "ATM"
        elif day_type == "RANGEBOUND":
            preferred_strikes = "OTM_ONLY"
        else:
            preferred_strikes = "SLIGHT_OTM"

        max_positions_map = {
            "TRENDING": 5, "RANGEBOUND": 3, "VOLATILE": 2,
            "EXPIRY": 2, "TRANSITIONAL": 3,
        }
        max_positions = max_positions_map.get(day_type, 3)

        hold_map = {
            "TRENDING": 60, "RANGEBOUND": 30, "VOLATILE": 20,
            "EXPIRY": 30, "TRANSITIONAL": 45,
        }
        max_holding_minutes = hold_map.get(day_type, 45)

        key_levels_for_day = self._build_key_levels(
            gex_profile.call_wall, gex_profile.put_wall, gex_profile.gex_flip_point
        )

        avoid_conditions: list[str] = []
        if high_vix:
            avoid_conditions.append("OTM options (IV too high)")
        if is_expiry_day:
            avoid_conditions.append("New positions after 2 PM")
        if vix_trend == "RISING":
            avoid_conditions.append("Opening range trades (VIX expanding)")
        if day_type == "RANGEBOUND":
            avoid_conditions.append("Chasing breakouts")

        call_wall_str = (
            f"Call wall at {gex_profile.call_wall:,.0f}, "
            if gex_profile.call_wall else ""
        )
        put_wall_str = (
            f"put wall at {gex_profile.put_wall:,.0f}. "
            if gex_profile.put_wall else ""
        )
        flip_str = (
            f"Flip point at {gex_profile.gex_flip_point:,.0f}. "
            if gex_profile.gex_flip_point else ""
        )
        vix_trend_str = vix_trend.lower()
        morning_brief = (
            f"{symbol}: {regime.replace('_', ' ')} ({net_gex_cr:+.0f} Cr), "
            f"VIX {vix_level:.1f} {vix_trend_str}. "
            f"{call_wall_str}{put_wall_str}{flip_str}"
            f"Day type: {day_type}. "
            f"Prefer {preferred_strikes} options, hold up to {max_holding_minutes} min. "
            f"{'Expiry day caution.' if is_expiry_day else ''}"
        ).strip().rstrip(".")

        return {
            "day_type": day_type,
            "recommended_strategy": recommended_strategy,
            "max_positions": max_positions,
            "position_size_multiplier": position_size_multiplier,
            "preferred_strikes": preferred_strikes,
            "max_holding_minutes": max_holding_minutes,
            "key_levels_for_day": key_levels_for_day,
            "avoid_conditions": avoid_conditions,
            "morning_brief": morning_brief,
            "net_gex_cr": net_gex_cr,
            "regime": regime,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _determine_strategy_type(
        regime: str,
        sub_regime: str,
        option_buying_quality: str,
    ) -> str:
        if sub_regime == "EXPIRY_PINNING":
            return "EXPIRY_SPECIFIC"
        if sub_regime == "FLIP_ZONE":
            return "RANGE_TRADE"
        if regime == "NEGATIVE_GEX":
            if option_buying_quality == "EXCELLENT":
                return "DIRECTIONAL_AGGRESSIVE"
            return "DIRECTIONAL_NORMAL"
        if regime == "POSITIVE_GEX":
            if option_buying_quality == "AVOID":
                return "AVOID_DIRECTIONAL"
            return "RANGE_TRADE"
        return "RANGE_TRADE"

    @staticmethod
    def _build_key_levels(
        call_wall: Optional[float],
        put_wall: Optional[float],
        flip_point: Optional[float],
    ) -> dict:
        levels: dict = {}
        if call_wall:
            levels["call_wall"] = call_wall
        if put_wall:
            levels["put_wall"] = put_wall
        if flip_point:
            levels["flip_point"] = flip_point
        return levels

    @staticmethod
    def _build_reason(
        regime: str, sub_regime: str, vix_level: float, adjustments: list[str]
    ) -> str:
        parts = [f"{regime}/{sub_regime}", f"VIX={vix_level:.1f}"]
        if adjustments:
            parts.append("; ".join(adjustments[:2]))
        return " | ".join(parts)

    def get_current_regime(self, symbol: str) -> Optional[str]:
        with self._lock:
            return self._current_regimes.get(symbol)

    def get_pending_change(self, symbol: str) -> Optional[dict]:
        with self._lock:
            return self._pending_regime_change.get(symbol)
