"""Hero Zero Detector — Proper gamma squeeze detection for expiry afternoon.

Replaces the naive threshold hack with actual GEX (Gamma Exposure) measurement,
max pain proximity, volume confirmation, IV skew, and momentum validation.

Only triggers when ALL conditions are met:
  1. Short gamma (dealers hedging by accelerating price)
  2. Spot within 1.5% of max pain (gravity active)
  3. Volume spike (1.5x 20-period avg)
  4. Momentum confirmed (last 3 bars directional alignment)

Hard controls:
  - Position size: 25% cap (no override)
  - Max 1 trade per expiry day
  - Circuit breaker: skip if last 2 HZ trades lost
  - SL hard cap: 50pts NIFTY, 100pts BANKNIFTY
  - Hard exit: 14:55 IST
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from src.database.db_manager import DatabaseManager
from src.data.options_chain import OptionsChainData

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")


@dataclass
class HeroZeroResult:
    """Full result of Hero Zero detection pass."""

    is_triggered: bool
    signal_direction: str  # "BUY_CALL" / "BUY_PUT" / "NO_TRADE"

    # GEX computation
    net_gex: float  # negative = short gamma = price accelerates
    gex_regime: str  # "SHORT_GAMMA" / "LONG_GAMMA" / "NEUTRAL"

    # Max pain proximity
    max_pain: float
    spot: float
    max_pain_proximity_pct: float
    max_pain_gravity_active: bool

    # Volume confirmation
    volume_ratio: float
    volume_confirmed: bool

    # IV Skew direction
    iv_skew: float
    skew_direction: str  # "BULLISH" / "BEARISH" / "NEUTRAL"

    # Momentum (last 3 bars)
    momentum_direction: str
    momentum_confirmed: bool

    # Risk controls status
    circuit_breaker_active: bool
    hz_trades_today: int
    stop_loss: float  # hard-cap derived
    position_size_cap: float  # always 0.25

    # Conditions audit
    conditions_met: dict[str, bool] = field(default_factory=dict)
    reasoning: str = ""


class HeroZeroDetector:
    """Detects gamma squeeze conditions on F&O expiry afternoon (14:00–14:50 IST)."""

    _WINDOW_START = (14, 0)
    _WINDOW_END = (14, 50)
    _MAX_PAIN_PROXIMITY_PCT = 1.5
    _VOLUME_RATIO_MIN = 1.5
    _MAX_HZ_TRADES_PER_DAY = 1
    _CB_LOOKBACK = 2
    _NIFTY_SL_PTS = 50
    _BANKNIFTY_SL_PTS = 100
    _POSITION_SIZE_CAP = 0.25
    _TTE_FLOOR_YEARS = 15.0 / (375 * 252)  # 15 trading minutes

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def detect(
        self,
        index_id: str,
        chain: Optional[OptionsChainData],
        price_bars: list[dict],
        volume_ratio: float,
        now: datetime,
        lot_size: int,
    ) -> HeroZeroResult:
        """Main entry point. Returns HeroZeroResult with is_triggered=False if any gate fails."""
        if chain is None:
            return self._no_trigger("options chain missing")

        # Gate 1: time window
        if not self._in_window(now):
            return self._no_trigger(f"outside 14:00-14:50 window (current: {now.strftime('%H:%M')})")

        # Gate 2: expiry today
        if not self._is_expiry_today(chain, now):
            return self._no_trigger(f"not expiry day (expiry: {chain.expiry_date})")

        # Gate 3: circuit breaker + daily cap
        cb_active, hz_today = self._check_circuit_breaker(index_id, now)
        if cb_active:
            return self._gate_failure("circuit breaker: last 2 HZ trades lost", cb_active=True)
        if hz_today >= self._MAX_HZ_TRADES_PER_DAY:
            return self._gate_failure(f"daily HZ cap reached ({hz_today}/{self._MAX_HZ_TRADES_PER_DAY})", hz_today=hz_today)

        # Compute all indicators
        try:
            net_gex, gex_breakdown = self._compute_net_gex(chain, lot_size, now)
            max_pain, proximity_pct, gravity_active = self._compute_max_pain_proximity(chain)
            iv_skew, skew_dir = self._compute_iv_skew(chain, chain.spot_price)
            mom_dir, mom_confirmed = self._compute_momentum(price_bars[-3:] if len(price_bars) >= 3 else price_bars)
            vol_confirmed = volume_ratio >= self._VOLUME_RATIO_MIN
        except Exception as e:
            logger.exception("Hero Zero indicator computation failed (non-fatal)")
            return self._no_trigger(f"computation error: {str(e)[:50]}")

        # All conditions required for trigger
        gex_ok = net_gex < 0  # short gamma = acceleration
        conditions = {
            "short_gamma": gex_ok,
            "max_pain_gravity": gravity_active,
            "volume_spike": vol_confirmed,
            "momentum": mom_confirmed,
        }

        # If any condition fails, return NO_TRIGGER with details
        if not all(conditions.values()):
            return self._no_trigger_with_conditions(
                conditions,
                net_gex,
                max_pain,
                proximity_pct,
                volume_ratio,
                iv_skew,
                skew_dir,
                mom_dir,
                mom_confirmed,
                cb_active,
                hz_today,
                chain.spot_price,
            )

        # All conditions met — determine direction
        direction = self._determine_direction(mom_dir, skew_dir)
        if direction == "NO_TRADE":
            return self._no_trigger("momentum and skew not aligned on direction")

        sl = self._compute_sl(index_id, chain.spot_price, direction)
        reasoning = self._build_reasoning(
            conditions, net_gex, max_pain, proximity_pct, volume_ratio, skew_dir, mom_dir, direction
        )

        logger.info(
            "HERO_ZERO triggered: %s %s | GEX=%.0f | proximity=%.1f%% | vol_ratio=%.2f | momentum=%s",
            index_id,
            direction,
            net_gex,
            proximity_pct,
            volume_ratio,
            mom_dir,
        )

        return HeroZeroResult(
            is_triggered=True,
            signal_direction=direction,
            net_gex=net_gex,
            gex_regime="SHORT_GAMMA" if net_gex < 0 else "LONG_GAMMA",
            max_pain=max_pain,
            spot=chain.spot_price,
            max_pain_proximity_pct=proximity_pct,
            max_pain_gravity_active=gravity_active,
            volume_ratio=volume_ratio,
            volume_confirmed=vol_confirmed,
            iv_skew=iv_skew,
            skew_direction=skew_dir,
            momentum_direction=mom_dir,
            momentum_confirmed=mom_confirmed,
            circuit_breaker_active=False,
            hz_trades_today=hz_today,
            stop_loss=sl,
            position_size_cap=self._POSITION_SIZE_CAP,
            conditions_met=conditions,
            reasoning=reasoning,
        )

    def _in_window(self, now: datetime) -> bool:
        """Return True if 14:00–14:50 IST."""
        h, m = now.hour, now.minute
        return h == 14 and self._WINDOW_START[1] <= m <= self._WINDOW_END[1]

    def _is_expiry_today(self, chain: OptionsChainData, now: datetime) -> bool:
        """Return True if expiry date matches today (IST)."""
        today = now.date()
        return chain.expiry_date == today

    def _compute_net_gex(self, chain: OptionsChainData, lot_size: int, now: datetime) -> tuple[float, list[dict]]:
        """
        Compute net dealer gamma exposure (GEX) using Black-Scholes.

        Returns (net_gex, per_strike_breakdown).
        net_gex < 0 → short gamma → price accelerates
        """
        from src.analysis.indicators.quant import QuantIndicators

        quant = QuantIndicators()
        spot = chain.spot_price

        # TTE: time until 15:30 IST (market close on expiry), in years
        close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        minutes_left = max(0, (close - now).total_seconds() / 60)
        tte = max(minutes_left / (375 * 252), self._TTE_FLOOR_YEARS)

        net_gex = 0.0
        breakdown = []

        for strike in chain.strikes:
            # Skip if IV missing or unreasonable
            if strike.ce_iv < 0.5 or strike.pe_iv < 0.5:
                continue

            try:
                # Black-Scholes: IV is percentage (e.g. 15.2), divide by 100 for decimal
                ce_bs = quant.calculate_bs_price(
                    spot=spot,
                    strike=strike.strike_price,
                    time_to_expiry_years=tte,
                    risk_free_rate=0.065,
                    volatility=strike.ce_iv / 100.0,
                    option_type="CE",
                )
                pe_bs = quant.calculate_bs_price(
                    spot=spot,
                    strike=strike.strike_price,
                    time_to_expiry_years=tte,
                    risk_free_rate=0.065,
                    volatility=strike.pe_iv / 100.0,
                    option_type="PE",
                )

                # GEX = OI × gamma × lot_size per side
                ce_gex = strike.ce_oi * ce_bs.gamma * lot_size
                pe_gex = strike.pe_oi * pe_bs.gamma * lot_size
                net = ce_gex - pe_gex
                net_gex += net

                breakdown.append(
                    {
                        "strike": strike.strike_price,
                        "ce_gamma": ce_bs.gamma,
                        "pe_gamma": pe_bs.gamma,
                        "ce_gex": ce_gex,
                        "pe_gex": pe_gex,
                        "net": net,
                    }
                )
            except Exception:
                continue

        return net_gex, breakdown

    def _compute_max_pain_proximity(self, chain: OptionsChainData) -> tuple[float, float, bool]:
        """
        Compute max pain (strike minimising total writer loss) and proximity percentage.

        Returns (max_pain_strike, proximity_pct, gravity_active).
        """
        strikes_list = chain.strikes
        if not strikes_list:
            return 0.0, 999.0, False

        min_loss = float("inf")
        max_pain = 0.0

        for candidate_strike in strikes_list:
            k = candidate_strike.strike_price
            # Total loss for writers if settlement at strike k
            loss = sum(
                max(0, k - s.strike_price) * s.ce_oi + max(0, s.strike_price - k) * s.pe_oi
                for s in strikes_list
            )
            if loss < min_loss:
                min_loss = loss
                max_pain = k

        spot = chain.spot_price
        if max_pain <= 0:
            proximity_pct = 999.0
        else:
            proximity_pct = abs(spot - max_pain) / max_pain * 100

        gravity_active = proximity_pct <= self._MAX_PAIN_PROXIMITY_PCT

        return max_pain, proximity_pct, gravity_active

    def _compute_iv_skew(self, chain: OptionsChainData, spot: float) -> tuple[float, str]:
        """
        Compute IV skew between OTM puts and calls.

        Returns (skew_value, direction).
        Positive skew = bearish (put IV > call IV = hedging demand).
        """
        # OTM puts: 2-4% below spot
        otm_put_ivs = [
            s.pe_iv
            for s in chain.strikes
            if spot > 0 and 0.02 <= (spot - s.strike_price) / spot <= 0.04
        ]
        # OTM calls: 2-4% above spot
        otm_call_ivs = [
            s.ce_iv
            for s in chain.strikes
            if spot > 0 and 0.02 <= (s.strike_price - spot) / spot <= 0.04
        ]

        if not otm_put_ivs or not otm_call_ivs:
            return 0.0, "NEUTRAL"

        put_avg = sum(otm_put_ivs) / len(otm_put_ivs)
        call_avg = sum(otm_call_ivs) / len(otm_call_ivs)
        skew = put_avg - call_avg

        # Threshold: 2% difference significant
        if skew > 2.0:
            direction = "BEARISH"
        elif skew < -2.0:
            direction = "BULLISH"
        else:
            direction = "NEUTRAL"

        return skew, direction

    def _compute_momentum(self, bars: list[dict]) -> tuple[str, bool]:
        """
        Check last 3 bars for directional momentum.

        Returns (direction, confirmed).
        """
        if len(bars) < 3:
            return "NEUTRAL", False

        closes = [float(b.get("close", 0) or 0) for b in bars[-3:]]
        if all(closes[i] > closes[i - 1] for i in range(1, 3)):
            return "BULLISH", True
        if all(closes[i] < closes[i - 1] for i in range(1, 3)):
            return "BEARISH", True

        return "NEUTRAL", False

    def _check_circuit_breaker(self, index_id: str, now: datetime) -> tuple[bool, int]:
        """
        Check circuit breaker and daily trade count.

        Returns (circuit_breaker_active, hz_trades_today).
        Circuit breaker: skip if last 2 HZ trades both lost.
        """
        try:
            rows = self._db.fetch_all(
                """SELECT outcome, signal_type, generated_at FROM trading_signals
                   WHERE index_id = ? AND reasoning LIKE '%"is_hero_zero": true%'
                   ORDER BY generated_at DESC LIMIT 10""",
                (index_id,),
            )

            # Count today's trades
            today = now.date()
            today_count = sum(
                1
                for r in rows
                if r.get("signal_type") in ("BUY_CALL", "BUY_PUT")
                and (r.get("generated_at") or "").split("T")[0] == today.isoformat()
            )

            # Check circuit breaker: last 2 completed trades both lost
            completed = [r for r in rows if r.get("outcome") in ("WIN", "LOSS")]
            cb_active = (
                len(completed) >= 2 and all(c.get("outcome") == "LOSS" for c in completed[: self._CB_LOOKBACK])
            )

            return cb_active, today_count
        except Exception:
            logger.exception("Circuit breaker query failed (non-fatal)")
            return False, 0

    def _compute_sl(self, index_id: str, spot: float, direction: str) -> float:
        """
        Compute stop loss with hard cap.

        Returns SL price (not distance).
        """
        pts = self._BANKNIFTY_SL_PTS if "BANK" in index_id else self._NIFTY_SL_PTS
        slippage = 2.0

        if direction == "BUY_CALL":
            return spot - pts - slippage
        else:  # BUY_PUT
            return spot + pts + slippage

    def _determine_direction(self, momentum_dir: str, skew_dir: str) -> str:
        """
        Determine trade direction from momentum and skew.

        Momentum is primary. Skew used only for tie-breaking if momentum is NEUTRAL.
        """
        if momentum_dir == "BULLISH":
            return "BUY_CALL"
        if momentum_dir == "BEARISH":
            return "BUY_PUT"
        # Momentum NEUTRAL — use skew
        if skew_dir == "BULLISH":
            return "BUY_CALL"
        if skew_dir == "BEARISH":
            return "BUY_PUT"
        return "NO_TRADE"

    def _no_trigger(self, reason: str) -> HeroZeroResult:
        """Return a NO_TRADE result."""
        return HeroZeroResult(
            is_triggered=False,
            signal_direction="NO_TRADE",
            net_gex=0.0,
            gex_regime="NEUTRAL",
            max_pain=0.0,
            spot=0.0,
            max_pain_proximity_pct=999.0,
            max_pain_gravity_active=False,
            volume_ratio=0.0,
            volume_confirmed=False,
            iv_skew=0.0,
            skew_direction="NEUTRAL",
            momentum_direction="NEUTRAL",
            momentum_confirmed=False,
            circuit_breaker_active=False,
            hz_trades_today=0,
            stop_loss=0.0,
            position_size_cap=self._POSITION_SIZE_CAP,
            conditions_met={"_reason": reason},
            reasoning=f"NO_TRIGGER: {reason}",
        )

    def _gate_failure(
        self, reason: str, cb_active: bool = False, hz_today: int = 0
    ) -> HeroZeroResult:
        """Return a NO_TRADE result for early gate failures (time, expiry, CB, cap)."""
        return HeroZeroResult(
            is_triggered=False,
            signal_direction="NO_TRADE",
            net_gex=0.0,
            gex_regime="NEUTRAL",
            max_pain=0.0,
            spot=0.0,
            max_pain_proximity_pct=999.0,
            max_pain_gravity_active=False,
            volume_ratio=0.0,
            volume_confirmed=False,
            iv_skew=0.0,
            skew_direction="NEUTRAL",
            momentum_direction="NEUTRAL",
            momentum_confirmed=False,
            circuit_breaker_active=cb_active,
            hz_trades_today=hz_today,
            stop_loss=0.0,
            position_size_cap=self._POSITION_SIZE_CAP,
            conditions_met={"_reason": reason},
            reasoning=f"NO_TRIGGER: {reason}",
        )

    def _no_trigger_with_conditions(
        self,
        conditions: dict[str, bool],
        net_gex: float,
        max_pain: float,
        proximity_pct: float,
        volume_ratio: float,
        iv_skew: float,
        skew_dir: str,
        mom_dir: str,
        mom_confirmed: bool,
        cb_active: bool,
        hz_today: int,
        spot: float,
    ) -> HeroZeroResult:
        """Return NO_TRADE with full diagnostic info."""
        failed = [k for k, v in conditions.items() if not v]
        reason = f"Conditions not met: {', '.join(failed)}"
        return HeroZeroResult(
            is_triggered=False,
            signal_direction="NO_TRADE",
            net_gex=net_gex,
            gex_regime="SHORT_GAMMA" if net_gex < 0 else "LONG_GAMMA",
            max_pain=max_pain,
            spot=spot,
            max_pain_proximity_pct=proximity_pct,
            max_pain_gravity_active=proximity_pct <= self._MAX_PAIN_PROXIMITY_PCT,
            volume_ratio=volume_ratio,
            volume_confirmed=volume_ratio >= self._VOLUME_RATIO_MIN,
            iv_skew=iv_skew,
            skew_direction=skew_dir,
            momentum_direction=mom_dir,
            momentum_confirmed=mom_confirmed,
            circuit_breaker_active=cb_active,
            hz_trades_today=hz_today,
            stop_loss=0.0,
            position_size_cap=self._POSITION_SIZE_CAP,
            conditions_met=conditions,
            reasoning=reason,
        )

    def _build_reasoning(
        self,
        conditions: dict[str, bool],
        net_gex: float,
        max_pain: float,
        proximity_pct: float,
        volume_ratio: float,
        skew_dir: str,
        mom_dir: str,
        direction: str,
    ) -> str:
        """Build human-readable reasoning for trigger."""
        lines = [
            f"HERO_ZERO triggered — {direction}",
            "",
            "Conditions met:",
            f"  ✓ Short gamma: net GEX = {net_gex:.0f} (dealers hedging by accelerating price)",
            f"  ✓ Max pain gravity: proximity = {proximity_pct:.1f}% (max pain at {max_pain:.0f})",
            f"  ✓ Volume spike: ratio = {volume_ratio:.2f}x (threshold 1.5x)",
            f"  ✓ Momentum: {mom_dir} (last 3 bars aligned)",
            "",
            "Context:",
            f"  IV skew: {skew_dir} (bearish hedge demand > bullish chase)",
            "",
            "Risk controls:",
            "  • Position size: 25% cap (hard)",
            "  • Stop loss: hard-capped (see signal SL)",
            "  • Hard exit: 14:55 IST",
            "  • Circuit breaker: skip if last 2 HZ trades lost",
        ]
        return "\n".join(lines)
