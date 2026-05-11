"""
Gamma Exposure (GEX) calculator for F&O symbols.

Computes per-strike gamma exposure, dealer positioning profiles,
and key structural levels (call wall, put wall, GEX flip point).

options_chain format expected from SmartAPIClient:
    chain[strike: float][expiry_date_str: str]['CE' | 'PE'] = {
        'openInterest': int,
        'lastPrice': float,
        'impliedVolatility': float,   # percentage, e.g. 15.0 for 15%
    }
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.analysis.gex.black_scholes import (
    BSMInputs,
    BlackScholesEngine,
    get_dte,
    get_risk_free_rate,
)
from src.analysis.gex.models import GEXProfile, GEXSnapshot, StrikeGEX

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")


class GEXCalculator:
    """
    Computes Gamma Exposure profiles for F&O symbols.

    Thread-safe: internal lock guards profile updates and history writes.
    """

    def __init__(self, smart_api=None, db=None) -> None:
        self.smart_api = smart_api
        self.db = db

        self.bs_engine = BlackScholesEngine()
        self.risk_free_rate = get_risk_free_rate()

        try:
            from config.settings import settings  # type: ignore[import]

            gex = settings.gex
            self._strikes_range: int = gex.gex_strikes_range
            self._expiries_to_include: int = gex.gex_expiries_to_include
            self._refresh_minutes: int = gex.gex_refresh_minutes
            self._near_expiry_weight: float = gex.near_expiry_weight
            self._next_expiry_weight: float = gex.next_expiry_weight
            self._atm_threshold_pct: float = gex.atm_threshold_pct
            self._call_wall_min_gex_cr: float = gex.call_wall_min_gex_cr
        except Exception:
            self._strikes_range = 20
            self._expiries_to_include = 2
            self._refresh_minutes = 5
            self._near_expiry_weight = 1.0
            self._next_expiry_weight = 0.5
            self._atm_threshold_pct = 0.5
            self._call_wall_min_gex_cr = 50.0

        # (timestamp, net_gex) pairs for velocity computation
        self._gex_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._latest_profiles: dict[str, GEXProfile] = {}
        self._calculation_lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _determine_strike_spacing(self, symbol: str, strikes: list[float]) -> float:
        sym = symbol.upper()
        if "BANKNIFTY" in sym or "BANK NIFTY" in sym:
            return 100.0
        if "NIFTY" in sym and "BANK" not in sym:
            return 50.0
        if len(strikes) >= 2:
            sorted_s = sorted(set(strikes))
            gaps = [
                sorted_s[i + 1] - sorted_s[i]
                for i in range(len(sorted_s) - 1)
                if sorted_s[i + 1] > sorted_s[i]
            ]
            if gaps:
                return min(gaps)
        return 100.0

    def _normalize_iv(self, iv_raw: float | None) -> float | None:
        """Convert percentage IV (15.0) → decimal (0.15) for BSMInputs."""
        if iv_raw is None or iv_raw <= 0:
            return None
        if iv_raw > 1.0:
            iv_raw = iv_raw / 100.0
        if not (0.001 <= iv_raw <= 9.0):
            return None
        return iv_raw

    def _solve_iv_from_ltp(
        self,
        ltp: float,
        spot: float,
        strike: float,
        dte_days: float,
        option_type: str,
    ) -> float:
        """Back-solve IV from market LTP; fall back to 0.20 on failure."""
        try:
            placeholder = BSMInputs(
                spot=spot,
                strike=strike,
                dte_days=dte_days,
                iv=0.20,
                risk_free_rate=self.risk_free_rate,
                option_type=option_type,
            )
            return self.bs_engine.calculate_iv(ltp, placeholder)
        except Exception:
            return 0.20

    def _find_flip_point(self, strikes: list[StrikeGEX], spot_price: float) -> float:
        """
        Find price level where cumulative GEX (summed high-to-low) crosses zero.

        Scans from the highest strike downward; linearly interpolates the exact
        crossing point between the two adjacent strikes where sign changes.
        """
        if not strikes:
            return spot_price

        sorted_strikes = sorted(strikes, key=lambda s: s.strike, reverse=True)
        cumulative = 0.0

        for i, s in enumerate(sorted_strikes):
            prev_cumulative = cumulative
            cumulative += s.net_gex

            if i > 0:
                prev_strike = sorted_strikes[i - 1].strike
                curr_strike = s.strike

                if prev_cumulative > 0 and cumulative < 0:
                    denom = prev_cumulative - cumulative
                    ratio = prev_cumulative / denom if denom != 0 else 0.5
                    return prev_strike + ratio * (curr_strike - prev_strike)

                if prev_cumulative < 0 and cumulative > 0:
                    denom = abs(prev_cumulative) + cumulative
                    ratio = abs(prev_cumulative) / denom if denom != 0 else 0.5
                    return sorted_strikes[i - 1].strike + ratio * (
                        curr_strike - sorted_strikes[i - 1].strike
                    )

        return spot_price

    def _compute_velocity(
        self, symbol: str, current_time: datetime, net_gex: float
    ) -> float:
        """Append to history and return net_gex change over last 30 min."""
        self._gex_history[symbol].append((current_time, net_gex))
        thirty_min_ago = current_time - timedelta(minutes=30)
        historical = [
            (t, g) for t, g in self._gex_history[symbol] if t <= thirty_min_ago
        ]
        if historical:
            return net_gex - historical[-1][1]
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Main calculation
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_gex_profile(
        self,
        symbol: str,
        spot_price: float,
        options_chain: dict,
        lot_size: int,
        current_time: datetime,
    ) -> GEXProfile:
        """
        Build a full GEX profile from raw options chain data.

        Parameters
        ----------
        symbol:
            Instrument symbol, e.g. 'NIFTY', 'BANKNIFTY', 'RELIANCE'.
        spot_price:
            Current underlying price.
        options_chain:
            Nested dict: chain[strike][expiry_iso_str]['CE'|'PE'] = {...}.
        lot_size:
            F&O lot size for the underlying.
        current_time:
            Timezone-aware datetime (IST) when this snapshot was taken.
        """
        # ── Step 1: strike range + expiries ───────────────────────────────────
        all_chain_strikes = [float(k) for k in options_chain]
        strike_spacing = self._determine_strike_spacing(symbol, all_chain_strikes)

        atm_strike = round(spot_price / strike_spacing) * strike_spacing
        min_strike = atm_strike - self._strikes_range * strike_spacing
        max_strike = atm_strike + self._strikes_range * strike_spacing

        expiry_keys: set[str] = set()
        for strike_data in options_chain.values():
            expiry_keys.update(strike_data.keys())

        expiry_dates: list[date] = []
        for ek in expiry_keys:
            try:
                expiry_dates.append(date.fromisoformat(ek))
            except ValueError:
                pass
        expiry_dates.sort()

        today = (
            current_time.date()
            if hasattr(current_time, "date")
            else date.today()
        )
        future_expiries = [e for e in expiry_dates if e >= today]
        expiries_to_use = future_expiries[: self._expiries_to_include]
        if not expiries_to_use:
            expiries_to_use = expiry_dates[: self._expiries_to_include]

        # ── Step 2: per-(strike, expiry) GEX, aggregated by strike ───────────
        strike_gex_map: dict[float, list[StrikeGEX]] = defaultdict(list)

        for expiry_idx, expiry in enumerate(expiries_to_use):
            weight = (
                self._near_expiry_weight
                if expiry_idx == 0
                else self._next_expiry_weight
            )
            expiry_str = expiry.isoformat()

            for raw_strike in sorted(all_chain_strikes):
                strike = float(raw_strike)
                if strike < min_strike or strike > max_strike:
                    continue

                strike_data = options_chain.get(raw_strike) or options_chain.get(strike) or {}
                expiry_data = strike_data.get(expiry_str, {})
                if not expiry_data:
                    continue

                ce_data = expiry_data.get("CE", {})
                pe_data = expiry_data.get("PE", {})

                ce_oi = int(ce_data.get("openInterest", 0))
                pe_oi = int(pe_data.get("openInterest", 0))
                if ce_oi == 0 and pe_oi == 0:
                    continue

                dte_days = get_dte(expiry, current_time) * 365.0

                ce_ltp = float(ce_data.get("lastPrice", 0.0))
                ce_iv = self._normalize_iv(ce_data.get("impliedVolatility"))
                if ce_iv is None:
                    ce_iv = (
                        self._solve_iv_from_ltp(ce_ltp, spot_price, strike, dte_days, "CE")
                        if ce_ltp > 0
                        else 0.20
                    )

                pe_ltp = float(pe_data.get("lastPrice", 0.0))
                pe_iv = self._normalize_iv(pe_data.get("impliedVolatility"))
                if pe_iv is None:
                    pe_iv = (
                        self._solve_iv_from_ltp(pe_ltp, spot_price, strike, dte_days, "PE")
                        if pe_ltp > 0
                        else 0.20
                    )

                # CE Greeks + GEX
                try:
                    ce_inputs = BSMInputs(
                        spot=spot_price,
                        strike=strike,
                        dte_days=dte_days,
                        iv=ce_iv,
                        risk_free_rate=self.risk_free_rate,
                        option_type="CE",
                    )
                    ce_greeks = self.bs_engine.calculate_greeks(ce_inputs)
                    ce_gamma = ce_greeks.gamma
                    ce_gex = (
                        self.bs_engine.calculate_gamma_dollars(ce_inputs, lot_size, ce_oi)
                        * weight
                    )  # positive
                except Exception as exc:
                    logger.debug("GEX[%s] CE failed strike=%.0f: %s", symbol, strike, exc)
                    ce_gamma, ce_gex = 0.0, 0.0

                # PE Greeks + GEX
                try:
                    pe_inputs = BSMInputs(
                        spot=spot_price,
                        strike=strike,
                        dte_days=dte_days,
                        iv=pe_iv,
                        risk_free_rate=self.risk_free_rate,
                        option_type="PE",
                    )
                    pe_greeks = self.bs_engine.calculate_greeks(pe_inputs)
                    pe_gamma = pe_greeks.gamma
                    pe_gex = -(
                        self.bs_engine.calculate_gamma_dollars(pe_inputs, lot_size, pe_oi)
                        * weight
                    )  # negative — dealers short gamma on puts
                except Exception as exc:
                    logger.debug("GEX[%s] PE failed strike=%.0f: %s", symbol, strike, exc)
                    pe_gamma, pe_gex = 0.0, 0.0

                dist_pct = abs(strike - spot_price) / spot_price * 100.0

                sgex = StrikeGEX(
                    strike=strike,
                    expiry=expiry,
                    ce_oi=ce_oi,
                    ce_iv=ce_iv,
                    ce_ltp=ce_ltp,
                    ce_gamma=ce_gamma,
                    ce_gex=ce_gex,
                    pe_oi=pe_oi,
                    pe_iv=pe_iv,
                    pe_ltp=pe_ltp,
                    pe_gamma=pe_gamma,
                    pe_gex=pe_gex,
                    net_gex=ce_gex + pe_gex,
                    is_call_dominant=(ce_gex > abs(pe_gex)),
                    distance_from_spot_pct=dist_pct,
                    is_atm=(dist_pct < self._atm_threshold_pct),
                )
                strike_gex_map[strike].append(sgex)

        # ── Merge expiries per strike (GEX already weighted, sum them) ────────
        merged_strikes: list[StrikeGEX] = []
        for strike_val in sorted(strike_gex_map):
            entries = strike_gex_map[strike_val]
            if not entries:
                continue
            if len(entries) == 1:
                merged_strikes.append(entries[0])
                continue

            base = entries[0]
            total_ce_gex = sum(e.ce_gex for e in entries)
            total_pe_gex = sum(e.pe_gex for e in entries)
            total_net = total_ce_gex + total_pe_gex
            merged = StrikeGEX(
                strike=base.strike,
                expiry=base.expiry,
                ce_oi=sum(e.ce_oi for e in entries),
                ce_iv=base.ce_iv,
                ce_ltp=base.ce_ltp,
                ce_gamma=base.ce_gamma,
                ce_gex=total_ce_gex,
                pe_oi=sum(e.pe_oi for e in entries),
                pe_iv=base.pe_iv,
                pe_ltp=base.pe_ltp,
                pe_gamma=base.pe_gamma,
                pe_gex=total_pe_gex,
                net_gex=total_net,
                is_call_dominant=(total_ce_gex > abs(total_pe_gex)),
                distance_from_spot_pct=base.distance_from_spot_pct,
                is_atm=base.is_atm,
            )
            merged_strikes.append(merged)

        # ── Step 3: aggregate ─────────────────────────────────────────────────
        net_gex = sum(s.net_gex for s in merged_strikes)
        min_gex_rupees = self._call_wall_min_gex_cr * 10_000_000

        strikes_above = [s for s in merged_strikes if s.strike > spot_price]
        call_wall: float | None = None
        if strikes_above:
            best_call = max(strikes_above, key=lambda s: s.ce_gex)
            if best_call.ce_gex >= min_gex_rupees:
                call_wall = best_call.strike

        strikes_below = [s for s in merged_strikes if s.strike < spot_price]
        put_wall: float | None = None
        if strikes_below:
            best_put = min(strikes_below, key=lambda s: s.pe_gex)  # most negative
            if abs(best_put.pe_gex) >= min_gex_rupees:
                put_wall = best_put.strike

        largest_positive_gex = max((s.ce_gex for s in merged_strikes), default=0.0)
        largest_negative_gex = min((s.pe_gex for s in merged_strikes), default=0.0)

        # ── Step 4: flip point ───────────────────────────────────────────────
        gex_flip_point = self._find_flip_point(merged_strikes, spot_price)

        # ── Step 5: regime ───────────────────────────────────────────────────
        total_abs_gex = sum(abs(s.net_gex) for s in merged_strikes)
        regime_ratio = net_gex / total_abs_gex if total_abs_gex > 0 else 0.0
        regime_strength = min(abs(regime_ratio) * 100.0, 100.0)

        if regime_ratio > 0.15:
            regime = "POSITIVE_GEX"
        elif regime_ratio < -0.15:
            regime = "NEGATIVE_GEX"
        else:
            regime = "NEUTRAL_GEX"

        # ── Step 6: velocity ─────────────────────────────────────────────────
        gex_change_30min = self._compute_velocity(symbol, current_time, net_gex)

        # ── Step 7: assemble profile ─────────────────────────────────────────
        total_oi = sum(s.ce_oi + s.pe_oi for s in merged_strikes)

        profile = GEXProfile(
            symbol=symbol,
            spot_price=spot_price,
            timestamp=current_time,
            strikes=sorted(merged_strikes, key=lambda s: s.strike),
            net_gex=net_gex,
            call_wall=call_wall,
            put_wall=put_wall,
            gex_flip_point=gex_flip_point,
            largest_positive_gex=largest_positive_gex,
            largest_negative_gex=largest_negative_gex,
            regime=regime,
            regime_strength=regime_strength,
            gex_change_30min=gex_change_30min,
            expiries_included=expiries_to_use,
            total_oi_analyzed=total_oi,
            lot_size=lot_size,
        )

        with self._calculation_lock:
            self._latest_profiles[symbol] = profile

        self._persist_to_db(profile)
        return profile

    # ─────────────────────────────────────────────────────────────────────────
    # DB persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _persist_to_db(self, profile: GEXProfile) -> None:
        if self.db is None:
            return
        try:
            cursor = self.db.execute(
                """
                INSERT INTO gex_snapshots
                    (symbol, timestamp, spot_price, net_gex, call_wall, put_wall,
                     flip_point, regime, regime_strength, gex_change_30min, date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.symbol,
                    profile.timestamp.isoformat(),
                    profile.spot_price,
                    profile.net_gex / 10_000_000,
                    profile.call_wall,
                    profile.put_wall,
                    profile.gex_flip_point,
                    profile.regime,
                    profile.regime_strength,
                    profile.gex_change_30min / 10_000_000,
                    profile.timestamp.date().isoformat(),
                ),
            )
            snapshot_id = cursor.lastrowid

            strike_rows = [
                (
                    snapshot_id,
                    s.strike,
                    s.expiry.isoformat(),
                    s.ce_oi,
                    s.ce_iv,
                    s.ce_gex / 10_000_000,
                    s.pe_oi,
                    s.pe_iv,
                    s.pe_gex / 10_000_000,
                    s.net_gex / 10_000_000,
                )
                for s in profile.strikes
            ]
            if strike_rows:
                self.db.execute_many(
                    """
                    INSERT INTO gex_strike_data
                        (snapshot_id, strike, expiry, ce_oi, ce_iv, ce_gex,
                         pe_oi, pe_iv, pe_gex, net_gex)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    strike_rows,
                )
        except Exception as exc:
            logger.warning("GEX DB persist failed [%s]: %s", profile.symbol, exc)

    def _persist_wall_break(self, symbol: str, event: dict) -> None:
        if self.db is None:
            return
        try:
            now = datetime.now(tz=_IST)
            self.db.execute(
                """
                INSERT INTO gex_wall_breaks
                    (symbol, wall_type, wall_price, break_time, volume_confirmed, date)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    event["wall_type"],
                    event["wall_price"],
                    now.isoformat(),
                    int(event["volume_confirmed"]),
                    now.date().isoformat(),
                ),
            )
        except Exception as exc:
            logger.warning("Wall break persist failed [%s]: %s", symbol, exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Derived signal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_call_wall_resistance(
        self, symbol: str, current_price: float
    ) -> float | None:
        """Call wall if it's above current price; None if price already broke above."""
        profile = self._latest_profiles.get(symbol)
        if not profile or profile.call_wall is None:
            return None
        return profile.call_wall if profile.call_wall > current_price else None

    def get_put_wall_support(
        self, symbol: str, current_price: float
    ) -> float | None:
        """Put wall if it's below current price; None if price already broke below."""
        profile = self._latest_profiles.get(symbol)
        if not profile or profile.put_wall is None:
            return None
        return profile.put_wall if profile.put_wall < current_price else None

    def is_price_above_flip_point(self, symbol: str, current_price: float) -> bool:
        """
        True  → above flip point → positive GEX territory → rangebound.
        False → below flip point → negative GEX territory → trending.
        """
        profile = self._latest_profiles.get(symbol)
        if not profile:
            return True
        return current_price > profile.gex_flip_point

    def get_distance_to_call_wall(self, symbol: str, current_price: float) -> float:
        """Points between current price and call wall. Returns inf if no valid wall."""
        cw = self.get_call_wall_resistance(symbol, current_price)
        return (cw - current_price) if cw is not None else float("inf")

    def get_distance_to_put_wall(self, symbol: str, current_price: float) -> float:
        """Points between current price and put wall. Returns inf if no valid wall."""
        pw = self.get_put_wall_support(symbol, current_price)
        return (current_price - pw) if pw is not None else float("inf")

    def get_gamma_at_price(self, symbol: str, price: float) -> float:
        """
        Net dealer gamma at an arbitrary price, interpolated between adjacent strikes.

        High absolute gamma at a price level → sticky price action.
        Low gamma between levels → price moves quickly through.
        """
        profile = self._latest_profiles.get(symbol)
        if not profile or not profile.strikes:
            return 0.0

        strikes = sorted(profile.strikes, key=lambda s: s.strike)

        if price <= strikes[0].strike:
            return strikes[0].net_gex
        if price >= strikes[-1].strike:
            return strikes[-1].net_gex

        for i in range(len(strikes) - 1):
            lo, hi = strikes[i], strikes[i + 1]
            if lo.strike <= price <= hi.strike:
                span = hi.strike - lo.strike
                if span == 0:
                    return lo.net_gex
                ratio = (price - lo.strike) / span
                return lo.net_gex + ratio * (hi.net_gex - lo.net_gex)

        return 0.0

    def get_gex_signal_for_engine(
        self, symbol: str, current_price: float
    ) -> dict:
        """
        Structured GEX signal for DecisionEngine consumption.

        Returns regime context, wall levels, and a strategy recommendation
        with a confidence_multiplier (0.5–1.5) to scale signal confidence.
        """
        profile = self._latest_profiles.get(symbol)

        if profile is None:
            return {
                "regime": "NEUTRAL_GEX",
                "regime_strength": 0.0,
                "net_gex": 0.0,
                "gex_trend": "STABLE",
                "call_wall": None,
                "put_wall": None,
                "flip_point": current_price,
                "price_vs_flip": "ABOVE",
                "strategy_recommendation": "NORMAL",
                "confidence_multiplier": 1.0,
                "reason": "No GEX data available",
            }

        change = profile.gex_change_30min
        gex_trend = "INCREASING" if change > 0 else ("DECREASING" if change < 0 else "STABLE")
        price_vs_flip = "ABOVE" if current_price > profile.gex_flip_point else "BELOW"

        regime = profile.regime
        strength = profile.regime_strength

        call_wall = self.get_call_wall_resistance(symbol, current_price)
        put_wall = self.get_put_wall_support(symbol, current_price)
        dist_cw = self.get_distance_to_call_wall(symbol, current_price)
        dist_pw = self.get_distance_to_put_wall(symbol, current_price)
        near_wall = (dist_cw < 30) or (dist_pw < 30)

        # ── Base strategy ──────────────────────────────────────────────────
        if regime == "NEGATIVE_GEX" and strength > 60:
            strategy = "DIRECTIONAL_AGGRESSIVE"
            mult = 1.25
            reason = "Strong negative GEX — dealers amplify moves, buy options"
        elif regime == "NEGATIVE_GEX" and strength >= 30:
            strategy = "DIRECTIONAL_NORMAL"
            mult = 1.10
            reason = "Moderate negative GEX — directional trades favored"
        elif regime == "NEGATIVE_GEX":
            strategy = "DIRECTIONAL_NORMAL"
            mult = 1.05
            reason = "Weak negative GEX — slight directional bias"
        elif regime == "POSITIVE_GEX" and strength > 60:
            if near_wall:
                strategy = "WALL_TRADE"
                mult = 1.15
                reason = "Price approaching GEX wall — strong rejection likely"
            else:
                strategy = "AVOID_DIRECTIONAL"
                mult = 0.60
                reason = "Strong positive GEX — dealers stabilize, options decay fast"
        elif regime == "POSITIVE_GEX" and strength >= 30:
            strategy = "RANGE_TRADE"
            mult = 0.80
            reason = "Moderate positive GEX — range-bound, tighter targets"
        else:
            strategy = "NORMAL"
            mult = 1.0
            reason = "Neutral GEX — no adjustment"

        # ── GEX trend modifier ─────────────────────────────────────────────
        if change < 0:
            if strategy == "AVOID_DIRECTIONAL":
                strategy = "RANGE_TRADE"
            mult += 0.10
            reason += " | GEX declining — volatility increasing"
        elif change > 0 and strategy in ("DIRECTIONAL_AGGRESSIVE", "DIRECTIONAL_NORMAL"):
            mult -= 0.05

        mult = max(0.5, min(1.5, mult))

        return {
            "regime": regime,
            "regime_strength": strength,
            "net_gex": profile.net_gex,
            "gex_trend": gex_trend,
            "call_wall": call_wall,
            "put_wall": put_wall,
            "flip_point": profile.gex_flip_point,
            "price_vs_flip": price_vs_flip,
            "strategy_recommendation": strategy,
            "confidence_multiplier": mult,
            "reason": reason,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Wall break detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_wall_break(
        self,
        symbol: str,
        previous_price: float,
        current_price: float,
        volume: int,
        avg_volume: float,
    ) -> dict | None:
        """
        Detect a GEX wall break between the previous and current price tick.

        Call wall break → dealers must BUY to re-hedge → gamma squeeze up → BUY_CALL.
        Put wall break  → dealers must SELL to re-hedge → gamma squeeze down → BUY_PUT.

        Wall breaks are rare and high-conviction. Every event is logged and persisted.
        """
        profile = self._latest_profiles.get(symbol)
        if not profile:
            return None

        call_wall = profile.call_wall
        put_wall = profile.put_wall
        event: dict | None = None

        if call_wall is not None and previous_price < call_wall <= current_price:
            vol_confirmed = avg_volume > 0 and volume > 1.5 * avg_volume
            significance = (
                "HIGH" if (avg_volume > 0 and volume > 2 * avg_volume) else "MEDIUM"
            )
            event = {
                "wall_type": "CALL_WALL_BREAK",
                "wall_price": call_wall,
                "break_distance": current_price - call_wall,
                "volume_confirmed": vol_confirmed,
                "signal": "BUY_CALL",
                "significance": significance,
                "reason": (
                    f"Price broke above call wall {call_wall:.0f} — "
                    "dealers must buy to re-hedge → gamma squeeze up"
                ),
            }
            logger.warning(
                "WALL BREAK [%s] CALL_WALL %.2f | prev=%.2f curr=%.2f vol_conf=%s",
                symbol,
                call_wall,
                previous_price,
                current_price,
                vol_confirmed,
            )

        elif put_wall is not None and previous_price > put_wall >= current_price:
            vol_confirmed = avg_volume > 0 and volume > 1.5 * avg_volume
            significance = (
                "HIGH" if (avg_volume > 0 and volume > 2 * avg_volume) else "MEDIUM"
            )
            event = {
                "wall_type": "PUT_WALL_BREAK",
                "wall_price": put_wall,
                "break_distance": put_wall - current_price,
                "volume_confirmed": vol_confirmed,
                "signal": "BUY_PUT",
                "significance": significance,
                "reason": (
                    f"Price broke below put wall {put_wall:.0f} — "
                    "dealers must sell to re-hedge → gamma squeeze down"
                ),
            }
            logger.warning(
                "WALL BREAK [%s] PUT_WALL %.2f | prev=%.2f curr=%.2f vol_conf=%s",
                symbol,
                put_wall,
                previous_price,
                current_price,
                vol_confirmed,
            )

        if event is not None:
            self._persist_wall_break(symbol, event)

        return event

    # ─────────────────────────────────────────────────────────────────────────
    # Snapshot builder
    # ─────────────────────────────────────────────────────────────────────────

    def build_snapshot(self, timestamp: Optional[datetime] = None) -> GEXSnapshot:
        """Wrap all latest profiles into a GEXSnapshot."""
        ts = timestamp or datetime.now(tz=_IST)
        profiles = dict(self._latest_profiles)

        index_gex = sum(
            p.net_gex
            for sym, p in profiles.items()
            if "NIFTY" in sym.upper()
        )

        total_abs = sum(abs(p.net_gex) for p in profiles.values())
        if profiles:
            ratio = index_gex / total_abs if total_abs > 0 else 0.0
            market_regime = (
                "POSITIVE_GEX"
                if ratio > 0.15
                else ("NEGATIVE_GEX" if ratio < -0.15 else "NEUTRAL_GEX")
            )
        else:
            market_regime = "NEUTRAL_GEX"

        return GEXSnapshot(
            timestamp=ts,
            profiles=profiles,
            market_wide_gex=index_gex,
            market_regime=market_regime,
        )
