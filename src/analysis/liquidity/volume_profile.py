"""
VolumeProfileAnalyzer — intraday volume profile with POC, Value Area, HVN, LVN.

Registers a 1-min candle callback on CandleStore.  Every 5 new candles the
running profile is recalculated.  Previous-day profiles are loaded from DB.

POC, VAH, VAL, HVNs, and LVNs are the most reliable intraday S/R levels
because they reflect actual transaction history, not just price patterns.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.analysis.liquidity.candle_store import CandleStore
from src.analysis.liquidity.models import Candle, VolumeProfile, VolumeProfileLevel
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)


class VolumeProfileAnalyzer:
    """
    Builds and maintains intraday volume profiles for tracked symbols.

    Registers a callback on CandleStore for every 1-min close.  Profile is
    recalculated every 5 candles once min_candles_for_profile is reached.
    """

    def __init__(self, candle_store: CandleStore, db: DatabaseManager) -> None:
        self._cs = candle_store
        self._db = db

        cfg = settings.volume_profile
        self._tick_size: float = cfg.tick_size
        self._bucket_size: int = cfg.profile_bucket_size
        self._value_area_pct: float = cfg.value_area_pct
        self._hvn_threshold_pct: float = cfg.hvn_threshold_pct
        self._lvn_threshold_pct: float = cfg.lvn_threshold_pct
        self._min_candles: int = cfg.min_candles_for_profile

        self._daily_profiles: dict[str, VolumeProfile] = {}
        self._previous_profiles: dict[str, VolumeProfile] = {}

        # Incremental state for running profile
        self._running_candles: dict[str, list[Candle]] = {}
        self._last_candle_date: dict[str, date] = {}
        self._candle_count: dict[str, int] = {}

        self._ensure_db_tables()
        self._cs.register_candle_callback("1min", self._on_1min_candle)

    # ── Profile calculation ────────────────────────────────────────────────────

    def calculate_profile(
        self,
        symbol: str,
        candles: list[Candle],
        date_obj: date,
    ) -> VolumeProfile:
        """Build a complete VolumeProfile from a list of 1-min candles."""
        if not candles:
            raise ValueError(f"calculate_profile: no candles for {symbol}")

        bs = float(self._bucket_size)

        # Step 1 — build price buckets aligned to bucket_size grid
        price_min = min(c.low for c in candles)
        price_max = max(c.high for c in candles)

        bucket_start = math.floor(price_min / bs) * bs
        bucket_end = math.ceil(price_max / bs) * bs

        buckets: list[float] = []
        b = bucket_start
        while b <= bucket_end:
            buckets.append(round(b, 2))
            b = round(b + bs, 2)

        if not buckets:
            buckets = [round(bucket_start, 2)]

        bucket_volume: dict[float, float] = {bp: 0.0 for bp in buckets}

        # Step 2 — distribute volume proportionally to buckets per candle
        for candle in candles:
            in_range = [bp for bp in buckets if candle.low <= bp <= candle.high]
            if not in_range:
                # Degenerate case (doji between bucket boundaries) — assign to closest
                in_range = [min(buckets, key=lambda bp: abs(bp - candle.close))]
            vol_per = candle.volume / len(in_range)
            for bp in in_range:
                bucket_volume[bp] += vol_per

        total_volume = sum(bucket_volume.values())
        if total_volume <= 0:
            total_volume = 1.0

        # Step 3 — Point of Control (highest volume bucket)
        poc_price = max(bucket_volume, key=lambda bp: bucket_volume[bp])

        # Step 4 — Value Area (70% of volume centred on POC)
        sorted_by_price = sorted(buckets)
        poc_index = sorted_by_price.index(poc_price)

        up_index = poc_index + 1
        down_index = poc_index - 1
        va_volume = bucket_volume[poc_price]
        target_volume = total_volume * self._value_area_pct

        while va_volume < target_volume:
            vol_above = (
                bucket_volume[sorted_by_price[up_index]]
                if up_index < len(sorted_by_price) else 0.0
            )
            vol_below = (
                bucket_volume[sorted_by_price[down_index]]
                if down_index >= 0 else 0.0
            )

            if vol_above == 0.0 and vol_below == 0.0:
                break

            if vol_above >= vol_below:
                va_volume += vol_above
                up_index += 1
            else:
                va_volume += vol_below
                down_index -= 1

        vah = sorted_by_price[min(up_index - 1, len(sorted_by_price) - 1)]
        val = sorted_by_price[max(down_index + 1, 0)]

        # Step 5 — HVN and LVN (top 5 each by proximity to POC)
        avg_vol = total_volume / len(buckets)
        raw_hvns = [bp for bp in buckets if bucket_volume[bp] > self._hvn_threshold_pct * avg_vol]
        raw_lvns = [bp for bp in buckets if bucket_volume[bp] < self._lvn_threshold_pct * avg_vol]

        hvns = sorted(raw_hvns, key=lambda bp: abs(bp - poc_price))[:5]
        lvns = sorted(raw_lvns, key=lambda bp: abs(bp - poc_price))[:5]

        hvn_set = set(hvns)
        lvn_set = set(lvns)

        # Step 6 — Build VolumeProfileLevel objects
        levels: list[VolumeProfileLevel] = []
        for bp in sorted_by_price:
            vol = int(bucket_volume[bp])
            vol_pct = round(bucket_volume[bp] / total_volume * 100.0, 4)

            if bp == poc_price:
                ltype = "POC"
            elif bp == vah and bp != poc_price:
                ltype = "VAH"
            elif bp == val and bp != poc_price and bp != vah:
                ltype = "VAL"
            elif bp in hvn_set:
                ltype = "HVN"
            elif bp in lvn_set:
                ltype = "LVN"
            else:
                ltype = "NORMAL"

            levels.append(
                VolumeProfileLevel(
                    symbol=symbol,
                    date=date_obj,
                    timeframe="1min",
                    price=bp,
                    volume=vol,
                    volume_pct=vol_pct,
                    level_type=ltype,
                    is_value_area=(val <= bp <= vah),
                )
            )

        return VolumeProfile(
            symbol=symbol,
            date=date_obj,
            timeframe="1min",
            poc=poc_price,
            vah=vah,
            val=val,
            total_volume=int(total_volume),
            levels=levels,
            hvns=sorted(hvns),
            lvns=sorted(lvns),
            calculated_at=datetime.now(_IST),
        )

    # ── Previous-day profile ───────────────────────────────────────────────────

    def load_previous_day_profile(self, symbol: str) -> Optional[VolumeProfile]:
        """Load yesterday's completed profile from DB and cache it."""
        yesterday = (datetime.now(_IST).date() - timedelta(days=1)).isoformat()
        try:
            from src.database import queries as Q

            row = self._db.fetch_one(Q.SELECT_VOLUME_PROFILE, (symbol, yesterday))
            if not row:
                return None

            profile_id = row["id"]
            level_rows = self._db.fetch_all(Q.LIST_VOLUME_PROFILE_LEVELS, (profile_id,))

            prev_date = date.fromisoformat(yesterday)
            levels = [
                VolumeProfileLevel(
                    symbol=symbol,
                    date=prev_date,
                    timeframe="1min",
                    price=r["price"],
                    volume=r["volume"],
                    volume_pct=r["volume_pct"],
                    level_type=r["level_type"],
                    is_value_area=bool(r["is_value_area"]),
                )
                for r in level_rows
            ]

            hvns = sorted(lvl.price for lvl in levels if lvl.level_type == "HVN")
            lvns = sorted(lvl.price for lvl in levels if lvl.level_type == "LVN")

            ts_raw = row["calculated_at"]
            if isinstance(ts_raw, str):
                calc_at = datetime.fromisoformat(ts_raw)
                if calc_at.tzinfo is None:
                    calc_at = calc_at.replace(tzinfo=_IST)
            else:
                calc_at = datetime.now(_IST)

            profile = VolumeProfile(
                symbol=symbol,
                date=prev_date,
                timeframe="1min",
                poc=row["poc"],
                vah=row["vah"],
                val=row["val"],
                total_volume=row["total_volume"],
                levels=levels,
                hvns=hvns,
                lvns=lvns,
                calculated_at=calc_at,
            )
            self._previous_profiles[symbol] = profile
            return profile

        except Exception as exc:
            logger.debug(
                "VolumeProfileAnalyzer: load_previous_day_profile failed for %s: %s",
                symbol, exc,
            )
            return None

    def get_previous_day_levels(self, symbol: str) -> dict:
        """Return yesterday's POC, VAH, VAL — the three most important carry-forward levels."""
        profile = self._previous_profiles.get(symbol)
        if not profile:
            return {"prev_poc": 0.0, "prev_vah": 0.0, "prev_val": 0.0}
        return {
            "prev_poc": profile.poc,
            "prev_vah": profile.vah,
            "prev_val": profile.val,
        }

    # ── Signal methods ─────────────────────────────────────────────────────────

    def get_price_context(self, symbol: str, current_price: float) -> dict:
        """
        Return volume profile context dict for current_price.

        Keys: location, poc, vah, val, nearest_hvn_above, nearest_hvn_below,
              nearest_lvn_above, nearest_lvn_below, signal, strength, reason.
        """
        profile = self._daily_profiles.get(symbol)
        if not profile:
            return {
                "location": "UNKNOWN",
                "poc": 0.0,
                "vah": 0.0,
                "val": 0.0,
                "nearest_hvn_above": 0.0,
                "nearest_hvn_below": 0.0,
                "nearest_lvn_above": 0.0,
                "nearest_lvn_below": 0.0,
                "signal": "NEUTRAL",
                "strength": 0.0,
                "reason": "No volume profile available",
            }

        poc, vah, val = profile.poc, profile.vah, profile.val
        hvns = sorted(profile.hvns)
        lvns = sorted(profile.lvns)

        # Use >= / <= so prices exactly at a node level are captured by both directions
        nearest_hvn_above = next((h for h in hvns if h >= current_price), 0.0)
        nearest_hvn_below = next((h for h in reversed(hvns) if h <= current_price), 0.0)
        nearest_lvn_above = next((lv for lv in lvns if lv >= current_price), 0.0)
        nearest_lvn_below = next((lv for lv in reversed(lvns) if lv <= current_price), 0.0)

        base = {
            "poc": poc,
            "vah": vah,
            "val": val,
            "nearest_hvn_above": nearest_hvn_above,
            "nearest_hvn_below": nearest_hvn_below,
            "nearest_lvn_above": nearest_lvn_above,
            "nearest_lvn_below": nearest_lvn_below,
        }

        poc_tolerance = poc * 0.001  # 0.1% of POC
        bucket_half = self._bucket_size / 2.0

        # AT_POC — highest priority
        if abs(current_price - poc) <= poc_tolerance:
            return {
                **base,
                "location": "AT_POC",
                "signal": "NEUTRAL",
                "strength": 30.0,
                "reason": "Price at POC — highest volume level, expect two-way moves",
            }

        if current_price > vah:
            # Check for breakout through LVN above VAH
            in_lvn_above = (
                nearest_lvn_above > 0.0
                and abs(current_price - nearest_lvn_above) <= bucket_half
            )
            if in_lvn_above:
                return {
                    **base,
                    "location": "IN_LVN",
                    "signal": "BULLISH",
                    "strength": 70.0,
                    "reason": f"Price above VAH ({vah:.2f}) in LVN — breakout through thin zone",
                }
            return {
                **base,
                "location": "ABOVE_VAH",
                "signal": "BEARISH",
                "strength": 55.0,
                "reason": f"Price above value area high ({vah:.2f}) — mean reversion probable",
            }

        if current_price < val:
            return {
                **base,
                "location": "BELOW_VAL",
                "signal": "BULLISH",
                "strength": 55.0,
                "reason": f"Price below value area low ({val:.2f}) — mean reversion probable",
            }

        # Within value area — check HVN proximity
        at_hvn = (
            (nearest_hvn_above > 0.0 and abs(current_price - nearest_hvn_above) <= bucket_half)
            or (nearest_hvn_below > 0.0 and abs(current_price - nearest_hvn_below) <= bucket_half)
        )
        if at_hvn:
            return {
                **base,
                "location": "AT_HVN",
                "signal": "NEUTRAL",
                "strength": 40.0,
                "reason": "Price at high volume node — expect consolidation",
            }

        # Check LVN proximity within value area
        in_lvn = (
            (nearest_lvn_above > 0.0 and abs(current_price - nearest_lvn_above) <= bucket_half)
            or (nearest_lvn_below > 0.0 and abs(current_price - nearest_lvn_below) <= bucket_half)
        )
        if in_lvn:
            direction = "BULLISH" if current_price > poc else "BEARISH"
            return {
                **base,
                "location": "IN_LVN",
                "signal": direction,
                "strength": 65.0,
                "reason": "Price in low volume node — expect fast directional move",
            }

        return {
            **base,
            "location": "IN_VALUE_AREA",
            "signal": "NEUTRAL",
            "strength": 20.0,
            "reason": f"Price within value area ({val:.2f} – {vah:.2f})",
        }

    def get_vp_targets(
        self,
        symbol: str,
        current_price: float,
        direction: str,
    ) -> list[float]:
        """
        Return ordered price targets (within 2% intraday) based on volume profile structure.

        For BULLISH: targets are HVNs above, VAH, POC (if above), prev-day VAH.
        For BEARISH: targets are HVNs below, VAL, POC (if below), prev-day VAL.
        Price will slow/reverse at each HVN and move quickly through LVNs.
        """
        profile = self._daily_profiles.get(symbol)
        if not profile:
            return []

        prev = self._previous_profiles.get(symbol)
        poc, vah, val = profile.poc, profile.vah, profile.val
        hvns = sorted(profile.hvns)
        max_dist = current_price * 0.02

        targets: list[float] = []

        if direction == "BULLISH":
            hvns_above = [h for h in hvns if h > current_price]
            if hvns_above:
                targets.append(hvns_above[0])
            if vah > current_price:
                targets.append(vah)
            if poc > current_price:
                targets.append(poc)
            if prev and prev.vah > current_price:
                targets.append(prev.vah)
        else:  # BEARISH
            hvns_below = sorted([h for h in hvns if h < current_price], reverse=True)
            if hvns_below:
                targets.append(hvns_below[0])
            if val < current_price:
                targets.append(val)
            if poc < current_price:
                targets.append(poc)
            if prev and prev.val < current_price:
                targets.append(prev.val)

        # Intraday filter: only within 2% of current price
        return [t for t in targets if abs(t - current_price) <= max_dist]

    def is_price_in_acceptance(self, symbol: str, current_price: float) -> bool:
        """
        True if price is within Value Area AND near POC.

        Indicates market is in 'acceptance' — balanced, efficient price discovery.
        In acceptance, avoid directional trades.  Outside VA = exploration mode.
        """
        profile = self._daily_profiles.get(symbol)
        if not profile:
            return False
        in_va = profile.val <= current_price <= profile.vah
        near_poc = abs(current_price - profile.poc) / profile.poc <= 0.005
        return in_va and near_poc

    # ── CandleStore callback ────────────────────────────────────────────────────

    def _on_1min_candle(self, candle: Candle) -> None:
        symbol = candle.symbol
        candle_date = candle.timestamp.astimezone(_IST).date()

        # Reset running state at start of each new trading day
        if self._last_candle_date.get(symbol) != candle_date:
            self._running_candles[symbol] = []
            self._candle_count[symbol] = 0
            self._last_candle_date[symbol] = candle_date

        self._running_candles[symbol].append(candle)
        self._candle_count[symbol] += 1

        if self._candle_count[symbol] % 5 != 0:
            return

        if len(self._running_candles[symbol]) < self._min_candles:
            return

        try:
            profile = self.calculate_profile(
                symbol, self._running_candles[symbol], candle_date
            )
            self._daily_profiles[symbol] = profile
            self._persist_profile(profile)
            logger.debug(
                "VolumeProfileAnalyzer: %s profile updated — POC=%.2f VAH=%.2f VAL=%.2f",
                symbol, profile.poc, profile.vah, profile.val,
            )
        except Exception as exc:
            logger.error(
                "VolumeProfileAnalyzer: profile calculation failed for %s: %s",
                symbol, exc,
            )

    # ── DB helpers ──────────────────────────────────────────────────────────────

    def _ensure_db_tables(self) -> None:
        try:
            from src.database import queries as Q

            self._db.execute(Q.CREATE_VOLUME_PROFILES)
            self._db.execute(Q.CREATE_VOLUME_PROFILES_IDX)
            self._db.execute(Q.CREATE_VOLUME_PROFILE_LEVELS)
            self._db.execute(Q.CREATE_VOLUME_PROFILE_LEVELS_IDX)
        except Exception as exc:
            logger.error("VolumeProfileAnalyzer: failed to ensure DB tables: %s", exc)

    def _persist_profile(self, profile: VolumeProfile) -> None:
        try:
            from src.database import queries as Q

            date_str = profile.date.isoformat()

            # Delete existing levels for this (symbol, date) before re-inserting
            existing = self._db.fetch_one(
                Q.SELECT_VOLUME_PROFILE, (profile.symbol, date_str)
            )
            if existing:
                self._db.execute(Q.DELETE_VOLUME_PROFILE_LEVELS, (existing["id"],))

            cur = self._db.execute(
                Q.UPSERT_VOLUME_PROFILE,
                (
                    profile.symbol,
                    date_str,
                    profile.poc,
                    profile.vah,
                    profile.val,
                    profile.total_volume,
                    profile.calculated_at.isoformat(),
                ),
            )
            profile_id = cur.lastrowid

            level_rows = [
                (
                    profile_id,
                    lvl.price,
                    lvl.volume,
                    lvl.volume_pct,
                    lvl.level_type,
                    int(lvl.is_value_area),
                )
                for lvl in profile.levels
            ]
            self._db.execute_many(Q.INSERT_VOLUME_PROFILE_LEVEL, level_rows)

        except Exception as exc:
            logger.error(
                "VolumeProfileAnalyzer: DB persist failed for %s: %s",
                profile.symbol, exc,
            )
