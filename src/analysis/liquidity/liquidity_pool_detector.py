"""
LiquidityPoolDetector — identifies and maintains all liquidity pools.

Scans multiple timeframes for equal highs/lows, previous day/week levels,
round numbers, and multi-touch clusters.  Updates on every 15-min candle
close via CandleStore callback.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.analysis.liquidity.candle_store import CandleStore
from src.analysis.liquidity.models import Candle, LiquidityLevel, LiquiditySnapshot
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# How many candles to scan per detection pass
_SWING_SCAN_CANDLES = 50
_MULTI_TOUCH_CANDLES = 100


class LiquidityPoolDetector:
    """
    Identifies liquidity pools (clustered stop-loss levels) for a set of symbols.

    Registers a callback on CandleStore for 15-min closes; on each close it
    re-runs all detection passes, merges results, and persists to DB.
    """

    def __init__(self, candle_store: CandleStore, db: DatabaseManager) -> None:
        self._cs = candle_store
        self._db = db

        cfg = settings.liquidity
        self._tolerance_pct: float = cfg.equal_high_low_tolerance_pct
        self._min_strength: int = cfg.min_level_strength
        self._max_per_side: int = cfg.max_levels_per_side
        self._expiry_hours: int = cfg.level_expiry_hours
        self._increments: list[int] = list(cfg.round_number_increments)
        self._rn_range_pct: float = cfg.round_number_range_pct
        self._mt_tolerance_pct: float = cfg.multi_touch_tolerance_pct
        self._bias_ratio: float = cfg.bias_proximity_ratio

        self._levels: dict[str, list[LiquidityLevel]] = {}
        self._previous_day_data: dict[str, dict] = {}

        self._ensure_db_tables()
        self._cs.register_candle_callback("15min", self._on_15min_candle)

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_previous_day_data(
        self,
        symbol: str,
        pdh: float,
        pdl: float,
        pdc: float,
        pwh: float = 0.0,
        pwl: float = 0.0,
    ) -> None:
        """Store previous day/week OHLC reference levels for *symbol*.

        Call this at 9:00 AM before the market opens.
        """
        self._previous_day_data[symbol] = {
            "pdh": pdh,
            "pdl": pdl,
            "pdc": pdc,
            "pwh": pwh,
            "pwl": pwl,
        }

    def update_levels(self, symbol: str, current_price: float) -> None:
        """Re-detect all level types, merge, rank, and persist to DB."""
        now = datetime.now(_IST)

        candles_15 = self._cs.get_candles(symbol, "15min", _SWING_SCAN_CANDLES)
        candles_5 = self._cs.get_candles(symbol, "5min", _SWING_SCAN_CANDLES)
        candles_15_long = self._cs.get_candles(symbol, "15min", _MULTI_TOUCH_CANDLES)

        all_levels: list[LiquidityLevel] = []

        all_levels.extend(self._detect_equal_highs(candles_15, self._tolerance_pct))
        all_levels.extend(self._detect_equal_highs(candles_5, self._tolerance_pct))
        all_levels.extend(self._detect_equal_lows(candles_15, self._tolerance_pct))
        all_levels.extend(self._detect_equal_lows(candles_5, self._tolerance_pct))
        all_levels.extend(self._detect_previous_day_levels(symbol))
        all_levels.extend(self._detect_round_numbers(current_price, symbol))
        all_levels.extend(self._detect_multi_touch_levels(candles_15_long, self._mt_tolerance_pct))

        merged = self._merge_levels(all_levels)

        # Mark swept
        for lvl in merged:
            if lvl.level_type in ("BSL", "EQUAL_HIGH", "PDH", "PWH", "ROUND_NUMBER"):
                lvl.is_swept = current_price > lvl.price
            else:
                lvl.is_swept = current_price < lvl.price

        # Remove expired
        cutoff = now - timedelta(hours=self._expiry_hours)
        merged = [
            lvl for lvl in merged
            if lvl.first_formed_at >= cutoff or lvl.level_type in ("PDH", "PDL", "PWH", "PWL")
        ]

        # Split, sort by proximity, cap per side
        bsl = sorted(
            [lvl for lvl in merged if lvl.price > current_price],
            key=lambda x: x.price,
        )
        ssl = sorted(
            [lvl for lvl in merged if lvl.price < current_price],
            key=lambda x: x.price,
            reverse=True,
        )

        active = bsl[: self._max_per_side] + ssl[: self._max_per_side]
        self._levels[symbol] = active
        self._persist_levels(symbol, active)

    def mark_level_swept(
        self,
        symbol: str,
        level: LiquidityLevel,
        sweep_candle: Candle,
        wick_beyond: float,
    ) -> None:
        """Mark *level* as swept and persist."""
        level.is_swept = True
        level.swept_at = sweep_candle.timestamp
        level.wick_beyond = wick_beyond

        try:
            from src.database import queries as Q

            today = datetime.now(_IST).date().isoformat()
            price_key = round(level.price)
            rows = self._db.fetch_all(
                Q.LIST_ACTIVE_LIQUIDITY_LEVELS, (symbol, today)
            )
            for row in rows:
                if row["level_type"] == level.level_type and row["price_key"] == price_key:
                    self._db.execute(
                        Q.UPDATE_LIQUIDITY_LEVEL_SWEPT,
                        (sweep_candle.timestamp.isoformat(), wick_beyond, row["id"]),
                    )
                    break
        except Exception as exc:
            logger.error("%s: mark_level_swept DB error: %s", symbol, exc)

        logger.info(
            "%s: %s at %.2f SWEPT (wick_beyond=%.2f)",
            symbol, level.level_type, level.price, wick_beyond,
        )

    def get_nearest_bsl(
        self, symbol: str, current_price: float
    ) -> Optional[LiquidityLevel]:
        """Closest un-swept BSL above *current_price*, or None."""
        candidates = [
            lvl for lvl in self._levels.get(symbol, [])
            if lvl.price > current_price and not lvl.is_swept
        ]
        return min(candidates, key=lambda x: x.price) if candidates else None

    def get_nearest_ssl(
        self, symbol: str, current_price: float
    ) -> Optional[LiquidityLevel]:
        """Closest un-swept SSL below *current_price*, or None."""
        candidates = [
            lvl for lvl in self._levels.get(symbol, [])
            if lvl.price < current_price and not lvl.is_swept
        ]
        return max(candidates, key=lambda x: x.price) if candidates else None

    def get_all_active_levels(
        self, symbol: str, current_price: float
    ) -> tuple[list[LiquidityLevel], list[LiquidityLevel]]:
        """Return (bsl_levels, ssl_levels) sorted nearest-first."""
        all_lvls = self._levels.get(symbol, [])
        bsl = sorted(
            [lvl for lvl in all_lvls if lvl.price > current_price and not lvl.is_swept],
            key=lambda x: x.price,
        )
        ssl = sorted(
            [lvl for lvl in all_lvls if lvl.price < current_price and not lvl.is_swept],
            key=lambda x: x.price,
            reverse=True,
        )
        return bsl, ssl

    def get_liquidity_snapshot(
        self, symbol: str, current_price: float
    ) -> LiquiditySnapshot:
        """Build a LiquiditySnapshot for *symbol* at *current_price*."""
        bsl, ssl = self.get_all_active_levels(symbol, current_price)
        nearest_bsl = bsl[0] if bsl else None
        nearest_ssl = ssl[0] if ssl else None

        dist_bsl = (
            abs(nearest_bsl.price - current_price) / current_price * 100
            if nearest_bsl else 0.0
        )
        dist_ssl = (
            abs(current_price - nearest_ssl.price) / current_price * 100
            if nearest_ssl else 0.0
        )

        bias, reason = self._calculate_liquidity_bias(symbol, current_price)
        now = datetime.now(_IST)

        return LiquiditySnapshot(
            symbol=symbol,
            timestamp=now,
            timeframe="15min",
            active_bsl_levels=bsl,
            active_ssl_levels=ssl,
            nearest_bsl=nearest_bsl,
            nearest_ssl=nearest_ssl,
            distance_to_bsl_pct=dist_bsl,
            distance_to_ssl_pct=dist_ssl,
            liquidity_bias=bias,
            bias_reason=reason,
        )

    # ── Detection methods ──────────────────────────────────────────────────────

    def _detect_equal_highs(
        self, candles: list[Candle], tolerance_pct: float
    ) -> list[LiquidityLevel]:
        if len(candles) < 3:
            return []

        swing_highs: list[tuple[float, datetime]] = []
        for i in range(1, len(candles) - 1):
            if (
                candles[i].high > candles[i - 1].high
                and candles[i].high > candles[i + 1].high
            ):
                swing_highs.append((candles[i].high, candles[i].timestamp))

        return self._group_into_levels(
            swing_highs,
            level_type="EQUAL_HIGH",
            tolerance_pct=tolerance_pct,
            price_selector=max,  # stops rest ABOVE the highest
            symbol=candles[0].symbol,
            timeframe=candles[0].timeframe,
        )

    def _detect_equal_lows(
        self, candles: list[Candle], tolerance_pct: float
    ) -> list[LiquidityLevel]:
        if len(candles) < 3:
            return []

        swing_lows: list[tuple[float, datetime]] = []
        for i in range(1, len(candles) - 1):
            if (
                candles[i].low < candles[i - 1].low
                and candles[i].low < candles[i + 1].low
            ):
                swing_lows.append((candles[i].low, candles[i].timestamp))

        return self._group_into_levels(
            swing_lows,
            level_type="EQUAL_LOW",
            tolerance_pct=tolerance_pct,
            price_selector=min,  # stops rest BELOW the lowest
            symbol=candles[0].symbol,
            timeframe=candles[0].timeframe,
        )

    def _group_into_levels(
        self,
        price_ts_pairs: list[tuple[float, datetime]],
        level_type: str,
        tolerance_pct: float,
        price_selector,   # min or max
        symbol: str,
        timeframe: str,
    ) -> list[LiquidityLevel]:
        if len(price_ts_pairs) < 2:
            return []

        # Sort ascending by price
        sorted_pts = sorted(price_ts_pairs, key=lambda x: x[0])

        groups: list[list[tuple[float, datetime]]] = []
        current: list[tuple[float, datetime]] = [sorted_pts[0]]

        for price, ts in sorted_pts[1:]:
            ref = current[-1][0]
            if abs(price - ref) / ref < tolerance_pct / 100.0:
                current.append((price, ts))
            else:
                groups.append(current)
                current = [(price, ts)]
        groups.append(current)

        levels: list[LiquidityLevel] = []
        for group in groups:
            if len(group) >= self._min_strength:
                level_price = price_selector(p for p, _ in group)
                first_ts = min(ts for _, ts in group)
                last_ts = max(ts for _, ts in group)
                levels.append(
                    LiquidityLevel(
                        symbol=symbol,
                        level_type=level_type,
                        price=level_price,
                        strength=len(group),
                        timeframe=timeframe,
                        first_formed_at=first_ts,
                        last_touched_at=last_ts,
                    )
                )
        return levels

    def _detect_previous_day_levels(self, symbol: str) -> list[LiquidityLevel]:
        data = self._previous_day_data.get(symbol)
        if not data:
            return []

        now = datetime.now(_IST)
        levels: list[LiquidityLevel] = []

        specs = [
            (data.get("pdh", 0.0), "PDH", 3),
            (data.get("pdl", 0.0), "PDL", 3),
            (data.get("pwh", 0.0), "PWH", 4),
            (data.get("pwl", 0.0), "PWL", 4),
        ]
        for price, ltype, strength in specs:
            if price > 0.0:
                levels.append(
                    LiquidityLevel(
                        symbol=symbol,
                        level_type=ltype,
                        price=price,
                        strength=strength,
                        timeframe="1day",
                        first_formed_at=now,
                        last_touched_at=now,
                    )
                )
        return levels

    def _detect_round_numbers(
        self, current_price: float, symbol: str
    ) -> list[LiquidityLevel]:
        now = datetime.now(_IST)
        range_frac = self._rn_range_pct / 100.0
        upper_bound = current_price * (1 + range_frac)
        lower_bound = current_price * (1 - range_frac)

        seen: set[float] = set()
        levels: list[LiquidityLevel] = []

        for increment in self._increments:
            # BSL — round numbers above current_price
            first_above = math.ceil(current_price / increment) * increment
            price = float(first_above)
            while price <= upper_bound:
                if price not in seen and price > current_price:
                    seen.add(price)
                    levels.append(
                        LiquidityLevel(
                            symbol=symbol,
                            level_type="ROUND_NUMBER",
                            price=price,
                            strength=self._round_number_strength(price),
                            timeframe="any",
                            first_formed_at=now,
                            last_touched_at=now,
                        )
                    )
                price += increment

            # SSL — round numbers below current_price
            first_below = math.floor(current_price / increment) * increment
            price = float(first_below)
            while price >= lower_bound:
                if price not in seen and price < current_price:
                    seen.add(price)
                    levels.append(
                        LiquidityLevel(
                            symbol=symbol,
                            level_type="ROUND_NUMBER",
                            price=price,
                            strength=self._round_number_strength(price),
                            timeframe="any",
                            first_formed_at=now,
                            last_touched_at=now,
                        )
                    )
                price -= increment

        return levels

    @staticmethod
    def _round_number_strength(price: float) -> int:
        p = int(price)
        if p % 1000 == 0:
            return 5
        if p % 500 == 0:
            return 4
        if p % 100 == 0:
            return 3
        if p % 50 == 0:
            return 2
        return 1

    def _detect_multi_touch_levels(
        self, candles: list[Candle], tolerance_pct: float = 0.1
    ) -> list[LiquidityLevel]:
        if len(candles) < 3:
            return []

        symbol = candles[0].symbol
        timeframe = candles[0].timeframe
        now = datetime.now(_IST)
        tol_frac = tolerance_pct / 100.0

        # Collect all highs and lows with timestamps
        price_ts: list[tuple[float, datetime]] = []
        for c in candles:
            price_ts.append((c.high, c.timestamp))
            price_ts.append((c.low, c.timestamp))

        price_ts.sort(key=lambda x: x[0])
        used: list[bool] = [False] * len(price_ts)
        levels: list[LiquidityLevel] = []

        for i, (pi, ti) in enumerate(price_ts):
            if used[i]:
                continue
            group: list[tuple[float, datetime]] = [(pi, ti)]
            for j in range(i + 1, len(price_ts)):
                pj = price_ts[j][0]
                if pj - pi > pi * tol_frac:
                    break
                if not used[j]:
                    group.append(price_ts[j])
                    used[j] = True
            used[i] = True

            if len(group) >= 3:
                avg_price = sum(p for p, _ in group) / len(group)
                first_ts = min(ts for _, ts in group)
                last_ts = max(ts for _, ts in group)
                levels.append(
                    LiquidityLevel(
                        symbol=symbol,
                        level_type="BSL",   # caller splits by price vs current
                        price=avg_price,
                        strength=len(group),
                        timeframe=timeframe,
                        first_formed_at=first_ts,
                        last_touched_at=last_ts,
                    )
                )

        return levels

    # ── Bias calculation ───────────────────────────────────────────────────────

    def _calculate_liquidity_bias(
        self, symbol: str, current_price: float
    ) -> tuple[str, str]:
        bsl = self.get_nearest_bsl(symbol, current_price)
        ssl = self.get_nearest_ssl(symbol, current_price)

        # Check recently swept levels for post-sweep reversal bias
        all_lvls = self._levels.get(symbol, [])
        recently_swept_bsl = any(
            lvl.is_swept and lvl.price > current_price
            for lvl in all_lvls
        )
        recently_swept_ssl = any(
            lvl.is_swept and lvl.price < current_price
            for lvl in all_lvls
        )

        if recently_swept_bsl:
            return "BEARISH", f"BSL at {bsl.price:.2f} just swept — expect reversal" if bsl else "BSL swept — expect reversal"
        if recently_swept_ssl:
            return "BULLISH", f"SSL at {ssl.price:.2f} just swept — expect reversal" if ssl else "SSL swept — expect reversal"

        if bsl is None and ssl is None:
            return "NEUTRAL", "No active levels detected"
        if bsl is None:
            return "BEARISH", f"No BSL above — price drifting toward SSL at {ssl.price:.2f}"
        if ssl is None:
            return "BULLISH", f"No SSL below — price drifting toward BSL at {bsl.price:.2f}"

        dist_bsl = abs(bsl.price - current_price) / current_price * 100
        dist_ssl = abs(current_price - ssl.price) / current_price * 100

        if dist_ssl < dist_bsl * self._bias_ratio:
            return "BEARISH", f"Price within {dist_ssl:.1f}% of SSL at {ssl.price:.2f}"
        if dist_bsl < dist_ssl * self._bias_ratio:
            return "BULLISH", f"Price within {dist_bsl:.1f}% of BSL at {bsl.price:.2f}"

        return "NEUTRAL", f"Equidistant — BSL {dist_bsl:.1f}% | SSL {dist_ssl:.1f}%"

    # ── Level merging ──────────────────────────────────────────────────────────

    def _merge_levels(
        self, levels: list[LiquidityLevel], dedup_pct: float = 0.1
    ) -> list[LiquidityLevel]:
        if not levels:
            return []

        sorted_lvls = sorted(levels, key=lambda x: x.price)
        merged: list[LiquidityLevel] = [sorted_lvls[0]]

        for lvl in sorted_lvls[1:]:
            prev = merged[-1]
            if abs(lvl.price - prev.price) / prev.price < dedup_pct / 100.0:
                # Keep higher-strength level; update last_touched if needed
                if lvl.strength > prev.strength:
                    merged[-1] = lvl
                else:
                    if lvl.last_touched_at > prev.last_touched_at:
                        prev.last_touched_at = lvl.last_touched_at
            else:
                merged.append(lvl)

        return merged

    # ── CandleStore callback ────────────────────────────────────────────────────

    def _on_15min_candle(self, candle: Candle) -> None:
        try:
            self.update_levels(candle.symbol, candle.close)
        except Exception as exc:
            logger.error(
                "LiquidityPoolDetector: update_levels failed for %s: %s",
                candle.symbol, exc,
            )

    # ── DB helpers ──────────────────────────────────────────────────────────────

    def _ensure_db_tables(self) -> None:
        try:
            from src.database import queries as Q

            self._db.execute(Q.CREATE_LIQUIDITY_LEVELS)
            self._db.execute(Q.CREATE_LIQUIDITY_LEVELS_IDX)
            self._db.execute(Q.CREATE_LEVEL_TOUCHES)
            self._db.execute(Q.CREATE_LEVEL_TOUCHES_IDX)
        except Exception as exc:
            logger.error("LiquidityPoolDetector: failed to ensure DB tables: %s", exc)

    def _persist_levels(self, symbol: str, levels: list[LiquidityLevel]) -> None:
        if not levels:
            return
        try:
            from src.database import queries as Q

            today = datetime.now(_IST).date().isoformat()
            rows = [
                (
                    lvl.symbol,
                    lvl.level_type,
                    lvl.price,
                    round(lvl.price),           # price_key for dedup
                    lvl.strength,
                    lvl.timeframe,
                    lvl.first_formed_at.isoformat(),
                    lvl.last_touched_at.isoformat(),
                    int(lvl.is_swept),
                    lvl.swept_at.isoformat() if lvl.swept_at else None,
                    lvl.wick_beyond,
                    int(lvl.is_active),
                    today,
                )
                for lvl in levels
            ]
            self._db.execute_many(Q.UPSERT_LIQUIDITY_LEVEL, rows)
        except Exception as exc:
            logger.error(
                "LiquidityPoolDetector: DB persist failed for %s: %s", symbol, exc
            )
