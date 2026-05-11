"""
ImbalanceDetector — Fair Value Gap and Order Block detection.

FVGs are 3-candle price imbalances (voids) that price returns to fill.
Order Blocks are the last opposing candle before a strong displacement;
institutional re-entry zones when price retests them.

Integration: call get_imbalance_signal() before DecisionEngine emits a
signal to incorporate SMC imbalance context.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.analysis.liquidity.candle_store import CandleStore
from src.analysis.liquidity.models import Candle, FairValueGap, OrderBlock
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

_SCAN_WINDOW = 100   # candles to scan for new structures


class ImbalanceDetector:
    """
    Detects FVGs and Order Blocks, tracks their fill/mitigation state,
    and produces imbalance-based signal contributions for DecisionEngine.
    """

    def __init__(self, candle_store: CandleStore, db: DatabaseManager) -> None:
        self._cs = candle_store
        self._db = db

        cfg = settings.imbalance
        self._min_fvg_size_pct: float = cfg.min_fvg_size_pct
        self._max_fvg_age_hours: int = cfg.max_fvg_age_hours
        self._min_ob_displacement_pct: float = cfg.min_ob_displacement_pct
        self._max_ob_age_hours: int = cfg.max_ob_age_hours
        self._fvg_fill_threshold_pct: float = cfg.fvg_fill_threshold_pct
        self._ob_mitigation_threshold_pct: float = cfg.ob_mitigation_threshold_pct

        # symbol → list of active (unmitigated) FVGs
        self._active_fvgs: dict[str, list[FairValueGap]] = {}
        # symbol → list of active (unmitigated) OBs
        self._active_obs: dict[str, list[OrderBlock]] = {}

        # DB row IDs for update operations: keyed by object id()
        self._fvg_db_ids: dict[int, int] = {}
        self._ob_db_ids: dict[int, int] = {}

        self._ensure_db_tables()
        candle_store.register_candle_callback("5min", self._on_5min_candle)
        candle_store.register_candle_callback("15min", self._on_15min_candle)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_fvg_at_price(
        self,
        symbol: str,
        price: float,
        timeframe: str = "5min",
    ) -> Optional[FairValueGap]:
        """Return the active FVG that contains *price*, or None."""
        for fvg in self._active_fvgs.get(symbol, []):
            if fvg.timeframe != timeframe:
                continue
            if fvg.gap_bottom <= price <= fvg.gap_top:
                return fvg
        return None

    def get_nearest_fvg(
        self,
        symbol: str,
        current_price: float,
        direction: str = "both",
    ) -> Optional[FairValueGap]:
        """
        Return the nearest unfilled FVG in *direction* ('above'|'below'|'both').
        Used by DecisionEngine to set price targets.
        """
        candidates = [
            fvg for fvg in self._active_fvgs.get(symbol, [])
            if not fvg.is_mitigated
        ]

        if direction == "above":
            candidates = [f for f in candidates if f.gap_bottom > current_price]
        elif direction == "below":
            candidates = [f for f in candidates if f.gap_top < current_price]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda f: min(
                abs(f.gap_bottom - current_price),
                abs(f.gap_top - current_price),
            ),
        )

    def get_ob_at_price(
        self,
        symbol: str,
        price: float,
    ) -> Optional[OrderBlock]:
        """Return the active unmitigated OB that contains *price*, or None."""
        for ob in self._active_obs.get(symbol, []):
            if ob.ob_low <= price <= ob.ob_high:
                return ob
        return None

    def get_nearest_ob(
        self,
        symbol: str,
        current_price: float,
        ob_type: str = "BULLISH",
    ) -> Optional[OrderBlock]:
        """Return nearest unmitigated OB of *ob_type*. Used for entry zones / targets."""
        candidates = [
            ob for ob in self._active_obs.get(symbol, [])
            if ob.ob_type == ob_type and not ob.is_mitigated
        ]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda o: min(
                abs(o.ob_low - current_price),
                abs(o.ob_high - current_price),
            ),
        )

    def get_imbalance_signal(
        self,
        symbol: str,
        current_price: float,
        current_candle: Candle,
    ) -> dict:
        """
        Return a signal contribution dict for DecisionEngine:
            signal:       'BULLISH' | 'BEARISH' | 'NEUTRAL'
            strength:     0–100
            reason:       human-readable explanation
            target_price: nearest FVG midpoint or OB midpoint
            invalidation: price that invalidates this analysis
        """
        # Update fill/mitigation state for all active structures
        self._update_all_fvg_fills(symbol, current_candle)
        self._update_all_ob_mitigations(symbol, current_candle)

        active_fvgs = [f for f in self._active_fvgs.get(symbol, []) if not f.is_mitigated]
        active_obs = [o for o in self._active_obs.get(symbol, []) if not o.is_mitigated]

        fvg_at = self.get_fvg_at_price(symbol, current_price)
        ob_at = self.get_ob_at_price(symbol, current_price)

        # Scenario 7 — OB and FVG confluence at same price level
        if fvg_at is not None and ob_at is not None:
            if fvg_at.gap_type == "BULLISH" and ob_at.ob_type == "BULLISH":
                base = max(70.0, 75.0)
                strength = min(base + 15.0, 100.0)
                target = _fvg_target_above(symbol, current_price, active_fvgs)
                return {
                    "signal": "BULLISH",
                    "strength": strength,
                    "reason": (
                        f"Bullish OB+FVG confluence at {ob_at.ob_low:.2f}–{ob_at.ob_high:.2f}"
                    ),
                    "target_price": target or fvg_at.gap_midpoint,
                    "invalidation": ob_at.ob_low,
                }
            if fvg_at.gap_type == "BEARISH" and ob_at.ob_type == "BEARISH":
                base = max(70.0, 75.0)
                strength = min(base + 15.0, 100.0)
                target = _fvg_target_below(symbol, current_price, active_fvgs)
                return {
                    "signal": "BEARISH",
                    "strength": strength,
                    "reason": (
                        f"Bearish OB+FVG confluence at {ob_at.ob_low:.2f}–{ob_at.ob_high:.2f}"
                    ),
                    "target_price": target or fvg_at.gap_midpoint,
                    "invalidation": ob_at.ob_high,
                }

        # Scenario 1 — Price inside Bullish FVG
        if fvg_at is not None and fvg_at.gap_type == "BULLISH":
            # Count stacked bullish FVGs for strength boost (Scenario 6)
            stacked = sum(
                1 for f in active_fvgs
                if f.gap_type == "BULLISH" and f is not fvg_at
                and _overlaps_or_adjacent(f, fvg_at)
            )
            strength = min(70.0 + stacked * 15.0, 100.0)
            target = _fvg_target_above(symbol, current_price, active_fvgs)
            invalidation = fvg_at.gap_bottom - fvg_at.gap_size * 0.5
            return {
                "signal": "BULLISH",
                "strength": strength,
                "reason": f"Price filling bullish FVG ({fvg_at.gap_bottom:.2f}–{fvg_at.gap_top:.2f})",
                "target_price": target or fvg_at.gap_top,
                "invalidation": invalidation,
            }

        # Scenario 2 — Price inside Bearish FVG
        if fvg_at is not None and fvg_at.gap_type == "BEARISH":
            stacked = sum(
                1 for f in active_fvgs
                if f.gap_type == "BEARISH" and f is not fvg_at
                and _overlaps_or_adjacent(f, fvg_at)
            )
            strength = min(70.0 + stacked * 15.0, 100.0)
            target = _fvg_target_below(symbol, current_price, active_fvgs)
            invalidation = fvg_at.gap_top + fvg_at.gap_size * 0.5
            return {
                "signal": "BEARISH",
                "strength": strength,
                "reason": f"Price filling bearish FVG ({fvg_at.gap_bottom:.2f}–{fvg_at.gap_top:.2f})",
                "target_price": target or fvg_at.gap_bottom,
                "invalidation": invalidation,
            }

        # Scenario 3 — Price entering Bullish OB from above
        if ob_at is not None and ob_at.ob_type == "BULLISH":
            atr = _estimate_atr(
                self._cs.get_candles(symbol, current_candle.timeframe, 14)
            )
            target = ob_at.displacement_candle.high
            return {
                "signal": "BULLISH",
                "strength": 75.0,
                "reason": f"Price at bullish OB ({ob_at.ob_low:.2f}–{ob_at.ob_high:.2f})",
                "target_price": target,
                "invalidation": ob_at.ob_low - atr,
            }

        # Scenario 4 — Price entering Bearish OB from below
        if ob_at is not None and ob_at.ob_type == "BEARISH":
            atr = _estimate_atr(
                self._cs.get_candles(symbol, current_candle.timeframe, 14)
            )
            target = ob_at.displacement_candle.low
            return {
                "signal": "BEARISH",
                "strength": 75.0,
                "reason": f"Price at bearish OB ({ob_at.ob_low:.2f}–{ob_at.ob_high:.2f})",
                "target_price": target,
                "invalidation": ob_at.ob_high + atr,
            }

        # Scenario 5 — Price just mitigated an OB (close through it = breakout direction)
        recently_mitigated = self._get_recently_mitigated_ob(symbol, current_candle)
        if recently_mitigated is not None:
            if recently_mitigated.ob_type == "BULLISH":
                # Bullish OB mitigated by bearish close → bearish breakout
                return {
                    "signal": "BEARISH",
                    "strength": 60.0,
                    "reason": f"Bullish OB mitigated at {recently_mitigated.ob_low:.2f} — bearish break",
                    "target_price": recently_mitigated.ob_low * 0.995,
                    "invalidation": recently_mitigated.ob_high,
                }
            else:
                # Bearish OB mitigated by bullish close → bullish breakout
                return {
                    "signal": "BULLISH",
                    "strength": 60.0,
                    "reason": f"Bearish OB mitigated at {recently_mitigated.ob_high:.2f} — bullish break",
                    "target_price": recently_mitigated.ob_high * 1.005,
                    "invalidation": recently_mitigated.ob_low,
                }

        return {
            "signal": "NEUTRAL",
            "strength": 0.0,
            "reason": "No active FVG or OB at current price",
            "target_price": 0.0,
            "invalidation": 0.0,
        }

    # ── FVG detection ──────────────────────────────────────────────────────────

    def _detect_fvgs(
        self,
        candles: list[Candle],
        timeframe: str,
    ) -> list[FairValueGap]:
        """Scan the most recent _SCAN_WINDOW candles for 3-candle FVG patterns."""
        window = candles[-_SCAN_WINDOW:]
        detected: list[FairValueGap] = []

        for i in range(2, len(window)):
            c1, c2, c3 = window[i - 2], window[i - 1], window[i]
            symbol = c1.symbol

            # Bullish FVG: gap between top of c1 and bottom of c3
            if c3.low > c1.high:
                gap_bottom = c1.high
                gap_top = c3.low
                gap_size = gap_top - gap_bottom
                gap_size_pct = gap_size / c3.close * 100.0 if c3.close else 0.0
                if gap_size_pct >= self._min_fvg_size_pct:
                    detected.append(FairValueGap(
                        symbol=symbol,
                        timeframe=timeframe,
                        gap_type="BULLISH",
                        gap_top=gap_top,
                        gap_bottom=gap_bottom,
                        gap_midpoint=(gap_top + gap_bottom) / 2.0,
                        gap_size=gap_size,
                        gap_size_pct=gap_size_pct,
                        formed_at=c3.timestamp,
                        candle_1=c1,
                        candle_2=c2,
                        candle_3=c3,
                    ))

            # Bearish FVG: gap between bottom of c1 and top of c3
            elif c3.high < c1.low:
                gap_top = c1.low
                gap_bottom = c3.high
                gap_size = gap_top - gap_bottom
                gap_size_pct = gap_size / c3.close * 100.0 if c3.close else 0.0
                if gap_size_pct >= self._min_fvg_size_pct:
                    detected.append(FairValueGap(
                        symbol=symbol,
                        timeframe=timeframe,
                        gap_type="BEARISH",
                        gap_top=gap_top,
                        gap_bottom=gap_bottom,
                        gap_midpoint=(gap_top + gap_bottom) / 2.0,
                        gap_size=gap_size,
                        gap_size_pct=gap_size_pct,
                        formed_at=c3.timestamp,
                        candle_1=c1,
                        candle_2=c2,
                        candle_3=c3,
                    ))

        # Return sorted newest-first
        detected.sort(key=lambda f: f.formed_at, reverse=True)
        return detected

    def _update_fvg_fill_status(
        self,
        fvg: FairValueGap,
        current_candle: Candle,
    ) -> FairValueGap:
        """Update fill_pct and mitigation state of *fvg* given *current_candle*."""
        changed = False

        if fvg.gap_type == "BULLISH":
            if (
                current_candle.low <= fvg.gap_top
                and current_candle.low >= fvg.gap_bottom
            ):
                fill_pct = (fvg.gap_top - current_candle.low) / fvg.gap_size * 100.0
                new_fill = max(fvg.fill_pct, fill_pct)
                if new_fill != fvg.fill_pct:
                    fvg.fill_pct = new_fill
                    fvg.is_filled = fvg.fill_pct >= self._fvg_fill_threshold_pct
                    changed = True

            if current_candle.close < fvg.gap_bottom and not fvg.is_mitigated:
                fvg.is_mitigated = True
                fvg.fill_pct = 100.0
                fvg.is_filled = True
                changed = True

        else:  # BEARISH
            if (
                current_candle.high >= fvg.gap_bottom
                and current_candle.high <= fvg.gap_top
            ):
                fill_pct = (current_candle.high - fvg.gap_bottom) / fvg.gap_size * 100.0
                new_fill = max(fvg.fill_pct, fill_pct)
                if new_fill != fvg.fill_pct:
                    fvg.fill_pct = new_fill
                    fvg.is_filled = fvg.fill_pct >= self._fvg_fill_threshold_pct
                    changed = True

            if current_candle.close > fvg.gap_top and not fvg.is_mitigated:
                fvg.is_mitigated = True
                fvg.fill_pct = 100.0
                fvg.is_filled = True
                changed = True

        if changed:
            self._persist_fvg_update(fvg)

        return fvg

    def _update_all_fvg_fills(self, symbol: str, current_candle: Candle) -> None:
        for fvg in self._active_fvgs.get(symbol, []):
            if not fvg.is_mitigated:
                self._update_fvg_fill_status(fvg, current_candle)

    # ── OB detection ──────────────────────────────────────────────────────────

    def _detect_order_blocks(
        self,
        candles: list[Candle],
        timeframe: str,
    ) -> list[OrderBlock]:
        """Scan the most recent _SCAN_WINDOW candles for Order Blocks."""
        window = candles[-_SCAN_WINDOW:]
        if len(window) < 3:
            return []

        detected: list[OrderBlock] = []

        for i in range(1, len(window)):
            # Displacement starts at i; OB is the last candle before it (i-1)
            displacement_candles = _find_displacement(
                window, i, self._min_ob_displacement_pct
            )
            if displacement_candles is None:
                continue

            disp_candles, disp_pct, direction = displacement_candles
            ob_candle = window[i - 1]

            if direction == "UP":
                # Bullish displacement → OB must be the last BEARISH candle before it
                if ob_candle.is_bullish:
                    continue
                detected.append(OrderBlock(
                    symbol=ob_candle.symbol,
                    timeframe=timeframe,
                    ob_type="BULLISH",
                    ob_high=ob_candle.high,
                    ob_low=ob_candle.low,
                    ob_midpoint=(ob_candle.high + ob_candle.low) / 2.0,
                    origin_candle=ob_candle,
                    displacement_candle=disp_candles[-1],
                    displacement_size_pct=disp_pct,
                    formed_at=ob_candle.timestamp,
                ))
            else:  # DOWN
                # Bearish displacement → OB must be the last BULLISH candle before it
                if not ob_candle.is_bullish:
                    continue
                detected.append(OrderBlock(
                    symbol=ob_candle.symbol,
                    timeframe=timeframe,
                    ob_type="BEARISH",
                    ob_high=ob_candle.high,
                    ob_low=ob_candle.low,
                    ob_midpoint=(ob_candle.high + ob_candle.low) / 2.0,
                    origin_candle=ob_candle,
                    displacement_candle=disp_candles[-1],
                    displacement_size_pct=disp_pct,
                    formed_at=ob_candle.timestamp,
                ))

        detected.sort(key=lambda o: o.formed_at, reverse=True)
        return detected

    def _update_ob_mitigation(
        self,
        ob: OrderBlock,
        current_candle: Candle,
    ) -> OrderBlock:
        """Update touch_count and mitigation state of *ob* given *current_candle*."""
        if ob.is_mitigated:
            return ob

        price_in_zone = ob.ob_low <= current_candle.close <= ob.ob_high

        if ob.ob_type == "BULLISH":
            if current_candle.close < ob.ob_low:
                ob.is_mitigated = True
                ob.mitigated_at = current_candle.timestamp
                self._persist_ob_mitigated(ob)
            elif price_in_zone:
                ob.touch_count += 1
                self._persist_ob_touch(ob)
        else:  # BEARISH
            if current_candle.close > ob.ob_high:
                ob.is_mitigated = True
                ob.mitigated_at = current_candle.timestamp
                self._persist_ob_mitigated(ob)
            elif price_in_zone:
                ob.touch_count += 1
                self._persist_ob_touch(ob)

        return ob

    def _update_all_ob_mitigations(self, symbol: str, current_candle: Candle) -> None:
        for ob in self._active_obs.get(symbol, []):
            if not ob.is_mitigated:
                self._update_ob_mitigation(ob, current_candle)

    def _get_recently_mitigated_ob(
        self, symbol: str, current_candle: Candle
    ) -> Optional[OrderBlock]:
        """Return an OB that was mitigated on this exact candle, if any."""
        for ob in self._active_obs.get(symbol, []):
            if ob.is_mitigated and ob.mitigated_at == current_candle.timestamp:
                return ob
        return None

    # ── CandleStore callbacks ──────────────────────────────────────────────────

    def _on_5min_candle(self, candle: Candle) -> None:
        self._process_candle(candle, "5min")

    def _on_15min_candle(self, candle: Candle) -> None:
        self._process_candle(candle, "15min")

    def _process_candle(self, candle: Candle, timeframe: str) -> None:
        symbol = candle.symbol
        try:
            candles = self._cs.get_candles(symbol, timeframe, _SCAN_WINDOW)
            if len(candles) < 3:
                return

            # Detect new structures
            new_fvgs = self._detect_fvgs(candles, timeframe)
            new_obs = self._detect_order_blocks(candles, timeframe)

            # Merge into active lists (skip duplicates by formed_at)
            self._merge_fvgs(symbol, new_fvgs)
            self._merge_obs(symbol, new_obs)

            # Expire old structures
            self._expire_old_structures(symbol)

            # Update fill/mitigation on the new candle
            self._update_all_fvg_fills(symbol, candle)
            self._update_all_ob_mitigations(symbol, candle)

        except Exception as exc:
            logger.error("ImbalanceDetector._process_candle %s/%s: %s", symbol, timeframe, exc)

    # ── In-memory merge helpers ────────────────────────────────────────────────

    def _merge_fvgs(self, symbol: str, new_fvgs: list[FairValueGap]) -> None:
        existing = self._active_fvgs.setdefault(symbol, [])
        existing_keys = {(f.timeframe, f.formed_at, f.gap_type) for f in existing}
        for fvg in new_fvgs:
            key = (fvg.timeframe, fvg.formed_at, fvg.gap_type)
            if key not in existing_keys:
                existing.append(fvg)
                existing_keys.add(key)
                db_id = self._persist_fvg(fvg)
                if db_id is not None:
                    self._fvg_db_ids[id(fvg)] = db_id

    def _merge_obs(self, symbol: str, new_obs: list[OrderBlock]) -> None:
        existing = self._active_obs.setdefault(symbol, [])
        existing_keys = {(o.timeframe, o.formed_at, o.ob_type) for o in existing}
        for ob in new_obs:
            key = (ob.timeframe, ob.formed_at, ob.ob_type)
            if key not in existing_keys:
                existing.append(ob)
                existing_keys.add(key)
                db_id = self._persist_ob(ob)
                if db_id is not None:
                    self._ob_db_ids[id(ob)] = db_id

    def _expire_old_structures(self, symbol: str) -> None:
        now = datetime.now(_IST)
        fvg_cutoff = now - timedelta(hours=self._max_fvg_age_hours)
        ob_cutoff = now - timedelta(hours=self._max_ob_age_hours)

        self._active_fvgs[symbol] = [
            f for f in self._active_fvgs.get(symbol, [])
            if f.formed_at >= fvg_cutoff and not f.is_mitigated
        ]
        self._active_obs[symbol] = [
            o for o in self._active_obs.get(symbol, [])
            if o.formed_at >= ob_cutoff and not o.is_mitigated
        ]

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _ensure_db_tables(self) -> None:
        try:
            from src.database import queries as Q
            self._db.execute(Q.CREATE_FAIR_VALUE_GAPS)
            self._db.execute(Q.CREATE_FAIR_VALUE_GAPS_IDX)
            self._db.execute(Q.CREATE_ORDER_BLOCKS)
            self._db.execute(Q.CREATE_ORDER_BLOCKS_IDX)
        except Exception as exc:
            logger.error("ImbalanceDetector: failed to ensure DB tables: %s", exc)

    def _persist_fvg(self, fvg: FairValueGap) -> Optional[int]:
        try:
            from src.database import queries as Q
            today = datetime.now(_IST).date().isoformat()
            cur = self._db.execute(
                Q.INSERT_FAIR_VALUE_GAP,
                (
                    fvg.symbol,
                    fvg.timeframe,
                    fvg.gap_type,
                    fvg.gap_top,
                    fvg.gap_bottom,
                    fvg.gap_midpoint,
                    fvg.gap_size_pct,
                    fvg.formed_at.isoformat(),
                    int(fvg.is_filled),
                    fvg.fill_pct,
                    int(fvg.is_mitigated),
                    today,
                ),
            )
            return cur.lastrowid
        except Exception as exc:
            logger.error("ImbalanceDetector: FVG DB persist failed: %s", exc)
            return None

    def _persist_fvg_update(self, fvg: FairValueGap) -> None:
        db_id = self._fvg_db_ids.get(id(fvg))
        if db_id is None:
            return
        try:
            from src.database import queries as Q
            self._db.execute(
                Q.UPDATE_FAIR_VALUE_GAP_FILL,
                (fvg.fill_pct, int(fvg.is_filled), int(fvg.is_mitigated), db_id),
            )
        except Exception as exc:
            logger.error("ImbalanceDetector: FVG update failed: %s", exc)

    def _persist_ob(self, ob: OrderBlock) -> Optional[int]:
        try:
            from src.database import queries as Q
            today = datetime.now(_IST).date().isoformat()
            cur = self._db.execute(
                Q.INSERT_ORDER_BLOCK,
                (
                    ob.symbol,
                    ob.timeframe,
                    ob.ob_type,
                    ob.ob_high,
                    ob.ob_low,
                    ob.displacement_size_pct,
                    ob.strength,
                    ob.formed_at.isoformat(),
                    ob.touch_count,
                    int(ob.is_mitigated),
                    ob.mitigated_at.isoformat() if ob.mitigated_at else None,
                    today,
                ),
            )
            return cur.lastrowid
        except Exception as exc:
            logger.error("ImbalanceDetector: OB DB persist failed: %s", exc)
            return None

    def _persist_ob_mitigated(self, ob: OrderBlock) -> None:
        db_id = self._ob_db_ids.get(id(ob))
        if db_id is None:
            return
        try:
            from src.database import queries as Q
            mitigated_at = ob.mitigated_at.isoformat() if ob.mitigated_at else None
            self._db.execute(Q.UPDATE_ORDER_BLOCK_MITIGATED, (mitigated_at, db_id))
        except Exception as exc:
            logger.error("ImbalanceDetector: OB mitigated update failed: %s", exc)

    def _persist_ob_touch(self, ob: OrderBlock) -> None:
        db_id = self._ob_db_ids.get(id(ob))
        if db_id is None:
            return
        try:
            from src.database import queries as Q
            self._db.execute(Q.UPDATE_ORDER_BLOCK_TOUCH, (db_id,))
        except Exception as exc:
            logger.error("ImbalanceDetector: OB touch update failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _find_displacement(
    candles: list[Candle],
    start_idx: int,
    min_displacement_pct: float,
) -> Optional[tuple[list[Candle], float, str]]:
    """
    Look for a strong displacement starting at *start_idx*.

    Returns (displacement_candles, displacement_pct, direction) or None.
    Direction: 'UP' | 'DOWN'.

    Strong displacement: >= 2 consecutive candles same direction,
    each body >= 60% of its total range, combined move >= min_displacement_pct.
    """
    if start_idx >= len(candles):
        return None

    # Try bullish displacement
    bull_candles: list[Candle] = []
    for i in range(start_idx, min(start_idx + 3, len(candles))):
        c = candles[i]
        if not c.is_bullish:
            break
        if c.total_range > 0 and c.body_size / c.total_range < 0.60:
            break
        bull_candles.append(c)

    if len(bull_candles) >= 2:
        start_price = bull_candles[0].open
        end_price = bull_candles[-1].close
        if start_price > 0:
            pct = (end_price - start_price) / start_price * 100.0
            if pct >= min_displacement_pct:
                return (bull_candles, pct, "UP")

    # Try bearish displacement
    bear_candles: list[Candle] = []
    for i in range(start_idx, min(start_idx + 3, len(candles))):
        c = candles[i]
        if c.is_bullish:
            break
        if c.total_range > 0 and c.body_size / c.total_range < 0.60:
            break
        bear_candles.append(c)

    if len(bear_candles) >= 2:
        start_price = bear_candles[0].open
        end_price = bear_candles[-1].close
        if start_price > 0:
            pct = (start_price - end_price) / start_price * 100.0
            if pct >= min_displacement_pct:
                return (bear_candles, pct, "DOWN")

    return None


def _estimate_atr(candles: list[Candle]) -> float:
    """Estimate Average True Range from last 14 candles."""
    if not candles:
        return 0.0
    trs = [c.total_range for c in candles[-14:]]
    return sum(trs) / len(trs) if trs else 0.0


def _overlaps_or_adjacent(a: FairValueGap, b: FairValueGap) -> bool:
    """True if two FVGs overlap or are directly adjacent (stacked)."""
    return a.gap_bottom <= b.gap_top and b.gap_bottom <= a.gap_top


def _fvg_target_above(
    symbol: str,
    current_price: float,
    active_fvgs: list[FairValueGap],
) -> Optional[float]:
    """Return midpoint of the nearest bearish FVG above current price."""
    candidates = [
        f for f in active_fvgs
        if f.gap_type == "BEARISH" and f.gap_bottom > current_price
    ]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda f: f.gap_bottom - current_price)
    return nearest.gap_midpoint


def _fvg_target_below(
    symbol: str,
    current_price: float,
    active_fvgs: list[FairValueGap],
) -> Optional[float]:
    """Return midpoint of the nearest bullish FVG below current price."""
    candidates = [
        f for f in active_fvgs
        if f.gap_type == "BULLISH" and f.gap_top < current_price
    ]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda f: current_price - f.gap_top)
    return nearest.gap_midpoint
