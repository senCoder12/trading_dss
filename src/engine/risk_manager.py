"""
Risk Manager — Phase 5.3 of the Trading Decision Support System.

Takes a raw TradingSignal from the SignalGenerator and makes it EXECUTABLE:
  - Validates and refines stop-loss and target levels
  - Sizes the position (lots) based on capital at risk
  - Selects the optimal option strike from the chain
  - Enforces daily loss limits and open position caps
  - Tracks open positions with trailing stop logic
  - Computes daily P&L and portfolio summary

Design philosophy: CONSERVATIVE — protect capital before maximising profit.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.data.index_registry import get_registry
from src.data.options_chain import OptionsChainData
from src.database.db_manager import DatabaseManager
from src.engine.option_contract import (
    ContractQuote,
    SelectionParams,
    estimate_premium,
    minutes_until,
    premium_at,
    select_contract,
)
from src.engine.option_costs import round_trip_cost
from src.engine.signal_generator import TradingSignal

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RiskConfig:
    """All configurable risk parameters — zero-cost conservative defaults."""

    # Capital & risk limits
    total_capital: float = 100_000.0          # ₹ total trading capital
    max_risk_per_trade_pct: float = 2.0        # % of capital at risk per trade (premium terms)
    # Daily limit must exceed the per-trade limit, otherwise one full-size
    # loss ends the day and the second trade can never be sized.
    max_risk_per_day_pct: float = 4.0          # % daily loss limit incl. open risk
    max_open_positions: int = 3                # simultaneous open positions
    max_positions_per_index: int = 1           # positions in same index
    max_lots_per_trade: int = 5                # absolute safety cap
    # Indian index options are highly correlated: NIFTY/BANKNIFTY/FINNIFTY
    # calls together are one bet, not three.
    max_same_direction_positions: int = 2
    # Premium outlay cap: an option buyer can lose the whole premium.
    max_premium_outlay_pct: float = 50.0       # % of capital across open positions

    # Risk-reward (measured in PREMIUM, net of spread and costs)
    min_risk_reward_ratio: float = 1.3         # reject if RR below this
    default_risk_reward: float = 1.5           # target RR

    # Execution costs — statutory charges come from src/engine/option_costs.
    slippage_points: float = 2.0               # (legacy spot path only)
    transaction_cost_per_lot: float = 120.0    # (legacy spot path only)

    # Option-buyer premium rules
    max_premium_sl_pct: float = 0.35           # never risk more than 35% of premium
    expected_hold_minutes: int = 60            # theta horizon for target/SL premiums
    target_abs_delta_high: float = 0.50        # strike choice by computed delta
    target_abs_delta_medium: float = 0.40
    target_abs_delta_hero_zero: float = 0.20
    min_option_premium: float = 5.0            # ₹ — below this, tick/spread dominate
    max_spread_pct: float = 0.05               # of mid, when bid/ask are known
    assumed_half_spread_pct: float = 0.005     # of mid, when bid/ask unknown
    hero_zero_risk_pct: float = 0.5            # % of capital — lottery-ticket sizing
    hero_zero_premium_sl_pct: float = 0.50
    hero_zero_target_multiple: float = 2.5     # premium target = 2.5× entry

    # Kelly sizing is OFF until enough premium-based trades exist; history
    # recorded before the premium model is in index points and unusable.
    use_kelly: bool = False
    kelly_min_trades: int = 30
    kelly_fraction: float = 0.25

    # Discipline controls
    loss_cooldown_minutes: int = 30            # per index, after a losing exit
    max_consecutive_losses: int = 3            # halt new entries for the day

    # Option-buyer time rules (IST wall clock)
    no_entry_before: str = "09:30"             # opening 15 min = noise + wide spreads
    force_exit_time: str = "15:15"             # ahead of broker MIS square-off
    time_stop_minutes: int = 60                # exit if not working by then …
    time_stop_min_r: float = 0.3               # … i.e. below +0.3R in premium

    # Stop-loss validation thresholds
    # Noise floor for the spot stop.  Must match the calibrated trade
    # definition (signal_calibration.DEFAULT_MIN_STOP_PCT) or the risk
    # manager widens the stop into a different, uncalibrated trade.
    min_sl_pct: float = 0.001                  # 0.1% ≈ one 5m ATR on NIFTY
    max_sl_pct: float = 0.030                  # 3.0% — too much risk
    sl_sr_buffer_pct: float = 0.002            # 0.2% buffer beyond S/R

    # Confidence-based position sizing
    high_confidence_size_pct: float = 100.0    # % of max risk
    medium_confidence_size_pct: float = 60.0
    low_confidence_size_pct: float = 30.0

    # Market timing rules
    no_entry_after_minutes_before_close: int = 30  # no new entries
    force_exit_minutes_before_close: int = 5       # must exit intraday


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RefinedSignal:
    """
    Executable version of a TradingSignal.

    Carries all original signal fields plus execution details produced by the
    Risk Manager: refined price levels, lot count, option strike, cost
    breakdown, and full validation results.
    """

    # ── Passed through from TradingSignal ──────────────────────────────────
    signal_id: str
    index_id: str
    generated_at: datetime
    signal_type: str          # "BUY_CALL" / "BUY_PUT" / "NO_TRADE"
    confidence_level: str     # "HIGH" / "MEDIUM" / "LOW"
    confidence_score: float
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    regime: str
    weighted_score: float
    vote_breakdown: dict
    risk_level: str
    position_size_modifier: float
    suggested_lot_count: int
    estimated_max_loss: float
    estimated_max_profit: float
    reasoning: str
    warnings: list[str]
    outcome: Optional[str]
    actual_exit_price: Optional[float]
    actual_pnl: Optional[float]
    closed_at: Optional[datetime]
    data_completeness: float
    signals_generated_today: int

    # ── Refined levels ─────────────────────────────────────────────────────
    refined_entry: float = 0.0
    refined_target: float = 0.0
    refined_stop_loss: float = 0.0

    # ── Position sizing ────────────────────────────────────────────────────
    lots: int = 0
    total_margin_required: float = 0.0
    max_loss_amount: float = 0.0        # ₹ if SL hit
    max_profit_amount: float = 0.0      # ₹ if target hit
    transaction_cost_total: float = 0.0
    breakeven_move: float = 0.0         # points to cover costs

    # ── Option selection ───────────────────────────────────────────────────
    recommended_strike: Optional[float] = None
    recommended_expiry: Optional[str] = None
    option_premium: Optional[float] = None
    option_greeks: Optional[dict] = None

    # ── Risk metrics ───────────────────────────────────────────────────────
    risk_amount: float = 0.0
    risk_pct_of_capital: float = 0.0
    daily_loss_remaining: float = 0.0

    # ── Validation ─────────────────────────────────────────────────────────
    is_valid: bool = True
    rejection_reasons: list[str] = field(default_factory=list)
    adjustments_made: list[str] = field(default_factory=list)

    # ── Execution instructions ─────────────────────────────────────────────
    execution_type: str = "LIMIT"       # "MARKET" / "LIMIT"
    limit_price: Optional[float] = None
    validity: str = "DAY"

    # ── Option contract (premium-native risk) ──────────────────────────────
    is_hero_zero: bool = False
    option_type: Optional[str] = None       # "CE" / "PE"
    lot_size: int = 0
    quantity: int = 0                       # lots × lot_size
    entry_premium: float = 0.0              # ₹/unit actually paid (mid + half spread)
    premium_mid: float = 0.0                # ₹/unit fair premium at entry
    half_spread: float = 0.0                # ₹/unit, each side
    spread_known: bool = False
    premium_stop: float = 0.0               # exit when mark ≤ this
    premium_target: float = 0.0             # exit when mark ≥ this
    premium_rr: float = 0.0                 # net reward / net risk in ₹
    option_iv: float = 0.0                  # decimal
    days_to_expiry: float = 0.0
    risk_per_lot: float = 0.0               # ₹ incl. spread + statutory costs
    premium_outlay: float = 0.0             # ₹ paid for the position


@dataclass
class _OpenPosition:
    """Internal state for a single tracked open position."""

    db_id: int
    signal_id: str
    index_id: str
    signal_type: str          # "BUY_CALL" / "BUY_PUT"
    entry_price: float        # spot at entry
    current_sl: float         # spot stop trigger (ratchets with the premium stop)
    target_price: float       # spot target trigger
    lots: int
    lot_size: int
    entry_time: datetime
    confidence_level: str
    is_hero_zero: bool = False
    # Option contract — None for legacy spot-only positions
    option_type: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[date] = None
    entry_premium: float = 0.0     # paid per unit
    premium_mid: float = 0.0       # fair premium at entry (model anchor)
    half_spread: float = 0.0
    premium_stop: float = 0.0      # ratchets up only
    initial_premium_stop: float = 0.0
    premium_target: float = 0.0
    entry_iv: float = 0.0
    max_loss_amount: float = 0.0
    last_premium: float = 0.0
    exchange: str = "NSE"

    @property
    def quantity(self) -> int:
        return self.lots * self.lot_size

    @property
    def is_option(self) -> bool:
        return self.option_type is not None and self.strike is not None and self.expiry is not None


@dataclass
class PositionUpdate:
    """Result of checking whether an open position should be exited."""

    signal_id: str
    action: str               # HOLD / EXIT_TARGET / EXIT_SL / EXIT_TRAILING_SL / EXIT_TIME / EXIT_TIME_STOP
    current_pnl: float        # ₹ mark-to-market (net of estimated exit costs for options)
    current_pnl_pct: float    # % of total capital
    time_in_trade_minutes: int
    mark_premium: Optional[float] = None   # option mark used for the decision
    premium_source: str = ""                # "quote" / "model"


@dataclass
class DailyPnL:
    """Daily trading performance summary."""

    date: date
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float      # % of capital
    largest_win: float
    largest_loss: float
    avg_win: float
    avg_loss: float
    risk_used_pct: float      # % of daily risk limit consumed
    remaining_risk: float     # ₹ of daily limit remaining
    open_positions: int


@dataclass
class PortfolioSummary:
    """Point-in-time portfolio snapshot."""

    total_capital: float
    current_exposure: float   # ₹ in open positions (margin)
    available_capital: float
    open_positions: list[dict]  # per-position dicts with key metrics
    today_pnl: float
    this_week_pnl: float
    this_month_pnl: float
    total_trades_today: int
    daily_limit_remaining_pct: float


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


class RiskManager:
    """
    Converts a raw TradingSignal into an executable RefinedSignal.

    Also manages the lifecycle of open positions: tracking, trailing
    stop updates, forced exits, and daily P&L accounting.

    Parameters
    ----------
    db:
        Live DatabaseManager instance (shared with the rest of the app).
    config:
        Optional :class:`RiskConfig` override; defaults are used when omitted.
    """

    def __init__(
        self,
        db: DatabaseManager,
        config: Optional[RiskConfig] = None,
    ) -> None:
        self._db = db
        self.config = config or RiskConfig()
        # The 60 s monitor job and the 300 s decision job run on different
        # scheduler threads; every read-check-write of position state holds
        # this lock (the position caps were a check-then-act race).
        self._lock = threading.RLock()

        # signal_id → _OpenPosition for instant lookup during the session
        self._open_positions: dict[str, _OpenPosition] = {}
        # Today's realised P&L (₹, negative = loss)
        self._daily_realised_pnl: float = 0.0
        # Date for which the daily cache was last refreshed
        self._cache_date: Optional[date] = None

        self._load_open_positions()
        self._refresh_daily_pnl()
        self._cleanup_on_startup()

    # ------------------------------------------------------------------
    # Startup cleanup
    # ------------------------------------------------------------------

    def _cleanup_on_startup(self) -> None:
        """Resolve orphaned open positions from previous sessions.

        On restart, any trading_signals with outcome='OPEN' from before today's
        market open are stale. We can't reliably track them anymore, so we resolve
        them based on current market state or mark them expired.
        """
        try:
            from src.utils.date_utils import get_ist_now

            now = get_ist_now()
            today_open = now.replace(hour=9, minute=15, second=0, microsecond=0)

            orphaned = self._db.fetch_all(
                """SELECT id, index_id, signal_type, entry_price, target_price, stop_loss, generated_at
                   FROM trading_signals
                   WHERE outcome = 'OPEN' AND generated_at < ?""",
                (today_open.isoformat(),)
            )

            if not orphaned:
                return

            resolved_count = 0
            for signal in orphaned:
                # An option position left open across sessions cannot be
                # valued from today's index price (theta, IV and the broker's
                # own square-off all intervene).  Writing a fabricated
                # index-point P&L as WIN/LOSS corrupted every statistic, so
                # orphans are marked EXPIRED with no P&L and flagged for
                # manual reconciliation against the broker contract note.
                resolution = "EXPIRED"
                exit_price = None
                pnl = None
                logger.warning(
                    "Startup cleanup: signal #%s (%s %s) was still OPEN from a previous "
                    "session — marked EXPIRED with no P&L. Reconcile with the broker.",
                    signal['id'], signal['index_id'], signal['signal_type'],
                )

                self._db.execute(
                    """UPDATE trading_signals
                       SET outcome = ?, actual_exit_price = ?, actual_pnl = ?, closed_at = ?
                       WHERE id = ?""",
                    (resolution, exit_price, pnl, now.isoformat(), signal['id'])
                )
                resolved_count += 1
                logger.info(
                    f"Startup cleanup: resolved signal #{signal['id']} "
                    f"({signal['index_id']} {signal['signal_type']}) as {resolution}"
                )

            if resolved_count > 0:
                logger.info(f"Startup cleanup: resolved {resolved_count} orphaned position(s)")

        except Exception as e:
            logger.error(f"Startup cleanup failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Public API — signal refinement
    # ------------------------------------------------------------------

    def validate_and_refine_signal(
        self,
        signal: TradingSignal,
        current_chain: Optional[OptionsChainData] = None,
        _now: Optional[datetime] = None,  # injectable for backtesting/tests
        next_chain: Optional[OptionsChainData] = None,
    ) -> RefinedSignal:
        """
        Turn a raw signal into an executable OPTION trade, or reject it.

        Risk per lot is a premium quantity, so the contract is chosen before
        sizing.  Steps, each of which can reject:

          0. entry gates      — time window, loss streak, per-index cooldown,
                                position caps (total / per index / same
                                direction), daily limit incl. open risk
          1. spot SL trigger  — side, noise floor, max width, OI support
          2. spot target      — OI cap only when the wall is on the right side
          3. contract         — expiry (next expiry on expiry day unless hero
                                zero), strike by computed delta, liquidity
          4. premium levels   — SL/target premiums re-priced at the spot
                                triggers after the expected holding time
                                (theta), minus spread and statutory costs;
                                reject if premium R:R < minimum (the edge check)
          5. sizing           — lots = ₹ budget / ₹ risk per lot; a trade that
                                cannot fit one lot is rejected, never forced;
                                capped by the daily limit and premium outlay

        Parameters
        ----------
        signal:
            Raw signal from SignalGenerator (spot entry/SL/target).
        current_chain:
            Options chain for the nearest tradeable expiry.
        _now:
            Decision time (default: wall clock).
        next_chain:
            Chain for the following expiry — used on expiry day.
        """
        with self._lock:
            return self._validate_and_refine(signal, current_chain, _now, next_chain)

    def _validate_and_refine(
        self,
        signal: TradingSignal,
        chain: Optional[OptionsChainData],
        _now: Optional[datetime],
        next_chain: Optional[OptionsChainData],
    ) -> RefinedSignal:
        self._maybe_refresh_daily_cache()
        refined = self._init_refined(signal)

        if signal.signal_type == "NO_TRADE":
            return self._reject(refined, "Signal type is NO_TRADE")

        now = _now if _now is not None else datetime.now(tz=_IST)
        if now.tzinfo is None:
            now = now.replace(tzinfo=_IST)

        # --- Step 0: entry gates ------------------------------------------
        gate = self._entry_gate_reason(refined, now)
        if gate:
            return self._reject(refined, gate)

        # --- Steps 1-2: spot trigger levels -------------------------------
        self._step_validate_sl(refined, chain)
        if not refined.is_valid:
            return refined
        self._step_validate_target(refined, chain)
        if not refined.is_valid:
            return refined

        # --- Step 3: the actual contract ----------------------------------
        quote, why = self._step_select_contract(refined, chain, next_chain, now)
        if quote is None:
            return self._reject(refined, f"No executable option contract: {why}")

        lot_size = self._contract_lot_size(refined.index_id, quote.expiry)
        if not lot_size:
            return self._reject(
                refined, f"{refined.index_id} has no F&O lot size — cannot size position",
            )

        # --- Step 4: premium levels + edge check --------------------------
        if not self._step_premium_levels(refined, quote, now, lot_size):
            return refined

        # --- Step 5: sizing, daily limit, outlay --------------------------
        if not self._step_size_premium(refined, lot_size):
            return refined

        refined.execution_type = "LIMIT"
        refined.limit_price = round(refined.entry_premium, 2)  # an OPTION price
        refined.validity = "DAY"
        refined.is_valid = not refined.rejection_reasons
        return refined

    # ------------------------------------------------------------------
    # Public API — position lifecycle
    # ------------------------------------------------------------------

    def track_open_position(self, signal: RefinedSignal) -> None:
        """
        Register a refined (risk-approved) signal as an open position.

        Persists the full contract and premium levels into the DB row so a
        restart restores the position exactly (strike, expiry, premium stop).
        """
        if not signal.is_valid:
            logger.warning(
                "track_open_position called with invalid signal %s — skipped",
                signal.signal_id,
            )
            return

        with self._lock:
            lot_size = signal.lot_size or self._get_lot_size(signal.index_id) or 1
            db_id = self._find_signal_db_id(signal)

            expiry = None
            if signal.recommended_expiry:
                try:
                    expiry = date.fromisoformat(str(signal.recommended_expiry)[:10])
                except ValueError:
                    expiry = None

            pos = _OpenPosition(
                db_id=db_id or -1,
                signal_id=signal.signal_id,
                index_id=signal.index_id,
                signal_type=signal.signal_type,
                entry_price=signal.refined_entry,
                current_sl=signal.refined_stop_loss,
                target_price=signal.refined_target,
                lots=signal.lots,
                lot_size=lot_size,
                entry_time=signal.generated_at,
                confidence_level=signal.confidence_level,
                is_hero_zero=signal.is_hero_zero,
                option_type=signal.option_type,
                strike=signal.recommended_strike if signal.option_type else None,
                expiry=expiry if signal.option_type else None,
                entry_premium=signal.entry_premium,
                premium_mid=signal.premium_mid,
                half_spread=signal.half_spread,
                premium_stop=signal.premium_stop,
                initial_premium_stop=signal.premium_stop,
                premium_target=signal.premium_target,
                entry_iv=signal.option_iv,
                max_loss_amount=signal.max_loss_amount,
                last_premium=signal.premium_mid,
                exchange=self._exchange_for(signal.index_id),
            )
            self._open_positions[signal.signal_id] = pos
            if db_id is not None:
                self._persist_risk_details(db_id, signal, pos)

        if pos.is_option:
            logger.info(
                "OPEN  %-12s %s %s %.0f%s  qty=%d  premium=%.2f  SL=%.2f  TGT=%.2f  "
                "(spot %.2f, SL %.2f, TGT %.2f)",
                signal.index_id, signal.signal_type, pos.expiry, pos.strike, pos.option_type,
                pos.quantity, pos.entry_premium, pos.premium_stop, pos.premium_target,
                signal.refined_entry, signal.refined_stop_loss, signal.refined_target,
            )
        else:
            logger.info(
                "OPEN  %-12s %s  Entry=%.2f  SL=%.2f  Target=%.2f  Lots=%d (spot-only)",
                signal.index_id, signal.signal_type,
                signal.refined_entry, signal.refined_stop_loss,
                signal.refined_target, signal.lots,
            )

    def update_position(
        self,
        signal_id: str,
        current_price: float,
        _now: Optional[datetime] = None,  # injectable for tests
        option_ltp: Optional[float] = None,
    ) -> PositionUpdate:
        """
        Check exit conditions for an open position and ratchet its stop.

        *current_price* is the underlying spot.  For option positions the
        decision is made on the option mark: *option_ltp* when a live quote
        is available, else a model re-price from spot (flagged ``model``).
        """
        with self._lock:
            pos = self._open_positions.get(signal_id)
            if pos is None:
                return PositionUpdate(
                    signal_id=signal_id, action="HOLD", current_pnl=0.0,
                    current_pnl_pct=0.0, time_in_trade_minutes=0,
                )
            now = _now if _now is not None else datetime.now(tz=_IST)
            if now.tzinfo is None:
                now = now.replace(tzinfo=_IST)
            if pos.is_option:
                return self._update_option_position(pos, current_price, now, option_ltp)
            return self._update_spot_position(pos, current_price, now)

    def _update_option_position(
        self,
        pos: _OpenPosition,
        spot: float,
        now: datetime,
        option_ltp: Optional[float],
    ) -> PositionUpdate:
        cfg = self.config
        minutes = int((now - pos.entry_time).total_seconds() / 60)

        if option_ltp is not None and option_ltp > 0:
            mark, source = float(option_ltp), "quote"
        else:
            mark = estimate_premium(
                pos.option_type, pos.strike, pos.expiry, pos.entry_iv or 0.15,
                spot, now, pos.premium_mid or pos.entry_premium, pos.entry_price, pos.entry_time,
            )
            source = "model"
        pos.last_premium = mark

        exit_px = max(mark - pos.half_spread, 0.05)
        costs = round_trip_cost(pos.entry_premium, exit_px, pos.quantity, pos.exchange).total
        pnl = (exit_px - pos.entry_premium) * pos.quantity - costs

        def result(action: str) -> PositionUpdate:
            return PositionUpdate(
                signal_id=pos.signal_id,
                action=action,
                current_pnl=round(pnl, 2),
                current_pnl_pct=round(pnl / cfg.total_capital * 100, 4),
                time_in_trade_minutes=minutes,
                mark_premium=round(mark, 2),
                premium_source=source,
            )

        is_call = pos.signal_type == "BUY_CALL"

        # Time exits first: an intraday option must not be carried.
        if pos.is_hero_zero and minutes_until(now, "14:55") <= 0:
            return result("EXIT_TIME")
        if minutes_until(now, cfg.force_exit_time) <= 0:
            return result("EXIT_TIME")

        # Stop (premium or spot trigger) before target — conservative.
        spot_sl_hit = spot <= pos.current_sl if is_call else spot >= pos.current_sl
        if mark <= pos.premium_stop or spot_sl_hit:
            trailing = pos.premium_stop >= pos.entry_premium
            return result("EXIT_TRAILING_SL" if trailing else "EXIT_SL")

        spot_tgt_hit = spot >= pos.target_price if is_call else spot <= pos.target_price
        if mark >= pos.premium_target or spot_tgt_hit:
            return result("EXIT_TARGET")

        # Time stop: a long option that is not working is bleeding theta.
        risk_unit = pos.entry_premium - pos.initial_premium_stop
        if (
            not pos.is_hero_zero
            and minutes >= cfg.time_stop_minutes
            and risk_unit > 0
            and (mark - pos.entry_premium) / risk_unit < cfg.time_stop_min_r
        ):
            return result("EXIT_TIME_STOP")

        if self._trail_premium_stop(pos, mark):
            self._persist_position_state(pos)
        return result("HOLD")

    def _update_spot_position(
        self,
        pos: _OpenPosition,
        current_price: float,
        now: datetime,
    ) -> PositionUpdate:
        """Legacy spot-only position (no contract recorded): delta-1 P&L."""
        signal_id = pos.signal_id
        time_in_trade = int((now - pos.entry_time).total_seconds() / 60)

        is_call = pos.signal_type == "BUY_CALL"
        sign = 1 if is_call else -1

        pnl_points = sign * (current_price - pos.entry_price)
        pnl_rupees = pnl_points * pos.lots * pos.lot_size

        def result(action: str) -> PositionUpdate:
            return PositionUpdate(
                signal_id=signal_id,
                action=action,
                current_pnl=round(pnl_rupees, 2),
                current_pnl_pct=round(pnl_rupees / self.config.total_capital * 100, 4),
                time_in_trade_minutes=time_in_trade,
            )

        if pos.is_hero_zero and minutes_until(now, "14:55") <= 0:
            return result("EXIT_TIME")
        if self._is_near_close(now, self.config.force_exit_minutes_before_close) or (
            minutes_until(now, self.config.force_exit_time) <= 0
        ):
            return result("EXIT_TIME")

        if is_call and current_price >= pos.target_price:
            return result("EXIT_TARGET")
        if not is_call and current_price <= pos.target_price:
            return result("EXIT_TARGET")

        if is_call and current_price <= pos.current_sl:
            return result("EXIT_TRAILING_SL" if pos.current_sl >= pos.entry_price else "EXIT_SL")
        if not is_call and current_price >= pos.current_sl:
            return result("EXIT_TRAILING_SL" if pos.current_sl <= pos.entry_price else "EXIT_SL")

        self._apply_trailing_sl(pos, current_price)
        return result("HOLD")

    def close_position(
        self,
        signal_id: str,
        exit_price: float,
        exit_reason: str,
        exit_premium: Optional[float] = None,
    ) -> Optional[float]:
        """
        Finalise an open position: compute net P&L, update DB, and log.

        Parameters
        ----------
        signal_id:
            UUID of the signal to close.
        exit_price:
            Underlying spot at exit.
        exit_reason:
            "TARGET", "SL", "TRAILING_SL", "TIME", "TIME_STOP", …
        exit_premium:
            Option mark at exit (default: the last mark seen by the monitor).

        Returns
        -------
        Net ₹ P&L after spread and statutory costs, or ``None`` if the
        position was not open.  WIN means net P&L > 0.
        """
        with self._lock:
            pos = self._open_positions.pop(signal_id, None)
            if pos is None:
                logger.warning("close_position: signal %s not in open positions", signal_id)
                return None

            if pos.is_option:
                mark = exit_premium if exit_premium and exit_premium > 0 else (
                    pos.last_premium or pos.premium_mid or pos.entry_premium
                )
                exit_px = max(mark - pos.half_spread, 0.05)
                costs = round_trip_cost(
                    pos.entry_premium, exit_px, pos.quantity, pos.exchange,
                ).total
                pnl_rupees = (exit_px - pos.entry_premium) * pos.quantity - costs
                recorded_exit = round(exit_px, 2)  # the traded instrument's price
            else:
                is_call = pos.signal_type == "BUY_CALL"
                sign = 1 if is_call else -1
                tx_cost = pos.lots * self.config.transaction_cost_per_lot * 2
                pnl_rupees = sign * (exit_price - pos.entry_price) * pos.quantity - tx_cost
                recorded_exit = exit_price

            outcome = "WIN" if pnl_rupees > 0 else "LOSS"
            self._daily_realised_pnl += pnl_rupees

            if pos.db_id > 0:
                now_ist = datetime.now(tz=_IST).isoformat()
                try:
                    self._db.execute(
                        """
                        UPDATE trading_signals
                        SET outcome = ?, actual_exit_price = ?, actual_pnl = ?,
                            closed_at = ?
                        WHERE id = ?
                        """,
                        (outcome, recorded_exit, round(pnl_rupees, 2), now_ist, pos.db_id),
                    )
                    self._persist_exit_details(pos, exit_price, recorded_exit, exit_reason)
                except Exception:
                    logger.exception("Failed to update signal %s in DB", signal_id)

        logger.info(
            "CLOSED %-12s %s: spot %.2f→%.2f  %s  net PnL=₹%.0f  (%s)  [%s]",
            pos.index_id, pos.signal_type, pos.entry_price, exit_price,
            f"premium {pos.entry_premium:.2f}→{recorded_exit:.2f}" if pos.is_option else "",
            pnl_rupees, outcome, exit_reason,
        )
        return round(pnl_rupees, 2)

    # ------------------------------------------------------------------
    # Public API — reporting
    # ------------------------------------------------------------------

    def get_daily_pnl_summary(self) -> DailyPnL:
        """Return a :class:`DailyPnL` for today."""
        today_str = date.today().isoformat()

        rows = self._db.fetch_all(
            """
            SELECT outcome, actual_pnl
            FROM   trading_signals
            WHERE  outcome IN ('WIN', 'LOSS')
              AND  date(closed_at) = ?
            """,
            (today_str,),
        )

        wins = [r["actual_pnl"] for r in rows if r["outcome"] == "WIN"]
        losses = [r["actual_pnl"] for r in rows if r["outcome"] == "LOSS"]
        total_trades = len(rows)
        total_pnl = sum(r["actual_pnl"] for r in rows if r["actual_pnl"] is not None)

        daily_limit = self.config.total_capital * self.config.max_risk_per_day_pct / 100
        unrealised = self._unrealised_pnl_total()
        risk_used = max(0.0, -(total_pnl + unrealised))

        return DailyPnL(
            date=date.today(),
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / total_trades if total_trades else 0.0,
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl / self.config.total_capital * 100, 4),
            largest_win=round(max(wins, default=0.0), 2),
            largest_loss=round(min(losses, default=0.0), 2),
            avg_win=round(sum(wins) / len(wins), 2) if wins else 0.0,
            avg_loss=round(sum(losses) / len(losses), 2) if losses else 0.0,
            risk_used_pct=round(risk_used / daily_limit * 100, 2) if daily_limit > 0 else 0.0,
            remaining_risk=round(max(0.0, daily_limit - risk_used), 2),
            open_positions=len(self._open_positions),
        )

    def get_portfolio_summary(self) -> PortfolioSummary:
        """Return a :class:`PortfolioSummary` snapshot."""
        today = date.today()
        week_start = (today - timedelta(days=today.weekday())).isoformat()
        month_start = today.replace(day=1).isoformat()
        today_str = today.isoformat()

        def _sum_pnl(start: str, end: str) -> float:
            row = self._db.fetch_one(
                """
                SELECT COALESCE(SUM(actual_pnl), 0) AS total
                FROM   trading_signals
                WHERE  outcome IN ('WIN', 'LOSS')
                  AND  date(closed_at) BETWEEN ? AND ?
                """,
                (start, end),
            )
            return float(row["total"]) if row else 0.0

        def _count_today() -> int:
            row = self._db.fetch_one(
                """
                SELECT COUNT(*) AS cnt
                FROM   trading_signals
                WHERE  date(generated_at) = ?
                  AND  signal_type != 'NO_TRADE'
                """,
                (today_str,),
            )
            return int(row["cnt"]) if row else 0

        today_pnl = _sum_pnl(today_str, today_str)
        week_pnl = _sum_pnl(week_start, today_str)
        month_pnl = _sum_pnl(month_start, today_str)
        trades_today = _count_today()

        # Build per-position dicts (option positions carry their last mark)
        open_list = []
        total_margin = 0.0
        with self._lock:
            positions = list(self._open_positions.values())
        for pos in positions:
            item = {
                "signal_id": pos.signal_id,
                "index": pos.index_id,
                "signal_type": pos.signal_type,
                "entry": pos.entry_price,
                "current_sl": pos.current_sl,
                "target": pos.target_price,
                "lots": pos.lots,
                "lot_size": pos.lot_size,
                "entry_time": pos.entry_time.isoformat(),
            }
            if pos.is_option:
                item.update({
                    "contract": f"{pos.expiry} {pos.strike:.0f}{pos.option_type}",
                    "entry_premium": pos.entry_premium,
                    "premium_stop": pos.premium_stop,
                    "premium_target": pos.premium_target,
                    "last_premium": pos.last_premium,
                })
                # A long option's capital at work is the premium paid.
                total_margin += pos.entry_premium * pos.quantity
            else:
                # Legacy spot-only record: rough 15 % of notional
                total_margin += pos.entry_price * pos.quantity * 0.15
            open_list.append(item)

        daily_limit = self.config.total_capital * self.config.max_risk_per_day_pct / 100
        realised_loss = abs(min(0.0, today_pnl))
        remaining = max(0.0, daily_limit - realised_loss)

        return PortfolioSummary(
            total_capital=self.config.total_capital,
            current_exposure=round(total_margin, 2),
            available_capital=round(self.config.total_capital - total_margin, 2),
            open_positions=open_list,
            today_pnl=round(today_pnl, 2),
            this_week_pnl=round(week_pnl, 2),
            this_month_pnl=round(month_pnl, 2),
            total_trades_today=trades_today,
            daily_limit_remaining_pct=round(remaining / daily_limit * 100, 2)
            if daily_limit > 0
            else 0.0,
        )

    # ------------------------------------------------------------------
    # Refinement steps (private)
    # ------------------------------------------------------------------

    def _step_validate_sl(
        self,
        refined: RefinedSignal,
        chain: Optional[OptionsChainData],
    ) -> None:
        """Step 1 — validate and if necessary adjust the stop-loss."""
        entry = refined.refined_entry
        sl = refined.refined_stop_loss
        is_call = refined.signal_type == "BUY_CALL"

        # --- Side check -------------------------------------------------
        if is_call and sl >= entry:
            sl = entry * (1.0 - self.config.min_sl_pct)
            refined.adjustments_made.append(
                f"SL was above/at entry for BUY_CALL — moved to entry − {self.config.min_sl_pct*100:.1f}%"
            )
        elif not is_call and sl <= entry:
            sl = entry * (1.0 + self.config.min_sl_pct)
            refined.adjustments_made.append(
                f"SL was below/at entry for BUY_PUT — moved to entry + {self.config.min_sl_pct*100:.1f}%"
            )

        dist_pct = abs(entry - sl) / entry

        # --- Too tight (< 0.3%) -----------------------------------------
        if dist_pct < self.config.min_sl_pct:
            new_dist = entry * self.config.min_sl_pct
            sl = entry - new_dist if is_call else entry + new_dist
            refined.adjustments_made.append(
                f"SL too tight ({dist_pct*100:.2f}%) — widened to {self.config.min_sl_pct*100:.1f}%"
            )
            dist_pct = self.config.min_sl_pct

        # --- Too wide (> 3%) — tighten (conservative) -------------------
        if dist_pct > self.config.max_sl_pct:
            new_dist = entry * self.config.max_sl_pct
            sl = entry - new_dist if is_call else entry + new_dist
            refined.adjustments_made.append(
                f"SL too wide ({dist_pct*100:.2f}%) — tightened to {self.config.max_sl_pct*100:.1f}%"
            )

        # --- OI-based support/resistance check --------------------------
        # An OI wall between the stop and entry is a defended level: if it
        # breaks, the trade thesis is wrong, so the stop TIGHTENS to just
        # beyond it.  It never WIDENS the stop — the old rule moved the stop
        # out to the max-OI strike (often 2 % away), turning a 90-point stop
        # into a 500-point one for an option buyer.
        if chain is not None and chain.strikes:
            support, resistance = self._oi_sr_levels(chain)
            min_dist = entry * self.config.min_sl_pct
            if is_call and support is not None and support < entry:
                sr_floor = support * (1.0 - self.config.sl_sr_buffer_pct)
                if sl < sr_floor and entry - sr_floor >= min_dist:
                    sl = sr_floor
                    refined.adjustments_made.append(
                        f"SL tightened to below OI support {support:.2f} "
                        f"(− {self.config.sl_sr_buffer_pct*100:.1f}% buffer)"
                    )
            elif not is_call and resistance is not None and resistance > entry:
                sr_ceil = resistance * (1.0 + self.config.sl_sr_buffer_pct)
                if sl > sr_ceil and sr_ceil - entry >= min_dist:
                    sl = sr_ceil
                    refined.adjustments_made.append(
                        f"SL tightened to above OI resistance {resistance:.2f} "
                        f"(+ {self.config.sl_sr_buffer_pct*100:.1f}% buffer)"
                    )

        # Final bounds sanity
        final_dist_pct = abs(entry - sl) / entry
        if final_dist_pct > self.config.max_sl_pct:
            refined.is_valid = False
            refined.rejection_reasons.append(
                f"SL distance {final_dist_pct*100:.2f}% exceeds max {self.config.max_sl_pct*100:.0f}% "
                "after adjustments"
            )
            return

        refined.refined_stop_loss = round(sl, 2)

    def _step_validate_target(
        self,
        refined: RefinedSignal,
        chain: Optional[OptionsChainData],
    ) -> None:
        """Step 2 — validate and if necessary adjust the target."""
        entry = refined.refined_entry
        sl = refined.refined_stop_loss
        target = refined.refined_target
        is_call = refined.signal_type == "BUY_CALL"

        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            refined.is_valid = False
            refined.rejection_reasons.append("SL distance is zero — cannot compute RR")
            return

        # --- OI resistance cap (non-HIGH confidence) --------------------
        # Only a wall on the profitable side of entry can cap the target: a
        # max-CE-OI strike *below* spot (already broken) used to drag a CALL
        # target under entry, and the abs()-based R:R let it through.
        if chain is not None and chain.strikes and refined.confidence_level != "HIGH":
            support, resistance = self._oi_sr_levels(chain)
            if resistance is not None and resistance <= entry:
                resistance = None
            if support is not None and support >= entry:
                support = None
            if is_call and resistance is not None and target > resistance:
                refined.adjustments_made.append(
                    f"Target {target:.2f} beyond OI resistance {resistance:.2f} "
                    f"(confidence {refined.confidence_level}) — pulled back"
                )
                target = resistance
            elif not is_call and support is not None and target < support:
                refined.adjustments_made.append(
                    f"Target {target:.2f} beyond OI support {support:.2f} "
                    f"(confidence {refined.confidence_level}) — pulled back"
                )
                target = support

        # Persist any OI adjustment to refined_target immediately so it is
        # visible even when we subsequently reject on RR grounds.
        refined.refined_target = round(target, 2)

        # --- Side check: target must be in the profitable direction ------
        if (is_call and target <= entry) or (not is_call and target >= entry):
            refined.is_valid = False
            refined.rejection_reasons.append(
                f"Target {target:.2f} is on the wrong side of entry {entry:.2f} "
                f"for {refined.signal_type}"
            )
            return

        # --- RR check ---------------------------------------------------
        tgt_dist = abs(target - entry)
        rr = tgt_dist / sl_dist if sl_dist > 0 else 0.0

        # Hard reject: RR < 1.0 is never acceptable regardless of confidence
        if rr < 1.0:
            refined.is_valid = False
            refined.rejection_reasons.append(f"RR {rr:.2f} < 1.0 — always rejected")
            return

        if rr < self.config.min_risk_reward_ratio:
            # Try to push target to meet minimum RR
            min_tgt_dist = sl_dist * self.config.min_risk_reward_ratio
            if is_call:
                new_target = entry + min_tgt_dist
            else:
                new_target = entry - min_tgt_dist

            # Accept only if no resistance was blocking us (already handled above)
            if chain is None or refined.confidence_level == "HIGH":
                target = new_target
                rr = self.config.min_risk_reward_ratio
                refined.adjustments_made.append(
                    f"Target stretched to achieve minimum RR {self.config.min_risk_reward_ratio}"
                )
                refined.refined_target = round(target, 2)
            else:
                refined.is_valid = False
                refined.rejection_reasons.append(
                    f"RR {rr:.2f} < minimum {self.config.min_risk_reward_ratio} "
                    "— target capped by OI resistance, cannot achieve min RR"
                )
                return

        refined.refined_target = round(target, 2)
        refined.risk_reward_ratio = round(rr, 2)

    # ------------------------------------------------------------------
    # Step 0 — entry gates
    # ------------------------------------------------------------------

    @staticmethod
    def _reject(refined: RefinedSignal, reason: str) -> RefinedSignal:
        refined.is_valid = False
        refined.rejection_reasons.append(reason)
        return refined

    def _entry_gate_reason(self, refined: RefinedSignal, now: datetime) -> Optional[str]:
        """Return a rejection reason, or ``None`` if a new entry is allowed."""
        cfg = self.config

        if minutes_until(now, cfg.no_entry_before) > 0:
            return f"No new entries before {cfg.no_entry_before} (opening noise, wide spreads)"
        close_buffer = 10 if refined.is_hero_zero else cfg.no_entry_after_minutes_before_close
        if self._is_near_close(now, close_buffer):
            return f"No new entries within {close_buffer} minutes of market close"

        closed = self._closed_trades_today(now)
        if closed is None:
            return "Cannot verify today's closed trades (DB error) — no new entries"
        streak = 0
        for row in reversed(closed):
            if row["outcome"] != "LOSS":
                break
            streak += 1
        if streak >= cfg.max_consecutive_losses:
            return f"{streak} consecutive losses today — new entries halted for the day"
        for row in reversed(closed):
            if row["index_id"] != refined.index_id or row["outcome"] != "LOSS":
                continue
            closed_at = self._parse_ts(row["closed_at"])
            if closed_at is not None:
                mins = (now - closed_at).total_seconds() / 60.0
                if 0 <= mins < cfg.loss_cooldown_minutes:
                    return (
                        f"Cooldown: last {refined.index_id} trade lost {mins:.0f} min ago "
                        f"(< {cfg.loss_cooldown_minutes} min)"
                    )
            break

        open_pos = list(self._open_positions.values())
        if len(open_pos) >= cfg.max_open_positions:
            return f"Max open positions reached ({len(open_pos)}/{cfg.max_open_positions})"
        index_open = sum(1 for p in open_pos if p.index_id == refined.index_id)
        if index_open >= cfg.max_positions_per_index:
            return (
                f"Max positions for {refined.index_id} reached "
                f"({index_open}/{cfg.max_positions_per_index})"
            )
        same_dir = sum(1 for p in open_pos if p.signal_type == refined.signal_type)
        if same_dir >= cfg.max_same_direction_positions:
            return (
                f"Max same-direction positions reached ({same_dir}/"
                f"{cfg.max_same_direction_positions}) — correlated indices are one bet"
            )

        remaining = self._compute_daily_remaining()
        refined.daily_loss_remaining = round(remaining, 2)
        if remaining <= 0:
            return "Daily loss limit reached (realised loss + risk reserved by open positions)"
        return None

    def _closed_trades_today(self, now: datetime) -> Optional[list[dict]]:
        try:
            return self._db.fetch_all(
                """
                SELECT index_id, outcome, closed_at FROM trading_signals
                WHERE outcome IN ('WIN', 'LOSS') AND date(closed_at) = ?
                ORDER BY closed_at ASC
                """,
                (now.date().isoformat(),),
            )
        except Exception:
            logger.exception("Could not load today's closed trades")
            return None

    @staticmethod
    def _parse_ts(value) -> Optional[datetime]:
        try:
            ts = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        return ts.replace(tzinfo=_IST) if ts.tzinfo is None else ts

    # ------------------------------------------------------------------
    # Step 3 — contract selection
    # ------------------------------------------------------------------

    def _step_select_contract(
        self,
        refined: RefinedSignal,
        chain: Optional[OptionsChainData],
        next_chain: Optional[OptionsChainData],
        now: datetime,
    ) -> tuple[Optional[ContractQuote], str]:
        cfg = self.config
        if chain is None or not chain.strikes:
            return None, "no live options chain (cannot price the trade)"

        use_chain = chain
        if chain.expiry_date == now.date() and not refined.is_hero_zero:
            # 0DTE: theta and gamma are extreme — directional buys go to the
            # next expiry, priced from THAT expiry's chain (not today's).
            if next_chain is None or not next_chain.strikes:
                return None, "expiry day and the next-expiry chain is unavailable"
            use_chain = next_chain
            refined.adjustments_made.append(
                f"Expiry day — using next expiry {next_chain.expiry_date}"
            )

        if refined.is_hero_zero:
            target_delta = cfg.target_abs_delta_hero_zero
        elif refined.confidence_level == "HIGH":
            target_delta = cfg.target_abs_delta_high
        else:
            target_delta = cfg.target_abs_delta_medium
        params = SelectionParams(
            target_abs_delta=target_delta,
            min_premium=cfg.min_option_premium,
            max_spread_pct=cfg.max_spread_pct,
            assumed_half_spread_pct=cfg.assumed_half_spread_pct,
        )
        quote, why = select_contract(use_chain, refined.signal_type, refined.refined_entry, now, params)
        if quote is None:
            return None, why
        quote.index_id = refined.index_id

        g = quote.greeks
        refined.recommended_strike = quote.strike
        refined.recommended_expiry = quote.expiry.isoformat()
        refined.option_type = quote.option_type
        refined.premium_mid = round(quote.mid, 2)
        refined.half_spread = round(quote.half_spread, 2)
        refined.spread_known = quote.spread_known
        refined.option_iv = round(quote.iv, 4)
        refined.days_to_expiry = float(max((quote.expiry - now.date()).days, 0))
        refined.option_greeks = {
            "delta": round(g.delta, 4),
            "gamma": round(g.gamma, 6),
            "theta": round(g.theta_per_day, 2),
            "vega": round(g.vega_per_pct, 2),
            "iv": round(quote.iv * 100, 2),
            "iv_source": quote.iv_source,
        }
        refined.adjustments_made.append(
            f"Contract {quote.expiry} {quote.strike:.0f}{quote.option_type}: mid ₹{quote.mid:.2f}, "
            f"Δ {g.delta:+.2f}, θ ₹{g.theta_per_day:.2f}/day, IV {quote.iv * 100:.1f}% ({quote.iv_source})"
        )
        if not quote.spread_known:
            refined.warnings.append(
                f"Bid/ask unavailable — spread assumed {cfg.assumed_half_spread_pct * 200:.1f}% "
                "of premium; place a LIMIT order"
            )
        return quote, ""

    # ------------------------------------------------------------------
    # Step 4 — premium levels and the edge check
    # ------------------------------------------------------------------

    def _step_premium_levels(
        self,
        refined: RefinedSignal,
        quote: ContractQuote,
        now: datetime,
        lot_size: int,
    ) -> bool:
        cfg = self.config
        mid, half = quote.mid, quote.half_spread
        entry_paid = mid + half
        exchange = self._exchange_for(refined.index_id)

        if refined.is_hero_zero:
            stop = mid * (1.0 - cfg.hero_zero_premium_sl_pct)
            target = mid * cfg.hero_zero_target_multiple
            refined.adjustments_made.append(
                f"HERO_ZERO: premium stop −{cfg.hero_zero_premium_sl_pct:.0%}, "
                f"target {cfg.hero_zero_target_multiple:.1f}× premium"
            )
        else:
            hold_until = now + timedelta(minutes=cfg.expected_hold_minutes)
            at_spot_sl = premium_at(quote, refined.refined_stop_loss, hold_until)
            floor = mid * (1.0 - cfg.max_premium_sl_pct)
            stop = max(at_spot_sl, floor)
            if floor > at_spot_sl:
                refined.adjustments_made.append(
                    f"Premium stop capped at −{cfg.max_premium_sl_pct:.0%} (₹{floor:.2f}); "
                    f"spot SL alone implied ₹{at_spot_sl:.2f}"
                )
            target = premium_at(quote, refined.refined_target, hold_until)

        exit_at_stop = max(stop - half, 0.05)
        exit_at_target = target - half
        cost_stop = round_trip_cost(entry_paid, exit_at_stop, lot_size, exchange).total
        cost_target = round_trip_cost(entry_paid, max(exit_at_target, 0.05), lot_size, exchange).total
        risk_lot = (entry_paid - exit_at_stop) * lot_size + cost_stop
        reward_lot = (exit_at_target - entry_paid) * lot_size - cost_target

        refined.entry_premium = round(entry_paid, 2)
        refined.option_premium = refined.entry_premium
        refined.premium_stop = round(stop, 2)
        refined.premium_target = round(target, 2)
        refined.risk_per_lot = round(risk_lot, 2)

        spot_rr = refined.risk_reward_ratio
        if reward_lot <= 0:
            self._reject(
                refined,
                f"No edge: spot target {refined.refined_target:.0f} lifts the premium only to "
                f"₹{target:.2f} after {cfg.expected_hold_minutes} min of theta — below entry "
                f"₹{entry_paid:.2f} + spread + costs",
            )
            return False
        rr = reward_lot / risk_lot if risk_lot > 0 else 0.0
        refined.premium_rr = round(rr, 2)
        if rr < cfg.min_risk_reward_ratio:
            self._reject(
                refined,
                f"Premium R:R {rr:.2f} < {cfg.min_risk_reward_ratio} after theta "
                f"({cfg.expected_hold_minutes} min), spread and costs (spot R:R was {spot_rr:.2f})",
            )
            return False
        refined.risk_reward_ratio = round(rr, 2)
        return True

    # ------------------------------------------------------------------
    # Step 5 — sizing on premium risk
    # ------------------------------------------------------------------

    def _risk_budget(self, refined: RefinedSignal) -> tuple[float, str]:
        cfg = self.config
        if refined.is_hero_zero:
            return cfg.total_capital * cfg.hero_zero_risk_pct / 100.0, "hero-zero fixed risk"
        cap = cfg.total_capital * cfg.max_risk_per_trade_pct / 100.0
        conf_pct = {
            "HIGH": cfg.high_confidence_size_pct,
            "MEDIUM": cfg.medium_confidence_size_pct,
            "LOW": cfg.low_confidence_size_pct,
        }.get(refined.confidence_level, cfg.low_confidence_size_pct)
        budget = cap * conf_pct / 100.0
        basis = f"{refined.confidence_level} confidence {conf_pct:.0f}% of {cfg.max_risk_per_trade_pct}%"
        if cfg.use_kelly:
            hist = self._fetch_historical_stats(refined.index_id)
            if hist["total_trades"] >= cfg.kelly_min_trades:
                k = self._kelly_raw(hist)
                if k <= 0:
                    return 0.0, f"negative historical edge (Kelly {k:.3f} ≤ 0)"
                budget = min(budget, cfg.kelly_fraction * k * cfg.total_capital)
                basis += f", capped at {cfg.kelly_fraction:.2f}× Kelly"
        budget *= refined.position_size_modifier
        return budget, basis

    def _step_size_premium(self, refined: RefinedSignal, lot_size: int) -> bool:
        cfg = self.config
        risk_lot = refined.risk_per_lot
        if risk_lot <= 0:
            self._reject(refined, "Non-positive risk per lot — cannot size")
            return False

        budget, basis = self._risk_budget(refined)
        refined.risk_amount = round(budget, 2)
        lots = math.floor(budget / risk_lot)
        if lots < 1:
            self._reject(
                refined,
                f"1 lot risks ₹{risk_lot:,.0f} > risk budget ₹{budget:,.0f} ({basis}) — "
                "skipped rather than forced to 1 lot",
            )
            return False
        if lots > cfg.max_lots_per_trade:
            refined.adjustments_made.append(f"Lots capped {lots} → {cfg.max_lots_per_trade}")
            lots = cfg.max_lots_per_trade

        remaining = refined.daily_loss_remaining
        lots_daily = math.floor(remaining / risk_lot)
        if lots_daily < 1:
            self._reject(
                refined,
                f"1 lot risks ₹{risk_lot:,.0f} > daily limit remaining ₹{remaining:,.0f} "
                "(realised loss + open risk)",
            )
            return False
        if lots_daily < lots:
            refined.adjustments_made.append(
                f"Lots reduced {lots} → {lots_daily} to stay within the daily limit"
            )
            lots = lots_daily

        per_lot_outlay = refined.entry_premium * lot_size
        open_outlay = sum(
            p.entry_premium * p.quantity for p in self._open_positions.values() if p.is_option
        )
        outlay_room = cfg.total_capital * cfg.max_premium_outlay_pct / 100.0 - open_outlay
        lots_outlay = math.floor(outlay_room / per_lot_outlay) if per_lot_outlay > 0 else 0
        if lots_outlay < 1:
            self._reject(
                refined,
                f"Premium outlay cap: 1 lot costs ₹{per_lot_outlay:,.0f}, room ₹{max(outlay_room, 0):,.0f} "
                f"({cfg.max_premium_outlay_pct:.0f}% of capital incl. open positions)",
            )
            return False
        if lots_outlay < lots:
            refined.adjustments_made.append(
                f"Lots reduced {lots} → {lots_outlay} by the premium outlay cap"
            )
            lots = lots_outlay

        qty = lots * lot_size
        exchange = self._exchange_for(refined.index_id)
        exit_stop = max(refined.premium_stop - refined.half_spread, 0.05)
        exit_target = refined.premium_target - refined.half_spread
        cost_stop = round_trip_cost(refined.entry_premium, exit_stop, qty, exchange).total
        cost_target = round_trip_cost(refined.entry_premium, max(exit_target, 0.05), qty, exchange).total

        refined.lots = lots
        refined.lot_size = lot_size
        refined.quantity = qty
        refined.max_loss_amount = round((refined.entry_premium - exit_stop) * qty + cost_stop, 2)
        refined.max_profit_amount = round((exit_target - refined.entry_premium) * qty - cost_target, 2)
        refined.transaction_cost_total = round(cost_stop, 2)
        refined.premium_outlay = round(refined.entry_premium * qty, 2)
        refined.total_margin_required = refined.premium_outlay
        refined.risk_pct_of_capital = round(refined.max_loss_amount / cfg.total_capital * 100, 4)
        refined.breakeven_move = round(2 * refined.half_spread + cost_stop / qty, 2)
        refined.adjustments_made.append(
            f"Size {lots} lot(s) × {lot_size} = {qty}: max loss ₹{refined.max_loss_amount:,.0f} "
            f"({refined.risk_pct_of_capital:.2f}% of ₹{cfg.total_capital:,.0f}), "
            f"outlay ₹{refined.premium_outlay:,.0f}, premium R:R {refined.premium_rr:.2f}"
        )
        return True

    # ------------------------------------------------------------------
    # Premium trailing stop
    # ------------------------------------------------------------------

    def _trail_premium_stop(self, pos: _OpenPosition, mark: float) -> bool:
        """Ratchet the premium stop (never loosens).  Returns True if moved.

        Progress = share of the entry→target premium distance achieved:
          ≥ 50 %  → breakeven (exit at bid covers entry + costs)
          ≥ 75 %  → lock 30 % of the distance
        """
        dist = pos.premium_target - pos.entry_premium
        if dist <= 0 or pos.quantity <= 0:
            return False
        progress = (mark - pos.entry_premium) / dist
        if progress >= 0.75:
            new_stop = pos.entry_premium + 0.3 * dist
        elif progress >= 0.5:
            costs = round_trip_cost(pos.entry_premium, pos.entry_premium, pos.quantity, pos.exchange).total
            new_stop = pos.entry_premium + pos.half_spread + costs / pos.quantity
        else:
            return False
        if new_stop > pos.premium_stop:
            pos.premium_stop = round(new_stop, 2)
            return True
        return False

    # ------------------------------------------------------------------
    # Trailing SL logic
    # ------------------------------------------------------------------

    def _apply_trailing_sl(
        self,
        pos: _OpenPosition,
        current_price: float,
    ) -> None:
        """
        Move the trailing stop only in the favourable direction.

        Milestones (based on % of target distance achieved):
          > 50%  → SL to entry (breakeven)
          > 75%  → SL to entry + 30% of target distance
          > 100% → SL to entry + 60% of target distance
        """
        is_call = pos.signal_type == "BUY_CALL"
        entry = pos.entry_price
        target = pos.target_price

        tgt_dist = abs(target - entry)
        if tgt_dist <= 0:
            return

        if is_call:
            profit_pct = (current_price - entry) / tgt_dist
        else:
            profit_pct = (entry - current_price) / tgt_dist

        if profit_pct >= 1.0:
            new_sl = (entry + 0.6 * tgt_dist) if is_call else (entry - 0.6 * tgt_dist)
        elif profit_pct >= 0.75:
            new_sl = (entry + 0.3 * tgt_dist) if is_call else (entry - 0.3 * tgt_dist)
        elif profit_pct >= 0.5:
            new_sl = entry  # breakeven
        else:
            return  # not yet profitable enough to trail

        # Only ratchet — never widen
        if is_call:
            if new_sl > pos.current_sl:
                pos.current_sl = round(new_sl, 2)
        else:
            if new_sl < pos.current_sl:
                pos.current_sl = round(new_sl, 2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_refined(self, signal: TradingSignal) -> RefinedSignal:
        """Build a RefinedSignal pre-populated with signal fields."""
        return RefinedSignal(
            signal_id=signal.signal_id,
            index_id=signal.index_id,
            generated_at=signal.generated_at,
            signal_type=signal.signal_type,
            confidence_level=signal.confidence_level,
            confidence_score=signal.confidence_score,
            entry_price=signal.entry_price,
            target_price=signal.target_price,
            stop_loss=signal.stop_loss,
            risk_reward_ratio=signal.risk_reward_ratio,
            regime=signal.regime,
            weighted_score=signal.weighted_score,
            vote_breakdown=signal.vote_breakdown,
            risk_level=signal.risk_level,
            position_size_modifier=max(signal.position_size_modifier, 0.1),
            suggested_lot_count=signal.suggested_lot_count,
            estimated_max_loss=signal.estimated_max_loss,
            estimated_max_profit=signal.estimated_max_profit,
            reasoning=signal.reasoning,
            warnings=list(signal.warnings),
            outcome=signal.outcome,
            actual_exit_price=signal.actual_exit_price,
            actual_pnl=signal.actual_pnl,
            closed_at=signal.closed_at,
            data_completeness=signal.data_completeness,
            signals_generated_today=signal.signals_generated_today,
            refined_entry=signal.entry_price,
            refined_target=signal.target_price,
            refined_stop_loss=signal.stop_loss,
            # The flag used to be dropped here, so the 14:55 hero-zero exit
            # never fired and history queries never matched.
            is_hero_zero=bool(getattr(signal, "is_hero_zero", False))
            or any("HERO_ZERO" in w for w in (signal.warnings or [])),
        )

    def _get_lot_size(self, index_id: str) -> Optional[int]:
        """Return lot size from IndexRegistry, or None if not F&O tradeable."""
        try:
            reg = get_registry()
            idx = reg.get(index_id)
            return idx.lot_size if idx else None
        except Exception:
            return None

    def _oi_sr_levels(
        self, chain: OptionsChainData
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Derive support and resistance from OI-dominant strikes.

        Support  = strike with highest aggregate PE OI (put writers defend it)
        Resistance = strike with highest aggregate CE OI (call writers cap it)
        """
        if not chain.strikes:
            return None, None

        max_pe_oi = max(chain.strikes, key=lambda s: s.pe_oi, default=None)
        max_ce_oi = max(chain.strikes, key=lambda s: s.ce_oi, default=None)

        support = max_pe_oi.strike_price if max_pe_oi and max_pe_oi.pe_oi > 0 else None
        resistance = max_ce_oi.strike_price if max_ce_oi and max_ce_oi.ce_oi > 0 else None

        return support, resistance

    def _compute_daily_remaining(self) -> float:
        """₹ of daily loss limit still available for a NEW trade.

        limit − realised net loss − risk already reserved by open positions.
        (The old version ignored open positions, so three open trades could
        lose ~3× the daily limit.)
        """
        daily_limit = self.config.total_capital * self.config.max_risk_per_day_pct / 100
        realised_loss = abs(min(0.0, self._daily_realised_pnl))
        return max(0.0, daily_limit - realised_loss - self._open_risk_reserved())

    def _open_risk_reserved(self) -> float:
        """₹ each open position can still lose from here to its stop."""
        total = 0.0
        for p in self._open_positions.values():
            if p.is_option:
                exit_stop = max(p.premium_stop - p.half_spread, 0.05)
                costs = round_trip_cost(p.entry_premium, exit_stop, p.quantity, p.exchange).total
                total += max(0.0, (p.entry_premium - exit_stop) * p.quantity) + costs
            elif p.max_loss_amount > 0:
                total += p.max_loss_amount
            else:
                total += abs(p.entry_price - p.current_sl) * p.quantity
        return total

    def _unrealised_pnl_total(self) -> float:
        """Mark-to-market P&L of open option positions at their last mark."""
        total = 0.0
        for p in self._open_positions.values():
            if p.is_option and p.last_premium > 0:
                total += (max(p.last_premium - p.half_spread, 0.05) - p.entry_premium) * p.quantity
        return total

    def _contract_lot_size(self, index_id: str, expiry: Optional[date]) -> Optional[int]:
        """Lot size of the specific contract (lot sizes are per expiry)."""
        try:
            from src.data.contract_master import get_contract_master  # noqa: PLC0415

            idx = get_registry().get(index_id)
            symbol = idx.option_symbol if idx and idx.option_symbol else index_id
            lot = get_contract_master().lot_size(symbol, expiry=expiry)
            if lot:
                return int(lot)
        except Exception:
            logger.debug("Contract lot-size lookup failed for %s", index_id, exc_info=True)
        return self._get_lot_size(index_id)

    @staticmethod
    def _exchange_for(index_id: str) -> str:
        try:
            idx = get_registry().get(index_id)
            return (idx.exchange if idx else "NSE") or "NSE"
        except Exception:
            return "NSE"

    @staticmethod
    def _is_near_close(now: datetime, minutes: int) -> bool:
        """True if fewer than *minutes* remain until 15:30 IST close."""
        h, m = settings.market_hours.market_close.split(":")
        close = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
        return now >= close - timedelta(minutes=minutes)

    def _fetch_historical_stats(self, index_id: str) -> dict:
        """Fetch win rate and payoff ratio from closed trades on *index_id*."""
        try:
            rows = self._db.fetch_all(
                """
                SELECT outcome, actual_pnl
                FROM   trading_signals
                WHERE  index_id = ?
                  AND  outcome IN ('WIN', 'LOSS')
                  AND  actual_pnl IS NOT NULL
                ORDER  BY closed_at DESC
                LIMIT  100
                """,
                (index_id,),
            )

            if not rows:
                return {"total_trades": 0, "win_rate": 0.5, "avg_win": 0.0, "avg_loss": 0.0}

            wins = [r["actual_pnl"] for r in rows if r["outcome"] == "WIN"]
            losses = [r["actual_pnl"] for r in rows if r["outcome"] == "LOSS"]

            if not wins or not losses:
                return {
                    "total_trades": len(rows),
                    "win_rate": len(wins) / len(rows) if rows else 0.5,
                    "avg_win": sum(wins) / len(wins) if wins else 0.0,
                    "avg_loss": abs(sum(losses) / len(losses)) if losses else 0.0,
                }

            return {
                "total_trades": len(rows),
                "win_rate": len(wins) / len(rows),
                "avg_win": sum(wins) / len(wins),
                "avg_loss": abs(sum(losses) / len(losses)),
            }
        except Exception:
            logger.debug(f"Could not fetch historical stats for {index_id}")
            return {"total_trades": 0, "win_rate": 0.5, "avg_win": 0.0, "avg_loss": 0.0}

    @staticmethod
    def _kelly_criterion(hist: dict) -> float:
        """
        Calculate optimal position size via Kelly criterion.
        Kelly% = (b*p - (1-p)) / b, where b = avg_win/avg_loss, p = win_rate.
        Capped at 25% for safety (Kelly itself can be aggressive).
        """
        p = hist.get("win_rate", 0.5)
        avg_win = hist.get("avg_win", 0.0)
        avg_loss = hist.get("avg_loss", 0.0)

        if avg_loss <= 0 or avg_win <= 0:
            return 0.02  # conservative fallback: 2% per trade

        b = avg_win / avg_loss
        kelly = (b * p - (1 - p)) / b if b > 0 else 0.02

        # Clamp to [0.5%, 25%] for practical trading
        kelly = max(0.005, min(kelly, 0.25))
        return kelly

    @staticmethod
    def _kelly_raw(hist: dict) -> float:
        """Unclamped Kelly fraction (≤ 0 means no historical edge)."""
        p = hist.get("win_rate", 0.0)
        avg_win = hist.get("avg_win", 0.0)
        avg_loss = hist.get("avg_loss", 0.0)
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        b = avg_win / avg_loss
        return (b * p - (1 - p)) / b

    def _load_open_positions(self) -> None:
        """Restore open positions from DB on startup (today's OPEN signals)."""
        today_str = date.today().isoformat()
        try:
            rows = self._db.fetch_all(
                """
                SELECT id, index_id, signal_type, entry_price, stop_loss,
                       target_price, confidence_level, generated_at, reasoning
                FROM   trading_signals
                WHERE  outcome = 'OPEN'
                  AND  date(generated_at) = ?
                  AND  signal_type != 'NO_TRADE'
                ORDER  BY generated_at ASC
                """,
                (today_str,),
            )
        except Exception:
            logger.warning("Could not load open positions from DB", exc_info=True)
            return

        for row in rows:
            # Parse extra risk details from reasoning JSON if present
            try:
                reasoning = json.loads(row["reasoning"])
                risk = reasoning.get("risk_details", {}) or {}
                signal_id = risk.get("signal_id") or reasoning.get("signal_id") or f"db_{row['id']}"
                lots = int(risk.get("lots", 1) or 1)
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                risk = {}
                signal_id = f"db_{row['id']}"
                lots = 1

            lot_size = int(risk.get("lot_size") or self._get_lot_size(row["index_id"]) or 1)
            try:
                entry_time = datetime.fromisoformat(row["generated_at"])
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=_IST)
            except (ValueError, TypeError):
                entry_time = datetime.now(tz=_IST)

            expiry = None
            if risk.get("expiry"):
                try:
                    expiry = date.fromisoformat(str(risk["expiry"])[:10])
                except ValueError:
                    expiry = None
            has_contract = bool(risk.get("option_type") and risk.get("strike") and expiry)

            pos = _OpenPosition(
                db_id=row["id"],
                signal_id=signal_id,
                index_id=row["index_id"],
                signal_type=row["signal_type"],
                entry_price=float(risk.get("spot_entry") or row["entry_price"]),
                current_sl=float(risk.get("spot_stop") or row["stop_loss"]),
                target_price=float(risk.get("spot_target") or row["target_price"]),
                lots=lots,
                lot_size=lot_size,
                entry_time=entry_time,
                confidence_level=row["confidence_level"],
                is_hero_zero=bool(risk.get("is_hero_zero", False)),
                option_type=risk.get("option_type") if has_contract else None,
                strike=float(risk["strike"]) if has_contract else None,
                expiry=expiry if has_contract else None,
                entry_premium=float(risk.get("entry_premium") or 0.0),
                premium_mid=float(risk.get("premium_mid") or 0.0),
                half_spread=float(risk.get("half_spread") or 0.0),
                # The trailing stop is persisted every time it moves, so a
                # restart resumes the ratcheted level, not the initial one.
                premium_stop=float(risk.get("premium_stop") or 0.0),
                initial_premium_stop=float(
                    risk.get("initial_premium_stop") or risk.get("premium_stop") or 0.0
                ),
                premium_target=float(risk.get("premium_target") or 0.0),
                entry_iv=float(risk.get("entry_iv") or 0.0),
                max_loss_amount=float(risk.get("max_loss_amount") or 0.0),
                last_premium=float(risk.get("last_premium") or risk.get("premium_mid") or 0.0),
                exchange=str(risk.get("exchange") or self._exchange_for(row["index_id"])),
            )
            self._open_positions[signal_id] = pos

        if self._open_positions:
            logger.info(
                "Loaded %d open position(s) from DB", len(self._open_positions)
            )

    def _refresh_daily_pnl(self) -> None:
        """Reload realised P&L from DB for today."""
        today_str = date.today().isoformat()
        try:
            row = self._db.fetch_one(
                """
                SELECT COALESCE(SUM(actual_pnl), 0) AS total
                FROM   trading_signals
                WHERE  outcome IN ('WIN', 'LOSS')
                  AND  date(closed_at) = ?
                """,
                (today_str,),
            )
            self._daily_realised_pnl = float(row["total"]) if row else 0.0
        except Exception:
            logger.warning("Could not load daily P&L from DB", exc_info=True)
            self._daily_realised_pnl = 0.0
        self._cache_date = date.today()

    def _maybe_refresh_daily_cache(self) -> None:
        """Refresh daily cache if the date has changed (midnight rollover)."""
        with self._lock:
            if self._cache_date != date.today():
                self._open_positions.clear()
                self._load_open_positions()
                self._refresh_daily_pnl()

    def get_open_positions(self) -> list[_OpenPosition]:
        """Snapshot of open positions (safe to iterate from another thread)."""
        with self._lock:
            return list(self._open_positions.values())

    def _find_signal_db_id(self, signal: RefinedSignal) -> Optional[int]:
        """
        Find the DB row id for a signal by matching index_id + generated_at.
        """
        try:
            row = self._db.fetch_one(
                """
                SELECT id FROM trading_signals
                WHERE  index_id     = ?
                  AND  generated_at = ?
                  AND  signal_type  = ?
                ORDER  BY id DESC
                LIMIT  1
                """,
                (
                    signal.index_id,
                    signal.generated_at.isoformat(),
                    signal.signal_type,
                ),
            )
            return int(row["id"]) if row else None
        except Exception:
            logger.debug("Could not find DB id for signal %s", signal.signal_id)
            return None

    def _update_reasoning_json(self, db_id: int, mutate) -> None:
        """Read-modify-write the reasoning JSON of one trading_signals row."""
        row = self._db.fetch_one("SELECT reasoning FROM trading_signals WHERE id = ?", (db_id,))
        if not row:
            return
        try:
            reasoning = json.loads(row["reasoning"]) if row["reasoning"] else {}
        except (json.JSONDecodeError, TypeError):
            reasoning = {"text": str(row["reasoning"])}
        mutate(reasoning)
        self._db.execute(
            "UPDATE trading_signals SET reasoning = ? WHERE id = ?",
            (json.dumps(reasoning, ensure_ascii=False, default=str), db_id),
        )

    @staticmethod
    def _position_state(pos: _OpenPosition) -> dict:
        return {
            "signal_id": pos.signal_id,
            "lots": pos.lots,
            "lot_size": pos.lot_size,
            "is_hero_zero": pos.is_hero_zero,
            "spot_entry": pos.entry_price,
            "spot_stop": pos.current_sl,
            "spot_target": pos.target_price,
            "option_type": pos.option_type,
            "strike": pos.strike,
            "expiry": pos.expiry.isoformat() if pos.expiry else None,
            "entry_premium": pos.entry_premium,
            "premium_mid": pos.premium_mid,
            "half_spread": pos.half_spread,
            "premium_stop": pos.premium_stop,
            "initial_premium_stop": pos.initial_premium_stop,
            "premium_target": pos.premium_target,
            "entry_iv": pos.entry_iv,
            "max_loss_amount": pos.max_loss_amount,
            "last_premium": pos.last_premium,
            "exchange": pos.exchange,
        }

    def _persist_risk_details(
        self, db_id: int, signal: RefinedSignal, pos: Optional[_OpenPosition] = None,
    ) -> None:
        """Store the full executable trade (contract + premium levels) on the row."""
        try:
            details = {
                "signal_id": signal.signal_id,
                "lots": signal.lots,
                "refined_entry": signal.refined_entry,
                "refined_target": signal.refined_target,
                "refined_stop_loss": signal.refined_stop_loss,
                "risk_amount": signal.risk_amount,
                "max_loss_amount": signal.max_loss_amount,
                "recommended_strike": signal.recommended_strike,
                "option_premium": signal.option_premium,
                "premium_rr": signal.premium_rr,
                "risk_pct_of_capital": signal.risk_pct_of_capital,
            }
            if pos is not None:
                details.update(self._position_state(pos))
            self._update_reasoning_json(db_id, lambda r: r.__setitem__("risk_details", details))
        except Exception:
            logger.warning(
                "Could not persist risk details for signal %s — a restart would "
                "not restore this position's contract", signal.signal_id, exc_info=True,
            )

    def _persist_position_state(self, pos: _OpenPosition) -> None:
        """Persist a moved trailing stop so a restart resumes from it."""
        if pos.db_id <= 0:
            return
        try:
            state = self._position_state(pos)

            def mutate(r: dict) -> None:
                r.setdefault("risk_details", {}).update(state)

            self._update_reasoning_json(pos.db_id, mutate)
        except Exception:
            logger.warning("Could not persist stop for %s", pos.signal_id, exc_info=True)

    def _persist_exit_details(
        self, pos: _OpenPosition, spot_exit: float, recorded_exit: float, reason: str,
    ) -> None:
        def mutate(r: dict) -> None:
            r["exit_details"] = {
                "reason": reason,
                "spot_exit": spot_exit,
                "exit_price": recorded_exit,
                "instrument": (
                    f"{pos.expiry} {pos.strike:.0f}{pos.option_type}" if pos.is_option else "spot"
                ),
            }

        try:
            self._update_reasoning_json(pos.db_id, mutate)
        except Exception:
            logger.debug("Could not persist exit details for %s", pos.signal_id, exc_info=True)
