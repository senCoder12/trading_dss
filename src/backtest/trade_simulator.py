"""
Trade Simulator — Phase 6 Step 6.2 of the Trading Decision Support System.

Simulates order execution, transaction costs, position tracking, trailing
stop-loss logic, and forced intraday exits.  Mirrors the RiskManager
(Phase 5.3) exactly so that backtest results faithfully predict live
performance.

Usage
-----
::

    sim = TradeSimulator(SimulatorConfig(initial_capital=500_000))
    # On each bar from DataReplayEngine:
    execution = sim.execute_entry(signal, current_bar)
    closed = sim.update_positions(current_bar, timestamp)
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Union

from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from src.data.index_registry import get_registry
from src.engine.signal_generator import TradingSignal

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# Market close in IST (15:30)
_MARKET_CLOSE_HOUR = 15
_MARKET_CLOSE_MINUTE = 30


# ---------------------------------------------------------------------------
# Variable Bid-Ask Spread Model
# ---------------------------------------------------------------------------


class SpreadModel:
    """Models bid-ask spread for Indian index options based on moneyness and time to expiry.

    The spread is the HIDDEN cost most retail traders ignore. It's the difference
    between the price you see (mid) and the price you actually get (bid for sells, ask for buys).
    Half the spread is effectively slippage on each side.
    """

    def estimate_spread(
        self,
        index_id: str,
        spot_price: float,
        strike_price: float | None,
        option_premium: float | None,
        days_to_expiry: int | None,
        is_expiry_day: bool = False,
        hour_of_day: int | None = None,
    ) -> float:
        """Estimate half-spread (one-side slippage) in index points.

        Returns estimated slippage in INDEX POINTS (not option premium points).
        This is what gets added/subtracted from the signal's entry/exit price.
        """
        # Base spread by index liquidity
        base_spreads = {
            "NIFTY50": 0.5,       # Most liquid
            "BANKNIFTY": 0.8,     # Very liquid
            "FINNIFTY": 1.5,      # Moderate
            "MIDCPNIFTY": 2.0,    # Less liquid
        }
        base = base_spreads.get(index_id, 2.0)  # Default for unknown indices

        # Moneyness multiplier (OTM options have wider spreads)
        moneyness_mult = 1.0
        if strike_price and spot_price:
            otm_pct = abs(strike_price - spot_price) / spot_price * 100
            if otm_pct < 1.0:
                moneyness_mult = 1.0    # ATM — tightest spread
            elif otm_pct < 2.0:
                moneyness_mult = 1.3    # Slightly OTM
            elif otm_pct < 3.0:
                moneyness_mult = 1.8    # OTM
            elif otm_pct < 5.0:
                moneyness_mult = 2.5    # Deep OTM
            else:
                moneyness_mult = 4.0    # Very deep OTM — very wide spreads

        # Time to expiry multiplier (spreads widen near expiry)
        expiry_mult = 1.0
        if days_to_expiry is not None:
            if days_to_expiry > 5:
                expiry_mult = 1.0     # Normal
            elif days_to_expiry > 2:
                expiry_mult = 1.2     # Slightly wider
            elif days_to_expiry > 0:
                expiry_mult = 1.5     # Expiry week
            else:
                expiry_mult = 2.5     # Expiry day — significantly wider

        # Expiry day last hour — spreads explode
        if is_expiry_day and hour_of_day is not None and hour_of_day >= 14:
            expiry_mult *= 1.5  # Additional 50% wider in last 1.5 hours

        # Option premium floor — very cheap options have proportionally huge spreads
        premium_mult = 1.0
        if option_premium is not None and option_premium > 0:
            if option_premium < 5:
                premium_mult = 3.0    # ₹5 option with ₹2 spread = 40% slippage!
            elif option_premium < 15:
                premium_mult = 2.0
            elif option_premium < 50:
                premium_mult = 1.3

        half_spread = base * moneyness_mult * expiry_mult * premium_mult

        # Cap: spread shouldn't exceed 5 points for NIFTY, 10 for others
        max_spread = 5.0 if index_id == "NIFTY50" else 10.0
        half_spread = min(half_spread, max_spread)

        return round(half_spread, 2)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimulatorConfig:
    """All configurable parameters for the trade simulator."""

    # Capital & risk limits
    initial_capital: float = 100_000.0
    max_risk_per_trade_pct: float = 2.0
    max_risk_per_day_pct: float = 5.0
    max_open_positions: int = 3
    max_positions_per_index: int = 1
    min_risk_reward_ratio: float = 1.3

    # Cost model (Indian market specific)
    brokerage_per_order: float = 20.0
    stt_rate: float = 0.000625           # 0.0625% on sell side for options
    exchange_charges_pct: float = 0.00053
    gst_pct: float = 0.18
    sebi_charges_pct: float = 0.000001
    stamp_duty_pct: float = 0.00003

    # Slippage model
    slippage_points: float = 2.0
    slippage_pct: float = 0.05
    use_percentage_slippage: bool = False
    use_variable_spread: bool = True       # Use SpreadModel instead of fixed slippage
    spread_model: Optional[SpreadModel] = None  # Custom model (uses default if None)

    # Execution rules
    intraday_only: bool = True
    force_exit_minutes_before_close: int = 5

    # Position sizing — confidence multipliers (% of max risk)
    high_confidence_size_pct: float = 100.0
    medium_confidence_size_pct: float = 60.0
    low_confidence_size_pct: float = 30.0

    # Safety caps
    max_lots_per_trade: int = 5

    # Transaction cost per lot (round-trip) — used for position sizing calc
    # (mirrors RiskConfig.transaction_cost_per_lot)
    transaction_cost_per_lot: float = 120.0


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TradeExecution:
    """An open (or pending-close) position created by the simulator."""

    trade_id: str
    index_id: str
    signal_id: str

    # Direction
    trade_type: str                      # "BUY_CALL" / "BUY_PUT"

    # Entry
    signal_entry_price: float
    actual_entry_price: float            # After slippage
    entry_timestamp: datetime
    entry_bar: dict

    # Levels
    stop_loss: float
    target: float
    trailing_sl: float                   # Starts same as stop_loss, ratchets
    risk_reward_ratio: float

    # Position size
    lots: int
    lot_size: int
    quantity: int                        # lots * lot_size

    # Option details
    strike_price: Optional[float] = None
    option_premium: Optional[float] = None
    expiry: Optional[str] = None

    # Cost
    entry_cost: float = 0.0

    # Risk
    risk_amount: float = 0.0
    confidence_level: str = "LOW"

    # Spread cost tracking
    spread_cost_entry: float = 0.0

    # Status
    status: str = "OPEN"

    # Tracking for MFE / MAE
    _max_favorable: float = field(default=0.0, repr=False)
    _max_adverse: float = field(default=0.0, repr=False)
    _bars_held: int = field(default=0, repr=False)


@dataclass
class ClosedTrade:
    """A fully resolved trade with complete P&L breakdown."""

    # Identity
    trade_id: str
    index_id: str
    signal_id: str
    trade_type: str

    # Entry
    signal_entry_price: float
    actual_entry_price: float
    entry_timestamp: datetime
    entry_bar: dict

    # Levels at entry
    original_stop_loss: float
    original_target: float

    # Exit
    actual_exit_price: float
    exit_timestamp: datetime
    exit_reason: str                     # TARGET_HIT / STOP_LOSS_HIT / TRAILING_SL_HIT / FORCED_EOD / MANUAL

    # Position size
    lots: int
    lot_size: int
    quantity: int
    confidence_level: str

    # P&L
    gross_pnl_points: float
    gross_pnl: float                     # ₹ before costs
    total_costs: float                   # Entry + exit costs
    net_pnl: float                       # ₹ after all costs
    net_pnl_pct: float                   # % of capital at time of trade

    # Performance
    duration_bars: int
    duration_minutes: int
    max_favorable_excursion: float       # Best unrealised profit (points)
    max_adverse_excursion: float         # Worst unrealised loss (points)

    # Outcome
    outcome: str                         # "WIN" / "LOSS" / "BREAKEVEN"

    # Option details
    strike_price: Optional[float] = None
    option_premium: Optional[float] = None
    expiry: Optional[str] = None

    # Spread cost tracking
    spread_cost_entry: float = 0.0
    spread_cost_exit: float = 0.0


@dataclass
class EquityPoint:
    """Single equity-curve observation."""

    timestamp: datetime
    capital: float                       # Cash + unrealised
    cash: float
    unrealized: float
    drawdown_pct: float
    open_positions: int


@dataclass
class PortfolioState:
    """Point-in-time portfolio snapshot."""

    current_capital: float
    initial_capital: float
    total_return: float
    open_positions: list[TradeExecution]
    open_position_count: int
    unrealized_pnl: float
    realized_pnl: float
    total_trades: int
    today_pnl: float
    today_trades: int
    daily_loss_limit_remaining: float
    max_capital_reached: float
    current_drawdown: float


# ---------------------------------------------------------------------------
# Trade Simulator
# ---------------------------------------------------------------------------


class TradeSimulator:
    """
    Simulates trade execution against bar-by-bar historical data.

    Mirrors RiskManager (Phase 5.3) logic for position sizing, trailing
    stop-loss ratcheting, and exit conditions so backtest results match
    expected live behaviour.
    """

    def __init__(self, config: Optional[SimulatorConfig] = None) -> None:
        self.config = config or SimulatorConfig()

        # State
        self.current_capital: float = self.config.initial_capital
        self.open_positions: list[TradeExecution] = []
        self.trade_history: list[ClosedTrade] = []
        self.equity_curve: list[EquityPoint] = []

        # Daily tracking
        self._daily_pnl: dict[str, float] = {}  # date_str → realised P&L
        self._daily_trades: dict[str, int] = {}
        self._current_date: Optional[str] = None

        # Drawdown tracking
        self._max_capital: float = self.config.initial_capital

        # Capital-depleted flag
        self._capital_depleted: bool = False

        # Duplicate signal filter: (index_id, bar_timestamp_str) → bool
        self._bar_signals: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # Public API — entry
    # ------------------------------------------------------------------

    def execute_entry(
        self,
        signal: Union[TradingSignal, object],
        current_bar: dict,
    ) -> Optional[TradeExecution]:
        """
        Simulate entering a position based on *signal*.

        Returns a :class:`TradeExecution` if the trade was taken, or
        ``None`` if any pre-execution check failed.
        """
        if self._capital_depleted:
            logger.warning("CAPITAL DEPLETED — cannot enter new trades")
            return None

        # --- Extract signal fields (support both TradingSignal & RefinedSignal) ---
        signal_id = getattr(signal, "signal_id", str(uuid.uuid4()))
        index_id = getattr(signal, "index_id", "")
        signal_type = getattr(signal, "signal_type", "")
        confidence_level = getattr(signal, "confidence_level", "LOW")
        entry_price = getattr(signal, "entry_price", 0.0)
        target_price = getattr(signal, "target_price", 0.0)
        stop_loss = getattr(signal, "stop_loss", 0.0)
        risk_reward_ratio = getattr(signal, "risk_reward_ratio", 0.0)
        position_size_modifier = getattr(signal, "position_size_modifier", 1.0)
        strike_price = getattr(signal, "recommended_strike", None) or getattr(signal, "strike_price", None)
        option_premium = getattr(signal, "option_premium", None)
        expiry = getattr(signal, "recommended_expiry", None) or getattr(signal, "expiry", None)

        # Use refined levels if available (RefinedSignal)
        entry_price = getattr(signal, "refined_entry", 0.0) or entry_price
        target_price = getattr(signal, "refined_target", 0.0) or target_price
        stop_loss = getattr(signal, "refined_stop_loss", 0.0) or stop_loss

        timestamp = self._bar_timestamp(current_bar)

        # --- Pre-execution checks ---

        # 1. Valid signal type
        if signal_type not in ("BUY_CALL", "BUY_PUT"):
            return None

        # 2. Risk-reward ratio check
        if risk_reward_ratio < self.config.min_risk_reward_ratio:
            logger.debug("Rejected %s: RR %.2f < min %.2f",
                         index_id, risk_reward_ratio, self.config.min_risk_reward_ratio)
            return None

        # 3. Daily loss limit
        date_str = self._date_str(timestamp)
        self._ensure_daily_tracker(date_str)
        daily_loss = self._daily_pnl.get(date_str, 0.0)
        daily_limit = self.config.initial_capital * self.config.max_risk_per_day_pct / 100
        if daily_loss <= -daily_limit:
            logger.info("Daily loss limit reached (₹%.0f) — trade rejected", daily_loss)
            return None

        # 4. Max open positions
        if len(self.open_positions) >= self.config.max_open_positions:
            logger.debug("Max open positions (%d) reached — trade rejected",
                         self.config.max_open_positions)
            return None

        # 5. Max positions per index
        idx_positions = sum(1 for p in self.open_positions if p.index_id == index_id)
        if idx_positions >= self.config.max_positions_per_index:
            logger.debug("Max positions for %s reached — trade rejected", index_id)
            return None

        # 6. Duplicate signal for same index on same bar
        bar_key = (index_id, str(timestamp))
        if bar_key in self._bar_signals:
            logger.debug("Duplicate signal for %s at %s — skipped", index_id, timestamp)
            return None
        self._bar_signals.add(bar_key)

        # --- Calculate slippage ---
        if self.config.use_variable_spread:
            model = self.config.spread_model or SpreadModel()
            slippage = model.estimate_spread(
                index_id=index_id,
                spot_price=current_bar["close"],
                strike_price=strike_price,
                option_premium=option_premium,
                days_to_expiry=getattr(signal, "days_to_expiry", None),
                is_expiry_day=self._is_expiry_day(timestamp),
                hour_of_day=self._get_hour(timestamp),
            )
        elif self.config.use_percentage_slippage:
            slippage = entry_price * self.config.slippage_pct / 100
        else:
            slippage = self.config.slippage_points

        entry_spread_cost = slippage  # Record for ClosedTrade

        if signal_type == "BUY_CALL":
            actual_entry = entry_price + slippage
        else:  # BUY_PUT
            actual_entry = entry_price - slippage

        # --- Position sizing (mirrors RiskManager._step_size_position) ---
        lot_size = self._get_lot_size(index_id)
        if lot_size is None:
            raise ValueError(
                f"No F&O lot size found for {index_id!r} in IndexRegistry"
            )

        max_risk = self.current_capital * self.config.max_risk_per_trade_pct / 100
        conf_map = {
            "HIGH": self.config.high_confidence_size_pct,
            "MEDIUM": self.config.medium_confidence_size_pct,
            "LOW": self.config.low_confidence_size_pct,
        }
        conf_pct = conf_map.get(confidence_level, self.config.low_confidence_size_pct)
        risk_amount = max_risk * (conf_pct / 100.0)
        risk_amount *= position_size_modifier
        risk_amount = max(risk_amount, 0.0)

        sl_distance = abs(actual_entry - stop_loss)
        loss_per_lot = sl_distance * lot_size + self.config.transaction_cost_per_lot
        if loss_per_lot <= 0:
            lots = 1
        else:
            lots = math.floor(risk_amount / loss_per_lot)
            lots = max(lots, 1)
            lots = min(lots, self.config.max_lots_per_trade)

        quantity = lots * lot_size

        # 7. Capital sufficiency — can we afford even 1 lot?
        min_margin = sl_distance * lot_size
        if min_margin > self.current_capital:
            logger.info("Cannot afford 1 lot for %s: margin ₹%.0f > capital ₹%.0f",
                        index_id, min_margin, self.current_capital)
            return None

        # --- Entry transaction cost ---
        turnover = sl_distance * quantity  # approximate turnover
        entry_brokerage = self.config.brokerage_per_order
        entry_exchange = turnover * self.config.exchange_charges_pct
        entry_gst = (entry_brokerage + entry_exchange) * self.config.gst_pct
        entry_stamp = turnover * self.config.stamp_duty_pct
        entry_sebi = turnover * self.config.sebi_charges_pct
        entry_cost = entry_brokerage + entry_exchange + entry_gst + entry_stamp + entry_sebi
        # STT is 0 on buy side for options

        # --- Build execution ---
        execution = TradeExecution(
            trade_id=str(uuid.uuid4()),
            index_id=index_id,
            signal_id=signal_id,
            trade_type=signal_type,
            signal_entry_price=entry_price,
            actual_entry_price=round(actual_entry, 2),
            entry_timestamp=timestamp,
            entry_bar=current_bar,
            stop_loss=stop_loss,
            target=target_price,
            trailing_sl=stop_loss,
            risk_reward_ratio=risk_reward_ratio,
            lots=lots,
            lot_size=lot_size,
            quantity=quantity,
            strike_price=strike_price,
            option_premium=option_premium,
            expiry=expiry,
            entry_cost=round(entry_cost, 2),
            risk_amount=round(risk_amount, 2),
            confidence_level=confidence_level,
            spread_cost_entry=round(entry_spread_cost, 2),
            status="OPEN",
        )

        self.open_positions.append(execution)
        logger.info(
            "ENTRY  %-12s %s  Sig=%.2f  Act=%.2f  SL=%.2f  Tgt=%.2f  Lots=%d",
            index_id, signal_type, entry_price, actual_entry,
            stop_loss, target_price, lots,
        )
        return execution

    # ------------------------------------------------------------------
    # Public API — position updates
    # ------------------------------------------------------------------

    def update_positions(
        self,
        current_bar: dict,
        timestamp: datetime,
    ) -> list[ClosedTrade]:
        """
        Check all open positions against *current_bar*.

        Called on EVERY bar during replay.  Returns list of positions
        closed this bar (SL, target, trailing SL, or forced EOD).
        """
        closed_this_bar: list[ClosedTrade] = []

        # Skip bars with NaN data
        for key in ("open", "high", "low", "close"):
            val = current_bar.get(key)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                logger.debug("Skipping bar at %s — NaN in %s", timestamp, key)
                self._record_equity(timestamp)
                return closed_this_bar

        bar_high = float(current_bar["high"])
        bar_low = float(current_bar["low"])
        bar_close = float(current_bar["close"])

        # Check forced EOD exit
        is_eod = self._is_forced_exit_time(timestamp)

        positions_to_close: list[tuple[TradeExecution, float, str]] = []

        for pos in list(self.open_positions):
            pos._bars_held += 1
            is_call = pos.trade_type == "BUY_CALL"

            # --- Track MFE/MAE ---
            if is_call:
                favorable = bar_high - pos.actual_entry_price
                adverse = pos.actual_entry_price - bar_low
            else:
                favorable = pos.actual_entry_price - bar_low
                adverse = bar_high - pos.actual_entry_price
            pos._max_favorable = max(pos._max_favorable, favorable)
            pos._max_adverse = max(pos._max_adverse, adverse)

            # --- Check SL hit (use low for CALL, high for PUT) ---
            sl_hit = False
            if is_call and bar_low <= pos.trailing_sl:
                sl_hit = True
            elif not is_call and bar_high >= pos.trailing_sl:
                sl_hit = True

            # --- Check target hit (use high for CALL, low for PUT) ---
            tgt_hit = False
            if is_call and bar_high >= pos.target:
                tgt_hit = True
            elif not is_call and bar_low <= pos.target:
                tgt_hit = True

            # --- If both SL and target hit in same bar → assume SL first (conservative) ---
            if sl_hit and tgt_hit:
                exit_price = pos.trailing_sl
                exit_reason = self._sl_exit_reason(pos)
                positions_to_close.append((pos, exit_price, exit_reason))
                continue

            if sl_hit:
                exit_price = pos.trailing_sl
                exit_reason = self._sl_exit_reason(pos)
                positions_to_close.append((pos, exit_price, exit_reason))
                continue

            if tgt_hit:
                exit_price = pos.target
                positions_to_close.append((pos, exit_price, "TARGET_HIT"))
                continue

            # --- Update trailing SL (mirrors RiskManager._apply_trailing_sl) ---
            self._apply_trailing_sl(pos, bar_close)

            # --- Check trailing SL after update (in case close itself triggers) ---
            if is_call and bar_close <= pos.trailing_sl:
                positions_to_close.append((pos, pos.trailing_sl, self._sl_exit_reason(pos)))
                continue
            if not is_call and bar_close >= pos.trailing_sl:
                positions_to_close.append((pos, pos.trailing_sl, self._sl_exit_reason(pos)))
                continue

            # --- Forced EOD exit ---
            if is_eod and self.config.intraday_only:
                positions_to_close.append((pos, bar_close, "FORCED_EOD"))
                continue

        # Execute all closes
        for pos, exit_price, reason in positions_to_close:
            closed = self.close_position(pos, exit_price, reason, timestamp)
            closed_this_bar.append(closed)

        # Record equity after all updates
        self._record_equity(timestamp)

        return closed_this_bar

    # ------------------------------------------------------------------
    # Public API — close a specific position
    # ------------------------------------------------------------------

    def close_position(
        self,
        position: TradeExecution,
        exit_price: float,
        exit_reason: str,
        exit_timestamp: datetime,
    ) -> ClosedTrade:
        """Close *position* at *exit_price*, computing full P&L."""
        cfg = self.config
        is_call = position.trade_type == "BUY_CALL"

        # --- Exit slippage (always adverse) ---
        if cfg.use_variable_spread:
            model = cfg.spread_model or SpreadModel()
            slippage = model.estimate_spread(
                index_id=position.index_id,
                spot_price=exit_price,
                strike_price=position.strike_price,
                option_premium=position.option_premium,
                days_to_expiry=None,  # Not tracked at exit
                is_expiry_day=self._is_expiry_day(exit_timestamp),
                hour_of_day=self._get_hour(exit_timestamp),
            )
        elif cfg.use_percentage_slippage:
            slippage = exit_price * cfg.slippage_pct / 100
        else:
            slippage = cfg.slippage_points

        exit_spread_cost = slippage  # Record for ClosedTrade

        if is_call:
            actual_exit = exit_price - slippage  # sell lower
        else:
            actual_exit = exit_price + slippage  # sell higher... adverse for PUT means you exit at higher index

        actual_exit = round(actual_exit, 2)

        # Guard: exit price should not be beyond bar bounds (sanity clamp)
        # (Not clamped to bar bounds here because exit_price is already from bar data)

        # --- Exit transaction cost ---
        turnover = abs(actual_exit - position.actual_entry_price) * position.quantity
        turnover = max(turnover, 1.0)  # avoid zero-turnover edge case
        exit_brokerage = cfg.brokerage_per_order
        exit_stt = turnover * cfg.stt_rate  # STT on sell side
        exit_exchange = turnover * cfg.exchange_charges_pct
        exit_gst = (exit_brokerage + exit_exchange) * cfg.gst_pct
        exit_sebi = turnover * cfg.sebi_charges_pct
        exit_cost = exit_brokerage + exit_stt + exit_exchange + exit_gst + exit_sebi

        # --- P&L ---
        if is_call:
            gross_pnl_points = actual_exit - position.actual_entry_price
        else:
            gross_pnl_points = position.actual_entry_price - actual_exit

        gross_pnl = gross_pnl_points * position.quantity
        total_costs = position.entry_cost + exit_cost
        net_pnl = gross_pnl - total_costs
        net_pnl_pct = net_pnl / self.current_capital * 100 if self.current_capital > 0 else 0.0

        # Duration
        duration_minutes = int((exit_timestamp - position.entry_timestamp).total_seconds() / 60)

        # Outcome
        if net_pnl > 0.5:
            outcome = "WIN"
        elif net_pnl < -0.5:
            outcome = "LOSS"
        else:
            outcome = "BREAKEVEN"

        closed = ClosedTrade(
            trade_id=position.trade_id,
            index_id=position.index_id,
            signal_id=position.signal_id,
            trade_type=position.trade_type,
            signal_entry_price=position.signal_entry_price,
            actual_entry_price=position.actual_entry_price,
            entry_timestamp=position.entry_timestamp,
            entry_bar=position.entry_bar,
            original_stop_loss=position.stop_loss,
            original_target=position.target,
            actual_exit_price=actual_exit,
            exit_timestamp=exit_timestamp,
            exit_reason=exit_reason,
            lots=position.lots,
            lot_size=position.lot_size,
            quantity=position.quantity,
            confidence_level=position.confidence_level,
            gross_pnl_points=round(gross_pnl_points, 2),
            gross_pnl=round(gross_pnl, 2),
            total_costs=round(total_costs, 2),
            net_pnl=round(net_pnl, 2),
            net_pnl_pct=round(net_pnl_pct, 4),
            duration_bars=position._bars_held,
            duration_minutes=duration_minutes,
            max_favorable_excursion=round(position._max_favorable, 2),
            max_adverse_excursion=round(position._max_adverse, 2),
            outcome=outcome,
            strike_price=position.strike_price,
            option_premium=position.option_premium,
            expiry=position.expiry,
            spread_cost_entry=position.spread_cost_entry,
            spread_cost_exit=round(exit_spread_cost, 2),
        )

        # --- Update state ---
        self.current_capital += net_pnl
        if position in self.open_positions:
            self.open_positions.remove(position)
        self.trade_history.append(closed)

        # Daily P&L tracking
        date_str = self._date_str(exit_timestamp)
        self._ensure_daily_tracker(date_str)
        self._daily_pnl[date_str] += net_pnl
        self._daily_trades[date_str] += 1

        # Update peak capital
        if self.current_capital > self._max_capital:
            self._max_capital = self.current_capital

        # Capital depletion check
        if self.current_capital <= 0:
            self._capital_depleted = True
            logger.error("CAPITAL DEPLETED at %s — trading halted", date_str)

        logger.info(
            "EXIT   %-12s %s  Entry=%.2f  Exit=%.2f  PnL=₹%.0f (%s)  [%s]",
            position.index_id, position.trade_type,
            position.actual_entry_price, actual_exit,
            net_pnl, outcome, exit_reason,
        )

        return closed

    # ------------------------------------------------------------------
    # Public API — portfolio & history
    # ------------------------------------------------------------------

    def get_portfolio_state(self) -> PortfolioState:
        """Return a point-in-time portfolio snapshot."""
        unrealized = self._compute_unrealized_pnl()
        realized = sum(t.net_pnl for t in self.trade_history)
        total_return = (
            (self.current_capital - self.config.initial_capital)
            / self.config.initial_capital * 100
        ) if self.config.initial_capital > 0 else 0.0

        today_str = self._current_date or ""
        today_pnl = self._daily_pnl.get(today_str, 0.0)
        today_trades = self._daily_trades.get(today_str, 0)

        daily_limit = self.config.initial_capital * self.config.max_risk_per_day_pct / 100
        daily_used = max(0.0, -today_pnl)
        daily_remaining = max(0.0, daily_limit - daily_used)

        drawdown = (
            (self._max_capital - self.current_capital) / self._max_capital * 100
        ) if self._max_capital > 0 else 0.0

        return PortfolioState(
            current_capital=round(self.current_capital, 2),
            initial_capital=self.config.initial_capital,
            total_return=round(total_return, 4),
            open_positions=list(self.open_positions),
            open_position_count=len(self.open_positions),
            unrealized_pnl=round(unrealized, 2),
            realized_pnl=round(realized, 2),
            total_trades=len(self.trade_history),
            today_pnl=round(today_pnl, 2),
            today_trades=today_trades,
            daily_loss_limit_remaining=round(daily_remaining, 2),
            max_capital_reached=round(self._max_capital, 2),
            current_drawdown=round(drawdown, 4),
        )

    def get_trade_history(self) -> list[ClosedTrade]:
        """Return all closed trades in chronological order."""
        return list(self.trade_history)

    def reset(self) -> None:
        """Reset everything to initial state for re-running."""
        self.current_capital = self.config.initial_capital
        self.open_positions.clear()
        self.trade_history.clear()
        self.equity_curve.clear()
        self._daily_pnl.clear()
        self._daily_trades.clear()
        self._current_date = None
        self._max_capital = self.config.initial_capital
        self._capital_depleted = False
        self._bar_signals.clear()

    # ------------------------------------------------------------------
    # Trailing SL — mirrors RiskManager._apply_trailing_sl exactly
    # ------------------------------------------------------------------

    def _apply_trailing_sl(self, pos: TradeExecution, current_price: float) -> None:
        """
        Move the trailing stop only in the favourable direction.

        Milestones (based on % of target distance achieved):
          > 50%  → SL to entry (breakeven)
          > 75%  → SL to entry + 30% of target distance
          > 100% → SL to entry + 60% of target distance
        """
        is_call = pos.trade_type == "BUY_CALL"
        entry = pos.actual_entry_price
        target = pos.target

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
            return  # not profitable enough to trail

        # Only ratchet — never widen
        if is_call:
            if new_sl > pos.trailing_sl:
                pos.trailing_sl = round(new_sl, 2)
                pos.stop_loss = pos.trailing_sl
        else:
            if new_sl < pos.trailing_sl:
                pos.trailing_sl = round(new_sl, 2)
                pos.stop_loss = pos.trailing_sl

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_lot_size(index_id: str) -> Optional[int]:
        """Look up lot_size from IndexRegistry."""
        registry = get_registry()
        index = registry.get_index(index_id)
        if index is None or index.lot_size is None:
            return None
        return index.lot_size

    @staticmethod
    def _bar_timestamp(bar: dict) -> datetime:
        """Extract timestamp from a bar dict."""
        ts = bar.get("timestamp") or bar.get("datetime") or bar.get("date")
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            return datetime.fromisoformat(ts)
        # Fallback — return now in IST
        return datetime.now(tz=_IST)

    @staticmethod
    def _date_str(ts: datetime) -> str:
        return ts.strftime("%Y-%m-%d")

    def _ensure_daily_tracker(self, date_str: str) -> None:
        if self._current_date != date_str:
            self._current_date = date_str
            # Reset bar-signal dedup on new day
            self._bar_signals.clear()
        self._daily_pnl.setdefault(date_str, 0.0)
        self._daily_trades.setdefault(date_str, 0)

    def _is_forced_exit_time(self, ts: datetime) -> bool:
        """True if timestamp is at or past forced-exit cutoff."""
        if not self.config.intraday_only:
            return False
        cutoff_minute = (
            _MARKET_CLOSE_HOUR * 60 + _MARKET_CLOSE_MINUTE
            - self.config.force_exit_minutes_before_close
        )
        bar_minute = ts.hour * 60 + ts.minute
        return bar_minute >= cutoff_minute

    def _sl_exit_reason(self, pos: TradeExecution) -> str:
        """Determine whether exit is via original SL or trailing SL."""
        is_call = pos.trade_type == "BUY_CALL"
        if is_call:
            return "TRAILING_SL_HIT" if pos.trailing_sl >= pos.actual_entry_price else "STOP_LOSS_HIT"
        else:
            return "TRAILING_SL_HIT" if pos.trailing_sl <= pos.actual_entry_price else "STOP_LOSS_HIT"

    @staticmethod
    def _is_expiry_day(ts: datetime) -> bool:
        """Check if timestamp falls on a weekly expiry day (Thursday for NSE)."""
        if not isinstance(ts, datetime):
            return False
        return ts.weekday() == 3  # Thursday

    @staticmethod
    def _get_hour(ts: datetime) -> Optional[int]:
        """Extract hour from timestamp."""
        if not isinstance(ts, datetime):
            return None
        return ts.hour

    def _compute_unrealized_pnl(self) -> float:
        """Sum unrealised P&L across open positions (using last known bar close)."""
        total = 0.0
        for pos in self.open_positions:
            last_close = pos.entry_bar.get("close", pos.actual_entry_price)
            if pos.trade_type == "BUY_CALL":
                total += (last_close - pos.actual_entry_price) * pos.quantity
            else:
                total += (pos.actual_entry_price - last_close) * pos.quantity
        return total

    def _record_equity(self, timestamp: datetime) -> None:
        """Append a single equity-curve observation."""
        unrealized = self._compute_unrealized_pnl()
        total = self.current_capital + unrealized

        peak = max(self._max_capital, total)
        dd = (peak - total) / peak * 100 if peak > 0 else 0.0

        self.equity_curve.append(EquityPoint(
            timestamp=timestamp,
            capital=round(total, 2),
            cash=round(self.current_capital, 2),
            unrealized=round(unrealized, 2),
            drawdown_pct=round(dd, 4),
            open_positions=len(self.open_positions),
        ))
