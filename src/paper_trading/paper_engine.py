"""
Paper Trading Engine — Phase 9.1 of the Trading Decision Support System.

Executes signals from the DecisionEngine against real-time market prices
without placing actual orders.  Key differences from backtesting:

- Data arrives in real time — never look-ahead
- Slippage uses the live SpreadModel on each entry and exit
- Position monitoring runs every 60 s against live prices
- All state persists to DB so restarts are seamless
- Missed signals are logged to answer: "are my risk filters too tight?"

Architecture
------------
::

    DecisionEngine.run_full_cycle()
        └─► PaperTradingEngine.on_signal(signal, current_market)
                └─► filter → risk check → price → create PaperPosition

    DataCollector (every 60 s)
        └─► PaperTradingEngine.update_positions(current_prices)
                └─► SL / target / trailing / EOD → PaperTrade
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional, Union
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import STRATEGY_VERSION
from src.analysis.options_pricing import OptionPnLEstimator
from src.backtest.trade_simulator import SpreadModel
from src.data.index_registry import get_registry
from src.database.db_manager import DatabaseManager
from src.engine.risk_manager import RefinedSignal
from src.engine.signal_generator import TradingSignal
from src.utils.date_utils import (
    get_current_expiry,
    get_ist_now,
    days_to_expiry as calc_days_to_expiry,
)

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# Confidence level ordering for comparison
_CONF_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# ── Market timing constants ─────────────────────────────────────────────────
_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MINUTE = 15
_WARMUP_END_MINUTE = 17        # Skip opening auction (9:15–9:17)
_NO_NEW_ENTRY_HOUR = 15
_NO_NEW_ENTRY_MINUTE = 20      # No new entries after 15:20
_FORCE_EXIT_HOUR = 15
_FORCE_EXIT_MINUTE = 25        # Close all positions by 15:25
_STALE_DATA_THRESHOLD_SECS = 300   # 5 minutes


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PaperTradingConfig:
    """All configurable parameters for the paper trading engine."""

    # Capital
    initial_capital: float = 100_000.0

    # Transaction cost model (mirrors SimulatorConfig)
    brokerage_per_order: float = 20.0
    stt_rate: float = 0.000625          # 0.0625% on sell-side options
    exchange_charges_pct: float = 0.00053
    gst_pct: float = 0.18
    sebi_charges_pct: float = 0.000001
    stamp_duty_pct: float = 0.00003

    # Slippage
    use_live_spread: bool = True
    fallback_slippage_points: float = 3.0

    # Position management
    max_risk_per_trade_pct: float = 2.0
    max_risk_per_day_pct: float = 5.0
    max_open_positions: int = 3
    intraday_only: bool = True

    # Confidence filter
    min_confidence: str = "HIGH"        # Start conservative; relax to MEDIUM later
    signal_types: list = field(default_factory=lambda: ["BUY_CALL", "BUY_PUT"])

    # Active indices
    active_indices: list = field(default_factory=lambda: ["NIFTY50", "BANKNIFTY"])

    # Sizing (mirrors RiskManager)
    max_lots_per_trade: int = 5
    transaction_cost_per_lot: float = 120.0
    high_confidence_size_pct: float = 100.0
    medium_confidence_size_pct: float = 60.0
    low_confidence_size_pct: float = 30.0

    # Tracking
    track_missed_signals: bool = True
    track_no_trade_reasons: bool = True

    # State persistence
    state_file: Optional[str] = None    # JSON path; None → use DB table

    # Greeks-based P&L (delta-adjusted instead of naive index movement)
    use_greeks_pnl: bool = True
    default_iv: Optional[float] = None  # Override per-index default IV


# ---------------------------------------------------------------------------
# Position and trade dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PaperPosition:
    """An open (unresolved) paper trade."""

    position_id: str
    signal_id: str
    index_id: str
    trade_type: str                     # "BUY_CALL" / "BUY_PUT"

    # Entry prices
    signal_entry_price: float           # What the signal said
    market_price_at_signal: float       # Actual market price when signal fired
    execution_price: float              # After slippage (what we "paid")
    entry_timestamp: datetime

    # Levels
    stop_loss: float
    original_stop_loss: float
    target: float
    trailing_sl: float

    # Size
    lots: int
    lot_size: int
    quantity: int                       # lots × lot_size

    # Option details (from RefinedSignal if available)
    strike: Optional[float]
    expiry: Optional[str]
    option_premium_at_entry: Optional[float]

    # Costs
    entry_cost: float

    # Risk
    risk_amount: float
    confidence: str
    regime: str

    # Live tracking (updated every tick)
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    bars_held: int = 0

    iv_at_entry: Optional[float] = None
    days_to_expiry_at_entry: Optional[float] = None

    status: str = "OPEN"
    opened_at: datetime = field(default_factory=get_ist_now)
    strategy_version: str = "1.0.0"


@dataclass
class PaperTrade:
    """A fully resolved paper trade with complete P&L breakdown."""

    # Identity
    position_id: str
    signal_id: str
    index_id: str
    trade_type: str

    # Entry
    signal_entry_price: float
    market_price_at_signal: float
    execution_price: float
    entry_timestamp: datetime

    # Levels at entry
    original_stop_loss: float
    original_target: float

    # Option details
    strike: Optional[float]
    expiry: Optional[str]
    option_premium_at_entry: Optional[float]

    # Size
    lots: int
    lot_size: int
    quantity: int

    # Exit
    exit_price: float                   # Signal/reference exit price
    actual_exit_price: float            # After exit slippage
    exit_timestamp: datetime
    exit_reason: str                    # TARGET_HIT / STOP_LOSS_HIT / TRAILING_SL_HIT / FORCED_EOD / MANUAL

    # P&L
    gross_pnl: float
    entry_cost: float
    exit_cost: float
    total_costs: float
    net_pnl: float
    net_pnl_pct: float                  # % of capital at entry

    # Trade quality
    max_favorable_excursion: float
    max_adverse_excursion: float
    capture_ratio: float                # net_pnl / (max_favorable × quantity)

    # Timing
    duration_seconds: int
    duration_bars: int

    # Context
    confidence: str
    regime: str
    risk_amount: float
    entry_cost_at_open: float

    outcome: str                        # "WIN" / "LOSS" / "BREAKEVEN"

    # Greeks P&L decomposition (populated when Greeks P&L is available)
    delta_pnl: Optional[float] = None
    theta_pnl: Optional[float] = None
    vega_pnl: Optional[float] = None
    naive_pnl: Optional[float] = None
    overestimation_pct: Optional[float] = None

    strategy_version: str = "1.0.0"


@dataclass
class PaperExecution:
    """Return value of on_signal() — confirmation of a paper trade entry."""

    position: PaperPosition
    slippage_applied: float
    entry_cost: float
    message: str


@dataclass
class MissedSignal:
    """A signal that passed confidence filters but couldn't be traded."""

    timestamp: datetime
    index_id: str
    signal_type: str
    confidence: str
    reason_missed: str                  # "daily_loss_limit" / "max_positions" / "insufficient_capital" / etc.
    signal_entry_price: float
    would_have_been_target: float
    would_have_been_sl: float
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class DailyRecord:
    """Internal per-day accounting ledger."""

    date: date
    starting_capital: float
    realized_pnl: float = 0.0
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    signals_generated: int = 0
    signals_filtered: int = 0          # Blocked by confidence/index filter
    signals_executed: int = 0
    signals_missed: int = 0            # Passed filters, blocked by risk checks
    max_drawdown_today: float = 0.0    # Peak-to-trough intraday
    peak_capital_today: float = 0.0
    anomaly_alerts_count: int = 0


@dataclass
class DailyPaperSummary:
    """End-of-day (or intraday) paper trading summary."""

    date: date
    starting_capital: float
    ending_capital: float

    trades_taken: int
    trades_won: int
    trades_lost: int
    win_rate: float

    total_pnl: float
    total_pnl_pct: float

    best_trade_pnl: float
    worst_trade_pnl: float

    signals_generated: int
    signals_filtered: int
    signals_executed: int
    signals_missed: int

    expected_daily_return: Optional[float]

    risk_used_pct: float
    max_drawdown_today: float

    open_positions_eod: int

    # System health
    data_freshness_ok: bool
    system_mode: str
    anomaly_alerts_count: int


@dataclass
class PaperTradingStats:
    """Cumulative paper trading statistics over N days."""

    period_days: int
    trading_days: int

    # Capital
    initial_capital: float
    current_capital: float
    total_return_pct: float
    annualized_return_pct: float

    # Trades
    total_trades: int
    win_rate: float
    profit_factor: float
    expected_value: float
    avg_win: float
    avg_loss: float

    # Risk
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float

    # Consistency
    profitable_days: int
    unprofitable_days: int
    daily_win_rate: float
    max_consecutive_losing_days: int

    # Confidence breakdown
    high_conf_trades: int
    high_conf_win_rate: float
    medium_conf_trades: int
    medium_conf_win_rate: float

    # Comparison with backtest
    backtest_expected_return: Optional[float]
    live_vs_backtest_gap: Optional[float]
    edge_status: str                    # "INTACT" / "DEGRADING" / "GONE"

    # Operational
    system_uptime_pct: float
    data_availability_pct: float
    total_signals: int
    signal_to_trade_ratio: float


# ---------------------------------------------------------------------------
# Paper Trading Engine
# ---------------------------------------------------------------------------


class PaperTradingEngine:
    """
    Simulates live order execution against real-time market prices.

    Wraps the DecisionEngine signal pipeline:
      1. ``on_signal()``       — called by the engine when a signal fires
      2. ``update_positions()`` — called every 60 s by the data collector
      3. ``get_daily_summary()`` / ``get_cumulative_stats()`` — reporting
      4. ``save_state()`` / ``load_state()`` — survive restarts

    Parameters
    ----------
    db:
        Shared DatabaseManager instance.
    config:
        PaperTradingConfig; defaults are conservative (HIGH confidence only).
    telegram_bot:
        Optional TelegramBot instance for trade alerts.
    """

    # ------------------------------------------------------------------
    # Construction & initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        db: DatabaseManager,
        config: Optional[PaperTradingConfig] = None,
        telegram_bot=None,
    ) -> None:
        self._db = db
        self.config = config or PaperTradingConfig()
        self._telegram = telegram_bot
        self._spread_model = SpreadModel()

        # State
        self.current_capital: float = self.config.initial_capital
        self.open_positions: dict[str, PaperPosition] = {}
        self.trade_history: list[PaperTrade] = []
        self.daily_ledger: dict[date, DailyRecord] = {}
        self.missed_signals: list[MissedSignal] = []

        # Kill switch (set externally by DecisionEngine or manually)
        self._kill_switch_active: bool = False
        self._kill_switch_reason: str = ""

        # Warm-up flag: skip signals until past opening auction
        self._warmed_up: bool = False

        # Data freshness
        self._last_price_update: Optional[datetime] = None

        # System uptime tracking
        self._session_start: datetime = get_ist_now()
        self._uptime_minutes_total: float = 0.0
        self._uptime_minutes_active: float = 0.0

        # Peak capital (for drawdown tracking)
        self._peak_capital: float = self.config.initial_capital

        # Benchmark buy-and-hold tracking
        self._benchmark_start_price: Optional[float] = None
        self._benchmark_current_price: Optional[float] = None
        self._benchmark_index: str = "NIFTY50"

        # Strategy version — alert when it changes between sessions
        self._active_strategy_version: str = STRATEGY_VERSION

        # Ensure the persistence table exists
        self._ensure_schema()

        # Load persisted state if it exists and is from today
        self.load_state()

        logger.info(
            "PaperTradingEngine initialised — capital ₹%.0f, config: "
            "min_conf=%s, indices=%s",
            self.current_capital,
            self.config.min_confidence,
            self.config.active_indices,
        )

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def on_signal(
        self,
        signal: Union[TradingSignal, RefinedSignal],
        current_market: dict,
    ) -> Optional[PaperExecution]:
        """
        Evaluate a signal and, if it passes all checks, open a paper position.

        Parameters
        ----------
        signal:
            TradingSignal or RefinedSignal from the DecisionEngine.
        current_market:
            Live market snapshot::

                {
                    "index_id": "NIFTY50",
                    "ltp":      22_450.0,
                    "bid":      22_449.5,   # optional
                    "ask":      22_450.5,   # optional
                    "timestamp": datetime,
                    "option_premium": 185.0,  # optional
                    "options_chain_snapshot": ...,  # optional
                }

        Returns
        -------
        PaperExecution on success, None if trade not taken (with reason logged).
        """
        now = get_ist_now()

        # ── Version change detection ─────────────────────────────────────
        if STRATEGY_VERSION != self._active_strategy_version:
            msg = (
                f"📌 Strategy version changed: {self._active_strategy_version} → {STRATEGY_VERSION}. "
                f"Performance will be tracked separately."
            )
            logger.warning("Strategy version changed: %s → %s", self._active_strategy_version, STRATEGY_VERSION)
            self._send_alert(msg)
            self._active_strategy_version = STRATEGY_VERSION

        today = now.date()
        self._ensure_daily_record(today)
        record = self.daily_ledger[today]
        record.signals_generated += 1

        # ── Step 1: Hard filters ────────────────────────────────────────
        sig_type = getattr(signal, "signal_type", "")
        index_id = getattr(signal, "index_id", "")
        confidence = getattr(signal, "confidence_level", "LOW")
        signal_id = getattr(signal, "signal_id", str(uuid.uuid4()))

        if sig_type not in self.config.signal_types:
            self._log_skip(record, signal, "signal_type_filtered",
                           f"signal type {sig_type!r} not in config")
            return None

        if index_id not in self.config.active_indices:
            self._log_skip(record, signal, "index_filtered",
                           f"index {index_id!r} not active")
            return None

        if _CONF_ORDER.get(confidence, 0) < _CONF_ORDER.get(self.config.min_confidence, 1):
            record.signals_filtered += 1
            self._maybe_track_missed(signal, "below_min_confidence", current_market)
            logger.info(
                "PAPER SKIP [%s %s] confidence=%s < min=%s",
                index_id, sig_type, confidence, self.config.min_confidence,
            )
            return None

        if self._kill_switch_active:
            self._log_skip(record, signal, "kill_switch",
                           f"kill switch active: {self._kill_switch_reason}")
            return None

        if not self._is_warmed_up(now):
            self._log_skip(record, signal, "warmup",
                           "system in warm-up (opening auction)")
            return None

        if self._is_too_late_for_entry(now):
            self._log_skip(record, signal, "too_late",
                           f"past entry cutoff ({_NO_NEW_ENTRY_HOUR}:{_NO_NEW_ENTRY_MINUTE:02d})")
            return None

        # ── Step 2: Risk checks ─────────────────────────────────────────
        daily_loss_limit = self.config.initial_capital * self.config.max_risk_per_day_pct / 100
        daily_realized = record.realized_pnl
        if daily_realized <= -daily_loss_limit:
            self._maybe_track_missed(signal, "daily_loss_limit", current_market)
            record.signals_missed += 1
            logger.info("PAPER SKIP [%s] daily loss limit reached (₹%.0f)", index_id, daily_realized)
            return None

        if len(self.open_positions) >= self.config.max_open_positions:
            self._maybe_track_missed(signal, "max_positions", current_market)
            record.signals_missed += 1
            logger.info("PAPER SKIP [%s] max open positions (%d) reached", index_id, self.config.max_open_positions)
            return None

        existing_for_index = [p for p in self.open_positions.values() if p.index_id == index_id]
        if existing_for_index:
            self._maybe_track_missed(signal, "existing_position_in_index", current_market)
            record.signals_missed += 1
            logger.info("PAPER SKIP [%s] already have open position in this index", index_id)
            return None

        # ── Step 3: Execution price (slippage) ──────────────────────────
        ltp = float(current_market.get("ltp", 0.0))
        if ltp <= 0:
            logger.warning("PAPER SKIP [%s] invalid LTP: %s", index_id, ltp)
            return None

        option_premium = current_market.get("option_premium") or getattr(signal, "option_premium", None)
        strike = getattr(signal, "recommended_strike", None) or getattr(signal, "strike", None)
        expiry_date = get_current_expiry(index_id)
        dte = calc_days_to_expiry(index_id)

        if self.config.use_live_spread:
            try:
                spread = self._spread_model.estimate_spread(
                    index_id=index_id,
                    spot_price=ltp,
                    strike_price=strike,
                    option_premium=option_premium,
                    days_to_expiry=dte,
                    is_expiry_day=(now.date() == expiry_date),
                    hour_of_day=now.hour,
                )
            except Exception as exc:
                logger.warning("SpreadModel failed (%s) — using fallback slippage", exc)
                spread = self.config.fallback_slippage_points
        else:
            spread = self.config.fallback_slippage_points

        signal_entry = getattr(signal, "refined_entry", 0.0) or getattr(signal, "entry_price", ltp)
        if sig_type == "BUY_CALL":
            execution_price = ltp + spread
        else:
            execution_price = ltp - spread

        execution_price = round(execution_price, 2)

        # ── Step 4: Position sizing ─────────────────────────────────────
        stop_loss = getattr(signal, "refined_stop_loss", 0.0) or getattr(signal, "stop_loss", 0.0)
        target = getattr(signal, "refined_target", 0.0) or getattr(signal, "target_price", 0.0)

        lot_size = self._get_lot_size(index_id)
        if lot_size is None:
            logger.error("PAPER SKIP [%s] no lot size in registry", index_id)
            return None

        max_risk = self.current_capital * self.config.max_risk_per_trade_pct / 100
        conf_pct_map = {
            "HIGH": self.config.high_confidence_size_pct,
            "MEDIUM": self.config.medium_confidence_size_pct,
            "LOW": self.config.low_confidence_size_pct,
        }
        conf_pct = conf_pct_map.get(confidence, self.config.low_confidence_size_pct)
        pos_modifier = getattr(signal, "position_size_modifier", 1.0)
        risk_amount = max(max_risk * (conf_pct / 100.0) * pos_modifier, 0.0)

        sl_distance = abs(execution_price - stop_loss)
        loss_per_lot = sl_distance * lot_size + self.config.transaction_cost_per_lot
        if loss_per_lot <= 0:
            lots = 1
        else:
            lots = math.floor(risk_amount / loss_per_lot)
            lots = max(lots, 1)
            lots = min(lots, self.config.max_lots_per_trade)

        quantity = lots * lot_size
        min_margin = sl_distance * lot_size

        if min_margin > self.current_capital:
            self._maybe_track_missed(signal, "insufficient_capital", current_market)
            record.signals_missed += 1
            logger.info(
                "PAPER SKIP [%s] insufficient capital: need ₹%.0f, have ₹%.0f",
                index_id, min_margin, self.current_capital,
            )
            return None

        # ── Step 5: Entry transaction cost ──────────────────────────────
        turnover = sl_distance * quantity
        entry_cost = self._calc_entry_cost(turnover)

        # ── Step 6: Create paper position ───────────────────────────────
        pos_id = str(uuid.uuid4())
        regime = getattr(signal, "regime", "UNKNOWN")
        expiry_str = getattr(signal, "recommended_expiry", None) or expiry_date.strftime("%d-%b-%Y").upper()

        iv_at_entry = getattr(signal, "entry_iv", None) or getattr(signal, "iv_at_entry", None)

        position = PaperPosition(
            position_id=pos_id,
            signal_id=signal_id,
            index_id=index_id,
            trade_type=sig_type,
            signal_entry_price=round(signal_entry, 2),
            market_price_at_signal=round(ltp, 2),
            execution_price=execution_price,
            entry_timestamp=now,
            stop_loss=round(stop_loss, 2),
            original_stop_loss=round(stop_loss, 2),
            target=round(target, 2),
            trailing_sl=round(stop_loss, 2),
            lots=lots,
            lot_size=lot_size,
            quantity=quantity,
            strike=strike,
            expiry=expiry_str,
            option_premium_at_entry=option_premium,
            iv_at_entry=iv_at_entry,
            days_to_expiry_at_entry=float(dte) if dte is not None else None,
            entry_cost=round(entry_cost, 2),
            risk_amount=round(risk_amount, 2),
            confidence=confidence,
            regime=regime,
            current_price=ltp,
            unrealized_pnl=0.0,
            max_favorable=0.0,
            max_adverse=0.0,
            bars_held=0,
            status="OPEN",
            opened_at=now,
            strategy_version=STRATEGY_VERSION,
        )

        # Deduct margin from available capital
        margin_held = execution_price * quantity * 0.1  # approximate option margin
        self.current_capital = max(0.0, self.current_capital - entry_cost)

        self.open_positions[pos_id] = position
        record.signals_executed += 1

        alert_msg = (
            f"📝 PAPER TRADE: {sig_type.replace('_', ' ')} {index_id} @ "
            f"{execution_price:,.0f} ({lots} lot{'s' if lots > 1 else ''}) | "
            f"SL: {stop_loss:,.0f}  Tgt: {target:,.0f} | "
            f"Risk: ₹{risk_amount:,.0f}"
        )
        logger.info(
            "PAPER ENTRY: %s %s @ %.2f (signal: %.2f, market: %.2f) | "
            "%d lot%s | Risk: ₹%.0f | Spread: %.2f pts",
            sig_type, index_id, execution_price,
            signal_entry, ltp, lots, "s" if lots > 1 else "",
            risk_amount, spread,
        )
        self._send_alert(alert_msg)
        self._queue_ws_event("paper_entry", {
            "position_id": pos_id,
            "index_id": index_id,
            "trade_type": sig_type,
            "execution_price": execution_price,
            "lots": lots,
            "stop_loss": stop_loss,
            "target": target,
            "confidence": confidence,
        })

        self.save_state()

        return PaperExecution(
            position=position,
            slippage_applied=round(spread, 2),
            entry_cost=round(entry_cost, 2),
            message=alert_msg,
        )

    def update_positions(self, current_prices: dict[str, float]) -> list[PaperTrade]:
        """
        Check all open positions against current prices; close those that
        hit SL, target, trailing SL, or the forced EOD cutoff.

        Parameters
        ----------
        current_prices:
            ``{"NIFTY50": 22_485.0, "BANKNIFTY": 48_200.0, ...}``

        Returns
        -------
        List of PaperTrade objects for positions closed this cycle.
        """
        now = get_ist_now()

        # Guard against stale data
        if self._is_data_stale(now):
            logger.warning("update_positions: data is stale — skipping SL/target checks")
            return []

        self._last_price_update = now

        # Update benchmark price (use NIFTY50 as primary benchmark)
        for bm_candidate in ["NIFTY50", "BANKNIFTY"]:
            if bm_candidate in current_prices and current_prices[bm_candidate] > 0:
                self._update_benchmark_price(float(current_prices[bm_candidate]), bm_candidate)
                break

        today = now.date()
        self._ensure_daily_record(today)
        record = self.daily_ledger[today]

        is_eod = self._is_forced_exit_time(now)
        positions_to_close: list[tuple[PaperPosition, float, str]] = []

        for pos in list(self.open_positions.values()):
            idx_price = current_prices.get(pos.index_id)
            if idx_price is None or idx_price <= 0:
                logger.debug("No price for %s — skipping position %s", pos.index_id, pos.position_id)
                continue

            current_price = float(idx_price)
            is_call = pos.trade_type == "BUY_CALL"

            # ── Update tracking ─────────────────────────────────────────
            pos.current_price = current_price
            pos.bars_held += 1

            if is_call:
                favorable_move = current_price - pos.execution_price
                adverse_move = pos.execution_price - current_price
            else:
                favorable_move = pos.execution_price - current_price
                adverse_move = current_price - pos.execution_price

            pos.max_favorable = max(pos.max_favorable, favorable_move)
            pos.max_adverse = max(pos.max_adverse, adverse_move)

            # Unrealised P&L (approximate — uses index points, not option premium)
            if is_call:
                pos.unrealized_pnl = (current_price - pos.execution_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.execution_price - current_price) * pos.quantity

            # ── Check SL ────────────────────────────────────────────────
            sl_hit = (
                (is_call and current_price <= pos.trailing_sl) or
                (not is_call and current_price >= pos.trailing_sl)
            )
            # ── Check target ────────────────────────────────────────────
            tgt_hit = (
                (is_call and current_price >= pos.target) or
                (not is_call and current_price <= pos.target)
            )

            # If both in same update cycle → conservative: assume SL first
            if sl_hit and tgt_hit:
                reason = self._sl_exit_reason(pos)
                positions_to_close.append((pos, pos.trailing_sl, reason))
                continue

            if sl_hit:
                reason = self._sl_exit_reason(pos)
                positions_to_close.append((pos, pos.trailing_sl, reason))
                continue

            if tgt_hit:
                positions_to_close.append((pos, pos.target, "TARGET_HIT"))
                continue

            # ── Update trailing SL ───────────────────────────────────────
            self._apply_trailing_sl(pos, current_price)

            # ── Re-check after trailing SL update ───────────────────────
            trailing_triggered = (
                (is_call and current_price <= pos.trailing_sl) or
                (not is_call and current_price >= pos.trailing_sl)
            )
            if trailing_triggered:
                positions_to_close.append((pos, pos.trailing_sl, "TRAILING_SL_HIT"))
                continue

            # ── Forced EOD exit ──────────────────────────────────────────
            if is_eod and self.config.intraday_only:
                positions_to_close.append((pos, current_price, "FORCED_EOD"))
                continue

        # Execute all closes
        closed_trades: list[PaperTrade] = []
        for pos, exit_price, reason in positions_to_close:
            trade = self.close_position(pos.position_id, exit_price, reason)
            if trade:
                closed_trades.append(trade)

        # Update daily drawdown
        if record.peak_capital_today < self.current_capital:
            record.peak_capital_today = self.current_capital
        current_drawdown = record.peak_capital_today - self.current_capital
        if current_drawdown > record.max_drawdown_today:
            record.max_drawdown_today = current_drawdown

        if closed_trades:
            self.save_state()

        return closed_trades

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[PaperTrade]:
        """
        Close a paper position and record the completed trade.

        Parameters
        ----------
        position_id:
            UUID of the open PaperPosition.
        exit_price:
            Reference exit price (before slippage).
        exit_reason:
            One of: TARGET_HIT / STOP_LOSS_HIT / TRAILING_SL_HIT / FORCED_EOD / MANUAL

        Returns
        -------
        PaperTrade on success, None if position not found.
        """
        pos = self.open_positions.get(position_id)
        if pos is None:
            logger.warning("close_position: position %s not found", position_id)
            return None

        now = get_ist_now()
        is_call = pos.trade_type == "BUY_CALL"

        # ── Exit slippage ────────────────────────────────────────────────
        if self.config.use_live_spread:
            try:
                exit_spread = self._spread_model.estimate_spread(
                    index_id=pos.index_id,
                    spot_price=exit_price,
                    strike_price=pos.strike,
                    option_premium=pos.option_premium_at_entry,
                    days_to_expiry=None,
                    is_expiry_day=(now.date() == get_current_expiry(pos.index_id)),
                    hour_of_day=now.hour,
                )
            except Exception as exc:
                logger.warning("Exit SpreadModel failed (%s) — using fallback", exc)
                exit_spread = self.config.fallback_slippage_points
        else:
            exit_spread = self.config.fallback_slippage_points

        # Exit is adverse: sell CALL lower, cover PUT higher
        if is_call:
            actual_exit_price = exit_price - exit_spread
        else:
            actual_exit_price = exit_price + exit_spread
        actual_exit_price = round(actual_exit_price, 2)

        # ── Exit costs ───────────────────────────────────────────────────
        turnover = abs(actual_exit_price - pos.execution_price) * pos.quantity
        turnover = max(turnover, 1.0)
        exit_cost = self._calc_exit_cost(turnover)

        # ── Gross P&L ────────────────────────────────────────────────────
        delta_pnl: Optional[float] = None
        theta_pnl: Optional[float] = None
        vega_pnl: Optional[float] = None
        naive_pnl: Optional[float] = None
        overestimation_pct: Optional[float] = None

        if self.config.use_greeks_pnl and pos.strike and pos.days_to_expiry_at_entry:
            try:
                estimator = OptionPnLEstimator()
                iv = (
                    pos.iv_at_entry
                    or self.config.default_iv
                    or estimator.get_default_iv(pos.index_id)
                )
                duration_days = pos.bars_held * (1.0 / 78.0)  # 75-min bars → days
                dte_at_exit = max(pos.days_to_expiry_at_entry - duration_days, 0.01)

                option_pnl = estimator.estimate_option_pnl(
                    trade_type=pos.trade_type,
                    spot_at_entry=pos.execution_price,
                    spot_at_exit=actual_exit_price,
                    strike=pos.strike,
                    entry_iv=iv,
                    exit_iv=None,
                    days_to_expiry_at_entry=pos.days_to_expiry_at_entry,
                    days_to_expiry_at_exit=dte_at_exit,
                    lot_size=pos.lot_size,
                    lots=pos.lots,
                )
                gross_pnl = option_pnl.gross_pnl
                delta_pnl = option_pnl.delta_pnl * pos.quantity
                theta_pnl = option_pnl.theta_pnl * pos.quantity
                vega_pnl = option_pnl.vega_pnl * pos.quantity
                naive_pnl = option_pnl.naive_pnl
                overestimation_pct = option_pnl.overestimation_pct
            except Exception as exc:
                logger.warning("Greeks P&L failed (%s) — falling back to naive", exc)
                if is_call:
                    gross_pnl = (actual_exit_price - pos.execution_price) * pos.quantity
                else:
                    gross_pnl = (pos.execution_price - actual_exit_price) * pos.quantity
        else:
            if is_call:
                gross_pnl = (actual_exit_price - pos.execution_price) * pos.quantity
            else:
                gross_pnl = (pos.execution_price - actual_exit_price) * pos.quantity

        total_costs = pos.entry_cost + exit_cost
        net_pnl = gross_pnl - total_costs
        net_pnl_pct = net_pnl / self.config.initial_capital * 100 if self.config.initial_capital > 0 else 0.0

        # ── Trade quality ─────────────────────────────────────────────────
        max_favorable_pts = pos.max_favorable
        max_favorable_value = max_favorable_pts * pos.quantity
        capture_ratio = (net_pnl / max_favorable_value) if max_favorable_value > 0.5 else 0.0

        duration_secs = int((now - pos.entry_timestamp).total_seconds())

        if net_pnl > 0.5:
            outcome = "WIN"
        elif net_pnl < -0.5:
            outcome = "LOSS"
        else:
            outcome = "BREAKEVEN"

        trade = PaperTrade(
            position_id=pos.position_id,
            signal_id=pos.signal_id,
            index_id=pos.index_id,
            trade_type=pos.trade_type,
            signal_entry_price=pos.signal_entry_price,
            market_price_at_signal=pos.market_price_at_signal,
            execution_price=pos.execution_price,
            entry_timestamp=pos.entry_timestamp,
            original_stop_loss=pos.original_stop_loss,
            original_target=pos.target,
            strike=pos.strike,
            expiry=pos.expiry,
            option_premium_at_entry=pos.option_premium_at_entry,
            lots=pos.lots,
            lot_size=pos.lot_size,
            quantity=pos.quantity,
            exit_price=round(exit_price, 2),
            actual_exit_price=actual_exit_price,
            exit_timestamp=now,
            exit_reason=exit_reason,
            gross_pnl=round(gross_pnl, 2),
            entry_cost=pos.entry_cost,
            exit_cost=round(exit_cost, 2),
            total_costs=round(total_costs, 2),
            net_pnl=round(net_pnl, 2),
            net_pnl_pct=round(net_pnl_pct, 4),
            max_favorable_excursion=round(max_favorable_pts, 2),
            max_adverse_excursion=round(pos.max_adverse, 2),
            capture_ratio=round(capture_ratio, 4),
            duration_seconds=duration_secs,
            duration_bars=pos.bars_held,
            confidence=pos.confidence,
            regime=pos.regime,
            risk_amount=pos.risk_amount,
            entry_cost_at_open=pos.entry_cost,
            outcome=outcome,
            delta_pnl=round(delta_pnl, 2) if delta_pnl is not None else None,
            theta_pnl=round(theta_pnl, 2) if theta_pnl is not None else None,
            vega_pnl=round(vega_pnl, 2) if vega_pnl is not None else None,
            naive_pnl=round(naive_pnl, 2) if naive_pnl is not None else None,
            overestimation_pct=round(overestimation_pct, 2) if overestimation_pct is not None else None,
            strategy_version=pos.strategy_version,
        )

        # ── Update state ──────────────────────────────────────────────────
        self.current_capital += net_pnl
        if self.current_capital > self._peak_capital:
            self._peak_capital = self.current_capital

        del self.open_positions[position_id]
        self.trade_history.append(trade)

        # Update daily record
        today = now.date()
        self._ensure_daily_record(today)
        rec = self.daily_ledger[today]
        rec.realized_pnl += net_pnl
        rec.trades_taken += 1
        if outcome == "WIN":
            rec.trades_won += 1
        elif outcome == "LOSS":
            rec.trades_lost += 1

        # Persist to DB (update signal outcome)
        self._persist_trade(trade)

        # Alerts
        icon = "✅" if outcome == "WIN" else ("❌" if outcome == "LOSS" else "➖")
        pnl_sign = "+" if net_pnl >= 0 else ""
        alert_msg = (
            f"📤 PAPER EXIT: {pos.index_id} "
            f"{'CALL' if is_call else 'PUT'} @ {actual_exit_price:,.0f} | "
            f"{pnl_sign}₹{net_pnl:,.0f} ({exit_reason.replace('_', ' ')} {icon})"
        )
        logger.info(
            "PAPER EXIT: %s %s  Entry=%.2f  Exit=%.2f  Net=₹%.0f (%s) [%s]",
            pos.index_id, pos.trade_type,
            pos.execution_price, actual_exit_price,
            net_pnl, outcome, exit_reason,
        )
        self._send_alert(alert_msg, priority="HIGH" if abs(net_pnl) > 1000 else "NORMAL")
        self._queue_ws_event("paper_exit", {
            "position_id": position_id,
            "index_id": pos.index_id,
            "trade_type": pos.trade_type,
            "entry_price": pos.execution_price,
            "exit_price": actual_exit_price,
            "net_pnl": trade.net_pnl,
            "outcome": outcome,
            "exit_reason": exit_reason,
            "duration_seconds": duration_secs,
        })

        return trade

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_daily_summary(self, for_date: Optional[date] = None) -> DailyPaperSummary:
        """Return a summary for *for_date* (defaults to today)."""
        today = for_date or get_ist_now().date()
        self._ensure_daily_record(today)
        rec = self.daily_ledger[today]

        # Today's completed trades
        today_trades = [
            t for t in self.trade_history
            if t.exit_timestamp.date() == today
        ]
        total_pnl = sum(t.net_pnl for t in today_trades)
        best = max((t.net_pnl for t in today_trades), default=0.0)
        worst = min((t.net_pnl for t in today_trades), default=0.0)

        win_rate = (rec.trades_won / rec.trades_taken) if rec.trades_taken > 0 else 0.0

        total_pnl_pct = total_pnl / rec.starting_capital * 100 if rec.starting_capital > 0 else 0.0
        ending_capital = rec.starting_capital + total_pnl

        daily_limit = self.config.initial_capital * self.config.max_risk_per_day_pct / 100
        risk_used_pct = (
            abs(min(rec.realized_pnl, 0.0)) / daily_limit * 100
        ) if daily_limit > 0 else 0.0

        # Data freshness check
        data_fresh = (
            self._last_price_update is not None and
            (get_ist_now() - self._last_price_update).total_seconds() < _STALE_DATA_THRESHOLD_SECS
        )

        return DailyPaperSummary(
            date=today,
            starting_capital=rec.starting_capital,
            ending_capital=ending_capital,
            trades_taken=rec.trades_taken,
            trades_won=rec.trades_won,
            trades_lost=rec.trades_lost,
            win_rate=round(win_rate, 4),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 4),
            best_trade_pnl=round(best, 2),
            worst_trade_pnl=round(worst, 2),
            signals_generated=rec.signals_generated,
            signals_filtered=rec.signals_filtered,
            signals_executed=rec.signals_executed,
            signals_missed=rec.signals_missed,
            expected_daily_return=None,             # Populated externally from backtest metrics
            risk_used_pct=round(risk_used_pct, 2),
            max_drawdown_today=round(rec.max_drawdown_today, 2),
            open_positions_eod=len(self.open_positions),
            data_freshness_ok=data_fresh,
            system_mode="FULL",
            anomaly_alerts_count=rec.anomaly_alerts_count,
        )

    def get_cumulative_stats(self, days: int = 30) -> PaperTradingStats:
        """Compute cumulative performance statistics over the last *days* days."""
        now = get_ist_now()
        cutoff = now.date() - timedelta(days=days)

        recent_trades = [
            t for t in self.trade_history
            if t.exit_timestamp.date() >= cutoff
        ]
        total = len(recent_trades)
        wins = [t for t in recent_trades if t.outcome == "WIN"]
        losses = [t for t in recent_trades if t.outcome == "LOSS"]

        win_rate = len(wins) / total if total > 0 else 0.0
        avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0.0

        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Capital metrics
        total_return_pct = (
            (self.current_capital - self.config.initial_capital) /
            self.config.initial_capital * 100
        ) if self.config.initial_capital > 0 else 0.0

        # Drawdown
        max_dd_pct = self._compute_max_drawdown_pct(recent_trades)

        # Sharpe / Sortino
        sharpe = self._compute_sharpe(recent_trades)
        sortino = self._compute_sortino(recent_trades)

        # Daily win/loss stats
        trading_days = len([d for d in self.daily_ledger if d >= cutoff])
        profitable_days = len([
            d for d, rec in self.daily_ledger.items()
            if d >= cutoff and rec.realized_pnl > 0
        ])
        unprofitable_days = len([
            d for d, rec in self.daily_ledger.items()
            if d >= cutoff and rec.realized_pnl < 0
        ])
        daily_win_rate = profitable_days / trading_days if trading_days > 0 else 0.0

        max_consec_loss = self._compute_max_consecutive_losing_days(cutoff)

        # Confidence breakdown
        high_trades = [t for t in recent_trades if t.confidence == "HIGH"]
        med_trades = [t for t in recent_trades if t.confidence == "MEDIUM"]
        high_wins = sum(1 for t in high_trades if t.outcome == "WIN")
        med_wins = sum(1 for t in med_trades if t.outcome == "WIN")

        # Annualised return
        period_trading_days = max(trading_days, 1)
        annualized = total_return_pct * (252 / period_trading_days) if period_trading_days > 0 else 0.0

        # Signal stats
        total_signals = sum(rec.signals_generated for rec in self.daily_ledger.values())
        signal_to_trade = total / total_signals if total_signals > 0 else 0.0

        return PaperTradingStats(
            period_days=days,
            trading_days=trading_days,
            initial_capital=self.config.initial_capital,
            current_capital=round(self.current_capital, 2),
            total_return_pct=round(total_return_pct, 4),
            annualized_return_pct=round(annualized, 4),
            total_trades=total,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            expected_value=round(expected_value, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            max_drawdown_pct=round(max_dd_pct, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            profitable_days=profitable_days,
            unprofitable_days=unprofitable_days,
            daily_win_rate=round(daily_win_rate, 4),
            max_consecutive_losing_days=max_consec_loss,
            high_conf_trades=len(high_trades),
            high_conf_win_rate=round(high_wins / len(high_trades), 4) if high_trades else 0.0,
            medium_conf_trades=len(med_trades),
            medium_conf_win_rate=round(med_wins / len(med_trades), 4) if med_trades else 0.0,
            backtest_expected_return=None,
            live_vs_backtest_gap=None,
            edge_status=self._evaluate_edge_status(win_rate, profit_factor),
            system_uptime_pct=self._uptime_pct(),
            data_availability_pct=self._data_availability_pct(),
            total_signals=total_signals,
            signal_to_trade_ratio=round(signal_to_trade, 4),
        )

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist current state to DB so the engine survives restarts."""
        now = get_ist_now()
        open_pos_json = json.dumps([self._position_to_dict(p) for p in self.open_positions.values()])

        # Daily ledger (serialise dates as strings)
        ledger_json = json.dumps({
            d.isoformat(): {
                "starting_capital": rec.starting_capital,
                "realized_pnl": rec.realized_pnl,
                "trades_taken": rec.trades_taken,
                "trades_won": rec.trades_won,
                "trades_lost": rec.trades_lost,
                "signals_generated": rec.signals_generated,
                "signals_filtered": rec.signals_filtered,
                "signals_executed": rec.signals_executed,
                "signals_missed": rec.signals_missed,
                "max_drawdown_today": rec.max_drawdown_today,
                "peak_capital_today": rec.peak_capital_today,
                "anomaly_alerts_count": rec.anomaly_alerts_count,
            }
            for d, rec in self.daily_ledger.items()
        })

        try:
            existing = self._db.fetch_one(
                "SELECT id FROM paper_trading_state WHERE id = 1", ()
            )
            if existing:
                self._db.execute(
                    """UPDATE paper_trading_state
                       SET current_capital=?, open_positions_json=?,
                           daily_ledger_json=?, updated_at=?
                       WHERE id=1""",
                    (self.current_capital, open_pos_json, ledger_json, now.isoformat()),
                )
            else:
                self._db.execute(
                    """INSERT INTO paper_trading_state
                       (id, current_capital, open_positions_json, daily_ledger_json, updated_at)
                       VALUES (1, ?, ?, ?, ?)""",
                    (self.current_capital, open_pos_json, ledger_json, now.isoformat()),
                )
        except Exception as exc:
            logger.error("save_state failed: %s", exc)

    def load_state(self) -> None:
        """
        Load persisted state on startup.

        - If saved state is from today → resume (restore open positions + capital)
        - If saved state is from a previous day → start fresh daily tracking
        - If no saved state → initialise fresh
        """
        try:
            row = self._db.fetch_one(
                "SELECT current_capital, open_positions_json, daily_ledger_json, updated_at "
                "FROM paper_trading_state WHERE id=1",
                (),
            )
        except Exception:
            row = None

        if not row:
            logger.info("load_state: no saved state — starting fresh")
            return

        try:
            saved_at = datetime.fromisoformat(row["updated_at"])
            today = get_ist_now().date()
            is_today = (saved_at.date() == today)

            # Always restore daily ledger (keeps cumulative stats)
            ledger_raw = json.loads(row["daily_ledger_json"] or "{}")
            for date_str, data in ledger_raw.items():
                d = date.fromisoformat(date_str)
                rec = DailyRecord(
                    date=d,
                    starting_capital=data.get("starting_capital", self.config.initial_capital),
                    realized_pnl=data.get("realized_pnl", 0.0),
                    trades_taken=data.get("trades_taken", 0),
                    trades_won=data.get("trades_won", 0),
                    trades_lost=data.get("trades_lost", 0),
                    signals_generated=data.get("signals_generated", 0),
                    signals_filtered=data.get("signals_filtered", 0),
                    signals_executed=data.get("signals_executed", 0),
                    signals_missed=data.get("signals_missed", 0),
                    max_drawdown_today=data.get("max_drawdown_today", 0.0),
                    peak_capital_today=data.get("peak_capital_today", 0.0),
                    anomaly_alerts_count=data.get("anomaly_alerts_count", 0),
                )
                self.daily_ledger[d] = rec

            if is_today:
                # Resume today's session: restore capital and open positions
                self.current_capital = float(row["current_capital"])
                open_raw = json.loads(row["open_positions_json"] or "[]")
                for pos_dict in open_raw:
                    pos = self._dict_to_position(pos_dict)
                    if pos:
                        self.open_positions[pos.position_id] = pos
                logger.info(
                    "load_state: resumed from today's state — capital ₹%.0f, %d open position(s)",
                    self.current_capital, len(self.open_positions),
                )
            else:
                # Previous day: fresh daily tracking but keep capital from last close
                self.current_capital = float(row["current_capital"])
                # Don't restore stale open positions — they expired at EOD
                logger.info(
                    "load_state: previous day's state — capital ₹%.0f, no open positions carried over",
                    self.current_capital,
                )
        except Exception as exc:
            logger.error("load_state failed (starting fresh): %s", exc)

    def reset(self, confirm: bool = False) -> None:
        """
        Reset paper trading completely.

        Parameters
        ----------
        confirm:
            Must be True — prevents accidental resets.
        """
        if not confirm:
            raise ValueError("Pass confirm=True to reset paper trading state — this clears all history")

        self.current_capital = self.config.initial_capital
        self._peak_capital = self.config.initial_capital
        self.open_positions.clear()
        self.trade_history.clear()
        self.daily_ledger.clear()
        self.missed_signals.clear()
        self._kill_switch_active = False

        try:
            self._db.execute("DELETE FROM paper_trading_state WHERE id=1", ())
        except Exception as exc:
            logger.warning("reset: could not clear DB state: %s", exc)

        logger.warning("PaperTradingEngine RESET — all state cleared, capital restored to ₹%.0f",
                       self.config.initial_capital)

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def activate_kill_switch(self, reason: str = "Manual") -> None:
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        logger.warning("PAPER TRADING KILL SWITCH ACTIVATED: %s", reason)

    def deactivate_kill_switch(self) -> None:
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        logger.info("Paper trading kill switch deactivated")

    # ------------------------------------------------------------------
    # Convenience helpers (called by the data collector / orchestrator)
    # ------------------------------------------------------------------

    def has_open_positions(self) -> bool:
        return bool(self.open_positions)

    def get_open_positions(self) -> list[PaperPosition]:
        return list(self.open_positions.values())

    def increment_anomaly_count(self) -> None:
        today = get_ist_now().date()
        self._ensure_daily_record(today)
        self.daily_ledger[today].anomaly_alerts_count += 1

    def get_benchmark_return_pct(self) -> Optional[float]:
        """Return cumulative buy-and-hold return since paper trading started."""
        if self._benchmark_start_price and self._benchmark_current_price and self._benchmark_start_price > 0:
            return (self._benchmark_current_price - self._benchmark_start_price) / self._benchmark_start_price * 100
        return None

    def _update_benchmark_price(self, price: float, index_id: str) -> None:
        if self._benchmark_start_price is None:
            self._benchmark_start_price = price
            self._benchmark_index = index_id
            logger.info("Benchmark start price set: %s @ %.2f", index_id, price)
        self._benchmark_current_price = price

    # ------------------------------------------------------------------
    # Private — trailing stop logic (mirrors TradeSimulator exactly)
    # ------------------------------------------------------------------

    def _apply_trailing_sl(self, pos: PaperPosition, current_price: float) -> None:
        """Ratchet trailing SL based on progress toward target."""
        is_call = pos.trade_type == "BUY_CALL"
        entry = pos.execution_price
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
            return

        # Only ratchet — never widen the stop
        if is_call:
            if new_sl > pos.trailing_sl:
                pos.trailing_sl = round(new_sl, 2)
                pos.stop_loss = pos.trailing_sl
        else:
            if new_sl < pos.trailing_sl:
                pos.trailing_sl = round(new_sl, 2)
                pos.stop_loss = pos.trailing_sl

    # ------------------------------------------------------------------
    # Private — cost calculations
    # ------------------------------------------------------------------

    def _calc_entry_cost(self, turnover: float) -> float:
        brokerage = self.config.brokerage_per_order
        exchange = turnover * self.config.exchange_charges_pct
        gst = (brokerage + exchange) * self.config.gst_pct
        stamp = turnover * self.config.stamp_duty_pct
        sebi = turnover * self.config.sebi_charges_pct
        # No STT on options buy
        return brokerage + exchange + gst + stamp + sebi

    def _calc_exit_cost(self, turnover: float) -> float:
        brokerage = self.config.brokerage_per_order
        stt = turnover * self.config.stt_rate      # STT on sell side
        exchange = turnover * self.config.exchange_charges_pct
        gst = (brokerage + exchange) * self.config.gst_pct
        sebi = turnover * self.config.sebi_charges_pct
        return brokerage + stt + exchange + gst + sebi

    # ------------------------------------------------------------------
    # Private — market timing checks
    # ------------------------------------------------------------------

    def _is_warmed_up(self, now: datetime) -> bool:
        if self._warmed_up:
            return True
        warmup_end = now.replace(
            hour=_MARKET_OPEN_HOUR,
            minute=_WARMUP_END_MINUTE,
            second=0, microsecond=0,
        )
        if now >= warmup_end:
            self._warmed_up = True
            return True
        return False

    def _is_too_late_for_entry(self, now: datetime) -> bool:
        cutoff_minutes = _NO_NEW_ENTRY_HOUR * 60 + _NO_NEW_ENTRY_MINUTE
        now_minutes = now.hour * 60 + now.minute
        return now_minutes >= cutoff_minutes

    def _is_forced_exit_time(self, now: datetime) -> bool:
        if not self.config.intraday_only:
            return False
        cutoff_minutes = _FORCE_EXIT_HOUR * 60 + _FORCE_EXIT_MINUTE
        now_minutes = now.hour * 60 + now.minute
        return now_minutes >= cutoff_minutes

    def _is_data_stale(self, now: datetime) -> bool:
        if self._last_price_update is None:
            return False  # First update — allow it
        age = (now - self._last_price_update).total_seconds()
        return age > _STALE_DATA_THRESHOLD_SECS

    # ------------------------------------------------------------------
    # Private — statistical helpers
    # ------------------------------------------------------------------

    def _sl_exit_reason(self, pos: PaperPosition) -> str:
        is_call = pos.trade_type == "BUY_CALL"
        if is_call:
            return "TRAILING_SL_HIT" if pos.trailing_sl >= pos.execution_price else "STOP_LOSS_HIT"
        else:
            return "TRAILING_SL_HIT" if pos.trailing_sl <= pos.execution_price else "STOP_LOSS_HIT"

    def _compute_max_drawdown_pct(self, trades: list[PaperTrade]) -> float:
        if not trades:
            return 0.0
        capital = self.config.initial_capital
        peak = capital
        max_dd = 0.0
        for t in sorted(trades, key=lambda x: x.exit_timestamp):
            capital += t.net_pnl
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def _compute_sharpe(self, trades: list[PaperTrade]) -> float:
        if len(trades) < 2:
            return 0.0
        daily: dict[date, float] = {}
        for t in trades:
            d = t.exit_timestamp.date()
            daily[d] = daily.get(d, 0.0) + t.net_pnl / self.config.initial_capital * 100
        returns = list(daily.values())
        n = len(returns)
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        daily_rf = 6.5 / 252 / 100
        return (mean - daily_rf) / std * math.sqrt(252)

    def _compute_sortino(self, trades: list[PaperTrade]) -> float:
        if len(trades) < 2:
            return 0.0
        daily: dict[date, float] = {}
        for t in trades:
            d = t.exit_timestamp.date()
            daily[d] = daily.get(d, 0.0) + t.net_pnl / self.config.initial_capital * 100
        returns = list(daily.values())
        n = len(returns)
        mean = sum(returns) / n
        neg_sq = [(r ** 2) for r in returns if r < 0]
        downside_std = math.sqrt(sum(neg_sq) / n) if neg_sq else 0.0
        if downside_std == 0:
            return 0.0
        daily_rf = 6.5 / 252 / 100
        return (mean - daily_rf) / downside_std * math.sqrt(252)

    def _compute_max_consecutive_losing_days(self, cutoff: date) -> int:
        sorted_days = sorted(
            [(d, rec) for d, rec in self.daily_ledger.items() if d >= cutoff],
            key=lambda x: x[0],
        )
        max_streak = 0
        streak = 0
        for _, rec in sorted_days:
            if rec.realized_pnl < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    def _evaluate_edge_status(self, win_rate: float, profit_factor: float) -> str:
        if win_rate >= 0.50 and profit_factor >= 1.3:
            return "INTACT"
        if win_rate >= 0.40 or profit_factor >= 1.0:
            return "DEGRADING"
        return "GONE"

    def _uptime_pct(self) -> float:
        total_market_minutes = len(self.daily_ledger) * 375  # 6h15m per day
        if total_market_minutes <= 0:
            return 100.0
        return min(self._uptime_minutes_active / total_market_minutes * 100, 100.0)

    def _data_availability_pct(self) -> float:
        # Approximate: if we have daily records, data was mostly available
        return 100.0 if self.daily_ledger else 0.0

    # ------------------------------------------------------------------
    # Private — daily record management
    # ------------------------------------------------------------------

    def _ensure_daily_record(self, d: date) -> None:
        if d not in self.daily_ledger:
            prev_capital = self.current_capital
            self.daily_ledger[d] = DailyRecord(
                date=d,
                starting_capital=prev_capital,
                peak_capital_today=prev_capital,
            )

    # ------------------------------------------------------------------
    # Private — missed signal tracking
    # ------------------------------------------------------------------

    def _maybe_track_missed(
        self,
        signal: Union[TradingSignal, RefinedSignal],
        reason: str,
        current_market: dict,
    ) -> None:
        if not self.config.track_missed_signals:
            return
        entry = getattr(signal, "refined_entry", 0.0) or getattr(signal, "entry_price", 0.0)
        target = getattr(signal, "refined_target", 0.0) or getattr(signal, "target_price", 0.0)
        sl = getattr(signal, "refined_stop_loss", 0.0) or getattr(signal, "stop_loss", 0.0)
        self.missed_signals.append(MissedSignal(
            timestamp=get_ist_now(),
            index_id=getattr(signal, "index_id", ""),
            signal_type=getattr(signal, "signal_type", ""),
            confidence=getattr(signal, "confidence_level", "LOW"),
            reason_missed=reason,
            signal_entry_price=entry,
            would_have_been_target=target,
            would_have_been_sl=sl,
            signal_id=getattr(signal, "signal_id", str(uuid.uuid4())),
        ))

    def _log_skip(
        self,
        record: DailyRecord,
        signal,
        reason: str,
        detail: str,
    ) -> None:
        record.signals_filtered += 1
        if self.config.track_no_trade_reasons:
            logger.debug(
                "PAPER SKIP [%s %s] %s: %s",
                getattr(signal, "index_id", "?"),
                getattr(signal, "signal_type", "?"),
                reason, detail,
            )

    # ------------------------------------------------------------------
    # Private — DB / notification helpers
    # ------------------------------------------------------------------

    def _persist_trade(self, trade: PaperTrade) -> None:
        """Update the trading_signals row with paper trade outcome."""
        try:
            self._db.execute(
                """UPDATE trading_signals
                   SET outcome=?, actual_exit_price=?, actual_pnl=?, closed_at=?
                   WHERE id IN (
                       SELECT id FROM trading_signals
                       WHERE index_id=? AND outcome='OPEN'
                       ORDER BY generated_at DESC LIMIT 1
                   )""",
                (
                    trade.outcome,
                    trade.actual_exit_price,
                    trade.net_pnl,
                    trade.exit_timestamp.isoformat(),
                    trade.index_id,
                ),
            )
        except Exception as exc:
            logger.warning("_persist_trade: DB update failed: %s", exc)

    def _send_alert(self, message: str, priority: str = "NORMAL") -> None:
        if self._telegram is not None:
            try:
                self._telegram.send_alert(message, priority=priority)
            except Exception as exc:
                logger.debug("Telegram alert failed: %s", exc)

    def _queue_ws_event(self, event_type: str, data: dict) -> None:
        try:
            from src.api.websocket import queue_event
            queue_event(f"paper_{event_type}", data)
        except Exception as exc:
            logger.debug("WebSocket event queue failed: %s", exc)

    def _get_lot_size(self, index_id: str) -> Optional[int]:
        try:
            registry = get_registry()
            index = registry.get_index(index_id)
            if index and index.lot_size:
                return index.lot_size
        except Exception:
            pass
        return None

    def _ensure_schema(self) -> None:
        """Create the paper_trading_state table if it doesn't exist."""
        try:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS paper_trading_state (
                    id                   INTEGER PRIMARY KEY,
                    current_capital      REAL NOT NULL,
                    open_positions_json  TEXT NOT NULL DEFAULT '[]',
                    daily_ledger_json    TEXT NOT NULL DEFAULT '{}',
                    updated_at           TEXT NOT NULL
                )""",
                (),
            )
        except Exception as exc:
            logger.warning("Could not create paper_trading_state table: %s", exc)

    # ------------------------------------------------------------------
    # Private — serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _position_to_dict(pos: PaperPosition) -> dict:
        return {
            "position_id": pos.position_id,
            "signal_id": pos.signal_id,
            "index_id": pos.index_id,
            "trade_type": pos.trade_type,
            "signal_entry_price": pos.signal_entry_price,
            "market_price_at_signal": pos.market_price_at_signal,
            "execution_price": pos.execution_price,
            "entry_timestamp": pos.entry_timestamp.isoformat(),
            "stop_loss": pos.stop_loss,
            "original_stop_loss": pos.original_stop_loss,
            "target": pos.target,
            "trailing_sl": pos.trailing_sl,
            "lots": pos.lots,
            "lot_size": pos.lot_size,
            "quantity": pos.quantity,
            "strike": pos.strike,
            "expiry": pos.expiry,
            "option_premium_at_entry": pos.option_premium_at_entry,
            "entry_cost": pos.entry_cost,
            "risk_amount": pos.risk_amount,
            "confidence": pos.confidence,
            "regime": pos.regime,
            "current_price": pos.current_price,
            "max_favorable": pos.max_favorable,
            "max_adverse": pos.max_adverse,
            "bars_held": pos.bars_held,
            "strategy_version": pos.strategy_version,
        }

    @staticmethod
    def _dict_to_position(d: dict) -> Optional[PaperPosition]:
        try:
            return PaperPosition(
                position_id=d["position_id"],
                signal_id=d["signal_id"],
                index_id=d["index_id"],
                trade_type=d["trade_type"],
                signal_entry_price=d["signal_entry_price"],
                market_price_at_signal=d["market_price_at_signal"],
                execution_price=d["execution_price"],
                entry_timestamp=datetime.fromisoformat(d["entry_timestamp"]),
                stop_loss=d["stop_loss"],
                original_stop_loss=d["original_stop_loss"],
                target=d["target"],
                trailing_sl=d["trailing_sl"],
                lots=d["lots"],
                lot_size=d["lot_size"],
                quantity=d["quantity"],
                strike=d.get("strike"),
                expiry=d.get("expiry"),
                option_premium_at_entry=d.get("option_premium_at_entry"),
                entry_cost=d["entry_cost"],
                risk_amount=d["risk_amount"],
                confidence=d["confidence"],
                regime=d.get("regime", "UNKNOWN"),
                current_price=d.get("current_price", 0.0),
                max_favorable=d.get("max_favorable", 0.0),
                max_adverse=d.get("max_adverse", 0.0),
                bars_held=d.get("bars_held", 0),
                status="OPEN",
                opened_at=datetime.fromisoformat(d["entry_timestamp"]),
                strategy_version=d.get("strategy_version", "1.0.0"),
            )
        except Exception as exc:
            logger.error("_dict_to_position failed: %s | data=%s", exc, d)
            return None
