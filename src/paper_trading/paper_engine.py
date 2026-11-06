"""
Paper Trading Engine — a premium-native paper ledger for index-option BUYING.

What it simulates
-----------------
The live :class:`~src.engine.risk_manager.RiskManager` turns a signal into an
option trade (contract, lots, premium stop/target).  This engine answers
"what would a trader who followed those alerts actually have made?":

* **Entry** — an order is created for every risk-approved ``RefinedSignal``
  (no re-sizing: lots/quantity are taken as approved).  It fills at the
  option **ASK** of the first ``options_chain_snapshot`` quote for that exact
  contract stamped at/after ``decision_time + fill_latency_seconds``
  (``LTP + half_spread`` if the quote has no book).  If no quote arrives
  within ``fill_window_minutes`` the order expires as **NO FILL** and the
  reason is recorded.  Cash is debited ``premium × qty + entry-leg costs``.
* **Marks** — every monitor tick marks each position from the freshest quote
  (≤ ``max_quote_age_seconds``; mid of bid/ask, else LTP), exactly like the
  live monitor.  Without a fresh quote the mark is a model re-price from
  spot (``option_contract.estimate_premium``) and is flagged ``model``.
* **Exits** — the same rules as ``RiskManager._update_option_position``:
  hero-zero cut-off 14:55, forced exit 15:15, premium stop or spot stop
  (stop first), premium target or spot target, 60-min time stop below
  +0.3 R, and the same premium-stop ratchet.  Exit *decisions* use the
  live RM's parameters (planned entry premium, decision time) so the paper
  trader exits when the alert says so; the *fill* is at the quote's **BID**
  (``mark − half_spread`` without a book).
* **P&L** — ``(exit fill − entry fill) × qty − round_trip_cost(...)``
  (statutory charges from ``option_costs``).  WIN iff net P&L > 0.

All prices on positions and trades are **option premium in ₹ per unit**
unless the field name says ``spot``.  The legacy field names are kept so
existing reports keep working; their meaning is documented on each class.

Integration
-----------
Use :class:`src.paper_trading.adapter.PaperTradingAdapter` — it feeds
``DecisionResult`` objects into :meth:`PaperTradingEngine.submit_order` and
calls :meth:`PaperTradingEngine.on_tick` from the 60 s monitor job.

Persistence
-----------
State lives in the (engine-owned) ``paper_trading_state`` row (id=1):
``current_capital`` holds **cash**; ``open_positions_json`` holds open
positions + pending orders; ``daily_ledger_json`` holds the daily ledger,
closed trades and today's unfilled orders.  No schema change is needed.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta
from typing import Optional, Union
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import STRATEGY_VERSION
from src.data.index_registry import get_registry
from src.database import queries as Q
from src.database.db_manager import DatabaseManager
from src.engine.option_contract import estimate_premium, minutes_until
from src.engine.option_costs import round_trip_cost, schedule_for
from src.engine.risk_manager import RefinedSignal, RiskConfig
from src.engine.signal_generator import TradingSignal
from src.paper_trading.edge_tracker import (
    annualised_sharpe,
    annualised_sortino,
    daily_equity_returns,
)
from src.utils.date_utils import get_ist_now

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# Confidence level ordering for comparison
_CONF_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

_STATE_SCHEMA = 2
_MIN_PREMIUM = 0.05                 # exchange tick — never fill below it
_STALE_DATA_THRESHOLD_SECS = 300    # daily summary "data fresh" flag

# Paper exit reason ← live RiskManager action (for parity checks / reports)
RM_ACTION_TO_EXIT_REASON = {
    "EXIT_SL": "STOP_LOSS_HIT",
    "EXIT_TRAILING_SL": "TRAILING_SL_HIT",
    "EXIT_TARGET": "TARGET_HIT",
    "EXIT_TIME_STOP": "TIME_STOP",
    "EXIT_TIME": "FORCED_EOD",          # also HERO_ZERO_CUTOFF (RM uses one action)
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PaperTradingConfig:
    """Configurable parameters.  Risk caps and exit rules default to the
    live :class:`RiskConfig` values so paper mirrors live."""

    # Capital
    initial_capital: float = 100_000.0

    # ── Fill model ──────────────────────────────────────────────────────
    fill_latency_seconds: float = 30.0          # alert → order at the exchange
    fill_window_minutes: float = 5.0            # no quote within this → NO FILL
    max_entry_slippage_pct: Optional[float] = None  # e.g. 0.05: skip if ask > planned × 1.05
    assumed_half_spread_pct: float = RiskConfig.assumed_half_spread_pct
    max_quote_age_seconds: int = 600            # same freshness rule as the live monitor
    max_spot_age_seconds: int = 600

    # ── Exit rules (mirror RiskManager) ────────────────────────────────
    force_exit_time: str = RiskConfig.force_exit_time
    hero_zero_exit_time: str = "14:55"
    time_stop_minutes: int = RiskConfig.time_stop_minutes
    time_stop_min_r: float = RiskConfig.time_stop_min_r
    intraday_only: bool = True                  # False disables the time exits

    # ── Caps (same as RiskManager; nothing is re-sized) ────────────────
    max_open_positions: int = RiskConfig.max_open_positions
    max_positions_per_index: int = RiskConfig.max_positions_per_index
    max_same_direction_positions: int = RiskConfig.max_same_direction_positions
    max_risk_per_day_pct: float = RiskConfig.max_risk_per_day_pct

    # ── Filters ─────────────────────────────────────────────────────────
    # "LOW" = take whatever the live risk manager approved.
    min_confidence: str = "LOW"
    signal_types: list = field(default_factory=lambda: ["BUY_CALL", "BUY_PUT"])
    active_indices: list = field(default_factory=lambda: ["NIFTY50", "BANKNIFTY"])

    # Tracking
    track_missed_signals: bool = True
    track_no_trade_reasons: bool = True
    max_trade_history: int = 5_000

    # Legacy (ignored; kept so old constructors keep working)
    state_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Order / position / trade dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PaperOrder:
    """A buy order for one risk-approved signal, waiting for a quote."""

    order_id: str
    signal_id: str
    index_id: str
    trade_type: str                     # BUY_CALL / BUY_PUT
    option_type: str                    # CE / PE
    strike: float
    expiry: str                         # ISO date
    lots: int
    lot_size: int
    quantity: int
    decision_time: datetime
    planned_entry_premium: float        # RM entry premium (mid + half spread)
    premium_mid: float                  # RM fair premium at decision
    half_spread: float
    premium_stop: float
    premium_target: float
    spot_at_signal: float
    spot_stop: float
    spot_target: float
    option_iv: float
    is_hero_zero: bool
    confidence: str
    regime: str
    risk_amount: float
    exchange: str = "NSE"
    status: str = "PENDING"             # PENDING / FILLED / NO_FILL
    reason: str = ""
    resolved_at: Optional[datetime] = None
    position_id: Optional[str] = None


@dataclass
class PaperPosition:
    """An open paper option position.  Prices are premium ₹/unit."""

    position_id: str
    signal_id: str
    index_id: str
    trade_type: str                     # "BUY_CALL" / "BUY_PUT"

    signal_entry_price: float           # RM planned entry premium (decision rules use it)
    market_price_at_signal: float       # underlying SPOT at the decision
    execution_price: float              # premium actually paid (ask at fill)
    entry_timestamp: datetime           # fill time (quote timestamp)

    stop_loss: float                    # current premium stop (ratchets up)
    original_stop_loss: float           # initial premium stop
    target: float                       # premium target
    trailing_sl: float                  # == stop_loss (legacy name)

    lots: int
    lot_size: int
    quantity: int

    strike: Optional[float]
    expiry: Optional[str]               # ISO date
    option_premium_at_entry: Optional[float]  # RM premium mid at decision (model anchor)

    entry_cost: float                   # ₹ statutory cost of the buy leg
    risk_amount: float                  # RM max loss (₹)
    confidence: str
    regime: str

    # Live tracking (updated every tick)
    current_price: float = 0.0          # last premium mark
    unrealized_pnl: float = 0.0         # at the exit fill estimate, net of all costs
    max_favorable: float = 0.0          # premium ₹/unit above execution_price
    max_adverse: float = 0.0            # premium ₹/unit below execution_price
    bars_held: int = 0                  # monitor ticks (informational only)

    iv_at_entry: Optional[float] = None
    days_to_expiry_at_entry: Optional[float] = None

    status: str = "OPEN"
    opened_at: datetime = field(default_factory=get_ist_now)
    strategy_version: str = "1.0.0"

    # Contract / rule details
    option_type: Optional[str] = None   # CE / PE
    decision_time: Optional[datetime] = None
    half_spread: float = 0.0
    spot_stop: float = 0.0
    spot_target: float = 0.0
    is_hero_zero: bool = False
    exchange: str = "NSE"
    entry_fill_source: str = ""         # "ask" / "ltp+half_spread"
    mark_source: str = ""               # "quote" / "model" / "last_known"
    mark_timestamp: Optional[datetime] = None
    last_spot: float = 0.0

    @property
    def cost_basis(self) -> float:
        """Cash paid to open: premium × qty + entry-leg costs."""
        return self.execution_price * self.quantity + self.entry_cost


@dataclass
class PaperTrade:
    """A closed paper trade.  Prices are premium ₹/unit."""

    position_id: str
    signal_id: str
    index_id: str
    trade_type: str

    signal_entry_price: float           # RM planned entry premium
    market_price_at_signal: float       # spot at decision
    execution_price: float              # entry fill premium (ask)
    entry_timestamp: datetime           # fill time

    original_stop_loss: float           # initial premium stop
    original_target: float              # premium target

    strike: Optional[float]
    expiry: Optional[str]
    option_premium_at_entry: Optional[float]

    lots: int
    lot_size: int
    quantity: int

    exit_price: float                   # premium mark that triggered the exit
    actual_exit_price: float            # exit fill premium (bid)
    exit_timestamp: datetime
    exit_reason: str                    # TARGET_HIT / STOP_LOSS_HIT / TRAILING_SL_HIT /
                                        # TIME_STOP / FORCED_EOD / HERO_ZERO_CUTOFF / MANUAL

    gross_pnl: float                    # (exit fill − entry fill) × qty
    entry_cost: float
    exit_cost: float
    total_costs: float                  # round_trip_cost(...).total
    net_pnl: float
    net_pnl_pct: float                  # % of initial capital

    max_favorable_excursion: float      # premium ₹/unit
    max_adverse_excursion: float
    capture_ratio: float                # net_pnl / (MFE × qty)

    duration_seconds: int
    duration_bars: int                  # monitor ticks (informational)

    confidence: str
    regime: str
    risk_amount: float
    entry_cost_at_open: float

    outcome: str                        # "WIN" iff net_pnl > 0, else "LOSS"

    strategy_version: str = "1.0.0"

    option_type: Optional[str] = None
    decision_time: Optional[datetime] = None
    entry_fill_source: str = ""
    exit_fill_source: str = ""          # "bid" / "ltp-half_spread" / "model-half_spread" / ...
    exit_mark_source: str = ""          # "quote" / "model" / "last_known" / "manual"
    spot_at_exit: Optional[float] = None
    is_hero_zero: bool = False


@dataclass
class PaperExecution:
    """Confirmation of a filled paper entry."""

    position: PaperPosition
    slippage_applied: float             # fill − planned entry premium (₹/unit)
    entry_cost: float
    message: str


@dataclass
class MissedSignal:
    """A signal that passed the filters but was not traded (cap, limit, no fill …)."""

    timestamp: datetime
    index_id: str
    signal_type: str
    confidence: str
    reason_missed: str                  # "daily_loss_limit" / "max_positions" / "no_fill:no_quote" / …
    signal_entry_price: float           # planned entry premium (spot for non-option signals)
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
    signals_filtered: int = 0          # Blocked by confidence/index/type filter
    signals_executed: int = 0          # Filled
    signals_missed: int = 0            # Passed filters, blocked by caps/limits or not filled
    max_drawdown_today: float = 0.0    # Peak-to-trough (mark-to-market)
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
    sharpe_ratio: float                 # annualised, DAILY returns on equity (zero days included)
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


@dataclass(frozen=True)
class _Quote:
    timestamp: datetime
    ltp: float
    bid: float
    ask: float

    @property
    def has_book(self) -> bool:
        return self.bid > 0 and self.ask >= self.bid

    @property
    def mark(self) -> Optional[float]:
        """Mid of bid/ask, else LTP — identical to DecisionEngine._option_mark."""
        if self.has_book:
            return (self.bid + self.ask) / 2
        return self.ltp if self.ltp > 0 else None


# ---------------------------------------------------------------------------
# Paper Trading Engine
# ---------------------------------------------------------------------------


class PaperTradingEngine:
    """
    Premium-native paper ledger fed by risk-approved signals.

    Public API
    ----------
    * :meth:`submit_order` / :meth:`on_signal` — a risk-approved signal → order
      (filled immediately if a qualifying quote already exists).
    * :meth:`on_tick` — fill pending orders, mark positions, apply exits.
    * :meth:`close_position` — manual close at a premium mark.
    * :meth:`get_daily_summary` / :meth:`get_cumulative_stats` — reporting.
    * :meth:`save_state` / :meth:`load_state` — survive restarts.
    """

    def __init__(
        self,
        db: DatabaseManager,
        config: Optional[PaperTradingConfig] = None,
        telegram_bot=None,
    ) -> None:
        self._db = db
        self.config = config if isinstance(config, PaperTradingConfig) else PaperTradingConfig()
        self._telegram = telegram_bot
        # Decision (300 s) and monitor (60 s) jobs run on different threads.
        self._lock = threading.RLock()

        # State
        self.cash: float = self.config.initial_capital
        self.open_positions: dict[str, PaperPosition] = {}
        self.pending_orders: dict[str, PaperOrder] = {}
        self.unfilled_orders: list[PaperOrder] = []
        self.trade_history: list[PaperTrade] = []
        self.daily_ledger: dict[date, DailyRecord] = {}
        self.missed_signals: list[MissedSignal] = []

        # Kill switch (manual) and watchdog safe mode
        self._kill_switch_active: bool = False
        self._kill_switch_reason: str = ""
        self.pause_new_trades: bool = False

        # Data freshness
        self._last_price_update: Optional[datetime] = None
        self._stops_moved: bool = False

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

        self._ensure_schema()
        self.load_state()

        logger.info(
            "PaperTradingEngine initialised — cash ₹%.0f, %d open, %d pending "
            "(min_conf=%s, indices=%s, latency=%.0fs, fill window=%.1f min)",
            self.cash, len(self.open_positions), len(self.pending_orders),
            self.config.min_confidence, self.config.active_indices,
            self.config.fill_latency_seconds, self.config.fill_window_minutes,
        )

    # ------------------------------------------------------------------
    # Capital
    # ------------------------------------------------------------------

    @property
    def current_capital(self) -> float:
        """Realised equity: cash + cost basis of open positions.

        Equals ``initial_capital + Σ net P&L of closed trades``; open
        positions are carried at cost (see :meth:`equity` for mark-to-market).
        """
        return self.cash + sum(p.cost_basis for p in self.open_positions.values())

    @current_capital.setter
    def current_capital(self, value: float) -> None:
        self.cash = float(value) - sum(p.cost_basis for p in self.open_positions.values())

    def equity(self) -> float:
        """Mark-to-market equity: cash + what open positions would fetch now
        (exit fill estimate minus exit-leg costs)."""
        total = self.cash
        for p in self.open_positions.values():
            total += p.cost_basis + p.unrealized_pnl
        return total

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def on_signal(
        self,
        signal: Union[TradingSignal, RefinedSignal],
        current_market: Optional[dict] = None,
        *,
        decision_time: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> Optional[PaperExecution]:
        """Submit *signal*; return a :class:`PaperExecution` if it filled now.

        ``None`` means filtered, rejected, not filled, or still pending —
        see :attr:`pending_orders` / :attr:`unfilled_orders` /
        :attr:`missed_signals`.  ``current_market`` is only used for its
        optional ``"timestamp"`` (the decision time).
        """
        if decision_time is None and current_market:
            decision_time = current_market.get("timestamp")
        order = self.submit_order(signal, decision_time=decision_time, now=now)
        if order is None or order.status != "FILLED":
            return None
        pos = self.open_positions.get(order.position_id or "")
        if pos is None:
            return None
        return PaperExecution(
            position=pos,
            slippage_applied=round(pos.execution_price - pos.signal_entry_price, 2),
            entry_cost=pos.entry_cost,
            message=self._entry_message(pos),
        )

    def submit_order(
        self,
        signal: Union[TradingSignal, RefinedSignal],
        decision_time: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> Optional[PaperOrder]:
        """Create a paper buy order for a risk-approved option signal.

        The order fills on the first qualifying quote (see module docstring);
        it is attempted immediately and then on every :meth:`on_tick`.
        Returns the order (``PENDING`` / ``FILLED`` / ``NO_FILL``) or
        ``None`` when the signal was filtered out or blocked by a cap.
        """
        with self._lock:
            now = _aware(now or get_ist_now())
            decision_time = _aware(decision_time or getattr(signal, "generated_at", None) or now)
            self._check_strategy_version()

            day = decision_time.date()
            self._ensure_daily_record(day)
            record = self.daily_ledger[day]
            record.signals_generated += 1

            sig_type = getattr(signal, "signal_type", "")
            index_id = getattr(signal, "index_id", "")
            confidence = getattr(signal, "confidence_level", "LOW") or "LOW"
            signal_id = getattr(signal, "signal_id", None) or str(uuid.uuid4())

            # ── Filters ─────────────────────────────────────────────────
            if sig_type not in self.config.signal_types:
                self._log_skip(record, signal, "signal_type_filtered", f"type {sig_type!r}")
                return None
            if index_id not in self.config.active_indices:
                self._log_skip(record, signal, "index_filtered", f"index {index_id!r} not active")
                return None
            if _CONF_ORDER.get(confidence, 0) < _CONF_ORDER.get(self.config.min_confidence, 0):
                record.signals_filtered += 1
                self._maybe_track_missed(signal, "below_min_confidence")
                return None
            if self._kill_switch_active:
                self._log_skip(record, signal, "kill_switch", self._kill_switch_reason)
                return None
            if self.pause_new_trades:
                self._log_skip(record, signal, "paused", "watchdog safe mode")
                return None
            contract_problem = _contract_problem(signal)
            if contract_problem:
                self._log_skip(record, signal, "not_option_native", contract_problem)
                return None
            if self._signal_seen(signal_id):
                logger.debug("PAPER SKIP [%s] duplicate signal %s", index_id, signal_id[:8])
                return None

            # ── Limits (same caps as the live RiskManager) ──────────────
            blocked = self._cap_reason(index_id, sig_type, record)
            if blocked:
                self._maybe_track_missed(signal, blocked)
                record.signals_missed += 1
                logger.info("PAPER SKIP [%s %s] %s", index_id, sig_type, blocked)
                return None

            lots, lot_size, quantity = _approved_size(signal)
            order = PaperOrder(
                order_id=str(uuid.uuid4()),
                signal_id=signal_id,
                index_id=index_id,
                trade_type=sig_type,
                option_type=str(signal.option_type),
                strike=float(signal.recommended_strike),
                expiry=str(signal.recommended_expiry)[:10],
                lots=lots,
                lot_size=lot_size,
                quantity=quantity,
                decision_time=decision_time,
                planned_entry_premium=float(signal.entry_premium),
                premium_mid=float(getattr(signal, "premium_mid", 0.0) or signal.entry_premium),
                half_spread=float(getattr(signal, "half_spread", 0.0) or 0.0),
                premium_stop=float(signal.premium_stop),
                premium_target=float(signal.premium_target),
                spot_at_signal=float(getattr(signal, "refined_entry", 0.0) or getattr(signal, "entry_price", 0.0) or 0.0),
                spot_stop=float(getattr(signal, "refined_stop_loss", 0.0) or 0.0),
                spot_target=float(getattr(signal, "refined_target", 0.0) or 0.0),
                option_iv=float(getattr(signal, "option_iv", 0.0) or 0.0),
                is_hero_zero=bool(getattr(signal, "is_hero_zero", False)),
                confidence=confidence,
                regime=str(getattr(signal, "regime", "UNKNOWN")),
                risk_amount=float(getattr(signal, "max_loss_amount", 0.0) or getattr(signal, "risk_amount", 0.0) or 0.0),
                exchange=self._exchange_for(index_id),
            )
            self.pending_orders[order.order_id] = order
            logger.info(
                "PAPER ORDER: %s %s %s %.0f%s exp %s × %d (planned %.2f) — fills on first quote "
                "after %s",
                index_id, sig_type, order.option_type, order.strike, order.option_type,
                order.expiry, order.quantity, order.planned_entry_premium,
                (decision_time + timedelta(seconds=self.config.fill_latency_seconds)).strftime("%H:%M:%S"),
            )
            self._try_fill(order, now)
            self.save_state()
            return order

    def _cap_reason(self, index_id: str, sig_type: str, record: DailyRecord) -> Optional[str]:
        cfg = self.config
        daily_loss_limit = cfg.initial_capital * cfg.max_risk_per_day_pct / 100.0
        if record.realized_pnl <= -daily_loss_limit:
            return "daily_loss_limit"
        live = [(p.index_id, p.trade_type) for p in self.open_positions.values()]
        live += [(o.index_id, o.trade_type) for o in self.pending_orders.values()]
        if len(live) >= cfg.max_open_positions:
            return "max_positions"
        if sum(1 for i, _ in live if i == index_id) >= cfg.max_positions_per_index:
            return "existing_position_in_index"
        if sum(1 for _, t in live if t == sig_type) >= cfg.max_same_direction_positions:
            return "max_same_direction"
        return None

    def _try_fill(self, order: PaperOrder, now: datetime) -> bool:
        """Fill *order* on the first quote in its window, or expire it."""
        cfg = self.config
        start = order.decision_time + timedelta(seconds=cfg.fill_latency_seconds)
        deadline = start + timedelta(minutes=cfg.fill_window_minutes)
        if now < start:
            return False

        quote = self._first_quote(order, start, min(now, deadline))
        if quote is None:
            if now >= deadline:
                self._no_fill(
                    order, "no_quote",
                    f"no quote for {order.index_id} {order.strike:.0f}{order.option_type} "
                    f"{order.expiry} between {start.strftime('%H:%M:%S')} and "
                    f"{deadline.strftime('%H:%M:%S')}",
                    now,
                )
            return False

        cutoff = cfg.hero_zero_exit_time if order.is_hero_zero else cfg.force_exit_time
        if cfg.intraday_only and minutes_until(quote.timestamp, cutoff) <= 0:
            self._no_fill(order, "past_force_exit", f"first quote at {quote.timestamp:%H:%M} ≥ {cutoff}", now)
            return False

        if quote.ask > 0 and (quote.bid <= 0 or quote.ask >= quote.bid):
            fill_px, source = quote.ask, "ask"
        else:
            half = order.half_spread or quote.ltp * cfg.assumed_half_spread_pct
            fill_px, source = quote.ltp + half, "ltp+half_spread"
        fill_px = round(max(fill_px, _MIN_PREMIUM), 2)

        if (
            cfg.max_entry_slippage_pct is not None
            and fill_px > order.planned_entry_premium * (1.0 + cfg.max_entry_slippage_pct)
        ):
            self._no_fill(
                order, "slippage_cap",
                f"ask {fill_px:.2f} > planned {order.planned_entry_premium:.2f} "
                f"+{cfg.max_entry_slippage_pct:.0%}",
                now,
            )
            return False

        entry_cost = self._entry_leg_cost(fill_px, order.quantity, order.exchange)
        outlay = fill_px * order.quantity + entry_cost
        if outlay > self.cash + 1e-9:
            self._no_fill(order, "insufficient_cash", f"need ₹{outlay:,.0f}, cash ₹{self.cash:,.0f}", now)
            return False

        self._open_position(order, quote, fill_px, source, entry_cost)
        return True

    def _open_position(
        self, order: PaperOrder, quote: _Quote, fill_px: float, source: str, entry_cost: float,
    ) -> PaperPosition:
        self.cash -= fill_px * order.quantity + entry_cost
        expiry_d = _parse_date(order.expiry)
        dte = (expiry_d - quote.timestamp.date()).days if expiry_d else None
        pos = PaperPosition(
            position_id=str(uuid.uuid4()),
            signal_id=order.signal_id,
            index_id=order.index_id,
            trade_type=order.trade_type,
            signal_entry_price=round(order.planned_entry_premium, 2),
            market_price_at_signal=round(order.spot_at_signal, 2),
            execution_price=fill_px,
            entry_timestamp=quote.timestamp,
            stop_loss=round(order.premium_stop, 2),
            original_stop_loss=round(order.premium_stop, 2),
            target=round(order.premium_target, 2),
            trailing_sl=round(order.premium_stop, 2),
            lots=order.lots,
            lot_size=order.lot_size,
            quantity=order.quantity,
            strike=order.strike,
            expiry=order.expiry,
            option_premium_at_entry=round(order.premium_mid, 2),
            entry_cost=round(entry_cost, 2),
            risk_amount=round(order.risk_amount, 2),
            confidence=order.confidence,
            regime=order.regime,
            current_price=quote.mark or fill_px,
            iv_at_entry=order.option_iv or None,
            days_to_expiry_at_entry=float(dte) if dte is not None else None,
            opened_at=quote.timestamp,
            strategy_version=STRATEGY_VERSION,
            option_type=order.option_type,
            decision_time=order.decision_time,
            half_spread=order.half_spread,
            spot_stop=order.spot_stop,
            spot_target=order.spot_target,
            is_hero_zero=order.is_hero_zero,
            exchange=order.exchange,
            entry_fill_source=source,
            mark_source="quote",
            mark_timestamp=quote.timestamp,
            last_spot=order.spot_at_signal,
        )
        self.open_positions[pos.position_id] = pos
        self.pending_orders.pop(order.order_id, None)
        order.status, order.position_id, order.resolved_at = "FILLED", pos.position_id, quote.timestamp

        day = order.decision_time.date()
        self._ensure_daily_record(day)
        self.daily_ledger[day].signals_executed += 1

        msg = self._entry_message(pos)
        logger.info(
            "PAPER FILL: %s %s %.0f%s @ %.2f (%s, planned %.2f, latency %ds) qty=%d cost ₹%.2f",
            pos.index_id, pos.trade_type, pos.strike, pos.option_type, fill_px, source,
            order.planned_entry_premium,
            int((quote.timestamp - order.decision_time).total_seconds()),
            pos.quantity, entry_cost,
        )
        self._send_alert(msg)
        self._queue_ws_event("entry", {
            "position_id": pos.position_id,
            "index_id": pos.index_id,
            "trade_type": pos.trade_type,
            "option_type": pos.option_type,
            "strike": pos.strike,
            "expiry": pos.expiry,
            "execution_price": fill_px,
            "lots": pos.lots,
            "stop_loss": pos.stop_loss,
            "target": pos.target,
            "confidence": pos.confidence,
        })
        return pos

    def _no_fill(self, order: PaperOrder, code: str, detail: str, now: datetime) -> None:
        order.status, order.reason, order.resolved_at = "NO_FILL", f"{code}: {detail}", now
        self.pending_orders.pop(order.order_id, None)
        self.unfilled_orders.append(order)
        day = order.decision_time.date()
        self._ensure_daily_record(day)
        self.daily_ledger[day].signals_missed += 1
        if self.config.track_missed_signals:
            self.missed_signals.append(MissedSignal(
                timestamp=now,
                index_id=order.index_id,
                signal_type=order.trade_type,
                confidence=order.confidence,
                reason_missed=f"no_fill:{code}",
                signal_entry_price=order.planned_entry_premium,
                would_have_been_target=order.premium_target,
                would_have_been_sl=order.premium_stop,
                signal_id=order.signal_id,
            ))
        logger.warning("PAPER NO FILL [%s %s]: %s", order.index_id, order.trade_type, order.reason)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def on_tick(
        self,
        now: Optional[datetime] = None,
        spot_prices: Optional[dict[str, float]] = None,
    ) -> list[PaperTrade]:
        """Fill pending orders, mark every position, apply exit rules.

        *spot_prices* (``{"NIFTY50": 24_510.0}``) overrides the DB spot
        lookup; normally omitted.  Returns trades closed this tick.
        """
        with self._lock:
            now = _aware(now or get_ist_now())
            changed = False
            for order in list(self.pending_orders.values()):
                before = order.status
                self._try_fill(order, now)
                changed |= order.status != before

            closed = self._settle_previous_day_positions(now)
            changed |= bool(closed)

            bm = self._spot(self._benchmark_index, now, spot_prices)
            if bm:
                self._update_benchmark_price(bm, self._benchmark_index)

            self._stops_moved = False
            closed += self._mark_and_exit(now, spot_prices)
            changed |= bool(closed) or self._stops_moved

            self._update_drawdown(now)
            # Persist only durable changes (fills, exits, ratcheted stops);
            # per-tick marks are cheap to lose and re-derive after a restart.
            if changed:
                self.save_state()
            return closed

    def update_positions(self, current_prices: dict[str, float]) -> list[PaperTrade]:
        """Legacy entry point: one tick with explicit spot prices."""
        return self.on_tick(spot_prices=current_prices)

    def _mark_and_exit(self, now: datetime, spot_prices: Optional[dict]) -> list[PaperTrade]:
        closed: list[PaperTrade] = []
        for pos in list(self.open_positions.values()):
            if pos.entry_timestamp > now:
                continue
            spot = self._spot(pos.index_id, now, spot_prices)
            quote = self._latest_quote(pos, now)
            mark = quote.mark if quote else None
            if mark is not None:
                source = "quote"
                if quote.has_book:
                    fill, fill_source = quote.bid, "bid"
                else:
                    fill, fill_source = mark - self._half_spread(pos, mark), "ltp-half_spread"
            elif spot:
                mark, source = self._model_mark(pos, spot, now), "model"
                fill, fill_source = mark - self._half_spread(pos, mark), "model-half_spread"
                logger.warning(
                    "PAPER MARK [%s %.0f%s]: no fresh option quote — MODEL mark ₹%.2f",
                    pos.index_id, pos.strike or 0, pos.option_type or "", mark,
                )
            else:
                reason = self._time_exit_reason(pos, now)
                if reason:
                    mark = pos.current_price or pos.option_premium_at_entry or pos.execution_price
                    fill = mark - self._half_spread(pos, mark)
                    logger.error(
                        "PAPER EXIT [%s] at last known mark ₹%.2f — no quote and no spot", pos.index_id, mark,
                    )
                    closed.append(self._close(
                        pos, fill, mark, reason, now, "last_known-half_spread", "last_known", None,
                    ))
                else:
                    logger.warning("PAPER MARK [%s]: no quote and no spot — position not evaluated", pos.index_id)
                continue

            self._last_price_update = now
            self._track(pos, mark, fill, source, now, spot)
            reason = self._exit_reason(pos, mark, spot, now)
            if reason:
                closed.append(self._close(pos, fill, mark, reason, now, fill_source, source, spot))
            elif self._trail_premium_stop(pos, mark):
                self._stops_moved = True
        return closed

    def _track(
        self, pos: PaperPosition, mark: float, fill: float, source: str,
        now: datetime, spot: Optional[float],
    ) -> None:
        pos.current_price = round(mark, 2)
        pos.mark_source = source
        pos.mark_timestamp = now
        if spot:
            pos.last_spot = spot
        pos.bars_held += 1
        pos.max_favorable = max(pos.max_favorable, mark - pos.execution_price)
        pos.max_adverse = max(pos.max_adverse, pos.execution_price - mark)
        fill = max(fill, _MIN_PREMIUM)
        costs = round_trip_cost(pos.execution_price, fill, pos.quantity, pos.exchange).total
        pos.unrealized_pnl = round((fill - pos.execution_price) * pos.quantity - costs, 2)

    # ── Exit rules: mirror RiskManager._update_option_position ──────────

    def _time_exit_reason(self, pos: PaperPosition, now: datetime) -> Optional[str]:
        cfg = self.config
        if not cfg.intraday_only:
            return None
        if pos.is_hero_zero and minutes_until(now, cfg.hero_zero_exit_time) <= 0:
            return "HERO_ZERO_CUTOFF"
        if minutes_until(now, cfg.force_exit_time) <= 0:
            return "FORCED_EOD"
        return None

    def _exit_reason(
        self, pos: PaperPosition, mark: float, spot: Optional[float], now: datetime,
    ) -> Optional[str]:
        """Exit decision on the option *mark* (and spot triggers when spot is known).

        Uses the live RM's parameters — planned entry premium for R and
        trailing, decision time for the time stop — so the paper trader exits
        exactly when the live alert says to.
        """
        cfg = self.config
        timed = self._time_exit_reason(pos, now)
        if timed:
            return timed

        is_call = pos.trade_type == "BUY_CALL"
        entry = pos.signal_entry_price
        spot_ok = spot is not None and spot > 0

        # Stop (premium or spot trigger) before target — conservative.
        spot_sl_hit = spot_ok and pos.spot_stop > 0 and (
            spot <= pos.spot_stop if is_call else spot >= pos.spot_stop
        )
        if mark <= pos.stop_loss or spot_sl_hit:
            return "TRAILING_SL_HIT" if pos.stop_loss >= entry else "STOP_LOSS_HIT"

        spot_tgt_hit = spot_ok and pos.spot_target > 0 and (
            spot >= pos.spot_target if is_call else spot <= pos.spot_target
        )
        if mark >= pos.target or spot_tgt_hit:
            return "TARGET_HIT"

        # Time stop: a long option that is not working is bleeding theta.
        started = pos.decision_time or pos.entry_timestamp
        minutes = int((now - started).total_seconds() / 60)
        risk_unit = entry - pos.original_stop_loss
        if (
            not pos.is_hero_zero
            and minutes >= cfg.time_stop_minutes
            and risk_unit > 0
            and (mark - entry) / risk_unit < cfg.time_stop_min_r
        ):
            return "TIME_STOP"
        return None

    @staticmethod
    def _trail_premium_stop(pos: PaperPosition, mark: float) -> bool:
        """Same ratchet as RiskManager._trail_premium_stop (never loosens)."""
        entry = pos.signal_entry_price
        dist = pos.target - entry
        if dist <= 0 or pos.quantity <= 0:
            return False
        progress = (mark - entry) / dist
        if progress >= 0.75:
            new_stop = entry + 0.3 * dist
        elif progress >= 0.5:
            costs = round_trip_cost(entry, entry, pos.quantity, pos.exchange).total
            new_stop = entry + pos.half_spread + costs / pos.quantity
        else:
            return False
        if new_stop > pos.stop_loss:
            pos.stop_loss = pos.trailing_sl = round(new_stop, 2)
            return True
        return False

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def close_position(
        self,
        position_id: str,
        exit_price: Optional[float],
        exit_reason: str,
        now: Optional[datetime] = None,
    ) -> Optional[PaperTrade]:
        """Close at a premium *exit_price* mark (fill = mark − half spread,
        the RiskManager.close_position convention).  ``None`` → last mark."""
        with self._lock:
            pos = self.open_positions.get(position_id)
            if pos is None:
                logger.warning("close_position: position %s not found", position_id)
                return None
            now = _aware(now or get_ist_now())
            mark = exit_price if exit_price and exit_price > 0 else (
                pos.current_price or pos.option_premium_at_entry or pos.execution_price
            )
            fill = mark - self._half_spread(pos, mark)
            trade = self._close(pos, fill, mark, exit_reason, now, "mark-half_spread", "manual", None)
            self.save_state()
            return trade

    def _close(
        self,
        pos: PaperPosition,
        fill: float,
        mark: float,
        reason: str,
        now: datetime,
        fill_source: str,
        mark_source: str,
        spot: Optional[float],
    ) -> PaperTrade:
        fill = round(max(fill, _MIN_PREMIUM), 2)
        qty = pos.quantity
        total_costs = round_trip_cost(pos.execution_price, fill, qty, pos.exchange).total
        exit_cost = total_costs - pos.entry_cost
        gross = (fill - pos.execution_price) * qty
        net = gross - total_costs
        self.cash += fill * qty - exit_cost

        mfe_value = pos.max_favorable * qty
        capture = (net / mfe_value) if mfe_value > 0.5 else 0.0
        outcome = "WIN" if net > 0 else "LOSS"

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
            quantity=qty,
            exit_price=round(mark, 2),
            actual_exit_price=fill,
            exit_timestamp=now,
            exit_reason=reason,
            gross_pnl=round(gross, 2),
            entry_cost=pos.entry_cost,
            exit_cost=round(exit_cost, 2),
            total_costs=round(total_costs, 2),
            net_pnl=round(net, 2),
            net_pnl_pct=round(net / self.config.initial_capital * 100, 4)
            if self.config.initial_capital > 0 else 0.0,
            max_favorable_excursion=round(pos.max_favorable, 2),
            max_adverse_excursion=round(pos.max_adverse, 2),
            capture_ratio=round(capture, 4),
            duration_seconds=int((now - pos.entry_timestamp).total_seconds()),
            duration_bars=pos.bars_held,
            confidence=pos.confidence,
            regime=pos.regime,
            risk_amount=pos.risk_amount,
            entry_cost_at_open=pos.entry_cost,
            outcome=outcome,
            strategy_version=pos.strategy_version,
            option_type=pos.option_type,
            decision_time=pos.decision_time,
            entry_fill_source=pos.entry_fill_source,
            exit_fill_source=fill_source,
            exit_mark_source=mark_source,
            spot_at_exit=round(spot, 2) if spot else None,
            is_hero_zero=pos.is_hero_zero,
        )

        del self.open_positions[pos.position_id]
        self.trade_history.append(trade)
        if len(self.trade_history) > self.config.max_trade_history:
            self.trade_history = self.trade_history[-self.config.max_trade_history:]
        if self.current_capital > self._peak_capital:
            self._peak_capital = self.current_capital

        day = now.date()
        self._ensure_daily_record(day)
        rec = self.daily_ledger[day]
        rec.realized_pnl += net
        rec.trades_taken += 1
        if outcome == "WIN":
            rec.trades_won += 1
        else:
            rec.trades_lost += 1

        icon = "✅" if outcome == "WIN" else "❌"
        alert_msg = (
            f"📤 PAPER EXIT: {pos.index_id} {pos.strike:.0f}{pos.option_type or ''} "
            f"@ ₹{fill:,.2f} (entry ₹{pos.execution_price:,.2f}) | "
            f"{'+' if net >= 0 else ''}₹{net:,.0f} net ({reason.replace('_', ' ')} {icon})"
        )
        logger.info(
            "PAPER EXIT: %s %s %s%s  %.2f→%.2f (%s, mark %.2f %s)  qty=%d  net ₹%.2f (%s) [%s]",
            pos.index_id, pos.trade_type, pos.strike, pos.option_type, pos.execution_price, fill,
            fill_source, mark, mark_source, qty, net, outcome, reason,
        )
        self._send_alert(alert_msg, priority="HIGH" if abs(net) > 1000 else "NORMAL")
        self._queue_ws_event("exit", {
            "position_id": pos.position_id,
            "index_id": pos.index_id,
            "trade_type": pos.trade_type,
            "entry_price": pos.execution_price,
            "exit_price": fill,
            "net_pnl": trade.net_pnl,
            "outcome": outcome,
            "exit_reason": reason,
            "mark_source": mark_source,
            "duration_seconds": trade.duration_seconds,
        })
        return trade

    def _settle_previous_day_positions(self, now: datetime) -> list[PaperTrade]:
        """A position carried past its entry day (process was down at 15:15)
        is closed as it would have been: at the last quote on its entry day
        at/before the forced-exit time, else at its last known mark."""
        closed: list[PaperTrade] = []
        for pos in list(self.open_positions.values()):
            entry_day = pos.entry_timestamp.date()
            if entry_day >= now.date():
                continue
            hhmm = self.config.hero_zero_exit_time if pos.is_hero_zero else self.config.force_exit_time
            h, m = (int(x) for x in hhmm.split(":"))
            cutoff = datetime(entry_day.year, entry_day.month, entry_day.day, h, m, tzinfo=_IST)
            quote = self._quote_between(pos, pos.entry_timestamp, cutoff, newest=True)
            if quote is not None and quote.mark is not None:
                mark, source, at = quote.mark, "quote", quote.timestamp
                fill, fill_source = (
                    (quote.bid, "bid") if quote.has_book
                    else (mark - self._half_spread(pos, mark), "ltp-half_spread")
                )
            else:
                mark = pos.current_price or pos.option_premium_at_entry or pos.execution_price
                source, at = "last_known", cutoff
                fill, fill_source = mark - self._half_spread(pos, mark), "last_known-half_spread"
            logger.warning(
                "PAPER: settling %s position %s from %s at %s mark ₹%.2f",
                pos.index_id, pos.position_id[:8], entry_day, source, mark,
            )
            closed.append(self._close(pos, fill, mark, "FORCED_EOD", at, fill_source, source, None))
        return closed

    # ------------------------------------------------------------------
    # Market data (options_chain_snapshot / price_data)
    # ------------------------------------------------------------------

    _QUOTE_SQL = (
        "SELECT timestamp, ltp, bid_price, ask_price FROM options_chain_snapshot "
        "WHERE index_id = ? AND expiry_date = ? AND strike_price = ? AND option_type = ? "
        "AND timestamp >= ? AND timestamp <= ? "
    )

    def _first_quote(self, order: PaperOrder, start: datetime, end: datetime) -> Optional[_Quote]:
        """First quote in [start, end] that has a usable price (ask or LTP)."""
        if end < start:
            return None
        try:
            row = self._db.fetch_one(
                self._QUOTE_SQL
                + "AND (COALESCE(ask_price, 0) > 0 OR COALESCE(ltp, 0) > 0) "
                "ORDER BY timestamp ASC LIMIT 1",
                (order.index_id, order.expiry, float(order.strike), order.option_type,
                 start.isoformat(), end.isoformat()),
            )
        except Exception:
            logger.exception("paper fill quote lookup failed for %s", order.signal_id[:8])
            return None
        return _row_to_quote(row)

    def _quote_between(
        self, pos: PaperPosition, start: datetime, end: datetime, newest: bool = True,
    ) -> Optional[_Quote]:
        if not pos.option_type or pos.strike is None or not pos.expiry:
            return None
        try:
            row = self._db.fetch_one(
                self._QUOTE_SQL + f"ORDER BY timestamp {'DESC' if newest else 'ASC'} LIMIT 1",
                (pos.index_id, str(pos.expiry)[:10], float(pos.strike), pos.option_type,
                 start.isoformat(), end.isoformat()),
            )
        except Exception:
            logger.exception("paper quote lookup failed for %s", pos.position_id[:8])
            return None
        return _row_to_quote(row)

    def _latest_quote(self, pos: PaperPosition, now: datetime) -> Optional[_Quote]:
        """Newest quote ≤ now and ≤ max_quote_age old (live-monitor rule)."""
        return self._quote_between(
            pos, now - timedelta(seconds=self.config.max_quote_age_seconds), now, newest=True,
        )

    def _spot(self, index_id: str, now: datetime, override: Optional[dict]) -> Optional[float]:
        if override is not None:
            v = override.get(index_id)
            return float(v) if v and float(v) > 0 else None
        bars = []
        for tf in ("1m", "5m"):
            try:
                bar = self._db.fetch_one(Q.GET_LATEST_PRICE, (index_id, tf, now.isoformat()))
            except Exception:
                logger.debug("spot lookup failed for %s %s", index_id, tf, exc_info=True)
                bar = None
            if bar:
                bars.append(bar)
        if not bars:
            return None
        bar = max(bars, key=lambda b: str(b.get("timestamp", "")))
        price = float(bar.get("close") or 0)
        ts = _parse_dt(bar.get("timestamp"))
        if price <= 0 or ts is None:
            return None
        if (now - ts).total_seconds() > self.config.max_spot_age_seconds:
            return None
        return price

    def _model_mark(self, pos: PaperPosition, spot: float, now: datetime) -> float:
        """Model re-price anchored exactly like the live RM's model mark."""
        anchor_at = pos.decision_time or pos.entry_timestamp
        return estimate_premium(
            pos.option_type, float(pos.strike), _parse_date(pos.expiry), pos.iv_at_entry or 0.15,
            spot, now, pos.option_premium_at_entry or pos.signal_entry_price,
            pos.market_price_at_signal or spot, anchor_at,
        )

    def _half_spread(self, pos: PaperPosition, mark: float) -> float:
        return pos.half_spread if pos.half_spread > 0 else mark * self.config.assumed_half_spread_pct

    @staticmethod
    def _entry_leg_cost(premium: float, quantity: int, exchange: str) -> float:
        """Statutory cost of the BUY order alone.

        ``round_trip_cost(buy, 0, qty)`` is the buy leg plus one empty sell
        order; removing that order's brokerage (+GST) leaves the buy leg.
        The exit leg is then ``round_trip_cost(buy, sell).total − entry leg``,
        so the two legs always sum to the statutory round trip.
        """
        s = schedule_for(exchange)
        rt = round_trip_cost(premium, 0.0, quantity, exchange).total
        return round(rt - s.brokerage_per_order * (1.0 + s.gst_pct), 2)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_daily_summary(self, for_date: Optional[date] = None) -> DailyPaperSummary:
        """Return a summary for *for_date* (defaults to today)."""
        today = for_date or get_ist_now().date()
        self._ensure_daily_record(today)
        rec = self.daily_ledger[today]

        today_trades = [t for t in self.trade_history if t.exit_timestamp.date() == today]
        total_pnl = sum(t.net_pnl for t in today_trades)
        best = max((t.net_pnl for t in today_trades), default=0.0)
        worst = min((t.net_pnl for t in today_trades), default=0.0)
        win_rate = (rec.trades_won / rec.trades_taken) if rec.trades_taken > 0 else 0.0
        total_pnl_pct = total_pnl / rec.starting_capital * 100 if rec.starting_capital > 0 else 0.0

        daily_limit = self.config.initial_capital * self.config.max_risk_per_day_pct / 100
        risk_used_pct = (abs(min(rec.realized_pnl, 0.0)) / daily_limit * 100) if daily_limit > 0 else 0.0

        data_fresh = (
            self._last_price_update is not None
            and (get_ist_now() - self._last_price_update).total_seconds() < _STALE_DATA_THRESHOLD_SECS
        )
        return DailyPaperSummary(
            date=today,
            starting_capital=rec.starting_capital,
            ending_capital=rec.starting_capital + total_pnl,
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
            expected_daily_return=None,
            risk_used_pct=round(risk_used_pct, 2),
            max_drawdown_today=round(rec.max_drawdown_today, 2),
            open_positions_eod=len(self.open_positions),
            data_freshness_ok=data_fresh,
            system_mode="FULL",
            anomaly_alerts_count=rec.anomaly_alerts_count,
        )

    def get_cumulative_stats(self, days: int = 30) -> PaperTradingStats:
        """Cumulative performance statistics over the last *days* days."""
        now = get_ist_now()
        today = now.date()
        cutoff = today - timedelta(days=days)

        recent = [t for t in self.trade_history if t.exit_timestamp.date() >= cutoff]
        total = len(recent)
        wins = [t for t in recent if t.outcome == "WIN"]
        losses = [t for t in recent if t.outcome == "LOSS"]

        win_rate = len(wins) / total if total > 0 else 0.0
        avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0.0
        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        total_return_pct = (
            (self.current_capital - self.config.initial_capital) / self.config.initial_capital * 100
        ) if self.config.initial_capital > 0 else 0.0

        # ── Daily returns on equity, zero-P&L days included ────────────
        pnl_by_day, start, start_equity = self._daily_pnl_window(cutoff, today)
        returns = daily_equity_returns(pnl_by_day, start, today, start_equity) if start else []
        sharpe = annualised_sharpe(returns)
        sortino = annualised_sortino(returns)
        trading_days = len(returns)
        profitable_days = sum(1 for r in returns if r > 0)
        unprofitable_days = sum(1 for r in returns if r < 0)
        daily_win_rate = profitable_days / trading_days if trading_days > 0 else 0.0
        streak = max_streak = 0
        for r in returns:
            streak = streak + 1 if r < 0 else 0
            max_streak = max(max_streak, streak)

        max_dd_pct = self._compute_max_drawdown_pct(recent, start_equity)

        high_trades = [t for t in recent if t.confidence == "HIGH"]
        med_trades = [t for t in recent if t.confidence == "MEDIUM"]
        high_wins = sum(1 for t in high_trades if t.outcome == "WIN")
        med_wins = sum(1 for t in med_trades if t.outcome == "WIN")

        annualized = total_return_pct * (252 / max(trading_days, 1))
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
            max_consecutive_losing_days=max_streak,
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

    def _daily_pnl_window(
        self, cutoff: date, end: date,
    ) -> tuple[dict[date, float], Optional[date], float]:
        """Net P&L by exit day in [start, end], the window start, and the
        equity at the start of the window.

        The window starts at the later of *cutoff* and the first day the
        paper account existed (first ledger day / first trade), so days
        before paper trading began are not counted as zero-return days.
        """
        known = [d for d in self.daily_ledger] + [t.exit_timestamp.date() for t in self.trade_history]
        if not known:
            return {}, None, self.config.initial_capital
        start = max(cutoff, min(known))
        pnl_by_day: dict[date, float] = {}
        before = 0.0
        for t in self.trade_history:
            d = t.exit_timestamp.date()
            if d < start:
                before += t.net_pnl
            elif d <= end:
                pnl_by_day[d] = pnl_by_day.get(d, 0.0) + t.net_pnl
        return pnl_by_day, start, self.config.initial_capital + before

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist cash, open positions, pending orders, ledger and trades."""
        with self._lock:
            now = get_ist_now()
            positions_json = json.dumps({
                "schema": _STATE_SCHEMA,
                "positions": [_dc_to_dict(p) for p in self.open_positions.values()],
                "pending": [_dc_to_dict(o) for o in self.pending_orders.values()],
            })
            ledger_json = json.dumps({
                "schema": _STATE_SCHEMA,
                "days": {d.isoformat(): _ledger_to_dict(r) for d, r in self.daily_ledger.items()},
                "trades": [_dc_to_dict(t) for t in self.trade_history],
                "unfilled": [_dc_to_dict(o) for o in self.unfilled_orders[-200:]],
            })
            try:
                existing = self._db.fetch_one("SELECT id FROM paper_trading_state WHERE id = 1", ())
                if existing:
                    self._db.execute(
                        """UPDATE paper_trading_state
                           SET current_capital=?, open_positions_json=?,
                               daily_ledger_json=?, updated_at=?
                           WHERE id=1""",
                        (self.cash, positions_json, ledger_json, now.isoformat()),
                    )
                else:
                    self._db.execute(
                        """INSERT INTO paper_trading_state
                           (id, current_capital, open_positions_json, daily_ledger_json, updated_at)
                           VALUES (1, ?, ?, ?, ?)""",
                        (self.cash, positions_json, ledger_json, now.isoformat()),
                    )
            except Exception as exc:
                logger.error("save_state failed: %s", exc)

    def load_state(self) -> None:
        """Restore persisted state.

        Open positions and pending orders are restored regardless of the
        day they were saved; a position left open from an earlier session
        is settled at that day's forced-exit mark (see
        :meth:`_settle_previous_day_positions`).  Legacy (index-point)
        positions cannot be valued in premium and are discarded.
        """
        with self._lock:
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
                self._restore(row)
            except Exception as exc:
                logger.error("load_state failed (starting fresh): %s", exc)
                return
            if self.open_positions and self._settle_previous_day_positions(_aware(get_ist_now())):
                self.save_state()
            logger.info(
                "load_state: cash ₹%.0f, %d open position(s), %d pending order(s), %d closed trade(s)",
                self.cash, len(self.open_positions), len(self.pending_orders), len(self.trade_history),
            )

    def _restore(self, row: dict) -> None:
        cash = float(row["current_capital"])
        positions_raw = json.loads(row["open_positions_json"] or "[]")
        ledger_raw = json.loads(row["daily_ledger_json"] or "{}")

        if isinstance(ledger_raw, dict) and ledger_raw.get("schema") == _STATE_SCHEMA:
            days_raw = ledger_raw.get("days", {})
            self.trade_history = [
                t for t in (_dc_from_dict(PaperTrade, d) for d in ledger_raw.get("trades", [])) if t
            ]
            self.unfilled_orders = [
                o for o in (_dc_from_dict(PaperOrder, d) for d in ledger_raw.get("unfilled", [])) if o
            ]
        else:
            days_raw = ledger_raw if isinstance(ledger_raw, dict) else {}
        for date_str, data in days_raw.items():
            d = date.fromisoformat(date_str)
            self.daily_ledger[d] = _ledger_from_dict(d, data, self.config.initial_capital)

        if isinstance(positions_raw, dict) and positions_raw.get("schema") == _STATE_SCHEMA:
            for d in positions_raw.get("positions", []):
                pos = _dc_from_dict(PaperPosition, d)
                if pos is not None:
                    self.open_positions[pos.position_id] = pos
            for d in positions_raw.get("pending", []):
                order = _dc_from_dict(PaperOrder, d)
                if order is not None:
                    self.pending_orders[order.order_id] = order
        elif positions_raw:
            logger.warning(
                "load_state: discarding %d legacy index-point paper position(s) — "
                "they cannot be valued in option premium",
                len(positions_raw),
            )
        self.cash = cash
        self._peak_capital = max(self.config.initial_capital, self.current_capital)

    def reset(self, confirm: bool = False) -> None:
        """Reset paper trading completely (``confirm=True`` required)."""
        if not confirm:
            raise ValueError("Pass confirm=True to reset paper trading state — this clears all history")
        with self._lock:
            self.open_positions.clear()
            self.pending_orders.clear()
            self.unfilled_orders.clear()
            self.trade_history.clear()
            self.daily_ledger.clear()
            self.missed_signals.clear()
            self.cash = self.config.initial_capital
            self._peak_capital = self.config.initial_capital
            self._kill_switch_active = False
            try:
                self._db.execute("DELETE FROM paper_trading_state WHERE id=1", ())
            except Exception as exc:
                logger.warning("reset: could not clear DB state: %s", exc)
        logger.warning("PaperTradingEngine RESET — capital restored to ₹%.0f", self.config.initial_capital)

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
    # Convenience helpers
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
        """Cumulative buy-and-hold return since paper trading started."""
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
    # Private — statistics
    # ------------------------------------------------------------------

    def _compute_max_drawdown_pct(self, trades: list[PaperTrade], start_equity: Optional[float] = None) -> float:
        if not trades:
            return 0.0
        capital = start_equity if start_equity is not None else self.config.initial_capital
        peak = capital
        max_dd = 0.0
        for t in sorted(trades, key=lambda x: x.exit_timestamp):
            capital += t.net_pnl
            peak = max(peak, capital)
            dd = (peak - capital) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def _evaluate_edge_status(self, win_rate: float, profit_factor: float) -> str:
        if win_rate >= 0.50 and profit_factor >= 1.3:
            return "INTACT"
        if win_rate >= 0.40 or profit_factor >= 1.0:
            return "DEGRADING"
        return "GONE"

    def _uptime_pct(self) -> float:
        total_market_minutes = len(self.daily_ledger) * 375
        if total_market_minutes <= 0:
            return 100.0
        return min(self._uptime_minutes_active / total_market_minutes * 100, 100.0)

    def _data_availability_pct(self) -> float:
        return 100.0 if self.daily_ledger else 0.0

    def _update_drawdown(self, now: datetime) -> None:
        day = now.date()
        self._ensure_daily_record(day)
        rec = self.daily_ledger[day]
        eq = self.equity()
        rec.peak_capital_today = max(rec.peak_capital_today, eq)
        rec.max_drawdown_today = max(rec.max_drawdown_today, rec.peak_capital_today - eq)

    # ------------------------------------------------------------------
    # Private — bookkeeping
    # ------------------------------------------------------------------

    def _ensure_daily_record(self, d: date) -> None:
        if d not in self.daily_ledger:
            cap = self.current_capital
            self.daily_ledger[d] = DailyRecord(date=d, starting_capital=cap, peak_capital_today=cap)

    def _signal_seen(self, signal_id: str) -> bool:
        if any(p.signal_id == signal_id for p in self.open_positions.values()):
            return True
        if any(o.signal_id == signal_id for o in self.pending_orders.values()):
            return True
        if any(o.signal_id == signal_id for o in self.unfilled_orders):
            return True
        return any(t.signal_id == signal_id for t in self.trade_history[-500:])

    def _check_strategy_version(self) -> None:
        if STRATEGY_VERSION != self._active_strategy_version:
            logger.warning("Strategy version changed: %s → %s", self._active_strategy_version, STRATEGY_VERSION)
            self._send_alert(
                f"📌 Strategy version changed: {self._active_strategy_version} → {STRATEGY_VERSION}. "
                f"Performance will be tracked separately."
            )
            self._active_strategy_version = STRATEGY_VERSION

    def _maybe_track_missed(self, signal, reason: str) -> None:
        if not self.config.track_missed_signals:
            return
        option = bool(getattr(signal, "option_type", None)) and (getattr(signal, "entry_premium", 0) or 0) > 0
        if option:
            entry = float(signal.entry_premium)
            target = float(getattr(signal, "premium_target", 0.0) or 0.0)
            sl = float(getattr(signal, "premium_stop", 0.0) or 0.0)
        else:
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
            signal_id=getattr(signal, "signal_id", None) or str(uuid.uuid4()),
        ))

    def _log_skip(self, record: DailyRecord, signal, reason: str, detail: str) -> None:
        record.signals_filtered += 1
        if self.config.track_no_trade_reasons:
            logger.debug(
                "PAPER SKIP [%s %s] %s: %s",
                getattr(signal, "index_id", "?"), getattr(signal, "signal_type", "?"), reason, detail,
            )

    def _entry_message(self, pos: PaperPosition) -> str:
        return (
            f"📝 PAPER TRADE: {pos.trade_type.replace('_', ' ')} {pos.index_id} "
            f"{pos.strike:.0f}{pos.option_type} {pos.expiry} @ ₹{pos.execution_price:,.2f} "
            f"({pos.lots} lot{'s' if pos.lots != 1 else ''}, qty {pos.quantity}) | "
            f"SL ₹{pos.stop_loss:,.2f}  Tgt ₹{pos.target:,.2f} | Risk ₹{pos.risk_amount:,.0f}"
        )

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

    @staticmethod
    def _exchange_for(index_id: str) -> str:
        try:
            idx = get_registry().get(index_id)
            exch = getattr(idx, "exchange", None) if idx else None
            return exch if isinstance(exch, str) and exch else "NSE"
        except Exception:
            return "NSE"

    def _ensure_schema(self) -> None:
        """Create the engine-owned paper_trading_state table if missing."""
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


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _contract_problem(signal) -> Optional[str]:
    """Why *signal* cannot be paper-traded in premium, or ``None``."""
    if not getattr(signal, "option_type", None):
        return "no option contract (not a premium-native RefinedSignal)"
    if getattr(signal, "recommended_strike", None) is None:
        return "no strike"
    if _parse_date(getattr(signal, "recommended_expiry", None)) is None:
        return "no/invalid expiry"
    if int(getattr(signal, "lots", 0) or 0) <= 0:
        return "zero lots"
    if not (getattr(signal, "quantity", 0) or getattr(signal, "lot_size", 0)):
        return "no quantity/lot size"
    for name in ("entry_premium", "premium_stop", "premium_target"):
        if not (getattr(signal, name, 0) or 0) > 0:
            return f"no {name}"
    if getattr(signal, "is_valid", True) is False:
        return "signal rejected by risk manager"
    return None


def _approved_size(signal) -> tuple[int, int, int]:
    """(lots, lot_size, quantity) exactly as the risk manager approved them."""
    lots = int(getattr(signal, "lots", 0) or 0)
    lot_size = int(getattr(signal, "lot_size", 0) or 0)
    quantity = int(getattr(signal, "quantity", 0) or 0)
    if not quantity:
        quantity = lots * lot_size
    if not lot_size and lots:
        lot_size = quantity // lots
    return lots, lot_size, quantity


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=_IST)


def _parse_dt(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _aware(value)
    try:
        return _aware(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


def _parse_date(value) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _row_to_quote(row: Optional[dict]) -> Optional[_Quote]:
    if not row:
        return None
    ts = _parse_dt(row.get("timestamp"))
    if ts is None:
        return None
    return _Quote(
        timestamp=ts,
        ltp=float(row.get("ltp") or 0.0),
        bid=float(row.get("bid_price") or 0.0),
        ask=float(row.get("ask_price") or 0.0),
    )


_DATETIME_FIELDS = {
    "entry_timestamp", "opened_at", "decision_time", "mark_timestamp",
    "exit_timestamp", "resolved_at",
}


def _dc_to_dict(obj) -> dict:
    out = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, (datetime, date)):
            v = v.isoformat()
        out[f.name] = v
    return out


def _dc_from_dict(cls, data: dict):
    try:
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in data.items() if k in names}
        for k in _DATETIME_FIELDS & kw.keys():
            kw[k] = _parse_dt(kw[k])
        return cls(**kw)
    except Exception as exc:
        logger.error("could not restore %s: %s | data=%s", cls.__name__, exc, data)
        return None


def _ledger_to_dict(rec: DailyRecord) -> dict:
    d = _dc_to_dict(rec)
    d.pop("date", None)
    return d


def _ledger_from_dict(d: date, data: dict, default_capital: float) -> DailyRecord:
    names = {f.name for f in fields(DailyRecord)} - {"date"}
    kw = {k: v for k, v in data.items() if k in names}
    kw.setdefault("starting_capital", default_capital)
    return DailyRecord(date=d, **kw)
