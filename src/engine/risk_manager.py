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
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.analysis.options_pricing import OptionPnLEstimator
from src.data.index_registry import get_registry
from src.data.options_chain import OptionsChainData
from src.database.db_manager import DatabaseManager
from src.engine.signal_generator import TradingSignal

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# Risk-free rate (approximate RBI repo rate)
_RISK_FREE_RATE: float = 0.065  # 6.5% p.a.
# Default IV used when chain data is unavailable
_DEFAULT_IV: float = 0.15       # 15% annualised


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RiskConfig:
    """All configurable risk parameters — zero-cost conservative defaults."""

    # Capital & risk limits
    total_capital: float = 100_000.0          # ₹ total trading capital
    max_risk_per_trade_pct: float = 2.0        # % of capital per trade
    max_risk_per_day_pct: float = 5.0          # % daily loss limit
    max_open_positions: int = 3                # simultaneous open positions
    max_positions_per_index: int = 1           # positions in same index
    max_lots_per_trade: int = 5                # absolute safety cap

    # Risk-reward
    min_risk_reward_ratio: float = 1.3         # reject if RR below this
    default_risk_reward: float = 1.5           # target RR

    # Execution costs
    slippage_points: float = 2.0               # expected slippage per trade
    transaction_cost_per_lot: float = 120.0    # ₹ round-trip per lot

    # Stop-loss validation thresholds
    min_sl_pct: float = 0.003                  # 0.3% — noise floor
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


@dataclass
class _OpenPosition:
    """Internal state for a single tracked open position."""

    db_id: int
    signal_id: str
    index_id: str
    signal_type: str          # "BUY_CALL" / "BUY_PUT"
    entry_price: float
    current_sl: float         # trailing stop (mutates over time)
    target_price: float
    lots: int
    lot_size: int
    entry_time: datetime
    confidence_level: str


@dataclass
class PositionUpdate:
    """Result of checking whether an open position should be exited."""

    signal_id: str
    action: str               # HOLD / EXIT_TARGET / EXIT_SL / EXIT_TRAILING_SL / EXIT_TIME / EXIT_REVERSAL
    current_pnl: float        # ₹ mark-to-market
    current_pnl_pct: float    # % of risk_amount
    time_in_trade_minutes: int


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
# Math helpers
# ---------------------------------------------------------------------------


def _ncdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun, max error 1.5e-7)."""
    if x >= 0.0:
        t = 1.0 / (1.0 + 0.2316419 * x)
        poly = t * (
            0.319381530
            + t * (-0.356563782
            + t * (1.781477937
            + t * (-1.821255978
            + t * 1.330274429)))
        )
        return 1.0 - math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi) * poly
    return 1.0 - _ncdf(-x)


def _bs_price(
    spot: float,
    strike: float,
    tte: float,       # years to expiry
    iv: float,        # annualised volatility (0.15 = 15%)
    option_type: str, # "CE" or "PE"
    rate: float = _RISK_FREE_RATE,
) -> tuple[float, dict]:
    """Black-Scholes price and first-order greeks (delta, theta/day)."""
    if tte <= 0.0 or iv <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return 0.0, {"delta": 0.0, "theta": 0.0}

    sqrt_t = math.sqrt(tte)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * tte) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    disc = math.exp(-rate * tte)

    if option_type == "CE":
        price = spot * _ncdf(d1) - strike * disc * _ncdf(d2)
        delta = _ncdf(d1)
    else:
        price = strike * disc * _ncdf(-d2) - spot * _ncdf(-d1)
        delta = _ncdf(d1) - 1.0

    # Approximate daily theta (negative = time decay cost)
    pdf_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    common_theta = -(spot * pdf_d1 * iv) / (2.0 * sqrt_t)
    if option_type == "CE":
        theta = (common_theta - rate * strike * disc * _ncdf(d2)) / 365.0
    else:
        theta = (common_theta + rate * strike * disc * _ncdf(-d2)) / 365.0

    return max(price, 0.0), {
        "delta": round(delta, 4),
        "theta": round(theta, 2),
    }


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
            from src.utils.market_hours import MarketHoursManager
            from src.utils.date_utils import get_ist_now

            mh = MarketHoursManager()
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
                resolution = "EXPIRED"
                exit_price = None
                pnl = None

                if mh.is_market_open():
                    try:
                        latest = self._db.fetch_one(
                            """SELECT close FROM price_data
                               WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1""",
                            (signal['index_id'],)
                        )
                        if latest and latest['close']:
                            current_price = latest['close']
                            exit_price = current_price

                            if signal['signal_type'] == 'BUY_CALL':
                                if current_price <= signal['stop_loss']:
                                    resolution = "LOSS"
                                    pnl = signal['stop_loss'] - signal['entry_price']
                                elif current_price >= signal['target_price']:
                                    resolution = "WIN"
                                    pnl = signal['target_price'] - signal['entry_price']
                            elif signal['signal_type'] == 'BUY_PUT':
                                if current_price >= signal['stop_loss']:
                                    resolution = "LOSS"
                                    pnl = signal['entry_price'] - signal['stop_loss']
                                elif current_price <= signal['target_price']:
                                    resolution = "WIN"
                                    pnl = signal['entry_price'] - signal['target_price']
                    except Exception as e:
                        logger.debug(f"Could not determine outcome for signal {signal['id']}: {e}")

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
    ) -> RefinedSignal:
        """
        Validate and refine a raw TradingSignal into an executable RefinedSignal.

        All eight refinement steps are applied in sequence.  Any rejection
        immediately short-circuits remaining steps and returns an invalid
        ``RefinedSignal`` with populated ``rejection_reasons``.

        Parameters
        ----------
        signal:
            Raw signal from SignalGenerator.
        current_chain:
            Live options chain (optional — used for strike selection and S/R).
        _now:
            Current time override (default: ``datetime.now(tz=_IST)``).
            Inject a fixed time in tests or backtesting to avoid clock dependency.
        """
        self._maybe_refresh_daily_cache()

        refined = self._init_refined(signal)

        if signal.signal_type == "NO_TRADE":
            refined.is_valid = False
            refined.rejection_reasons.append("Signal type is NO_TRADE")
            return refined

        # --- Step 0: time gate -------------------------------------------
        now = _now if _now is not None else datetime.now(tz=_IST)
        if self._is_near_close(now, self.config.no_entry_after_minutes_before_close):
            refined.is_valid = False
            refined.rejection_reasons.append(
                f"No new entries within {self.config.no_entry_after_minutes_before_close} "
                "minutes of market close"
            )
            return refined

        # --- Step 1: validate / refine stop-loss -------------------------
        self._step_validate_sl(refined, current_chain)
        if not refined.is_valid:
            return refined

        # --- Step 2: validate / refine target ----------------------------
        self._step_validate_target(refined, current_chain)
        if not refined.is_valid:
            return refined

        # --- Step 3: position sizing -------------------------------------
        lot_size = self._get_lot_size(signal.index_id)
        if lot_size is None:
            refined.is_valid = False
            refined.rejection_reasons.append(
                f"{signal.index_id} has no F&O lot size — cannot size position"
            )
            return refined

        # Early capital adequacy check — fail fast before daily-limit math
        sl_dist_early = abs(refined.refined_entry - refined.refined_stop_loss)
        min_1lot_cost = sl_dist_early * lot_size + self.config.transaction_cost_per_lot * 2
        if min_1lot_cost > self.config.total_capital:
            refined.is_valid = False
            refined.rejection_reasons.append(
                f"Insufficient capital: 1-lot risk ₹{min_1lot_cost:.0f} "
                f"> total capital ₹{self.config.total_capital:.0f}"
            )
            return refined

        daily_remaining = self._compute_daily_remaining()
        refined.daily_loss_remaining = daily_remaining

        lots, risk_amount = self._step_size_position(
            refined, lot_size, daily_remaining
        )
        refined.lots = lots
        refined.risk_amount = risk_amount

        # Recompute after lot clamping
        sl_dist = abs(refined.refined_entry - refined.refined_stop_loss)
        tgt_dist = abs(refined.refined_target - refined.refined_entry)
        tx_total = lots * self.config.transaction_cost_per_lot * 2  # entry + exit
        refined.transaction_cost_total = tx_total
        refined.max_loss_amount = lots * lot_size * sl_dist + tx_total
        refined.max_profit_amount = lots * lot_size * tgt_dist - tx_total
        refined.risk_pct_of_capital = (
            refined.max_loss_amount / self.config.total_capital * 100
        )
        refined.total_margin_required = 0.0  # filled in step 6 if chain available

        # --- Step 4: daily loss gate -------------------------------------
        if not self._step_check_daily_limit(refined, daily_remaining):
            return refined

        # --- Step 5: open positions gate ---------------------------------
        if not self._step_check_open_positions(refined):
            return refined

        # --- Step 6: option strike selection -----------------------------
        self._step_select_strike(refined, current_chain, lot_size)

        # --- Step 7: breakeven -------------------------------------------
        if lots > 0 and lot_size > 0:
            refined.breakeven_move = tx_total / (lots * lot_size)
            tgt_distance = abs(refined.refined_target - refined.refined_entry)
            if tgt_distance > 0 and refined.breakeven_move > tgt_distance * 0.3:
                refined.warnings.append(
                    f"High cost-to-target ratio: breakeven needs {refined.breakeven_move:.1f} pts "
                    f"({refined.breakeven_move / tgt_distance * 100:.0f}% of target distance)"
                )

        # --- Step 8: final validation ------------------------------------
        self._step_final_validation(refined)

        # Execution instructions
        refined.execution_type = "LIMIT"
        refined.limit_price = refined.refined_entry
        refined.validity = "DAY"

        return refined

    # ------------------------------------------------------------------
    # Public API — position lifecycle
    # ------------------------------------------------------------------

    def track_open_position(self, signal: RefinedSignal) -> None:
        """
        Register a refined signal as an open position.

        Updates the DB row (outcome → 'OPEN', refined details stored in
        reasoning JSON) and creates an in-memory :class:`_OpenPosition`.
        """
        if not signal.is_valid:
            logger.warning(
                "track_open_position called with invalid signal %s — skipped",
                signal.signal_id,
            )
            return

        lot_size = self._get_lot_size(signal.index_id) or 1

        # Persist risk details into the existing DB row
        db_id = self._find_signal_db_id(signal)
        if db_id is not None:
            self._persist_risk_details(db_id, signal)

        # In-memory tracker
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
        )
        self._open_positions[signal.signal_id] = pos

        logger.info(
            "OPEN  %-12s %s  Entry=%.2f  SL=%.2f  Target=%.2f  Lots=%d",
            signal.index_id, signal.signal_type,
            signal.refined_entry, signal.refined_stop_loss,
            signal.refined_target, signal.lots,
        )

    def update_position(
        self,
        signal_id: str,
        current_price: float,
        _now: Optional[datetime] = None,  # injectable for tests
    ) -> PositionUpdate:
        """
        Check exit conditions for an open position and apply trailing SL.

        Returns a :class:`PositionUpdate` describing the recommended action.
        """
        pos = self._open_positions.get(signal_id)
        if pos is None:
            return PositionUpdate(
                signal_id=signal_id,
                action="HOLD",
                current_pnl=0.0,
                current_pnl_pct=0.0,
                time_in_trade_minutes=0,
            )

        now = _now if _now is not None else datetime.now(tz=_IST)
        time_in_trade = int((now - pos.entry_time).total_seconds() / 60)

        is_call = pos.signal_type == "BUY_CALL"
        sign = 1 if is_call else -1

        pnl_points = sign * (current_price - pos.entry_price)
        pnl_rupees = pnl_points * pos.lots * pos.lot_size

        # --- Force exit near close (intraday) ---
        if self._is_near_close(now, self.config.force_exit_minutes_before_close):
            return PositionUpdate(
                signal_id=signal_id,
                action="EXIT_TIME",
                current_pnl=round(pnl_rupees, 2),
                current_pnl_pct=round(
                    pnl_rupees / self.config.total_capital * 100, 4
                ),
                time_in_trade_minutes=time_in_trade,
            )

        # --- Target hit ---
        if is_call and current_price >= pos.target_price:
            return PositionUpdate(
                signal_id=signal_id,
                action="EXIT_TARGET",
                current_pnl=round(pnl_rupees, 2),
                current_pnl_pct=round(
                    pnl_rupees / self.config.total_capital * 100, 4
                ),
                time_in_trade_minutes=time_in_trade,
            )
        if not is_call and current_price <= pos.target_price:
            return PositionUpdate(
                signal_id=signal_id,
                action="EXIT_TARGET",
                current_pnl=round(pnl_rupees, 2),
                current_pnl_pct=round(
                    pnl_rupees / self.config.total_capital * 100, 4
                ),
                time_in_trade_minutes=time_in_trade,
            )

        # --- Stop loss / trailing SL hit ---
        # TRAILING_SL = SL has been ratcheted into profitable territory
        # (for CALL: SL >= entry; for PUT: SL <= entry)
        if is_call and current_price <= pos.current_sl:
            action = (
                "EXIT_TRAILING_SL"
                if pos.current_sl >= pos.entry_price
                else "EXIT_SL"
            )
            return PositionUpdate(
                signal_id=signal_id,
                action=action,
                current_pnl=round(pnl_rupees, 2),
                current_pnl_pct=round(
                    pnl_rupees / self.config.total_capital * 100, 4
                ),
                time_in_trade_minutes=time_in_trade,
            )
        if not is_call and current_price >= pos.current_sl:
            action = (
                "EXIT_TRAILING_SL"
                if pos.current_sl <= pos.entry_price
                else "EXIT_SL"
            )
            return PositionUpdate(
                signal_id=signal_id,
                action=action,
                current_pnl=round(pnl_rupees, 2),
                current_pnl_pct=round(
                    pnl_rupees / self.config.total_capital * 100, 4
                ),
                time_in_trade_minutes=time_in_trade,
            )

        # --- Apply trailing SL adjustments (only moves in favour) --------
        self._apply_trailing_sl(pos, current_price)

        return PositionUpdate(
            signal_id=signal_id,
            action="HOLD",
            current_pnl=round(pnl_rupees, 2),
            current_pnl_pct=round(
                pnl_rupees / self.config.total_capital * 100, 4
            ),
            time_in_trade_minutes=time_in_trade,
        )

    def close_position(
        self,
        signal_id: str,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """
        Finalise an open position: compute P&L, update DB, and log.

        Parameters
        ----------
        signal_id:
            UUID of the signal to close.
        exit_price:
            Actual exit price (index level for index options).
        exit_reason:
            Human-readable reason: "TARGET", "SL", "TRAILING_SL", "TIME", etc.
        """
        pos = self._open_positions.pop(signal_id, None)
        if pos is None:
            logger.warning("close_position: signal %s not in open positions", signal_id)
            return

        is_call = pos.signal_type == "BUY_CALL"
        sign = 1 if is_call else -1
        pnl_points = sign * (exit_price - pos.entry_price)
        tx_cost = pos.lots * self.config.transaction_cost_per_lot * 2
        pnl_rupees = pnl_points * pos.lots * pos.lot_size - tx_cost

        outcome = "WIN" if pnl_rupees >= 0 else "LOSS"
        self._daily_realised_pnl += pnl_rupees

        # Update DB
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
                    (outcome, exit_price, round(pnl_rupees, 2), now_ist, pos.db_id),
                )
            except Exception:
                logger.exception("Failed to update signal %s in DB", signal_id)

        logger.info(
            "CLOSED %-12s %s: Entry=%.2f  Exit=%.2f  PnL=₹%.0f  (%s)  [%s]",
            pos.index_id, pos.signal_type,
            pos.entry_price, exit_price,
            pnl_rupees, outcome, exit_reason,
        )

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

        # Build per-position dicts (no live price — mark entry as current)
        open_list = []
        total_margin = 0.0
        for pos in self._open_positions.values():
            open_list.append({
                "signal_id": pos.signal_id,
                "index": pos.index_id,
                "signal_type": pos.signal_type,
                "entry": pos.entry_price,
                "current_sl": pos.current_sl,
                "target": pos.target_price,
                "lots": pos.lots,
                "lot_size": pos.lot_size,
                "entry_time": pos.entry_time.isoformat(),
            })
            # Very rough margin estimate: entry × lot_size × lots × 0.15
            total_margin += pos.entry_price * pos.lot_size * pos.lots * 0.15

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
        # Only apply if the resulting SL stays within the max allowed distance
        # (avoids widening SL to a far-away OI level that exceeds max_sl_pct).
        if chain is not None and chain.strikes:
            support, resistance = self._oi_sr_levels(chain)
            if is_call and support is not None:
                sr_floor = support * (1.0 - self.config.sl_sr_buffer_pct)
                new_dist_pct = abs(entry - sr_floor) / entry
                if sl > sr_floor and new_dist_pct <= self.config.max_sl_pct:
                    sl = sr_floor
                    refined.adjustments_made.append(
                        f"SL moved to below OI support {support:.2f} "
                        f"(− {self.config.sl_sr_buffer_pct*100:.1f}% buffer)"
                    )
            elif not is_call and resistance is not None:
                sr_ceil = resistance * (1.0 + self.config.sl_sr_buffer_pct)
                new_dist_pct = abs(entry - sr_ceil) / entry
                if sl < sr_ceil and new_dist_pct <= self.config.max_sl_pct:
                    sl = sr_ceil
                    refined.adjustments_made.append(
                        f"SL moved to above OI resistance {resistance:.2f} "
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
        if chain is not None and chain.strikes and refined.confidence_level != "HIGH":
            support, resistance = self._oi_sr_levels(chain)
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

    def _step_size_position(
        self,
        refined: RefinedSignal,
        lot_size: int,
        daily_remaining: float,
    ) -> tuple[int, float]:
        """
        Step 3 — calculate lots based on risk budget.

        Returns ``(lots, risk_amount_₹)``.
        """
        cfg = self.config
        max_risk = cfg.total_capital * cfg.max_risk_per_trade_pct / 100

        # Confidence modifier
        conf_map = {
            "HIGH": cfg.high_confidence_size_pct,
            "MEDIUM": cfg.medium_confidence_size_pct,
            "LOW": cfg.low_confidence_size_pct,
        }
        conf_pct = conf_map.get(refined.confidence_level, cfg.low_confidence_size_pct)
        risk_amount = max_risk * (conf_pct / 100.0)

        # Regime/anomaly position modifier from signal
        risk_amount *= refined.position_size_modifier

        # Clamp to a meaningful minimum
        risk_amount = max(risk_amount, 0.0)

        sl_dist = abs(refined.refined_entry - refined.refined_stop_loss)

        # Delta-adjusted SL: option moves less than the index (delta < 1).
        # Use entry delta from option_greeks if available; otherwise attempt a
        # quick BS estimate; fall back to delta=1 (conservative) if unavailable.
        entry_delta = 1.0
        if refined.option_greeks and "delta" in refined.option_greeks:
            entry_delta = abs(float(refined.option_greeks["delta"]))
        elif refined.recommended_strike:
            try:
                estimator = OptionPnLEstimator()
                days_to_expiry = getattr(refined, "days_to_expiry", None) or 7.0
                iv = estimator.get_default_iv(refined.index_id)
                entry_delta = estimator.get_entry_delta(
                    trade_type=refined.signal_type,
                    spot=refined.refined_entry,
                    strike=refined.recommended_strike,
                    days_to_expiry=float(days_to_expiry),
                    iv=iv,
                )
            except Exception:
                entry_delta = 1.0

        # Clamp delta to a sensible range (deep OTM ≥ 0.1, deep ITM ≤ 0.95)
        entry_delta = max(0.1, min(entry_delta, 0.95))

        effective_sl_distance = sl_dist * entry_delta
        loss_per_lot = effective_sl_distance * lot_size + cfg.transaction_cost_per_lot

        if loss_per_lot <= 0:
            return 1, risk_amount

        lots = math.floor(risk_amount / loss_per_lot)
        lots = max(lots, 1)           # at least 1 lot
        lots = min(lots, cfg.max_lots_per_trade)  # safety cap

        if entry_delta < 0.99:
            refined.adjustments_made.append(
                f"Delta-adjusted sizing: delta={entry_delta:.2f}, "
                f"effective SL={effective_sl_distance:.1f} pts (raw {sl_dist:.1f})"
            )

        # Verify we can actually afford 1 lot
        one_lot_cost = loss_per_lot
        if one_lot_cost > cfg.total_capital * 0.20:  # > 20% of capital for 1 lot
            refined.adjustments_made.append(
                f"Insufficient capital for even 1 lot: 1-lot risk ₹{one_lot_cost:.0f} "
                f"> 20% of capital ₹{cfg.total_capital:.0f}"
            )

        return lots, round(risk_amount, 2)

    def _step_check_daily_limit(
        self,
        refined: RefinedSignal,
        daily_remaining: float,
    ) -> bool:
        """Step 4 — enforce daily loss limit.  Returns False if trade rejected."""
        if daily_remaining <= 0:
            refined.is_valid = False
            refined.rejection_reasons.append(
                f"Daily loss limit reached — ₹{daily_remaining:.0f} remaining"
            )
            return False

        max_loss = refined.max_loss_amount
        if max_loss > daily_remaining:
            # Try reducing lots
            lot_size = self._get_lot_size(refined.index_id) or 1
            sl_dist = abs(refined.refined_entry - refined.refined_stop_loss)
            tx_per_lot = self.config.transaction_cost_per_lot * 2
            loss_per_lot = sl_dist * lot_size + tx_per_lot
            max_lots = math.floor(daily_remaining / loss_per_lot) if loss_per_lot > 0 else 0
            if max_lots < 1:
                refined.is_valid = False
                refined.rejection_reasons.append(
                    f"Trade risk ₹{max_loss:.0f} exceeds daily limit "
                    f"remaining ₹{daily_remaining:.0f} — cannot fit even 1 lot"
                )
                return False
            # Reduce lots and recalculate
            refined.lots = max_lots
            refined.max_loss_amount = max_lots * lot_size * sl_dist + max_lots * tx_per_lot
            refined.max_profit_amount = (
                max_lots * lot_size * abs(refined.refined_target - refined.refined_entry)
                - max_lots * tx_per_lot
            )
            refined.adjustments_made.append(
                f"Lots reduced from {refined.lots} to {max_lots} to stay within daily limit"
            )

        return True

    def _step_check_open_positions(self, refined: RefinedSignal) -> bool:
        """Step 5 — check open position caps.  Returns False if trade rejected."""
        total_open = len(self._open_positions)
        if total_open >= self.config.max_open_positions:
            refined.is_valid = False
            refined.rejection_reasons.append(
                f"Max open positions reached ({total_open}/{self.config.max_open_positions})"
            )
            return False

        index_open = sum(
            1 for p in self._open_positions.values()
            if p.index_id == refined.index_id
        )
        if index_open >= self.config.max_positions_per_index:
            refined.is_valid = False
            refined.rejection_reasons.append(
                f"Max positions for {refined.index_id} reached "
                f"({index_open}/{self.config.max_positions_per_index})"
            )
            return False

        return True

    def _step_select_strike(
        self,
        refined: RefinedSignal,
        chain: Optional[OptionsChainData],
        lot_size: int,
    ) -> None:
        """Step 6 — select option strike, expiry, and estimate premium."""
        if chain is None or not chain.strikes:
            # Fallback: Black-Scholes estimate with default IV
            self._bs_fallback(refined)
            return

        # --- Expiry selection -------------------------------------------
        today = date.today()
        expiry = chain.expiry_date
        days_to_expiry = (expiry - today).days

        if days_to_expiry < 1 and len(chain.available_expiries) > 1:
            expiry = chain.available_expiries[1]
            refined.adjustments_made.append(
                f"Expiry day — using next expiry {expiry} instead"
            )
        elif refined.confidence_level == "HIGH" and days_to_expiry < 3:
            if len(chain.available_expiries) > 1:
                expiry = chain.available_expiries[1]

        dte = (expiry - today).days
        refined.recommended_expiry = expiry.isoformat()

        # --- Strike selection -------------------------------------------
        spot = chain.spot_price
        is_call = refined.signal_type == "BUY_CALL"

        sorted_strikes = sorted(chain.strikes, key=lambda s: s.strike_price)
        strike_prices = [s.strike_price for s in sorted_strikes]

        # ATM = closest strike to spot
        atm_strike = min(strike_prices, key=lambda k: abs(k - spot))
        atm_idx = strike_prices.index(atm_strike)

        if refined.confidence_level == "HIGH":
            chosen_idx = atm_idx  # ATM for HIGH
        else:
            # 1 OTM for MEDIUM/LOW
            if is_call:
                chosen_idx = min(atm_idx + 1, len(strike_prices) - 1)
            else:
                chosen_idx = max(atm_idx - 1, 0)

        chosen_strike = strike_prices[chosen_idx]
        refined.recommended_strike = chosen_strike

        # --- Premium from chain ------------------------------------------
        strike_data = sorted_strikes[chosen_idx]
        ltp = strike_data.ce_ltp if is_call else strike_data.pe_ltp
        iv = strike_data.ce_iv if is_call else strike_data.pe_iv

        # Skip zero-liquidity strikes, try adjacent
        attempts = 0
        while ltp <= 0.0 and attempts < 3:
            attempts += 1
            alt_idx = (
                chosen_idx + attempts if is_call
                else chosen_idx - attempts
            )
            alt_idx = max(0, min(alt_idx, len(sorted_strikes) - 1))
            strike_data = sorted_strikes[alt_idx]
            ltp = strike_data.ce_ltp if is_call else strike_data.pe_ltp
            iv = strike_data.ce_iv if is_call else strike_data.pe_iv
            if ltp > 0:
                chosen_strike = strike_prices[alt_idx]
                refined.recommended_strike = chosen_strike
                refined.adjustments_made.append(
                    f"Zero LTP at chosen strike — shifted to {chosen_strike}"
                )

        if ltp <= 0.0:
            # All adjacent strikes illiquid — use BS
            self._bs_fallback(refined)
            return

        refined.option_premium = round(ltp, 2)

        # Margin = premium × lot_size × lots (option buyer pays premium)
        refined.total_margin_required = round(
            ltp * lot_size * refined.lots, 2
        )

        # Greeks
        if iv and iv > 0:
            tte = max(dte, 1) / 365.0
            _, greeks = _bs_price(spot, chosen_strike, tte, iv / 100.0,
                                   "CE" if is_call else "PE")
            refined.option_greeks = greeks
        else:
            refined.option_greeks = {"delta": None, "theta": None}

    def _step_final_validation(self, refined: RefinedSignal) -> None:
        """Step 8 — compile all final checks into is_valid / rejection_reasons."""
        cfg = self.config
        reasons = refined.rejection_reasons  # mutated in place

        sl_dist = abs(refined.refined_entry - refined.refined_stop_loss)
        tgt_dist = abs(refined.refined_target - refined.refined_entry)
        rr = tgt_dist / sl_dist if sl_dist > 0 else 0.0

        # Only reject on risk_pct when lots > 1.  If lots == 1 (minimum) and
        # the cost still exceeds budget it means the instrument requires more
        # capital — surfaced as a warning, not a hard rejection.
        if refined.risk_pct_of_capital > cfg.max_risk_per_trade_pct:
            if refined.lots > 1:
                reasons.append(
                    f"Risk {refined.risk_pct_of_capital:.2f}% > max {cfg.max_risk_per_trade_pct}%"
                )
            else:
                refined.warnings.append(
                    f"Risk {refined.risk_pct_of_capital:.2f}% exceeds {cfg.max_risk_per_trade_pct}% budget "
                    "at minimum 1-lot size — consider adding capital"
                )

        if rr < cfg.min_risk_reward_ratio:
            reasons.append(
                f"Final RR {rr:.2f} < minimum {cfg.min_risk_reward_ratio}"
            )

        if refined.lots < 1:
            reasons.append("Position size rounds to zero lots")

        if refined.daily_loss_remaining <= 0:
            reasons.append("Daily loss limit already reached")

        # Hard reject: RR < 1.0 is never acceptable
        if rr < 1.0 and sl_dist > 0:
            reasons.append(f"RR {rr:.2f} < 1.0 — always rejected")

        # Check capital sufficiency for 1 lot
        lot_size = self._get_lot_size(refined.index_id) or 1
        min_cost = sl_dist * lot_size + cfg.transaction_cost_per_lot * 2
        if min_cost > cfg.total_capital:
            reasons.append(
                f"Insufficient capital: 1-lot risk ₹{min_cost:.0f} "
                f"> total capital ₹{cfg.total_capital:.0f}"
            )

        refined.is_valid = len(reasons) == 0

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

    def _bs_fallback(self, refined: RefinedSignal) -> None:
        """Estimate option premium via Black-Scholes when chain is unavailable."""
        entry = refined.refined_entry
        if entry <= 0:
            return

        # Estimate ATM strike (round to nearest 50 for NIFTY-like indices)
        # For a rough estimate, use entry as both spot and strike (ATM)
        dte_assumption = 4  # assume mid-week, ~4 days to expiry
        tte = dte_assumption / 365.0
        is_call = refined.signal_type == "BUY_CALL"

        premium, greeks = _bs_price(
            entry, entry, tte, _DEFAULT_IV,
            "CE" if is_call else "PE",
        )

        refined.option_premium = round(premium, 2) if premium > 0 else None
        refined.option_greeks = greeks
        if refined.recommended_expiry is None:
            # Generic label when no chain is available
            refined.recommended_expiry = "current_weekly"
        if refined.recommended_strike is None:
            refined.recommended_strike = round(entry / 50) * 50  # nearest 50

        if premium > 0 and refined.lots > 0:
            lot_size = self._get_lot_size(refined.index_id) or 1
            refined.total_margin_required = round(
                premium * lot_size * refined.lots, 2
            )
        refined.adjustments_made.append("Option premium estimated via Black-Scholes (no live chain)")

    def _compute_daily_remaining(self) -> float:
        """Return how much ₹ of daily loss limit is still available."""
        daily_limit = self.config.total_capital * self.config.max_risk_per_day_pct / 100
        unrealised = self._unrealised_pnl_total()
        # Only negative unrealised counts against limit
        unrealised_loss = abs(min(0.0, unrealised))
        return max(0.0, daily_limit - abs(min(0.0, self._daily_realised_pnl)) - unrealised_loss)

    def _unrealised_pnl_total(self) -> float:
        """Sum of unrealised P&L across all open positions (no live price)."""
        # Without a live price feed, return 0 (conservative — don't assume gain)
        return 0.0

    @staticmethod
    def _is_near_close(now: datetime, minutes: int) -> bool:
        """True if fewer than *minutes* remain until 15:30 IST close."""
        h, m = settings.market_hours.market_close.split(":")
        close = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
        return now >= close - timedelta(minutes=minutes)

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
                risk = reasoning.get("risk_details", {})
                signal_id = risk.get("signal_id", f"db_{row['id']}")
                lots = int(risk.get("lots", 1))
            except (json.JSONDecodeError, TypeError, KeyError):
                signal_id = f"db_{row['id']}"
                lots = 1

            lot_size = self._get_lot_size(row["index_id"]) or 1
            try:
                entry_time = datetime.fromisoformat(row["generated_at"])
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=_IST)
            except (ValueError, TypeError):
                entry_time = datetime.now(tz=_IST)

            pos = _OpenPosition(
                db_id=row["id"],
                signal_id=signal_id,
                index_id=row["index_id"],
                signal_type=row["signal_type"],
                entry_price=float(row["entry_price"]),
                current_sl=float(row["stop_loss"]),
                target_price=float(row["target_price"]),
                lots=lots,
                lot_size=lot_size,
                entry_time=entry_time,
                confidence_level=row["confidence_level"],
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
        if self._cache_date != date.today():
            self._open_positions.clear()
            self._load_open_positions()
            self._refresh_daily_pnl()

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

    def _persist_risk_details(self, db_id: int, signal: RefinedSignal) -> None:
        """Append risk_details to the reasoning JSON in the DB row."""
        try:
            row = self._db.fetch_one(
                "SELECT reasoning FROM trading_signals WHERE id = ?",
                (db_id,),
            )
            if row:
                try:
                    reasoning = json.loads(row["reasoning"])
                except (json.JSONDecodeError, TypeError):
                    reasoning = {}
                reasoning["risk_details"] = {
                    "signal_id": signal.signal_id,
                    "lots": signal.lots,
                    "refined_entry": signal.refined_entry,
                    "refined_target": signal.refined_target,
                    "refined_stop_loss": signal.refined_stop_loss,
                    "risk_amount": signal.risk_amount,
                    "max_loss_amount": signal.max_loss_amount,
                    "recommended_strike": signal.recommended_strike,
                    "option_premium": signal.option_premium,
                }
                self._db.execute(
                    "UPDATE trading_signals SET reasoning = ? WHERE id = ?",
                    (json.dumps(reasoning, ensure_ascii=False), db_id),
                )
        except Exception:
            logger.debug(
                "Could not persist risk details for signal %s", signal.signal_id,
                exc_info=True,
            )
