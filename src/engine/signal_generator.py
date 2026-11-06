"""
Signal Generator — Phase 5.2 of the Trading Decision Support System.

Combines Phase 2 (Technical), Phase 3 (News), Phase 4 (Anomaly), and Phase 5.1
(MarketRegime) inputs into a final TradingSignal with entry/exit levels,
confidence scoring, and human-readable reasoning.

Algorithm overview
------------------
1. Safety checks  — bail early on unsafe market conditions
2. Vote collection — convert each input to a numeric score in [-2, +2]
3. Weighted score  — regime-aware weighted sum, with special-case boosts
4. Direction       — BUY_CALL / BUY_PUT / NO_TRADE from score thresholds
5. Confidence      — base from score magnitude, boosted/reduced by context
6. Levels          — entry = spot, stop/target from ATR + S/R
7. Reasoning       — full multi-line human-readable explanation
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from config.settings import settings
from src.analysis.anomaly.anomaly_aggregator import AnomalyVote
from src.analysis.news.news_engine import NewsVote
from src.analysis.technical_aggregator import TechnicalAnalysisResult
from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from src.engine.regime_detector import MarketRegime, RegimeDetector, SignalWeights
from src.engine.signal_calibration import SignalCalibration, get_signal_calibration

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Rich in-memory TradingSignal (superset of the DB model)
# ---------------------------------------------------------------------------


@dataclass
class TradingSignal:
    """Full trading signal produced by SignalGenerator.

    The DB ``trading_signals`` table stores a simplified subset; this class
    carries the complete in-memory representation used by the API and tests.
    """

    # Identification
    signal_id: str           # UUID
    index_id: str
    generated_at: datetime

    # Direction & confidence
    signal_type: str         # "BUY_CALL" / "BUY_PUT" / "NO_TRADE"
    confidence_level: str    # "HIGH" / "MEDIUM" / "LOW"
    confidence_score: float  # 0–1

    # Price levels
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float

    # Context
    regime: str              # MarketRegime.regime string
    weighted_score: float    # Raw weighted score (pre-direction)

    # Per-component transparency
    vote_breakdown: dict     # {component: {score, confidence, weight, weighted_contribution}}

    # Risk parameters
    risk_level: str               # From AnomalyVote (NORMAL / ELEVATED / HIGH / EXTREME)
    position_size_modifier: float # Combined regime × anomaly × volatility modifier
    suggested_lot_count: int      # Rounded from position_size_modifier
    estimated_max_loss: float     # Points lost if stop hit × lot count
    estimated_max_profit: float   # Points gained if target hit × lot count

    # Human-readable output
    reasoning: str
    warnings: list[str] = field(default_factory=list)

    # Outcome tracking (filled post-trade)
    outcome: Optional[str] = None           # "WIN" / "LOSS" / "OPEN" / "EXPIRED"
    actual_exit_price: Optional[float] = None
    actual_pnl: Optional[float] = None
    closed_at: Optional[datetime] = None

    # Structured audit trail for replay/debug (exact intermediate values)
    signal_audit: Optional[dict] = None

    # Metadata
    data_completeness: float = 1.0      # 0–1, fraction of data sources available
    signals_generated_today: int = 0    # Running count for this index today

    # Hero Zero metadata (populated only when this is a HZ signal)
    is_hero_zero: bool = False
    hero_zero_result: Optional[object] = None  # HeroZeroResult (avoid circular import)

    # Strike selection (populated by DecisionEngine._select_optimal_strike)
    recommended_strike: Optional[float] = None   # Optimal strike price (~0.4 delta)
    recommended_delta: float = 0.0               # Approximate delta of recommended strike
    recommended_expiry: Optional[str] = None     # Nearest expiry date string


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confidence = calibrated P(win) (see src/engine/signal_calibration.py).
# The calibration labels a win as "target 1.5×ATR reached before stop
# 1.0×ATR within 60 min", so break-even is P = 0.40 before costs; MEDIUM
# demands 0.50 to leave room for option costs, theta and model error.
_CONF_HIGH: float = 0.58
_CONF_MEDIUM: float = 0.50
_CONF_LOW: float = 0.10
_CONF_FLOOR: float = 0.05    # absolute minimum (never return 0)
_CONF_CEILING: float = 0.92  # absolute maximum (never return 1)

# Weighted-score → direction thresholds
_SCORE_CALL_THRESHOLD: float = 0.25
_SCORE_PUT_THRESHOLD: float = -0.25

# Bar staleness: a bar may be at most this many intervals old (plus a fixed
# allowance for collection latency) before the decision is refused.
_TIMEFRAME_SECONDS: dict[str, int] = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "1h": 3600}
_STALE_BAR_INTERVALS: int = 2
_STALE_MIN_SECONDS: int = 600


def _max_bar_age_seconds(timeframe: str) -> float:
    """Maximum acceptable age of the newest bar for an intraday decision.

    Unknown / daily timeframes get the fixed floor, so a daily bar (stamped
    at midnight) is always stale intraday — daily ATR/levels must not drive
    an intraday option entry.
    """
    tf_secs = _TIMEFRAME_SECONDS.get((timeframe or "").lower())
    if tf_secs is None:
        return float(_STALE_MIN_SECONDS)
    return float(max(_STALE_MIN_SECONDS, _STALE_BAR_INTERVALS * tf_secs))

# Map MarketRegime.regime → DB regime CHECK constraint values
_REGIME_TO_DB: dict[str, str] = {
    "STRONG_TREND_UP": "TRENDING",
    "TREND_UP": "TRENDING",
    "STRONG_TREND_DOWN": "TRENDING",
    "TREND_DOWN": "TRENDING",
    "EVENT_DRIVEN": "EVENT",
    "RANGE_BOUND": "RANGE_BOUND",
    "VOLATILE_CHOPPY": "RANGE_BOUND",
    "BREAKOUT": "RANGE_BOUND",
    "CRASH": "RANGE_BOUND",
}

# Numeric vote lookup  (case-insensitive via .upper() at call site)
_VOTE_NUMERIC: dict[str, float] = {
    "STRONG_BULLISH": 2.0,
    "STRONGLY_BULLISH": 2.0,
    "STRONG_BUY": 2.0,
    "BULLISH": 1.0,
    "BUY": 1.0,
    "SLIGHTLY_BULLISH": 0.5,
    "NEUTRAL": 0.0,
    "NO_TRADE": 0.0,
    "CAUTION": 0.0,          # directionally neutral
    "SLIGHTLY_BEARISH": -0.5,
    "BEARISH": -1.0,
    "SELL": -1.0,
    "STRONG_BEARISH": -2.0,
    "STRONGLY_BEARISH": -2.0,
    "STRONG_SELL": -2.0,
}

# Attribute name on SignalWeights for each component
# News engine excluded: adds lag, not edge. Weight redistributed to options + smart_money.
_WEIGHT_ATTR: dict[str, str] = {
    "trend": "trend_weight",
    "momentum": "momentum_weight",
    "options": "options_weight",
    "volume": "volume_weight",
    "smart_money": "smart_money_weight",
    "anomaly": "anomaly_weight",
}

# All component names (used for redistribution logic)
_ALL_COMPONENTS: frozenset[str] = frozenset(_WEIGHT_ATTR.keys())

# Components that form the calibrated score.  Options and anomaly have no
# point-in-time history to calibrate against; they veto or haircut instead.
_SCORED_COMPONENTS: frozenset[str] = frozenset({"trend", "momentum", "volume", "smart_money"})

# Probability haircuts for conditions the calibration does not see.  Only
# penalties: un-measured boosts are how the old confidence inflated itself.
_HAIRCUT_ANOMALY_HIGH = 0.05
_HAIRCUT_ANOMALY_CAUTION = 0.03
_HAIRCUT_REGIME_CHANGING = 0.02
_HAIRCUT_EXTREME_VOL = 0.05
_HAIRCUT_IV_EXPENSIVE = 0.05
_HAIRCUT_BANKNIFTY_LAGGING = 0.03


# ---------------------------------------------------------------------------
# Signal Deduplicator
# ---------------------------------------------------------------------------


class SignalDeduplicator:
    """Suppress a repeat of the same signal on the same index within a window.

    Keyed by ``(index_id, signal_type)`` with the time of the last accepted
    signal.  The old version keyed on 3-minute clock buckets (a 5-minute
    decision cycle never lands in the same bucket twice, so nothing was ever
    suppressed) and expired entries against the wall clock (so replays never
    matched either).  All arithmetic now uses the signal's own timestamp.
    """

    def __init__(self, window_seconds: int = 300) -> None:
        self.window_seconds = window_seconds
        self._last: dict[tuple[str, str], datetime] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _aware(ts: datetime) -> datetime:
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=_IST)

    def is_duplicate(self, index_id: str, signal_type: str, timestamp: datetime) -> bool:
        """True if the same signal was accepted < window_seconds before *timestamp*."""
        if signal_type == "NO_TRADE":
            return False
        ts = self._aware(timestamp)
        with self._lock:
            last = self._last.get((index_id, signal_type))
            if last is None:
                return False
            age = (ts - last).total_seconds()
            if 0 <= age < self.window_seconds:
                logger.warning(
                    "Duplicate signal suppressed: %s %s (previous at %s)",
                    index_id, signal_type, last.isoformat(),
                )
                return True
            return False

    def record(self, index_id: str, signal_type: str, timestamp: datetime) -> None:
        """Record an accepted signal."""
        if signal_type == "NO_TRADE":
            return
        with self._lock:
            self._last[(index_id, signal_type)] = self._aware(timestamp)


# ---------------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------------


class SignalGenerator:
    """Combines Phase 2–5.1 inputs into a final TradingSignal.

    Parameters
    ----------
    db:
        Connected DatabaseManager used for persisting signals and loading
        historical counts.
    """

    def __init__(self, db: DatabaseManager, calibration: Optional[SignalCalibration] = None) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._calibration = calibration or get_signal_calibration()

        # Minimum confidence thresholds (could be loaded from settings)
        self._min_conf_high = _CONF_HIGH
        self._min_conf_medium = _CONF_MEDIUM
        self._min_conf_low = _CONF_LOW

        # In-memory cache of signals generated today per index.
        # Keyed by index_id; reset at midnight.
        self._today_signals: dict[str, list[TradingSignal]] = {}
        self._today_date: Optional[str] = None  # "YYYY-MM-DD" IST

        # Dynamic whitelist for F&O stocks (top priority symbols)
        self._dynamic_allowed_symbols: set[str] = set()

        # Deduplication — suppress duplicate signals within a 5-minute window
        # Same signal on the same index within 15 min (three 5-min cycles) is a repeat
        self.deduplicator = SignalDeduplicator(window_seconds=900)

    def add_allowed_symbol(self, symbol: str) -> None:
        """Dynamically whitelist a symbol for signal generation.
        
        This allows F&O stocks from the priority watchlist to pass the
        safety gate that usually restricts signals to major indices.
        """
        with self._lock:
            self._dynamic_allowed_symbols.add(symbol.upper())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        index_id: str,
        technical: TechnicalAnalysisResult,
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
        regime: MarketRegime,
        current_spot_price: float,
        mode: str = "FULL",
        hero_zero_result: Optional[object] = None,
        timestamp: Optional[datetime] = None,
    ) -> TradingSignal:
        """Generate a TradingSignal from all available analysis inputs.

        This is the main entry point.  For NO_TRADE outcomes the signal is
        not persisted to the database; actionable signals (BUY_CALL /
        BUY_PUT) are inserted into ``trading_signals``.

        Parameters
        ----------
        index_id:
            Index identifier, e.g. ``"NIFTY50"``.
        technical:
            Phase 2 output — always required.
        news:
            Phase 3 NewsVote — may be ``None`` (weight redistributed).
        anomaly:
            Phase 4 AnomalyVote — may be ``None`` (weight redistributed).
        regime:
            Phase 5.1 MarketRegime — always required.
        current_spot_price:
            Latest spot price used as the entry reference.
        mode:
            Backtest/trading mode — ``"TECHNICAL_ONLY"``, ``"TECHNICAL_OPTIONS"``,
            or ``"FULL"`` (default). Controls which missing-source penalties apply.
        timestamp:
            Override for the current time (used in backtesting so that
            time-based checks like market-close proximity use the bar's
            timestamp rather than the real wall-clock time).
        """
        now = timestamp if timestamp is not None else datetime.now(tz=_IST)
        self._refresh_daily_cache(now)
        warnings: list[str] = []

        # ── Step 1: Safety Checks ──────────────────────────────────────
        no_trade = self._run_safety_checks(
            index_id, current_spot_price, technical, news, anomaly, regime, now, warnings,
        )
        if no_trade is not None:
            return no_trade

        # ── Step 1.5: Hero Zero Override ───────────────────────────────
        if hero_zero_result is not None and getattr(hero_zero_result, "is_triggered", False):
            return self._build_hero_zero_signal(
                index_id, now, current_spot_price, hero_zero_result, regime, technical, news, anomaly
            )

        # ── Step 2: Model score (measured components only) ────────────
        # Only components the calibration replay can measure contribute to
        # the score: trend, momentum, and volume/smart-money when the series
        # has real volume.  Options and anomaly have no historical record to
        # calibrate against, so they act as vetoes/haircuts (below), not as
        # additive weight that silently changes what the score means.
        weighted_score, votes, _scored = self.model_score(technical, regime)

        # ── Step 3: Direction ──────────────────────────────────────────
        direction = self._determine_direction(weighted_score, regime.regime)
        if direction == "NO_TRADE":
            return self._make_no_trade(
                index_id, now,
                f"Score {weighted_score:+.2f} in dead zone "
                f"(|score| < {_SCORE_CALL_THRESHOLD:.2f}) — no clear edge",
                regime, technical, news, anomaly, warnings,
                weighted_score=weighted_score,
            )

        # ── Step 4: Vetoes from unmeasured components ──────────────────
        veto = self._veto_reason(direction, technical, anomaly)
        if veto:
            return self._make_no_trade(
                index_id, now, veto, regime, technical, news, anomaly, warnings,
                weighted_score=weighted_score,
            )

        # ── Step 5: Calibrated confidence ──────────────────────────────
        confidence_score, confidence_level, cal_reason = self._calibrated_confidence(
            index_id, direction, weighted_score, technical, anomaly, regime, warnings,
        )
        if confidence_score is None or confidence_level == "LOW":
            reason = (
                f"Model not tradeable: {cal_reason}" if confidence_score is None
                else f"Calibrated P(win) {confidence_score:.2f} < {self._min_conf_medium:.2f}"
            )
            return self._make_no_trade(
                index_id, now, reason, regime, technical, news, anomaly, warnings,
                weighted_score=weighted_score,
            )

        # ── Smart money & expiry warnings (post-direction) ─────────────
        warnings.extend(self._smart_money_warnings(direction, technical))
        warnings.extend(self._expiry_warnings(direction, technical, regime, now))

        if self._has_conflicting_strong_signals(votes):
            return self._make_no_trade(
                index_id, now,
                "Conflicting strong signals detected — direction unclear",
                regime, technical, news, anomaly, warnings,
                weighted_score=weighted_score,
            )

        # ── Cooldown: recent duplicate signal within 15 min ────────────
        self._check_cooldown(index_id, direction, now, warnings)

        # ── Step 6: Entry / Exit Levels ────────────────────────────────
        stop_atr, target_atr, min_stop_pct = self._calibration.label_multiples()
        entry_price, target_price, stop_loss, rr_ratio = self._calculate_levels(
            direction, current_spot_price, technical, regime, confidence_level,
            stop_atr=stop_atr, target_atr=target_atr, min_stop_pct=min_stop_pct,
        )

        # ── R:R Gate — reject signals with insufficient risk/reward ────
        min_rr_ratio = 1.3  # minimum 1:1.3 (spot); premium R:R is re-checked in risk
        if rr_ratio < min_rr_ratio:
            return self._make_no_trade(
                index_id, now,
                f"Insufficient R:R ({rr_ratio:.2f} < {min_rr_ratio:.2f} minimum) — skip",
                regime, technical, news, anomaly, warnings,
                weighted_score=weighted_score,
            )

        # ── Position sizing ────────────────────────────────────────────
        pos_mod, lot_count, max_loss, max_profit = self._position_sizing(
            regime, anomaly, technical, entry_price, target_price, stop_loss,
        )

        # ── Misc context ───────────────────────────────────────────────
        risk_level = anomaly.risk_level if anomaly else "NORMAL"
        data_completeness = self._data_completeness(technical, news, anomaly)
        vote_breakdown = self._build_vote_breakdown(votes)
        today_count = self._get_signals_today_count(index_id, now)

        # ── Step 7: Reasoning ──────────────────────────────────────────
        reasoning = self._build_reasoning(
            index_id, direction, confidence_level, confidence_score,
            weighted_score, regime, votes, entry_price, target_price,
            stop_loss, rr_ratio, technical, news, anomaly, warnings,
        )

        signal = TradingSignal(
            signal_id=str(uuid.uuid4()),
            index_id=index_id,
            generated_at=now,
            signal_type=direction,
            confidence_level=confidence_level,
            confidence_score=round(confidence_score, 4),
            entry_price=round(entry_price, 2),
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            risk_reward_ratio=round(rr_ratio, 2),
            regime=regime.regime,
            weighted_score=round(weighted_score, 4),
            vote_breakdown=vote_breakdown,
            risk_level=risk_level,
            position_size_modifier=round(pos_mod, 4),
            suggested_lot_count=lot_count,
            estimated_max_loss=round(max_loss, 2),
            estimated_max_profit=round(max_profit, 2),
            reasoning=reasoning,
            warnings=list(warnings),
            data_completeness=round(data_completeness, 4),
            signals_generated_today=today_count + 1,
        )

        # Attach recommended expiry from options chain if available
        if signal.signal_type in ("BUY_CALL", "BUY_PUT") and technical.options is not None:
            try:
                expiry = getattr(technical.options, "expiry_date", None)
                if expiry:
                    object.__setattr__(signal, "recommended_expiry", str(expiry))
            except (AttributeError, TypeError):
                pass

        # ── Deduplication check ───────────────────────────────────────
        if signal.signal_type != "NO_TRADE":
            if self.deduplicator.is_duplicate(index_id, signal.signal_type, now):
                signal.signal_type = "NO_TRADE"
                signal.reasoning = (
                    "Signal suppressed: same signal within the last 15 minutes\n"
                    + signal.reasoning
                )
                signal.warnings.append("DUPLICATE_SUPPRESSED")
                return signal
            self.deduplicator.record(index_id, signal.signal_type, now)

        # ── Structured audit trail ────────────────────────────────────
        signal.signal_audit = self._build_audit_trail(
            index_id, current_spot_price, confidence_score,
            technical, news, anomaly, regime, signal, votes,
        )

        # Cache only — persistence is handled by DecisionEngine via SignalTracker
        self._today_signals.setdefault(index_id, []).append(signal)

        logger.info(
            "Signal generated for %s: %s (conf=%s %.2f, score=%.3f)",
            index_id, direction, confidence_level, confidence_score, weighted_score,
        )
        return signal

    def get_signals_today(self, index_id: str) -> list[TradingSignal]:
        """Return all signals generated today for *index_id* (in-memory cache)."""
        self._refresh_daily_cache(datetime.now(tz=_IST))
        return list(self._today_signals.get(index_id, []))

    def _get_signals_today_count(self, index_id: str, timestamp: datetime) -> int:
        """Taken-trade count for *index_id* on *timestamp*'s date (display use).

        Never raises; returns 0 if the count cannot be read.  The daily-cap
        safety check uses :meth:`_count_taken_trades_today`, which fails closed.
        """
        count = self._count_taken_trades_today(index_id, timestamp)
        return count if count is not None else 0

    def _count_taken_trades_today(self, index_id: str, timestamp: datetime) -> Optional[int]:
        """Taken-trade count, or ``None`` when the DB cannot be read.

        ``fetch_one`` returns a dict; the old ``result[0]`` raised KeyError on
        every call, was swallowed, and returned 0 — so the per-regime daily
        trade cap was never enforced.
        """
        try:
            ts = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
            result = self._db.fetch_one(Q.COUNT_SIGNALS_TODAY, (index_id, ts))
            return int(result["count"]) if result else 0
        except Exception:
            logger.exception("Failed to count today's trades for %s", index_id)
            return None

    def get_all_active_signals(self) -> list[TradingSignal]:
        """Return all open signals across all indices."""
        return [
            s
            for signals in self._today_signals.values()
            for s in signals
            if s.signal_type != "NO_TRADE" and (s.outcome is None or s.outcome == "OPEN")
        ]

    # ------------------------------------------------------------------
    # Step 1 — Safety Checks
    # ------------------------------------------------------------------

    def _run_safety_checks(
        self,
        index_id: str,
        current_spot_price: float,
        technical: Optional[TechnicalAnalysisResult],
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
        regime: MarketRegime,
        now: datetime,
        warnings: list[str],
    ) -> Optional[TradingSignal]:
        """Return a NO_TRADE signal on failure, or ``None`` if all checks pass."""

        # Data freshness: technical.timestamp is the newest *bar* time.  Reject
        # when that bar is older than allowed for its timeframe (a daily bar
        # is always stale for an intraday decision).
        if technical and technical.timestamp:
            ts = technical.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_IST)
            age_secs = (now - ts).total_seconds()
            max_age = _max_bar_age_seconds(getattr(technical, "timeframe", ""))
            if age_secs > max_age:
                return self._make_no_trade(
                    index_id, now,
                    f"Data stale: last {getattr(technical, 'timeframe', '?')} bar "
                    f"{age_secs:.0f}s old (> {max_age:.0f}s)",
                    regime, technical, news, anomaly, warnings,
                )

        # Invalid spot price
        if not current_spot_price or current_spot_price <= 0:
            return self._make_no_trade(
                index_id, now, "Invalid spot price (zero or negative)",
                regime, technical, news, anomaly, warnings,
            )

        # Regime safety
        is_safe, reason = RegimeDetector.is_safe_to_trade(regime)
        if not is_safe:
            return self._make_no_trade(
                index_id, now, f"Regime unsafe: {reason}",
                regime, technical, news, anomaly, warnings,
            )

        # Whitelist top performers: BANKNIFTY, NIFTY50, FINNIFTY, MIDCPNIFTY
        # Plus any dynamically allowed priority stocks from F&O Universe
        allowed_indices = {"BANKNIFTY", "NIFTY50", "FINNIFTY", "MIDCPNIFTY"}
        with self._lock:
            allowed_indices.update(self._dynamic_allowed_symbols)

        if index_id not in allowed_indices:
            return self._make_no_trade(
                index_id, now, f"Index {index_id} rejected — restricted universe only",
                regime, technical, news, anomaly, warnings,
            )

        # Daily signal cap (query DB instead of in-memory cache).  Fails
        # closed: if the count cannot be read, the cap cannot be verified.
        today_count = self._count_taken_trades_today(index_id, now)
        if today_count is None:
            return self._make_no_trade(
                index_id, now, "Cannot verify today's trade count (DB error) — no new entries",
                regime, technical, news, anomaly, warnings,
            )
        if today_count >= regime.max_trades_today:
            return self._make_no_trade(
                index_id, now,
                f"Max signals for today reached ({today_count}/{regime.max_trades_today})",
                regime, technical, news, anomaly, warnings,
            )

        # Extreme anomaly risk
        if anomaly and anomaly.risk_level == "EXTREME":
            return self._make_no_trade(
                index_id, now, "Extreme anomaly risk — no new positions",
                regime, technical, news, anomaly, warnings,
            )

        # Near market close (< 30 minutes until 15:30 IST)
        # RELAXED for Hero Zero candidates until 15:20 (10 mins before close)
        is_hero_zero = any("HERO_ZERO" in w for w in warnings)
        limit_mins = 10 if is_hero_zero else 30
        
        if self._is_near_market_close(now, minutes=limit_mins):
            msg = f"Too close to market close (< {limit_mins} min) — no new entries"
            return self._make_no_trade(
                index_id, now, msg,
                regime, technical, news, anomaly, warnings,
            )

        # Expiry day: no new long option positions after 2:30 PM (theta acceleration)
        if self._is_expiry_day(index_id, now, technical):
            if now.hour > 14 or (now.hour == 14 and now.minute >= 30):
                return self._make_no_trade(
                    index_id, now,
                    "Expiry day: no new entries after 14:30 — theta acceleration risk",
                    regime, technical, news, anomaly, warnings,
                )

        # GIFT NIFTY large gap: enforce no-trade for first 15 min
        if self._is_in_opening_window(now):
            try:
                gap_row = self._db.fetch_one(
                    "SELECT wait_15min, gap_classification, implied_gap_pts FROM gift_nifty ORDER BY timestamp DESC LIMIT 1",
                    ()
                )
                if gap_row and int(gap_row.get("wait_15min", 0)):
                    gap_pts = gap_row.get("implied_gap_pts", 0)
                    gap_cls = gap_row.get("gap_classification", "")
                    return self._make_no_trade(
                        index_id, now,
                        f"Large gap open ({gap_cls}, {gap_pts:+.0f} pts) — wait 15 min before first entry",
                        regime, technical, news, anomaly, warnings,
                    )
            except Exception:
                pass  # Non-fatal — proceed without the gap check

        return None

    @staticmethod
    def _is_near_market_close(now: datetime, minutes: int = 30) -> bool:
        """Return True if fewer than *minutes* remain until market close."""
        h, m = settings.market_hours.market_close.split(":")
        close = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
        return now >= close - timedelta(minutes=minutes)

    @staticmethod
    def _is_expiry_day(
        index_id: str,
        now: datetime,
        technical: Optional[TechnicalAnalysisResult] = None,
    ) -> bool:
        """Return True if *index_id*'s nearest option contract expires today.

        Expiry weekdays change by exchange circular and holidays shift
        individual expiries, so this never uses a weekday rule.  Sources, in
        order: the loaded options chain's expiry, then the contract master.
        Unknown → False (the chain-based theta rules simply do not apply).
        """
        opts = getattr(technical, "options", None) if technical is not None else None
        expiry = getattr(opts, "expiry_date", None) if opts is not None else None
        if expiry is not None:
            try:
                expiry_day = expiry.date() if isinstance(expiry, datetime) else expiry
                if not isinstance(expiry_day, date):
                    expiry_day = date.fromisoformat(str(expiry_day)[:10])
                return expiry_day == now.date()
            except (TypeError, ValueError):
                pass
        try:
            from src.data.contract_master import get_contract_master  # noqa: PLC0415
            from src.data.index_registry import get_registry  # noqa: PLC0415

            idx = get_registry().get_index(index_id)
            symbol = (idx.option_symbol if idx and idx.option_symbol else index_id)
            return bool(get_contract_master().is_expiry_day(symbol, now.date()))
        except Exception:
            logger.debug("Expiry-day lookup failed for %s", index_id, exc_info=True)
            return False

    @staticmethod
    def _is_in_opening_window(now: datetime) -> bool:
        """Return True if within the first 15 minutes after market open (9:15–9:30)."""
        return now.hour == 9 and 15 <= now.minute < 30

    # ------------------------------------------------------------------
    # Step 2 — Vote Collection
    # ------------------------------------------------------------------

    @staticmethod
    def _available_components(
        technical: TechnicalAnalysisResult,
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
    ) -> set[str]:
        failed = set(getattr(technical, "unavailable_components", None) or [])
        available = {"trend", "momentum"} - failed
        # Without real volume (NSE spot indices) the volume layer and the
        # smart-money composite carry no information: volume terms are zero
        # and OI history is not supplied, leaving only a candle-wick
        # oscillator.  Mark them unavailable so their weight is redistributed
        # instead of letting noise (or a forced NEUTRAL) vote.
        has_volume = getattr(technical, "volume_available", True) and "volume" not in failed
        if has_volume:
            available.add("volume")
        if technical.options is not None:
            available.add("options")
        if technical.smart_money is not None and has_volume:
            available.add("smart_money")
        if anomaly is not None:
            available.add("anomaly")
        return available

    @staticmethod
    def _collect_votes(
        technical: TechnicalAnalysisResult,
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
        weights: SignalWeights,
        available: Optional[set[str]] = None,
    ) -> dict[str, tuple[float, float, float]]:
        """Return ``{component: (numeric_score, confidence, weight)}``.

        Components outside *available* are omitted entirely, so they cannot
        count towards agreement or conflict checks either.
        """
        v: dict[str, tuple[float, float, float]] = {}

        def num(s: str) -> float:
            return _VOTE_NUMERIC.get((s or "NEUTRAL").upper(), 0.0)

        def ok(component: str) -> bool:
            return available is None or component in available

        if ok("trend"):
            v["trend"] = (
                num(technical.trend.trend_vote),
                technical.trend.trend_confidence,
                weights.trend_weight,
            )
        if ok("momentum"):
            v["momentum"] = (
                num(technical.momentum.momentum_vote),
                technical.momentum.momentum_confidence,
                weights.momentum_weight,
            )
        if ok("volume"):
            v["volume"] = (
                num(technical.volume.volume_vote),
                technical.volume.volume_confidence,
                weights.volume_weight,
            )
        if technical.options is not None and ok("options"):
            v["options"] = (
                num(technical.options.options_vote),
                technical.options.options_confidence,
                weights.options_weight,
            )
        if technical.smart_money is not None and ok("smart_money"):
            v["smart_money"] = (
                num(technical.smart_money.smart_money_bias),
                technical.smart_money.confidence,
                weights.smart_money_weight,
            )
        if anomaly is not None and ok("anomaly"):
            v["anomaly"] = (
                num(anomaly.vote),
                anomaly.confidence,
                weights.anomaly_weight,
            )
        return v

    # ------------------------------------------------------------------
    # Step 3 — Weighted Score
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_weighted_score(
        votes: dict[str, tuple[float, float, float]],
        regime: MarketRegime,
        tf_confluence_score: float = 0.0,
        tf_agreement: str = "UNKNOWN",
    ) -> float:
        """Compute the regime-adjusted weighted score.

        Formula: Σ(score × confidence × weight) / Σ(weight)

        Since redistributed weights already sum to 1.0, the denominator is
        effectively 1.0; the division guards against floating-point drift.
        """
        if not votes:
            return 0.0

        total_weight = sum(w for _, _, w in votes.values())
        if total_weight <= 0:
            return 0.0

        raw = sum(s * c * w for s, c, w in votes.values()) / total_weight

        regime_name = regime.regime

        # BREAKOUT: boost when score direction matches the trend direction
        if regime_name == "BREAKOUT":
            trend_score = votes.get("trend", (0.0, 0.0, 0.0))[0]
            if raw > 0 and trend_score > 0:
                raw += 0.3
            elif raw < 0 and trend_score < 0:
                raw -= 0.3

        # RANGE_BOUND: dampen extreme signals (less reliable in ranges)
        if regime_name == "RANGE_BOUND" and abs(raw) > 1.5:
            raw = raw - 0.2 * (1.0 if raw > 0 else -1.0)

        # Multi-timeframe confluence adjustment
        # MIXED TFs dampen (conflicting signal), ALIGNED boost
        if tf_agreement == "ALIGNED_BULL" and raw > 0:
            raw = raw * 1.12  # 12% boost when higher TF confirms bullish
        elif tf_agreement == "ALIGNED_BEAR" and raw < 0:
            raw = raw * 1.12  # 12% boost when higher TF confirms bearish
        elif tf_agreement == "MIXED" and abs(raw) > 0:
            raw = raw * 0.75  # 25% dampen for conflicting timeframes

        return raw

    # ------------------------------------------------------------------
    # Step 4 — Direction
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_direction(score: float, regime: str = "RANGE_BOUND") -> str:
        # Same thresholds in every regime.  (VOLATILE_CHOPPY used to LOWER
        # them to 0.15 — more trades exactly where option buyers bleed; the
        # regime is now blocked by RegimeDetector.is_safe_to_trade.)
        call_thresh = _SCORE_CALL_THRESHOLD
        put_thresh = _SCORE_PUT_THRESHOLD

        # Symmetric direction logic — both BUY_CALL and BUY_PUT apply in all regimes
        # Exception: CRASH regime — only BUY_PUT allowed (never buy calls in a crash)
        if regime == "CRASH":
            if score <= put_thresh:
                return "BUY_PUT"
            return "NO_TRADE"

        if score >= call_thresh:
            return "BUY_CALL"
        if score <= put_thresh:
            return "BUY_PUT"
        return "NO_TRADE"

    # ------------------------------------------------------------------
    # Step 5 — Confidence
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Model score, vetoes, calibrated confidence
    # ------------------------------------------------------------------

    @classmethod
    def model_score(
        cls,
        technical: TechnicalAnalysisResult,
        regime: MarketRegime,
    ) -> tuple[float, dict[str, tuple[float, float, float]], set[str]]:
        """The calibrated scoring function: ``(score, votes, components)``.

        ``scripts/calibrate_signal_model.py`` calls exactly this method, so
        the live score and the calibrated score are the same quantity.
        """
        available = cls._available_components(technical, None, None) & _SCORED_COMPONENTS
        weights = cls._redistribute_weights(regime.weight_adjustments, available)
        votes = cls._collect_votes(technical, None, None, weights, available)
        tf_confluence = getattr(technical, "timeframe_confluence_score", 0.0) or 0.0
        tf_agreement = getattr(technical, "tf_agreement", "UNKNOWN") or "UNKNOWN"
        score = cls._calculate_weighted_score(votes, regime, tf_confluence, tf_agreement)

        # Near-month futures buildup + order flow: the only volume/OI source an
        # index has.  Blended in only when available, so history without
        # futures data is scored exactly as before.
        flow = getattr(technical, "futures_flow", None)
        if flow is not None and getattr(flow, "available", False):
            from src.analysis.futures_flow import FLOW_WEIGHT, blend_score  # noqa: PLC0415

            score = blend_score(score, flow, _VOTE_NUMERIC)
            votes = dict(votes)
            votes["futures_flow"] = (
                _VOTE_NUMERIC.get(flow.vote, 0.0), flow.confidence, FLOW_WEIGHT,
            )
            available = set(available) | {"futures_flow"}
        return score, votes, available

    @staticmethod
    def _veto_reason(
        direction: str,
        technical: TechnicalAnalysisResult,
        anomaly: Optional[AnomalyVote],
    ) -> Optional[str]:
        """Unmeasured components can stop a trade, never start one."""
        is_call = direction == "BUY_CALL"

        def against(vote: str) -> bool:
            v = _VOTE_NUMERIC.get((vote or "NEUTRAL").upper(), 0.0)
            return v <= -1.0 if is_call else v >= 1.0

        opts = technical.options
        if opts is not None and against(opts.options_vote) and (opts.options_confidence or 0) >= 0.6:
            return (
                f"Options positioning opposes {direction} "
                f"({opts.options_vote}, conf {opts.options_confidence:.2f}) — vetoed"
            )
        if anomaly is not None and anomaly.risk_level in ("HIGH", "EXTREME") and against(anomaly.vote):
            return f"Anomaly engine ({anomaly.vote}, risk {anomaly.risk_level}) opposes {direction} — vetoed"
        return None

    def _calibrated_confidence(
        self,
        index_id: str,
        direction: str,
        weighted_score: float,
        technical: TechnicalAnalysisResult,
        anomaly: Optional[AnomalyVote],
        regime: MarketRegime,
        warnings: list[str],
    ) -> tuple[Optional[float], str, str]:
        """Return ``(P(win) | None, level, reason)``.

        P(win) comes from the out-of-sample-validated calibration, then only
        PENALTIES are applied for conditions the calibration did not see.
        """
        p, reason = self._calibration.lookup(index_id, abs(weighted_score))
        if p is None:
            return None, "LOW", reason

        haircut = 0.0
        notes: list[str] = []
        if anomaly is not None:
            if anomaly.risk_level == "HIGH":
                haircut += _HAIRCUT_ANOMALY_HIGH
                notes.append("anomaly risk HIGH")
            if anomaly.vote == "CAUTION":
                haircut += _HAIRCUT_ANOMALY_CAUTION
                notes.append("anomaly CAUTION")
        if regime.regime_changing:
            haircut += _HAIRCUT_REGIME_CHANGING
            notes.append("regime transitioning")
        if regime.volatility_regime == "EXTREME":
            haircut += _HAIRCUT_EXTREME_VOL
            notes.append("extreme volatility")
        opts = technical.options
        if opts is not None and (getattr(opts, "iv_strategy_bias", "") or "") == "SELL_PREMIUM":
            haircut += _HAIRCUT_IV_EXPENSIVE
            notes.append("IV expensive (IVP > 75)")
            warnings.append("IV expensive (IVP >75%) — option buyers face IV-crush risk")
        if direction == "BUY_CALL" and any(
            "BANKNIFTY_LAGGING" in w for w in (getattr(regime, "warnings", []) or [])
        ):
            haircut += _HAIRCUT_BANKNIFTY_LAGGING
            notes.append("BankNIFTY lagging")

        prob = max(_CONF_FLOOR, min(_CONF_CEILING, p - haircut))
        if notes:
            warnings.append(
                f"Calibrated P(win) {p:.2f} − {haircut:.2f} ({', '.join(notes)}) = {prob:.2f}"
            )
        if prob >= _CONF_HIGH:
            level = "HIGH"
        elif prob >= self._min_conf_medium:
            level = "MEDIUM"
        else:
            level = "LOW"
        return prob, level, "ok"

    # ------------------------------------------------------------------
    # Step 6 — Levels
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_levels(
        direction: str,
        entry_price: float,
        technical: TechnicalAnalysisResult,
        regime: MarketRegime,
        confidence_level: str,
        stop_atr: float = 1.0,
        target_atr: float = 1.5,
        min_stop_pct: float = 0.001,
    ) -> tuple[float, float, float, float]:
        """Return ``(entry, target, stop_loss, risk_reward_ratio)``.

        The stop and target follow the SAME rule the calibration labelled
        a win/loss with (``trade_distances``: stop = max(stop_atr × ATR,
        min_stop_pct × spot), target = stop × target_atr / stop_atr), so the
        calibrated P(win) describes this exact trade and the RiskManager's
        noise floor never has to widen it.  The old levels mixed a support-based stop with an
        ATR × regime-multiplier target; on 5m data that gave R:R ≈ 0.3–0.5
        and ~70 % of otherwise-valid signals failed the R:R gate — and the
        probability, had it existed, would have described a different trade.
        OI walls still cap the target, in the RiskManager, on a real chain.
        """
        import math
        atr = technical.volatility.atr_value
        if not atr or not math.isfinite(atr) or atr <= 0:
            atr = entry_price * 0.002  # fallback: ~a 5m ATR on an index
        from src.engine.signal_calibration import trade_distances  # noqa: PLC0415

        stop_dist, tgt_dist = trade_distances(atr, entry_price, stop_atr, target_atr, min_stop_pct)
        if direction == "BUY_CALL":
            stop_loss, target = entry_price - stop_dist, entry_price + tgt_dist
        else:
            stop_loss, target = entry_price + stop_dist, entry_price - tgt_dist
        stop_loss = max(stop_loss, 0.01)
        target = max(target, 0.01)
        rr_ratio = round(tgt_dist / stop_dist, 2) if stop_dist > 0 else 0.0
        return entry_price, round(target, 2), round(stop_loss, 2), rr_ratio

    # ------------------------------------------------------------------
    # Special overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _smart_money_warnings(
        direction: str,
        technical: TechnicalAnalysisResult,
    ) -> list[str]:
        """Return warnings when smart money direction conflicts with signal."""
        if technical.smart_money is None:
            return []
        sm = technical.smart_money
        sm_bull = sm.smart_money_bias in ("BULLISH", "STRONGLY_BULLISH")
        sm_bear = sm.smart_money_bias in ("BEARISH", "STRONGLY_BEARISH")
        if direction == "BUY_CALL" and sm_bear:
            return [
                f"Smart money ({sm.smart_money_bias}) disagrees with CALL — "
                "institutional flow is bearish"
            ]
        if direction == "BUY_PUT" and sm_bull:
            return [
                f"Smart money ({sm.smart_money_bias}) disagrees with PUT — "
                "institutional flow is bullish"
            ]
        return []

    @staticmethod
    def _expiry_warnings(
        direction: str,
        technical: TechnicalAnalysisResult,
        regime: MarketRegime,
        now: datetime,
    ) -> list[str]:
        """Warn about max-pain gravitational pull on expiry days after 14:00."""
        if regime.event_regime != "EVENT_DAY":
            return []
        if technical.options is None:
            return []
        if now.hour < 14:
            return []
        opts = technical.options
        max_pain = opts.max_pain
        if max_pain <= 0:
            return []
        return [
            f"Expiry day after 14:00 — max pain at {max_pain:,.0f} "
            f"({opts.days_to_expiry} days to expiry). "
            "Max-pain gravity increases near close."
        ]

    @staticmethod
    def _get_orb_context(
        current_price: float,
        technical: TechnicalAnalysisResult,
    ) -> Optional[str]:
        """Return ORB breakout context if available in technical result."""
        # Check if technical result carries ORB data (populated by data layer)
        orb_high = getattr(technical, 'orb_high', None)
        orb_low = getattr(technical, 'orb_low', None)
        if not orb_high or not orb_low:
            return None
        if current_price > orb_high:
            return f"Price {current_price:.0f} above ORB high {orb_high:.0f} — breakout confirmed"
        if current_price < orb_low:
            return f"Price {current_price:.0f} below ORB low {orb_low:.0f} — breakdown confirmed"
        return f"Price {current_price:.0f} inside ORB ({orb_low:.0f}–{orb_high:.0f}) — no breakout yet"

    @staticmethod
    def _has_conflicting_strong_signals(
        votes: dict[str, tuple[float, float, float]],
    ) -> bool:
        """Return True when 2+ strong bullish AND 2+ strong bearish votes exist."""
        strong_bull = sum(1 for s, _, _ in votes.values() if s >= 2.0)
        strong_bear = sum(1 for s, _, _ in votes.values() if s <= -2.0)
        return strong_bull >= 2 and strong_bear >= 2

    def _check_cooldown(
        self,
        index_id: str,
        direction: str,
        now: datetime,
        warnings: list[str],
    ) -> None:
        """Add a cooldown warning if a matching signal was generated < 15 min ago."""
        recent = self._today_signals.get(index_id, [])
        for sig in reversed(recent):
            elapsed = (now - sig.generated_at.replace(tzinfo=_IST)).total_seconds()
            if elapsed < 15 * 60 and sig.signal_type == direction:
                warnings.append(
                    f"Cooldown: same direction ({direction}) signal was generated "
                    f"{int(elapsed / 60)} min ago — trade with caution"
                )
                break

    # ------------------------------------------------------------------
    # Weight redistribution
    # ------------------------------------------------------------------

    @staticmethod
    def _redistribute_weights(
        weights: SignalWeights,
        available_components: set[str],
    ) -> SignalWeights:
        """Redistribute missing-component weights proportionally to available ones.

        Sets missing component weights to 0 then normalises so all weights
        still sum to 1.0.

        Example
        -------
        Options missing (weight=0.25), remaining 0.75 → each remaining
        component scaled by 1/0.75 = 1.333×.
        """
        missing = _ALL_COMPONENTS - available_components
        if not missing:
            return weights  # nothing to do

        new = SignalWeights(
            trend_weight=weights.trend_weight if "trend" in available_components else 0.0,
            momentum_weight=weights.momentum_weight if "momentum" in available_components else 0.0,
            options_weight=weights.options_weight if "options" in available_components else 0.0,
            volume_weight=weights.volume_weight if "volume" in available_components else 0.0,
            smart_money_weight=(
                weights.smart_money_weight if "smart_money" in available_components else 0.0
            ),
            anomaly_weight=weights.anomaly_weight if "anomaly" in available_components else 0.0,
        )
        new.normalise()
        return new

    # ------------------------------------------------------------------
    # Static conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_vote_to_numeric(vote_string: str) -> float:
        """Map a vote string to [-2, +2].  Unknown strings return 0.0."""
        return _VOTE_NUMERIC.get((vote_string or "NEUTRAL").upper(), 0.0)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _position_sizing(
        regime: MarketRegime,
        anomaly: Optional[AnomalyVote],
        technical: TechnicalAnalysisResult,
        entry_price: float,
        target_price: float,
        stop_loss: float,
    ) -> tuple[float, int, float, float]:
        """Return ``(combined_modifier, lot_count, max_loss_pts, max_profit_pts)``."""
        regime_mod = regime.position_size_multiplier
        anomaly_mod = anomaly.position_size_modifier if anomaly else 1.0
        vol_mod = technical.volatility.position_size_modifier

        combined = regime_mod * anomaly_mod * vol_mod
        combined = max(0.1, min(1.5, combined))

        lot_count = max(1, round(combined))
        sl_dist = abs(entry_price - stop_loss)
        tgt_dist = abs(target_price - entry_price)

        return combined, lot_count, sl_dist * lot_count, tgt_dist * lot_count

    # ------------------------------------------------------------------
    # Step 7 — Reasoning
    # ------------------------------------------------------------------

    def _build_reasoning(
        self,
        index_id: str,
        direction: str,
        confidence_level: str,
        confidence_score: float,
        weighted_score: float,
        regime: MarketRegime,
        votes: dict[str, tuple[float, float, float]],
        entry_price: float,
        target_price: float,
        stop_loss: float,
        rr_ratio: float,
        technical: TechnicalAnalysisResult,
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
        warnings: list[str],
    ) -> str:
        lines: list[str] = []

        # Header
        dir_label = {"BUY_CALL": "BUY CALL", "BUY_PUT": "BUY PUT"}.get(direction, direction)
        lines.append(
            f"{dir_label} — {index_id} (Confidence: {confidence_level} — {confidence_score:.2f})"
        )
        lines.append("")

        # Regime
        lines.append(f"Regime: {regime.description}")
        lines.append("")

        # Signal breakdown
        lines.append("Signal Breakdown:")
        component_order = ["trend", "momentum", "options", "volume", "smart_money", "news", "anomaly"]
        labels = {
            "trend": "Trend",
            "momentum": "Momentum",
            "options": "Options",
            "volume": "Volume",
            "smart_money": "Smart Money",
            "news": "News",
            "anomaly": "Anomaly",
        }
        for comp in component_order:
            if comp in votes:
                score, conf, _ = votes[comp]
                vote_label = self._score_to_label(score)
                detail = self._vote_detail(comp, technical, news, anomaly)
                lines.append(f"  ● {labels[comp]}: {vote_label} ({conf:.2f}) — {detail}")
            else:
                lines.append(f"  ● {labels[comp]}: N/A (data unavailable)")

        lines.append("")
        lines.append(f"Weighted Score: {weighted_score:+.2f}")
        lines.append("")

        # Key levels
        sl_dist = abs(entry_price - stop_loss)
        tgt_dist = abs(target_price - entry_price)
        sign = "+" if direction == "BUY_CALL" else "-"
        lines.append("Key Levels:")
        lines.append(f"  ● Entry: {entry_price:,.0f} (market)")
        lines.append(f"  ● Target: {target_price:,.0f} ({sign}{tgt_dist:.0f} pts)")
        lines.append(f"  ● Stop Loss: {stop_loss:,.0f} ({sl_dist:.0f} pts)")
        lines.append(f"  ● Risk:Reward = 1:{rr_ratio:.2f}")

        # Caution notes
        if warnings:
            lines.append("")
            lines.append("Caution:")
            for w in warnings:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 1.5:
            return "STRONG BULLISH"
        if score >= 0.75:
            return "BULLISH"
        if score >= 0.25:
            return "SLIGHTLY BULLISH"
        if score <= -1.5:
            return "STRONG BEARISH"
        if score <= -0.75:
            return "BEARISH"
        if score <= -0.25:
            return "SLIGHTLY BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _vote_detail(
        comp: str,
        technical: TechnicalAnalysisResult,
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
    ) -> str:
        if comp == "trend":
            t = technical.trend
            parts: list[str] = []
            if t.ema_alignment:
                parts.append(f"EMA {t.ema_alignment}")
            if t.macd_signal and t.macd_signal != "NEUTRAL":
                parts.append(f"MACD {t.macd_signal}")
            if t.golden_cross:
                parts.append("golden cross")
            elif t.death_cross:
                parts.append("death cross")
            return ", ".join(parts) or t.trend_vote

        if comp == "momentum":
            m = technical.momentum
            parts = [f"RSI {m.rsi_value:.0f}"]
            if m.rsi_divergence:
                parts.append(m.rsi_divergence)
            if m.overbought_consensus:
                parts.append("overbought")
            elif m.oversold_consensus:
                parts.append("oversold")
            return ", ".join(parts)

        if comp == "options" and technical.options is not None:
            o = technical.options
            return (
                f"PCR {o.pcr:.2f}, max pain {o.max_pain:,.0f}, {o.dominant_buildup}"
            )

        if comp == "volume":
            v = technical.volume
            return f"VWAP {v.price_vs_vwap}, OBV {v.obv_trend}"

        if comp == "smart_money" and technical.smart_money is not None:
            sm = technical.smart_money
            return f"score {sm.score:.0f}, grade {sm.grade}, {sm.key_finding}"

        if comp == "news" and news is not None:
            return (
                f"{news.active_article_count} articles, "
                f"sentiment {news.weighted_sentiment:+.2f}, "
                f"regime {news.event_regime}"
            )

        if comp == "anomaly" and anomaly is not None:
            msg = anomaly.primary_alert_message or "No alert"
            return f"risk={anomaly.risk_level}, {msg}"

        return "No detail available"

    # ------------------------------------------------------------------
    # Vote breakdown (transparency dict)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_vote_breakdown(
        votes: dict[str, tuple[float, float, float]],
    ) -> dict:
        return {
            comp: {
                "score": round(score, 2),
                "confidence": round(conf, 4),
                "weight": round(weight, 4),
                "weighted_contribution": round(score * conf * weight, 4),
            }
            for comp, (score, conf, weight) in votes.items()
        }

    # ------------------------------------------------------------------
    # Audit trail (persistence handled by SignalTracker only)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_audit_trail(
        index_id: str,
        spot_price: float,
        base_confidence: float,
        technical: TechnicalAnalysisResult,
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
        regime: MarketRegime,
        signal: TradingSignal,
        votes: dict[str, tuple[float, float, float]],
    ) -> dict:
        """Build a structured audit dict capturing every intermediate value."""
        return {
            "generated_at": signal.generated_at.isoformat(),
            "index_id": index_id,
            "spot_price": spot_price,

            "technical": {
                "overall_signal": technical.overall_signal,
                "overall_confidence": technical.overall_confidence,
                "trend_vote": technical.trend.trend_vote,
                "ema_20": float(technical.trend.price_vs_ema20 == "ABOVE") if hasattr(technical.trend, "price_vs_ema20") else None,
                "price_vs_ema20": technical.trend.price_vs_ema20 if hasattr(technical.trend, "price_vs_ema20") else None,
                "macd_crossover": technical.trend.macd_crossover if hasattr(technical.trend, "macd_crossover") else None,
                "adx_value": technical.trend.trend_strength if hasattr(technical.trend, "trend_strength") else None,
                "rsi_value": technical.momentum.rsi_value,
                "rsi_zone": technical.momentum.rsi_zone,
                "rsi_divergence": technical.momentum.rsi_divergence,
                "bb_position": technical.volatility.bb_position,
                "bb_squeeze": technical.volatility.bb_squeeze,
                "atr_value": technical.volatility.atr_value,
                "vwap_position": technical.volume.price_vs_vwap,
                "obv_divergence": technical.volume.obv_divergence,
            },

            "options": {
                "pcr": technical.options.pcr,
                "max_pain": technical.options.max_pain,
                "oi_support": technical.options.oi_support,
                "oi_resistance": technical.options.oi_resistance,
                "atm_iv": technical.options.atm_iv,
                "iv_regime": technical.options.iv_regime,
                "options_vote": technical.options.options_vote,
            } if technical.options else None,

            "smart_money": {
                "score": technical.smart_money.score,
                "grade": technical.smart_money.grade,
                "smfi_signal": technical.smart_money.smfi_component,
            } if technical.smart_money else None,

            "news": {
                "vote": news.vote,
                "confidence": news.confidence,
                "article_count": news.active_article_count,
                "event_regime": news.event_regime,
            } if news else None,

            "anomaly": {
                "vote": anomaly.vote,
                "risk_level": anomaly.risk_level,
                "active_alerts": anomaly.active_alerts,
            } if anomaly else None,

            "regime": {
                "regime": regime.regime,
                "trend_regime": regime.trend_regime,
                "volatility_regime": regime.volatility_regime,
                "market_phase": regime.market_phase,
            },

            "computation": {
                "weighted_score": signal.weighted_score,
                "vote_breakdown": signal.vote_breakdown,
                "base_confidence": round(base_confidence, 4),
                "final_confidence": signal.confidence_score,
            },

            "levels": {
                "support_levels": technical.support_levels[:3] if technical.support_levels else [],
                "resistance_levels": technical.resistance_levels[:3] if technical.resistance_levels else [],
                "immediate_support": technical.immediate_support,
                "immediate_resistance": technical.immediate_resistance,
                "atr_based_sl": technical.suggested_stop_loss_distance,
            },

            "data_completeness": signal.data_completeness,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_no_trade(
        self,
        index_id: str,
        now: datetime,
        reason: str,
        regime: MarketRegime,
        technical: Optional[TechnicalAnalysisResult],
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
        warnings: list[str],
        weighted_score: float = 0.0,
    ) -> TradingSignal:
        risk_level = anomaly.risk_level if anomaly else "NORMAL"
        today_count = self._get_signals_today_count(index_id, now)
        return TradingSignal(
            signal_id=str(uuid.uuid4()),
            index_id=index_id,
            generated_at=now,
            signal_type="NO_TRADE",
            confidence_level="LOW",
            confidence_score=0.0,
            entry_price=0.0,
            target_price=0.0,
            stop_loss=0.0,
            risk_reward_ratio=0.0,
            regime=regime.regime,
            weighted_score=weighted_score,
            vote_breakdown={},
            risk_level=risk_level,
            position_size_modifier=0.0,
            suggested_lot_count=0,
            estimated_max_loss=0.0,
            estimated_max_profit=0.0,
            reasoning=f"NO_TRADE: {reason}",
            warnings=list(warnings) + [reason],
            data_completeness=self._data_completeness(technical, news, anomaly),
            signals_generated_today=today_count,
        )

    def _build_hero_zero_signal(
        self,
        index_id: str,
        now: datetime,
        spot: float,
        hz_result: object,
        regime: MarketRegime,
        technical: Optional[TechnicalAnalysisResult],
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
    ) -> TradingSignal:
        """Build a Hero Zero signal from HeroZeroResult.

        Bypasses the normal vote/score pipeline.
        Hard-caps position size at 25% regardless of regime/volatility modifiers.
        """
        rr = 1.5
        direction = getattr(hz_result, "signal_direction", "NO_TRADE")
        # Default stop on the correct side (it used to be spot − 50 for puts
        # too, i.e. a put "stop" below entry).  The executable stop for a
        # hero-zero trade is set on the premium by the RiskManager.
        default_sl = spot - 50 if direction == "BUY_CALL" else spot + 50
        sl = getattr(hz_result, "stop_loss", None) or default_sl

        if direction == "BUY_CALL":
            target = spot + abs(spot - sl) * rr
        else:  # BUY_PUT
            target = spot - abs(sl - spot) * rr

        sl_dist = abs(spot - sl)
        tgt_dist = abs(target - spot)
        rr_ratio = round(tgt_dist / sl_dist, 2) if sl_dist > 0 else rr
        today_count = self._get_signals_today_count(index_id, now)

        signal = TradingSignal(
            signal_id=str(uuid.uuid4()),
            index_id=index_id,
            generated_at=now,
            signal_type=direction,
            confidence_level="MEDIUM",
            confidence_score=0.55,
            entry_price=round(spot, 2),
            target_price=round(target, 2),
            stop_loss=round(sl, 2),
            risk_reward_ratio=rr_ratio,
            regime=regime.regime,
            weighted_score=0.0,
            vote_breakdown={},
            risk_level="ELEVATED",
            position_size_modifier=0.25,  # Hard cap
            suggested_lot_count=1,
            estimated_max_loss=round(sl_dist, 2),
            estimated_max_profit=round(tgt_dist, 2),
            reasoning=getattr(hz_result, "reasoning", "Hero Zero triggered"),
            warnings=["HERO_ZERO: gamma squeeze detected"],
            data_completeness=self._data_completeness(technical, news, anomaly),
            signals_generated_today=today_count + 1,
            is_hero_zero=True,
            hero_zero_result=hz_result,
        )
        self._today_signals.setdefault(index_id, []).append(signal)
        return signal

    def _refresh_daily_cache(self, now: datetime) -> None:
        """Reset the in-memory cache when the IST calendar day rolls over."""
        today = now.strftime("%Y-%m-%d")
        if self._today_date != today:
            self._today_signals = {}
            self._today_date = today

    @staticmethod
    def _data_completeness(
        technical: Optional[TechnicalAnalysisResult],
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
    ) -> float:
        if technical is None:
            return 0.0
        score = technical.data_completeness * 0.5
        if news is not None:
            score += 0.25
        if anomaly is not None:
            score += 0.25
        return min(1.0, score)
