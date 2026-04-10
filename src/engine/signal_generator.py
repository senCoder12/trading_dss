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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from config.settings import settings
from src.analysis.anomaly.anomaly_aggregator import AnomalyVote
from src.analysis.news.news_engine import NewsVote
from src.analysis.technical_aggregator import TechnicalAnalysisResult
from src.database.db_manager import DatabaseManager
from src.engine.regime_detector import MarketRegime, RegimeDetector, SignalWeights

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

    # Metadata
    data_completeness: float = 1.0      # 0–1, fraction of data sources available
    signals_generated_today: int = 0    # Running count for this index today


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum confidence thresholds
_CONF_HIGH: float = 0.65
_CONF_MEDIUM: float = 0.45
_CONF_LOW: float = 0.30
_CONF_FLOOR: float = 0.15    # absolute minimum (never return 0)
_CONF_CEILING: float = 0.92  # absolute maximum (never return 1)

# Weighted-score → direction thresholds
_SCORE_CALL_THRESHOLD: float = 0.40
_SCORE_PUT_THRESHOLD: float = -0.40

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
_WEIGHT_ATTR: dict[str, str] = {
    "trend": "trend_weight",
    "momentum": "momentum_weight",
    "options": "options_weight",
    "volume": "volume_weight",
    "news": "news_weight",
    "smart_money": "smart_money_weight",
    "anomaly": "anomaly_weight",
}

# All component names (used for redistribution logic)
_ALL_COMPONENTS: frozenset[str] = frozenset(_WEIGHT_ATTR.keys())


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

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

        # Minimum confidence thresholds (could be loaded from settings)
        self._min_conf_high = _CONF_HIGH
        self._min_conf_medium = _CONF_MEDIUM
        self._min_conf_low = _CONF_LOW

        # In-memory cache of signals generated today per index.
        # Keyed by index_id; reset at midnight.
        self._today_signals: dict[str, list[TradingSignal]] = {}
        self._today_date: Optional[str] = None  # "YYYY-MM-DD" IST

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

        # ── Step 2: Collect All Votes ──────────────────────────────────
        available = self._available_components(technical, news, anomaly)
        adj_weights = self._redistribute_weights(regime.weight_adjustments, available)
        votes = self._collect_votes(technical, news, anomaly, adj_weights)

        # ── Step 3: Weighted Score ─────────────────────────────────────
        weighted_score = self._calculate_weighted_score(votes, regime)

        # ── Special overrides applied to the score ─────────────────────
        weighted_score, div_warns = self._apply_divergence_override(
            weighted_score, technical,
        )
        warnings.extend(div_warns)

        gap_score, gap_warns = self._apply_gap_fill_override(weighted_score, anomaly)

        # ── Step 4: Direction ──────────────────────────────────────────
        direction = self._determine_direction(weighted_score)

        # Apply gap-fill only when the primary signal is indeterminate or weak
        if direction == "NO_TRADE" or abs(weighted_score) < 1.0:
            gap_direction = self._determine_direction(gap_score)
            if gap_direction != "NO_TRADE":
                weighted_score = gap_score
                direction = gap_direction
                warnings.extend(gap_warns)

        if direction == "NO_TRADE":
            return self._make_no_trade(
                index_id, now,
                "Score in dead zone (|score| < 0.40) — no clear edge",
                regime, technical, news, anomaly, warnings,
                weighted_score=weighted_score,
            )

        # ── Step 5: Confidence ─────────────────────────────────────────
        confidence_score, confidence_level = self._calculate_confidence(
            direction, weighted_score, votes, technical, news, anomaly, regime, warnings, mode,
        )

        if confidence_score < self._min_conf_low:
            return self._make_no_trade(
                index_id, now,
                f"Insufficient confidence ({confidence_score:.2f} < {self._min_conf_low:.2f})",
                regime, technical, news, anomaly, warnings,
                weighted_score=weighted_score,
            )

        # ── Smart money & expiry warnings (post-direction) ─────────────
        warnings.extend(self._smart_money_warnings(direction, technical))
        warnings.extend(self._expiry_warnings(direction, technical, regime, now))

        # ── Conflicting signals guard ──────────────────────────────────
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
        entry_price, target_price, stop_loss, rr_ratio = self._calculate_levels(
            direction, current_spot_price, technical, regime, confidence_level,
        )

        # ── Position sizing ────────────────────────────────────────────
        pos_mod, lot_count, max_loss, max_profit = self._position_sizing(
            regime, anomaly, technical, entry_price, target_price, stop_loss,
        )

        # ── Misc context ───────────────────────────────────────────────
        risk_level = anomaly.risk_level if anomaly else "NORMAL"
        data_completeness = self._data_completeness(technical, news, anomaly)
        vote_breakdown = self._build_vote_breakdown(votes)
        today_count = len(self._today_signals.get(index_id, []))

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

        # Persist & cache
        self._save_signal(signal, technical, news, anomaly)
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

        # Daily signal cap
        today_count = len(self._today_signals.get(index_id, []))
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
        if self._is_near_market_close(now, minutes=30):
            return self._make_no_trade(
                index_id, now, "Too close to market close (< 30 min) — no new entries",
                regime, technical, news, anomaly, warnings,
            )

        return None

    @staticmethod
    def _is_near_market_close(now: datetime, minutes: int = 30) -> bool:
        """Return True if fewer than *minutes* remain until market close."""
        h, m = settings.market_hours.market_close.split(":")
        close = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
        return now >= close - timedelta(minutes=minutes)

    # ------------------------------------------------------------------
    # Step 2 — Vote Collection
    # ------------------------------------------------------------------

    @staticmethod
    def _available_components(
        technical: TechnicalAnalysisResult,
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
    ) -> set[str]:
        available = {"trend", "momentum", "volume"}
        if technical.options is not None:
            available.add("options")
        if technical.smart_money is not None:
            available.add("smart_money")
        if news is not None:
            available.add("news")
        if anomaly is not None:
            available.add("anomaly")
        return available

    @staticmethod
    def _collect_votes(
        technical: TechnicalAnalysisResult,
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
        weights: SignalWeights,
    ) -> dict[str, tuple[float, float, float]]:
        """Return ``{component: (numeric_score, confidence, weight)}``."""
        v: dict[str, tuple[float, float, float]] = {}

        def num(s: str) -> float:
            return _VOTE_NUMERIC.get((s or "NEUTRAL").upper(), 0.0)

        v["trend"] = (
            num(technical.trend.trend_vote),
            technical.trend.trend_confidence,
            weights.trend_weight,
        )
        v["momentum"] = (
            num(technical.momentum.momentum_vote),
            technical.momentum.momentum_confidence,
            weights.momentum_weight,
        )
        v["volume"] = (
            num(technical.volume.volume_vote),
            technical.volume.volume_confidence,
            weights.volume_weight,
        )
        if technical.options is not None:
            v["options"] = (
                num(technical.options.options_vote),
                technical.options.options_confidence,
                weights.options_weight,
            )
        if technical.smart_money is not None:
            v["smart_money"] = (
                num(technical.smart_money.smart_money_bias),
                technical.smart_money.confidence,
                weights.smart_money_weight,
            )
        if news is not None:
            v["news"] = (
                num(news.vote),
                news.confidence,
                weights.news_weight,
            )
        if anomaly is not None:
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

        return raw

    # ------------------------------------------------------------------
    # Step 4 — Direction
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_direction(score: float) -> str:
        if score > _SCORE_CALL_THRESHOLD:
            return "BUY_CALL"
        if score < _SCORE_PUT_THRESHOLD:
            return "BUY_PUT"
        return "NO_TRADE"

    # ------------------------------------------------------------------
    # Step 5 — Confidence
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self,
        direction: str,
        weighted_score: float,
        votes: dict[str, tuple[float, float, float]],
        technical: TechnicalAnalysisResult,
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
        regime: MarketRegime,
        warnings: list[str],
        mode: str = "FULL",
    ) -> tuple[float, str]:
        """Return ``(confidence_score, confidence_level)``."""
        # In FULL mode, weighted scores routinely exceed 1.0 (many components in
        # agreement), so we halve to keep base ≤ 1.0.  In TECHNICAL_ONLY mode,
        # only 3 components contribute, so scores rarely exceed 0.8; dividing by
        # 2 makes the confidence systematically too low.  Use the raw score
        # directly so the confidence reflects the available evidence.
        if mode == "TECHNICAL_ONLY":
            base = min(abs(weighted_score), 1.0)
        else:
            base = min(abs(weighted_score) / 2.0, 1.0)
        adj = 0.0

        is_bullish = direction == "BUY_CALL"

        # ── Boosters ──────────────────────────────────────────────────

        # 5+ categories agree on direction
        agreeing = sum(
            1
            for s, _, _ in votes.values()
            if (is_bullish and s > 0) or (not is_bullish and s < 0)
        )
        if agreeing >= 5:
            adj += 0.10

        # Institutional activity confirms direction
        if anomaly and anomaly.institutional_activity:
            anomaly_score = votes.get("anomaly", (0.0, 0.0, 0.0))[0]
            aligned = (is_bullish and anomaly_score >= 0) or (
                not is_bullish and anomaly_score <= 0
            )
            if aligned:
                adj += 0.05

        # Strong trend regime matches direction
        if regime.regime == "STRONG_TREND_UP" and is_bullish:
            adj += 0.05
        elif regime.regime == "STRONG_TREND_DOWN" and not is_bullish:
            adj += 0.05

        # News severity HIGH/EXTREME matching direction
        if news and news.event_regime in ("HIGH", "EXTREME"):
            news_score = votes.get("news", (0.0, 0.0, 0.0))[0]
            if (is_bullish and news_score > 0) or (not is_bullish and news_score < 0):
                adj += 0.05

        # Smart money A+ grade alignment
        if technical.smart_money is not None:
            sm = technical.smart_money
            if sm.grade == "A+":
                sm_bullish = sm.smart_money_bias in ("BULLISH", "STRONGLY_BULLISH")
                sm_bearish = sm.smart_money_bias in ("BEARISH", "STRONGLY_BEARISH")
                if (is_bullish and sm_bullish) or (not is_bullish and sm_bearish):
                    adj += 0.10

        # ── Reducers ──────────────────────────────────────────────────

        # Anomaly risk level
        if anomaly:
            if anomaly.risk_level == "HIGH":
                adj -= 0.10
            elif anomaly.risk_level == "EXTREME":
                adj -= 0.15
            if anomaly.vote == "CAUTION":
                adj -= 0.10

        # Regime is transitioning
        if regime.regime_changing:
            adj -= 0.05

        # Low regime confidence
        if regime.regime_confidence < 0.5:
            adj -= 0.05

        # Extreme volatility (VIX or ATR)
        if regime.volatility_regime == "EXTREME":
            adj -= 0.10
            warnings.append("Extreme volatility — confidence reduced")

        # Thin news base (1-2 articles only)
        if news and news.active_article_count <= 2:
            adj -= 0.05

        # Penalty per missing data source — only count sources expected in this mode.
        # TECHNICAL_ONLY: options and news are intentionally absent, no penalty.
        # TECHNICAL_OPTIONS: news is intentionally absent; penalise missing options.
        # FULL: all three sources are expected.
        if mode == "TECHNICAL_ONLY":
            missing = sum([technical.smart_money is None])
        elif mode == "TECHNICAL_OPTIONS":
            missing = sum([
                technical.options is None,
                technical.smart_money is None,
            ])
        else:
            missing = sum([
                technical.options is None,
                news is None,
                technical.smart_money is None,
            ])
        adj -= 0.05 * missing

        score = max(_CONF_FLOOR, min(_CONF_CEILING, base + adj))

        if score >= _CONF_HIGH:
            level = "HIGH"
        elif score >= _CONF_MEDIUM:
            level = "MEDIUM"
        else:
            level = "LOW"

        return score, level

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
    ) -> tuple[float, float, float, float]:
        """Return ``(entry, target, stop_loss, risk_reward_ratio)``."""
        import math
        atr = technical.volatility.atr_value
        if not atr or not math.isfinite(atr) or atr <= 0:
            atr = entry_price * 0.005  # fallback: 0.5% of price

        sl_mult = regime.stop_loss_multiplier
        if not sl_mult or not math.isfinite(sl_mult) or sl_mult <= 0:
            sl_mult = 1.5  # fallback
        rr = 1.5 if confidence_level == "HIGH" else 2.0

        support = technical.immediate_support
        resistance = technical.immediate_resistance
        if not support or not math.isfinite(support) or support <= 0:
            support = entry_price * 0.98   # fallback: 2% below entry
        if not resistance or not math.isfinite(resistance) or resistance <= 0:
            resistance = entry_price * 1.02  # fallback: 2% above entry

        if direction == "BUY_CALL":
            stop_loss = support - atr * sl_mult * 0.5
            target = entry_price + atr * sl_mult * rr
        else:  # BUY_PUT
            stop_loss = resistance + atr * sl_mult * 0.5
            target = entry_price - atr * sl_mult * rr

        stop_loss = max(stop_loss, 0.01)
        target = max(target, 0.01)

        sl_dist = abs(entry_price - stop_loss)
        tgt_dist = abs(entry_price - target)
        rr_ratio = round(tgt_dist / sl_dist, 2) if sl_dist > 0 else rr

        return entry_price, round(target, 2), round(stop_loss, 2), rr_ratio

    # ------------------------------------------------------------------
    # Special overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_divergence_override(
        score: float,
        technical: TechnicalAnalysisResult,
    ) -> tuple[float, list[str]]:
        """Boost score toward CALL/PUT when RSI or OBV divergence is detected.

        Divergences are leading indicators — they can flip a neutral/mild
        signal toward a higher-conviction direction.
        """
        warnings: list[str] = []
        if not technical.momentum.divergence_detected:
            return score, warnings

        rsi_div = (technical.momentum.rsi_divergence or "").upper()
        obv_div = (technical.volume.obv_divergence or "").upper()

        if "BULLISH" in rsi_div or "BULLISH" in obv_div:
            if score <= _SCORE_CALL_THRESHOLD:  # neutral or mildly bearish
                score += 0.3
                warnings.append(
                    "Bullish divergence detected — score boosted +0.3 toward CALL"
                )
        elif "BEARISH" in rsi_div or "BEARISH" in obv_div:
            if score >= _SCORE_PUT_THRESHOLD:  # neutral or mildly bullish
                score -= 0.3
                warnings.append(
                    "Bearish divergence detected — score adjusted -0.3 toward PUT"
                )

        return score, warnings

    @staticmethod
    def _apply_gap_fill_override(
        score: float,
        anomaly: Optional[AnomalyVote],
    ) -> tuple[float, list[str]]:
        """Apply a mild gap-fill bias for weak signals (≤ 1.0 strength).

        Gaps fill 60-70 % of the time in Indian markets — this nudges the
        signal direction toward the fill direction when the primary signal
        is already weak or absent.  Strong signals are left unchanged.
        """
        warnings: list[str] = []
        if anomaly is None:
            return score, warnings

        msg = (anomaly.primary_alert_message or "").upper()
        if "GAP" not in msg:
            return score, warnings

        gap_up = "GAP_UP" in msg or "GAP UP" in msg
        gap_down = "GAP_DOWN" in msg or "GAP DOWN" in msg
        if not (gap_up or gap_down):
            return score, warnings

        # Don't override strong signals
        if abs(score) >= 1.0:
            return score, warnings

        if gap_up:
            score -= 0.2
            warnings.append(
                "Gap-up detected — mild bearish bias for gap fill (60-70 % fill rate)"
            )
        else:
            score += 0.2
            warnings.append(
                "Gap-down detected — mild bullish bias for gap fill (60-70 % fill rate)"
            )

        return score, warnings

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
            news_weight=weights.news_weight if "news" in available_components else 0.0,
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
    # Database persistence
    # TODO: remove _save_signal - violates single persistence path.
    # SignalTracker is the sole writer for trading_signals.
    # This method should be removed and the caller should let
    # SignalTracker.record_signal() handle persistence instead.
    # ------------------------------------------------------------------

    def _save_signal(
        self,
        signal: TradingSignal,
        technical: Optional[TechnicalAnalysisResult],
        news: Optional[NewsVote],
        anomaly: Optional[AnomalyVote],
    ) -> None:
        """Insert the signal into the ``trading_signals`` table."""
        regime_db = _REGIME_TO_DB.get(signal.regime, "RANGE_BOUND")

        technical_vote = technical.overall_signal if technical else "NEUTRAL"
        options_vote = (
            technical.options.options_vote
            if technical and technical.options
            else "NEUTRAL"
        )
        news_vote_str = news.vote if news else "NEUTRAL"
        anomaly_vote_str = anomaly.vote if anomaly else "NEUTRAL"

        reasoning_json = json.dumps({
            "text": signal.reasoning,
            "vote_breakdown": signal.vote_breakdown,
            "weighted_score": signal.weighted_score,
            "warnings": signal.warnings,
        }, ensure_ascii=False)

        try:
            self._db.execute(
                """
                INSERT INTO trading_signals
                    (index_id, generated_at, signal_type, confidence_level,
                     entry_price, target_price, stop_loss, risk_reward_ratio,
                     regime, technical_vote, options_vote, news_vote,
                     anomaly_vote, reasoning, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.index_id,
                    signal.generated_at.isoformat(),
                    signal.signal_type,
                    signal.confidence_level,
                    signal.entry_price,
                    signal.target_price,
                    signal.stop_loss,
                    signal.risk_reward_ratio,
                    regime_db,
                    technical_vote,
                    options_vote,
                    news_vote_str,
                    anomaly_vote_str,
                    reasoning_json,
                    "OPEN",
                ),
            )
        except Exception:
            logger.exception("Failed to persist signal %s to DB", signal.signal_id)

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
        today_count = len(self._today_signals.get(index_id, []))
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
