"""
Signal Tracker — lifecycle and performance analytics for all trading signals.

Records every signal produced by the Decision Engine, tracks outcomes when
positions are closed, and computes performance statistics for calibration
and continuous improvement.

Thread-safe: all public methods acquire ``_lock`` before touching shared state.
"""

# ================================================================
# TABLE OWNERSHIP: This class is the SOLE writer for `trading_signals`.
# No other module should INSERT, UPDATE, or DELETE from trading_signals.
# SignalGenerator returns TradingSignal → SignalTracker persists them.
# RiskManager may call SignalTracker methods to update outcomes.
# ================================================================

from __future__ import annotations

import json
import logging
import math
import statistics
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.database.db_manager import DatabaseManager
from src.database import queries as Q

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PerformanceStats:
    """Aggregated performance statistics over a rolling window."""

    period_days: int

    # Volume
    total_signals: int
    actionable_signals: int       # BUY_CALL + BUY_PUT (excluding NO_TRADE)
    total_trades: int             # closed (WIN or LOSS)

    # Win rate
    wins: int
    losses: int
    win_rate: float               # wins / total_trades

    # P&L
    total_pnl: float              # ₹
    avg_pnl_per_trade: float
    largest_win: float
    largest_loss: float
    avg_win: float
    avg_loss: float
    profit_factor: float          # abs(sum_wins) / abs(sum_losses)

    # Risk
    max_drawdown: float           # ₹ peak-to-trough on cumulative PnL series
    max_drawdown_pct: float       # as % of capital (uses 100 000 default)
    sharpe_ratio: float           # annualised, using daily returns
    avg_risk_reward: float        # actual achieved R:R

    # By confidence
    high_confidence_win_rate: float
    medium_confidence_win_rate: float
    low_confidence_win_rate: float

    # Breakdowns
    win_rate_by_regime: dict      # {regime: float}
    win_rate_by_index: dict       # {index_id: float}
    pnl_by_index: dict            # {index_id: float}

    # Signal quality
    avg_confidence_of_wins: float
    avg_confidence_of_losses: float
    no_trade_accuracy: float      # stub — hard to measure without ground truth

    # Edge
    expected_value_per_trade: float
    is_profitable: bool
    edge_comment: str


@dataclass
class CalibrationReport:
    """Are our confidence levels calibrated?"""

    high_confidence_expected_win_rate: float = 0.70
    high_confidence_actual_win_rate: float = 0.0
    high_confidence_calibration: str = "UNCALIBRATED"   # OVER_CONFIDENT / CALIBRATED / UNDER_CONFIDENT

    medium_confidence_expected: float = 0.55
    medium_confidence_actual: float = 0.0
    medium_confidence_calibration: str = "UNCALIBRATED"

    low_confidence_expected: float = 0.45
    low_confidence_actual: float = 0.0
    low_confidence_calibration: str = "UNCALIBRATED"

    overall_calibration: str = "NEEDS_ADJUSTMENT"       # WELL_CALIBRATED / NEEDS_ADJUSTMENT
    suggested_adjustments: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Signal Tracker
# ---------------------------------------------------------------------------


class SignalTracker:
    """
    Records every signal and its outcome; produces performance analytics.

    Parameters
    ----------
    db:
        Shared :class:`DatabaseManager` instance.
    capital:
        Total trading capital used for drawdown % calculations.
    """

    # Bands for calibration judgement
    _CALIBRATION_BAND: float = 0.08  # ±8% around expected = CALIBRATED

    def __init__(self, db: DatabaseManager, capital: float = 100_000.0) -> None:
        self._db = db
        self._capital = capital
        self._lock = threading.Lock()

        # In-memory cache of recent signals for fast look-ups.
        # {signal_id: {row dict from DB}}
        self._recent: dict[str, dict] = {}

        logger.info("SignalTracker initialised (capital=₹%.0f)", capital)
        self._cleanup_on_startup()

    # ------------------------------------------------------------------
    # Startup cleanup
    # ------------------------------------------------------------------

    def _cleanup_on_startup(self) -> None:
        """Resolve any OPEN signals that RiskManager may have missed.

        Safety net: if RiskManager cleanup already resolved a signal (closed_at is set),
        we skip it here. Only process signals that are still truly orphaned.
        """
        try:
            from src.utils.date_utils import get_ist_now
            now = get_ist_now()
            today_open = now.replace(hour=9, minute=15, second=0, microsecond=0)

            still_orphaned = self._db.fetch_all(
                """SELECT id, index_id, signal_type FROM trading_signals
                   WHERE outcome = 'OPEN' AND generated_at < ? AND closed_at IS NULL""",
                (today_open.isoformat(),)
            )

            for signal in still_orphaned:
                self._db.execute(
                    """UPDATE trading_signals
                       SET outcome = 'EXPIRED', closed_at = ?
                       WHERE id = ?""",
                    (now.isoformat(), signal['id'])
                )
                logger.info(
                    f"SignalTracker cleanup: expired signal #{signal['id']} "
                    f"({signal['index_id']} {signal['signal_type']})"
                )

            if still_orphaned:
                logger.info(f"SignalTracker cleanup: expired {len(still_orphaned)} orphaned signal(s)")

        except Exception as e:
            logger.error(f"SignalTracker startup cleanup failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Public API — signal recording
    # ------------------------------------------------------------------

    def record_signal(self, signal) -> None:
        """
        Persist a TradingSignal or RefinedSignal to the ``trading_signals`` table.

        NO_TRADE signals are recorded for traceability but marked with outcome=None.
        Actionable signals get outcome='OPEN'.

        Parameters
        ----------
        signal:
            A ``TradingSignal`` or ``RefinedSignal`` from the engine.
        """
        with self._lock:
            self._do_record_signal(signal)

    def record_outcome(
        self,
        signal_id: str,
        exit_price: float,
        exit_reason: str,
        pnl: float,
    ) -> None:
        """
        Update outcome for a closed position.

        Parameters
        ----------
        signal_id:
            UUID of the signal.
        exit_price:
            Actual exit price of the index.
        exit_reason:
            Human-readable reason (e.g. "TARGET", "SL", "TIME").
        pnl:
            Realised P&L in ₹ (negative = loss).
        """
        with self._lock:
            outcome = "WIN" if pnl > 0 else "LOSS"
            now_ist = datetime.now(tz=_IST).isoformat()

            # Find DB row by signal UUID stored in reasoning JSON
            db_id = self._find_db_id(signal_id)
            if db_id is None:
                logger.warning("record_outcome: signal_id %s not found in DB", signal_id)
                return

            try:
                self._db.execute(
                    Q.UPDATE_SIGNAL_OUTCOME,
                    (outcome, exit_price, round(pnl, 2), now_ist, db_id),
                )
                logger.info(
                    "Outcome recorded: signal=%s outcome=%s pnl=₹%.0f reason=%s",
                    signal_id, outcome, pnl, exit_reason,
                )
            except Exception:
                logger.exception("Failed to record outcome for signal %s", signal_id)

            # Evict from in-memory cache so next read is fresh
            self._recent.pop(signal_id, None)

    # ------------------------------------------------------------------
    # Public API — analytics
    # ------------------------------------------------------------------

    def get_performance_stats(self, days: int = 30) -> PerformanceStats:
        """
        Compute performance statistics over the last *days* calendar days.

        Parameters
        ----------
        days:
            Rolling window in calendar days.

        Returns
        -------
        PerformanceStats
        """
        since = (datetime.now(tz=_IST) - timedelta(days=days)).isoformat()

        rows = self._db.fetch_all(
            """
            SELECT signal_type, confidence_level, entry_price, target_price,
                   stop_loss, risk_reward_ratio, regime, outcome,
                   actual_exit_price, actual_pnl, index_id,
                   reasoning
            FROM   trading_signals
            WHERE  generated_at >= ?
            ORDER  BY generated_at ASC
            """,
            (since,),
        )

        total_signals = len(rows)
        actionable = [r for r in rows if r["signal_type"] != "NO_TRADE"]
        closed = [r for r in rows if r["outcome"] in ("WIN", "LOSS")]

        wins = [r for r in closed if r["outcome"] == "WIN"]
        losses = [r for r in closed if r["outcome"] == "LOSS"]

        win_pnls = [r["actual_pnl"] for r in wins if r["actual_pnl"] is not None]
        loss_pnls = [r["actual_pnl"] for r in losses if r["actual_pnl"] is not None]
        all_pnls = [r["actual_pnl"] for r in closed if r["actual_pnl"] is not None]

        total_trades = len(closed)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades else 0.0

        total_pnl = sum(all_pnls)
        avg_pnl = total_pnl / total_trades if total_trades else 0.0
        largest_win = max(win_pnls, default=0.0)
        largest_loss = min(loss_pnls, default=0.0)
        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0

        sum_wins = sum(win_pnls)
        sum_losses = abs(sum(loss_pnls))
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

        max_dd, max_dd_pct = self._compute_max_drawdown(all_pnls)
        sharpe = self._compute_sharpe(all_pnls, days)

        avg_rr = (
            sum(r["risk_reward_ratio"] for r in closed if r["risk_reward_ratio"])
            / total_trades
            if total_trades
            else 0.0
        )

        # By confidence
        hcwr = self._wr_for_confidence(closed, "HIGH")
        mcwr = self._wr_for_confidence(closed, "MEDIUM")
        lcwr = self._wr_for_confidence(closed, "LOW")

        # By regime
        wr_by_regime = self._wr_by_field(closed, "regime")

        # By index
        wr_by_index: dict[str, float] = {}
        pnl_by_index: dict[str, float] = {}
        for idx in {r["index_id"] for r in closed}:
            idx_rows = [r for r in closed if r["index_id"] == idx]
            idx_wins = [r for r in idx_rows if r["outcome"] == "WIN"]
            wr_by_index[idx] = len(idx_wins) / len(idx_rows) if idx_rows else 0.0
            pnl_by_index[idx] = sum(
                r["actual_pnl"] for r in idx_rows if r["actual_pnl"] is not None
            )

        # Signal quality
        avg_conf_wins = self._avg_confidence_score(wins)
        avg_conf_losses = self._avg_confidence_score(losses)

        # Expected value
        ev = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)) if total_trades else 0.0
        is_profitable = ev > 0 and win_rate > 0.45

        if ev > 50:
            edge_comment = "Positive edge detected — strategy is performing well."
        elif ev > 0:
            edge_comment = "Slight positive edge — monitor closely."
        elif total_trades < 10:
            edge_comment = "Insufficient data — need more trades for reliable statistics."
        elif ev > -50:
            edge_comment = "No clear edge — review signal thresholds."
        else:
            edge_comment = "Losing strategy — immediate review needed."

        return PerformanceStats(
            period_days=days,
            total_signals=total_signals,
            actionable_signals=len(actionable),
            total_trades=total_trades,
            wins=win_count,
            losses=loss_count,
            win_rate=round(win_rate, 4),
            total_pnl=round(total_pnl, 2),
            avg_pnl_per_trade=round(avg_pnl, 2),
            largest_win=round(largest_win, 2),
            largest_loss=round(largest_loss, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 4),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 4),
            sharpe_ratio=round(sharpe, 4),
            avg_risk_reward=round(avg_rr, 4),
            high_confidence_win_rate=round(hcwr, 4),
            medium_confidence_win_rate=round(mcwr, 4),
            low_confidence_win_rate=round(lcwr, 4),
            win_rate_by_regime=wr_by_regime,
            win_rate_by_index=wr_by_index,
            pnl_by_index=pnl_by_index,
            avg_confidence_of_wins=round(avg_conf_wins, 4),
            avg_confidence_of_losses=round(avg_conf_losses, 4),
            no_trade_accuracy=0.0,  # requires external ground-truth data
            expected_value_per_trade=round(ev, 2),
            is_profitable=is_profitable,
            edge_comment=edge_comment,
        )

    def get_signal_history(
        self,
        index_id: Optional[str] = None,
        days: int = 30,
        signal_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Return signal history (with outcomes) for review.

        Parameters
        ----------
        index_id:
            If given, filter to this index only.
        days:
            Rolling window.
        signal_type:
            If given, filter to this signal type (BUY_CALL / BUY_PUT / NO_TRADE).
        """
        since = (datetime.now(tz=_IST) - timedelta(days=days)).isoformat()

        clauses = ["generated_at >= ?"]
        params: list = [since]

        if index_id:
            clauses.append("index_id = ?")
            params.append(index_id)
        if signal_type:
            clauses.append("signal_type = ?")
            params.append(signal_type)

        where = " AND ".join(clauses)
        rows = self._db.fetch_all(
            f"""
            SELECT * FROM trading_signals
            WHERE {where}
            ORDER BY generated_at DESC
            LIMIT 500
            """,
            tuple(params),
        )
        return rows

    def get_calibration_report(self) -> CalibrationReport:
        """
        Assess whether HIGH/MEDIUM/LOW confidence signals achieve their
        expected win rates.

        Returns
        -------
        CalibrationReport
        """
        since = (datetime.now(tz=_IST) - timedelta(days=90)).isoformat()

        rows = self._db.fetch_all(
            """
            SELECT confidence_level, outcome
            FROM   trading_signals
            WHERE  generated_at >= ?
              AND  signal_type != 'NO_TRADE'
              AND  outcome IN ('WIN', 'LOSS')
            """,
            (since,),
        )

        report = CalibrationReport()

        def _wr(level: str) -> float:
            subset = [r for r in rows if r["confidence_level"] == level]
            if not subset:
                return -1.0  # sentinel — no data
            wins = sum(1 for r in subset if r["outcome"] == "WIN")
            return wins / len(subset)

        def _calibrate(actual: float, expected: float) -> str:
            if actual < 0:
                return "NO_DATA"
            diff = actual - expected
            if diff > self._CALIBRATION_BAND:
                return "UNDER_CONFIDENT"   # signals are better than we claim
            if diff < -self._CALIBRATION_BAND:
                return "OVER_CONFIDENT"    # signals under-deliver
            return "CALIBRATED"

        h_wr = _wr("HIGH")
        m_wr = _wr("MEDIUM")
        l_wr = _wr("LOW")

        report.high_confidence_actual_win_rate = max(h_wr, 0.0)
        report.high_confidence_calibration = _calibrate(h_wr, report.high_confidence_expected_win_rate)

        report.medium_confidence_actual = max(m_wr, 0.0)
        report.medium_confidence_calibration = _calibrate(m_wr, report.medium_confidence_expected)

        report.low_confidence_actual = max(l_wr, 0.0)
        report.low_confidence_calibration = _calibrate(l_wr, report.low_confidence_expected)

        calibrations = [
            report.high_confidence_calibration,
            report.medium_confidence_calibration,
            report.low_confidence_calibration,
        ]
        non_data = [c for c in calibrations if c not in ("NO_DATA",)]
        miscalibrated = [c for c in non_data if c != "CALIBRATED"]

        report.overall_calibration = (
            "WELL_CALIBRATED" if len(miscalibrated) == 0 and len(non_data) > 0
            else "NEEDS_ADJUSTMENT"
        )

        # Suggestions
        suggestions: list[str] = []
        if report.high_confidence_calibration == "OVER_CONFIDENT":
            suggestions.append(
                "HIGH confidence threshold is too loose — raise score threshold or "
                "tighten confirmation requirements."
            )
        elif report.high_confidence_calibration == "UNDER_CONFIDENT":
            suggestions.append(
                "HIGH confidence signals are outperforming expectations — "
                "consider increasing position size for HIGH signals."
            )

        if report.medium_confidence_calibration == "OVER_CONFIDENT":
            suggestions.append(
                "MEDIUM confidence win rate is below expectation — "
                "consider increasing threshold from MEDIUM to HIGH."
            )

        if report.low_confidence_calibration == "OVER_CONFIDENT":
            suggestions.append(
                "LOW confidence signals are underperforming — "
                "consider suppressing LOW confidence signals entirely."
            )

        if not suggestions:
            suggestions.append("No immediate adjustments required — keep monitoring.")

        report.suggested_adjustments = suggestions
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_record_signal(self, signal) -> None:
        """Insert signal into DB.  Caller must hold ``_lock``."""
        now_ist = datetime.now(tz=_IST).isoformat()

        signal_type = getattr(signal, "signal_type", "NO_TRADE")
        is_actionable = signal_type in ("BUY_CALL", "BUY_PUT")
        outcome = "OPEN" if is_actionable else None

        # Extract component votes from vote_breakdown if present
        vb = getattr(signal, "vote_breakdown", {}) or {}
        tech_vote = (vb.get("trend") or {}).get("vote", "NEUTRAL")
        opt_vote = (vb.get("options") or {}).get("vote", "NEUTRAL")
        news_vote_str = (vb.get("news") or {}).get("vote", "NEUTRAL")
        anomaly_vote_str = (vb.get("anomaly") or {}).get("vote", "NEUTRAL")

        # Regime DB value
        raw_regime = getattr(signal, "regime", "RANGE_BOUND") or "RANGE_BOUND"
        db_regime = self._regime_to_db(raw_regime)

        # Store signal_id in reasoning JSON so we can look it up later
        signal_id = getattr(signal, "signal_id", "")
        reasoning_raw = getattr(signal, "reasoning", "")
        try:
            reasoning_dict = json.loads(reasoning_raw) if isinstance(reasoning_raw, str) and reasoning_raw.startswith("{") else {"text": reasoning_raw}
        except (json.JSONDecodeError, TypeError):
            reasoning_dict = {"text": str(reasoning_raw)}
        reasoning_dict["signal_id"] = signal_id

        # Stash option-trade fields (strike/expiry/premium/lots/risk) for the
        # REST API to surface in the UI. These aren't first-class DB columns,
        # so they ride inside the reasoning JSON blob.
        if is_actionable:
            strike = getattr(signal, "recommended_strike", None)
            expiry = getattr(signal, "recommended_expiry", None)
            if strike or expiry:
                reasoning_dict["option_trade"] = {
                    "strike": float(strike) if strike is not None else None,
                    "expiry": expiry,
                    "option_type": "CE" if signal_type == "BUY_CALL" else "PE",
                    "premium": (
                        float(getattr(signal, "option_premium", 0) or 0) or None
                    ),
                    "lots": int(
                        getattr(signal, "lots", 0)
                        or getattr(signal, "suggested_lot_count", 0)
                        or 0
                    ) or None,
                    "max_loss_amount": (
                        float(
                            getattr(signal, "max_loss_amount", 0)
                            or getattr(signal, "estimated_max_loss", 0)
                            or 0
                        ) or None
                    ),
                    "risk_pct_of_capital": (
                        float(getattr(signal, "risk_pct_of_capital", 0) or 0) or None
                    ),
                }

        reasoning_json = json.dumps(reasoning_dict)

        # Structured audit trail (if present)
        audit_data = getattr(signal, "signal_audit", None)
        audit_json_str = (
            json.dumps(audit_data, ensure_ascii=False, default=str)
            if audit_data else None
        )

        try:
            self._db.execute(
                Q.INSERT_TRADING_SIGNAL,
                (
                    getattr(signal, "index_id", ""),
                    now_ist,
                    signal_type,
                    getattr(signal, "confidence_level", "LOW"),
                    float(getattr(signal, "entry_price", 0) or 0),
                    float(getattr(signal, "target_price", 0) or 0),
                    float(getattr(signal, "stop_loss", 0) or 0),
                    float(getattr(signal, "risk_reward_ratio", 0) or 0),
                    db_regime,
                    str(tech_vote),
                    str(opt_vote),
                    str(news_vote_str),
                    str(anomaly_vote_str),
                    reasoning_json,
                    outcome,
                    None,  # actual_exit_price
                    None,  # actual_pnl
                    None,  # closed_at
                ),
            )

            # Persist audit trail if the column exists
            if audit_json_str:
                try:
                    row = self._db.fetch_one(
                        "SELECT id FROM trading_signals WHERE index_id = ? AND generated_at = ? ORDER BY id DESC LIMIT 1",
                        (getattr(signal, "index_id", ""), now_ist),
                    )
                    if row:
                        self._db.execute(
                            "UPDATE trading_signals SET audit_json = ? WHERE id = ?",
                            (audit_json_str, row["id"]),
                        )
                except Exception:
                    logger.debug("SignalTracker: audit_json column not yet available — skipping")

            logger.debug(
                "SignalTracker: recorded %s %s conf=%s",
                signal_type,
                getattr(signal, "index_id", ""),
                getattr(signal, "confidence_level", ""),
            )
        except Exception:
            logger.exception("SignalTracker: failed to record signal %s", signal_id)

    def _find_db_id(self, signal_id: str) -> Optional[int]:
        """Find the DB integer PK for a signal UUID stored in reasoning JSON."""
        # Try in-memory cache first
        cached = self._recent.get(signal_id)
        if cached:
            return cached.get("id")

        # Search recent rows (reasoning contains signal_id as JSON field)
        since = (datetime.now(tz=_IST) - timedelta(days=7)).isoformat()
        rows = self._db.fetch_all(
            """
            SELECT id, reasoning FROM trading_signals
            WHERE generated_at >= ?
            ORDER BY generated_at DESC
            LIMIT 1000
            """,
            (since,),
        )
        for row in rows:
            try:
                d = json.loads(row["reasoning"] or "{}")
                if d.get("signal_id") == signal_id:
                    return int(row["id"])
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        return None

    @staticmethod
    def _regime_to_db(regime: str) -> str:
        """Map full regime name to DB CHECK constraint value."""
        _MAP = {
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
        return _MAP.get(regime, "RANGE_BOUND")

    @staticmethod
    def _compute_max_drawdown(pnl_series: list[float]) -> tuple[float, float]:
        """
        Compute maximum peak-to-trough drawdown on a running P&L series.

        Returns (max_drawdown_₹, max_drawdown_pct_of_100k).
        """
        if not pnl_series:
            return 0.0, 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for pnl in pnl_series:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        capital = 100_000.0
        pct = max_dd / capital * 100 if capital > 0 else 0.0
        return max_dd, pct

    @staticmethod
    def _compute_sharpe(pnl_series: list[float], days: int) -> float:
        """
        Annualised Sharpe ratio from a sequence of per-trade P&L values (₹).

        Uses a ₹100 000 capital base and 6.5% annualised risk-free rate.
        """
        if len(pnl_series) < 2:
            return 0.0

        capital = 100_000.0
        risk_free_daily = 0.065 / 252

        returns = [p / capital for p in pnl_series]
        avg_ret = statistics.mean(returns)
        std_ret = statistics.stdev(returns)

        if std_ret == 0:
            return 0.0

        # Scale to daily: assume `days` calendar span, ~252 trading days/year
        trading_days = max(1, int(days * 252 / 365))
        trades_per_day = len(pnl_series) / trading_days

        daily_excess = (avg_ret - risk_free_daily) * trades_per_day
        daily_std = std_ret * math.sqrt(trades_per_day)

        if daily_std == 0:
            return 0.0

        return (daily_excess / daily_std) * math.sqrt(252)

    @staticmethod
    def _wr_for_confidence(closed_rows: list[dict], level: str) -> float:
        subset = [r for r in closed_rows if r["confidence_level"] == level]
        if not subset:
            return 0.0
        wins = sum(1 for r in subset if r["outcome"] == "WIN")
        return wins / len(subset)

    @staticmethod
    def _wr_by_field(closed_rows: list[dict], field_name: str) -> dict[str, float]:
        result: dict[str, float] = {}
        values = {r[field_name] for r in closed_rows if r.get(field_name)}
        for val in values:
            subset = [r for r in closed_rows if r.get(field_name) == val]
            wins = sum(1 for r in subset if r["outcome"] == "WIN")
            result[val] = round(wins / len(subset), 4) if subset else 0.0
        return result

    @staticmethod
    def _avg_confidence_score(rows: list[dict]) -> float:
        """
        Extract confidence score from reasoning JSON and return mean.

        Falls back to level-based approximation if no numeric score stored.
        """
        _LEVEL_MAP = {"HIGH": 0.75, "MEDIUM": 0.55, "LOW": 0.35}
        scores: list[float] = []
        for row in rows:
            try:
                d = json.loads(row.get("reasoning") or "{}")
                s = d.get("confidence_score")
                if s is not None:
                    scores.append(float(s))
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
            level = row.get("confidence_level", "LOW")
            scores.append(_LEVEL_MAP.get(level, 0.35))
        return sum(scores) / len(scores) if scores else 0.0
