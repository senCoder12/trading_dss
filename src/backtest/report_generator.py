"""
Report Generator — Phase 6 Step 6.5.

Generates human-readable text reports for backtest results and
walk-forward validation runs.

Usage
-----
::

    rg = ReportGenerator()
    report = ReportGenerator.generate_backtest_report(result, metrics)
    ReportGenerator.save_report(report, index_id="NIFTY50")

    wf_report = ReportGenerator.generate_walk_forward_report(wf_result)
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

from src.backtest.metrics import BacktestMetrics
from src.backtest.strategy_runner import BacktestResult
from src.backtest.trade_simulator import ClosedTrade
from src.backtest.walk_forward import WalkForwardResult

_BOX_WIDTH = 56   # inner width of the report box


def _center(text: str, width: int = _BOX_WIDTH) -> str:
    return text.center(width)


def _fmt_month(month_key: str) -> str:
    """Convert '2024-08' → 'Aug 2024'."""
    try:
        return datetime.strptime(month_key, "%Y-%m").strftime("%b %Y")
    except ValueError:
        return month_key


def _find_best_worst_month(monthly_returns: list[dict]) -> tuple[str, str]:
    """Return (best_month_label, worst_month_label) from monthly_returns list."""
    if not monthly_returns:
        return "N/A", "N/A"
    best = max(monthly_returns, key=lambda m: m["return_pct"])
    worst = min(monthly_returns, key=lambda m: m["return_pct"])
    return _fmt_month(best["month"]), _fmt_month(worst["month"])


def _exit_avg_pnl(trade_history: list[ClosedTrade], exit_reason: str) -> float:
    """Average net PnL for trades with the given exit_reason."""
    trades = [t for t in trade_history if t.exit_reason == exit_reason]
    if not trades:
        return 0.0
    return sum(t.net_pnl for t in trades) / len(trades)


class ReportGenerator:
    """Static helper that generates formatted text reports."""

    # ------------------------------------------------------------------
    # Backtest report
    # ------------------------------------------------------------------

    @staticmethod
    def generate_backtest_report(
        result: BacktestResult, metrics: BacktestMetrics
    ) -> str:
        cfg = result.config
        scfg = cfg.simulator_config

        capital = scfg.initial_capital
        # max_risk_per_trade_pct is stored as a percentage (e.g. 2.0 = 2%)
        risk_per_trade = scfg.max_risk_per_trade_pct
        risk_per_day = scfg.max_risk_per_day_pct

        grade_desc = {
            "A": "A (Excellent)",
            "B": "B (Good)",
            "C": "C (Average)",
            "D": "D (Below Average)",
            "F": "F (Failing)",
        }.get(metrics.strategy_grade, f"{metrics.strategy_grade} (Unknown)")

        ret_sign = "+" if metrics.total_return_pct >= 0 else ""
        ev_sign = "+" if metrics.expected_value_per_trade >= 0 else ""

        best_month_label, worst_month_label = _find_best_worst_month(
            metrics.monthly_returns
        )

        # Exit reason avg PnL (computed from raw trade history)
        target_avg = _exit_avg_pnl(result.trade_history, "TARGET_HIT")
        sl_avg = _exit_avg_pnl(result.trade_history, "STOP_LOSS_HIT")
        tsl_avg = _exit_avg_pnl(result.trade_history, "TRAILING_SL_HIT")
        eod_avg = _exit_avg_pnl(result.trade_history, "FORCED_EOD")

        total = max(metrics.total_trades, 1)
        eod_pct = metrics.forced_eod_count / total * 100

        # Confidence level totals
        h_total = metrics.high_confidence_trades * metrics.high_confidence_avg_pnl
        m_total = metrics.medium_confidence_trades * metrics.medium_confidence_avg_pnl
        l_total = metrics.low_confidence_trades * metrics.low_confidence_avg_pnl

        # Pre-compute box header lines (avoids nested f-string backslash issue on Py 3.9)
        _mdash = "\u2014"
        _arrow = "\u2192"
        _box_title = _center(f"BACKTEST REPORT {_mdash} {cfg.index_id}")
        _box_period = _center(f"Period: {cfg.start_date} {_arrow} {cfg.end_date}")
        _box_mode = _center(f"Mode: {cfg.mode} | Timeframe: {cfg.timeframe}")

        # Config hash for reproducibility
        config_hash = getattr(result, "config_hash", None) or ""
        config_hash_line = (
            _center(f"Config Hash: {config_hash}  (use for reproducibility)")
            if config_hash
            else ""
        )

        # Slippage description
        if scfg.use_variable_spread:
            slippage_desc = "Variable spread model (moneyness + expiry + premium)"
        else:
            slippage_desc = f"{scfg.slippage_points} pts fixed"

        border = "\u2550" * _BOX_WIDTH
        lines: list[str] = [
            f"\u2554{border}\u2557",
            f"\u2551{_box_title}\u2551",
            f"\u2551{_box_period}\u2551",
            f"\u2551{_box_mode}\u2551",
        ]
        if config_hash_line:
            lines.append(f"\u2551{config_hash_line}\u2551")
        lines += [
            f"\u2560{border}\u2563",
            "",
            "CONFIGURATION:",
            f"  Capital: \u20b9{capital:,.0f} | Risk/Trade: {risk_per_trade:.1f}% | Risk/Day: {risk_per_day:.1f}%",
            f"  Max Positions: {scfg.max_open_positions} | Min Confidence: {cfg.min_confidence}",
            f"  Slippage: {slippage_desc} | Costs: Realistic (STT+brokerage+GST)",
            "",
            "RETURNS:",
            f"  Total Return: {ret_sign}{metrics.total_return_pct:.1f}% (\u20b9{metrics.total_return_amount:,.0f})",
            f"  Annualized: ~{metrics.annualized_return_pct:.1f}%",
            f"  Best Month: +{metrics.best_month_pct:.1f}% ({best_month_label})",
            f"  Worst Month: {metrics.worst_month_pct:.1f}% ({worst_month_label})",
            f"  Monthly Win Rate: {metrics.monthly_win_rate:.1f}%"
            f" ({metrics.positive_months}/{len(metrics.monthly_returns)} months profitable)",
            "",
            "TRADES:",
            f"  Total: {metrics.total_trades} | Wins: {metrics.winning_trades} | Losses: {metrics.losing_trades}",
            f"  Win Rate: {metrics.win_rate:.1f}%",
            f"  Avg Win: \u20b9{metrics.avg_win_amount:,.0f} | Avg Loss: \u20b9{metrics.avg_loss_amount:,.0f}",
            f"  Largest Win: \u20b9{metrics.largest_win:,.0f} | Largest Loss: \u20b9{metrics.largest_loss:,.0f}",
            f"  Profit Factor: {metrics.profit_factor:.2f}",
            f"  Expected Value: {ev_sign}\u20b9{metrics.expected_value_per_trade:,.0f}/trade",
            "",
            "RISK:",
            f"  Max Drawdown: -{metrics.max_drawdown_pct:.1f}% (\u20b9{metrics.max_drawdown_amount:,.0f})",
            f"  Max DD Duration: {metrics.max_drawdown_duration_bars} bars",
            f"  Max Consecutive Losses: {metrics.max_consecutive_losses}",
            f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}",
            f"  Sortino Ratio: {metrics.sortino_ratio:.2f}",
            f"  Calmar Ratio: {metrics.calmar_ratio:.2f}",
            "",
            "BY CONFIDENCE:",
            f"  HIGH:   {metrics.high_confidence_trades:<3} trades | {metrics.high_confidence_win_rate:>5.1f}% WR"
            f" | Avg {metrics.high_confidence_avg_pnl:>+7.0f} | Total {h_total:>+8.0f}",
            f"  MEDIUM: {metrics.medium_confidence_trades:<3} trades | {metrics.medium_confidence_win_rate:>5.1f}% WR"
            f" | Avg {metrics.medium_confidence_avg_pnl:>+7.0f} | Total {m_total:>+8.0f}",
            f"  LOW:    {metrics.low_confidence_trades:<3} trades | {metrics.low_confidence_win_rate:>5.1f}% WR"
            f" | Avg {metrics.low_confidence_avg_pnl:>+7.0f} | Total {l_total:>+8.0f}",
            "",
            "BY SIGNAL TYPE:",
            f"  CALL: {metrics.call_trades:<3} trades | {metrics.call_win_rate:>5.1f}% WR | \u20b9{metrics.call_total_pnl:,.0f}",
            f"  PUT:  {metrics.put_trades:<3} trades | {metrics.put_win_rate:>5.1f}% WR | \u20b9{metrics.put_total_pnl:,.0f}",
            "",
            "BY EXIT REASON:",
            f"  Target Hit:    {metrics.target_hit_count:<3} ({metrics.target_hit_pct:>5.1f}%) | Avg {target_avg:>+8.0f}",
            f"  Stop Loss:     {metrics.sl_hit_count:<3} ({metrics.sl_hit_pct:>5.1f}%) | Avg {sl_avg:>+8.0f}",
            f"  Trailing SL:   {metrics.trailing_sl_count:<3} ({metrics.trailing_sl_pct:>5.1f}%) | Avg {tsl_avg:>+8.0f}",
            f"  Forced EOD:    {metrics.forced_eod_count:<3} ({eod_pct:>5.1f}%) | Avg {eod_avg:>+8.0f}",
            "",
            f"GRADE: {grade_desc}",
            "",
            "RECOMMENDATIONS:",
        ]

        # Recommendations
        recs = ReportGenerator._backtest_recommendations(metrics)
        lines.extend(recs)

        lines += [
            "",
            f"\u255a{border}\u255d",
        ]

        return "\n".join(lines)

    @staticmethod
    def _backtest_recommendations(metrics: BacktestMetrics) -> list[str]:
        recs: list[str] = []

        if metrics.total_trades == 0:
            recs.append("  \u2022 No trades were generated in this period.")
            return recs

        if metrics.has_edge:
            recs.append(
                "  \u2022 Positive edge detected \u2014 proceed to walk-forward validation"
            )
        else:
            recs.append("  \u2022 No clear statistical edge — review signal parameters")

        if metrics.low_confidence_avg_pnl < 0 and metrics.low_confidence_trades > 0:
            recs.append(
                "  \u2022 Consider removing LOW confidence signals (negative EV)"
            )

        if (
            metrics.call_total_pnl > metrics.put_total_pnl
            and metrics.put_total_pnl < 0
            and metrics.put_trades > 0
        ):
            recs.append(
                "  \u2022 CALL signals stronger than PUT \u2014 review PUT entry criteria"
            )
        elif (
            metrics.put_total_pnl > metrics.call_total_pnl
            and metrics.call_total_pnl < 0
            and metrics.call_trades > 0
        ):
            recs.append(
                "  \u2022 PUT signals stronger than CALL \u2014 review CALL entry criteria"
            )

        if metrics.forced_eod_count > 0 and metrics.forced_eod_avg_pnl < 0:
            recs.append(
                "  \u2022 Forced EOD exits are slightly negative \u2014 review timing rules"
            )

        if metrics.max_drawdown_pct > 20:
            recs.append(
                f"  \u2022 Warning: Max drawdown is high ({metrics.max_drawdown_pct:.1f}%)"
                " \u2014 review risk parameters"
            )

        if not recs:
            recs.append("  \u2022 No specific warnings at this time.")

        return recs

    # ------------------------------------------------------------------
    # Walk-forward report
    # ------------------------------------------------------------------

    @staticmethod
    def generate_walk_forward_report(wf_result: WalkForwardResult) -> str:
        cfg = wf_result.config

        # Pre-compute box header lines (avoids nested f-string backslash issue on Py 3.9)
        _mdash = "\u2014"
        _arrow = "\u2192"
        _wf_title = _center(f"WALK-FORWARD VALIDATION REPORT {_mdash} {cfg.index_id}")
        _wf_period = _center(
            f"Full Period: {cfg.full_start_date} {_arrow} {cfg.full_end_date}"
        )

        border = "\u2550" * _BOX_WIDTH
        lines: list[str] = [
            f"\u2554{border}\u2557",
            f"\u2551{_wf_title}\u2551",
            f"\u2551{_wf_period}\u2551",
            f"\u2560{border}\u2563",
            "",
            "WINDOWS:",
            "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510",
            "\u2502  #   \u2502 Train Period        \u2502 Test Period          \u2502 Train% \u2502 Test%  \u2502",
            "\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524",
        ]

        for w in wf_result.windows:
            train_period = (
                f"{w.train_start.strftime('%Y-%m')} \u2192 {w.train_end.strftime('%Y-%m')}"
            )
            test_period = (
                f"{w.test_start.strftime('%Y-%m')} \u2192 {w.test_end.strftime('%Y-%m')}"
            )
            tr_pct = f"{w.train_metrics.total_return_pct:>+6.1f}%"
            te_pct = f"{w.test_metrics.total_return_pct:>+6.1f}%"
            lines.append(
                f"\u2502 {w.window_id:^4} \u2502 {train_period:<19} \u2502"
                f" {test_period:<20} \u2502 {tr_pct:<6} \u2502 {te_pct:<6} \u2502"
            )

        lines += [
            "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518",
            "",
            "AGGREGATE:",
            f"  Avg Train Return: {wf_result.avg_train_return:>+6.2f}%"
            f"  |  Avg Test Return: {wf_result.avg_test_return:>+6.2f}%",
            f"  Avg Degradation: {wf_result.avg_degradation:.1f} percentage points",
            f"  Test Profitability: {wf_result.test_profitability_rate*100:.0f}%"
            f" ({wf_result.profitable_test_windows}/{wf_result.total_windows} windows profitable)",
            f"  Combined Test Return: {wf_result.combined_test_return:>+6.1f}%",
            f"  Combined Test Win Rate: {wf_result.combined_test_win_rate:.1f}%",
            f"  Worst Test Drawdown: -{wf_result.max_test_drawdown:.1f}%",
            "",
            "OVERFITTING ASSESSMENT:",
            f"  Score: {wf_result.overfitting_score:.2f} / 1.0 \u2014 {wf_result.overfitting_assessment}",
            "",
        ]

        # Plain-language interpretation
        interp = ReportGenerator._overfitting_interpretation(wf_result)
        for ln in interp:
            lines.append(f"  {ln}")

        lines += [
            "",
            "VERDICT:",
        ]
        for ln in wf_result.verdict.splitlines():
            lines.append(f"  {ln}")

        lines += [
            "",
            f"\u255a{border}\u255d",
        ]

        return "\n".join(lines)

    @staticmethod
    def _overfitting_interpretation(wf_result: WalkForwardResult) -> list[str]:
        deg = wf_result.avg_degradation
        prof_rate = wf_result.test_profitability_rate * 100

        if wf_result.overfitting_assessment == "LOW_RISK":
            return [
                f"Return degradation of {deg:.1f}pp is minimal.",
                f"Test periods are profitable {prof_rate:.0f}% of the time.",
                "The strategy appears to generalise well out-of-sample.",
            ]
        elif wf_result.overfitting_assessment == "MODERATE_RISK":
            return [
                f"Return degradation is significant ({deg:.1f}pp) but test periods"
                f" are mostly profitable ({prof_rate:.0f}%).",
                "The strategy retains an edge out-of-sample, though smaller than in-sample.",
            ]
        elif wf_result.overfitting_assessment == "HIGH_RISK":
            return [
                f"High degradation detected ({deg:.1f}pp).",
                f"Only {prof_rate:.0f}% of test periods are profitable.",
                "The strategy shows signs of overfitting to training data.",
            ]
        else:  # SEVERE
            return [
                f"Severe overfitting: {deg:.1f}pp degradation.",
                f"Only {prof_rate:.0f}% of test periods are profitable.",
                "The strategy does not generalise — do not trade live.",
            ]

    # ------------------------------------------------------------------
    # Save to file
    # ------------------------------------------------------------------

    @staticmethod
    def save_report(
        report: str,
        filepath: Optional[str] = None,
        index_id: str = "BACKTEST",
    ) -> str:
        """Save *report* to *filepath* (auto-generated if not provided).

        Returns the path where the file was written.
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                "data", "reports", f"backtest_{index_id}_{timestamp}.txt"
            )

        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(report)

        return filepath

    # ------------------------------------------------------------------
    # Trade log
    # ------------------------------------------------------------------

    @staticmethod
    def generate_trade_log(trade_history: list[ClosedTrade]) -> str:
        """Generate a detailed trade-by-trade log table."""
        header = (
            f"{'#':<4} | {'Date':<10} | {'Type':<9} | {'Entry':>7}"
            f" | {'Exit':>7} | {'SL':>7} | {'TGT':>7}"
            f" | {'PnL':>8} | {'Conf':<5} | {'Exit Reason':<16} | Duration"
        )
        separator = "-" * len(header)

        rows = [header, separator]

        for i, t in enumerate(trade_history, 1):
            date_str = t.entry_timestamp.strftime("%Y-%m-%d")
            pnl_str = f"{t.net_pnl:>+,.0f}"
            rows.append(
                f"{i:<4} | {date_str:<10} | {t.trade_type:<9}"
                f" | {t.actual_entry_price:>7,.0f}"
                f" | {t.actual_exit_price:>7,.0f}"
                f" | {t.original_stop_loss:>7,.0f}"
                f" | {t.original_target:>7,.0f}"
                f" | \u20b9{pnl_str:<7}"
                f" | {t.confidence_level[:4]:<5}"
                f" | {t.exit_reason:<16}"
                f" | {t.duration_bars} bars"
            )

        return "\n".join(rows)
