"""
Optimization Report Generator — Step 7.4 of the Trading Decision Support System.

Generates comprehensive, human-readable optimization reports with:
- Search summary and timing
- Baseline vs optimized comparison
- Top-N ranked results with robustness badges
- Detailed robustness breakdown for the recommended set
- Walk-forward validation results
- Key insights and actionable recommendations
- Heatmap data for Phase 8 dashboard

Usage
-----
::

    gen = OptimizationReportGenerator()
    report = gen.generate_report(opt_result, robustness_reports)
    gen.save_report(report)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Optional

from src.backtest.metrics import BacktestMetrics
from src.backtest.optimizer.optimization_engine import (
    EvaluationResult,
    OptimizationResult,
    _params_to_key,
)
from src.backtest.optimizer.robustness import RobustnessReport
from src.utils.date_utils import get_ist_now

logger = logging.getLogger(__name__)

# Display names for known parameters (optimizer key → short label)
_PARAM_SHORT_NAMES: dict[str, str] = {
    "stop_loss_atr_multiplier": "SL",
    "target_atr_multiplier": "TGT",
    "min_confidence_filter": "Conf",
    "signal_cooldown_bars": "Cool",
    "risk_per_trade_pct": "Risk%",
    "trailing_sl_activation_pct": "Trail",
    "max_positions": "MaxP",
}

# Width of the report box
_BOX_W = 65


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_hash(params: dict[str, Any]) -> str:
    """Short 8-char hash of a parameter dict for traceability."""
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def _fmt_pct(val: float) -> str:
    """Format a percentage with sign: '+5.2%' / '-3.1%'."""
    return f"{val:+.1f}%"


def _fmt_val(val: Any) -> str:
    """Format a parameter value for table display."""
    if isinstance(val, float):
        return f"{val:.1f}"
    if isinstance(val, str) and val in ("LOW", "MEDIUM", "HIGH"):
        return {"LOW": "LOW", "MEDIUM": "MED", "HIGH": "HIGH"}[val]
    return str(val)


def _robustness_badge(score: float | None, is_approved: bool | None) -> str:
    """Return a robustness badge string: check/warning/cross + score."""
    if score is None:
        return "  N/A  "
    if is_approved is None:
        is_approved = score >= 0.5
    if score >= 0.5:
        return f"\u2705 {score:.2f}"
    if score >= 0.3:
        return f"\u26a0\ufe0f {score:.2f}"
    return f"\u274c {score:.2f}"


def _separator() -> str:
    return "\u2501" * _BOX_W


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------


class OptimizationReportGenerator:
    """Generates comprehensive optimization reports and heatmap data."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        opt_result: OptimizationResult,
        robustness_reports: dict[str, RobustnessReport],
    ) -> str:
        """Generate a comprehensive optimization report.

        Parameters
        ----------
        opt_result:
            The completed ``OptimizationResult`` from ``OptimizationEngine.run()``.
        robustness_reports:
            Mapping of ``_params_to_key(params) -> RobustnessReport`` for the
            top-N parameter sets that were robustness-tested.

        Returns
        -------
        str
            The full report text, ready to print or save.
        """
        cfg = opt_result.config
        lines: list[str] = []

        # -- Header --------------------------------------------------------
        lines.append(self._header(cfg))

        # -- Search summary ------------------------------------------------
        lines.append(self._search_summary(opt_result))
        lines.append("")
        lines.append(_separator())

        # -- Baseline ------------------------------------------------------
        lines.append(self._baseline_section(opt_result))
        lines.append("")
        lines.append(_separator())

        # -- Top N table ---------------------------------------------------
        top_n = min(5, len(opt_result.ranked_results))
        if top_n > 0:
            lines.append(
                self._top_n_table(opt_result, robustness_reports, top_n)
            )
            lines.append("")
            lines.append(_separator())

        # -- Recommended parameters ----------------------------------------
        if opt_result.ranked_results:
            lines.append(
                self._recommended_section(opt_result, robustness_reports)
            )
            lines.append("")
            lines.append(_separator())

        # -- Key insights --------------------------------------------------
        if opt_result.ranked_results:
            lines.append(self._insights_section(opt_result, robustness_reports))
            lines.append("")
            lines.append(_separator())

        # -- Final recommendation ------------------------------------------
        lines.append(
            self._recommendation_section(opt_result, robustness_reports)
        )

        # -- Footer --------------------------------------------------------
        best_params = opt_result.best_params or {}
        now = get_ist_now()
        lines.append("")
        lines.append(
            f"Config Hash: {_config_hash(best_params)} | "
            f"Generated: {now.strftime('%Y-%m-%d %H:%M')} IST"
        )
        lines.append(
            "\u255a" + "\u2550" * (_BOX_W - 2) + "\u255d"
        )

        return "\n".join(lines)

    def generate_parameter_heatmap_data(
        self,
        results: list[EvaluationResult],
        param1: str,
        param2: str,
        metric: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        """Generate data for a 2D heatmap showing how two parameters interact.

        Parameters
        ----------
        results:
            List of ``EvaluationResult`` (typically ``opt_result.all_results``).
        param1, param2:
            Parameter names to plot on the two axes.
        metric:
            The metric to display in the grid (default: ``sharpe_ratio``).

        Returns
        -------
        dict with keys:
            ``param1_values``, ``param2_values``, ``metric_grid``,
            ``metric_name``, ``best_cell``.
        """
        # Collect unique values for each parameter
        p1_vals: set[Any] = set()
        p2_vals: set[Any] = set()
        lookup: dict[tuple[Any, Any], float] = {}

        for r in results:
            if r.metrics is None:
                continue
            v1 = r.params.get(param1)
            v2 = r.params.get(param2)
            if v1 is None or v2 is None:
                continue
            p1_vals.add(v1)
            p2_vals.add(v2)
            val = getattr(r.metrics, metric, None)
            if val is not None:
                lookup[(v1, v2)] = val

        if not lookup:
            return {
                "param1_values": [],
                "param2_values": [],
                "metric_grid": [],
                "metric_name": metric,
                "best_cell": None,
            }

        # Sort values
        p1_sorted = sorted(p1_vals, key=lambda x: (isinstance(x, str), x))
        p2_sorted = sorted(p2_vals, key=lambda x: (isinstance(x, str), x))

        # Build grid (rows = param1, cols = param2)
        grid: list[list[float | None]] = []
        best_val: float | None = None
        best_cell: tuple[int, int] | None = None

        for ri, v1 in enumerate(p1_sorted):
            row: list[float | None] = []
            for ci, v2 in enumerate(p2_sorted):
                val = lookup.get((v1, v2))
                row.append(val)
                if val is not None and (best_val is None or val > best_val):
                    best_val = val
                    best_cell = (ri, ci)
            grid.append(row)

        return {
            "param1_values": p1_sorted,
            "param2_values": p2_sorted,
            "metric_grid": grid,
            "metric_name": metric,
            "best_cell": best_cell,
        }

    def save_report(
        self,
        report: str,
        filepath: str | None = None,
        index_id: str = "UNKNOWN",
    ) -> Path:
        """Save report to file.

        Default path: ``data/reports/optimization_{index}_{date}.txt``

        Returns the path the file was written to.
        """
        if filepath is not None:
            out = Path(filepath)
        else:
            reports_dir = Path("data/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            stamp = get_ist_now().strftime("%Y%m%d_%H%M%S")
            out = reports_dir / f"optimization_{index_id}_{stamp}.txt"

        out.write_text(report, encoding="utf-8")
        logger.info("Optimization report saved to %s", out)
        return out

    # ------------------------------------------------------------------
    # Private section builders
    # ------------------------------------------------------------------

    def _header(self, cfg: Any) -> str:
        """Build the top banner."""
        title = "STRATEGY OPTIMIZATION REPORT"
        subtitle = (
            f"{cfg.index_id} \u2014 "
            f"{cfg.start_date.isoformat()} to {cfg.end_date.isoformat()}"
        )
        lines = [
            "\u2554" + "\u2550" * (_BOX_W - 2) + "\u2557",
            "\u2551" + title.center(_BOX_W - 2) + "\u2551",
            "\u2551" + subtitle.center(_BOX_W - 2) + "\u2551",
            "\u2560" + "\u2550" * (_BOX_W - 2) + "\u2563",
        ]
        return "\n".join(lines)

    def _search_summary(self, opt: OptimizationResult) -> str:
        """Search summary block."""
        cfg = opt.config
        space = cfg.parameter_space
        mins = opt.optimization_duration_seconds / 60

        profile_name = _guess_profile_name(space.num_parameters)

        lines = [
            "",
            "SEARCH SUMMARY:",
            f"  Profile: {profile_name} ({space.num_parameters} parameters)",
            f"  Method: {'Grid Search' if cfg.search_method == 'grid' else 'Random Search'}",
            f"  Combinations tested: {opt.total_evaluations} | "
            f"Passed constraints: {opt.passed_constraints} | "
            f"Failed: {opt.failed_constraints}",
            f"  Duration: {mins:.1f} minutes",
            f"  Objective: {_pretty_metric(cfg.primary_objective)} (maximize)",
        ]
        return "\n".join(lines)

    def _baseline_section(self, opt: OptimizationResult) -> str:
        """Baseline (default parameters) block."""
        lines = ["", "BASELINE (Default Parameters):"]

        default_r = opt.default_params_result
        if default_r is None or default_r.metrics is None:
            lines.append("  Not available (default combination was filtered).")
            return "\n".join(lines)

        m = default_r.metrics
        params = default_r.params

        # Param values
        param_strs = [
            f"{_PARAM_SHORT_NAMES.get(k, k)}: {_fmt_val(v)}"
            for k, v in params.items()
        ]
        lines.append("  " + " | ".join(param_strs))

        # Metrics
        lines.append(
            f"  Return: {_fmt_pct(m.total_return_pct)} | "
            f"Sharpe: {m.sharpe_ratio:.2f} | "
            f"Win Rate: {m.win_rate:.1f}% | "
            f"Drawdown: {_fmt_pct(-abs(m.max_drawdown_pct))}"
        )
        lines.append(f"  Trades: {m.total_trades}")

        return "\n".join(lines)

    def _top_n_table(
        self,
        opt: OptimizationResult,
        robustness_reports: dict[str, RobustnessReport],
        top_n: int,
    ) -> str:
        """Ranked table of top N parameter sets with robustness badges."""
        lines: list[str] = ["", f"TOP {top_n} PARAMETER SETS:", ""]

        # Determine which parameters to show (from the first result's keys)
        first = opt.ranked_results[0]
        param_keys = list(first.params.keys())
        param_headers = [_PARAM_SHORT_NAMES.get(k, k[:5]) for k in param_keys]

        # Build header
        hdr_parts = ["Rank"] + param_headers + [
            "Return", "Sharpe", "WinR%", "MaxDD", "Robust?",
        ]
        hdr = " | ".join(f"{h:>6}" for h in hdr_parts)
        sep = "-+-".join("-" * 6 for _ in hdr_parts)

        lines.append(f"  {hdr}")
        lines.append(f"  {sep}")

        # Identify overfitting trap candidates
        trap_ranks: list[int] = []

        for r in opt.ranked_results[:top_n]:
            if r.metrics is None:
                continue

            m = r.metrics
            key = _params_to_key(r.params)
            rob = robustness_reports.get(key)

            rob_score = rob.robustness_score if rob else None
            rob_approved = rob.is_approved if rob else None
            badge = _robustness_badge(rob_score, rob_approved)

            vals = [_fmt_val(r.params.get(k, "")) for k in param_keys]
            row_parts = [f"#{r.rank or 0}"] + vals + [
                _fmt_pct(m.total_return_pct),
                f"{m.sharpe_ratio:.2f}",
                f"{m.win_rate:.1f}%",
                _fmt_pct(-abs(m.max_drawdown_pct)),
                badge,
            ]
            row = " | ".join(f"{p:>6}" for p in row_parts)
            lines.append(f"  {row}")

            # Detect trap: high return but failed robustness
            if rob and not rob.is_approved and m.total_return_pct > 0:
                trap_ranks.append(r.rank or 0)

        # Trap warnings
        for rank in trap_ranks:
            lines.append("")
            lines.append(
                f"  \u26a1 Note: Rank #{rank} has high return but FAILED "
                f"robustness testing."
            )
            lines.append(
                "     This is a classic overfitting signal \u2014 "
                "high return + fragile parameters."
            )

        return "\n".join(lines)

    def _recommended_section(
        self,
        opt: OptimizationResult,
        robustness_reports: dict[str, RobustnessReport],
    ) -> str:
        """Detailed breakdown of the recommended (rank #1) parameters."""
        best = opt.ranked_results[0]
        lines = ["", "RECOMMENDED PARAMETERS (Rank #1):"]

        # Display parameter values
        for k, v in best.params.items():
            display = _PARAM_SHORT_NAMES.get(k, k)
            if k == "stop_loss_atr_multiplier":
                lines.append(f"  Stop Loss: {v}x ATR")
            elif k == "target_atr_multiplier":
                lines.append(f"  Target: {v}x ATR")
            elif k == "min_confidence_filter":
                lines.append(f"  Min Confidence: {v}")
            elif k == "signal_cooldown_bars":
                lines.append(f"  Signal Cooldown: {v} bars")
            elif k == "risk_per_trade_pct":
                lines.append(f"  Risk Per Trade: {v}%")
            elif k == "trailing_sl_activation_pct":
                lines.append(f"  Trailing SL Activation: {v}")
            elif k == "max_positions":
                lines.append(f"  Max Positions: {v}")
            else:
                lines.append(f"  {display}: {v}")

        # Optimization lift
        if opt.optimization_lift_pct is not None and opt.default_params_result:
            dm = opt.default_params_result.metrics
            bm = best.metrics
            if dm and bm:
                abs_lift = bm.total_return_pct - dm.total_return_pct
                lines.append("")
                lines.append(
                    f"  Optimization Lift: {_fmt_pct(abs_lift)} over baseline "
                    f"({_fmt_pct(opt.optimization_lift_pct)} improvement)"
                )

        # Robustness details
        key = _params_to_key(best.params)
        rob = robustness_reports.get(key)
        if rob:
            lines.append("")
            lines.append("ROBUSTNESS DETAILS (Rank #1):")

            if rob.sensitivity_result:
                sr = rob.sensitivity_result
                status = "\u2705" if sr.is_robust else "\u274c"
                lines.append(
                    f"  Sensitivity:    {sr.overall_sensitivity_score:.2f} {status} "
                    f"\u2014 Parameters {'are stable' if sr.is_robust else 'are SENSITIVE'} "
                    f"under \u00b120% variation"
                )

            if rob.stability_result:
                st = rob.stability_result
                total = st.profitable_periods + st.unprofitable_periods
                status = "\u2705" if st.is_stable else "\u274c"
                lines.append(
                    f"  Data Stability: {st.consistency_score:.2f} {status} "
                    f"\u2014 Profitable in {st.profitable_periods}/{total} sub-periods"
                )

            if rob.monte_carlo_result:
                mc = rob.monte_carlo_result
                status = "\u2705" if mc.is_statistically_profitable else "\u274c"
                lines.append(
                    f"  Monte Carlo:    {mc.reliability_score:.2f} {status} "
                    f"\u2014 {mc.pct_profitable:.0%} of simulations profitable, "
                    f"luck factor {mc.luck_factor:.2f}"
                )

            if rob.regime_result:
                rr = rob.regime_result
                status = "\u274c" if rr.is_regime_dependent else "\u2705"
                lines.append(
                    f"  Regime:         {rr.regime_diversity_score:.2f} {status} "
                    f"\u2014 {'Regime dependent' if rr.is_regime_dependent else 'Works across market regimes'}"
                )

            lines.append("")
            lines.append(
                f"  Overall: {rob.robustness_score:.2f} \u2014 "
                f"{rob.robustness_grade} "
                f"({'approved for paper trading' if rob.is_approved else 'NOT approved'})"
            )

        # Walk-forward results
        wf_key = _params_to_key(best.params)
        wf = opt.walk_forward_results.get(wf_key)
        if wf:
            lines.append("")
            lines.append("WALK-FORWARD RESULTS (Rank #1):")
            lines.append(
                f"  Windows: {wf.total_windows} | "
                f"Test profitability: {wf.test_profitability_rate:.0%} "
                f"({wf.profitable_test_windows}/{wf.total_windows} profitable)"
            )
            lines.append(
                f"  Avg train return: {_fmt_pct(wf.avg_train_return)} | "
                f"Avg test return: {_fmt_pct(wf.avg_test_return)}"
            )
            degradation = wf.avg_degradation
            deg_note = "within acceptable range" if degradation < 70 else "HIGH \u2014 possible overfit"
            lines.append(
                f"  Degradation: {degradation:.1f} pct pts "
                f"\u2014 {deg_note}"
            )
            lines.append(
                f"  Overfitting score: {wf.overfitting_score:.2f} \u2014 "
                f"{wf.overfitting_assessment}"
            )

        return "\n".join(lines)

    def _insights_section(
        self,
        opt: OptimizationResult,
        robustness_reports: dict[str, RobustnessReport],
    ) -> str:
        """Key insights derived from the optimization results."""
        lines = ["", "KEY INSIGHTS:", ""]
        insights = self._derive_insights(opt, robustness_reports)

        for i, insight in enumerate(insights, 1):
            lines.append(f"  {i}. {insight}")
            lines.append("")

        if not insights:
            lines.append("  No specific insights available.")

        return "\n".join(lines)

    def _recommendation_section(
        self,
        opt: OptimizationResult,
        robustness_reports: dict[str, RobustnessReport],
    ) -> str:
        """Final recommendation block."""
        lines = ["", "RECOMMENDATION:"]

        if not opt.ranked_results:
            lines.append(
                "  \u274c No viable parameter sets found. "
                "Consider expanding search space or relaxing constraints."
            )
            return "\n".join(lines)

        best = opt.ranked_results[0]
        key = _params_to_key(best.params)
        rob = robustness_reports.get(key)

        if rob and rob.is_approved:
            lines.append(
                "  \u2705 Apply Rank #1 parameters for paper trading."
            )
            lines.append(
                "     Expected live performance: 30-50% below backtest "
                "(market impact, slippage)."
            )
            lines.append(
                "     Monitor for 4 weeks minimum before increasing "
                "position sizes."
            )
            lines.append(
                "     Re-run optimization monthly to detect parameter decay."
            )
        elif rob and not rob.is_approved:
            lines.append(
                "  \u26a0\ufe0f Rank #1 parameters did NOT pass robustness testing."
            )
            lines.append(
                f"     Robustness score: {rob.robustness_score:.2f} "
                f"({rob.robustness_grade})"
            )
            lines.append(
                "     Do NOT deploy to live trading. "
                "Review concerns and re-optimize with different profile."
            )
            if rob.recommendations:
                for rec in rob.recommendations[:3]:
                    lines.append(f"     \u2022 {rec}")
        else:
            lines.append(
                "  \u26a0\ufe0f Robustness testing was not performed."
            )
            lines.append(
                "     Run robustness tests before deploying to paper trading."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Insight derivation
    # ------------------------------------------------------------------

    def _derive_insights(
        self,
        opt: OptimizationResult,
        robustness_reports: dict[str, RobustnessReport],
    ) -> list[str]:
        """Derive actionable insights from the results."""
        insights: list[str] = []

        if not opt.ranked_results:
            return insights

        # Insight: confidence filter impact
        insights.extend(self._confidence_filter_insight(opt))

        # Insight: target size analysis
        insights.extend(self._target_size_insight(opt))

        # Insight: stop loss sweet spot
        insights.extend(self._stop_loss_insight(opt))

        # Insight: overfitting traps
        insights.extend(self._overfitting_trap_insight(opt, robustness_reports))

        return insights[:4]  # Cap at 4 insights

    def _confidence_filter_insight(self, opt: OptimizationResult) -> list[str]:
        """Check if confidence filter significantly impacts results."""
        insights: list[str] = []
        passing = [
            r for r in opt.all_results
            if r.passes_constraints and r.metrics is not None
        ]
        if not passing:
            return insights

        by_conf: dict[str, list[EvaluationResult]] = {}
        for r in passing:
            conf = r.params.get("min_confidence_filter")
            if conf is not None:
                by_conf.setdefault(str(conf), []).append(r)

        if len(by_conf) >= 2:
            avg_wr: dict[str, float] = {}
            for conf, results in by_conf.items():
                avg_wr[conf] = sum(
                    r.metrics.win_rate for r in results if r.metrics
                ) / len(results)

            best_conf = max(avg_wr, key=avg_wr.get)  # type: ignore[arg-type]
            worst_conf = min(avg_wr, key=avg_wr.get)  # type: ignore[arg-type]

            if avg_wr[best_conf] - avg_wr[worst_conf] > 3.0:
                insights.append(
                    f"{best_conf} confidence filter improves results. "
                    f"Avg win rate {avg_wr[best_conf]:.0f}% vs "
                    f"{avg_wr[worst_conf]:.0f}% for {worst_conf}."
                )

        return insights

    def _target_size_insight(self, opt: OptimizationResult) -> list[str]:
        """Analyse target ATR multiplier impact."""
        insights: list[str] = []
        passing = [
            r for r in opt.all_results
            if r.passes_constraints and r.metrics is not None
        ]
        if not passing:
            return insights

        by_tgt: dict[float, list[EvaluationResult]] = {}
        for r in passing:
            tgt = r.params.get("target_atr_multiplier")
            if tgt is not None:
                by_tgt.setdefault(float(tgt), []).append(r)

        if len(by_tgt) >= 2:
            avg_sharpe: dict[float, float] = {}
            for tgt, results in by_tgt.items():
                avg_sharpe[tgt] = sum(
                    r.metrics.sharpe_ratio for r in results if r.metrics
                ) / len(results)

            best_tgt = max(avg_sharpe, key=avg_sharpe.get)  # type: ignore[arg-type]
            worst_tgt = min(avg_sharpe, key=avg_sharpe.get)  # type: ignore[arg-type]

            if best_tgt > worst_tgt:
                direction = "Wider" if best_tgt > worst_tgt else "Tighter"
                insights.append(
                    f"{direction} targets ({best_tgt:.1f} ATR) outperform "
                    f"narrower targets ({worst_tgt:.1f} ATR) on Sharpe ratio."
                )

        return insights

    def _stop_loss_insight(self, opt: OptimizationResult) -> list[str]:
        """Identify stop loss sweet spot."""
        insights: list[str] = []
        passing = [
            r for r in opt.all_results
            if r.passes_constraints and r.metrics is not None
        ]
        if not passing:
            return insights

        by_sl: dict[float, list[EvaluationResult]] = {}
        for r in passing:
            sl = r.params.get("stop_loss_atr_multiplier")
            if sl is not None:
                by_sl.setdefault(float(sl), []).append(r)

        if len(by_sl) >= 2:
            avg_sharpe: dict[float, float] = {}
            for sl, results in by_sl.items():
                avg_sharpe[sl] = sum(
                    r.metrics.sharpe_ratio for r in results if r.metrics
                ) / len(results)

            best_sl = max(avg_sharpe, key=avg_sharpe.get)  # type: ignore[arg-type]
            insights.append(
                f"Stop loss sweet spot is {best_sl:.1f} ATR "
                f"(best avg Sharpe: {avg_sharpe[best_sl]:.2f})."
            )

        return insights

    def _overfitting_trap_insight(
        self,
        opt: OptimizationResult,
        robustness_reports: dict[str, RobustnessReport],
    ) -> list[str]:
        """Flag high-return sets that fail robustness."""
        insights: list[str] = []

        for r in opt.ranked_results[:5]:
            if r.metrics is None:
                continue
            key = _params_to_key(r.params)
            rob = robustness_reports.get(key)
            if rob and not rob.is_approved and r.metrics.total_return_pct > 0:
                insights.append(
                    f"Rank #{r.rank} is a TRAP \u2014 highest return "
                    f"but breaks under robustness testing. "
                    f"Parameters are overfit to specific historical patterns."
                )
                break  # One trap warning is enough

        return insights


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _guess_profile_name(num_params: int) -> str:
    """Guess the profile name from the parameter count."""
    if num_params <= 4:
        return "Conservative"
    if num_params <= 7:
        return "Moderate"
    return "Aggressive"


def _pretty_metric(metric_name: str) -> str:
    """Human-friendly metric name."""
    names = {
        "sharpe_ratio": "Sharpe Ratio",
        "total_return_pct": "Total Return",
        "profit_factor": "Profit Factor",
        "win_rate": "Win Rate",
        "sortino_ratio": "Sortino Ratio",
        "calmar_ratio": "Calmar Ratio",
    }
    return names.get(metric_name, metric_name)
