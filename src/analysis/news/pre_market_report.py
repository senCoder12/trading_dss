"""
Pre-market intelligence report generator.

Consumes an :class:`OvernightData` snapshot (or a dict loaded from the
``overnight_data`` DB table) and renders a Telegram-friendly summary that
is dispatched ~08:50 IST, just before the Indian market open.

The generator is tolerant of missing inputs — when the system was offline
overnight, or a specific data source (US, Asian, FII) is unavailable,
the corresponding section is annotated rather than omitted entirely.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from src.analysis.news.overnight_engine import OvernightData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(data: Any, key: str, default: Any = None) -> Any:
    """Attribute-or-dict lookup so both OvernightData and dict payloads work."""
    if data is None:
        return default
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _trend_emoji(change_pct: Optional[float]) -> str:
    if change_pct is None:
        return "\u26aa"  # white circle
    if change_pct > 0.1:
        return "\U0001f7e2"  # green circle
    if change_pct < -0.1:
        return "\U0001f534"  # red circle
    return "\u26aa"


def _gap_emoji(direction: Optional[str]) -> str:
    if direction == "GAP_UP":
        return "\U0001f7e2"
    if direction == "GAP_DOWN":
        return "\U0001f534"
    return "\u26aa"


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


class PreMarketReportGenerator:
    """Render a pre-market intelligence report for Telegram / dashboard.

    Designed to work on either:
      - an :class:`OvernightData` dataclass instance, or
      - a dict returned from :func:`load_latest_overnight_data`
        (which exposes top-level cols plus a ``payload`` sub-dict).
    """

    def generate_report(self, data: Union[OvernightData, dict, None]) -> str:
        """Render a full report. Returns a fallback banner when ``data`` is None."""
        if data is None:
            return self._empty_report()

        # If we got a DB row dict, unwrap the payload so field names align.
        view = data
        if isinstance(data, dict) and "payload" in data:
            view = dict(data["payload"])
            # Lift through the aggregated columns so missing payload keys
            # still produce a usable report.
            view.setdefault("us_impact", data.get("us_impact"))
            view.setdefault("fii_bias", data.get("fii_bias"))
            view.setdefault("overall_sentiment", data.get("overall_sentiment"))
            view.setdefault("regime_expectation", data.get("regime_expectation"))
            view.setdefault(
                "gap_prediction",
                {
                    "direction": data.get("gap_direction"),
                    "confidence": data.get("gap_confidence"),
                }
                if data.get("gap_direction")
                else None,
            )

        gap = _get(view, "gap_prediction") or {}
        gap_dir = gap.get("direction") if isinstance(gap, dict) else None
        gap_conf = gap.get("confidence") if isinstance(gap, dict) else None
        gap_reason = gap.get("reasoning") if isinstance(gap, dict) else None

        sep = "\u2501" * 24
        lines: list[str] = [
            "\U0001f305 Pre-Market Intelligence",
            sep,
            "",
            f"{_gap_emoji(gap_dir)} Gap Prediction: {gap_dir or 'UNKNOWN'} "
            f"(conf: {(gap_conf or 0):.0%})",
        ]
        if gap_reason:
            lines.append(gap_reason)
        lines.append("")

        # ── US markets ───────────────────────────────────────────────────
        lines.append("\U0001f1fa\U0001f1f8 US Markets (overnight):")
        us_market = _get(view, "us_market") or {}
        if us_market:
            for name, vals in us_market.items():
                change_pct = vals.get("change_pct") if isinstance(vals, dict) else None
                lines.append(
                    f"  {_trend_emoji(change_pct)} {name}: "
                    f"{(change_pct if change_pct is not None else 0):+.1f}%"
                )
            us_impact = _get(view, "us_impact") or "N/A"
            lines.append(f"  Impact: {us_impact}")
        else:
            lines.append("  Data unavailable (system may have been offline overnight)")
        lines.append("")

        # ── Asian markets ────────────────────────────────────────────────
        lines.append("\U0001f30f Asian Markets:")
        asian = _get(view, "asian_markets") or {}
        if asian:
            for name, vals in asian.items():
                change_pct = vals.get("change_pct") if isinstance(vals, dict) else None
                lines.append(
                    f"  {_trend_emoji(change_pct)} {name}: "
                    f"{(change_pct if change_pct is not None else 0):+.1f}%"
                )
        else:
            lines.append("  Data unavailable (markets may not have opened yet)")
        lines.append("")

        # ── FII / DII ────────────────────────────────────────────────────
        lines.append("\U0001f3e6 FII/DII (yesterday):")
        fii_data = _get(view, "fii_data")
        if fii_data:
            fii_net = _get(fii_data, "fii_net_value", 0) or 0
            dii_net = _get(fii_data, "dii_net_value", 0) or 0
            lines.append(f"  FII: \u20b9{fii_net:+,.0f} Cr")
            lines.append(f"  DII: \u20b9{dii_net:+,.0f} Cr")
            bias = _get(view, "fii_bias") or "NEUTRAL"
            lines.append(f"  Bias: {bias}")
        else:
            lines.append("  Data not yet available")

        # ── Key announcements ────────────────────────────────────────────
        announcements = _get(view, "corporate_announcements") or []
        if announcements:
            lines.append("")
            lines.append(f"\U0001f4cb Key Announcements ({len(announcements)}):")
            for a in announcements[:5]:
                sev = _get(a, "impact_severity", "?")
                title = _get(a, "title", None)
                if title is None:
                    article = _get(a, "article", None)
                    title = _get(article, "title", "") or ""
                lines.append(f"  \u2022 [{sev}] {title[:70]}")

        # ── Overnight sentiment ──────────────────────────────────────────
        sentiment = _get(view, "overall_sentiment")
        lines.append("")
        lines.append("\U0001f4f0 Overnight Sentiment: " + self._sentiment_label(sentiment))

        regime = _get(view, "regime_expectation")
        if regime and regime != "NORMAL":
            lines.append("")
            lines.append(f"\u26a0\ufe0f Expected Regime: {regime}")

        # ── Recommendation ───────────────────────────────────────────────
        lines.append("")
        lines.append("\U0001f4a1 Recommendation:")
        lines.append(self._generate_recommendation(view))
        lines.append("")
        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------

    @staticmethod
    def _sentiment_label(value: Optional[float]) -> str:
        if value is None:
            return "UNAVAILABLE \u26aa"
        if value > 0.3:
            return f"BULLISH \U0001f7e2 ({value:+.2f})"
        if value < -0.3:
            return f"BEARISH \U0001f534 ({value:+.2f})"
        return f"NEUTRAL \u26aa ({value:+.2f})"

    @staticmethod
    def _generate_recommendation(data: Any) -> str:
        gap = _get(data, "gap_prediction") or {}
        direction = gap.get("direction") if isinstance(gap, dict) else None
        confidence = gap.get("confidence") if isinstance(gap, dict) else 0
        regime = _get(data, "regime_expectation")

        if direction == "GAP_UP" and (confidence or 0) > 0.6:
            return (
                "  Market likely to open higher. Watch for gap fill in first 15 min.\n"
                "  If gap holds (no fill) \u2192 favor CALL entries.\n"
                "  If gap fills \u2192 wait for support confirmation before entering."
            )
        if direction == "GAP_DOWN" and (confidence or 0) > 0.6:
            return (
                "  Market likely to open lower. Expect volatility in first 30 min.\n"
                "  Don't rush into PUTs \u2014 gap downs often see a bounce attempt.\n"
                "  Wait for 10:00 AM for clearer direction."
            )
        if regime == "EVENT_DRIVEN":
            return (
                "  Event-driven day expected. Reduce position sizes.\n"
                "  Wait for event reaction to settle before trading.\n"
                "  Keep wide stop losses."
            )
        if regime == "VOLATILE_CHOPPY":
            return (
                "  High volatility expected (US market declined significantly).\n"
                "  Consider sitting out first hour.\n"
                "  Only take HIGH confidence signals today."
            )
        return (
            "  Normal conditions expected.\n"
            "  Follow system signals as usual.\n"
            "  Watch for any morning surprises in first 15 min."
        )

    @staticmethod
    def _empty_report() -> str:
        sep = "\u2501" * 24
        return (
            "\U0001f305 Pre-Market Intelligence\n"
            f"{sep}\n\n"
            "No overnight data available. System may have been offline overnight.\n"
            "Trade with standard live-data signals only \u2014 no pre-market bias applied.\n"
            f"\n{sep}"
        )
