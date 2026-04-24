"""
Post-Market Decision Engine Analysis
Runs the full 8-step pipeline on today's backfilled data for all F&O indices
and reports any actionable signals, votes, and regime info.
"""

import sys
import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# Silence noisy loggers; keep our own output clean
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
for noisy in ("yfinance", "urllib3", "httpx", "peewee", "apscheduler"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import DatabaseManager
from src.engine.decision_engine import DecisionEngine
from config.settings import settings

_IST = ZoneInfo("Asia/Kolkata")

BANNER = "=" * 72
DIVIDER = "-" * 72
THIN = "·" * 72


def fmt_price(v):
    if v is None or v == 0:
        return "—"
    return f"₹{v:,.2f}"


def run_analysis():
    print(f"\n{'█'*72}")
    print(f"  POST-MARKET ANALYSIS — {datetime.now(_IST).strftime('%A, %d %B %Y  %H:%M IST')}")
    print(f"{'█'*72}\n")

    # ── Init ──────────────────────────────────────────────────────────────
    db = DatabaseManager(settings.database.path)
    db.connect()
    engine = DecisionEngine(db)

    # Skip live news freshness requirement — we're in post-market
    engine._last_news_cycle = datetime.now(tz=_IST)

    # ── Run all F&O indices ───────────────────────────────────────────────
    print("⏳  Running decision engine across all F&O indices...\n")
    results = engine.run_all_indices()

    if not results:
        print("❌  No results returned. Check if kill-switch is active or F&O registry is empty.")
        return

    # ── Categorise ───────────────────────────────────────────────────────
    actionable   = [r for r in results if r.is_actionable]
    buy_calls    = [r for r in results if getattr(r.signal, "signal_type", "") == "BUY_CALL"]
    buy_puts     = [r for r in results if getattr(r.signal, "signal_type", "") == "BUY_PUT"]
    no_trades    = [r for r in results if getattr(r.signal, "signal_type", "") == "NO_TRADE"]

    # ── Summary header ────────────────────────────────────────────────────
    print(BANNER)
    print(f"  SUMMARY — {len(results)} indices analysed")
    print(BANNER)
    print(f"  ✅  Actionable signals  : {len(actionable)}")
    print(f"  📈  BUY CALL            : {len(buy_calls)}  →  {', '.join(r.index_id for r in buy_calls) or '—'}")
    print(f"  📉  BUY PUT             : {len(buy_puts)}  →  {', '.join(r.index_id for r in buy_puts) or '—'}")
    print(f"  ⏸   NO TRADE            : {len(no_trades)}")
    print(BANNER)

    # ── Detail per index ──────────────────────────────────────────────────
    for res in results:
        sig = res.signal
        stype    = getattr(sig, "signal_type",     "N/A")
        clevel   = getattr(sig, "confidence_level","?")
        cscore   = getattr(sig, "confidence_score", 0.0) or 0.0
        regime   = res.regime.regime if res.regime else "UNKNOWN"
        entry    = getattr(sig, "refined_entry",     None) or getattr(sig, "entry_price",  None)
        target   = getattr(sig, "refined_target",    None) or getattr(sig, "target_price", None)
        sl       = getattr(sig, "refined_stop_loss", None) or getattr(sig, "stop_loss",    None)
        rr       = getattr(sig, "risk_reward_ratio", 0.0) or 0.0
        strike   = getattr(sig, "recommended_strike", None)
        expiry   = getattr(sig, "recommended_expiry", None)
        premium  = getattr(sig, "option_premium",     None)
        wscore   = getattr(sig, "weighted_score",     0.0) or 0.0
        reasoning= getattr(sig, "reasoning",          "—")

        # Emoji prefix
        if stype == "BUY_CALL":
            icon = "🟢 BUY CALL"
        elif stype == "BUY_PUT":
            icon = "🔴 BUY PUT "
        else:
            icon = "⬜ NO TRADE"

        print(f"\n{DIVIDER}")
        print(f"  {icon}  ──  {res.index_id:<20}  Conf: {clevel} ({cscore:.2f})  Regime: {regime}")
        print(THIN)

        # Votes
        tech_sig = "—"
        if res.technical_result:
            tech_sig = res.technical_result.overall_signal or "—"
        news_v = res.news_vote.vote  if res.news_vote  else "—"
        anom_v = res.anomaly_vote.vote if res.anomaly_vote else "—"
        print(f"  Votes     → Technical: {tech_sig:<12}  News: {news_v:<12}  Anomaly: {anom_v}")
        print(f"  Weighted score: {wscore:.4f}")

        # Levels (only if not NO_TRADE)
        if stype != "NO_TRADE":
            print(THIN)
            print(f"  📊 Levels:")
            print(f"     Entry    : {fmt_price(entry)}")
            print(f"     Target   : {fmt_price(target)}")
            print(f"     Stop Loss: {fmt_price(sl)}")
            print(f"     R/R Ratio: 1:{rr:.2f}")
            if strike and expiry:
                opt = "CE" if stype == "BUY_CALL" else "PE"
                print(f"  📋 Option  : {res.index_id} {strike:.0f} {opt}  Expiry: {expiry}  Premium: {fmt_price(premium)}")

        # Reasoning (truncated)
        if reasoning and reasoning != "—":
            short = reasoning[:240].replace("\n", " ")
            print(THIN)
            print(f"  Reasoning: {short}{'…' if len(reasoning)>240 else ''}")

        # Warnings
        all_warnings = list(res.warnings or []) + list(getattr(sig, "warnings", []) or [])
        if all_warnings:
            print(f"  ⚠️  Warnings: {' | '.join(all_warnings[:4])}")

        # Timings
        total = res.total_duration_ms
        print(f"  ⏱  Duration: {total}ms")

    # ── Actionable signal detail ──────────────────────────────────────────
    if actionable:
        print(f"\n{'█'*72}")
        print(f"  ✅  ACTIONABLE SIGNALS — TRADE-READY")
        print(f"{'█'*72}")
        for res in actionable:
            msg = engine.generate_alert_message(res)
            if msg:
                print()
                for line in msg.splitlines():
                    print(f"  {line}")
    else:
        print(f"\n{DIVIDER}")
        print("  ℹ️   No actionable signals generated today.")
        print("       This may be due to: market closed, low confidence, data staleness,")
        print("       or risk filters blocking marginal setups.")
        print(DIVIDER)

    # ── Timing summary ────────────────────────────────────────────────────
    if results:
        avg_ms = sum(r.total_duration_ms for r in results) / len(results)
        print(f"\n  Average engine cycle time: {avg_ms:.0f}ms per index")

    print(f"\n{'█'*72}")
    print("  Analysis complete.")
    print(f"{'█'*72}\n")


if __name__ == "__main__":
    run_analysis()
