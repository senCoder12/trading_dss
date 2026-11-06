#!/usr/bin/env python3
"""
Replay the LIVE decision engine over stored 5-minute bars.

Runs ``DecisionEngine.run_full_cycle`` once per 5m bar (the live cadence)
with fills at the next bar's open, premium-based exits and statutory costs.
See ``src/backtest/engine_replay.py`` for every assumption.

The --db file is copied to a temporary file first; it is never opened or
written.

Examples
--------
    # the real model (config/signal_calibration.json), last 20 sessions
    python scripts/replay_decision_engine.py --db /path/snapshot.db --index NIFTY50 \\
        --sessions 20 --out /tmp/replay_live

    # DIAGNOSTIC: permissive calibration to exercise the trade path — NOT edge
    python scripts/replay_decision_engine.py --db /path/snapshot.db --index NIFTY50 \\
        --sessions 20 --diagnostic --out /tmp/replay_diag
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Contract facts come from the replay's rules; never download today's master.
os.environ.setdefault("TRADING_DSS_CONTRACT_MASTER_DISABLED", "1")

from src.backtest.engine_replay import (  # noqa: E402
    DIAGNOSTIC_BANNER, run_replay, write_permissive_calibration,
)


def _banner(text: str) -> str:
    line = "!" * 100
    return f"{line}\n!! {text}\n{line}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="SQLite DB to replay (copied, never written)")
    ap.add_argument("--index", default="NIFTY50")
    ap.add_argument("--start", help="first session YYYY-MM-DD (default: --sessions back from --end)")
    ap.add_argument("--end", help="last session YYYY-MM-DD (default: last stored session)")
    ap.add_argument("--sessions", type=int, default=None, help="replay the last N sessions (when --start is omitted)")
    ap.add_argument("--stride", type=int, default=1, help="decide every N bars (1 = live 5-min cycle)")
    ap.add_argument("--calibration", help="signal calibration JSON (default: config/signal_calibration.json)")
    ap.add_argument("--diagnostic", action="store_true",
                    help="use a PERMISSIVE calibration (all bins P=0.60, validated) — exercises the "
                         "trade path; results are NOT evidence of edge")
    ap.add_argument("--capital", type=float, default=500_000)
    ap.add_argument("--out", help="directory for replay_result.json / trades.csv / daily_pnl.csv")
    ap.add_argument("--progress", type=int, default=250, help="log every N decisions (0 = off)")
    ap.add_argument("--verbose", action="store_true", help="engine INFO logging")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("src.backtest.engine_replay").setLevel(logging.INFO if args.verbose else logging.WARNING)

    if args.diagnostic and args.calibration:
        ap.error("--diagnostic and --calibration are mutually exclusive")
    calibration = args.calibration
    if args.diagnostic:
        target_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="replay_diag_"))
        calibration = str(write_permissive_calibration(target_dir / "DIAGNOSTIC_permissive_calibration.json"))
        print(_banner(DIAGNOSTIC_BANNER))
    if not args.start and not args.sessions and not args.end:
        args.sessions = 20

    result = run_replay(
        args.db, args.index, start=args.start, end=args.end,
        calibration_path=calibration, capital=args.capital, stride=args.stride,
        out_dir=args.out, progress_every=args.progress, last_sessions=args.sessions,
    )
    s = result.summary()
    m = result.meta

    print()
    if m.get("mode") == "diagnostic":
        print(_banner(DIAGNOSTIC_BANNER))
    elif m.get("warning"):
        print(f"NOTE: {m['warning']}")
    print(f"Mode               : {m['mode']}  (calibration: {m['calibration_path']} — {m['calibration_status']})")
    print(f"Index / period     : {s['index_id']}  {s['period']}  ({s['sessions']} sessions, "
          f"{m['bars']} bars, stride {m['stride_bars']})")
    print(f"Decisions          : {s['decisions']}   ({m['cycle_ms_mean']} ms/cycle mean, "
          f"p95 {m['cycle_ms_p95']} ms; runtime {m['runtime_seconds']} s)")
    win_rate = "-" if s["win_rate"] is None else f"{s['win_rate']:.1%}"
    print(f"Trades             : {s['trades']}   wins {s['wins']}  losses {s['losses']}  "
          f"win rate {win_rate}")
    print(f"Net P&L            : ₹{s['net_pnl']:,.2f}   (gross ₹{s['gross_pnl']:,.2f}, costs ₹{s['costs']:,.2f})")
    print(f"Max drawdown       : ₹{s['max_drawdown']:,.2f}  ({s['max_drawdown_pct']}% of ₹{m['capital']:,.0f})")
    if s["exit_reasons"]:
        print(f"Exit reasons       : {json.dumps(s['exit_reasons'])}")
    print(f"IV source mix      : {json.dumps(m['iv_source_mix'])}  (synthetic chains: {m['synthetic_chains_built']})")
    print("Decision outcomes  :")
    total = sum(result.decision_counts.values()) or 1
    for key, n in sorted(result.decision_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:6d}  {n / total:6.1%}  {key}")
    if result.trades:
        print("Trades:")
        for t in result.trades:
            print(f"  {t['entry_time'][:16]} → {t['exit_time'][11:16]}  {t['direction']:8s} "
                  f"{t['strike']:.0f}{t['option_type']} {t['expiry']}  {t['lots']}×{t['qty'] // max(t['lots'], 1)}  "
                  f"₹{t['entry_premium']:.2f} → ₹{t['exit_premium']:.2f}  net ₹{t['net_pnl']:,.0f}  "
                  f"[{t['exit_reason']}/{t['exit_trigger']}]")
    if args.out:
        print(f"Written to         : {args.out}")
    if m.get("mode") == "diagnostic":
        print(_banner("REMINDER: DIAGNOSTIC RESULTS ARE NOT EVIDENCE OF EDGE."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
