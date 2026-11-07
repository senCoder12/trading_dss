#!/usr/bin/env python3
"""
Measure real option trading costs from recorded quotes.

Reads data/forward_test/option_quotes.db (scripts/record_option_quotes.py) and
reports, near the money (|strike - spot| <= one strike step):

  * relative spread (ask - bid) / mid, by index and time of day
  * the same spread expressed in R for the frozen rule: a round trip pays the
    full spread once; one R is |delta| x stop, with |delta| ~ 0.5 and the stop
    at its 0.1 %-of-spot floor.  Real stops are usually wider, so this is an
    UPPER bound on the spread cost in R.
  * theta: premium change per hour after removing the delta effect of the
    spot move (least squares on 60-minute windows of ATM mids).  Needs days of
    data; it prints how many windows it used.

Compare the total with the 0.10 R per trade that every backtest assumed.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "forward_test" / "option_quotes.db"
BUCKETS = (("09:15-11:30", "09:15", "11:30"), ("11:30-13:30", "11:30", "13:30"), ("13:30-15:30", "13:30", "15:30"))
STOP_FLOOR_PCT = 0.001
DELTA = 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT ts, index_id, strike, opt_type, spot, bid, ask, ltp, lot_size FROM option_quotes "
        "WHERE bid > 0 AND ask >= bid"
    ).fetchall()
    total = con.execute("SELECT COUNT(*), COUNT(DISTINCT substr(ts,1,10)) FROM option_quotes").fetchone()
    print(f"{total[0]} quotes over {total[1]} day(s); {len(rows)} with a two-sided book\n")
    if not rows:
        return 0

    step = {"NIFTY50": 50, "BANKNIFTY": 100, "SENSEX": 100}
    print("SPREAD near the money (ATM ± 1 strike), relative to mid:")
    for idx in sorted({r[1] for r in rows}):
        for label, lo, hi in BUCKETS:
            sel = [r for r in rows if r[1] == idx and lo <= r[0][11:16] < hi
                   and abs(r[2] - r[4]) <= step.get(idx, 100)]
            if len(sel) < 5:
                continue
            rel = np.array([(r[6] - r[5]) / ((r[6] + r[5]) / 2) for r in sel])
            absd = np.array([r[6] - r[5] for r in sel])
            mid = np.array([(r[6] + r[5]) / 2 for r in sel])
            spot = np.array([r[4] for r in sel])
            r_cost = absd / (DELTA * STOP_FLOOR_PCT * spot)
            print(f"  {idx:9s} {label}: n={len(sel):5d}  median {np.median(rel) * 100:5.2f}%  "
                  f"p90 {np.percentile(rel, 90) * 100:5.2f}%  median ₹{np.median(absd):.2f} on ₹{np.median(mid):.0f}"
                  f"  → ≤ {np.median(r_cost):.3f} R per round trip")

    print("\nTHETA (ATM mid change per hour, delta-adjusted, 60-minute windows):")
    by = {}
    for r in rows:
        if abs(r[2] - r[4]) <= step.get(r[1], 100) / 2 + 1e-9:
            by.setdefault((r[1], r[3], r[2]), []).append((r[0], r[4], (r[5] + r[6]) / 2))
    est = {}
    for (idx, opt, strike), series in by.items():
        series.sort()
        for a in range(len(series)):
            for b in range(a + 1, len(series)):
                if series[a][0][:10] != series[b][0][:10]:
                    break
                mins = (int(series[b][0][11:13]) * 60 + int(series[b][0][14:16])
                        - int(series[a][0][11:13]) * 60 - int(series[a][0][14:16]))
                if mins == 60:
                    sgn = 1 if opt == "CE" else -1
                    est.setdefault(idx, []).append((series[b][2] - series[a][2] - sgn * DELTA * (series[b][1] - series[a][1])))
    if not est:
        print("  not enough multi-minute data yet (needs several market-hours samples)")
    for idx, v in est.items():
        print(f"  {idx:9s} windows={len(v):5d}  median ₹{np.median(v):+.2f} / hour (negative = decay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
