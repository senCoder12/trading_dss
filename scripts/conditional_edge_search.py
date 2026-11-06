#!/usr/bin/env python3
"""
Conditional-edge search: does the model's direction pay under SOME condition?

Uses the replayed samples cached by scripts/calibration_grid.py (live score,
3 years, NIFTY50 + BANKNIFTY) and tests a small, PRE-SPECIFIED set of
conditions, each over the same 12 trade definitions:

  conditions: regime-aligned / counter-trend / range, time of day,
              strong score, and participant positioning (NSE participant-wise
              futures OI: FII, CLIENT) known BEFORE the session (date < D).

Honesty rules
-------------
* Selection and evaluation are separated in time. Every (definition x
  condition) is ranked on the chronologically FIRST 70 % only; the top picks
  are then evaluated ONCE on the last 30 %.
* One position at a time per index (overlapping signals are not extra trades).
* A flat COST_R is deducted per trade (option costs + theta + spread are
  roughly 0.05-0.15 R at these stop distances; 0.10 is assumed).
* Every combination tried counts as a trial in the Deflated Sharpe.
* A pick is "validated" only if OOS net mean R > 0, OOS trades >= 30 and the
  deflated Sharpe >= 0.95.

Usage:
    python scripts/conditional_edge_search.py --db DB_COPY --cache CACHE_DIR [--out report.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
import sys
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import calibrate_signal_model as cal  # noqa: E402
import calibration_grid as g  # noqa: E402
from src.backtest.overfitting import deflated_sharpe_ratio, return_moments, sharpe_ratio  # noqa: E402

COST_R = 0.10
MIN_TRAIN_TRADES = 150
MIN_OOS_TRADES = 30
TOP_K = 5
Z_WINDOW = 60


# ── participant positioning, point-in-time ────────────────────────────────


def load_positions(db: Path, participant: str) -> tuple[list[str], list[float]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT trade_date, net_contracts FROM fii_futures_positions "
        "WHERE participant = ? AND instrument = 'INDEX_FUT' ORDER BY trade_date",
        (participant,),
    ).fetchall()
    con.close()
    return [r[0] for r in rows], [float(r[1]) for r in rows]


def features_for(day: str, dates: list[str], nets: list[float]) -> dict | None:
    """Positioning known BEFORE `day`: latest date strictly before it."""
    i = bisect_left(dates, day) - 1          # last date < day
    if i < 2:
        return None
    net, prev = nets[i], nets[i - 1]
    hist = nets[max(0, i - Z_WINDOW + 1): i + 1]
    sd = float(np.std(hist)) if len(hist) > 10 else 0.0
    z = (net - float(np.mean(hist))) / sd if sd > 0 else 0.0
    return {"net": net, "chg": net - prev, "z": z, "age_days": None}


# ── conditions ────────────────────────────────────────────────────────────

TREND_UP = {"TREND_UP", "STRONG_TREND_UP"}
TREND_DN = {"TREND_DOWN", "STRONG_TREND_DOWN"}


def _call(s):
    return s["direction"] == "BUY_CALL"


def trend_aligned(s):
    return (_call(s) and s["regime"] in TREND_UP) or (not _call(s) and s["regime"] in TREND_DN)


def counter_trend(s):
    return (_call(s) and s["regime"] in TREND_DN) or (not _call(s) and s["regime"] in TREND_UP)


def hhmm(s):
    return s["ts"][11:16]


def fii_chg_aligned(s):
    f = s.get("fii")
    return bool(f) and ((f["chg"] > 0) == _call(s)) and f["chg"] != 0


def fii_net_aligned(s):
    f = s.get("fii")
    return bool(f) and ((f["net"] > 0) == _call(s)) and f["net"] != 0


def fii_z_aligned(s):
    f = s.get("fii")
    return bool(f) and ((f["z"] >= 1.0 and _call(s)) or (f["z"] <= -1.0 and not _call(s)))


def client_contra(s):
    c = s.get("client")
    return bool(c) and ((c["chg"] < 0) == _call(s)) and c["chg"] != 0


CONDITIONS = {
    "ALL": lambda s: True,
    "TREND_ALIGNED": trend_aligned,
    "COUNTER_TREND": counter_trend,
    "RANGE": lambda s: s["regime"] == "RANGE_BOUND",
    "MORNING": lambda s: hhmm(s) < "11:30",
    "MIDDAY": lambda s: "11:30" <= hhmm(s) < "13:30",
    "AFTERNOON": lambda s: hhmm(s) >= "13:30",
    "SCORE_HI": lambda s: s["abs_score"] >= 0.9,
    "FII_NET_ALIGNED": fii_net_aligned,
    "FII_CHG_ALIGNED": fii_chg_aligned,
    "FII_Z_ALIGNED": fii_z_aligned,
    "CLIENT_CONTRA": client_contra,
    "FII_CHG+TREND": lambda s: fii_chg_aligned(s) and trend_aligned(s),
    "FII_Z+TREND": lambda s: fii_z_aligned(s) and trend_aligned(s),
    "FII_CHG+CLIENT": lambda s: fii_chg_aligned(s) and client_contra(s),
    "MORNING+TREND": lambda s: hhmm(s) < "11:30" and trend_aligned(s),
}


def non_overlapping(rows: list[dict]) -> list[dict]:
    out, busy = [], {}
    for x in sorted(rows, key=lambda r: r["ts"]):
        if x["t"] <= busy.get(x["index_id"], -1):
            continue
        busy[x["index_id"]] = x["exit_t"]
        out.append(x)
    return out


def stats(trades: list[dict]) -> dict:
    r = np.array([t["r"] - COST_R for t in trades], dtype=float)
    gross = np.array([t["r"] for t in trades], dtype=float)
    n = len(r)
    if n < 2:
        return {"n": n, "mean_net": None, "mean_gross": None, "t": 0.0, "hit": None, "sr": 0.0}
    sd = float(np.std(r, ddof=1))
    return {
        "n": n, "mean_net": float(np.mean(r)), "mean_gross": float(np.mean(gross)),
        "t": float(np.mean(r) / (sd / np.sqrt(n))) if sd > 0 else 0.0,
        "hit": float(np.mean(gross >= 0.999 * max(gross))) if n else None,
        "sr": sharpe_ratio(r),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--indices", default="NIFTY50,BANKNIFTY")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    indices = [i.strip().upper() for i in args.indices.split(",") if i.strip()]
    bars_by_index = {i: cal.load_bars(args.db, i) for i in indices}
    f_dates, f_nets = load_positions(args.db, "FII")
    c_dates, c_nets = load_positions(args.db, "CLIENT")
    print(f"participant history: FII {len(f_dates)} sessions {f_dates[0]} → {f_dates[-1]}")

    samples = []
    for i in indices:
        samples += json.loads((args.cache / f"samples_{i}.json").read_text())
    for s in samples:
        day = s["ts"][:10]
        s["fii"] = features_for(day, f_dates, f_nets)
        s["client"] = features_for(day, c_dates, c_nets)
    samples.sort(key=lambda s: s["ts"])
    cut_ts = samples[int(len(samples) * 0.7)]["ts"]
    print(f"{len(samples)} samples · train < {cut_ts[:10]} ≤ OOS · "
          f"{samples[0]['ts'][:10]} → {samples[-1]['ts'][:10]}")

    defs = list(itertools.product(g.HORIZONS, g.TARGET_RS))
    outcomes = {}
    for h, tr in defs:
        rows = []
        for s in samples:
            res = g.outcome_r(bars_by_index[s["index_id"]], s["t"], s["direction"], s["atr"], h, tr)
            if res is None:
                continue
            rows.append({**s, "r": res[0], "exit_t": res[1]})
        outcomes[(h, tr)] = rows

    table = []
    for (h, tr), rows in outcomes.items():
        for cname, pred in CONDITIONS.items():
            sel = [x for x in rows if pred(x)]
            train = non_overlapping([x for x in sel if x["ts"] < cut_ts])
            oos = non_overlapping([x for x in sel if x["ts"] >= cut_ts])
            table.append({
                "h": h, "R": tr, "cond": cname,
                "train": stats(train), "oos": stats(oos),
                "oos_rs": [t["r"] - COST_R for t in oos],
            })
    n_trials = len(table)

    eligible = [t for t in table if t["train"]["n"] >= MIN_TRAIN_TRADES and (t["train"]["mean_net"] or -9) > 0]
    eligible.sort(key=lambda t: -t["train"]["t"])
    picks = eligible[:TOP_K]

    var_sr = float(np.var([t["oos"]["sr"] for t in table if t["oos"]["n"] > 5], ddof=1) or 0.0)
    print(f"\n{n_trials} (definition × condition) trials · cost {COST_R} R/trade · "
          f"{len(eligible)} had positive net mean R in TRAIN with ≥{MIN_TRAIN_TRADES} trades")
    print("\nSELECTED ON TRAIN (top by t-stat), EVALUATED ONCE OUT OF SAMPLE:")
    print(f"{'def':>10} {'condition':<16} {'trainN':>6} {'trainR':>7} {'t':>5} | "
          f"{'oosN':>5} {'oosR(net)':>9} {'oosR(gross)':>11} {'DSR':>6} valid")
    for t in picks:
        o = t["oos"]
        rs = t["oos_rs"]
        if o["n"] > 3:
            sk, ku = return_moments(rs)
            dsr = deflated_sharpe_ratio(o["sr"], n_trials, o["n"], sk, ku, var_sr)
        else:
            dsr = 0.0
        t["dsr"] = dsr
        t["validated"] = bool(o["n"] >= MIN_OOS_TRADES and (o["mean_net"] or -9) > 0 and dsr >= g.DSR_THRESHOLD)
        print(f"{t['h'] * 5:>6}m/{t['R']:<3} {t['cond']:<16} {t['train']['n']:>6} "
              f"{t['train']['mean_net']:>+7.3f} {t['train']['t']:>5.1f} | "
              f"{o['n']:>5} {('%+.3f' % o['mean_net']) if o['mean_net'] is not None else '-':>9} "
              f"{('%+.3f' % o['mean_gross']) if o['mean_gross'] is not None else '-':>11} {dsr:>6.2f} "
              f"{'YES' if t['validated'] else 'no'}")

    # context: baseline + each condition averaged over definitions (OOS, net)
    print("\nCONTEXT — OOS net mean R by condition, averaged over the 12 definitions "
          "(positive and consistent would be needed):")
    for cname in CONDITIONS:
        rows = [t for t in table if t["cond"] == cname and t["oos"]["n"] >= 20]
        if rows:
            m = np.mean([t["oos"]["mean_net"] for t in rows])
            pos = sum(1 for t in rows if t["oos"]["mean_net"] > 0)
            print(f"  {cname:<16} mean {m:+.3f}   positive in {pos}/{len(rows)} definitions")

    if args.out:
        slim = [{k: v for k, v in t.items() if k != "oos_rs"} for t in table]
        args.out.write_text(json.dumps(
            {"generated_at": datetime.now().isoformat(), "n_trials": n_trials, "cost_r": COST_R,
             "cut_ts": cut_ts, "picks": [{k: v for k, v in p.items() if k != "oos_rs"} for p in picks],
             "table": slim}, indent=2, default=str))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
