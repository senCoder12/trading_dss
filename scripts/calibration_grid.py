#!/usr/bin/env python3
"""
Search trade definitions for the signal model — honestly.

The live score is replayed ONCE per index (expensive; cached to JSON), then
each candidate trade definition is labelled cheaply from the stored bars:

    stop   = max(STOP_ATR × ATR, MIN_STOP_PCT × entry)
    target = TARGET_R × stop
    exit   = target / stop first, else mark-to-market at HORIZON bars
    entry  = next bar's open (decision at bar close)

Each definition gets the same treatment as scripts/calibrate_signal_model.py
(monotone shrunk bins fitted on the first 70 % chronologically) and is then
judged OUT OF SAMPLE on the last 30 %:

  * trades are taken only where the fitted P(target first) clears that
    definition's break-even + margin,
  * overlapping trades are removed (one position at a time per index), and
  * per-trade R-multiples give a Sharpe that is DEFLATED for the number of
    definitions tried (Bailey & López de Prado DSR).

A definition is only "validated" if DSR ≥ 0.95, mean OOS R > 0 and there are
enough OOS trades.  R is before option costs/theta (≈ 0.05-0.15 R per trade
for ATM index options at these distances), so a positive mean R below that
is still not an edge.

Usage:
    python scripts/calibration_grid.py --db data/db/trading_dss.db --cache /tmp/cal_cache
    python scripts/calibration_grid.py --cache /tmp/cal_cache --skip-replay   # re-grid only
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import calibrate_signal_model as cal  # noqa: E402
from src.backtest.overfitting import (  # noqa: E402
    deflated_sharpe_ratio, return_moments, sharpe_ratio,
)
from src.engine.signal_calibration import fit_bins  # noqa: E402

HORIZONS = [6, 12, 24]            # 30 / 60 / 120 minutes
TARGET_RS = [1.0, 1.5, 2.0, 3.0]
STOP_ATR = 1.0
MIN_STOP_PCT = 0.001
P_MARGIN = 0.05                   # over break-even 1/(1+R)
MIN_OOS_TRADES = 30
DSR_THRESHOLD = 0.95


def outcome_r(bars, t, direction, atr, horizon, target_r):
    """(R multiple, exit bar index, target_hit) or None if not labelable."""
    if t + horizon >= len(bars):
        return None
    day = bars[t]["timestamp"].date()
    path = bars[t + 1: t + 1 + horizon]
    if any(b["timestamp"].date() != day for b in path):
        return None
    entry = path[0]["open"]
    stop_d = max(STOP_ATR * atr, MIN_STOP_PCT * entry)
    up = direction == "BUY_CALL"
    stop = entry - stop_d if up else entry + stop_d
    target = entry + target_r * stop_d if up else entry - target_r * stop_d
    for i, b in enumerate(path):
        if (b["low"] <= stop) if up else (b["high"] >= stop):
            return -1.0, t + 1 + i, False          # stop first (also if both)
        if (b["high"] >= target) if up else (b["low"] <= target):
            return target_r, t + 1 + i, True
    last = path[-1]["close"]
    r = ((last - entry) if up else (entry - last)) / stop_d
    return max(-1.0, min(target_r, r)), t + horizon, False


def evaluate(samples, bars_by_index, horizon, target_r):
    rows = []
    for s in samples:
        res = outcome_r(bars_by_index[s["index_id"]], s["t"], s["direction"], s["atr"], horizon, target_r)
        if res is None:
            continue
        r, exit_t, hit = res
        rows.append({**s, "r": r, "exit_t": exit_t, "hit": int(hit)})
    rows.sort(key=lambda x: x["ts"])
    if len(rows) < 100:
        return None
    cut = int(len(rows) * 0.7)
    train, test = rows[:cut], rows[cut:]
    bins = fit_bins([x["abs_score"] for x in train], [x["hit"] for x in train], cal.EDGES)

    def p_of(v):
        for b in bins:
            if b.lo <= v < b.hi:
                return b.p_win
        return bins[-1].p_win

    p_min = 1.0 / (1.0 + target_r) + P_MARGIN
    # one position at a time per index, OOS only
    taken, busy_until = [], {}
    for x in test:
        if p_of(x["abs_score"]) < p_min:
            continue
        if x["t"] <= busy_until.get(x["index_id"], -1):
            continue
        busy_until[x["index_id"]] = x["exit_t"]
        taken.append(x["r"])
    all_oos = [x["r"] for x in test]
    res = {
        "horizon_bars": horizon, "target_r": target_r, "p_min": round(p_min, 3),
        "train_hit_rate": round(sum(x["hit"] for x in train) / len(train), 3),
        "oos_mean_r_all": round(float(np.mean(all_oos)), 4),
        "oos_trades": len(taken),
        "oos_mean_r": round(float(np.mean(taken)), 4) if taken else None,
        "oos_hit_rate": round(sum(1 for r in taken if r >= target_r) / len(taken), 3) if taken else None,
        "sr": sharpe_ratio(taken) if len(taken) > 1 else 0.0,
        "bins": [b.__dict__ for b in bins],
    }
    res["skew"], res["kurt"] = return_moments(taken) if len(taken) > 3 else (0.0, 3.0)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "db" / "trading_dss.db")
    ap.add_argument("--indices", default="NIFTY50,BANKNIFTY")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--cache", type=Path, required=True, help="dir for replayed samples")
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write the grid report JSON here")
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.ERROR)
    args.cache.mkdir(parents=True, exist_ok=True)
    indices = [i.strip().upper() for i in args.indices.split(",") if i.strip()]
    bars_by_index, samples = {}, []
    for idx in indices:
        bars = cal.load_bars(args.db, idx)
        bars_by_index[idx] = bars
        f = args.cache / f"samples_{idx}.json"
        if args.skip_replay and f.exists():
            s = json.loads(f.read_text())
        else:
            t0 = time.time()
            s = cal.replay(idx, bars, args.stride, cal.load_futures(args.db, idx))
            f.write_text(json.dumps(s))
            print(f"{idx}: {len(bars)} bars → {len(s)} samples ({time.time() - t0:.0f}s)", flush=True)
        samples.extend(s)

    results = []
    for h, tr in itertools.product(HORIZONS, TARGET_RS):
        r = evaluate(samples, bars_by_index, h, tr)
        if r:
            results.append(r)
    n_trials = len(results)
    srs = [r["sr"] for r in results]
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    for r in results:
        n = r["oos_trades"]
        r["dsr"] = (
            deflated_sharpe_ratio(r["sr"], n_trials, n, r["skew"], r["kurt"], var_sr)
            if n > 3 else 0.0
        )
        r["validated"] = bool(
            n >= MIN_OOS_TRADES and (r["oos_mean_r"] or 0) > 0 and r["dsr"] >= DSR_THRESHOLD
        )

    print(f"\n{n_trials} definitions tried · samples {len(samples)} · "
          f"{samples[0]['ts'][:10] if samples else '-'} → {samples[-1]['ts'][:10] if samples else '-'}")
    print(f"{'horizon':>8} {'R':>4} {'p_min':>6} {'trainHit':>8} {'OOS meanR(all)':>14} "
          f"{'trades':>6} {'hit':>5} {'meanR':>7} {'SR/trade':>8} {'DSR':>6}  valid")
    for r in sorted(results, key=lambda x: -(x["oos_mean_r"] if x["oos_mean_r"] is not None else -9)):
        print(f"{r['horizon_bars'] * 5:>6}m {r['target_r']:>4} {r['p_min']:>6} {r['train_hit_rate']:>8} "
              f"{r['oos_mean_r_all']:>14} {r['oos_trades']:>6} {str(r['oos_hit_rate']):>5} "
              f"{str(r['oos_mean_r']):>7} {r['sr']:>8.3f} {r['dsr']:>6.3f}  {'YES' if r['validated'] else 'no'}")
    if args.out:
        args.out.write_text(json.dumps({
            "generated_at": datetime.now().isoformat(), "n_trials": n_trials,
            "results": results,
        }, indent=2, default=str))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
