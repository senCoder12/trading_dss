#!/usr/bin/env python3
"""
Forward test of ONE pre-registered rule — nothing is tuned here.

RULE (frozen at registration, 2026-11-07; found by a post-hoc search, so it
is a hypothesis, not a validated edge):

    index      NIFTY50, BANKNIFTY
    decision   5-minute bars starting 11:30 <= t < 13:30 IST
    direction  the live score's direction (SignalGenerator.model_score,
               regime-safe, |score| >= 0.25)
    trade      entry at the NEXT bar's open;
               stop  = max(1.0 x ATR, 0.1 % of entry)
               target = 3.0 x stop
               exit  = target / stop, else mark-to-market after 24 bars (120 min)
    sizing     one position at a time per index
    cost       0.10 R deducted per trade (assumed: option costs + theta + spread)

Each run labels completed sessions and appends them to a JSON-lines ledger,
idempotently (one row per index/decision-time).  Rows are tagged
``forward=true`` ONLY for sessions after the registration date, so the
in-sample history can never be mixed into the forward result.

Judge it only after enough forward trades (about 150, i.e. roughly three
months).  Until then the summary prints the numbers with that warning.

Usage:
    python scripts/forward_test_midday.py --db data/db/trading_dss.db --update
    python scripts/forward_test_midday.py --summary
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import calibrate_signal_model as cal  # noqa: E402
import calibration_grid as g  # noqa: E402
from src.backtest.overfitting import probabilistic_sharpe_ratio, return_moments, sharpe_ratio  # noqa: E402

_IST = ZoneInfo("Asia/Kolkata")
RULE_ID = "midday_120m_3R_v1"
REGISTERED_ON = "2026-11-07"
INDICES = ("NIFTY50", "BANKNIFTY")
WINDOW_START, WINDOW_END = "11:30", "13:30"
HORIZON_BARS, TARGET_R, COST_R = 24, 3.0, 0.10
DEFAULT_LEDGER = ROOT / "data" / "forward_test" / f"{RULE_ID}.jsonl"
MIN_FORWARD_TRADES = 150


def load_bars_filled(db: Path, index_id: str) -> list[dict]:
    """Stored 5m bars, plus 5m bars built from 1m for sessions after the last stored one."""
    bars = cal.load_bars(db, index_id)
    last_day = bars[-1]["timestamp"].date() if bars else date(2000, 1, 1)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT timestamp, open, high, low, close FROM price_data "
        "WHERE index_id = ? AND timeframe = '1m' AND timestamp >= ? ORDER BY timestamp",
        (index_id, f"{last_day.isoformat()}T23:59:59"),
    ).fetchall()
    con.close()
    buckets: dict[datetime, list[float]] = {}
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["timestamp"])
        except (TypeError, ValueError):
            continue
        ts = ts.replace(tzinfo=_IST) if ts.tzinfo is None else ts.astimezone(_IST)
        if not (dtime(9, 15) <= ts.time() < dtime(15, 30)) or ts.date() <= last_day:
            continue
        k = ts.replace(minute=ts.minute - ts.minute % 5, second=0, microsecond=0)
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        if k not in buckets:
            buckets[k] = [o, h, l, c]
        else:
            b = buckets[k]
            b[1], b[2], b[3] = max(b[1], h), min(b[2], l), c
    for k in sorted(buckets):
        o, h, l, c = buckets[k]
        bars.append({"timestamp": k, "open": o, "high": h, "low": l, "close": c, "volume": 0.0})
    return bars


def label_sessions(db: Path, index_id: str, days: set[date] | None) -> list[dict]:
    bars = load_bars_filled(db, index_id)
    if len(bars) < cal.WINDOW + HORIZON_BARS:
        return []
    first = next((i for i, b in enumerate(bars) if days is None or b["timestamp"].date() in days), None)
    if first is None:
        return []
    start = max(0, first - (cal.WINDOW - 1))
    samples = cal.replay(index_id, bars[start:], 1)
    rows = []
    for s in samples:
        day = date.fromisoformat(s["ts"][:10])
        if days is not None and day not in days:
            continue
        hhmm = s["ts"][11:16]
        if not (WINDOW_START <= hhmm < WINDOW_END):
            continue
        t_global = start + s["t"]
        res = g.outcome_r(bars, t_global, s["direction"], s["atr"], HORIZON_BARS, TARGET_R)
        if res is None:
            continue                                   # session incomplete: label later
        r, exit_t, _ = res
        rows.append({**s, "t": t_global, "exit_t": exit_t, "r": r})
    return rows


def non_overlapping(rows: list[dict]) -> list[dict]:
    out, busy = [], -1
    for x in sorted(rows, key=lambda r: r["ts"]):
        if x["t"] <= busy:
            continue
        busy = x["exit_t"]
        out.append(x)
    return out


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def update(db: Path, ledger: Path, since: str | None) -> int:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["index_id"], r["day"]) for r in read_ledger(ledger)}
    today = datetime.now(_IST).date()
    new = []
    for idx in INDICES:
        bars = load_bars_filled(db, idx)
        days = sorted({b["timestamp"].date() for b in bars})
        todo = {d for d in days if (idx, d.isoformat()) not in done and d < today
                and (since is None or d.isoformat() >= since)}
        if not todo:
            continue
        # a day is "complete" only once its 15:25 bar exists
        full = {d for d in todo if any(b["timestamp"].date() == d and b["timestamp"].time() >= dtime(15, 20) for b in bars)}
        labelled = label_sessions(db, idx, full)
        by_day: dict[date, list[dict]] = {d: [] for d in full}
        for row in labelled:
            by_day[date.fromisoformat(row["ts"][:10])].append(row)
        for d, rows in sorted(by_day.items()):
            for t in non_overlapping(rows):
                new.append({
                    "rule": RULE_ID, "index_id": idx, "day": d.isoformat(), "ts": t["ts"],
                    "direction": t["direction"], "score": t["score"], "regime": t["regime"],
                    "r_gross": round(t["r"], 4), "r_net": round(t["r"] - COST_R, 4),
                    "forward": d.isoformat() > REGISTERED_ON,
                })
            new.append({"rule": RULE_ID, "index_id": idx, "day": d.isoformat(), "ts": None,
                        "no_trade": not by_day[d], "forward": d.isoformat() > REGISTERED_ON})
    with ledger.open("a") as fh:
        for r in new:
            fh.write(json.dumps(r) + "\n")
    print(f"ledger {ledger}: +{len(new)} rows")
    return 0


def summary(ledger: Path) -> int:
    rows = [r for r in read_ledger(ledger) if r.get("ts")]
    for label, sel in (("FORWARD (after registration)", [r for r in rows if r["forward"]]),
                       ("history in ledger (NOT forward)", [r for r in rows if not r["forward"]])):
        n = len(sel)
        print(f"\n{label}: {n} trades")
        if n < 2:
            continue
        net = np.array([r["r_net"] for r in sel]); gross = np.array([r["r_gross"] for r in sel])
        sk, ku = return_moments(net)
        sr = sharpe_ratio(net)
        print(f"  gross mean {gross.mean():+.3f} R · net mean {net.mean():+.3f} R · "
              f"hit(3R) {np.mean(gross >= TARGET_R - 1e-9):.2f} · PSR(SR>0) {probabilistic_sharpe_ratio(sr, 0.0, n, sk, ku):.2f}")
        for idx in INDICES:
            v = np.array([r["r_net"] for r in sel if r["index_id"] == idx])
            if len(v):
                print(f"  {idx}: n={len(v)} net {v.mean():+.3f}")
        if label.startswith("FORWARD") and n < MIN_FORWARD_TRADES:
            print(f"  ! only {n}/{MIN_FORWARD_TRADES} forward trades — too few to conclude anything.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "db" / "trading_dss.db")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--since", default=None, help="only label sessions on/after YYYY-MM-DD")
    ap.add_argument("--update", action="store_true", help="label completed sessions into the ledger")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    import logging
    logging.basicConfig(level=logging.ERROR)
    if args.update:
        update(args.db, args.ledger, args.since)
    if args.summary or not args.update:
        summary(args.ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
