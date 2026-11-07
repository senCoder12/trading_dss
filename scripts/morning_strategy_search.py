#!/usr/bin/env python3
"""
Morning (09:15-12:00) strategy search: opening-range breakout and gap fade.

Price-chart indicators have no edge in the morning (the live score's direction
is no better than the opposite direction there), so this tests structurally
different morning ideas.  All of them are PRE-SPECIFIED below; nothing is
tuned after seeing results.

Strategies (one trade per day per index at most)
------------------------------------------------
ORB   Opening range = high/low of the first OR_LEN minutes (15 or 30), verified
      to start at 09:15 with contiguous 5-minute bars.
      Signal: the first 5-minute CLOSE beyond the range, decided no later
      than the 10:30 bar's close (so at least 85 minutes remain before 12:00).
      Entry = next contiguous 5-minute bar's open.
      Stop = opposite end of the range ("OPP") or its midpoint ("MID").
      Target = R x stop distance, R in {1, 1.5, 2, 3}.
      Filters:
        ALL          — every valid breakout
        GAP_ALIGNED  — overnight gap >= 0.15 % in the breakout direction
        NARROW_OR    — range width <= 40th percentile of the prior 60 days
        NO_EXHAUST   — |gap| + OR width <= prior 20-day median daily range
        IMPULSE      — breakout bar range >= 1.3x OR mean bar range and closes
                       in the extreme 30 % of its range (or volume >= 1.5x OR mean)
FADE  Gap >= 0.3 % / 0.5 % vs the previous close (tracked across all sessions,
      including short/special sessions): trade back toward the previous close.
      Modes:
        RAW — enter at 09:30 open after the first 15 minutes
        REJ — enter at 09:30 open only if the 15-minute range rejected the gap
              (15m close < open on gap-up, > open on gap-down)
        BRK — enter on the first 5-minute close after 09:30 (up to 10:30) that
              breaks the first 5-minute bar's extreme toward the previous close
      Stop = far end of the range prior to entry, target = previous close
      (requires 1 <= R <= 6).
Exit  Target / stop touched first (both in one bar = stop; gap-through-stop on
      a bar open records the actual open loss < -1R), else un-clamped mark-to-
      market at the close of the last bar before 12:00.

Honesty rules (same as the other searches)
------------------------------------------
* Select on the chronologically FIRST 70 % of days (NIFTY50 + BANKNIFTY),
  evaluate the top picks ONCE on the last 30 %.
* Same-day trades across NIFTY50 and BANKNIFTY are aggregated into a single
  daily equal-weight return so t-stats, Sharpe, and Deflated Sharpe count
  independent trading days rather than correlated index duplicates.
* Trial Sharpe variance for DSR is measured across TRAIN trials (not OOS).
* SENSEX is never used for selection: it is an extra, uncontaminated (but
  highly correlated) check of the picks.
* Cost per trade = (COST_FRAC + THETA_FRAC_PER_HOUR x hours_held) x entry price
  in index points (round-trip spread + fees + slippage plus morning ATM
  theta/IV decay), divided by the stop distance to get R.
* Every strategy variant tried counts as a trial in the Deflated Sharpe.
* "validated" = OOS net mean R > 0, >= 20 OOS active days, DSR >= 0.95.

Usage:
    python scripts/morning_strategy_search.py --db DB_COPY [--out report.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from datetime import date, datetime, time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import calibrate_signal_model as cal  # noqa: E402
from src.backtest.overfitting import deflated_sharpe_ratio, return_moments, sharpe_ratio  # noqa: E402

COST_FRAC = 0.00025                  # round-trip spread + slippage + fees in spot fraction
THETA_FRAC_PER_HOUR = 0.00010        # morning ATM option theta + IV crush in spot fraction / hr
MIN_STOP_PCT, MAX_STOP_PCT = 0.001, 0.012
SESSION_OPEN = dtime(9, 15)
DECISION_UNTIL = dtime(10, 30)       # last bar START allowed for a morning signal (leaves >=85m to 12:00)
EXIT_BEFORE = dtime(12, 0)
OR_LENS = (15, 30)
STOPS = ("OPP", "MID")
TARGETS = (1.0, 1.5, 2.0, 3.0)
FILTERS = ("ALL", "GAP_ALIGNED", "NARROW_OR", "NO_EXHAUST", "IMPULSE")
FADE_GAPS = (0.003, 0.005)
FADE_MODES = ("RAW", "REJ", "BRK")
ALL_STRATEGIES = tuple(
    [f"ORB{n}/{s}/{R}R/{flt}" for n, s, R, flt in itertools.product(OR_LENS, STOPS, TARGETS, FILTERS)]
    + [f"FADE/{m}/gap>={thr * 100:.1f}%" for thr, m in itertools.product(FADE_GAPS, FADE_MODES)]
)
SELECT_INDICES = ("NIFTY50", "BANKNIFTY")
HOLDOUT_INDEX = "SENSEX"
MIN_TRAIN, MIN_OOS, TOP_K, DSR_MIN = 80, 20, 5, 0.95


def all_sessions(bars: list[dict]) -> dict[date, list[dict]]:
    """Group all bars by session date, sorted chronologically within each day."""
    out: dict[date, list[dict]] = defaultdict(list)
    for b in bars:
        out[b["timestamp"].date()].append(b)
    for d in out:
        out[d].sort(key=lambda b: b["timestamp"])
    return dict(sorted(out.items()))


def is_regular_session(day_bars: list[dict]) -> bool:
    """True when the session has >=60 bars and starts promptly at 09:15."""
    return len(day_bars) >= 60 and day_bars[0]["timestamp"].time() == SESSION_OPEN


def by_day(bars: list[dict]) -> dict[date, list[dict]]:
    return {d: v for d, v in all_sessions(bars).items() if is_regular_session(v)}


def _has_contiguous_orb(day_bars: list[dict], nb: int) -> bool:
    """Verify the first `nb` bars form a contiguous 5-minute opening range from 09:15."""
    if len(day_bars) <= nb or day_bars[0]["timestamp"].time() != SESSION_OPEN:
        return False
    for k in range(1, nb):
        if (day_bars[k]["timestamp"] - day_bars[k - 1]["timestamp"]).total_seconds() != 300:
            return False
    return True


def _is_next_5m_bar(day_bars: list[dict], sig_i: int) -> bool:
    """Ensure the entry bar immediately follows the signal bar by 5 minutes."""
    return (
        sig_i + 1 < len(day_bars)
        and (day_bars[sig_i + 1]["timestamp"] - day_bars[sig_i]["timestamp"]).total_seconds() == 300
    )


def simulate(day_bars: list[dict], entry_i: int, direction: int, stop: float, target: float, entry: float) -> tuple[float, float, int]:
    """(r_multiple, stop_distance, hold_bars) walking bars from entry_i until 12:00.

    * Accounts for gap-through-stop on subsequent bar opens (losses can exceed -1R).
    * When both stop and target are touched inside one bar, stop wins.
    * Time exit at 12:00 uses the actual mark-to-market close without clamping.
    """
    stop_d = abs(entry - stop)
    up = direction > 0
    last_close = entry
    hold_bars = 0
    for k in range(entry_i, len(day_bars)):
        b = day_bars[k]
        if b["timestamp"].time() >= EXIT_BEFORE:
            break
        hold_bars += 1
        # Gap-through-stop on bar open after entry bar
        if k > entry_i:
            open_px = b["open"]
            if (up and open_px <= stop) or (not up and open_px >= stop):
                gap_r = ((open_px - entry) if up else (entry - open_px)) / stop_d
                return min(-1.0, gap_r), stop_d, hold_bars
        last_close = b["close"]
        hit_stop = b["low"] <= stop if up else b["high"] >= stop
        hit_tgt = b["high"] >= target if up else b["low"] <= target
        if hit_stop:
            return -1.0, stop_d, hold_bars
        if hit_tgt:
            return abs(target - entry) / stop_d, stop_d, hold_bars
    r = ((last_close - entry) if up else (entry - last_close)) / stop_d
    return r, stop_d, max(1, hold_bars)


def _trade_cost_r(entry: float, stop_d: float, hold_bars: int) -> float:
    hold_hours = (hold_bars * 5.0) / 60.0
    return (COST_FRAC + THETA_FRAC_PER_HOUR * hold_hours) * entry / stop_d


def build_trades(index_id: str, bars: list[dict]) -> list[dict]:
    sessions = all_sessions(bars)
    trades: list[dict] = []
    width_hist: dict[int, list[float]] = {n: [] for n in OR_LENS}
    day_range_hist: list[float] = []
    prev_close: float | None = None

    for d, db in sessions.items():
        if not db:
            continue
        day_open = db[0]["open"]
        day_hi = max(b["high"] for b in db)
        day_lo = min(b["low"] for b in db)
        gap = (day_open - prev_close) / prev_close if prev_close else None
        prev_close_today = prev_close
        # Always advance prev_close across every session (including short/special days)
        prev_close = db[-1]["close"]

        if not is_regular_session(db):
            continue

        med_day_range = float(np.median(day_range_hist[-20:])) if len(day_range_hist) >= 10 else None
        day_range_hist.append((day_hi - day_lo) / day_open)

        for n in OR_LENS:
            nb = n // 5
            if not _has_contiguous_orb(db, nb):
                continue
            orb = db[:nb]
            hi, lo = max(b["high"] for b in orb), min(b["low"] for b in orb)
            width = (hi - lo) / day_open
            hist = width_hist[n][-60:]
            narrow = len(hist) >= 30 and width <= float(np.percentile(hist, 40))
            width_hist[n].append(width)
            no_exhaust = med_day_range is not None and (abs(gap or 0.0) + width) <= med_day_range
            orb_mean_vol = float(np.mean([b.get("volume", 0.0) for b in orb]))
            orb_mean_rng = float(np.mean([b["high"] - b["low"] for b in orb]))

            # first breakout close
            sig = None
            for j in range(nb, len(db) - 1):
                if db[j]["timestamp"].time() > DECISION_UNTIL:
                    break
                if db[j]["close"] > hi:
                    sig = (j, 1)
                    break
                if db[j]["close"] < lo:
                    sig = (j, -1)
                    break
            if sig and _is_next_5m_bar(db, sig[0]):
                j, direction = sig
                sb = db[j]
                sb_rng = sb["high"] - sb["low"]
                close_loc = ((sb["close"] - sb["low"]) if direction > 0 else (sb["high"] - sb["close"])) / sb_rng if sb_rng > 0 else 0.0
                sig_vol = float(sb.get("volume", 0.0))
                impulse_ok = (
                    (orb_mean_vol > 0.0 and sig_vol >= 1.5 * orb_mean_vol)
                    or (orb_mean_rng > 0.0 and sb_rng >= 1.3 * orb_mean_rng and close_loc >= 0.70)
                )
                entry_i = j + 1
                entry = db[entry_i]["open"]
                for stop_kind, R, flt in itertools.product(STOPS, TARGETS, FILTERS):
                    if flt == "GAP_ALIGNED" and not (gap is not None and direction * gap >= 0.0015):
                        continue
                    if flt == "NARROW_OR" and not narrow:
                        continue
                    if flt == "NO_EXHAUST" and not no_exhaust:
                        continue
                    if flt == "IMPULSE" and not impulse_ok:
                        continue
                    stop = (lo if direction > 0 else hi) if stop_kind == "OPP" else (hi + lo) / 2
                    stop_d = abs(entry - stop)
                    if not (MIN_STOP_PCT * entry <= stop_d <= MAX_STOP_PCT * entry):
                        continue
                    if (direction > 0 and stop >= entry) or (direction < 0 and stop <= entry):
                        continue
                    target = entry + direction * R * stop_d
                    r, sd, hold_bars = simulate(db, entry_i, direction, stop, target, entry)
                    trades.append({"index_id": index_id, "day": d.isoformat(),
                                   "strategy": f"ORB{n}/{stop_kind}/{R}R/{flt}",
                                   "r": r, "cost_r": _trade_cost_r(entry, sd, hold_bars), "dir": direction})

        # gap fade (after the first 15 minutes, with RAW / REJ / BRK modes)
        if gap is not None and prev_close_today and _has_contiguous_orb(db, 3):
            nb = 3
            orb = db[:nb]
            direction = -1 if gap > 0 else 1
            orb_close = orb[-1]["close"]
            rej_ok = (orb_close < day_open) if direction < 0 else (orb_close > day_open)
            first_bar_ext = orb[0]["low"] if direction < 0 else orb[0]["high"]

            for thr, mode in itertools.product(FADE_GAPS, FADE_MODES):
                if abs(gap) < thr:
                    continue
                entry_i = None
                if mode == "RAW" and _is_next_5m_bar(db, nb - 1):
                    entry_i = nb
                elif mode == "REJ" and rej_ok and _is_next_5m_bar(db, nb - 1):
                    entry_i = nb
                elif mode == "BRK":
                    for j in range(nb - 1, len(db) - 1):
                        if db[j]["timestamp"].time() > DECISION_UNTIL:
                            break
                        broke = (db[j]["close"] < first_bar_ext) if direction < 0 else (db[j]["close"] > first_bar_ext)
                        if broke and _is_next_5m_bar(db, j):
                            entry_i = j + 1
                            break
                if entry_i is None or entry_i >= len(db):
                    continue
                prior_bars = db[:entry_i]
                hi = max(b["high"] for b in prior_bars)
                lo = min(b["low"] for b in prior_bars)
                entry = db[entry_i]["open"]
                stop = hi if direction < 0 else lo
                target = prev_close_today
                stop_d = abs(entry - stop)
                dist = abs(target - entry)
                if stop_d <= 0 or not (MIN_STOP_PCT * entry <= stop_d <= MAX_STOP_PCT * entry):
                    continue
                if (direction > 0 and (stop >= entry or target <= entry)) or \
                   (direction < 0 and (stop <= entry or target >= entry)):
                    continue
                if not (1.0 <= dist / stop_d <= 6.0):
                    continue
                r, sd, hold_bars = simulate(db, entry_i, direction, stop, target, entry)
                trades.append({"index_id": index_id, "day": d.isoformat(),
                               "strategy": f"FADE/{mode}/gap>={thr * 100:.1f}%",
                               "r": r, "cost_r": _trade_cost_r(entry, sd, hold_bars), "dir": direction})
    return trades


def daily_net_returns(rows: list[dict]) -> np.ndarray:
    """Aggregate same-day trades across indices into 1 equal-weight return per active day."""
    by_d: dict[str, list[float]] = defaultdict(list)
    for x in rows:
        by_d[x["day"]].append(x["r"] - x["cost_r"])
    return np.array([float(np.mean(by_d[d])) for d in sorted(by_d)], dtype=float)


def stats(rows: list[dict]) -> dict:
    """Compute day-clustered statistics so correlated indices on the same day count once."""
    n_trades = len(rows)
    day_net = daily_net_returns(rows)
    n = int(day_net.size)
    if n < 2:
        return {"n": n, "n_trades": n_trades, "net": None, "gross": None, "t": 0.0, "sr": 0.0, "hit": None}
    by_d_gross: dict[str, list[float]] = defaultdict(list)
    for x in rows:
        by_d_gross[x["day"]].append(x["r"])
    day_gross = np.array([float(np.mean(by_d_gross[d])) for d in sorted(by_d_gross)], dtype=float)
    sd = float(np.std(day_net, ddof=1))
    return {"n": n, "n_trades": n_trades, "net": float(day_net.mean()), "gross": float(day_gross.mean()),
            "t": float(day_net.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0,
            "sr": sharpe_ratio(day_net), "hit": float(np.mean(day_gross > 0))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    trades: list[dict] = []
    for idx in (*SELECT_INDICES, HOLDOUT_INDEX):
        bars = cal.load_bars(args.db, idx)
        t = build_trades(idx, bars)
        print(f"{idx}: {len(bars)} bars, {len(by_day(bars))} regular sessions, {len(t)} strategy-trades")
        trades += t

    sel = sorted([x for x in trades if x["index_id"] in SELECT_INDICES], key=lambda x: x["day"])
    days = sorted({x["day"] for x in sel})
    cut = days[int(len(days) * 0.7)]
    print(f"\ntrain < {cut} <= OOS · {days[0]} → {days[-1]}   "
          f"cost {COST_FRAC * 100:.3f}% + {THETA_FRAC_PER_HOUR * 100:.3f}%/hr of spot")

    names = sorted(set(ALL_STRATEGIES) | {x["strategy"] for x in trades})
    n_trials = len(names)
    table = []
    for nm in names:
        s_ = [x for x in sel if x["strategy"] == nm]
        tr_rows = [x for x in s_ if x["day"] < cut]
        oo_rows = [x for x in s_ if x["day"] >= cut]
        tr = stats(tr_rows)
        oo = stats(oo_rows)
        ho = stats([x for x in trades if x["index_id"] == HOLDOUT_INDEX and x["strategy"] == nm])
        ho_oos = stats([x for x in trades if x["index_id"] == HOLDOUT_INDEX and x["strategy"] == nm and x["day"] >= cut])
        table.append({"name": nm, "train": tr, "oos": oo, "sensex": ho, "sensex_oos": ho_oos,
                      "oos_net_rs": daily_net_returns(oo_rows).tolist()})
    # Cross-sectional variance of trial Sharpe ratios must come from the TRAIN selection set
    train_srs = [t["train"]["sr"] for t in table if t["train"]["n"] > 5]
    var_sr = float(np.var(train_srs, ddof=1) if len(train_srs) > 1 else 0.0)

    elig = [t for t in table if t["train"]["n"] >= MIN_TRAIN and (t["train"]["net"] or -9) > 0]
    elig.sort(key=lambda t: -t["train"]["t"])
    print(f"\n{n_trials} strategy variants · {len(elig)} with positive net mean R in TRAIN (active days >= {MIN_TRAIN})")
    print("\nSELECTED ON TRAIN, EVALUATED ONCE OUT OF SAMPLE (NIFTY50+BANKNIFTY daily EW), then SENSEX:")
    print(f"{'strategy':<28} {'trD':>4} {'trR':>7} | {'oosD':>4} {'oosNet':>7} {'oosGross':>8} {'DSR':>5} | "
          f"{'SENSEX d':>8} {'net':>7} {'oos-net':>7}  valid")
    picks = elig[:TOP_K]
    for t in picks:
        o = t["oos"]
        dsr = 0.0
        if o["n"] > 3:
            sk, ku = return_moments(t["oos_net_rs"])
            dsr = deflated_sharpe_ratio(o["sr"], n_trials, o["n"], sk, ku, var_sr)
        t["dsr"] = dsr
        t["validated"] = bool(o["n"] >= MIN_OOS and (o["net"] or -9) > 0 and dsr >= DSR_MIN
                              and (t["sensex"]["net"] or -9) > 0)
        f = lambda v: "-" if v is None else f"{v:+.3f}"
        print(f"{t['name']:<28} {t['train']['n']:>4} {t['train']['net']:>+7.3f} | {o['n']:>4} {f(o['net']):>7} "
              f"{f(o['gross']):>8} {dsr:>5.2f} | {t['sensex']['n']:>8} {f(t['sensex']['net']):>7} "
              f"{f(t['sensex_oos']['net']):>7}  {'YES' if t['validated'] else 'no'}")
    if not picks:
        print("  (nothing had positive net mean R in train)")

    print("\nCONTEXT — OOS net mean R by family (mean over variants with >= 15 OOS days; share positive):")
    fam = defaultdict(list)
    for t in table:
        if t["oos"]["n"] >= 15 and t["oos"]["net"] is not None:
            parts = t["name"].split("/")
            key = f"{parts[0]}/{parts[1]}" if parts[0] == "FADE" else f"{parts[0]}/{parts[-1]}"
            fam[key].append(t["oos"]["net"])
    for k in sorted(fam):
        v = fam[k]
        print(f"  {k:<24} mean {np.mean(v):+.3f}   positive {sum(1 for x in v if x > 0)}/{len(v)}")
    gross_best = sorted((t for t in table if t["train"]["n"] >= MIN_TRAIN), key=lambda t: -(t["train"]["gross"] or -9))[:3]
    print("\nbest TRAIN gross (before cost) for reference:")
    for t in gross_best:
        print(f"  {t['name']:<28} train gross {t['train']['gross']:+.3f} (net {t['train']['net']:+.3f}, d={t['train']['n']}) · "
              f"OOS gross {t['oos']['gross'] if t['oos']['gross'] is None else round(t['oos']['gross'], 3)} "
              f"net {t['oos']['net'] if t['oos']['net'] is None else round(t['oos']['net'], 3)}")
    if args.out:
        slim = [{k: v for k, v in t.items() if k != "oos_net_rs"} for t in table]
        args.out.write_text(json.dumps({"generated_at": datetime.now().isoformat(), "cut": cut,
                                        "n_trials": n_trials, "cost_frac": COST_FRAC,
                                        "theta_frac_per_hour": THETA_FRAC_PER_HOUR,
                                        "var_sr_train": var_sr, "table": slim}, indent=2, default=str))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
