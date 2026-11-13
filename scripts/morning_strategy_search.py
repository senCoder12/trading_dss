#!/usr/bin/env python3
"""
Morning (09:15-12:00) strategy search: opening-range breakout and gap fade.

Research only.  This script reads a COPY of the price database read-only and
never touches the live engine, its signals or the production database.

Every rule below is fixed before the data is looked at; nothing is tuned on
out-of-sample (OOS) or SENSEX results.

Data
----
* 5-minute index bars (``price_data``, timeframe ``5m``) via
  ``calibrate_signal_model.load_bars`` and daily bars (timeframe ``1d``) for
  the trading calendar and the official previous close.
* A day is usable only when its MORNING is complete: 33 contiguous 5-minute
  bars 09:15 ... 11:55.  Nothing after 12:00 is used to decide eligibility
  (no look-ahead through "the session had N bars").
* Previous close = the previous trading day's ``1d`` close (the exchange's
  official close).  If the previous trading day in the data is more than
  MAX_PREV_GAP_DAYS calendar days back (missing data), gap-based rules skip
  the day.  Muhurat sessions (e.g. 2023-11-12, 2024-11-01, 2025-10-21) ARE in
  the ``1d`` table, so the first session after Diwali measures its gap
  against the muhurat close, which is the exchange's official previous close.
  Muhurat dates count as calendar dates in the split; their mornings are
  never usable.

Strategies (one trade per day per index per variant, enforced)
--------------------------------------------------------------
ORB   Opening range = high/low of the first OR_LEN minutes (15 or 30).
      Signal: the first 5-minute bar whose CLOSE is beyond the range; the
      signal bar must START no later than 10:55 (SIGNAL_CUTOFF).
      Entry = the next 5-minute bar's open.  Stop/target checking starts on
      the entry bar.
      Stop = opposite end of the range ("OPP") or its midpoint ("MID").
      Target = R x stop distance, R in {1, 1.5, 2, 3}.
      Filters:
        ALL          every valid breakout
        GAP_ALIGNED  overnight gap >= 0.15 % in the breakout direction
        NARROW_OR    range width (fraction of the 09:15 open) <= 40th
                     percentile of the widths of the PRIOR 60 usable days
                     (today excluded; needs all 60, else the filter is off)
        NO_EXHAUST   |gap| + OR width <= median full-day range of the
                     prior 20 days (needs all 20)
        IMPULSE      signal bar range >= 1.3 x the OR's mean bar range and
                     closes in the extreme 30 % of its range (or volume >=
                     1.5 x OR mean volume; index volume is usually 0)
FADE  Gap = (09:15 open - previous close) / previous close, |gap| >= 0.3 % or
      0.5 %.  Gap-up => SHORT, gap-down => LONG, target = previous close.
      Opening range = 09:15-09:30 (first three 5-minute bars).
      Modes (entry is always a bar open at or after 09:30):
        RAW  enter at the 09:30 bar's open
        REJ  as RAW, only if the 15-minute range rejected the gap
             (09:25 close < 09:15 open on gap-up, > on gap-down)
        BRK  enter at the open after the first bar from 09:25 on (signal
             bar start <= 10:55) whose close breaks the first bar's
             extreme toward the previous close
      Stop = the far extreme of all bars before entry.
      Skipped when the previous close was touched by any bar before entry
      (the gap already filled), or when the entry open is at/through it.
      Fade R = |entry - previous close| / stop distance; trade only when
      1 <= R <= 6.
All   Stop distance must be within 0.1 % ... 1.2 % of the entry price.

Execution (``simulate``)
------------------------
* Bars are walked from the entry bar.  On the entry bar the open IS the fill,
  so only its high/low can hit stop or target.
* Adverse gap through the stop on a later bar's open: filled at that OPEN,
  i.e. a loss larger than 1 R (conservative).
* Favourable gap through the target on a bar's open: filled at the TARGET
  price, no gap bonus (conservative).
* Stop and target both inside one bar: stop first.
* Otherwise exit at the close of the last bar before 12:00 (the 11:55 bar),
  marked to market, unclamped.

Costs
-----
COST_FRAC is the assumed ROUND-TRIP cost of one trade (spread + brokerage +
fees + slippage, entry and exit together) as a fraction of the index level
at entry.  It is converted to R by dividing by the stop distance in points.
A fixed theta allowance THETA_FRAC_PER_HOUR x hours held is added in every
scenario (R is measured on the index; a long option also bleeds time value).

    base    0.025 %   (selection uses ONLY this)
    stress  0.050 %, 0.075 %   (reported for the frozen picks only)

Split, selection, freeze
------------------------
* The trading calendar = the union of ``1d`` dates of all three indices.  The
  first 70 % of those dates is TRAIN, the last 30 % OOS; a date's partition is
  the same for every index.
* Selection uses NIFTY50 + BANKNIFTY TRAIN trades only.  Same-day trades of
  the two indices are averaged into one equal-weight daily return, so all
  statistics count trading days, not correlated duplicates.
* Ranking metric: train t-stat of the mean daily net R at the base cost,
  among variants with >= MIN_TRAIN active train days (ties by name).  The
  TOP_K = 5 best are frozen (names, train stats, SHA-256) and written to disk
  BEFORE any OOS or SENSEX row is read.  The frozen list is never changed.
* SENSEX is never used for selection; it is an extra (correlated) check.

Deflated Sharpe (Bailey & Lopez de Prado 2014; src/backtest/overfitting.py)
--------------------------------------------------------------------------
* N trials = every variant in ALL_STRATEGIES (all OR lengths x stops x
  targets x filters, plus all fade gap x mode combinations), whether or not
  it produced trades.  The variants share trades, so N overstates the
  independent trials: conservative.
* Trial Sharpe variance V = sample variance of the per-day train Sharpe of
  every variant with >= MIN_TRAIN active train days.
* SR0 = E[max SR] = sqrt(V) ((1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e))).
* DSR = PSR(SR, SR0, n, skew, Pearson kurtosis) of the daily net R series.
  In-sample DSR uses the train series; OOS DSR the OOS series with the SAME
  N and V (conservative: the OOS check itself only chose among TOP_K).
* Sharpe is per ACTIVE day (days with at least one trade), not annualised.

* Earlier configurations were run on the SAME data and their results seen:
  a 50-variant version (2026-11-07) and the 86-variant commit e9edfa9 (signal
  cutoff 10:30), which printed OOS numbers for every variant.  Those count
  as PRIOR_TRIALS, so the DSR uses N = N_TRIALS + PRIOR_TRIALS.

OOS is NOT a clean first look: the window from the cut to 2026-11-06 was
already inspected by those earlier versions.  Only sessions after
2026-11-06 (a forward test of the frozen picks) are a true one-time OOS.

Validated only if EVERY criterion holds (fixed before the run):
  train net mean > 0 (base) with >= MIN_TRAIN days; >= MIN_OOS OOS days;
  OOS net mean > 0 at the base AND at 0.05 % cost; OOS DSR >= 0.95 (base);
  SENSEX net mean > 0 on OOS dates only (base, >= MIN_OOS days).  SENSEX on
  train dates is reported as context only (same sessions as the selection).

"hit" = share of trades with R > 0 after that row's cost.

Usage:
    python scripts/morning_strategy_search.py --db DB_COPY --out report.json [--md report.md]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import calibrate_signal_model as cal  # noqa: E402
from src.backtest.overfitting import deflate_selected_sharpe, return_moments, sharpe_ratio  # noqa: E402

_IST = ZoneInfo("Asia/Kolkata")

# ── costs ───────────────────────────────────────────────────────────────────
BASE_COST_FRAC = 0.00025             # round trip, fraction of entry index level
STRESS_COST_FRACS = (0.0005, 0.00075)
COST_SCENARIOS = (BASE_COST_FRAC, *STRESS_COST_FRACS)
VALIDATION_STRESS_COST = 0.0005      # OOS must also be net positive here
THETA_FRAC_PER_HOUR = 0.00010        # ATM option time decay, fraction of index / hour held

# ── session / execution ─────────────────────────────────────────────────────
SESSION_OPEN = dtime(9, 15)
FADE_ENTRY_FROM = dtime(9, 30)
SIGNAL_CUTOFF = dtime(10, 55)        # last signal-bar START time (entry by 11:00)
EXIT_BEFORE = dtime(12, 0)
BAR_SECONDS = 300
MORNING_BARS = 33                    # 09:15 ... 11:55
MAX_PREV_GAP_DAYS = 4                # Fri -> Mon plus one holiday
MIN_STOP_PCT, MAX_STOP_PCT = 0.001, 0.012

# ── strategy grid ───────────────────────────────────────────────────────────
OR_LENS = (15, 30)
STOPS = ("OPP", "MID")
TARGETS = (1.0, 1.5, 2.0, 3.0)
FILTERS = ("ALL", "GAP_ALIGNED", "NARROW_OR", "NO_EXHAUST", "IMPULSE")
GAP_ALIGNED_MIN = 0.0015
NARROW_LOOKBACK, NARROW_PCTL = 60, 40
EXHAUST_LOOKBACK = 20
FADE_OR_BARS = 3                     # 09:15-09:30
FADE_GAPS = (0.003, 0.005)
FADE_MODES = ("RAW", "REJ", "BRK")
FADE_R_MIN, FADE_R_MAX = 1.0, 6.0


def orb_name(n: int, stop_kind: str, r: float, flt: str) -> str:
    return f"ORB{n}/{stop_kind}/{r}R/{flt}"


def fade_name(mode: str, thr: float) -> str:
    return f"FADE/{mode}/gap>={thr * 100:.1f}%"


ALL_STRATEGIES = tuple(
    [orb_name(n, s, r, f) for n, s, r, f in itertools.product(OR_LENS, STOPS, TARGETS, FILTERS)]
    + [fade_name(m, thr) for thr, m in itertools.product(FADE_GAPS, FADE_MODES)]
)
N_TRIALS = len(ALL_STRATEGIES)
PRIOR_TRIALS = 50 + 86               # earlier configurations run on the same data (see docstring)
DSR_N = N_TRIALS + PRIOR_TRIALS
OOS_DISCLOSURE = ("OOS window up to 2026-11-06 was already inspected by earlier versions "
                  "(50-variant run 2026-11-07, 86-variant commit e9edfa9); not a clean holdout. "
                  "Only a forward test after 2026-11-06 is a true one-time OOS.")

# ── split / selection / validation ──────────────────────────────────────────
SELECT_INDICES = ("NIFTY50", "BANKNIFTY")
HOLDOUT_INDEX = "SENSEX"
TRAIN_FRAC = 0.70
MIN_TRAIN, MIN_OOS, TOP_K, DSR_MIN = 80, 30, 5, 0.95
RANK_METRIC = "train t-stat of the mean daily equal-weight net R at the base cost (ties: name)"


# ════════════════════════════════════════════════════════════════════════════
# data
# ════════════════════════════════════════════════════════════════════════════
def load_daily(db: Path, index_id: str) -> dict[date, dict]:
    """``1d`` bars keyed by session date: official open/high/low/close."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT timestamp, open, high, low, close FROM price_data "
        "WHERE index_id = ? AND timeframe = '1d' ORDER BY timestamp", (index_id,)
    ).fetchall()
    con.close()
    out: dict[date, dict] = {}
    for ts, o, h, lo, c in rows:
        try:
            d = datetime.fromisoformat(ts)
        except (TypeError, ValueError):
            continue
        d = (d.astimezone(_IST) if d.tzinfo else d).date()
        out[d] = {"open": float(o), "high": float(h), "low": float(lo), "close": float(c)}
    return dict(sorted(out.items()))


def sessions_by_day(bars: list[dict]) -> dict[date, list[dict]]:
    out: dict[date, list[dict]] = defaultdict(list)
    for b in bars:
        out[b["timestamp"].date()].append(b)
    for d in out:
        out[d].sort(key=lambda b: b["timestamp"])
    return dict(sorted(out.items()))


def morning_bars(day_bars: list[dict]) -> list[dict] | None:
    """The 33 contiguous 5-minute bars 09:15 ... 11:55, or None if any is missing."""
    mb = [b for b in day_bars if b["timestamp"].time() < EXIT_BEFORE]
    if len(mb) != MORNING_BARS or mb[0]["timestamp"].time() != SESSION_OPEN:
        return None
    for k in range(1, len(mb)):
        if (mb[k]["timestamp"] - mb[k - 1]["timestamp"]).total_seconds() != BAR_SECONDS:
            return None
    return mb


def previous_close(daily: dict[date, dict], d: date) -> float | None:
    """Official close of the previous trading day, if it is not suspiciously far back."""
    prev = [p for p in daily if p < d]
    if not prev:
        return None
    p = prev[-1]
    if (d - p).days > MAX_PREV_GAP_DAYS:
        return None
    return daily[p]["close"]


# ════════════════════════════════════════════════════════════════════════════
# execution
# ════════════════════════════════════════════════════════════════════════════
def simulate(mb: list[dict], entry_i: int, direction: int, stop: float, target: float) -> dict:
    """Walk bars from the entry bar until a stop, a target, or the last bar before 12:00.

    Returns ``{"r", "exit_i", "reason", "hold_bars", "entry", "stop_d"}`` where
    ``r`` is the gross R multiple and reason is one of stop, stop_gap, target,
    target_gap, time.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    entry = mb[entry_i]["open"]
    stop_d = abs(entry - stop)
    if stop_d <= 0:
        raise ValueError("stop distance must be positive")
    up = direction > 0
    if (up and not (stop < entry < target)) or (not up and not (target < entry < stop)):
        raise ValueError("need stop < entry < target (long) or target < entry < stop (short)")
    target_r = abs(target - entry) / stop_d

    def done(r: float, k: int, reason: str) -> dict:
        return {"r": float(r), "exit_i": k, "reason": reason, "hold_bars": k - entry_i + 1,
                "entry": entry, "stop_d": stop_d}

    last = None
    for k in range(entry_i, len(mb)):
        b = mb[k]
        if b["timestamp"].time() >= EXIT_BEFORE:
            break
        last = k
        if k > entry_i:
            o = b["open"]
            if (o <= stop) if up else (o >= stop):
                return done((o - entry) * direction / stop_d, k, "stop_gap")
            if (o >= target) if up else (o <= target):
                return done(target_r, k, "target_gap")
        hit_stop = b["low"] <= stop if up else b["high"] >= stop
        hit_target = b["high"] >= target if up else b["low"] <= target
        if hit_stop:
            return done(-1.0, k, "stop")
        if hit_target:
            return done(target_r, k, "target")
    if last is None:
        raise ValueError("entry bar is at or after the 12:00 exit")
    return done((mb[last]["close"] - entry) * direction / stop_d, last, "time")


def cost_r(entry: float, stop_d: float, hold_bars: int, cost_frac: float) -> float:
    """Round-trip cost plus theta allowance, in R."""
    hours = hold_bars * BAR_SECONDS / 3600.0
    return (cost_frac + THETA_FRAC_PER_HOUR * hours) * entry / stop_d


def _stop_ok(entry: float, stop_d: float) -> bool:
    return MIN_STOP_PCT * entry <= stop_d <= MAX_STOP_PCT * entry


# ════════════════════════════════════════════════════════════════════════════
# trade generation
# ════════════════════════════════════════════════════════════════════════════
def build_trades(index_id: str, bars: list[dict], daily: dict[date, dict]) -> tuple[list[dict], Counter]:
    """All strategy trades of one index.  At most one per (day, variant); raises otherwise."""
    trades: list[dict] = []
    diag: Counter = Counter()
    seen: set[tuple[str, str, str]] = set()
    width_hist: dict[int, list[float]] = {n: [] for n in OR_LENS}
    range_hist: list[float] = []

    def add(d: date, name: str, direction: int, sim: dict) -> None:
        key = (index_id, d.isoformat(), name)
        if key in seen:
            raise AssertionError(f"second trade for {key}")
        seen.add(key)
        trades.append({"index_id": index_id, "day": d.isoformat(), "strategy": name, "dir": direction,
                       "r": sim["r"], "entry": sim["entry"], "stop_d": sim["stop_d"],
                       "hold_bars": sim["hold_bars"], "reason": sim["reason"]})

    for d, day in sessions_by_day(bars).items():
        mb = morning_bars(day)
        # full-day range enters the NO_EXHAUST history only AFTER today's decisions
        day_range = (max(b["high"] for b in day) - min(b["low"] for b in day)) / day[0]["open"]
        if mb is None:
            diag["day_incomplete_morning"] += 1
            range_hist.append(day_range)
            continue
        diag["day_usable"] += 1
        day_open = mb[0]["open"]
        pc = previous_close(daily, d)
        gap = (day_open - pc) / pc if pc else None
        if gap is None:
            diag["day_no_prev_close"] += 1
        exhaust_ref = (float(np.median(range_hist[-EXHAUST_LOOKBACK:]))
                       if len(range_hist) >= EXHAUST_LOOKBACK else None)

        # ── opening-range breakout ────────────────────────────────────────
        for n in OR_LENS:
            nb = n // 5
            orb = mb[:nb]
            hi, lo = max(b["high"] for b in orb), min(b["low"] for b in orb)
            width = (hi - lo) / day_open
            hist = width_hist[n][-NARROW_LOOKBACK:]
            narrow = len(hist) >= NARROW_LOOKBACK and width <= float(np.percentile(hist, NARROW_PCTL))
            width_hist[n].append(width)
            no_exhaust = gap is not None and exhaust_ref is not None and abs(gap) + width <= exhaust_ref

            sig = None
            for j in range(nb, len(mb) - 1):
                if mb[j]["timestamp"].time() > SIGNAL_CUTOFF:
                    break
                if mb[j]["close"] > hi:
                    sig = (j, 1)
                    break
                if mb[j]["close"] < lo:
                    sig = (j, -1)
                    break
            if sig is None:
                diag[f"orb{n}_no_signal"] += 1
                continue
            j, direction = sig
            sb = mb[j]
            sb_rng = sb["high"] - sb["low"]
            close_loc = (((sb["close"] - sb["low"]) if direction > 0 else (sb["high"] - sb["close"])) / sb_rng
                         if sb_rng > 0 else 0.0)
            orb_vol = float(np.mean([b.get("volume", 0.0) for b in orb]))
            orb_rng = float(np.mean([b["high"] - b["low"] for b in orb]))
            impulse = ((orb_vol > 0 and float(sb.get("volume", 0.0)) >= 1.5 * orb_vol)
                       or (orb_rng > 0 and sb_rng >= 1.3 * orb_rng and close_loc >= 0.70))
            entry_i = j + 1
            entry = mb[entry_i]["open"]
            for stop_kind, r_mult, flt in itertools.product(STOPS, TARGETS, FILTERS):
                if flt == "GAP_ALIGNED" and not (gap is not None and direction * gap >= GAP_ALIGNED_MIN):
                    continue
                if (flt == "NARROW_OR" and not narrow) or (flt == "NO_EXHAUST" and not no_exhaust) \
                        or (flt == "IMPULSE" and not impulse):
                    continue
                stop = (lo if direction > 0 else hi) if stop_kind == "OPP" else (hi + lo) / 2
                if (direction > 0 and stop >= entry) or (direction < 0 and stop <= entry):
                    diag["orb_stop_wrong_side"] += 1
                    continue
                stop_d = abs(entry - stop)
                if not _stop_ok(entry, stop_d):
                    diag["orb_stop_out_of_range"] += 1
                    continue
                target = entry + direction * r_mult * stop_d
                add(d, orb_name(n, stop_kind, r_mult, flt), direction,
                    simulate(mb, entry_i, direction, stop, target))

        # ── gap fade ──────────────────────────────────────────────────────
        if gap is not None and abs(gap) >= min(FADE_GAPS):
            direction = -1 if gap > 0 else 1
            orb = mb[:FADE_OR_BARS]
            rejected = orb[-1]["close"] < day_open if direction < 0 else orb[-1]["close"] > day_open
            first_ext = orb[0]["low"] if direction < 0 else orb[0]["high"]
            for thr, mode in itertools.product(FADE_GAPS, FADE_MODES):
                if abs(gap) < thr:
                    continue
                entry_i = None
                if mode == "RAW" or (mode == "REJ" and rejected):
                    entry_i = FADE_OR_BARS
                elif mode == "BRK":
                    for j in range(FADE_OR_BARS - 1, len(mb) - 1):
                        if mb[j]["timestamp"].time() > SIGNAL_CUTOFF:
                            break
                        if (mb[j]["close"] < first_ext) if direction < 0 else (mb[j]["close"] > first_ext):
                            entry_i = j + 1
                            break
                if entry_i is None:
                    continue
                assert mb[entry_i]["timestamp"].time() >= FADE_ENTRY_FROM
                pre = mb[:entry_i]
                touched = (min(b["low"] for b in pre) <= pc) if direction < 0 else (max(b["high"] for b in pre) >= pc)
                if touched:
                    diag["fade_prev_close_touched"] += 1
                    continue
                entry = mb[entry_i]["open"]
                if (direction < 0 and entry <= pc) or (direction > 0 and entry >= pc):
                    diag["fade_entry_through_prev_close"] += 1
                    continue
                stop = max(b["high"] for b in pre) if direction < 0 else min(b["low"] for b in pre)
                if (direction < 0 and stop <= entry) or (direction > 0 and stop >= entry):
                    diag["fade_stop_wrong_side"] += 1
                    continue
                stop_d = abs(entry - stop)
                if not _stop_ok(entry, stop_d):
                    diag["fade_stop_out_of_range"] += 1
                    continue
                fade_r = abs(entry - pc) / stop_d
                if not (FADE_R_MIN <= fade_r <= FADE_R_MAX):
                    diag["fade_r_out_of_range"] += 1
                    continue
                add(d, fade_name(mode, thr), direction, simulate(mb, entry_i, direction, stop, pc))

        range_hist.append(day_range)

    unknown = {t["strategy"] for t in trades} - set(ALL_STRATEGIES)
    if unknown:
        raise AssertionError(f"strategies outside the trial list: {sorted(unknown)}")
    return trades, diag


# ════════════════════════════════════════════════════════════════════════════
# split / statistics
# ════════════════════════════════════════════════════════════════════════════
def split_dates(dates, train_frac: float = TRAIN_FRAC) -> tuple[set[str], set[str], str]:
    """Chronological split of unique trading dates: (train, oos, first OOS date)."""
    uniq = sorted({str(d) for d in dates})
    if len(uniq) < 2:
        raise ValueError("need at least two trading dates")
    k = int(len(uniq) * train_frac)
    k = min(max(k, 1), len(uniq) - 1)
    return set(uniq[:k]), set(uniq[k:]), uniq[k]


def daily_series(rows: list[dict], cost_frac: float, net: bool = True) -> np.ndarray:
    """One equal-weight return per active day (same-day trades of several indices averaged)."""
    by_d: dict[str, list[float]] = defaultdict(list)
    for x in rows:
        r = x["r"] - (cost_r(x["entry"], x["stop_d"], x["hold_bars"], cost_frac) if net else 0.0)
        by_d[x["day"]].append(r)
    return np.array([float(np.mean(by_d[d])) for d in sorted(by_d)], dtype=float)


def max_drawdown(series: np.ndarray) -> float:
    """Largest peak-to-trough fall of the cumulative sum (R), starting from 0."""
    if series.size == 0:
        return 0.0
    cum = np.concatenate([[0.0], np.cumsum(series)])
    return float(np.max(np.maximum.accumulate(cum) - cum))


def stats(rows: list[dict], cost_frac: float) -> dict:
    net = daily_series(rows, cost_frac)
    gross = daily_series(rows, cost_frac, net=False)
    n = int(net.size)
    out = {"n_days": n, "n_trades": len(rows), "gross": None, "net": None, "t": 0.0, "sr": 0.0,
           "hit": None, "max_dd": max_drawdown(net), "total_net": float(net.sum()) if n else 0.0}
    if n >= 1:
        out.update(gross=float(gross.mean()), net=float(net.mean()),
                   hit=float(np.mean([x["r"] - cost_r(x["entry"], x["stop_d"], x["hold_bars"], cost_frac) > 0
                                      for x in rows])))
    if n >= 2:
        sd = float(np.std(net, ddof=1))
        out.update(t=float(net.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0, sr=sharpe_ratio(net))
    return out


def dsr_block(series: np.ndarray, trial_srs: list[float], n_trials: int = DSR_N) -> dict:
    n = int(series.size)
    sk, ku = return_moments(series)
    res = deflate_selected_sharpe(sharpe_ratio(series), n, trial_srs, n_trials=n_trials,
                                  skew=sk, kurtosis=ku, threshold=DSR_MIN, assumed_normal=n < 3)
    mintrl = res.min_track_record_length
    return {"sr_per_active_day": res.sharpe, "n_obs": res.n_obs,
            "n_trials": res.n_trials, "trial_sr_variance": res.sr_trials_variance,
            "expected_max_sr": res.expected_max_sharpe, "skew": res.skew, "kurtosis": res.kurtosis,
            "psr": res.psr, "dsr": res.dsr, "min_trl": mintrl if np.isfinite(mintrl) else None,
            "passes": res.passes}


# ════════════════════════════════════════════════════════════════════════════
# selection (train only) and freeze
# ════════════════════════════════════════════════════════════════════════════
def train_table(train_rows: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for x in train_rows:
        by[x["strategy"]].append(x)
    return {nm: stats(by.get(nm, []), BASE_COST_FRAC) for nm in ALL_STRATEGIES}


def select_picks(table: dict[str, dict]) -> list[str]:
    """Top TOP_K variants by RANK_METRIC among those with >= MIN_TRAIN train days."""
    elig = [nm for nm, s in table.items() if s["n_days"] >= MIN_TRAIN]
    elig.sort(key=lambda nm: (-table[nm]["t"], nm))
    return elig[:TOP_K]


def trial_sharpes(table: dict[str, dict]) -> list[float]:
    return [s["sr"] for s in table.values() if s["n_days"] >= MIN_TRAIN]


def freeze(picks: list[str], table: dict[str, dict], cut: str) -> dict:
    body = {"picks": picks, "rank_metric": RANK_METRIC, "first_oos_date": cut, "top_k": TOP_K,
            "base_cost_frac": BASE_COST_FRAC, "n_trials": N_TRIALS, "dsr_n": DSR_N,
            "train": {nm: table[nm] for nm in picks}}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def validate(train: dict, oos: dict, oos_stress: dict, oos_dsr: dict, sensex: dict) -> dict:
    checks = {
        "train_net_positive": train["n_days"] >= MIN_TRAIN and (train["net"] or -9) > 0,
        "oos_days_enough": oos["n_days"] >= MIN_OOS,
        "oos_net_positive_base": (oos["net"] or -9) > 0,
        "oos_net_positive_stress_0.05pct": (oos_stress["net"] or -9) > 0,
        "oos_dsr_ok": oos["n_days"] >= 3 and oos_dsr["dsr"] >= DSR_MIN,
        "sensex_oos_net_positive": sensex["n_days"] >= MIN_OOS and (sensex["net"] or -9) > 0,
    }
    return {"checks": checks, "validated": all(checks.values())}


def evaluate_frozen(frozen: dict, trades: list[dict], train_days: set[str], oos_days: set[str],
                    trial_srs: list[float]) -> list[dict]:
    out = []
    for nm in frozen["picks"]:
        rows = [x for x in trades if x["strategy"] == nm]
        sel = [x for x in rows if x["index_id"] in SELECT_INDICES]
        tr = [x for x in sel if x["day"] in train_days]
        oo = [x for x in sel if x["day"] in oos_days]
        sx = [x for x in rows if x["index_id"] == HOLDOUT_INDEX]
        sx_oo = [x for x in sx if x["day"] in oos_days]
        res = {"name": nm, "by_cost": {}}
        for c in COST_SCENARIOS:
            res["by_cost"][f"{c * 100:.3f}%"] = {
                "train": stats(tr, c), "oos": stats(oo, c),
                "sensex_all": stats(sx, c), "sensex_oos": stats(sx_oo, c)}
        res["dsr_is"] = dsr_block(daily_series(tr, BASE_COST_FRAC), trial_srs)
        res["dsr_oos"] = dsr_block(daily_series(oo, BASE_COST_FRAC), trial_srs)
        base = res["by_cost"][f"{BASE_COST_FRAC * 100:.3f}%"]
        res["validation"] = validate(base["train"], base["oos"], stats(oo, VALIDATION_STRESS_COST),
                                     res["dsr_oos"], base["sensex_oos"])
        res["exit_reasons_oos"] = dict(Counter(x["reason"] for x in oo))
        out.append(res)
    return out


# ════════════════════════════════════════════════════════════════════════════
# report
# ════════════════════════════════════════════════════════════════════════════
def _f(v, p: int = 3) -> str:
    return "-" if v is None else f"{v:+.{p}f}"


def markdown(report: dict) -> str:
    lines = [
        "# Morning strategy search", "",
        f"Data {report['first_date']} → {report['last_date']} · train {report['n_train_days']} days · "
        f"OOS {report['n_oos_days']} days from {report['first_oos_date']} · {report['n_trials']} variants "
        f"(DSR N = {report['dsr_n']} incl. {PRIOR_TRIALS} earlier configurations)",
        f"**Caveat:** {OOS_DISCLOSURE}",
        f"Cost = round trip {', '.join(f'{c * 100:.3f}%' for c in COST_SCENARIOS)} of spot (first = base) "
        f"+ theta {THETA_FRAC_PER_HOUR * 100:.3f}% of spot per hour held.",
        f"Frozen picks (sha256 {report['frozen']['sha256'][:12]}…), ranked by {RANK_METRIC}.", "",
    ]
    if not report["picks"]:
        lines += ["No variant had enough train days to be selected.", ""]
    for p in report["picks"]:
        lines += [f"## {p['name']} — {'VALIDATED' if p['validation']['validated'] else 'not validated'}", "",
                  "| cost | set | days | trades | gross R | net R | t | net hit | max DD R |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c, blk in p["by_cost"].items():
            for k in ("train", "oos", "sensex_all", "sensex_oos"):
                s = blk[k]
                hit = "-" if s["hit"] is None else f"{s['hit']:.2f}"
                lines.append(f"| {c} | {k} | {s['n_days']} | {s['n_trades']} | {_f(s['gross'])} | {_f(s['net'])} | "
                             f"{s['t']:+.2f} | {hit} | {s['max_dd']:.2f} |")
        lines.append("")
        for k in ("dsr_is", "dsr_oos"):
            d = p[k]
            lines.append(f"- {k}: SR {d['sr_per_active_day']:+.3f}/active day (n={d['n_obs']}, skew {d['skew']:+.2f}, "
                         f"kurt {d['kurtosis']:.2f}) · N={d['n_trials']} V={d['trial_sr_variance']:.4f} "
                         f"SR0={d['expected_max_sr']:.3f} · PSR {d['psr']:.2f} · DSR {d['dsr']:.2f}")
        lines.append("- checks: " + ", ".join(f"{k} {'✓' if v else '✗'}"
                                              for k, v in p["validation"]["checks"].items()))
        lines += [f"- OOS exits: {p['exit_reasons_oos']}",
                  "- sensex_all overlaps the train dates: context only, not a validation input.", ""]
    lines += ["## Train-only context (base cost; unselected variants get no OOS numbers)", "",
              "| family | variants | mean train net R | positive |", "|---|---:|---:|---:|"]
    for k, v in report["train_families"].items():
        lines.append(f"| {k} | {v['n']} | {v['mean']:+.3f} | {v['positive']} |")
    lines += ["", "Diagnostics: " + "; ".join(f"{i}: {dict(sorted(d.items()))}"
                                              for i, d in report["diagnostics"].items())]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, required=True, help="a COPY of the price database (read-only)")
    ap.add_argument("--out", type=Path, required=True, help="JSON report; frozen picks go next to it")
    ap.add_argument("--md", type=Path, default=None, help="optional markdown report")
    args = ap.parse_args()

    trades: list[dict] = []
    diagnostics: dict[str, dict] = {}
    calendar: set[str] = set()
    for idx in (*SELECT_INDICES, HOLDOUT_INDEX):
        daily = load_daily(args.db, idx)
        calendar |= {d.isoformat() for d in daily}
        t, diag = build_trades(idx, cal.load_bars(args.db, idx), daily)
        diagnostics[idx] = dict(diag)
        print(f"{idx}: {len(daily)} trading days, {diag['day_usable']} usable mornings, {len(t)} strategy-trades")
        trades += t

    train_days, oos_days, cut = split_dates(calendar)
    print(f"\n{len(calendar)} trading dates · train {len(train_days)} · OOS {len(oos_days)} from {cut}")

    # ── selection: train rows of the selection indices ONLY ──
    train_rows = [x for x in trades if x["index_id"] in SELECT_INDICES and x["day"] in train_days]
    table = train_table(train_rows)
    srs = trial_sharpes(table)
    picks = select_picks(table)
    frozen = freeze(picks, table, cut)
    frozen_path = args.out.with_name(args.out.stem + ".frozen_picks.json")
    frozen_path.write_text(json.dumps(frozen, indent=2))
    print(f"frozen {len(picks)} picks before OOS → {frozen_path} (sha256 {frozen['sha256'][:12]}…)")
    print(f"DSR: N={DSR_N} ({N_TRIALS} variants + {PRIOR_TRIALS} earlier), trial-SR variance "
          f"{float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0:.5f} over {len(srs)} variants "
          f"with >= {MIN_TRAIN} train days")

    # ── one-time evaluation of the frozen picks ──
    evaluated = evaluate_frozen(frozen, trades, train_days, oos_days, srs)

    fam: dict[str, list[float]] = defaultdict(list)
    for nm, s in table.items():
        if s["n_days"] >= 15 and s["net"] is not None:
            parts = nm.split("/")
            fam[f"{parts[0]}/{parts[1]}" if parts[0] == "FADE" else f"{parts[0]}/{parts[-1]}"].append(s["net"])
    families = {k: {"n": len(v), "mean": float(np.mean(v)), "positive": f"{sum(x > 0 for x in v)}/{len(v)}"}
                for k, v in sorted(fam.items())}

    report = {"generated_at": datetime.now().isoformat(), "db": str(args.db),
              "first_date": min(calendar), "last_date": max(calendar), "first_oos_date": cut,
              "n_train_days": len(train_days), "n_oos_days": len(oos_days), "n_trials": N_TRIALS,
              "dsr_n": DSR_N, "prior_trials": PRIOR_TRIALS, "oos_disclosure": OOS_DISCLOSURE,
              "cost_scenarios": COST_SCENARIOS, "theta_frac_per_hour": THETA_FRAC_PER_HOUR,
              "frozen": frozen, "picks": evaluated, "train_table": table,
              "train_families": families, "diagnostics": diagnostics}
    args.out.write_text(json.dumps(report, indent=2, default=str, allow_nan=False))
    md = markdown(report)
    if args.md:
        args.md.write_text(md)
    print("\n" + md)
    n_ok = sum(p["validation"]["validated"] for p in evaluated)
    print(f"{n_ok}/{len(evaluated)} frozen picks validated.  Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
