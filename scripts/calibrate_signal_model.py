#!/usr/bin/env python3
"""
Calibrate the signal model: map |score| → P(win), validated out of sample.

Replays the LIVE scoring path bar by bar on stored 5-minute bars:

    TechnicalAggregator.analyze  (300-bar window, 15m bars resampled for MTF)
    RegimeDetector.detect_regime (as_of = bar time; hysteresis state carried)
    SignalGenerator.model_score  (the exact function the live engine uses)

Each tradeable bar (09:30-14:25 IST, regime safe, |score| above the dead
zone) is labelled by what happened next, as an option buyer experiences it:

    entry  = next bar's open
    win    = price reaches entry ± TARGET_ATR × ATR before entry ∓ STOP_ATR × ATR
             within HORIZON bars of the same session
    loss   = stop first, both in the same bar, or neither by the horizon

Fitting: bins of |score| with Beta(2,2) shrinkage and a monotone (PAV) fit on
the chronologically FIRST 70 % of samples.  The last 30 % is out-of-sample:
the model is marked ``validated`` only if, out of sample,

  * at least --min-oos-trades samples fall in tradeable bins (P ≥ 0.50),
  * their realised win rate is ≥ --min-oos-winrate (break-even is 0.40 at
    1.5:1 before costs), and
  * the Brier score beats the constant base-rate forecast.

Options/OI and anomaly inputs have no point-in-time history and are not part
of the score (live, they can only veto or haircut).

Usage:
    python scripts/calibrate_signal_model.py                  # write config/signal_calibration.json
    python scripts/calibrate_signal_model.py --report-only    # print, write nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.analysis.technical_aggregator import TechnicalAggregator  # noqa: E402
from src.engine.decision_engine import resample_session_bars  # noqa: E402
from src.engine.regime_detector import RegimeDetector  # noqa: E402
from src.engine.signal_calibration import (  # noqa: E402
    CALIBRATION_VERSION, DEFAULT_PATH, POOLED_KEY, brier, fit_bins,
)
from src.engine.signal_generator import SignalGenerator  # noqa: E402

_IST = ZoneInfo("Asia/Kolkata")
WINDOW = 300
HORIZON = 12          # 5m bars = 60 minutes
TARGET_ATR = 1.5
STOP_ATR = 1.0
FIRST_ENTRY = dtime(9, 30)
LAST_ENTRY = dtime(14, 25)
EDGES = [0.25, 0.40, 0.55, 0.70, 0.90, 1.20, 99.0]
TRADEABLE_P = 0.50


def load_bars(db_path: Path, index_id: str) -> list[dict]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT timestamp, open, high, low, close, volume FROM price_data "
        "WHERE index_id = ? AND timeframe = '5m' ORDER BY timestamp ASC",
        (index_id,),
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["timestamp"])
        except (TypeError, ValueError):
            continue
        ts = ts.replace(tzinfo=_IST) if ts.tzinfo is None else ts.astimezone(_IST)
        # session bars only (some stores contain flat post-market rows)
        if not (dtime(9, 15) <= ts.time() <= dtime(15, 30)):
            continue
        out.append({"timestamp": ts, "open": float(r["open"]), "high": float(r["high"]),
                    "low": float(r["low"]), "close": float(r["close"]),
                    "volume": float(r["volume"] or 0)})
    return out


def label(bars: list[dict], t: int, direction: str, atr: float) -> int | None:
    """1 = target first, 0 = stop first / both / timeout, None = not labelable."""
    if t + HORIZON >= len(bars):
        return None
    day = bars[t]["timestamp"].date()
    path = bars[t + 1: t + 1 + HORIZON]
    if any(b["timestamp"].date() != day for b in path):
        return None
    entry = path[0]["open"]
    up = direction == "BUY_CALL"
    target = entry + TARGET_ATR * atr if up else entry - TARGET_ATR * atr
    stop = entry - STOP_ATR * atr if up else entry + STOP_ATR * atr
    for b in path:
        hit_stop = b["low"] <= stop if up else b["high"] >= stop
        hit_target = b["high"] >= target if up else b["low"] <= target
        if hit_stop:
            return 0          # includes "both in one bar" — conservative
        if hit_target:
            return 1
    return 0                  # timeout: an option buyer bled theta


def replay(index_id: str, bars: list[dict], stride: int) -> list[dict]:
    agg = TechnicalAggregator()
    rd = RegimeDetector(db=None)  # the detector does not query the DB
    samples: list[dict] = []
    for t in range(WINDOW - 1, len(bars) - 1, stride):
        ts = bars[t]["timestamp"]
        if not (FIRST_ENTRY <= ts.time() <= LAST_ENTRY):
            continue
        df = pd.DataFrame(bars[t - WINDOW + 1: t + 1])
        try:
            tech = agg.analyze(
                index_id=index_id, price_df=df, timeframe="5m",
                htf_df=resample_session_bars(df, "15min"),
            )
            regime = rd.detect_regime(index_id, df, tech, as_of=ts)
        except Exception as exc:  # one bad window must not end the run
            logging.getLogger(__name__).debug("skip %s: %s", ts, exc)
            continue
        safe, _ = RegimeDetector.is_safe_to_trade(regime)
        if not safe:
            continue
        score, _, comps = SignalGenerator.model_score(tech, regime)
        direction = SignalGenerator._determine_direction(score, regime.regime)
        if direction == "NO_TRADE":
            continue
        atr = float(tech.volatility.atr_value or 0)
        if not math.isfinite(atr) or atr <= 0:
            continue
        y = label(bars, t, direction, atr)
        if y is None:
            continue
        samples.append({
            "ts": ts.isoformat(), "index_id": index_id, "score": round(score, 4),
            "abs_score": round(abs(score), 4), "direction": direction, "win": y,
            "regime": regime.regime, "components": sorted(comps),
        })
    return samples


def fit_and_validate(samples: list[dict], args) -> dict:
    samples = sorted(samples, key=lambda s: s["ts"])
    n = len(samples)
    cut = int(n * 0.7)
    train, test = samples[:cut], samples[cut:]
    bins = fit_bins([s["abs_score"] for s in train], [s["win"] for s in train], EDGES)

    def p_of(x: float) -> float:
        for b in bins:
            if b.lo <= x < b.hi:
                return b.p_win
        return bins[-1].p_win

    base = sum(s["win"] for s in train) / len(train) if train else 0.5
    preds = [p_of(s["abs_score"]) for s in test]
    ys = [s["win"] for s in test]
    b_model, b_base = brier(preds, ys), brier([base] * len(ys), ys)
    tradeable = [y for p, y in zip(preds, ys) if p >= TRADEABLE_P]
    oos_wr = (sum(tradeable) / len(tradeable)) if tradeable else float("nan")
    validated = (
        len(tradeable) >= args.min_oos_trades
        and oos_wr >= args.min_oos_winrate
        and b_model <= b_base
    )
    if validated:
        note = f"OOS {len(tradeable)} tradeable, win {oos_wr:.2f}, Brier {b_model:.4f} ≤ base {b_base:.4f}"
    else:
        parts = []
        if len(tradeable) < args.min_oos_trades:
            parts.append(f"only {len(tradeable)} OOS tradeable samples (< {args.min_oos_trades})")
        if not (oos_wr >= args.min_oos_winrate):
            parts.append(f"OOS win rate {oos_wr:.2f} < {args.min_oos_winrate}")
        if b_model > b_base:
            parts.append(f"Brier {b_model:.4f} worse than base-rate {b_base:.4f}")
        note = "; ".join(parts)

    oos_by_bin = []
    for b in bins:
        ys_b = [s["win"] for s in test if b.lo <= s["abs_score"] < b.hi]
        oos_by_bin.append({"lo": b.lo, "hi": b.hi, "fit_p": b.p_win, "train_n": b.n,
                           "oos_n": len(ys_b),
                           "oos_win": round(sum(ys_b) / len(ys_b), 3) if ys_b else None})
    by_dir = {}
    for d in ("BUY_CALL", "BUY_PUT"):
        ys_d = [s["win"] for s in samples if s["direction"] == d]
        by_dir[d] = {"n": len(ys_d), "win": round(sum(ys_d) / len(ys_d), 3) if ys_d else None}
    return {
        "bins": [b.__dict__ for b in bins],
        "validated": validated,
        "meta": {
            "validation_note": note,
            "n_samples": n, "train_n": len(train), "test_n": len(test),
            "train_base_rate": round(base, 4),
            "oos_brier": round(b_model, 5), "oos_brier_base": round(b_base, 5),
            "oos_tradeable_n": len(tradeable),
            "oos_tradeable_win_rate": None if math.isnan(oos_wr) else round(oos_wr, 4),
            "oos_by_bin": oos_by_bin, "by_direction": by_dir,
            "first_ts": samples[0]["ts"] if samples else None,
            "last_ts": samples[-1]["ts"] if samples else None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "db" / "trading_dss.db")
    ap.add_argument("--indices", default="NIFTY50,BANKNIFTY")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth bar")
    ap.add_argument("--min-oos-trades", type=int, default=30)
    ap.add_argument("--min-oos-winrate", type=float, default=0.45)
    ap.add_argument("--min-index-samples", type=int, default=300)
    ap.add_argument("--output", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    all_samples: list[dict] = []
    models: dict = {}
    for index_id in [i.strip().upper() for i in args.indices.split(",") if i.strip()]:
        bars = load_bars(args.db, index_id)
        t0 = time.time()
        samples = replay(index_id, bars, args.stride) if len(bars) > WINDOW + HORIZON else []
        print(f"{index_id}: {len(bars)} bars → {len(samples)} labelled samples "
              f"({time.time() - t0:.0f}s)")
        all_samples.extend(samples)
        if len(samples) >= args.min_index_samples:
            models[index_id] = fit_and_validate(samples, args)

    if not all_samples:
        print("No samples — nothing to calibrate.")
        return 1
    models[POOLED_KEY] = fit_and_validate(all_samples, args)

    for key, m in models.items():
        meta = m["meta"]
        print(f"\n[{key}] validated={m['validated']}  {meta['validation_note']}")
        print(f"  samples {meta['n_samples']} (train {meta['train_n']} / OOS {meta['test_n']}), "
              f"train base rate {meta['train_base_rate']:.3f}")
        for row in meta["oos_by_bin"]:
            print(f"  |score| [{row['lo']:.2f},{row['hi']:.2f}): fit P={row['fit_p']:.3f} "
                  f"(train n={row['train_n']})  OOS n={row['oos_n']} win={row['oos_win']}")
        print(f"  by direction: {meta['by_direction']}")

    payload = {
        "version": CALIBRATION_VERSION,
        "generated_at": datetime.now(tz=_IST).isoformat(),
        "label": {"horizon_bars": HORIZON, "bar": "5m", "target_atr": TARGET_ATR,
                  "stop_atr": STOP_ATR, "entry": "next bar open",
                  "window_bars": WINDOW, "entry_window": f"{FIRST_ENTRY}-{LAST_ENTRY}"},
        "models": models,
    }
    if args.report_only:
        print("\n--report-only: nothing written")
    else:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
