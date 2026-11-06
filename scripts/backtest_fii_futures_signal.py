"""
EXPERIMENTAL backtest of FII index-futures positioning as a NIFTY signal.
Read-only: changes no data, and nothing here touches the live engine.

    .venv/bin/python scripts/backtest_fii_futures_signal.py [--db PATH] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from scripts.backfill_fii_futures import DEFAULT_CACHE  # noqa: E402
from src.analysis.experimental import fii_futures_signal as ex  # noqa: E402
from src.database.db_manager import DatabaseManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()

    db = DatabaseManager(db_path=args.db or settings.database.path)
    db.connect()
    d = ex.add_features_and_targets(ex.load_dataset(db, args.cache_dir))
    pd.set_option("display.width", 160)
    pd.set_option("display.precision", 3)

    usable = d.dropna(subset=["ret_1d_fwd"])
    print(f"Sessions with positions + NIFTY close: {len(d)} "
          f"({d.index.min().date()} → {d.index.max().date()}); with next-day return: {len(usable)}\n")

    print("== Proposed ±40 rule: how often does it fire? ==")
    for k, v in ex.boost_distribution(d).items():
        print(f"  {k:<16} {v:,.3f}" if isinstance(v, float) else f"  {k:<16} {v}")

    print("\n== Next-day NIFTY return by sign of proposed boost ==")
    print(ex.conditional_returns(d).to_string())

    print("\n== Information coefficients (Spearman; 5d uses non-overlapping windows) ==")
    for col in ("proposed_boost", "fii_net", "fii_net_chg", "candidate_score", "client_net"):
        for h in (1, 5):
            r = ex.information_coefficient(d, col, h)
            print(f"  {col:<16} h={h}  n={r.n:<4} IC={r.ic:+.3f}  t={r.t_stat:+.2f}  p={r.p_value:.3f}  hit={r.hit_rate:.3f}")

    print("\n== Contemporaneous: does FII positioning *follow* price? (same-session return) ==")
    x = d[["fii_net_chg", "ret_1d_same"]].dropna()
    print(f"  Spearman(fii_net_chg_D, NIFTY return_D) = {x.corr(method='spearman').iloc[0, 1]:+.3f}  (n={len(x)})")

    print("\n== Overlap with OI inputs already used (Spearman) ==")
    print(ex.overlap_matrix(d).to_string())

    print("\n== Toy next-session strategy vs buy & hold (no costs) ==")
    for col in ("proposed_boost", "candidate_score"):
        s = ex.strategy_stats(d, col)
        print(f"  {col:<16} strat Sharpe={s['strategy']['sharpe']:+.2f} ann_ret={s['strategy']['ann_return']:+.3f} "
              f"exposure={s['strategy']['exposure']:.2f} | buy&hold Sharpe={s['buy_and_hold']['sharpe']:+.2f} "
              f"ann_ret={s['buy_and_hold']['ann_return']:+.3f}")

    print("\n== Stability by year (proposed boost, 1-day) ==")
    print(ex.by_year(d, "proposed_boost").to_string(index=False))
    print("\n== Stability by year (candidate score, 1-day) ==")
    print(ex.by_year(d, "candidate_score").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
