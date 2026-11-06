"""
EXPERIMENTAL: FII index-futures positioning as a NIFTY signal.

NOT wired into the decision engine. It exists to evaluate, on history:

* the *proposed* rule already coded in SmartMoneyIndicators.calculate_smart_money_score:
      boost = clip(|net| / 500, 0, 40) · sign(net)   when |net| ≥ 5000 contracts
  which is added to the OI-momentum (OIMI) component;
* candidate alternatives that are stationary (changes / z-scores) and bounded;
* overlap ("double counting") with OI-based inputs the engine already uses.

Point-in-time: positions for session D are published after D's close, so
they are aligned to the return from D's close to D+1's close (and D+h).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ── signal definitions ────────────────────────────────────────────────────────


def proposed_boost(net: float) -> float:
    """Exact replica of the rule in smart_money.calculate_smart_money_score."""
    if net is None or abs(net) < 5000:
        return 0.0
    return min(40.0, abs(net) / 500.0) * (1 if net > 0 else -1)


def candidate_score(d: pd.DataFrame, window: int = 60, cap: float = 15.0) -> pd.Series:
    """Bounded, stationary alternative: z-score of FII long-ratio change vs its own history.

    Uses the *change* in positioning (what FIIs did today), not the level (which
    sits on one side for months). Clipped to ±2σ → ±cap points.
    """
    x = d["fii_long_ratio"].diff()
    z = (x - x.rolling(window, min_periods=window // 2).mean()) / x.rolling(window, min_periods=window // 2).std()
    return (z.clip(-2, 2) / 2 * cap).rename("candidate_score")


# ── data ──────────────────────────────────────────────────────────────────────


def _total_options_oi(cache_dir: Path) -> pd.Series:
    """Market-wide index options OI per session from the cached NSE participant files."""
    import csv
    import io
    from datetime import datetime

    from src.data.fii_dii_data import _TITLE_DATE_RE

    out = {}
    for f in cache_dir.glob("fao_participant_oi_*.csv"):
        text = f.read_text()
        if text.startswith("__404__"):
            continue
        lines = [ln for ln in text.splitlines() if ln.strip()]
        m = _TITLE_DATE_RE.search(lines[0]) if lines else None
        if not m:
            continue
        d = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y").date()
        rows = list(csv.reader(io.StringIO("\n".join(lines[1:]))))
        header = [h.strip() for h in rows[0]]
        tot = next((r for r in rows[1:] if r and r[0].strip().upper() == "TOTAL"), None)
        if tot is None:
            continue
        ci, pi = header.index("Option Index Call Long"), header.index("Option Index Put Long")
        out[pd.Timestamp(d)] = float(tot[ci]) + float(tot[pi])
    return pd.Series(out, name="total_idx_opt_oi").sort_index()


def load_dataset(db, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    pos = pd.DataFrame(db.fetch_all(
        "SELECT trade_date, participant, instrument, long_contracts, short_contracts "
        "FROM fii_futures_positions WHERE instrument = 'INDEX_FUT'"
    ))
    pos["trade_date"] = pd.to_datetime(pos["trade_date"])
    wide = pos.pivot_table(index="trade_date", columns="participant",
                           values=["long_contracts", "short_contracts"], aggfunc="first")
    d = pd.DataFrame(index=wide.index)
    for p in ("FII", "CLIENT", "PRO", "DII"):
        d[f"{p.lower()}_long"] = wide[("long_contracts", p)]
        d[f"{p.lower()}_short"] = wide[("short_contracts", p)]
        d[f"{p.lower()}_net"] = d[f"{p.lower()}_long"] - d[f"{p.lower()}_short"]
    d["total_idx_fut_oi"] = d[[f"{p}_long" for p in ("fii", "client", "pro", "dii")]].sum(axis=1)
    d["fii_long_ratio"] = d["fii_long"] / (d["fii_long"] + d["fii_short"])

    px = pd.DataFrame(db.fetch_all(
        "SELECT substr(timestamp,1,10) AS d, close FROM price_data "
        "WHERE index_id = 'NIFTY50' AND timeframe = '1d' ORDER BY timestamp"
    ))
    px["d"] = pd.to_datetime(px["d"])
    close = px.groupby("d")["close"].last()
    d = d.join(close.rename("nifty_close"), how="inner")
    if cache_dir is not None:
        d = d.join(_total_options_oi(cache_dir), how="left")
    return d.sort_index()


# ── evaluation ────────────────────────────────────────────────────────────────


@dataclass
class ICResult:
    name: str
    horizon: int
    n: int
    ic: float
    t_stat: float
    p_value: float
    hit_rate: float


def add_features_and_targets(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["ret_1d_fwd"] = d["nifty_close"].shift(-1) / d["nifty_close"] - 1
    d["ret_5d_fwd"] = d["nifty_close"].shift(-5) / d["nifty_close"] - 1
    d["ret_1d_same"] = d["nifty_close"].pct_change()  # contemporaneous (same session D)
    d["fii_net_chg"] = d["fii_net"].diff()
    d["proposed_boost"] = d["fii_net"].map(proposed_boost)
    d["candidate_score"] = candidate_score(d)
    # Daily OIMI proxy: same formula as SmartMoney OIMI (0.4·price ROC5 + 0.6·OI ROC5, ×10, clip ±100)
    if "total_idx_opt_oi" in d:
        oimi = (d["nifty_close"].pct_change(5) * 100 * 0.4 + d["total_idx_opt_oi"].pct_change(5) * 100 * 0.6) * 10
        d["oimi_proxy"] = oimi.clip(-100, 100)
    d["price_roc5"] = d["nifty_close"].pct_change(5) * 100
    d["fut_oi_roc5"] = d["total_idx_fut_oi"].pct_change(5) * 100
    return d


def information_coefficient(d: pd.DataFrame, col: str, horizon: int) -> ICResult:
    target = "ret_1d_fwd" if horizon == 1 else "ret_5d_fwd"
    x = d[[col, target]].dropna()
    if horizon > 1:
        x = x.iloc[::horizon]  # non-overlapping windows → honest t-stat
    rho, p = stats.spearmanr(x[col], x[target])
    n = len(x)
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    nz = x[x[col] != 0]
    hit = float((np.sign(nz[col]) == np.sign(nz[target])).mean()) if len(nz) else float("nan")
    return ICResult(col, horizon, n, float(rho), float(t), float(p), hit)


def boost_distribution(d: pd.DataFrame) -> dict:
    b = d["proposed_boost"].dropna()
    return {
        "sessions": int(len(b)),
        "at_+40": float((b == 40).mean()),
        "at_-40": float((b == -40).mean()),
        "zero": float((b == 0).mean()),
        "partial": float(((b != 0) & (b.abs() < 40)).mean()),
        "sign_flips": int((np.sign(b).diff().fillna(0) != 0).sum()),
        "fii_net_min": float(d["fii_net"].min()),
        "fii_net_median": float(d["fii_net"].median()),
        "fii_net_max": float(d["fii_net"].max()),
    }


def conditional_returns(d: pd.DataFrame) -> pd.DataFrame:
    x = d[["proposed_boost", "ret_1d_fwd"]].dropna()
    g = x.groupby(np.sign(x["proposed_boost"]))["ret_1d_fwd"]
    out = g.agg(["count", "mean", "std"])
    out["mean_bp"] = out["mean"] * 1e4
    out["t_stat"] = out["mean"] / (out["std"] / np.sqrt(out["count"]))
    out["up_day_rate"] = g.apply(lambda s: float((s > 0).mean()))
    return out[["count", "mean_bp", "t_stat", "up_day_rate"]]


def strategy_stats(d: pd.DataFrame, col: str) -> dict:
    """Toy: hold NIFTY in the direction of sign(col) for the next session; compare to buy & hold."""
    x = d[[col, "ret_1d_fwd"]].dropna()
    strat = np.sign(x[col]) * x["ret_1d_fwd"]
    bh = x["ret_1d_fwd"]

    def ann(r: pd.Series) -> dict:
        return {"ann_return": float(r.mean() * 252), "ann_vol": float(r.std() * np.sqrt(252)),
                "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
                "exposure": float((r != 0).mean())}

    return {"strategy": ann(strat), "buy_and_hold": ann(bh)}


def overlap_matrix(d: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("proposed_boost", "fii_net", "fii_net_chg", "candidate_score",
                        "oimi_proxy", "price_roc5", "fut_oi_roc5", "ret_1d_same") if c in d]
    return d[cols].corr(method="spearman")


def by_year(d: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for y, g in d.groupby(d.index.year):
        r = information_coefficient(g, col, 1)
        b = g["proposed_boost"]
        rows.append({"year": y, "n": r.n, "ic_1d": round(r.ic, 3), "t": round(r.t_stat, 2),
                     "hit": round(r.hit_rate, 3), "share_-40": round(float((b == -40).mean()), 2),
                     "share_+40": round(float((b == 40).mean()), 2)})
    return pd.DataFrame(rows)
