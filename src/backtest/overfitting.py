"""
Multiple-testing and backtest-overfitting statistics.

Implements the selection-bias corrections that an optimiser which ranks many
parameter combinations by Sharpe ratio needs before any "best" configuration
can be trusted:

* :func:`probabilistic_sharpe_ratio` (PSR) — probability that the true Sharpe
  exceeds a benchmark, given sample length, skewness and kurtosis
  (Bailey & López de Prado, 2012, "The Sharpe Ratio Efficient Frontier").
* :func:`expected_max_sharpe` — expected maximum Sharpe among ``N`` unskilled
  trials (False Strategy Theorem).
* :func:`deflated_sharpe_ratio` (DSR) — PSR whose benchmark is the expected
  maximum Sharpe of ``N`` trials (Bailey & López de Prado, 2014, "The Deflated
  Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and
  Non-Normality", Journal of Portfolio Management 40(5)).
* :func:`minimum_track_record_length` (MinTRL).
* :func:`probability_of_backtest_overfitting` (PBO) via Combinatorially
  Symmetric Cross-Validation (Bailey, Borwein, López de Prado & Zhu, 2017,
  "The Probability of Backtest Overfitting", Journal of Computational
  Finance 20(4)).

Conventions
-----------
* Every Sharpe ratio in this module is **per period** (NOT annualised) and
  must be measured on the same returns whose count is ``n_obs`` — e.g. a daily
  Sharpe with ``n_obs`` = number of daily returns.  Use
  :func:`annualise_sharpe` only for display.
* ``kurtosis`` is Pearson (non-excess) kurtosis: a normal distribution has
  kurtosis 3.
* ``sr_trials_variance`` is the cross-sectional variance of the per-period
  Sharpe ratios of all trials that were tried (not only the ones reported).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329

ArrayLike = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def _clean(returns: ArrayLike) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    return r[np.isfinite(r)]


def sharpe_ratio(returns: ArrayLike, rf_per_period: float = 0.0, ddof: int = 1) -> float:
    """Per-period (non-annualised) Sharpe ratio ``mean(r - rf) / std(r)``.

    Returns 0.0 when fewer than two observations or zero dispersion.
    """
    r = _clean(returns) - rf_per_period
    if r.size < 2:
        return 0.0
    sd = float(r.std(ddof=ddof))
    if not math.isfinite(sd) or sd <= 0.0:
        return 0.0
    return float(r.mean() / sd)


def annualise_sharpe(sr: float, periods_per_year: float = 252.0) -> float:
    """Convert a per-period Sharpe to annual units (``sr * sqrt(ppy)``)."""
    return float(sr) * math.sqrt(periods_per_year)


def return_moments(returns: ArrayLike) -> tuple[float, float]:
    """Return ``(skewness, pearson_kurtosis)`` of a return series.

    Falls back to the normal values ``(0.0, 3.0)`` when the series is too short
    or has zero dispersion.
    """
    r = _clean(returns)
    if r.size < 3 or float(r.std()) <= 0.0:
        return 0.0, 3.0
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    if not (math.isfinite(skew) and math.isfinite(kurt)):
        return 0.0, 3.0
    return skew, kurt


def _sr_variance_factor(sr: float, skew: float, kurtosis: float) -> float:
    """``1 - γ3·SR + (γ4 - 1)/4 · SR²`` (Mertens / Bailey-LdP SR variance term).

    For any valid moment pair (γ4 ≥ 1 + γ3²) this is ≥ (1 - γ3·SR/2)² ≥ 0.
    It is clamped away from zero to keep the z-score finite.
    """
    v = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    return max(v, 1e-12)


# ---------------------------------------------------------------------------
# PSR / DSR / MinTRL
# ---------------------------------------------------------------------------


def probabilistic_sharpe_ratio(
    sr: float,
    sr_benchmark: float = 0.0,
    n_obs: int = 252,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio: ``P(true SR > sr_benchmark)``.

    PSR = Φ( (SR - SR*) · sqrt(T - 1) / sqrt(1 - γ3·SR + (γ4 - 1)/4 · SR²) )

    Args:
        sr: observed per-period Sharpe ratio.
        sr_benchmark: per-period benchmark Sharpe ``SR*``.
        n_obs: number of return observations ``T`` used to estimate ``sr``.
        skew: skewness of the returns (γ3).
        kurtosis: Pearson kurtosis of the returns (γ4; normal = 3).
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(
        _sr_variance_factor(sr, skew, kurtosis)
    )
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, variance: float, mean: float = 0.0) -> float:
    """Expected maximum of ``n_trials`` Sharpe ratios drawn with zero skill.

    E[max SR] ≈ mean + sqrt(V) · ((1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N·e)))

    where γ is the Euler–Mascheroni constant.  With a single trial there is no
    selection, so the expectation is ``mean``.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_trials == 1 or variance <= 0.0:
        return float(mean)
    n = float(n_trials)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(mean + math.sqrt(variance) * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    sr: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_trials_variance: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    The PSR of ``sr`` against the benchmark ``SR0 = expected_max_sharpe(
    n_trials, sr_trials_variance)``, i.e. the probability that the selected
    strategy's true Sharpe is positive after accounting for having picked the
    best of ``n_trials`` attempts.  Values ≥ 0.95 are conventionally required.
    """
    sr0 = expected_max_sharpe(n_trials, sr_trials_variance)
    return probabilistic_sharpe_ratio(sr, sr0, n_obs, skew, kurtosis)


def minimum_track_record_length(
    sr: float,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Minimum number of observations needed for ``PSR(sr_benchmark) ≥ confidence``.

    MinTRL = 1 + (1 - γ3·SR + (γ4 - 1)/4 · SR²) · (Z_α / (SR - SR*))²

    Returns ``inf`` when ``sr <= sr_benchmark`` (no track record suffices).
    """
    if sr <= sr_benchmark:
        return float("inf")
    z = stats.norm.ppf(confidence)
    return float(1.0 + _sr_variance_factor(sr, skew, kurtosis) * (z / (sr - sr_benchmark)) ** 2)


@dataclass
class DeflatedSharpeResult:
    """Selection-bias-adjusted assessment of a chosen strategy."""

    sharpe: float                    # per-period SR of the selected strategy
    sharpe_annualised: float
    n_obs: int
    n_trials: int
    sr_trials_variance: float        # cross-sectional variance of per-period SRs
    expected_max_sharpe: float       # SR0 (per period) — the deflation benchmark
    skew: float
    kurtosis: float                  # Pearson
    psr: float                       # PSR vs 0 (no multiple-testing correction)
    dsr: float                       # PSR vs SR0
    min_track_record_length: float   # obs needed for PSR(SR0) ≥ threshold
    threshold: float
    passes: bool                     # dsr >= threshold
    assumed_normal: bool = False     # True if moments were not estimated from data


def deflate_selected_sharpe(
    sr: float,
    n_obs: int,
    trial_sharpes: ArrayLike,
    n_trials: Optional[int] = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: float = 252.0,
    threshold: float = 0.95,
    assumed_normal: bool = False,
) -> DeflatedSharpeResult:
    """Convenience wrapper: DSR of a selected strategy given all trial Sharpes.

    Args:
        sr: per-period Sharpe of the selected strategy.
        n_obs: number of return observations behind ``sr``.
        trial_sharpes: per-period Sharpe of every trial evaluated (used for the
            cross-sectional variance).
        n_trials: number of independent trials; defaults to ``len(trial_sharpes)``.
    """
    srs = _clean(trial_sharpes)
    n = int(n_trials) if n_trials is not None else max(int(srs.size), 1)
    var = float(srs.var(ddof=1)) if srs.size >= 2 else 0.0
    sr0 = expected_max_sharpe(n, var)
    if n_obs >= 2:
        psr = probabilistic_sharpe_ratio(sr, 0.0, n_obs, skew, kurtosis)
        dsr = probabilistic_sharpe_ratio(sr, sr0, n_obs, skew, kurtosis)
    else:
        psr = dsr = 0.0
    mintrl = minimum_track_record_length(sr, sr0, skew, kurtosis, threshold)
    return DeflatedSharpeResult(
        sharpe=float(sr),
        sharpe_annualised=annualise_sharpe(sr, periods_per_year),
        n_obs=int(n_obs),
        n_trials=n,
        sr_trials_variance=var,
        expected_max_sharpe=sr0,
        skew=float(skew),
        kurtosis=float(kurtosis),
        psr=psr,
        dsr=dsr,
        min_track_record_length=mintrl,
        threshold=threshold,
        passes=bool(dsr >= threshold),
        assumed_normal=assumed_normal,
    )


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (CSCV)
# ---------------------------------------------------------------------------


@dataclass
class PBOResult:
    """Outcome of Combinatorially Symmetric Cross-Validation."""

    pbo: float                      # P(IS-best ranks at/below OOS median)
    logits: np.ndarray              # λ_c = ln(ω̄_c / (1 - ω̄_c)) per combination
    n_splits: int
    n_combinations: int
    n_strategies: int
    n_obs_used: int
    is_performance: np.ndarray      # IS metric of the IS-best per combination
    oos_performance: np.ndarray     # OOS metric of that same strategy
    oos_relative_rank: np.ndarray   # ω̄_c in (0, 1)
    degradation_slope: float        # OLS slope of OOS on IS performance
    degradation_intercept: float
    degradation_r2: float
    prob_oos_loss: float            # P(OOS metric of IS-best < 0)


def _block_stats(m: np.ndarray, n_splits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    t, _ = m.shape
    rows = t // n_splits
    trimmed = m[t - rows * n_splits:]          # drop the oldest remainder rows
    blocks = trimmed.reshape(n_splits, rows, -1)
    return blocks.sum(axis=1), (blocks ** 2).sum(axis=1), np.full(n_splits, rows, dtype=float), rows * n_splits


def _perf_from_sums(sums: np.ndarray, sumsq: np.ndarray, counts: np.ndarray, metric: str) -> np.ndarray:
    n = counts[:, None]
    mean = sums / n
    if metric == "mean":
        return mean
    var = (sumsq - n * mean ** 2) / np.maximum(n - 1.0, 1.0)
    sd = np.sqrt(np.clip(var, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        perf = np.where(sd > 1e-15, mean / np.where(sd > 1e-15, sd, 1.0), np.sign(mean) * np.inf)
    return np.nan_to_num(perf, nan=0.0)


def _relative_rank(row_values: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """Average rank (1..N) of ``chosen[c]`` within ``row_values[c]``, scaled to (0,1)."""
    n = row_values.shape[1]
    less = (row_values < chosen[:, None]).sum(axis=1)
    equal = (row_values == chosen[:, None]).sum(axis=1)
    rank = less + (equal + 1) / 2.0
    return rank / (n + 1.0)


def probability_of_backtest_overfitting(
    perf_matrix: ArrayLike,
    n_splits: int = 16,
    metric: Union[str, Callable[[np.ndarray], np.ndarray]] = "sharpe",
    max_combinations: Optional[int] = None,
    seed: Optional[int] = 0,
) -> PBOResult:
    """Probability of Backtest Overfitting via CSCV.

    Args:
        perf_matrix: ``T × N`` matrix of per-period returns (rows chronological,
            one column per strategy / parameter combination).
        n_splits: number ``S`` of equal, contiguous row blocks (even, ≥ 2).
            All ``C(S, S/2)`` train/test block combinations are evaluated.
        metric: ``"sharpe"`` (mean/std, default), ``"mean"``, or a callable
            mapping a ``t × N`` sub-matrix to an ``N`` vector of performance.
        max_combinations: if given and smaller than ``C(S, S/2)``, a random
            subset (seeded) of combinations is evaluated.

    Returns:
        :class:`PBOResult`.  ``pbo`` is the fraction of combinations whose
        in-sample-best strategy ranks at or below the median out of sample
        (logit ≤ 0).  A skill-less selection gives PBO ≈ 0.5, a selection that
        systematically picks OOS losers gives PBO → 1, and a genuinely dominant
        strategy gives PBO → 0.
    """
    m = np.asarray(perf_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T observations × N strategies)")
    t, n = m.shape
    if n < 2:
        raise ValueError("PBO needs at least 2 strategies (columns)")
    if n_splits < 2 or n_splits % 2:
        raise ValueError(f"n_splits must be an even integer >= 2, got {n_splits}")
    if t < 2 * n_splits:
        raise ValueError(f"Need at least {2 * n_splits} rows for n_splits={n_splits}, got {t}")
    if not np.all(np.isfinite(m)):
        raise ValueError("perf_matrix contains non-finite values")

    half = n_splits // 2
    combos = list(itertools.combinations(range(n_splits), half))
    if max_combinations is not None and max_combinations < len(combos):
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in sorted(pick)]
    ind = np.zeros((len(combos), n_splits))
    for i, c in enumerate(combos):
        ind[i, list(c)] = 1.0
    oos_ind = 1.0 - ind

    if callable(metric):
        rows = t // n_splits
        trimmed = m[t - rows * n_splits:]
        block_rows = trimmed.reshape(n_splits, rows, n)
        is_perf = np.empty((len(combos), n))
        oos_perf = np.empty((len(combos), n))
        for i, c in enumerate(combos):
            mask = np.zeros(n_splits, dtype=bool)
            mask[list(c)] = True
            is_perf[i] = np.asarray(metric(block_rows[mask].reshape(-1, n)), dtype=float)
            oos_perf[i] = np.asarray(metric(block_rows[~mask].reshape(-1, n)), dtype=float)
        used = rows * n_splits
    else:
        if metric not in ("sharpe", "mean"):
            raise ValueError(f"Unknown metric {metric!r}; use 'sharpe', 'mean' or a callable")
        sums, sumsq, counts, used = _block_stats(m, n_splits)
        # Some BLAS builds (e.g. Apple Accelerate with numpy 2.0) emit spurious
        # floating-point warnings from matmul on finite inputs; silence them.
        with np.errstate(all="ignore"):
            is_sums, is_sq, is_n = ind @ sums, ind @ sumsq, ind @ counts
            oos_sums, oos_sq, oos_n = oos_ind @ sums, oos_ind @ sumsq, oos_ind @ counts
        is_perf = _perf_from_sums(is_sums, is_sq, is_n, metric)
        oos_perf = _perf_from_sums(oos_sums, oos_sq, oos_n, metric)

    best = np.argmax(is_perf, axis=1)
    idx = np.arange(len(combos))
    best_is = is_perf[idx, best]
    best_oos = oos_perf[idx, best]
    omega = _relative_rank(oos_perf, best_oos)
    logits = np.log(omega / (1.0 - omega))
    pbo = float(np.mean(logits <= 0.0))

    finite = np.isfinite(best_is) & np.isfinite(best_oos)
    slope = intercept = r2 = float("nan")
    if finite.sum() >= 2 and np.ptp(best_is[finite]) > 0:
        slope, intercept = (float(v) for v in np.polyfit(best_is[finite], best_oos[finite], 1))
        pred = intercept + slope * best_is[finite]
        ss_res = float(((best_oos[finite] - pred) ** 2).sum())
        ss_tot = float(((best_oos[finite] - best_oos[finite].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return PBOResult(
        pbo=pbo,
        logits=logits,
        n_splits=n_splits,
        n_combinations=len(combos),
        n_strategies=n,
        n_obs_used=int(used),
        is_performance=best_is,
        oos_performance=best_oos,
        oos_relative_rank=omega,
        degradation_slope=slope,
        degradation_intercept=intercept,
        degradation_r2=r2,
        prob_oos_loss=float(np.mean(best_oos < 0.0)),
    )
