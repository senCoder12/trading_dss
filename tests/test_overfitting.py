"""Tests for multiple-testing / overfitting statistics (src/backtest/overfitting.py)."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

from src.backtest.overfitting import (
    deflate_selected_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    return_moments,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# PSR
# ---------------------------------------------------------------------------


class TestProbabilisticSharpe:
    @pytest.mark.parametrize("n_obs", [2, 30, 252, 5000])
    @pytest.mark.parametrize("skew,kurt", [(0.0, 3.0), (-2.0, 9.0), (1.0, 5.0)])
    def test_sr_equal_benchmark_is_half(self, n_obs, skew, kurt):
        assert probabilistic_sharpe_ratio(0.0, 0.0, n_obs, skew, kurt) == pytest.approx(0.5)
        assert probabilistic_sharpe_ratio(0.1, 0.1, n_obs, skew, kurt) == pytest.approx(0.5)

    def test_increases_with_track_record(self):
        vals = [probabilistic_sharpe_ratio(0.1, 0.0, n) for n in (20, 60, 250, 1000)]
        assert vals == sorted(vals)
        assert vals[-1] > 0.99

    def test_normal_closed_form(self):
        # Normal returns: z = SR·sqrt(T-1)/sqrt(1 + SR²/2)
        sr, n = 0.1, 253
        z = sr * math.sqrt(n - 1) / math.sqrt(1 + 0.5 * sr * sr)
        from scipy.stats import norm
        assert probabilistic_sharpe_ratio(sr, 0.0, n) == pytest.approx(norm.cdf(z))

    def test_negative_skew_fat_tails_lower_confidence(self):
        base = probabilistic_sharpe_ratio(0.1, 0.0, 250, 0.0, 3.0)
        bad = probabilistic_sharpe_ratio(0.1, 0.0, 250, -3.0, 10.0)
        assert bad < base

    def test_requires_two_obs(self):
        with pytest.raises(ValueError):
            probabilistic_sharpe_ratio(0.1, 0.0, 1)


# ---------------------------------------------------------------------------
# Expected max Sharpe / DSR
# ---------------------------------------------------------------------------


class TestDeflatedSharpe:
    def test_expected_max_single_trial_is_mean(self):
        assert expected_max_sharpe(1, 1.0) == 0.0
        assert expected_max_sharpe(1, 1.0, mean=0.3) == 0.3

    def test_expected_max_known_value(self):
        # Standard-normal maxima: E[max of 1000] ≈ 3.26 (False Strategy Theorem)
        assert expected_max_sharpe(1000, 1.0) == pytest.approx(3.255, abs=0.01)

    def test_expected_max_increasing_in_trials_and_variance(self):
        ns = [2, 10, 100, 1000, 10000]
        vals = [expected_max_sharpe(n, 0.01) for n in ns]
        assert vals == sorted(vals)
        assert expected_max_sharpe(100, 0.04) == pytest.approx(2 * expected_max_sharpe(100, 0.01))

    def test_paper_numerical_example(self):
        # Bailey & López de Prado (2014), Section 6: annual SR 2.5 on 1250
        # daily obs (250/yr), 100 trials with annual-SR variance 0.5,
        # skew -3, kurtosis 10  →  DSR ≈ 0.9004.
        sr = 2.5 / math.sqrt(250)
        dsr = deflated_sharpe_ratio(sr, 100, 1250, skew=-3.0, kurtosis=10.0,
                                    sr_trials_variance=0.5 / 250)
        assert dsr == pytest.approx(0.9004, abs=1e-3)

    def test_dsr_decreases_with_n_trials(self):
        vals = [
            deflated_sharpe_ratio(0.15, n, 500, sr_trials_variance=0.003)
            for n in (1, 2, 5, 20, 100, 1000, 10000)
        ]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_dsr_single_trial_equals_psr(self):
        assert deflated_sharpe_ratio(0.1, 1, 300, sr_trials_variance=0.5) == pytest.approx(
            probabilistic_sharpe_ratio(0.1, 0.0, 300)
        )

    def test_deflate_selected_sharpe_wrapper(self):
        rng = np.random.default_rng(3)
        trial_srs = rng.normal(0.0, 0.05, 200)
        res = deflate_selected_sharpe(float(trial_srs.max()), 250, trial_srs)
        assert res.n_trials == 200
        assert res.sr_trials_variance == pytest.approx(float(np.var(trial_srs, ddof=1)))
        # Best of 200 noise strategies must not look significant after deflation
        assert res.dsr < 0.95 and not res.passes
        assert res.psr > res.dsr
        assert res.expected_max_sharpe > 0


class TestMinTRL:
    def test_infinite_when_not_above_benchmark(self):
        assert math.isinf(minimum_track_record_length(0.05, 0.05))
        assert math.isinf(minimum_track_record_length(-0.1, 0.0))

    def test_psr_at_mintrl_equals_confidence(self):
        sr, skew, kurt = 0.12, -1.0, 6.0
        n = minimum_track_record_length(sr, 0.0, skew, kurt, confidence=0.95)
        assert probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt) == pytest.approx(0.95, abs=1e-9)

    def test_longer_for_fat_tails(self):
        assert minimum_track_record_length(0.1, 0, -3, 10) > minimum_track_record_length(0.1)


class TestHelpers:
    def test_sharpe_ratio_per_period(self):
        r = [0.01, -0.005, 0.02, 0.0]
        assert sharpe_ratio(r) == pytest.approx(np.mean(r) / np.std(r, ddof=1))
        assert sharpe_ratio([0.01]) == 0.0
        assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0

    def test_return_moments_normal_fallback(self):
        assert return_moments([1.0, 1.0, 1.0]) == (0.0, 3.0)
        rng = np.random.default_rng(0)
        skew, kurt = return_moments(rng.normal(size=200_000))
        assert skew == pytest.approx(0.0, abs=0.03)
        assert kurt == pytest.approx(3.0, abs=0.05)


# ---------------------------------------------------------------------------
# PBO (CSCV)
# ---------------------------------------------------------------------------


class TestPBO:
    def _noise(self, t=800, n=40, seed=7):
        return np.random.default_rng(seed).normal(0.0, 0.01, (t, n))

    def test_overfit_selection_pbo_near_one(self):
        # Every strategy has exactly zero full-sample mean: whatever looks best
        # in-sample must look worse out-of-sample → PBO ≈ 1.
        m = self._noise()
        m = m - m.mean(axis=0)
        res = probability_of_backtest_overfitting(m, n_splits=10)
        assert res.pbo > 0.9
        assert res.degradation_slope < 0
        assert res.prob_oos_loss > 0.9

    def test_skill_less_selection_pbo_near_half(self):
        # i.i.d. noise: the IS winner is a random draw OOS → PBO ≈ 0.5
        pbos = [probability_of_backtest_overfitting(self._noise(seed=s), 10).pbo for s in range(5)]
        assert 0.3 < float(np.mean(pbos)) < 0.7

    def test_dominant_strategy_pbo_near_zero(self):
        m = self._noise()
        m[:, 5] += 0.004
        res = probability_of_backtest_overfitting(m, n_splits=10)
        assert res.pbo < 0.05
        assert np.all(res.logits > 0)

    def test_structure(self):
        m = self._noise(t=203, n=6)
        res = probability_of_backtest_overfitting(m, n_splits=8)
        assert res.n_combinations == len(list(combinations(range(8), 4)))
        assert res.logits.shape == (res.n_combinations,)
        assert res.n_obs_used == (203 // 8) * 8
        assert np.all((res.oos_relative_rank > 0) & (res.oos_relative_rank < 1))
        assert 0.0 <= res.pbo <= 1.0

    def test_callable_metric_matches_builtin_mean(self):
        m = self._noise(t=160, n=5)
        a = probability_of_backtest_overfitting(m, 8, metric="mean")
        b = probability_of_backtest_overfitting(m, 8, metric=lambda x: x.mean(axis=0))
        assert a.pbo == pytest.approx(b.pbo)
        np.testing.assert_allclose(a.logits, b.logits)

    def test_max_combinations_subsamples(self):
        res = probability_of_backtest_overfitting(self._noise(t=320, n=5), 16, max_combinations=100)
        assert res.n_combinations == 100

    @pytest.mark.parametrize("kwargs", [
        {"n_splits": 7},
        {"n_splits": 0},
    ])
    def test_invalid_splits(self, kwargs):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(self._noise(t=100, n=4), **kwargs)

    def test_invalid_shapes(self):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.zeros((100, 1)), 4)
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.zeros((10, 4)), 16)
        bad = self._noise(t=100, n=4)
        bad[3, 1] = np.nan
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(bad, 4)
