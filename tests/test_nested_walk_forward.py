"""Tests for the generic nested walk-forward and legacy-validator fixes."""

from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import MetricsCalculator, session_days
from src.backtest.strategy_runner import BacktestConfig
from src.backtest.trade_simulator import SimulatorConfig
from src.backtest.walk_forward import (
    FixedConfigWalkForwardWarning,
    WalkForwardConfig,
    WalkForwardValidator,
    make_walk_forward_folds,
    nested_walk_forward,
)
from src.database.db_manager import DatabaseManager

DAYS = session_days(date(2023, 1, 2), date(2024, 3, 29))  # ~320 weekdays


class _Recorder:
    def __init__(self):
        self.fit_calls: list[list] = []
        self.eval_calls: list[tuple] = []

    def fit(self, train):
        train = list(train)
        self.fit_calls.append(train)
        return {"max_train_day": max(train), "n": len(train)}

    def evaluate(self, model, test):
        test = list(test)
        self.eval_calls.append((model, test))
        # fit_fn only ever saw data strictly before the test fold
        assert model["max_train_day"] < min(test)
        return {"n_test": len(test), "score": float(len(test)) / model["n"]}


class TestFoldConstruction:
    def test_chronological_no_overlap_embargo_respected(self):
        folds, hold = make_walk_forward_folds(DAYS, n_folds=5, embargo_days=3, holdout_fraction=0.2)
        idx = {d: i for i, d in enumerate(DAYS)}
        assert len(folds) == 5
        for f in folds:
            assert not set(f.train_days) & set(f.test_days)
            assert f.train_end < f.test_start
            assert idx[f.test_start] - idx[f.train_end] - 1 == 3      # embargo gap
        # test folds are contiguous, ordered and non-overlapping
        for a, b in zip(folds, folds[1:]):
            assert a.test_end < b.test_start
        # holdout = last ceil(20%) days, untouched by every fold
        n_hold = math.ceil(len(DAYS) * 0.2)
        assert hold.test_days == DAYS[-n_hold:]
        for f in folds:
            assert not set(f.train_days + f.test_days) & set(hold.test_days)
        assert idx[hold.test_start] - idx[hold.train_end] - 1 == 3
        assert hold.train_days == DAYS[: len(DAYS) - n_hold - 3]

    def test_rolling_window_length(self):
        folds, _ = make_walk_forward_folds(
            DAYS, n_folds=4, train_days=60, test_days=20, embargo_days=2,
            holdout_fraction=0.1, expanding=False,
        )
        assert all(len(f.train_days) == 60 for f in folds)
        assert all(len(f.test_days) == 20 for f in folds)

    def test_expanding_train_grows(self):
        folds, _ = make_walk_forward_folds(DAYS, n_folds=4, embargo_days=1)
        sizes = [len(f.train_days) for f in folds]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
        assert all(f.train_start == DAYS[0] for f in folds)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            make_walk_forward_folds(DAYS, holdout_fraction=1.0)
        with pytest.raises(ValueError):
            make_walk_forward_folds(DAYS[:5], n_folds=10)
        with pytest.raises(ValueError):
            make_walk_forward_folds(DAYS, expanding=False)     # rolling needs train_days


class TestNestedWalkForward:
    def test_fit_sees_only_train_and_holdout_evaluated_once(self):
        rec = _Recorder()
        res = nested_walk_forward(DAYS, rec.fit, rec.evaluate, n_folds=4,
                                  embargo_days=2, holdout_fraction=0.25)
        n_hold = math.ceil(len(DAYS) * 0.25)
        holdout_days = set(DAYS[-n_hold:])

        assert len(rec.fit_calls) == 5 and len(rec.eval_calls) == 5   # 4 folds + holdout
        # holdout data never reaches fit_fn
        for train in rec.fit_calls:
            assert not set(train) & holdout_days
        # holdout is evaluated exactly once
        assert sum(1 for _, t in rec.eval_calls if set(t) & holdout_days) == 1
        assert set(rec.eval_calls[-1][1]) == holdout_days
        # holdout model fitted on ALL pre-holdout data except the embargo
        assert rec.fit_calls[-1] == DAYS[: len(DAYS) - n_hold - 2]

        assert res.n_folds == 4 and res.holdout is not None
        assert res.holdout_metrics["n_test"] == n_hold
        assert res.oos_summary["n_test"]["n"] == 4.0
        assert res.oos_summary["score"]["mean"] == pytest.approx(
            np.mean([fr.metrics["score"] for fr in res.folds])
        )
        assert "HOLDOUT" in res.summary()

    def test_no_holdout(self):
        rec = _Recorder()
        res = nested_walk_forward(DAYS, rec.fit, rec.evaluate, n_folds=3, holdout_fraction=0.0)
        assert res.holdout is None and res.holdout_metrics is None
        assert len(rec.eval_calls) == 3

    def test_intraday_timestamps_never_split_a_day(self):
        ts = [datetime(d.year, d.month, d.day, 9, 15) + timedelta(minutes=5 * k)
              for d in DAYS[:120] for k in range(6)]
        seen = []

        def fit(train):
            seen.append(("fit", list(train)))
            return max(train)

        def ev(model, test):
            test = list(test)
            assert model < min(test)
            seen.append(("eval", test))
            return {"bars": len(test)}

        res = nested_walk_forward(ts, fit, ev, n_folds=3, embargo_days=1, holdout_fraction=0.2)
        for kind, chunk in seen:
            assert len(chunk) % 6 == 0                     # whole days only
        fit_days = {t.date() for kind, c in seen if kind == "fit" for t in c}
        hold_days = {t.date() for t in seen[-1][1]}
        assert not fit_days & hold_days
        assert res.holdout_metrics["bars"] == len(hold_days) * 6

    def test_pandas_inputs_preserve_type(self):
        idx = pd.DatetimeIndex(DAYS[:150])
        types = []
        nested_walk_forward(
            idx, lambda tr: types.append(type(tr)) or 0, lambda m, te: {"x": 1.0},
            n_folds=2, holdout_fraction=0.2,
        )
        assert all(issubclass(t, pd.DatetimeIndex) for t in types)

        frame = pd.DataFrame({"r": np.arange(150.0)}, index=idx)
        got = []
        nested_walk_forward(
            frame, lambda tr: got.append(tr) or tr.index.max(),
            lambda m, te: {"ok": float(m < te.index.min())},
            n_folds=2, holdout_fraction=0.2,
        )
        assert all(isinstance(g, pd.DataFrame) for g in got)

    def test_unsorted_input_rejected(self):
        with pytest.raises(ValueError):
            nested_walk_forward(list(reversed(DAYS)), lambda t: 0, lambda m, t: {})

    def test_eval_must_return_dict(self):
        with pytest.raises(TypeError):
            nested_walk_forward(DAYS, lambda t: 0, lambda m, t: 1.0, n_folds=2)


class TestLegacyValidatorFixes:
    def _validator(self):
        v = WalkForwardValidator(MagicMock(spec=DatabaseManager))
        v._get_trading_days = MagicMock(
            return_value=[date(2022, 1, 1) + timedelta(days=i) for i in range(504)]
        )
        res = MagicMock()
        res.trade_history = []
        res.equity_curve = []
        v.strategy_runner.run = MagicMock(return_value=res)
        v.metrics_calculator.calculate_all = MagicMock(
            return_value=MetricsCalculator.calculate_all([], [], 100_000)
        )
        return v

    def test_warns_and_flags_not_out_of_sample(self):
        v = self._validator()
        cfg = WalkForwardConfig("NIFTY50", date(2022, 1, 1), date(2023, 5, 19))
        with pytest.warns(FixedConfigWalkForwardWarning):
            result = v.run_walk_forward(cfg)
        assert result.is_out_of_sample is False
        assert "NOT out-of-sample" in result.verdict

    def test_template_preserves_all_optimised_params(self):
        v = self._validator()
        sim = SimulatorConfig()
        template = BacktestConfig(
            index_id="NIFTY50", start_date=date(2022, 1, 1), end_date=date(2023, 5, 19),
            min_confidence="HIGH", signal_cooldown_bars=9, mode="TECHNICAL_ONLY",
            simulator_config=sim,
        )
        cfg = WalkForwardConfig(
            "NIFTY50", date(2022, 1, 1), date(2023, 5, 19),
            simulator_config=sim, backtest_template=template,
        )
        with pytest.warns(FixedConfigWalkForwardWarning):
            v.run_walk_forward(cfg)
        runs = [c.args[0] for c in v.strategy_runner.run.call_args_list]
        assert runs
        for bt in runs:
            assert bt.min_confidence == "HIGH"
            assert bt.signal_cooldown_bars == 9
            assert bt.show_progress is False
        assert len({(bt.start_date, bt.end_date) for bt in runs}) == len(runs)

    def test_embargo_shifts_test_window(self):
        v = self._validator()
        days = [date(2022, 1, 1) + timedelta(days=i) for i in range(504)]
        cfg = WalkForwardConfig("NIFTY50", days[0], days[-1], embargo_days=5)
        w = v._calculate_windows(cfg, days)[0]
        assert w["train_end"] == days[251]
        assert w["test_start"] == days[257]
