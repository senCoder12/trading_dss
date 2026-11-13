"""Deterministic tests for scripts/morning_strategy_search.py (research script)."""

from __future__ import annotations

import importlib.util
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from src.backtest.overfitting import expected_max_sharpe

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "morning_strategy_search.py"
_spec = importlib.util.spec_from_file_location("morning_strategy_search", _PATH)
mss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mss)

IST = ZoneInfo("Asia/Kolkata")
PX = 10000.0


def make_bars(day: date, ohlc: list[tuple[float, float, float, float]], start: time = time(9, 15)) -> list[dict]:
    t0 = datetime.combine(day, start, tzinfo=IST)
    return [{"timestamp": t0 + timedelta(minutes=5 * i), "open": o, "high": h, "low": lo, "close": c, "volume": 0.0}
            for i, (o, h, lo, c) in enumerate(ohlc)]


def flat(n: int, px: float = PX, w: float = 1.0) -> list[tuple[float, float, float, float]]:
    return [(px, px + w, px - w, px)] * n


def idx_of(t: time) -> int:
    return ((t.hour * 60 + t.minute) - (9 * 60 + 15)) // 5


D0 = date(2025, 1, 6)  # a Monday


# ── simulate ────────────────────────────────────────────────────────────────
def _mb(rows):
    return make_bars(D0, rows + flat(33 - len(rows)))


class TestSimulate:
    # long: entry 10000 at bar 5 open, stop 9970 (30 pts = 1 R), target 10060 (2 R)
    def _run(self, rows, direction=1, stop=9970.0, target=10060.0, entry_i=5):
        return mss.simulate(_mb(rows), entry_i, direction, stop, target)

    def test_adverse_gap_through_stop_fills_at_open(self):
        rows = flat(5) + [(PX, PX + 5, PX - 5, PX - 5), (9950.0, 9955.0, 9940.0, 9950.0)]
        res = self._run(rows)
        assert res["reason"] == "stop_gap"
        assert res["r"] == pytest.approx(-50 / 30)
        assert res["r"] < -1.0
        assert res["exit_i"] == 6

    def test_favourable_gap_through_target_fills_at_target(self):
        rows = flat(5) + [(PX, PX + 5, PX - 5, PX + 5), (10080.0, 10090.0, 9960.0, 10085.0)]
        res = self._run(rows)
        assert res["reason"] == "target_gap"
        assert res["r"] == pytest.approx(2.0)          # no gap bonus, and the gap beats the in-bar stop touch

    def test_stop_and_target_in_same_candle_is_stop(self):
        rows = flat(5) + [(PX, PX + 5, PX - 5, PX), (PX, 10070.0, 9960.0, PX)]
        res = self._run(rows)
        assert res["reason"] == "stop" and res["r"] == -1.0 and res["exit_i"] == 6

    def test_checking_starts_on_entry_bar_stop(self):
        rows = flat(5) + [(PX, PX + 5, 9965.0, 9990.0)]
        res = self._run(rows)
        assert res["reason"] == "stop" and res["exit_i"] == 5 and res["hold_bars"] == 1

    def test_checking_starts_on_entry_bar_target(self):
        rows = flat(5) + [(PX, 10061.0, PX - 5, 10050.0)]
        res = self._run(rows)
        assert res["reason"] == "target" and res["r"] == pytest.approx(2.0) and res["exit_i"] == 5

    def test_time_exit_at_last_bar_before_noon(self):
        rows = flat(32) + [(PX, PX + 9, PX - 9, PX + 9)]          # 11:55 bar closes +9
        mb = make_bars(D0, rows + flat(10, px=PX + 50))             # 12:00+ bars must be ignored
        res = mss.simulate(mb, 5, 1, 9970.0, 10060.0)
        assert res["reason"] == "time"
        assert mb[res["exit_i"]]["timestamp"].time() == time(11, 55)
        assert res["r"] == pytest.approx(9 / 30)
        assert res["hold_bars"] == 33 - 5

    def test_short_side_gap_through_stop(self):
        rows = flat(5) + [(PX, PX + 5, PX - 5, PX), (10040.0, 10045.0, 10035.0, 10040.0)]
        res = self._run(rows, direction=-1, stop=10030.0, target=9940.0)
        assert res["reason"] == "stop_gap" and res["r"] == pytest.approx(-40 / 30)

    def test_short_side_both_in_candle_is_stop(self):
        rows = flat(5) + [(PX, PX + 5, PX - 5, PX), (PX, 10031.0, 9930.0, PX)]
        res = self._run(rows, direction=-1, stop=10030.0, target=9940.0)
        assert res["reason"] == "stop" and res["r"] == -1.0

    def test_rejects_inconsistent_levels(self):
        with pytest.raises(ValueError):
            self._run(flat(6), direction=1, stop=10010.0, target=10060.0)
        with pytest.raises(ValueError):
            self._run(flat(6), direction=0)


# ── data conventions ────────────────────────────────────────────────────────
class TestMorningBars:
    def test_complete_morning_ignores_afternoon(self):
        assert len(mss.morning_bars(make_bars(D0, flat(75)))) == 33
        assert len(mss.morning_bars(make_bars(D0, flat(33)))) == 33    # afternoon missing: still usable

    def test_missing_morning_bar_rejected(self):
        bars = make_bars(D0, flat(75))
        del bars[idx_of(time(10, 0))]
        assert mss.morning_bars(bars) is None

    def test_late_start_rejected(self):
        assert mss.morning_bars(make_bars(D0, flat(75), start=time(9, 20))) is None


def test_previous_close_uses_daily_and_rejects_big_gaps():
    daily = {date(2025, 1, 3): {"close": 9950.0}, date(2025, 1, 6): {"close": 1.0}}
    assert mss.previous_close(daily, date(2025, 1, 6)) == 9950.0           # Fri -> Mon
    assert mss.previous_close(daily, date(2025, 1, 3)) is None              # no history
    assert mss.previous_close({date(2024, 12, 27): {"close": 1.0}}, date(2025, 1, 6)) is None


# ── trade generation ────────────────────────────────────────────────────────
def orb_day(day: date, signal_at: time, or_half: float = 10.0) -> list[dict]:
    """OR 9990-10010; flat inside it; the bar starting `signal_at` closes at 10015."""
    rows = [(PX, PX + or_half, PX - or_half, PX)] * 6
    j = idx_of(signal_at)
    rows += flat(j - 6, w=2.0)
    rows += [(PX, 10016.0, PX - 2, 10015.0)]
    rows += flat(75 - len(rows), px=10015.0, w=2.0)
    return make_bars(day, rows)


def _daily_for(*days, close=PX):
    return {d: {"open": PX, "high": PX, "low": PX, "close": close} for d in days}


class TestORB:
    def test_signal_at_cutoff_enters_next_bar_open(self):
        d_prev, d = date(2025, 1, 3), D0
        trades, _ = mss.build_trades("NIFTY50", orb_day(d, time(10, 55)), _daily_for(d_prev, d))
        orb = [t for t in trades if t["strategy"] == "ORB15/OPP/1.0R/ALL"]
        assert len(orb) == 1
        assert orb[0]["dir"] == 1 and orb[0]["entry"] == pytest.approx(10015.0)   # 11:00 bar open
        assert orb[0]["stop_d"] == pytest.approx(10015.0 - 9990.0)

    def test_signal_after_cutoff_ignored(self):
        trades, diag = mss.build_trades("NIFTY50", orb_day(D0, time(11, 0)), _daily_for(date(2025, 1, 3), D0))
        assert not [t for t in trades if t["strategy"].startswith("ORB")]
        assert diag["orb15_no_signal"] == 1

    def test_one_trade_per_day_per_variant(self):
        days = [D0 + timedelta(days=k) for k in range(5)]
        bars = [b for d in days for b in orb_day(d, time(9, 50))]
        trades, _ = mss.build_trades("NIFTY50", bars, _daily_for(date(2025, 1, 3), *days))
        keys = [(t["index_id"], t["day"], t["strategy"]) for t in trades]
        assert keys and len(keys) == len(set(keys))

    def test_duplicate_variant_names_raise(self, monkeypatch):
        monkeypatch.setattr(mss, "FADE_GAPS", (0.003, 0.003))
        trades_bars = fade_day(D0, open_px=10000.0)
        with pytest.raises(AssertionError):
            mss.build_trades("NIFTY50", trades_bars, _daily_for(date(2025, 1, 3), close=9950.0) | _daily_for(D0))


def narrow_series(widths: list[float]) -> tuple[list[dict], dict]:
    days, d = [], date(2024, 1, 1)
    while len(days) < len(widths):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    bars = [b for dd, w in zip(days, widths) for b in orb_day(dd, time(9, 50), or_half=w)]
    return bars, (days, _daily_for(*days))


class TestNarrowOR:
    def test_needs_full_prior_window_and_no_lookahead(self):
        widths = [15.0] * 60 + [10.0]
        bars, (days, daily) = narrow_series(widths)
        trades, _ = mss.build_trades("NIFTY50", bars, daily)
        narrow_days = {t["day"] for t in trades if t["strategy"] == "ORB15/OPP/1.0R/NARROW_OR"}
        assert narrow_days == {days[60].isoformat()}            # day 60 has exactly 60 prior days

        bars59, (days59, daily59) = narrow_series([15.0] * 59 + [10.0])
        t59, _ = mss.build_trades("NIFTY50", bars59, daily59)
        assert not [t for t in t59 if t["strategy"].endswith("NARROW_OR")]   # 59 prior days: filter off

        # future days (even extreme ones) must not change earlier decisions
        bars_f, (days_f, daily_f) = narrow_series(widths + [1.0] * 5 + [40.0] * 5)
        t_f, _ = mss.build_trades("NIFTY50", bars_f, daily_f)
        before = sorted((t["day"], t["strategy"], t["r"]) for t in trades)
        after = sorted((t["day"], t["strategy"], t["r"]) for t in t_f if t["day"] <= days[60].isoformat())
        assert before == after


# ── gap fade ────────────────────────────────────────────────────────────────
def fade_day(day: date, open_px: float, or_lo: float = 9990.0, or_hi: float = 10020.0,
             entry_px: float = 9995.0, touch: float | None = None) -> list[dict]:
    rows = [(open_px, or_hi, or_lo, 10000.0), (10000.0, 10005.0, or_lo if touch is None else touch, 9998.0),
            (9998.0, 10000.0, 9992.0, entry_px)]
    rows += [(entry_px, entry_px + 3, entry_px - 3, entry_px)] * (75 - len(rows))
    return make_bars(day, rows)


class TestFade:
    prev = date(2025, 1, 3)

    def test_gap_up_shorts_at_0930_open(self):
        daily = _daily_for(self.prev, close=9950.0) | _daily_for(D0)
        trades, _ = mss.build_trades("NIFTY50", fade_day(D0, 10000.0), daily)
        raw = [t for t in trades if t["strategy"] == "FADE/RAW/gap>=0.3%"]
        assert len(raw) == 1
        t = raw[0]
        assert t["dir"] == -1 and t["entry"] == pytest.approx(9995.0)     # 09:30 bar open
        assert t["stop_d"] == pytest.approx(10020.0 - 9995.0)
        assert {x["strategy"] for x in trades if x["strategy"].startswith("FADE/RAW")} == \
            {"FADE/RAW/gap>=0.3%", "FADE/RAW/gap>=0.5%"}                   # gap 0.50 %

    def test_gap_down_goes_long(self):
        rows = [(10000.0, 10010.0, 9980.0, 10000.0), (10000.0, 10008.0, 9995.0, 10002.0),
                (10002.0, 10008.0, 9998.0, 10005.0)]
        rows += [(10005.0, 10008.0, 10002.0, 10005.0)] * 72
        daily = _daily_for(self.prev, close=10050.0) | _daily_for(D0)
        trades, _ = mss.build_trades("NIFTY50", make_bars(D0, rows), daily)
        raw = [t for t in trades if t["strategy"] == "FADE/RAW/gap>=0.3%"]
        assert len(raw) == 1 and raw[0]["dir"] == 1

    def test_skip_when_prev_close_touched_before_entry(self):
        daily = _daily_for(self.prev, close=9950.0) | _daily_for(D0)
        trades, diag = mss.build_trades("NIFTY50", fade_day(D0, 10000.0, touch=9949.0), daily)
        assert not [t for t in trades if t["strategy"].startswith("FADE")]
        assert diag["fade_prev_close_touched"] >= 1

    def test_r_bounds(self):
        # previous close 9800 (gap 2 %): R = (9995-9800)/25 = 7.8 > 6 → skipped
        daily = _daily_for(self.prev, close=9800.0) | _daily_for(D0)
        trades, diag = mss.build_trades("NIFTY50", fade_day(D0, 10000.0), daily)
        assert not [t for t in trades if t["strategy"].startswith("FADE")]
        assert diag["fade_r_out_of_range"] >= 1

    def test_gap_uses_official_daily_close_not_last_5m_close(self):
        prev_bars = make_bars(self.prev, flat(75, px=10000.0))         # 5m close says "no gap"
        daily = _daily_for(self.prev, close=9950.0) | _daily_for(D0)   # official close 9950
        trades, _ = mss.build_trades("NIFTY50", prev_bars + fade_day(D0, 10000.0), daily)
        assert [t for t in trades if t["day"] == D0.isoformat() and t["strategy"] == "FADE/RAW/gap>=0.3%"]


# ── split, statistics, selection, DSR ───────────────────────────────────────
def test_split_on_unique_dates():
    dates = [f"2025-01-{k:02d}" for k in range(1, 11)]
    train, oos, cut = mss.split_dates(dates + dates[::-1])          # duplicates / order don't matter
    assert len(train) == 7 and len(oos) == 3 and cut == "2025-01-08"
    assert max(train) < min(oos) and not train & oos
    # a date belongs to one partition for every index
    rows = [{"index_id": i, "day": "2025-01-08"} for i in ("NIFTY50", "BANKNIFTY", "SENSEX")]
    assert {r["day"] in oos for r in rows} == {True}


def test_cost_definition_and_scenarios():
    assert mss.COST_SCENARIOS == (0.00025, 0.0005, 0.00075)
    # 12 bars = 1 h: (0.025 % + 0.010 %) x 10000 / 20 points
    assert mss.cost_r(10000.0, 20.0, 12, 0.00025) == pytest.approx(0.175)
    assert mss.cost_r(10000.0, 20.0, 12, 0.00075) > mss.cost_r(10000.0, 20.0, 12, 0.0005)


def _row(day, r, strategy="X", index_id="NIFTY50"):
    return {"day": day, "r": r, "entry": 10000.0, "stop_d": 1e9, "hold_bars": 1,
            "strategy": strategy, "index_id": index_id, "reason": "time"}


def test_daily_equal_weight_and_drawdown():
    rows = [_row("d1", 1.0), _row("d1", -1.0, index_id="BANKNIFTY"), _row("d2", 2.0)]
    s = mss.daily_series(rows, 0.0)
    assert s.tolist() == pytest.approx([0.0, 2.0], abs=1e-6)
    assert mss.max_drawdown(np.array([1.0, -2.0, 1.0, -1.0])) == pytest.approx(2.0)
    st = mss.stats(rows, 0.0)
    assert st["n_days"] == 2 and st["n_trades"] == 3


def test_selection_uses_train_only_and_is_frozen(monkeypatch):
    monkeypatch.setattr(mss, "MIN_TRAIN", 3)
    a, b = mss.ALL_STRATEGIES[0], mss.ALL_STRATEGIES[1]
    train = [_row(f"t{k}", 1.0 + 0.1 * k, a) for k in range(5)] + [_row(f"t{k}", (-1) ** k, b) for k in range(5)]
    picks = mss.select_picks(mss.train_table(train))
    assert picks[0] == a and set(picks) == {a, b}
    frozen = mss.freeze(picks, mss.train_table(train), "o0")
    oos = [_row(f"o{k}", 50.0, b) for k in range(40)]                 # b is spectacular OOS
    evaluated = mss.evaluate_frozen(frozen, train + oos, {f"t{k}" for k in range(5)},
                                    {f"o{k}" for k in range(40)}, [0.1, 0.2])
    assert [e["name"] for e in evaluated] == picks                     # order and members unchanged
    assert frozen["picks"] == picks and len(frozen["sha256"]) == 64


def test_dsr_counts_every_variant():
    assert mss.N_TRIALS == len(mss.ALL_STRATEGIES) == 2 * 2 * 4 * 5 + 2 * 3
    rng = np.random.default_rng(0)
    series = rng.normal(0.05, 1.0, 200)
    srs = [0.0, 0.05, -0.05, 0.1]
    blk = mss.dsr_block(series, srs)
    assert mss.DSR_N == mss.N_TRIALS + mss.PRIOR_TRIALS
    assert blk["n_trials"] == mss.DSR_N and blk["n_obs"] == 200
    assert blk["expected_max_sr"] == pytest.approx(expected_max_sharpe(mss.DSR_N, float(np.var(srs, ddof=1))))
    assert blk["min_trl"] is None or np.isfinite(blk["min_trl"])
    assert 0.0 <= blk["dsr"] <= blk["psr"] <= 1.0


@pytest.mark.parametrize("which, bad, check", [
    ("train", {"n_days": 100, "net": -0.01}, "train_net_positive"),
    ("train", {"n_days": mss.MIN_TRAIN - 1, "net": 0.1}, "train_net_positive"),
    ("oos", {"n_days": mss.MIN_OOS - 1, "net": 0.1}, "oos_days_enough"),
    ("oos", {"n_days": 100, "net": -0.01}, "oos_net_positive_base"),
    ("stress", {"n_days": 100, "net": -0.01}, "oos_net_positive_stress_0.05pct"),
    ("dsr", {"dsr": 0.5}, "oos_dsr_ok"),
    ("sensex", {"n_days": 100, "net": -0.01}, "sensex_oos_net_positive"),
    ("sensex", {"n_days": mss.MIN_OOS - 1, "net": 0.1}, "sensex_oos_net_positive"),
])
def test_validation_requires_every_criterion(which, bad, check):
    good = {"n_days": 100, "net": 0.1}
    args = {"train": good, "oos": good, "stress": good, "dsr": {"dsr": 0.99}, "sensex": good}
    assert mss.validate(*args.values())["validated"]
    args[which] = bad
    res = mss.validate(*args.values())
    assert not res["validated"] and not res["checks"][check]
    assert sum(not v for v in res["checks"].values()) == 1


def test_sensex_check_uses_oos_dates_only(monkeypatch):
    monkeypatch.setattr(mss, "MIN_TRAIN", 1)
    monkeypatch.setattr(mss, "MIN_OOS", 1)
    a = mss.ALL_STRATEGIES[0]
    rows = [_row("t1", 1.0, a), _row("t2", 1.0, a), _row("o1", 1.0, a),
            _row("t1", 5.0, a, "SENSEX"), _row("t2", 5.0, a, "SENSEX"), _row("o1", -1.0, a, "SENSEX")]
    frozen = mss.freeze([a], mss.train_table(rows[:2]), "o1")
    ev = mss.evaluate_frozen(frozen, rows, {"t1", "t2"}, {"o1"}, [0.0, 0.1])[0]
    assert not ev["validation"]["checks"]["sensex_oos_net_positive"]     # SENSEX train days can't rescue it


def test_hit_rate_is_net_of_the_row_cost():
    rows = [{"day": "d1", "r": 0.01, "entry": 10000.0, "stop_d": 10.0, "hold_bars": 1,
             "strategy": "X", "index_id": "NIFTY50", "reason": "time"}]
    assert mss.stats(rows, mss.BASE_COST_FRAC)["hit"] == 0.0           # +0.01 R gross, negative after cost


# ── additional execution / mode edge cases ──────────────────────────────────
class TestSimulateMore:
    def test_entry_bar_touching_both_is_stop_long_and_short(self):
        rows = flat(5) + [(PX, 10061.0, 9969.0, PX)]
        res = mss.simulate(_mb(rows), 5, 1, 9970.0, 10060.0)
        assert (res["reason"], res["r"], res["hold_bars"]) == ("stop", -1.0, 1)
        res = mss.simulate(_mb(rows), 5, -1, 10030.0, 9940.0)
        assert (res["reason"], res["r"], res["hold_bars"]) == ("stop", -1.0, 1)

    def test_short_time_exit_sign(self):
        rows = flat(32) + [(PX, PX + 9, 9991.0, 9991.0)]
        res = mss.simulate(make_bars(D0, rows), 5, -1, 10030.0, 9940.0)
        assert res["reason"] == "time" and res["r"] == pytest.approx(9 / 30)

    def test_short_target_gap_no_bonus(self):
        rows = flat(5) + [(PX, PX + 5, PX - 5, PX - 5), (9930.0, 10040.0, 9920.0, 9925.0)]
        res = mss.simulate(_mb(rows), 5, -1, 10030.0, 9940.0)
        assert res["reason"] == "target_gap" and res["r"] == pytest.approx(2.0)

    def test_open_exactly_at_stop_is_a_stop_gap(self):
        rows = flat(5) + [(PX, PX + 5, PX - 5, PX), (9970.0, 9975.0, 9965.0, 9970.0)]
        res = mss.simulate(_mb(rows), 5, 1, 9970.0, 10060.0)
        assert res["reason"] == "stop_gap" and res["r"] == pytest.approx(-1.0)


def test_orb30_and_mid_stop_are_distinct_from_orb15_opp():
    rows = [(PX, 10010.0, 9990.0, PX)] * 3                       # 09:15-09:25: 9990-10010
    rows += [(PX, 10020.0, 9980.0, PX), (PX, 10008.0, 9992.0, 10015.0)]   # 09:30 widens, 09:35 closes 10015
    rows += [(10012.0, 10014.0, 10010.0, 10012.0)] * 70            # 09:40 opens 10012
    daily = _daily_for(date(2025, 1, 3), D0)
    trades, _ = mss.build_trades("NIFTY50", make_bars(D0, rows), daily)
    by = {t["strategy"]: t for t in trades}
    assert by["ORB15/OPP/1.0R/ALL"]["entry"] == pytest.approx(10012.0)
    assert by["ORB15/OPP/1.0R/ALL"]["stop_d"] == pytest.approx(10012.0 - 9990.0)
    assert by["ORB15/MID/1.0R/ALL"]["stop_d"] == pytest.approx(10012.0 - 10000.0)
    assert not [k for k in by if k.startswith("ORB30/")]           # 10015 is inside the 30-minute range


def test_orb_cutoff_entry_is_next_bar_open_not_signal_close():
    rows = [(PX, PX + 10, PX - 10, PX)] * 6 + flat(14, w=2.0)
    rows += [(PX, 10016.0, PX - 2, 10015.0)]                       # 10:55 signal closes 10015
    rows += [(10018.0, 10020.0, 10016.0, 10018.0)] * 54            # 11:00 opens 10018
    trades, _ = mss.build_trades("NIFTY50", make_bars(D0, rows), _daily_for(date(2025, 1, 3), D0))
    t = next(x for x in trades if x["strategy"] == "ORB15/OPP/1.0R/ALL")
    assert t["entry"] == pytest.approx(10018.0)


class TestFadeModes:
    prev = date(2025, 1, 3)

    def _daily(self, pc=9950.0):
        return _daily_for(self.prev, close=pc) | _daily_for(D0)

    def _brk_day(self, signal_at: time):
        # gap-up; first bar low 9990; the bar starting `signal_at` is the first to close below 9990
        rows = [(10000.0, 10020.0, 9990.0, 10000.0), (10000.0, 10025.0, 9995.0, 10005.0),
                (10005.0, 10010.0, 9996.0, 10002.0)]
        j = idx_of(signal_at)
        rows += [(10002.0, 10006.0, 9996.0, 10002.0)] * (j - 3)
        rows += [(10002.0, 10004.0, 9985.0, 9988.0), (9986.0, 9990.0, 9980.0, 9984.0)]
        rows += [(9984.0, 9987.0, 9981.0, 9984.0)] * (75 - len(rows))
        return make_bars(D0, rows)

    def test_brk_enters_on_open_after_first_break(self):
        trades, _ = mss.build_trades("NIFTY50", self._brk_day(time(9, 40)), self._daily(9900.0))   # R = 86/39
        t = next(x for x in trades if x["strategy"] == "FADE/BRK/gap>=0.3%")
        assert t["dir"] == -1 and t["entry"] == pytest.approx(9986.0)      # 09:45 open
        assert t["stop_d"] == pytest.approx(10025.0 - 9986.0)             # highest high before entry

    def test_brk_respects_cutoff(self):
        trades, _ = mss.build_trades("NIFTY50", self._brk_day(time(10, 55)), self._daily(9900.0))
        assert [x for x in trades if x["strategy"] == "FADE/BRK/gap>=0.3%"]
        trades, _ = mss.build_trades("NIFTY50", self._brk_day(time(11, 0)), self._daily(9900.0))
        assert not [x for x in trades if x["strategy"] == "FADE/BRK/gap>=0.3%"]

    def test_rej_only_when_first_15m_rejects_gap(self):
        rejected = fade_day(D0, 10000.0)                                  # 09:25 close 9995 < open 10000
        not_rej = fade_day(D0, 10000.0, entry_px=10005.0)                 # 09:25 close 10005 >= open
        t1, _ = mss.build_trades("NIFTY50", rejected, self._daily())
        t2, _ = mss.build_trades("NIFTY50", not_rej, self._daily())
        assert [x for x in t1 if x["strategy"] == "FADE/REJ/gap>=0.3%"]
        assert not [x for x in t2 if x["strategy"] == "FADE/REJ/gap>=0.3%"]
        assert [x for x in t2 if x["strategy"] == "FADE/RAW/gap>=0.3%"]

    @pytest.mark.parametrize("pc, trades_expected", [(9975.0, False), (9970.0, True), (9845.0, True), (9844.0, False)])
    def test_r_bounds_inclusive(self, pc, trades_expected):
        # open 10010 keeps the gap >= 0.3 % in every case, so only R decides.
        # entry 9995, stop 10020 (stop_d 25): R = (9995 - pc) / 25 → 0.8, 1.0, 6.0, 6.04
        bars = fade_day(D0, 10010.0)
        assert (10010.0 - pc) / pc >= 0.003
        trades, diag = mss.build_trades("NIFTY50", bars, self._daily(pc))
        got = [x for x in trades if x["strategy"] == "FADE/RAW/gap>=0.3%"]
        assert bool(got) is trades_expected
        if not trades_expected:
            assert diag["fade_r_out_of_range"] >= 1

    def test_gap_down_skip_when_prev_close_touched(self):
        rows = [(10000.0, 10051.0, 9980.0, 10000.0), (10000.0, 10008.0, 9995.0, 10002.0),
                (10002.0, 10008.0, 9998.0, 10005.0)]
        rows += [(10005.0, 10008.0, 10002.0, 10005.0)] * 72
        trades, diag = mss.build_trades("NIFTY50", make_bars(D0, rows), self._daily(10050.0))
        assert not [t for t in trades if t["strategy"].startswith("FADE")]
        assert diag["fade_prev_close_touched"] >= 1

    def test_entry_through_prev_close_skipped(self):
        bars = fade_day(D0, 10000.0, or_lo=9993.0, entry_px=9990.0)
        bars[2]["low"] = 9993.0
        trades, diag = mss.build_trades("NIFTY50", bars, self._daily(9960.0))
        assert [t for t in trades if t["strategy"] == "FADE/RAW/gap>=0.3%"]     # control: trades normally
        bars2 = fade_day(D0, 10040.0, or_lo=9993.0, or_hi=10045.0, entry_px=9990.0)
        trades2, diag2 = mss.build_trades("NIFTY50", bars2, self._daily(9991.0))   # entry 9990 already past pc 9991
        assert not [t for t in trades2 if t["strategy"].startswith("FADE")]
        assert diag2["fade_entry_through_prev_close"] >= 1


def test_narrow_or_uses_rolling_60_day_40th_percentile():
    hist = [6.0 + (k % 20) for k in range(60)]                   # varied widths (half-ranges)
    widths_pct = [2 * h / PX for h in hist]
    thr_half = float(np.percentile(widths_pct, 40)) * PX / 2
    # 30 extremely wide days BEFORE the 60-day window must not matter (rolling, not expanding)
    for prefix in ([], [60.0] * 30):
        for today, expect in ((thr_half - 0.01, True), (thr_half + 0.01, False)):
            bars, (days, daily) = narrow_series(prefix + hist + [today])
            trades, _ = mss.build_trades("NIFTY50", bars, daily)
            got = any(t["day"] == days[-1].isoformat() and t["strategy"] == "ORB15/OPP/1.0R/NARROW_OR"
                      for t in trades)
            assert got is expect, (len(prefix), today, thr_half)
