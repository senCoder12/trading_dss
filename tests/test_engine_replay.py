"""
Phase 3A — replay of the LIVE decision engine (src/backtest/engine_replay.py).

The end-to-end tests run the real ``DecisionEngine.run_full_cycle`` over a
small synthetic DB (a sawtooth up-trend) with a DIAGNOSTIC permissive
calibration, so the trade path is exercised.  Nothing here says anything
about edge.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.backtest.engine_replay import (
    Bar,
    ReplayContractMaster,
    ReplayExecutor,
    ReplayRiskManager,
    VolatilityModel,
    build_synthetic_chain,
    expiry_on_or_after,
    following_expiry,
    nearest_expiry,
    reason_key,
    run_replay,
    synthetic_half_spread,
    write_permissive_calibration,
)
from src.database.db_manager import DatabaseManager
from src.engine.option_contract import (
    ContractQuote, bs_price_greeks, estimate_premium, premium_at, select_contract,
    time_to_expiry_years,
)
from src.engine.option_costs import round_trip_cost
from src.engine.risk_manager import RiskConfig, _OpenPosition

_IST = ZoneInfo("Asia/Kolkata")
SESSIONS = [date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2)]
REPLAY_START, REPLAY_END = "2026-10-30", "2026-11-02"


# ── Synthetic DB ──────────────────────────────────────────────────────────


def _seed_db(path) -> dict:
    """3 sessions of 5m NIFTY50 bars: +0.08 % bars with a −0.12 % pullback
    every 4th bar (a clear up-trend whose closes sit near the prior low)."""
    db = DatabaseManager(db_path=path, wal_mode=False)
    db.connect()
    db.initialise_schema()
    ts0 = datetime(2026, 10, 1, tzinfo=_IST).isoformat()
    db.execute(
        """
        INSERT OR IGNORE INTO index_master
            (id, display_name, nse_symbol, yahoo_symbol, exchange, lot_size,
             has_options, option_symbol, sector_category, is_active,
             created_at, updated_at)
        VALUES ('NIFTY50','NIFTY 50','NIFTY 50','^NSEI','NSE',65,1,'NIFTY',
                'broad_market',1,?,?)
        """,
        (ts0, ts0),
    )
    bars: dict = {}
    px, k = 22000.0, 0
    for day in SESSIONS:
        for j in range(75):
            t = datetime.combine(day, dtime(9, 15), tzinfo=_IST) + timedelta(minutes=5 * j)
            o = px
            c = o * (1 - 0.0012) if k % 4 == 3 else o * (1 + 0.0008)
            h, lo = max(o, c) * 1.0008, min(o, c) * 0.9992
            db.execute(
                "INSERT INTO price_data (index_id, timestamp, open, high, low, close, volume, "
                "source, timeframe) VALUES ('NIFTY50', ?, ?, ?, ?, ?, 0, 'test', '5m')",
                (t.isoformat(), o, h, lo, c),
            )
            bars[t] = {"open": o, "high": h, "low": lo, "close": c}
            px, k = c, k + 1
    db.close()
    return bars


def _sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def replay_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("engine_replay")
    src = base / "source.db"
    bars = _seed_db(src)
    os.chmod(src, 0o444)  # any write attempt would fail loudly
    digest, mtime = _sha256(src), os.stat(src).st_mtime_ns
    cal = write_permissive_calibration(base / "diag_cal.json")
    result = run_replay(
        src, "NIFTY50", REPLAY_START, REPLAY_END,
        calibration_path=cal, capital=500_000, out_dir=base / "out",
    )
    return {"src": src, "bars": bars, "digest": digest, "mtime": mtime,
            "result": result, "base": base}


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


# ── End-to-end on the live engine ─────────────────────────────────────────


class TestReplayLifecycle:
    def test_source_db_byte_for_byte_unchanged(self, replay_env):
        assert _sha256(replay_env["src"]) == replay_env["digest"]
        assert os.stat(replay_env["src"]).st_mtime_ns == replay_env["mtime"]
        assert not os.path.exists(str(replay_env["src"]) + "-wal")
        assert not os.path.exists(str(replay_env["src"]) + "-journal")

    def test_at_least_one_full_trade_lifecycle(self, replay_env):
        r = replay_env["result"]
        assert r.meta["mode"] == "diagnostic"
        assert r.trades, r.decision_counts
        assert r.decision_counts.get("TRADE", 0) == len(r.trades)
        for t in r.trades:
            assert t["exit_time"] > t["entry_time"] > t["decision_time"]
            assert t["exit_reason"] in {"SL", "TRAILING_SL", "TARGET", "TIME", "TIME_STOP"}
            assert t["premium_source"] == "model"
            assert t["option_type"] == ("CE" if t["direction"] == "BUY_CALL" else "PE")
            assert t["strike"] % 50 == 0
            assert t["qty"] == t["lots"] * 65
            # NIFTY weekly expiry: a Tuesday on/after the trade date
            exp = date.fromisoformat(t["expiry"])
            assert exp.weekday() == 1 and exp >= _ts(t["entry_time"]).date()

    def test_entry_fills_at_next_bar_open_repriced(self, replay_env):
        bars = replay_env["bars"]
        for t in replay_env["result"].trades:
            decision, entry = _ts(t["decision_time"]), _ts(t["entry_time"])
            # decision at the close of bar s; fill at the open of bar s+5m
            assert entry == decision + timedelta(seconds=1)
            assert entry in bars
            assert t["entry_spot"] == pytest.approx(bars[entry]["open"])
            tte = time_to_expiry_years(date.fromisoformat(t["expiry"]), decision)
            quote = ContractQuote(
                index_id="NIFTY50", expiry=date.fromisoformat(t["expiry"]), strike=t["strike"],
                option_type=t["option_type"], ltp=t["decision_mid"], bid=0, ask=0,
                mid=t["decision_mid"], half_spread=0, spread_known=False, iv=t["iv"],
                iv_source="chain", oi=0,
                greeks=bs_price_greeks(t["decision_spot"], t["strike"], tte, t["iv"], t["option_type"]),
                priced_at=decision, spot=t["decision_spot"],
            )
            expected_mid = premium_at(quote, t["entry_spot"], entry)
            assert t["entry_mid"] == pytest.approx(expected_mid, abs=0.02)
            assert t["entry_premium"] == pytest.approx(
                expected_mid + max(expected_mid * 0.005, 0.05), abs=0.02,
            )

    def test_premium_pnl_net_of_spread_and_costs(self, replay_env):
        r = replay_env["result"]
        for t in r.trades:
            assert t["exit_premium"] == pytest.approx(max(t["exit_mark"] - t["half_spread"], 0.05), abs=1e-3)
            costs = round_trip_cost(t["entry_premium"], t["exit_premium"], t["qty"], "NSE").total
            assert t["costs"] == pytest.approx(costs, abs=0.02)
            assert t["costs"] > 40  # 2 × ₹20 brokerage + statutory
            expected = (t["exit_premium"] - t["entry_premium"]) * t["qty"] - t["costs"]
            assert t["net_pnl"] == pytest.approx(expected, abs=0.05)
        assert sum(r.daily_pnl.values()) == pytest.approx(sum(t["net_pnl"] for t in r.trades), abs=0.05)
        assert r.equity_curve[-1]["equity"] == pytest.approx(
            500_000 + sum(t["net_pnl"] for t in r.trades), abs=0.05,
        )

    def test_entry_window_and_flat_by_1515(self, replay_env):
        for t in replay_env["result"].trades:
            decision, entry, exit_ = (_ts(t[k]) for k in ("decision_time", "entry_time", "exit_time"))
            assert decision.time() >= dtime(9, 30)
            assert decision.time() < dtime(15, 0)
            assert dtime(9, 30) <= entry.time() <= dtime(15, 0)
            assert exit_.date() == entry.date()
            assert exit_.time() <= dtime(15, 15)

    def test_no_trade_reasons_are_counted(self, replay_env):
        counts = replay_env["result"].decision_counts
        no_trade = {k: v for k, v in counts.items() if k != "TRADE"}
        assert no_trade and all(v > 0 for v in no_trade.values())
        # the daily cap and the risk manager's position cap are live rules too
        assert any(k.startswith("RISK: ") for k in counts) or "Max signals for today reached" in counts
        assert sum(counts.values()) == replay_env["result"].meta["decisions"]
        assert replay_env["result"].meta["decisions"] == 2 * 75

    def test_outputs_written(self, replay_env):
        out = replay_env["base"] / "out"
        payload = json.loads((out / "replay_result.json").read_text())
        assert payload["summary"]["trades"] == len(replay_env["result"].trades)
        assert "NOT evidence of edge" in payload["meta"]["warning"]
        assert (out / "trades.csv").read_text().count("\n") == len(replay_env["result"].trades) + 1

    def test_unvalidated_calibration_trades_nothing(self, replay_env, tmp_path):
        cal = write_permissive_calibration(tmp_path / "unvalidated.json")
        payload = json.loads(cal.read_text())
        for m in payload["models"].values():
            m["validated"] = False
        payload.pop("diagnostic")
        cal.write_text(json.dumps(payload))
        r = run_replay(replay_env["src"], "NIFTY50", "2026-11-02", "2026-11-02",
                       calibration_path=cal, stride=3)
        assert r.trades == []
        assert any(k.startswith("Model not tradeable") for k in r.decision_counts)
        assert r.meta["mode"] == "custom"
        assert _sha256(replay_env["src"]) == replay_env["digest"]

    def test_deterministic_across_runs(self, replay_env):
        cal = write_permissive_calibration(replay_env["base"] / "det_cal.json")
        runs = [
            run_replay(replay_env["src"], "NIFTY50", "2026-11-02", "2026-11-02",
                       calibration_path=cal, stride=2)
            for _ in range(2)
        ]
        strip = lambda r: [{k: v for k, v in t.items() if k != "signal_id"} for t in r.trades]  # noqa: E731
        assert strip(runs[0]) == strip(runs[1])
        assert runs[0].decision_counts == runs[1].decision_counts


# ── Bar-level execution (intrabar triggers) ───────────────────────────────


def _rm(tmp_path) -> ReplayRiskManager:
    db = DatabaseManager(db_path=tmp_path / "rm.db", wal_mode=False)
    db.connect()
    db.initialise_schema()
    return ReplayRiskManager(db, config=RiskConfig(total_capital=500_000))


def _call_position(rm: ReplayRiskManager, ex: ReplayExecutor, entry_at: datetime,
                   premium_stop_frac: float = 0.97) -> _OpenPosition:
    expiry = date(2026, 11, 10)
    iv, spot, strike = 0.15, 22000.0, 22000.0
    mid = bs_price_greeks(spot, strike, time_to_expiry_years(expiry, entry_at), iv, "CE").price
    sl, tgt = 21934.0, 22200.0
    stop_prem = estimate_premium("CE", strike, expiry, iv, sl, entry_at + timedelta(minutes=10),
                                 mid, spot, entry_at) * premium_stop_frac
    half = synthetic_half_spread(mid)
    pos = _OpenPosition(
        db_id=-1, signal_id="T1", index_id="NIFTY50", signal_type="BUY_CALL",
        entry_price=spot, current_sl=sl, target_price=tgt, lots=1, lot_size=65,
        entry_time=entry_at, confidence_level="HIGH", option_type="CE", strike=strike,
        expiry=expiry, entry_premium=mid + half, premium_mid=mid, half_spread=half,
        premium_stop=stop_prem, initial_premium_stop=stop_prem, premium_target=mid * 1.6,
        entry_iv=iv, last_premium=mid,
    )
    rm._open_positions["T1"] = pos
    ex._filled.append("T1")
    return pos


def _bar(start: datetime, o, h, lo, c) -> Bar:
    return Bar(start, o, h, lo, c)


class TestIntrabar:
    T0 = datetime(2026, 11, 4, 10, 0, tzinfo=_IST)

    def test_low_pierces_spot_stop_but_closes_above(self, tmp_path):
        rm = _rm(tmp_path)
        ex = ReplayExecutor(rm)
        pos = _call_position(rm, ex, self.T0)
        bar = _bar(self.T0 + timedelta(minutes=5), 21995.0, 22010.0, 21900.0, 21990.0)
        assert bar.close > pos.current_sl  # a close-only check would HOLD
        ex.on_open(bar)
        assert not ex.trades
        ex.on_intrabar(bar)
        assert len(ex.trades) == 1
        t = ex.trades[0]
        assert t["exit_reason"] == "SL" and t["exit_trigger"] == "intrabar_spot_sl"
        assert t["exit_spot"] == pytest.approx(21934.0)
        expected_mark = estimate_premium("CE", 22000.0, date(2026, 11, 10), 0.15, 21934.0,
                                         bar.end, pos.premium_mid, 22000.0, self.T0)
        assert t["exit_mark"] == pytest.approx(expected_mark, abs=1e-3)
        assert t["exit_premium"] == pytest.approx(expected_mark - pos.half_spread, abs=1e-3)
        assert t["exit_time"] == bar.end.isoformat()
        assert "T1" not in rm._open_positions
        # the loss is on the replay ledger → cooldown/loss-streak see it
        assert rm._closed_trades_today(bar.end)[0]["outcome"] == "LOSS"

    def test_both_touched_in_one_bar_assumes_stop_first(self, tmp_path):
        rm = _rm(tmp_path)
        ex = ReplayExecutor(rm)
        _call_position(rm, ex, self.T0)
        bar = _bar(self.T0 + timedelta(minutes=5), 22000.0, 22250.0, 21900.0, 22100.0)
        ex.on_intrabar(bar)
        assert [t["exit_reason"] for t in ex.trades] == ["SL"]

    def test_premium_stop_tighter_than_spot_stop(self, tmp_path):
        rm = _rm(tmp_path)
        ex = ReplayExecutor(rm)
        pos = _call_position(rm, ex, self.T0, premium_stop_frac=1.15)  # above mark at spot SL
        bar = _bar(self.T0 + timedelta(minutes=5), 21995.0, 22000.0, 21920.0, 21990.0)
        ex.on_intrabar(bar)
        t = ex.trades[0]
        assert t["exit_trigger"] == "intrabar_premium_sl"
        assert t["exit_mark"] == pytest.approx(pos.premium_stop, abs=1e-3)

    def test_intrabar_target(self, tmp_path):
        rm = _rm(tmp_path)
        ex = ReplayExecutor(rm)
        pos = _call_position(rm, ex, self.T0)
        bar = _bar(self.T0 + timedelta(minutes=5), 22010.0, 22230.0, 21990.0, 22150.0)
        ex.on_intrabar(bar)
        t = ex.trades[0]
        assert t["exit_reason"] == "TARGET"
        # the first trigger touched on the way up is the lower premium
        spot_tgt_mark = estimate_premium("CE", 22000.0, date(2026, 11, 10), 0.15, 22200.0,
                                         bar.end, pos.premium_mid, 22000.0, self.T0)
        assert t["exit_mark"] == pytest.approx(min(spot_tgt_mark, pos.premium_target), abs=1e-3)
        assert t["net_pnl"] > 0

    def test_gap_through_stop_exits_at_open(self, tmp_path):
        rm = _rm(tmp_path)
        ex = ReplayExecutor(rm)
        pos = _call_position(rm, ex, self.T0)
        bar = _bar(self.T0 + timedelta(minutes=5), 21850.0, 21880.0, 21800.0, 21870.0)
        ex.on_open(bar)
        t = ex.trades[0]
        assert t["exit_trigger"] == "open" and t["exit_time"] == bar.start.isoformat()
        at_open = estimate_premium("CE", 22000.0, date(2026, 11, 10), 0.15, 21850.0,
                                   bar.start, pos.premium_mid, 22000.0, self.T0)
        assert t["exit_mark"] == pytest.approx(at_open, abs=0.01)  # monitor rounds to paise

    def test_force_exit_at_1515_on_bar_close(self, tmp_path):
        rm = _rm(tmp_path)
        ex = ReplayExecutor(rm)
        t0 = datetime(2026, 11, 4, 14, 30, tzinfo=_IST)
        _call_position(rm, ex, t0)
        bar = _bar(datetime(2026, 11, 4, 15, 10, tzinfo=_IST), 22000.0, 22010.0, 21990.0, 22005.0)
        ex.on_open(bar)
        ex.on_intrabar(bar)
        assert not ex.trades
        ex.on_close(bar)
        assert ex.trades[0]["exit_reason"] == "TIME"
        assert ex.trades[0]["exit_time"] == "2026-11-04T15:15:00+05:30"

    def test_fill_reprices_to_next_open(self, tmp_path):
        rm = _rm(tmp_path)
        ex = ReplayExecutor(rm)
        decision = datetime(2026, 11, 4, 10, 4, 59, tzinfo=_IST)
        pos = _call_position(rm, ex, decision)
        ex._filled.remove("T1")
        decision_mid = pos.premium_mid
        nxt = _bar(datetime(2026, 11, 4, 10, 5, tzinfo=_IST), 22040.0, 22050.0, 22030.0, 22045.0)
        assert ex.fill("T1", nxt)
        expected = estimate_premium("CE", 22000.0, date(2026, 11, 10), 0.15, 22040.0, nxt.start,
                                    decision_mid, 22000.0, decision)
        assert pos.premium_mid == pytest.approx(expected, abs=1e-3)
        assert pos.entry_premium == pytest.approx(expected + synthetic_half_spread(expected), abs=1e-3)
        assert pos.entry_price == 22040.0 and pos.entry_time == nxt.start


# ── Calendar, chain, volatility ───────────────────────────────────────────


class TestExpiryCalendar:
    def test_nifty_weekly_tuesday(self):
        assert expiry_on_or_after("NIFTY50", date(2026, 11, 5)) == date(2026, 11, 10)  # Thu → Tue
        assert expiry_on_or_after("NIFTY50", date(2026, 11, 10)) == date(2026, 11, 10)

    def test_banknifty_monthly_last_tuesday(self):
        assert expiry_on_or_after("BANKNIFTY", date(2026, 11, 5)) == date(2026, 11, 24)
        assert expiry_on_or_after("BANKNIFTY", date(2026, 11, 25)) == date(2026, 12, 29)

    def test_roll_after_1530_on_expiry_day(self):
        tue = date(2026, 11, 10)
        assert nearest_expiry("NIFTY50", datetime.combine(tue, dtime(15, 29), tzinfo=_IST)) == tue
        assert nearest_expiry("NIFTY50", datetime.combine(tue, dtime(15, 30), tzinfo=_IST)) == date(2026, 11, 17)
        assert following_expiry("BANKNIFTY", date(2026, 11, 24)) == date(2026, 12, 29)

    def test_contract_master_stand_in(self):
        cm = ReplayContractMaster()
        assert cm.lot_size("NIFTY") == 65 and cm.lot_size("BANKNIFTY") == 30
        assert cm.is_expiry_day("NIFTY", date(2026, 11, 10)) is True
        assert cm.is_expiry_day("NIFTY", date(2026, 11, 11)) is False


class TestSyntheticChain:
    def test_chain_is_tradeable_and_consistent(self):
        now = datetime(2026, 11, 4, 11, 0, tzinfo=_IST)
        chain = build_synthetic_chain("NIFTY50", 22013.0, now, 0.14, date(2026, 11, 10))
        ks = [s.strike_price for s in chain.strikes]
        assert ks == sorted(ks) and {b - a for a, b in zip(ks, ks[1:])} == {50.0}
        assert 22000.0 in ks and len(ks) == 41
        for s in chain.strikes:
            assert s.ce_oi > 0 and s.pe_oi > 0
            assert s.ce_iv == pytest.approx(14.0) and s.pe_iv == pytest.approx(14.0)
            for ltp, bid, ask in ((s.ce_ltp, s.ce_bid_price, s.ce_ask_price),
                                  (s.pe_ltp, s.pe_bid_price, s.pe_ask_price)):
                half = synthetic_half_spread(ltp)
                assert half >= 0.05
                assert ask == pytest.approx(ltp + half)
                assert bid == pytest.approx(max(ltp - half, 0.0))
        q, why = select_contract(chain, "BUY_CALL", 22013.0, now)
        assert q is not None, why
        assert q.spread_known and q.iv == pytest.approx(0.14)
        assert abs(q.greeks.delta - 0.45) < 0.1

    def test_banknifty_interval(self):
        now = datetime(2026, 11, 4, 11, 0, tzinfo=_IST)
        chain = build_synthetic_chain("BANKNIFTY", 50040.0, now, 0.16, date(2026, 11, 24))
        ks = [s.strike_price for s in chain.strikes]
        assert {b - a for a, b in zip(ks, ks[1:])} == {100.0}


class TestVolatilityModel:
    def _bars(self, n_days=2):
        out, px = [], 22000.0
        for d in (date(2026, 11, 2), date(2026, 11, 3))[:n_days]:
            for j in range(75):
                t = datetime.combine(d, dtime(9, 15), tzinfo=_IST) + timedelta(minutes=5 * j)
                c = px * (1.001 if j % 2 else 0.999)
                out.append(Bar(t, px, max(px, c), min(px, c), c))
                px = c
        return out

    def test_vix_previous_session_eod_used_not_same_day(self):
        bars = self._bars()
        vix = [(datetime(2026, 11, 2, 0, 0, tzinfo=_IST), 14.0),
               (datetime(2026, 11, 3, 0, 0, tzinfo=_IST), 30.0)]  # 11-03 close: unknown intraday
        vm = VolatilityModel(bars, vix)
        iv, src = vm.iv_at(datetime(2026, 11, 3, 11, 0, tzinfo=_IST))
        assert (iv, src) == (pytest.approx(0.14), "vix")

    def test_realised_vol_fallback_with_floor(self):
        bars = self._bars()
        vm = VolatilityModel(bars, [])
        iv, src = vm.iv_at(datetime(2026, 11, 3, 14, 0, tzinfo=_IST))
        assert src == "realised_vol"
        # 0.1 % per bar alternating → σ ≈ 0.001 × √(75×252) ≈ 13.7 %; × 1.1
        assert iv == pytest.approx(0.001 * (75 * 252) ** 0.5 * 1.1, rel=0.05)
        flat = [Bar(b.start, 100.0, 100.0, 100.0, 100.0) for b in bars]
        assert VolatilityModel(flat, []).iv_at(bars[-1].end)[0] == pytest.approx(0.10)


class TestReasonKeys:
    def test_numbers_bucketed_names_kept(self):
        assert reason_key("Insufficient R:R (0.38 < 1.30 minimum) — skip") == "Insufficient R:R"
        assert reason_key("Max positions for NIFTY50 reached (1/1)") == "Max positions for NIFTY50 reached"
        assert reason_key("Score +0.12 in dead zone (|score| < 0.25)") == "Score # in dead zone"
