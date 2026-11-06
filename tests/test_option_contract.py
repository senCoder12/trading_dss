"""Tests for option pricing/selection (option_contract) and costs (option_costs)."""

from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.data.options_chain import OptionsChainData, OptionStrike
from src.engine.option_contract import (
    SelectionParams,
    bs_price_greeks,
    implied_vol,
    premium_at,
    select_contract,
    strike_interval,
    time_to_expiry_years,
)
from src.engine.option_costs import BSE_OPTIONS, round_trip_cost

_IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 11, 5, 10, 30, tzinfo=_IST)
EXPIRY = date(2026, 11, 10)
SPOT = 24_500.0
_STRIKE_FIELDS = {f.name for f in dataclasses.fields(OptionStrike)}


def _strike(k: float, iv: float = 0.14, oi: int = 100_000, spread: float = 0.01, **over) -> OptionStrike:
    t = time_to_expiry_years(EXPIRY, NOW)
    ce = bs_price_greeks(SPOT, k, t, iv, "CE").price
    pe = bs_price_greeks(SPOT, k, t, iv, "PE").price
    kw = {n: 0 for n in _STRIKE_FIELDS}
    kw.update(
        strike_price=float(k), ce_ltp=round(ce, 2), pe_ltp=round(pe, 2),
        ce_iv=iv * 100, pe_iv=iv * 100, ce_oi=oi, pe_oi=oi,
        ce_bid_price=round(ce * (1 - spread / 2), 2), ce_ask_price=round(ce * (1 + spread / 2), 2),
        pe_bid_price=round(pe * (1 - spread / 2), 2), pe_ask_price=round(pe * (1 + spread / 2), 2),
    )
    kw.update(over)
    return OptionStrike(**kw)


def _chain(strikes=None) -> OptionsChainData:
    strikes = strikes or [_strike(k) for k in range(24_000, 25_050, 50)]
    return OptionsChainData(
        index_id="NIFTY50", spot_price=SPOT, timestamp=NOW, expiry_date=EXPIRY,
        strikes=tuple(strikes), available_expiries=(EXPIRY,),
    )


# ── Pricing primitives ────────────────────────────────────────────────────


class TestPricing:
    def test_put_call_parity(self):
        t, k, r = 30 / 365, 24_500.0, 0.065
        c = bs_price_greeks(SPOT, k, t, 0.15, "CE", r).price
        p = bs_price_greeks(SPOT, k, t, 0.15, "PE", r).price
        assert c - p == pytest.approx(SPOT - k * math.exp(-r * t), abs=1e-6)

    def test_delta_signs_and_theta_negative(self):
        t = 5 / 365
        ce = bs_price_greeks(SPOT, SPOT, t, 0.14, "CE")
        pe = bs_price_greeks(SPOT, SPOT, t, 0.14, "PE")
        assert 0.45 < ce.delta < 0.6 and -0.55 < pe.delta < -0.4
        assert ce.theta_per_day < 0 and pe.theta_per_day < 0
        assert ce.gamma > 0

    def test_implied_vol_round_trip(self):
        t = 5 / 365
        price = bs_price_greeks(SPOT, 24_600, t, 0.17, "CE").price
        assert implied_vol(price, SPOT, 24_600, t, "CE") == pytest.approx(0.17, abs=1e-3)

    def test_implied_vol_rejects_price_below_intrinsic(self):
        assert implied_vol(1.0, SPOT, 24_000, 5 / 365, "CE") is None

    def test_time_to_expiry_is_calendar_to_1530_and_floored(self):
        t = time_to_expiry_years(EXPIRY, NOW)
        expected_days = (datetime(2026, 11, 10, 15, 30, tzinfo=_IST) - NOW).total_seconds() / 86400
        assert t * 365 == pytest.approx(expected_days)
        after = datetime(2026, 11, 10, 15, 29, tzinfo=_IST)
        assert time_to_expiry_years(EXPIRY, after) == pytest.approx(15 * 60 / (365 * 86400))

    def test_strike_interval_inferred(self):
        assert strike_interval([24000, 24050, 24100, 24200]) == 50
        assert strike_interval([81000, 81100, 81200]) == 100
        assert strike_interval([12000, 12025, 12050]) == 25
        assert strike_interval([]) is None


# ── Contract selection ────────────────────────────────────────────────────


class TestSelectContract:
    def test_picks_strike_closest_to_target_delta(self):
        q, why = select_contract(_chain(), "BUY_CALL", SPOT, NOW, SelectionParams(target_abs_delta=0.40))
        assert why == "" and q.option_type == "CE"
        assert q.strike > SPOT  # 0.40Δ call is OTM
        assert abs(q.greeks.delta - 0.40) < 0.1

    def test_put_uses_pe_and_negative_delta(self):
        q, _ = select_contract(_chain(), "BUY_PUT", SPOT, NOW, SelectionParams(target_abs_delta=0.40))
        assert q.option_type == "PE" and q.greeks.delta < 0 and q.strike < SPOT

    def test_illiquid_and_wide_spread_strikes_skipped(self):
        strikes = [_strike(k) for k in range(24_000, 25_050, 50)]
        for i, s in enumerate(strikes):
            if s.strike_price in (24_500, 24_550):
                strikes[i] = _strike(s.strike_price, oi=0)            # no open interest
            if s.strike_price == 24_600:
                strikes[i] = _strike(s.strike_price, spread=0.20)     # 20 % spread
        q, _ = select_contract(_chain(strikes), "BUY_CALL", SPOT, NOW, SelectionParams(target_abs_delta=0.50))
        assert q.strike not in (24_500, 24_550, 24_600)

    def test_missing_iv_is_solved_from_price(self):
        strikes = [_strike(k, ce_iv=0.0) for k in range(24_000, 25_050, 50)]
        q, _ = select_contract(_chain(strikes), "BUY_CALL", SPOT, NOW, SelectionParams())
        assert q.iv_source == "solved" and q.iv == pytest.approx(0.14, abs=0.01)

    def test_unknown_spread_is_assumed_and_flagged(self):
        strikes = [_strike(k, ce_bid_price=0.0, ce_ask_price=0.0) for k in range(24_000, 25_050, 50)]
        q, _ = select_contract(_chain(strikes), "BUY_CALL", SPOT, NOW, SelectionParams())
        assert q.spread_known is False
        assert q.half_spread == pytest.approx(q.mid * 0.005)

    def test_no_tradeable_strike_reports_reason(self):
        strikes = [_strike(k, ce_ltp=1.0, ce_bid_price=0.0, ce_ask_price=0.0) for k in range(24_000, 25_050, 50)]
        q, why = select_contract(_chain(strikes), "BUY_CALL", SPOT, NOW, SelectionParams())
        assert q is None and "premium below minimum" in why


class TestPremiumAt:
    def test_anchored_to_market_mid(self):
        q, _ = select_contract(_chain(), "BUY_CALL", SPOT, NOW, SelectionParams())
        assert premium_at(q, SPOT, NOW) == pytest.approx(q.mid)

    def test_theta_lowers_premium_at_unchanged_spot(self):
        q, _ = select_contract(_chain(), "BUY_CALL", SPOT, NOW, SelectionParams())
        assert premium_at(q, SPOT, NOW + timedelta(hours=1)) < q.mid

    def test_moves_with_delta(self):
        q, _ = select_contract(_chain(), "BUY_CALL", SPOT, NOW, SelectionParams())
        up = premium_at(q, SPOT + 100, NOW)
        assert up - q.mid == pytest.approx(q.greeks.delta * 100, rel=0.15)


# ── Costs ─────────────────────────────────────────────────────────────────


class TestOptionCosts:
    def test_nse_round_trip_components(self):
        c = round_trip_cost(100.0, 120.0, 65)
        buy, sell = 6_500.0, 7_800.0
        assert c.brokerage == 40.0
        assert c.stt == pytest.approx(sell * 0.001)
        assert c.exchange == pytest.approx((buy + sell) * 0.0003503, abs=0.01)
        assert c.stamp == pytest.approx(buy * 0.00003, abs=0.01)
        assert c.gst == pytest.approx((40.0 + c.exchange + c.sebi) * 0.18, abs=0.02)
        assert c.total == pytest.approx(
            c.brokerage + c.stt + c.exchange + c.sebi + c.stamp + c.gst
        )

    def test_stt_only_on_sell_side(self):
        assert round_trip_cost(100.0, 0.0, 65).stt == 0.0

    def test_bse_schedule(self):
        assert round_trip_cost(100.0, 100.0, 20, exchange="BSE").exchange == pytest.approx(
            4_000 * BSE_OPTIONS.exchange_txn_pct, abs=0.01
        )

    def test_zero_quantity_costs_nothing(self):
        assert round_trip_cost(100.0, 100.0, 0).total == 0.0
