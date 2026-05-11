"""Tests for Black-Scholes pricing engine."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, datetime

import pytest

from src.analysis.gex.black_scholes import (
    BSMInputs,
    BlackScholesEngine,
    OptionGreeks,
    get_dte,
    get_risk_free_rate,
)


@pytest.fixture()
def engine() -> BlackScholesEngine:
    return BlackScholesEngine()


@pytest.fixture()
def atm_ce() -> BSMInputs:
    return BSMInputs(
        spot=24000.0,
        strike=24000.0,
        dte_days=7.0,
        iv=0.15,
        risk_free_rate=0.067,
        option_type="CE",
    )


@pytest.fixture()
def atm_pe(atm_ce: BSMInputs) -> BSMInputs:
    return replace(atm_ce, option_type="PE")


# ---------------------------------------------------------------------------
# Test 1 — ATM call price
# ---------------------------------------------------------------------------


class TestATMCallPrice:
    def test_price_in_expected_range(self, engine: BlackScholesEngine, atm_ce: BSMInputs) -> None:
        price = engine.calculate_price(atm_ce)
        # BSM reference: S=24000 K=24000 T=7d σ=15% r=6.7% → ~214.55
        assert 212.0 <= price <= 217.0, f"ATM call price {price:.2f} outside [212, 217]"

    def test_price_non_negative(self, engine: BlackScholesEngine, atm_ce: BSMInputs) -> None:
        assert engine.calculate_price(atm_ce) >= 0.0


# ---------------------------------------------------------------------------
# Test 2 — Deep ITM call delta
# ---------------------------------------------------------------------------


class TestDeepITMDelta:
    def test_delta_near_one(self, engine: BlackScholesEngine) -> None:
        deep_itm = BSMInputs(
            spot=24000.0,
            strike=22000.0,
            dte_days=7.0,
            iv=0.15,
            risk_free_rate=0.067,
            option_type="CE",
        )
        greeks = engine.calculate_greeks(deep_itm)
        assert 0.98 <= greeks.delta <= 1.0, f"Deep ITM CE delta {greeks.delta:.4f} outside [0.98, 1.0]"


# ---------------------------------------------------------------------------
# Test 3 — ATM gamma symmetry (CE == PE)
# ---------------------------------------------------------------------------


class TestGammaSymmetry:
    def test_ce_pe_gamma_equal(
        self,
        engine: BlackScholesEngine,
        atm_ce: BSMInputs,
        atm_pe: BSMInputs,
    ) -> None:
        ce_greeks = engine.calculate_greeks(atm_ce)
        pe_greeks = engine.calculate_greeks(atm_pe)
        assert math.isclose(ce_greeks.gamma, pe_greeks.gamma, rel_tol=1e-9), (
            f"CE gamma {ce_greeks.gamma} != PE gamma {pe_greeks.gamma}"
        )


# ---------------------------------------------------------------------------
# Test 4 — IV solver round-trip
# ---------------------------------------------------------------------------


class TestIVSolverRoundTrip:
    def test_roundtrip_accuracy(self, engine: BlackScholesEngine, atm_ce: BSMInputs) -> None:
        true_iv = 0.18
        inputs_at_true_iv = replace(atm_ce, iv=true_iv)
        market_price = engine.calculate_price(inputs_at_true_iv)

        solved_iv = engine.calculate_iv(
            market_price=market_price,
            inputs_without_iv=atm_ce,  # iv field ignored by solver
            initial_guess=0.20,
        )
        assert abs(solved_iv - true_iv) < 0.001, (
            f"IV solver error: expected {true_iv}, got {solved_iv:.6f}"
        )

    def test_pe_roundtrip(self, engine: BlackScholesEngine, atm_pe: BSMInputs) -> None:
        true_iv = 0.22
        market_price = engine.calculate_price(replace(atm_pe, iv=true_iv))
        solved_iv = engine.calculate_iv(
            market_price=market_price,
            inputs_without_iv=atm_pe,
            initial_guess=0.20,
        )
        assert abs(solved_iv - true_iv) < 0.001


# ---------------------------------------------------------------------------
# Test 5 — Theta is negative for long options
# ---------------------------------------------------------------------------


class TestThetaSign:
    def test_ce_theta_negative(self, engine: BlackScholesEngine, atm_ce: BSMInputs) -> None:
        greeks = engine.calculate_greeks(atm_ce)
        assert greeks.theta < 0, f"CE theta should be negative, got {greeks.theta}"

    def test_pe_theta_negative(self, engine: BlackScholesEngine, atm_pe: BSMInputs) -> None:
        greeks = engine.calculate_greeks(atm_pe)
        assert greeks.theta < 0, f"PE theta should be negative, got {greeks.theta}"


# ---------------------------------------------------------------------------
# Test 6 — Vega is positive for both CE and PE
# ---------------------------------------------------------------------------


class TestVegaSign:
    def test_ce_vega_positive(self, engine: BlackScholesEngine, atm_ce: BSMInputs) -> None:
        greeks = engine.calculate_greeks(atm_ce)
        assert greeks.vega > 0

    def test_pe_vega_positive(self, engine: BlackScholesEngine, atm_pe: BSMInputs) -> None:
        greeks = engine.calculate_greeks(atm_pe)
        assert greeks.vega > 0

    def test_higher_iv_gives_higher_price(self, engine: BlackScholesEngine, atm_ce: BSMInputs) -> None:
        price_low_iv = engine.calculate_price(replace(atm_ce, iv=0.10))
        price_high_iv = engine.calculate_price(replace(atm_ce, iv=0.25))
        assert price_high_iv > price_low_iv


# ---------------------------------------------------------------------------
# Test 7 — Put-call parity
# ---------------------------------------------------------------------------


class TestPutCallParity:
    def test_parity(
        self,
        engine: BlackScholesEngine,
        atm_ce: BSMInputs,
        atm_pe: BSMInputs,
    ) -> None:
        ce_price = engine.calculate_price(atm_ce)
        pe_price = engine.calculate_price(atm_pe)

        T = atm_ce.dte_days / 365.0
        # C - P = S*exp(-q*T) - K*exp(-r*T)  (assuming q=0 for index)
        lhs = ce_price - pe_price
        rhs = atm_ce.spot - atm_ce.strike * math.exp(-atm_ce.risk_free_rate * T)

        assert math.isclose(lhs, rhs, abs_tol=0.5), (
            f"Put-call parity violated: C-P={lhs:.4f}, S-K*exp(-rT)={rhs:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 8 — Zero DTE handling
# ---------------------------------------------------------------------------


class TestZeroDTE:
    def test_zero_dte_no_division_error(self, engine: BlackScholesEngine, atm_ce: BSMInputs) -> None:
        zero_dte = replace(atm_ce, dte_days=0)
        # Should not raise
        price = engine.calculate_price(zero_dte)
        assert price >= 0.0

    def test_deep_itm_zero_dte_returns_intrinsic(self, engine: BlackScholesEngine) -> None:
        deep_itm = BSMInputs(
            spot=24000.0,
            strike=22000.0,
            dte_days=0,
            iv=0.15,
            risk_free_rate=0.067,
            option_type="CE",
        )
        price = engine.calculate_price(deep_itm)
        intrinsic = 24000.0 - 22000.0
        # At near-zero DTE, price should be very close to intrinsic
        assert abs(price - intrinsic) < 10.0, f"Near-expiry deep ITM price {price} far from intrinsic {intrinsic}"

    def test_otm_zero_dte_near_zero(self, engine: BlackScholesEngine) -> None:
        otm = BSMInputs(
            spot=24000.0,
            strike=26000.0,
            dte_days=0,
            iv=0.15,
            risk_free_rate=0.067,
            option_type="CE",
        )
        price = engine.calculate_price(otm)
        assert price < 5.0, f"Near-expiry OTM call price {price} should be near zero"


# ---------------------------------------------------------------------------
# Test 9 — Bisection fallback
# ---------------------------------------------------------------------------


class TestBisectionFallback:
    def test_bisection_returns_valid_iv(self, engine: BlackScholesEngine, atm_ce: BSMInputs) -> None:
        market_price = engine.calculate_price(replace(atm_ce, iv=0.20))
        iv = engine._bisection_iv(market_price, atm_ce)
        assert engine.MIN_IV <= iv <= engine.MAX_IV

    def test_bisection_accuracy(self, engine: BlackScholesEngine, atm_ce: BSMInputs) -> None:
        true_iv = 0.20
        market_price = engine.calculate_price(replace(atm_ce, iv=true_iv))
        iv = engine._bisection_iv(market_price, atm_ce)
        assert abs(iv - true_iv) < 0.005


# ---------------------------------------------------------------------------
# BSMInputs validation
# ---------------------------------------------------------------------------


class TestBSMInputsValidation:
    def test_rejects_zero_spot(self) -> None:
        with pytest.raises(ValueError, match="spot"):
            BSMInputs(spot=0, strike=24000, dte_days=7, iv=0.15, risk_free_rate=0.067, option_type="CE")

    def test_rejects_negative_dte(self) -> None:
        with pytest.raises(ValueError, match="dte_days"):
            BSMInputs(spot=24000, strike=24000, dte_days=-1, iv=0.15, risk_free_rate=0.067, option_type="CE")

    def test_rejects_zero_iv(self) -> None:
        with pytest.raises(ValueError, match="iv"):
            BSMInputs(spot=24000, strike=24000, dte_days=7, iv=0.0, risk_free_rate=0.067, option_type="CE")

    def test_rejects_invalid_option_type(self) -> None:
        with pytest.raises(ValueError, match="option_type"):
            BSMInputs(spot=24000, strike=24000, dte_days=7, iv=0.15, risk_free_rate=0.067, option_type="CALL")

    def test_zero_dte_gets_floored(self) -> None:
        inputs = BSMInputs(spot=24000, strike=24000, dte_days=0, iv=0.15, risk_free_rate=0.067, option_type="CE")
        assert inputs.dte_days > 0


# ---------------------------------------------------------------------------
# Gamma dollars
# ---------------------------------------------------------------------------


class TestGammaDollars:
    def test_returns_positive(self, engine: BlackScholesEngine, atm_ce: BSMInputs) -> None:
        gd = engine.calculate_gamma_dollars(atm_ce, lot_size=50, oi=1000)
        assert gd > 0

    def test_scales_with_oi(self, engine: BlackScholesEngine, atm_ce: BSMInputs) -> None:
        gd_100 = engine.calculate_gamma_dollars(atm_ce, lot_size=50, oi=100)
        gd_200 = engine.calculate_gamma_dollars(atm_ce, lot_size=50, oi=200)
        assert math.isclose(gd_200, gd_100 * 2.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# get_dte utility
# ---------------------------------------------------------------------------


class TestGetDTE:
    def test_positive_dte_before_expiry(self) -> None:
        expiry = date(2026, 5, 15)
        now = datetime(2026, 5, 12, 10, 0, 0)
        dte = get_dte(expiry, now)
        assert dte > 0

    def test_minimum_floor_at_expiry(self) -> None:
        expiry = date(2026, 5, 12)
        after_close = datetime(2026, 5, 12, 16, 0, 0)
        dte = get_dte(expiry, after_close)
        # Should hit the floor, not negative
        min_dte = 30.0 / (365.0 * 24.0 * 60.0)
        assert dte >= min_dte

    def test_naive_datetime_treated_as_ist(self) -> None:
        expiry = date(2026, 5, 15)
        now = datetime(2026, 5, 12, 10, 0, 0)  # naive
        dte = get_dte(expiry, now)
        assert dte > 0


# ---------------------------------------------------------------------------
# get_risk_free_rate utility
# ---------------------------------------------------------------------------


def test_risk_free_rate_in_range() -> None:
    rfr = get_risk_free_rate()
    assert 0.01 <= rfr <= 0.20, f"Unexpected risk-free rate {rfr}"
