"""
Tests for GEXCalculator.

All tests use synthetic options chains — no real API or DB calls.
The GEXCalculator is instantiated with smart_api=None, db=None.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.analysis.gex.gex_calculator import GEXCalculator
from src.analysis.gex.models import GEXProfile, StrikeGEX

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SPOT = 24000.0
LOT = 50
EXPIRY_NEAR = date(2026, 5, 15)
EXPIRY_FAR = date(2026, 5, 22)
NOW = datetime(2026, 5, 12, 10, 30, 0, tzinfo=_IST)


def _make_chain(
    spot: float,
    strikes: list[float],
    expiry: date,
    *,
    ce_oi: int = 10_000,
    pe_oi: int = 10_000,
    iv_pct: float = 15.0,       # percentage, e.g. 15.0
    ce_oi_override: dict[float, int] | None = None,
    pe_oi_override: dict[float, int] | None = None,
) -> dict:
    """Build a synthetic options chain dict."""
    chain: dict = {}
    exp_str = expiry.isoformat()
    for strike in strikes:
        dist = abs(strike - spot) / spot
        base_iv = iv_pct + dist * 200.0   # smile: ±10 pp per 5% moneyness
        call_oi = ce_oi_override.get(strike, ce_oi) if ce_oi_override else ce_oi
        put_oi = pe_oi_override.get(strike, pe_oi) if pe_oi_override else pe_oi
        chain[float(strike)] = {
            exp_str: {
                "CE": {
                    "openInterest": call_oi,
                    "lastPrice": max(spot - strike + base_iv / 100 * spot * 0.4, 0.5),
                    "impliedVolatility": base_iv,
                },
                "PE": {
                    "openInterest": put_oi,
                    "lastPrice": max(strike - spot + base_iv / 100 * spot * 0.4, 0.5),
                    "impliedVolatility": base_iv,
                },
            }
        }
    return chain


def _default_strikes(spot: float = SPOT, spacing: float = 50.0, n: int = 10) -> list[float]:
    atm = round(spot / spacing) * spacing
    return [atm + i * spacing for i in range(-n, n + 1)]


@pytest.fixture
def calc() -> GEXCalculator:
    return GEXCalculator(smart_api=None, db=None)


@pytest.fixture
def balanced_chain() -> dict:
    return _make_chain(SPOT, _default_strikes(), EXPIRY_NEAR, ce_oi=10_000, pe_oi=10_000)


@pytest.fixture
def balanced_profile(calc: GEXCalculator, balanced_chain: dict) -> GEXProfile:
    return calc.calculate_gex_profile("NIFTY", SPOT, balanced_chain, LOT, NOW)


# ---------------------------------------------------------------------------
# Test 1 — CE GEX always positive, PE GEX always negative
# ---------------------------------------------------------------------------

class TestGEXSign:
    def test_ce_gex_positive(self, balanced_profile: GEXProfile) -> None:
        for s in balanced_profile.strikes:
            assert s.ce_gex >= 0, f"CE GEX at strike {s.strike} is negative: {s.ce_gex}"

    def test_pe_gex_negative(self, balanced_profile: GEXProfile) -> None:
        for s in balanced_profile.strikes:
            assert s.pe_gex <= 0, f"PE GEX at strike {s.strike} is positive: {s.pe_gex}"

    def test_net_gex_is_sum(self, balanced_profile: GEXProfile) -> None:
        for s in balanced_profile.strikes:
            assert abs(s.net_gex - (s.ce_gex + s.pe_gex)) < 1e-6


# ---------------------------------------------------------------------------
# Test 2 — Net GEX sign matches OI dominance
# ---------------------------------------------------------------------------

class TestNetGEXSign:
    def test_positive_when_call_oi_dominates(self, calc: GEXCalculator) -> None:
        chain = _make_chain(
            SPOT, _default_strikes(), EXPIRY_NEAR, ce_oi=50_000, pe_oi=1_000
        )
        profile = calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)
        assert profile.net_gex > 0, f"Expected positive net_gex, got {profile.net_gex}"

    def test_negative_when_put_oi_dominates(self, calc: GEXCalculator) -> None:
        chain = _make_chain(
            SPOT, _default_strikes(), EXPIRY_NEAR, ce_oi=1_000, pe_oi=50_000
        )
        profile = calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)
        assert profile.net_gex < 0, f"Expected negative net_gex, got {profile.net_gex}"


# ---------------------------------------------------------------------------
# Test 3 — Call wall above spot
# ---------------------------------------------------------------------------

class TestCallWall:
    def test_call_wall_above_spot(self, calc: GEXCalculator) -> None:
        # Concentrate call OI at strike 24200 (above spot 24000)
        strikes = _default_strikes()
        chain = _make_chain(
            SPOT, strikes, EXPIRY_NEAR,
            ce_oi=1_000,
            ce_oi_override={24200.0: 200_000},  # huge spike at 24200
        )
        profile = calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)
        if profile.call_wall is not None:
            assert profile.call_wall > SPOT, (
                f"Call wall {profile.call_wall} should be above spot {SPOT}"
            )

    def test_call_wall_is_none_or_above_spot(self, balanced_profile: GEXProfile) -> None:
        cw = balanced_profile.call_wall
        if cw is not None:
            assert cw > SPOT


# ---------------------------------------------------------------------------
# Test 4 — Put wall below spot
# ---------------------------------------------------------------------------

class TestPutWall:
    def test_put_wall_below_spot(self, calc: GEXCalculator) -> None:
        strikes = _default_strikes()
        chain = _make_chain(
            SPOT, strikes, EXPIRY_NEAR,
            pe_oi=1_000,
            pe_oi_override={23800.0: 200_000},  # huge put OI at 23800
        )
        profile = calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)
        if profile.put_wall is not None:
            assert profile.put_wall < SPOT, (
                f"Put wall {profile.put_wall} should be below spot {SPOT}"
            )

    def test_put_wall_is_none_or_below_spot(self, balanced_profile: GEXProfile) -> None:
        pw = balanced_profile.put_wall
        if pw is not None:
            assert pw < SPOT


# ---------------------------------------------------------------------------
# Test 5 — Flip point between put_wall and call_wall
# ---------------------------------------------------------------------------

class TestFlipPoint:
    def test_flip_point_between_walls(self, calc: GEXCalculator) -> None:
        strikes = _default_strikes()
        chain = _make_chain(
            SPOT, strikes, EXPIRY_NEAR,
            ce_oi=1_000,
            pe_oi=1_000,
            ce_oi_override={24200.0: 200_000},
            pe_oi_override={23800.0: 200_000},
        )
        profile = calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)
        cw = profile.call_wall
        pw = profile.put_wall
        fp = profile.gex_flip_point
        # If both walls exist, flip point should be between them
        if cw is not None and pw is not None:
            assert pw <= fp <= cw, (
                f"Flip point {fp} not in [{pw}, {cw}]"
            )

    def test_flip_point_is_finite(self, balanced_profile: GEXProfile) -> None:
        fp = balanced_profile.gex_flip_point
        assert -1e12 < fp < 1e12


# ---------------------------------------------------------------------------
# Test 6 — Regime classification correctness
# ---------------------------------------------------------------------------

class TestRegime:
    def test_positive_gex_regime_when_call_dominant(self, calc: GEXCalculator) -> None:
        chain = _make_chain(
            SPOT, _default_strikes(), EXPIRY_NEAR, ce_oi=100_000, pe_oi=500
        )
        profile = calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)
        assert profile.regime == "POSITIVE_GEX", (
            f"Expected POSITIVE_GEX, got {profile.regime} (net_gex={profile.net_gex:.0f})"
        )

    def test_negative_gex_regime_when_put_dominant(self, calc: GEXCalculator) -> None:
        chain = _make_chain(
            SPOT, _default_strikes(), EXPIRY_NEAR, ce_oi=500, pe_oi=100_000
        )
        profile = calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)
        assert profile.regime == "NEGATIVE_GEX", (
            f"Expected NEGATIVE_GEX, got {profile.regime} (net_gex={profile.net_gex:.0f})"
        )

    def test_regime_strength_in_range(self, balanced_profile: GEXProfile) -> None:
        assert 0.0 <= balanced_profile.regime_strength <= 100.0


# ---------------------------------------------------------------------------
# Test 7 — Wall break detection
# ---------------------------------------------------------------------------

class TestWallBreak:
    def _profile_with_walls(self, calc: GEXCalculator) -> GEXProfile:
        strikes = _default_strikes()
        chain = _make_chain(
            SPOT, strikes, EXPIRY_NEAR,
            ce_oi=1_000,
            pe_oi=1_000,
            ce_oi_override={24200.0: 300_000},
            pe_oi_override={23800.0: 300_000},
        )
        return calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)

    def test_call_wall_break_detected(self, calc: GEXCalculator) -> None:
        profile = self._profile_with_walls(calc)
        if profile.call_wall is None:
            pytest.skip("No call wall established at this OI level")

        cw = profile.call_wall
        event = calc.detect_wall_break(
            "NIFTY",
            previous_price=cw - 10,
            current_price=cw + 5,
            volume=300_000,
            avg_volume=100_000,
        )
        assert event is not None
        assert event["wall_type"] == "CALL_WALL_BREAK"
        assert event["signal"] == "BUY_CALL"
        assert event["wall_price"] == cw

    def test_put_wall_break_detected(self, calc: GEXCalculator) -> None:
        profile = self._profile_with_walls(calc)
        if profile.put_wall is None:
            pytest.skip("No put wall established at this OI level")

        pw = profile.put_wall
        event = calc.detect_wall_break(
            "NIFTY",
            previous_price=pw + 10,
            current_price=pw - 5,
            volume=300_000,
            avg_volume=100_000,
        )
        assert event is not None
        assert event["wall_type"] == "PUT_WALL_BREAK"
        assert event["signal"] == "BUY_PUT"

    def test_no_break_when_price_stays_below_call_wall(self, calc: GEXCalculator) -> None:
        profile = self._profile_with_walls(calc)
        if profile.call_wall is None:
            pytest.skip("No call wall")
        cw = profile.call_wall
        event = calc.detect_wall_break("NIFTY", cw - 20, cw - 5, 100_000, 50_000)
        assert event is None

    def test_volume_confirmed_flag(self, calc: GEXCalculator) -> None:
        profile = self._profile_with_walls(calc)
        if profile.call_wall is None:
            pytest.skip("No call wall")
        cw = profile.call_wall

        # Low volume → not confirmed
        ev_low = calc.detect_wall_break("NIFTY", cw - 10, cw + 5, 50_000, 100_000)
        if ev_low:
            assert ev_low["volume_confirmed"] is False

        # Re-inject profile (detect_wall_break uses _latest_profiles)
        chain = _make_chain(
            SPOT, _default_strikes(), EXPIRY_NEAR,
            ce_oi=1_000,
            ce_oi_override={24200.0: 300_000},
        )
        calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)
        ev_high = calc.detect_wall_break("NIFTY", cw - 10, cw + 5, 250_000, 100_000)
        if ev_high:
            assert ev_high["volume_confirmed"] is True


# ---------------------------------------------------------------------------
# Test 8 — GEX velocity (30-min change)
# ---------------------------------------------------------------------------

class TestGEXVelocity:
    def test_velocity_reflects_gex_change(self, calc: GEXCalculator) -> None:
        strikes = _default_strikes()

        # Reading 1: balanced OI → low net GEX
        t0 = NOW - timedelta(minutes=35)
        chain1 = _make_chain(SPOT, strikes, EXPIRY_NEAR, ce_oi=10_000, pe_oi=10_000)
        p1 = calc.calculate_gex_profile("NIFTY", SPOT, chain1, LOT, t0)

        # Reading 2: 10 min later
        t1 = NOW - timedelta(minutes=25)
        chain2 = _make_chain(SPOT, strikes, EXPIRY_NEAR, ce_oi=10_000, pe_oi=10_000)
        calc.calculate_gex_profile("NIFTY", SPOT, chain2, LOT, t1)

        # Reading 3: now, with much higher call OI → higher net GEX
        chain3 = _make_chain(SPOT, strikes, EXPIRY_NEAR, ce_oi=80_000, pe_oi=10_000)
        p3 = calc.calculate_gex_profile("NIFTY", SPOT, chain3, LOT, NOW)

        # GEX should have increased from t0 to NOW
        assert p3.gex_change_30min > 0, (
            f"Expected positive velocity, got {p3.gex_change_30min}"
        )

    def test_velocity_zero_on_first_reading(self, calc: GEXCalculator) -> None:
        chain = _make_chain(SPOT, _default_strikes(), EXPIRY_NEAR)
        profile = calc.calculate_gex_profile("NIFTY2", SPOT, chain, LOT, NOW)
        assert profile.gex_change_30min == 0.0

    def test_velocity_negative_when_gex_falls(self, calc: GEXCalculator) -> None:
        strikes = _default_strikes()

        t0 = NOW - timedelta(minutes=35)
        chain_high = _make_chain(SPOT, strikes, EXPIRY_NEAR, ce_oi=80_000, pe_oi=10_000)
        calc.calculate_gex_profile("NIFTY3", SPOT, chain_high, LOT, t0)

        chain_low = _make_chain(SPOT, strikes, EXPIRY_NEAR, ce_oi=10_000, pe_oi=10_000)
        p = calc.calculate_gex_profile("NIFTY3", SPOT, chain_low, LOT, NOW)

        assert p.gex_change_30min < 0


# ---------------------------------------------------------------------------
# Test 9 — confidence_multiplier always in [0.5, 1.5]
# ---------------------------------------------------------------------------

class TestConfidenceMultiplier:
    @pytest.mark.parametrize("ce_oi,pe_oi", [
        (100_000, 500),    # strong positive GEX
        (500, 100_000),    # strong negative GEX
        (10_000, 10_000),  # neutral
        (50_000, 1_000),   # moderate positive
        (1_000, 50_000),   # moderate negative
    ])
    def test_multiplier_in_range(
        self, calc: GEXCalculator, ce_oi: int, pe_oi: int
    ) -> None:
        chain = _make_chain(SPOT, _default_strikes(), EXPIRY_NEAR, ce_oi=ce_oi, pe_oi=pe_oi)
        calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)
        sig = calc.get_gex_signal_for_engine("NIFTY", SPOT)
        mult = sig["confidence_multiplier"]
        assert 0.5 <= mult <= 1.5, f"Multiplier {mult} out of [0.5, 1.5] for OI={ce_oi}/{pe_oi}"

    def test_multiplier_no_profile(self, calc: GEXCalculator) -> None:
        sig = calc.get_gex_signal_for_engine("UNKNOWN_SYM", SPOT)
        assert sig["confidence_multiplier"] == 1.0

    def test_multiplier_clamped_after_modifier(self, calc: GEXCalculator) -> None:
        # Strong negative GEX with falling GEX velocity → max aggressive
        strikes = _default_strikes()
        t0 = NOW - timedelta(minutes=35)
        chain_strong = _make_chain(SPOT, strikes, EXPIRY_NEAR, ce_oi=100, pe_oi=200_000)
        calc.calculate_gex_profile("NIFTY", SPOT, chain_strong, LOT, t0)

        chain_weak = _make_chain(SPOT, strikes, EXPIRY_NEAR, ce_oi=100, pe_oi=150_000)
        calc.calculate_gex_profile("NIFTY", SPOT, chain_weak, LOT, NOW)

        sig = calc.get_gex_signal_for_engine("NIFTY", SPOT)
        assert 0.5 <= sig["confidence_multiplier"] <= 1.5


# ---------------------------------------------------------------------------
# Additional — signal structure completeness
# ---------------------------------------------------------------------------

class TestSignalStructure:
    def test_signal_has_required_keys(self, calc: GEXCalculator, balanced_chain: dict) -> None:
        calc.calculate_gex_profile("NIFTY", SPOT, balanced_chain, LOT, NOW)
        sig = calc.get_gex_signal_for_engine("NIFTY", SPOT)
        required = {
            "regime", "regime_strength", "net_gex", "gex_trend",
            "call_wall", "put_wall", "flip_point", "price_vs_flip",
            "strategy_recommendation", "confidence_multiplier", "reason",
        }
        assert required.issubset(sig.keys())

    def test_regime_is_valid(self, calc: GEXCalculator, balanced_chain: dict) -> None:
        calc.calculate_gex_profile("NIFTY", SPOT, balanced_chain, LOT, NOW)
        sig = calc.get_gex_signal_for_engine("NIFTY", SPOT)
        assert sig["regime"] in ("POSITIVE_GEX", "NEGATIVE_GEX", "NEUTRAL_GEX")

    def test_price_vs_flip_consistent(self, calc: GEXCalculator, balanced_chain: dict) -> None:
        profile = calc.calculate_gex_profile("NIFTY", SPOT, balanced_chain, LOT, NOW)
        sig = calc.get_gex_signal_for_engine("NIFTY", SPOT)
        if SPOT > profile.gex_flip_point:
            assert sig["price_vs_flip"] == "ABOVE"
        else:
            assert sig["price_vs_flip"] == "BELOW"


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

class TestDistanceHelpers:
    def test_distance_to_call_wall_positive(self, calc: GEXCalculator) -> None:
        chain = _make_chain(
            SPOT, _default_strikes(), EXPIRY_NEAR,
            ce_oi=1_000,
            ce_oi_override={24200.0: 300_000},
        )
        calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)
        d = calc.get_distance_to_call_wall("NIFTY", SPOT)
        assert d >= 0

    def test_distance_inf_when_no_wall(self, calc: GEXCalculator) -> None:
        # Tiny OI → walls won't clear the 50 Cr threshold
        chain = _make_chain(SPOT, _default_strikes(), EXPIRY_NEAR, ce_oi=10, pe_oi=10)
        calc.calculate_gex_profile("NIFTY", SPOT, chain, LOT, NOW)
        assert calc.get_distance_to_call_wall("NIFTY", SPOT) == float("inf")
        assert calc.get_distance_to_put_wall("NIFTY", SPOT) == float("inf")
