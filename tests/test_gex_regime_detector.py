"""Tests for GEXRegimeDetector."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.gex.gex_regime_detector import GEXRegimeDetector
from src.analysis.gex.models import GEXProfile

_IST = ZoneInfo("Asia/Kolkata")


def _make_profile(
    regime: str = "NEGATIVE_GEX",
    regime_strength: float = 70.0,
    net_gex: float = -200_000_000,
    call_wall: float = 24500.0,
    put_wall: float = 23800.0,
    flip_point: float = 24150.0,
    spot: float = 24100.0,
) -> GEXProfile:
    import datetime as dt
    today = dt.date.today()
    now = datetime.now(tz=_IST)
    return GEXProfile(
        symbol="NIFTY",
        spot_price=spot,
        timestamp=now,
        strikes=[],
        net_gex=net_gex,
        call_wall=call_wall,
        put_wall=put_wall,
        gex_flip_point=flip_point,
        largest_positive_gex=500_000_000,
        largest_negative_gex=-800_000_000,
        regime=regime,
        regime_strength=regime_strength,
        gex_change_30min=0.0,
        expiries_included=[today],
        total_oi_analyzed=100_000,
        lot_size=50,
    )


@pytest.fixture
def detector() -> GEXRegimeDetector:
    return GEXRegimeDetector(gex_calculator=None, db=None)


# ── Test 1 ───────────────────────────────────────────────────────────────────

class TestOptionBuyingQuality:
    """Low VIX + Negative GEX → 'EXCELLENT' option buying."""

    def test_low_vix_negative_gex_excellent(self, detector: GEXRegimeDetector) -> None:
        # flip_point far from current_price to avoid FLIP_ZONE override
        profile = _make_profile(
            regime="NEGATIVE_GEX", regime_strength=70,
            flip_point=23000.0, spot=24100.0,
        )
        fake_now = datetime(2026, 5, 12, 11, 0, tzinfo=_IST)
        with patch("src.analysis.gex.gex_regime_detector.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = detector.classify_regime(
                symbol="NIFTY",
                gex_profile=profile,
                vix_level=12.0,
                vix_trend="FLAT",
                current_price=24100.0,
                price_trend_15min="UP",
                is_expiry_day=False,
                minutes_to_close=270,
            )
        assert result["option_buying_quality"] == "EXCELLENT"
        assert result["sub_regime"] == "COMPLACENT_TRENDING"


# ── Test 2 ───────────────────────────────────────────────────────────────────

class TestSubRegime:
    """High VIX + Positive GEX → 'FEAR_WITH_STABILIZATION'."""

    def test_high_vix_positive_gex(self, detector: GEXRegimeDetector) -> None:
        # flip_point far above current_price to avoid FLIP_ZONE override
        profile = _make_profile(
            regime="POSITIVE_GEX", net_gex=150_000_000, regime_strength=60,
            flip_point=26000.0, spot=24100.0,
        )
        fake_now = datetime(2026, 5, 12, 11, 0, tzinfo=_IST)
        with patch("src.analysis.gex.gex_regime_detector.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = detector.classify_regime(
                symbol="NIFTY",
                gex_profile=profile,
                vix_level=22.0,
                vix_trend="FLAT",
                current_price=24100.0,
                price_trend_15min="SIDEWAYS",
                is_expiry_day=False,
                minutes_to_close=270,
            )
        assert result["sub_regime"] == "FEAR_WITH_STABILIZATION"


# ── Test 3 ───────────────────────────────────────────────────────────────────

class TestExpiryPinning:
    """Expiry day after 2 PM → 'EXPIRY_PINNING'."""

    def test_expiry_day_after_2pm(self, detector: GEXRegimeDetector) -> None:
        profile = _make_profile(regime="NEGATIVE_GEX", regime_strength=80)
        fake_now = datetime(2026, 5, 12, 14, 30, tzinfo=_IST)
        with patch("src.analysis.gex.gex_regime_detector.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = detector.classify_regime(
                symbol="NIFTY",
                gex_profile=profile,
                vix_level=15.0,
                vix_trend="FLAT",
                current_price=24100.0,
                price_trend_15min="UP",
                is_expiry_day=True,
                minutes_to_close=60,
            )
        assert result["sub_regime"] == "EXPIRY_PINNING"
        assert result["strategy_type"] == "EXPIRY_SPECIFIC"

    def test_expiry_day_before_2pm_no_pinning(self, detector: GEXRegimeDetector) -> None:
        profile = _make_profile(regime="NEGATIVE_GEX", regime_strength=80)
        fake_now = datetime(2026, 5, 12, 12, 0, tzinfo=_IST)
        with patch("src.analysis.gex.gex_regime_detector.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = detector.classify_regime(
                symbol="NIFTY",
                gex_profile=profile,
                vix_level=15.0,
                vix_trend="FLAT",
                current_price=24100.0,
                price_trend_15min="UP",
                is_expiry_day=True,
                minutes_to_close=210,
            )
        assert result["sub_regime"] != "EXPIRY_PINNING"


# ── Test 4 ───────────────────────────────────────────────────────────────────

class TestFlipPointConfidence:
    """Near flip point reduces confidence by 20."""

    def test_near_flip_point_reduces_confidence(self, detector: GEXRegimeDetector) -> None:
        # 24104 is ~0.19% below 24150 → within 0.3% threshold
        spot_near = 24104.0
        profile_near = _make_profile(
            regime="NEGATIVE_GEX",
            regime_strength=70.0,
            flip_point=24150.0,
            spot=spot_near,
        )
        # 23900 is ~1.05% below 24150 → outside 0.3%
        spot_far = 23900.0
        profile_far = _make_profile(
            regime="NEGATIVE_GEX",
            regime_strength=70.0,
            flip_point=24150.0,
            spot=spot_far,
        )
        fake_now = datetime(2026, 5, 12, 11, 0, tzinfo=_IST)
        with patch("src.analysis.gex.gex_regime_detector.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result_near = detector.classify_regime(
                symbol="NIFTY",
                gex_profile=profile_near,
                vix_level=13.0,
                vix_trend="FLAT",
                current_price=spot_near,
                price_trend_15min="SIDEWAYS",
                is_expiry_day=False,
                minutes_to_close=270,
            )
        with patch("src.analysis.gex.gex_regime_detector.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result_far = detector.classify_regime(
                symbol="NIFTY2",
                gex_profile=profile_far,
                vix_level=13.0,
                vix_trend="FLAT",
                current_price=spot_far,
                price_trend_15min="DOWN",
                is_expiry_day=False,
                minutes_to_close=270,
            )

        assert result_near["confidence"] < result_far["confidence"]
        assert result_near["sub_regime"] == "FLIP_ZONE"


# ── Test 5 ───────────────────────────────────────────────────────────────────

class TestRegimeChangeConfirmation:
    """Regime change requires confirmation_minutes to confirm."""

    def test_confirmation_delay(self, detector: GEXRegimeDetector) -> None:
        detector._regime_change_confirmation_minutes = 15
        detector._current_regimes["NIFTY"] = "POSITIVE_GEX"

        profile_neg = _make_profile(regime="NEGATIVE_GEX", regime_strength=70)

        # Call 1 at t1 — starts pending change, not yet confirmed
        t1 = datetime(2026, 5, 12, 11, 5, tzinfo=_IST)
        with patch("src.analysis.gex.gex_regime_detector.datetime") as mock_dt:
            mock_dt.now.return_value = t1
            detector.classify_regime(
                "NIFTY", profile_neg, 13.0, "FLAT", 24050.0, "DOWN", False, 265
            )
        assert detector._current_regimes.get("NIFTY") == "POSITIVE_GEX"
        assert detector._pending_regime_change.get("NIFTY") is not None

        # Call 2 at t2 (5 min later) — still not confirmed
        t2 = datetime(2026, 5, 12, 11, 10, tzinfo=_IST)
        with patch("src.analysis.gex.gex_regime_detector.datetime") as mock_dt:
            mock_dt.now.return_value = t2
            detector.classify_regime(
                "NIFTY", profile_neg, 13.0, "FLAT", 24050.0, "DOWN", False, 260
            )
        assert detector._current_regimes.get("NIFTY") == "POSITIVE_GEX"

        # Call 3 at t3 (16 min from t1) — confirmed
        t3 = datetime(2026, 5, 12, 11, 21, tzinfo=_IST)
        with patch("src.analysis.gex.gex_regime_detector.datetime") as mock_dt:
            mock_dt.now.return_value = t3
            # Must also mock timedelta-based comparison: pending["first_detected"] was t1
            # We need datetime.now() to return t3 so elapsed = t3-t1 = 16min >= 15min
            detector.classify_regime(
                "NIFTY", profile_neg, 13.0, "FLAT", 24050.0, "DOWN", False, 249
            )
        assert detector._current_regimes.get("NIFTY") == "NEGATIVE_GEX"


# ── Test 6 ───────────────────────────────────────────────────────────────────

class TestMorningBrief:
    """get_day_trading_plan() returns complete dict with non-empty morning_brief."""

    def test_morning_plan_keys_and_brief(self, detector: GEXRegimeDetector) -> None:
        profile = _make_profile(regime="NEGATIVE_GEX", regime_strength=70)
        plan = detector.get_day_trading_plan(
            symbol="NIFTY",
            gex_profile=profile,
            vix_level=13.2,
            vix_trend="FALLING",
            is_expiry_day=False,
        )
        required_keys = {
            "day_type", "recommended_strategy", "max_positions",
            "position_size_multiplier", "preferred_strikes",
            "max_holding_minutes", "key_levels_for_day",
            "avoid_conditions", "morning_brief",
        }
        assert required_keys.issubset(plan.keys())
        assert len(plan["morning_brief"]) > 20
        assert plan["day_type"] == "TRENDING"
        assert plan["preferred_strikes"] == "ATM"


# ── Test 7 ───────────────────────────────────────────────────────────────────

class TestAvoidDirectional:
    """Low VIX + Positive GEX → CALM_RANGEBOUND → AVOID_DIRECTIONAL."""

    def test_calm_rangebound_avoid_directional(self, detector: GEXRegimeDetector) -> None:
        profile = _make_profile(
            regime="POSITIVE_GEX",
            regime_strength=75,
            net_gex=150_000_000,
            # push flip_point far away to avoid FLIP_ZONE
            flip_point=22000.0,
            call_wall=26000.0,
            put_wall=22000.0,
        )
        fake_now = datetime(2026, 5, 12, 11, 0, tzinfo=_IST)
        with patch("src.analysis.gex.gex_regime_detector.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = detector.classify_regime(
                symbol="NIFTY",
                gex_profile=profile,
                vix_level=12.0,
                vix_trend="FLAT",
                current_price=24100.0,
                price_trend_15min="SIDEWAYS",
                is_expiry_day=False,
                minutes_to_close=270,
            )
        assert result["strategy_type"] == "AVOID_DIRECTIONAL"
        assert result["sub_regime"] == "CALM_RANGEBOUND"
        assert result["option_buying_quality"] == "AVOID"


# ── Test 8 ───────────────────────────────────────────────────────────────────

class TestWallBreak:
    """Price within 0.5% of call_wall → WALL_TRADE strategy."""

    def test_near_call_wall_triggers_wall_trade(self, detector: GEXRegimeDetector) -> None:
        # Price at 24490, call_wall at 24500 → 0.04% → within 0.5%
        profile = _make_profile(
            regime="POSITIVE_GEX",
            regime_strength=65,
            call_wall=24500.0,
            put_wall=23000.0,
            flip_point=22000.0,
        )
        fake_now = datetime(2026, 5, 12, 13, 0, tzinfo=_IST)
        with patch("src.analysis.gex.gex_regime_detector.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = detector.classify_regime(
                symbol="NIFTY",
                gex_profile=profile,
                vix_level=14.0,
                vix_trend="FLAT",
                current_price=24490.0,
                price_trend_15min="UP",
                is_expiry_day=False,
                minutes_to_close=150,
            )
        assert result["strategy_type"] == "WALL_TRADE"
        assert "call_wall" in result["key_levels"]
        assert result["key_levels"]["call_wall"] == 24500.0
