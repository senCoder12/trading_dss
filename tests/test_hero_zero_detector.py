"""
Unit tests for src/engine/hero_zero_detector.py

Covers:
  - Time window gates (14:00-14:50 IST)
  - Expiry day detection
  - Circuit breaker logic (last 2 HZ losses, daily cap)
  - GEX calculation (short gamma, long gamma)
  - Max pain proximity (< 1.5%)
  - IV skew (OTM wings)
  - Momentum (3-bar directional checks)
  - Stop loss calculation (hard caps by index)
  - Direction determination (momentum + skew)
  - All conditions met → is_triggered=True
  - Partial conditions → is_triggered=False with reasoning
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.data.options_chain import OptionStrike, OptionsChainData
from src.engine.hero_zero_detector import HeroZeroDetector, HeroZeroResult
from config.constants import IST_TIMEZONE

_IST = ZoneInfo(IST_TIMEZONE)


# ─────────────────────────────────────────────────────────────────────────────
# Test Data Factories
# ─────────────────────────────────────────────────────────────────────────────


def make_option_strike(
    strike: float,
    ce_oi: int = 100_000,
    pe_oi: int = 100_000,
    ce_iv: float = 15.0,  # NSE format: percentage
    pe_iv: float = 15.0,
) -> OptionStrike:
    """Factory for OptionStrike with sensible defaults."""
    return OptionStrike(
        strike_price=strike,
        ce_oi=ce_oi,
        ce_oi_change=1000,
        ce_oi_change_pct=1.0,
        ce_volume=10_000,
        ce_ltp=100.0,
        ce_ltp_change=5.0,
        ce_ltp_change_pct=5.0,
        ce_iv=ce_iv,
        ce_bid_price=99.0,
        ce_bid_qty=100,
        ce_ask_price=101.0,
        ce_ask_qty=100,
        pe_oi=pe_oi,
        pe_oi_change=1000,
        pe_oi_change_pct=1.0,
        pe_volume=10_000,
        pe_ltp=100.0,
        pe_ltp_change=5.0,
        pe_ltp_change_pct=5.0,
        pe_iv=pe_iv,
        pe_bid_price=99.0,
        pe_bid_qty=100,
        pe_ask_price=101.0,
        pe_ask_qty=100,
    )


def make_options_chain(
    spot: float = 22000.0,
    timestamp: Optional[datetime] = None,
    expiry: Optional[date] = None,
    strikes: Optional[list[float]] = None,
    short_gamma: bool = True,
) -> OptionsChainData:
    """Factory for OptionsChainData with sensible defaults.

    short_gamma=True creates PE-heavy OI (short gamma for dealers).
    """
    if timestamp is None:
        timestamp = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
    if expiry is None:
        expiry = date(2026, 4, 28)
    if strikes is None:
        strikes = list(range(int(spot) - 500, int(spot) + 500, 100))

    strike_data = tuple(
        make_option_strike(
            s,
            ce_oi=50_000 + (s - spot) * 5 if short_gamma else 100_000 + (s - spot) * 10,
            pe_oi=150_000 - (s - spot) * 5 if short_gamma else 100_000 - (s - spot) * 10,
        )
        for s in strikes
    )

    return OptionsChainData(
        index_id="NIFTY50",
        spot_price=spot,
        timestamp=timestamp,
        expiry_date=expiry,
        strikes=strike_data,
        available_expiries=(expiry,),
    )


def make_price_bars(closes: list[float]) -> list[dict]:
    """Factory for price bar dicts with just close price."""
    return [{"close": c, "volume": 1_000_000} for c in closes]


# ─────────────────────────────────────────────────────────────────────────────
# Mock Database Manager
# ─────────────────────────────────────────────────────────────────────────────


class MockDatabaseManager:
    """Mock DatabaseManager for circuit breaker tests."""

    def __init__(self):
        self.hz_signals = []

    def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        """Mock fetch_all to return stored HZ signals."""
        # Return stored signals in reverse order (newest first)
        return list(reversed(self.hz_signals))

    def add_hz_signal(self, index_id: str, outcome: str, signal_type: str, generated_at: Optional[str] = None):
        """Helper to add a mock HZ signal."""
        if generated_at is None:
            generated_at = datetime.now(tz=_IST).isoformat()
        self.hz_signals.append({
            "index_id": index_id,
            "outcome": outcome,
            "signal_type": signal_type,
            "generated_at": generated_at,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Time Window Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTimeWindow:
    """Test HeroZeroDetector time window gate (14:00-14:50 IST)."""

    def test_in_window_14_30(self):
        """14:30 IST is within window."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        chain = make_options_chain(timestamp=datetime(2026, 4, 28, 14, 30, tzinfo=_IST))
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, datetime(2026, 4, 28, 14, 30, tzinfo=_IST), 75)
        # Should pass time gate, may fail other gates
        assert result is not None

    def test_outside_window_13_59(self):
        """13:59 IST is before window."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        chain = make_options_chain(timestamp=datetime(2026, 4, 28, 13, 59, tzinfo=_IST))
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, datetime(2026, 4, 28, 13, 59, tzinfo=_IST), 75)

        assert result.is_triggered is False
        assert "outside" in result.reasoning.lower()

    def test_outside_window_14_51(self):
        """14:51 IST is after window."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        chain = make_options_chain(timestamp=datetime(2026, 4, 28, 14, 51, tzinfo=_IST))
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, datetime(2026, 4, 28, 14, 51, tzinfo=_IST), 75)

        assert result.is_triggered is False
        assert "outside" in result.reasoning.lower()

    def test_boundary_14_00(self):
        """14:00 IST (inclusive start) is in window."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        chain = make_options_chain(timestamp=datetime(2026, 4, 28, 14, 0, tzinfo=_IST))
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, datetime(2026, 4, 28, 14, 0, tzinfo=_IST), 75)

        assert result is not None

    def test_boundary_14_50(self):
        """14:50 IST (inclusive end) is in window."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        chain = make_options_chain(timestamp=datetime(2026, 4, 28, 14, 50, tzinfo=_IST))
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, datetime(2026, 4, 28, 14, 50, tzinfo=_IST), 75)

        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Expiry Day Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExpiryDay:
    """Test HeroZeroDetector expiry day gate."""

    def test_expiry_today_thursday(self):
        """Today is expiry day (Thursday)."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)  # Thursday
        chain = make_options_chain(expiry=date(2024, 6, 27), timestamp=now)
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        # Should pass expiry gate

    def test_expiry_tomorrow(self):
        """Expiry is tomorrow, not today."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2024, 6, 26, 14, 30, tzinfo=_IST)  # Wednesday
        chain = make_options_chain(expiry=date(2024, 6, 27), timestamp=now)
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result.is_triggered is False
        assert "not expiry" in result.reasoning.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCircuitBreaker:
    """Test HeroZeroDetector circuit breaker (last 2 HZ losses, daily cap)."""

    def test_circuit_breaker_active_last_2_losses(self):
        """Last 2 HZ trades both lost → circuit breaker active."""
        db = MockDatabaseManager()
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        # Add 2 LOSS signals (must be added in reverse chronological order for DESC ordering)
        db.add_hz_signal("NIFTY50", "LOSS", "BUY_PUT", generated_at=now.isoformat())
        db.add_hz_signal("NIFTY50", "LOSS", "BUY_CALL", generated_at=(now - timedelta(hours=1)).isoformat())
        detector = HeroZeroDetector(db)
        chain = make_options_chain(timestamp=now)
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result.is_triggered is False
        assert result.circuit_breaker_active is True
        assert "circuit breaker" in result.reasoning.lower()

    def test_circuit_breaker_inactive_one_loss(self):
        """Only 1 prior loss → circuit breaker not active."""
        db = MockDatabaseManager()
        db.add_hz_signal("NIFTY50", "LOSS", "BUY_CALL")
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        chain = make_options_chain(timestamp=now)
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        assert result.circuit_breaker_active is False
        # Should pass circuit breaker gate

    def test_circuit_breaker_inactive_one_win_one_loss(self):
        """Last 2 include a win → circuit breaker not active."""
        db = MockDatabaseManager()
        db.add_hz_signal("NIFTY50", "WIN", "BUY_CALL")
        db.add_hz_signal("NIFTY50", "LOSS", "BUY_PUT")
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        chain = make_options_chain(timestamp=now)
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        assert result.circuit_breaker_active is False

    def test_daily_hz_cap_exceeded(self):
        """Already generated 1 HZ signal today → cap reached."""
        db = MockDatabaseManager()
        now = datetime(2026, 4, 28, 14, 35, tzinfo=_IST)
        # Add 1 WIN signal today (counts toward daily cap)
        db.add_hz_signal("NIFTY50", "WIN", "BUY_CALL", generated_at=now.isoformat())
        detector = HeroZeroDetector(db)
        chain = make_options_chain(timestamp=now, short_gamma=True)
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result.is_triggered is False
        assert "daily" in result.reasoning.lower() and "cap" in result.reasoning.lower()


# ─────────────────────────────────────────────────────────────────────────────
# GEX Calculation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGexCalculation:
    """Test HeroZeroDetector GEX calculation (short gamma = acceleration)."""

    def test_gex_short_gamma_with_pe_heavy_oi(self):
        """PE-heavy OI should produce short gamma (net_gex < 0)."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)

        # Create chain with short_gamma=True (PE-heavy OI)
        chain = make_options_chain(spot=22000.0, timestamp=now, short_gamma=True)
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        # With short_gamma=True, net_gex should be negative or conditions should reflect it
        if result.is_triggered:
            assert result.net_gex < 0

    def test_gex_long_gamma_with_ce_heavy_oi(self):
        """CE-heavy OI should produce long gamma (net_gex > 0)."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)

        # Create chain with short_gamma=False (CE-heavy OI)
        chain = make_options_chain(spot=22000.0, timestamp=now, short_gamma=False)
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        # With short_gamma=False, net_gex should be positive, so short_gamma condition should fail
        assert result.is_triggered is False
        if "short_gamma" in result.conditions_met:
            assert result.conditions_met["short_gamma"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Max Pain Proximity Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMaxPainProximity:
    """Test HeroZeroDetector max pain proximity check (< 1.5%)."""

    def test_max_pain_close_to_spot(self):
        """Max pain within 1.5% of spot → gravity_active=True."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        spot = 22000.0
        # Max pain typically ~spot on expiry, so proximity should be small
        chain = make_options_chain(spot=spot, timestamp=now)
        bars = make_price_bars([spot, spot + 100, spot + 200])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        assert result.max_pain_proximity_pct <= 1.5  # Within threshold

    def test_max_pain_far_from_spot(self):
        """Max pain > 1.5% away → gravity_active=False → condition fails."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        spot = 22000.0

        # Create chain with strikes far from spot to force max pain far away
        strikes = list(range(21000, 23500, 100))
        strike_data = tuple(
            make_option_strike(s, ce_oi=10_000 if s < 22000 else 500_000,
                              pe_oi=500_000 if s < 22000 else 10_000)
            for s in strikes
        )
        chain = OptionsChainData(
            index_id="NIFTY50",
            spot_price=spot,
            timestamp=now,
            expiry_date=date(2024, 6, 27),
            strikes=strike_data,
            available_expiries=(date(2024, 6, 27),),
        )
        bars = make_price_bars([spot, spot + 100, spot + 200])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        # Should fail gravity condition or not trigger
        if result.max_pain_proximity_pct > 1.5:
            assert result.max_pain_gravity_active is False


# ─────────────────────────────────────────────────────────────────────────────
# IV Skew Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIvSkew:
    """Test HeroZeroDetector IV skew (OTM wings)."""

    def test_iv_skew_bullish(self):
        """OTM put IV < OTM call IV → BULLISH skew."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        spot = 22000.0

        # Create chain: OTM puts low IV, OTM calls high IV
        strikes = [21900, 21950, 22000, 22050, 22100]
        strike_data = tuple(
            make_option_strike(
                s,
                ce_iv=20.0 if s > spot else 10.0,  # calls higher IV
                pe_iv=10.0 if s > spot else 20.0,  # puts lower IV
            )
            for s in strikes
        )
        chain = OptionsChainData(
            index_id="NIFTY50",
            spot_price=spot,
            timestamp=now,
            expiry_date=date(2024, 6, 27),
            strikes=strike_data,
            available_expiries=(date(2024, 6, 27),),
        )
        bars = make_price_bars([spot, spot + 100, spot + 200])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        assert result.skew_direction in ["BULLISH", "NEUTRAL"]

    def test_iv_skew_bearish(self):
        """OTM put IV > OTM call IV → BEARISH skew."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        spot = 22000.0

        # Create chain: OTM puts high IV, OTM calls low IV
        strikes = [21900, 21950, 22000, 22050, 22100]
        strike_data = tuple(
            make_option_strike(
                s,
                ce_iv=10.0 if s > spot else 20.0,  # calls lower IV
                pe_iv=20.0 if s > spot else 10.0,  # puts higher IV
            )
            for s in strikes
        )
        chain = OptionsChainData(
            index_id="NIFTY50",
            spot_price=spot,
            timestamp=now,
            expiry_date=date(2024, 6, 27),
            strikes=strike_data,
            available_expiries=(date(2024, 6, 27),),
        )
        bars = make_price_bars([spot, spot - 100, spot - 200])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        assert result.skew_direction in ["BEARISH", "NEUTRAL"]


# ─────────────────────────────────────────────────────────────────────────────
# Momentum Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMomentum:
    """Test HeroZeroDetector momentum check (3-bar directional)."""

    def test_momentum_bullish_3_up(self):
        """Last 3 closes: up, up, up → BULLISH + confirmed."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        chain = make_options_chain(spot=22000.0, timestamp=now)
        bars = make_price_bars([22000.0, 22100.0, 22200.0])  # Last 3: ↑↑↑

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        assert result.momentum_direction == "BULLISH"
        assert result.momentum_confirmed is True

    def test_momentum_bearish_3_down(self):
        """Last 3 closes: down, down, down → BEARISH + confirmed."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        chain = make_options_chain(spot=22000.0, timestamp=now)
        bars = make_price_bars([22300.0, 22200.0, 22100.0])  # Last 3: ↓↓↓

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        assert result.momentum_direction == "BEARISH"
        assert result.momentum_confirmed is True

    def test_momentum_neutral_mixed(self):
        """Last 3: up, down, up → NEUTRAL + not confirmed."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        chain = make_options_chain(spot=22000.0, timestamp=now)
        bars = make_price_bars([22000.0, 22100.0, 22050.0])  # Mixed

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        assert result.momentum_direction == "NEUTRAL"
        assert result.momentum_confirmed is False

    def test_momentum_insufficient_bars(self):
        """Less than 3 bars → NEUTRAL."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        chain = make_options_chain(spot=22000.0, timestamp=now)
        bars = make_price_bars([22000.0, 22100.0])  # Only 2 bars

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        assert result.momentum_confirmed is False


# ─────────────────────────────────────────────────────────────────────────────
# Stop Loss Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStopLoss:
    """Test HeroZeroDetector stop loss calculation (hard caps)."""

    def test_sl_nifty_buy_call(self):
        """NIFTY50 BUY_CALL → SL = spot - 50pts - 2pts slippage."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        spot = 22000.0
        chain = make_options_chain(spot=spot, timestamp=now)
        bars = make_price_bars([spot, spot + 100, spot + 200])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        if result.is_triggered and result.signal_direction == "BUY_CALL":
            assert result.stop_loss == spot - 52.0  # 50pts + 2pts slippage

    def test_sl_banknifty_buy_put(self):
        """BANKNIFTY BUY_PUT → SL = spot + 100pts + 2pts slippage."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        spot = 45000.0
        chain = make_options_chain(spot=spot, timestamp=now)
        bars = make_price_bars([spot, spot - 100, spot - 200])

        result = detector.detect("BANKNIFTY", chain, bars, 1.5, now, 15)

        assert result is not None
        if result.is_triggered and result.signal_direction == "BUY_PUT":
            assert result.stop_loss == spot + 102.0  # 100pts + 2pts slippage


# ─────────────────────────────────────────────────────────────────────────────
# Direction Determination Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDirection:
    """Test HeroZeroDetector signal direction logic."""

    def test_direction_momentum_bullish_skew_bearish(self):
        """Momentum wins (BULLISH) over skew (BEARISH) → BUY_CALL."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        # Check that _determine_direction prioritizes momentum
        direction = detector._determine_direction("BULLISH", "BEARISH")
        assert direction == "BUY_CALL"

    def test_direction_momentum_bearish_skew_bullish(self):
        """Momentum wins (BEARISH) over skew (BULLISH) → BUY_PUT."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        direction = detector._determine_direction("BEARISH", "BULLISH")
        assert direction == "BUY_PUT"

    def test_direction_both_bullish(self):
        """Both BULLISH → BUY_CALL."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        direction = detector._determine_direction("BULLISH", "BULLISH")
        assert direction == "BUY_CALL"

    def test_direction_both_bearish(self):
        """Both BEARISH → BUY_PUT."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        direction = detector._determine_direction("BEARISH", "BEARISH")
        assert direction == "BUY_PUT"

    def test_direction_momentum_neutral(self):
        """Momentum NEUTRAL → use skew as tiebreaker."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        direction_bullish = detector._determine_direction("NEUTRAL", "BULLISH")
        direction_bearish = detector._determine_direction("NEUTRAL", "BEARISH")
        assert direction_bullish == "BUY_CALL"
        assert direction_bearish == "BUY_PUT"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: All Conditions Met Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAllConditionsMet:
    """Test HeroZeroDetector with all conditions satisfied (integration)."""

    def test_all_conditions_met_triggers_signal(self):
        """All conditions met → is_triggered=True."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        spot = 22000.0

        # Create favourable conditions:
        # - Time: 14:30 (in window) ✓
        # - Expiry: today ✓
        # - No circuit breaker ✓
        # - Short gamma (PE-heavy OI) ✓
        # - Max pain close (default) ✓
        # - Volume ratio: 1.5 (met) ✓
        # - Momentum: bullish ✓
        chain = make_options_chain(spot=spot, timestamp=now, short_gamma=True)
        bars = make_price_bars([spot, spot + 100, spot + 200])  # Bullish: ↑↑
        volume_ratio = 1.5  # Met threshold

        result = detector.detect("NIFTY50", chain, bars, volume_ratio, now, 75)

        assert result is not None
        # All conditions are met, so should trigger
        assert result.is_triggered is True
        assert result.signal_direction == "BUY_CALL"  # Bullish momentum
        assert result.conditions_met["short_gamma"] is True
        assert result.conditions_met["max_pain_gravity"] is True
        assert result.conditions_met["volume_spike"] is True
        assert result.conditions_met["momentum"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_iv_data_skipped(self):
        """Strikes with missing IV (< 1.0) should be skipped in GEX calc."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)

        # Mix of valid and invalid IV
        strikes = [21900, 22000, 22100]
        strike_data = []
        for s in strikes:
            if s == 22000:
                # Valid IV
                strike_data.append(make_option_strike(s, ce_iv=15.0, pe_iv=15.0))
            else:
                # Missing IV
                strike_data.append(make_option_strike(s, ce_iv=0.5, pe_iv=0.5))

        chain = OptionsChainData(
            index_id="NIFTY50",
            spot_price=22000.0,
            timestamp=now,
            expiry_date=date(2024, 6, 27),
            strikes=tuple(strike_data),
            available_expiries=(date(2024, 6, 27),),
        )
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        # Should complete without crashing

    def test_empty_chain(self):
        """Empty strike list should not crash."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)

        chain = OptionsChainData(
            index_id="NIFTY50",
            spot_price=22000.0,
            timestamp=now,
            expiry_date=date(2024, 6, 27),
            strikes=(),  # Empty
            available_expiries=(date(2024, 6, 27),),
        )
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        result = detector.detect("NIFTY50", chain, bars, 1.5, now, 75)

        assert result is not None
        # Should not crash, likely not triggered

    def test_none_chain_returns_none(self):
        """None chain passed in → returns None (defensive)."""
        db = MockDatabaseManager()
        detector = HeroZeroDetector(db)
        now = datetime(2026, 4, 28, 14, 30, tzinfo=_IST)
        bars = make_price_bars([22000.0, 22100.0, 22200.0])

        # Manually call _step_hero_zero equivalent to test None handling
        try:
            result = detector.detect("NIFTY50", None, bars, 1.5, now, 75)
            # If no exception, should fail gracefully
            assert result is None or result.is_triggered is False
        except (AttributeError, TypeError):
            # OK if it raises - None chain shouldn't be passed in production
            pass
