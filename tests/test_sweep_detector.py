"""
Tests for SweepDetector.

Uses MagicMock for CandleStore, LiquidityPoolDetector, and DatabaseManager.
All tests are fully isolated — no real DB, network, or file I/O.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.analysis.liquidity.models import Candle, LiquidityLevel, LiquiditySweeep
from src.analysis.liquidity.sweep_detector import SweepDetector, _avg_volume, _wick_fraction

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 5, 11, 9, 15, tzinfo=_IST)
    return base + timedelta(minutes=offset_minutes)


def _candle(
    symbol: str = "NIFTY",
    timeframe: str = "1min",
    open: float = 100.0,
    high: float = 100.0,
    low: float = 100.0,
    close: float = 100.0,
    volume: int = 1000,
    offset: int = 0,
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=_ts(offset),
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _level(
    price: float = 100.0,
    level_type: str = "BSL",
    symbol: str = "NIFTY",
    timeframe: str = "1min",
) -> LiquidityLevel:
    now = _ts()
    return LiquidityLevel(
        symbol=symbol,
        level_type=level_type,
        price=price,
        strength=2,
        timeframe=timeframe,
        first_formed_at=now,
        last_touched_at=now,
    )


def _make_detector() -> SweepDetector:
    mock_db = MagicMock()
    mock_db.execute.return_value = MagicMock()
    mock_db.execute_many.return_value = None
    mock_db.fetch_all.return_value = []

    mock_cs = MagicMock()
    mock_cs.get_candles.return_value = []

    mock_pool = MagicMock()
    mock_pool.get_all_active_levels.return_value = ([], [])

    return SweepDetector(candle_store=mock_cs, pool_detector=mock_pool, db=mock_db)


def _recent_candles(avg_volume: int = 500, count: int = 10) -> list[Candle]:
    return [
        _candle(volume=avg_volume, offset=i)
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# classify_level_interaction — NONE
# ---------------------------------------------------------------------------

def test_classify_none_bsl_candle_below():
    """Candle never touched BSL level — should return NONE."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="BSL")
    # high=98, well below level at 100
    c = _candle(open=96, high=98, low=95, close=97, volume=1000)
    result = det.classify_level_interaction(c, lvl, _recent_candles(500), 97.0)
    assert result == "NONE"


def test_classify_none_ssl_candle_above():
    """Candle never touched SSL level — should return NONE."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="SSL")
    # low=102, well above level at 100
    c = _candle(open=102, high=105, low=102, close=103, volume=1000)
    result = det.classify_level_interaction(c, lvl, _recent_candles(500), 103.0)
    assert result == "NONE"


# ---------------------------------------------------------------------------
# classify_level_interaction — BOUNCE
# ---------------------------------------------------------------------------

def test_classify_bounce_bsl():
    """Candle barely touches BSL but no meaningful wick beyond."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="BSL")
    # high=100.01 (0.01% beyond 100) — below min_wick_beyond_pct=0.03
    c = _candle(open=99, high=100.01, low=98, close=99, volume=1000)
    result = det.classify_level_interaction(c, lvl, _recent_candles(500), 99.0)
    assert result == "BOUNCE"


# ---------------------------------------------------------------------------
# classify_level_interaction — SWEEP (genuine stop hunt)
# ---------------------------------------------------------------------------

def test_classify_sweep_bsl():
    """
    BSL sweep: wick goes above level (0.1%), closes back below,
    large upper wick, volume spike.
    """
    det = _make_detector()
    lvl = _level(price=100.0, level_type="BSL")

    # wick_beyond = 100.10 - 100.0 = 0.10  (0.10% of 100)
    # wick_beyond_pct = 0.10%  → between 0.03 and 0.5
    # close = 99.8 < 100.0  → closed back inside
    # upper_wick = 100.10 - 99.9 = 0.20, lower_wick = 99.9 - 99.7 = 0.20
    # total_range = 0.40, upper_wick_fraction = 0.5 — need >= 0.60
    # Let's make a pin-bar: open=close=99.7, high=100.15, low=99.65
    # upper_wick = 100.15 - 99.7 = 0.45, lower_wick = 99.7 - 99.65 = 0.05
    # total_range = 0.50, wick_fraction = 0.90 — OK
    c = _candle(open=99.7, high=100.15, low=99.65, close=99.7, volume=2000)
    recent = _recent_candles(avg_volume=1000, count=10)  # volume_spike_ratio=1.5 → need 1500
    result = det.classify_level_interaction(c, lvl, recent, 99.7)
    assert result == "SWEEP"
    # pending sweep should have been added
    assert "NIFTY" in det._pending_sweeps
    assert len(det._pending_sweeps["NIFTY"]) == 1


def test_classify_sweep_ssl():
    """SSL sweep: wick goes below level, closes back above, volume spike."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="SSL")

    # pin bar below SSL: open=close=100.3, low=99.85, high=100.35
    # wick_beyond = 100.0 - 99.85 = 0.15 (0.15%) → in range
    # lower_wick = 100.3 - 99.85 = 0.45, total_range = 0.50
    # wick_fraction = 0.90 — OK
    # close = 100.3 > 100.0 → closed back inside (above)
    c = _candle(open=100.3, high=100.35, low=99.85, close=100.3, volume=2000)
    recent = _recent_candles(avg_volume=1000, count=10)
    result = det.classify_level_interaction(c, lvl, recent, 100.3)
    assert result == "SWEEP"


def test_classify_no_sweep_low_volume():
    """Wick qualifies geometrically but volume too low — should be TEST."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="BSL")
    c = _candle(open=99.7, high=100.15, low=99.65, close=99.7, volume=300)
    recent = _recent_candles(avg_volume=1000, count=10)  # need >= 1500
    result = det.classify_level_interaction(c, lvl, recent, 99.7)
    assert result == "TEST"


# ---------------------------------------------------------------------------
# classify_level_interaction — BREAKOUT
# ---------------------------------------------------------------------------

def test_classify_breakout_bsl():
    """
    Price closes above BSL with strong body and volume — BREAKOUT.
    """
    det = _make_detector()
    lvl = _level(price=100.0, level_type="BSL")
    # Closes above 100: open=99.5, high=100.8, low=99.4, close=100.7
    # body = 1.2, range = 1.4, body_ratio = 0.857 >= 0.40
    # volume spike
    c = _candle(open=99.5, high=100.8, low=99.4, close=100.7, volume=2500)
    recent = _recent_candles(avg_volume=1000, count=10)
    result = det.classify_level_interaction(c, lvl, recent, 100.7)
    assert result == "BREAKOUT"


def test_classify_breakout_ssl():
    """Price closes below SSL with strong body and volume — BREAKOUT."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="SSL")
    c = _candle(open=100.5, high=100.6, low=99.2, close=99.3, volume=2500)
    recent = _recent_candles(avg_volume=1000, count=10)
    result = det.classify_level_interaction(c, lvl, recent, 99.3)
    assert result == "BREAKOUT"


# ---------------------------------------------------------------------------
# _confirm_sweep — confirmation logic
# ---------------------------------------------------------------------------

def test_confirm_sweep_confirmed():
    """BSL sweep confirmed by bearish candle closing below level."""
    det = _make_detector()
    det._cs.get_candles.return_value = _recent_candles(1000)

    sweep_c = _candle(open=99.7, high=100.15, low=99.65, close=99.7, volume=2000)
    lvl = _level(price=100.0, level_type="BSL")

    pending = {
        "sweep_candle": sweep_c,
        "level": lvl,
        "wick_beyond_pct": 0.15,
        "is_bsl": True,
        "candles_seen": 1,
        "sweep_type": "BSL_SWEEP",
    }

    # Bearish confirmation candle closes well below 100
    # body = 100.2 - 99.5 = 0.7, range = 0.8, ratio = 0.875 >= 0.3
    conf_c = _candle(open=100.2, high=100.25, low=99.45, close=99.5, volume=1500, offset=1)
    result = det._confirm_sweep("NIFTY", pending, conf_c)

    assert result is not None
    assert isinstance(result, LiquiditySweeep)
    assert result.sweep_type == "BSL_SWEEP"
    assert result.reversion_confirmed is True


def test_confirm_sweep_rejected_wrong_direction():
    """Candle closes in same direction as sweep → not confirmed."""
    det = _make_detector()
    sweep_c = _candle(open=99.7, high=100.15, low=99.65, close=99.7, volume=2000)
    lvl = _level(price=100.0, level_type="BSL")

    pending = {
        "sweep_candle": sweep_c,
        "level": lvl,
        "wick_beyond_pct": 0.15,
        "is_bsl": True,
        "candles_seen": 1,
        "sweep_type": "BSL_SWEEP",
    }

    # Bullish candle — wrong direction for BSL sweep confirmation
    conf_c = _candle(open=99.5, high=100.3, low=99.4, close=100.2, volume=1500, offset=1)
    result = det._confirm_sweep("NIFTY", pending, conf_c)
    assert result is None


def test_confirm_sweep_rejected_doji():
    """Doji confirmation candle (tiny body) is rejected."""
    det = _make_detector()
    det._cs.get_candles.return_value = _recent_candles(1000)

    sweep_c = _candle(open=99.7, high=100.15, low=99.65, close=99.7, volume=2000)
    lvl = _level(price=100.0, level_type="BSL")

    pending = {
        "sweep_candle": sweep_c,
        "level": lvl,
        "wick_beyond_pct": 0.15,
        "is_bsl": True,
        "candles_seen": 1,
        "sweep_type": "BSL_SWEEP",
    }

    # Doji: open ≈ close, big wicks → body_ratio tiny
    conf_c = _candle(open=99.8, high=100.1, low=99.5, close=99.81, volume=1500, offset=1)
    result = det._confirm_sweep("NIFTY", pending, conf_c)
    assert result is None


# ---------------------------------------------------------------------------
# Pending sweep expiry
# ---------------------------------------------------------------------------

def test_pending_sweep_expires():
    """Pending sweep removed after sweep_confirmation_candles with no confirmation."""
    det = _make_detector()
    det._cs.get_candles.return_value = _recent_candles(1000)

    sweep_c = _candle(open=99.7, high=100.15, low=99.65, close=99.7, volume=2000)
    lvl = _level(price=100.0, level_type="BSL")

    # Manually inject a pending sweep
    det._pending_sweeps["NIFTY"] = [{
        "sweep_candle": sweep_c,
        "level": lvl,
        "wick_beyond_pct": 0.15,
        "is_bsl": True,
        "candles_seen": 0,
        "sweep_type": "BSL_SWEEP",
    }]

    # Feed confirmation_candles (default=3) non-confirming candles
    non_conf = _candle(open=99.9, high=100.05, low=99.85, close=99.95, volume=800, offset=1)
    for i in range(det._confirmation_candles):
        det._try_confirm_pending_sweeps("NIFTY", non_conf, is_5min=False)

    # Pending should be cleared
    assert det._pending_sweeps.get("NIFTY", []) == []


# ---------------------------------------------------------------------------
# validate_breakout
# ---------------------------------------------------------------------------

def test_validate_breakout_genuine():
    """Breakout followed by 3 confirming candles → GENUINE."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="BSL")

    breakout_c = _candle(open=99.5, high=101.0, low=99.4, close=100.8, volume=3000)

    candles_after = [
        _candle(open=100.7, high=101.2, low=100.5, close=101.0, volume=1500, offset=i)
        for i in range(1, 4)
    ]
    recent_for_avg = _recent_candles(avg_volume=1000, count=15)

    result = det.validate_breakout("NIFTY", lvl, breakout_c, candles_after + recent_for_avg)
    assert result["breakout_type"] == "GENUINE"
    assert result["is_valid"] is True
    assert result["confidence"] > 50


def test_validate_breakout_failed():
    """Price breaks above then returns below within 5 candles → FAILED."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="BSL")

    breakout_c = _candle(open=99.5, high=101.0, low=99.4, close=100.8, volume=3000)

    # First candle above, then comes back below
    candles_after = [
        _candle(open=100.7, high=101.0, low=100.5, close=100.9, volume=1200, offset=1),
        _candle(open=100.8, high=101.0, low=99.5, close=99.7, volume=2000, offset=2),  # back below
    ]

    result = det.validate_breakout("NIFTY", lvl, breakout_c, candles_after)
    assert result["breakout_type"] == "FAILED"
    assert result["is_valid"] is False


def test_validate_breakout_pending_not_enough_candles():
    """Fewer than confirmation_candles → PENDING."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="BSL")

    breakout_c = _candle(open=99.5, high=101.0, low=99.4, close=100.8, volume=3000)
    candles_after = [
        _candle(open=100.7, high=101.0, low=100.5, close=100.9, volume=1200, offset=1),
    ]

    result = det.validate_breakout("NIFTY", lvl, breakout_c, candles_after)
    assert result["breakout_type"] == "PENDING"


def test_validate_breakout_oi_boosts_confidence():
    """options_oi_confirms=True should boost confidence."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="BSL")

    breakout_c = _candle(open=99.5, high=101.0, low=99.4, close=100.8, volume=3000)
    candles_after = [
        _candle(open=100.7, high=101.2, low=100.5, close=101.0, volume=1500, offset=i)
        for i in range(1, 4)
    ]
    recent = _recent_candles(avg_volume=1000, count=15)

    without_oi = det.validate_breakout("NIFTY", lvl, breakout_c, candles_after + recent, False)
    with_oi = det.validate_breakout("NIFTY", lvl, breakout_c, candles_after + recent, True)

    assert with_oi["confidence"] > without_oi["confidence"]


# ---------------------------------------------------------------------------
# get_signal_override
# ---------------------------------------------------------------------------

def _make_sweep(
    sweep_type: str = "BSL_SWEEP",
    quality: str = "HIGH",
    minutes_ago: int = 5,
    reversion: bool = True,
) -> LiquiditySweeep:
    lvl = _level(price=100.0, level_type="BSL" if "BSL" in sweep_type else "SSL")
    sc = _candle(open=99.7, high=100.15, low=99.65, close=99.7, volume=2000)
    rc = _candle(open=100.0, high=100.1, low=99.4, close=99.5, volume=1500)
    ts = datetime.now(_IST) - timedelta(minutes=minutes_ago)
    return LiquiditySweeep(
        symbol="NIFTY",
        timeframe="5min",
        level=lvl,
        sweep_candle=sc,
        sweep_type=sweep_type,
        sweep_high=100.15,
        sweep_low=99.65,
        reversion_confirmed=reversion,
        timestamp=ts,
        reversion_candle=rc if reversion else None,
        signal_quality=quality,
    )


def test_signal_override_block_call_after_bsl_sweep():
    """BUY_CALL blocked after HIGH quality BSL sweep within 15 min (no reversion yet)."""
    det = _make_detector()
    sweep = _make_sweep("BSL_SWEEP", "HIGH", minutes_ago=8, reversion=False)
    result = det.get_signal_override("NIFTY", "BUY_CALL", 99.5, [sweep])
    assert result["action"] == "BLOCK"


def test_signal_override_reverse_call_after_confirmed_bsl_sweep():
    """BUY_CALL reversed after HIGH quality confirmed BSL sweep within 10 min."""
    det = _make_detector()
    sweep = _make_sweep("BSL_SWEEP", "HIGH", minutes_ago=5, reversion=True)
    result = det.get_signal_override("NIFTY", "BUY_CALL", 99.5, [sweep])
    assert result["action"] == "REVERSE"
    assert result["confidence_adjustment"] == pytest.approx(10.0)


def test_signal_override_boost_call_after_ssl_sweep():
    """BUY_CALL boosted after HIGH quality confirmed SSL sweep within 20 min."""
    det = _make_detector()
    sweep = _make_sweep("SSL_SWEEP", "HIGH", minutes_ago=12, reversion=True)
    result = det.get_signal_override("NIFTY", "BUY_CALL", 100.3, [sweep])
    assert result["action"] == "ALLOW"
    assert result["confidence_adjustment"] == pytest.approx(15.0)


def test_signal_override_allow_no_sweeps():
    """No recent sweeps → ALLOW with zero adjustment."""
    det = _make_detector()
    result = det.get_signal_override("NIFTY", "BUY_CALL", 100.0, [])
    assert result["action"] == "ALLOW"
    assert result["confidence_adjustment"] == pytest.approx(0.0)


def test_signal_override_low_quality_sweep_does_not_block():
    """LOW quality BSL sweep within 15 min should NOT block BUY_CALL."""
    det = _make_detector()
    sweep = _make_sweep("BSL_SWEEP", "LOW", minutes_ago=8, reversion=False)
    result = det.get_signal_override("NIFTY", "BUY_CALL", 99.5, [sweep])
    # LOW quality → falls through to ALLOW (no overriding condition)
    assert result["action"] == "ALLOW"


# ---------------------------------------------------------------------------
# _calculate_signal_quality
# ---------------------------------------------------------------------------

def test_signal_quality_high():
    """Strong volume, precise wick, strong reversal body → HIGH."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="BSL")

    # vol_ratio = 4.0 → 40pts; wick_beyond = 0.05% → 30pts; body = 0.8 → 30pts = 100
    sweep_c = _candle(open=99.7, high=100.05, low=99.65, close=99.7, volume=4000)
    # strong bearish reversal: body = open - close = 1.0, range = 1.1 → 0.91
    rev_c = _candle(open=100.1, high=100.15, low=99.05, close=99.1, volume=2000)
    recent = _recent_candles(avg_volume=1000, count=10)

    quality = det._calculate_signal_quality(sweep_c, rev_c, lvl, recent)
    assert quality == "HIGH"


def test_signal_quality_low():
    """Weak volume, large wick, doji reversal → LOW."""
    det = _make_detector()
    lvl = _level(price=100.0, level_type="BSL")

    # vol_ratio = 1.0 → 0pts; wick = 0.4% → 5pts; body ≈ 0.05 → 5pts = 10
    sweep_c = _candle(open=99.7, high=100.40, low=99.65, close=99.7, volume=1000)
    rev_c = _candle(open=99.8, high=100.1, low=99.5, close=99.81, volume=900)
    recent = _recent_candles(avg_volume=1000, count=10)

    quality = det._calculate_signal_quality(sweep_c, rev_c, lvl, recent)
    assert quality == "LOW"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_avg_volume_empty():
    assert _avg_volume([]) == 0.0


def test_avg_volume_correct():
    candles = [_candle(volume=v) for v in [1000, 2000, 3000]]
    assert _avg_volume(candles) == pytest.approx(2000.0)


def test_wick_fraction_bsl():
    # upper_wick = 100.2 - 100.0 = 0.2, total_range = 100.2 - 99.6 = 0.6
    c = _candle(open=100.0, high=100.2, low=99.6, close=100.0)
    assert _wick_fraction(c, is_bsl=True) == pytest.approx(0.2 / 0.6)


def test_wick_fraction_ssl():
    # lower_wick = 99.8 - 99.6 = 0.2, total_range = 100.4 - 99.6 = 0.8
    c = _candle(open=99.8, high=100.4, low=99.6, close=99.8)
    assert _wick_fraction(c, is_bsl=False) == pytest.approx(0.2 / 0.8)
