"""
Tests for ImbalanceDetector.

Uses MagicMock for CandleStore and DatabaseManager.
All tests are fully isolated — no real DB, network, or file I/O.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.analysis.liquidity.imbalance_detector import (
    ImbalanceDetector,
    _estimate_atr,
    _find_displacement,
    _overlaps_or_adjacent,
)
from src.analysis.liquidity.models import Candle, FairValueGap, OrderBlock

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 5, 11, 9, 15, tzinfo=_IST)
    return base + timedelta(minutes=offset_minutes)


def _candle(
    symbol: str = "NIFTY",
    timeframe: str = "5min",
    open: float = 100.0,
    high: float = 102.0,
    low: float = 98.0,
    close: float = 101.0,
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


def _make_detector() -> ImbalanceDetector:
    mock_db = MagicMock()
    mock_db.execute.return_value = MagicMock(lastrowid=1)
    mock_db.execute_many.return_value = None
    mock_db.fetch_all.return_value = []

    mock_cs = MagicMock()
    mock_cs.get_candles.return_value = []

    with patch("src.analysis.liquidity.imbalance_detector.settings") as mock_settings:
        mock_settings.imbalance.min_fvg_size_pct = 0.05
        mock_settings.imbalance.max_fvg_age_hours = 24
        mock_settings.imbalance.min_ob_displacement_pct = 0.3
        mock_settings.imbalance.max_ob_age_hours = 48
        mock_settings.imbalance.fvg_fill_threshold_pct = 50.0
        mock_settings.imbalance.ob_mitigation_threshold_pct = 100.0
        det = ImbalanceDetector(candle_store=mock_cs, db=mock_db)

    return det


# ---------------------------------------------------------------------------
# FVG detection — Bullish
# ---------------------------------------------------------------------------

def test_detect_bullish_fvg():
    """3-candle pattern where c3.low > c1.high → bullish FVG."""
    det = _make_detector()

    # c1: high=100, c2: impulse up, c3: low=101 → gap between 100 and 101 = 1 (1% of 101)
    c1 = _candle(open=98, high=100, low=97, close=99, offset=0)
    c2 = _candle(open=99, high=105, low=98, close=104, offset=5)
    c3 = _candle(open=103, high=107, low=101, close=106, offset=10)

    fvgs = det._detect_fvgs([c1, c2, c3], "5min")

    assert len(fvgs) == 1
    fvg = fvgs[0]
    assert fvg.gap_type == "BULLISH"
    assert fvg.gap_bottom == pytest.approx(100.0)  # c1.high
    assert fvg.gap_top == pytest.approx(101.0)      # c3.low
    assert fvg.gap_midpoint == pytest.approx(100.5)
    assert fvg.gap_size == pytest.approx(1.0)
    assert fvg.candle_1 is c1
    assert fvg.candle_2 is c2
    assert fvg.candle_3 is c3


# ---------------------------------------------------------------------------
# FVG detection — Bearish
# ---------------------------------------------------------------------------

def test_detect_bearish_fvg():
    """3-candle pattern where c3.high < c1.low → bearish FVG."""
    det = _make_detector()

    # c1: low=100, c3: high=99 → gap between 99 and 100 = 1 (1% of 99)
    c1 = _candle(open=103, high=105, low=100, close=101, offset=0)
    c2 = _candle(open=101, high=101, low=95, close=96, offset=5)
    c3 = _candle(open=97, high=99, low=94, close=95, offset=10)

    fvgs = det._detect_fvgs([c1, c2, c3], "5min")

    assert len(fvgs) == 1
    fvg = fvgs[0]
    assert fvg.gap_type == "BEARISH"
    assert fvg.gap_top == pytest.approx(100.0)    # c1.low
    assert fvg.gap_bottom == pytest.approx(99.0)  # c3.high
    assert fvg.gap_midpoint == pytest.approx(99.5)


# ---------------------------------------------------------------------------
# FVG minimum size filter
# ---------------------------------------------------------------------------

def test_fvg_minimum_size_filter_rejects_tiny_gap():
    """Gap of 0.03% should be rejected (< min_fvg_size_pct=0.05%)."""
    det = _make_detector()

    # gap = 0.03 on price ~100 → 0.03%
    c1 = _candle(open=98, high=100.00, low=97, close=99, offset=0)
    c2 = _candle(open=99, high=103, low=98, close=102, offset=5)
    c3 = _candle(open=101, high=104, low=100.03, close=103, offset=10)

    fvgs = det._detect_fvgs([c1, c2, c3], "5min")
    assert len(fvgs) == 0


def test_fvg_minimum_size_filter_accepts_valid_gap():
    """Gap of 1% should pass the filter."""
    det = _make_detector()

    c1 = _candle(open=98, high=100.0, low=97, close=99, offset=0)
    c2 = _candle(open=99, high=105, low=98, close=104, offset=5)
    c3 = _candle(open=103, high=108, low=101.0, close=107, offset=10)

    fvgs = det._detect_fvgs([c1, c2, c3], "5min")
    assert len(fvgs) == 1


# ---------------------------------------------------------------------------
# FVG fill tracking
# ---------------------------------------------------------------------------

def test_fvg_partial_fill_bullish():
    """Price enters bullish FVG but does not close below → partial fill."""
    det = _make_detector()

    c1 = _candle(open=98, high=100.0, low=97, close=99, offset=0)
    c2 = _candle(open=99, high=105, low=98, close=104, offset=5)
    c3 = _candle(open=103, high=108, low=102.0, close=107, offset=10)

    fvgs = det._detect_fvgs([c1, c2, c3], "5min")
    fvg = fvgs[0]
    # gap_bottom=100, gap_top=102, gap_size=2

    # Candle whose low enters the gap (low=101 = halfway in)
    c_fill = _candle(open=104, high=105, low=101.0, close=103.5, offset=15)
    updated = det._update_fvg_fill_status(fvg, c_fill)

    # fill_pct = (102 - 101) / 2 * 100 = 50%
    assert updated.fill_pct == pytest.approx(50.0)
    assert updated.is_filled is True       # 50 >= threshold 50%
    assert updated.is_mitigated is False


def test_fvg_full_mitigation_bullish():
    """Price closes below gap_bottom → fully mitigated."""
    det = _make_detector()

    c1 = _candle(open=98, high=100.0, low=97, close=99, offset=0)
    c2 = _candle(open=99, high=105, low=98, close=104, offset=5)
    c3 = _candle(open=103, high=108, low=102.0, close=107, offset=10)

    fvgs = det._detect_fvgs([c1, c2, c3], "5min")
    fvg = fvgs[0]

    # Close below gap_bottom=100
    c_mit = _candle(open=102, high=103, low=98.5, close=99.0, offset=20)
    updated = det._update_fvg_fill_status(fvg, c_mit)

    assert updated.is_mitigated is True
    assert updated.fill_pct == pytest.approx(100.0)


def test_fvg_partial_fill_bearish():
    """Price enters bearish FVG from below → partial fill."""
    det = _make_detector()

    c1 = _candle(open=103, high=105, low=100.0, close=101, offset=0)
    c2 = _candle(open=101, high=101, low=95, close=96, offset=5)
    c3 = _candle(open=97, high=99.0, low=94, close=95, offset=10)

    fvgs = det._detect_fvgs([c1, c2, c3], "5min")
    fvg = fvgs[0]
    # gap_bottom=99, gap_top=100, gap_size=1

    # Price retraces up, high enters gap at 99.5 (halfway)
    c_fill = _candle(open=96, high=99.5, low=95.5, close=96.5, offset=15)
    updated = det._update_fvg_fill_status(fvg, c_fill)

    # fill_pct = (99.5 - 99) / 1 * 100 = 50%
    assert updated.fill_pct == pytest.approx(50.0)
    assert updated.is_mitigated is False


def test_fvg_fill_does_not_decrease():
    """fill_pct should only ever increase (max tracking)."""
    det = _make_detector()

    c1 = _candle(open=98, high=100.0, low=97, close=99, offset=0)
    c2 = _candle(open=99, high=105, low=98, close=104, offset=5)
    c3 = _candle(open=103, high=108, low=102.0, close=107, offset=10)

    fvgs = det._detect_fvgs([c1, c2, c3], "5min")
    fvg = fvgs[0]
    # gap: 100–102

    # First touch: low=101 → 50%
    c1f = _candle(open=104, high=105, low=101.0, close=103, offset=15)
    det._update_fvg_fill_status(fvg, c1f)
    assert fvg.fill_pct == pytest.approx(50.0)

    # Second touch: low=101.5 (less deep) → still 50%, not decreased
    c2f = _candle(open=104, high=105, low=101.5, close=103.5, offset=20)
    det._update_fvg_fill_status(fvg, c2f)
    assert fvg.fill_pct == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# OB detection — Bullish
# ---------------------------------------------------------------------------

def test_detect_bullish_order_block():
    """Last bearish candle before strong bullish displacement → Bullish OB."""
    det = _make_detector()

    # OB candle: bearish (close < open)
    ob_c = _candle(open=102.0, high=103.0, low=100.0, close=100.5, offset=0)

    # Displacement: 2 consecutive strong bullish candles
    # Each has body >= 60% of range
    d1 = _candle(open=100.5, high=103.5, low=100.4, close=103.4, offset=5)
    # body=2.9, range=3.1 → 93% body — OK
    d2 = _candle(open=103.4, high=106.5, low=103.3, close=106.4, offset=10)
    # body=3.0, range=3.2 → 93% — OK
    # combined move: (106.4 - 100.5) / 100.5 * 100 = ~5.87% ≫ 0.3% threshold

    candles = [ob_c, d1, d2]
    obs = det._detect_order_blocks(candles, "5min")

    assert len(obs) >= 1
    bullish_obs = [o for o in obs if o.ob_type == "BULLISH"]
    assert len(bullish_obs) >= 1
    ob = bullish_obs[0]
    assert ob.ob_high == pytest.approx(103.0)
    assert ob.ob_low == pytest.approx(100.0)
    assert ob.strength in ("STRONG", "MEDIUM", "WEAK")


def test_detect_bearish_order_block():
    """Last bullish candle before strong bearish displacement → Bearish OB."""
    det = _make_detector()

    ob_c = _candle(open=100.0, high=103.0, low=99.5, close=102.5, offset=0)

    d1 = _candle(open=102.5, high=102.6, low=99.4, close=99.5, offset=5)
    d2 = _candle(open=99.5, high=99.6, low=96.3, close=96.4, offset=10)

    candles = [ob_c, d1, d2]
    obs = det._detect_order_blocks(candles, "5min")

    bearish_obs = [o for o in obs if o.ob_type == "BEARISH"]
    assert len(bearish_obs) >= 1
    ob = bearish_obs[0]
    assert ob.ob_high == pytest.approx(103.0)
    assert ob.ob_low == pytest.approx(99.5)


# ---------------------------------------------------------------------------
# OB mitigation
# ---------------------------------------------------------------------------

def _make_bullish_ob(symbol: str = "NIFTY") -> OrderBlock:
    origin = _candle(open=102, high=103, low=100, close=100.5, offset=0)
    disp = _candle(open=100.5, high=106, low=100.4, close=105.9, offset=5)
    return OrderBlock(
        symbol=symbol,
        timeframe="5min",
        ob_type="BULLISH",
        ob_high=103.0,
        ob_low=100.0,
        ob_midpoint=101.5,
        origin_candle=origin,
        displacement_candle=disp,
        displacement_size_pct=5.0,
        formed_at=_ts(0),
    )


def _make_bearish_ob(symbol: str = "NIFTY") -> OrderBlock:
    origin = _candle(open=100, high=103, low=99.5, close=102.5, offset=0)
    disp = _candle(open=102.5, high=102.6, low=96, close=96.1, offset=5)
    return OrderBlock(
        symbol=symbol,
        timeframe="5min",
        ob_type="BEARISH",
        ob_high=103.0,
        ob_low=99.5,
        ob_midpoint=101.25,
        origin_candle=origin,
        displacement_candle=disp,
        displacement_size_pct=6.0,
        formed_at=_ts(0),
    )


def test_ob_mitigation_bullish():
    """Bullish OB mitigated when price closes below ob_low."""
    det = _make_detector()
    ob = _make_bullish_ob()

    mitigation_candle = _candle(open=101, high=101.5, low=99.0, close=99.5, offset=15)
    updated = det._update_ob_mitigation(ob, mitigation_candle)

    assert updated.is_mitigated is True
    assert updated.mitigated_at == mitigation_candle.timestamp


def test_ob_not_mitigated_bullish_when_close_above_low():
    """Bullish OB NOT mitigated when close stays above ob_low."""
    det = _make_detector()
    ob = _make_bullish_ob()

    # close=100.5 > ob_low=100 → no mitigation
    candle = _candle(open=102, high=103, low=100.2, close=100.5, offset=15)
    updated = det._update_ob_mitigation(ob, candle)

    assert updated.is_mitigated is False


def test_ob_touch_count_increments():
    """Touch count increments when price enters OB zone without mitigating."""
    det = _make_detector()
    ob = _make_bullish_ob()

    # Price closes inside zone: 100 < 101.5 < 103
    touch_candle = _candle(open=103, high=103.5, low=100.5, close=101.5, offset=15)
    det._update_ob_mitigation(ob, touch_candle)

    assert ob.touch_count == 1


def test_ob_mitigation_bearish():
    """Bearish OB mitigated when price closes above ob_high."""
    det = _make_detector()
    ob = _make_bearish_ob()

    candle = _candle(open=101, high=104.5, low=100.5, close=103.5, offset=15)
    updated = det._update_ob_mitigation(ob, candle)

    assert updated.is_mitigated is True


# ---------------------------------------------------------------------------
# get_imbalance_signal — scenario tests
# ---------------------------------------------------------------------------

def _inject_fvg(
    det: ImbalanceDetector,
    symbol: str,
    gap_type: str,
    gap_bottom: float,
    gap_top: float,
    timeframe: str = "5min",
) -> FairValueGap:
    c1 = _candle(offset=0)
    c2 = _candle(offset=5)
    c3 = _candle(offset=10)
    fvg = FairValueGap(
        symbol=symbol,
        timeframe=timeframe,
        gap_type=gap_type,
        gap_top=gap_top,
        gap_bottom=gap_bottom,
        gap_midpoint=(gap_top + gap_bottom) / 2,
        gap_size=gap_top - gap_bottom,
        gap_size_pct=(gap_top - gap_bottom) / gap_top * 100,
        formed_at=_ts(0),
        candle_1=c1,
        candle_2=c2,
        candle_3=c3,
    )
    det._active_fvgs.setdefault(symbol, []).append(fvg)
    return fvg


def _inject_ob(
    det: ImbalanceDetector,
    symbol: str,
    ob_type: str,
    ob_low: float,
    ob_high: float,
    disp_pct: float = 1.0,
) -> OrderBlock:
    origin = _candle(offset=0)
    disp = _candle(offset=5)
    ob = OrderBlock(
        symbol=symbol,
        timeframe="5min",
        ob_type=ob_type,
        ob_high=ob_high,
        ob_low=ob_low,
        ob_midpoint=(ob_high + ob_low) / 2,
        origin_candle=origin,
        displacement_candle=disp,
        displacement_size_pct=disp_pct,
        formed_at=_ts(0),
    )
    det._active_obs.setdefault(symbol, []).append(ob)
    return ob


def test_signal_bullish_fvg():
    """Price inside bullish FVG → BULLISH signal, strength=70."""
    det = _make_detector()
    _inject_fvg(det, "NIFTY", "BULLISH", gap_bottom=100.0, gap_top=102.0)

    candle = _candle(open=101, high=102, low=100.5, close=101.0, offset=15)
    det._cs.get_candles.return_value = [candle]
    result = det.get_imbalance_signal("NIFTY", 101.0, candle)

    assert result["signal"] == "BULLISH"
    assert result["strength"] == pytest.approx(70.0)
    assert "bullish FVG" in result["reason"]
    assert result["invalidation"] < 100.0


def test_signal_bearish_fvg():
    """Price inside bearish FVG → BEARISH signal, strength=70."""
    det = _make_detector()
    _inject_fvg(det, "NIFTY", "BEARISH", gap_bottom=98.0, gap_top=100.0)

    candle = _candle(open=99, high=99.5, low=98.5, close=99.0, offset=15)
    det._cs.get_candles.return_value = [candle]
    result = det.get_imbalance_signal("NIFTY", 99.0, candle)

    assert result["signal"] == "BEARISH"
    assert result["strength"] == pytest.approx(70.0)
    assert "bearish FVG" in result["reason"]


def test_signal_bullish_ob():
    """Price inside bullish OB → BULLISH signal, strength=75."""
    det = _make_detector()
    _inject_ob(det, "NIFTY", "BULLISH", ob_low=100.0, ob_high=103.0)

    candle = _candle(open=102, high=102.5, low=101.0, close=101.5, offset=15)
    det._cs.get_candles.return_value = [candle]
    result = det.get_imbalance_signal("NIFTY", 101.5, candle)

    assert result["signal"] == "BULLISH"
    assert result["strength"] == pytest.approx(75.0)
    assert "bullish OB" in result["reason"]


def test_signal_bearish_ob():
    """Price inside bearish OB → BEARISH signal, strength=75."""
    det = _make_detector()
    _inject_ob(det, "NIFTY", "BEARISH", ob_low=100.0, ob_high=103.0)

    candle = _candle(open=101, high=102, low=100.5, close=101.0, offset=15)
    det._cs.get_candles.return_value = [candle]
    result = det.get_imbalance_signal("NIFTY", 101.0, candle)

    assert result["signal"] == "BEARISH"
    assert result["strength"] == pytest.approx(75.0)
    assert "bearish OB" in result["reason"]


def test_signal_neutral_no_structures():
    """No active FVGs or OBs → NEUTRAL signal."""
    det = _make_detector()
    candle = _candle(open=100, high=101, low=99, close=100, offset=0)
    det._cs.get_candles.return_value = [candle]
    result = det.get_imbalance_signal("NIFTY", 100.0, candle)

    assert result["signal"] == "NEUTRAL"
    assert result["strength"] == pytest.approx(0.0)


def test_signal_stacked_bullish_fvgs_boost():
    """Two stacked bullish FVGs → strength boosted by 15 per additional."""
    det = _make_detector()
    # FVG1: 98–100, FVG2: 97–99  (overlapping)
    _inject_fvg(det, "NIFTY", "BULLISH", gap_bottom=98.0, gap_top=100.0)
    _inject_fvg(det, "NIFTY", "BULLISH", gap_bottom=97.0, gap_top=99.0)

    # Price at 99.0 — inside FVG1; close=99 keeps both FVGs alive (above both gap_bottoms)
    candle = _candle(open=99.5, high=100.0, low=98.5, close=99.0, offset=15)
    det._cs.get_candles.return_value = [candle]
    result = det.get_imbalance_signal("NIFTY", 99.0, candle)

    assert result["signal"] == "BULLISH"
    # strength = 70 + 1 * 15 = 85
    assert result["strength"] == pytest.approx(85.0)


def test_signal_ob_fvg_confluence_boost():
    """OB and FVG at same price → strength = max(75, 70) + 15 = 90."""
    det = _make_detector()
    _inject_fvg(det, "NIFTY", "BULLISH", gap_bottom=100.0, gap_top=102.0)
    _inject_ob(det, "NIFTY", "BULLISH", ob_low=100.0, ob_high=103.0)

    candle = _candle(open=101, high=102, low=100.5, close=101.0, offset=15)
    det._cs.get_candles.return_value = [candle]
    result = det.get_imbalance_signal("NIFTY", 101.0, candle)

    assert result["signal"] == "BULLISH"
    assert result["strength"] == pytest.approx(90.0)
    assert "confluence" in result["reason"]


# ---------------------------------------------------------------------------
# get_ob_at_price / get_fvg_at_price
# ---------------------------------------------------------------------------

def test_get_ob_at_price_found():
    det = _make_detector()
    _inject_ob(det, "NIFTY", "BULLISH", ob_low=100.0, ob_high=103.0)
    ob = det.get_ob_at_price("NIFTY", 101.5)
    assert ob is not None
    assert ob.ob_type == "BULLISH"


def test_get_ob_at_price_not_found():
    det = _make_detector()
    _inject_ob(det, "NIFTY", "BULLISH", ob_low=100.0, ob_high=103.0)
    ob = det.get_ob_at_price("NIFTY", 105.0)
    assert ob is None


def test_get_fvg_at_price_found():
    det = _make_detector()
    _inject_fvg(det, "NIFTY", "BULLISH", gap_bottom=100.0, gap_top=102.0)
    fvg = det.get_fvg_at_price("NIFTY", 101.0)
    assert fvg is not None
    assert fvg.gap_type == "BULLISH"


def test_get_fvg_at_price_not_found():
    det = _make_detector()
    _inject_fvg(det, "NIFTY", "BULLISH", gap_bottom=100.0, gap_top=102.0)
    fvg = det.get_fvg_at_price("NIFTY", 105.0)
    assert fvg is None


# ---------------------------------------------------------------------------
# get_nearest_fvg
# ---------------------------------------------------------------------------

def test_get_nearest_fvg_above():
    det = _make_detector()
    _inject_fvg(det, "NIFTY", "BEARISH", gap_bottom=105.0, gap_top=107.0)
    _inject_fvg(det, "NIFTY", "BEARISH", gap_bottom=110.0, gap_top=112.0)

    fvg = det.get_nearest_fvg("NIFTY", 100.0, direction="above")
    assert fvg is not None
    assert fvg.gap_bottom == pytest.approx(105.0)


def test_get_nearest_fvg_below():
    det = _make_detector()
    _inject_fvg(det, "NIFTY", "BULLISH", gap_bottom=90.0, gap_top=92.0)
    _inject_fvg(det, "NIFTY", "BULLISH", gap_bottom=85.0, gap_top=87.0)

    fvg = det.get_nearest_fvg("NIFTY", 100.0, direction="below")
    assert fvg is not None
    assert fvg.gap_top == pytest.approx(92.0)


# ---------------------------------------------------------------------------
# get_nearest_ob
# ---------------------------------------------------------------------------

def test_get_nearest_ob():
    det = _make_detector()
    _inject_ob(det, "NIFTY", "BULLISH", ob_low=90.0, ob_high=92.0)
    _inject_ob(det, "NIFTY", "BULLISH", ob_low=80.0, ob_high=82.0)

    ob = det.get_nearest_ob("NIFTY", 100.0, ob_type="BULLISH")
    assert ob is not None
    assert ob.ob_low == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_find_displacement_bullish():
    candles = [
        _candle(open=100.0, high=103.0, low=99.9, close=102.9, offset=0),   # strong bullish
        _candle(open=102.9, high=106.0, low=102.8, close=105.9, offset=5),  # strong bullish
    ]
    result = _find_displacement(candles, 0, 0.3)
    assert result is not None
    disp_candles, pct, direction = result
    assert direction == "UP"
    assert pct > 0.3


def test_find_displacement_none_on_weak_body():
    """Candles with body < 60% of range should not count as displacement."""
    candles = [
        _candle(open=100, high=105, low=99, close=101, offset=0),  # body=1, range=6 → 17%
        _candle(open=101, high=106, low=100, close=102, offset=5),
    ]
    result = _find_displacement(candles, 0, 0.3)
    assert result is None


def test_estimate_atr_basic():
    candles = [_candle(open=100, high=102, low=98, close=101, offset=i) for i in range(14)]
    atr = _estimate_atr(candles)
    assert atr == pytest.approx(4.0)


def test_estimate_atr_empty():
    assert _estimate_atr([]) == 0.0


def test_overlaps_or_adjacent_true():
    c1 = _candle(offset=0)
    c2 = _candle(offset=5)
    c3 = _candle(offset=10)
    a = FairValueGap(
        symbol="NIFTY", timeframe="5min", gap_type="BULLISH",
        gap_top=102.0, gap_bottom=100.0, gap_midpoint=101.0,
        gap_size=2.0, gap_size_pct=2.0, formed_at=_ts(0),
        candle_1=c1, candle_2=c2, candle_3=c3,
    )
    b = FairValueGap(
        symbol="NIFTY", timeframe="5min", gap_type="BULLISH",
        gap_top=103.0, gap_bottom=101.5, gap_midpoint=102.25,
        gap_size=1.5, gap_size_pct=1.5, formed_at=_ts(5),
        candle_1=c1, candle_2=c2, candle_3=c3,
    )
    assert _overlaps_or_adjacent(a, b) is True


def test_overlaps_or_adjacent_false():
    c1 = _candle(offset=0)
    c2 = _candle(offset=5)
    c3 = _candle(offset=10)
    a = FairValueGap(
        symbol="NIFTY", timeframe="5min", gap_type="BULLISH",
        gap_top=102.0, gap_bottom=100.0, gap_midpoint=101.0,
        gap_size=2.0, gap_size_pct=2.0, formed_at=_ts(0),
        candle_1=c1, candle_2=c2, candle_3=c3,
    )
    b = FairValueGap(
        symbol="NIFTY", timeframe="5min", gap_type="BULLISH",
        gap_top=108.0, gap_bottom=106.0, gap_midpoint=107.0,
        gap_size=2.0, gap_size_pct=2.0, formed_at=_ts(5),
        candle_1=c1, candle_2=c2, candle_3=c3,
    )
    assert _overlaps_or_adjacent(a, b) is False
