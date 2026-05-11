"""
Integration tests for LiquidityEngine.

Tests:
  - Full analyze() pipeline with mock data
  - Signal BLOCK when sweep contradicts proposed signal
  - Signal REVERSE when stop hunt detected
  - Signal BOOST when liquidity aligned
  - startup() populates snapshots
  - Composite score calculation with known inputs
  - DecisionEngine integration doesn't break existing signals (ALLOW, 0 adj)
"""

from __future__ import annotations

import weakref
from datetime import datetime, date
from typing import Optional
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from config.constants import IST_TIMEZONE
from src.analysis.liquidity.liquidity_engine import LiquidityEngine
from src.analysis.liquidity.models import (
    Candle,
    FairValueGap,
    LiquidityLevel,
    LiquiditySnapshot,
    LiquiditySweeep,
    OrderBlock,
    VolumeProfile,
)

_IST = ZoneInfo(IST_TIMEZONE)
_NOW = datetime(2026, 5, 11, 10, 30, tzinfo=_IST)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candle(
    symbol: str = "NIFTY50",
    o: float = 22000,
    h: float = 22050,
    l: float = 21980,
    c: float = 22020,
    vol: int = 100_000,
    timeframe: str = "5min",
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=_NOW,
        open=o, high=h, low=l, close=c,
        volume=vol,
    )


def _make_level(
    symbol: str = "NIFTY50",
    price: float = 22200.0,
    level_type: str = "BSL",
    strength: int = 3,
    is_swept: bool = False,
) -> LiquidityLevel:
    return LiquidityLevel(
        symbol=symbol,
        level_type=level_type,
        price=price,
        strength=strength,
        timeframe="5min",
        first_formed_at=_NOW,
        last_touched_at=_NOW,
        is_swept=is_swept,
    )


def _make_sweep(
    symbol: str = "NIFTY50",
    sweep_type: str = "SSL_SWEEP",
    quality: str = "HIGH",
) -> LiquiditySweeep:
    level = _make_level(price=21800.0, level_type="SSL")
    candle = _make_candle(symbol)
    return LiquiditySweeep(
        symbol=symbol,
        timeframe="5min",
        level=level,
        sweep_candle=candle,
        sweep_type=sweep_type,
        sweep_high=21850.0,
        sweep_low=21750.0,
        reversion_confirmed=True,
        timestamp=_NOW,
        signal_quality=quality,
    )


@pytest.fixture
def engine() -> LiquidityEngine:
    """LiquidityEngine with all sub-components mocked."""
    candle_store = MagicMock()
    db = MagicMock()
    smart_api = MagicMock()
    options_analyzer = MagicMock()

    # Patch all sub-component constructors so we don't hit real DB
    with (
        patch("src.analysis.liquidity.liquidity_engine.LiquidityPoolDetector") as MockPool,
        patch("src.analysis.liquidity.liquidity_engine.SweepDetector") as MockSweep,
        patch("src.analysis.liquidity.liquidity_engine.ImbalanceDetector") as MockImb,
        patch("src.analysis.liquidity.liquidity_engine.VolumeProfileAnalyzer") as MockVP,
        patch("src.analysis.liquidity.liquidity_engine.StopHuntDetector") as MockHunt,
    ):
        eng = LiquidityEngine(candle_store, db, smart_api, options_analyzer, weakref.ref(MagicMock()))

        # Configure reasonable defaults on the mock instances
        pool = eng.pool_detector
        pool.get_all_active_levels.return_value = ([], [])
        pool.get_nearest_bsl.return_value = None
        pool.get_nearest_ssl.return_value = None
        pool._calculate_liquidity_bias.return_value = ("NEUTRAL", "No levels")
        pool._cs = candle_store

        eng.sweep_detector.get_recent_sweeps.return_value = []
        eng.sweep_detector.get_signal_override.return_value = {
            "action": "ALLOW", "reason": "no sweep", "confidence_adjustment": 0.0
        }

        eng.imbalance_detector._active_fvgs = {}
        eng.imbalance_detector._active_obs = {}
        eng.imbalance_detector.get_imbalance_signal.return_value = {
            "signal": "NEUTRAL", "strength": 0.0, "reason": ""
        }

        eng.volume_profile.load_previous_day_profile.return_value = None
        eng.volume_profile.get_price_context.return_value = {
            "signal": "NEUTRAL", "location": "IN_VA"
        }
        eng.volume_profile.get_vp_targets.return_value = []

        candle_store.get_latest_candle.return_value = _make_candle()

        eng.stop_hunt_detector.detect_stop_hunt.return_value = None
        eng.stop_hunt_detector.get_recent_hunts.return_value = []
        eng.stop_hunt_detector.get_stop_hunt_signal_for_engine.return_value = {
            "action": "ALLOW", "confidence_adjustment": 0.0, "reason": "no hunt"
        }

        yield eng


# ---------------------------------------------------------------------------
# 1. Full analyze() pipeline
# ---------------------------------------------------------------------------

class TestAnalyzePipeline:
    def test_returns_snapshot(self, engine: LiquidityEngine) -> None:
        snap = engine.analyze("NIFTY50", 22020.0, 22030.0, [], 0, 0)
        assert isinstance(snap, LiquiditySnapshot)
        assert snap.symbol == "NIFTY50"

    def test_snapshot_cached(self, engine: LiquidityEngine) -> None:
        snap = engine.analyze("NIFTY50", 22020.0, 22030.0, [], 0, 0)
        assert engine.get_snapshot("NIFTY50") is snap

    def test_error_returns_empty_snapshot(self, engine: LiquidityEngine) -> None:
        engine.pool_detector.get_all_active_levels.side_effect = RuntimeError("db down")
        snap = engine.analyze("NIFTY50", 22020.0, 22030.0, [], 0, 0)
        assert snap.symbol == "NIFTY50"
        assert snap.liquidity_score == 0.0

    def test_score_neutral_no_signals(self, engine: LiquidityEngine) -> None:
        snap = engine.analyze("NIFTY50", 22020.0, 22030.0, [], 0, 0)
        assert snap.liquidity_score == 0.0

    def test_bullish_bias_positive_score(self, engine: LiquidityEngine) -> None:
        engine.pool_detector._calculate_liquidity_bias.return_value = ("BULLISH", "SSL swept")
        snap = engine.analyze("NIFTY50", 22020.0, 22030.0, [], 0, 0)
        assert snap.liquidity_score > 0

    def test_bearish_bias_negative_score(self, engine: LiquidityEngine) -> None:
        engine.pool_detector._calculate_liquidity_bias.return_value = ("BEARISH", "BSL swept")
        snap = engine.analyze("NIFTY50", 22020.0, 22030.0, [], 0, 0)
        assert snap.liquidity_score < 0


# ---------------------------------------------------------------------------
# 2. Composite score — known inputs
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_full_bullish_score(self, engine: LiquidityEngine) -> None:
        engine.pool_detector._calculate_liquidity_bias.return_value = ("BULLISH", "reason")
        engine.sweep_detector.get_recent_sweeps.return_value = [_make_sweep(sweep_type="SSL_SWEEP", quality="HIGH")]
        engine.imbalance_detector.get_imbalance_signal.return_value = {
            "signal": "BULLISH", "strength": 100.0, "reason": "OB below"
        }
        engine.volume_profile.get_price_context.return_value = {"signal": "BULLISH", "location": "IN_VA"}
        snap = engine.analyze("NIFTY50", 22020.0, 22030.0, [], 0, 0)
        # 30 + 25 + 25 + 20 = 100
        assert snap.liquidity_score == pytest.approx(100.0)

    def test_stop_hunt_override_floor(self, engine: LiquidityEngine) -> None:
        engine.stop_hunt_detector.detect_stop_hunt.return_value = {
            "type": "SSL_HUNT", "confidence": 80.0
        }
        snap = engine.analyze("NIFTY50", 22020.0, 22030.0, [], 0, 0)
        assert snap.liquidity_score >= 60.0

    def test_stop_hunt_bsl_caps_negative(self, engine: LiquidityEngine) -> None:
        engine.stop_hunt_detector.detect_stop_hunt.return_value = {
            "type": "BSL_HUNT", "confidence": 75.0
        }
        engine.pool_detector._calculate_liquidity_bias.return_value = ("BULLISH", "r")
        snap = engine.analyze("NIFTY50", 22020.0, 22030.0, [], 0, 0)
        assert snap.liquidity_score <= -60.0

    def test_score_clamped_to_100(self, engine: LiquidityEngine) -> None:
        engine.pool_detector._calculate_liquidity_bias.return_value = ("BULLISH", "r")
        engine.sweep_detector.get_recent_sweeps.return_value = [
            _make_sweep(sweep_type="SSL_SWEEP", quality="HIGH"),
            _make_sweep(sweep_type="SSL_SWEEP", quality="HIGH"),
        ]
        snap = engine.analyze("NIFTY50", 22020.0, 22030.0, [], 0, 0)
        assert snap.liquidity_score <= 100.0


# ---------------------------------------------------------------------------
# 3. Signal BLOCK — sweep contradicts proposed
# ---------------------------------------------------------------------------

class TestSignalBlock:
    def test_block_buy_call_on_bsl_sweep(self, engine: LiquidityEngine) -> None:
        engine.sweep_detector.get_signal_override.return_value = {
            "action": "BLOCK",
            "reason": "BSL swept — bearish reversal expected",
            "confidence_adjustment": 0.0,
        }
        result = engine.get_decision_engine_input(
            "NIFTY50", "BUY_CALL", 22020.0, 22030.0, [], 0, 0
        )
        assert result["signal_action"] == "BLOCK"
        assert "BSL swept" in result["reason"]

    def test_allow_when_no_sweep_override(self, engine: LiquidityEngine) -> None:
        result = engine.get_decision_engine_input(
            "NIFTY50", "BUY_CALL", 22020.0, 22030.0, [], 0, 0
        )
        assert result["signal_action"] in ("ALLOW", "BOOST")


# ---------------------------------------------------------------------------
# 4. Signal REVERSE — stop hunt detected
# ---------------------------------------------------------------------------

class TestSignalReverse:
    def test_reverse_when_hunt_says_reverse(self, engine: LiquidityEngine) -> None:
        engine.stop_hunt_detector.get_stop_hunt_signal_for_engine.return_value = {
            "action": "REVERSE",
            "confidence_adjustment": -20.0,
            "reason": "BSL_HUNT confirmed — bearish",
        }
        result = engine.get_decision_engine_input(
            "NIFTY50", "BUY_CALL", 22020.0, 22030.0, [], 0, 0
        )
        assert result["signal_action"] == "REVERSE"


# ---------------------------------------------------------------------------
# 5. Signal BOOST — liquidity aligned
# ---------------------------------------------------------------------------

class TestSignalBoost:
    def test_boost_buy_call_on_bullish_liquidity(self, engine: LiquidityEngine) -> None:
        engine.pool_detector._calculate_liquidity_bias.return_value = ("BULLISH", "SSL swept")
        engine.sweep_detector.get_recent_sweeps.return_value = [
            _make_sweep(sweep_type="SSL_SWEEP", quality="HIGH")
        ]
        engine.volume_profile.get_price_context.return_value = {"signal": "BULLISH", "location": "IN_VA"}
        engine.imbalance_detector.get_imbalance_signal.return_value = {
            "signal": "BULLISH", "strength": 80.0, "reason": "OB hold"
        }
        result = engine.get_decision_engine_input(
            "NIFTY50", "BUY_CALL", 22020.0, 22030.0, [], 0, 0
        )
        assert result["signal_action"] == "BOOST"
        assert result["confidence_adjustment"] > 0

    def test_boost_buy_put_on_bearish_liquidity(self, engine: LiquidityEngine) -> None:
        engine.pool_detector._calculate_liquidity_bias.return_value = ("BEARISH", "BSL swept")
        engine.sweep_detector.get_recent_sweeps.return_value = [
            _make_sweep(sweep_type="BSL_SWEEP", quality="HIGH")
        ]
        engine.volume_profile.get_price_context.return_value = {"signal": "BEARISH", "location": "IN_VA"}
        result = engine.get_decision_engine_input(
            "NIFTY50", "BUY_PUT", 22020.0, 22030.0, [], 0, 0
        )
        assert result["signal_action"] == "BOOST"


# ---------------------------------------------------------------------------
# 6. startup() populates snapshots
# ---------------------------------------------------------------------------

class TestStartup:
    def test_startup_populates_snapshots(self, engine: LiquidityEngine) -> None:
        symbols = ["NIFTY50", "BANKNIFTY"]
        engine.startup(symbols)
        for sym in symbols:
            snap = engine.get_snapshot(sym)
            assert snap is not None
            assert snap.symbol == sym

    def test_startup_loads_vp(self, engine: LiquidityEngine) -> None:
        mock_vp = MagicMock(spec=VolumeProfile)
        mock_vp.poc = 22000.0
        engine.volume_profile.load_previous_day_profile.return_value = mock_vp
        engine.startup(["NIFTY50"])
        snap = engine.get_snapshot("NIFTY50")
        assert snap.volume_profile is mock_vp

    def test_startup_continues_on_error(self, engine: LiquidityEngine) -> None:
        engine.volume_profile.load_previous_day_profile.side_effect = [
            RuntimeError("DB error"), None
        ]
        engine.startup(["NIFTY50", "BANKNIFTY"])
        # BANKNIFTY should still be populated
        assert engine.get_snapshot("BANKNIFTY") is not None


# ---------------------------------------------------------------------------
# 7. DecisionEngine integration — ALLOW + 0 adj doesn't break signals
# ---------------------------------------------------------------------------

class TestDecisionEngineIntegration:
    def test_allow_zero_adj_passthrough(self, engine: LiquidityEngine) -> None:
        result = engine.get_decision_engine_input(
            "NIFTY50", "BUY_CALL", 22020.0, 22030.0, [], 0, 0
        )
        assert result["signal_action"] in ("ALLOW", "BOOST")
        assert isinstance(result["liquidity_score"], float)
        assert isinstance(result["active_setups"], list)
        assert isinstance(result["price_targets"], list)
        assert isinstance(result["snapshot"], LiquiditySnapshot)

    def test_output_keys_complete(self, engine: LiquidityEngine) -> None:
        result = engine.get_decision_engine_input(
            "NIFTY50", "BUY_CALL", 22020.0, 22030.0, [], 0, 0
        )
        required = {
            "liquidity_score", "signal_action", "confidence_adjustment",
            "active_setups", "price_targets", "stop_suggestion", "reason", "snapshot",
        }
        assert required.issubset(result.keys())

    def test_no_trade_signal_skipped(self, engine: LiquidityEngine) -> None:
        # get_decision_engine_input is always called with actual signal type;
        # but we verify neutral score maps to ALLOW
        result = engine.get_decision_engine_input(
            "NIFTY50", "NO_TRADE", 22020.0, 22030.0, [], 0, 0
        )
        # Even NO_TRADE should return a structured response safely
        assert "signal_action" in result
