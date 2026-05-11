"""Unit tests for CandleStore and Candle model."""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from config.constants import IST_TIMEZONE
from src.analysis.liquidity.candle_store import (
    CandleStore,
    _floor_to_period,
    _merge_candles,
    _parse_timestamp,
)
from src.analysis.liquidity.models import Candle

_IST = ZoneInfo(IST_TIMEZONE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candle(
    symbol: str = "NIFTY",
    timeframe: str = "1min",
    timestamp: datetime | None = None,
    open: float = 100.0,
    high: float = 105.0,
    low: float = 98.0,
    close: float = 103.0,
    volume: int = 1000,
) -> Candle:
    ts = timestamp or datetime(2026, 1, 2, 9, 15, tzinfo=_IST)
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_store() -> tuple[CandleStore, MagicMock, MagicMock]:
    """Return a CandleStore backed by in-memory SQLite mocks."""
    db = MagicMock()
    db.execute.return_value = MagicMock()
    db.execute_many.return_value = None
    db.fetch_all.return_value = []
    smart_api = MagicMock()
    store = CandleStore(db=db, smart_api=smart_api)
    return store, db, smart_api


# ---------------------------------------------------------------------------
# Candle model
# ---------------------------------------------------------------------------

class TestCandleModel:
    def test_derived_fields_bullish(self):
        c = _make_candle(open=100.0, high=110.0, low=98.0, close=108.0)
        assert c.is_bullish is True
        assert c.body_size == pytest.approx(8.0)
        assert c.upper_wick == pytest.approx(2.0)   # 110 - 108
        assert c.lower_wick == pytest.approx(2.0)   # 100 - 98
        assert c.total_range == pytest.approx(12.0)

    def test_derived_fields_bearish(self):
        c = _make_candle(open=108.0, high=110.0, low=98.0, close=100.0)
        assert c.is_bullish is False
        assert c.body_size == pytest.approx(8.0)
        assert c.upper_wick == pytest.approx(2.0)   # 110 - 108
        assert c.lower_wick == pytest.approx(2.0)   # 100 - 98

    def test_doji_candle(self):
        c = _make_candle(open=100.0, high=102.0, low=98.0, close=100.0)
        assert c.is_bullish is False
        assert c.body_size == pytest.approx(0.0)
        assert c.upper_wick == pytest.approx(2.0)
        assert c.lower_wick == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Tick aggregation into 1-min candle
# ---------------------------------------------------------------------------

class TestTickAggregation:
    def test_first_tick_opens_candle(self):
        store, _, _ = _make_store()
        ts = datetime(2026, 1, 2, 9, 15, 30, tzinfo=_IST)
        store.add_tick("NIFTY", 100.0, 500, ts)
        oc = store.get_open_candle("NIFTY", "1min")
        assert oc is not None
        assert oc.open == pytest.approx(100.0)
        assert oc.close == pytest.approx(100.0)
        assert oc.volume == 500

    def test_second_tick_same_period_updates_high_low(self):
        store, _, _ = _make_store()
        t0 = datetime(2026, 1, 2, 9, 15, 10, tzinfo=_IST)
        t1 = datetime(2026, 1, 2, 9, 15, 45, tzinfo=_IST)
        store.add_tick("NIFTY", 100.0, 500, t0)
        store.add_tick("NIFTY", 105.0, 300, t1)
        oc = store.get_open_candle("NIFTY", "1min")
        assert oc.high == pytest.approx(105.0)
        assert oc.low == pytest.approx(100.0)
        assert oc.close == pytest.approx(105.0)
        assert oc.volume == 800

    def test_tick_crossing_boundary_closes_candle(self):
        store, _, _ = _make_store()
        t0 = datetime(2026, 1, 2, 9, 15, 30, tzinfo=_IST)
        t1 = datetime(2026, 1, 2, 9, 16, 5, tzinfo=_IST)
        store.add_tick("NIFTY", 100.0, 500, t0)
        store.add_tick("NIFTY", 102.0, 200, t1)

        closed = store.get_latest_candle("NIFTY", "1min")
        assert closed is not None
        assert closed.timestamp == datetime(2026, 1, 2, 9, 15, tzinfo=_IST)
        assert closed.close == pytest.approx(100.0)

        # New open candle should exist
        oc = store.get_open_candle("NIFTY", "1min")
        assert oc is not None
        assert oc.open == pytest.approx(102.0)

    def test_callback_fires_on_close(self):
        store, _, _ = _make_store()
        fired: list[Candle] = []
        store.register_candle_callback("1min", lambda c: fired.append(c))

        t0 = datetime(2026, 1, 2, 9, 15, 0, tzinfo=_IST)
        t1 = datetime(2026, 1, 2, 9, 16, 0, tzinfo=_IST)
        store.add_tick("NIFTY", 100.0, 100, t0)
        assert len(fired) == 0
        store.add_tick("NIFTY", 101.0, 100, t1)
        assert len(fired) == 1
        assert fired[0].symbol == "NIFTY"
        assert fired[0].timeframe == "1min"


# ---------------------------------------------------------------------------
# 1-min to 5-min aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def _five_1min_candles(self) -> list[Candle]:
        base = datetime(2026, 1, 2, 9, 15, tzinfo=_IST)
        return [
            _make_candle(
                timeframe="1min",
                timestamp=base + timedelta(minutes=i),
                open=100.0 + i,
                high=105.0 + i,
                low=98.0 + i,
                close=103.0 + i,
                volume=1000,
            )
            for i in range(5)
        ]

    def test_aggregate_5_min_from_1_min(self):
        candles = self._five_1min_candles()
        result = CandleStore.aggregate_to_timeframe(candles, "5min")
        assert len(result) == 1
        agg = result[0]
        assert agg.timeframe == "5min"
        assert agg.open == pytest.approx(100.0)    # first open
        assert agg.close == pytest.approx(107.0)   # last close (103 + 4)
        assert agg.high == pytest.approx(109.0)    # max high (105 + 4)
        assert agg.low == pytest.approx(98.0)      # min low
        assert agg.volume == 5000

    def test_aggregate_two_periods(self):
        base = datetime(2026, 1, 2, 9, 15, tzinfo=_IST)
        candles = [
            _make_candle(timeframe="1min", timestamp=base + timedelta(minutes=i))
            for i in range(10)
        ]
        result = CandleStore.aggregate_to_timeframe(candles, "5min")
        assert len(result) == 2

    def test_aggregate_unknown_timeframe_returns_empty(self):
        candles = self._five_1min_candles()
        result = CandleStore.aggregate_to_timeframe(candles, "99min")
        assert result == []

    def test_aggregate_empty_list_returns_empty(self):
        result = CandleStore.aggregate_to_timeframe([], "5min")
        assert result == []


# ---------------------------------------------------------------------------
# Deque maxlen behaviour
# ---------------------------------------------------------------------------

class TestDequeMaxlen:
    def test_deque_does_not_exceed_maxlen(self):
        store, _, _ = _make_store()
        store.init_symbol("NIFTY")

        from src.analysis.liquidity.candle_store import _DEQUE_MAXLEN

        # Pre-fill the deque beyond maxlen
        base = datetime(2026, 1, 2, 9, 0, tzinfo=_IST)
        for i in range(_DEQUE_MAXLEN + 10):
            c = _make_candle(
                timestamp=base + timedelta(minutes=i),
                volume=i + 1,
            )
            store._candles["NIFTY"]["1min"].append(c)

        assert len(store._candles["NIFTY"]["1min"]) == _DEQUE_MAXLEN

    def test_oldest_candle_evicted_when_full(self):
        store, _, _ = _make_store()
        store.init_symbol("NIFTY")

        from src.analysis.liquidity.candle_store import _DEQUE_MAXLEN

        base = datetime(2026, 1, 2, 9, 0, tzinfo=_IST)
        first_ts = base
        for i in range(_DEQUE_MAXLEN + 5):
            c = _make_candle(timestamp=base + timedelta(minutes=i))
            store._candles["NIFTY"]["1min"].append(c)

        # The original first candle must be gone
        oldest = store.get_candles("NIFTY", "1min")[0]
        assert oldest.timestamp != first_ts


# ---------------------------------------------------------------------------
# Late tick handling
# ---------------------------------------------------------------------------

class TestLateTick:
    def test_late_tick_merged_not_reopened(self):
        store, _, _ = _make_store()

        # Send a normal tick that builds a candle
        t0 = datetime(2026, 1, 2, 9, 15, 30, tzinfo=_IST)
        t1 = datetime(2026, 1, 2, 9, 16, 5, tzinfo=_IST)
        store.add_tick("NIFTY", 100.0, 500, t0)
        store.add_tick("NIFTY", 102.0, 200, t1)  # closes 9:15 candle

        closed_before = store.get_latest_candle("NIFTY", "1min")
        assert closed_before is not None

        # Late tick for the already-closed 9:15 period
        late = datetime(2026, 1, 2, 9, 15, 59, tzinfo=_IST)
        store.add_tick("NIFTY", 110.0, 100, late)

        # Still only one closed candle at 9:15
        closed_after = store.get_latest_candle("NIFTY", "1min")
        assert closed_after.timestamp == closed_before.timestamp
        # late tick high should be merged
        assert closed_after.high >= 110.0

    def test_late_tick_does_not_fire_callback(self):
        store, _, _ = _make_store()
        fired: list[Candle] = []
        store.register_candle_callback("1min", lambda c: fired.append(c))

        t0 = datetime(2026, 1, 2, 9, 15, 0, tzinfo=_IST)
        t1 = datetime(2026, 1, 2, 9, 16, 0, tzinfo=_IST)
        store.add_tick("NIFTY", 100.0, 100, t0)
        store.add_tick("NIFTY", 101.0, 100, t1)
        assert len(fired) == 1

        # Now send a late tick for 9:15
        late = datetime(2026, 1, 2, 9, 15, 50, tzinfo=_IST)
        store.add_tick("NIFTY", 99.0, 50, late)
        # No additional callback fired
        assert len(fired) == 1


# ---------------------------------------------------------------------------
# get_candles / get_latest_candle guards
# ---------------------------------------------------------------------------

class TestGetCandles:
    def test_unknown_symbol_returns_empty(self):
        store, _, _ = _make_store()
        assert store.get_candles("UNKNOWN", "1min") == []
        assert store.get_latest_candle("UNKNOWN", "1min") is None
        assert store.get_open_candle("UNKNOWN", "1min") is None

    def test_count_slices_correctly(self):
        store, _, _ = _make_store()
        store.init_symbol("NIFTY")
        base = datetime(2026, 1, 2, 9, 0, tzinfo=_IST)
        for i in range(50):
            store._candles["NIFTY"]["1min"].append(
                _make_candle(timestamp=base + timedelta(minutes=i))
            )
        result = store.get_candles("NIFTY", "1min", count=10)
        assert len(result) == 10
        # Most recent candle is last
        assert result[-1].timestamp == base + timedelta(minutes=49)

    def test_get_latest_candle_is_most_recent(self):
        store, _, _ = _make_store()
        store.init_symbol("NIFTY")
        base = datetime(2026, 1, 2, 9, 0, tzinfo=_IST)
        for i in range(5):
            store._candles["NIFTY"]["1min"].append(
                _make_candle(timestamp=base + timedelta(minutes=i))
            )
        latest = store.get_latest_candle("NIFTY", "1min")
        assert latest.timestamp == base + timedelta(minutes=4)


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_multiple_callbacks_all_fire(self):
        store, _, _ = _make_store()
        results: list[str] = []
        store.register_candle_callback("5min", lambda c: results.append("A"))
        store.register_candle_callback("5min", lambda c: results.append("B"))

        t0 = datetime(2026, 1, 2, 9, 15, 0, tzinfo=_IST)
        t1 = datetime(2026, 1, 2, 9, 20, 0, tzinfo=_IST)
        store.add_tick("NIFTY", 100.0, 100, t0)
        store.add_tick("NIFTY", 101.0, 100, t1)

        assert "A" in results
        assert "B" in results

    def test_unknown_timeframe_callback_ignored(self):
        store, _, _ = _make_store()
        store.register_candle_callback("99min", lambda c: None)
        # Should not raise; just log warning

    def test_callback_exception_does_not_crash_store(self):
        store, _, _ = _make_store()

        def bad_cb(c: Candle) -> None:
            raise RuntimeError("callback exploded")

        store.register_candle_callback("1min", bad_cb)
        t0 = datetime(2026, 1, 2, 9, 15, 0, tzinfo=_IST)
        t1 = datetime(2026, 1, 2, 9, 16, 0, tzinfo=_IST)
        # Should not raise
        store.add_tick("NIFTY", 100.0, 100, t0)
        store.add_tick("NIFTY", 101.0, 100, t1)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_floor_to_period_1min(self):
        ts = datetime(2026, 1, 2, 9, 17, 43, tzinfo=_IST)
        floored = _floor_to_period(ts, 1)
        assert floored == datetime(2026, 1, 2, 9, 17, 0, tzinfo=_IST)

    def test_floor_to_period_5min(self):
        ts = datetime(2026, 1, 2, 9, 18, 30, tzinfo=_IST)
        floored = _floor_to_period(ts, 5)
        assert floored == datetime(2026, 1, 2, 9, 15, 0, tzinfo=_IST)

    def test_floor_to_period_15min(self):
        ts = datetime(2026, 1, 2, 9, 29, 59, tzinfo=_IST)
        floored = _floor_to_period(ts, 15)
        assert floored == datetime(2026, 1, 2, 9, 15, 0, tzinfo=_IST)

    def test_floor_to_period_1hour(self):
        ts = datetime(2026, 1, 2, 10, 45, 0, tzinfo=_IST)
        floored = _floor_to_period(ts, 60)
        assert floored == datetime(2026, 1, 2, 10, 0, 0, tzinfo=_IST)

    def test_parse_timestamp_datetime_passthrough(self):
        dt = datetime(2026, 1, 2, 9, 15, tzinfo=_IST)
        result = _parse_timestamp(dt)
        assert result == dt

    def test_parse_timestamp_string_smartapi_format(self):
        result = _parse_timestamp("2026-01-02 09:15")
        assert result.year == 2026
        assert result.minute == 15
        assert result.tzinfo is not None

    def test_parse_timestamp_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_timestamp("not-a-date")
