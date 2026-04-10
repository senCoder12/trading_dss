"""
Tests for the historical data replay engine.

Covers:
- prepare_replay: correct bar counts, warmup, data quality
- get_current_slice: anti-look-ahead-bias guarantee
- advance / reset: state transitions
- ReplayIterator: correct iteration count
- Multi-index replay: timestamp synchronization
- Edge cases: insufficient data, missing options, etc.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from src.backtest.data_replay import (
    DataReplayEngine,
    ReplayIterator,
    ReplaySession,
    TimeSlice,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """Ephemeral in-memory-style DB (actually a temp file for FK support)."""
    db_path = tmp_path / "test_replay.db"
    mgr = DatabaseManager(db_path=db_path)
    mgr.connect()
    mgr.initialise_schema()
    return mgr


def _seed_index(db: DatabaseManager, index_id: str = "NIFTY50") -> None:
    """Insert a minimal index_master row."""
    now = datetime.now().isoformat()
    db.execute(
        Q.INSERT_INDEX_MASTER,
        (index_id, f"Test {index_id}", index_id, f"^{index_id}",
         "NSE", 75, 1, index_id, "broad_market", 1, now, now),
    )


def _seed_price_data(
    db: DatabaseManager,
    index_id: str = "NIFTY50",
    start: date = date(2023, 1, 2),
    days: int = 500,
    timeframe: str = "1d",
    base_price: float = 18000.0,
) -> pd.DatetimeIndex:
    """Insert synthetic daily OHLCV rows and return the dates generated."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start=start, periods=days)

    close = base_price + rng.standard_normal(days).cumsum() * 50
    high = close + rng.uniform(20, 100, days)
    low = close - rng.uniform(20, 100, days)
    open_ = low + rng.uniform(0, 1, days) * (high - low)
    volume = rng.integers(500_000, 5_000_000, days)

    params = []
    for i, dt in enumerate(dates):
        ts = dt.isoformat()
        params.append((
            index_id, ts,
            float(open_[i]), float(high[i]), float(low[i]), float(close[i]),
            int(volume[i]), None, "test", timeframe,
        ))
    db.execute_many(Q.INSERT_PRICE_DATA, params)
    return dates


def _seed_vix(db: DatabaseManager, start: date, days: int = 100) -> None:
    """Insert synthetic VIX readings."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range(start=start, periods=days)
    for dt in dates:
        val = 12.0 + rng.standard_normal() * 3
        db.execute(Q.INSERT_VIX_DATA, (dt.isoformat(), round(val, 2), 0.0, 0.0))


def _seed_fii(db: DatabaseManager, start: date, days: int = 100) -> None:
    """Insert synthetic FII/DII daily flows."""
    rng = np.random.default_rng(77)
    dates = pd.bdate_range(start=start, periods=days)
    for dt in dates:
        d = dt.date().isoformat()
        fii_buy = round(rng.uniform(5000, 15000), 2)
        fii_sell = round(rng.uniform(5000, 15000), 2)
        db.execute(Q.INSERT_FII_DII_ACTIVITY,
                   (d, "FII", fii_buy, fii_sell, round(fii_buy - fii_sell, 2), "CASH"))
        dii_buy = round(rng.uniform(4000, 12000), 2)
        dii_sell = round(rng.uniform(4000, 12000), 2)
        db.execute(Q.INSERT_FII_DII_ACTIVITY,
                   (d, "DII", dii_buy, dii_sell, round(dii_buy - dii_sell, 2), "CASH"))


def _seed_options(db: DatabaseManager, index_id: str, start: date, days: int = 50) -> None:
    """Insert synthetic options chain snapshots."""
    rng = np.random.default_rng(55)
    dates = pd.bdate_range(start=start, periods=days)
    expiry = "2024-06-27"
    for dt in dates:
        ts = dt.isoformat()
        for strike in range(17500, 18500, 100):
            for opt_type in ("CE", "PE"):
                db.execute(Q.INSERT_OPTIONS_CHAIN, (
                    index_id, ts, expiry, float(strike), opt_type,
                    int(rng.integers(1000, 200000)),  # oi
                    int(rng.integers(-5000, 20000)),   # oi_change
                    int(rng.integers(100, 50000)),      # volume
                    round(rng.uniform(5, 500), 2),      # ltp
                    round(rng.uniform(10, 30), 2),      # iv
                    None, None,
                ))


# ---------------------------------------------------------------------------
# Tests: prepare_replay
# ---------------------------------------------------------------------------

class TestPrepareReplay:

    def test_basic_replay_setup(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=500)
        engine = DataReplayEngine(db)

        # Request replay starting ~250 days in, leaving room for warmup
        start = dates[260].date()
        end = dates[490].date()
        session = engine.prepare_replay("NIFTY50", start, end, warmup_bars=250)

        assert session.index_id == "NIFTY50"
        assert session.timeframe == "1d"
        assert session.warmup_bars <= 260  # at most 260 bars before start
        assert session.total_bars > 0
        assert not session.is_complete
        assert session.actual_start >= start or session.actual_start == start
        assert session.actual_end <= end
        assert session.data_quality_score > 0

    def test_warmup_bars_count(self, db: DatabaseManager) -> None:
        """Warmup must provide the requested number of bars (when available)."""
        _seed_index(db)
        dates = _seed_price_data(db, days=500)
        engine = DataReplayEngine(db)

        start = dates[260].date()
        end = dates[490].date()
        session = engine.prepare_replay("NIFTY50", start, end, warmup_bars=250)

        # The warmup_bars stored should be >= 250 (since we have 260 bars before start)
        assert session.warmup_bars >= 250

        # The first tradeable bar index should equal warmup_bars
        tradeable_start = session.full_price_data.index[session.warmup_bars]
        assert tradeable_start.date() >= start

    def test_total_bars_matches_tradeable_range(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=500)
        engine = DataReplayEngine(db)

        start = dates[260].date()
        end = dates[490].date()
        session = engine.prepare_replay("NIFTY50", start, end, warmup_bars=250)

        # total_bars = len(full_price_data) - warmup_bars
        assert session.total_bars == len(session.full_price_data) - session.warmup_bars

    def test_insufficient_data_raises(self, db: DatabaseManager) -> None:
        _seed_index(db)
        _seed_price_data(db, days=30)  # Only 30 bars total
        engine = DataReplayEngine(db)

        with pytest.raises(ValueError, match="Minimum 50 required"):
            engine.prepare_replay(
                "NIFTY50", date(2023, 1, 2), date(2023, 2, 15), warmup_bars=10,
            )

    def test_index_not_found_raises(self, db: DatabaseManager) -> None:
        _seed_index(db)
        engine = DataReplayEngine(db)

        with pytest.raises(ValueError, match="no price data"):
            engine.prepare_replay("NONEXISTENT", date(2024, 1, 1), date(2024, 6, 30))

    def test_timeframe_not_available_raises(self, db: DatabaseManager) -> None:
        _seed_index(db)
        _seed_price_data(db, days=300, timeframe="1d")
        engine = DataReplayEngine(db)

        with pytest.raises(ValueError, match="not available"):
            engine.prepare_replay(
                "NIFTY50", date(2023, 6, 1), date(2023, 12, 31), timeframe="5m",
            )

    def test_insufficient_warmup_warns(self, db: DatabaseManager) -> None:
        """When not enough history exists for warmup, issue a warning."""
        _seed_index(db)
        dates = _seed_price_data(db, days=100, start=date(2024, 1, 2))
        engine = DataReplayEngine(db)

        # Start date is very close to the beginning — can't fit 250 warmup bars
        start = dates[10].date()
        end = dates[99].date()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            session = engine.prepare_replay("NIFTY50", start, end, warmup_bars=250)
            # Should have triggered a warning about insufficient warmup
            warmup_warnings = [x for x in w if "warmup" in str(x.message).lower()]
            assert len(warmup_warnings) > 0

        # Should still create the session with whatever warmup is available
        assert session.warmup_bars < 250

    def test_benchmark_loading(self, db: DatabaseManager) -> None:
        _seed_index(db, "NIFTY50")
        _seed_index(db, "BANKNIFTY")
        _seed_price_data(db, "NIFTY50", days=400)
        _seed_price_data(db, "BANKNIFTY", days=400, base_price=42000)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "BANKNIFTY", date(2023, 6, 1), date(2024, 6, 30),
            benchmark_id="NIFTY50", warmup_bars=50,
        )
        assert session.benchmark_data is not None
        assert len(session.benchmark_data) > 0

    def test_auxiliary_data_flags(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=400)
        start = dates[260].date()
        end = dates[390].date()

        # Seed VIX and FII
        _seed_vix(db, start, days=50)
        _seed_fii(db, start, days=50)

        engine = DataReplayEngine(db)
        session = engine.prepare_replay("NIFTY50", start, end, warmup_bars=50)

        assert session.has_vix_data is True
        assert session.has_fii_data is True
        # No options seeded
        assert session.has_options_data is False

    def test_data_quality_score(self, db: DatabaseManager) -> None:
        _seed_index(db)
        _seed_price_data(db, days=500)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", date(2023, 6, 1), date(2024, 6, 30), warmup_bars=50,
        )
        # With continuous business-day data, quality should be high
        assert session.data_quality_score >= 0.9


# ---------------------------------------------------------------------------
# Tests: get_current_slice — LOOK-AHEAD BIAS PREVENTION
# ---------------------------------------------------------------------------

class TestGetCurrentSlice:

    def test_first_slice_has_warmup_plus_one_rows(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=400)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[260].date(), dates[390].date(), warmup_bars=250,
        )
        ts = engine.get_current_slice(session)

        # price_history should have warmup_bars + 1 rows (warmup + current bar)
        expected = session.warmup_bars + 1
        assert len(ts.price_history) == expected

    def test_no_look_ahead_bias(self, db: DatabaseManager) -> None:
        """At bar N, price_history must NOT contain any bar after N."""
        _seed_index(db)
        dates = _seed_price_data(db, days=400)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[100].date(), dates[390].date(), warmup_bars=50,
        )

        # Check several bars
        for _ in range(20):
            ts = engine.get_current_slice(session)

            # The last row in price_history must be the current bar
            last_ts_in_history = ts.price_history.index[-1]
            assert last_ts_in_history == ts.timestamp

            # No future data
            future_mask = ts.price_history.index > ts.timestamp
            assert not future_mask.any(), (
                f"Look-ahead bias detected! price_history contains "
                f"{future_mask.sum()} bars after {ts.timestamp}"
            )

            # price_history length should grow by 1 each step
            engine.advance(session)

    def test_price_history_grows_by_one_each_bar(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=400)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[100].date(), dates[390].date(), warmup_bars=50,
        )

        ts1 = engine.get_current_slice(session)
        len1 = len(ts1.price_history)

        engine.advance(session)
        ts2 = engine.get_current_slice(session)
        len2 = len(ts2.price_history)

        assert len2 == len1 + 1

    def test_current_bar_matches_last_history_row(self, db: DatabaseManager) -> None:
        _seed_index(db)
        _seed_price_data(db, days=300)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", date(2023, 6, 1), date(2024, 1, 31), warmup_bars=50,
        )
        ts = engine.get_current_slice(session)

        last_row = ts.price_history.iloc[-1]
        assert ts.current_bar["open"] == float(last_row["open"])
        assert ts.current_bar["close"] == float(last_row["close"])
        assert ts.current_bar["timestamp"] == ts.timestamp

    def test_previous_bar_is_correct(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=300)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[100].date(), dates[290].date(), warmup_bars=50,
        )
        # Advance once so there IS a previous bar in the tradeable range
        engine.advance(session)
        ts = engine.get_current_slice(session)

        assert ts.previous_bar is not None
        # Previous bar's close should match the second-to-last in history
        second_last = ts.price_history.iloc[-2]
        assert ts.previous_bar["close"] == float(second_last["close"])

    def test_benchmark_history_no_lookahead(self, db: DatabaseManager) -> None:
        _seed_index(db, "NIFTY50")
        _seed_index(db, "BANKNIFTY")
        _seed_price_data(db, "NIFTY50", days=400)
        _seed_price_data(db, "BANKNIFTY", days=400, base_price=42000)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "BANKNIFTY", date(2023, 6, 1), date(2024, 6, 30),
            benchmark_id="NIFTY50", warmup_bars=50,
        )
        ts = engine.get_current_slice(session)

        assert ts.benchmark_history is not None
        # Benchmark must not have data beyond current timestamp
        assert ts.benchmark_history.index[-1] <= ts.timestamp

    def test_progress_pct(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=200)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[60].date(), dates[199].date(), warmup_bars=50,
        )

        ts = engine.get_current_slice(session)
        assert ts.progress_pct == pytest.approx(0.0, abs=1.0)

        # Advance to the end
        while not session.is_complete:
            engine.advance(session)

        # Check last slice before completion
        engine.reset(session)
        last_ts = None
        for slice_ in ReplayIterator(engine, session):
            last_ts = slice_
        assert last_ts is not None
        assert last_ts.progress_pct == pytest.approx(100.0, abs=1.0)
        assert last_ts.bars_remaining == 0

    def test_vix_at_current_time(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=400)
        start = dates[260].date()
        end = dates[390].date()
        _seed_vix(db, start, days=50)

        engine = DataReplayEngine(db)
        session = engine.prepare_replay("NIFTY50", start, end, warmup_bars=50)
        ts = engine.get_current_slice(session)

        assert ts.vix_value is not None
        assert isinstance(ts.vix_value, float)

    def test_fii_data_is_previous_day(self, db: DatabaseManager) -> None:
        """FII data should be from the PREVIOUS trading day, not current."""
        _seed_index(db)
        dates = _seed_price_data(db, days=400)
        start = dates[260].date()
        end = dates[390].date()
        _seed_fii(db, start, days=50)

        engine = DataReplayEngine(db)
        session = engine.prepare_replay("NIFTY50", start, end, warmup_bars=50)

        # Advance a few bars so FII data is available
        for _ in range(5):
            engine.advance(session)

        ts = engine.get_current_slice(session)
        if ts.fii_data is not None:
            fii_date = date.fromisoformat(ts.fii_data["date"])
            current_date = ts.timestamp.date()
            assert fii_date < current_date, (
                f"FII data date {fii_date} must be before current date {current_date}"
            )

    def test_options_snapshot_no_future(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=400)
        start = dates[260].date()
        end = dates[390].date()
        _seed_options(db, "NIFTY50", start, days=30)

        engine = DataReplayEngine(db)
        session = engine.prepare_replay("NIFTY50", start, end, warmup_bars=50)

        ts = engine.get_current_slice(session)
        if ts.options_snapshot is not None:
            snap_ts = ts.options_snapshot["snapshot_timestamp"]
            assert snap_ts <= ts.timestamp.isoformat()


# ---------------------------------------------------------------------------
# Tests: advance / reset
# ---------------------------------------------------------------------------

class TestAdvanceReset:

    def test_advance_moves_forward(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=200)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[60].date(), dates[199].date(), warmup_bars=50,
        )

        ts1 = engine.get_current_slice(session)
        ts2 = engine.advance(session)

        assert ts2 is not None
        assert ts2.bar_index == ts1.bar_index + 1
        assert ts2.timestamp > ts1.timestamp

    def test_advance_returns_none_at_end(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=200)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[60].date(), dates[199].date(), warmup_bars=50,
        )

        # Advance until done
        count = 0
        while engine.advance(session) is not None:
            count += 1
        assert session.is_complete is True
        assert count == session.total_bars - 1  # first bar is at warmup_bars

    def test_reset_returns_to_start(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=200)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[60].date(), dates[199].date(), warmup_bars=50,
        )
        start_idx = session.current_bar_index

        # Advance several bars
        for _ in range(10):
            engine.advance(session)
        assert session.current_bar_index == start_idx + 10

        # Reset
        engine.reset(session)
        assert session.current_bar_index == session.warmup_bars
        assert session.is_complete is False

        # First slice should be the same as original
        ts = engine.get_current_slice(session)
        assert ts.bar_index == session.warmup_bars

    def test_reset_allows_rerun(self, db: DatabaseManager) -> None:
        """After reset, can iterate the full session again."""
        _seed_index(db)
        dates = _seed_price_data(db, days=200)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[60].date(), dates[199].date(), warmup_bars=50,
        )

        # First run
        count1 = sum(1 for _ in ReplayIterator(engine, session))

        # Second run after reset (ReplayIterator calls reset internally)
        count2 = sum(1 for _ in ReplayIterator(engine, session))

        assert count1 == count2
        assert count1 == session.total_bars


# ---------------------------------------------------------------------------
# Tests: ReplayIterator
# ---------------------------------------------------------------------------

class TestReplayIterator:

    def test_iterator_produces_correct_count(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=300)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[100].date(), dates[290].date(), warmup_bars=50,
        )

        count = 0
        for ts in ReplayIterator(engine, session):
            count += 1
            assert isinstance(ts, TimeSlice)
        assert count == session.total_bars

    def test_iterator_len(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=300)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[100].date(), dates[290].date(), warmup_bars=50,
        )
        it = ReplayIterator(engine, session)
        assert len(it) == session.total_bars

    def test_iterator_timestamps_are_monotonic(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=300)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[100].date(), dates[290].date(), warmup_bars=50,
        )

        prev_ts = None
        for ts in ReplayIterator(engine, session):
            if prev_ts is not None:
                assert ts.timestamp > prev_ts
            prev_ts = ts.timestamp

    def test_iterator_look_ahead_bias_full_run(self, db: DatabaseManager) -> None:
        """Verify no look-ahead bias across the entire replay."""
        _seed_index(db)
        dates = _seed_price_data(db, days=300)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[100].date(), dates[290].date(), warmup_bars=50,
        )

        full_data = session.full_price_data
        for ts in ReplayIterator(engine, session):
            # The current bar index must not allow seeing future bars
            assert ts.price_history.index[-1] == ts.timestamp

            # Verify the actual values match the full dataset
            actual_bar = full_data.iloc[ts.bar_index]
            assert ts.current_bar["close"] == float(actual_bar["close"])

            # Verify NO future bar exists in history
            future = full_data.iloc[ts.bar_index + 1:]
            if len(future) > 0:
                future_ts = future.index[0]
                assert future_ts not in ts.price_history.index


# ---------------------------------------------------------------------------
# Tests: multi-index replay
# ---------------------------------------------------------------------------

class TestMultiIndexReplay:

    def test_multi_replay_synchronized_timestamps(self, db: DatabaseManager) -> None:
        _seed_index(db, "NIFTY50")
        _seed_index(db, "BANKNIFTY")
        _seed_price_data(db, "NIFTY50", days=400)
        _seed_price_data(db, "BANKNIFTY", days=400, base_price=42000)
        engine = DataReplayEngine(db)

        sessions = engine.prepare_multi_replay(
            ["NIFTY50", "BANKNIFTY"],
            date(2023, 6, 1), date(2024, 6, 30),
            warmup_bars=50,
        )

        assert "NIFTY50" in sessions
        assert "BANKNIFTY" in sessions

        # Both should have the same number of tradeable bars
        assert sessions["NIFTY50"].total_bars == sessions["BANKNIFTY"].total_bars

        # Iterate both and verify timestamps match
        it_nifty = ReplayIterator(engine, sessions["NIFTY50"])
        it_bank = ReplayIterator(engine, sessions["BANKNIFTY"])
        for ts_n, ts_b in zip(it_nifty, it_bank):
            assert ts_n.timestamp == ts_b.timestamp

    def test_multi_replay_insufficient_common_bars(self, db: DatabaseManager) -> None:
        _seed_index(db, "NIFTY50")
        _seed_index(db, "NIFTY_IT")
        _seed_price_data(db, "NIFTY50", start=date(2023, 1, 2), days=300)
        # NIFTY_IT has data in a totally different range with minimal overlap
        _seed_price_data(db, "NIFTY_IT", start=date(2025, 1, 2), days=100, base_price=35000)
        engine = DataReplayEngine(db)

        with pytest.raises(ValueError, match="common trading bars"):
            engine.prepare_multi_replay(
                ["NIFTY50", "NIFTY_IT"],
                date(2023, 1, 2), date(2025, 12, 31),
                warmup_bars=10,
            )

    def test_multi_replay_skips_invalid_indices(self, db: DatabaseManager) -> None:
        _seed_index(db, "NIFTY50")
        _seed_index(db, "BANKNIFTY")
        _seed_price_data(db, "NIFTY50", days=400)
        _seed_price_data(db, "BANKNIFTY", days=400, base_price=42000)
        engine = DataReplayEngine(db)

        # Include a nonexistent index — should be skipped with a warning
        sessions = engine.prepare_multi_replay(
            ["NIFTY50", "BANKNIFTY", "DOESNOTEXIST"],
            date(2023, 6, 1), date(2024, 6, 30),
            warmup_bars=50,
        )
        assert "DOESNOTEXIST" not in sessions
        assert len(sessions) == 2


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_no_data_in_range(self, db: DatabaseManager) -> None:
        _seed_index(db)
        _seed_price_data(db, days=100, start=date(2023, 1, 2))
        engine = DataReplayEngine(db)

        with pytest.raises(ValueError, match="No bars on or after|No data"):
            engine.prepare_replay("NIFTY50", date(2025, 1, 1), date(2025, 12, 31), warmup_bars=10)

    def test_missing_options_returns_none(self, db: DatabaseManager) -> None:
        """When no options data exists, slices should have None for options fields."""
        _seed_index(db)
        dates = _seed_price_data(db, days=300)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[100].date(), dates[290].date(), warmup_bars=50,
        )
        assert session.has_options_data is False

        ts = engine.get_current_slice(session)
        assert ts.options_snapshot is None
        assert ts.oi_summary is None

    def test_missing_vix_returns_none(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=300)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[100].date(), dates[290].date(), warmup_bars=50,
        )
        assert session.has_vix_data is False

        ts = engine.get_current_slice(session)
        assert ts.vix_value is None

    def test_missing_fii_returns_none(self, db: DatabaseManager) -> None:
        _seed_index(db)
        dates = _seed_price_data(db, days=300)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[100].date(), dates[290].date(), warmup_bars=50,
        )
        assert session.has_fii_data is False

        ts = engine.get_current_slice(session)
        assert ts.fii_data is None

    def test_benchmark_same_as_index_skips_loading(self, db: DatabaseManager) -> None:
        """When benchmark_id equals index_id, benchmark_data should be None."""
        _seed_index(db)
        dates = _seed_price_data(db, days=300)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[100].date(), dates[290].date(),
            benchmark_id="NIFTY50", warmup_bars=50,
        )
        assert session.benchmark_data is None

    def test_list_available_indices(self, db: DatabaseManager) -> None:
        _seed_index(db, "NIFTY50")
        _seed_index(db, "BANKNIFTY")
        _seed_price_data(db, "NIFTY50", days=100)
        _seed_price_data(db, "BANKNIFTY", days=100, base_price=42000)
        engine = DataReplayEngine(db)

        indices = engine.list_available_indices()
        assert "NIFTY50" in indices
        assert "BANKNIFTY" in indices

    def test_get_date_range(self, db: DatabaseManager) -> None:
        _seed_index(db)
        _seed_price_data(db, days=100, start=date(2024, 1, 2))
        engine = DataReplayEngine(db)

        dr = engine.get_date_range("NIFTY50", "1d")
        assert dr["bar_count"] == 100
        assert dr["first_ts"] is not None

    def test_list_timeframes(self, db: DatabaseManager) -> None:
        _seed_index(db)
        _seed_price_data(db, days=100, timeframe="1d")
        engine = DataReplayEngine(db)

        tfs = engine.list_timeframes("NIFTY50")
        assert "1d" in tfs

    def test_day_context_fields(self, db: DatabaseManager) -> None:
        """Verify is_first_bar_of_day and day_open for daily timeframe."""
        _seed_index(db)
        dates = _seed_price_data(db, days=200)
        engine = DataReplayEngine(db)

        session = engine.prepare_replay(
            "NIFTY50", dates[60].date(), dates[199].date(), warmup_bars=50,
        )
        ts = engine.get_current_slice(session)

        # For daily data, each bar IS a full day
        assert ts.is_first_bar_of_day is True
        assert ts.is_last_bar_of_day is True
        assert ts.day_open == ts.current_bar["open"]


# ---------------------------------------------------------------------------
# Tests: direct DB lookup methods
# ---------------------------------------------------------------------------

class TestDirectLookups:

    def test_get_options_at_time(self, db: DatabaseManager) -> None:
        _seed_index(db)
        _seed_options(db, "NIFTY50", date(2024, 3, 1), days=10)
        engine = DataReplayEngine(db)

        # Query at a time after the seeded options
        result = engine.get_options_at_time(
            "NIFTY50", datetime(2024, 3, 15, 15, 30),
        )
        assert result is not None
        assert "strikes" in result
        assert len(result["strikes"]) > 0

    def test_get_options_at_time_returns_none(self, db: DatabaseManager) -> None:
        _seed_index(db)
        engine = DataReplayEngine(db)

        result = engine.get_options_at_time(
            "NIFTY50", datetime(2024, 1, 1, 9, 15),
        )
        assert result is None

    def test_get_fii_at_time(self, db: DatabaseManager) -> None:
        _seed_index(db)
        _seed_fii(db, date(2024, 3, 1), days=10)
        engine = DataReplayEngine(db)

        result = engine.get_fii_at_time(datetime(2024, 3, 15, 12, 0))
        assert result is not None
        assert "fii_net" in result
        assert "dii_net" in result
        # The date should be before 2024-03-15
        assert result["date"] < "2024-03-15"

    def test_get_vix_at_time(self, db: DatabaseManager) -> None:
        _seed_index(db)
        _seed_vix(db, date(2024, 3, 1), days=10)
        engine = DataReplayEngine(db)

        result = engine.get_vix_at_time(datetime(2024, 3, 15, 12, 0))
        assert result is not None
        assert isinstance(result, float)

    def test_get_vix_at_time_returns_none(self, db: DatabaseManager) -> None:
        engine = DataReplayEngine(db)
        result = engine.get_vix_at_time(datetime(2020, 1, 1))
        assert result is None
