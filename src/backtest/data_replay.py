"""
Historical data replay engine for backtesting.

Replays market data bar-by-bar, enforcing strict look-ahead bias prevention:
at any point in time T, only data from time <= T is visible.

Usage
-----
::

    engine = DataReplayEngine(db)
    session = engine.prepare_replay("NIFTY50", date(2024, 1, 1), date(2024, 12, 31))
    for ts in ReplayIterator(engine, session):
        # ts.price_history only contains bars up to the current bar
        result = aggregator.analyze("NIFTY50", ts.price_history, ...)
"""

from __future__ import annotations

import bisect
import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from src.database.db_manager import DatabaseManager
from src.database import queries as Q

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ReplaySession:
    """All state needed for a single-index replay run."""

    index_id: str
    benchmark_id: str
    start_date: date
    end_date: date
    timeframe: str
    warmup_bars: int

    # Pre-loaded data (full period including warmup)
    full_price_data: pd.DataFrame
    benchmark_data: Optional[pd.DataFrame]

    # Replay state
    total_bars: int                     # Tradeable bars (excluding warmup)
    current_bar_index: int              # Position in full_price_data (starts at warmup_bars)
    is_complete: bool = False

    # Actual date range after data loading
    actual_start: date = field(default_factory=lambda: date.min)
    actual_end: date = field(default_factory=lambda: date.min)
    trading_days: int = 0

    # Auxiliary data availability
    has_options_data: bool = False
    has_fii_data: bool = False
    has_vix_data: bool = False

    # Data quality
    missing_days: list[date] = field(default_factory=list)
    data_quality_score: float = 1.0

    # Pre-loaded auxiliary caches (sorted by timestamp for binary search)
    _options_timestamps: list[str] = field(default_factory=list, repr=False)
    _options_cache: dict[str, list[dict]] = field(default_factory=dict, repr=False)
    _oi_timestamps: list[str] = field(default_factory=list, repr=False)
    _oi_cache: dict[str, dict] = field(default_factory=dict, repr=False)
    _vix_timestamps: list[str] = field(default_factory=list, repr=False)
    _vix_cache: dict[str, dict] = field(default_factory=dict, repr=False)
    _fii_dates: list[str] = field(default_factory=list, repr=False)
    _fii_cache: dict[str, list[dict]] = field(default_factory=dict, repr=False)


@dataclass
class TimeSlice:
    """Point-in-time view of market data — no future data visible."""

    timestamp: datetime
    bar_index: int

    # Current bar
    current_bar: dict

    # Historical data up to and including current bar
    price_history: pd.DataFrame

    # Previous bar (for gap detection, overnight analysis)
    previous_bar: Optional[dict]

    # Benchmark data up to current bar
    benchmark_history: Optional[pd.DataFrame]

    # Day context
    is_first_bar_of_day: bool
    is_last_bar_of_day: bool
    day_open: float

    # Options data at this point (if available)
    options_snapshot: Optional[dict]
    oi_summary: Optional[dict]

    # VIX at this point
    vix_value: Optional[float]

    # FII/DII (previous day's data — FII is always delayed)
    fii_data: Optional[dict]

    # Progress
    bars_remaining: int
    progress_pct: float


# ---------------------------------------------------------------------------
# Helper: build business day set for Indian markets
# ---------------------------------------------------------------------------

def _expected_trading_days(start: date, end: date) -> set[date]:
    """Return weekdays between *start* and *end* inclusive (no holiday calendar)."""
    days: set[date] = set()
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon–Fri
            days.add(d)
        d += timedelta(days=1)
    return days


def _ts_to_date(ts_str: str) -> date:
    """Parse an ISO-8601 timestamp to a date."""
    return datetime.fromisoformat(ts_str).date()


def _ts_to_datetime(ts_str: str) -> datetime:
    """Parse an ISO-8601 timestamp to datetime."""
    return datetime.fromisoformat(ts_str)


# ---------------------------------------------------------------------------
# DataReplayEngine
# ---------------------------------------------------------------------------

class DataReplayEngine:
    """
    Loads historical data and replays it bar-by-bar with strict
    look-ahead bias prevention.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Data availability helpers
    # ------------------------------------------------------------------

    def list_available_indices(self) -> list[str]:
        """Return index_ids that have price data in the DB."""
        rows = self._db.fetch_all(Q.LIST_INDICES_WITH_DATA, ())
        return [r["index_id"] for r in rows]

    def get_date_range(self, index_id: str, timeframe: str = "1d") -> dict:
        """Return first/last timestamp and bar count for an index+timeframe."""
        row = self._db.fetch_one(Q.GET_PRICE_DATE_RANGE, (index_id, timeframe))
        if row is None or row["bar_count"] == 0:
            return {"first_ts": None, "last_ts": None, "bar_count": 0}
        return dict(row)

    def list_timeframes(self, index_id: str) -> list[str]:
        """Return available timeframes for an index."""
        rows = self._db.fetch_all(Q.LIST_AVAILABLE_TIMEFRAMES, (index_id,))
        return [r["timeframe"] for r in rows]

    # ------------------------------------------------------------------
    # prepare_replay
    # ------------------------------------------------------------------

    def prepare_replay(
        self,
        index_id: str,
        start_date: date,
        end_date: date,
        timeframe: str = "1d",
        benchmark_id: str = "NIFTY50",
        warmup_bars: int = 250,
    ) -> ReplaySession:
        """
        Load all data for *index_id* and create a :class:`ReplaySession`.

        Raises
        ------
        ValueError
            If the index has no data, if the timeframe is unavailable,
            or if there are fewer than 50 tradeable bars.
        """
        # --- Validate index exists in DB ---
        available = self.list_available_indices()
        if index_id not in available:
            raise ValueError(
                f"Index '{index_id}' has no price data. "
                f"Available indices: {', '.join(sorted(available)) or '(none)'}"
            )

        available_tf = self.list_timeframes(index_id)
        if timeframe not in available_tf:
            raise ValueError(
                f"Timeframe '{timeframe}' not available for {index_id}. "
                f"Available: {', '.join(available_tf)}"
            )

        # --- Determine warmup start date ---
        # We need `warmup_bars` bars before start_date. Load extra margin
        # (weekends, holidays) so we're likely to have enough.
        margin_days = int(warmup_bars * 1.6) + 30  # generous margin
        warmup_start = start_date - timedelta(days=margin_days)

        # --- Load price data ---
        price_rows = self._db.fetch_all(
            Q.LIST_PRICE_HISTORY,
            (index_id, timeframe, warmup_start.isoformat(), end_date.isoformat() + "T23:59:59"),
        )
        if not price_rows:
            dr = self.get_date_range(index_id, timeframe)
            raise ValueError(
                f"No data for {index_id}/{timeframe} between "
                f"{start_date} and {end_date}. "
                f"Available range: {dr.get('first_ts')} → {dr.get('last_ts')}"
            )

        price_df = self._rows_to_ohlcv(price_rows)

        # --- Determine actual warmup split ---
        # Find the index of the first bar on or after start_date
        start_ts = pd.Timestamp(start_date)
        if price_df.index.tz is not None:
            start_ts = start_ts.tz_localize(price_df.index.tz)
        mask = price_df.index >= start_ts
        if not mask.any():
            raise ValueError(
                f"No bars on or after {start_date} for {index_id}/{timeframe}."
            )
        first_tradeable_pos = int(mask.argmax())

        actual_warmup = first_tradeable_pos
        if actual_warmup < warmup_bars:
            warnings.warn(
                f"Requested {warmup_bars} warmup bars but only {actual_warmup} "
                f"available before {start_date} for {index_id}. "
                f"Indicators requiring long lookback may be unreliable.",
                stacklevel=2,
            )

        total_bars = len(price_df) - first_tradeable_pos
        if total_bars < 50:
            raise ValueError(
                f"Only {total_bars} tradeable bars for {index_id}/{timeframe} "
                f"between {start_date} and {end_date}. "
                f"Minimum 50 required for a meaningful backtest."
            )

        # --- Load benchmark data ---
        benchmark_df: Optional[pd.DataFrame] = None
        if benchmark_id and benchmark_id != index_id and benchmark_id in available:
            bench_rows = self._db.fetch_all(
                Q.LIST_PRICE_HISTORY,
                (benchmark_id, timeframe, warmup_start.isoformat(), end_date.isoformat() + "T23:59:59"),
            )
            if bench_rows:
                benchmark_df = self._rows_to_ohlcv(bench_rows)

        # --- Data quality assessment ---
        tradeable_dates = set(price_df.index[first_tradeable_pos:].date)
        expected_days = _expected_trading_days(start_date, end_date)
        # Days that are expected (weekdays) but have no data (could be holidays — flag anyway)
        missing = sorted(expected_days - tradeable_dates)
        # Indian markets have ~250 trading days/year vs ~260 weekdays → ~15 holidays
        # Treat as gap only if > 5% beyond expected holiday rate
        actual_trading_days = len(tradeable_dates)
        expected_weekdays = len(expected_days)
        quality = actual_trading_days / max(expected_weekdays, 1)
        quality = min(quality, 1.0)

        if quality < 0.95:
            pct_missing = (1.0 - quality) * 100
            logger.warning(
                "%s/%s: %.1f%% of expected trading days missing (%d/%d)",
                index_id, timeframe, pct_missing,
                expected_weekdays - actual_trading_days, expected_weekdays,
            )

        # --- Check auxiliary data availability ---
        ts_start = start_date.isoformat()
        ts_end = end_date.isoformat() + "T23:59:59"

        has_options = self._count_exists(Q.AGG_OPTIONS_DATA_EXISTS, (index_id, ts_start, ts_end))
        has_fii = self._count_exists(Q.AGG_FII_DII_DATA_EXISTS, (ts_start[:10], ts_end[:10]))
        has_vix = self._count_exists(Q.AGG_VIX_DATA_EXISTS, (ts_start, ts_end))

        # --- Pre-load auxiliary data into sorted caches ---
        options_ts, options_cache = self._preload_options(index_id, warmup_start, end_date) if has_options else ([], {})
        oi_ts, oi_cache = self._preload_oi(index_id, warmup_start, end_date) if has_options else ([], {})
        vix_ts, vix_cache = self._preload_vix(warmup_start, end_date) if has_vix else ([], {})
        fii_dates, fii_cache = self._preload_fii(warmup_start, end_date) if has_fii else ([], {})

        actual_start_date = price_df.index[first_tradeable_pos].date()
        actual_end_date = price_df.index[-1].date()

        return ReplaySession(
            index_id=index_id,
            benchmark_id=benchmark_id,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            warmup_bars=actual_warmup,
            full_price_data=price_df,
            benchmark_data=benchmark_df,
            total_bars=total_bars,
            current_bar_index=first_tradeable_pos,
            is_complete=False,
            actual_start=actual_start_date,
            actual_end=actual_end_date,
            trading_days=actual_trading_days,
            has_options_data=has_options,
            has_fii_data=has_fii,
            has_vix_data=has_vix,
            missing_days=missing,
            data_quality_score=round(quality, 4),
            _options_timestamps=options_ts,
            _options_cache=options_cache,
            _oi_timestamps=oi_ts,
            _oi_cache=oi_cache,
            _vix_timestamps=vix_ts,
            _vix_cache=vix_cache,
            _fii_dates=fii_dates,
            _fii_cache=fii_cache,
        )

    # ------------------------------------------------------------------
    # get_current_slice — THE anti-look-ahead-bias method
    # ------------------------------------------------------------------

    def get_current_slice(self, session: ReplaySession) -> TimeSlice:
        """
        Return a :class:`TimeSlice` containing ONLY data visible at the
        current bar.  No future data is exposed.
        """
        idx = session.current_bar_index
        df = session.full_price_data

        # Historical data up to and including current bar — iloc is exclusive on end
        price_history = df.iloc[: idx + 1]

        current_row = df.iloc[idx]
        ts: datetime = current_row.name.to_pydatetime()  # type: ignore[union-attr]
        ts_str = ts.isoformat()

        current_bar = {
            "open": float(current_row["open"]),
            "high": float(current_row["high"]),
            "low": float(current_row["low"]),
            "close": float(current_row["close"]),
            "volume": float(current_row["volume"]),
            "timestamp": ts,
        }

        previous_bar: Optional[dict] = None
        if idx > 0:
            prev = df.iloc[idx - 1]
            previous_bar = {
                "open": float(prev["open"]),
                "high": float(prev["high"]),
                "low": float(prev["low"]),
                "close": float(prev["close"]),
                "volume": float(prev["volume"]),
                "timestamp": prev.name.to_pydatetime(),  # type: ignore[union-attr]
            }

        # Benchmark up to current bar
        benchmark_history: Optional[pd.DataFrame] = None
        if session.benchmark_data is not None:
            bench = session.benchmark_data
            bench_mask = bench.index <= ts
            if bench_mask.any():
                benchmark_history = bench.loc[bench_mask]

        # Day context
        current_date = ts.date()
        is_first = (idx == 0) or (df.iloc[idx - 1].name.date() != current_date)  # type: ignore[union-attr]
        is_last = (idx == len(df) - 1) or (df.iloc[idx + 1].name.date() != current_date)  # type: ignore[union-attr]

        # Day's opening price: first bar of this date
        day_bars = price_history[price_history.index.date == current_date]  # type: ignore[attr-defined]
        day_open = float(day_bars.iloc[0]["open"]) if len(day_bars) > 0 else current_bar["open"]

        # Options snapshot at this point in time
        options_snapshot = self._lookup_options_at(session, ts_str)
        oi_summary = self._lookup_oi_at(session, ts_str)
        vix_value = self._lookup_vix_at(session, ts_str)
        fii_data = self._lookup_fii_before(session, current_date.isoformat())

        # Progress
        last_idx = len(df) - 1
        bars_remaining = last_idx - idx
        total_tradeable = session.total_bars
        bars_done = idx - session.warmup_bars
        progress = (bars_done / max(total_tradeable - 1, 1)) * 100.0

        return TimeSlice(
            timestamp=ts,
            bar_index=idx,
            current_bar=current_bar,
            price_history=price_history,
            previous_bar=previous_bar,
            benchmark_history=benchmark_history,
            is_first_bar_of_day=is_first,
            is_last_bar_of_day=is_last,
            day_open=day_open,
            options_snapshot=options_snapshot,
            oi_summary=oi_summary,
            vix_value=vix_value,
            fii_data=fii_data,
            bars_remaining=bars_remaining,
            progress_pct=round(min(progress, 100.0), 2),
        )

    # ------------------------------------------------------------------
    # advance / reset
    # ------------------------------------------------------------------

    def advance(self, session: ReplaySession) -> Optional[TimeSlice]:
        """Move to the next bar and return the new time slice, or *None*
        when the replay is complete."""
        session.current_bar_index += 1
        if session.current_bar_index >= len(session.full_price_data):
            session.is_complete = True
            return None
        return self.get_current_slice(session)

    def reset(self, session: ReplaySession) -> None:
        """Reset session to the start of the tradeable period."""
        session.current_bar_index = session.warmup_bars
        session.is_complete = False

    # ------------------------------------------------------------------
    # Point-in-time auxiliary data lookups
    # ------------------------------------------------------------------

    def get_options_at_time(self, index_id: str, timestamp: datetime) -> Optional[dict]:
        """Query DB for nearest options snapshot at or before *timestamp*."""
        ts_str = timestamp.isoformat()
        rows = self._db.fetch_all(
            Q.GET_OPTIONS_SNAPSHOT_AT_TIME, (index_id, ts_str),
        )
        if not rows:
            return None

        oi_row = self._db.fetch_one(Q.GET_OI_AGGREGATED_AT_TIME, (index_id, ts_str))
        return {
            "strikes": rows,
            "oi_summary": dict(oi_row) if oi_row else None,
            "snapshot_timestamp": rows[0]["timestamp"],
        }

    def get_fii_at_time(self, timestamp: datetime) -> Optional[dict]:
        """Return latest FII/DII data strictly before *timestamp*'s date."""
        date_str = timestamp.date().isoformat()
        rows = self._db.fetch_all(Q.GET_FII_DII_BEFORE_DATE, (date_str,))
        if not rows:
            return None
        # Group by date — take the most recent date
        latest_date = rows[0]["date"]
        entries = [r for r in rows if r["date"] == latest_date]
        return self._aggregate_fii_rows(entries, latest_date)

    def get_vix_at_time(self, timestamp: datetime) -> Optional[float]:
        """Return the nearest VIX reading at or before *timestamp*."""
        ts_str = timestamp.isoformat()
        row = self._db.fetch_one(Q.GET_VIX_AT_TIME, (ts_str,))
        return float(row["vix_value"]) if row else None

    # ------------------------------------------------------------------
    # Multi-index replay
    # ------------------------------------------------------------------

    def prepare_multi_replay(
        self,
        index_ids: list[str],
        start_date: date,
        end_date: date,
        timeframe: str = "1d",
        benchmark_id: str = "NIFTY50",
        warmup_bars: int = 250,
    ) -> dict[str, ReplaySession]:
        """
        Create synchronized replay sessions for multiple indices.

        All sessions are aligned to the *intersection* of their trading
        dates so that ``advance()`` moves them to the same timestamp.
        """
        sessions: dict[str, ReplaySession] = {}
        for idx_id in index_ids:
            try:
                s = self.prepare_replay(
                    idx_id, start_date, end_date, timeframe,
                    benchmark_id, warmup_bars,
                )
                sessions[idx_id] = s
            except ValueError as exc:
                logger.warning("Skipping %s: %s", idx_id, exc)

        if not sessions:
            raise ValueError("No valid sessions could be created for the given indices.")

        # Synchronize: find common trading dates in the tradeable range
        date_sets = []
        for s in sessions.values():
            tradeable = s.full_price_data.iloc[s.warmup_bars:]
            date_sets.append(set(tradeable.index))

        common_dates = sorted(set.intersection(*date_sets))
        if len(common_dates) < 50:
            raise ValueError(
                f"Only {len(common_dates)} common trading bars across "
                f"{list(sessions.keys())}. Minimum 50 required."
            )

        # Re-slice each session to only include common dates (plus warmup)
        for idx_id, s in sessions.items():
            df = s.full_price_data
            warmup_df = df.iloc[: s.warmup_bars]
            tradeable_df = df.iloc[s.warmup_bars:]
            # Keep only rows whose timestamp is in common_dates
            aligned = tradeable_df[tradeable_df.index.isin(common_dates)]
            combined = pd.concat([warmup_df, aligned])
            s.full_price_data = combined
            s.total_bars = len(aligned)
            s.current_bar_index = s.warmup_bars
            s.trading_days = len(aligned)
            if len(aligned) > 0:
                s.actual_start = aligned.index[0].date()
                s.actual_end = aligned.index[-1].date()

        return sessions

    # ------------------------------------------------------------------
    # Private: data loading helpers
    # ------------------------------------------------------------------

    def _rows_to_ohlcv(self, rows: list[dict]) -> pd.DataFrame:
        """Convert DB rows to a DatetimeIndex OHLCV DataFrame."""
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        # Keep standard OHLCV columns; drop DB metadata
        keep = ["open", "high", "low", "close", "volume"]
        extra = [c for c in ("vwap",) if c in df.columns]
        return df[keep + extra]

    def _count_exists(self, query: str, params: tuple) -> bool:
        """Return True if a COUNT(*) query returns > 0."""
        row = self._db.fetch_one(query, params)
        return bool(row and row["cnt"] > 0)

    # --- Options pre-loading ---

    def _preload_options(
        self, index_id: str, start: date, end: date,
    ) -> tuple[list[str], dict[str, list[dict]]]:
        """Pre-load all options snapshots into a timestamp-keyed dict."""
        rows = self._db.fetch_all(
            Q.LIST_OPTIONS_CHAIN_AT_TIME,
            (index_id, "", start.isoformat(), end.isoformat() + "T23:59:59"),
        )
        if not rows:
            return [], {}

        cache: dict[str, list[dict]] = {}
        for r in rows:
            ts = r["timestamp"]
            cache.setdefault(ts, []).append(dict(r))

        timestamps = sorted(cache.keys())
        return timestamps, cache

    def _preload_oi(
        self, index_id: str, start: date, end: date,
    ) -> tuple[list[str], dict[str, dict]]:
        """Pre-load OI aggregated into a timestamp-keyed dict."""
        rows = self._db.fetch_all(
            Q.LIST_OI_AGGREGATED_HISTORY,
            (index_id, "", start.isoformat(), end.isoformat() + "T23:59:59"),
        )
        if not rows:
            return [], {}

        cache: dict[str, dict] = {}
        for r in rows:
            cache[r["timestamp"]] = dict(r)

        timestamps = sorted(cache.keys())
        return timestamps, cache

    def _preload_vix(
        self, start: date, end: date,
    ) -> tuple[list[str], dict[str, dict]]:
        """Pre-load VIX data into a timestamp-keyed dict."""
        rows = self._db.fetch_all(
            Q.LIST_VIX_HISTORY,
            (start.isoformat(), end.isoformat() + "T23:59:59"),
        )
        if not rows:
            return [], {}

        cache: dict[str, dict] = {}
        for r in rows:
            cache[r["timestamp"]] = dict(r)

        timestamps = sorted(cache.keys())
        return timestamps, cache

    def _preload_fii(
        self, start: date, end: date,
    ) -> tuple[list[str], dict[str, list[dict]]]:
        """Pre-load FII/DII data into a date-keyed dict."""
        rows = self._db.fetch_all(
            Q.LIST_FII_DII_RECENT,
            (start.isoformat(),),
        )
        if not rows:
            return [], {}

        cache: dict[str, list[dict]] = {}
        for r in rows:
            d = r["date"]
            if d <= end.isoformat():
                cache.setdefault(d, []).append(dict(r))

        dates = sorted(cache.keys())
        return dates, cache

    # --- Binary-search lookups on pre-loaded caches ---

    @staticmethod
    def _bisect_le(sorted_keys: list[str], target: str) -> Optional[str]:
        """Return the largest key in *sorted_keys* that is <= *target*,
        or None if no such key exists."""
        if not sorted_keys:
            return None
        pos = bisect.bisect_right(sorted_keys, target)
        if pos == 0:
            return None
        return sorted_keys[pos - 1]

    @staticmethod
    def _bisect_lt(sorted_keys: list[str], target: str) -> Optional[str]:
        """Return the largest key in *sorted_keys* that is < *target*."""
        if not sorted_keys:
            return None
        pos = bisect.bisect_left(sorted_keys, target)
        if pos == 0:
            return None
        return sorted_keys[pos - 1]

    def _lookup_options_at(
        self, session: ReplaySession, ts_str: str,
    ) -> Optional[dict]:
        """Look up pre-loaded options snapshot at or before *ts_str*."""
        key = self._bisect_le(session._options_timestamps, ts_str)
        if key is None:
            return None
        return {"strikes": session._options_cache[key], "snapshot_timestamp": key}

    def _lookup_oi_at(
        self, session: ReplaySession, ts_str: str,
    ) -> Optional[dict]:
        key = self._bisect_le(session._oi_timestamps, ts_str)
        if key is None:
            return None
        return session._oi_cache[key]

    def _lookup_vix_at(
        self, session: ReplaySession, ts_str: str,
    ) -> Optional[float]:
        key = self._bisect_le(session._vix_timestamps, ts_str)
        if key is None:
            return None
        return float(session._vix_cache[key]["vix_value"])

    def _lookup_fii_before(
        self, session: ReplaySession, date_str: str,
    ) -> Optional[dict]:
        """FII/DII data is delayed — use strictly before the given date."""
        key = self._bisect_lt(session._fii_dates, date_str)
        if key is None:
            return None
        entries = session._fii_cache[key]
        return self._aggregate_fii_rows(entries, key)

    @staticmethod
    def _aggregate_fii_rows(entries: list[dict], as_of_date: str) -> dict:
        """Bundle FII/DII rows for a single date into a summary dict."""
        fii_net = sum(r["net_value"] for r in entries if r["category"] == "FII")
        dii_net = sum(r["net_value"] for r in entries if r["category"] == "DII")
        return {
            "date": as_of_date,
            "fii_net": fii_net,
            "dii_net": dii_net,
            "total_net": fii_net + dii_net,
            "details": entries,
        }


# ---------------------------------------------------------------------------
# ReplayIterator — convenience wrapper for bar-by-bar looping
# ---------------------------------------------------------------------------

class ReplayIterator:
    """
    Iterate through a :class:`ReplaySession` bar-by-bar.

    Usage::

        engine = DataReplayEngine(db)
        session = engine.prepare_replay(...)
        for time_slice in ReplayIterator(engine, session):
            result = aggregator.analyze(session.index_id, time_slice.price_history)
    """

    def __init__(self, engine: DataReplayEngine, session: ReplaySession) -> None:
        self._engine = engine
        self._session = session

    def __iter__(self) -> ReplayIterator:
        self._engine.reset(self._session)
        # Yield the first bar (at warmup_bars position) on the first __next__
        self._started = False
        return self

    def __next__(self) -> TimeSlice:
        if not self._started:
            self._started = True
            return self._engine.get_current_slice(self._session)
        ts = self._engine.advance(self._session)
        if ts is None:
            raise StopIteration
        return ts

    def __len__(self) -> int:
        return self._session.total_bars
