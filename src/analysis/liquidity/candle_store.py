"""
CandleStore — rolling in-memory OHLCV store for all liquidity analyzers.

Single data source for all SMC analyzers. Ticks arrive via add_tick(),
candles are built in real time, callbacks fire on close, and DB backs
the store for crash recovery within the same trading day.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from src.analysis.liquidity.models import Candle
from src.database.db_manager import DatabaseManager
from src.data.smart_api_client import SmartAPIClient

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# Minutes per timeframe — used for candle boundary calculations
_TF_MINUTES: dict[str, int] = {
    "1min": 1,
    "3min": 3,
    "5min": 5,
    "15min": 15,
    "1hour": 60,
}

_SUPPORTED_TIMEFRAMES = list(_TF_MINUTES.keys())
_DEQUE_MAXLEN = 500


# ---------------------------------------------------------------------------
# _OpenCandle — mutable accumulator for a forming candle
# ---------------------------------------------------------------------------

class _OpenCandle:
    """Accumulates ticks until the candle's time boundary closes."""

    __slots__ = (
        "symbol", "timeframe", "open", "high", "low", "close",
        "volume", "open_time",
    )

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        price: float,
        volume: int,
        open_time: datetime,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = volume
        self.open_time = open_time

    def update(self, price: float, volume: int) -> None:
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume += volume

    def to_candle(self) -> Candle:
        return Candle(
            symbol=self.symbol,
            timeframe=self.timeframe,
            timestamp=self.open_time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


# ---------------------------------------------------------------------------
# CandleStore
# ---------------------------------------------------------------------------

class CandleStore:
    """
    Rolling in-memory OHLCV store for multiple symbols and timeframes.

    Usage
    -----
    At 9:00 AM call load_historical() for all tracked symbols, then feed
    every WebSocket tick through add_tick(). Register callbacks via
    register_candle_callback() to react to closed candles.

    Parameters
    ----------
    db:
        DatabaseManager for crash-recovery persistence.
    smart_api:
        SmartAPIClient used to back-fill historical candles.
    """

    def __init__(self, db: DatabaseManager, smart_api: SmartAPIClient) -> None:
        self._db = db
        self._smart_api = smart_api

        # {symbol: {timeframe: deque[Candle]}}
        self._candles: dict[str, dict[str, deque[Candle]]] = {}

        # {symbol: {timeframe: _OpenCandle | None}}
        self._open: dict[str, dict[str, Optional[_OpenCandle]]] = {}

        # {timeframe: [callback, ...]}
        self._callbacks: dict[str, list[Callable[[Candle], None]]] = {
            tf: [] for tf in _SUPPORTED_TIMEFRAMES
        }

        self._ensure_db_table()

    # ── Public API ────────────────────────────────────────────────────────────

    def init_symbol(self, symbol: str) -> None:
        """
        Allocate empty deques for a symbol on all supported timeframes.

        Call this before load_historical() or add_tick() for a new symbol.
        """
        if symbol not in self._candles:
            self._candles[symbol] = {
                tf: deque(maxlen=_DEQUE_MAXLEN) for tf in _SUPPORTED_TIMEFRAMES
            }
            self._open[symbol] = {tf: None for tf in _SUPPORTED_TIMEFRAMES}

    def load_historical(
        self,
        symbol: str,
        timeframe: str,
        count: int = 200,
        exchange: str = "NSE",
        token: str = "",
    ) -> None:
        """
        Populate a symbol/timeframe deque from SmartAPI historical data or DB.

        Tries the DB first (same trading day). Falls back to SmartAPI.

        Parameters
        ----------
        symbol:
            Instrument symbol string.
        timeframe:
            One of the supported timeframes.
        count:
            Number of candles to load (up to _DEQUE_MAXLEN).
        exchange:
            'NSE' | 'NFO' — passed to SmartAPIClient.
        token:
            Instrument token for SmartAPI.
        """
        if timeframe not in _TF_MINUTES:
            logger.warning("Unsupported timeframe '%s' — skip load_historical", timeframe)
            return

        self.init_symbol(symbol)

        # Try DB first — fast path within same trading day
        loaded = self._load_from_db(symbol, timeframe, count)
        if loaded:
            logger.info(
                "CandleStore: loaded %d candles from DB for %s/%s",
                len(loaded), symbol, timeframe,
            )
            for c in loaded:
                self._candles[symbol][timeframe].append(c)
            return

        # Fall back to SmartAPI
        if not token:
            logger.warning(
                "CandleStore: no DB data and no token for %s/%s — cannot load",
                symbol, timeframe,
            )
            return

        api_interval = _tf_to_api_interval(timeframe)
        bars = self._smart_api.get_historical_ohlcv(
            exchange=exchange,
            token=token,
            interval=api_interval,
            count=count,
        )
        if not bars:
            logger.warning(
                "CandleStore: SmartAPI returned no data for %s/%s",
                symbol, timeframe,
            )
            return

        candles: list[Candle] = []
        for bar in bars:
            try:
                ts = _parse_timestamp(bar["timestamp"])
                candle = Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=float(bar["open"]),
                    high=float(bar["high"]),
                    low=float(bar["low"]),
                    close=float(bar["close"]),
                    volume=int(bar["volume"]),
                )
                candles.append(candle)
            except Exception as exc:
                logger.debug("CandleStore: skip malformed bar for %s: %s", symbol, exc)

        for c in candles:
            self._candles[symbol][timeframe].append(c)

        self._persist_candles(candles)
        logger.info(
            "CandleStore: loaded %d candles from SmartAPI for %s/%s",
            len(candles), symbol, timeframe,
        )

    def add_tick(
        self,
        symbol: str,
        ltp: float,
        volume: int,
        timestamp: datetime,
    ) -> None:
        """
        Ingest a live tick and update open candles for all timeframes.

        When a time boundary is crossed the open candle is finalised,
        appended to the deque, persisted, and all registered callbacks
        are fired. Late ticks (past candle boundary) are merged into the
        already-closed candle rather than re-opening it.

        Parameters
        ----------
        symbol:
            Instrument symbol.
        ltp:
            Last traded price.
        volume:
            Tick volume (incremental, not cumulative).
        timestamp:
            Timezone-aware datetime of the tick.
        """
        self.init_symbol(symbol)

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=_IST)

        for tf in _SUPPORTED_TIMEFRAMES:
            self._process_tick_for_tf(symbol, tf, ltp, volume, timestamp)

    def register_candle_callback(
        self,
        timeframe: str,
        callback: Callable[[Candle], None],
    ) -> None:
        """
        Register a callback invoked when a candle closes on *timeframe*.

        Multiple callbacks per timeframe are allowed.

        Parameters
        ----------
        timeframe:
            One of the supported timeframe strings.
        callback:
            Callable receiving the just-closed Candle.
        """
        if timeframe not in _SUPPORTED_TIMEFRAMES:
            logger.warning(
                "CandleStore: unknown timeframe '%s' for callback", timeframe
            )
            return
        self._callbacks[timeframe].append(callback)

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
    ) -> list[Candle]:
        """
        Return the last *count* closed candles, oldest first.

        Returns empty list if symbol/timeframe not initialised.
        """
        try:
            dq = self._candles[symbol][timeframe]
        except KeyError:
            return []
        candles = list(dq)
        return candles[-count:] if count < len(candles) else candles

    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[Candle]:
        """Return the most recently closed candle, or None."""
        try:
            dq = self._candles[symbol][timeframe]
            return dq[-1] if dq else None
        except KeyError:
            return None

    def get_open_candle(self, symbol: str, timeframe: str) -> Optional[Candle]:
        """Return the currently forming (not yet closed) candle, or None."""
        try:
            oc = self._open[symbol][timeframe]
            return oc.to_candle() if oc is not None else None
        except KeyError:
            return None

    @staticmethod
    def aggregate_to_timeframe(
        candles_1min: list[Candle],
        target_timeframe: str,
    ) -> list[Candle]:
        """
        Aggregate 1-minute candles into *target_timeframe* candles.

        Parameters
        ----------
        candles_1min:
            List of 1-min Candle objects in chronological order.
        target_timeframe:
            Target timeframe string, e.g. '5min', '15min', '1hour'.

        Returns
        -------
        list[Candle]:
            Aggregated candles in chronological order.
        """
        if target_timeframe not in _TF_MINUTES:
            logger.warning("aggregate_to_timeframe: unknown timeframe '%s'", target_timeframe)
            return []
        if not candles_1min:
            return []

        period = _TF_MINUTES[target_timeframe]
        result: list[Candle] = []
        bucket: list[Candle] = []
        bucket_open_time: Optional[datetime] = None

        for c in candles_1min:
            # Align to period boundary (floor to period-minute mark)
            aligned = _floor_to_period(c.timestamp, period)

            if bucket_open_time is None:
                bucket_open_time = aligned

            if aligned != bucket_open_time:
                if bucket:
                    result.append(_merge_candles(bucket, target_timeframe))
                bucket = []
                bucket_open_time = aligned

            bucket.append(c)

        if bucket:
            result.append(_merge_candles(bucket, target_timeframe))

        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _process_tick_for_tf(
        self,
        symbol: str,
        tf: str,
        ltp: float,
        volume: int,
        timestamp: datetime,
    ) -> None:
        """Update the open candle for one timeframe; close and emit if needed."""
        period_minutes = _TF_MINUTES[tf]
        candle_open_time = _floor_to_period(timestamp, period_minutes)
        candle_close_time = candle_open_time + timedelta(minutes=period_minutes)

        oc = self._open[symbol][tf]

        if oc is None:
            # No open candle — start one
            self._open[symbol][tf] = _OpenCandle(
                symbol, tf, ltp, volume, candle_open_time
            )
            return

        if candle_open_time == oc.open_time:
            # Same period — update
            oc.update(ltp, volume)
            return

        if candle_open_time > oc.open_time:
            # Boundary crossed — close the old candle
            closed = oc.to_candle()
            self._candles[symbol][tf].append(closed)
            self._persist_candles([closed])
            self._fire_callbacks(tf, closed)
            logger.debug(
                "CandleStore: closed %s %s candle at %s",
                symbol, tf, closed.timestamp,
            )

            # Start a fresh open candle for the new period
            self._open[symbol][tf] = _OpenCandle(
                symbol, tf, ltp, volume, candle_open_time
            )
        else:
            # Late tick — belongs to the already-closed previous candle.
            # Merge into the last closed candle rather than reopening.
            dq = self._candles[symbol][tf]
            if dq:
                last = dq[-1]
                if last.timestamp == candle_open_time:
                    # Merge: extend high/low, add volume
                    merged = Candle(
                        symbol=last.symbol,
                        timeframe=last.timeframe,
                        timestamp=last.timestamp,
                        open=last.open,
                        high=max(last.high, ltp),
                        low=min(last.low, ltp),
                        close=last.close,   # close stays as original
                        volume=last.volume + volume,
                    )
                    dq[-1] = merged
                    logger.debug(
                        "CandleStore: late tick merged into closed candle %s %s @%s",
                        symbol, tf, last.timestamp,
                    )

    def _fire_callbacks(self, timeframe: str, candle: Candle) -> None:
        for cb in self._callbacks.get(timeframe, []):
            try:
                cb(candle)
            except Exception as exc:
                logger.error(
                    "CandleStore: callback error on %s/%s: %s",
                    candle.symbol, timeframe, exc,
                )

    # ── DB persistence ─────────────────────────────────────────────────────────

    def _ensure_db_table(self) -> None:
        try:
            from src.database import queries as Q
            self._db.execute(Q.CREATE_CANDLE_CACHE)
        except Exception as exc:
            logger.error("CandleStore: failed to create candle_cache table: %s", exc)

    def _persist_candles(self, candles: list[Candle]) -> None:
        if not candles:
            return
        try:
            from src.database import queries as Q
            rows = [
                (
                    c.symbol,
                    c.timeframe,
                    c.timestamp.isoformat(),
                    c.open,
                    c.high,
                    c.low,
                    c.close,
                    c.volume,
                )
                for c in candles
            ]
            self._db.execute_many(Q.UPSERT_CANDLE_CACHE, rows)
        except Exception as exc:
            logger.error("CandleStore: DB persist failed: %s", exc)

    def _load_from_db(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> list[Candle]:
        """Load candles from DB for the current trading day."""
        try:
            from src.database import queries as Q
            today = datetime.now(_IST).date().isoformat()
            rows = self._db.fetch_all(
                Q.LIST_CANDLE_CACHE,
                (symbol, timeframe, today, count),
            )
            candles: list[Candle] = []
            for row in rows:
                ts = _parse_timestamp(row["timestamp"])
                candles.append(
                    Candle(
                        symbol=row["symbol"],
                        timeframe=row["timeframe"],
                        timestamp=ts,
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                    )
                )
            return candles
        except Exception as exc:
            logger.debug("CandleStore: DB load skipped: %s", exc)
            return []

    def purge_old_candles(self, trading_days: int = 5) -> int:
        """
        Delete candles older than *trading_days* from DB.

        Returns number of rows deleted.
        """
        try:
            from src.database import queries as Q
            cutoff = (
                datetime.now(_IST) - timedelta(days=trading_days * 2)
            ).isoformat()
            cur = self._db.execute(Q.DELETE_CANDLE_CACHE_OLD, (cutoff,))
            deleted = cur.rowcount
            logger.info("CandleStore: purged %d old candle rows", deleted)
            return deleted
        except Exception as exc:
            logger.error("CandleStore: purge failed: %s", exc)
            return 0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _floor_to_period(ts: datetime, period_minutes: int) -> datetime:
    """Floor *ts* down to the nearest *period_minutes* boundary."""
    total_minutes = ts.hour * 60 + ts.minute
    floored_minutes = (total_minutes // period_minutes) * period_minutes
    return ts.replace(
        hour=floored_minutes // 60,
        minute=floored_minutes % 60,
        second=0,
        microsecond=0,
    )


def _merge_candles(candles: list[Candle], timeframe: str) -> Candle:
    """Merge a list of candles into a single higher-timeframe candle."""
    return Candle(
        symbol=candles[0].symbol,
        timeframe=timeframe,
        timestamp=candles[0].timestamp,
        open=candles[0].open,
        high=max(c.high for c in candles),
        low=min(c.low for c in candles),
        close=candles[-1].close,
        volume=sum(c.volume for c in candles),
    )


def _tf_to_api_interval(timeframe: str) -> str:
    """Map internal timeframe string to SmartAPI interval string."""
    return {
        "1min": "1",
        "3min": "3",
        "5min": "5",
        "15min": "15",
        "1hour": "60",
    }.get(timeframe, "1")


def _parse_timestamp(raw: str | datetime) -> datetime:
    """Parse a timestamp string or pass through a datetime, ensuring IST tz."""
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=_IST)
    # SmartAPI format: "2024-01-15 09:15"
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=_IST)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {raw!r}")
