"""
Backfill index OHLCV from Angel One historical candles into price_data.

Fills gaps left while the system was offline. Re-runnable: rows are UPSERTed on
(index_id, timestamp, timeframe), so overlapping runs overwrite with Angel data.

Usage
-----
::

    python scripts/backfill_angel.py                                  # NIFTY50 + SENSEX, from 2026-06-01
    python scripts/backfill_angel.py --from 2026-06-01 --to 2026-11-05
    python scripts/backfill_angel.py --indices NIFTY50 BANKNIFTY --timeframes 1d 5m
    python scripts/backfill_angel.py --dry-run                        # fetch + count, no DB writes
    python scripts/backfill_angel.py --indices --vix                  # India VIX daily closes only

Stop run_system.py first: this script opens its own Angel One session.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import IST_TIMEZONE  # noqa: E402
from src.data.broker.angel_one import (  # noqa: E402
    CANDLE_MAX_DAYS,
    INDEX_REST_TOKENS,
    INDIA_VIX_REST_TOKEN,
    AngelOneClient,
)
from src.database import queries as Q  # noqa: E402
from src.database.db_manager import DatabaseManager  # noqa: E402

logger = logging.getLogger("backfill_angel")
_IST = ZoneInfo(IST_TIMEZONE)

SOURCE = "ANGEL_HIST"
TIMEFRAME_TO_INTERVAL = {"1d": "ONE_DAY", "5m": "FIVE_MINUTE", "1m": "ONE_MINUTE"}
REQUEST_GAP_SEC = 0.4  # getCandleData allows ~3 req/s


def fetch_range(client: AngelOneClient, exchange: str, token: str, interval: str,
                start: datetime, end: datetime) -> list[list]:
    """Fetch [start, end] in chunks that respect the per-interval span limit."""
    span = timedelta(days=CANDLE_MAX_DAYS[interval] - 1)
    rows: list[list] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + span, end)
        data = client.get_candle_data(exchange, token, interval, cur, chunk_end)
        if data is None:
            raise RuntimeError(f"candle fetch failed for {exchange}:{token} {interval} {cur:%Y-%m-%d}")
        rows.extend(data)
        cur = chunk_end + timedelta(minutes=1)
        time.sleep(REQUEST_GAP_SEC)
    return rows


def to_params(index_id: str, timeframe: str, rows: list[list]) -> list[tuple]:
    """Convert Angel candle rows to UPSERT_PRICE_DATA params, dropping malformed bars."""
    params = []
    seen = set()
    for r in rows:
        if len(r) < 6:
            continue
        ts, o, h, l, c, v = r[:6]
        if ts in seen or min(o, h, l, c) <= 0 or h < l:
            continue
        seen.add(ts)
        params.append((index_id, ts, float(o), float(h), float(l), float(c), float(v or 0), None, SOURCE, timeframe))
    return params


def backfill_vix(client: AngelOneClient, db: DatabaseManager | None, start: datetime, end: datetime) -> bool:
    """Write one vix_data row per trading day (stamped 00:00, like the existing daily seeds).

    vix_data has no unique key, so days that already have a 00:00 row are skipped.
    """
    exchange, token = INDIA_VIX_REST_TOKEN
    try:
        rows = fetch_range(client, exchange, token, "ONE_DAY", start, end)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return False
    rows = [r for r in rows if len(r) >= 5 and r[4] and r[4] > 0]
    if not rows:
        logger.warning("INDIA VIX: no candles returned")
        return False

    # Previous close for the first row's change
    first_ts = rows[0][0]
    prev = None
    existing: set[str] = set()
    if db:
        r = db.fetch_one(
            "SELECT vix_value FROM vix_data WHERE substr(timestamp, 1, 10) < substr(?, 1, 10) "
            "ORDER BY timestamp DESC LIMIT 1",
            (first_ts,),
        )
        prev = r["vix_value"] if r else None
        existing = {
            x["timestamp"] for x in db.fetch_all(
                "SELECT timestamp FROM vix_data WHERE timestamp >= ?", (first_ts,)
            )
        }

    params = []
    for ts, _o, _h, _l, close, *_ in rows:
        change = round(close - prev, 2) if prev else 0.0
        change_pct = round(change / prev * 100, 2) if prev else 0.0
        prev = close
        if ts not in existing:
            params.append((ts, float(close), change, change_pct))

    if db and params:
        db.execute_many(Q.INSERT_VIX_DATA, params)
    logger.info("INDIA VIX  1d  %6d new rows (%d fetched)  %s → %s%s", len(params), len(rows),
                rows[0][0], rows[-1][0], "  (dry run)" if db is None else "")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indices", nargs="*", default=["NIFTY50", "SENSEX"], choices=sorted(INDEX_REST_TOKENS))
    p.add_argument("--vix", action="store_true", help="also backfill India VIX daily closes into vix_data")
    p.add_argument("--timeframes", nargs="+", default=["1d", "5m", "1m"], choices=list(TIMEFRAME_TO_INTERVAL))
    p.add_argument("--from", dest="from_date", default="2026-06-01", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--to", dest="to_date", default=None, help="YYYY-MM-DD (inclusive, default today)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Whole days: daily bars are stamped 00:00 and todate is exclusive of the last minute
    start = datetime.strptime(args.from_date, "%Y-%m-%d")
    end_day = datetime.strptime(args.to_date, "%Y-%m-%d") if args.to_date else datetime.now(tz=_IST).replace(tzinfo=None)
    end = end_day.replace(hour=23, minute=59, second=0, microsecond=0)

    client = AngelOneClient()
    if not client.login():
        logger.error("Angel One login failed")
        return 1

    db = None
    if not args.dry_run:
        db = DatabaseManager()
        db.connect()

    failed = False
    for index_id in args.indices:
        exchange, token = INDEX_REST_TOKENS[index_id]
        for tf in args.timeframes:
            try:
                rows = fetch_range(client, exchange, token, TIMEFRAME_TO_INTERVAL[tf], start, end)
            except RuntimeError as exc:
                logger.error("%s", exc)
                failed = True
                continue
            params = to_params(index_id, tf, rows)
            span = f"{params[0][1]} → {params[-1][1]}" if params else "none"
            if db and params:
                db.execute_many(Q.UPSERT_PRICE_DATA, params)
            logger.info("%-10s %-3s %6d bars  %s%s", index_id, tf, len(params), span,
                        "  (dry run)" if args.dry_run else "")

    if args.vix and not backfill_vix(client, db, start, end):
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
