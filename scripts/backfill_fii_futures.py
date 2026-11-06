"""
Backfill fii_futures_positions from NSE's participant-wise OI archive
(fao_participant_oi_DDMMYYYY.csv: one file per trading session, kept historically).

Writes ONLY to fii_futures_positions, never to fii_dii_activity. The target DB
must already have migration v20 applied. The script refuses to create the table.
Idempotent: re-running a range rewrites identical rows.

Every calendar day is tried, because NSE holds occasional Saturday sessions
(e.g. Budget day 2025-02-01). Non-session days return 404. Raw files are
cached under --cache-dir: this gives an audit trail, and re-runs need no
network.

Usage:
    .venv/bin/python scripts/backfill_fii_futures.py --start 2024-01-01 --end 2026-11-05 [--db PATH]
    .venv/bin/python scripts/backfill_fii_futures.py --start ... --end ... --dry-run
    .venv/bin/python scripts/backfill_fii_futures.py --start ... --end ... --validate
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import MARKET_HOLIDAYS  # noqa: E402
from config.settings import PROJECT_ROOT, settings  # noqa: E402
from src.data.fii_dii_data import FIIDIIFetcher, _http_get_text  # noqa: E402
from src.database.db_manager import DatabaseManager  # noqa: E402

DEFAULT_CACHE = PROJECT_ROOT / "data" / "cache" / "nse_participant_oi"
_MISSING = "__404__"


def cached_getter(cache_dir: Path, today: date, pause: float) -> Callable[[str], Optional[str]]:
    """HTTP GET with an on-disk cache.

    Misses are cached only for weekends. NSE occasionally serves a transient
    404 for a real session (seen for 2025-09-19), so weekday misses are always
    retried.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats = {"network": 0, "cache": 0}

    def get(url: str) -> Optional[str]:
        name = url.rsplit("/", 1)[-1]
        path = cache_dir / name
        ddmmyyyy = name.split("_")[-1].split(".")[0]
        day = date(int(ddmmyyyy[4:]), int(ddmmyyyy[2:4]), int(ddmmyyyy[:2]))
        if path.exists():
            text = path.read_text()
            if text != _MISSING or day.weekday() >= 5:
                stats["cache"] += 1
                return None if text == _MISSING else text
            path.unlink()  # stale weekday miss from an older run → retry
        stats["network"] += 1
        text = _http_get_text(url)
        time.sleep(pause)
        if text is not None:
            path.write_text(text)
        elif day.weekday() >= 5 and (today - day).days > 5:
            path.write_text(_MISSING)
        return text

    get.stats = stats  # type: ignore[attr-defined]
    return get


def expected_sessions(db: DatabaseManager, start: date, end: date) -> set[date]:
    """Trading sessions from NIFTY daily bars where available, else weekdays minus holidays."""
    rows = db.fetch_all(
        "SELECT DISTINCT substr(timestamp,1,10) AS d FROM price_data "
        "WHERE index_id='NIFTY50' AND timeframe='1d' AND substr(timestamp,1,10) BETWEEN ? AND ?",
        (start.isoformat(), end.isoformat()),
    )
    bar_days = {date.fromisoformat(r["d"]) for r in rows}
    first_bar = min(bar_days) if bar_days else end + timedelta(days=1)
    days, d = set(bar_days), start
    while d < first_bar and d <= end:
        if d.weekday() < 5 and d not in MARKET_HOLIDAYS:
            days.add(d)
        d += timedelta(days=1)
    return days


def table_digest(db: DatabaseManager, sql: str) -> tuple[int, str]:
    rows = db._conn().execute(sql).fetchall()
    h = hashlib.sha256(repr([tuple(r) for r in rows]).encode()).hexdigest()[:16]
    return len(rows), h


FUT_DIGEST_SQL = ("SELECT trade_date, participant, instrument, long_contracts, short_contracts, "
                  "net_contracts, source FROM fii_futures_positions ORDER BY 1, 2, 3")
CASH_DIGEST_SQL = "SELECT * FROM fii_dii_activity ORDER BY id"


def validate(db: DatabaseManager, start: date, end: date) -> bool:
    ok = True

    def check(cond: bool, label: str, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")

    s, e = start.isoformat(), end.isoformat()
    loaded = {date.fromisoformat(r["trade_date"]) for r in db.fetch_all(
        "SELECT DISTINCT trade_date FROM fii_futures_positions WHERE trade_date BETWEEN ? AND ?", (s, e))}
    expected = expected_sessions(db, start, end)
    missing = sorted(expected - loaded)
    extra = sorted(loaded - expected)
    print(f"  sessions expected={len(expected)} loaded={len(loaded)}")
    check(not missing, "every expected trading session loaded",
          ", ".join(map(str, missing[:20])) + (" …" if len(missing) > 20 else ""))
    if extra:
        print(f"  [INFO] sessions loaded beyond calendar (special sessions?): {', '.join(map(str, extra))}")

    bad_shape = db.fetch_all(
        "SELECT trade_date, instrument, COUNT(*) AS n FROM fii_futures_positions "
        "WHERE trade_date BETWEEN ? AND ? GROUP BY trade_date, instrument HAVING n != 4", (s, e))
    check(not bad_shape, "4 participants per session and instrument", str([dict(r) for r in bad_shape[:5]]))

    unbalanced = db.fetch_all(
        "SELECT trade_date, instrument, SUM(long_contracts) AS l, SUM(short_contracts) AS sh "
        "FROM fii_futures_positions WHERE trade_date BETWEEN ? AND ? "
        "GROUP BY trade_date, instrument HAVING l != sh", (s, e))
    check(not unbalanced, "market-wide long OI == short OI for every session/instrument",
          str([dict(r) for r in unbalanced[:5]]))

    bad_net = db.fetch_one(
        "SELECT COUNT(*) AS n FROM fii_futures_positions WHERE net_contracts != long_contracts - short_contracts")
    check(bad_net["n"] == 0, "net = long − short on every row")

    leaked = db.fetch_one(
        "SELECT COUNT(*) AS n FROM fii_dii_activity WHERE segment NOT IN ('CASH','FO')")
    check(leaked["n"] == 0, "fii_dii_activity holds no futures/unknown segments")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, type=date.fromisoformat)
    ap.add_argument("--end", required=True, type=date.fromisoformat)
    ap.add_argument("--db", type=Path, default=None, help="SQLite path (default: settings.database.path)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--dry-run", action="store_true", help="fetch + validate files only, no writes")
    ap.add_argument("--validate", action="store_true", help="only run the DB validation checks")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between NSE requests")
    args = ap.parse_args()
    if args.end < args.start:
        ap.error("--end is before --start")

    db_path = args.db or settings.database.path
    db = DatabaseManager(db_path=db_path)
    db.connect()
    if not db.table_exists("fii_futures_positions"):
        print(f"ERROR: fii_futures_positions not found in {db_path} — apply migration v20 first.")
        return 2
    if args.validate:
        print(f"Validating {args.start} → {args.end} in {db_path}")
        return 0 if validate(db, args.start, args.end) else 1

    cash_before = table_digest(db, CASH_DIGEST_SQL)
    getter = cached_getter(args.cache_dir, date.today(), args.pause)
    fetcher = FIIDIIFetcher(scraper=object(), http_get=getter)  # cash scraper unused here

    loaded, rows, rejected = [], 0, []
    day = args.start
    while day <= args.end:
        positions = fetcher.fetch_futures_positions(day)
        if positions:
            if not args.dry_run:
                rows += fetcher.save_futures_positions_to_db(positions, db)
            loaded.append(day)
        elif (args.cache_dir / f"fao_participant_oi_{day:%d%m%Y}.csv").exists() and \
                (args.cache_dir / f"fao_participant_oi_{day:%d%m%Y}.csv").read_text() != _MISSING:
            rejected.append(day)  # file exists but failed validation
        day += timedelta(days=1)

    cash_after = table_digest(db, CASH_DIGEST_SQL)
    print(f"{'DRY RUN — ' if args.dry_run else ''}sessions loaded: {len(loaded)}, rows written: {rows}, "
          f"requests: network={getter.stats['network']} cache={getter.stats['cache']}")
    if rejected:
        print("FILES REJECTED BY VALIDATION:", ", ".join(map(str, rejected)))
    print(f"fii_dii_activity unchanged: {cash_before == cash_after} (rows={cash_after[0]}, digest={cash_after[1]})")
    print(f"fii_futures_positions: rows={table_digest(db, FUT_DIGEST_SQL)[0]}, "
          f"digest={table_digest(db, FUT_DIGEST_SQL)[1]}")
    db.close()
    return 0 if (cash_before == cash_after and not rejected) else 1


if __name__ == "__main__":
    sys.exit(main())
