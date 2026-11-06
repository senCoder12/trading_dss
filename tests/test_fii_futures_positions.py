"""
fii_futures_positions: separate table for participant-wise futures OI.

Covers parsing of NSE's fao_participant_oi archive, idempotent upserts,
constraints, the gap-filling sync, and — most importantly — that futures
positions (contract counts) never land in fii_dii_activity (₹ crore turnover)
and never show up in the cash readers.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.data.fii_dii_data import (
    PARTICIPANT_OI_URL,
    FIIDIIData,
    FIIDIIFetcher,
    parse_participant_oi_csv,
)
from src.database import queries as Q
from src.database.db_manager import DatabaseManager, QueryError
from src.database.migrations import _v20_fii_futures_positions

# Verbatim NSE file for 2026-11-05 (CRLF line endings as served)
CSV_2026_11_05 = "\r\n".join([
    '""Participant wise Open Interest (no. of contracts) in Equity Derivatives as on Nov 05, 2026"",,,,,,,,,,,,,,',
    "Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short       ,Option Index Call Long,Option Index Put Long,Option Index Call Short,Option Index Put Short,Option Stock Call Long,Option Stock Put Long,Option Stock Call Short,Option Stock Put Short,Total Long Contracts      ,Total Short Contracts",
    "Client,187440,82799,3141463,306451,2760218,3318042,2669313,3745090,1604934,1042838,1036875,1031293,12054935,8871822",
    "DII,53917,8791,322570,4416167,8583,104414,653,603,2577,41457,358497,31427,533518,4816138",
    "FII,52037,189475,3610905,2523555,802826,1236704,956574,810619,280284,350521,338192,262953,6333277,5081368",
    "Pro,28693,41022,676814,505579,1096242,1289384,1041328,1392233,1152292,924125,1306523,1033268,5167550,5319953",
    "TOTAL,322087,322087,7751752,7751752,4667868,5948545,4667868,5948545,3040087,2358941,3040087,2358941,24089280,24089280",
]) + "\r\n"

NOV5 = date(2026, 11, 5)
IST = ZoneInfo("Asia/Kolkata")


def _csv_for(day: date) -> str:
    return CSV_2026_11_05.replace("Nov 05, 2026", day.strftime("%b %d, %Y"))


@pytest.fixture()
def db(tmp_path):
    database = DatabaseManager(db_path=tmp_path / "fii.db")
    database.connect()
    database.initialise_schema()
    _v20_fii_futures_positions(database)
    yield database
    database.close()


def _fetcher(pages: dict[str, str] | None = None) -> FIIDIIFetcher:
    pages = pages or {}
    calls: list[str] = []

    def http_get(url: str):
        calls.append(url)
        return pages.get(url)

    f = FIIDIIFetcher(scraper=object(), http_get=http_get)  # scraper unused for futures
    f.calls = calls  # type: ignore[attr-defined]
    return f


def _url(day: date) -> str:
    return PARTICIPANT_OI_URL.format(ddmmyyyy=day.strftime("%d%m%Y"))


# ── Parsing ───────────────────────────────────────────────────────────────────


def test_parse_real_file() -> None:
    positions = parse_participant_oi_csv(CSV_2026_11_05, expected_date=NOV5)
    assert positions is not None
    assert len(positions) == 8  # 4 participants × {INDEX_FUT, STOCK_FUT}
    fii = next(p for p in positions if p.participant == "FII" and p.instrument == "INDEX_FUT")
    assert (fii.long_contracts, fii.short_contracts, fii.net_contracts) == (52037, 189475, -137438)
    assert {p.trade_date for p in positions} == {NOV5}
    assert {p.participant for p in positions} == {"FII", "DII", "PRO", "CLIENT"}


@pytest.mark.parametrize("text", [
    "<!DOCTYPE html><html><head><title>Resource not found</title></head></html>",
    "",
    CSV_2026_11_05.replace("FII,52037", "FII,52038"),        # rows no longer sum to TOTAL
    CSV_2026_11_05.replace("Future Index Long", "Fut Idx L"),  # unexpected header
])
def test_parse_rejects_bad_input(text: str) -> None:
    assert parse_participant_oi_csv(text, expected_date=NOV5) is None


def test_parse_pre_2025_format() -> None:
    """Older files: 'as on Jan 02,2024' (no space after comma) and a TAB in one header."""
    text = "\r\n".join([
        '""Participant wise Open Interest (no. of contracts) in Equity Derivatives as on Jan 02,2024"",,,,,,,,,,,,,,',
        "Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short\t,Option Index Call Long,Option Index Put Long,Option Index Call Short,Option Index Put Short,Option Stock Call Long,Option Stock Put Long,Option Stock Call Short,Option Stock Put Short,Total Long Contracts\t,Total Short Contracts",
        "Client,195331,237965,1626579,249785,6280912,3736471,6250446,4306826,1089322,338453,661653,542985,13267067,12249660",
        "DII,61533,74099,75446,2059977,0,0,0,0,0,0,0,0,0,0",
        "FII,150027,70562,1425213,1093493,0,0,0,0,0,0,0,0,0,0",
        "Pro,31786,56051,448364,172347,0,0,0,0,0,0,0,0,0,0",
        "TOTAL,438677,438677,3575602,3575602,0,0,0,0,0,0,0,0,0,0",
    ])
    positions = parse_participant_oi_csv(text, expected_date=date(2024, 1, 2))
    assert positions is not None
    fii = next(p for p in positions if p.participant == "FII" and p.instrument == "INDEX_FUT")
    assert fii.net_contracts == 150027 - 70562


def test_parse_quoted_year_title() -> None:
    # Some 2024 files quote the year in the title: as on Feb 22,"2024" followed by stray quotes
    text = CSV_2026_11_05.replace(
        '""Participant wise Open Interest (no. of contracts) in Equity Derivatives as on Nov 05, 2026""',
        'Participant wise Open Interest (no. of contracts) in Equity Derivatives as on Nov 05,"2026"""""',
    )
    assert parse_participant_oi_csv(text, expected_date=NOV5) is not None


def test_sync_retries_settled_miss_after_ttl(db) -> None:
    f = _fetcher()  # 2026-11-04 missing on NSE (transient 404)
    t0 = datetime(2026, 11, 9, 10, 0, tzinfo=IST)
    f.sync_futures_positions(db, lookback_days=5, now=t0, pause_seconds=0)
    assert _url(date(2026, 11, 4)) in f.calls

    f.calls.clear()
    f.sync_futures_positions(db, lookback_days=5, now=t0 + timedelta(hours=1), pause_seconds=0)
    assert _url(date(2026, 11, 4)) not in f.calls        # remembered for a while

    f.calls.clear()
    f.sync_futures_positions(db, lookback_days=5, now=t0 + timedelta(hours=7), pause_seconds=0)
    assert _url(date(2026, 11, 4)) in f.calls            # …then retried


def test_parse_rejects_date_mismatch() -> None:
    assert parse_participant_oi_csv(CSV_2026_11_05, expected_date=date(2026, 11, 4)) is None


# ── Table constraints & idempotent upsert ─────────────────────────────────────


def test_upsert_is_idempotent(db) -> None:
    f = _fetcher()
    positions = parse_participant_oi_csv(CSV_2026_11_05)
    assert f.save_futures_positions_to_db(positions, db) == 8
    assert f.save_futures_positions_to_db(positions, db) == 8

    rows = db.fetch_all("SELECT * FROM fii_futures_positions")
    assert len(rows) == 8
    fii = db.fetch_one(Q.GET_FII_INDEX_FUT_NET_BEFORE_DATE, ("2026-11-06",))
    assert dict(fii) == {"trade_date": "2026-11-05", "net_contracts": -137438}


def test_upsert_updates_revised_values(db) -> None:
    f = _fetcher()
    f.save_futures_positions_to_db(parse_participant_oi_csv(CSV_2026_11_05), db)
    revised = CSV_2026_11_05.replace("FII,52037,189475", "FII,52137,189375") \
                            .replace("Client,187440,82799", "Client,187340,82899")
    f.save_futures_positions_to_db(parse_participant_oi_csv(revised), db)

    row = db.fetch_one(
        "SELECT long_contracts, short_contracts, net_contracts FROM fii_futures_positions "
        "WHERE trade_date='2026-11-05' AND participant='FII' AND instrument='INDEX_FUT'"
    )
    assert (row["long_contracts"], row["short_contracts"], row["net_contracts"]) == (52137, 189375, -137238)
    assert db.fetch_one("SELECT COUNT(*) AS n FROM fii_futures_positions")["n"] == 8


def test_unique_and_check_constraints(db) -> None:
    insert = (
        "INSERT INTO fii_futures_positions (trade_date, participant, instrument, long_contracts, "
        "short_contracts, net_contracts, source, fetched_at) VALUES (?, ?, ?, ?, ?, ?, 'test', 'now')"
    )
    db.execute(insert, ("2026-11-05", "FII", "INDEX_FUT", 10, 4, 6))
    with pytest.raises(QueryError):  # UNIQUE (trade_date, participant, instrument)
        db.execute(insert, ("2026-11-05", "FII", "INDEX_FUT", 10, 4, 6))
    with pytest.raises(QueryError):  # net must equal long − short
        db.execute(insert, ("2026-11-04", "FII", "INDEX_FUT", 10, 4, 7))
    with pytest.raises(QueryError):  # unknown participant
        db.execute(insert, ("2026-11-04", "FPI", "INDEX_FUT", 10, 4, 6))
    with pytest.raises(QueryError):  # negative contracts
        db.execute(insert, ("2026-11-04", "FII", "INDEX_FUT", -1, 4, -5))


def test_lookup_index_exists(db) -> None:
    plan = db.fetch_all(
        "EXPLAIN QUERY PLAN " + Q.GET_FII_INDEX_FUT_NET_BEFORE_DATE, ("2026-11-06",)
    )
    assert any("idx_fii_fut_pos_lookup" in str(r["detail"]) for r in plan)


def test_point_in_time_read_excludes_same_day(db) -> None:
    _fetcher().save_futures_positions_to_db(parse_participant_oi_csv(CSV_2026_11_05), db)
    assert db.fetch_one(Q.GET_FII_INDEX_FUT_NET_BEFORE_DATE, ("2026-11-05",)) is None


# ── Never mixed with cash ─────────────────────────────────────────────────────


def _save_cash(f: FIIDIIFetcher, db) -> None:
    f.save_to_db(FIIDIIData(
        date=NOV5, fii_buy_value=20000.0, fii_sell_value=5025.32, fii_net_value=14974.68,
        dii_buy_value=9000.0, dii_sell_value=19444.54, dii_net_value=-10444.54,
    ), db)


def test_futures_write_never_touches_fii_dii_activity(db) -> None:
    f = _fetcher()
    _save_cash(f, db)
    before = db.fetch_all("SELECT * FROM fii_dii_activity ORDER BY id")

    f.save_futures_positions_to_db(parse_participant_oi_csv(CSV_2026_11_05), db)

    assert db.fetch_all("SELECT * FROM fii_dii_activity ORDER BY id") == before
    segments = {r["segment"] for r in db.fetch_all("SELECT DISTINCT segment FROM fii_dii_activity")}
    assert segments == {"CASH"}


def test_cash_write_never_touches_futures_table(db) -> None:
    f = _fetcher()
    _save_cash(f, db)
    assert db.fetch_one("SELECT COUNT(*) AS n FROM fii_futures_positions")["n"] == 0


def test_cash_readers_unchanged_after_futures_write(db) -> None:
    f = _fetcher()
    _save_cash(f, db)
    recent_before = db.fetch_all(Q.LIST_FII_DII_RECENT, ("2026-11-01",))
    before_date = db.fetch_all(Q.GET_FII_DII_BEFORE_DATE, ("2026-11-06",))
    agg_before = db.fetch_all(Q.AGG_FII_DII_NET_BY_DATE, ("2026-11-01",))
    # Telegram /fii reads these two directly
    tg_latest_before = db.fetch_all("SELECT * FROM fii_dii_activity ORDER BY date DESC LIMIT 1")
    tg_day_before = db.fetch_all("SELECT * FROM fii_dii_activity WHERE date = ? ORDER BY category", ("2026-11-05",))

    f.save_futures_positions_to_db(parse_participant_oi_csv(CSV_2026_11_05), db)

    assert db.fetch_all(Q.LIST_FII_DII_RECENT, ("2026-11-01",)) == recent_before
    assert db.fetch_all(Q.GET_FII_DII_BEFORE_DATE, ("2026-11-06",)) == before_date
    assert db.fetch_all(Q.AGG_FII_DII_NET_BY_DATE, ("2026-11-01",)) == agg_before
    assert db.fetch_all("SELECT * FROM fii_dii_activity ORDER BY date DESC LIMIT 1") == tg_latest_before
    assert db.fetch_all(
        "SELECT * FROM fii_dii_activity WHERE date = ? ORDER BY category", ("2026-11-05",)
    ) == tg_day_before
    # Cash values are crore turnover, untouched by the −137,438 contract position
    assert {r["net_value"] for r in before_date} == {14974.68, -10444.54}


def test_fii_dii_activity_check_constraint_unchanged(db) -> None:
    with pytest.raises(QueryError):
        db.execute(Q.INSERT_FII_DII_ACTIVITY, ("2026-11-05", "FII", 1.0, 2.0, -1.0, "F&O_FUTURES"))
    ddl = db.fetch_one("SELECT sql FROM sqlite_master WHERE name = 'fii_dii_activity'")["sql"]
    assert "CHECK (segment IN ('CASH','FO'))" in ddl


# ── Migration not applied → graceful skip, still no cash-table writes ─────────


def test_missing_table_skips_without_writing_anywhere(tmp_path) -> None:
    database = DatabaseManager(db_path=tmp_path / "pre_v20.db")
    database.connect()
    database.initialise_schema()  # must NOT create fii_futures_positions
    assert not database.table_exists("fii_futures_positions")

    f = _fetcher({_url(NOV5): CSV_2026_11_05})
    assert f.save_futures_positions_to_db(parse_participant_oi_csv(CSV_2026_11_05), database) == 0
    assert f.sync_futures_positions(database, now=datetime(2026, 11, 6, 10, 0, tzinfo=IST), pause_seconds=0) == []
    assert f.calls == []  # no network either
    assert database.fetch_one("SELECT COUNT(*) AS n FROM fii_dii_activity")["n"] == 0
    database.close()


# ── Gap-filling sync ──────────────────────────────────────────────────────────


def test_sync_fills_missing_weekdays_and_remembers_old_holidays(db) -> None:
    # Mon 2026-11-09 10:00: lookback covers Mon 11-02 … Fri 11-06; 11-04 is a "holiday" (404)
    pages = {_url(d): _csv_for(d) for d in (date(2026, 11, 2), date(2026, 11, 3),
                                             date(2026, 11, 5), date(2026, 11, 6))}
    f = _fetcher(pages)
    now = datetime(2026, 11, 9, 10, 0, tzinfo=IST)

    written = f.sync_futures_positions(db, lookback_days=7, now=now, pause_seconds=0)

    assert written == [date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 5), date(2026, 11, 6)]
    assert _url(date(2026, 11, 9)) not in f.calls  # today, before publication cutoff
    dates = {r["trade_date"] for r in db.fetch_all("SELECT DISTINCT trade_date FROM fii_futures_positions")}
    assert dates == {"2026-11-02", "2026-11-03", "2026-11-05", "2026-11-06"}

    # Second run: everything present, the 5-day-old holiday remembered → zero requests
    f.calls.clear()
    assert f.sync_futures_positions(db, lookback_days=7, now=now, pause_seconds=0) == []
    assert f.calls == []


def test_sync_fetches_today_only_after_cutoff_and_retries_recent_misses(db) -> None:
    f = _fetcher()  # nothing published yet
    before = datetime(2026, 11, 6, 20, 0, tzinfo=IST)
    after = datetime(2026, 11, 6, 21, 15, tzinfo=IST)

    f.sync_futures_positions(db, lookback_days=1, now=before, pause_seconds=0)
    assert f.calls == [_url(date(2026, 11, 5))]

    f.calls.clear()
    f.sync_futures_positions(db, lookback_days=1, now=after, pause_seconds=0)
    # yesterday's miss is recent → retried; today requested after the cutoff
    assert f.calls == [_url(date(2026, 11, 5)), _url(date(2026, 11, 6))]


def test_sync_skips_weekends(db) -> None:
    f = _fetcher()
    # Mon 11-09 before cutoff: window Sat 11-07 … Sun 11-08 → nothing requested
    f.sync_futures_positions(db, lookback_days=2, now=datetime(2026, 11, 9, 10, 0, tzinfo=IST), pause_seconds=0)
    assert f.calls == []
