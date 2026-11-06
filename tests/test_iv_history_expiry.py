"""
collect_iv_history must read the nearest *live* expiry, not the oldest expiry
ever stored in options_chain_snapshot (which keeps expired contracts forever).
"""

from __future__ import annotations

from datetime import datetime

import pytest

import src.data.data_collector as dc
from src.data.data_collector import DataCollector
from src.database import queries as Q
from src.database.db_manager import DatabaseManager


@pytest.fixture()
def db(tmp_path):
    database = DatabaseManager(db_path=tmp_path / "iv.db")
    database.connect()
    database.initialise_schema()
    database.execute(Q.CREATE_IV_HISTORY)
    now = "2026-11-01T09:00:00+05:30"
    database.execute(
        Q.INSERT_INDEX_MASTER,
        ("NIFTY50", "NIFTY 50", "NIFTY 50", "^NSEI", "NSE", 75, 1, "NIFTY",
         "broad_market", 1, now, now),
    )
    database.execute(
        "INSERT INTO price_data (index_id, timestamp, open, high, low, close, volume, source, timeframe) "
        "VALUES ('NIFTY50', '2026-11-06T09:59:00+05:30', 24010, 24010, 24010, 24010, 0, 'test', '1m')"
    )
    # ATM 24000 CE/PE for three expiries; IV identifies which expiry was read.
    for expiry, ce_iv, pe_iv in (
        ("2026-10-27", 50.0, 50.0),   # long expired
        ("2026-11-06", 20.0, 22.0),   # expires today
        ("2026-11-10", 12.0, 14.0),   # next weekly
    ):
        for opt, iv in (("CE", ce_iv), ("PE", pe_iv)):
            database.execute(Q.INSERT_OPTIONS_CHAIN, (
                "NIFTY50", "2026-11-06T09:55:00+05:30", expiry, 24000.0, opt,
                1000, 0, 0.0, 0, 100.0, 0.0, 0.0, iv, 99.0, 75, 101.0, 75,
            ))
    yield database
    database.close()


def _collector(db: DatabaseManager) -> DataCollector:
    collector = DataCollector.__new__(DataCollector)  # skip scraper/broker init
    collector.db = db
    collector.dry_run = False
    collector.force_start = True
    return collector


def _freeze(monkeypatch, hour: int, minute: int) -> None:
    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 6, hour, minute, tzinfo=tz)

    monkeypatch.setattr(dc, "datetime", _Fixed)


def _atm_iv(db: DatabaseManager):
    rows = db.fetch_all("SELECT atm_iv FROM iv_history WHERE index_id = 'NIFTY50'")
    return [r["atm_iv"] for r in rows]


def test_uses_todays_expiry_before_close(db, monkeypatch) -> None:
    _freeze(monkeypatch, 10, 0)
    _collector(db).collect_iv_history()
    assert _atm_iv(db) == [pytest.approx(21.0)]


def test_skips_todays_expiry_after_close(db, monkeypatch) -> None:
    _freeze(monkeypatch, 15, 45)
    _collector(db).collect_iv_history()
    assert _atm_iv(db) == [pytest.approx(13.0)]


def test_only_expired_chains_writes_nothing(db, monkeypatch) -> None:
    db.execute("DELETE FROM options_chain_snapshot WHERE expiry_date >= '2026-11-06'")
    _freeze(monkeypatch, 10, 0)
    _collector(db).collect_iv_history()
    assert _atm_iv(db) == []
