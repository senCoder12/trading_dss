"""
FII data freshness: session arithmetic, the neutrality rule, stored-data
checks, alert de-duplication, and the DataCollector job wiring.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.data.data_collector import DataCollector
from src.database.db_manager import DatabaseManager
from src.database.migrations import _v20_fii_futures_positions
from src.database import queries as Q
from src.monitoring.data_freshness import (
    FreshnessMonitor,
    check_fii_cash,
    check_fii_futures,
    is_fresh_for_decision,
    latest_published_session,
    previous_trading_day,
    sessions_behind,
)

IST = ZoneInfo("Asia/Kolkata")


def at(y, m, d, hh=10, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=IST)


# ── Session arithmetic ────────────────────────────────────────────────────────


def test_previous_trading_day_skips_weekend_and_holiday() -> None:
    assert previous_trading_day(date(2026, 11, 9)) == date(2026, 11, 6)   # Mon → Fri
    assert previous_trading_day(date(2026, 1, 27)) == date(2026, 1, 23)   # Tue → Fri (Mon 26 Jan holiday)


def test_latest_published_session_respects_cutoff() -> None:
    from src.monitoring.data_freshness import CASH_PUBLISH_CUTOFF as C

    assert latest_published_session(at(2026, 11, 5, 18, 0), C) == date(2026, 11, 4)
    assert latest_published_session(at(2026, 11, 5, 19, 45), C) == date(2026, 11, 5)
    assert latest_published_session(at(2026, 11, 7, 12, 0), C) == date(2026, 11, 6)  # Saturday


def test_sessions_behind_counts_trading_days_only() -> None:
    assert sessions_behind(date(2026, 11, 5), date(2026, 11, 5)) == 0
    assert sessions_behind(date(2026, 11, 6), date(2026, 11, 9)) == 1    # weekend not counted
    assert sessions_behind(None, date(2026, 11, 9)) is None


@pytest.mark.parametrize("latest, as_of, fresh", [
    (date(2026, 11, 6), at(2026, 11, 9, 9, 30), True),    # Mon morning, Fri data
    (date(2026, 11, 5), at(2026, 11, 9, 9, 30), False),   # Mon morning, Thu data → stale
    (date(2026, 11, 9), at(2026, 11, 9, 19, 0), True),    # same-day data after close
    (date(2026, 6, 5), at(2026, 11, 5, 10, 0), False),    # the real June→Nov gap
    (None, at(2026, 11, 9), False),
    ("2026-11-06", at(2026, 11, 9), False),               # undated/garbage → neutral
])
def test_neutrality_rule(latest, as_of, fresh) -> None:
    assert is_fresh_for_decision(latest, as_of) is fresh


# ── Stored-data checks ────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path):
    database = DatabaseManager(db_path=tmp_path / "fresh.db")
    database.connect()
    database.initialise_schema()
    yield database
    database.close()


def _cash(db, d: str) -> None:
    db.execute(Q.INSERT_FII_DII_ACTIVITY, (d, "FII", 1.0, 2.0, -1.0, "CASH"))
    db.execute(Q.INSERT_FII_DII_ACTIVITY, (d, "DII", 2.0, 1.0, 1.0, "CASH"))


def test_check_fii_cash_reports_stale_gap(db) -> None:
    _cash(db, "2026-06-05")
    st = check_fii_cash(db, at(2026, 11, 6, 10, 0))
    assert st.status == "STALE"
    assert st.latest == date(2026, 6, 5) and st.expected == date(2026, 11, 5)
    assert st.behind > 100


def test_check_fii_cash_ok_and_missing(db) -> None:
    assert check_fii_cash(db, at(2026, 11, 6)).status == "MISSING"
    _cash(db, "2026-11-05")
    assert check_fii_cash(db, at(2026, 11, 6, 10, 0)).status == "OK"


def test_check_fii_futures_missing_table_and_ok(db) -> None:
    assert check_fii_futures(db, at(2026, 11, 6)).status == "MISSING"   # pre-v20
    _v20_fii_futures_positions(db)
    db.execute(Q.UPSERT_FII_FUTURES_POSITION,
               ("2026-11-05", "FII", "INDEX_FUT", 10, 4, 6, "nse_archive", "now"))
    assert check_fii_futures(db, at(2026, 11, 6, 10, 0)).status == "OK"
    assert check_fii_futures(db, at(2026, 11, 6, 21, 30)).status == "STALE"  # today's file due


# ── Alert de-duplication ──────────────────────────────────────────────────────


def test_monitor_alerts_once_reminds_and_recovers(db) -> None:
    sent: list[tuple[str, str]] = []
    mon = FreshnessMonitor(db, lambda msg, prio: sent.append((msg, prio)))
    mon.CHECKS = (check_fii_cash,)
    _cash(db, "2026-06-05")

    t0 = at(2026, 11, 6, 10, 0)
    mon.run(t0)
    assert len(sent) == 1 and sent[0][1] == "HIGH" and "NEUTRAL" in sent[0][0]

    mon.run(t0 + timedelta(minutes=30))            # same problem → no repeat
    assert len(sent) == 1
    mon.run(t0 + timedelta(hours=6, minutes=1))    # reminder after 6 h
    assert len(sent) == 2

    _cash(db, "2026-11-05")
    mon.run(t0 + timedelta(hours=7))               # recovered → one NORMAL notice
    assert len(sent) == 3 and sent[2][1] == "NORMAL" and "current again" in sent[2][0]
    mon.run(t0 + timedelta(hours=8))
    assert len(sent) == 3


def test_monitor_survives_failing_check(db) -> None:
    sent = []
    mon = FreshnessMonitor(db, lambda m, p: sent.append(m))

    def boom(_db, _now):
        raise RuntimeError("db locked")

    mon.CHECKS = (boom, check_fii_cash)
    assert [s.dataset for s in mon.run(at(2026, 11, 6))] == ["FII cash (fii_dii_activity)"]


# ── DataCollector wiring ──────────────────────────────────────────────────────


def test_collector_job_sends_via_telegram(db) -> None:
    collector = DataCollector.__new__(DataCollector)
    collector.db = db
    collector.telegram_bot = MagicMock()
    _cash(db, "2026-06-05")

    collector.check_data_freshness()

    calls = collector.telegram_bot.send_alert.call_args_list
    assert calls and calls[0].kwargs["priority"] == "HIGH"
    assert "FII cash" in calls[0].args[0]


def test_collector_job_without_telegram_only_logs(db, caplog) -> None:
    collector = DataCollector.__new__(DataCollector)
    collector.db = db
    _cash(db, "2026-06-05")
    collector.check_data_freshness()  # must not raise
    assert any("Ops alert" in r.message for r in caplog.records)
