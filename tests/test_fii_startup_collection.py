"""
FII/DII cash collection must run once on every start, even when the scheduler
starts seconds after the jobs were registered (a next_run_time set at
add_job() time is silently dropped as a misfire in that case).
"""

from __future__ import annotations

import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.data.data_collector import DataCollector, _IST


def _collector(force_start: bool) -> tuple[DataCollector, list[str]]:
    calls: list[str] = []
    collector = DataCollector.__new__(DataCollector)  # no scrapers / broker
    collector.force_start = force_start
    collector.scheduler = BackgroundScheduler(timezone=_IST)
    for job_id in ("collect_fii_dii", "update_historical", "collect_technical_indicators", "decision_cycle"):
        collector.scheduler.add_job(
            (lambda j=job_id: calls.append(j)),
            trigger=CronTrigger(hour=18, minute=0, timezone=_IST),
            id=job_id,
        )
    return collector, calls


def _wait(cond, timeout=3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def test_fii_runs_on_start_despite_registration_gap() -> None:
    collector, calls = _collector(force_start=False)
    time.sleep(1.5)  # components initialise between DataCollector() and start()
    collector.start()
    try:
        assert _wait(lambda: "collect_fii_dii" in calls)
        assert calls.count("collect_fii_dii") == 1
        assert "update_historical" not in calls  # only FII is forced on a normal start
    finally:
        collector.scheduler.shutdown(wait=False)


def test_force_start_runs_fii_exactly_once() -> None:
    collector, calls = _collector(force_start=True)
    collector.start()
    try:
        time.sleep(0.5)
        assert calls.count("collect_fii_dii") == 1
        assert "update_historical" in calls
    finally:
        collector.scheduler.shutdown(wait=False)


def test_fii_job_keeps_18h_schedule_after_startup_run() -> None:
    collector, calls = _collector(force_start=False)
    collector.start()
    try:
        assert _wait(lambda: "collect_fii_dii" in calls)
        nxt = collector.scheduler.get_job("collect_fii_dii").next_run_time
        assert (nxt.hour, nxt.minute) == (18, 0)
    finally:
        collector.scheduler.shutdown(wait=False)
