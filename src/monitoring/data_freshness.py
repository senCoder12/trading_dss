"""
FII data freshness, measured in trading sessions rather than wall-clock time.

Two jobs:

1. **Monitoring.** How many sessions behind the latest stored FII cash
   (fii_dii_activity, CASH) and FII futures (fii_futures_positions) data are,
   relative to the latest session NSE should have published by now.
   FreshnessMonitor turns that into deduplicated Telegram alerts.

2. **Neutrality rule.** FII data may contribute *direction* only if it comes
   from the latest session knowable at decision time, i.e. the previous
   trading session or later. Anything older is treated as absent: callers
   return NEUTRAL instead of a bias computed from stale numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from src.utils.market_hours import is_trading_day

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# NSE publication times, with margin: provisional FII/DII cash figures appear
# in the evening; the participant-wise OI archive file somewhat later.
CASH_PUBLISH_CUTOFF = time(19, 30)
FUTURES_PUBLISH_CUTOFF = time(21, 0)

REALERT_AFTER = timedelta(hours=6)


# ── Session arithmetic ────────────────────────────────────────────────────────


def previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def latest_published_session(now: datetime, cutoff: time) -> date:
    """Most recent session whose end-of-day data should be published by *now*."""
    now = now.astimezone(_IST)
    d = now.date()
    if is_trading_day(d) and now.time() >= cutoff:
        return d
    return previous_trading_day(d)


def sessions_behind(latest: Optional[date], expected: date) -> Optional[int]:
    """Trading sessions in (latest, expected]; None when there is no data at all."""
    if latest is None:
        return None
    n, d = 0, latest + timedelta(days=1)
    while d <= expected:
        if is_trading_day(d):
            n += 1
        d += timedelta(days=1)
    return n


def is_fresh_for_decision(latest: Optional[date], as_of: Optional[datetime] = None) -> bool:
    """FII data is usable directionally only if it is from the previous session or later."""
    if isinstance(latest, datetime):
        latest = latest.date()
    if not isinstance(latest, date):
        return False  # missing or undated data never carries direction
    as_of = (as_of or datetime.now(tz=_IST)).astimezone(_IST)
    return latest >= previous_trading_day(as_of.date())


# ── Stored-data checks ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FreshnessStatus:
    dataset: str
    latest: Optional[date]
    expected: date
    behind: Optional[int]

    @property
    def status(self) -> str:
        if self.latest is None:
            return "MISSING"
        return "OK" if self.behind == 0 else "STALE"


def _as_date(v) -> Optional[date]:
    return date.fromisoformat(str(v)[:10]) if v else None


def latest_fii_cash_date(db) -> Optional[date]:
    row = db.fetch_one(
        "SELECT MAX(date) AS d FROM fii_dii_activity WHERE category = 'FII' AND segment = 'CASH'"
    )
    return _as_date(row["d"] if row else None)


def latest_fii_futures_date(db) -> Optional[date]:
    if not db.table_exists("fii_futures_positions"):
        return None
    row = db.fetch_one(
        "SELECT MAX(trade_date) AS d FROM fii_futures_positions "
        "WHERE participant = 'FII' AND instrument = 'INDEX_FUT'"
    )
    return _as_date(row["d"] if row else None)


def check_fii_cash(db, now: Optional[datetime] = None) -> FreshnessStatus:
    now = now or datetime.now(tz=_IST)
    expected = latest_published_session(now, CASH_PUBLISH_CUTOFF)
    latest = latest_fii_cash_date(db)
    return FreshnessStatus("FII cash (fii_dii_activity)", latest, expected, sessions_behind(latest, expected))


def check_fii_futures(db, now: Optional[datetime] = None) -> FreshnessStatus:
    now = now or datetime.now(tz=_IST)
    expected = latest_published_session(now, FUTURES_PUBLISH_CUTOFF)
    latest = latest_fii_futures_date(db)
    return FreshnessStatus("FII futures (fii_futures_positions)", latest, expected, sessions_behind(latest, expected))


# ── Alerting ──────────────────────────────────────────────────────────────────


class FreshnessMonitor:
    """Turns freshness checks into deduplicated alerts.

    Alerts on the first stale/missing observation, again at most every
    REALERT_AFTER while the problem persists or whenever the expected session
    moves on, and once on recovery.
    """

    CHECKS: tuple[Callable, ...] = (check_fii_cash, check_fii_futures)

    def __init__(self, db, send: Callable[[str, str], object]) -> None:
        self.db = db
        self.send = send  # (message, priority) -> anything
        self._alerted: dict[str, tuple[date, datetime]] = {}  # dataset → (expected, sent_at)

    def run(self, now: Optional[datetime] = None) -> list[FreshnessStatus]:
        now = now or datetime.now(tz=_IST)
        results = []
        for check in self.CHECKS:
            try:
                st = check(self.db, now)
            except Exception:
                logger.exception("Freshness check %s failed", getattr(check, "__name__", check))
                continue
            results.append(st)
            self._maybe_alert(st, now)
        return results

    def _maybe_alert(self, st: FreshnessStatus, now: datetime) -> None:
        prev = self._alerted.get(st.dataset)
        if st.status == "OK":
            if prev is not None:
                self.send(f"✅ {st.dataset} is current again (latest session {st.latest}).", "NORMAL")
                del self._alerted[st.dataset]
            return
        if prev is not None and prev[0] == st.expected and now - prev[1] < REALERT_AFTER:
            return
        if st.status == "MISSING":
            detail = "no data stored at all"
        else:
            detail = f"latest {st.latest}, expected {st.expected} ({st.behind} session(s) behind)"
        self.send(
            f"⚠️ DATA STALE — {st.dataset}: {detail}. "
            f"FII inputs are treated as NEUTRAL until this is fixed.",
            "HIGH",
        )
        logger.warning("Freshness: %s %s — %s", st.dataset, st.status, detail)
        self._alerted[st.dataset] = (st.expected, now)
