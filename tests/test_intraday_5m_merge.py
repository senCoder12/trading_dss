"""Live intraday series: stored 5m history + 5m bars built from live 1m bars."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.engine.decision_engine import merge_intraday_5m

_IST = ZoneInfo("Asia/Kolkata")


def _m1(ts: datetime, o: float, h: float, l: float, c: float) -> dict:
    return {"timestamp": ts.isoformat(), "open": o, "high": h, "low": l, "close": c, "volume": 0}


def _desc(rows):
    return sorted(rows, key=lambda r: r["timestamp"], reverse=True)


def test_builds_closed_5m_bars_from_1m_and_drops_forming_bar():
    start = datetime(2026, 11, 6, 9, 15, tzinfo=_IST)
    ones = [_m1(start + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(12)]
    now = start + timedelta(minutes=12)          # 09:27 → 09:25 bucket still forming
    out = merge_intraday_5m([], _desc(ones), now, 100)
    assert [r["timestamp"] for r in out] == [
        (start + timedelta(minutes=5)).isoformat(), start.isoformat(),
    ]
    first = out[-1]                               # 09:15 bar (oldest is last: newest-first order)
    assert first["open"] == 100 and first["close"] == 104.5
    assert first["high"] == 105 and first["low"] == 99


def test_post_market_1m_rows_ignored():
    late = datetime(2026, 11, 5, 22, 50, tzinfo=_IST)
    out = merge_intraday_5m([], _desc([_m1(late, 1, 2, 0.5, 1.5)]), late + timedelta(hours=1), 100)
    assert out == []


def test_appends_to_stored_history_without_overlap():
    stored = [{"timestamp": (datetime(2026, 11, 5, 15, 25, tzinfo=_IST) - timedelta(minutes=5 * i)).isoformat(),
               "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0} for i in range(3)]
    start = datetime(2026, 11, 6, 9, 15, tzinfo=_IST)
    ones = [_m1(start + timedelta(minutes=i), 2, 2, 2, 2) for i in range(10)]
    out = merge_intraday_5m(_desc(stored), _desc(ones), start + timedelta(minutes=10), 100)
    ts = [r["timestamp"] for r in out]
    assert ts == sorted(ts, reverse=True)         # newest first, contiguous
    assert ts[0] == (start + timedelta(minutes=5)).isoformat()
    assert ts[2] == "2026-11-05T15:25:00+05:30"
    assert len(out) == 5


def test_no_1m_returns_stored_and_limit_applies():
    stored = [{"timestamp": f"2026-11-05T10:{m:02d}:00+05:30", "close": 1} for m in range(55, 0, -5)]
    assert merge_intraday_5m(stored, [], datetime(2026, 11, 6, tzinfo=_IST), 3) == stored[:3]
