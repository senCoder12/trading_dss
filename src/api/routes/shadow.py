"""
Shadow-model API: read-only views of the shadow decision store.

Every payload is tagged ``mode: "SHADOW"`` and ``traded: false``. Shadow
decisions are never acted on. The live engine's decision for the same moment
is included alongside each one for comparison.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Query

from src.shadow.store import DEFAULT_SHADOW_DB, open_readonly

router = APIRouter()

_TAG = {"mode": "SHADOW", "traded": False,
        "notice": "Shadow model output — for evaluation only, never traded."}


def _row(r) -> dict:
    d = dict(r)
    for k in ("components_json", "warnings_json", "data_freshness_json"):
        try:
            d[k.replace("_json", "")] = json.loads(d.pop(k) or "null")
        except (TypeError, ValueError):
            d[k.replace("_json", "")] = None
    d["forced_neutral"] = bool(d.get("forced_neutral"))
    d["is_actionable"] = bool(d.get("is_actionable"))
    d["agree"] = None if d.get("agree") is None else bool(d["agree"])
    return d


def _conn():
    return open_readonly(DEFAULT_SHADOW_DB)


@router.get("/latest")
def latest() -> dict:
    """Most recent shadow run with one LIVE-vs-SHADOW row per index."""
    conn = _conn()
    if conn is None:
        return {**_TAG, "run": None, "decisions": []}
    try:
        run = conn.execute("SELECT * FROM shadow_runs ORDER BY run_id DESC LIMIT 1").fetchone()
        if run is None:
            return {**_TAG, "run": None, "decisions": []}
        rows = conn.execute(
            "SELECT * FROM shadow_decisions WHERE run_id = ? ORDER BY index_id", (run["run_id"],)
        ).fetchall()
        return {**_TAG, "run": dict(run), "decisions": [_row(r) for r in rows]}
    finally:
        conn.close()


@router.get("/history")
def history(index_id: Optional[str] = None, limit: int = Query(200, ge=1, le=5000)) -> dict:
    conn = _conn()
    if conn is None:
        return {**_TAG, "decisions": []}
    try:
        rows = conn.execute(
            "SELECT * FROM shadow_decisions WHERE (? IS NULL OR index_id = ?) ORDER BY as_of DESC LIMIT ?",
            (index_id, index_id, limit),
        ).fetchall()
        return {**_TAG, "decisions": [_row(r) for r in rows]}
    finally:
        conn.close()


@router.get("/summary")
def summary(date: Optional[str] = None) -> dict:
    """Agreement and directional counts, overall or for one IST date (YYYY-MM-DD)."""
    conn = _conn()
    if conn is None:
        return {**_TAG, "summary": None}
    try:
        s = conn.execute(
            "SELECT COUNT(*) AS decisions, SUM(agree IS NOT NULL) AS compared, SUM(agree) AS agreed, "
            "SUM(decision != 'NEUTRAL') AS shadow_directional, "
            "SUM(live_decision IS NOT NULL AND live_decision != 'NEUTRAL') AS live_directional, "
            "SUM(forced_neutral) AS forced_neutral, MIN(as_of) AS first_as_of, MAX(as_of) AS last_as_of "
            "FROM shadow_decisions WHERE (? IS NULL OR substr(as_of,1,10) = ?)", (date, date),
        ).fetchone()
        runs = conn.execute(
            "SELECT status, COUNT(*) AS n FROM shadow_runs WHERE (? IS NULL OR substr(as_of,1,10) = ?) "
            "GROUP BY status", (date, date),
        ).fetchall()
        return {**_TAG, "summary": dict(s), "runs": {r["status"]: r["n"] for r in runs}}
    finally:
        conn.close()
