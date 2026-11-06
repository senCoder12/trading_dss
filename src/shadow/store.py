"""
Storage for shadow-model decisions, in a separate SQLite file.

The shadow DB is NOT part of the production schema (no migration). It is owned
by the shadow runner, which creates it on first use. Readers (API, reports)
open it read-only.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from config.settings import settings

# Lives next to the production DB (settings.database.path / DB_PATH), so the live
# app and the shadow runner agree on it without extra configuration.
DEFAULT_SHADOW_DB = Path(settings.database.path).parent / "shadow" / "shadow_decisions.db"

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS shadow_runs (
        run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        as_of             TEXT    NOT NULL,             -- IST ISO; decision time
        snapshot_taken_at TEXT    NOT NULL,
        model_commit      TEXT    NOT NULL,
        model_branch      TEXT,
        live_commit       TEXT,
        status            TEXT    NOT NULL CHECK (status IN ('OK','PARTIAL','FAILED','SKIPPED')),
        error             TEXT,
        duration_ms       INTEGER,
        n_indices         INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS shadow_decisions (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id                INTEGER NOT NULL REFERENCES shadow_runs(run_id),
        as_of                 TEXT    NOT NULL,
        index_id              TEXT    NOT NULL,
        model_commit          TEXT    NOT NULL,
        decision              TEXT    NOT NULL CHECK (decision IN ('BUY','SELL','NEUTRAL')),
        signal_type           TEXT,                       -- BUY_CALL / BUY_PUT / NO_TRADE
        is_actionable         INTEGER NOT NULL DEFAULT 0,  -- would have been a trade (never placed)
        confidence_score      REAL,
        confidence_level      TEXT,
        weighted_score        REAL,
        regime                TEXT,
        components_json       TEXT,                        -- vote_breakdown
        reasoning             TEXT,
        warnings_json         TEXT,                        -- model + data warnings
        data_freshness_json   TEXT,
        forced_neutral        INTEGER NOT NULL DEFAULT 0,   -- 1 = stale/missing inputs overrode the model
        spot_price            REAL,
        live_signal_type      TEXT,
        live_decision         TEXT,
        live_confidence_level TEXT,
        live_generated_at     TEXT,
        agree                 INTEGER,                     -- shadow decision == live decision
        created_at            TEXT    NOT NULL,
        UNIQUE (run_id, index_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_shadow_dec_idx_asof ON shadow_decisions (index_id, as_of)",
    "CREATE INDEX IF NOT EXISTS idx_shadow_dec_asof ON shadow_decisions (as_of)",
]

DECISION_OF = {"BUY_CALL": "BUY", "BUY_PUT": "SELL", "NO_TRADE": "NEUTRAL"}


def to_decision(signal_type: Optional[str]) -> str:
    return DECISION_OF.get((signal_type or "").upper(), "NEUTRAL")


class ShadowStore:
    def __init__(self, path: Path = DEFAULT_SHADOW_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        with self.conn:
            for stmt in _DDL:
                self.conn.execute(stmt)

    def start_run(self, as_of: str, snapshot_taken_at: str, model_commit: str,
                  model_branch: Optional[str], live_commit: Optional[str]) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO shadow_runs (as_of, snapshot_taken_at, model_commit, model_branch, live_commit, status) "
                "VALUES (?, ?, ?, ?, ?, 'FAILED')",
                (as_of, snapshot_taken_at, model_commit, model_branch, live_commit),
            )
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, n: int, duration_ms: int, error: Optional[str] = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE shadow_runs SET status=?, n_indices=?, duration_ms=?, error=? WHERE run_id=?",
                (status, n, duration_ms, error, run_id),
            )

    def add_decision(self, run_id: int, row: dict[str, Any]) -> None:
        cols = [
            "as_of", "index_id", "model_commit", "decision", "signal_type", "is_actionable",
            "confidence_score", "confidence_level", "weighted_score", "regime", "components_json",
            "reasoning", "warnings_json", "data_freshness_json", "forced_neutral", "spot_price",
            "live_signal_type", "live_decision", "live_confidence_level", "live_generated_at",
            "agree", "created_at",
        ]
        vals = [row.get(c) for c in cols]
        for i, c in enumerate(cols):
            if c.endswith("_json") and not isinstance(vals[i], (str, type(None))):
                vals[i] = json.dumps(vals[i], default=str)
        with self.conn:
            self.conn.execute(
                f"INSERT OR REPLACE INTO shadow_decisions (run_id, {', '.join(cols)}) "
                f"VALUES (?, {', '.join('?' * len(cols))})",
                [run_id, *vals],
            )

    def close(self) -> None:
        self.conn.close()


def open_readonly(path: Path = DEFAULT_SHADOW_DB) -> Optional[sqlite3.Connection]:
    """Read-only connection for API/report readers; None when no shadow DB exists yet."""
    path = Path(path)
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn
