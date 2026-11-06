"""
ShadowRunner: one shadow cycle = snapshot → live decisions + freshness → candidate model → store.

Isolation guarantees:
* the production DB is only *read* here (by SQLite's online backup API);
* the candidate model runs in a subprocess whose DB_PATH is the snapshot copy;
* outputs go to the separate shadow DB.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from src.shadow.store import ShadowStore, to_decision

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")
WORKER = Path(__file__).with_name("worker.py")

# Staleness thresholds while the market is open (minutes)
PRICE_STALE_MIN = 10        # price stale → decision forced NEUTRAL
CHAIN_STALE_MIN = 15        # options chain stale → warning (model gates on it too)
DEPTH_STALE_MIN = 10        # futures depth stale → warning
LIVE_MATCH_WINDOW_MIN = 15  # live decision must be this recent to be compared


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except Exception:
        return None


def _age_min(ts: Optional[str], as_of: datetime) -> Optional[float]:
    if not ts:
        return None
    t = datetime.fromisoformat(str(ts))
    if t.tzinfo is None:
        t = t.replace(tzinfo=_IST)
    return round((as_of - t).total_seconds() / 60.0, 1)


class ShadowRunner:
    def __init__(
        self,
        prod_db: Path,
        model_root: Path,
        store: ShadowStore,
        work_dir: Path,
        model_branch: Optional[str] = None,
        live_root: Optional[Path] = None,
        python: str = sys.executable,
        indices: Optional[list[str]] = None,
        worker_timeout_s: int = 900,
    ) -> None:
        self.prod_db = Path(prod_db).resolve()
        self.model_root = Path(model_root).resolve()
        self.store = store
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.model_branch = model_branch
        self.model_commit = _git(self.model_root, "rev-parse", "--short", "HEAD") or "unknown"
        self.live_commit = _git(live_root, "rev-parse", "--short", "HEAD") if live_root else None
        self.python = python
        self.indices = indices or []
        self.worker_timeout_s = worker_timeout_s
        if self.model_root == Path(__file__).resolve().parents[2]:
            raise ValueError("model_root must be the candidate model's worktree, not the live code")

    # ── steps ────────────────────────────────────────────────────────────────

    def snapshot(self) -> Path:
        """Consistent copy of the production DB (safe while the live app writes)."""
        final = self.work_dir / "snapshot.db"
        tmp = self.work_dir / "snapshot.tmp.db"
        for p in (tmp, Path(f"{tmp}-wal"), Path(f"{tmp}-shm")):
            p.unlink(missing_ok=True)
        src = sqlite3.connect(f"file:{self.prod_db}?mode=ro", uri=True)
        dst = sqlite3.connect(str(tmp))
        try:
            src.backup(dst)
            # The copy inherits WAL mode; a read-only reader cannot create -wal/-shm, so switch.
            dst.execute("PRAGMA journal_mode=DELETE")
        finally:
            dst.close()
            src.close()
        for p in (final, Path(f"{final}-wal"), Path(f"{final}-shm")):
            p.unlink(missing_ok=True)
        tmp.rename(final)
        return final

    @staticmethod
    def live_decisions(conn: sqlite3.Connection, as_of: datetime) -> dict[str, dict]:
        since = (as_of - timedelta(minutes=LIVE_MATCH_WINDOW_MIN)).isoformat()
        rows = conn.execute(
            "SELECT index_id, signal_type, confidence_level, generated_at FROM trading_signals "
            "WHERE generated_at > ? AND generated_at <= ? ORDER BY generated_at DESC",
            (since, as_of.isoformat()),
        ).fetchall()
        out: dict[str, dict] = {}
        for r in rows:
            out.setdefault(r[0], {"signal_type": r[1], "confidence_level": r[2], "generated_at": r[3]})
        return out

    @staticmethod
    def freshness(conn: sqlite3.Connection, index_id: str, as_of: datetime, market_open: bool) -> dict:
        """Input freshness for one index. Stale price → forced NEUTRAL (requirement: neutral + explicit warning)."""
        f: dict = {"warnings": [], "forced_neutral": False}
        row = conn.execute(
            "SELECT timestamp, close FROM price_data WHERE index_id = ? AND timeframe IN ('1m','5m') "
            "AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1", (index_id, as_of.isoformat()),
        ).fetchone()
        f["price_ts"], f["spot"] = (row[0], row[1]) if row else (None, None)
        f["price_age_min"] = _age_min(f["price_ts"], as_of)
        chain = conn.execute(
            "SELECT MAX(timestamp) FROM options_chain_snapshot WHERE index_id = ? AND timestamp <= ?",
            (index_id, as_of.isoformat()),
        ).fetchone()[0]
        f["chain_age_min"] = _age_min(chain, as_of)
        has_depth = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='futures_depth_1m'").fetchone()
        depth = conn.execute(
            "SELECT MAX(timestamp) FROM futures_depth_1m WHERE index_id = ? AND timestamp <= ?",
            (index_id, as_of.isoformat()),
        ).fetchone()[0] if has_depth else None
        f["depth_age_min"] = _age_min(depth, as_of)

        if not market_open:
            f["forced_neutral"] = True
            f["warnings"].append("Market closed: inputs are from the last session, so the decision is forced NEUTRAL")
        if f["price_age_min"] is None:
            f["forced_neutral"] = True
            f["warnings"].append("No intraday price data: decision forced NEUTRAL")
        elif market_open and f["price_age_min"] > PRICE_STALE_MIN:
            f["forced_neutral"] = True
            f["warnings"].append(f"Price data stale ({f['price_age_min']} min): decision forced NEUTRAL")
        if market_open and (f["chain_age_min"] is None or f["chain_age_min"] > CHAIN_STALE_MIN):
            f["warnings"].append(f"Options chain stale/missing (age {f['chain_age_min']} min): options inputs neutral")
        if market_open and (f["depth_age_min"] is None or f["depth_age_min"] > DEPTH_STALE_MIN):
            f["warnings"].append(f"Futures depth stale/missing (age {f['depth_age_min']} min)")
        return f

    @staticmethod
    def fii_freshness(db_path: Path, as_of: datetime) -> list[str]:
        class _ReadOnlyDB:
            """Just enough of DatabaseManager for the freshness helpers; never writes."""

            def __init__(self, path: Path) -> None:
                self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                self.conn.row_factory = sqlite3.Row

            def fetch_one(self, sql: str, params: tuple = ()):
                return self.conn.execute(sql, params).fetchone()

            def table_exists(self, name: str) -> bool:
                return self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

            def close(self) -> None:
                self.conn.close()

        try:
            from src.monitoring.data_freshness import (
                is_fresh_for_decision, latest_fii_cash_date, latest_fii_futures_date,
            )
            db = _ReadOnlyDB(db_path)
            out = []
            cash, fut = latest_fii_cash_date(db), latest_fii_futures_date(db)
            if not is_fresh_for_decision(cash, as_of):
                out.append(f"FII cash data stale (latest {cash}): FII inputs neutral")
            if not is_fresh_for_decision(fut, as_of):
                out.append(f"FII futures data stale (latest {fut}): FII inputs neutral")
            db.close()
            return out
        except Exception as exc:
            return [f"FII freshness check failed: {exc}"]

    def run_model(self, snapshot: Path, as_of: datetime) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".json", dir=self.work_dir, delete=False) as fh:
            out_path = fh.name
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ANGEL", "TELEGRAM", "BROKER"))}
        env.update({"PYTHONPATH": str(self.model_root), "DB_PATH": str(snapshot),
                    "USE_BROKER_API": "false", "TELEGRAM_ENABLED": "false"})
        code = ("import runpy, sys; sys.argv = ['shadow_worker', %r, %r, %r]; "
                "runpy.run_path(%r, run_name='__main__')") % (
            out_path, as_of.isoformat(), ",".join(self.indices), str(WORKER))
        proc = subprocess.run([self.python, "-c", code], cwd=str(self.model_root), env=env,
                              capture_output=True, text=True, timeout=self.worker_timeout_s)
        try:
            with open(out_path) as fh:
                payload = json.load(fh)
        except Exception:
            payload = {"fatal": f"worker produced no output (rc={proc.returncode})",
                       "stderr": proc.stderr[-3000:]}
        finally:
            Path(out_path).unlink(missing_ok=True)
        payload["returncode"] = proc.returncode
        return payload

    # ── one cycle ────────────────────────────────────────────────────────────

    def run_once(self, as_of: Optional[datetime] = None, market_open: Optional[bool] = None) -> int:
        from src.utils.market_hours import is_market_open

        t0 = time.monotonic()
        snap = self.snapshot()
        snap_at = datetime.now(tz=_IST)
        as_of = as_of or snap_at
        market_open = is_market_open(as_of) if market_open is None else market_open
        run_id = self.store.start_run(as_of.isoformat(), snap_at.isoformat(), self.model_commit,
                                      self.model_branch, self.live_commit)
        try:
            with sqlite3.connect(f"file:{snap}?mode=ro", uri=True) as conn:
                live = self.live_decisions(conn, as_of)
                fresh = {}
                for idx in (self.indices or sorted(live) or []):
                    fresh[idx] = self.freshness(conn, idx, as_of, market_open)
            fii_warn = self.fii_freshness(snap, as_of)

            payload = self.run_model(snap, as_of)
            if payload.get("fatal"):
                raise RuntimeError(f"{payload['fatal']}\n{payload.get('traceback') or payload.get('stderr', '')}")

            n_ok = 0
            with sqlite3.connect(f"file:{snap}?mode=ro", uri=True) as conn:
                for r in payload["results"]:
                    idx = r["index_id"]
                    f = fresh.get(idx) or self.freshness(conn, idx, as_of, market_open)
                    lv = live.get(idx, {})
                    warnings = []
                    if r.get("error"):
                        warnings.append(f"Model error: {r['error']}")
                    warnings += r.get("result_warnings", []) + r.get("signal_warnings", []) \
                        + r.get("regime_warnings", []) + f["warnings"] + fii_warn
                    if r.get("unavailable_components"):
                        warnings.append(f"Unavailable components (neutral): {', '.join(r['unavailable_components'])}")
                    model_decision = to_decision(r.get("signal_type"))
                    decision = "NEUTRAL" if (f["forced_neutral"] or r.get("error")) else model_decision
                    live_decision = to_decision(lv.get("signal_type")) if lv else None
                    self.store.add_decision(run_id, {
                        "as_of": as_of.isoformat(), "index_id": idx, "model_commit": self.model_commit,
                        "decision": decision, "signal_type": r.get("signal_type"),
                        "is_actionable": int(bool(r.get("is_actionable")) and decision != "NEUTRAL"),
                        "confidence_score": r.get("confidence_score"),
                        "confidence_level": r.get("confidence_level"),
                        "weighted_score": r.get("weighted_score"), "regime": r.get("regime"),
                        "components_json": r.get("vote_breakdown") or {},
                        "reasoning": r.get("reasoning") or (r.get("traceback") or ""),
                        "warnings_json": list(dict.fromkeys(w for w in warnings if w)),
                        "data_freshness_json": {k: v for k, v in f.items() if k != "warnings"},
                        "forced_neutral": int(f["forced_neutral"]), "spot_price": f.get("spot"),
                        "live_signal_type": lv.get("signal_type"), "live_decision": live_decision,
                        "live_confidence_level": lv.get("confidence_level"),
                        "live_generated_at": lv.get("generated_at"),
                        "agree": None if live_decision is None else int(live_decision == decision),
                        "created_at": datetime.now(tz=_IST).isoformat(),
                    })
                    n_ok += 0 if r.get("error") else 1
            n = len(payload["results"])
            status = "OK" if n and n_ok == n else ("PARTIAL" if n_ok else "FAILED")
            self.store.finish_run(run_id, status, n, int((time.monotonic() - t0) * 1000))
            logger.info("Shadow run %d (%s @ %s): %s, %d/%d indices", run_id, self.model_commit,
                        as_of.isoformat(timespec="seconds"), status, n_ok, n)
        except Exception as exc:
            self.store.finish_run(run_id, "FAILED", 0, int((time.monotonic() - t0) * 1000), str(exc)[:4000])
            logger.exception("Shadow run %d failed", run_id)
        return run_id
