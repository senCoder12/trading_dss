"""
Shadow mode: the candidate model runs against a snapshot, never the production
DB; results land in a separate store with the live decision alongside;
stale/missing inputs force NEUTRAL with explicit warnings; the API is
read-only and tagged SHADOW.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.shadow.runner import ShadowRunner
from src.shadow.store import ShadowStore, open_readonly, to_decision

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parent.parent
AS_OF = datetime(2026, 11, 6, 10, 0, tzinfo=IST)


def _digest(path: Path) -> str:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = []
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        rows.append((t, conn.execute(f'SELECT * FROM "{t}" ORDER BY 1').fetchall()))
    conn.close()
    return hashlib.sha256(repr(rows).encode()).hexdigest()


@pytest.fixture()
def prod_db(tmp_path) -> Path:
    p = tmp_path / "prod.db"
    c = sqlite3.connect(p)
    c.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE trading_signals (id INTEGER PRIMARY KEY, index_id TEXT, generated_at TEXT,
            signal_type TEXT, confidence_level TEXT);
        CREATE TABLE price_data (index_id TEXT, timestamp TEXT, close REAL, timeframe TEXT);
        CREATE TABLE options_chain_snapshot (index_id TEXT, timestamp TEXT);
        CREATE TABLE fii_dii_activity (date TEXT, category TEXT, segment TEXT);
    """)
    c.execute("INSERT INTO trading_signals (index_id, generated_at, signal_type, confidence_level) "
              "VALUES ('NIFTY50', ?, 'NO_TRADE', 'LOW')", ((AS_OF - timedelta(minutes=2)).isoformat(),))
    c.execute("INSERT INTO trading_signals (index_id, generated_at, signal_type, confidence_level) "
              "VALUES ('BANKNIFTY', ?, 'BUY_PUT', 'MEDIUM')", ((AS_OF - timedelta(minutes=3)).isoformat(),))
    c.execute("INSERT INTO price_data VALUES ('NIFTY50', ?, 24500.0, '1m')", ((AS_OF - timedelta(minutes=1)).isoformat(),))
    c.execute("INSERT INTO price_data VALUES ('BANKNIFTY', ?, 52000.0, '1m')", ((AS_OF - timedelta(minutes=45)).isoformat(),))
    c.execute("INSERT INTO options_chain_snapshot VALUES ('NIFTY50', ?)", ((AS_OF - timedelta(minutes=3)).isoformat(),))
    c.commit()
    c.close()
    return p


@pytest.fixture()
def fake_model(tmp_path) -> Path:
    """A minimal 'candidate worktree' whose engine also WRITES to its DB (to prove isolation)."""
    root = tmp_path / "candidate"
    files = {
        "config/__init__.py": "",
        "config/settings.py": """
            import os
            from types import SimpleNamespace
            settings = SimpleNamespace(database=SimpleNamespace(path=os.environ['DB_PATH']),
                                       trading=SimpleNamespace(restrict_to_major_indices=True))
        """,
        "src/__init__.py": "",
        "src/database/__init__.py": "",
        "src/database/db_manager.py": """
            import os, sqlite3
            class DatabaseManager:
                def connect(self):
                    self.conn = sqlite3.connect(os.environ['DB_PATH'])
        """,
        "src/engine/__init__.py": "",
        "src/engine/decision_engine.py": """
            from types import SimpleNamespace
            SIGNALS = {'NIFTY50': 'BUY_CALL', 'BANKNIFTY': 'BUY_CALL'}
            class DecisionEngine:
                def __init__(self, db):
                    self.db = db
                    self.registry = SimpleNamespace(get_trading_universe=lambda major_only: [
                        SimpleNamespace(id='NIFTY50'), SimpleNamespace(id='BANKNIFTY')])
                def run_full_cycle(self, index_id, as_of=None):
                    if index_id == 'FINNIFTY':   # model short-circuits: no vote_breakdown
                        sig = SimpleNamespace(signal_type='NO_TRADE', confidence_score=0.0, confidence_level='LOW',
                            weighted_score=1.5, vote_breakdown={}, reasoning='NO_TRADE: Model not tradeable', warnings=[])
                        tech = SimpleNamespace(unavailable_components=[], volume_available=True,
                            votes={'trend': 'BULLISH', 'momentum': 'NEUTRAL'}, overall_signal='BUY',
                            smart_money=SimpleNamespace(score=42.0))
                        return SimpleNamespace(index_id=index_id, signal=sig, is_actionable=False, warnings=[],
                            regime=SimpleNamespace(regime='TREND_UP', warnings=[]), technical_result=tech)
                    # side effect like the real engine: write a signal row
                    self.db.conn.execute("INSERT INTO trading_signals (index_id, generated_at, signal_type, confidence_level) VALUES (?, ?, 'SHADOW_WRITE', 'X')", (index_id, as_of.isoformat()))
                    self.db.conn.commit()
                    sig = SimpleNamespace(signal_type=SIGNALS[index_id], confidence_score=0.61,
                        confidence_level='MEDIUM', weighted_score=0.42,
                        vote_breakdown={'trend': {'weighted_contribution': 0.3}, 'momentum': {'weighted_contribution': 0.12}},
                        reasoning='Trend and momentum aligned', warnings=['model warning'])
                    return SimpleNamespace(index_id=index_id, signal=sig, is_actionable=True,
                        warnings=[], regime=SimpleNamespace(regime='TREND_UP', warnings=[]),
                        technical_result=SimpleNamespace(unavailable_components=['volume'], volume_available=False))
        """,
    }
    for rel, body in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(textwrap.dedent(body))
    return root


def _runner(prod_db, fake_model, tmp_path):
    store = ShadowStore(tmp_path / "shadow" / "shadow.db")
    return ShadowRunner(prod_db=prod_db, model_root=fake_model, store=store,
                        work_dir=tmp_path / "shadow" / "work", model_branch="feat/test",
                        python=sys.executable, indices=["NIFTY50", "BANKNIFTY"]), store


def test_decision_mapping() -> None:
    assert to_decision("BUY_CALL") == "BUY"
    assert to_decision("BUY_PUT") == "SELL"
    assert to_decision("NO_TRADE") == "NEUTRAL"
    assert to_decision(None) == "NEUTRAL"


def test_shadow_run_never_touches_production(prod_db, fake_model, tmp_path) -> None:
    before = _digest(prod_db)
    runner, store = _runner(prod_db, fake_model, tmp_path)
    run_id = runner.run_once(as_of=AS_OF, market_open=True)
    assert _digest(prod_db) == before, "production DB changed during a shadow run"

    snap = sqlite3.connect(tmp_path / "shadow" / "work" / "snapshot.db")
    assert snap.execute("SELECT COUNT(*) FROM trading_signals WHERE signal_type='SHADOW_WRITE'").fetchone()[0] == 2

    rows = {r["index_id"]: r for r in store.conn.execute(
        "SELECT * FROM shadow_decisions WHERE run_id = ?", (run_id,))}
    run = store.conn.execute("SELECT * FROM shadow_runs WHERE run_id = ?", (run_id,)).fetchone()
    assert run["status"] == "OK" and run["model_branch"] == "feat/test"

    n = rows["NIFTY50"]
    assert n["decision"] == "BUY" and n["forced_neutral"] == 0
    assert n["live_decision"] == "NEUTRAL" and n["agree"] == 0
    assert n["confidence_score"] == pytest.approx(0.61) and n["regime"] == "TREND_UP"
    assert json.loads(n["components_json"])["trend"]["weighted_contribution"] == 0.3
    warnings = json.loads(n["warnings_json"])
    assert "model warning" in warnings
    assert any("Unavailable components (neutral): volume" in w for w in warnings)
    assert n["spot_price"] == 24500.0


def test_stale_price_forces_neutral_with_warning(prod_db, fake_model, tmp_path) -> None:
    runner, store = _runner(prod_db, fake_model, tmp_path)
    run_id = runner.run_once(as_of=AS_OF, market_open=True)
    b = store.conn.execute("SELECT * FROM shadow_decisions WHERE run_id=? AND index_id='BANKNIFTY'",
                           (run_id,)).fetchone()
    assert b["signal_type"] == "BUY_CALL"          # what the model said …
    assert b["decision"] == "NEUTRAL"              # … overridden: price 45 min old
    assert b["forced_neutral"] == 1 and b["is_actionable"] == 0
    assert any("Price data stale" in w for w in json.loads(b["warnings_json"]))
    assert b["live_decision"] == "SELL" and b["agree"] == 0


def test_market_closed_forces_neutral(prod_db, fake_model, tmp_path) -> None:
    runner, store = _runner(prod_db, fake_model, tmp_path)
    run_id = runner.run_once(as_of=AS_OF, market_open=False)
    rows = store.conn.execute("SELECT decision, forced_neutral FROM shadow_decisions WHERE run_id=?",
                              (run_id,)).fetchall()
    assert {tuple(r) for r in rows} == {("NEUTRAL", 1)}


def test_worker_crash_is_recorded_not_raised(prod_db, tmp_path) -> None:
    broken = tmp_path / "broken"
    (broken / "src").mkdir(parents=True)
    runner, store = _runner(prod_db, broken, tmp_path)
    run_id = runner.run_once(as_of=AS_OF, market_open=True)
    run = store.conn.execute("SELECT status, error FROM shadow_runs WHERE run_id=?", (run_id,)).fetchone()
    assert run["status"] == "FAILED" and run["error"]


def test_refuses_live_code_as_model_root(prod_db, tmp_path) -> None:
    with pytest.raises(ValueError):
        ShadowRunner(prod_db=prod_db, model_root=ROOT, store=ShadowStore(tmp_path / "s.db"),
                     work_dir=tmp_path / "w")


def test_api_is_readonly_and_tagged(prod_db, fake_model, tmp_path, monkeypatch) -> None:
    import src.api.routes.shadow as route

    runner, store = _runner(prod_db, fake_model, tmp_path)
    runner.run_once(as_of=AS_OF, market_open=True)
    monkeypatch.setattr(route, "DEFAULT_SHADOW_DB", store.path)

    latest = route.latest()
    assert latest["mode"] == "SHADOW" and latest["traded"] is False
    assert {d["index_id"] for d in latest["decisions"]} == {"NIFTY50", "BANKNIFTY"}
    assert isinstance(latest["decisions"][0]["components"], dict)
    s = route.summary(date="2026-11-06")
    assert s["summary"]["decisions"] == 2 and s["runs"] == {"OK": 1}

    ro = open_readonly(store.path)
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("DELETE FROM shadow_decisions")


def test_live_code_does_not_import_shadow() -> None:
    for path in list((ROOT / "src" / "engine").rglob("*.py")) + [
        ROOT / "src" / "data" / "data_collector.py",
        ROOT / "src" / "analysis" / "technical_aggregator.py",
        ROOT / "src" / "paper_trading" / "paper_engine.py",
        ROOT / "scripts" / "run_system.py",
    ]:
        assert "src.shadow" not in path.read_text(), f"{path} imports shadow code"


def test_short_circuit_keeps_component_votes(prod_db, fake_model, tmp_path) -> None:
    store = ShadowStore(tmp_path / "shadow" / "shadow.db")
    runner = ShadowRunner(prod_db=prod_db, model_root=fake_model, store=store,
                          work_dir=tmp_path / "shadow" / "work", python=sys.executable, indices=["FINNIFTY"])
    run_id = runner.run_once(as_of=AS_OF, market_open=True)
    r = store.conn.execute("SELECT * FROM shadow_decisions WHERE run_id=?", (run_id,)).fetchone()
    comps = json.loads(r["components_json"])
    assert comps == {"trend": {"vote": "BULLISH"}, "momentum": {"vote": "NEUTRAL"}, "smart_money": {"score": 42.0}}
    assert r["decision"] == "NEUTRAL" and r["weighted_score"] == 1.5
