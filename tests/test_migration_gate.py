"""
Migration approval gate: an existing database is never migrated implicitly
(no auto-apply on restart, no table creation via initialise_schema). A fresh
database is still built automatically. Also covers per-migration rollback and
the scripts/migrate.py rehearse/apply/rollback flow.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import src.database.db_manager as dbm
import src.database.migrations as mig
from src.database.db_manager import DatabaseManager
from src.database.migrations import MigrationRunner, PendingMigrationsError, rollback_migration

ROOT = Path(__file__).resolve().parent.parent


def _fresh_migrated(path: Path) -> DatabaseManager:
    db = DatabaseManager(db_path=path)
    db.connect()
    db.initialise_schema()
    MigrationRunner(db).run_pending()  # fresh → applies everything, no approval needed
    return db


def _add_migration(monkeypatch, version: int, table: str) -> None:
    def fn(db):
        with db._conn() as conn:
            conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")

    monkeypatch.setattr(mig, "_REGISTRY", list(mig._REGISTRY) + [(version, f"add {table}", fn)])
    monkeypatch.setitem(mig.ROLLBACK_SQL, version, [f"DROP TABLE IF EXISTS {table}"])
    from src.database import models  # a real migration also declares its table here
    monkeypatch.setitem(models.EXPECTED_SCHEMA, table, {"id": "INTEGER"})


def test_fresh_database_migrates_automatically(tmp_path) -> None:
    db = _fresh_migrated(tmp_path / "fresh.db")
    assert MigrationRunner(db).pending() == []
    assert db.table_exists("fii_futures_positions")  # v20 included for fresh DBs
    db.close()


def test_existing_database_refuses_unapproved_migration(tmp_path, monkeypatch) -> None:
    db = _fresh_migrated(tmp_path / "prod.db")
    _add_migration(monkeypatch, 999, "new_feature")

    with pytest.raises(PendingMigrationsError) as exc:
        MigrationRunner(db).run_pending()
    assert "v999" in str(exc.value) and "migrate.py apply --approve 999" in str(exc.value)
    assert not db.table_exists("new_feature")

    with pytest.raises(PendingMigrationsError):  # approving a different version is not enough
        MigrationRunner(db).run_pending(approve={998})
    assert not db.table_exists("new_feature")

    assert MigrationRunner(db).run_pending(approve={999}) == 1
    assert db.table_exists("new_feature")
    db.close()


def test_initialise_schema_is_noop_on_existing_database(tmp_path, monkeypatch) -> None:
    db = _fresh_migrated(tmp_path / "prod.db")
    monkeypatch.setitem(dbm.TABLE_DDL, "sneaky", ["CREATE TABLE IF NOT EXISTS sneaky (id INTEGER)"])
    monkeypatch.setattr(dbm, "TABLE_CREATION_ORDER", list(dbm.TABLE_CREATION_ORDER) + ["sneaky"])

    db.initialise_schema()

    assert not db.table_exists("sneaky")
    db.close()


def test_rollback_migration_requires_reapproval(tmp_path) -> None:
    db = _fresh_migrated(tmp_path / "prod.db")
    rollback_migration(db, 20)

    assert not db.table_exists("fii_futures_positions")
    assert [v for v, _ in MigrationRunner(db).pending()] == [20]
    with pytest.raises(PendingMigrationsError):  # a restart cannot silently re-create it
        MigrationRunner(db).run_pending()
    assert MigrationRunner(db).run_pending(approve={20}) == 1
    assert db.table_exists("fii_futures_positions")
    db.close()


def test_every_registered_migration_from_v19_has_rollback_sql() -> None:
    for version, _, _ in mig._REGISTRY:
        if version >= 19:
            assert version in mig.ROLLBACK_SQL, f"v{version} has no ROLLBACK_SQL"


# ── scripts/migrate.py end-to-end on a temp "production" DB ───────────────────


@pytest.fixture()
def migrate_cli():
    spec = importlib.util.spec_from_file_location("migrate_cli", ROOT / "scripts" / "migrate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["migrate_cli"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("migrate_cli", None)


def test_rehearse_apply_rollback_flow(tmp_path, monkeypatch, migrate_cli, capsys) -> None:
    prod = tmp_path / "prod.db"
    db = _fresh_migrated(prod)
    db.execute("INSERT INTO vix_data (timestamp, vix_value, vix_change, vix_change_pct) "
               "VALUES ('2026-11-05T15:30:00+05:30', 12.5, 0.1, 0.8)")
    db.close()
    _add_migration(monkeypatch, 999, "new_feature")

    # rehearse works on a copy; production is untouched
    assert migrate_cli.cmd_rehearse(prod, {999}, tmp_path / "rehearsal") == 0
    out = capsys.readouterr().out
    assert "REHEARSAL PASSED" in out and "[FAIL]" not in out
    check = DatabaseManager(db_path=prod)
    check.connect()
    assert not check.table_exists("new_feature")
    check.close()

    # approve must match exactly what is pending
    with pytest.raises(SystemExit):
        migrate_cli.cmd_apply(prod, {998}, tmp_path / "backups")

    assert migrate_cli.cmd_apply(prod, {999}, tmp_path / "backups") == 0
    out = capsys.readouterr().out
    assert "APPLY SUCCEEDED" in out
    backups = list((tmp_path / "backups").glob("prod.pre_v999_*.db"))
    assert len(backups) == 1
    after = DatabaseManager(db_path=prod)
    after.connect()
    assert after.table_exists("new_feature")
    assert after.fetch_one("SELECT COUNT(*) AS n FROM vix_data")["n"] == 1
    after.close()

    assert migrate_cli.cmd_rollback(prod, 999, {999}, tmp_path / "backups") == 0
    final = DatabaseManager(db_path=prod)
    final.connect()
    assert not final.table_exists("new_feature")
    assert [v for v, _ in MigrationRunner(final).pending()] == [999]
    final.close()
