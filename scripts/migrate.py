"""
Explicit, reviewed schema migrations for an existing database.

Startup never migrates an existing database. Use this tool instead:

    status    list applied / pending migrations
    rehearse  copy the DB, apply on the copy, validate, roll back, re-apply
    apply     back up the DB, apply the approved migrations, validate
    rollback  back up the DB, undo one additive migration (ROLLBACK_SQL)

--approve must list every version being applied or rolled back. Nothing is
applied unless it is named explicitly.

Examples:
    .venv/bin/python scripts/migrate.py status
    .venv/bin/python scripts/migrate.py rehearse --approve 20
    .venv/bin/python scripts/migrate.py apply    --approve 20
    .venv/bin/python scripts/migrate.py rollback --version 20 --approve 20
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import IST_TIMEZONE  # noqa: E402
from config.settings import settings  # noqa: E402
from src.database.db_manager import DatabaseManager  # noqa: E402
from src.database.migrations import (  # noqa: E402
    ROLLBACK_SQL,
    MigrationRunner,
    rollback_migration,
)

_IST = ZoneInfo(IST_TIMEZONE)


# ── helpers ───────────────────────────────────────────────────────────────────


def _stamp() -> str:
    return datetime.now(tz=_IST).strftime("%Y%m%d_%H%M%S")


def backup(src: Path, dest: Path) -> Path:
    """Consistent online copy via SQLite's backup API (safe while the app runs)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(src)) as s, sqlite3.connect(str(dest)) as d:
        s.backup(d)
    return dest


def _open(path: Path) -> DatabaseManager:
    db = DatabaseManager(db_path=path)
    db.connect()
    return db


def snapshot(db: DatabaseManager, counts: bool = True, full_check: bool = True) -> dict:
    conn = db._conn()
    objects = {
        (r["type"], r["name"]): r["sql"]
        for r in conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    tables = sorted(n for (t, n) in objects if t == "table")
    row_counts = (
        {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
        if counts else {}
    )
    check = conn.execute("PRAGMA integrity_check" if full_check else "PRAGMA quick_check").fetchone()[0]
    _, issues = db.validate_schema()
    return {"objects": objects, "counts": row_counts, "check": check, "issues": set(issues)}


def _pending_or_fail(db: DatabaseManager, approve: set[int]) -> list[tuple[int, str]]:
    pending = MigrationRunner(db).pending()
    pending_v = {v for v, _ in pending}
    if not pending:
        raise SystemExit("Nothing pending — schema is up to date.")
    if pending_v != approve:
        raise SystemExit(
            f"--approve {sorted(approve)} does not match pending {sorted(pending_v)}. "
            "Approve exactly the pending versions after reviewing them."
        )
    return pending


class Report:
    def __init__(self) -> None:
        self.ok = True

    def check(self, cond: bool, label: str, detail: str = "") -> None:
        self.ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


def _compare_after(
    rep: Report, before: dict, after: dict, compare_counts: bool, expected_new_versions: int = 0
) -> set:
    rep.check(after["check"] == "ok", "integrity check after migration", after["check"])
    changed = [k for k in before["objects"] if after["objects"].get(k) != before["objects"][k]]
    rep.check(not changed, "no existing table/index altered or dropped", ", ".join(map(str, changed)))
    if compare_counts:
        diffs = {t: (n, after["counts"].get(t)) for t, n in before["counts"].items()
                 if t != "schema_version" and after["counts"].get(t) != n}
        rep.check(not diffs, "row counts of existing tables unchanged", str(diffs) if diffs else "")
        added = after["counts"]["schema_version"] - before["counts"]["schema_version"]
        rep.check(added == expected_new_versions, "schema_version gained exactly the approved versions",
                  f"+{added}")
    new_issues = after["issues"] - before["issues"]
    rep.check(not new_issues, "validate_schema: no new issues", "; ".join(sorted(new_issues)))
    resolved = before["issues"] - after["issues"]
    if resolved:
        print(f"  [INFO] validate_schema issues resolved: {'; '.join(sorted(resolved))}")
    return set(after["objects"]) - set(before["objects"])


# ── commands ──────────────────────────────────────────────────────────────────


def cmd_status(db_path: Path) -> int:
    db = _open(db_path)
    for m in MigrationRunner(db).status():
        state = f"applied {m['applied_at']}" if m["applied"] else "PENDING"
        print(f"  v{m['version']:>3}  {state:<45} {m['description']}")
    return 0


def cmd_rehearse(db_path: Path, approve: set[int], work_dir: Path) -> int:
    copy = backup(db_path, work_dir / f"rehearsal_v{'_'.join(map(str, sorted(approve)))}_{_stamp()}.db")
    print(f"Rehearsal copy: {copy}")
    db = _open(copy)
    pending = _pending_or_fail(db, approve)
    print("Pending:", ", ".join(f"v{v} ({d})" for v, d in pending))

    rep = Report()
    before = snapshot(db)
    rep.check(before["check"] == "ok", "integrity check of copy before migration", before["check"])

    MigrationRunner(db).run_pending(approve=approve)
    after = snapshot(db)
    new = _compare_after(rep, before, after, compare_counts=True, expected_new_versions=len(approve))
    print(f"  [INFO] new objects: {sorted(n for _, n in new)}")
    rep.check(not MigrationRunner(db).pending(), "no migrations left pending")

    for v in sorted(approve, reverse=True):
        rollback_migration(db, v)
    rolled = snapshot(db)
    rep.check(rolled["objects"] == before["objects"], "rollback restores the exact original schema")
    rep.check(rolled["counts"] == before["counts"], "rollback leaves all row counts unchanged")
    rep.check({v for v, _ in MigrationRunner(db).pending()} == approve,
              "after rollback the migrations are pending again (re-approval required)")

    MigrationRunner(db).run_pending(approve=approve)
    rep.check(snapshot(db, counts=False)["objects"] == after["objects"], "re-apply after rollback is identical")

    print(f"\nREHEARSAL {'PASSED' if rep.ok else 'FAILED'} — copy kept at {copy} (migrated state)")
    return 0 if rep.ok else 1


def cmd_apply(db_path: Path, approve: set[int], backup_dir: Path) -> int:
    db = _open(db_path)
    pending = _pending_or_fail(db, approve)
    print("Applying:", ", ".join(f"v{v} ({d})" for v, d in pending))
    missing_rb = [v for v in approve if v not in ROLLBACK_SQL]
    if missing_rb:
        raise SystemExit(f"Refusing: no ROLLBACK_SQL for {missing_rb}")

    rep = Report()
    # Live DB may be written concurrently → compare schema and integrity, not row counts
    before = snapshot(db, counts=False, full_check=False)
    bak = backup(db_path, backup_dir / f"{db_path.stem}.pre_v{'_'.join(map(str, sorted(approve)))}_{_stamp()}.db")
    bdb = _open(bak)
    bsnap = snapshot(bdb, counts=False)
    rep.check(bsnap["check"] == "ok", f"backup integrity check ({bak})", bsnap["check"])
    rep.check(bsnap["objects"] == before["objects"], "backup schema identical to production")
    bdb.close()
    if not rep.ok:
        print("Backup verification failed — NOT applying.")
        return 1

    MigrationRunner(db).run_pending(approve=approve)
    after = snapshot(db, counts=False, full_check=False)
    new = _compare_after(rep, before, after, compare_counts=False)
    print(f"  [INFO] new objects: {sorted(n for _, n in new)}")
    rep.check(not MigrationRunner(db).pending(), "no migrations left pending")

    print(f"\nAPPLY {'SUCCEEDED' if rep.ok else 'HAD FAILURES'}. Backup: {bak}")
    print("Rollback (keeps all other data):")
    for v in sorted(approve, reverse=True):
        print(f"  .venv/bin/python scripts/migrate.py rollback --version {v} --approve {v}")
    print(f"Full restore (last resort, stop the app first): copy {bak} over {db_path} and delete its -wal/-shm files")
    return 0 if rep.ok else 1


def cmd_rollback(db_path: Path, version: int, approve: set[int], backup_dir: Path) -> int:
    if approve != {version}:
        raise SystemExit("--approve must equal --version for a rollback")
    db = _open(db_path)
    applied = {m["version"] for m in MigrationRunner(db).status() if m["applied"]}
    if version not in applied:
        raise SystemExit(f"v{version} is not applied")
    bak = backup(db_path, backup_dir / f"{db_path.stem}.pre_rollback_v{version}_{_stamp()}.db")
    print(f"Backup: {bak}")
    rollback_migration(db, version)
    print(f"v{version} rolled back. It is now pending and will NOT be re-applied on restart without approval.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "rehearse", "apply", "rollback"])
    ap.add_argument("--db", type=Path, default=None, help="SQLite path (default: settings.database.path)")
    ap.add_argument("--approve", default="", help="comma-separated versions, e.g. 20 or 20,21")
    ap.add_argument("--version", type=int, help="rollback: version to undo")
    ap.add_argument("--work-dir", type=Path, default=None, help="rehearsal copies (default: <db dir>/rehearsal)")
    ap.add_argument("--backup-dir", type=Path, default=None, help="backups (default: <db dir>/backups)")
    args = ap.parse_args()

    db_path = (args.db or settings.database.path).resolve()
    approve = {int(v) for v in args.approve.split(",") if v.strip()}
    if args.command == "status":
        return cmd_status(db_path)
    if not approve:
        ap.error("--approve is required")
    if args.command == "rehearse":
        return cmd_rehearse(db_path, approve, args.work_dir or db_path.parent / "rehearsal")
    backup_dir = args.backup_dir or db_path.parent / "backups"
    if args.command == "apply":
        return cmd_apply(db_path, approve, backup_dir)
    if args.version is None:
        ap.error("rollback needs --version")
    return cmd_rollback(db_path, args.version, approve, backup_dir)


if __name__ == "__main__":
    sys.exit(main())
