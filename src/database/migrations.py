"""
Schema migration manager.

Migrations are plain Python functions decorated with :func:`@migration`.
They are applied in version order exactly once, with each applied version
recorded in the ``schema_version`` table.

Adding a new migration
----------------------
1. Write a function that takes a :class:`~src.database.db_manager.DatabaseManager`.
2. Decorate it with ``@migration(version=<next_int>, description="…")``.
3. On the next app startup :func:`run_migrations` picks it up automatically.

Rollback
--------
SQLite does not support ``DROP COLUMN``, so rollbacks are destructive.
:meth:`MigrationRunner.rollback` drops the **entire** schema and re-applies
all migrations up to (but not including) *target_version*.  Use with care —
only in development environments.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from src.database.db_manager import DatabaseError, DatabaseManager, get_db_manager

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# Type alias for a migration function
MigrationFn = Callable[[DatabaseManager], None]

# Registry: list of (version, description, fn) tuples — populated by @migration
_REGISTRY: list[tuple[int, str, MigrationFn]] = []


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def migration(version: int, description: str) -> Callable[[MigrationFn], MigrationFn]:
    """
    Register a migration function.

    Parameters
    ----------
    version:
        Monotonically increasing integer.  Must be unique across all
        migrations in this module.
    description:
        One-line human description stored in ``schema_version``.

    Example
    -------
    ::

        @migration(version=4, description="Add iv column to options chain")
        def v4_add_iv_column(db: DatabaseManager) -> None:
            db.execute("ALTER TABLE options_chain_snapshot ADD COLUMN iv REAL")
    """
    def decorator(fn: MigrationFn) -> MigrationFn:
        _REGISTRY.append((version, description, fn))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Migration definitions
# ---------------------------------------------------------------------------

@migration(version=1, description="Initial schema — all tables and indexes")
def _v1_initial_schema(db: DatabaseManager) -> None:
    """
    Create all tables defined in :data:`~src.database.models.TABLE_DDL`.

    Delegates to :meth:`DatabaseManager.initialise_schema` so the DDL
    lives in a single place (``models.py``).
    """
    db.initialise_schema()


@migration(version=2, description="Seed index_master from indices.json")
def _v2_seed_index_master(db: DatabaseManager) -> None:
    """
    Populate ``index_master`` from ``config/indices.json``.

    Uses INSERT … ON CONFLICT DO UPDATE so re-running this migration
    on an already-seeded database is safe (idempotent).
    """
    import json
    from datetime import datetime
    from config.settings import settings
    from src.database import queries as Q

    path = settings.indices_config_path
    if not path.exists():
        logger.warning("indices.json not found at %s — skipping seed", path)
        return

    raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(tz=_IST).isoformat()

    rows = [
        (
            entry["id"],
            entry["display_name"],
            entry.get("nse_symbol"),
            entry.get("yahoo_symbol"),
            entry["exchange"],
            entry.get("lot_size"),
            int(bool(entry.get("has_options", False))),
            entry.get("option_symbol"),
            entry.get("sector_category", "unknown"),
            int(bool(entry.get("is_active", True))),
            now,
            now,
        )
        for entry in raw
    ]
    count = db.execute_many(Q.INSERT_INDEX_MASTER, rows)
    logger.info("Seeded %d indices into index_master", len(rows))


@migration(version=3, description="Add composite index on oi_aggregated (index_id, expiry_date, timestamp)")
def _v3_oia_composite_index(db: DatabaseManager) -> None:
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_oia_id_exp_ts
        ON oi_aggregated (index_id, expiry_date, timestamp)
    """)


@migration(version=4, description="Add partial index on trading_signals for open signals")
def _v4_signals_open_index(db: DatabaseManager) -> None:
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_sig_open
        ON trading_signals (index_id, generated_at)
        WHERE outcome IS NULL OR outcome = 'OPEN'
    """)


@migration(version=5, description="Add index on news_index_impact (index_id)")
def _v5_news_impact_index_id(db: DatabaseManager) -> None:
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_nii_index_id
        ON news_index_impact (index_id)
    """)


@migration(version=6, description="Add index on anomaly_events (index_id, timestamp)")
def _v6_anomaly_index(db: DatabaseManager) -> None:
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_anomaly_id_ts
        ON anomaly_events (index_id, timestamp)
    """)


@migration(version=7, description="Add index on fii_dii_activity (date)")
def _v7_fii_dii_date_index(db: DatabaseManager) -> None:
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_fii_dii_date
        ON fii_dii_activity (date)
    """)


@migration(version=8, description="Expand anomaly_events category CHECK to include FII_DII and DIVERGENCE")
def _v8_anomaly_category_expand(db: DatabaseManager) -> None:
    # SQLite cannot ALTER CHECK constraints — rebuild the table.
    db.execute("ALTER TABLE anomaly_events RENAME TO _anomaly_events_old")
    db.execute("""
        CREATE TABLE anomaly_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id     TEXT    NOT NULL REFERENCES index_master(id),
            timestamp    TEXT    NOT NULL,
            anomaly_type TEXT    NOT NULL,
            severity     TEXT    NOT NULL DEFAULT 'MEDIUM'
                         CHECK (severity IN ('HIGH','MEDIUM','LOW')),
            category     TEXT    NOT NULL DEFAULT 'OTHER'
                         CHECK (category IN ('VOLUME','PRICE','OI','FII_DII','DIVERGENCE','OTHER')),
            details      TEXT    NOT NULL DEFAULT '{}',
            message      TEXT    NOT NULL DEFAULT '',
            is_active    INTEGER NOT NULL DEFAULT 1
        )
    """)
    db.execute("""
        INSERT INTO anomaly_events (id, index_id, timestamp, anomaly_type,
                                    severity, category, details, message, is_active)
        SELECT id, index_id, timestamp, anomaly_type,
               severity,
               CASE WHEN category IN ('VOLUME','PRICE','OI','FII_DII','DIVERGENCE','OTHER')
                    THEN category ELSE 'OTHER' END,
               details, message, is_active
        FROM _anomaly_events_old
    """)
    db.execute("DROP TABLE _anomaly_events_old")
    db.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_id_ts       ON anomaly_events (index_id, timestamp)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_type_active ON anomaly_events (anomaly_type, is_active)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_cat_active  ON anomaly_events (category, is_active)")


@migration(version=9, description="Fix anomaly_events schema: add cooldown_key, expand severity CHECK, ensure all columns")
def _v9_fix_anomaly_events_schema(db: DatabaseManager) -> None:
    """Ensure anomaly_events has all required columns and no restrictive CHECK on anomaly_type.

    Bug: Direct schema modifications caused drift. This migration ensures
    the table matches the expected schema regardless of current state.
    """
    # Check current columns
    current_cols = {row['name'] for row in db.fetch_all("PRAGMA table_info(anomaly_events)")}

    added: list[str] = []
    if 'category' not in current_cols:
        db.execute("ALTER TABLE anomaly_events ADD COLUMN category TEXT DEFAULT 'UNKNOWN'")
        added.append('category')
    if 'message' not in current_cols:
        db.execute("ALTER TABLE anomaly_events ADD COLUMN message TEXT DEFAULT ''")
        added.append('message')
    if 'cooldown_key' not in current_cols:
        db.execute("ALTER TABLE anomaly_events ADD COLUMN cooldown_key TEXT DEFAULT ''")
        added.append('cooldown_key')

    # Test if there's a restrictive CHECK constraint on anomaly_type
    # by attempting an insert with a non-standard type
    needs_rebuild = False
    try:
        db.execute(
            """INSERT INTO anomaly_events
               (index_id, timestamp, anomaly_type, severity, details, is_active)
               VALUES ('__TEST__', datetime('now'), 'EXTENSIBILITY_TEST', 'LOW', '{}', 0)"""
        )
        db.execute("DELETE FROM anomaly_events WHERE index_id = '__TEST__'")
    except Exception:
        needs_rebuild = True

    # Also check if severity CHECK is missing EXTREME
    if not needs_rebuild:
        try:
            db.execute(
                """INSERT INTO anomaly_events
                   (index_id, timestamp, anomaly_type, severity, details, is_active)
                   VALUES ('__TEST__', datetime('now'), 'TEST', 'EXTREME', '{}', 0)"""
            )
            db.execute("DELETE FROM anomaly_events WHERE index_id = '__TEST__'")
        except Exception:
            needs_rebuild = True

    if needs_rebuild:
        # Rebuild table without restrictive CHECK constraint on anomaly_type
        # and with expanded severity CHECK
        db.execute("ALTER TABLE anomaly_events RENAME TO _anomaly_events_backup")
        db.execute("""
            CREATE TABLE anomaly_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                index_id     TEXT    NOT NULL,
                timestamp    TEXT    NOT NULL,
                anomaly_type TEXT    NOT NULL,
                severity     TEXT    NOT NULL DEFAULT 'MEDIUM'
                             CHECK (severity IN ('LOW','MEDIUM','HIGH','EXTREME')),
                category     TEXT    NOT NULL DEFAULT 'OTHER'
                             CHECK (category IN ('VOLUME','PRICE','OI','FII_DII','DIVERGENCE','OTHER')),
                details      TEXT    NOT NULL DEFAULT '{}',
                message      TEXT    NOT NULL DEFAULT '',
                cooldown_key TEXT    NOT NULL DEFAULT '',
                is_active    INTEGER NOT NULL DEFAULT 1
            )
        """)
        # Migrate data — handle column mismatch gracefully
        backup_cols = {row['name'] for row in db.fetch_all("PRAGMA table_info(_anomaly_events_backup)")}
        common_cols = backup_cols & {
            'id', 'index_id', 'timestamp', 'anomaly_type', 'severity',
            'details', 'is_active', 'category', 'message', 'cooldown_key',
        }
        cols_str = ', '.join(sorted(common_cols))
        db.execute(f"INSERT INTO anomaly_events ({cols_str}) SELECT {cols_str} FROM _anomaly_events_backup")
        db.execute("DROP TABLE _anomaly_events_backup")
        added.append('TABLE_REBUILT')

    # Ensure indices exist
    db.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_active    ON anomaly_events(is_active, index_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_events(timestamp)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_type      ON anomaly_events(anomaly_type)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_id_ts       ON anomaly_events (index_id, timestamp)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_type_active ON anomaly_events (anomaly_type, is_active)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_cat_active  ON anomaly_events (category, is_active)")

    if added:
        logger.info(f"Migration v9: anomaly_events schema fixed — {', '.join(added)}")


@migration(version=10, description="Add audit_json to trading_signals, expand system_health status for kill switch")
def _v10_audit_json_and_kill_switch(db: DatabaseManager) -> None:
    """Add structured audit trail column and expand system_health for kill switch.

    - trading_signals.audit_json: stores exact intermediate values for signal replay.
    - system_health.status CHECK: expanded to include ACTIVE / INACTIVE for kill switch.
    """
    changes: list[str] = []

    # -- trading_signals.audit_json --
    ts_cols = {row["name"] for row in db.fetch_all("PRAGMA table_info(trading_signals)")}
    if "audit_json" not in ts_cols:
        db.execute("ALTER TABLE trading_signals ADD COLUMN audit_json TEXT DEFAULT NULL")
        changes.append("trading_signals.audit_json")

    # -- Expand system_health status CHECK to include ACTIVE / INACTIVE --
    # Test if the current CHECK constraint allows 'ACTIVE'
    needs_rebuild = False
    try:
        db.execute(
            """INSERT INTO system_health (timestamp, component, status, message)
               VALUES (datetime('now'), '__kill_switch_test__', 'ACTIVE', 'test')"""
        )
        db.execute("DELETE FROM system_health WHERE component = '__kill_switch_test__'")
    except Exception:
        needs_rebuild = True

    if needs_rebuild:
        db.execute("ALTER TABLE system_health RENAME TO _system_health_backup")
        db.execute("""
            CREATE TABLE system_health (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                component        TEXT    NOT NULL,
                status           TEXT    NOT NULL DEFAULT 'OK'
                                 CHECK (status IN ('OK','WARNING','ERROR','ACTIVE','INACTIVE')),
                message          TEXT,
                response_time_ms INTEGER
            )
        """)
        backup_cols = {row["name"] for row in db.fetch_all("PRAGMA table_info(_system_health_backup)")}
        common_cols = backup_cols & {"id", "timestamp", "component", "status", "message", "response_time_ms"}
        cols_str = ", ".join(sorted(common_cols))
        db.execute(f"INSERT INTO system_health ({cols_str}) SELECT {cols_str} FROM _system_health_backup")
        db.execute("DROP TABLE _system_health_backup")
        changes.append("system_health.status CHECK expanded")

    if changes:
        logger.info("Migration v10: %s", ", ".join(changes))


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------

class MigrationRunner:
    """
    Applies pending migrations in version order.

    Parameters
    ----------
    db:
        A connected :class:`~src.database.db_manager.DatabaseManager`.
        The ``schema_version`` table must exist before calling
        :meth:`run_pending` — it is created by :meth:`~DatabaseManager.initialise_schema`.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_applied_versions(self) -> set[int]:
        """Return the set of migration versions already recorded in the DB."""
        rows = self._db.fetch_all("SELECT version FROM schema_version")
        return {row["version"] for row in rows}

    def _record_migration(self, version: int, description: str) -> None:
        """Write an applied-migration record."""
        from src.database import queries as Q

        now = datetime.now(tz=_IST).isoformat()
        self._db.execute(Q.INSERT_SCHEMA_VERSION, (version, now, description))

    # ── Public interface ──────────────────────────────────────────────────────

    def run_pending(self) -> int:
        """
        Apply all registered migrations not yet recorded in ``schema_version``.

        Migrations run inside individual transactions.  If one fails, it is
        rolled back but previously-applied migrations in the same call are
        **not** rolled back (they are already committed).

        Returns
        -------
        int:
            Number of migrations applied during this call.
        """
        applied = self._get_applied_versions()
        pending = sorted(
            ((v, d, fn) for v, d, fn in _REGISTRY if v not in applied),
            key=lambda t: t[0],
        )

        if not pending:
            logger.debug("No pending migrations — schema is up to date")
            return 0

        count = 0
        for version, description, fn in pending:
            logger.info("Applying migration v%d: %s", version, description)
            try:
                fn(self._db)
                self._record_migration(version, description)
                count += 1
                logger.info("Migration v%d applied successfully", version)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Migration v%d FAILED — %s — rolling back this migration only",
                    version, exc,
                )
                raise DatabaseError(
                    f"Migration v{version} failed: {exc}"
                ) from exc

        logger.info("Applied %d migration(s); schema now at v%d", count, pending[-1][0])
        return count

    def current_version(self) -> int:
        """
        Return the highest migration version recorded in the database.

        Returns
        -------
        int:
            0 if no migrations have been applied yet.
        """
        versions = self._get_applied_versions()
        return max(versions) if versions else 0

    def rollback(self, target_version: int = 0) -> None:
        """
        Destructive rollback to *target_version*.

        Drops **all tables** in the database and re-applies all migrations
        up to and including *target_version*.

        .. warning::
            This permanently deletes all data.  Use only in development.

        Parameters
        ----------
        target_version:
            Re-apply migrations with ``version <= target_version``.
            Pass ``0`` to reset to an empty schema.
        """
        logger.warning(
            "ROLLBACK requested to v%d — ALL DATA WILL BE LOST", target_version
        )
        conn = self._db._conn()
        # Drop every user table
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        with conn:
            for row in tables:
                conn.execute(f"DROP TABLE IF EXISTS {row['name']}")
        logger.warning("All tables dropped")

        # Re-create schema_version first, then re-apply up to target_version
        from src.database.models import TABLE_DDL
        with conn:
            for stmt in TABLE_DDL["schema_version"]:
                conn.execute(stmt)

        to_apply = sorted(
            ((v, d, fn) for v, d, fn in _REGISTRY if v <= target_version),
            key=lambda t: t[0],
        )
        for version, description, fn in to_apply:
            logger.info("Re-applying v%d: %s", version, description)
            fn(self._db)
            self._record_migration(version, description)

        logger.info(
            "Rollback complete — schema is now at v%d",
            target_version if to_apply else 0,
        )

    def status(self) -> list[dict]:
        """
        Return a status report of all registered migrations.

        Returns
        -------
        list[dict]:
            Each entry has ``version``, ``description``, ``applied``,
            and ``applied_at`` (or ``None`` if pending).
        """
        from src.database import queries as Q

        applied_rows = {
            row["version"]: row
            for row in self._db.fetch_all(Q.LIST_ALL_MIGRATIONS)
        }
        report = []
        for version, description, _ in sorted(_REGISTRY, key=lambda t: t[0]):
            applied_info = applied_rows.get(version)
            report.append({
                "version": version,
                "description": description,
                "applied": applied_info is not None,
                "applied_at": applied_info["applied_at"] if applied_info else None,
            })
        return report


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def run_migrations(db: DatabaseManager | None = None) -> int:
    """
    Connect to the DB (if needed), initialise the schema_version table,
    and apply all pending migrations.

    This is the one-liner called from app startup scripts::

        from src.database.migrations import run_migrations
        run_migrations()

    Parameters
    ----------
    db:
        Optional :class:`DatabaseManager` to use.  Defaults to the
        process singleton from :func:`~src.database.db_manager.get_db_manager`.

    Returns
    -------
    int:
        Number of migrations applied.
    """
    if db is None:
        db = get_db_manager()

    # Ensure we're connected and the schema_version table exists
    db.connect()

    # schema_version must exist before MigrationRunner queries it
    from src.database.models import TABLE_DDL
    conn = db._conn()
    with conn:
        for stmt in TABLE_DDL["schema_version"]:
            conn.execute(stmt)

    runner = MigrationRunner(db)
    applied = runner.run_pending()
    return applied
