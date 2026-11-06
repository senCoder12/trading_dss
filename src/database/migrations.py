"""
Schema migration manager.

Migrations are plain Python functions decorated with :func:`@migration`.
They are applied in version order exactly once, with each applied version
recorded in the ``schema_version`` table.

Adding a new migration
----------------------
1. Write a function that takes a :class:`~src.database.db_manager.DatabaseManager`.
2. Decorate it with ``@migration(version=<next_int>, description="…")``.
3. Add its inverse to :data:`ROLLBACK_SQL`.
4. Deploy explicitly. Startup never migrates an existing database:
   ``scripts/migrate.py rehearse --approve N`` validates on a copy of the
   database, then ``scripts/migrate.py apply --approve N`` backs up and applies.

Approval gate
-------------
A *fresh* database (no migrations recorded yet) is built and migrated
automatically. There is nothing to protect. On an *existing* database,
:meth:`MigrationRunner.run_pending` raises :class:`PendingMigrationsError`
unless every pending version is passed in ``approve``. An app restart can
therefore never change the production schema by surprise.

Rollback
--------
Per-migration: :func:`rollback_migration` runs the inverse statements in
:data:`ROLLBACK_SQL` and removes the ``schema_version`` row. The migration
then shows as pending again and needs re-approval.
:meth:`MigrationRunner.rollback` is the old *destructive* full reset, for
development only.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Iterable, Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from src.database.db_manager import DatabaseError, DatabaseManager, get_db_manager

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# Type alias for a migration function
MigrationFn = Callable[[DatabaseManager], None]

# Registry: list of (version, description, fn) tuples — populated by @migration
_REGISTRY: list[tuple[int, str, MigrationFn]] = []

# Inverse statements per migration (only additive migrations can be undone
# without data loss). Used by rollback_migration() and scripts/migrate.py.
ROLLBACK_SQL: dict[int, list[str]] = {
    19: ["DROP TABLE IF EXISTS futures_depth_1m"],
    20: [
        "DROP INDEX IF EXISTS idx_fii_fut_pos_lookup",
        "DROP TABLE IF EXISTS fii_futures_positions",
    ],
}


class PendingMigrationsError(DatabaseError):
    """Existing database has migrations that were not explicitly approved."""

    def __init__(self, pending: list[tuple[int, str]]) -> None:
        self.pending = pending
        listed = "\n".join(f"  v{v}: {d}" for v, d in pending)
        versions = ",".join(str(v) for v, _ in pending)
        super().__init__(
            "Database has unapplied migrations that need explicit approval:\n"
            f"{listed}\n"
            "Schema changes are never applied on startup. Review them, then run:\n"
            f"  .venv/bin/python scripts/migrate.py rehearse --approve {versions}\n"
            f"  .venv/bin/python scripts/migrate.py apply --approve {versions}"
        )


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


@migration(version=11, description="Add overnight_data table for pre-market intelligence snapshots")
def _v11_overnight_data(db: DatabaseManager) -> None:
    """Create the overnight_data table.

    Stores one compiled snapshot per trading day from the overnight news /
    pre-market pipeline. The Decision Engine loads the latest row at market
    open to seed its news vote and regime expectation.
    """
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS overnight_data (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date           TEXT    NOT NULL UNIQUE,
            compiled_at          TEXT    NOT NULL,
            is_complete          INTEGER NOT NULL DEFAULT 0,
            overall_sentiment    REAL,
            gap_direction        TEXT,
            gap_confidence       REAL,
            regime_expectation   TEXT,
            us_impact            TEXT,
            fii_bias             TEXT,
            payload_json         TEXT    NOT NULL
        )
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_overnight_date ON overnight_data (trade_date)"
    )


@migration(version=12, description="Add missing columns to options_chain_snapshot (oi_change_pct, ltp_change, ltp_change_pct, bid_qty, ask_qty)")
def _v12_options_chain_columns(db: DatabaseManager) -> None:
    cols_to_add = [
        ("oi_change_pct",  "REAL    NOT NULL DEFAULT 0"),
        ("ltp_change",     "REAL    NOT NULL DEFAULT 0"),
        ("ltp_change_pct", "REAL    NOT NULL DEFAULT 0"),
        ("bid_qty",        "INTEGER NOT NULL DEFAULT 0"),
        ("ask_qty",        "INTEGER NOT NULL DEFAULT 0"),
    ]
    existing = {row["name"] for row in db.fetch_all("PRAGMA table_info(options_chain_snapshot)", ())}
    for col, col_def in cols_to_add:
        if col not in existing:
            db.execute(f"ALTER TABLE options_chain_snapshot ADD COLUMN {col} {col_def}")


@migration(version=13, description="Add fo_universe and fo_universe_history tables")
def _v13_fo_universe_tables(db: DatabaseManager) -> None:
    from src.database.models import TABLE_DDL

    conn = db._conn()
    with conn:
        for table_name in ["fo_universe", "fo_universe_history"]:
            for stmt in TABLE_DDL[table_name]:
                conn.execute(stmt)


@migration(version=14, description="Add fo_avg_volumes table for cached 20-day volumes")
def _v14_fo_avg_volumes_table(db: DatabaseManager) -> None:
    from src.database.models import TABLE_DDL

    conn = db._conn()
    with conn:
        for stmt in TABLE_DDL["fo_avg_volumes"]:
            conn.execute(stmt)


@migration(version=15, description="Add fo_universe_snapshots table for daily universe orchestration")
def _v15_fo_universe_snapshots_table(db: DatabaseManager) -> None:
    from src.database.models import TABLE_DDL

    conn = db._conn()
    with conn:
        for stmt in TABLE_DDL["fo_universe_snapshots"]:
            conn.execute(stmt)


@migration(version=16, description="Add candle_cache table for liquidity module CandleStore")
def _v16_candle_cache_table(db: DatabaseManager) -> None:
    from src.database import queries as Q

    db.execute(Q.CREATE_CANDLE_CACHE)
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_candle_cache_symbol_tf "
        "ON candle_cache (symbol, timeframe, timestamp)"
    )


@migration(version=18, description="Add GEX tables: gex_snapshots, gex_strike_data, gex_wall_breaks")
def _v18_gex_tables(db: DatabaseManager) -> None:
    from src.database.models import TABLE_DDL

    conn = db._conn()
    with conn:
        for table_name in ["gex_snapshots", "gex_strike_data", "gex_wall_breaks"]:
            for stmt in TABLE_DDL[table_name]:
                conn.execute(stmt)


@migration(version=19, description="Add futures_depth_1m table for order-book depth features")
def _v19_futures_depth_table(db: DatabaseManager) -> None:
    from src.database.models import TABLE_DDL

    conn = db._conn()
    with conn:
        for stmt in TABLE_DDL["futures_depth_1m"]:
            conn.execute(stmt)


# Approved 2026-11-06. Purely additive: one table plus one index; no existing table altered.
@migration(version=20, description="Add fii_futures_positions table (participant-wise futures OI)")
def _v20_fii_futures_positions(db: DatabaseManager) -> None:
    from src.database import queries as Q

    conn = db._conn()
    with conn:
        conn.execute(Q.CREATE_FII_FUTURES_POSITIONS)
        conn.execute(Q.CREATE_FII_FUTURES_POSITIONS_IDX)


@migration(version=17, description="Add liquidity_levels and level_touches tables for SMC pool detector")
def _v17_liquidity_tables(db: DatabaseManager) -> None:
    from src.database import queries as Q

    db.execute(Q.CREATE_LIQUIDITY_LEVELS)
    db.execute(Q.CREATE_LIQUIDITY_LEVELS_IDX)
    db.execute(Q.CREATE_LEVEL_TOUCHES)
    db.execute(Q.CREATE_LEVEL_TOUCHES_IDX)


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

    def pending(self) -> list[tuple[int, str]]:
        """Registered migrations not yet recorded in ``schema_version``."""
        applied = self._get_applied_versions()
        return sorted((v, d) for v, d, _ in _REGISTRY if v not in applied)

    def run_pending(self, approve: Optional[Iterable[int]] = None) -> int:
        """
        Apply all registered migrations not yet recorded in ``schema_version``.

        On a fresh database (no migrations recorded) everything is applied.
        On an existing database every pending version must be listed in
        *approve*. Otherwise :class:`PendingMigrationsError` is raised and
        nothing is changed.

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

        if applied:  # existing database: never migrate implicitly
            approved = set(approve or ())
            unapproved = [(v, d) for v, d, _ in pending if v not in approved]
            if unapproved:
                raise PendingMigrationsError(unapproved)

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


def rollback_migration(db: DatabaseManager, version: int) -> None:
    """Undo one additive migration: run its ROLLBACK_SQL, drop its version row.

    The migration stays registered, so it reappears as pending and must be
    re-approved. A restart will not silently re-apply it.
    """
    stmts = ROLLBACK_SQL.get(version)
    if stmts is None:
        raise DatabaseError(f"No rollback defined for migration v{version}")
    conn = db._conn()
    with conn:
        for stmt in stmts:
            conn.execute(stmt)
        conn.execute("DELETE FROM schema_version WHERE version = ?", (version,))
    logger.warning("Migration v%d rolled back", version)


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
