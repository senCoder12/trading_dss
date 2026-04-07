"""
SQLite database manager — singleton, thread-safe, WAL-mode.

Provides synchronous helpers used directly by data-collection scripts
and async wrappers (via ``asyncio.run_in_executor``) used by FastAPI.

Thread safety
-------------
SQLite connections are **not** safe to share across threads.  This module
uses a ``threading.local`` store so that each thread transparently gets its
own connection to the same database file.  The WAL journal mode allows one
writer and many readers to proceed concurrently without locking.

Usage
-----
Acquire the singleton via :func:`get_db_manager` or use the class directly::

    with DatabaseManager() as db:
        db.execute(Q.INSERT_PRICE_DATA, (params...))

    # or as a singleton
    db = DatabaseManager.instance()
    db.connect()
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.database.models import TABLE_DDL, TABLE_CREATION_ORDER

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DatabaseError(Exception):
    """Base exception for all database layer failures."""


class ConnectionError(DatabaseError):  # noqa: A001  (shadows built-in only in this module)
    """Raised when a database connection cannot be established."""


class QueryError(DatabaseError):
    """Raised when a SQL query fails."""


# ---------------------------------------------------------------------------
# Singleton manager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """
    Thread-safe SQLite connection manager with WAL mode.

    Each thread gets its own ``sqlite3.Connection`` stored in
    ``threading.local``.  A single ``DatabaseManager`` instance is shared
    across the application via :meth:`instance`.

    Parameters
    ----------
    db_path:
        Absolute path to the ``.db`` file.  Created (along with any
        parent directories) if it does not exist.
    wal_mode:
        Enable WAL journal mode.  Defaults to ``settings.database.wal_mode``.
    timeout:
        SQLite busy-timeout in seconds.
    """

    _instance: Optional["DatabaseManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        db_path: Optional[Path] = None,
        wal_mode: Optional[bool] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self._db_path: Path = Path(db_path) if db_path is not None else settings.database.path
        self._wal_mode: bool = wal_mode if wal_mode is not None else settings.database.wal_mode
        self._timeout: int = timeout or settings.database.timeout_seconds
        self._local: threading.local = threading.local()

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def instance(
        cls,
        db_path: Optional[Path] = None,
        wal_mode: Optional[bool] = None,
    ) -> "DatabaseManager":
        """
        Return the process-wide singleton, creating it on first call.

        Parameters
        ----------
        db_path:
            Only used on the very first call; ignored on subsequent calls.
        wal_mode:
            Only used on the very first call.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path=db_path, wal_mode=wal_mode)
        return cls._instance

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Open (or re-open) the per-thread connection.

        Safe to call multiple times — a no-op if already connected.
        Creates the database file and parent directories as needed.
        """
        if getattr(self._local, "conn", None) is not None:
            return  # already connected on this thread

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,   # we guard threads ourselves
                timeout=self._timeout,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
        except sqlite3.Error as exc:
            raise ConnectionError(f"Cannot open SQLite at {self._db_path}: {exc}") from exc

        conn.row_factory = sqlite3.Row
        # Performance pragmas
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -32000")    # 32 MB page cache
        conn.execute("PRAGMA temp_store = MEMORY")
        if self._wal_mode:
            result = conn.execute("PRAGMA journal_mode = WAL").fetchone()
            if result[0] != "wal":
                logger.warning("WAL mode could not be enabled (got %s)", result[0])

        self._local.conn = conn
        logger.info(
            "SQLite connected [thread=%d]: %s (WAL=%s)",
            threading.get_ident(), self._db_path.name, self._wal_mode,
        )

    def close(self) -> None:
        """Close the per-thread connection, if open."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
            logger.debug("SQLite connection closed [thread=%d]", threading.get_ident())

    def _conn(self) -> sqlite3.Connection:
        """Return the per-thread connection, auto-connecting if needed."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is None:
            self.connect()
            conn = self._local.conn
        return conn  # type: ignore[return-value]

    # ── Schema initialisation ─────────────────────────────────────────────────

    def initialise_schema(self) -> None:
        """
        Create all tables and indexes defined in :data:`~src.database.models.TABLE_DDL`.

        Safe to call on every startup — uses ``CREATE TABLE IF NOT EXISTS``.
        """
        conn = self._conn()
        try:
            with conn:
                for table_name in TABLE_CREATION_ORDER:
                    for statement in TABLE_DDL[table_name]:
                        conn.execute(statement)
            logger.info("Database schema initialised (%d tables)", len(TABLE_CREATION_ORDER))
        except sqlite3.Error as exc:
            raise DatabaseError(f"Schema initialisation failed: {exc}") from exc

    def table_exists(self, table_name: str) -> bool:
        """
        Return ``True`` if *table_name* exists in the database.

        Parameters
        ----------
        table_name:
            Exact SQLite table name (case-sensitive).
        """
        row = self._conn().execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    # ── Core query helpers ────────────────────────────────────────────────────

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager yielding the per-thread connection.

        The caller is responsible for committing or rolling back; the
        context manager does **not** issue ``COMMIT``/``ROLLBACK``
        automatically so that callers can batch multiple statements.

        Example
        -------
        ::

            with db.get_connection() as conn:
                conn.execute(Q.INSERT_PRICE_DATA, params)
                conn.commit()
        """
        yield self._conn()

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """
        Execute a single DML/DDL statement inside an implicit transaction.

        Parameters
        ----------
        query:
            Parameterised SQL string — use ``?`` placeholders.
        params:
            Positional parameters bound to the placeholders.

        Returns
        -------
        sqlite3.Cursor:
            Cursor after execution (use ``.lastrowid`` for inserts).

        Raises
        ------
        QueryError:
            On any SQLite error.
        """
        conn = self._conn()
        try:
            with conn:
                return conn.execute(query, params)
        except sqlite3.IntegrityError as exc:
            raise QueryError(f"Integrity violation: {exc}\nSQL: {query.strip()[:120]}") from exc
        except sqlite3.Error as exc:
            raise QueryError(f"Query failed: {exc}\nSQL: {query.strip()[:120]}") from exc

    def execute_many(
        self,
        query: str,
        params_list: list[tuple[Any, ...]],
    ) -> int:
        """
        Batch-execute *query* for each parameter tuple in *params_list*.

        Uses a single transaction for performance.

        Parameters
        ----------
        query:
            Parameterised SQL string.
        params_list:
            List of parameter tuples — one per row.

        Returns
        -------
        int:
            Total number of rows affected.

        Raises
        ------
        QueryError:
            On any SQLite error.
        """
        if not params_list:
            return 0
        conn = self._conn()
        try:
            with conn:
                cursor = conn.executemany(query, params_list)
                return cursor.rowcount
        except sqlite3.Error as exc:
            raise QueryError(f"Batch execute failed: {exc}\nSQL: {query.strip()[:120]}") from exc

    def fetch_one(
        self,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> Optional[dict[str, Any]]:
        """
        Execute a SELECT and return the first matching row as a dict.

        Parameters
        ----------
        query:
            Parameterised SELECT statement.
        params:
            Bound parameters.

        Returns
        -------
        dict or None:
            Row as ``{column_name: value}``, or ``None`` if no rows found.
        """
        conn = self._conn()
        try:
            row: Optional[sqlite3.Row] = conn.execute(query, params).fetchone()
            return dict(row) if row is not None else None
        except sqlite3.Error as exc:
            raise QueryError(f"fetch_one failed: {exc}\nSQL: {query.strip()[:120]}") from exc

    def fetch_all(
        self,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        """
        Execute a SELECT and return all matching rows as a list of dicts.

        Parameters
        ----------
        query:
            Parameterised SELECT statement.
        params:
            Bound parameters.

        Returns
        -------
        list[dict]:
            Every row as ``{column_name: value}``.  Empty list if no rows.
        """
        conn = self._conn()
        try:
            rows: list[sqlite3.Row] = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            raise QueryError(f"fetch_all failed: {exc}\nSQL: {query.strip()[:120]}") from exc

    # ── Maintenance ───────────────────────────────────────────────────────────

    def get_db_size(self) -> str:
        """
        Return the database file size as a human-readable string.

        Returns
        -------
        str:
            E.g. ``"12.4 MB"`` or ``"934.0 KB"``.
        """
        try:
            size_bytes = self._db_path.stat().st_size
        except FileNotFoundError:
            return "0 B"

        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024  # type: ignore[assignment]
        return f"{size_bytes:.1f} TB"

    def cleanup_old_data(self, days: int = 90) -> dict[str, int]:
        """
        Delete rows older than *days* days from non-critical tables.

        The following tables are pruned:
        - ``price_data`` (intraday timeframes only: 1m, 5m, 15m, 1h)
        - ``options_chain_snapshot``
        - ``news_articles`` (+ cascade to ``news_index_impact``)
        - ``anomaly_events``
        - ``system_health``

        Daily (``1d``) price data and all signals are **never** deleted.

        Parameters
        ----------
        days:
            Retention window.  Rows with ``timestamp`` or ``published_at``
            older than ``now − days`` are removed.

        Returns
        -------
        dict[str, int]:
            Mapping of ``table_name → rows_deleted``.
        """
        cutoff = datetime.now(tz=_IST).strftime("%Y-%m-%dT00:00:00")
        # Compute cutoff by going back `days` calendar days
        from datetime import timedelta
        cutoff_dt = datetime.now(tz=_IST) - timedelta(days=days)
        cutoff = cutoff_dt.isoformat()

        deleted: dict[str, int] = {}

        cleanup_ops: list[tuple[str, str, tuple[Any, ...]]] = [
            (
                "price_data (intraday)",
                "DELETE FROM price_data WHERE timestamp < ? AND timeframe != '1d'",
                (cutoff,),
            ),
            (
                "options_chain_snapshot",
                "DELETE FROM options_chain_snapshot WHERE timestamp < ?",
                (cutoff,),
            ),
            (
                "news_articles",
                "DELETE FROM news_articles WHERE published_at < ?",
                (cutoff,),
            ),
            (
                "anomaly_events",
                "DELETE FROM anomaly_events WHERE timestamp < ? AND is_active = 0",
                (cutoff,),
            ),
            (
                "system_health",
                "DELETE FROM system_health WHERE timestamp < ?",
                (cutoff,),
            ),
        ]

        for label, sql, params in cleanup_ops:
            try:
                cursor = self.execute(sql, params)
                count = cursor.rowcount
                deleted[label] = count
                if count:
                    logger.info("Cleaned %d rows from %s (cutoff %s)", count, label, cutoff[:10])
            except QueryError as exc:
                logger.error("Cleanup failed for %s: %s", label, exc)
                deleted[label] = -1

        return deleted

    def vacuum(self) -> None:
        """
        Run ``VACUUM`` to reclaim space and defragment the database file.

        This is a long-running operation — only call it during maintenance
        windows, not during normal trading hours.
        """
        conn = self._conn()
        logger.info("Starting VACUUM on %s (size before: %s)", self._db_path.name, self.get_db_size())
        t0 = time.monotonic()
        try:
            conn.execute("VACUUM")
        except sqlite3.Error as exc:
            raise DatabaseError(f"VACUUM failed: {exc}") from exc
        elapsed = time.monotonic() - t0
        logger.info("VACUUM complete in %.1fs (size after: %s)", elapsed, self.get_db_size())

    def write_health(
        self,
        component: str,
        status: str,
        message: Optional[str] = None,
        response_time_ms: Optional[int] = None,
    ) -> None:
        """
        Persist a system health check record.

        Parameters
        ----------
        component:
            Component name, e.g. ``"nse_scraper"``.
        status:
            ``"OK"``, ``"WARNING"``, or ``"ERROR"``.
        message:
            Optional detail string.
        response_time_ms:
            Measured response time, if applicable.
        """
        from src.database import queries as Q

        now = datetime.now(tz=_IST).isoformat()
        try:
            self.execute(Q.INSERT_SYSTEM_HEALTH, (now, component, status, message, response_time_ms))
        except QueryError as exc:
            # Health writes must never raise — just log
            logger.warning("Failed to write health record for %s: %s", component, exc)

    # ── Async wrappers ────────────────────────────────────────────────────────

    async def async_execute(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> sqlite3.Cursor:
        """Non-blocking execute via the default thread-pool executor."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, query, params)

    async def async_execute_many(
        self, query: str, params_list: list[tuple[Any, ...]]
    ) -> int:
        """Non-blocking batch execute."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_many, query, params_list)

    async def async_fetch_one(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> Optional[dict[str, Any]]:
        """Non-blocking fetch_one."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fetch_one, query, params)

    async def async_fetch_all(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        """Non-blocking fetch_all."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fetch_all, query, params)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "DatabaseManager":
        """Connect and initialise schema on entry."""
        self.connect()
        self.initialise_schema()
        return self

    def __exit__(self, *_: object) -> None:
        """Close the per-thread connection on exit."""
        self.close()

    def __repr__(self) -> str:
        connected = getattr(self._local, "conn", None) is not None
        return (
            f"DatabaseManager(path={self._db_path.name!r}, "
            f"wal={self._wal_mode}, connected={connected})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience accessor
# ---------------------------------------------------------------------------

def get_db_manager() -> DatabaseManager:
    """
    Return the process-wide :class:`DatabaseManager` singleton.

    The singleton is initialised from ``settings.database`` on the first
    call.  Subsequent calls return the cached instance.

    Example
    -------
    ::

        db = get_db_manager()
        row = db.fetch_one(Q.GET_LATEST_PRICE, ("NIFTY50",))
    """
    return DatabaseManager.instance()
