"""
Index registry — loads, validates, and serves all Indian index definitions.

The registry reads ``config/indices.json`` as its source of truth and
optionally syncs the ``index_master`` database table on startup.

Design decisions
----------------
- **Singleton**: one load per process; call :func:`get_registry` everywhere.
- **Pydantic model**: ``Index`` validates every field on construction.
- **Hot-reload**: :meth:`IndexRegistry.reload` re-reads JSON without restart.
- **DB sync**: :meth:`IndexRegistry.sync_to_db` upserts into ``index_master``,
  marking any IDs that disappeared from JSON as ``is_active=False``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, field_validator, model_validator

from config.constants import IST_TIMEZONE
from config.settings import settings

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class IndexRegistryError(Exception):
    """Raised when the index registry cannot be loaded or validated."""


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class Index(BaseModel):
    """
    Immutable, fully-validated definition of a single Indian index.

    All fields map directly to columns in ``index_master`` and fields in
    ``indices.json``.  Validation runs at construction time so bad config
    is caught at startup, not during trading.
    """

    model_config = {"frozen": True}  # immutable after construction

    id: str
    display_name: str
    nse_symbol: Optional[str] = None
    yahoo_symbol: Optional[str] = None
    exchange: str
    lot_size: Optional[int] = None
    has_options: bool = False
    option_symbol: Optional[str] = None
    sector_category: str = "unknown"
    is_active: bool = True
    description: str = ""

    @field_validator("exchange")
    @classmethod
    def _exchange_valid(cls, v: str) -> str:
        if v not in {"NSE", "BSE"}:
            raise ValueError(f"exchange must be 'NSE' or 'BSE', got {v!r}")
        return v

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must not be empty")
        return v.strip().upper()

    @field_validator("lot_size")
    @classmethod
    def _lot_size_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(f"lot_size must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _options_consistency(self) -> "Index":
        if self.has_options and self.option_symbol is None:
            raise ValueError(
                f"Index {self.id!r}: has_options=True but option_symbol is null"
            )
        return self

    # ── Convenience properties ──────────────────────────────────────────────

    @property
    def is_fo_enabled(self) -> bool:
        """True when both lot_size and option_symbol are set."""
        return self.has_options and self.lot_size is not None

    @property
    def is_nse(self) -> bool:
        return self.exchange == "NSE"

    @property
    def is_bse(self) -> bool:
        return self.exchange == "BSE"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class IndexRegistry:
    """
    Thread-safe in-memory registry of all Indian market indices.

    Load from JSON with :meth:`from_file`, or use the process singleton
    via :func:`get_registry`.

    Parameters
    ----------
    indices:
        Pre-parsed list of :class:`Index` objects.
    source_path:
        Path of the JSON file the indices were loaded from (used by
        :meth:`reload`).
    """

    def __init__(
        self,
        indices: list[Index],
        source_path: Optional[Path] = None,
    ) -> None:
        self._source_path: Optional[Path] = source_path
        self._lock = threading.RLock()
        self._indices: dict[str, Index] = {}
        self._load_time: Optional[datetime] = None
        self._set_indices(indices)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_indices(self, indices: list[Index]) -> None:
        """Replace the internal index map (called under write lock)."""
        self._indices = {idx.id: idx for idx in indices}
        self._load_time = datetime.now(tz=_IST)
        logger.info(
            "IndexRegistry: loaded %d indices (%d active)",
            len(self._indices),
            sum(1 for i in self._indices.values() if i.is_active),
        )

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "IndexRegistry":
        """
        Load the registry from *path* (JSON array of index objects).

        Parameters
        ----------
        path:
            Absolute path to ``indices.json``.  Defaults to
            ``settings.indices_config_path``.

        Raises
        ------
        IndexRegistryError:
            If the file is missing, unparseable, or any entry is invalid.
        """
        resolved = path or settings.indices_config_path
        if not resolved.exists():
            raise IndexRegistryError(
                f"Indices config not found: {resolved}. "
                "Set INDICES_CONFIG_PATH or ensure config/indices.json exists."
            )

        try:
            raw_text = resolved.read_text(encoding="utf-8")
            raw: list[dict] = json.loads(raw_text)
        except (json.JSONDecodeError, OSError) as exc:
            raise IndexRegistryError(
                f"Failed to read/parse {resolved}: {exc}"
            ) from exc

        if not isinstance(raw, list):
            raise IndexRegistryError(
                f"{resolved} must contain a JSON array, got {type(raw).__name__}"
            )

        indices: list[Index] = []
        errors: list[str] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                errors.append(f"Entry #{i}: expected object, got {type(entry).__name__}")
                continue
            idx_id = entry.get("id", f"<entry #{i}>")
            try:
                indices.append(Index.model_validate(entry))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Entry {idx_id!r}: {exc}")

        if errors:
            raise IndexRegistryError(
                f"Validation failed for {len(errors)} index entries:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

        return cls(indices, source_path=resolved)

    # ── Hot-reload ────────────────────────────────────────────────────────────

    def reload(self) -> int:
        """
        Re-read the source JSON file and replace the in-memory registry.

        Returns
        -------
        int:
            Number of indices after reload.

        Raises
        ------
        IndexRegistryError:
            If the reload fails.  The registry is **not** modified on error.
        """
        if self._source_path is None:
            raise IndexRegistryError("Cannot reload: registry was not loaded from a file")

        new_registry = IndexRegistry.from_file(self._source_path)
        with self._lock:
            self._set_indices(list(new_registry._indices.values()))
        logger.info("IndexRegistry reloaded from %s", self._source_path)
        return len(self._indices)

    # ── DB sync ───────────────────────────────────────────────────────────────

    def sync_to_db(self, db: "DatabaseManager") -> dict[str, int]:  # type: ignore[name-defined]
        """
        Upsert all indices into ``index_master`` and mark removed IDs inactive.

        Parameters
        ----------
        db:
            Connected :class:`~src.database.db_manager.DatabaseManager`.

        Returns
        -------
        dict[str, int]:
            ``{"upserted": N, "deactivated": N}``
        """
        from src.database import queries as Q

        now = datetime.now(tz=_IST).isoformat()

        rows = [
            (
                idx.id,
                idx.display_name,
                idx.nse_symbol,
                idx.yahoo_symbol,
                idx.exchange,
                idx.lot_size,
                int(idx.has_options),
                idx.option_symbol,
                idx.sector_category,
                int(idx.is_active),
                now,
                now,
            )
            for idx in self._indices.values()
        ]
        upserted = db.execute_many(Q.INSERT_INDEX_MASTER, rows)

        # Mark IDs present in DB but not in current JSON as inactive
        existing_rows = db.fetch_all("SELECT id FROM index_master WHERE is_active = 1")
        existing_ids = {r["id"] for r in existing_rows}
        current_ids = set(self._indices.keys())
        removed_ids = existing_ids - current_ids

        deactivated = 0
        for removed_id in removed_ids:
            db.execute(
                "UPDATE index_master SET is_active = 0, updated_at = ? WHERE id = ?",
                (now, removed_id),
            )
            deactivated += 1
            logger.warning("IndexRegistry: marked %s as inactive (removed from JSON)", removed_id)

        logger.info(
            "DB sync complete — upserted=%d, deactivated=%d",
            len(rows), deactivated,
        )
        return {"upserted": len(rows), "deactivated": deactivated}

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_all_indices(self, active_only: bool = False) -> list[Index]:
        """
        Return all indices in the registry.

        Parameters
        ----------
        active_only:
            When ``True``, only return indices with ``is_active=True``.
        """
        with self._lock:
            indices = list(self._indices.values())
        if active_only:
            indices = [i for i in indices if i.is_active]
        return indices

    def get_active_indices(self) -> list[Index]:
        """Return only active indices (``is_active=True``)."""
        return self.get_all_indices(active_only=True)

    def get_index(self, index_id: str) -> Optional[Index]:
        """
        Return the :class:`Index` for *index_id*, or ``None`` if not found.

        Parameters
        ----------
        index_id:
            Registry ID (case-insensitive).
        """
        with self._lock:
            return self._indices.get(index_id.upper())

    def get_index_or_raise(self, index_id: str) -> Index:
        """
        Return the :class:`Index` for *index_id*.

        Raises
        ------
        KeyError:
            If *index_id* is not in the registry.
        """
        idx = self.get_index(index_id)
        if idx is None:
            raise KeyError(f"Unknown index id: {index_id!r}")
        return idx

    def get_indices_by_exchange(self, exchange: str) -> list[Index]:
        """
        Return active indices for *exchange*.

        Parameters
        ----------
        exchange:
            ``"NSE"`` or ``"BSE"`` (case-insensitive).
        """
        ex = exchange.upper()
        return [i for i in self.get_active_indices() if i.exchange == ex]

    def get_indices_with_options(self) -> list[Index]:
        """Return active indices that have options (weekly or monthly)."""
        return [i for i in self.get_active_indices() if i.has_options]

    def get_index_by_option_symbol(self, option_symbol: str) -> Optional[Index]:
        """Return the Index matching the given option_symbol (e.g. 'NIFTY')."""
        sym = option_symbol.upper()
        for idx in self.get_active_indices():
            if idx.option_symbol and idx.option_symbol.upper() == sym:
                return idx
        return None

    def get_indices_by_sector(self, sector_category: str) -> list[Index]:
        """
        Return active indices matching *sector_category*.

        Parameters
        ----------
        sector_category:
            Value from ``sector_category`` field (e.g. ``"sectoral"``,
            ``"broad_market"``, ``"thematic"``).
        """
        return [
            i for i in self.get_active_indices()
            if i.sector_category == sector_category
        ]

    def search_indices(self, query: str) -> list[Index]:
        """
        Fuzzy search across ``id``, ``display_name``, ``nse_symbol``,
        ``option_symbol``, and ``sector_category``.

        Matching is case-insensitive; returns active indices only.
        Results are ranked: exact id match first, then contains matches.

        Parameters
        ----------
        query:
            Search string (e.g. ``"bank"``, ``"IT"``, ``"midcap"``).
        """
        q = query.strip().lower()
        if not q:
            return self.get_active_indices()

        exact: list[Index] = []
        partial: list[Index] = []

        for idx in self.get_active_indices():
            searchable = " ".join(filter(None, [
                idx.id,
                idx.display_name,
                idx.nse_symbol or "",
                idx.option_symbol or "",
                idx.sector_category,
                idx.description,
            ])).lower()

            if idx.id.lower() == q:
                exact.append(idx)
            elif q in searchable:
                partial.append(idx)

        return exact + partial

    # ── Convenience filters (backward compat) ─────────────────────────────────

    def get(self, index_id: str) -> Index:
        """Alias for :meth:`get_index_or_raise`."""
        return self.get_index_or_raise(index_id)

    def get_or_none(self, index_id: str) -> Optional[Index]:
        """Alias for :meth:`get_index`."""
        return self.get_index(index_id)

    def filter(
        self,
        *,
        exchange: Optional[str] = None,
        has_options: Optional[bool] = None,
        sector_category: Optional[str] = None,
        active_only: bool = True,
    ) -> list[Index]:
        """
        Filter indices by multiple criteria simultaneously.

        Parameters
        ----------
        exchange:
            ``"NSE"`` or ``"BSE"``.
        has_options:
            Filter by options availability.
        sector_category:
            Exact sector category match.
        active_only:
            Exclude inactive indices.
        """
        result = self.get_all_indices(active_only=active_only)
        if exchange is not None:
            result = [i for i in result if i.exchange == exchange.upper()]
        if has_options is not None:
            result = [i for i in result if i.has_options == has_options]
        if sector_category is not None:
            result = [i for i in result if i.sector_category == sector_category]
        return result

    def ids(self, active_only: bool = True) -> list[str]:
        """Return all index IDs."""
        return [i.id for i in self.get_all_indices(active_only=active_only)]

    def yahoo_symbols(self, active_only: bool = True) -> dict[str, str]:
        """Return ``{index_id: yahoo_symbol}`` for indices that have a Yahoo ticker."""
        return {
            i.id: i.yahoo_symbol
            for i in self.get_all_indices(active_only=active_only)
            if i.yahoo_symbol
        }

    # ── Meta ──────────────────────────────────────────────────────────────────

    @property
    def load_time(self) -> Optional[datetime]:
        """Datetime when the registry was last loaded/reloaded."""
        return self._load_time

    def __len__(self) -> int:
        with self._lock:
            return len(self._indices)

    def __contains__(self, index_id: str) -> bool:
        with self._lock:
            return index_id.upper() in self._indices

    def __repr__(self) -> str:
        return (
            f"IndexRegistry(total={len(self)}, "
            f"active={len(self.get_active_indices())}, "
            f"loaded_at={self._load_time and self._load_time.strftime('%H:%M:%S')})"
        )


# ---------------------------------------------------------------------------
# Process singleton
# ---------------------------------------------------------------------------

_singleton: Optional[IndexRegistry] = None
_singleton_lock = threading.Lock()


def get_registry(path: Optional[Path] = None) -> IndexRegistry:
    """
    Return the process-wide :class:`IndexRegistry` singleton.

    Thread-safe.  Loads from ``config/indices.json`` on the first call
    (or from *path* if provided).  Subsequent calls ignore *path* and return
    the cached instance.

    Parameters
    ----------
    path:
        Override path — only honoured on the very first call.

    Example
    -------
    ::

        registry = get_registry()
        nifty = registry.get_index("NIFTY50")
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = IndexRegistry.from_file(path)
    return _singleton


def reset_registry() -> None:
    """
    Clear the singleton — forces a fresh load on the next :func:`get_registry` call.

    Intended for use in tests only.
    """
    global _singleton
    with _singleton_lock:
        _singleton = None
