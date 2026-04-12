"""
FastAPI shared dependencies (injected via Depends).
"""

from __future__ import annotations

from functools import lru_cache

from config.settings import settings
from src.data.index_registry import IndexRegistry
from src.database.db_manager import DatabaseManager


@lru_cache(maxsize=1)
def get_index_registry() -> IndexRegistry:
    """
    Return the module-level IndexRegistry singleton.

    Loaded once on first call and cached for the application lifetime.
    New indices added to ``indices.json`` require a server restart.
    """
    return IndexRegistry.from_file(settings.indices_config_path)


@lru_cache(maxsize=1)
def get_db() -> DatabaseManager:
    """
    Return the process-wide DatabaseManager singleton.

    Connected and ready for use; cached for the application lifetime.
    """
    db = DatabaseManager()
    db.connect()
    return db
