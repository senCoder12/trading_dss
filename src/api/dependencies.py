"""
FastAPI shared dependencies (injected via Depends).

Singletons are created once and cached for the application lifetime.
Heavy components (DecisionEngine, SignalTracker) are optional — the API
can start in read-only mode when only the DB is available.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Optional

from config.settings import settings
from src.data.index_registry import IndexRegistry
from src.database.db_manager import DatabaseManager
from src.utils.market_hours import MarketHoursManager


# ── Core singletons ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_index_registry() -> IndexRegistry:
    """Return the module-level IndexRegistry singleton."""
    return IndexRegistry.from_file(settings.indices_config_path)


@lru_cache(maxsize=1)
def get_db() -> DatabaseManager:
    """Return the process-wide DatabaseManager singleton."""
    db = DatabaseManager()
    db.connect()
    return db


@lru_cache(maxsize=1)
def get_market_hours() -> MarketHoursManager:
    """Return a MarketHoursManager singleton."""
    return MarketHoursManager()


# ── Optional heavy components ────────────────────────────────────────────────
# These may not be running if only the API server is started.

_decision_engine = None
_signal_tracker = None
_kill_switch_lock = threading.Lock()
_kill_switch_active = False
_kill_switch_reason: Optional[str] = None


def get_decision_engine():
    return _decision_engine


def set_decision_engine(engine):
    global _decision_engine
    _decision_engine = engine


def get_signal_tracker():
    """Return the SignalTracker if one has been wired up."""
    return _signal_tracker


def set_signal_tracker(tracker):
    global _signal_tracker
    _signal_tracker = tracker


# ── Kill switch ──────────────────────────────────────────────────────────────

def activate_kill_switch(reason: str) -> None:
    global _kill_switch_active, _kill_switch_reason
    with _kill_switch_lock:
        _kill_switch_active = True
        _kill_switch_reason = reason


def deactivate_kill_switch() -> None:
    global _kill_switch_active, _kill_switch_reason
    with _kill_switch_lock:
        _kill_switch_active = False
        _kill_switch_reason = None


def is_kill_switch_active() -> tuple[bool, Optional[str]]:
    with _kill_switch_lock:
        return _kill_switch_active, _kill_switch_reason
