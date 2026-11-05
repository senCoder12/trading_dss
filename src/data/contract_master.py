"""
Contract master — single source of truth for F&O contract facts.

Lot sizes and expiry dates are *not* stable facts:

* NSE/BSE revise lot sizes twice a year, and a revision applies per expiry
  (e.g. SENSEX was 20 for Nov/Dec 2026 contracts and 25 from Jan 2027), so
  two live contracts on the same underlying can have different lot sizes.
* Expiry weekdays change by circular (NIFTY moved Thursday → Tuesday,
  SENSEX Tuesday → Thursday, BANKNIFTY/FINNIFTY/MIDCPNIFTY went monthly-only)
  and holidays shift individual expiries to another weekday.

Hardcoding either one, or deriving expiries from a weekday rule, silently
corrupts every ₹ figure and every expiry-day rule.  This module reads both
from Angel One's public instrument master (``OpenAPIScripMaster.json``),
which the broker refreshes before every session.

Only option contracts (``OPTIDX`` / ``OPTSTK`` on NFO/BFO) are kept,
compacted to ``{underlying: {expiry_iso: lot_size}}`` and cached on disk per
IST day, so the ~36 MB file is downloaded at most once per day.

Environment switches
--------------------
``TRADING_DSS_OFFLINE=1``
    Never download; use the newest on-disk cache only.
``TRADING_DSS_CONTRACT_MASTER_DISABLED=1``
    Disable the master entirely (all lookups return ``None``).  Used by the
    test suite so results never depend on network or on a cache file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Callable, Optional, Union
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")
_SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)
_OPTION_INSTRUMENTS = frozenset({"OPTIDX", "OPTSTK"})
_OPTION_SEGMENTS = frozenset({"NFO", "BFO"})
_CACHE_PREFIX = "contract_master_"
_CACHE_VERSION = 1
# Options stop trading at 15:30 IST; after that today's expiry is dead.
_EXPIRY_CUTOFF = dtime(15, 30)

# Registry ids whose scrip-master underlying name differs from the id.
_UNDERLYING_ALIASES: dict[str, str] = {
    "NIFTY50": "NIFTY",
    "NIFTY_50": "NIFTY",
    "NIFTYBANK": "BANKNIFTY",
}

Contracts = dict[str, dict[str, int]]  # {underlying: {expiry_iso: lot_size}}


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "cache"


def _download_scrip_master() -> list[dict]:
    """Fetch the raw instrument master (list of dicts) from Angel One."""
    import requests  # noqa: PLC0415 — optional at import time

    resp = requests.get(_SCRIP_MASTER_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected scrip master payload type: {type(data).__name__}")
    return data


def parse_scrip_master(rows: list[dict]) -> Contracts:
    """Compact raw scrip-master rows into ``{underlying: {expiry_iso: lot}}``.

    Rows are external data: anything malformed is skipped, never trusted.
    """
    contracts: Contracts = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("instrumenttype") not in _OPTION_INSTRUMENTS:
            continue
        if row.get("exch_seg") not in _OPTION_SEGMENTS:
            continue
        name = str(row.get("name") or "").strip().upper()
        if not name:
            continue
        try:
            expiry = datetime.strptime(str(row.get("expiry", "")), "%d%b%Y").date()
            lot = int(float(row.get("lotsize") or 0))
        except (ValueError, TypeError):
            continue
        if lot <= 0:
            continue
        contracts.setdefault(name, {}).setdefault(expiry.isoformat(), lot)
    return contracts


def _ist_now() -> datetime:
    return datetime.now(tz=_IST)


def _as_ist(ts: Union[datetime, date, None]) -> datetime:
    """Normalise *ts* to an aware IST datetime (date → start of that day)."""
    if ts is None:
        return _ist_now()
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=_IST) if ts.tzinfo is None else ts.astimezone(_IST)
    return datetime.combine(ts, dtime(0, 0), tzinfo=_IST)


class ContractMaster:
    """Daily-cached lot sizes and expiry calendar for every option underlying.

    Parameters
    ----------
    cache_dir:
        Where per-day compact cache files live (default ``data/cache``).
    fetcher:
        Callable returning raw scrip-master rows.  Injectable for tests.
    allow_download:
        Override the ``TRADING_DSS_OFFLINE`` switch.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        fetcher: Optional[Callable[[], list[dict]]] = None,
        allow_download: Optional[bool] = None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self._fetcher = fetcher or _download_scrip_master
        if allow_download is None:
            allow_download = os.environ.get("TRADING_DSS_OFFLINE") != "1"
        self._allow_download = allow_download
        self._disabled = os.environ.get("TRADING_DSS_CONTRACT_MASTER_DISABLED") == "1"
        self._lock = threading.RLock()
        self._contracts: Contracts = {}
        self._loaded_for: Optional[date] = None
        self._source: str = "none"
        self._warned_missing: set[str] = set()

    # ── Loading ──────────────────────────────────────────────────────────

    @property
    def source(self) -> str:
        """``download`` / ``cache`` / ``stale_cache:<date>`` / ``none``."""
        return self._source

    @property
    def is_loaded(self) -> bool:
        return bool(self._contracts)

    def refresh(self, today: Optional[date] = None, force: bool = False) -> bool:
        """Load today's contract facts.  Returns ``True`` if any data is loaded.

        Order: today's cache file → download (if allowed) → newest older cache
        (logged as stale).  Never raises.
        """
        if self._disabled:
            return False
        today = today or _ist_now().date()
        with self._lock:
            if not force and self._loaded_for == today and self._contracts:
                return True

            cached = self._read_cache(self._cache_path(today))
            if cached:
                self._set(cached, today, "cache")
                return True

            if self._allow_download:
                try:
                    contracts = parse_scrip_master(self._fetcher())
                    if contracts:
                        self._write_cache(self._cache_path(today), contracts, today)
                        self._set(contracts, today, "download")
                        return True
                    logger.error("ContractMaster: scrip master parsed to zero option contracts")
                except Exception as exc:
                    logger.error("ContractMaster: scrip master download failed: %s", exc)

            stale_path = self._newest_cache_before(today)
            if stale_path is not None:
                stale = self._read_cache(stale_path)
                if stale:
                    stale_day = stale_path.stem.replace(_CACHE_PREFIX, "")
                    logger.warning(
                        "ContractMaster: using STALE contract cache from %s — "
                        "lot sizes/expiries may be outdated", stale_day,
                    )
                    self._set(stale, today, f"stale_cache:{stale_day}")
                    return True

            logger.error(
                "ContractMaster: no contract data available — lot sizes fall back "
                "to config/indices.json and expiry checks to the options chain"
            )
            self._loaded_for = today
            self._source = "none"
            return False

    def _ensure_loaded(self) -> None:
        if self._disabled:
            return
        today = _ist_now().date()
        if self._loaded_for != today:
            self.refresh(today)

    def _set(self, contracts: Contracts, today: date, source: str) -> None:
        self._contracts = contracts
        self._loaded_for = today
        self._source = source
        logger.info(
            "ContractMaster: %d option underlyings loaded (source=%s)",
            len(contracts), source,
        )

    # ── Cache I/O ────────────────────────────────────────────────────────

    def _cache_path(self, day: date) -> Path:
        return self._cache_dir / f"{_CACHE_PREFIX}{day.isoformat()}.json"

    def _newest_cache_before(self, day: date) -> Optional[Path]:
        try:
            files = sorted(self._cache_dir.glob(f"{_CACHE_PREFIX}*.json"))
        except OSError:
            return None
        older = [p for p in files if p.stem.replace(_CACHE_PREFIX, "") < day.isoformat()]
        return older[-1] if older else None

    @staticmethod
    def _read_cache(path: Path) -> Optional[Contracts]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
            return None
        contracts = payload.get("contracts")
        if not isinstance(contracts, dict):
            return None
        clean: Contracts = {}
        for name, expiries in contracts.items():
            if not isinstance(expiries, dict):
                continue
            good = {
                str(e): int(l) for e, l in expiries.items()
                if isinstance(l, (int, float)) and int(l) > 0
            }
            if good:
                clean[str(name).upper()] = good
        return clean or None

    def _write_cache(self, path: Path, contracts: Contracts, day: date) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"version": _CACHE_VERSION, "day": day.isoformat(),
                            "contracts": contracts}),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("ContractMaster: could not write cache %s: %s", path, exc)

    # ── Lookups ──────────────────────────────────────────────────────────

    @staticmethod
    def underlying_for(symbol: str) -> str:
        sym = (symbol or "").strip().upper()
        return _UNDERLYING_ALIASES.get(sym, sym)

    def _contracts_for(self, symbol: str) -> dict[str, int]:
        self._ensure_loaded()
        name = self.underlying_for(symbol)
        found = self._contracts.get(name, {})
        if not found and self._contracts and name not in self._warned_missing:
            self._warned_missing.add(name)
            logger.warning("ContractMaster: no option contracts for underlying %r", name)
        return found

    def expiries(self, symbol: str, on_or_after: Optional[date] = None) -> list[date]:
        """All listed option expiries for *symbol*, ascending."""
        out = sorted(date.fromisoformat(e) for e in self._contracts_for(symbol))
        if on_or_after is not None:
            out = [e for e in out if e >= on_or_after]
        return out

    def nearest_expiry(
        self, symbol: str, as_of: Union[datetime, date, None] = None,
    ) -> Optional[date]:
        """Nearest *tradeable* expiry at *as_of*.

        On an expiry day the contract is tradeable until 15:30 IST; after
        that the next expiry is the nearest one.
        """
        ts = _as_ist(as_of)
        first_day = ts.date()
        if isinstance(as_of, datetime) and ts.time() >= _EXPIRY_CUTOFF:
            first_day = date.fromordinal(first_day.toordinal() + 1)
        listed = self.expiries(symbol, on_or_after=first_day)
        return listed[0] if listed else None

    def lot_size(
        self,
        symbol: str,
        expiry: Optional[date] = None,
        as_of: Union[datetime, date, None] = None,
    ) -> Optional[int]:
        """Lot size of the *expiry* contract (default: nearest tradeable)."""
        contracts = self._contracts_for(symbol)
        if not contracts:
            return None
        if expiry is None:
            expiry = self.nearest_expiry(symbol, as_of)
            if expiry is None:
                return None
        return contracts.get(expiry.isoformat())

    def is_expiry_day(self, symbol: str, day: Optional[date] = None) -> Optional[bool]:
        """``True``/``False`` if known, ``None`` when no data for *symbol*."""
        contracts = self._contracts_for(symbol)
        if not contracts:
            return None
        day = day or _ist_now().date()
        return day.isoformat() in contracts


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def sync_registry_lot_sizes(
    registry,
    master: Optional[ContractMaster] = None,
    as_of: Union[datetime, date, None] = None,
) -> dict[str, int]:
    """Overwrite registry lot sizes with the nearest-expiry contract lot size.

    Returns ``{index_id: lot_size}`` for every index that changed.
    """
    master = master or get_contract_master()
    if not master.refresh():
        return {}
    updates: dict[str, int] = {}
    for idx in registry.get_indices_with_options():
        symbol = idx.option_symbol or idx.id
        lot = master.lot_size(symbol, as_of=as_of)
        if lot and lot != idx.lot_size:
            updates[idx.id] = lot
    if updates:
        registry.apply_lot_sizes(updates)
    return updates


_master_singleton: Optional[ContractMaster] = None
_singleton_lock = threading.Lock()


def get_contract_master() -> ContractMaster:
    """Process-wide :class:`ContractMaster` (lazy; no I/O until first lookup)."""
    global _master_singleton
    with _singleton_lock:
        if _master_singleton is None:
            _master_singleton = ContractMaster()
        return _master_singleton


def reset_contract_master() -> None:
    """Drop the singleton (tests)."""
    global _master_singleton
    with _singleton_lock:
        _master_singleton = None
