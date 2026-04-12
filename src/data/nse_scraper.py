"""
NSE India live data scraper.

Uses the public NSE JSON API (no API key required).  Maintains a session
with cookies because NSE requires a prior page visit before API calls.
All requests go through :class:`~src.data.rate_limiter.RateLimiter` and
results are cached with :class:`~src.utils.cache.TTLCache`.

Anti-scraping mitigations
-------------------------
- Homepage visit before any API call (cookie seeding)
- Per-request User-Agent rotation from :data:`~config.constants.USER_AGENT_ROTATION`
- Session auto-refresh when it expires (NSE invalidates sessions ~5 min)
- Exponential-backoff retry on transient network errors via *tenacity*
- 403/429 handling: clear session flag and surface as non-retried error
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests
from curl_cffi import requests as cffi_requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.constants import IST_TIMEZONE, NSE_BASE_URL, NSE_HEADERS, USER_AGENT_ROTATION

# Browser to impersonate for TLS fingerprinting (curl_cffi)
_IMPERSONATE: str = "chrome131"
from config.settings import settings
from src.data.rate_limiter import (
    GlobalCircuitBreaker,
    GlobalRateLimiter,
    RateLimiter,
    create_nse_limiter,
    freshness_tracker,
)
from src.utils.cache import TTLCache

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SESSION_TTL: int = 4 * 60          # Re-warm before NSE's ~5-min session timeout
_CACHE_TTL_PRICE: int = 30          # Seconds to cache price data
_CACHE_TTL_STATUS: int = 60         # Seconds to cache market status
_MAX_RESPONSE_TIMES: int = 100      # Rolling window for avg-response-time calc

# NSE API endpoints (relative to NSE_BASE_URL)
_ENDPOINTS = {
    "all_indices": "/api/allIndices",
    "market_status": "/api/marketStatus",
    "option_chain": "/api/option-chain-indices",
    "equity_quote": "/api/quote-equity",
    "fii_dii": "/api/fiidiiTradeReact",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NSEScraperError(Exception):
    """Raised for NSE scraper failures."""


class _NSETransientError(NSEScraperError):
    """Internal — transient failure (network, timeout). Retried by tenacity."""


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class NSEScraper:
    """
    Session-based scraper for NSE India public JSON APIs.

    Maintains a ``requests.Session`` with cookies and realistic browser headers.
    The session is refreshed automatically when it expires or when NSE returns 401/403.

    Parameters
    ----------
    rate_limiter:
        Shared :class:`~src.data.rate_limiter.RateLimiter` instance.
        Defaults to :func:`~src.data.rate_limiter.create_nse_limiter` (25 req/60s).
    cache:
        :class:`~src.utils.cache.TTLCache` for caching API responses.
        Defaults to a new 30-second cache.

    Examples
    --------
    ::

        with NSEScraper() as nse:
            nse.warm_up()
            quote = nse.get_index_quote("NIFTY 50")
            vix = nse.get_vix()
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[TTLCache] = None,
    ) -> None:
        self._session = cffi_requests.Session(impersonate=_IMPERSONATE)
        self._rate_limiter = rate_limiter or GlobalRateLimiter.get(
            "nseindia.com", max_requests=25, window_seconds=60,
        )
        self._cache: TTLCache = cache or TTLCache(default_ttl=_CACHE_TTL_PRICE)
        self._warmed_up: bool = False
        self._session_created_at: float = 0.0

        # Health tracking
        self._total_requests: int = 0
        self._successful: int = 0
        self._failed: int = 0
        self._response_times: list[float] = []
        self._consecutive_failures: int = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_session_valid(self) -> bool:
        """Return True if the session has been warmed up and is not expired."""
        if not self._warmed_up:
            return False
        age = time.monotonic() - self._session_created_at
        return age < _SESSION_TTL

    def _build_headers(self, for_api: bool = False) -> dict[str, str]:
        """Return NSE headers with a randomly chosen User-Agent.

        Parameters
        ----------
        for_api:
            When True, set Sec-Fetch headers appropriate for an XHR/API
            call (same-site fetch) rather than a top-level page navigation.
        """
        headers = dict(NSE_HEADERS)
        headers["User-Agent"] = random.choice(USER_AGENT_ROTATION)
        if for_api:
            headers["Accept"] = "*/*"
            headers["Sec-Fetch-Dest"] = "empty"
            headers["Sec-Fetch-Mode"] = "cors"
            headers["Sec-Fetch-Site"] = "same-origin"
            headers.pop("Sec-Fetch-User", None)
            headers.pop("Upgrade-Insecure-Requests", None)
            headers["Referer"] = "https://www.nseindia.com/"
            headers["X-Requested-With"] = "XMLHttpRequest"
        return headers

    # ── Session management ────────────────────────────────────────────────────

    def _warm_up_internal(self) -> None:
        """
        Hit the NSE homepage to obtain session cookies.

        Raises
        ------
        NSEScraperError:
            On any network or HTTP failure during warm-up.
        """
        headers = self._build_headers(for_api=False)
        try:
            resp = self._session.get(NSE_BASE_URL, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            raise NSEScraperError(f"NSE warm-up failed: {exc}") from exc

        self._warmed_up = True
        self._session_created_at = time.monotonic()
        logger.info(
            "NSE session warmed up (cookies: %d, UA: %s…)",
            len(self._session.cookies),
            headers["User-Agent"][:40],
        )

    def warm_up(self) -> None:
        """
        Seed the session with NSE cookies (public method).

        NSE requires a real browser visit before the JSON APIs respond.
        Called automatically by :meth:`_get` if the session is not valid.
        """
        self._warm_up_internal()

    def _refresh_session(self) -> None:
        """Close the current session and open a fresh one, then warm it up."""
        logger.info("NSE: refreshing session")
        self._session.close()
        self._session = cffi_requests.Session(impersonate=_IMPERSONATE)
        self._warmed_up = False
        self._warm_up_internal()

    # ── Core HTTP ─────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(_NSETransientError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _get(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
    ) -> Any:
        """
        Rate-limited, cached, retrying GET.

        Retries up to 3 times (with exponential back-off) for transient network
        errors.  HTTP 401/403 are **not** retried — the session flag is cleared
        so the next call triggers a re-warm.

        Parameters
        ----------
        endpoint:
            Relative path (e.g. ``"/api/allIndices"``).
        params:
            Optional query parameters.
        cache_key:
            If provided, return cached result when available and store on success.
        cache_ttl:
            Override default TTL for this cache entry.

        Returns
        -------
        Any:
            Parsed JSON.

        Raises
        ------
        NSEScraperError:
            On non-transient failure (HTTP error, 403, parse error).
        _NSETransientError:
            On transient failure; tenacity will retry.
        """
        if cache_key:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        if not self._is_session_valid():
            self._warm_up_internal()

        url = NSE_BASE_URL + endpoint
        headers = self._build_headers(for_api=True)
        self._rate_limiter.wait_and_acquire()

        t0 = time.monotonic()
        try:
            resp = self._session.get(url, headers=headers, params=params, timeout=10)
        except Exception as exc:
            # curl_cffi raises its own exception types for timeouts/connection errors
            exc_name = type(exc).__name__.lower()
            if "timeout" in exc_name or "connection" in exc_name:
                raise _NSETransientError(f"Transient network error: {exc}") from exc
            raise NSEScraperError(f"Request failed: {exc}") from exc

        elapsed_ms = (time.monotonic() - t0) * 1000
        label = endpoint.split("?")[0].rsplit("/", 1)[-1]
        logger.debug("NSE %s → HTTP %d (%.0f ms)", label, resp.status_code, elapsed_ms)

        # Session-expiry signals — do not retry this request
        if resp.status_code in (401, 403):
            self._warmed_up = False
            raise NSEScraperError(
                f"NSE returned {resp.status_code} (session expired) — refreshed on next call"
            )

        # Rate-limit — transient, let tenacity retry
        if resp.status_code == 429:
            raise _NSETransientError("NSE rate limit exceeded (429)")

        if resp.status_code >= 400:
            raise NSEScraperError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
        except Exception as exc:
            raise _NSETransientError(f"JSON parse error: {exc}") from exc

        # Record timing
        self._response_times.append(elapsed_ms)
        if len(self._response_times) > _MAX_RESPONSE_TIMES:
            self._response_times.pop(0)

        if cache_key:
            self._cache.set(cache_key, data, ttl=cache_ttl)

        return data

    def _call(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Wrapper around :meth:`_get` that tracks health stats, honours the
        circuit breaker, and returns ``None`` instead of raising on failure.
        """
        cb = GlobalCircuitBreaker.get("nse_api", failure_threshold=5, recovery_timeout=120)

        if not cb.can_execute():
            logger.warning("NSE circuit breaker OPEN — skipping request for %s", endpoint)
            return None

        self._total_requests += 1
        try:
            result = self._get(
                endpoint, params=params, cache_key=cache_key, cache_ttl=cache_ttl
            )
            self._successful += 1
            self._consecutive_failures = 0
            cb.record_success()
            return result
        except NSEScraperError as exc:
            self._failed += 1
            self._consecutive_failures += 1
            cb.record_failure()
            if self._consecutive_failures >= 5:
                logger.critical(
                    "NSE scraper: %d consecutive failures — component health ERROR. Last: %s",
                    self._consecutive_failures,
                    exc,
                )
            else:
                logger.error("NSE request to %s failed: %s", endpoint, exc)
            return None

    # ── Normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _normalize_index_entry(self, entry: dict) -> dict:
        """Convert a raw NSE ``allIndices`` entry to a standard dict."""
        f = self._float
        return {
            "index_name": entry.get("indexSymbol", ""),
            "ltp": f(entry.get("last")),
            "open": f(entry.get("open")),
            "high": f(entry.get("high")),
            "low": f(entry.get("low")),
            "close": f(entry.get("previousClose")),
            "change": f(entry.get("variation", entry.get("change"))),
            "change_pct": f(entry.get("percentChange", entry.get("pChange"))),
            "previousClose": f(entry.get("previousClose")),
            "last": f(entry.get("last")),
            "advances": int(entry.get("advances") or 0),
            "declines": int(entry.get("declines") or 0),
            "unchanged": int(entry.get("unchanged") or 0),
            "volume": f(entry.get("totalTradedVolume")),
            "timestamp": datetime.now(tz=_IST).isoformat(),
        }

    # ── Public data methods ───────────────────────────────────────────────────

    def get_all_indices(self) -> Optional[list[dict[str, Any]]]:
        """
        Fetch live quotes for all NSE indices in a single API call.

        This is the preferred method for bulk price collection — one request
        instead of N individual calls.

        Returns
        -------
        list[dict] or None:
            Normalised index entries, or ``None`` on failure.
            Each dict has: ``index_name``, ``ltp``, ``open``, ``high``,
            ``low``, ``close``, ``change``, ``change_pct``, ``advances``,
            ``declines``, ``unchanged``, ``volume``, ``timestamp``.
        """
        raw = self._call(
            _ENDPOINTS["all_indices"],
            cache_key="nse:all_indices",
        )
        if raw is None:
            return None
        result = [self._normalize_index_entry(e) for e in raw.get("data", [])]
        if result:
            freshness_tracker.update("index_prices")
        return result

    def get_index_quote(self, nse_symbol: str) -> Optional[dict[str, Any]]:
        """
        Return the live quote dict for a single NSE index.

        Fetches :meth:`get_all_indices` (cached 30s) and filters by
        ``indexSymbol``.  Comparison is case-insensitive.

        Parameters
        ----------
        nse_symbol:
            Symbol as stored in ``indices.json`` (e.g. ``"NIFTY 50"``).

        Returns
        -------
        dict or None:
            Normalised quote, or ``None`` if not found or on failure.
        """
        raw = self._call(
            _ENDPOINTS["all_indices"],
            cache_key="nse:all_indices",
        )
        if raw is None:
            return None

        target = nse_symbol.strip().upper()
        for entry in raw.get("data", []):
            if entry.get("indexSymbol", "").upper() == target:
                return self._normalize_index_entry(entry)

        logger.warning("NSE: index symbol not found in allIndices: %r", nse_symbol)
        return None

    def get_advances_declines(self, nse_symbol: str) -> Optional[dict[str, int]]:
        """
        Return advances/declines/unchanged for *nse_symbol*.

        Returns
        -------
        dict or None:
            ``{"advances": N, "declines": N, "unchanged": N}``, or ``None``.
        """
        quote = self.get_index_quote(nse_symbol)
        if quote is None:
            return None
        return {
            "advances": quote["advances"],
            "declines": quote["declines"],
            "unchanged": quote["unchanged"],
        }

    def get_vix(self) -> Optional[dict[str, float]]:
        """
        Return the current India VIX value.

        Filters the ``allIndices`` response for an entry whose symbol
        contains both ``"INDIA"`` and ``"VIX"``.

        Returns
        -------
        dict or None:
            ``{"vix_value": float, "vix_change": float, "vix_change_pct": float}``,
            or ``None`` if not found or on failure.
        """
        raw = self._call(
            _ENDPOINTS["all_indices"],
            cache_key="nse:all_indices",
        )
        if raw is None:
            return None

        f = self._float
        for entry in raw.get("data", []):
            sym = entry.get("indexSymbol", "").upper()
            if "VIX" in sym and "INDIA" in sym:
                freshness_tracker.update("vix")
                return {
                    "vix_value": f(entry.get("last")),
                    "vix_change": f(entry.get("change")),
                    "vix_change_pct": f(entry.get("pChange")),
                }

        logger.warning("NSE: India VIX entry not found in allIndices response")
        return None

    def get_market_status(self) -> Optional[dict[str, Any]]:
        """
        Fetch current market open/close status from NSE.

        Returns
        -------
        dict or None:
            Raw NSE market status dict, or ``None`` on failure.
        """
        return self._call(
            _ENDPOINTS["market_status"],
            cache_key="nse:market_status",
            cache_ttl=_CACHE_TTL_STATUS,
        )

    def get_option_chain(self, symbol: str) -> Optional[dict[str, Any]]:
        """
        Fetch the full options chain for *symbol* (e.g. ``"NIFTY"``).

        Returns
        -------
        dict or None:
            Raw NSE option chain response, or ``None`` on failure.
        """
        return self._call(
            _ENDPOINTS["option_chain"],
            params={"symbol": symbol},
        )

    def get_fii_dii_activity(self) -> Optional[list[dict[str, Any]]]:
        """
        Fetch latest FII/DII provisional trading activity.

        Returns
        -------
        list[dict] or None:
            Daily buy/sell/net for FII and DII, or ``None`` on failure.
        """
        result = self._call(_ENDPOINTS["fii_dii"])
        if result:
            freshness_tracker.update("fii_dii")
        return result

    # ── Health ────────────────────────────────────────────────────────────────

    def get_health_stats(self) -> dict[str, Any]:
        """
        Return a snapshot of scraper health metrics.

        Returns
        -------
        dict:
            ``total_requests``, ``successful``, ``failed``,
            ``success_rate`` (0–1), ``avg_response_ms``,
            ``consecutive_failures``.
        """
        avg_ms = (
            sum(self._response_times) / len(self._response_times)
            if self._response_times
            else 0.0
        )
        total = self._total_requests
        cb = GlobalCircuitBreaker.get("nse_api", failure_threshold=5, recovery_timeout=120)
        return {
            "total_requests": total,
            "successful": self._successful,
            "failed": self._failed,
            "success_rate": self._successful / total if total > 0 else 0.0,
            "avg_response_ms": round(avg_ms, 1),
            "consecutive_failures": self._consecutive_failures,
            "circuit_breaker": cb.get_status(),
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "NSEScraper":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        stats = self.get_health_stats()
        return (
            f"NSEScraper(warmed={self._warmed_up}, "
            f"requests={stats['total_requests']}, "
            f"success_rate={stats['success_rate']:.0%})"
        )
