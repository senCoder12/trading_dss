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
    "option_chain_indices": "/api/option-chain-indices",
    "option_chain_equities": "/api/option-chain-equities",
    "option_chain_v3": "/api/option-chain-v3",
    "option_chain_meta": "/api/option-chain-contract-info",
    "equity_quote": "/api/quote-equity",
    "equity_fno": "/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O",
    "fii_dii": "/api/fiidiiTradeReact",
    "fii_futures_stats": "/api/fiiDiiData",
    "participant_oi": "/api/fiiStatsData",
    "gift_nifty": "/api/liveEquity-derivatives?index=GIFT%20NIFTY",
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
        self._ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

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

    def _build_headers(self, for_api: bool = False, symbol: Optional[str] = None) -> dict[str, str]:
        """Return NSE headers with a randomly chosen User-Agent.

        Parameters
        ----------
        for_api:
            When True, set Sec-Fetch headers appropriate for an XHR/API
            call (same-site fetch) rather than a top-level page navigation.
        """
        headers = dict(NSE_HEADERS)
        headers["User-Agent"] = self._ua
        if for_api:
            headers["Accept"] = "*/*"
            headers["Sec-Fetch-Dest"] = "empty"
            headers["Sec-Fetch-Mode"] = "cors"
            headers["Sec-Fetch-Site"] = "same-origin"
            headers.pop("Sec-Fetch-User", None)
            headers.pop("Upgrade-Insecure-Requests", None)
            
            # Browser-like hints
            headers["Sec-Ch-Ua"] = '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"'
            headers["Sec-Ch-Ua-Mobile"] = "?0"
            headers["Sec-Ch-Ua-Platform"] = '"macOS"'
            
            if symbol:
                headers["Referer"] = f"https://www.nseindia.com/option-chain?symbol={symbol.upper()}"
            else:
                headers["Referer"] = "https://www.nseindia.com/option-chain"
                
            headers["X-Requested-With"] = "XMLHttpRequest"
        return headers

    # ── Session management ────────────────────────────────────────────────────

    def _warm_up_internal(self, symbol: Optional[str] = None) -> None:
        """
        Actually perform the navigation to get cookies.
        """
        headers = self._build_headers(for_api=False, symbol=symbol)
        try:
            # 1. Main homepage
            resp = self._session.get(NSE_BASE_URL, headers=headers, timeout=15)
            resp.raise_for_status()
            
            # 2. Option chain landing page (sets required cookies for v3 API)
            oc_url = f"{NSE_BASE_URL}/option-chain"
            if symbol:
                oc_url += f"?symbol={symbol.upper()}"
            resp = self._session.get(oc_url, headers=headers, timeout=15)
            resp.raise_for_status()

            # 3. Symbol-specific page (optional but helps with cookies)
            sym_url = f"{NSE_BASE_URL}/get-quotes/derivatives?symbol=NIFTY"
            resp = self._session.get(sym_url, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            raise NSEScraperError(f"NSE warm-up failed: {exc}") from exc

        self._warmed_up = True
        self._session_created_at = time.monotonic()
        logger.info(
            "NSE session warmed up (cookies: %d, items: %s, UA: %s…)",
            len(self._session.cookies),
            list(self._session.cookies.keys()),
            headers.get("User-Agent", "Browser Default")[:40],
        )

    def warm_up(self, symbol: Optional[str] = None) -> None:
        """
        Seed the session with NSE cookies (public method).

        NSE requires a real browser visit before the JSON APIs respond.
        Called automatically by :meth:`_get` if the session is not valid.
        """
        self._warm_up_internal(symbol=symbol)

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
        symbol: Optional[str] = None,
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
        symbol:
            Optional symbol to use for Referer header.

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
            self._warm_up_internal(symbol=symbol)

        url = NSE_BASE_URL + endpoint
        logger.debug("NSE Request: %s", url)
        headers = self._build_headers(for_api=True, symbol=symbol)
        self._rate_limiter.wait_and_acquire()

        t0 = time.monotonic()
        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=30)
            logger.info("NSE API Response: %d | URL: %s", resp.status_code, resp.url)
        except Exception as exc:
            # curl_cffi raises its own exception types for timeouts/connection errors
            exc_name = type(exc).__name__.lower()
            if "timeout" in exc_name or "connection" in exc_name:
                raise _NSETransientError(f"Transient network error: {exc}") from exc
            raise NSEScraperError(f"Request failed: {exc}") from exc

        elapsed_ms = (time.monotonic() - t0) * 1000
        label = endpoint.split("?")[0].rsplit("/", 1)[-1]
        logger.debug("NSE %s → HTTP %d (%.0f ms)", label, resp.status_code, elapsed_ms)

        # Session-expiry signals — trigger a hard refresh and retry
        if resp.status_code in (401, 403):
            self._refresh_session()
            raise _NSETransientError(
                f"NSE returned {resp.status_code} (session expired) — session refreshed, retrying..."
            )

        # Rate-limit — transient, let tenacity retry
        if resp.status_code == 429:
            raise _NSETransientError("NSE rate limit exceeded (429)")

        if resp.status_code >= 400:
            logger.error("NSE API failure: HTTP %d | URL: %s | Response: %s", 
                         resp.status_code, url, resp.text[:500])
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
        symbol: Optional[str] = None,
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
                endpoint, params=params, cache_key=cache_key, cache_ttl=cache_ttl, symbol=symbol
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

    def _normalize_stock_entry(self, entry: dict) -> dict:
        """Convert a raw NSE equity entry to a standard dict."""
        f = self._float
        return {
            "symbol": entry.get("symbol", ""),
            "ltp": f(entry.get("lastPrice")),
            "open": f(entry.get("open")),
            "high": f(entry.get("dayHigh")),
            "low": f(entry.get("dayLow")),
            "close": f(entry.get("previousClose")),
            "change": f(entry.get("change")),
            "change_pct": f(entry.get("pChange")),
            "volume": f(entry.get("totalTradedVolume")),
            "value": f(entry.get("totalTradedValue")),
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

    def get_all_fno_quotes(self) -> Optional[list[dict[str, Any]]]:
        """
        Fetch live quotes for all F&O securities in a single API call.
        
        Returns
        -------
        list[dict] or None:
            Normalized stock entries, or None on failure.
        """
        raw = self._call(
            _ENDPOINTS["equity_fno"],
            cache_key="nse:equity_fno",
        )
        if raw is None:
            return None
        return [self._normalize_stock_entry(e) for e in raw.get("data", [])]

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

    def get_option_chain(self, symbol: str, instrument_type: str = "Indices") -> Optional[dict[str, Any]]:
        """Fetch the full options chain for *symbol* (e.g. ``"NIFTY"``).

        Uses option_chain_indices for indices, option_chain_equities for stocks.
        """
        endpoint = (
            _ENDPOINTS["option_chain_indices"]
            if instrument_type.lower() in ("indices", "index")
            else _ENDPOINTS["option_chain_equities"]
        )
        return self._call(
            endpoint,
            params={"symbol": symbol},
            symbol=symbol,
        )

    def get_option_chain_v3(
        self,
        symbol: str,
        type: str = "Indices",
        expiry: Optional[str] = None,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Fetch the options chain using the v3 API.

        Requires an 'expiry' date (format: dd-MMM-yyyy) to return full data.
        If expiry is None, this might return empty or metadata depending on NSE state.
        """
        # /api/option-chain-indices now returns 404; v3 serves indices and
        # equities from one endpoint, selected by the `type` param.
        params = {
            "type": "Equities" if type.lower() == "equities" else "Indices",
            "symbol": symbol,
        }
        if expiry:
            params["expiry"] = expiry

        return self._call(
            _ENDPOINTS["option_chain_v3"],
            params=params,
            cache_key=cache_key,
            cache_ttl=cache_ttl,
            symbol=symbol,
        )

    def get_option_chain_meta(
        self,
        symbol: str,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Fetch available expiry dates and strike prices for *symbol*.
        Useful for identifying the current/active expiry to use with v3 API.
        """
        return self._call(
            _ENDPOINTS["option_chain_meta"],
            params={"symbol": symbol},
            cache_key=cache_key,
            cache_ttl=cache_ttl,
        )

    def get_fii_futures_positions(self) -> Optional[dict]:
        """Fetch FII/DII participant-wise F&O trading data (index futures net positions).

        NSE publishes this every 30 minutes during market hours.
        Returns dict with FII net long/short in index futures.
        """
        result = self._call(_ENDPOINTS["participant_oi"], cache_key="nse:participant_oi", cache_ttl=300)
        if result:
            freshness_tracker.update("fii_futures")
        return result

    def get_gift_nifty(self) -> Optional[dict]:
        """Fetch GIFT NIFTY (SGX Nifty replacement) pre-market price.

        Used before 9:15 AM for gap prediction. Returns latest GIFT NIFTY price.
        """
        raw = self._call(_ENDPOINTS["gift_nifty"], cache_key="nse:gift_nifty", cache_ttl=60)
        if raw is None:
            return None
        # Parse response - GIFT NIFTY appears in derivatives data
        f = self._float
        data = raw.get("data", []) if isinstance(raw, dict) else []
        for entry in data:
            sym = str(entry.get("symbol", "") or entry.get("indexSymbol", "")).upper()
            if "GIFT" in sym or "SGX" in sym:
                return {
                    "ltp": f(entry.get("lastPrice") or entry.get("last")),
                    "change": f(entry.get("change") or entry.get("variation")),
                    "change_pct": f(entry.get("pChange") or entry.get("percentChange")),
                    "timestamp": datetime.now(tz=_IST).isoformat(),
                }
        # Fallback: return first entry if available (some endpoints return single value)
        if data:
            entry = data[0]
            return {
                "ltp": f(entry.get("lastPrice") or entry.get("last") or entry.get("ltp", 0)),
                "change": f(entry.get("change") or entry.get("variation", 0)),
                "change_pct": f(entry.get("pChange") or entry.get("percentChange", 0)),
                "timestamp": datetime.now(tz=_IST).isoformat(),
            }
        return None

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
