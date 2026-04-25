"""
BSE India live data scraper.

Uses the BSE public JSON API (``api.bseindia.com``).  BSE has simpler
cookie requirements than NSE but the same rate-limiting and caching
patterns apply.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.constants import BSE_BASE_URL, BSE_HEADERS, IST_TIMEZONE, USER_AGENT_ROTATION
from config.settings import settings
from src.data.rate_limiter import (
    GlobalCircuitBreaker,
    GlobalRateLimiter,
    RateLimiter,
    create_bse_limiter,
)
from src.utils.cache import TTLCache

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BSE_API_BASE = "https://api.bseindia.com/BseIndiaAPI/api"
_CACHE_TTL_PRICE: int = 30
_CACHE_TTL_STATUS: int = 60
_MAX_RESPONSE_TIMES: int = 100

_ENDPOINTS = {
    "sensex": f"{_BSE_API_BASE}/SensexView/w",
    "all_indices": f"{_BSE_API_BASE}/BseIndices/w",
    "index_data": f"{_BSE_API_BASE}/IndicesData/w",
    "market_status": f"{_BSE_API_BASE}/marketStatus/w",
    "index_members": f"{_BSE_API_BASE}/MSectors/w",
    "option_chain_expiry": f"{_BSE_API_BASE}/ddlExpiry_Options/w",
    "option_chain": f"{_BSE_API_BASE}/getoptionchain/w",
}

_BSE_SCRIP_CODES: dict[str, str] = {
    "SENSEX": "1",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BSEScraperError(Exception):
    """Raised for BSE scraper failures."""


class _BSETransientError(BSEScraperError):
    """Internal — transient failure retried by tenacity."""


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class BSEScraper:
    """
    Scraper for BSE India public JSON API.

    Parameters
    ----------
    rate_limiter:
        Shared :class:`~src.data.rate_limiter.RateLimiter` instance.
        Defaults to :func:`~src.data.rate_limiter.create_bse_limiter` (15 req/60s).
    cache:
        :class:`~src.utils.cache.TTLCache` for caching responses.
        Defaults to a new 30-second cache.

    Examples
    --------
    ::

        with BSEScraper() as bse:
            quote = bse.get_sensex_quote()
            indices = bse.get_all_indices()
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[TTLCache] = None,
    ) -> None:
        self._session = requests.Session()
        self._rate_limiter = rate_limiter or GlobalRateLimiter.get(
            "bseindia.com", max_requests=15, window_seconds=60,
        )
        self._cache: TTLCache = cache or TTLCache(default_ttl=_CACHE_TTL_PRICE)

        # Health tracking
        self._total_requests: int = 0
        self._successful: int = 0
        self._failed: int = 0
        self._response_times: list[float] = []
        self._consecutive_failures: int = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        headers = dict(BSE_HEADERS)
        headers["User-Agent"] = random.choice(USER_AGENT_ROTATION)
        return headers

    @staticmethod
    def _float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    # ── Core HTTP ─────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(_BSETransientError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _get(
        self,
        url: str,
        params: Optional[dict] = None,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
    ) -> Any:
        """
        Rate-limited, cached GET with exponential-backoff retry on transient errors.

        Parameters
        ----------
        url:
            Full URL (BSE API uses absolute URLs).
        params:
            Optional query parameters.
        cache_key:
            Cache store key; skipped if ``None``.
        cache_ttl:
            Override default cache TTL.

        Returns
        -------
        Any:
            Parsed JSON.

        Raises
        ------
        BSEScraperError:
            On non-transient HTTP failure.
        _BSETransientError:
            On transient failure; retried by tenacity.
        """
        if cache_key:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        headers = self._build_headers()
        self._rate_limiter.wait_and_acquire()

        t0 = time.monotonic()
        try:
            resp = self._session.get(url, headers=headers, params=params, timeout=10)
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise _BSETransientError(f"Transient network error: {exc}") from exc
        except requests.RequestException as exc:
            raise BSEScraperError(f"Request failed ({url}): {exc}") from exc

        elapsed_ms = (time.monotonic() - t0) * 1000
        label = url.rsplit("/", 1)[-1].split("?")[0]
        logger.debug("BSE %s → HTTP %d (%.0f ms)", label, resp.status_code, elapsed_ms)

        if resp.status_code == 429:
            raise _BSETransientError("BSE rate limit exceeded (429)")

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise BSEScraperError(f"HTTP error from BSE ({url}): {exc}") from exc

        try:
            data = resp.json()
        except Exception as exc:
            raise _BSETransientError(f"JSON parse error: {exc}") from exc

        self._response_times.append(elapsed_ms)
        if len(self._response_times) > _MAX_RESPONSE_TIMES:
            self._response_times.pop(0)

        if cache_key:
            self._cache.set(cache_key, data, ttl=cache_ttl)

        return data

    def _call(
        self,
        url: str,
        params: Optional[dict] = None,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
    ) -> Optional[Any]:
        """Wrapper tracking health stats, honouring the circuit breaker;
        returns ``None`` instead of raising."""
        cb = GlobalCircuitBreaker.get("bse_api", failure_threshold=5, recovery_timeout=60)

        if not cb.can_execute():
            logger.warning("BSE circuit breaker OPEN — skipping request for %s", url)
            return None

        self._total_requests += 1
        try:
            result = self._get(url, params=params, cache_key=cache_key, cache_ttl=cache_ttl)
            self._successful += 1
            self._consecutive_failures = 0
            cb.record_success()
            return result
        except BSEScraperError as exc:
            self._failed += 1
            self._consecutive_failures += 1
            cb.record_failure()
            if self._consecutive_failures >= 5:
                logger.critical(
                    "BSE scraper: %d consecutive failures — health ERROR. Last: %s",
                    self._consecutive_failures,
                    exc,
                )
            else:
                logger.error("BSE request to %s failed: %s", url, exc)
            return None

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _normalize_sensex_entry(self, raw: dict) -> dict:
        """Normalise a ``SensexView`` response to the standard format."""
        f = self._float
        return {
            "index_name": raw.get("Index", "SENSEX"),
            "ltp": f(raw.get("Value")),
            "open": f(raw.get("Open")),
            "high": f(raw.get("High")),
            "low": f(raw.get("Low")),
            "close": f(raw.get("PrevClose")),
            "change": f(raw.get("Change")),
            "change_pct": f(raw.get("PerChange")),
            "timestamp": datetime.now(tz=_IST).isoformat(),
        }

    def _normalize_bse_index(self, raw: dict) -> dict:
        """Normalise an entry from the ``BseIndices`` list response."""
        f = self._float
        return {
            "index_name": raw.get("indxName", raw.get("IndexName", "")),
            "ltp": f(raw.get("currValue", raw.get("CurrValue"))),
            "open": f(raw.get("open", raw.get("Open"))),
            "high": f(raw.get("high", raw.get("High"))),
            "low": f(raw.get("low", raw.get("Low"))),
            "close": f(raw.get("prevClose", raw.get("PrevClose"))),
            "change": f(raw.get("chg", raw.get("Change"))),
            "change_pct": f(raw.get("pChg", raw.get("PerChange"))),
            "timestamp": datetime.now(tz=_IST).isoformat(),
        }

    # ── Public data methods ───────────────────────────────────────────────────

    def get_sensex_quote(self) -> Optional[dict[str, Any]]:
        """
        Fetch live SENSEX quote.

        Returns
        -------
        dict or None:
            Normalised SENSEX quote, or ``None`` on failure.
        """
        raw = self._call(_ENDPOINTS["sensex"], cache_key="bse:sensex")
        if raw is None:
            return None
        return self._normalize_sensex_entry(raw)

    def get_all_indices(self) -> Optional[list[dict[str, Any]]]:
        """
        Fetch live quotes for all BSE indices.

        Returns
        -------
        list[dict] or None:
            Normalised index entries, or ``None`` on failure.
        """
        raw = self._call(_ENDPOINTS["all_indices"], cache_key="bse:all_indices")
        if raw is None:
            return None

        # BSE may return a list or a dict with a list under a key
        entries = raw if isinstance(raw, list) else raw.get("Table", raw.get("data", []))
        return [self._normalize_bse_index(e) for e in entries]

    def get_index_quote(self, bse_symbol: str) -> Optional[dict[str, Any]]:
        """
        Fetch a specific BSE index quote.

        First checks ``get_all_indices()`` (cached).  Falls back to the
        ``IndicesData`` endpoint if not found in the bulk response.

        Parameters
        ----------
        bse_symbol:
            BSE index name or code (e.g. ``"BSE500"``, ``"SENSEX"``).

        Returns
        -------
        dict or None:
            Normalised quote, or ``None`` on failure / not found.
        """
        # Try bulk first (avoids an extra request)
        all_indices = self.get_all_indices()
        if all_indices is not None:
            target = bse_symbol.strip().upper()
            for entry in all_indices:
                if entry.get("index_name", "").upper() == target:
                    return entry

        # Fallback: individual endpoint
        raw = self._call(
            _ENDPOINTS["index_data"],
            params={"index": bse_symbol},
            cache_key=f"bse:index:{bse_symbol}",
        )
        if raw is None:
            return None
        return self._normalize_bse_index(raw if isinstance(raw, dict) else {})

    def get_market_status(self) -> Optional[dict[str, Any]]:
        """
        Fetch BSE market status.

        Returns
        -------
        dict or None:
            Raw BSE market status, or ``None`` on failure.
        """
        return self._call(
            _ENDPOINTS["market_status"],
            cache_key="bse:market_status",
            cache_ttl=_CACHE_TTL_STATUS,
        )

    def get_option_chain_meta(self, symbol: str) -> Optional[dict[str, Any]]:
        """
        Fetch available expiry dates for a BSE F&O symbol (e.g., "SENSEX").

        Returns
        -------
        dict or None:
            Dict with `expiryDates: list[str]` (formatted dd-MMM-yyyy), or ``None`` on failure.
        """
        scrip_cd = _BSE_SCRIP_CODES.get(symbol.upper())
        if not scrip_cd:
            return None

        raw = self._call(
            _ENDPOINTS["option_chain_expiry"],
            params={"Scrip_cd": scrip_cd, "type": "Index"},
            cache_key=f"bse:optmeta:{symbol.upper()}",
            cache_ttl=300,
        )
        if not raw:
            return None

        expiry_list = raw if isinstance(raw, list) else raw.get("Table", [])
        expiry_dates = []
        for e in expiry_list:
            exp_str = e.get("ExpiryDate", "")
            if not exp_str:
                continue
            try:
                parsed = self._parse_bse_date(exp_str)
                if parsed:
                    expiry_dates.append(parsed)
            except Exception:
                continue
        return {"expiryDates": expiry_dates}

    def get_option_chain(self, symbol: str, expiry: str) -> Optional[dict[str, Any]]:
        """
        Fetch the full BSE options chain for a symbol + expiry.

        Parameters
        ----------
        symbol:
            BSE F&O symbol e.g. "SENSEX"
        expiry:
            Expiry date in BSE format (ddMONYYYY), e.g. "25APR2025"

        Returns
        -------
        dict or None:
            Normalized options chain in NSE-compatible shape with ``records`` key, or ``None``.
        """
        scrip_cd = _BSE_SCRIP_CODES.get(symbol.upper())
        if not scrip_cd:
            return None

        raw = self._call(
            _ENDPOINTS["option_chain"],
            params={
                "scrip_cd": scrip_cd,
                "type": "Index",
                "expiry": expiry,
                "optiontype": "Both",
                "strikeprice": "",
            },
            cache_key=f"bse:optchain:{symbol.upper()}:{expiry}",
            cache_ttl=60,
        )
        if not raw:
            return None
        return self._normalize_option_chain(symbol, raw)

    def _parse_bse_date(self, date_str: str) -> Optional[str]:
        """
        Parse BSE date string to dd-MMM-yyyy format for compatibility with NSE parser.
        Accepts ddMONYYYY (e.g., 25APR2025), dd-MON-YYYY, or ISO datetime.
        """
        if not date_str:
            return None

        from datetime import datetime

        date_str = date_str.strip()

        formats = ["%d%b%Y", "%d-%b-%Y", "%d-%B-%Y"]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.upper(), fmt)
                return dt.strftime("%d-%b-%Y")
            except ValueError:
                continue

        try:
            dt = datetime.fromisoformat(date_str.split("T")[0])
            return dt.strftime("%d-%b-%Y")
        except (ValueError, AttributeError):
            return None

    def _normalize_option_chain(self, symbol: str, raw: dict) -> dict:
        """
        Normalize BSE options chain JSON to NSE-compatible shape.
        Converts BSE field names to match `OptionsChainFetcher` expectations.
        """
        f = self._float

        underlying = f(raw.get("UnderlyingValue", raw.get("UndVal", 0.0)))

        expiry_table = raw.get("Table1", raw.get("Table2", []))
        if not isinstance(expiry_table, list):
            expiry_table = []
        expiry_dates = []
        for e in expiry_table:
            exp_str = e.get("ExpiryDate", "")
            if exp_str:
                parsed = self._parse_bse_date(exp_str)
                if parsed:
                    expiry_dates.append(parsed)

        strikes_table = raw.get("Table", [])
        if not isinstance(strikes_table, list):
            strikes_table = []

        data = []
        for s in strikes_table:
            strike_price = f(s.get("StrikePrice", 0.0))
            expiry_str = s.get("ExpiryDate", "")
            if expiry_str:
                expiry_str = self._parse_bse_date(expiry_str) or expiry_str

            ce = {
                "openInterest": int(f(s.get("OI_CE", s.get("CE_OI", 0.0)))),
                "changeinOpenInterest": int(f(s.get("OIChg_CE", s.get("CE_OIChg", 0.0)))),
                "pchangeinOpenInterest": f(s.get("POIChg_CE", s.get("CE_POIChg", 0.0))),
                "totalTradedVolume": int(f(s.get("Vol_CE", s.get("CE_VOL", s.get("Volume_CE", 0.0))))),
                "impliedVolatility": f(s.get("IV_CE", s.get("CE_IV", 0.0))),
                "lastPrice": f(s.get("LTP_CE", s.get("CE_LTP", 0.0))),
                "change": f(s.get("Chng_CE", s.get("CE_PriceChng", s.get("CE_Chng", 0.0)))),
                "pChange": f(s.get("PChng_CE", s.get("CE_ChngPcnt", s.get("CE_PChng", 0.0)))),
                "totalBuyQuantity": int(f(s.get("BidQty_CE", s.get("CE_BidQty", 0.0)))),
                "totalSellQuantity": int(f(s.get("AskQty_CE", s.get("CE_AskQty", 0.0)))),
                "bidQty": int(f(s.get("BidQty_CE", s.get("CE_BidQty", 0.0)))),
                "bidprice": f(s.get("BidPrice_CE", s.get("CE_BidPrice", 0.0))),
                "askPrice": f(s.get("AskPrice_CE", s.get("CE_AskPrice", 0.0))),
                "askQty": int(f(s.get("AskQty_CE", s.get("CE_AskQty", 0.0)))),
                "expiryDate": expiry_str,
            }

            pe = {
                "openInterest": int(f(s.get("OI_PE", s.get("PE_OI", 0.0)))),
                "changeinOpenInterest": int(f(s.get("OIChg_PE", s.get("PE_OIChg", 0.0)))),
                "pchangeinOpenInterest": f(s.get("POIChg_PE", s.get("PE_POIChg", 0.0))),
                "totalTradedVolume": int(f(s.get("Vol_PE", s.get("PE_VOL", s.get("Volume_PE", 0.0))))),
                "impliedVolatility": f(s.get("IV_PE", s.get("PE_IV", 0.0))),
                "lastPrice": f(s.get("LTP_PE", s.get("PE_LTP", 0.0))),
                "change": f(s.get("Chng_PE", s.get("PE_PriceChng", s.get("PE_Chng", 0.0)))),
                "pChange": f(s.get("PChng_PE", s.get("PE_ChngPcnt", s.get("PE_PChng", 0.0)))),
                "totalBuyQuantity": int(f(s.get("BidQty_PE", s.get("PE_BidQty", 0.0)))),
                "totalSellQuantity": int(f(s.get("AskQty_PE", s.get("PE_AskQty", 0.0)))),
                "bidQty": int(f(s.get("BidQty_PE", s.get("PE_BidQty", 0.0)))),
                "bidprice": f(s.get("BidPrice_PE", s.get("PE_BidPrice", 0.0))),
                "askPrice": f(s.get("AskPrice_PE", s.get("PE_AskPrice", 0.0))),
                "askQty": int(f(s.get("AskQty_PE", s.get("PE_AskQty", 0.0)))),
                "expiryDate": expiry_str,
            }

            data.append({
                "strikePrice": strike_price,
                "expiryDate": expiry_str,
                "CE": ce,
                "PE": pe,
            })

        return {
            "records": {
                "underlyingValue": underlying,
                "expiryDates": expiry_dates,
                "data": data,
            }
        }

    # ── Health ────────────────────────────────────────────────────────────────

    def get_health_stats(self) -> dict[str, Any]:
        """
        Return scraper health metrics.

        Returns
        -------
        dict:
            ``total_requests``, ``successful``, ``failed``,
            ``success_rate``, ``avg_response_ms``, ``consecutive_failures``.
        """
        avg_ms = (
            sum(self._response_times) / len(self._response_times)
            if self._response_times
            else 0.0
        )
        total = self._total_requests
        cb = GlobalCircuitBreaker.get("bse_api", failure_threshold=5, recovery_timeout=60)
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

    def __enter__(self) -> "BSEScraper":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        stats = self.get_health_stats()
        return (
            f"BSEScraper(requests={stats['total_requests']}, "
            f"success_rate={stats['success_rate']:.0%})"
        )
