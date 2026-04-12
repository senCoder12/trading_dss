"""
Token-bucket rate limiter for controlling outbound HTTP request rates.

Each data source (NSE, BSE, yfinance, news) has its own limiter instance
to avoid IP bans.

The :class:`RateLimiter` implements a true token-bucket algorithm:
tokens are added at a constant rate up to a maximum bucket capacity.
This allows short bursts while enforcing a long-term average.

API
---
- :meth:`RateLimiter.acquire` — non-blocking: consume one token or fail
- :meth:`RateLimiter.wait_and_acquire` — blocking: wait until a token is available
- :meth:`RateLimiter.remaining` — query available tokens without consuming

Factory helpers: :func:`create_nse_limiter`, :func:`create_bse_limiter`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from threading import Lock
from typing import Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synchronous token-bucket limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.

    Tokens refill continuously at *max_requests* / *window_seconds* tokens per
    second, up to a maximum of *max_requests*.  The bucket starts full.

    Parameters
    ----------
    max_requests:
        Maximum number of requests (= bucket capacity) allowed per
        *window_seconds*.
    window_seconds:
        Rolling window duration in seconds (default 60).
    name:
        Identifier used in log messages.

    Examples
    --------
    ::

        limiter = create_nse_limiter()

        # Non-blocking: try to acquire
        if limiter.acquire():
            response = requests.get(url)

        # Blocking: wait for a token (use as context manager)
        with limiter:
            response = requests.get(url)
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        name: str = "default",
    ) -> None:
        self._max_tokens: float = float(max_requests)
        self._refill_rate: float = max_requests / window_seconds  # tokens/second
        self._tokens: float = float(max_requests)  # bucket starts full
        self._last_refill: float = time.monotonic()
        self._name = name
        self._lock = Lock()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _refill(self) -> None:
        """Refill the bucket based on elapsed time (must be called under lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    # ── Public API ────────────────────────────────────────────────────────────

    def acquire(self) -> bool:
        """
        Non-blocking: consume one token if available.

        Returns
        -------
        bool:
            ``True`` if a token was consumed; ``False`` if the bucket is empty.
        """
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                logger.debug("[%s] Token acquired (%.1f remaining)", self._name, self._tokens)
                return True
            logger.debug("[%s] Bucket empty — acquire() returned False", self._name)
            return False

    def wait_and_acquire(self) -> None:
        """
        Blocking: wait until a token is available, then consume it.

        Sleeps in short increments to avoid busy-waiting.
        """
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    logger.debug("[%s] Token acquired after wait", self._name)
                    return
                # Time until 1 token becomes available
                wait = (1.0 - self._tokens) / self._refill_rate

            logger.debug("[%s] Rate limit — sleeping %.3fs", self._name, wait)
            time.sleep(min(wait, 0.05))  # poll at most every 50ms

    def remaining(self) -> int:
        """
        Return the approximate number of tokens currently available.

        The value is floored and may be slightly stale by the time it is read.
        """
        with self._lock:
            self._refill()
            return int(self._tokens)

    # ── Context manager (blocking) ────────────────────────────────────────────

    def __enter__(self) -> "RateLimiter":
        self.wait_and_acquire()
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"RateLimiter(name={self._name!r}, "
            f"max={int(self._max_tokens)}, "
            f"remaining={self.remaining()})"
        )


# ---------------------------------------------------------------------------
# Async token-bucket limiter
# ---------------------------------------------------------------------------


class AsyncRateLimiter:
    """
    Asyncio-compatible token-bucket rate limiter.

    Drop-in async replacement for :class:`RateLimiter`. Use with ``async with``.

    Parameters
    ----------
    max_requests:
        Bucket capacity.
    window_seconds:
        Refill window in seconds (default 60).
    name:
        Identifier used in log messages.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        name: str = "default",
    ) -> None:
        self._max_tokens: float = float(max_requests)
        self._refill_rate: float = max_requests / window_seconds
        self._tokens: float = float(max_requests)
        self._last_refill: float = time.monotonic()
        self._name = name
        self._lock: Optional[asyncio.Lock] = None  # created lazily inside the event loop

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Suspend until a request token is available, then consume it."""
        async with self._get_lock():
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    logger.debug("[%s] Async token acquired", self._name)
                    return
                wait = (1.0 - self._tokens) / self._refill_rate
                logger.debug("[%s] Async rate limit — sleeping %.3fs", self._name, wait)
                await asyncio.sleep(min(wait, 0.05))

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    def __repr__(self) -> str:
        return f"AsyncRateLimiter(name={self._name!r}, max={int(self._max_tokens)})"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_nse_limiter(name: str = "nse") -> RateLimiter:
    """
    Return a :class:`RateLimiter` configured for NSE API calls.

    Limit: 25 requests per 60 seconds.
    """
    return RateLimiter(max_requests=25, window_seconds=60.0, name=name)


def create_bse_limiter(name: str = "bse") -> RateLimiter:
    """
    Return a :class:`RateLimiter` configured for BSE API calls.

    Limit: 15 requests per 60 seconds.
    """
    return RateLimiter(max_requests=15, window_seconds=60.0, name=name)


def create_async_nse_limiter(name: str = "nse_async") -> AsyncRateLimiter:
    """Async variant of :func:`create_nse_limiter`."""
    return AsyncRateLimiter(max_requests=25, window_seconds=60.0, name=name)


def create_async_bse_limiter(name: str = "bse_async") -> AsyncRateLimiter:
    """Async variant of :func:`create_bse_limiter`."""
    return AsyncRateLimiter(max_requests=15, window_seconds=60.0, name=name)


# ---------------------------------------------------------------------------
# Global per-domain rate limiter (singleton per domain)
# ---------------------------------------------------------------------------


class GlobalRateLimiter:
    """Single rate limiter per external domain, shared across all components.

    NSE sees one IP, not separate 'components'. All code hitting nseindia.com
    MUST acquire from the same limiter instance.
    """

    _instances: dict[str, RateLimiter] = {}
    _lock = threading.Lock()

    @classmethod
    def get(
        cls,
        domain: str,
        max_requests: int = 25,
        window_seconds: int = 60,
    ) -> RateLimiter:
        """Get or create a rate limiter for *domain*. Returns same instance for same domain."""
        with cls._lock:
            if domain not in cls._instances:
                cls._instances[domain] = RateLimiter(
                    max_requests, window_seconds, name=domain,
                )
                logger.info(
                    "GlobalRateLimiter: created limiter for %s (%d req/%ds)",
                    domain, max_requests, window_seconds,
                )
            return cls._instances[domain]

    @classmethod
    def reset_all(cls) -> None:
        """Reset all limiters (for testing)."""
        with cls._lock:
            cls._instances.clear()


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Prevents cascading failures when an external service is down.

    States:
    - CLOSED: Normal operation. Requests pass through.
    - OPEN: Service is down. Requests fail immediately without hitting the service.
    - HALF_OPEN: Testing if service recovered. One request allowed through.

    Transitions:
    - CLOSED -> OPEN: After *failure_threshold* consecutive failures.
    - OPEN -> HALF_OPEN: After *recovery_timeout* seconds.
    - HALF_OPEN -> CLOSED: If test request succeeds (*success_threshold* times).
    - HALF_OPEN -> OPEN: If test request fails (reset recovery timer).
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 120,
        success_threshold: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state: str = "CLOSED"
        self.failure_count: int = 0
        self.success_count: int = 0
        self.last_failure_time: datetime | None = None
        self.opened_at: datetime | None = None
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        """Check if a request should be allowed through."""
        with self._lock:
            if self.state == "CLOSED":
                return True

            if self.state == "OPEN":
                elapsed = (datetime.now() - self.opened_at).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                    logger.info(
                        "CircuitBreaker [%s]: OPEN -> HALF_OPEN (testing recovery)", self.name,
                    )
                    return True
                return False

            if self.state == "HALF_OPEN":
                return True

            return False

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self.failure_count = 0
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = "CLOSED"
                    logger.info(
                        "CircuitBreaker [%s]: HALF_OPEN -> CLOSED (service recovered)", self.name,
                    )

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == "HALF_OPEN":
                self.state = "OPEN"
                self.opened_at = datetime.now()
                logger.warning(
                    "CircuitBreaker [%s]: HALF_OPEN -> OPEN (recovery failed)", self.name,
                )
            elif self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.opened_at = datetime.now()
                logger.warning(
                    "CircuitBreaker [%s]: CLOSED -> OPEN (%d consecutive failures). "
                    "Will retry in %ds.",
                    self.name, self.failure_count, self.recovery_timeout,
                )

    def get_status(self) -> dict:
        """Return a status dict suitable for health dashboards."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state,
                "failure_count": self.failure_count,
                "last_failure": (
                    self.last_failure_time.isoformat() if self.last_failure_time else None
                ),
            }


class GlobalCircuitBreaker:
    """Singleton registry for :class:`CircuitBreaker` instances."""

    _instances: dict[str, CircuitBreaker] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, name: str, **kwargs: int) -> CircuitBreaker:
        """Get or create a circuit breaker by *name*."""
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = CircuitBreaker(name, **kwargs)
            return cls._instances[name]

    @classmethod
    def reset_all(cls) -> None:
        """Reset all breakers (for testing)."""
        with cls._lock:
            cls._instances.clear()


# ---------------------------------------------------------------------------
# Data Freshness Tracker
# ---------------------------------------------------------------------------

_IST = ZoneInfo(IST_TIMEZONE)


def _get_ist_now() -> datetime:
    return datetime.now(tz=_IST)


class DataFreshnessTracker:
    """Tracks when each data type was last successfully updated.

    Call :meth:`update` after every successful scrape.  Downstream consumers
    call :meth:`is_stale` to decide whether to trust the data.
    """

    # Maximum acceptable age per data type (seconds)
    _MAX_AGES: dict[str, float] = {
        "index_prices": 120,       # 2 min (normally updates every 60s)
        "options_chain": 360,      # 6 min (normally updates every 180s)
        "vix": 240,                # 4 min (normally updates every 120s)
        "fii_dii": 86400,          # 24 hours (updates daily)
    }

    def __init__(self) -> None:
        self._timestamps: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def update(self, data_type: str) -> None:
        """Record successful data fetch."""
        with self._lock:
            self._timestamps[data_type] = _get_ist_now()

    def get_age_seconds(self, data_type: str) -> float | None:
        """Get age of data in seconds. ``None`` if never fetched."""
        with self._lock:
            ts = self._timestamps.get(data_type)
            if ts is None:
                return None
            return (_get_ist_now() - ts).total_seconds()

    def is_stale(self, data_type: str, max_age_seconds: float) -> bool:
        """Check if data is older than *max_age_seconds*."""
        age = self.get_age_seconds(data_type)
        if age is None:
            return True  # Never fetched = stale
        return age > max_age_seconds

    def get_all_status(self) -> dict[str, dict]:
        """Return freshness status of all tracked data types."""
        with self._lock:
            result: dict[str, dict] = {}
            now = _get_ist_now()
            for key, ts in self._timestamps.items():
                age = (now - ts).total_seconds()
                result[key] = {
                    "last_update": ts.isoformat(),
                    "age_seconds": age,
                    "is_stale": age > self._MAX_AGES.get(key, 300),
                }
            return result


# Module-level singleton — import and use from anywhere.
freshness_tracker = DataFreshnessTracker()
