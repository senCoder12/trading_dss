"""
System Watchdog — Phase 9.2 of the Trading Decision Support System.

Monitors all system components every 30 seconds and automatically recovers
from failures.  On repeated failure it enters safe mode (no new trades) and
sends a CRITICAL Telegram alert.
"""
from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from typing import Optional

from src.utils.date_utils import get_ist_now
from src.utils.market_hours import MarketHoursManager

logger = logging.getLogger(__name__)

_NETWORK_TIMEOUT_SECS = 5
_NSE_URL = "https://www.nseindia.com"

# Error-rate thresholds (errors per 60-second window)
_ERROR_RATE_WARN = 5
_ERROR_RATE_CRITICAL = 20


# ---------------------------------------------------------------------------
# HealthCheck result
# ---------------------------------------------------------------------------


@dataclass
class HealthCheck:
    """Result of a single component health check."""

    component: str
    status: str                # "OK" / "WARNING" / "CRITICAL" / "UNKNOWN"
    message: str
    timestamp: datetime
    details: Optional[dict]
    requires_action: bool      # True if auto-recovery should be attempted
    action_type: Optional[str] # "restart_component" / "clear_cache" / "alert_only"


# ---------------------------------------------------------------------------
# SystemWatchdog
# ---------------------------------------------------------------------------


class SystemWatchdog:
    """Monitors all system components and automatically recovers from failures.

    Runs as a background thread, checking every 30 seconds:
    - Is the data collector alive and collecting?
    - Is the decision engine generating signals?
    - Is the paper trading engine tracking positions?
    - Is the database accessible?
    - Is disk space sufficient?
    - Are there unhandled errors accumulating?
    - Can we reach NSE?

    On failure detection:
    - Attempts auto-recovery (restart component)
    - Logs the failure with full context
    - Sends Telegram alert
    - After ``max_recovery_attempts`` failed recovery attempts → sends CRITICAL
      alert and enters safe mode (sets ``paper_engine.pause_new_trades = True``).
    """

    def __init__(self, db, components: dict) -> None:
        """
        Parameters
        ----------
        db:
            DatabaseManager instance.
        components:
            dict with optional keys:

            ``'data_collector'``  DataCollector instance
            ``'decision_engine'`` DecisionEngine instance
            ``'paper_engine'``    PaperTradingEngine instance
            ``'telegram_bot'``    TelegramBot instance
            ``'api_server'``      True / False
        """
        self.db = db
        self.components = components
        self.check_interval = 30          # seconds between check cycles
        self.failure_counts: dict[str, int] = {}
        self.max_recovery_attempts = 3
        self.is_running = False
        self._thread: Optional[Thread] = None

        # Health history for trend detection / dashboard API
        self.health_log: list[dict] = []

        # Sliding-window error-rate counter
        self._error_timestamps: list[float] = []
        self._error_window_seconds: int = 60

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring thread."""
        self.is_running = True
        self._thread = Thread(
            target=self._monitor_loop, daemon=True, name="Watchdog"
        )
        self._thread.start()
        logger.info(
            "Watchdog started (interval=%ds, max_recovery=%d)",
            self.check_interval,
            self.max_recovery_attempts,
        )

    def stop(self) -> None:
        """Signal the monitoring thread to exit on its next wake-up."""
        self.is_running = False
        logger.info("Watchdog stop requested")

    def report_error(self, source: str = "unknown") -> None:
        """
        Record an error event from any component.

        Called by other parts of the system to increment the error-rate
        counter used by ``_check_error_rate()``.
        """
        self._error_timestamps.append(time.monotonic())
        logger.debug("Watchdog: error reported from '%s'", source)

    def get_health_summary(self) -> dict:
        """Return the most recent health snapshot (used by dashboard / API)."""
        if not self.health_log:
            return {"status": "UNKNOWN", "checks": []}
        return self.health_log[-1]

    # ------------------------------------------------------------------
    # Monitor loop
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        while self.is_running:
            try:
                checks = self._run_all_checks()
                self._process_results(checks)
                self._save_health_snapshot(checks)
            except Exception as exc:
                logger.error("Watchdog itself failed: %s", exc)

            time.sleep(self.check_interval)

    # ------------------------------------------------------------------
    # Individual health checks
    # ------------------------------------------------------------------

    def _run_all_checks(self) -> list[HealthCheck]:
        return [
            self._check_database(),
            self._check_data_freshness(),
            self._check_decision_engine(),
            self._check_paper_engine(),
            self._check_disk_space(),
            self._check_memory(),
            self._check_error_rate(),
            self._check_network(),
        ]

    # ── 1. Database ─────────────────────────────────────────────────────

    def _check_database(self) -> HealthCheck:
        """Verify the database is accessible."""
        try:
            self.db.execute("SELECT 1", ())
            return HealthCheck(
                "database", "OK", "Database accessible",
                get_ist_now(), None, False, None,
            )
        except Exception as exc:
            return HealthCheck(
                "database", "CRITICAL", f"Database unreachable: {exc}",
                get_ist_now(), {"error": str(exc)}, True, "alert_only",
            )

    # ── 2. Data freshness ───────────────────────────────────────────────

    def _check_data_freshness(self) -> HealthCheck:
        """Check if price data is still flowing during market hours."""
        from src.data.rate_limiter import freshness_tracker

        if not MarketHoursManager().is_market_open():
            return HealthCheck(
                "data_freshness", "OK",
                "Market closed — data check skipped",
                get_ist_now(), None, False, None,
            )

        price_age = freshness_tracker.get_age_seconds("index_prices")

        if price_age is None:
            return HealthCheck(
                "data_freshness", "CRITICAL",
                "No price data received since startup",
                get_ist_now(), {"age": None}, True, "restart_component",
            )

        if price_age > 300:
            return HealthCheck(
                "data_freshness", "CRITICAL",
                f"Price data is {price_age:.0f}s old (limit: 300s)",
                get_ist_now(), {"age": price_age}, True, "restart_component",
            )

        if price_age > 120:
            return HealthCheck(
                "data_freshness", "WARNING",
                f"Price data is {price_age:.0f}s old (limit: 120s)",
                get_ist_now(), {"age": price_age}, False, "alert_only",
            )

        return HealthCheck(
            "data_freshness", "OK",
            f"Price data is {price_age:.0f}s old",
            get_ist_now(), {"age": price_age}, False, None,
        )

    # ── 3. Decision engine ──────────────────────────────────────────────

    def _check_decision_engine(self) -> HealthCheck:
        """Check if the decision engine has cycled recently during market hours."""
        engine = self.components.get("decision_engine")
        if engine is None:
            return HealthCheck(
                "decision_engine", "UNKNOWN",
                "No decision_engine in components",
                get_ist_now(), None, False, None,
            )

        if not MarketHoursManager().is_market_open():
            return HealthCheck(
                "decision_engine", "OK",
                "Market closed — engine check skipped",
                get_ist_now(), None, False, None,
            )

        # Support both public and private naming conventions
        last_cycle: Optional[datetime] = (
            getattr(engine, "last_cycle_time", None)
            or getattr(engine, "_last_cycle_time", None)
        )

        if last_cycle is None:
            return HealthCheck(
                "decision_engine", "WARNING",
                "Decision engine has not completed any cycle yet",
                get_ist_now(), None, False, "alert_only",
            )

        now = get_ist_now()
        try:
            age_secs = (now - last_cycle).total_seconds()
        except TypeError:
            # Naive vs aware comparison
            return HealthCheck(
                "decision_engine", "OK",
                "Decision engine cycled (timezone mismatch — skipping age check)",
                get_ist_now(), None, False, None,
            )

        if age_secs > 600:
            return HealthCheck(
                "decision_engine", "CRITICAL",
                f"Decision engine last cycled {age_secs:.0f}s ago (limit: 600s)",
                get_ist_now(), {"age_seconds": age_secs}, True, "alert_only",
            )
        if age_secs > 300:
            return HealthCheck(
                "decision_engine", "WARNING",
                f"Decision engine last cycled {age_secs:.0f}s ago (limit: 300s)",
                get_ist_now(), {"age_seconds": age_secs}, False, "alert_only",
            )

        return HealthCheck(
            "decision_engine", "OK",
            f"Decision engine cycled {age_secs:.0f}s ago",
            get_ist_now(), {"age_seconds": age_secs}, False, None,
        )

    # ── 4. Paper engine ─────────────────────────────────────────────────

    def _check_paper_engine(self) -> HealthCheck:
        """Check that the paper trading engine is in a healthy state."""
        engine = self.components.get("paper_engine")
        if engine is None:
            return HealthCheck(
                "paper_engine", "UNKNOWN",
                "No paper_engine in components",
                get_ist_now(), None, False, None,
            )

        if getattr(engine, "_kill_switch_active", False):
            reason = getattr(engine, "_kill_switch_reason", "unknown")
            return HealthCheck(
                "paper_engine", "WARNING",
                f"Kill switch active: {reason}",
                get_ist_now(), {"reason": reason}, False, "alert_only",
            )

        if getattr(engine, "pause_new_trades", False):
            return HealthCheck(
                "paper_engine", "WARNING",
                "Paper engine paused (safe mode active)",
                get_ist_now(), None, False, "alert_only",
            )

        # Detect stale intraday positions (should never exceed 24 h)
        open_positions = getattr(engine, "open_positions", {})
        now = get_ist_now()
        stale = 0
        for pos in open_positions.values():
            entry_ts = getattr(pos, "entry_timestamp", None)
            if entry_ts is None:
                continue
            try:
                if (now - entry_ts).total_seconds() > 86_400:
                    stale += 1
            except TypeError:
                pass

        if stale:
            return HealthCheck(
                "paper_engine", "WARNING",
                f"{stale} stale open position(s) detected (>24h)",
                get_ist_now(), {"stale_count": stale}, True, "alert_only",
            )

        return HealthCheck(
            "paper_engine", "OK",
            f"Engine healthy — {len(open_positions)} open position(s)",
            get_ist_now(), {"open_positions": len(open_positions)}, False, None,
        )

    # ── 5. Disk space ───────────────────────────────────────────────────

    def _check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        import shutil

        total, _used, free = shutil.disk_usage(".")
        free_gb = free / (1024 ** 3)
        free_pct = free / total * 100

        if free_pct < 5 or free_gb < 1:
            return HealthCheck(
                "disk_space", "CRITICAL",
                f"Low disk: {free_gb:.1f} GB free ({free_pct:.0f}%)",
                get_ist_now(), {"free_gb": free_gb, "free_pct": free_pct},
                True, "alert_only",
            )

        if free_pct < 15:
            return HealthCheck(
                "disk_space", "WARNING",
                f"Disk filling: {free_gb:.1f} GB free ({free_pct:.0f}%)",
                get_ist_now(), {"free_gb": free_gb, "free_pct": free_pct},
                False, "alert_only",
            )

        return HealthCheck(
            "disk_space", "OK",
            f"{free_gb:.1f} GB free ({free_pct:.0f}%)",
            get_ist_now(), None, False, None,
        )

    # ── 6. Memory ───────────────────────────────────────────────────────

    def _check_memory(self) -> HealthCheck:
        """Check process RSS memory usage.  Requires psutil; skips if unavailable."""
        try:
            import psutil

            rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)

            if rss_mb > 2_000:
                return HealthCheck(
                    "memory", "CRITICAL", f"High memory: {rss_mb:.0f} MB",
                    get_ist_now(), {"rss_mb": rss_mb}, True, "alert_only",
                )
            if rss_mb > 1_000:
                return HealthCheck(
                    "memory", "WARNING", f"Memory: {rss_mb:.0f} MB",
                    get_ist_now(), {"rss_mb": rss_mb}, False, None,
                )
            return HealthCheck(
                "memory", "OK", f"{rss_mb:.0f} MB RSS",
                get_ist_now(), None, False, None,
            )
        except ImportError:
            return HealthCheck(
                "memory", "UNKNOWN", "psutil not available",
                get_ist_now(), None, False, None,
            )

    # ── 7. Error rate ───────────────────────────────────────────────────

    def _check_error_rate(self) -> HealthCheck:
        """Check if component errors are accumulating in the rolling window."""
        now_mono = time.monotonic()
        cutoff = now_mono - self._error_window_seconds
        self._error_timestamps = [t for t in self._error_timestamps if t > cutoff]
        count = len(self._error_timestamps)

        if count >= _ERROR_RATE_CRITICAL:
            return HealthCheck(
                "error_rate", "CRITICAL",
                f"{count} errors in last {self._error_window_seconds}s",
                get_ist_now(), {"count": count}, True, "alert_only",
            )
        if count >= _ERROR_RATE_WARN:
            return HealthCheck(
                "error_rate", "WARNING",
                f"{count} errors in last {self._error_window_seconds}s",
                get_ist_now(), {"count": count}, False, "alert_only",
            )

        return HealthCheck(
            "error_rate", "OK",
            f"{count} errors in last {self._error_window_seconds}s",
            get_ist_now(), None, False, None,
        )

    # ── 8. Network ──────────────────────────────────────────────────────

    def _check_network(self) -> HealthCheck:
        """Check network connectivity to NSE via a HEAD request."""
        try:
            req = urllib.request.Request(
                _NSE_URL,
                headers={"User-Agent": "Mozilla/5.0"},
                method="HEAD",
            )
            with urllib.request.urlopen(req, timeout=_NETWORK_TIMEOUT_SECS):
                pass
            return HealthCheck(
                "network", "OK", "NSE reachable",
                get_ist_now(), None, False, None,
            )
        except Exception as exc:
            return HealthCheck(
                "network", "WARNING",
                f"NSE unreachable: {type(exc).__name__}",
                get_ist_now(), {"error": str(exc)}, False, "alert_only",
            )

    # ------------------------------------------------------------------
    # Result processing
    # ------------------------------------------------------------------

    def _process_results(self, checks: list[HealthCheck]) -> None:
        for check in checks:
            if check.status == "CRITICAL" and check.requires_action:
                count = self.failure_counts.get(check.component, 0) + 1
                self.failure_counts[check.component] = count

                if count <= self.max_recovery_attempts:
                    logger.warning(
                        "Watchdog: %s CRITICAL — attempting recovery (%d/%d): %s",
                        check.component, count, self.max_recovery_attempts, check.message,
                    )
                    self._attempt_recovery(check)
                else:
                    logger.critical(
                        "Watchdog: %s — max recovery attempts exceeded. Entering safe mode.",
                        check.component,
                    )
                    self._enter_safe_mode(check)

            elif check.status in ("CRITICAL", "WARNING"):
                # CRITICAL without requires_action, or WARNING
                if check.action_type == "alert_only":
                    self._send_alert(
                        f"⚠️ WATCHDOG {check.status}: {check.component} — {check.message}"
                    )
                logger.warning(
                    "Watchdog: %s %s — %s",
                    check.component, check.status, check.message,
                )

            elif check.status == "OK":
                if check.component in self.failure_counts:
                    logger.info("Watchdog: %s recovered", check.component)
                    del self.failure_counts[check.component]

    def _attempt_recovery(self, check: HealthCheck) -> None:
        """Try to automatically recover a failed component."""
        if check.action_type == "restart_component":
            if check.component == "data_freshness":
                try:
                    collector = self.components.get("data_collector")
                    if collector:
                        scraper = getattr(collector, "nse_scraper", None)
                        refresh = (
                            getattr(scraper, "_refresh_session", None)
                            or getattr(scraper, "refresh_session", None)
                        )
                        if refresh:
                            refresh()
                            logger.info("Watchdog: NSE session refreshed")
                        else:
                            logger.warning(
                                "Watchdog: nse_scraper has no refresh_session method"
                            )
                except Exception as exc:
                    logger.error("Watchdog: NSE session refresh failed: %s", exc)

        self._send_alert(
            f"⚠️ WATCHDOG: {check.component} — {check.message}\n"
            f"Recovery attempted "
            f"({self.failure_counts.get(check.component, '?')}/{self.max_recovery_attempts})."
        )

    def _enter_safe_mode(self, check: HealthCheck) -> None:
        """When recovery fails repeatedly, halt new trades and alert."""
        self._send_alert(
            f"🚨 WATCHDOG CRITICAL: {check.component}\n"
            f"{check.message}\n"
            f"Recovery failed {self.max_recovery_attempts} times.\n"
            f"System entering SAFE MODE — no new trades.\n"
            f"Check system manually."
        )
        paper_engine = self.components.get("paper_engine")
        if paper_engine is not None:
            # NOTE: PaperTradingEngine must check `pause_new_trades` in on_signal()
            paper_engine.pause_new_trades = True
            logger.critical(
                "Watchdog: paper engine new-trade intake PAUSED (safe mode)"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_alert(self, message: str) -> None:
        telegram = self.components.get("telegram_bot")
        if telegram is None:
            return
        try:
            telegram.send_alert(message, priority="HIGH")
        except Exception as exc:
            logger.error("Watchdog: failed to send Telegram alert: %s", exc)

    def _save_health_snapshot(self, checks: list[HealthCheck]) -> None:
        """Append current check results to the bounded in-memory health log."""
        overall = "OK"
        if any(c.status == "CRITICAL" for c in checks):
            overall = "CRITICAL"
        elif any(c.status == "WARNING" for c in checks):
            overall = "WARNING"

        snapshot = {
            "timestamp": get_ist_now().isoformat(),
            "overall": overall,
            "checks": [
                {
                    "component": c.component,
                    "status": c.status,
                    "message": c.message,
                    "details": c.details,
                }
                for c in checks
            ],
        }
        self.health_log.append(snapshot)

        # Keep memory bounded: retain the last 500 snapshots (≈ 4 h at 30 s interval)
        if len(self.health_log) > 1_000:
            self.health_log = self.health_log[-500:]
