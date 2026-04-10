"""
Phase 4 — Alert Lifecycle Manager.

Manages the full lifecycle of anomaly alerts: creation → active → resolved/expired.
Provides filtering, history, statistics, and automatic stale-alert resolution.
"""

# ================================================================
# TABLE OWNERSHIP: This class is the SOLE writer for `anomaly_events`.
# No other module should INSERT, UPDATE, or DELETE from anomaly_events.
# Sub-detectors return AnomalyEvent objects → AlertManager persists them.
# ================================================================

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from zoneinfo import ZoneInfo

from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from src.analysis.anomaly.volume_price_detector import AnomalyEvent

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Auto-resolve windows by category (minutes)
# ---------------------------------------------------------------------------
_STALE_MINUTES: dict[str, int] = {
    "VOLUME": 30,
    "PRICE": 30,
    "OI": 60,
    "DIVERGENCE": 120,
    # FII alerts resolve after next trading day — handled specially
}

_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


# ---------------------------------------------------------------------------
# Statistics dataclass
# ---------------------------------------------------------------------------


@dataclass
class AlertStats:
    """Summary statistics for anomaly alerts over a given period."""

    total_alerts: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    avg_alerts_per_day: float = 0.0
    most_alerted_index: str = ""
    most_common_anomaly_type: str = ""
    false_positive_estimate: float = 0.0


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class AlertManager:
    """Manages the lifecycle of anomaly alerts.

    Thread-safe — uses a lock around the in-memory alert cache.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self._lock = threading.Lock()
        self._active_cache: dict[int, AnomalyEvent] = {}
        self._load_active_alerts()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _load_active_alerts(self) -> None:
        """Load currently active alerts from the database into memory.

        Also auto-resolves any stale alerts left over from a prior process
        that exited without cleaning up.  Resolution criteria:
        - VOLUME/PRICE alerts older than 30 minutes
        - OI alerts older than 60 minutes
        - DIVERGENCE alerts older than 120 minutes
        - ANY alert from a previous trading day
        Sets ``resolution_reason`` in the details JSON for audit trail.
        """
        rows = self.db.fetch_all(Q.LIST_ACTIVE_ANOMALIES)
        now = datetime.now(tz=_IST)
        today = now.date()
        loaded = 0
        stale_resolved = 0

        with self._lock:
            self._active_cache.clear()
            for row in rows:
                alert_id = row["id"]
                ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=_IST)
                category = row.get("category", "OTHER") or "OTHER"

                # Determine if stale: from a previous trading day OR exceeded TTL
                is_previous_day = ts.date() < today
                max_age = self._max_age_for_category(category, now)
                is_ttl_expired = (now - ts) > max_age

                if is_previous_day or is_ttl_expired:
                    stale_resolved += 1
                    reason = (
                        "auto_resolved_on_startup_previous_day"
                        if is_previous_day
                        else "auto_resolved_on_startup"
                    )
                    # Merge resolution_reason into details JSON
                    try:
                        details = json.loads(row.get("details") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        details = {}
                    details["resolution_reason"] = reason
                    details["resolved_at"] = now.isoformat()
                    self.db.execute(
                        Q.UPDATE_ANOMALY_RESOLVE,
                        (json.dumps(details), alert_id),
                    )
                    continue

                self._active_cache[alert_id] = AnomalyEvent(
                    index_id=row["index_id"],
                    timestamp=ts,
                    anomaly_type=row["anomaly_type"],
                    severity=row["severity"],
                    category=category,
                    details=row["details"],
                    message=row.get("message", "") or "",
                    is_active=True,
                    cooldown_key=f"{row['index_id']}_{row['anomaly_type']}",
                )
                loaded += 1

        if stale_resolved > 0:
            logger.info(
                "Resolved %d stale alerts from prior session on startup",
                stale_resolved,
            )
        logger.info("Loaded %d active alerts from database", loaded)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_alert(self, event: AnomalyEvent) -> int:
        """Insert a new alert into the database and cache.

        Returns the new alert row ID.
        """
        ts_str = event.timestamp.isoformat()
        cursor = self.db.execute(
            Q.INSERT_ANOMALY_EVENT,
            (
                event.index_id,
                ts_str,
                event.anomaly_type,
                event.severity,
                event.category,
                event.details,
                event.message,
                1,
            ),
        )
        alert_id = cursor.lastrowid or 0

        with self._lock:
            self._active_cache[alert_id] = event

        logger.info(
            "NEW ALERT [%s] %s %s: %s",
            event.severity,
            event.index_id,
            event.anomaly_type,
            event.message[:120],
        )
        if event.severity == "HIGH":
            logger.warning(
                "HIGH SEVERITY ALERT #%d — %s %s",
                alert_id,
                event.index_id,
                event.anomaly_type,
            )

        return alert_id

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------

    def resolve_alert(self, alert_id: int, resolution_reason: str) -> None:
        """Mark an alert as resolved with a reason."""
        with self._lock:
            event = self._active_cache.pop(alert_id, None)

        if event is None:
            logger.debug("Alert #%d not in active cache — resolving in DB only", alert_id)
            self.db.execute(Q.UPDATE_ANOMALY_DEACTIVATE, (alert_id,))
            return

        # Merge resolution reason into existing details JSON
        try:
            details = json.loads(event.details) if event.details else {}
        except (json.JSONDecodeError, TypeError):
            details = {}
        details["resolution_reason"] = resolution_reason
        details["resolved_at"] = datetime.now(tz=_IST).isoformat()

        self.db.execute(
            Q.UPDATE_ANOMALY_RESOLVE,
            (json.dumps(details), alert_id),
        )

        logger.info("RESOLVED ALERT #%d: %s", alert_id, resolution_reason)

    # ------------------------------------------------------------------
    # Auto-resolve stale alerts
    # ------------------------------------------------------------------

    def auto_resolve_stale_alerts(self) -> int:
        """Resolve alerts that have exceeded their category's time-to-live.

        Returns the number of alerts resolved.
        """
        now = datetime.now(tz=_IST)
        resolved_count = 0

        with self._lock:
            stale_ids: list[tuple[int, str]] = []
            for alert_id, event in self._active_cache.items():
                max_age = self._max_age_for_category(event.category, now)
                age = now - event.timestamp
                if age > max_age:
                    stale_ids.append((alert_id, event.category))

        for alert_id, category in stale_ids:
            self.resolve_alert(alert_id, f"Auto-resolved: {category} alert exceeded TTL")
            resolved_count += 1

        if resolved_count > 0:
            logger.info("Auto-resolved %d stale alerts", resolved_count)

        return resolved_count

    @staticmethod
    def _max_age_for_category(category: str, now: datetime) -> timedelta:
        """Return the maximum alert age before auto-resolution."""
        if category == "FII_DII":
            # Resolve after next trading day — approximate as 18 hours
            # (FII data arrives ~18:00, valid until next day ~15:30)
            return timedelta(hours=18)
        minutes = _STALE_MINUTES.get(category, 60)
        return timedelta(minutes=minutes)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_active_alerts(
        self,
        index_id: Optional[str] = None,
        category: Optional[str] = None,
        min_severity: Optional[str] = None,
    ) -> list[AnomalyEvent]:
        """Return active alerts, optionally filtered.

        Sorted by severity (HIGH first), then timestamp (newest first).
        """
        with self._lock:
            alerts = list(self._active_cache.values())

        if index_id is not None:
            alerts = [a for a in alerts if a.index_id == index_id]
        if category is not None:
            alerts = [a for a in alerts if a.category == category]
        if min_severity is not None:
            threshold = _SEVERITY_ORDER.get(min_severity, 2)
            alerts = [
                a for a in alerts if _SEVERITY_ORDER.get(a.severity, 2) <= threshold
            ]

        alerts.sort(
            key=lambda a: (
                _SEVERITY_ORDER.get(a.severity, 2),
                -a.timestamp.timestamp(),
            )
        )
        return alerts

    def get_active_alert_ids(self) -> dict[int, AnomalyEvent]:
        """Return a snapshot of the active cache {alert_id: event}."""
        with self._lock:
            return dict(self._active_cache)

    def get_alert_history(
        self,
        index_id: str,
        hours: int = 24,
    ) -> list[AnomalyEvent]:
        """Return all alerts (active + resolved) for an index over the last N hours."""
        cutoff = (datetime.now(tz=_IST) - timedelta(hours=hours)).isoformat()
        rows = self.db.fetch_all(Q.LIST_ANOMALY_HISTORY, (index_id, cutoff))

        events: list[AnomalyEvent] = []
        for row in rows:
            ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_IST)
            events.append(
                AnomalyEvent(
                    index_id=row["index_id"],
                    timestamp=ts,
                    anomaly_type=row["anomaly_type"],
                    severity=row["severity"],
                    category=row["category"],
                    details=row["details"],
                    message=row["message"],
                    is_active=bool(row["is_active"]),
                )
            )
        return events

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_alert_statistics(self, days: int = 7) -> AlertStats:
        """Compute summary statistics for anomaly alerts over the last N days."""
        cutoff = (datetime.now(tz=_IST) - timedelta(days=days)).isoformat()
        rows = self.db.fetch_all(Q.AGG_ANOMALY_STATS, (cutoff,))

        stats = AlertStats()
        index_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}

        for row in rows:
            count = row["total"]
            stats.total_alerts += count

            cat = row["category"]
            stats.by_category[cat] = stats.by_category.get(cat, 0) + count

            sev = row["severity"]
            stats.by_severity[sev] = stats.by_severity.get(sev, 0) + count

            idx = row["index_id"]
            index_counts[idx] = index_counts.get(idx, 0) + count

            atype = row["anomaly_type"]
            type_counts[atype] = type_counts.get(atype, 0) + count

        if days > 0:
            stats.avg_alerts_per_day = stats.total_alerts / days

        if index_counts:
            stats.most_alerted_index = max(index_counts, key=index_counts.get)  # type: ignore[arg-type]
        if type_counts:
            stats.most_common_anomaly_type = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]

        # False positive estimate: alerts that resolved within 5 minutes
        stats.false_positive_estimate = self._estimate_false_positives(cutoff)

        return stats

    def _estimate_false_positives(self, cutoff: str) -> float:
        """Fraction of resolved alerts that lived less than 5 minutes."""
        rows = self.db.fetch_all(Q.LIST_SHORT_LIVED_ALERTS, (cutoff,))
        if not rows:
            return 0.0

        total_resolved = len(rows)
        short_lived = 0

        for row in rows:
            try:
                details_str = row.get("details") or "{}"
                details = json.loads(details_str)
                resolved_at_str = details.get("resolved_at")
                if resolved_at_str:
                    resolved_at = pd.Timestamp(resolved_at_str).to_pydatetime()
                    created_at = pd.Timestamp(row["timestamp"]).to_pydatetime()
                    if (resolved_at - created_at) < timedelta(minutes=5):
                        short_lived += 1
            except Exception:
                continue

        return short_lived / total_resolved if total_resolved > 0 else 0.0

    @property
    def active_count(self) -> int:
        """Number of currently active alerts."""
        with self._lock:
            return len(self._active_cache)
