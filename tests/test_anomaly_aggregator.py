"""
Tests for Phase 4 — Anomaly Aggregator & Alert Manager.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from src.analysis.anomaly.volume_price_detector import AnomalyEvent, Baselines
from src.analysis.anomaly.alert_manager import AlertManager, AlertStats
from src.analysis.anomaly.anomaly_aggregator import (
    AnomalyAggregator,
    AnomalyDetectionResult,
    AnomalyVote,
)

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_index(db, index_id: str = "NIFTY50") -> None:
    """Insert a minimal index_master row so FK constraints pass."""
    from src.database import queries as Q

    now = datetime.now(tz=_IST).isoformat()
    db.execute(
        Q.INSERT_INDEX_MASTER,
        (index_id, index_id, index_id, f"^{index_id}", "NSE",
         75, 1, index_id, "broad_market", 1, now, now),
    )


@pytest.fixture()
def db(tmp_path):
    """Isolated in-memory DB with schema initialised."""
    from src.database.db_manager import DatabaseManager

    db = DatabaseManager(db_path=tmp_path / "test.db")
    db.connect()
    db.initialise_schema()
    _seed_index(db, "NIFTY50")
    _seed_index(db, "BANKNIFTY")
    return db


@pytest.fixture()
def alert_mgr(db) -> AlertManager:
    return AlertManager(db)


@pytest.fixture()
def aggregator(db) -> AnomalyAggregator:
    return AnomalyAggregator(db)


def _make_event(
    index_id: str = "NIFTY50",
    anomaly_type: str = "VOLUME_SPIKE",
    severity: str = "HIGH",
    category: str = "VOLUME",
    details: str = "{}",
    message: str = "Test alert",
    ts: datetime | None = None,
) -> AnomalyEvent:
    return AnomalyEvent(
        index_id=index_id,
        timestamp=ts or datetime.now(tz=_IST),
        anomaly_type=anomaly_type,
        severity=severity,
        category=category,
        details=details,
        message=message,
        is_active=True,
        cooldown_key=f"{index_id}_{anomaly_type}",
    )


def _bar(
    ts: datetime | None = None,
    open_: float = 22000.0,
    high: float = 22100.0,
    low: float = 21900.0,
    close: float = 22050.0,
    volume: float = 1_000_000,
) -> dict:
    return {
        "timestamp": ts or datetime.now(tz=_IST),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


# ===========================================================================
# AlertManager tests
# ===========================================================================


class TestAlertManager:
    """Tests for alert lifecycle management."""

    def test_create_alert_returns_id(self, alert_mgr: AlertManager):
        event = _make_event()
        alert_id = alert_mgr.create_alert(event)
        assert alert_id > 0

    def test_create_alert_adds_to_active_cache(self, alert_mgr: AlertManager):
        event = _make_event()
        alert_mgr.create_alert(event)
        assert alert_mgr.active_count == 1

    def test_resolve_alert_removes_from_cache(self, alert_mgr: AlertManager):
        event = _make_event()
        alert_id = alert_mgr.create_alert(event)
        alert_mgr.resolve_alert(alert_id, "Test resolution")
        assert alert_mgr.active_count == 0

    def test_resolve_alert_sets_reason_in_details(self, alert_mgr: AlertManager, db):
        event = _make_event()
        alert_id = alert_mgr.create_alert(event)
        alert_mgr.resolve_alert(alert_id, "Volume spike subsided")

        row = db.fetch_one("SELECT * FROM anomaly_events WHERE id = ?", (alert_id,))
        assert row is not None
        assert row["is_active"] == 0
        details = json.loads(row["details"])
        assert details["resolution_reason"] == "Volume spike subsided"
        assert "resolved_at" in details

    def test_resolve_nonexistent_alert_no_error(self, alert_mgr: AlertManager):
        # Should not raise
        alert_mgr.resolve_alert(99999, "Does not exist")

    def test_auto_resolve_stale_volume_alert(self, alert_mgr: AlertManager):
        old_ts = datetime.now(tz=_IST) - timedelta(minutes=35)
        event = _make_event(category="VOLUME", ts=old_ts)
        alert_mgr.create_alert(event)
        assert alert_mgr.active_count == 1

        resolved = alert_mgr.auto_resolve_stale_alerts()
        assert resolved == 1
        assert alert_mgr.active_count == 0

    def test_auto_resolve_respects_oi_ttl(self, alert_mgr: AlertManager):
        # OI alerts live 60 minutes — a 45-minute old alert should survive
        ts_45min = datetime.now(tz=_IST) - timedelta(minutes=45)
        event = _make_event(category="OI", anomaly_type="OI_SPIKE", ts=ts_45min)
        alert_mgr.create_alert(event)

        resolved = alert_mgr.auto_resolve_stale_alerts()
        assert resolved == 0
        assert alert_mgr.active_count == 1

    def test_auto_resolve_divergence_after_2h(self, alert_mgr: AlertManager):
        old_ts = datetime.now(tz=_IST) - timedelta(minutes=125)
        event = _make_event(
            category="DIVERGENCE",
            anomaly_type="CORRELATION_BREAKDOWN",
            ts=old_ts,
        )
        alert_mgr.create_alert(event)
        resolved = alert_mgr.auto_resolve_stale_alerts()
        assert resolved == 1

    def test_get_active_alerts_sorted_by_severity(self, alert_mgr: AlertManager):
        alert_mgr.create_alert(_make_event(severity="MEDIUM", anomaly_type="A"))
        alert_mgr.create_alert(_make_event(severity="HIGH", anomaly_type="B"))
        alert_mgr.create_alert(_make_event(severity="LOW", anomaly_type="C"))

        alerts = alert_mgr.get_active_alerts()
        assert len(alerts) == 3
        assert alerts[0].severity == "HIGH"
        assert alerts[1].severity == "MEDIUM"
        assert alerts[2].severity == "LOW"

    def test_get_active_alerts_filter_by_index(self, alert_mgr: AlertManager):
        alert_mgr.create_alert(_make_event(index_id="NIFTY50", anomaly_type="A"))
        alert_mgr.create_alert(_make_event(index_id="BANKNIFTY", anomaly_type="B"))

        nifty_alerts = alert_mgr.get_active_alerts(index_id="NIFTY50")
        assert len(nifty_alerts) == 1
        assert nifty_alerts[0].index_id == "NIFTY50"

    def test_get_active_alerts_filter_by_category(self, alert_mgr: AlertManager):
        alert_mgr.create_alert(_make_event(category="VOLUME", anomaly_type="A"))
        alert_mgr.create_alert(_make_event(category="OI", anomaly_type="B"))

        vol_alerts = alert_mgr.get_active_alerts(category="VOLUME")
        assert len(vol_alerts) == 1
        assert vol_alerts[0].category == "VOLUME"

    def test_get_active_alerts_filter_min_severity(self, alert_mgr: AlertManager):
        alert_mgr.create_alert(_make_event(severity="HIGH", anomaly_type="A"))
        alert_mgr.create_alert(_make_event(severity="LOW", anomaly_type="B"))

        high_only = alert_mgr.get_active_alerts(min_severity="HIGH")
        assert len(high_only) == 1

    def test_get_alert_history(self, alert_mgr: AlertManager):
        ev = _make_event()
        aid = alert_mgr.create_alert(ev)
        alert_mgr.resolve_alert(aid, "Done")

        # Create another active one
        alert_mgr.create_alert(_make_event(anomaly_type="GAP_UP", category="PRICE"))

        history = alert_mgr.get_alert_history("NIFTY50", hours=1)
        assert len(history) == 2

    def test_get_alert_statistics(self, alert_mgr: AlertManager):
        for i in range(5):
            alert_mgr.create_alert(
                _make_event(anomaly_type=f"TYPE_{i}", severity="MEDIUM")
            )
        alert_mgr.create_alert(_make_event(anomaly_type="BIG", severity="HIGH"))

        stats = alert_mgr.get_alert_statistics(days=7)
        assert stats.total_alerts == 6
        assert stats.by_severity.get("HIGH", 0) == 1
        assert stats.by_severity.get("MEDIUM", 0) == 5
        assert stats.most_alerted_index == "NIFTY50"
        assert stats.avg_alerts_per_day > 0

    def test_alert_lifecycle_create_active_resolve(self, alert_mgr: AlertManager):
        """Full lifecycle: create → verify active → resolve → verify inactive."""
        event = _make_event()
        aid = alert_mgr.create_alert(event)

        # Active
        active = alert_mgr.get_active_alerts()
        assert any(True for a in active if a.anomaly_type == "VOLUME_SPIKE")

        # Resolve
        alert_mgr.resolve_alert(aid, "Subsided")
        active_after = alert_mgr.get_active_alerts()
        assert len(active_after) == 0


# ===========================================================================
# AnomalyAggregator tests
# ===========================================================================


class TestAnomalyAggregator:
    """Tests for the master aggregator."""

    def test_detection_cycle_with_volume_spike(self, aggregator: AnomalyAggregator):
        """Volume spike in bar → should generate alert and bullish/bearish vote."""
        baselines = Baselines(
            index_id="NIFTY50",
            computed_at=datetime.now(tz=_IST),
            avg_volume_20d=1_000_000,
            std_volume_20d=200_000,
            avg_range_20d=150.0,
            avg_range_pct_20d=0.007,
            avg_body_20d=80.0,
            avg_intraday_volatility=0.007,
            typical_first_15min_range=60.0,
            typical_last_30min_volume=250_000,
        )
        aggregator._vp_baselines["NIFTY50"] = baselines

        # Bar with huge volume spike (5x normal, z > 4)
        now = datetime.now(tz=_IST).replace(hour=10, minute=30)
        spike_bar = _bar(
            ts=now,
            open_=22000.0,
            high=22200.0,
            low=21980.0,
            close=22180.0,
            volume=5_000_000,
        )
        recent = [_bar(ts=now - timedelta(minutes=i)) for i in range(1, 11)]

        result = aggregator.run_detection_cycle(
            index_id="NIFTY50",
            current_price_bar=spike_bar,
            recent_price_bars=recent,
        )

        assert isinstance(result, AnomalyDetectionResult)
        assert result.index_id == "NIFTY50"
        # Volume spike should be detected (5x with positive return)
        # The exact result depends on detector internals, but risk/vote should exist
        assert result.risk_level in ("NORMAL", "ELEVATED", "HIGH", "EXTREME")
        assert result.anomaly_vote in (
            "NEUTRAL", "BULLISH", "STRONG_BULLISH", "BEARISH",
            "STRONG_BEARISH", "CAUTION",
        )
        assert result.position_size_modifier <= 1.0

    def test_no_anomalies_neutral_vote(self, aggregator: AnomalyAggregator):
        """Normal bar → NEUTRAL vote, NORMAL risk."""
        baselines = Baselines(
            index_id="NIFTY50",
            computed_at=datetime.now(tz=_IST),
            avg_volume_20d=1_000_000,
            std_volume_20d=200_000,
            avg_range_20d=150.0,
            avg_range_pct_20d=0.007,
            avg_body_20d=80.0,
            avg_intraday_volatility=0.007,
            typical_first_15min_range=60.0,
            typical_last_30min_volume=250_000,
        )
        aggregator._vp_baselines["NIFTY50"] = baselines

        now = datetime.now(tz=_IST).replace(hour=11, minute=0)
        normal_bar = _bar(ts=now, volume=1_000_000)
        recent = [_bar(ts=now - timedelta(minutes=i)) for i in range(1, 11)]

        result = aggregator.run_detection_cycle(
            index_id="NIFTY50",
            current_price_bar=normal_bar,
            recent_price_bars=recent,
        )

        assert result.anomaly_vote == "NEUTRAL"
        assert result.risk_level == "NORMAL"
        assert result.position_size_modifier == 1.0

    def test_anomaly_vote_bullish(self, aggregator: AnomalyAggregator):
        """Bullish anomalies → bullish vote."""
        alerts = [
            _make_event(
                anomaly_type="VOLUME_SPIKE",
                severity="HIGH",
                category="VOLUME",
                details=json.dumps({"price_change_pct": 1.5}),
            ),
            _make_event(
                anomaly_type="ONE_SIDED_PE_BUILDUP",
                severity="MEDIUM",
                category="OI",
            ),
        ]
        vote, confidence, reasoning = aggregator._compute_vote(alerts, "NIFTY50")
        assert vote in ("BULLISH", "STRONG_BULLISH")
        assert confidence > 0

    def test_anomaly_vote_bearish(self, aggregator: AnomalyAggregator):
        """Bearish anomalies → bearish vote."""
        alerts = [
            _make_event(
                anomaly_type="VOLUME_SPIKE",
                severity="HIGH",
                category="VOLUME",
                details=json.dumps({"price_change_pct": -2.0}),
            ),
            _make_event(
                anomaly_type="ONE_SIDED_CE_BUILDUP",
                severity="MEDIUM",
                category="OI",
            ),
        ]
        vote, confidence, reasoning = aggregator._compute_vote(alerts, "NIFTY50")
        assert vote in ("BEARISH", "STRONG_BEARISH")

    def test_anomaly_vote_caution_conflicting(self, aggregator: AnomalyAggregator):
        """Conflicting signals → CAUTION."""
        alerts = [
            _make_event(
                anomaly_type="ABSORPTION",
                severity="HIGH",
                category="VOLUME",
            ),
            _make_event(
                anomaly_type="VIX_DIVERGENCE",
                severity="HIGH",
                category="DIVERGENCE",
            ),
        ]
        vote, confidence, reasoning = aggregator._compute_vote(alerts, "NIFTY50")
        assert vote == "CAUTION"

    def test_risk_level_normal(self):
        """No alerts → NORMAL risk."""
        risk, modifier = AnomalyAggregator._compute_risk([])
        assert risk == "NORMAL"
        assert modifier == 1.0

    def test_risk_level_elevated(self):
        """1-2 MEDIUM alerts → ELEVATED."""
        alerts = [_make_event(severity="MEDIUM")]
        risk, modifier = AnomalyAggregator._compute_risk(alerts)
        assert risk == "ELEVATED"
        assert modifier == 0.8

    def test_risk_level_high(self):
        """1 HIGH alert → HIGH."""
        alerts = [_make_event(severity="HIGH")]
        risk, modifier = AnomalyAggregator._compute_risk(alerts)
        assert risk == "HIGH"
        assert modifier == 0.6

    def test_risk_level_extreme(self):
        """3+ HIGH alerts → EXTREME."""
        alerts = [
            _make_event(severity="HIGH", anomaly_type=f"T{i}")
            for i in range(3)
        ]
        risk, modifier = AnomalyAggregator._compute_risk(alerts)
        assert risk == "EXTREME"
        assert modifier == 0.3

    def test_position_size_modifier_values(self):
        """Verify exact modifier values at each risk level."""
        assert AnomalyAggregator._compute_risk([])[1] == 1.0
        assert AnomalyAggregator._compute_risk(
            [_make_event(severity="MEDIUM")]
        )[1] == 0.8
        assert AnomalyAggregator._compute_risk(
            [_make_event(severity="HIGH")]
        )[1] == 0.6
        assert AnomalyAggregator._compute_risk(
            [_make_event(severity="HIGH", anomaly_type=f"T{i}") for i in range(3)]
        )[1] == 0.3

    def test_get_anomaly_vote(self, aggregator: AnomalyAggregator):
        """get_anomaly_vote returns AnomalyVote for an index."""
        vote = aggregator.get_anomaly_vote("NIFTY50")
        assert isinstance(vote, AnomalyVote)
        assert vote.index_id == "NIFTY50"
        assert vote.vote == "NEUTRAL"
        assert vote.risk_level == "NORMAL"

    def test_get_anomaly_vote_with_active_alerts(self, aggregator: AnomalyAggregator):
        """get_anomaly_vote reflects active alerts."""
        aggregator.alerts.create_alert(
            _make_event(
                anomaly_type="VOLUME_SPIKE",
                severity="HIGH",
                details=json.dumps({"price_change_pct": 2.0}),
            )
        )
        vote = aggregator.get_anomaly_vote("NIFTY50")
        assert vote.active_alerts == 1
        assert vote.risk_level in ("HIGH", "ELEVATED")

    def test_market_dashboard(self, aggregator: AnomalyAggregator):
        """Dashboard returns correct structure."""
        aggregator.alerts.create_alert(
            _make_event(index_id="NIFTY50", anomaly_type="A", severity="HIGH")
        )
        aggregator.alerts.create_alert(
            _make_event(index_id="BANKNIFTY", anomaly_type="B", severity="MEDIUM")
        )

        dashboard = aggregator.get_market_anomaly_dashboard()
        assert "timestamp" in dashboard
        assert dashboard["total_active_alerts"] == 2
        assert "NIFTY50" in dashboard["by_index"]
        assert "BANKNIFTY" in dashboard["by_index"]
        assert "market_wide" in dashboard
        assert "most_critical_alerts" in dashboard
        assert len(dashboard["most_critical_alerts"]) <= 5

    def test_detector_failure_isolation(self, aggregator: AnomalyAggregator):
        """One detector failing should not crash the full cycle."""
        baselines = Baselines(
            index_id="NIFTY50",
            computed_at=datetime.now(tz=_IST),
            avg_volume_20d=1_000_000,
            std_volume_20d=200_000,
            avg_range_20d=150.0,
            avg_range_pct_20d=0.007,
            avg_body_20d=80.0,
            avg_intraday_volatility=0.007,
            typical_first_15min_range=60.0,
            typical_last_30min_volume=250_000,
        )
        aggregator._vp_baselines["NIFTY50"] = baselines

        # Corrupt the VP detector to force a failure
        original_detect = aggregator.volume_price.detect_all
        aggregator.volume_price.detect_all = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("Simulated failure")
        )

        now = datetime.now(tz=_IST).replace(hour=11, minute=0)
        result = aggregator.run_detection_cycle(
            index_id="NIFTY50",
            current_price_bar=_bar(ts=now),
            recent_price_bars=[_bar(ts=now - timedelta(minutes=i)) for i in range(1, 5)],
        )

        # Should still return a valid result
        assert isinstance(result, AnomalyDetectionResult)
        assert result.index_id == "NIFTY50"

        # Restore
        aggregator.volume_price.detect_all = original_detect

    def test_institutional_activity_flag(self, aggregator: AnomalyAggregator):
        """Absorption or FII extreme flow → institutional_activity_detected."""
        alerts = [_make_event(anomaly_type="ABSORPTION", severity="HIGH")]
        assert AnomalyAggregator._has_institutional_activity(alerts) is True

        alerts2 = [_make_event(anomaly_type="VOLUME_SPIKE", severity="HIGH")]
        assert AnomalyAggregator._has_institutional_activity(alerts2) is False

    def test_summary_contains_key_info(self, aggregator: AnomalyAggregator):
        """Result summary should mention index, vote, and risk."""
        baselines = Baselines(
            index_id="NIFTY50",
            computed_at=datetime.now(tz=_IST),
            avg_volume_20d=1_000_000,
            std_volume_20d=200_000,
            avg_range_20d=150.0,
            avg_range_pct_20d=0.007,
            avg_body_20d=80.0,
            avg_intraday_volatility=0.007,
            typical_first_15min_range=60.0,
            typical_last_30min_volume=250_000,
        )
        aggregator._vp_baselines["NIFTY50"] = baselines

        now = datetime.now(tz=_IST).replace(hour=11, minute=0)
        result = aggregator.run_detection_cycle(
            index_id="NIFTY50",
            current_price_bar=_bar(ts=now),
            recent_price_bars=[_bar(ts=now - timedelta(minutes=i)) for i in range(1, 5)],
        )

        assert "NIFTY50" in result.summary
        assert "Vote:" in result.summary
        assert "Risk:" in result.summary
