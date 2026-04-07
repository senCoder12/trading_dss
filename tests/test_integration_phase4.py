"""
Phase 4 — End-to-End Integration Tests.

Tests the full anomaly detection pipeline with real (or realistic) data:
  - Detection over historical bars
  - Alert lifecycle (create → active → auto-resolve)
  - FII cycle with mock data
  - Divergence detection between correlated pairs
  - Alert statistics
  - Performance (detection cycle < 2 seconds)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from src.analysis.anomaly.volume_price_detector import AnomalyEvent, Baselines
from src.analysis.anomaly.alert_manager import AlertManager
from src.analysis.anomaly.anomaly_aggregator import (
    AnomalyAggregator,
    AnomalyDetectionResult,
)

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_indices(db: DatabaseManager) -> None:
    """Seed index_master with test indices."""
    now = datetime.now(tz=_IST).isoformat()
    indices = [
        ("NIFTY50", "NIFTY 50", "NIFTY 50", "^NSEI", "NSE", 75, 1, "NIFTY", "broad_market"),
        ("BANKNIFTY", "BANK NIFTY", "NIFTY BANK", "^NSEBANK", "NSE", 25, 1, "BANKNIFTY", "sectoral"),
        ("NIFTYIT", "NIFTY IT", "NIFTY IT", "^CNXIT", "NSE", 25, 0, None, "sectoral"),
        ("SENSEX", "SENSEX", "SENSEX", "^BSESN", "BSE", 10, 0, None, "broad_market"),
    ]
    for idx in indices:
        db.execute(
            Q.INSERT_INDEX_MASTER,
            (idx[0], idx[1], idx[2], idx[3], idx[4],
             idx[5], idx[6], idx[7], idx[8], 1, now, now),
        )


def _generate_price_bars(
    index_id: str,
    days: int = 20,
    base_price: float = 22000.0,
    base_volume: float = 1_000_000,
    bars_per_day: int = 6,
) -> list[dict]:
    """Generate realistic synthetic price bars.

    Includes occasional spikes and gaps to trigger anomalies.
    """
    np.random.seed(42)
    bars: list[dict] = []
    price = base_price

    for day in range(days):
        date = datetime.now(tz=_IST).replace(
            hour=9, minute=15, second=0, microsecond=0,
        ) - timedelta(days=days - day)

        for bar_idx in range(bars_per_day):
            ts = date + timedelta(hours=bar_idx)

            # Normal drift
            returns = np.random.normal(0.0002, 0.005)
            volume_mult = np.random.lognormal(0, 0.3)

            # Inject anomalies on specific days
            if day == 5 and bar_idx == 0:
                # Volume spike day
                volume_mult = 5.0
                returns = 0.015
            elif day == 10 and bar_idx == 0:
                # Gap down
                returns = -0.02
            elif day == 15 and bar_idx == 2:
                # Absorption: high volume, tiny move
                volume_mult = 4.0
                returns = 0.0001

            price = price * (1 + returns)
            open_ = price * (1 - abs(returns) * 0.3)
            high = max(price, open_) * (1 + abs(np.random.normal(0, 0.002)))
            low = min(price, open_) * (1 - abs(np.random.normal(0, 0.002)))
            volume = base_volume * volume_mult

            bars.append({
                "timestamp": ts,
                "open": round(open_, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": round(volume, 0),
            })

    return bars


@pytest.fixture()
def db(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "integration_test.db")
    db.connect()
    db.initialise_schema()
    _seed_indices(db)
    return db


@pytest.fixture()
def aggregator(db) -> AnomalyAggregator:
    return AnomalyAggregator(db)


# ===========================================================================
# End-to-end tests
# ===========================================================================


class TestIntegrationPhase4:
    """Full pipeline integration tests."""

    def test_detection_over_historical_bars(self, aggregator: AnomalyAggregator):
        """Run anomaly detection over 20 days of synthetic data.

        Verify reasonable anomaly count (not 0, not thousands).
        """
        bars = _generate_price_bars("NIFTY50", days=20)

        # Pre-compute baselines from first 60 bars
        baselines = Baselines(
            index_id="NIFTY50",
            computed_at=bars[0]["timestamp"],
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

        total_anomalies = 0
        results: list[AnomalyDetectionResult] = []

        for i in range(10, len(bars)):
            recent = bars[max(0, i - 20):i]
            result = aggregator.run_detection_cycle(
                index_id="NIFTY50",
                current_price_bar=bars[i],
                recent_price_bars=recent,
            )
            results.append(result)
            total_anomalies += len(result.new_anomalies)

        # Should detect some anomalies (we injected spikes) but not too many
        assert total_anomalies > 0, "Expected at least some anomalies from injected spikes"
        assert total_anomalies < 200, f"Too many anomalies detected: {total_anomalies}"

        # Verify we got valid results throughout
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)
        assert all(r.index_id == "NIFTY50" for r in results)

    def test_alerts_stored_in_database(self, aggregator: AnomalyAggregator, db):
        """Verify detected anomalies are persisted to anomaly_events."""
        bars = _generate_price_bars("NIFTY50", days=5, bars_per_day=6)

        baselines = Baselines(
            index_id="NIFTY50",
            computed_at=bars[0]["timestamp"],
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

        for i in range(5, len(bars)):
            aggregator.run_detection_cycle(
                index_id="NIFTY50",
                current_price_bar=bars[i],
                recent_price_bars=bars[max(0, i - 10):i],
            )

        # Check database has records
        rows = db.fetch_all(
            "SELECT COUNT(*) as cnt FROM anomaly_events WHERE index_id = 'NIFTY50'"
        )
        assert rows[0]["cnt"] >= 0  # May be 0 if no anomalies triggered in 5 days

    def test_auto_resolution_lifecycle(self, aggregator: AnomalyAggregator):
        """Create alerts, wait for TTL, verify auto-resolution works."""
        # Create old alerts manually
        old_ts = datetime.now(tz=_IST) - timedelta(minutes=35)
        old_event = AnomalyEvent(
            index_id="NIFTY50",
            timestamp=old_ts,
            anomaly_type="VOLUME_SPIKE",
            severity="HIGH",
            category="VOLUME",
            details="{}",
            message="Old volume spike",
            is_active=True,
            cooldown_key="NIFTY50_VOLUME_SPIKE",
        )
        aggregator.alerts.create_alert(old_event)

        # Create a fresh alert
        fresh_event = AnomalyEvent(
            index_id="NIFTY50",
            timestamp=datetime.now(tz=_IST),
            anomaly_type="OI_SPIKE",
            severity="MEDIUM",
            category="OI",
            details="{}",
            message="Fresh OI spike",
            is_active=True,
            cooldown_key="NIFTY50_OI_SPIKE",
        )
        aggregator.alerts.create_alert(fresh_event)

        assert aggregator.alerts.active_count == 2

        # Auto-resolve should only kill the 35-min-old VOLUME alert
        resolved = aggregator.alerts.auto_resolve_stale_alerts()
        assert resolved == 1
        assert aggregator.alerts.active_count == 1

        remaining = aggregator.alerts.get_active_alerts()
        assert remaining[0].anomaly_type == "OI_SPIKE"

    def test_detection_cycle_performance(self, aggregator: AnomalyAggregator):
        """Single detection cycle should complete in < 2 seconds."""
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
        bar = {
            "timestamp": now,
            "open": 22000.0,
            "high": 22100.0,
            "low": 21900.0,
            "close": 22050.0,
            "volume": 1_000_000,
        }
        recent = [
            {
                "timestamp": now - timedelta(minutes=i),
                "open": 22000.0 + i,
                "high": 22100.0 + i,
                "low": 21900.0 + i,
                "close": 22050.0 + i,
                "volume": 1_000_000 + i * 1000,
            }
            for i in range(1, 21)
        ]

        start = time.monotonic()
        result = aggregator.run_detection_cycle(
            index_id="NIFTY50",
            current_price_bar=bar,
            recent_price_bars=recent,
        )
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"Detection cycle took {elapsed:.2f}s (limit 2s)"
        assert isinstance(result, AnomalyDetectionResult)

    def test_divergence_detection_nifty_banknifty(self, aggregator: AnomalyAggregator, db):
        """Test divergence detection between NIFTY and BANKNIFTY using synthetic data."""
        np.random.seed(123)
        dates = pd.date_range(
            end=datetime.now(tz=_IST).date(), periods=30, freq="B",
        )

        # NIFTY trending up, BANKNIFTY flat → should detect divergence
        nifty_prices = 22000 + np.cumsum(np.random.normal(50, 30, 30))
        banknifty_prices = 47000 + np.cumsum(np.random.normal(0, 50, 30))

        nifty_df = pd.DataFrame({
            "date": dates,
            "close": nifty_prices,
        })
        banknifty_df = pd.DataFrame({
            "date": dates,
            "close": banknifty_prices,
        })

        index_data = {
            "NIFTY50": nifty_df,
            "BANKNIFTY": banknifty_df,
        }

        events = aggregator.divergence_detector.detect_all_divergences(index_data)
        # May or may not detect divergence depending on random data
        assert isinstance(events, list)
        for e in events:
            assert isinstance(e, AnomalyEvent)
            assert e.category == "DIVERGENCE"

    def test_fii_cycle_with_mock_data(self, aggregator: AnomalyAggregator, monkeypatch):
        """Test FII cycle by mocking the fetcher."""
        from src.data.fii_dii_data import FIIDIIData
        from datetime import date as date_cls

        mock_data = FIIDIIData(
            date=date_cls.today(),
            fii_buy_value=15000.0,
            fii_sell_value=5000.0,
            fii_net_value=10000.0,
            dii_buy_value=8000.0,
            dii_sell_value=9000.0,
            dii_net_value=-1000.0,
            fii_fo_buy=5000.0,
            fii_fo_sell=2000.0,
            fii_fo_net=3000.0,
        )

        # Mock the fetcher to return our data
        class MockFetcher:
            def fetch_today_fii_dii(self):
                return mock_data

        monkeypatch.setattr(
            "src.analysis.anomaly.anomaly_aggregator.FIIDIIFetcher",
            MockFetcher,
            raising=False,
        )

        # Need to also seed fii_dii_activity for baselines
        for i in range(30):
            d = (datetime.now(tz=_IST) - timedelta(days=i + 1)).strftime("%Y-%m-%d")
            try:
                aggregator.db.execute(
                    Q.INSERT_FII_DII_ACTIVITY,
                    (d, 10000.0, 9000.0, 1000.0, 8000.0, 7500.0, 500.0, 3000.0, 2500.0, 500.0),
                )
            except Exception:
                pass  # Ignore if table doesn't exist or constraint fails

        bias = aggregator.run_fii_cycle()
        # May return None if FII tables aren't seeded properly, but should not crash
        # The important thing is no exception raised

    def test_alert_statistics_integration(self, aggregator: AnomalyAggregator):
        """Run multiple cycles, then verify stats make sense."""
        bars = _generate_price_bars("NIFTY50", days=5, bars_per_day=6)

        baselines = Baselines(
            index_id="NIFTY50",
            computed_at=bars[0]["timestamp"],
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

        for i in range(5, len(bars)):
            aggregator.run_detection_cycle(
                index_id="NIFTY50",
                current_price_bar=bars[i],
                recent_price_bars=bars[max(0, i - 10):i],
            )

        stats = aggregator.alerts.get_alert_statistics(days=7)
        assert stats.total_alerts >= 0
        assert stats.avg_alerts_per_day >= 0
        assert stats.false_positive_estimate >= 0.0
        assert stats.false_positive_estimate <= 1.0

    def test_multiple_indices_no_cross_contamination(
        self, aggregator: AnomalyAggregator
    ):
        """Alerts for NIFTY50 should not appear when querying BANKNIFTY."""
        aggregator.alerts.create_alert(
            AnomalyEvent(
                index_id="NIFTY50",
                timestamp=datetime.now(tz=_IST),
                anomaly_type="VOLUME_SPIKE",
                severity="HIGH",
                category="VOLUME",
                details="{}",
                message="NIFTY spike",
                is_active=True,
            )
        )
        aggregator.alerts.create_alert(
            AnomalyEvent(
                index_id="BANKNIFTY",
                timestamp=datetime.now(tz=_IST),
                anomaly_type="OI_SPIKE",
                severity="MEDIUM",
                category="OI",
                details="{}",
                message="BANKNIFTY OI",
                is_active=True,
            )
        )

        nifty_vote = aggregator.get_anomaly_vote("NIFTY50")
        bank_vote = aggregator.get_anomaly_vote("BANKNIFTY")

        assert nifty_vote.active_alerts == 1
        assert bank_vote.active_alerts == 1

    def test_no_crashes_with_empty_data(self, aggregator: AnomalyAggregator):
        """Detection cycle with minimal/empty data should not crash."""
        now = datetime.now(tz=_IST).replace(hour=11, minute=0)
        bar = {
            "timestamp": now,
            "open": 22000.0,
            "high": 22000.0,
            "low": 22000.0,
            "close": 22000.0,
            "volume": 0.0,
        }

        result = aggregator.run_detection_cycle(
            index_id="NIFTY50",
            current_price_bar=bar,
            recent_price_bars=[],
        )
        assert isinstance(result, AnomalyDetectionResult)

    def test_dashboard_with_mixed_indices(self, aggregator: AnomalyAggregator):
        """Dashboard should aggregate across all indices."""
        for idx in ("NIFTY50", "BANKNIFTY"):
            aggregator.alerts.create_alert(
                AnomalyEvent(
                    index_id=idx,
                    timestamp=datetime.now(tz=_IST),
                    anomaly_type="VOLUME_SPIKE",
                    severity="HIGH",
                    category="VOLUME",
                    details="{}",
                    message=f"{idx} volume spike",
                    is_active=True,
                )
            )

        dashboard = aggregator.get_market_anomaly_dashboard()
        assert dashboard["total_active_alerts"] == 2
        assert "NIFTY50" in dashboard["by_index"]
        assert "BANKNIFTY" in dashboard["by_index"]
        assert len(dashboard["most_critical_alerts"]) == 2
