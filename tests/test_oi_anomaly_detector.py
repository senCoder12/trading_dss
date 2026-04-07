"""
Tests for Phase 4 -- Step 4.2: OI & Options Anomaly Detector.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from src.analysis.anomaly.oi_anomaly_detector import (
    OIAnomalyDetector,
    OIAnomaly,
    OIBaselines,
    OIAnomalySummary,
)
from src.analysis.anomaly.volume_price_detector import AnomalyEvent
from src.data.options_chain import OptionsChainData, OptionStrike, OISummary

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_baselines(**overrides) -> OIBaselines:
    defaults = dict(
        index_id="NIFTY50",
        avg_total_oi=5_000_000.0,
        std_total_oi=500_000.0,
        avg_oi_change_per_3min=20_000.0,
        std_oi_change_per_3min=8_000.0,
        avg_pcr=1.0,
        std_pcr=0.15,
        avg_max_pain_shift_per_day=50.0,
        typical_oi_at_atm=100_000.0,
        typical_iv_range=(12.0, 20.0),
    )
    defaults.update(overrides)
    return OIBaselines(**defaults)


def _make_summary(
    total_ce_oi: int = 2_500_000,
    total_pe_oi: int = 2_500_000,
    pcr: float = 1.0,
    max_pain_strike: float = 22000.0,
    **overrides,
) -> OISummary:
    return OISummary(
        total_ce_oi=total_ce_oi,
        total_pe_oi=total_pe_oi,
        total_ce_oi_change=overrides.get("total_ce_oi_change", 0),
        total_pe_oi_change=overrides.get("total_pe_oi_change", 0),
        total_ce_volume=overrides.get("total_ce_volume", 100_000),
        total_pe_volume=overrides.get("total_pe_volume", 100_000),
        pcr=pcr,
        pcr_change=overrides.get("pcr_change", 0.0),
        max_pain_strike=max_pain_strike,
        highest_ce_oi_strike=overrides.get("highest_ce_oi_strike", 22500.0),
        highest_pe_oi_strike=overrides.get("highest_pe_oi_strike", 21500.0),
        top_5_ce_oi_strikes=overrides.get("top_5_ce_oi_strikes", ()),
        top_5_pe_oi_strikes=overrides.get("top_5_pe_oi_strikes", ()),
    )


def _make_strike(
    strike_price: float,
    ce_oi: int = 10_000,
    pe_oi: int = 10_000,
    ce_iv: float = 16.0,
    pe_iv: float = 16.0,
    ce_oi_change: int = 0,
    pe_oi_change: int = 0,
) -> OptionStrike:
    return OptionStrike(
        strike_price=strike_price,
        ce_oi=ce_oi,
        ce_oi_change=ce_oi_change,
        ce_volume=5000,
        ce_ltp=100.0,
        ce_iv=ce_iv,
        pe_oi=pe_oi,
        pe_oi_change=pe_oi_change,
        pe_volume=5000,
        pe_ltp=100.0,
        pe_iv=pe_iv,
    )


def _make_chain(
    strikes: tuple[OptionStrike, ...] | None = None,
    spot_price: float = 22000.0,
    expiry_date: date | None = None,
) -> OptionsChainData:
    if strikes is None:
        strikes = tuple(
            _make_strike(sp)
            for sp in [21500, 21750, 22000, 22250, 22500]
        )
    return OptionsChainData(
        index_id="NIFTY50",
        spot_price=spot_price,
        timestamp=datetime.now(tz=_IST),
        expiry_date=expiry_date or date(2024, 6, 20),
        strikes=strikes,
        available_expiries=(date(2024, 6, 20),),
    )


@pytest.fixture()
def detector(tmp_path) -> OIAnomalyDetector:
    from src.database.db_manager import DatabaseManager
    from src.database import queries as Q

    db = DatabaseManager(db_path=tmp_path / "test.db")
    db.connect()
    db.initialise_schema()

    now = datetime.now(tz=_IST).isoformat()
    db.execute(
        Q.INSERT_INDEX_MASTER,
        ("NIFTY50", "NIFTY 50", "NIFTY 50", "^NSEI", "NSE",
         75, 1, "NIFTY", "broad_market", 1, now, now),
    )
    return OIAnomalyDetector(db)


@pytest.fixture()
def baselines() -> OIBaselines:
    return _make_baselines()


# ---------------------------------------------------------------------------
# OI spike detection
# ---------------------------------------------------------------------------


class TestOISpikeDetection:
    """Tests for detect_oi_spike."""

    def test_massive_oi_spike(self, detector, baselines):
        """Z > 4.0 triggers MASSIVE_OI_SPIKE."""
        prev = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        # Change of 60_000 → z = (60000 - 20000) / 8000 = 5.0
        curr = _make_summary(total_ce_oi=2_530_000, total_pe_oi=2_530_000)
        results = detector.detect_oi_spike("NIFTY50", curr, prev, baselines)
        massive = [r for r in results if r.anomaly_type == "MASSIVE_OI_SPIKE"]
        assert len(massive) == 1
        assert massive[0].severity == "HIGH"
        assert massive[0].oi_change_zscore >= 4.0

    def test_oi_spike_medium(self, detector, baselines):
        """Z > 2.5 but < 4.0 triggers OI_SPIKE at MEDIUM."""
        prev = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        # Change of 45_000 → z = (45000 - 20000) / 8000 = 3.125
        curr = _make_summary(total_ce_oi=2_522_500, total_pe_oi=2_522_500)
        results = detector.detect_oi_spike("NIFTY50", curr, prev, baselines)
        spikes = [r for r in results if r.anomaly_type == "OI_SPIKE"]
        assert len(spikes) == 1
        assert spikes[0].severity == "MEDIUM"

    def test_strike_oi_concentration(self, detector, baselines):
        """Single strike OI jumping >20% triggers STRIKE_OI_CONCENTRATION."""
        prev_strikes = (
            _make_strike(22000, ce_oi=10_000, pe_oi=10_000),
            _make_strike(22250, ce_oi=5_000, pe_oi=5_000),
        )
        curr_strikes = (
            _make_strike(22000, ce_oi=13_000, pe_oi=10_000),  # 30% CE jump
            _make_strike(22250, ce_oi=5_000, pe_oi=5_000),
        )
        prev_chain = _make_chain(strikes=prev_strikes)
        curr_chain = _make_chain(strikes=curr_strikes)

        prev_sum = _make_summary()
        curr_sum = _make_summary()

        results = detector.detect_oi_spike(
            "NIFTY50", curr_sum, prev_sum, baselines,
            current_chain=curr_chain, previous_chain=prev_chain,
        )
        conc = [r for r in results if r.anomaly_type == "STRIKE_OI_CONCENTRATION"]
        assert len(conc) >= 1
        assert conc[0].affected_strikes[0]["oi_change_pct"] == 30.0

    def test_one_sided_ce_buildup(self, detector, baselines):
        """CE OI change >> PE OI change triggers ONE_SIDED_CE_BUILDUP."""
        prev = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        curr = _make_summary(total_ce_oi=2_505_000, total_pe_oi=2_500_200)
        results = detector.detect_oi_spike("NIFTY50", curr, prev, baselines)
        ce_buildup = [r for r in results if r.anomaly_type == "ONE_SIDED_CE_BUILDUP"]
        assert len(ce_buildup) == 1
        assert ce_buildup[0].directional_implication == "BEARISH"

    def test_one_sided_pe_buildup(self, detector, baselines):
        """PE OI change >> CE OI change triggers ONE_SIDED_PE_BUILDUP."""
        prev = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        curr = _make_summary(total_ce_oi=2_500_200, total_pe_oi=2_505_000)
        results = detector.detect_oi_spike("NIFTY50", curr, prev, baselines)
        pe_buildup = [r for r in results if r.anomaly_type == "ONE_SIDED_PE_BUILDUP"]
        assert len(pe_buildup) == 1
        assert pe_buildup[0].directional_implication == "BULLISH"

    def test_small_oi_strike_ignored(self, detector, baselines):
        """Strikes with OI < 1000 should not trigger STRIKE_OI_CONCENTRATION."""
        prev_strikes = (_make_strike(22000, ce_oi=500, pe_oi=500),)
        curr_strikes = (_make_strike(22000, ce_oi=800, pe_oi=500),)  # 60% but small absolute
        prev_chain = _make_chain(strikes=prev_strikes)
        curr_chain = _make_chain(strikes=curr_strikes)

        results = detector.detect_oi_spike(
            "NIFTY50", _make_summary(), _make_summary(), baselines,
            current_chain=curr_chain, previous_chain=prev_chain,
        )
        conc = [r for r in results if r.anomaly_type == "STRIKE_OI_CONCENTRATION"]
        assert len(conc) == 0

    def test_expiry_day_raises_thresholds(self, detector, baselines):
        """On expiry day, thresholds are 2x higher."""
        prev = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        # Change of 45_000 → z = 3.125 — normally triggers OI_SPIKE
        # With 2x threshold: needs z > 5.0 for MEDIUM, so this should NOT trigger
        curr = _make_summary(total_ce_oi=2_522_500, total_pe_oi=2_522_500)
        results = detector.detect_oi_spike(
            "NIFTY50", curr, prev, baselines, is_expiry=True,
        )
        spikes = [r for r in results if r.anomaly_type in ("OI_SPIKE", "MASSIVE_OI_SPIKE")]
        assert len(spikes) == 0


# ---------------------------------------------------------------------------
# PCR anomaly detection
# ---------------------------------------------------------------------------


class TestPCRAnomalyDetection:
    """Tests for detect_pcr_anomaly."""

    def test_pcr_extreme_high(self, detector, baselines):
        """PCR z > 2.0 triggers PCR_EXTREME_HIGH."""
        # PCR = 1.8, avg = 1.0, std = 0.15 → z = 5.33
        results = detector.detect_pcr_anomaly("NIFTY50", 1.8, [], baselines)
        high = [r for r in results if r.anomaly_type == "PCR_EXTREME_HIGH"]
        assert len(high) == 1
        assert high[0].directional_implication == "BULLISH"

    def test_pcr_extreme_low(self, detector, baselines):
        """PCR z < -2.0 triggers PCR_EXTREME_LOW."""
        # PCR = 0.6, avg = 1.0, std = 0.15 → z = -2.67
        results = detector.detect_pcr_anomaly("NIFTY50", 0.6, [], baselines)
        low = [r for r in results if r.anomaly_type == "PCR_EXTREME_LOW"]
        assert len(low) == 1
        assert low[0].directional_implication == "BEARISH"

    def test_pcr_rapid_shift(self, detector, baselines):
        """PCR moving >0.15 within 10 snapshots triggers PCR_RAPID_SHIFT."""
        history = [0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.02, 1.05, 1.08, 1.10]
        current_pcr = 1.15  # shift = 1.15 - 0.95 = 0.20
        results = detector.detect_pcr_anomaly("NIFTY50", current_pcr, history, baselines)
        rapid = [r for r in results if r.anomaly_type == "PCR_RAPID_SHIFT"]
        assert len(rapid) == 1
        assert rapid[0].severity == "HIGH"

    def test_pcr_extreme_level_high(self, detector, baselines):
        """PCR > 1.5 triggers PCR_EXTREME (contrarian)."""
        # Use baselines with high std so z-score check doesn't also fire
        bl = _make_baselines(avg_pcr=1.4, std_pcr=0.5)
        results = detector.detect_pcr_anomaly("NIFTY50", 1.6, [], bl)
        extreme = [r for r in results if r.anomaly_type == "PCR_EXTREME"]
        assert len(extreme) == 1
        assert extreme[0].directional_implication == "BULLISH"

    def test_pcr_extreme_level_low(self, detector, baselines):
        """PCR < 0.5 triggers PCR_EXTREME (contrarian)."""
        bl = _make_baselines(avg_pcr=0.6, std_pcr=0.5)
        results = detector.detect_pcr_anomaly("NIFTY50", 0.4, [], bl)
        extreme = [r for r in results if r.anomaly_type == "PCR_EXTREME"]
        assert len(extreme) == 1
        assert extreme[0].directional_implication == "BEARISH"


# ---------------------------------------------------------------------------
# Max pain anomaly detection
# ---------------------------------------------------------------------------


class TestMaxPainAnomalyDetection:
    """Tests for detect_max_pain_anomaly."""

    def test_max_pain_jump(self, detector):
        """Max pain shifting >1% in a session triggers MAX_PAIN_JUMP."""
        # 200 point shift on 20000 = 1.0%... use 22000 → 22300 = 1.36%
        results = detector.detect_max_pain_anomaly(
            "NIFTY50",
            current_max_pain=22300.0,
            previous_max_pain=22000.0,
            spot_price=22100.0,
        )
        jumps = [r for r in results if r.anomaly_type == "MAX_PAIN_JUMP"]
        assert len(jumps) == 1
        assert jumps[0].severity == "HIGH"
        assert jumps[0].directional_implication == "BULLISH"

    def test_spot_max_pain_divergence(self, detector):
        """Spot >2% away from max pain triggers SPOT_MAX_PAIN_DIVERGENCE."""
        results = detector.detect_max_pain_anomaly(
            "NIFTY50",
            current_max_pain=22000.0,
            previous_max_pain=22000.0,
            spot_price=22500.0,  # 2.27% above max pain
            days_to_expiry=1,
        )
        div = [r for r in results if r.anomaly_type == "SPOT_MAX_PAIN_DIVERGENCE"]
        assert len(div) == 1
        assert div[0].details["pull_direction"] == "DOWN"
        assert div[0].details["convergence_probability"] == 0.75

    def test_no_anomaly_when_close(self, detector):
        """No anomaly when spot is close to max pain."""
        results = detector.detect_max_pain_anomaly(
            "NIFTY50",
            current_max_pain=22000.0,
            previous_max_pain=21990.0,
            spot_price=22050.0,
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# IV anomaly detection
# ---------------------------------------------------------------------------


class TestIVAnomalyDetection:
    """Tests for detect_iv_anomaly."""

    def test_iv_crush(self, detector, baselines):
        """ATM IV dropping >20% from baseline triggers IV_CRUSH."""
        # Baseline mid = (12+20)/2 = 16. IV at 12 → drop = 25%
        chain = _make_chain(
            strikes=(_make_strike(22000, ce_iv=12.0, pe_iv=12.0),),
        )
        results = detector.detect_iv_anomaly("NIFTY50", chain, baselines)
        crush = [r for r in results if r.anomaly_type == "IV_CRUSH"]
        assert len(crush) == 1
        assert crush[0].severity == "MEDIUM"

    def test_iv_explosion(self, detector, baselines):
        """ATM IV spiking >30% above baseline triggers IV_EXPLOSION."""
        # Baseline mid = 16. IV at 22 → spike = 37.5%
        chain = _make_chain(
            strikes=(_make_strike(22000, ce_iv=22.0, pe_iv=22.0),),
        )
        results = detector.detect_iv_anomaly("NIFTY50", chain, baselines)
        explosion = [r for r in results if r.anomaly_type == "IV_EXPLOSION"]
        assert len(explosion) == 1
        assert explosion[0].severity == "HIGH"

    def test_iv_skew_extreme(self, detector, baselines):
        """Put IV >> Call IV triggers IV_SKEW_EXTREME."""
        chain = _make_chain(
            strikes=(_make_strike(22000, ce_iv=10.0, pe_iv=25.0),),
        )
        results = detector.detect_iv_anomaly("NIFTY50", chain, baselines)
        skew = [r for r in results if r.anomaly_type == "IV_SKEW_EXTREME"]
        assert len(skew) == 1
        assert skew[0].directional_implication == "BEARISH"

    def test_iv_term_inversion(self, detector, baselines):
        """ATM IV significantly above OTM IV triggers IV_TERM_INVERSION."""
        strikes = (
            _make_strike(21000, ce_iv=10.0, pe_iv=10.0),
            _make_strike(21500, ce_iv=11.0, pe_iv=11.0),
            _make_strike(21750, ce_iv=12.0, pe_iv=12.0),
            _make_strike(22000, ce_iv=25.0, pe_iv=25.0),  # ATM — high IV
            _make_strike(22250, ce_iv=12.0, pe_iv=12.0),
            _make_strike(22500, ce_iv=10.0, pe_iv=10.0),
            _make_strike(23000, ce_iv=9.0, pe_iv=9.0),
            _make_strike(23500, ce_iv=8.0, pe_iv=8.0),
        )
        chain = _make_chain(strikes=strikes)
        # Baseline with no range so IV crush/explosion don't fire
        bl = _make_baselines(typical_iv_range=(0.0, 0.0))
        results = detector.detect_iv_anomaly("NIFTY50", chain, bl)
        inv = [r for r in results if r.anomaly_type == "IV_TERM_INVERSION"]
        assert len(inv) == 1

    def test_no_iv_anomaly_when_normal(self, detector, baselines):
        """No IV anomaly when IV is within normal range."""
        chain = _make_chain(
            strikes=(_make_strike(22000, ce_iv=16.0, pe_iv=16.0),),
        )
        results = detector.detect_iv_anomaly("NIFTY50", chain, baselines)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Cooldown system
# ---------------------------------------------------------------------------


class TestCooldownSystem:
    """Tests for cooldown suppression."""

    def test_cooldown_suppresses_duplicate(self, detector, baselines):
        """Same anomaly detected twice within cooldown window is suppressed."""
        prev = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        curr = _make_summary(total_ce_oi=2_530_000, total_pe_oi=2_530_000)

        # First detection
        events1 = detector.detect_all_oi_anomalies(
            "NIFTY50", None, curr, prev, baselines,
        )
        massive1 = [e for e in events1 if e.anomaly_type == "MASSIVE_OI_SPIKE"]

        # Second detection (within cooldown)
        events2 = detector.detect_all_oi_anomalies(
            "NIFTY50", None, curr, prev, baselines,
        )
        massive2 = [e for e in events2 if e.anomaly_type == "MASSIVE_OI_SPIKE"]

        assert len(massive1) >= 1
        assert len(massive2) == 0  # suppressed by cooldown

    def test_cooldown_expires(self, detector, baselines):
        """After cooldown expires, same anomaly is detected again."""
        prev = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        curr = _make_summary(total_ce_oi=2_530_000, total_pe_oi=2_530_000)

        # First detection
        events1 = detector.detect_all_oi_anomalies(
            "NIFTY50", None, curr, prev, baselines,
        )
        assert any(e.anomaly_type == "MASSIVE_OI_SPIKE" for e in events1)

        # Manually expire cooldowns
        detector._cooldowns.clear()

        # Second detection (cooldown cleared)
        events2 = detector.detect_all_oi_anomalies(
            "NIFTY50", None, curr, prev, baselines,
        )
        assert any(e.anomaly_type == "MASSIVE_OI_SPIKE" for e in events2)


# ---------------------------------------------------------------------------
# First snapshot of day (no previous)
# ---------------------------------------------------------------------------


class TestFirstSnapshotOfDay:
    """Tests for edge case: no previous snapshot available."""

    def test_no_previous_snapshot_skips_comparison(self, detector, baselines):
        """With no previous snapshot, comparison-based checks are skipped."""
        curr = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        events = detector.detect_all_oi_anomalies(
            "NIFTY50", None, curr, None, baselines,
        )
        # No OI spike or one-sided buildup (need previous)
        # No max pain jump (need previous)
        # PCR checks may still fire if PCR is extreme
        oi_types = {"OI_SPIKE", "MASSIVE_OI_SPIKE", "ONE_SIDED_CE_BUILDUP",
                     "ONE_SIDED_PE_BUILDUP", "MAX_PAIN_JUMP", "SPOT_MAX_PAIN_DIVERGENCE"}
        for ev in events:
            assert ev.anomaly_type not in oi_types


# ---------------------------------------------------------------------------
# Unified detection and summary
# ---------------------------------------------------------------------------


class TestUnifiedDetection:
    """Tests for detect_all_oi_anomalies and get_oi_anomaly_summary."""

    def test_detect_all_returns_anomaly_events(self, detector, baselines):
        """detect_all_oi_anomalies returns list[AnomalyEvent] with category OPTIONS."""
        prev = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        curr = _make_summary(total_ce_oi=2_530_000, total_pe_oi=2_530_000)
        events = detector.detect_all_oi_anomalies(
            "NIFTY50", None, curr, prev, baselines,
        )
        for ev in events:
            assert isinstance(ev, AnomalyEvent)
            assert ev.category == "OI"
            assert ev.cooldown_key.startswith("NIFTY50_")

    def test_summary_with_anomalies(self, detector, baselines):
        """Summary reflects detected anomalies."""
        prev = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        curr = _make_summary(total_ce_oi=2_530_000, total_pe_oi=2_530_000)
        detector.detect_all_oi_anomalies(
            "NIFTY50", None, curr, prev, baselines,
        )
        summary = detector.get_oi_anomaly_summary("NIFTY50")
        assert isinstance(summary, OIAnomalySummary)
        assert summary.anomaly_intensity != "NONE"
        assert len(summary.active_anomalies) > 0

    def test_summary_no_anomalies(self, detector):
        """Summary is clean when no anomalies detected."""
        summary = detector.get_oi_anomaly_summary("NIFTY50")
        assert summary.anomaly_intensity == "NONE"
        assert summary.dominant_signal == "NEUTRAL"
        assert len(summary.active_anomalies) == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    """Tests for save_anomalies."""

    def test_save_anomalies(self, detector, baselines):
        """Anomaly events are persisted to the database."""
        prev = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        curr = _make_summary(total_ce_oi=2_530_000, total_pe_oi=2_530_000)
        events = detector.detect_all_oi_anomalies(
            "NIFTY50", None, curr, prev, baselines,
        )
        inserted = detector.save_anomalies(events)
        assert inserted >= 1

    def test_save_dedup(self, detector, baselines):
        """Saving same event twice within cooldown only inserts once."""
        prev = _make_summary(total_ce_oi=2_500_000, total_pe_oi=2_500_000)
        curr = _make_summary(total_ce_oi=2_530_000, total_pe_oi=2_530_000)

        events = detector.detect_all_oi_anomalies(
            "NIFTY50", None, curr, prev, baselines,
        )
        n1 = detector.save_anomalies(events)
        assert n1 >= 1

        # Save again — should be deduped at DB level
        n2 = detector.save_anomalies(events)
        assert n2 == 0
