"""
Phase 4 — Anomaly Aggregator & Master Detection Engine.

Single entry point for ALL anomaly detection.  The Data Collector calls
:meth:`AnomalyAggregator.run_detection_cycle` once per minute per index
during market hours.  The aggregator orchestrates sub-detectors, manages
alerts, computes an anomaly vote, and returns a unified result for the
Decision Engine.

Integration with Data Collector
-------------------------------
.. code-block:: python

    # In data_collector.py, add these scheduled jobs:
    #
    # self.anomaly_engine = AnomalyAggregator(self.db)
    #
    # Job: detect_anomalies — runs every 60 seconds during market hours
    #      (after price collection)
    # def _detect_anomalies(self):
    #     for index_id in self.registry.get_indices_with_options():
    #         result = self.anomaly_engine.run_detection_cycle(
    #             index_id=index_id,
    #             current_price_bar=self._get_latest_bar(index_id),
    #             recent_price_bars=self._get_recent_bars(index_id, count=20),
    #             options_chain=self._get_latest_chain(index_id),
    #             ...
    #         )
    #         if result.high_severity_count > 0:
    #             self.telegram.send_alert(result.primary_alert.message)
    #
    # Job: detect_divergences — runs every 30 minutes during market hours
    # def _detect_divergences(self):
    #     self.anomaly_engine.run_divergence_cycle()
    #
    # Job: analyze_fii — runs at 18:30 IST (after market close)
    # def _analyze_fii(self):
    #     self.anomaly_engine.run_fii_cycle()
"""

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
from src.analysis.anomaly.volume_price_detector import (
    AnomalyEvent,
    Baselines,
    VolumePriceDetector,
)
from src.analysis.anomaly.oi_anomaly_detector import (
    OIAnomalyDetector,
    OIBaselines,
)
from src.analysis.anomaly.flow_divergence_detector import (
    CrossIndexDivergenceDetector,
    FIIFlowDetector,
    FIIBias,
)
from src.analysis.anomaly.alert_manager import AlertManager

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnomalyVote:
    """Quick anomaly assessment for the Decision Engine."""

    index_id: str
    vote: str = "NEUTRAL"  # STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG_BEARISH / CAUTION
    confidence: float = 0.0
    risk_level: str = "NORMAL"  # NORMAL / ELEVATED / HIGH / EXTREME
    position_size_modifier: float = 1.0
    active_alerts: int = 0
    primary_alert_message: Optional[str] = None
    institutional_activity: bool = False
    reasoning: str = ""


@dataclass
class AnomalyDetectionResult:
    """Full result from a single detection cycle."""

    index_id: str
    timestamp: datetime

    # All detected anomalies this cycle
    new_anomalies: list[AnomalyEvent] = field(default_factory=list)

    # Currently active alerts (new + previously active)
    active_alerts: list[AnomalyEvent] = field(default_factory=list)
    active_alert_count: int = 0

    # Categorised counts
    volume_anomalies: int = 0
    price_anomalies: int = 0
    oi_anomalies: int = 0
    divergence_anomalies: int = 0

    # Severity summary
    high_severity_count: int = 0
    medium_severity_count: int = 0

    # Anomaly vote (for Decision Engine)
    anomaly_vote: str = "NEUTRAL"
    anomaly_confidence: float = 0.0

    # Risk modifier
    risk_level: str = "NORMAL"
    position_size_modifier: float = 1.0

    # Most significant finding
    primary_alert: Optional[AnomalyEvent] = None

    # Human-readable summary
    summary: str = ""

    # Quick flags for Decision Engine
    has_volume_shock: bool = False
    has_oi_spike: bool = False
    has_divergence: bool = False
    has_breakout_trap_risk: bool = False
    institutional_activity_detected: bool = False


# ---------------------------------------------------------------------------
# Anomaly Aggregator
# ---------------------------------------------------------------------------


class AnomalyAggregator:
    """Single entry point for ALL anomaly detection.

    Orchestrates volume/price, OI, FII flow, and cross-index divergence
    detectors.  Thread-safe — each public method acquires a lock so
    concurrent scheduler calls do not corrupt shared state.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self._lock = threading.Lock()

        # Sub-detectors
        self.volume_price = VolumePriceDetector(db)
        self.oi_detector = OIAnomalyDetector(db)
        self.fii_detector = FIIFlowDetector(db)
        self.divergence_detector = CrossIndexDivergenceDetector(db)

        # Alert manager
        self.alerts = AlertManager(db)

        # Baseline caches
        self._vp_baselines: dict[str, Baselines] = {}
        self._oi_baselines: dict[str, OIBaselines] = {}

        # Latest FII bias (updated after market close)
        self._fii_bias: Optional[FIIBias] = None

        # Last divergence results (updated every 30 minutes)
        self._latest_divergence_events: list[AnomalyEvent] = []
        self._last_divergence_run: Optional[datetime] = None

        logger.info("AnomalyAggregator initialised with all sub-detectors")

    # ------------------------------------------------------------------
    # Main detection cycle
    # ------------------------------------------------------------------

    def run_detection_cycle(
        self,
        index_id: str,
        current_price_bar: dict,
        recent_price_bars: list[dict],
        options_chain=None,
        options_summary=None,
        previous_options_summary=None,
        previous_chain=None,
        expiry_date=None,
        days_to_expiry: int = 0,
    ) -> AnomalyDetectionResult:
        """Run ALL relevant detectors for a single index in one call.

        Parameters
        ----------
        index_id : str
            Index identifier (e.g. ``"NIFTY50"``).
        current_price_bar : dict
            Latest OHLCV bar with keys: timestamp, open, high, low, close, volume.
        recent_price_bars : list[dict]
            Last ~20 bars for context.
        options_chain : OptionsChainData, optional
            Current options chain snapshot.
        options_summary : OISummary, optional
            Current OI summary.
        previous_options_summary : OISummary, optional
            Previous OI summary for delta calculations.
        previous_chain : OptionsChainData, optional
            Previous chain for IV comparisons.
        expiry_date : date, optional
            Nearest expiry date.
        days_to_expiry : int
            Days until nearest expiry.

        Returns
        -------
        AnomalyDetectionResult
            Unified result with all anomalies, vote, and risk assessment.
        """
        now = datetime.now(tz=_IST)
        all_new: list[AnomalyEvent] = []

        with self._lock:
            # 1. Ensure baselines
            vp_bl = self._ensure_vp_baselines(index_id, recent_price_bars)

            # 2. Volume & Price detection
            vp_events = self._run_volume_price(
                index_id, current_price_bar, recent_price_bars, vp_bl
            )
            all_new.extend(vp_events)

            # 3. OI detection (if options data available)
            oi_events: list[AnomalyEvent] = []
            if options_summary is not None:
                oi_bl = self._ensure_oi_baselines(index_id)
                oi_events = self._run_oi_detection(
                    index_id,
                    options_chain,
                    options_summary,
                    previous_options_summary,
                    oi_bl,
                    previous_chain=previous_chain,
                    expiry_date=expiry_date,
                    days_to_expiry=days_to_expiry,
                )
                all_new.extend(oi_events)

            # 4. Create alerts for new detections
            for event in all_new:
                try:
                    self.alerts.create_alert(event)
                except Exception:
                    logger.exception(
                        "Failed to create alert for %s %s",
                        event.index_id,
                        event.anomaly_type,
                    )

            # 5. Auto-resolve stale alerts
            try:
                self.alerts.auto_resolve_stale_alerts()
            except Exception:
                logger.exception("Failed to auto-resolve stale alerts")

        # 6. Build result
        active = self.alerts.get_active_alerts(index_id=index_id)
        all_active = self.alerts.get_active_alerts()

        # Include divergence events from last run if relevant to this index
        div_events = [
            e
            for e in self._latest_divergence_events
            if index_id in e.index_id
        ]

        result = self._build_result(
            index_id=index_id,
            timestamp=now,
            new_anomalies=all_new,
            active_alerts=active,
            divergence_events=div_events,
        )

        if result.high_severity_count > 0:
            logger.warning(
                "Index %s: %d HIGH severity alerts active",
                index_id,
                result.high_severity_count,
            )
        else:
            logger.debug(
                "Index %s: %d new anomalies, %d active alerts, vote=%s risk=%s",
                index_id,
                len(all_new),
                result.active_alert_count,
                result.anomaly_vote,
                result.risk_level,
            )

        return result

    # ------------------------------------------------------------------
    # FII cycle (post-market)
    # ------------------------------------------------------------------

    def run_fii_cycle(self) -> Optional[FIIBias]:
        """Run FII analysis after market close.

        Fetches today's FII/DII data, runs anomaly detection, and returns
        the FII bias for the next trading day.
        """
        try:
            from src.data.fii_dii_data import FIIDIIData, FIIDIIFetcher

            fetcher = FIIDIIFetcher()
            today_data = fetcher.fetch_today_fii_dii()
            if today_data is None:
                logger.warning("No FII/DII data available for today")
                return self._fii_bias

            # Compute baselines
            baselines = self.fii_detector.compute_flow_baselines()

            # Detect anomalies
            fii_events = self.fii_detector.detect_fii_anomaly(today_data, baselines)
            for event in fii_events:
                try:
                    self.alerts.create_alert(event)
                except Exception:
                    logger.exception("Failed to create FII alert")

            # Compute bias
            self._fii_bias = self.fii_detector.get_fii_bias(today_data)
            logger.info("FII cycle complete: bias=%s", self._fii_bias.bias)

            return self._fii_bias

        except Exception:
            logger.exception("FII cycle failed")
            return self._fii_bias

    # ------------------------------------------------------------------
    # Divergence cycle (every 30 minutes)
    # ------------------------------------------------------------------

    def run_divergence_cycle(self) -> list[AnomalyEvent]:
        """Run cross-index divergence checks.

        Should be called every 30 minutes during market hours.
        """
        try:
            events = self.divergence_detector.detect_all_divergences()
            for event in events:
                try:
                    self.alerts.create_alert(event)
                except Exception:
                    logger.exception(
                        "Failed to create divergence alert for %s",
                        event.anomaly_type,
                    )

            self._latest_divergence_events = events
            self._last_divergence_run = datetime.now(tz=_IST)

            logger.info(
                "Divergence cycle complete: %d anomalies detected", len(events)
            )
            return events

        except Exception:
            logger.exception("Divergence cycle failed")
            return []

    # ------------------------------------------------------------------
    # Quick vote for Decision Engine
    # ------------------------------------------------------------------

    def get_anomaly_vote(self, index_id: str) -> AnomalyVote:
        """Quick method for the Decision Engine to get current anomaly assessment."""
        active = self.alerts.get_active_alerts(index_id=index_id)
        div_events = [
            e for e in self._latest_divergence_events if index_id in e.index_id
        ]
        all_relevant = active + div_events

        vote, confidence, reasoning_parts = self._compute_vote(
            all_relevant, index_id
        )
        risk_level, modifier = self._compute_risk(all_relevant)

        primary = active[0] if active else None
        institutional = self._has_institutional_activity(all_relevant)

        return AnomalyVote(
            index_id=index_id,
            vote=vote,
            confidence=confidence,
            risk_level=risk_level,
            position_size_modifier=modifier,
            active_alerts=len(active),
            primary_alert_message=primary.message if primary else None,
            institutional_activity=institutional,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "No anomalies",
        )

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_market_anomaly_dashboard(self) -> dict:
        """Overview of all anomalies across all indices for the frontend."""
        now = datetime.now(tz=_IST)
        all_active = self.alerts.get_active_alerts()

        # Group by index
        by_index: dict[str, dict] = {}
        for alert in all_active:
            idx = alert.index_id
            if idx not in by_index:
                by_index[idx] = {"alerts": 0, "risk": "NORMAL", "top_alert": "", "events": []}
            by_index[idx]["alerts"] += 1
            by_index[idx]["events"].append(alert)

        # Compute per-index risk and top alert
        for idx, info in by_index.items():
            risk, _ = self._compute_risk(info["events"])
            info["risk"] = risk
            if info["events"]:
                info["top_alert"] = info["events"][0].message[:100]
            del info["events"]  # Don't include raw events in JSON output

        # Market-wide signals
        fii_bias = self._fii_bias.bias if self._fii_bias else "UNKNOWN"
        sector_rotation = self._get_sector_rotation_signal()
        vix_signal = self._get_vix_signal(all_active)

        # Top 5 most critical alerts across all indices
        most_critical = sorted(
            all_active,
            key=lambda a: (
                0 if a.severity == "HIGH" else 1,
                -a.timestamp.timestamp(),
            ),
        )[:5]

        return {
            "timestamp": now.isoformat(),
            "total_active_alerts": len(all_active),
            "by_index": by_index,
            "market_wide": {
                "fii_bias": fii_bias,
                "sector_rotation": sector_rotation,
                "vix_signal": vix_signal,
            },
            "most_critical_alerts": [
                {
                    "index_id": a.index_id,
                    "type": a.anomaly_type,
                    "severity": a.severity,
                    "message": a.message[:150],
                }
                for a in most_critical
            ],
        }

    # ==================================================================
    # Private helpers
    # ==================================================================

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def _ensure_vp_baselines(
        self, index_id: str, recent_bars: list[dict]
    ) -> Baselines:
        """Get or compute volume/price baselines."""
        bl = self._vp_baselines.get(index_id)
        if bl is not None:
            # Recompute if older than 4 hours
            age = datetime.now(tz=_IST) - bl.computed_at
            if age < timedelta(hours=4) and bl.avg_volume_20d > 0:
                return bl

        # Try loading from detector cache first
        bl = self.volume_price.get_baselines(index_id)
        if bl is not None and bl.avg_volume_20d > 0:
            self._vp_baselines[index_id] = bl
            return bl

        # Compute from recent bars if we have enough data
        if len(recent_bars) >= 5:
            df = pd.DataFrame(recent_bars)
            bl = self.volume_price.update_baselines(index_id, df)
            self._vp_baselines[index_id] = bl
            return bl

        # Fallback: empty baselines — NOT cached so we recompute next cycle
        # when more bars are available
        return Baselines(
            index_id=index_id,
            computed_at=datetime.now(tz=_IST),
        )

    def _ensure_oi_baselines(self, index_id: str) -> OIBaselines:
        """Get or compute OI baselines."""
        bl = self._oi_baselines.get(index_id)
        if bl is not None:
            return bl

        bl = self.oi_detector.get_baselines(index_id)
        if bl is not None:
            self._oi_baselines[index_id] = bl
            return bl

        # Compute from history
        bl = self.oi_detector.update_oi_baselines(index_id)
        self._oi_baselines[index_id] = bl
        return bl

    # ------------------------------------------------------------------
    # Sub-detector runners (isolated error handling)
    # ------------------------------------------------------------------

    def _run_volume_price(
        self,
        index_id: str,
        current_bar: dict,
        recent_bars: list[dict],
        baselines: Baselines,
    ) -> list[AnomalyEvent]:
        """Run volume/price detection with error isolation.

        Does NOT call ``save_anomalies`` — persistence is handled by the
        AlertManager in ``run_detection_cycle`` to avoid duplicate rows.
        """
        try:
            return self.volume_price.detect_all(
                index_id, current_bar, recent_bars, baselines
            )
        except Exception:
            logger.exception("VolumePriceDetector failed for %s", index_id)
            return []

    def _run_oi_detection(
        self,
        index_id: str,
        current_chain,
        current_summary,
        previous_summary,
        baselines: OIBaselines,
        previous_chain=None,
        expiry_date=None,
        days_to_expiry: int = 0,
    ) -> list[AnomalyEvent]:
        """Run OI detection with error isolation.

        Does NOT call ``save_anomalies`` — persistence is handled by the
        AlertManager in ``run_detection_cycle`` to avoid duplicate rows.
        """
        try:
            return self.oi_detector.detect_all_oi_anomalies(
                index_id=index_id,
                current_chain=current_chain,
                current_summary=current_summary,
                previous_summary=previous_summary,
                baselines=baselines,
                previous_chain=previous_chain,
                expiry_date=expiry_date,
                days_to_expiry=days_to_expiry,
            )
        except Exception:
            logger.exception("OIAnomalyDetector failed for %s", index_id)
            return []

    # ------------------------------------------------------------------
    # Vote logic
    # ------------------------------------------------------------------

    def _compute_vote(
        self,
        active_alerts: list[AnomalyEvent],
        index_id: str,
    ) -> tuple[str, float, list[str]]:
        """Compute the anomaly vote from active alerts.

        Returns (vote, confidence, reasoning_parts).
        """
        if not active_alerts:
            return "NEUTRAL", 0.0, ["No anomalies detected"]

        bullish_score = 0.0
        bearish_score = 0.0
        caution_score = 0.0
        reasoning: list[str] = []

        for alert in active_alerts:
            weight = 2.0 if alert.severity == "HIGH" else 1.0
            details = self._parse_details(alert.details)
            direction = details.get("directional_implication", "").upper()
            atype = alert.anomaly_type

            # Volume spike + price direction
            if atype == "VOLUME_SPIKE":
                price_chg = details.get("price_change_pct", 0.0)
                if price_chg > 0:
                    bullish_score += weight
                    reasoning.append(f"Volume spike with price UP ({price_chg:+.2f}%)")
                elif price_chg < 0:
                    bearish_score += weight
                    reasoning.append(f"Volume spike with price DOWN ({price_chg:+.2f}%)")
                else:
                    caution_score += weight

            # Absorption — direction unclear, big move coming
            elif atype == "ABSORPTION":
                caution_score += weight * 1.5
                reasoning.append("Volume absorption detected — big move likely")

            # OI spikes
            elif atype in ("OI_SPIKE", "MASSIVE_OI_SPIKE"):
                if direction == "BULLISH":
                    bullish_score += weight
                    reasoning.append(f"OI spike with {direction} implication")
                elif direction == "BEARISH":
                    bearish_score += weight
                    reasoning.append(f"OI spike with {direction} implication")
                else:
                    caution_score += weight * 0.5

            # One-sided OI buildup
            elif atype == "ONE_SIDED_PE_BUILDUP":
                bullish_score += weight
                reasoning.append("PE buildup — bullish signal")
            elif atype == "ONE_SIDED_CE_BUILDUP":
                bearish_score += weight
                reasoning.append("CE buildup — bearish signal")

            # PCR extremes
            elif atype in ("PCR_EXTREME_HIGH", "PCR_EXTREME"):
                pcr = details.get("pcr", 0.0)
                if pcr > 1.0:
                    bullish_score += weight
                    reasoning.append(f"High PCR ({pcr:.2f}) — bullish")
                elif pcr < 0.7:
                    bearish_score += weight
                    reasoning.append(f"Low PCR ({pcr:.2f}) — bearish")
            elif atype == "PCR_EXTREME_LOW":
                bearish_score += weight
                reasoning.append("Extreme low PCR — bearish")

            # IV explosion
            elif atype == "IV_EXPLOSION":
                caution_score += weight
                reasoning.append("IV explosion — elevated uncertainty")

            # IV skew
            elif atype == "IV_SKEW_EXTREME":
                bearish_score += weight
                reasoning.append("Extreme IV skew — bearish hedging")

            # FII flows
            elif atype == "FII_EXTREME_FLOW":
                fii_net = details.get("fii_net", 0.0)
                if fii_net > 0:
                    bullish_score += weight * 1.5
                    reasoning.append("FII extreme buying")
                else:
                    bearish_score += weight * 1.5
                    reasoning.append("FII extreme selling")

            elif atype == "FII_TREND_REVERSAL":
                caution_score += weight
                reasoning.append("FII trend reversal — direction shift")

            # Divergence
            elif atype == "VIX_DIVERGENCE":
                caution_score += weight * 1.5
                reasoning.append("VIX divergence — market instability")

            elif atype in ("CORRELATION_BREAKDOWN", "DIRECTIONAL_DIVERGENCE"):
                caution_score += weight
                reasoning.append(f"{atype.replace('_', ' ').title()}")

            # Gap signals
            elif atype == "GAP_UP":
                bullish_score += weight * 0.5
                reasoning.append("Gap up detected")
            elif atype == "GAP_DOWN":
                bearish_score += weight * 0.5
                reasoning.append("Gap down detected")

            # Price movements
            elif atype in ("EXTREME_INTRADAY_MOVE", "LARGE_INTRADAY_MOVE"):
                price_chg = details.get("price_change_pct", 0.0)
                if price_chg > 0:
                    bullish_score += weight * 0.8
                else:
                    bearish_score += weight * 0.8
                reasoning.append(f"Large intraday move ({price_chg:+.2f}%)")

            # Range compression
            elif atype == "COMPRESSION":
                caution_score += weight * 0.5
                reasoning.append("Range compression — breakout imminent")

            # Catch-all for unclassified
            else:
                caution_score += weight * 0.3

        # Also incorporate FII bias
        if self._fii_bias:
            if self._fii_bias.bias in ("STRONG_BUYING", "BUYING"):
                bullish_score += self._fii_bias.confidence * 1.5
                reasoning.append(f"FII bias: {self._fii_bias.bias}")
            elif self._fii_bias.bias in ("STRONG_SELLING", "SELLING"):
                bearish_score += self._fii_bias.confidence * 1.5
                reasoning.append(f"FII bias: {self._fii_bias.bias}")

        # Determine vote
        total = bullish_score + bearish_score + caution_score
        if total == 0:
            return "NEUTRAL", 0.0, reasoning

        # Multiple conflicting signals → CAUTION
        if (
            bullish_score > 1
            and bearish_score > 1
            and abs(bullish_score - bearish_score) < max(bullish_score, bearish_score) * 0.3
        ):
            return "CAUTION", min(0.8, caution_score / total + 0.3), reasoning

        # Caution dominates
        if caution_score > bullish_score and caution_score > bearish_score:
            return "CAUTION", min(0.9, caution_score / total), reasoning

        # Bullish
        if bullish_score > bearish_score:
            net = bullish_score - bearish_score
            confidence = min(0.95, net / total)
            if net > 4:
                return "STRONG_BULLISH", confidence, reasoning
            return "BULLISH", confidence, reasoning

        # Bearish
        if bearish_score > bullish_score:
            net = bearish_score - bullish_score
            confidence = min(0.95, net / total)
            if net > 4:
                return "STRONG_BEARISH", confidence, reasoning
            return "BEARISH", confidence, reasoning

        return "NEUTRAL", 0.2, reasoning

    # ------------------------------------------------------------------
    # Risk computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_risk(
        active_alerts: list[AnomalyEvent],
    ) -> tuple[str, float]:
        """Compute risk level and position size modifier.

        Returns (risk_level, modifier).
        """
        if not active_alerts:
            return "NORMAL", 1.0

        high_count = sum(1 for a in active_alerts if a.severity == "HIGH")
        medium_count = sum(1 for a in active_alerts if a.severity == "MEDIUM")

        # Check for EXTREME severity in details (some detectors flag this)
        has_extreme = any(
            "EXTREME" in (a.details or "")
            for a in active_alerts
        )

        if high_count >= 3 or has_extreme:
            return "EXTREME", 0.3
        if high_count >= 1:
            return "HIGH", 0.6
        if medium_count >= 1:
            return "ELEVATED", 0.8
        return "NORMAL", 1.0

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(
        self,
        index_id: str,
        timestamp: datetime,
        new_anomalies: list[AnomalyEvent],
        active_alerts: list[AnomalyEvent],
        divergence_events: list[AnomalyEvent],
    ) -> AnomalyDetectionResult:
        """Assemble a complete detection result."""
        all_relevant = active_alerts + divergence_events

        # Category counts
        vol_count = sum(1 for a in all_relevant if a.category == "VOLUME")
        price_count = sum(1 for a in all_relevant if a.category == "PRICE")
        oi_count = sum(1 for a in all_relevant if a.category == "OI")
        div_count = sum(
            1 for a in all_relevant if a.category == "DIVERGENCE"
        )

        # Severity counts
        high_count = sum(1 for a in all_relevant if a.severity == "HIGH")
        medium_count = sum(1 for a in all_relevant if a.severity == "MEDIUM")

        # Vote
        vote, confidence, reasoning = self._compute_vote(all_relevant, index_id)

        # Risk
        risk_level, modifier = self._compute_risk(all_relevant)

        # Primary alert (highest severity, most recent)
        primary = active_alerts[0] if active_alerts else None

        # Flags
        has_volume_shock = any(
            a.anomaly_type in ("VOLUME_SPIKE", "SUDDEN_ACCELERATION")
            for a in all_relevant
        )
        has_oi_spike = any(
            a.anomaly_type in ("OI_SPIKE", "MASSIVE_OI_SPIKE")
            for a in all_relevant
        )
        has_divergence = len(divergence_events) > 0
        has_breakout_trap = any(
            "breakout_quality" in (a.details or "")
            and self._parse_details(a.details).get("breakout_quality", 10) < 4
            for a in all_relevant
        )
        institutional = self._has_institutional_activity(all_relevant)

        # Summary
        summary_lines = [f"Anomaly Report for {index_id} at {timestamp.strftime('%H:%M:%S')}"]
        summary_lines.append(f"Active alerts: {len(active_alerts)} ({high_count} HIGH, {medium_count} MEDIUM)")
        summary_lines.append(f"Vote: {vote} (confidence: {confidence:.0%})")
        summary_lines.append(f"Risk: {risk_level} (position modifier: {modifier:.0%})")
        if reasoning:
            summary_lines.append("Signals: " + "; ".join(reasoning[:5]))
        if primary:
            summary_lines.append(f"Primary: [{primary.severity}] {primary.message[:100]}")

        return AnomalyDetectionResult(
            index_id=index_id,
            timestamp=timestamp,
            new_anomalies=new_anomalies,
            active_alerts=active_alerts,
            active_alert_count=len(active_alerts),
            volume_anomalies=vol_count,
            price_anomalies=price_count,
            oi_anomalies=oi_count,
            divergence_anomalies=div_count,
            high_severity_count=high_count,
            medium_severity_count=medium_count,
            anomaly_vote=vote,
            anomaly_confidence=confidence,
            risk_level=risk_level,
            position_size_modifier=modifier,
            primary_alert=primary,
            summary="\n".join(summary_lines),
            has_volume_shock=has_volume_shock,
            has_oi_spike=has_oi_spike,
            has_divergence=has_divergence,
            has_breakout_trap_risk=has_breakout_trap,
            institutional_activity_detected=institutional,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_details(details: str) -> dict:
        """Safely parse the JSON details field."""
        if not details:
            return {}
        try:
            return json.loads(details)
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _has_institutional_activity(alerts: list[AnomalyEvent]) -> bool:
        """Check for signs of institutional activity."""
        institutional_types = {
            "ABSORPTION",
            "FII_EXTREME_FLOW",
            "FII_UNUSUAL_FLOW",
            "FII_FO_EXTREME",
            "MASSIVE_OI_SPIKE",
        }
        return any(a.anomaly_type in institutional_types for a in alerts)

    def _get_sector_rotation_signal(self) -> str:
        """Extract sector rotation from latest divergence events."""
        for event in self._latest_divergence_events:
            if event.anomaly_type == "SECTOR_ROTATION":
                details = self._parse_details(event.details)
                return details.get("rotation_type", "UNKNOWN")
        return "NONE"

    def _get_vix_signal(self, alerts: list[AnomalyEvent]) -> str:
        """Extract VIX signal from active alerts."""
        for alert in alerts:
            if alert.anomaly_type == "VIX_DIVERGENCE":
                details = self._parse_details(alert.details)
                return details.get("implication", "ELEVATED")
        return "NORMAL"
