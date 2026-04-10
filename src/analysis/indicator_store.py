"""
Indicator store — save and load technical analysis results to/from database.

Handles persistence of ``TechnicalAnalysisResult`` into the
``technical_indicators`` and ``anomaly_events`` tables.  Batch operations
for efficiency.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from src.analysis.technical_aggregator import TechnicalAnalysisResult, Alert
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class IndicatorStore:
    """Save and load technical analysis results to/from the database.

    All methods are stateless — the ``DatabaseManager`` instance is passed
    in per-call, making this class thread-safe.
    """

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @staticmethod
    def save_analysis(result: TechnicalAnalysisResult, db: DatabaseManager) -> None:
        """Persist a ``TechnicalAnalysisResult`` to the database.

        Saves indicator values to ``technical_indicators`` (upsert) and
        any alerts to ``anomaly_events``.

        Parameters
        ----------
        result:
            The analysis result to persist.
        db:
            An already-connected ``DatabaseManager``.
        """
        db.connect()

        timestamp_str = result.timestamp.isoformat()

        # Map overall_signal to the DB-compatible technical_signal
        signal_map = {
            "STRONG_BUY": "BULLISH",
            "BUY": "BULLISH",
            "NEUTRAL": "NEUTRAL",
            "SELL": "BEARISH",
            "STRONG_SELL": "BEARISH",
        }
        tech_signal = signal_map.get(result.overall_signal, "NEUTRAL")

        # ── Upsert into technical_indicators ─────────────────────────
        db.execute(
            """
            INSERT INTO technical_indicators
                (index_id, timestamp, timeframe, vwap, ema_20, ema_50,
                 rsi_14, support_1, resistance_1, support_2, resistance_2,
                 avg_volume_20, technical_signal, signal_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (index_id, timestamp, timeframe) DO UPDATE SET
                vwap = excluded.vwap,
                ema_20 = excluded.ema_20,
                ema_50 = excluded.ema_50,
                rsi_14 = excluded.rsi_14,
                support_1 = excluded.support_1,
                resistance_1 = excluded.resistance_1,
                support_2 = excluded.support_2,
                resistance_2 = excluded.resistance_2,
                avg_volume_20 = excluded.avg_volume_20,
                technical_signal = excluded.technical_signal,
                signal_strength = excluded.signal_strength
            """,
            (
                result.index_id,
                timestamp_str,
                result.timeframe,
                result.volume.poc if result.volume.poc > 0 else None,  # VWAP proxy from volume
                None,  # ema_20 not stored in summary; available from trend recompute
                None,  # ema_50
                result.momentum.rsi_value,
                result.support_levels[0] if result.support_levels else None,
                result.resistance_levels[0] if result.resistance_levels else None,
                result.support_levels[1] if len(result.support_levels) > 1 else None,
                result.resistance_levels[1] if len(result.resistance_levels) > 1 else None,
                None,  # avg_volume_20 not directly in result
                tech_signal,
                result.overall_confidence,
            ),
        )

        # ── Save alerts to anomaly_events ────────────────────────────
        # Map alert types to anomaly_type enum values the DB accepts
        alert_type_map = {
            "DIVERGENCE": "BREAKOUT",
            "BREAKOUT_TRAP": "BREAKOUT",
            "OI_SPIKE": "OI_SPIKE",
            "VOLUME_CLIMAX": "VOLUME_SPIKE",
            "BB_SQUEEZE": "BREAKOUT",
            "SMART_MONEY_SIGNAL": "FII_UNUSUAL",
            "VIX_EXTREME": "VOLUME_SPIKE",
            "REVERSAL_WARNING": "BREAKOUT",
        }

        # TODO: remove direct INSERT INTO anomaly_events - violates single
        # persistence path. AlertManager is the sole writer for anomaly_events.
        # These alerts should be returned as AnomalyEvent objects and routed
        # through AlertManager.create_alert() instead.
        if result.alerts:
            alert_params = []
            for alert in result.alerts:
                anomaly_type = alert_type_map.get(alert.type, "BREAKOUT")
                details = json.dumps({
                    "original_type": alert.type,
                    "message": alert.message,
                    "source": alert.source,
                    "overall_signal": result.overall_signal,
                    "confidence": result.overall_confidence,
                })
                alert_params.append((
                    result.index_id,
                    timestamp_str,
                    anomaly_type,
                    alert.severity,
                    details,
                    1,  # is_active
                ))

            db.execute_many(
                """
                INSERT INTO anomaly_events
                    (index_id, timestamp, anomaly_type, severity, details, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                alert_params,
            )

        logger.info(
            "Saved analysis for %s at %s (%d alerts)",
            result.index_id,
            timestamp_str,
            len(result.alerts),
        )

    # ------------------------------------------------------------------
    # Load latest
    # ------------------------------------------------------------------

    @staticmethod
    def get_latest_analysis(
        index_id: str,
        db: DatabaseManager,
    ) -> Optional[dict]:
        """Load the most recent analysis from the database.

        Returns a dict with indicator values and signal info, or ``None``
        if no data exists.

        Parameters
        ----------
        index_id:
            Index to query.
        db:
            An already-connected ``DatabaseManager``.
        """
        db.connect()
        row = db.fetch_one(
            """
            SELECT *
              FROM technical_indicators
             WHERE index_id = ?
             ORDER BY timestamp DESC
             LIMIT 1
            """,
            (index_id,),
        )
        if row is None:
            return None

        return dict(row)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    @staticmethod
    def get_analysis_history(
        index_id: str,
        start_date: str,
        end_date: str,
        db: DatabaseManager,
    ) -> list[dict]:
        """Load historical analysis results for backtesting.

        Parameters
        ----------
        index_id:
            Index to query.
        start_date:
            ISO-8601 start date (inclusive).
        end_date:
            ISO-8601 end date (inclusive).
        db:
            An already-connected ``DatabaseManager``.

        Returns
        -------
        list[dict]
            List of indicator rows as dicts, ordered by timestamp ascending.
        """
        db.connect()
        rows = db.fetch_all(
            """
            SELECT *
              FROM technical_indicators
             WHERE index_id = ?
               AND timestamp >= ?
               AND timestamp <= ?
             ORDER BY timestamp ASC
            """,
            (index_id, start_date, end_date),
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Indicator series
    # ------------------------------------------------------------------

    @staticmethod
    def get_indicator_series(
        index_id: str,
        indicator_name: str,
        start_date: str,
        end_date: str,
        db: DatabaseManager,
    ) -> pd.Series:
        """Get a time series of a specific indicator (e.g. RSI over 30 days).

        Parameters
        ----------
        index_id:
            Index to query.
        indicator_name:
            Column name in ``technical_indicators`` (e.g. ``"rsi_14"``).
        start_date:
            ISO-8601 start date (inclusive).
        end_date:
            ISO-8601 end date (inclusive).
        db:
            An already-connected ``DatabaseManager``.

        Returns
        -------
        pd.Series
            Time series indexed by timestamp with the indicator values.
        """
        valid_columns = {
            "vwap", "ema_20", "ema_50", "rsi_14",
            "support_1", "resistance_1", "support_2", "resistance_2",
            "avg_volume_20", "signal_strength",
        }
        if indicator_name not in valid_columns:
            raise ValueError(
                f"Invalid indicator name '{indicator_name}'. "
                f"Valid options: {sorted(valid_columns)}"
            )

        db.connect()
        rows = db.fetch_all(
            f"""
            SELECT timestamp, {indicator_name}
              FROM technical_indicators
             WHERE index_id = ?
               AND timestamp >= ?
               AND timestamp <= ?
             ORDER BY timestamp ASC
            """,
            (index_id, start_date, end_date),
        )

        if not rows:
            return pd.Series(dtype=float, name=indicator_name)

        timestamps = [r["timestamp"] for r in rows]
        values = [r[indicator_name] for r in rows]
        return pd.Series(values, index=pd.to_datetime(timestamps), name=indicator_name)
