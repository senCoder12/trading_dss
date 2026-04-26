"""
Batch signal generation for all indices using historical data.

Generates trading signals for all available indices in the database
and persists them to the trading_signals table.
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import IST_TIMEZONE
from src.database.db_manager import DatabaseManager
from src.database.migrations import MigrationRunner
from src.data.index_registry import IndexRegistry
from src.analysis.technical_aggregator import TechnicalAggregator
from src.engine.regime_detector import RegimeDetector
from src.engine.signal_generator import SignalGenerator
from src.analysis.news.news_engine import NewsVote
from src.analysis.anomaly.anomaly_aggregator import AnomalyVote
from config.settings import settings

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

def map_regime_to_db(regime_name: str) -> str:
    """Map regime names to DB valid values."""
    regime_map = {
        "TRENDING": "TRENDING",
        "RANGE_BOUND": "RANGE_BOUND",
        "EVENT": "EVENT",
        "VOLATILE_CHOPPY": "RANGE_BOUND",  # Map volatile/choppy to range-bound
        "CALM": "TRENDING",
        "STRONG_TRENDING": "TRENDING",
        "WEAK_TRENDING": "RANGE_BOUND",
    }
    return regime_map.get(regime_name, "RANGE_BOUND")

def get_dataframe(db, index_id: str, timeframe: str = "1d", limit: int = 250) -> pd.DataFrame:
    """Fetch bars as DataFrame for an index."""
    rows = db.fetch_all(
        """SELECT timestamp, open, high, low, close, volume
           FROM price_data
           WHERE index_id = ? AND timeframe = ?
           ORDER BY timestamp ASC
           LIMIT ?""",
        (index_id, timeframe, limit)
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df

def get_vix_value(db) -> float:
    """Get latest VIX value."""
    row = db.fetch_one("SELECT vix_value FROM vix_data ORDER BY timestamp DESC LIMIT 1")
    return float(row['vix_value']) if row else 15.0

def generate_signals_for_all(db, registry: IndexRegistry, signal_gen: SignalGenerator):
    """Generate signals for all active indices."""
    indices = registry.get_active_indices()
    print(f"\n🔄 Generating signals for {len(indices)} indices…\n")

    tech_agg = TechnicalAggregator()
    regime_detector = RegimeDetector(db)
    vix = get_vix_value(db)

    count = 0
    signals_saved = 0

    for idx in indices:
        try:
            df = get_dataframe(db, idx.id, timeframe="1d", limit=250)
            if df.empty or len(df) < 20:
                print(f"  [{count+1:3d}/{len(indices)}] {idx.id:<20} SKIP (< 20 bars)")
                count += 1
                continue

            # Run Phase 2: Technical analysis
            tech_result = tech_agg.analyze(idx.id, df, vix_value=vix, timeframe="1d")

            # Run Phase 5.1: Regime detection
            regime = regime_detector.detect_regime(idx.id, df, tech_result, vix_value=vix)

            # Mock Phase 3 & 4: News & Anomaly votes
            news_vote = NewsVote(
                index_id=idx.id,
                timestamp=datetime.now(tz=_IST),
                vote="NEUTRAL",
                confidence=0.5,
                active_article_count=0,
                weighted_sentiment=0.0,
                top_headline="",
                event_regime="NORMAL",
                reasoning="No news data"
            )

            anomaly_vote = AnomalyVote(
                index_id=idx.id,
                vote="NEUTRAL",
                confidence=0.5,
                risk_level="NORMAL",
                position_size_modifier=1.0,
                active_alerts=0,
                primary_alert_message=None,
                institutional_activity=False,
                reasoning="No anomaly data"
            )

            # Run Phase 5.2: Signal generation
            signal = signal_gen.generate_signal(
                index_id=idx.id,
                technical=tech_result,
                regime=regime,
                news=news_vote,
                anomaly=anomaly_vote,
                current_spot_price=float(df['close'].iloc[-1]),
                timestamp=datetime.now(tz=_IST)
            )

            if signal and signal.signal_type != "NO_TRADE":
                # Manually persist signal to DB (generate_signal() caches only)
                db.execute(
                    """INSERT INTO trading_signals
                       (index_id, generated_at, signal_type, confidence_level, entry_price,
                        target_price, stop_loss, risk_reward_ratio, regime, technical_vote,
                        options_vote, news_vote, anomaly_vote, reasoning)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        signal.index_id,
                        signal.generated_at.isoformat(),
                        signal.signal_type,
                        signal.confidence_level,
                        signal.entry_price,
                        signal.target_price,
                        signal.stop_loss,
                        signal.risk_reward_ratio,
                        map_regime_to_db(signal.regime),
                        "NEUTRAL",
                        "NEUTRAL",
                        news_vote.vote if news_vote else "NEUTRAL",
                        anomaly_vote.vote if anomaly_vote else "NEUTRAL",
                        signal.reasoning
                    )
                )
                print(f"  [{count+1:3d}/{len(indices)}] {idx.id:<20} {signal.signal_type:<12} ({signal.confidence_level}) ✓")
                signals_saved += 1
            else:
                print(f"  [{count+1:3d}/{len(indices)}] {idx.id:<20} {signal.signal_type if signal else 'SKIP'}")

            count += 1
        except Exception as e:
            print(f"  [{count+1:3d}/{len(indices)}] {idx.id:<20} ERROR: {str(e)[:50]}")
            count += 1

    print(f"\n✅ Signals generated: {signals_saved}/{len(indices)}")
    sig_count = db.fetch_one('SELECT COUNT(*) as c FROM trading_signals')['c']
    print(f"📊 Database: {sig_count} signals total\n")

def main():
    db = DatabaseManager()
    db.connect()
    db.initialise_schema()
    MigrationRunner(db).run_pending()

    registry = IndexRegistry.from_file(settings.indices_config_path)
    registry.sync_to_db(db)

    signal_gen = SignalGenerator(db)

    generate_signals_for_all(db, registry, signal_gen)

if __name__ == "__main__":
    main()
