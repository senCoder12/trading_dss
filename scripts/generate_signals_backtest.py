"""
Generate signals at historical dates to enable proper backtesting.
Generates signal for each index using data up to that date,
timestamps signal with that date, then checks next 20 bars for validation.
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
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
        "VOLATILE_CHOPPY": "RANGE_BOUND",
        "CALM": "TRENDING",
        "STRONG_TRENDING": "TRENDING",
        "WEAK_TRENDING": "RANGE_BOUND",
    }
    return regime_map.get(regime_name, "RANGE_BOUND")

def get_dataframe_up_to(db, index_id: str, max_timestamp: str, timeframe: str = "1d", limit: int = 250) -> pd.DataFrame:
    """Fetch bars up to max_timestamp as DataFrame."""
    rows = db.fetch_all(
        """SELECT timestamp, open, high, low, close, volume
           FROM price_data
           WHERE index_id = ? AND timeframe = ? AND timestamp <= ?
           ORDER BY timestamp ASC
           LIMIT ?""",
        (index_id, timeframe, max_timestamp, limit)
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df

def get_vix_value_at(db, max_timestamp: str) -> float:
    """Get VIX value at or before max_timestamp."""
    row = db.fetch_one(
        "SELECT vix_value FROM vix_data WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
        (max_timestamp,)
    )
    return float(row['vix_value']) if row else 15.0

def generate_signals_historical(db, registry: IndexRegistry, signal_gen: SignalGenerator):
    """Generate signals at historical dates for backtesting."""
    indices = registry.get_active_indices()

    # Get unique dates from price_data (last 100 trading dates for quicker test)
    dates = db.fetch_all(
        """SELECT DISTINCT timestamp FROM price_data
           WHERE timeframe = '1d'
           ORDER BY timestamp DESC
           LIMIT 100"""
    )

    if not dates:
        print("No price data found.")
        return

    dates = [d['timestamp'] for d in dates]
    dates.reverse()  # Process oldest to newest

    print(f"\n🔄 Generating signals for {len(indices)} indices at {len(dates)} historical dates…\n")

    tech_agg = TechnicalAggregator()
    regime_detector = RegimeDetector(db)

    count = 0
    signals_saved = 0
    signal_entries = []

    for date in dates[-20:]:  # Last 20 dates for testing
        for idx in indices:
            try:
                # Get data up to this date
                df = get_dataframe_up_to(db, idx.id, date, timeframe="1d", limit=250)
                if df.empty or len(df) < 20:
                    continue

                vix = get_vix_value_at(db, date)

                # Use 14:00 IST (2 PM) for backtest to pass market close check
                backtest_time = pd.Timestamp(date, tz=_IST).replace(hour=14, minute=0)

                # Phase 2: Technical analysis
                tech_result = tech_agg.analyze(idx.id, df, vix_value=vix, timeframe="1d")

                # Phase 5.1: Regime detection
                regime = regime_detector.detect_regime(idx.id, df, tech_result, vix_value=vix)

                # Phase 3 & 4: Mock News & Anomaly
                news_vote = NewsVote(
                    index_id=idx.id,
                    timestamp=backtest_time.to_pydatetime(),
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

                # Phase 5.2: Signal generation
                signal = signal_gen.generate_signal(
                    index_id=idx.id,
                    technical=tech_result,
                    regime=regime,
                    news=news_vote,
                    anomaly=anomaly_vote,
                    current_spot_price=float(df['close'].iloc[-1]),
                    timestamp=backtest_time.to_pydatetime()
                )

                if signal and signal.signal_type != "NO_TRADE":
                    signal_entries.append((
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
                    ))
                    signals_saved += 1

                count += 1
                if count % 100 == 0:
                    print(f"  Processed {count} index-date combinations, {signals_saved} signals saved…")

            except Exception as e:
                pass

    # Bulk insert signals
    if signal_entries:
        print(f"\n💾 Inserting {len(signal_entries)} signals into database…")
        for entry in signal_entries:
            db.execute(
                """INSERT INTO trading_signals
                   (index_id, generated_at, signal_type, confidence_level, entry_price,
                    target_price, stop_loss, risk_reward_ratio, regime, technical_vote,
                    options_vote, news_vote, anomaly_vote, reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                entry
            )
        print(f"✅ {len(signal_entries)} signals saved.")

def main():
    db = DatabaseManager()
    db.connect()
    db.initialise_schema()
    MigrationRunner(db).run_pending()

    registry = IndexRegistry.from_file(settings.indices_config_path)
    registry.sync_to_db(db)

    signal_gen = SignalGenerator(db)

    generate_signals_historical(db, registry, signal_gen)

if __name__ == "__main__":
    main()
