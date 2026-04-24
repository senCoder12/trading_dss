
import sys
import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.database.db_manager import DatabaseManager
from src.data.index_registry import IndexRegistry
from src.data.historical_data import HistoricalDataManager
from src.data.data_collector import DataCollector
from src.data.nse_scraper import NSEScraper
from src.data.fii_dii_data import FIIDIIFetcher
from src.data.vix_data import VIXTracker

def main():
    db = DatabaseManager.instance()
    db.connect()
    
    registry = IndexRegistry.from_file(settings.indices_config_path)
    manager = HistoricalDataManager(registry, db)
    
    print("\n--- Phase 1: Backfilling 1m Intraday Data (NIFTY50, BANKNIFTY, FINNIFTY) ---")
    indices_to_fetch = ["NIFTY50", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]
    for idx_id in indices_to_fetch:
        print(f"Fetching 1m data for {idx_id}...")
        definition = registry.get_index(idx_id)
        if not definition or not definition.yahoo_symbol: continue
        try:
            ticker = yf.Ticker(definition.yahoo_symbol)
            df = ticker.history(period="5d", interval="1m", auto_adjust=True)
            if df.empty: continue
            df = df.rename(columns=str.lower)
            df.index.name = "timestamp"
            df.reset_index(inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            if df.index.tz is None: df.index = df.index.tz_localize("UTC").tz_convert(IST_TIMEZONE)
            else: df.index = df.index.tz_convert(IST_TIMEZONE)
            df = df.reset_index()
            today_str = "2026-04-21"
            df_today = df[df['timestamp'].dt.strftime('%Y-%m-%d') == today_str]
            if not df_today.empty:
                manager._save_dataframe_to_db(idx_id, df_today, timeframe="1m")
                print(f"  ✓ Saved {len(df_today)} 1m records for {idx_id}")
        except Exception as e:
            print(f"  ❌ Failed for {idx_id}: {e}")
        time.sleep(1)

    print("\n--- Phase 2: Updating Daily (1d) History ---")
    try:
        manager.update_daily_data()
        print("  ✓ Daily data updated for all active indices.")
    except Exception as e:
        print(f"  ❌ Failed to update daily data: {e}")

    print("\n--- Phase 3: Fetching FII/DII and VIX for Today ---")
    scraper = NSEScraper()
    try:
        fii_fetcher = FIIDIIFetcher(scraper=scraper)
        fii_data = fii_fetcher.fetch_today_fii_dii()
        if fii_data:
            fii_fetcher.save_to_db(fii_data, db)
            print(f"  ✓ FII/DII data saved for today. Net: {fii_data.fii_net_value}")
        else:
            print("  ⚠️ FII/DII data not yet available on NSE.")
    except Exception as e:
        print(f"  ❌ FII/DII fetch failed: {e}")

    try:
        vix_tracker = VIXTracker(scraper=scraper)
        vix_data = vix_tracker.get_current_vix()
        if vix_data:
            vix_tracker.save_to_db(vix_data, db)
            print(f"  ✓ VIX data saved for today: {vix_data.value}")
    except Exception as e:
        print(f"  ❌ VIX fetch failed: {e}")

    print("\n--- Phase 4: Recomputing Technical Indicators ---")
    try:
        collector = DataCollector(db=db, registry=registry)
        # Force start to bypass market hours gate if needed, but collect_technical_indicators 
        # doesn't strictly check for "open" if called manually, but let's be safe.
        collector.force_start = True
        collector.collect_technical_indicators()
        print("  ✓ Technical indicators recomputed based on new data.")
    except Exception as e:
        print(f"  ❌ Technical indicator recomputation failed: {e}")

    print("\n--- Backfill Complete ---")

if __name__ == "__main__":
    main()
