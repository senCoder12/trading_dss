import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
import logging
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backfill")

_IST = ZoneInfo("Asia/Kolkata")
DB_PATH = "data/db/trading_dss.db"

def backfill():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get active indices with yahoo symbols
    cursor.execute("SELECT id, yahoo_symbol FROM index_master WHERE is_active = 1 AND yahoo_symbol IS NOT NULL")
    indices = cursor.fetchall()
    
    today = date.today()
    tomorrow = today + timedelta(days=1)
    
    for idx_id, symbol in indices:
        logger.info(f"Backfilling {idx_id} ({symbol})...")
        
        try:
            # Fetch 1m data for today
            df = yf.download(symbol, start=today.isoformat(), end=tomorrow.isoformat(), interval="1m", progress=False)
            
            if df.empty:
                logger.warning(f"No data for {symbol} on {today}")
                continue
            
            # Flatten multi-index columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            inserts = []
            for ts, row in df.iterrows():
                ts_ist = ts.astimezone(_IST)
                ts_str = ts_ist.isoformat()
                
                inserts.append((
                    idx_id,
                    ts_str,
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row.get('Volume', 0)),
                    0.0, # vwap
                    "YAHOO",
                    "1m"
                ))
            
            if inserts:
                cursor.executemany("""
                    INSERT INTO price_data (index_id, timestamp, open, high, low, close, volume, vwap, source, timeframe)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(index_id, timestamp, timeframe) DO UPDATE SET
                        open=excluded.open, high=excluded.high, low=excluded.low, 
                        close=excluded.close, volume=excluded.volume, source='BACKFILL'
                """, inserts)
                conn.commit()
                logger.info(f"Inserted {len(inserts)} rows (1m) for {idx_id}")
                
                # Resample to 5m
                df_5m = df.resample('5min').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                inserts_5m = []
                for ts, row in df_5m.iterrows():
                    ts_ist = ts.astimezone(_IST)
                    ts_str = ts_ist.isoformat()
                    inserts_5m.append((
                        idx_id, ts_str,
                        float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']),
                        float(row.get('Volume', 0)), 0.0, "YAHOO", "5m"
                    ))
                
                if inserts_5m:
                    cursor.executemany("""
                        INSERT INTO price_data (index_id, timestamp, open, high, low, close, volume, vwap, source, timeframe)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(index_id, timestamp, timeframe) DO UPDATE SET
                            open=excluded.open, high=excluded.high, low=excluded.low, 
                            close=excluded.close, volume=excluded.volume, source='BACKFILL'
                    """, inserts_5m)
                    conn.commit()
                    logger.info(f"Inserted {len(inserts_5m)} rows (5m) for {idx_id}")
                
        except Exception as e:
            logger.error(f"Failed to backfill {idx_id}: {e}")
            
    conn.close()

if __name__ == "__main__":
    backfill()
