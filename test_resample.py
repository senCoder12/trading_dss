import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

# Load DB directly
conn = sqlite3.connect("data/db/trading_dss.db")
conn.row_factory = sqlite3.Row

# Get since 30 days ago
now = datetime.now(ZoneInfo("Asia/Kolkata"))
since = (now - timedelta(days=30)).isoformat()

def get_resampled():
    c = conn.cursor()
    c.execute("SELECT * FROM price_data WHERE index_id='BANKNIFTY' AND timeframe='5m' AND timestamp BETWEEN ? AND ?", (since, now.isoformat()))
    source_rows = c.fetchall()
    
    if not source_rows:
        print("No source rows found")
        return
        
    df = pd.DataFrame([dict(r) for r in source_rows])
    df["dt"] = pd.to_datetime(df["timestamp"])
    df.set_index("dt", inplace=True)
    
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "vwap": "mean" # It might be null in DB, pandas handles it?
    }
    
    res = df.resample("15min", label="left", closed="left").agg(agg_dict).dropna(subset=['open'])
    print(f"Resampled to {len(res)} rows")
    if len(res) > 0:
        print(res.iloc[0])

get_resampled()
