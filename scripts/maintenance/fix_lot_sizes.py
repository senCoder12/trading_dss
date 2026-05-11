import json
import logging
import requests
import pandas as pd
from pathlib import Path
import sqlite3
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import settings
from src.database.db_manager import get_db_manager
from src.data.index_registry import IndexRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix_lot_sizes")

_SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

def update_lot_sizes():
    # 1. Download Master Contract
    logger.info("Downloading Angel One master contract...")
    try:
        resp = requests.get(_SCRIP_MASTER_URL, timeout=60)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        logger.info(f"Loaded {len(df)} instruments")
    except Exception as e:
        logger.error(f"Failed to download master contract: {e}")
        return

    # 2. Extract Lot Sizes
    # For Indices, we look at 'name' and 'exch_seg' == 'NFO'
    # Actually, for indices, it's better to look at specific rows.
    
    index_map = {
        "NIFTY": "NIFTY50",
        "BANKNIFTY": "BANKNIFTY",
        "FINNIFTY": "FINNIFTY",
        "MIDCPNIFTY": "MIDCPNIFTY",
        "SENSEX": "SENSEX"
    }
    
    found_lot_sizes = {}
    
    # Filter NFO instruments
    nfo_df = df[df["exch_seg"] == "NFO"].copy()
    
    # Get lot sizes for indices
    for name, index_id in index_map.items():
        subset = nfo_df[nfo_df["name"] == name]
        if not subset.empty:
            lot = int(subset.iloc[0]["lotsize"])
            found_lot_sizes[index_id] = lot
            logger.info(f"Index {index_id} lot size: {lot}")

    # Get lot sizes for stocks
    # For stocks, 'name' is the symbol. We can group by name.
    stock_lots = nfo_df.groupby("name")["lotsize"].first().to_dict()
    
    # 3. Update config/indices.json
    indices_path = Path("config/indices.json")
    if indices_path.exists():
        logger.info("Updating config/indices.json...")
        with open(indices_path, "r") as f:
            indices = json.load(f)
            
        updated = False
        for idx in indices:
            idx_id = idx["id"]
            if idx_id in found_lot_sizes:
                if idx["lot_size"] != found_lot_sizes[idx_id]:
                    logger.info(f"Updating {idx_id} lot size: {idx['lot_size']} -> {found_lot_sizes[idx_id]}")
                    idx["lot_size"] = found_lot_sizes[idx_id]
                    updated = True
            elif idx.get("has_options") and idx_id in stock_lots:
                 lot = int(stock_lots[idx_id])
                 if idx["lot_size"] != lot:
                    logger.info(f"Updating stock index {idx_id} lot size: {idx['lot_size']} -> {lot}")
                    idx["lot_size"] = lot
                    updated = True
                    
        if updated:
            with open(indices_path, "w") as f:
                json.dump(indices, f, indent=2)
            logger.info("config/indices.json updated.")
        else:
            logger.info("config/indices.json already up to date.")

    # 4. Update Database
    db = get_db_manager()
    db.connect()
    
    # Update index_master
    logger.info("Updating index_master table...")
    for idx_id, lot in found_lot_sizes.items():
        db.execute("UPDATE index_master SET lot_size = ? WHERE id = ?", (lot, idx_id))
    
    # Update fo_universe
    logger.info("Updating fo_universe table...")
    update_count = 0
    for symbol, lot in stock_lots.items():
        # Check if it's a stock (not one of the indices)
        if symbol not in index_map:
            cursor = db.execute("UPDATE fo_universe SET lot_size = ? WHERE symbol = ?", (int(lot), symbol))
            if cursor.rowcount > 0:
                update_count += 1
                
    logger.info(f"Updated lot sizes for {update_count} stocks in fo_universe.")
    
    # 5. Sync Registry
    # This ensures the in-memory registry is also updated if we are running.
    # But this script is standalone.
    
    db.close()
    logger.info("Lot size update complete.")

if __name__ == "__main__":
    update_lot_sizes()
