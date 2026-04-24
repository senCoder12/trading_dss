import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import get_db_manager
from src.data.index_registry import IndexRegistry
from src.data.data_collector import DataCollector
from config.settings import settings

def force_sync():
    db = get_db_manager()
    db.connect()
    registry = IndexRegistry.from_file(settings.indices_config_path)
    
    # We turn OFF broker API just for this sync to force Scraper data into DB
    collector = DataCollector(db, registry, force_start=True)
    print("Force collecting index prices from NSE Scraper...")
    collector.collect_index_prices()
    print("Price sync complete.")

if __name__ == "__main__":
    force_sync()
