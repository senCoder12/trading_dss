import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.broker.angel_one import AngelOneClient
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_login():
    logger.info(f"Testing Angel One Login for Client: {settings.broker.angel_one_client_id}")
    client = AngelOneClient()
    
    if client.login():
        logger.info("Login Successful!")
        # Fetch Nifty 50 LTP as a test
        # Note: We need a valid token. For Nifty 50 Index on NSE, token is usually '99926000' or similar.
        # Let's try to get master contract for tokens.
        logger.info("Fetching master contract...")
        df = client.get_master_contract("NSE")
        if df is not None:
            nifty_row = df[df['symbol'] == 'Nifty 50']
            if not nifty_row.empty:
                token = nifty_row.iloc[0]['token']
                symbol = nifty_row.iloc[0]['symbol']
                logger.info(f"Fetching LTP for {symbol} (Token: {token})...")
                ltp = client.get_ltp("NSE", symbol, token)
                logger.info(f"LTP: {ltp}")
            else:
                logger.warning("Nifty 50 not found in master contract. Trying direct token '99926000'")
                ltp = client.get_ltp("NSE", "Nifty 50", "99926000")
                logger.info(f"LTP: {ltp}")
    else:
        logger.error("Login Failed!")

if __name__ == "__main__":
    test_login()
