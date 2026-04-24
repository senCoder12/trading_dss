import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.broker.angel_one import AngelOneClient
import pandas as pd

def check_tokens():
    client = AngelOneClient()
    print("Loading NSE Master Contract...")
    df_nse = client.get_master_contract("NSE")
    
    tokens = [
        "3045", "26000", "1", "12", "99926000", 
        "99926009", "99926037", "99926074", "26009"
    ]
    print("\nToken Analysis:")
    for t in tokens:
        row = df_nse[df_nse['token'] == t]
        if not row.empty:
            print(f"Token {t}: Symbol={row.iloc[0]['symbol']}, Name={row.iloc[0]['name']}")
        else:
            print(f"Token {t}: Not found in NSE master")

if __name__ == "__main__":
    check_tokens()
