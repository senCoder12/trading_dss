import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.broker.angel_one import AngelOneClient

def verify_prices():
    client = AngelOneClient()
    if client.login():
        indices = {
            "NIFTY50": ("NSE", "Nifty 50", "26000"),
            "BANKNIFTY": ("NSE", "Nifty Bank", "26009"),
            "FINNIFTY": ("NSE", "Nifty Fin Service", "99926037"),
            "MIDCPNIFTY": ("NSE", "NIFTY MID SELECT", "99926074")
        }
        
        print("\nPrice Verification (REST LTP):")
        for name, (exch, sym, token) in indices.items():
            ltp = client.get_ltp(exch, sym, token)
            print(f"{name:<12} | Token: {token:<8} | LTP: {ltp}")
    else:
        print("Login Failed")

if __name__ == "__main__":
    verify_prices()
