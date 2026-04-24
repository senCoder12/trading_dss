
import time
import random
from curl_cffi import requests
import json

NSE_BASE_URL = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

def test_option_chain(symbol="NIFTY"):
    session = requests.Session(impersonate="chrome131")
    
    # 1. Warm up
    print(f"Warming up session by visiting {NSE_BASE_URL}...")
    r1 = session.get(NSE_BASE_URL, headers=NSE_HEADERS, timeout=15)
    print(f"Warm up status: {r1.status_code}")
    
    # 2. Try to get option chain
    api_url = f"{NSE_BASE_URL}/api/option-chain-indices"
    api_headers = NSE_HEADERS.copy()
    api_headers.update({
        "Accept": "*/*",
        "Referer": f"https://www.nseindia.com/get-quotes/derivatives?symbol={symbol}",
        "X-Requested-With": "XMLHttpRequest",
    })
    
    print(f"Fetching option chain for {symbol}...")
    params = {"symbol": symbol}
    r2 = session.get(api_url, headers=api_headers, params=params, timeout=15)
    
    print(f"Response status: {r2.status_code}")
    print(f"Response headers: {json.dumps(dict(r2.headers), indent=2)}")
    
    try:
        data = r2.json()
        if not data:
            print("Response is empty JSON.")
        else:
            print(f"Data keys: {list(data.keys())}")
            if "records" in data:
                print(f"Number of strikes: {len(data['records'].get('data', []))}")
                print(f"Expiry dates: {data['records'].get('expiryDates')[:5]}")
            else:
                print("No 'records' key in data.")
                print(f"Full data: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Response text (first 500 chars): {r2.text[:500]}")

if __name__ == "__main__":
    test_option_chain("NIFTY")
