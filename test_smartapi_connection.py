#!/usr/bin/env python3
"""
Test SmartAPI connection, authentication, and basic API calls.

Run: python test_smartapi_connection.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_angel_one_client():
    """Test AngelOneClient authentication."""
    print("\n" + "="*60)
    print("TEST 1: AngelOneClient Authentication")
    print("="*60)

    try:
        from src.data.broker.angel_one import AngelOneClient

        print("\n1. Creating AngelOneClient...")
        client = AngelOneClient()
        print(f"   ✅ Client created")
        print(f"   API Key: {client.api_key[:20]}..." if client.api_key else "   ⚠️  API Key missing")
        print(f"   Client ID: {client.client_id}" if client.client_id else "   ⚠️  Client ID missing")
        print(f"   Auth Token: {client.auth_token}")

        print("\n2. Attempting login...")
        success = client.login()
        if success:
            print(f"   ✅ Login successful")
            print(f"   Auth Token: {client.auth_token[:50]}..." if client.auth_token else "   ⚠️  No auth token")
            print(f"   Feed Token: {client.feed_token}")
            return client
        else:
            print(f"   ❌ Login failed")
            return None
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_smart_api_client(angel_client=None):
    """Test SmartAPIClient wrapper."""
    print("\n" + "="*60)
    print("TEST 2: SmartAPIClient Wrapper")
    print("="*60)

    try:
        from src.data.smart_api_client import SmartAPIClient

        print("\n1. Creating SmartAPIClient...")
        if angel_client:
            client = SmartAPIClient(angel_client=angel_client)
            print(f"   ✅ Client created (reusing authenticated AngelOneClient)")
        else:
            client = SmartAPIClient()
            print(f"   ✅ Client created (new AngelOneClient)")

        print("\n2. Ensuring authentication...")
        auth_ok = client.ensure_authenticated()
        if auth_ok:
            print(f"   ✅ Authentication check passed")
            return client
        else:
            print(f"   ❌ Authentication check failed")
            return None
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_master_contract(smart_api):
    """Test master contract download."""
    print("\n" + "="*60)
    print("TEST 3: Master Contract Download")
    print("="*60)

    try:
        print("\n1. Downloading NSE master contract...")
        df = smart_api.get_master_contract("NSE")
        if df is not None:
            print(f"   ✅ Downloaded {len(df)} records")
            print(f"   Columns: {list(df.columns)[:5]}...")
            if len(df) > 0:
                print(f"   First row: {df.iloc[0].to_dict()}")
        else:
            print(f"   ❌ Failed to download (returned None)")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()


def test_bulk_quotes(smart_api):
    """Test bulk quotes API."""
    print("\n" + "="*60)
    print("TEST 4: Bulk Market Data API")
    print("="*60)

    try:
        print("\n1. Fetching LTP for sample symbols...")
        exchange_tokens = {
            "NSE": ["RELIANCE", "INFY", "TCS"]  # Try symbol names first
        }

        print(f"   Request: {exchange_tokens}")
        result = smart_api.get_bulk_quotes("LTP", exchange_tokens)

        if result:
            print(f"   ✅ Got response")
            print(f"   Response keys: {list(result.keys())}")
            print(f"   Fetched: {len(result.get('fetched', []))} symbols")
            print(f"   Unfetched: {len(result.get('unfetched', []))} symbols")
            if result.get('fetched'):
                print(f"   First entry: {result['fetched'][0]}")
        else:
            print(f"   ❌ No response (returned None)")

        print("\n2. Trying QUOTE mode...")
        result = smart_api.get_bulk_quotes("QUOTE", exchange_tokens)
        if result:
            print(f"   ✅ Got response")
            print(f"   Fetched: {len(result.get('fetched', []))} symbols")
        else:
            print(f"   ❌ No response (returned None)")

    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SmartAPI Integration Test Suite")
    print("="*60)

    # Test 1: Angel One Client
    angel_client = test_angel_one_client()

    # Test 2: SmartAPI Client wrapper (pass authenticated angel_client to avoid rate limiting)
    smart_client = test_smart_api_client(angel_client=angel_client)

    # Test 3: Master contract
    if smart_client:
        test_master_contract(smart_client)

    # Test 4: Bulk quotes
    if smart_client:
        test_bulk_quotes(smart_client)

    print("\n" + "="*60)
    print("Test suite complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
