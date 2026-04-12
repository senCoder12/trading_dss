"""
Shared pytest fixtures for the trading_dss test suite.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from src.data.index_registry import IndexRegistry
from src.data.rate_limiter import GlobalCircuitBreaker, GlobalRateLimiter


# ── Reset global singletons between tests ────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_global_singletons():
    """Ensure each test starts with fresh circuit breakers and rate limiters."""
    GlobalCircuitBreaker.reset_all()
    GlobalRateLimiter.reset_all()
    yield
    GlobalCircuitBreaker.reset_all()
    GlobalRateLimiter.reset_all()


# ── Minimal valid indices fixture ─────────────────────────────────────────────

SAMPLE_INDICES = [
    {
        "id": "NIFTY50",
        "display_name": "NIFTY 50",
        "nse_symbol": "NIFTY 50",
        "yahoo_symbol": "^NSEI",
        "exchange": "NSE",
        "lot_size": 75,
        "has_options": True,
        "option_symbol": "NIFTY",
        "sector_category": "broad_market",
        "is_active": True,
        "description": "Test NIFTY 50",
    },
    {
        "id": "BANKNIFTY",
        "display_name": "NIFTY BANK",
        "nse_symbol": "NIFTY BANK",
        "yahoo_symbol": "^NSEBANK",
        "exchange": "NSE",
        "lot_size": 15,
        "has_options": True,
        "option_symbol": "BANKNIFTY",
        "sector_category": "sectoral",
        "is_active": True,
        "description": "Test BankNifty",
    },
    {
        "id": "SENSEX",
        "display_name": "BSE SENSEX",
        "nse_symbol": None,
        "yahoo_symbol": "^BSESN",
        "exchange": "BSE",
        "lot_size": 10,
        "has_options": True,
        "option_symbol": "SENSEX",
        "sector_category": "broad_market",
        "is_active": True,
        "description": "Test SENSEX",
    },
    {
        "id": "NIFTY_IT",
        "display_name": "NIFTY IT",
        "nse_symbol": "NIFTY IT",
        "yahoo_symbol": "^CNXIT",
        "exchange": "NSE",
        "lot_size": None,
        "has_options": False,
        "option_symbol": None,
        "sector_category": "sectoral",
        "is_active": True,
        "description": "Test NIFTY IT",
    },
]


@pytest.fixture()
def indices_json_path(tmp_path: Path) -> Path:
    """Write sample indices to a temp file and return its path."""
    p = tmp_path / "indices.json"
    p.write_text(json.dumps(SAMPLE_INDICES), encoding="utf-8")
    return p


@pytest.fixture()
def registry(indices_json_path: Path) -> IndexRegistry:
    """Return a test IndexRegistry loaded from sample data."""
    return IndexRegistry.from_file(indices_json_path)


# ── Sample OHLCV fixture ──────────────────────────────────────────────────────

@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Return 100 rows of synthetic daily OHLCV data."""
    start = date(2024, 1, 1)
    dates = pd.date_range(start=start, periods=100, freq="B")
    import numpy as np

    rng = np.random.default_rng(42)
    close = 20000.0 + (rng.standard_normal(100).cumsum() * 100)
    high = close + rng.uniform(50, 200, 100)
    low = close - rng.uniform(50, 200, 100)
    open_ = low + rng.uniform(0, 1, 100) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, 100)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


# ── Sample options chain fixture ──────────────────────────────────────────────

@pytest.fixture()
def sample_options_chain() -> pd.DataFrame:
    """Return a minimal options chain DataFrame around 22000."""
    strikes = list(range(21000, 23500, 100))
    import numpy as np
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "strike": strikes,
        "expiry": "27-Jun-2024",
        "ce_oi": rng.integers(5000, 200000, len(strikes)),
        "ce_chg_oi": rng.integers(-10000, 50000, len(strikes)),
        "ce_volume": rng.integers(1000, 50000, len(strikes)),
        "ce_iv": rng.uniform(10, 30, len(strikes)),
        "ce_ltp": rng.uniform(5, 500, len(strikes)),
        "pe_oi": rng.integers(5000, 200000, len(strikes)),
        "pe_chg_oi": rng.integers(-10000, 50000, len(strikes)),
        "pe_volume": rng.integers(1000, 50000, len(strikes)),
        "pe_iv": rng.uniform(10, 30, len(strikes)),
        "pe_ltp": rng.uniform(5, 500, len(strikes)),
    })
