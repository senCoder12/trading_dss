"""
SmartAPI client wrapper — abstracts Angel One broker specifics.

Provides a clean interface for market data operations:
- Bulk quotes for multiple symbols
- Historical OHLCV data for volume calculations
- Options chain data with implied volatility
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.data.broker.angel_one import AngelOneClient

if TYPE_CHECKING:
    from src.data.broker.angel_one import AngelOneClient as AngelOneClientType

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)


class SmartAPIClient:
    """
    SmartAPI client for authenticated market data access.

    Wraps AngelOneClient to provide:
    - Bulk market data for multiple instruments
    - Historical OHLCV (for avg volume calculations)
    - Options chain with Greeks/IV
    """

    def __init__(self, angel_client: Optional[AngelOneClientType] = None) -> None:
        """
        Initialize SmartAPIClient.

        Parameters
        ----------
        angel_client:
            Optional pre-authenticated AngelOneClient. If not provided, creates new one.
            Pass DataRouter.broker_client to reuse existing authenticated session and
            avoid rate-limiting issues.
        """
        self._angel = angel_client if angel_client is not None else AngelOneClient()

    def ensure_authenticated(self) -> bool:
        """Ensure session is active; re-login if needed."""
        return self._angel.ensure_authenticated()

    def get_bulk_quotes(
        self, mode: str, exchange_tokens: dict[str, list[str]]
    ) -> Optional[dict]:
        """
        Fetch bulk market data for multiple instruments.

        Parameters
        ----------
        mode:
            "LTP" (last traded price), "QUOTE" (OHLCV), or "SNAP_QUOTE" (includes depth + OI)
        exchange_tokens:
            {"NSE": ["token1", "token2"], "NFO": ["token3"]}

        Returns
        -------
        dict or None:
            Response data if successful, None if API call failed.
        """
        return self._angel.get_market_data(mode, exchange_tokens)

    def get_historical_ohlcv(
        self,
        exchange: str,
        token: str,
        interval: str = "day",
        count: int = 20,
    ) -> Optional[list[dict]]:
        """
        Fetch historical OHLCV data for volume calculations.

        Parameters
        ----------
        exchange:
            "NSE" | "NFO"
        token:
            Instrument token
        interval:
            "1" (minute) | "5" | "15" | "60" | "day" | "week" | "month"
        count:
            Number of candles to fetch (default 20 for 20-day avg)

        Returns
        -------
        list[dict] or None:
            List of OHLCV bars if successful, None if failed.
        """
        self.ensure_authenticated()
        try:
            smart_api = self._angel._get_smart_api()
            data = smart_api.getCandleData(exchange, token, interval, count)
            if data.get("status"):
                return data.get("data", [])
            logger.warning("Historical fetch failed: %s", data.get("message"))
            return None
        except Exception as exc:
            logger.error("Historical OHLCV fetch failed: %s", exc)
            return None

    def get_option_chain(
        self, name: str, expiry: str, strike_price: str
    ) -> Optional[dict]:
        """
        Fetch options chain with greeks and IV.

        Parameters
        ----------
        name:
            "NIFTY" | "BANKNIFTY" | "RELIANCE" etc.
        expiry:
            "18JAN2024" format
        strike_price:
            Strike (e.g., "22000")

        Returns
        -------
        dict or None:
            Options chain data if successful, None if failed.
        """
        return self._angel.get_option_chain(name, expiry, strike_price)

    def get_master_contract(self, exchange: str = "NFO") -> Optional[object]:
        """
        Download instrument master (contracts, tokens, specs).

        Called once at market open to build token mappings.

        Parameters
        ----------
        exchange:
            "NFO" (F&O) or "NSE" (equity)

        Returns
        -------
        pd.DataFrame or None:
            DataFrame of contracts if successful, None if failed.
        """
        return self._angel.get_master_contract(exchange)
