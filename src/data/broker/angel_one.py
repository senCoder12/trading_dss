"""
Angel One SmartAPI client.

Handles authentication (login + TOTP 2FA), session refresh, and REST data
access.  Import lazily: SmartApi and pyotp are only required when USE_BROKER_API
is True, so missing the package does not break the scraper-only path.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# URL for the daily instrument master (stable across API versions)
_SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)

# Index tokens for REST calls (getCandleData / getMarketData).
# These differ from the WebSocket tokens: NIFTY 50 is "26000" on the feed but
# "99926000" on REST — the feed token returns no candles.
INDEX_REST_TOKENS: dict[str, tuple[str, str]] = {
    "NIFTY50": ("NSE", "99926000"),
    "BANKNIFTY": ("NSE", "99926009"),
    "FINNIFTY": ("NSE", "99926037"),
    "MIDCPNIFTY": ("NSE", "99926074"),
    "SENSEX": ("BSE", "99919000"),
}
INDIA_VIX_REST_TOKEN: tuple[str, str] = ("NSE", "99926017")

# Max span per getCandleData request, by interval (Angel One API limits)
CANDLE_MAX_DAYS: dict[str, int] = {
    "ONE_MINUTE": 30,
    "FIVE_MINUTE": 100,
    "FIFTEEN_MINUTE": 200,
    "ONE_DAY": 2000,
}

# Max tokens per getMarketData request
MARKET_DATA_MAX_TOKENS = 50


class AngelOneClient:
    """Angel One SmartAPI client for authenticated market data access.

    Covers:
    - TOTP-based login and automatic session refresh
    - LTP and bulk market data (REST)
    - Options chain via optionGreeks
    - NFO master contract download (for token lookups)
    """

    def __init__(self) -> None:
        self.api_key = settings.broker.angel_one_api_key
        self.client_id = settings.broker.angel_one_client_id
        self.password = settings.broker.angel_one_password
        self.totp_secret = settings.broker.angel_one_totp_secret

        self._smart_api = None
        self.auth_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.feed_token: Optional[str] = None  # Required for WebSocket
        self._last_login: Optional[datetime] = None
        # Full scrip master, cached per IST day (the file is ~40 MB)
        self._master_df: Optional["pd.DataFrame"] = None
        self._master_date = None

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_smart_api(self):
        """Lazily initialise SmartConnect (raises clearly if package missing)."""
        if self._smart_api is None:
            try:
                from SmartApi import SmartConnect  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "smartapi-python not installed — run: pip install smartapi-python pyotp"
                ) from exc
            self._smart_api = SmartConnect(api_key=self.api_key)
        return self._smart_api

    # ── Authentication ─────────────────────────────────────────────────────

    def login(self) -> bool:
        """Authenticate with Angel One using TOTP 2FA. Returns True on success."""
        try:
            import pyotp  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "pyotp not installed — run: pip install smartapi-python pyotp"
            ) from exc

        try:
            totp = pyotp.TOTP(self.totp_secret).now()
            smart_api = self._get_smart_api()
            data = smart_api.generateSession(self.client_id, self.password, totp)

            if data["status"]:
                self.auth_token = data["data"]["jwtToken"]
                self.refresh_token = data["data"]["refreshToken"]
                self.feed_token = smart_api.getfeedToken()
                self._last_login = datetime.now(tz=_IST)
                logger.info("Angel One: logged in successfully")
                return True

            logger.error("Angel One login failed: %s", data.get("message"))
            return False

        except Exception as exc:
            logger.error("Angel One login error: %s", exc)
            return False

    def ensure_authenticated(self) -> bool:
        """Re-login if session is absent or older than 20 hours."""
        if not self.auth_token:
            return self.login()
        if self._last_login:
            age_hours = (datetime.now(tz=_IST) - self._last_login).total_seconds() / 3600
            if age_hours > 20:
                return self.login()
        return True

    # ── REST data methods ──────────────────────────────────────────────────

    def get_ltp(self, exchange: str, symbol: str, token: str) -> Optional[float]:
        """Return last traded price for a single instrument."""
        self.ensure_authenticated()
        try:
            data = self._get_smart_api().ltpData(exchange, symbol, token)
            if data["status"]:
                return float(data["data"]["ltp"])
        except Exception as exc:
            logger.error("LTP fetch failed for %s: %s", symbol, exc)
        return None

    def get_market_data(self, mode: str, exchange_tokens: dict) -> Optional[dict]:
        """Bulk market data for multiple instruments.

        mode: "LTP" | "QUOTE" | "SNAP_QUOTE" (SNAP_QUOTE includes depth + OI)
        exchange_tokens: {"NSE": ["3045", "26000"], "NFO": ["35003"]}
        """
        self.ensure_authenticated()
        try:
            data = self._get_smart_api().getMarketData(mode, exchange_tokens)
            logger.debug("getMarketData response status=%s, keys=%s",
                         data.get("status"), list(data.keys()) if data else None)
            if data["status"]:
                return data["data"]
            logger.warning("Market data fetch failed: %s", data.get("message"))
        except Exception as exc:
            logger.error("Market data fetch failed: %s", exc)
        return None

    def get_candle_data(
        self, exchange: str, token: str, interval: str, from_dt: datetime, to_dt: datetime
    ) -> Optional[list[list]]:
        """Historical OHLCV candles for one instrument.

        interval: "ONE_MINUTE" | "FIVE_MINUTE" | "FIFTEEN_MINUTE" | "ONE_DAY"
        Span must not exceed CANDLE_MAX_DAYS[interval]; callers chunk longer ranges.
        Returns rows of [iso_timestamp, open, high, low, close, volume], or None on failure.
        """
        self.ensure_authenticated()
        try:
            data = self._get_smart_api().getCandleData({
                "exchange": exchange,
                "symboltoken": str(token),
                "interval": interval,
                "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
                "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
            })
            if data and data.get("status"):
                return data.get("data") or []
            logger.warning("Candle fetch failed for %s:%s: %s", exchange, token,
                           data.get("message") if data else "no response")
        except Exception as exc:
            logger.error("Candle fetch failed for %s:%s: %s", exchange, token, exc)
        return None

    def get_option_chain(
        self, name: str, expiry: str, strike_price: str
    ) -> Optional[dict]:
        """Options chain with greeks.

        name: "NIFTY" | "BANKNIFTY" etc.
        expiry: "18JAN2024" format
        strike_price: "22000"
        """
        self.ensure_authenticated()
        try:
            params = {
                "name": name,
                "expirydate": expiry,
                "strikeprice": strike_price
            }
            data = self._get_smart_api().optionGreek(params)
            if data["status"]:
                return data["data"]
        except Exception as exc:
            logger.error("Options chain fetch failed: %s", exc)
        return None

    def get_master_contract(self, exchange: str = "NFO") -> Optional["pd.DataFrame"]:
        """Return today's instrument master rows for one exchange segment.

        The full file is downloaded once per IST day and cached; Angel One
        refreshes it before market open.
        """
        try:
            import pandas as pd  # noqa: PLC0415
            import requests  # noqa: PLC0415

            today = datetime.now(tz=_IST).date()
            if self._master_df is None or self._master_date != today:
                resp = requests.get(_SCRIP_MASTER_URL, timeout=30)
                resp.raise_for_status()
                self._master_df = pd.DataFrame(resp.json())
                self._master_date = today
            seg = self._master_df[self._master_df["exch_seg"] == exchange].copy()
            logger.info("Master contract loaded: %d %s instruments", len(seg), exchange)
            return seg
        except Exception as exc:
            logger.error("Master contract download failed: %s", exc)
            return None

    def get_token_for_symbol(
        self,
        symbol: str,
        expiry: str,
        strike: int,
        option_type: str,
        master_df: "pd.DataFrame",
    ) -> Optional[str]:
        """Resolve Angel One token for a specific options contract.

        Angel One stores strikes as integer × 100 (e.g. 22500 → 2250000).
        option_type: "CE" | "PE"
        """
        mask = (
            (master_df["name"] == symbol)
            & (master_df["expiry"] == expiry)
            & (master_df["strike"] == strike * 100)
            & (master_df["symbol"].str.endswith(option_type))
        )
        result = master_df[mask]
        if not result.empty:
            return str(result.iloc[0]["token"])
        return None
