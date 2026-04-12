"""
India VIX data tracker.

Tracks live India VIX and determines volatility regimes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data.nse_scraper import NSEScraper
from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from config.constants import IST_TIMEZONE
from config.settings import settings
from src.data.rate_limiter import freshness_tracker
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)


class VIXRegime(str):
    """VIX-based volatility regime label."""
    LOW_VOL = "LOW_VOL"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH_VOL = "HIGH_VOL"


@dataclass
class VIXData:
    """A single VIX reading."""
    value: float
    change: float
    change_pct: float
    timestamp: datetime


@dataclass
class VIXSnapshot:
    """VIX reading with a pre-computed regime classification."""
    value: float
    change: float
    change_pct: float
    timestamp: datetime
    regime: str  # one of VIXRegime constants


class VIXTracker:
    """
    Fetches India VIX history and current values, and determines volatility regime.
    """

    def __init__(self, scraper: NSEScraper) -> None:
        self._scraper = scraper

    def get_current_vix(self) -> Optional[VIXData]:
        """Fetch the current VIX value from NSE."""
        raw = self._scraper.get_vix()
        if not raw:
            logger.warning("VIX data not available from NSEScraper")
            return None

        freshness_tracker.update("vix")
        return VIXData(
            value=raw.get("vix_value", 0.0),
            change=raw.get("vix_change", 0.0),
            change_pct=raw.get("vix_change_pct", 0.0),
            timestamp=datetime.now(tz=_IST)
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def get_vix_history(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch VIX history from yfinance using ^INDIAVIX.
        Returns a DataFrame with lowercase columns.
        """
        logger.info(f"Fetching {days} days of VIX history from yfinance...")
        ticker = yf.Ticker("^INDIAVIX")
        df = ticker.history(period=f"{days}d", auto_adjust=True)
        if df.empty:
            logger.warning("yfinance returned empty dataframe for ^INDIAVIX")
            return pd.DataFrame()

        df = df.rename(columns=str.lower)
        df.index.name = "timestamp"
        df.reset_index(inplace=True)
        return df

    def get_vix_regime(self) -> str:
        """
        Classifies the current market volatility regime based on VIX.

        Thresholds are configurable via ``settings.thresholds``:

        - Below ``vix_normal_threshold`` (default 13.0)  → ``"LOW_VOL"``
        - Below ``vix_elevated_threshold`` (default 18.0) → ``"NORMAL"``
        - Below ``vix_panic_threshold`` (default 25.0)    → ``"ELEVATED"``
        - At or above ``vix_panic_threshold``             → ``"HIGH_VOL"``
        """
        vix_data = self.get_current_vix()
        if not vix_data:
            logger.warning("Could not determine VIX regime because VIX data is missing")
            return "NORMAL"

        val = vix_data.value
        t = settings.thresholds
        if val < t.vix_normal_threshold:
            return "LOW_VOL"
        if val < t.vix_elevated_threshold:
            return "NORMAL"
        if val < t.vix_panic_threshold:
            return "ELEVATED"
        return "HIGH_VOL"

    def save_to_db(self, data: VIXData, db: DatabaseManager) -> None:
        """Save VIX data into the vix_data table."""
        try:
            db.execute(
                Q.INSERT_VIX_DATA, 
                (data.timestamp.isoformat(), data.value, data.change, data.change_pct)
            )
            logger.debug(f"Saved VIX Data: {data.value} at {data.timestamp.isoformat()}")
        except Exception as e:
            logger.error(f"Failed to save VIX data to database: {e}")
