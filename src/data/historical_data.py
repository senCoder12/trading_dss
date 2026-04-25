"""
Historical OHLCV downloader via yfinance.

Fetches data, normalizes frames, and persists daily and intraday data efficiently 
to the local database.
"""

from __future__ import annotations

import logging
import time
import sys
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.data.index_registry import IndexRegistry, Index
from src.data.data_validator import validate_price_data
from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from config.constants import IST_TIMEZONE, NSE_BASE_URL
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class HistoricalDataError(Exception):
    """Raised when historical data cannot be fetched or is invalid."""


class HistoricalDataManager:
    """
    Downloads OHLCV history for any index using its Yahoo Finance symbol.
    Provides methods to perform initial seeding, and daily updates.
    """

    def __init__(
        self,
        registry: Optional[IndexRegistry] = None,
        db: Optional[DatabaseManager] = None,
    ) -> None:
        from src.data.index_registry import get_registry  # lazy to avoid circular import
        # Use `is not None` — IndexRegistry defines __len__, so a MagicMock mock
        # with spec=IndexRegistry evaluates as falsy (len==0), breaking `registry or`.
        self._registry = registry if registry is not None else get_registry()
        self._db = db

    def _convert_to_ist(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert a dataframe's DatetimeIndex to IST."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST_TIMEZONE)
        else:
            df.index = df.index.tz_convert(IST_TIMEZONE)
        return df

    _NSE_ARCHIVE_BASE = "https://archives.nseindia.com/content/indices/ind_close_all_{date}.csv"
    _NSE_ARCHIVE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain,*/*",
    }

    def _download_from_nse_history(
        self, archive_name: str, from_date: date, to_date: date
    ) -> pd.DataFrame:
        import requests

        ist = ZoneInfo(IST_TIMEZONE)
        rows: list[dict] = []
        current = from_date

        while current <= to_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            url = self._NSE_ARCHIVE_BASE.format(date=current.strftime("%d%m%Y"))
            try:
                resp = requests.get(
                    url, headers=self._NSE_ARCHIVE_HEADERS, timeout=15
                )
                if resp.status_code == 404:
                    current += timedelta(days=1)
                    continue
                resp.raise_for_status()
            except Exception as exc:
                logger.debug("NSE archive miss for %s: %s", current, exc)
                current += timedelta(days=1)
                time.sleep(0.3)
                continue

            found = False
            for line in resp.text.splitlines()[1:]:
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                row_name = parts[0].strip()
                if row_name.lower() != archive_name.lower():
                    continue
                try:
                    ts = datetime.strptime(parts[1].strip(), "%d-%m-%Y").replace(
                        hour=15, minute=30, tzinfo=ist
                    )
                    o = float(parts[2]) if parts[2].strip() not in ("-", "") else 0.0
                    h = float(parts[3]) if parts[3].strip() not in ("-", "") else 0.0
                    lo = float(parts[4]) if parts[4].strip() not in ("-", "") else 0.0
                    cl = float(parts[5]) if parts[5].strip() not in ("-", "") else 0.0
                    if cl > 0:
                        rows.append({"timestamp": ts, "open": o, "high": h,
                                     "low": lo, "close": cl, "volume": 0.0})
                    found = True
                except (ValueError, IndexError):
                    pass
                break

            if not found:
                logger.debug("'%s' not found in NSE archive for %s", archive_name, current)

            current += timedelta(days=1)
            time.sleep(0.25)

        if not rows:
            raise HistoricalDataError(
                f"NSE archives returned no data for '{archive_name}' "
                f"({from_date} – {to_date})"
            )

        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df[df["close"] > 0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_not_exception_type(HistoricalDataError),
        reraise=True,
    )
    def download_index_history(
        self, index_id: str, period: str = "2y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data using yfinance, with NSE archive fallback.

        Parameters
        ----------
        index_id: str
            Index id from the registry.
        period: str
            yfinance period string, e.g., "ytd", "max", "2y". Defaults to "2y".
        interval: str
            "1d", "1wk", "1mo" etc.
        """
        definition = self._registry.get_index(index_id)
        if definition is None:
            raise HistoricalDataError(f"Unknown index: {index_id}")

        df = None
        if definition.yahoo_symbol and interval == "1d":
            try:
                ticker = yf.Ticker(definition.yahoo_symbol)
                df = ticker.history(period=period, interval=interval, auto_adjust=True)
                if len(df) >= 10:
                    df = df.rename(columns=str.lower)
                    df.index.name = "timestamp"
                    df.reset_index(inplace=True)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                    df = self._convert_to_ist(df)
                    df = df.reset_index()
                    df = df.replace([np.inf, -np.inf], np.nan)
                    if "dividends" in df.columns:
                        df = df.drop(columns=["dividends"])
                    if "stock splits" in df.columns:
                        df = df.drop(columns=["stock splits"])
                    df = df.dropna(subset=["close"])
                    return df
            except Exception as exc:
                logger.warning("yfinance failed for %s: %s", index_id, exc)

        if definition.nse_archive_name:
            days_map = {"1y": 365, "2y": 730, "max": 3650}
            base_days = days_map.get(period, 365)
            if period == "ytd":
                from_date = date(date.today().year, 1, 1)
                to_date = date.today()
            else:
                to_date = date.today()
                from_date = to_date - timedelta(days=base_days)
            return self._download_from_nse_history(definition.nse_archive_name, from_date, to_date)

        raise HistoricalDataError(
            f"Cannot download {index_id}: no yahoo_symbol or nse_archive_name"
        )

    def download_intraday(self, index_id: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
        """Download intraday 5m data. yfinance supports max 60d for 5m."""
        return self.download_index_history(index_id, period=period, interval=interval)

    def download_all_active_indices(self, period: str = "2y", interval: str = "1d") -> dict:
        """
        Iterates and downloads historical records for all active indices, 
        saving the results straight to the database.
        """
        if not self._db:
            raise ValueError("Database connection required to save bulk history.")

        indices = self._registry.get_active_indices()
        summary = {"downloaded": 0, "failed": 0, "total_records": 0}

        for idx in indices:
            if not idx.yahoo_symbol and not idx.nse_archive_name:
                continue

            try:
                df = self.download_index_history(idx.id, period=period, interval=interval)
                records_count = len(df)
                self._save_dataframe_to_db(idx.id, df, timeframe=interval)
                
                logger.info(f"Downloaded {idx.id}: {records_count} {interval} candles")
                summary["downloaded"] += 1
                summary["total_records"] += records_count
            except Exception as e:
                logger.error(f"Failed handling {idx.id}: {e}")
                summary["failed"] += 1
                
            time.sleep(1.5)  # respectful yfinance rate limit padding

        return summary

    def update_daily_data(self) -> None:
        """
        Check database for missing records and download incrementally to keep pricing updated.
        Called typically daily after market close.
        """
        if not self._db:
            raise ValueError("Database connection required to perform daily updates.")
            
        indices = self._registry.get_active_indices()
        
        for idx in indices:
            if not idx.yahoo_symbol and not idx.nse_archive_name:
                continue

            try:
                # Find the latest stored date
                now_iso = datetime.now(tz=ZoneInfo(IST_TIMEZONE)).isoformat()
                latest_row = self._db.fetch_one(Q.GET_LATEST_PRICE, (idx.id, "1d", now_iso))
                
                if latest_row:
                    last_ts = pd.to_datetime(latest_row["timestamp"]).tz_convert(IST_TIMEZONE)
                    now_ts = datetime.now(tz=ZoneInfo(IST_TIMEZONE))
                    
                    diff_days = (now_ts.date() - last_ts.date()).days
                    if diff_days <= 1:
                        # Up to date
                        continue
                        
                    period = f"{diff_days + 1}d"
                else:
                    period = "1y" # Data missing completely, backfill short history
                    
                df = self.download_index_history(idx.id, period=period, interval="1d")
                self._save_dataframe_to_db(idx.id, df, timeframe="1d")
                logger.info(f"Updated {idx.id} with recent daily entries")
                
                time.sleep(1) # rate limiting
                
            except Exception as e:
                logger.error(f"Failed to update daily data for {idx.id}: {e}")

    def get_stored_history(self, index_id: str, start_date: str, end_date: str, timeframe: str = "1d") -> pd.DataFrame:
        """
        Query OHLCV dataset from local SQLite storage instead of redownloading via YFinance.
        Arguments start_date/end_date should be isoformat strings.
        """
        if not self._db:
            raise ValueError("Database connection required to read history.")
            
        # Get data backwards from limit assuming robust date tracking
        rows = self._db.fetch_all(Q.LIST_PRICE_HISTORY, (index_id, timeframe, start_date, end_date))
        if not rows:
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)
        # Parse timestamp string values properly from SQLite representation
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        return df

    def seed_historical_data(self) -> None:
        """
        One-time function for initial database structural setup filling table `price_data`.
        Has progress bars via standard print to stdout.
        """
        if not self._db:
            raise ValueError("Database connection required to seed historical data.")
            
        indices = self._registry.get_active_indices()
        total = len(indices)
        
        print("\n[Seeding Phase 1] Backwards historical daily tracking (2Y)...")
        for i, idx in enumerate(indices):
            print(f"\rProgress Daily: {i}/{total} [{idx.id}]        ", end="")
            sys.stdout.flush()
            if not idx.yahoo_symbol:
                continue
            try:
                df_daily = self.download_index_history(idx.id, period="2y", interval="1d")
                self._save_dataframe_to_db(idx.id, df_daily, timeframe="1d")
            except Exception:
                pass 
            time.sleep(0.5)

        print("\n\n[Seeding Phase 2] Backwards intraday FO tracking (5d / 5m)...")
        fo_indices = self._registry.get_indices_with_options()
        total_fo = len(fo_indices)
        for i, idx in enumerate(fo_indices):    
            print(f"\rProgress Intraday: {i}/{total_fo} [{idx.id}]        ", end="")
            sys.stdout.flush()
            if not idx.yahoo_symbol:
                continue
            try:
                df_intra = self.download_intraday(idx.id, period="5d", interval="5m")
                self._save_dataframe_to_db(idx.id, df_intra, timeframe="5m")
            except Exception:
                pass
            time.sleep(0.5)
            
        print("\nSeeding completely finished!")

    def _save_dataframe_to_db(self, index_id: str, df: pd.DataFrame, timeframe: str) -> None:
        """Batch UPSERT price rows into the database, skipping invalid rows."""
        if df.empty or not self._db:
            return

        inserts = []
        skipped = 0
        for _, row in df.iterrows():
            close_val = float(row.get("close", 0.0))
            price_dict = {
                "ltp":    close_val,   # closing price is the last traded price for a bar
                "open":   float(row.get("open",   0.0)),
                "high":   float(row.get("high",   0.0)),
                "low":    float(row.get("low",    0.0)),
                "close":  close_val,
                "volume": float(row.get("volume", 0.0)),
            }
            result = validate_price_data(price_dict)
            if not result.is_valid:
                logger.debug(
                    "Skipping invalid row for %s at %s: %s",
                    index_id, row.get("timestamp"), result.errors,
                )
                skipped += 1
                continue

            ts_str = row["timestamp"].isoformat()
            inserts.append((
                index_id, ts_str,
                price_dict["open"], price_dict["high"],
                price_dict["low"],  price_dict["close"],
                price_dict["volume"],
                0.0,          # vwap — not available from yfinance OHLCV
                "YFINANCE", timeframe,
            ))

        if skipped:
            logger.warning("Skipped %d invalid rows for %s", skipped, index_id)

        if inserts:
            try:
                self._db.execute_many(Q.UPSERT_PRICE_DATA, inserts)
            except Exception as e:
                logger.error("Failed upserting prices for %s: %s", index_id, e)
