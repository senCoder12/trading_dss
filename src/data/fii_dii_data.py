"""
FII / DII data module.

Fetches FII/DII institutional activity data from NSE, tracks trends
and saves to the database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from src.data.nse_scraper import NSEScraper
from src.database.db_manager import DatabaseManager
from src.database import queries as Q

logger = logging.getLogger(__name__)


@dataclass
class FIIDIIData:
    """Parsed FII and DII activity for a specific date."""
    date: date
    fii_buy_value: float
    fii_sell_value: float
    fii_net_value: float
    dii_buy_value: float
    dii_sell_value: float
    dii_net_value: float
    fii_fo_buy: float = 0.0
    fii_fo_sell: float = 0.0
    fii_fo_net: float = 0.0


class FIIDIIFetcher:
    """
    Fetches FII/DII trading data, generates trends and caches locally.
    """

    def __init__(self, scraper: Optional[NSEScraper] = None) -> None:
        self._scraper = scraper or NSEScraper()

    def _parse_date(self, date_str: str) -> date:
        try:
            return pd.to_datetime(date_str).date()
        except:
            return date.today()

    def fetch_today_fii_dii(self) -> Optional[FIIDIIData]:
        """
        Fetch the most recent FII/DII activity entry.

        NSE returns a list of records with ``category`` (``"FII/FPI"`` or
        ``"DII"``), ``date``, ``buyValue``, ``sellValue``, ``netValue``.
        """
        raw = self._scraper.get_fii_dii_activity()
        if not raw:
            logger.warning("FII_DII API returned no response")
            return None

        records = raw if isinstance(raw, list) else raw.get("data", [])
        if not records:
            return None

        fii_rec: Optional[dict] = None
        dii_rec: Optional[dict] = None
        trade_date = date.today()

        for rec in records:
            cat = (rec.get("category") or "").upper()
            if "FII" in cat or "FPI" in cat:
                fii_rec = rec
            elif "DII" in cat:
                dii_rec = rec

        if fii_rec:
            trade_date = self._parse_date(fii_rec.get("date", ""))
        elif dii_rec:
            trade_date = self._parse_date(dii_rec.get("date", ""))

        def _val(rec: Optional[dict], key: str) -> float:
            if rec is None:
                return 0.0
            v = rec.get(key, 0.0)
            try:
                return float(str(v).replace(",", ""))
            except (TypeError, ValueError):
                return 0.0

        try:
            return FIIDIIData(
                date=trade_date,
                fii_buy_value=_val(fii_rec, "buyValue"),
                fii_sell_value=_val(fii_rec, "sellValue"),
                fii_net_value=_val(fii_rec, "netValue"),
                dii_buy_value=_val(dii_rec, "buyValue"),
                dii_sell_value=_val(dii_rec, "sellValue"),
                dii_net_value=_val(dii_rec, "netValue"),
            )
        except Exception as e:
            logger.error(f"Failed to parse FII/DII data: {e} | raw: {records}")
            return None

    def fetch_historical_fii_dii(self, start_date: date, end_date: date) -> list[FIIDIIData]:
        """
        Fetch historical FII/DII from NSE for seeding purposes.
        (Note: the default NSE endpoint may only return a few days of history. 
        This attempts to parse all available records up to start_date).
        """
        raw = self._scraper.get_fii_dii_activity()
        records = raw if isinstance(raw, list) else (raw.get("data", []) if isinstance(raw, dict) else [])
        
        historical = []
        for rec in records:
            try:
                trade_date = self._parse_date(rec.get("date", ""))
                if start_date <= trade_date <= end_date:
                    historical.append(FIIDIIData(
                        date=trade_date,
                        fii_buy_value=float(rec.get("fiiBuyValue", 0.0) or 0.0),
                        fii_sell_value=float(rec.get("fiiSellValue", 0.0) or 0.0),
                        fii_net_value=float(rec.get("fiiNetValue", 0.0) or 0.0),
                        dii_buy_value=float(rec.get("diiBuyValue", 0.0) or 0.0),
                        dii_sell_value=float(rec.get("diiSellValue", 0.0) or 0.0),
                        dii_net_value=float(rec.get("diiNetValue", 0.0) or 0.0),
                    ))
            except Exception as e:
                logger.debug(f"Skipping record due to parse error: {e}")
                
        return historical

    def get_fii_trend(self, days: int = 5) -> dict:
        """
        Calculates the FII trend over the last recent `days` using available data.
        Returns empty placeholders if not enough data.
        """
        raw = self._scraper.get_fii_dii_activity()
        records = raw if isinstance(raw, list) else (raw.get("data", []) if isinstance(raw, dict) else [])
        
        recent = records[:days]
        if not recent:
            return {
                "total_net": 0.0,
                "avg_daily_net": 0.0,
                "trend": "NEUTRAL",
                "consecutive_buy_days": 0,
                "consecutive_sell_days": 0
            }

        total_net = 0.0
        consecutive_buy = 0
        consecutive_sell = 0
        streak_broken = False
        
        # the list is latest first in standard NSE responses
        for idx, rec in enumerate(recent):
            net = float(rec.get("fiiNetValue", 0.0) or 0.0)
            total_net += net
            
            if not streak_broken:
                if net > 0 and consecutive_sell == 0:
                    consecutive_buy += 1
                elif net < 0 and consecutive_buy == 0:
                    consecutive_sell += 1
                else:
                    streak_broken = True

        avg_daily = total_net / len(recent)
        
        if avg_daily > 500:
            trend = "BUYING"
        elif avg_daily < -500:
            trend = "SELLING"
        else:
            trend = "NEUTRAL"

        return {
            "total_net": round(total_net, 2),
            "avg_daily_net": round(avg_daily, 2),
            "trend": trend,
            "consecutive_buy_days": consecutive_buy,
            "consecutive_sell_days": consecutive_sell
        }

    def save_to_db(self, data: FIIDIIData, db: DatabaseManager) -> None:
        """
        Saves DII and FII activities cleanly into DB. Supports UPSERT/ON CONFLICT.
        """
        try:
            # Save FII (Cash segment by default)
            db.execute(Q.INSERT_FII_DII_ACTIVITY, (
                data.date.isoformat(), "FII", data.fii_buy_value, 
                data.fii_sell_value, data.fii_net_value, "CASH"
            ))
            # Save DII
            db.execute(Q.INSERT_FII_DII_ACTIVITY, (
                data.date.isoformat(), "DII", data.dii_buy_value, 
                data.dii_sell_value, data.dii_net_value, "CASH"
            ))
            
            # Save F&O if available
            if data.fii_fo_buy > 0 or data.fii_fo_sell > 0:
                db.execute(Q.INSERT_FII_DII_ACTIVITY, (
                    data.date.isoformat(), "FII", data.fii_fo_buy, 
                    data.fii_fo_sell, data.fii_fo_net, "F&O"
                ))
            
            logger.info(f"Saved FII/DII data for {data.date}")
        except Exception as e:
            logger.error(f"Failed saving FII/DII data: {e}")
