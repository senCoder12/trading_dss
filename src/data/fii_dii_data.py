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

    def fetch_futures_participant_positions(self) -> Optional[dict]:
        """Fetch FII net futures/options positions from NSE participant-wise data.

        This is the KEY institutional signal — FII net long/short in index futures
        is the primary directional bias indicator. Published every 30 min by NSE.

        Returns dict with keys:
            fii_index_futures_net_long: FII net long contracts in index futures
            fii_index_futures_net_short: FII net short contracts
            fii_net_position: net (long - short)
            fii_fo_trend: BULLISH / BEARISH / NEUTRAL
            dii_index_futures_net_long: DII net long
            dii_net_position: DII net
        """
        raw = self._scraper.get_fii_futures_positions()
        if not raw:
            return None

        try:
            data = raw if isinstance(raw, list) else raw.get("data", [])
            if not data:
                return None

            fii_futures: dict = {}
            dii_futures: dict = {}

            for rec in data:
                cat = str(rec.get("clientType") or rec.get("category") or "").upper()
                seg = str(rec.get("instrumentType") or rec.get("instrument") or "").upper()

                is_fii = "FII" in cat or "FPI" in cat or "FOREIGN" in cat
                is_dii = "DII" in cat or "DOMESTIC" in cat
                is_index_futures = "INDEX" in seg and ("FUT" in seg or "FUTURE" in seg)

                if is_fii and is_index_futures:
                    fii_futures = rec
                elif is_dii and is_index_futures:
                    dii_futures = rec

            def _n(d: dict, *keys) -> float:
                for k in keys:
                    v = d.get(k)
                    if v is not None:
                        try:
                            return float(str(v).replace(",", ""))
                        except (TypeError, ValueError):
                            pass
                return 0.0

            fii_long = _n(fii_futures, "longContracts", "buyContracts", "buyQuantity")
            fii_short = _n(fii_futures, "shortContracts", "sellContracts", "sellQuantity")
            fii_net = fii_long - fii_short

            dii_long = _n(dii_futures, "longContracts", "buyContracts", "buyQuantity")
            dii_short = _n(dii_futures, "shortContracts", "sellContracts", "sellQuantity")
            dii_net = dii_long - dii_short

            # FII trend based on net position magnitude
            if fii_net > 5000:
                trend = "BULLISH"
            elif fii_net < -5000:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            return {
                "fii_index_futures_net_long": fii_long,
                "fii_index_futures_net_short": fii_short,
                "fii_net_position": fii_net,
                "fii_fo_trend": trend,
                "dii_index_futures_net_long": dii_long,
                "dii_net_position": dii_net,
                "raw_records_count": len(data),
            }
        except Exception as e:
            logger.error("Failed to parse futures participant positions: %s", e)
            return None

    def save_futures_positions_to_db(self, positions: dict, db: DatabaseManager) -> None:
        """Save FII futures positions to DB for trend analysis."""
        try:
            from src.data.rate_limiter import freshness_tracker
            db.execute(
                """INSERT OR REPLACE INTO fii_dii_activity
                   (date, category, buy_value, sell_value, net_value, segment)
                   VALUES (?, 'FII', ?, ?, ?, 'F&O_FUTURES')""",
                (
                    date.today().isoformat(),
                    float(positions.get("fii_index_futures_net_long", 0)),
                    float(positions.get("fii_index_futures_net_short", 0)),
                    float(positions.get("fii_net_position", 0)),
                )
            )
            freshness_tracker.update("fii_futures")
            logger.info("FII futures positions saved: net=%s trend=%s",
                       positions.get("fii_net_position"), positions.get("fii_fo_trend"))
        except Exception as e:
            logger.error("Failed saving FII futures positions: %s", e)

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
