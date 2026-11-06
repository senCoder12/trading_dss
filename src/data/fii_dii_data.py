"""
FII / DII data module.

Fetches FII/DII institutional activity data from NSE, tracks trends
and saves to the database.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from config.constants import IST_TIMEZONE
from src.data.nse_scraper import NSEScraper
from src.database.db_manager import DatabaseManager
from src.database import queries as Q

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# NSE end-of-day participant-wise open interest (contracts). Published after
# the close for each trading day; historical files remain available, so the
# same URL serves both the daily job and backfills. Non-trading days are 404.
PARTICIPANT_OI_URL = (
    "https://archives.nseindia.com/content/nsccl/fao_participant_oi_{ddmmyyyy}.csv"
)
_MISS_RETRY = timedelta(hours=6)
_PARTICIPANTS = {"CLIENT": "CLIENT", "DII": "DII", "FII": "FII", "PRO": "PRO"}
_FUT_COLUMNS = {  # instrument → (long column, short column)
    "INDEX_FUT": ("Future Index Long", "Future Index Short"),
    "STOCK_FUT": ("Future Stock Long", "Future Stock Short"),
}
# Title variants seen in the archive:
#   'as on Nov 05, 2026'  (since May 2025)
#   'as on Jan 02,2024'   (older)
#   'as on Feb 22,"2024"""""'  (some 2024 files quote the year)
_TITLE_DATE_RE = re.compile(r'as on\s+([A-Za-z]{3})\s+(\d{1,2}),\s*"*\s*(\d{4})')


@dataclass(frozen=True)
class FuturesPosition:
    """One participant's futures open interest (contracts) at a day's close."""
    trade_date: date
    participant: str      # FII | DII | PRO | CLIENT
    instrument: str       # INDEX_FUT | STOCK_FUT
    long_contracts: int
    short_contracts: int

    @property
    def net_contracts(self) -> int:
        return self.long_contracts - self.short_contracts


def parse_participant_oi_csv(
    text: str, expected_date: Optional[date] = None
) -> Optional[list[FuturesPosition]]:
    """Parse NSE fao_participant_oi CSV into futures positions.

    Returns None for anything that is not a well-formed file (e.g. the HTML
    404 page served for holidays), when the title date does not match
    ``expected_date``, or when participant rows do not add up to TOTAL.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None

    m = _TITLE_DATE_RE.search(lines[0])
    if not m:
        return None
    try:
        file_date = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y").date()
    except ValueError:
        return None
    if expected_date is not None and file_date != expected_date:
        logger.warning("Participant OI file date %s != requested %s", file_date, expected_date)
        return None

    reader = csv.reader(io.StringIO("\n".join(lines[1:])))
    header = [h.strip() for h in next(reader)]
    try:
        col = {name: header.index(name) for pair in _FUT_COLUMNS.values() for name in pair}
        type_idx = header.index("Client Type")
    except ValueError:
        return None

    positions: list[FuturesPosition] = []
    totals: dict[str, int] = {}
    for row in reader:
        if len(row) <= max(col.values()):
            continue
        who = row[type_idx].strip().upper()
        try:
            values = {name: int(row[i].strip().replace(",", "")) for name, i in col.items()}
        except ValueError:
            return None
        if who == "TOTAL":
            totals = values
            continue
        participant = _PARTICIPANTS.get(who)
        if participant is None:
            continue
        for instrument, (long_col, short_col) in _FUT_COLUMNS.items():
            positions.append(FuturesPosition(
                file_date, participant, instrument, values[long_col], values[short_col],
            ))

    if {p.participant for p in positions} != set(_PARTICIPANTS.values()):
        return None
    # Integrity: the four participants must add up to the TOTAL row
    for name in col:
        instrument = next(i for i, pair in _FUT_COLUMNS.items() if name in pair)
        side = "long" if name == _FUT_COLUMNS[instrument][0] else "short"
        summed = sum(
            getattr(p, f"{side}_contracts") for p in positions if p.instrument == instrument
        )
        if totals and summed != totals[name]:
            logger.warning("Participant OI %s: %s sums to %d, TOTAL says %d",
                           file_date, name, summed, totals[name])
            return None
    return positions


def _http_get_text(url: str) -> Optional[str]:
    """GET helper; None on 404 or any network error."""
    import requests  # noqa: PLC0415

    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    except requests.RequestException as exc:
        logger.warning("Participant OI fetch failed (%s): %s", url, exc)
        return None
    if resp.status_code != 200:
        return None
    return resp.text


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

    def __init__(
        self,
        scraper: Optional[NSEScraper] = None,
        http_get: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self._scraper = scraper or NSEScraper()
        self._http_get = http_get or _http_get_text
        # Days NSE had no file for (holidays, or a transient 404) → time of the miss.
        # Re-requested after _MISS_RETRY, so a transient miss never becomes permanent.
        self._missing_oi_days: dict[date, datetime] = {}

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

    # ── Futures positions (participant-wise OI, contracts) ─────────────────

    def fetch_futures_positions(self, trade_date: date) -> Optional[list[FuturesPosition]]:
        """Fetch one trading day's participant-wise futures OI from the NSE archive.

        Returns None when NSE has no file for that day (holiday, weekend, or
        not yet published) or the file fails validation.
        """
        url = PARTICIPANT_OI_URL.format(ddmmyyyy=trade_date.strftime("%d%m%Y"))
        text = self._http_get(url)
        if not text:
            return None
        return parse_participant_oi_csv(text, expected_date=trade_date)

    def save_futures_positions_to_db(
        self,
        positions: list[FuturesPosition],
        db: DatabaseManager,
        source: str = "nse_archive",
    ) -> int:
        """Upsert positions into fii_futures_positions (never fii_dii_activity).

        Returns rows written; 0 if the table does not exist yet (migration v20
        not applied), so callers degrade gracefully instead of erroring.
        """
        if not positions:
            return 0
        if not db.table_exists("fii_futures_positions"):
            logger.warning("fii_futures_positions missing (migration v20 not applied) — skipping write")
            return 0
        fetched_at = datetime.now(tz=_IST).isoformat()
        rows = [
            (p.trade_date.isoformat(), p.participant, p.instrument,
             p.long_contracts, p.short_contracts, p.net_contracts, source, fetched_at)
            for p in positions
        ]
        db.execute_many(Q.UPSERT_FII_FUTURES_POSITION, rows)
        return len(rows)

    def sync_futures_positions(
        self,
        db: DatabaseManager,
        lookback_days: int = 7,
        now: Optional[datetime] = None,
        pause_seconds: float = 0.3,
    ) -> list[date]:
        """Fill any missing weekdays from (today − lookback) up to the latest published session.

        Today's file is requested only after FUTURES_PUBLISH_CUTOFF, since NSE
        publishes it after the close. Point-in-time use is enforced by readers:
        GET_FII_INDEX_FUT_NET_BEFORE_DATE only returns sessions before the
        decision date. Returns the dates written.
        """
        from src.monitoring.data_freshness import FUTURES_PUBLISH_CUTOFF  # noqa: PLC0415

        if not db.table_exists("fii_futures_positions"):
            logger.warning("fii_futures_positions missing (migration v20 not applied) — skipping sync")
            return []
        now = (now or datetime.now(tz=_IST)).astimezone(_IST)
        today = now.date()
        last = today if now.time() >= FUTURES_PUBLISH_CUTOFF else today - timedelta(days=1)
        start = today - timedelta(days=lookback_days)
        have = {
            r["trade_date"]
            for r in db.fetch_all(
                Q.LIST_FII_FUTURES_TRADE_DATES, (start.isoformat(), today.isoformat())
            )
        }
        written: list[date] = []
        day = start
        while day <= last:
            missed_at = self._missing_oi_days.get(day)
            recently_missed = missed_at is not None and now - missed_at < _MISS_RETRY
            if day.weekday() < 5 and day.isoformat() not in have and not recently_missed:
                positions = self.fetch_futures_positions(day)
                if positions:
                    self.save_futures_positions_to_db(positions, db)
                    written.append(day)
                elif (today - day).days >= 3:
                    # Settled days only; recent days may still be published any minute.
                    self._missing_oi_days[day] = now
                if pause_seconds:
                    time.sleep(pause_seconds)
            day += timedelta(days=1)
        if written:
            row = db.fetch_one(
                Q.GET_FII_INDEX_FUT_NET_BEFORE_DATE, ((today + timedelta(days=1)).isoformat(),)
            )
            logger.info(
                "FII futures positions synced for %s; latest FII index-futures net=%s (%s)",
                [d.isoformat() for d in written],
                row["net_contracts"] if row else None,
                row["trade_date"] if row else None,
            )
        return written

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
