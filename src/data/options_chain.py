"""
Options chain data fetcher and parser.

Fetches raw option chain from NSE and parses it into structured
strong-typed data classes for analysis, including PCR calculation,
Max Pain calculation, and Buildup / Spike detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

from src.data.nse_scraper import NSEScraper
from src.data.index_registry import get_registry
from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from config.constants import IST_TIMEZONE
from src.data.data_validator import validate_options_chain, DataValidationError
from src.data.rate_limiter import freshness_tracker

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class OptionStrike:
    strike_price: float
    ce_oi: int
    ce_oi_change: int
    ce_volume: int
    ce_ltp: float
    ce_iv: float
    pe_oi: int
    pe_oi_change: int
    pe_volume: int
    pe_ltp: float
    pe_iv: float


@dataclass(frozen=True)
class OptionsChainData:
    index_id: str
    spot_price: float
    timestamp: datetime
    expiry_date: date
    strikes: tuple[OptionStrike, ...]
    available_expiries: tuple[date, ...]


@dataclass(frozen=True)
class OISummary:
    total_ce_oi: int
    total_pe_oi: int
    total_ce_oi_change: int
    total_pe_oi_change: int
    total_ce_volume: int
    total_pe_volume: int
    pcr: float
    pcr_change: float
    max_pain_strike: float
    highest_ce_oi_strike: float
    highest_pe_oi_strike: float
    top_5_ce_oi_strikes: tuple[tuple[float, int], ...]
    top_5_pe_oi_strikes: tuple[tuple[float, int], ...]


@dataclass(frozen=True)
class OIBuildup:
    strike_price: float
    option_type: str  # "CE" or "PE"
    buildup_type: str  # LONG_BUILDUP, SHORT_BUILDUP, LONG_UNWINDING, SHORT_COVERING
    oi_change: int
    oi_change_pct: float
    price_change: float
    significance: str  # HIGH, MEDIUM, LOW


@dataclass(frozen=True)
class OISpike:
    strike_price: float
    option_type: str  # "CE" or "PE"
    previous_oi: int
    current_oi: int
    change_pct: float
    is_new_position: bool
    timestamp: datetime


# =============================================================================
# Fetcher Class
# =============================================================================

class OptionsChainFetcher:
    """
    Fetch and parse NSE options chains for any F&O-enabled index.

    Includes tracking for short-term snapshots to detect OI buildup without
    needing database round-trips.
    """

    def __init__(self, scraper: Optional[NSEScraper] = None) -> None:
        self._scraper = scraper or NSEScraper()
        # Memory storage for last two snapshots per symbol/expiry to do quick comparisons.
        # Key format: f"{symbol}_{expiry.isoformat()}" -> list[OptionsChainData] (max 2 items)
        self._previous_chains: dict[str, list[OptionsChainData]] = {}

    def _parse_date(self, date_str: str) -> date:
        """Parse NSE date format 'dd-MMM-yyyy'."""
        try:
            return datetime.strptime(date_str, "%d-%b-%Y").date()
        except ValueError:
            logger.warning(f"Failed to parse date string: {date_str}")
            return date.today()

    def fetch_raw_chain(self, option_symbol: str) -> Optional[dict]:
        """
        Hit NSE option chain API and return the raw parsed JSON response.

        Validates that *option_symbol* belongs to an F&O-enabled index before
        making a network call.  Caches the raw response for 60 seconds.
        """
        registry = get_registry()
        # Check by option_symbol field across all F&O indices
        fo_symbols = {
            idx.option_symbol.upper()
            for idx in registry.get_indices_with_options()
            if idx.option_symbol
        }
        if option_symbol.upper() not in fo_symbols:
            logger.warning(
                "fetch_raw_chain: %r is not an F&O-enabled index (known: %s)",
                option_symbol,
                sorted(fo_symbols),
            )
            return None

        cache_key = f"nse:option_chain:{option_symbol.upper()}"
        raw = self._scraper._call(
            "/api/option-chain-indices",
            params={"symbol": option_symbol},
            cache_key=cache_key,
            cache_ttl=60,
        )
        if raw:
            freshness_tracker.update("options_chain")
        else:
            # NSE's options-chain endpoint returns HTTP 200 with an empty
            # {} outside market hours. Surface this so "no chain" doesn't
            # look like a bug when inspecting logs.
            logger.debug(
                "Options chain API returned empty for %s "
                "(NSE likely closed or symbol not yet tradable)",
                option_symbol,
            )
        return raw

    def get_options_chain(
        self, option_symbol: str, expiry_date: Optional[date] = None
    ) -> Optional[OptionsChainData]:
        """
        Parse raw options chain into structured OptionsChainData.
        Defaults to nearest expiry if expiry_date is None.
        """
        raw = self.fetch_raw_chain(option_symbol)
        if not raw:
            return None

        result = validate_options_chain(raw, symbol=option_symbol)
        if not result.is_valid:
            logger.error(f"Options chain validation failed for {option_symbol}: {result.errors}")
            return None

        records = raw.get("records", {})
        spot_price = float(records.get("underlyingValue", 0.0))
        
        raw_expiries = records.get("expiryDates", [])
        available_expiries = sorted(self._parse_date(d) for d in raw_expiries)

        if not available_expiries:
            return None

        # Determine target expiry
        target_expiry = expiry_date if expiry_date else available_expiries[0]
        # NSE expects date in dd-MMM-yyyy format
        target_expiry_str = target_expiry.strftime("%d-%b-%Y").lstrip("0")  # e.g. 4-Apr-2024 instead of 04-Apr
        # Workaround for NSE's exact string matching:
        target_expiry_str_alt = target_expiry.strftime("%d-%b-%Y")
        
        raw_data = records.get("data", [])
        
        strikes: list[OptionStrike] = []
        for item in raw_data:
            expiry_val = item.get("expiryDate")
            if expiry_val not in (target_expiry_str, target_expiry_str_alt):
                continue

            strike_price = float(item.get("strikePrice", 0.0))
            ce = item.get("CE", {})
            pe = item.get("PE", {})

            strikes.append(OptionStrike(
                strike_price=strike_price,
                ce_oi=int(ce.get("openInterest", 0)),
                ce_oi_change=int(ce.get("changeinOpenInterest", 0)),
                ce_volume=int(ce.get("totalTradedVolume", 0)),
                ce_ltp=float(ce.get("lastPrice", 0.0)),
                ce_iv=float(ce.get("impliedVolatility", 0.0)),
                pe_oi=int(pe.get("openInterest", 0)),
                pe_oi_change=int(pe.get("changeinOpenInterest", 0)),
                pe_volume=int(pe.get("totalTradedVolume", 0)),
                pe_ltp=float(pe.get("lastPrice", 0.0)),
                pe_iv=float(pe.get("impliedVolatility", 0.0)),
            ))

        strikes.sort(key=lambda s: s.strike_price)

        # Use current time — NSE's own timestamp field is brittle
        fetch_ts = datetime.now(tz=_IST)

        chain_data = OptionsChainData(
            index_id=option_symbol,
            spot_price=spot_price,
            timestamp=fetch_ts,
            expiry_date=target_expiry,
            strikes=tuple(strikes),
            available_expiries=tuple(available_expiries),
        )

        # Retain last 2 snapshots in memory for diff calculations
        mem_key = f"{option_symbol}_{target_expiry.isoformat()}"
        if mem_key not in self._previous_chains:
            self._previous_chains[mem_key] = []
            
        history = self._previous_chains[mem_key]
        if not history or history[-1].timestamp < chain_data.timestamp:
            history.append(chain_data)
        if len(history) > 2:
            history.pop(0)

        return chain_data

    def get_available_expiries(self, option_symbol: str) -> list[date]:
        """Return all available expiry dates sorted ascending."""
        raw = self.fetch_raw_chain(option_symbol)
        if not raw:
            return []
        raw_expiries = raw.get("records", {}).get("expiryDates", [])
        return sorted(self._parse_date(d) for d in raw_expiries)

    def _compute_oi_summary(self, chain: OptionsChainData) -> Optional[OISummary]:
        """Compute OI summary from an already-fetched OptionsChainData (no API call)."""
        if not chain.strikes:
            return None

        total_ce_oi = sum(s.ce_oi for s in chain.strikes)
        total_pe_oi = sum(s.pe_oi for s in chain.strikes)
        total_ce_oi_change = sum(s.ce_oi_change for s in chain.strikes)
        total_pe_oi_change = sum(s.pe_oi_change for s in chain.strikes)
        total_ce_vol = sum(s.ce_volume for s in chain.strikes)
        total_pe_vol = sum(s.pe_volume for s in chain.strikes)

        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0

        prev_ce_oi = total_ce_oi - total_ce_oi_change
        prev_pe_oi = total_pe_oi - total_pe_oi_change
        prev_pcr = prev_pe_oi / prev_ce_oi if prev_ce_oi > 0 else 0.0
        pcr_change = pcr - prev_pcr

        max_pain = self.calculate_max_pain(chain.strikes)

        ce_strikes_sorted = sorted(chain.strikes, key=lambda x: x.ce_oi, reverse=True)
        pe_strikes_sorted = sorted(chain.strikes, key=lambda x: x.pe_oi, reverse=True)

        highest_ce = ce_strikes_sorted[0].strike_price if ce_strikes_sorted else 0.0
        highest_pe = pe_strikes_sorted[0].strike_price if pe_strikes_sorted else 0.0

        top_5_ce = tuple((s.strike_price, s.ce_oi) for s in ce_strikes_sorted[:5])
        top_5_pe = tuple((s.strike_price, s.pe_oi) for s in pe_strikes_sorted[:5])

        return OISummary(
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            total_ce_oi_change=total_ce_oi_change,
            total_pe_oi_change=total_pe_oi_change,
            total_ce_volume=total_ce_vol,
            total_pe_volume=total_pe_vol,
            pcr=round(pcr, 4),
            pcr_change=round(pcr_change, 4),
            max_pain_strike=max_pain,
            highest_ce_oi_strike=highest_ce,
            highest_pe_oi_strike=highest_pe,
            top_5_ce_oi_strikes=top_5_ce,
            top_5_pe_oi_strikes=top_5_pe,
        )

    def get_oi_summary(self, option_symbol: str, expiry_date: Optional[date] = None) -> Optional[OISummary]:
        """Calculate high-level OI summary metrics for the given chain."""
        chain = self.get_options_chain(option_symbol, expiry_date)
        if not chain:
            return None
        return self._compute_oi_summary(chain)

    def calculate_max_pain(self, strikes: list[OptionStrike]) -> float:
        """
        Max Pain algorithm: strike where option writers lose the least amount of money.
        """
        if not strikes:
            return 0.0

        min_loss = float('inf')
        max_pain_strike = 0.0

        target_strikes = [s.strike_price for s in strikes]
        
        # Test every strike as a potential expiry spot price
        for assumed_expiry_price in target_strikes:
            total_loss = 0.0
            
            for option in strikes:
                # If Market closes at assumed_expiry_price:
                # Call writers lose if assumed_expiry_price > option.strike_price
                if assumed_expiry_price > option.strike_price:
                    total_loss += option.ce_oi * (assumed_expiry_price - option.strike_price)
                
                # Put writers lose if assumed_expiry_price < option.strike_price
                if assumed_expiry_price < option.strike_price:
                    total_loss += option.pe_oi * (option.strike_price - assumed_expiry_price)
            
            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = assumed_expiry_price

        return max_pain_strike

    def detect_oi_buildup(
        self, current_chain: OptionsChainData, previous_chain: OptionsChainData
    ) -> list[OIBuildup]:
        """
        Compare two chains to determine buildup trends per strike.
        Price ↑, OI ↑ = Long Buildup
        Price ↓, OI ↑ = Short Buildup
        Price ↓, OI ↓ = Long Unwinding
        Price ↑, OI ↓ = Short Covering
        """
        buildups = []
        prev_map = {s.strike_price: s for s in previous_chain.strikes}

        def categorize_buildup(price_chg: float, oi_chg: int) -> str:
            if price_chg > 0 and oi_chg > 0: return "LONG_BUILDUP"
            if price_chg < 0 and oi_chg > 0: return "SHORT_BUILDUP"
            if price_chg < 0 and oi_chg < 0: return "LONG_UNWINDING"
            if price_chg > 0 and oi_chg < 0: return "SHORT_COVERING"
            return "NEUTRAL"

        def evaluate(option_type: str, cur_price: float, prev_price: float, cur_oi: int, prev_oi: int, strike: float):
            price_chg = cur_price - prev_price
            oi_chg = cur_oi - prev_oi
            
            if prev_oi == 0:
                oi_chg_pct = 100.0 if oi_chg > 0 else 0.0
            else:
                oi_chg_pct = (oi_chg / prev_oi) * 100.0
                
            b_type = categorize_buildup(price_chg, oi_chg)
            if b_type == "NEUTRAL":
                return
                
            abs_pct = abs(oi_chg_pct)
            if abs_pct > 20: sig = "HIGH"
            elif abs_pct > 10: sig = "MEDIUM"
            else: sig = "LOW"

            buildups.append(OIBuildup(
                strike_price=strike,
                option_type=option_type,
                buildup_type=b_type,
                oi_change=oi_chg,
                oi_change_pct=round(oi_chg_pct, 2),
                price_change=round(price_chg, 2),
                significance=sig
            ))

        for cur in current_chain.strikes:
            prev = prev_map.get(cur.strike_price)
            if not prev:
                continue
                
            evaluate("CE", cur.ce_ltp, prev.ce_ltp, cur.ce_oi, prev.ce_oi, cur.strike_price)
            evaluate("PE", cur.pe_ltp, prev.pe_ltp, cur.pe_oi, prev.pe_oi, cur.strike_price)

        return buildups

    def detect_oi_spikes(
        self, current_chain: OptionsChainData, previous_chain: OptionsChainData, threshold_pct: float = 10.0
    ) -> list[OISpike]:
        """Detect any strikes where OI spiked by more than the threshold."""
        spikes = []
        prev_map = {s.strike_price: s for s in previous_chain.strikes}

        def check_spike(otype: str, cur_oi: int, prev_oi: int, strike: float):
            if prev_oi == 0:
                return
            change_pct = ((cur_oi - prev_oi) / prev_oi) * 100.0
            if abs(change_pct) >= threshold_pct:
                spikes.append(OISpike(
                    strike_price=strike,
                    option_type=otype,
                    previous_oi=prev_oi,
                    current_oi=cur_oi,
                    change_pct=round(change_pct, 2),
                    is_new_position=(change_pct > 0),
                    timestamp=current_chain.timestamp
                ))

        for cur in current_chain.strikes:
            prev = prev_map.get(cur.strike_price)
            if prev:
                check_spike("CE", cur.ce_oi, prev.ce_oi, cur.strike_price)
                check_spike("PE", cur.pe_oi, prev.pe_oi, cur.strike_price)

        return spikes
        
    def get_memory_snapshots(self, option_symbol: str, expiry_date: date) -> list[OptionsChainData]:
        """Helper to retrieve cached memory snapshots for comparison."""
        mem_key = f"{option_symbol}_{expiry_date.isoformat()}"
        return self._previous_chains.get(mem_key, [])

    def save_to_db(self, chain_data: OptionsChainData, db: DatabaseManager) -> None:
        """
        Persist the OptionsChainData (detailed strikes) and its generated 
        OISummary directly into the database using execute_many and batch sizes.
        """
        ts = chain_data.timestamp.isoformat()
        exp_str = chain_data.expiry_date.isoformat()

        # 1. Insert options_chain_snapshot
        snapshot_records = []
        for strike in chain_data.strikes:
            # CE
            snapshot_records.append((
                chain_data.index_id, ts, exp_str, strike.strike_price, "CE",
                strike.ce_oi, strike.ce_oi_change, strike.ce_volume,
                strike.ce_ltp, strike.ce_iv, 0.0, 0.0  # bid/ask not captured yet
            ))
            # PE
            snapshot_records.append((
                chain_data.index_id, ts, exp_str, strike.strike_price, "PE",
                strike.pe_oi, strike.pe_oi_change, strike.pe_volume,
                strike.pe_ltp, strike.pe_iv, 0.0, 0.0
            ))

        if snapshot_records:
            try:
                db.execute_many(Q.INSERT_OPTIONS_CHAIN, snapshot_records)
            except Exception as e:
                logger.error(f"Failed inserting options_chain_snapshot: {e}")

        # 2. Compute OI summary directly from the already-fetched chain (no re-fetch)
        summary = self._compute_oi_summary(chain_data)
        if summary:
            record = (
                chain_data.index_id, ts, exp_str,
                summary.total_ce_oi, summary.total_pe_oi, 
                summary.total_ce_oi_change, summary.total_pe_oi_change,
                summary.pcr, summary.max_pain_strike,
                summary.highest_ce_oi_strike, summary.highest_pe_oi_strike
            )
            try:
                db.execute(Q.INSERT_OI_AGGREGATED, record)
            except Exception as e:
                logger.error(f"Failed inserting oi_aggregated: {e}")
