"""
Options chain data fetcher and parser.

Fetches options chain from NSE (via NSEScraper) or BSE (via BSEScraper) based on
index exchange, then parses into structured data classes for analysis, including
PCR calculation, Max Pain calculation, and Buildup / Spike detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

from src.data.nse_scraper import NSEScraper

if TYPE_CHECKING:
    from src.data.bse_scraper import BSEScraper
from src.data.index_registry import get_registry
from src.database.db_manager import DatabaseManager
from src.database import queries as Q
from config.constants import IST_TIMEZONE
from src.data.data_validator import validate_options_chain, DataValidationError
from src.data.rate_limiter import freshness_tracker

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)

# Strikes nearest to spot fetched on the broker fallback (x2 for CE+PE tokens)
_BROKER_MAX_STRIKES = 50


def _quote_time(quotes) -> Optional[datetime]:
    """Latest exchange feed time across Angel One quotes ("05-Nov-2026 15:40:00"), in IST."""
    latest = None
    for q in quotes:
        try:
            t = datetime.strptime(q.get("exchFeedTime") or "", "%d-%b-%Y %H:%M:%S").replace(tzinfo=_IST)
        except ValueError:
            continue
        if latest is None or t > latest:
            latest = t
    return latest


def _implied_vol_pct(
    ltp: float, spot: float, strike: float, opt_type: str, expiry: date, as_of: datetime
) -> float:
    """Black-Scholes IV in percent (NSE's convention), or 0.0 when it can't be solved reliably."""
    if ltp <= 0 or spot <= 0:
        return 0.0
    intrinsic = max(0.0, spot - strike) if opt_type == "CE" else max(0.0, strike - spot)
    if ltp <= intrinsic:
        return 0.0  # stale or below-parity print — no meaningful IV
    try:
        from src.analysis.gex.black_scholes import (  # noqa: PLC0415 — avoid data→analysis import at load
            BlackScholesEngine,
            BSMInputs,
            get_dte,
            get_risk_free_rate,
        )
        engine = BlackScholesEngine()
        iv = engine.calculate_iv(ltp, BSMInputs(
            spot=spot,
            strike=strike,
            dte_days=get_dte(expiry, as_of) * 365.0,
            iv=0.20,
            risk_free_rate=get_risk_free_rate(),
            option_type=opt_type,
        ))
    except Exception:
        return 0.0
    # Solver pinned to its bounds means it did not converge on a real value
    if iv <= engine.MIN_IV * 1.01 or iv >= engine.MAX_IV * 0.99:
        return 0.0
    return round(iv * 100, 2)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class OptionStrike:
    strike_price: float
    ce_oi: int
    ce_oi_change: int
    ce_oi_change_pct: float
    ce_volume: int
    ce_ltp: float
    ce_ltp_change: float
    ce_ltp_change_pct: float
    ce_iv: float
    ce_bid_price: float
    ce_bid_qty: int
    ce_ask_price: float
    ce_ask_qty: int
    
    pe_oi: int
    pe_oi_change: int
    pe_oi_change_pct: float
    pe_volume: int
    pe_ltp: float
    pe_ltp_change: float
    pe_ltp_change_pct: float
    pe_iv: float
    pe_bid_price: float
    pe_bid_qty: int
    pe_ask_price: float
    pe_ask_qty: int


@dataclass(frozen=True)
class OptionsChainData:
    index_id: str
    spot_price: float
    timestamp: datetime
    expiry_date: date
    strikes: tuple[OptionStrike, ...]
    available_expiries: tuple[date, ...]
    total_market_buy_qty: int = 0
    total_market_sell_qty: int = 0


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

    def __init__(
        self,
        scraper: Optional[NSEScraper] = None,
        bse_scraper=None,
        broker_client=None,
        db: Optional[DatabaseManager] = None,
    ) -> None:
        self._scraper = scraper or NSEScraper()
        self._bse_scraper = bse_scraper  # Optional[BSEScraper]
        self._broker = broker_client  # Optional[AngelOneClient]
        self._db = db  # Used by the broker path for prior-session OI (change in OI)
        # Memory storage for last two snapshots per symbol/expiry to do quick comparisons.
        # Key format: f"{symbol}_{expiry.isoformat()}" -> list[OptionsChainData] (max 2 items)
        self._previous_chains: dict[str, list[OptionsChainData]] = {}

    def _parse_date(self, date_str: str) -> date:
        """Parse date format supporting NSE ('dd-MMM-yyyy'), numeric ('dd-mm-yyyy'), and BSE ('ddMONYYYY')."""
        if not date_str:
            return date.today()
        # Try 'dd-MMM-yyyy' (e.g. 23-Apr-2026)
        try:
            return datetime.strptime(date_str, "%d-%b-%Y").date()
        except ValueError:
            pass
        # Try 'dd-mm-yyyy' (e.g. 23-04-2026)
        try:
            return datetime.strptime(date_str, "%d-%m-%Y").date()
        except ValueError:
            pass
        # Try 'ddMONYYYY' BSE format (e.g. 23APR2026)
        try:
            return datetime.strptime(date_str, "%d%b%Y").date()
        except ValueError:
            logger.warning(f"Failed to parse date string with known formats: {date_str}")
            return date.today()

    def _fetch_bse_raw_chain(self, option_symbol: str, expiry: Optional[str] = None) -> Optional[dict]:
        """Fetch raw options chain from BSE for BSE-listed indices (SENSEX)."""
        if not self._bse_scraper:
            logger.warning("BSE scraper not configured; %s options unavailable", option_symbol)
            return None

        meta = self._bse_scraper.get_option_chain_meta(option_symbol)
        expiry_candidates = meta.get("expiryDates", []) if meta else []

        target_expiry = expiry
        if not target_expiry and expiry_candidates:
            d = self._parse_date(expiry_candidates[0])
            target_expiry = d.strftime("%d%b%Y").upper()

        if not target_expiry:
            return None

        raw = self._bse_scraper.get_option_chain(option_symbol, target_expiry)
        if raw:
            freshness_tracker.update("options_chain")
        return raw

    def fetch_raw_chain(self, option_symbol: str, expiry: Optional[str] = None) -> Optional[dict]:
        """
        Fetch options chain from NSE or BSE based on exchange.
        Returns raw parsed JSON response.
        """
        registry = get_registry()
        fo_symbols = {
            idx.option_symbol.upper()
            for idx in registry.get_indices_with_options()
            if idx.option_symbol
        }
        if option_symbol.upper() not in fo_symbols:
            return None

        # Route to BSE for BSE-listed indices (SENSEX)
        idx_info = registry.get_index_by_option_symbol(option_symbol)
        if idx_info and idx_info.exchange == "BSE":
            return self._fetch_bse_raw_chain(option_symbol, expiry)

        # 1. Get Expiry metadata if not provided (NSE path)
        meta = self._scraper.get_option_chain_meta(option_symbol)
        expiry_candidates = meta.get("expiryDates", []) if meta else []
        
        target_expiry = expiry
        if not target_expiry and expiry_candidates:
            target_expiry = expiry_candidates[0]

        # 2. Determine if it's an equity or index for NSE v3
        INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
        is_index = option_symbol.upper() in INDEX_SYMBOLS
        chain_type = "Indices" if is_index else "Equities"

        # 3. Fetch using v3 with target expiry
        def fetch_v3(exp):
            cache_key = f"nse:option_chain_v3:{option_symbol.upper()}:{exp}"
            return self._scraper.get_option_chain_v3(
                symbol=option_symbol, type=chain_type, expiry=exp, cache_key=cache_key, cache_ttl=60
            )

        raw = None
        if target_expiry:
            raw = fetch_v3(target_expiry)
        
        # 3. If first attempt returns 0 data, try the next candidate
        if (not raw or not raw.get("records", {}).get("data")) and len(expiry_candidates) > 1:
            next_exp = expiry_candidates[1]
            logger.info("First expiry (%s) returned 0 data for %s, trying %s...", target_expiry, option_symbol, next_exp)
            raw = fetch_v3(next_exp)

        if not raw or not raw.get("records", {}).get("data"):
            raw_keys = list(raw.keys()) if raw else "None"
            logger.warning("NSE chain empty for %s expiry=%s — response keys: %s", option_symbol, target_expiry, raw_keys)

        if raw:
            freshness_tracker.update("options_chain")
        return raw

    def get_options_chain(
        self, option_symbol: str, expiry_date: Optional[date] = None
    ) -> Optional[OptionsChainData]:
        """
        Parse raw options chain into structured OptionsChainData.
        Defaults to nearest expiry if expiry_date is None.
        """
        from src.data.index_registry import get_registry
        registry = get_registry()
        idx_info = registry.get_index_by_option_symbol(option_symbol)

        target_expiry_str = None
        if expiry_date:
            if idx_info and idx_info.exchange == "BSE":
                target_expiry_str = expiry_date.strftime("%d%b%Y").upper()
            else:
                target_expiry_str = expiry_date.strftime("%d-%b-%Y").lstrip("0")

        raw = self.fetch_raw_chain(option_symbol, expiry=target_expiry_str)
        if not raw or not raw.get("records", {}).get("data"):
            # Attempt broker fallback if primary scraper fails or returns no data
            if self._broker:
                logger.info(f"NSE chain empty for {option_symbol}; attempting broker fallback...")
                broker_raw = self._fetch_from_broker(option_symbol, expiry_date)
                if broker_raw:
                    raw = broker_raw
            
        if not raw:
            return None

        # Resolve canonical index_id for DB foreign keys (e.g. NIFTY -> NIFTY50)
        canonical_id = option_symbol
        if idx_info:
            canonical_id = idx_info.id

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
        total_market_buy_qty = 0
        total_market_sell_qty = 0
        
        strikes: list[OptionStrike] = []
        for item in raw_data:
            expiry_val_str = item.get("expiryDate") or item.get("CE", {}).get("expiryDate") or item.get("PE", {}).get("expiryDate")
            if not expiry_val_str:
                continue
            
            expiry_val = self._parse_date(expiry_val_str)
            if expiry_val != target_expiry:
                continue

            strike_price = float(item.get("strikePrice", 0.0))
            ce = item.get("CE", {})
            pe = item.get("PE", {})

            # Aggregate market quantities (v3 structure has these per strike usually, but we sum them)
            total_market_buy_qty += int(ce.get("totalBuyQuantity", 0)) + int(pe.get("totalBuyQuantity", 0))
            total_market_sell_qty += int(ce.get("totalSellQuantity", 0)) + int(pe.get("totalSellQuantity", 0))

            strikes.append(OptionStrike(
                strike_price=strike_price,
                ce_oi=int(ce.get("openInterest", 0)),
                ce_oi_change=int(ce.get("changeinOpenInterest", 0)),
                ce_oi_change_pct=float(ce.get("pchangeinOpenInterest", 0.0)),
                ce_volume=int(ce.get("totalTradedVolume", 0)),
                ce_ltp=float(ce.get("lastPrice", 0.0)),
                ce_ltp_change=float(ce.get("change", 0.0)),
                ce_ltp_change_pct=float(ce.get("pChange", 0.0)),
                ce_iv=float(ce.get("impliedVolatility", 0.0)),
                ce_bid_price=float(ce.get("bidprice", 0.0)),
                ce_bid_qty=int(ce.get("bidQty", 0)),
                ce_ask_price=float(ce.get("askPrice", 0.0)),
                ce_ask_qty=int(ce.get("askQty", 0)),
                
                pe_oi=int(pe.get("openInterest", 0)),
                pe_oi_change=int(pe.get("changeinOpenInterest", 0)),
                pe_oi_change_pct=float(pe.get("pchangeinOpenInterest", 0.0)),
                pe_volume=int(pe.get("totalTradedVolume", 0)),
                pe_ltp=float(pe.get("lastPrice", 0.0)),
                pe_ltp_change=float(pe.get("change", 0.0)),
                pe_ltp_change_pct=float(pe.get("pChange", 0.0)),
                pe_iv=float(pe.get("impliedVolatility", 0.0)),
                pe_bid_price=float(pe.get("bidprice", 0.0)),
                pe_bid_qty=int(pe.get("bidQty", 0)),
                pe_ask_price=float(pe.get("askPrice", 0.0)),
                pe_ask_qty=int(pe.get("askQty", 0)),
            ))

        strikes.sort(key=lambda s: s.strike_price)

        # Use current time — NSE's own timestamp field is brittle
        fetch_ts = datetime.now(tz=_IST)

        chain_data = OptionsChainData(
            index_id=canonical_id,
            spot_price=spot_price,
            timestamp=fetch_ts,
            expiry_date=target_expiry,
            strikes=tuple(strikes),
            available_expiries=tuple(available_expiries),
            total_market_buy_qty=total_market_buy_qty,
            total_market_sell_qty=total_market_sell_qty,
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
        for s in chain_data.strikes:
            # CE
            snapshot_records.append((
                chain_data.index_id, ts, exp_str, s.strike_price, "CE",
                s.ce_oi, s.ce_oi_change, s.ce_oi_change_pct, s.ce_volume,
                s.ce_ltp, s.ce_ltp_change, s.ce_ltp_change_pct, s.ce_iv,
                s.ce_bid_price, s.ce_bid_qty, s.ce_ask_price, s.ce_ask_qty
            ))
            # PE
            snapshot_records.append((
                chain_data.index_id, ts, exp_str, s.strike_price, "PE",
                s.pe_oi, s.pe_oi_change, s.pe_oi_change_pct, s.pe_volume,
                s.pe_ltp, s.pe_ltp_change, s.pe_ltp_change_pct, s.pe_iv,
                s.pe_bid_price, s.pe_bid_qty, s.pe_ask_price, s.pe_ask_qty
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

    # ── Broker Fallback ───────────────────────────────────────────────────────

    def _prior_session_oi(self, index_id: str, expiry: date, session_date: date) -> dict[tuple[float, str], int]:
        """OI per (strike, option_type) from the last snapshot taken before session_date's open."""
        if self._db is None:
            return {}
        try:
            cutoff = f"{session_date.isoformat()}T09:15"
            rows = self._db.fetch_all(
                "SELECT strike_price, option_type, open_interest FROM options_chain_snapshot "
                "WHERE index_id = ? AND expiry_date = ? AND timestamp = ("
                "  SELECT MAX(timestamp) FROM options_chain_snapshot"
                "  WHERE index_id = ? AND expiry_date = ? AND timestamp < ?)",
                (index_id, expiry.isoformat(), index_id, expiry.isoformat(), cutoff),
            )
            return {(float(r["strike_price"]), r["option_type"]): int(r["open_interest"]) for r in rows}
        except Exception as exc:
            logger.warning("Prior-session OI lookup failed for %s: %s", index_id, exc)
            return {}

    def _fetch_from_broker(self, option_symbol: str, target_date: Optional[date] = None) -> Optional[dict]:
        """
        Build an options chain from Angel One quotes, shaped like the NSE raw response.

        Contracts come from the scrip master (NFO, or BFO for BSE indices); quotes
        come from getMarketData FULL, which carries LTP, OI and volume. Angel One
        reports OI and volume in units, so both are divided by lot size to match
        NSE's contract counts.

        Angel One supplies neither IV nor change in OI, so they are derived:
        IV is back-solved from LTP with Black-Scholes; change in OI is measured
        against the last stored snapshot taken before the quotes' session opened
        (needs ``db``; 0 when no such snapshot exists).
        """
        if not self._broker:
            return None

        try:
            import time  # noqa: PLC0415

            import pandas as pd  # noqa: PLC0415

            from src.data.broker.angel_one import INDEX_REST_TOKENS, MARKET_DATA_MAX_TOKENS  # noqa: PLC0415

            registry = get_registry()
            idx = registry.get_index_by_option_symbol(option_symbol)
            if not idx:
                return None

            # 1. Option contracts for the target (or nearest live) expiry
            segment = "BFO" if idx.exchange == "BSE" else "NFO"
            master = self._broker.get_master_contract(segment)
            if master is None or master.empty:
                return None
            opts = master[
                (master["name"] == option_symbol.upper())
                & (master["instrumenttype"].isin(["OPTIDX", "OPTSTK"]))
            ].copy()
            if opts.empty:
                return None
            opts["expiry_dt"] = pd.to_datetime(opts["expiry"], format="%d%b%Y").dt.date
            if target_date:
                opts = opts[opts["expiry_dt"] == target_date]
            else:
                live = opts[opts["expiry_dt"] >= datetime.now(tz=_IST).date()]
                opts = live[live["expiry_dt"] == live["expiry_dt"].min()] if not live.empty else live
            if opts.empty:
                return None
            opts = opts.copy()
            expiry_dt = opts["expiry_dt"].iloc[0]
            opts["strike_val"] = opts["strike"].astype(float) / 100.0
            opts["opt_type"] = opts["symbol"].str[-2:]

            # 2. Spot, to centre the strike window
            spot = 0.0
            rest = INDEX_REST_TOKENS.get(idx.id)
            if rest:
                spot_data = self._broker.get_market_data("LTP", {rest[0]: [rest[1]]})
                fetched = (spot_data or {}).get("fetched") or []
                if fetched:
                    spot = float(fetched[0].get("ltp") or 0.0)

            strikes = sorted(opts["strike_val"].unique())
            if spot > 0:
                strikes = sorted(strikes, key=lambda s: abs(s - spot))[:_BROKER_MAX_STRIKES]
            else:
                mid = len(strikes) // 2
                half = _BROKER_MAX_STRIKES // 2
                strikes = strikes[max(0, mid - half): mid + half]
            opts = opts[opts["strike_val"].isin(strikes)]

            # 3. Quotes in batches of MARKET_DATA_MAX_TOKENS
            tokens = opts["token"].astype(str).tolist()
            quotes: dict[str, dict] = {}
            for i in range(0, len(tokens), MARKET_DATA_MAX_TOKENS):
                if i:
                    time.sleep(0.35)  # stay well under the market-data rate limit
                batch = self._broker.get_market_data("FULL", {segment: tokens[i:i + MARKET_DATA_MAX_TOKENS]})
                for q in (batch or {}).get("fetched") or []:
                    quotes[str(q.get("symbolToken"))] = q
            if not quotes:
                return None

            # Quotes' own session time: after hours it is still the last close,
            # which keeps DTE (for IV) and the OI baseline consistent with the prices.
            as_of = _quote_time(quotes.values()) or datetime.now(tz=_IST)
            base_oi = self._prior_session_oi(idx.id, expiry_dt, as_of.date())

            # 4. Map to NSE-style records
            expiry_str = expiry_dt.strftime("%d-%b-%Y")
            strikes_map: dict[float, dict] = {}
            for row in opts.itertuples(index=False):
                q = quotes.get(str(row.token))
                if not q or row.opt_type not in ("CE", "PE"):
                    continue
                lot = float(row.lotsize or 1) or 1.0
                depth = q.get("depth") or {}
                best_bid = (depth.get("buy") or [{}])[0]
                best_ask = (depth.get("sell") or [{}])[0]
                strike = float(row.strike_val)
                oi = int(float(q.get("opnInterest") or 0) / lot)
                ltp = float(q.get("ltp") or 0.0)
                prior = base_oi.get((strike, row.opt_type))
                oi_change = oi - prior if prior is not None else 0
                strikes_map.setdefault(strike, {"strikePrice": strike, "expiryDate": expiry_str})
                strikes_map[strike][row.opt_type] = {
                    "strikePrice": strike,
                    "expiryDate": expiry_str,
                    "underlying": option_symbol,
                    "identifier": row.symbol,
                    "openInterest": oi,
                    "changeinOpenInterest": oi_change,
                    "pchangeinOpenInterest": round(oi_change / prior * 100, 2) if prior else 0.0,
                    "totalTradedVolume": int(float(q.get("tradeVolume") or 0) / lot),
                    "impliedVolatility": _implied_vol_pct(ltp, spot, strike, row.opt_type, expiry_dt, as_of),
                    "lastPrice": ltp,
                    "change": float(q.get("netChange") or 0.0),
                    "pChange": float(q.get("percentChange") or 0.0),
                    "totalBuyQuantity": int(q.get("totBuyQuan") or 0),
                    "totalSellQuantity": int(q.get("totSellQuan") or 0),
                    "bidQty": int(best_bid.get("quantity") or 0),
                    "bidprice": float(best_bid.get("price") or 0.0),
                    "askQty": int(best_ask.get("quantity") or 0),
                    "askPrice": float(best_ask.get("price") or 0.0),
                }

            data = [strikes_map[k] for k in sorted(strikes_map)]
            if not data:
                return None
            logger.info("Broker chain for %s %s: %d strikes, %d quotes", option_symbol, expiry_str, len(data), len(quotes))
            return {
                "records": {
                    "data": data,
                    "underlyingValue": spot,
                    "expiryDates": [expiry_str],
                    "timestamp": datetime.now(tz=_IST).strftime("%d-%b-%Y %H:%M:%S"),
                },
                "filtered": {"data": data},
            }

        except Exception as exc:
            logger.error(f"Broker fallback failed for {option_symbol}: {exc}")
            return None

