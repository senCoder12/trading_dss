"""GEX profile dataclasses — pure data, no computation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class StrikeGEX:
    """Gamma exposure at a single strike (may aggregate across expiries)."""

    strike: float
    expiry: date          # primary expiry (nearest week)

    # CE side
    ce_oi: int
    ce_iv: float          # decimal (e.g. 0.15)
    ce_ltp: float
    ce_gamma: float       # raw BSM gamma per unit
    ce_gex: float         # dealer gamma rupees for calls — positive

    # PE side
    pe_oi: int
    pe_iv: float          # decimal
    pe_ltp: float
    pe_gamma: float
    pe_gex: float         # dealer gamma rupees for puts — negative

    # Net
    net_gex: float        # ce_gex + pe_gex
    is_call_dominant: bool

    # Context
    distance_from_spot_pct: float
    is_atm: bool          # within atm_threshold_pct of spot


@dataclass
class GEXProfile:
    """Full GEX profile for one symbol at one timestamp."""

    symbol: str
    spot_price: float
    timestamp: datetime

    strikes: list[StrikeGEX]   # sorted ascending by strike

    # Summary
    net_gex: float              # total net GEX (positive = dealers net long gamma)
    call_wall: float | None     # strike with highest CE GEX above spot
    put_wall: float | None      # strike with highest abs PE GEX below spot
    gex_flip_point: float       # price where cumulative GEX crosses zero

    largest_positive_gex: float  # max CE GEX at any single strike
    largest_negative_gex: float  # most negative PE GEX at any single strike

    # Regime
    regime: str             # 'POSITIVE_GEX' | 'NEGATIVE_GEX' | 'NEUTRAL_GEX'
    regime_strength: float  # 0–100

    # Intraday velocity
    gex_change_30min: float  # negative = becoming more volatile

    # Metadata
    expiries_included: list[date]
    total_oi_analyzed: int
    lot_size: int


@dataclass
class GEXSnapshot:
    """Market-wide GEX snapshot across all tracked symbols."""

    timestamp: datetime
    profiles: dict[str, GEXProfile]  # symbol → GEXProfile
    market_wide_gex: float            # NIFTY + BANKNIFTY net GEX
    market_regime: str                # overall GEX regime
