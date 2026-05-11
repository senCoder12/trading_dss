"""
Data models for the liquidity analysis module.

All dataclasses used by liquidity analyzers are defined here.
Every other file in this module imports from models.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE

_IST = ZoneInfo(IST_TIMEZONE)

# ---------------------------------------------------------------------------
# Candle
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    """Single OHLCV candle with pre-computed structural fields."""

    symbol: str
    timeframe: str          # '1min' | '3min' | '5min' | '15min' | '1hour'
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    # Derived — populated by __post_init__
    is_bullish: bool = field(init=False)
    body_size: float = field(init=False)
    upper_wick: float = field(init=False)
    lower_wick: float = field(init=False)
    total_range: float = field(init=False)

    def __post_init__(self) -> None:
        self.is_bullish = self.close > self.open
        self.body_size = abs(self.close - self.open)
        self.upper_wick = self.high - max(self.open, self.close)
        self.lower_wick = min(self.open, self.close) - self.low
        self.total_range = self.high - self.low


# ---------------------------------------------------------------------------
# LiquidityLevel
# ---------------------------------------------------------------------------

@dataclass
class LiquidityLevel:
    """
    A price level where liquidity pools (stop-losses / pending orders) rest.

    level_type values:
        'BSL'          — Buy-Side Liquidity (above price, where shorts have stops)
        'SSL'          — Sell-Side Liquidity (below price, where longs have stops)
        'PDH'          — Previous Day High
        'PDL'          — Previous Day Low
        'PWH'          — Previous Week High
        'PWL'          — Previous Week Low
        'ROUND_NUMBER' — Psychological round number
        'EQUAL_HIGH'   — Two or more equal highs (resting buy-side liquidity)
        'EQUAL_LOW'    — Two or more equal lows (resting sell-side liquidity)
    """

    symbol: str
    level_type: str
    price: float
    strength: int           # 1–5: how many times touched / confirmed
    timeframe: str
    first_formed_at: datetime
    last_touched_at: datetime
    is_swept: bool = False
    swept_at: Optional[datetime] = None
    wick_beyond: float = 0.0
    is_active: bool = True


# ---------------------------------------------------------------------------
# LiquiditySweeep  (intentional spelling per spec)
# ---------------------------------------------------------------------------

@dataclass
class LiquiditySweeep:
    """
    A liquidity sweep event — price wicked through a LiquidityLevel and
    reversed, indicating smart-money stop-hunt behaviour.

    sweep_type values:  'BSL_SWEEP' | 'SSL_SWEEP'
    signal_quality:     'HIGH' | 'MEDIUM' | 'LOW'
    """

    symbol: str
    timeframe: str
    level: LiquidityLevel
    sweep_candle: Candle
    sweep_type: str
    sweep_high: float
    sweep_low: float
    reversion_confirmed: bool
    timestamp: datetime
    reversion_candle: Optional[Candle] = None
    oi_confirms: bool = False
    futures_confirms: bool = False
    signal_quality: str = "LOW"


# ---------------------------------------------------------------------------
# FairValueGap
# ---------------------------------------------------------------------------

@dataclass
class FairValueGap:
    """
    A 3-candle imbalance (Fair Value Gap / FVG).

    gap_type:  'BULLISH' (price gapped up, gap acts as support)
               'BEARISH' (price gapped down, gap acts as resistance)

    Geometry:
        BULLISH FVG — gap_bottom = candle_1.high, gap_top = candle_3.low
        BEARISH FVG — gap_bottom = candle_3.high, gap_top = candle_1.low
    """

    symbol: str
    timeframe: str
    gap_type: str
    gap_top: float
    gap_bottom: float
    gap_midpoint: float
    gap_size: float
    gap_size_pct: float
    formed_at: datetime
    candle_1: Candle        # first candle of the 3-candle pattern
    candle_2: Candle        # impulse candle (middle)
    candle_3: Candle        # third candle confirming the gap
    is_filled: bool = False
    filled_at: Optional[datetime] = None
    fill_pct: float = 0.0   # 0–100
    is_mitigated: bool = False


# ---------------------------------------------------------------------------
# OrderBlock
# ---------------------------------------------------------------------------

@dataclass
class OrderBlock:
    """
    An Order Block — the last opposing candle before a strong displacement.

    ob_type:  'BULLISH' (last bearish candle before bullish displacement)
              'BEARISH' (last bullish candle before bearish displacement)

    strength is derived from displacement_size_pct in __post_init__:
        >= 1.5% → 'STRONG', >= 0.75% → 'MEDIUM', else → 'WEAK'
    """

    symbol: str
    timeframe: str
    ob_type: str
    ob_high: float
    ob_low: float
    ob_midpoint: float
    origin_candle: Candle
    displacement_candle: Candle
    displacement_size_pct: float
    formed_at: datetime
    is_mitigated: bool = False
    mitigated_at: Optional[datetime] = None
    touch_count: int = 0
    strength: str = field(init=False)

    def __post_init__(self) -> None:
        if self.displacement_size_pct >= 1.5:
            self.strength = "STRONG"
        elif self.displacement_size_pct >= 0.75:
            self.strength = "MEDIUM"
        else:
            self.strength = "WEAK"


# ---------------------------------------------------------------------------
# VolumeProfileLevel
# ---------------------------------------------------------------------------

@dataclass
class VolumeProfileLevel:
    """Single price level within a volume profile histogram."""

    symbol: str
    date: date
    timeframe: str
    price: float        # rounded to tick size
    volume: int
    volume_pct: float   # % of total day volume
    level_type: str     # 'POC' | 'VAH' | 'VAL' | 'HVN' | 'LVN'
    is_value_area: bool # True if within 70% value area


# ---------------------------------------------------------------------------
# VolumeProfile
# ---------------------------------------------------------------------------

@dataclass
class VolumeProfile:
    """
    Complete volume profile for a symbol on a given date/timeframe.

    POC — Point of Control (highest volume price)
    VAH — Value Area High (70% of volume boundary, upper)
    VAL — Value Area Low  (70% of volume boundary, lower)
    HVNs — High Volume Node prices (support/resistance)
    LVNs — Low Volume Node prices (fast-move zones)
    """

    symbol: str
    date: date
    timeframe: str
    poc: float
    vah: float
    val: float
    total_volume: int
    levels: list[VolumeProfileLevel]
    hvns: list[float]
    lvns: list[float]
    calculated_at: datetime


# ---------------------------------------------------------------------------
# LiquiditySnapshot
# ---------------------------------------------------------------------------

@dataclass
class LiquiditySnapshot:
    """
    Full liquidity picture for a symbol at a point in time.

    This is the primary output fed to DecisionEngine.

    liquidity_score: -100 to +100
        positive  → bullish bias (more BSL above, price likely to sweep up)
        negative  → bearish bias (more SSL below, price likely to sweep down)
        near zero → neutral / ranging

    active_setups: list of named setups currently valid, e.g.:
        'BSL_SWEEP_REVERSAL', 'FVG_FILL', 'OB_RETEST', 'SSL_SWEEP_REVERSAL'
    """

    symbol: str
    timestamp: datetime
    timeframe: str

    # Active liquidity levels
    active_bsl_levels: list[LiquidityLevel] = field(default_factory=list)
    active_ssl_levels: list[LiquidityLevel] = field(default_factory=list)

    # Nearest actionable levels
    nearest_bsl: Optional[LiquidityLevel] = None
    nearest_ssl: Optional[LiquidityLevel] = None
    distance_to_bsl_pct: float = 0.0
    distance_to_ssl_pct: float = 0.0

    # Directional bias
    liquidity_bias: str = "NEUTRAL"     # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    bias_reason: str = ""

    # Active SMC structures
    unfilled_fvgs: list[FairValueGap] = field(default_factory=list)
    active_order_blocks: list[OrderBlock] = field(default_factory=list)
    volume_profile: Optional[VolumeProfile] = None
    recent_sweeps: list[LiquiditySweeep] = field(default_factory=list)

    # Signal output for DecisionEngine
    liquidity_score: float = 0.0
    active_setups: list[str] = field(default_factory=list)
    invalidation_price: float = 0.0
