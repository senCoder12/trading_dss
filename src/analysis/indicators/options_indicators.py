"""
Options & Derivatives indicators for trading analysis.

Builds analytical layers on top of the raw options data from Phase 1.
Does NOT fetch data — receives OptionsChainData and OISummary objects from
Phase 1 and computes deeper insights: OI structure, buildup analysis,
enhanced max pain, IV analysis, IV rank/percentile, and a composite summary.

All methods are pure functions — no database calls or side effects.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import helper — OptionsChainData lives in src.data.options_chain.
# We use a TYPE_CHECKING guard so the module doesn't pull in NSEScraper
# at import time.
# ---------------------------------------------------------------------------
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.options_chain import OptionsChainData, OptionStrike


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OIStructureAnalysis:
    """Enhanced OI structure analysis beyond Phase 1 raw summary.

    The Put-Call Ratio (PCR) measures the ratio of put OI to call OI.
    A high PCR (> 1.0) is generally bullish — more puts being written
    means market makers are hedging and expect the underlying to stay
    above those put strikes.

    OI-based support and resistance:
      - Highest PE OI strike = support (put writers defend this level)
      - Highest CE OI strike = resistance (call writers defend this level)

    OI concentration measures how concentrated the OI is in the top 3
    strikes.  High concentration (> 0.5) indicates a strong wall.
    """

    total_ce_oi: int
    total_pe_oi: int
    pcr: float  # PE OI / CE OI
    pcr_interpretation: str  # VERY_BULLISH / BULLISH / NEUTRAL / BEARISH / VERY_BEARISH

    # OI-based support and resistance
    max_ce_oi_strike: float  # Highest CE OI = resistance
    max_pe_oi_strike: float  # Highest PE OI = support
    oi_based_range: tuple[float, float]  # (support, resistance)

    # OI concentration
    ce_oi_concentration: float  # Top 3 strikes CE OI / total CE OI
    pe_oi_concentration: float  # Same for PE

    # OI change analysis
    ce_oi_change_net: int  # Total CE OI change
    pe_oi_change_net: int
    oi_change_signal: str  # BULLISH / BEARISH / NEUTRAL


@dataclass
class StrikeBuildup:
    """Per-strike OI buildup classification.

    Buildup logic (using the option premium as the price):
      Price ↑, OI ↑ = LONG_BUILDUP   (new longs opening)
      Price ↓, OI ↑ = SHORT_BUILDUP  (new shorts opening)
      Price ↓, OI ↓ = LONG_UNWINDING (longs exiting)
      Price ↑, OI ↓ = SHORT_COVERING (shorts exiting)
    """

    strike: float
    option_type: str  # "CE" / "PE"
    oi_change: int
    oi_change_pct: float
    price_change: float
    buildup_type: str  # LONG_BUILDUP / SHORT_BUILDUP / LONG_UNWINDING / SHORT_COVERING
    significance: str  # HIGH / MEDIUM / LOW


@dataclass
class OIChangeAnalysis:
    """OI change / buildup analysis across the options chain.

    Analyses the top 10 most active strikes by absolute OI change to
    determine the dominant buildup type and net sentiment.

    ATM (At The Money) strike analysis is the most important signal —
    what's happening at the money reflects near-term expectations.
    """

    buildups: list[StrikeBuildup]
    dominant_buildup: str  # Most prevalent buildup type
    net_sentiment: str  # BULLISH / BEARISH / NEUTRAL

    # ATM strike analysis
    atm_ce_oi_change: int
    atm_pe_oi_change: int
    atm_signal: str  # What's happening at the money


@dataclass
class MaxPainAnalysis:
    """Enhanced Max Pain analysis with gravitational pull assessment.

    Max Pain is the strike at which option writers collectively lose
    the least money.  Markets tend to gravitate towards max pain,
    especially as expiry approaches.

    ``gravitational_pull`` strength depends on:
      - Distance from spot (closer = stronger pull)
      - Days to expiry (fewer days = stronger pull)
    """

    max_pain_strike: float
    distance_from_spot: float  # Points away from current price
    distance_pct: float  # Percentage away
    pain_curve: list[tuple[float, float]]  # (strike, total_pain) for visualization
    gravitational_pull: str  # STRONG / MODERATE / WEAK
    days_to_expiry: int
    max_pain_shift: Optional[float]  # How much max pain moved since last calculation


@dataclass
class IVAnalysis:
    """Implied Volatility analysis across the options chain.

    ``iv_skew`` measures the difference between OTM put IV and OTM call IV.
    A positive skew means puts are more expensive (downside protection demand),
    indicating fear in the market.

    ``iv_smile`` is the volatility smile curve — plots IV against strike prices.
    """

    atm_iv: float  # IV of ATM options (average of ATM CE and PE)
    iv_skew: float  # IV of OTM puts - IV of OTM calls
    iv_smile: list[tuple[float, float]]  # (strike, iv) for the smile curve

    # IV mean by option type
    avg_ce_iv: float
    avg_pe_iv: float
    iv_put_call_spread: float  # avg PE IV - avg CE IV


@dataclass
class IVRankResult:
    """IV Rank and IV Percentile result.

    IV Rank = (current_iv - 52w_low) / (52w_high - 52w_low) × 100
    IV Percentile = % of days where IV was below current IV

    Trading implications:
      - LOW IV Rank: options are cheap → favour buying strategies (long straddles)
      - HIGH IV Rank: options are expensive → favour selling strategies (iron condors)
    """

    iv_rank: float  # 0-100
    iv_percentile: float  # 0-100
    iv_regime: str  # LOW / NORMAL / HIGH / VERY_HIGH / insufficient_history
    trading_implication: str


@dataclass
class OptionsSummary:
    """Composite options reading from all sub-indicators.

    ``options_vote`` is derived from per-indicator bullish / bearish votes.
    """

    timestamp: datetime
    index_id: str
    expiry_date: date
    days_to_expiry: int

    # PCR
    pcr: float
    pcr_signal: str

    # OI-based levels
    oi_support: float  # Max PE OI strike
    oi_resistance: float  # Max CE OI strike
    expected_range: tuple[float, float]

    # Max Pain
    max_pain: float
    max_pain_pull: str

    # OI Change
    oi_change_signal: str
    dominant_buildup: str

    # IV
    atm_iv: float
    iv_regime: Optional[str]
    iv_skew: float

    # Overall options verdict
    options_vote: str  # STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG_BEARISH
    options_confidence: float  # 0.0 to 1.0

    # Warnings / notes
    warnings: list[str] = field(default_factory=list)

    # Advanced analytics (from run_advanced_options_analytics)
    gex: float = 0.0
    gex_regime: str = "NEUTRAL"           # "RANGEBOUND" / "TRENDING" / "NEUTRAL"
    iv_percentile: float = 50.0
    iv_rank: float = 50.0
    iv_strategy_bias: str = "NEUTRAL"     # "BUY_OPTIONS" / "SELL_PREMIUM" / "NEUTRAL"
    atm_straddle_price: float = 0.0
    expected_daily_move_pct: float = 0.0
    uoa_direction: str = "NONE"           # "BULLISH" / "BEARISH" / "MIXED" / "NONE"
    uoa_strikes: list = field(default_factory=list)
    nearest_ce_wall: float = 0.0          # Nearest massive CE OI resistance above spot
    nearest_pe_wall: float = 0.0          # Nearest massive PE OI support below spot
    ce_wall_strikes: list = field(default_factory=list)
    pe_wall_strikes: list = field(default_factory=list)
    put_call_skew: float = 0.0            # ATM PE IV - CE IV (positive = put skew)
    skew_regime: str = "BALANCED"         # "PUT_SKEW" / "CALL_SKEW" / "BALANCED"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _interpret_pcr(pcr: float) -> str:
    """Classify PCR into a sentiment label.

    PCR > 1.2  → VERY_BULLISH (more puts = hedging, market expects up)
    PCR 1.0-1.2 → BULLISH
    PCR 0.8-1.0 → NEUTRAL
    PCR 0.6-0.8 → BEARISH
    PCR < 0.6  → VERY_BEARISH
    """
    if pcr > 1.2:
        return "VERY_BULLISH"
    if pcr >= 1.0:
        return "BULLISH"
    if pcr >= 0.8:
        return "NEUTRAL"
    if pcr >= 0.6:
        return "BEARISH"
    return "VERY_BEARISH"


def _classify_buildup(price_change: float, oi_change: int) -> str:
    """Classify OI buildup type based on price and OI change direction."""
    if price_change > 0 and oi_change > 0:
        return "LONG_BUILDUP"
    if price_change < 0 and oi_change > 0:
        return "SHORT_BUILDUP"
    if price_change < 0 and oi_change < 0:
        return "LONG_UNWINDING"
    if price_change > 0 and oi_change < 0:
        return "SHORT_COVERING"
    return "NEUTRAL"


def _classify_significance(oi_change_pct: float) -> str:
    """Classify OI change significance based on percentage magnitude."""
    abs_pct = abs(oi_change_pct)
    if abs_pct > 20:
        return "HIGH"
    if abs_pct > 10:
        return "MEDIUM"
    return "LOW"


def _find_atm_strike(strikes: tuple, spot_price: float):
    """Find the At-The-Money strike closest to the spot price."""
    if not strikes:
        return None
    return min(strikes, key=lambda s: abs(s.strike_price - spot_price))


def _calculate_days_to_expiry(expiry_date: date) -> int:
    """Calculate trading days remaining to expiry (calendar days)."""
    today = date.today()
    delta = (expiry_date - today).days
    return max(0, delta)


# ---------------------------------------------------------------------------
# OptionsIndicators
# ---------------------------------------------------------------------------


class OptionsIndicators:
    """Pure-function options indicator calculations.

    Receives ``OptionsChainData`` from Phase 1 and computes deeper
    analytical layers: OI structure, buildup analysis, enhanced max pain,
    IV analysis, IV rank/percentile, and composite summary.

    All public methods are stateless.  The only mutable state is
    ``_previous_max_pain`` which tracks the last max pain value to detect
    shifts across successive calls.
    """

    def __init__(self) -> None:
        # Track previous max pain per index/expiry for shift detection
        self._previous_max_pain: dict[str, float] = {}

    # ------------------------------------------------------------------
    # OI Structure Analysis
    # ------------------------------------------------------------------

    def analyze_oi_structure(self, chain) -> OIStructureAnalysis:
        """Analyse the OI structure of an options chain.

        Computes PCR, identifies OI-based support/resistance levels,
        measures OI concentration, and analyses OI changes.

        Args:
            chain: OptionsChainData from Phase 1.

        Returns:
            OIStructureAnalysis with all fields populated.
        """
        strikes = chain.strikes

        # --- Total OI ---
        total_ce_oi = sum(s.ce_oi for s in strikes)
        total_pe_oi = sum(s.pe_oi for s in strikes)

        # --- PCR ---
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0
        pcr = round(pcr, 4)
        pcr_interpretation = _interpret_pcr(pcr)

        # --- Max OI strikes (support / resistance) ---
        # Filter out strikes with 0 OI for meaningful analysis
        ce_with_oi = [s for s in strikes if s.ce_oi > 0]
        pe_with_oi = [s for s in strikes if s.pe_oi > 0]

        max_ce_oi_strike = (
            max(ce_with_oi, key=lambda s: s.ce_oi).strike_price
            if ce_with_oi
            else 0.0
        )
        max_pe_oi_strike = (
            max(pe_with_oi, key=lambda s: s.pe_oi).strike_price
            if pe_with_oi
            else 0.0
        )
        oi_based_range = (max_pe_oi_strike, max_ce_oi_strike)

        # --- OI concentration (top 3 strikes / total) ---
        ce_sorted = sorted(strikes, key=lambda s: s.ce_oi, reverse=True)
        pe_sorted = sorted(strikes, key=lambda s: s.pe_oi, reverse=True)

        top3_ce_oi = sum(s.ce_oi for s in ce_sorted[:3])
        top3_pe_oi = sum(s.pe_oi for s in pe_sorted[:3])

        ce_oi_concentration = round(top3_ce_oi / total_ce_oi, 4) if total_ce_oi > 0 else 0.0
        pe_oi_concentration = round(top3_pe_oi / total_pe_oi, 4) if total_pe_oi > 0 else 0.0

        # --- OI change analysis ---
        ce_oi_change_net = sum(s.ce_oi_change for s in strikes)
        pe_oi_change_net = sum(s.pe_oi_change for s in strikes)

        # PE OI adding > CE OI adding → bullish
        oi_change_diff = pe_oi_change_net - ce_oi_change_net
        if oi_change_diff > 0:
            oi_change_signal = "BULLISH"
        elif oi_change_diff < 0:
            oi_change_signal = "BEARISH"
        else:
            oi_change_signal = "NEUTRAL"

        return OIStructureAnalysis(
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            pcr=pcr,
            pcr_interpretation=pcr_interpretation,
            max_ce_oi_strike=max_ce_oi_strike,
            max_pe_oi_strike=max_pe_oi_strike,
            oi_based_range=oi_based_range,
            ce_oi_concentration=ce_oi_concentration,
            pe_oi_concentration=pe_oi_concentration,
            ce_oi_change_net=ce_oi_change_net,
            pe_oi_change_net=pe_oi_change_net,
            oi_change_signal=oi_change_signal,
        )

    # ------------------------------------------------------------------
    # OI Change / Buildup Analysis
    # ------------------------------------------------------------------

    def analyze_oi_change(self, chain, spot_price: float) -> OIChangeAnalysis:
        """Analyse OI changes and buildup patterns across the chain.

        Uses the intra-day OI change and price (LTP) data already embedded
        in each ``OptionStrike`` to classify buildup type for the top 10
        most active strikes.

        Buildup logic uses the option *premium* change as the price signal:
          - Premium ↑ + OI ↑ → LONG_BUILDUP
          - Premium ↓ + OI ↑ → SHORT_BUILDUP
          - Premium ↓ + OI ↓ → LONG_UNWINDING
          - Premium ↑ + OI ↓ → SHORT_COVERING

        Since we only have single-snapshot data (OI change and LTP, not
        the previous LTP), we approximate: if OI is increasing and the
        premium is relatively high (above intrinsic), it suggests long
        buildup.  For a more precise classification, ``ce_oi_change`` and
        ``pe_oi_change`` from the NSE API already represent the *change*
        in OI since the previous day.  We use ``ce_ltp`` directly as a
        proxy for current premium direction — higher premium relative to
        intrinsic value suggests buying pressure (long buildup), lower
        suggests selling pressure (short buildup).

        For the premium-change proxy we use:
          CE: if LTP > max(0, spot - strike) + 5 → premium is "rich" → price_change > 0
          PE: if LTP > max(0, strike - spot) + 5 → premium is "rich" → price_change > 0

        Args:
            chain: OptionsChainData from Phase 1.
            spot_price: Current spot price of the underlying.

        Returns:
            OIChangeAnalysis with all fields populated.
        """
        buildups: list[StrikeBuildup] = []

        for strike in chain.strikes:
            # --- CE buildup ---
            if strike.ce_oi_change != 0:
                ce_intrinsic = max(0.0, spot_price - strike.strike_price)
                ce_time_value = strike.ce_ltp - ce_intrinsic
                # Positive time value above a small threshold → premium is rich
                ce_price_signal = 1.0 if ce_time_value > 2.0 else -1.0
                if strike.ce_ltp == 0.0:
                    ce_price_signal = 0.0

                ce_prev_oi = strike.ce_oi - strike.ce_oi_change
                ce_oi_change_pct = (
                    (strike.ce_oi_change / ce_prev_oi * 100.0)
                    if ce_prev_oi > 0
                    else (100.0 if strike.ce_oi_change > 0 else 0.0)
                )

                buildup_type = _classify_buildup(ce_price_signal, strike.ce_oi_change)
                if buildup_type != "NEUTRAL":
                    buildups.append(StrikeBuildup(
                        strike=strike.strike_price,
                        option_type="CE",
                        oi_change=strike.ce_oi_change,
                        oi_change_pct=round(ce_oi_change_pct, 2),
                        price_change=round(ce_price_signal, 2),
                        buildup_type=buildup_type,
                        significance=_classify_significance(ce_oi_change_pct),
                    ))

            # --- PE buildup ---
            if strike.pe_oi_change != 0:
                pe_intrinsic = max(0.0, strike.strike_price - spot_price)
                pe_time_value = strike.pe_ltp - pe_intrinsic
                pe_price_signal = 1.0 if pe_time_value > 2.0 else -1.0
                if strike.pe_ltp == 0.0:
                    pe_price_signal = 0.0

                pe_prev_oi = strike.pe_oi - strike.pe_oi_change
                pe_oi_change_pct = (
                    (strike.pe_oi_change / pe_prev_oi * 100.0)
                    if pe_prev_oi > 0
                    else (100.0 if strike.pe_oi_change > 0 else 0.0)
                )

                buildup_type = _classify_buildup(pe_price_signal, strike.pe_oi_change)
                if buildup_type != "NEUTRAL":
                    buildups.append(StrikeBuildup(
                        strike=strike.strike_price,
                        option_type="PE",
                        oi_change=strike.pe_oi_change,
                        oi_change_pct=round(pe_oi_change_pct, 2),
                        price_change=round(pe_price_signal, 2),
                        buildup_type=buildup_type,
                        significance=_classify_significance(pe_oi_change_pct),
                    ))

        # Sort by absolute OI change and take top 10
        buildups.sort(key=lambda b: abs(b.oi_change), reverse=True)
        top_buildups = buildups[:10]

        # --- Dominant buildup ---
        if top_buildups:
            type_counts: dict[str, int] = {}
            for b in top_buildups:
                type_counts[b.buildup_type] = type_counts.get(b.buildup_type, 0) + 1
            dominant_buildup = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]
        else:
            dominant_buildup = "NEUTRAL"

        # --- Net sentiment from buildup mix ---
        bullish_types = {"SHORT_COVERING", "LONG_BUILDUP"}
        bearish_types = {"SHORT_BUILDUP", "LONG_UNWINDING"}

        bullish_count = sum(1 for b in top_buildups if b.buildup_type in bullish_types)
        bearish_count = sum(1 for b in top_buildups if b.buildup_type in bearish_types)

        if bullish_count > bearish_count + 2:
            net_sentiment = "BULLISH"
        elif bearish_count > bullish_count + 2:
            net_sentiment = "BEARISH"
        else:
            net_sentiment = "NEUTRAL"

        # --- ATM analysis ---
        atm_strike = _find_atm_strike(chain.strikes, spot_price)
        atm_ce_oi_change = atm_strike.ce_oi_change if atm_strike else 0
        atm_pe_oi_change = atm_strike.pe_oi_change if atm_strike else 0

        # ATM signal interpretation
        if atm_pe_oi_change > atm_ce_oi_change and atm_pe_oi_change > 0:
            atm_signal = "BULLISH_ATM_PE_ADDING"
        elif atm_ce_oi_change > atm_pe_oi_change and atm_ce_oi_change > 0:
            atm_signal = "BEARISH_ATM_CE_ADDING"
        elif atm_ce_oi_change < 0 and atm_pe_oi_change < 0:
            atm_signal = "UNWINDING_BOTH_SIDES"
        else:
            atm_signal = "NEUTRAL"

        return OIChangeAnalysis(
            buildups=top_buildups,
            dominant_buildup=dominant_buildup,
            net_sentiment=net_sentiment,
            atm_ce_oi_change=atm_ce_oi_change,
            atm_pe_oi_change=atm_pe_oi_change,
            atm_signal=atm_signal,
        )

    # ------------------------------------------------------------------
    # Max Pain (enhanced)
    # ------------------------------------------------------------------

    def calculate_max_pain_detailed(
        self, chain, previous_max_pain: Optional[float] = None
    ) -> MaxPainAnalysis:
        """Calculate enhanced Max Pain with gravitational pull assessment.

        Max Pain is the strike at which option writers collectively lose
        the least money.  For every candidate strike, we compute the total
        intrinsic value (loss to writers) of all CE and PE positions as if
        the underlying expired at that strike.

        Gravitational pull classification:
          - STRONG: spot within 0.5% of max pain
          - MODERATE: spot 0.5-1.5% from max pain
          - WEAK: spot > 1.5% from max pain

        Args:
            chain: OptionsChainData from Phase 1.
            previous_max_pain: Previous max pain value for shift detection.

        Returns:
            MaxPainAnalysis with all fields populated.
        """
        strikes = chain.strikes
        spot_price = chain.spot_price

        if not strikes:
            return MaxPainAnalysis(
                max_pain_strike=0.0,
                distance_from_spot=0.0,
                distance_pct=0.0,
                pain_curve=[],
                gravitational_pull="WEAK",
                days_to_expiry=0,
                max_pain_shift=None,
            )

        # --- Build pain curve ---
        pain_curve: list[tuple[float, float]] = []
        min_pain = float("inf")
        max_pain_strike = 0.0

        target_strikes = [s.strike_price for s in strikes]

        for assumed_expiry_price in target_strikes:
            total_pain = 0.0

            for option in strikes:
                # CE writers lose if price > strike
                if assumed_expiry_price > option.strike_price:
                    total_pain += option.ce_oi * (assumed_expiry_price - option.strike_price)

                # PE writers lose if price < strike
                if assumed_expiry_price < option.strike_price:
                    total_pain += option.pe_oi * (option.strike_price - assumed_expiry_price)

            pain_curve.append((assumed_expiry_price, total_pain))

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = assumed_expiry_price

        # --- Distance metrics ---
        distance_from_spot = max_pain_strike - spot_price
        distance_pct = (
            round(abs(distance_from_spot) / spot_price * 100, 4)
            if spot_price > 0
            else 0.0
        )

        # --- Gravitational pull ---
        if distance_pct <= 0.5:
            gravitational_pull = "STRONG"
        elif distance_pct <= 1.5:
            gravitational_pull = "MODERATE"
        else:
            gravitational_pull = "WEAK"

        # --- Days to expiry ---
        days_to_expiry = _calculate_days_to_expiry(chain.expiry_date)

        # --- Max pain shift ---
        cache_key = f"{chain.index_id}_{chain.expiry_date.isoformat()}"
        if previous_max_pain is not None:
            max_pain_shift = max_pain_strike - previous_max_pain
        elif cache_key in self._previous_max_pain:
            max_pain_shift = max_pain_strike - self._previous_max_pain[cache_key]
        else:
            max_pain_shift = None

        # Update cache for next call
        self._previous_max_pain[cache_key] = max_pain_strike

        return MaxPainAnalysis(
            max_pain_strike=max_pain_strike,
            distance_from_spot=round(distance_from_spot, 2),
            distance_pct=distance_pct,
            pain_curve=pain_curve,
            gravitational_pull=gravitational_pull,
            days_to_expiry=days_to_expiry,
            max_pain_shift=round(max_pain_shift, 2) if max_pain_shift is not None else None,
        )

    # ------------------------------------------------------------------
    # IV Analysis
    # ------------------------------------------------------------------

    def analyze_iv(self, chain) -> IVAnalysis:
        """Analyse Implied Volatility across the options chain.

        Computes ATM IV, IV skew, and the volatility smile curve.
        Strikes with IV = 0 (deep OTM with no trading activity) are
        excluded from calculations.

        IV skew = avg OTM put IV − avg OTM call IV.
        Positive skew means puts are more expensive (fear premium).

        Args:
            chain: OptionsChainData from Phase 1.

        Returns:
            IVAnalysis with all fields populated.
        """
        spot_price = chain.spot_price
        strikes = chain.strikes

        # --- Filter out strikes with IV = 0 ---
        valid_ce = [(s.strike_price, s.ce_iv) for s in strikes if s.ce_iv > 0]
        valid_pe = [(s.strike_price, s.pe_iv) for s in strikes if s.pe_iv > 0]

        # --- ATM IV (average of ATM CE and PE IV) ---
        atm_strike = _find_atm_strike(chain.strikes, spot_price)
        if atm_strike and atm_strike.ce_iv > 0 and atm_strike.pe_iv > 0:
            atm_iv = round((atm_strike.ce_iv + atm_strike.pe_iv) / 2, 2)
        elif atm_strike and atm_strike.ce_iv > 0:
            atm_iv = round(atm_strike.ce_iv, 2)
        elif atm_strike and atm_strike.pe_iv > 0:
            atm_iv = round(atm_strike.pe_iv, 2)
        else:
            atm_iv = 0.0

        # --- IV Skew (OTM puts vs OTM calls) ---
        otm_ce_ivs = [iv for strike, iv in valid_ce if strike > spot_price]
        otm_pe_ivs = [iv for strike, iv in valid_pe if strike < spot_price]

        avg_otm_ce_iv = sum(otm_ce_ivs) / len(otm_ce_ivs) if otm_ce_ivs else 0.0
        avg_otm_pe_iv = sum(otm_pe_ivs) / len(otm_pe_ivs) if otm_pe_ivs else 0.0
        iv_skew = round(avg_otm_pe_iv - avg_otm_ce_iv, 4)

        # --- IV Smile ---
        iv_smile: list[tuple[float, float]] = []
        for strike in strikes:
            # Use average of CE and PE IV for the smile, or whichever is available
            ce_iv = strike.ce_iv if strike.ce_iv > 0 else None
            pe_iv = strike.pe_iv if strike.pe_iv > 0 else None
            if ce_iv is not None and pe_iv is not None:
                avg_iv = (ce_iv + pe_iv) / 2
            elif ce_iv is not None:
                avg_iv = ce_iv
            elif pe_iv is not None:
                avg_iv = pe_iv
            else:
                continue  # Skip strikes with no IV data
            iv_smile.append((strike.strike_price, round(avg_iv, 2)))

        # --- Average IV by option type ---
        avg_ce_iv = round(sum(iv for _, iv in valid_ce) / len(valid_ce), 2) if valid_ce else 0.0
        avg_pe_iv = round(sum(iv for _, iv in valid_pe) / len(valid_pe), 2) if valid_pe else 0.0
        iv_put_call_spread = round(avg_pe_iv - avg_ce_iv, 4)

        return IVAnalysis(
            atm_iv=atm_iv,
            iv_skew=iv_skew,
            iv_smile=iv_smile,
            avg_ce_iv=avg_ce_iv,
            avg_pe_iv=avg_pe_iv,
            iv_put_call_spread=iv_put_call_spread,
        )

    # ------------------------------------------------------------------
    # IV Rank & IV Percentile
    # ------------------------------------------------------------------

    def calculate_iv_rank(
        self, current_iv: float, iv_history: list[float]
    ) -> Optional[IVRankResult]:
        """Calculate IV Rank and IV Percentile from historical IV data.

        IV Rank = (current - 52w_low) / (52w_high - 52w_low) × 100
        IV Percentile = % of days where IV was below current IV

        Requires at least 30 data points for meaningful calculation.
        Returns ``None`` if history is insufficient.

        Args:
            current_iv: Current ATM IV.
            iv_history: List of daily ATM IV snapshots (most recent last).

        Returns:
            IVRankResult or None if insufficient history.
        """
        if len(iv_history) < 30:
            return IVRankResult(
                iv_rank=0.0,
                iv_percentile=0.0,
                iv_regime="insufficient_history",
                trading_implication="Insufficient IV history for meaningful analysis.",
            )

        # Filter out zero/invalid entries
        valid_history = [iv for iv in iv_history if iv > 0]
        if len(valid_history) < 30:
            return IVRankResult(
                iv_rank=0.0,
                iv_percentile=0.0,
                iv_regime="insufficient_history",
                trading_implication="Insufficient valid IV data points.",
            )

        iv_high = max(valid_history)
        iv_low = min(valid_history)

        # IV Rank
        if iv_high == iv_low:
            iv_rank = 50.0  # Flat IV history
        else:
            iv_rank = round((current_iv - iv_low) / (iv_high - iv_low) * 100, 2)

        # Clamp to 0-100
        iv_rank = max(0.0, min(100.0, iv_rank))

        # IV Percentile
        days_below = sum(1 for iv in valid_history if iv < current_iv)
        iv_percentile = round(days_below / len(valid_history) * 100, 2)

        # IV Regime
        if iv_rank < 25:
            iv_regime = "LOW"
            trading_implication = "Options cheap, favor buying strategies (long straddles, debit spreads)."
        elif iv_rank < 50:
            iv_regime = "NORMAL"
            trading_implication = "IV is moderate. No strong directional bias from IV alone."
        elif iv_rank < 75:
            iv_regime = "HIGH"
            trading_implication = "Options expensive, favor selling strategies (iron condors, credit spreads)."
        else:
            iv_regime = "VERY_HIGH"
            trading_implication = "Options very expensive, strong edge in selling premium. Watch for mean reversion."

        return IVRankResult(
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            iv_regime=iv_regime,
            trading_implication=trading_implication,
        )

    # ------------------------------------------------------------------
    # Options Summary
    # ------------------------------------------------------------------

    def get_options_summary(
        self,
        chain,
        iv_history: Optional[list[float]] = None,
        previous_max_pain: Optional[float] = None,
        advanced_metrics=None,   # AdvancedOptionsMetrics from options_analysis.py
    ) -> Optional[OptionsSummary]:
        """Produce a composite options summary from all sub-indicators.

        Voting system:

        Bullish votes:
          - PCR > 1.0                                                 → +1
          - PE OI adding > CE OI adding                               → +1
          - Spot near max PE OI support (within 1%)                   → +1
          - Max pain above spot                                       → +0.5
          - Long buildup dominant at ATM PE                           → +1
          - High IV skew (puts expensive) = contrarian bullish        → +0.5

        Bearish votes:
          - PCR < 0.7                                                 → -1
          - CE OI adding > PE OI adding                               → -1
          - Spot near max CE OI resistance (within 1%)                → -1
          - Max pain below spot                                       → -0.5
          - Short buildup dominant at ATM CE                          → -1

        Adjustments:
          - Days to expiry < 2: increase max pain weight (×2)

        Score mapping:
          net >= +3   → STRONG_BULLISH  (confidence 0.85)
          net >= +1.5 → BULLISH         (confidence 0.70)
          net >= -1   → NEUTRAL         (confidence 0.50)
          net >= -2.5 → BEARISH         (confidence 0.70)
          net < -2.5  → STRONG_BEARISH  (confidence 0.85)

        Args:
            chain: OptionsChainData from Phase 1.
            iv_history: Optional list of daily ATM IV snapshots.
            previous_max_pain: Previous max pain value for shift detection.

        Returns:
            OptionsSummary with all fields populated, or None if the
            chain has no strikes.
        """
        if not chain.strikes:
            return None

        now = datetime.utcnow()
        warnings_list: list[str] = []
        spot_price = chain.spot_price

        # --- Sub-indicator calculations ---
        oi_structure = self.analyze_oi_structure(chain)
        oi_change = self.analyze_oi_change(chain, spot_price)
        max_pain = self.calculate_max_pain_detailed(chain, previous_max_pain)
        iv_analysis = self.analyze_iv(chain)

        # IV Rank (optional)
        iv_regime: Optional[str] = None
        iv_rank_pct = 50.0
        iv_percentile_from_rank = 50.0
        if iv_history is not None:
            iv_rank_result = self.calculate_iv_rank(iv_analysis.atm_iv, iv_history)
            if iv_rank_result is not None:
                iv_regime = iv_rank_result.iv_regime
                iv_rank_pct = iv_rank_result.iv_rank
                iv_percentile_from_rank = iv_rank_result.iv_percentile
        else:
            warnings_list.append("No IV history provided — IV regime unavailable.")

        # --- Days to expiry ---
        days_to_expiry = _calculate_days_to_expiry(chain.expiry_date)

        # --- Advanced metrics defaults (initialised before voting so they are
        #     always available for the return statement regardless of branch) ---
        adv_gex = 0.0
        adv_gex_regime = "NEUTRAL"
        adv_iv_percentile = 50.0
        adv_iv_rank = 50.0
        adv_iv_strategy_bias = "NEUTRAL"
        adv_straddle = 0.0
        adv_expected_move_pct = 0.0
        adv_uoa_direction = "NONE"
        adv_uoa_strikes: list = []
        adv_nearest_ce_wall = 0.0
        adv_nearest_pe_wall = 0.0
        adv_ce_walls: list = []
        adv_pe_walls: list = []
        adv_put_call_skew = 0.0
        adv_skew_regime = "BALANCED"

        # Seed IV rank/percentile from iv_history if advanced_metrics is absent
        if advanced_metrics is None and iv_history is not None:
            adv_iv_percentile = iv_percentile_from_rank
            adv_iv_rank = iv_rank_pct
            if adv_iv_percentile >= 75:
                adv_iv_strategy_bias = "SELL_PREMIUM"
            elif adv_iv_percentile <= 25:
                adv_iv_strategy_bias = "BUY_OPTIONS"

        # --- Voting ---
        score = 0.0

        # PCR signal
        if oi_structure.pcr > 1.0:
            score += 1.0
        elif oi_structure.pcr < 0.7:
            score -= 1.0

        # OI change direction
        if oi_structure.pe_oi_change_net > oi_structure.ce_oi_change_net:
            score += 1.0
        elif oi_structure.ce_oi_change_net > oi_structure.pe_oi_change_net:
            score -= 1.0

        # Proximity to OI-based support/resistance
        if spot_price > 0 and oi_structure.max_pe_oi_strike > 0:
            dist_to_support_pct = abs(spot_price - oi_structure.max_pe_oi_strike) / spot_price * 100
            if dist_to_support_pct <= 1.0:
                score += 1.0  # Near support = bullish (support holding)

        if spot_price > 0 and oi_structure.max_ce_oi_strike > 0:
            dist_to_resistance_pct = abs(spot_price - oi_structure.max_ce_oi_strike) / spot_price * 100
            if dist_to_resistance_pct <= 1.0:
                score -= 1.0  # Near resistance = bearish

        # Max pain pull
        max_pain_weight = 0.5
        if days_to_expiry < 2:
            max_pain_weight = 1.0  # Gravitational pull strongest near expiry
            warnings_list.append("Near expiry — max pain gravitational pull strengthened.")

        if max_pain.max_pain_strike > spot_price:
            score += max_pain_weight  # Bullish pull (price should move up towards max pain)
        elif max_pain.max_pain_strike < spot_price:
            score -= max_pain_weight  # Bearish pull

        # ATM buildup signal
        if oi_change.atm_signal == "BEARISH_ATM_CE_ADDING":
            score -= 1.0  # Short buildup at ATM CE = bearish
        elif oi_change.atm_signal == "BULLISH_ATM_PE_ADDING":
            score += 1.0  # Long buildup at ATM PE = bullish

        # IV skew — high skew (puts expensive) = fear = contrarian bullish
        if iv_analysis.iv_skew > 5.0:
            score += 0.5
            warnings_list.append("High IV skew: puts are expensive — contrarian bullish signal.")

        # ── Advanced Metrics Votes (when available) ────────────────────────
        if advanced_metrics is not None:
            try:
                adv_gex = float(getattr(advanced_metrics, "gex", 0.0) or 0.0)
                adv_gex_regime = getattr(advanced_metrics, "gex_regime", "NEUTRAL") or "NEUTRAL"
                adv_iv_percentile = float(getattr(advanced_metrics, "iv_percentile", 50.0) or 50.0)
                adv_iv_rank = float(getattr(advanced_metrics, "iv_rank", 50.0) or 50.0)
                adv_iv_strategy_bias = getattr(advanced_metrics, "iv_strategy_bias", "NEUTRAL") or "NEUTRAL"
                adv_straddle = float(getattr(advanced_metrics, "atm_straddle_price", 0.0) or 0.0)
                adv_expected_move_pct = float(getattr(advanced_metrics, "expected_daily_move", 0.0) or 0.0)
                adv_uoa_direction = getattr(advanced_metrics, "uoa_direction", "NONE") or "NONE"
                adv_uoa_strikes = list(getattr(advanced_metrics, "uoa_strikes", []) or [])
                adv_nearest_ce_wall = float(getattr(advanced_metrics, "nearest_ce_wall", 0.0) or 0.0)
                adv_nearest_pe_wall = float(getattr(advanced_metrics, "nearest_pe_wall", 0.0) or 0.0)
                adv_ce_walls = list(getattr(advanced_metrics, "ce_wall_strikes", []) or [])
                adv_pe_walls = list(getattr(advanced_metrics, "pe_wall_strikes", []) or [])
                adv_put_call_skew = float(getattr(advanced_metrics, "put_call_skew", 0.0) or 0.0)
                adv_skew_regime = getattr(advanced_metrics, "skew_regime", "BALANCED") or "BALANCED"

                # GEX regime vote: positive GEX = rangebound = dampen directional signal
                if adv_gex_regime == "RANGEBOUND":
                    score *= 0.7  # Dampen existing score — market makers will pin price
                    warnings_list.append("GEX positive (rangebound) — dealer hedging dampens directional move")
                elif adv_gex_regime == "TRENDING":
                    score *= 1.15  # Amplify — GEX negative means dealers amplify moves
                    warnings_list.append("GEX negative (trending) — dealer flow amplifies directional move")

                # UOA: unusual options activity
                if adv_uoa_direction == "BULLISH":
                    score += 1.0
                    warnings_list.append(f"UOA: unusual bullish activity at {len(adv_uoa_strikes)} strikes")
                elif adv_uoa_direction == "BEARISH":
                    score -= 1.0
                    warnings_list.append(f"UOA: unusual bearish activity at {len(adv_uoa_strikes)} strikes")

                # OI walls: proximity adjustment
                if spot_price > 0:
                    if adv_nearest_ce_wall > 0:
                        dist_to_ce_wall = (adv_nearest_ce_wall - spot_price) / spot_price * 100
                        if 0 < dist_to_ce_wall < 0.5:
                            score -= 0.5  # Hitting resistance wall = bearish friction
                            warnings_list.append(f"CE wall at {adv_nearest_ce_wall:.0f} — {dist_to_ce_wall:.2f}% above spot, resistance likely")
                    if adv_nearest_pe_wall > 0:
                        dist_to_pe_wall = (spot_price - adv_nearest_pe_wall) / spot_price * 100
                        if 0 < dist_to_pe_wall < 0.5:
                            score += 0.5  # Resting on support wall = bullish
                            warnings_list.append(f"PE wall at {adv_nearest_pe_wall:.0f} — support {dist_to_pe_wall:.2f}% below spot")

                # IV percentile: adjust confidence direction
                # High IVP = options expensive = mean reversion risk for buyers
                if adv_iv_percentile >= 75:
                    warnings_list.append(f"IV Percentile {adv_iv_percentile:.0f}% — options expensive, buyers face IV crush risk")
                elif adv_iv_percentile <= 25:
                    warnings_list.append(f"IV Percentile {adv_iv_percentile:.0f}% — options cheap, low risk for option buyers")

                # Put-call skew: deep put skew = institutional hedging = usually pre-crash
                if adv_skew_regime == "PUT_SKEW" and adv_put_call_skew > 3.0:
                    score -= 0.5
                    warnings_list.append(f"Strong put skew ({adv_put_call_skew:.1f}) — institutional downside hedging")
                elif adv_skew_regime == "CALL_SKEW" and adv_put_call_skew < -3.0:
                    score += 0.5
                    warnings_list.append(f"Unusual call skew ({adv_put_call_skew:.1f}) — institutional upside positioning")

            except Exception as _adv_e:
                logger.debug("Advanced metrics integration failed: %s", _adv_e)

        # --- Vote mapping ---
        if score >= 3.0:
            options_vote, base_confidence = "STRONG_BULLISH", 0.85
        elif score >= 1.5:
            options_vote, base_confidence = "BULLISH", 0.70
        elif score >= -1.0:
            options_vote, base_confidence = "NEUTRAL", 0.50
        elif score >= -2.5:
            options_vote, base_confidence = "BEARISH", 0.70
        else:
            options_vote, base_confidence = "STRONG_BEARISH", 0.85

        options_confidence = round(base_confidence, 2)

        return OptionsSummary(
            timestamp=chain.timestamp if hasattr(chain, "timestamp") else now,
            index_id=chain.index_id,
            expiry_date=chain.expiry_date,
            days_to_expiry=days_to_expiry,
            pcr=oi_structure.pcr,
            pcr_signal=oi_structure.pcr_interpretation,
            oi_support=oi_structure.max_pe_oi_strike,
            oi_resistance=oi_structure.max_ce_oi_strike,
            expected_range=oi_structure.oi_based_range,
            max_pain=max_pain.max_pain_strike,
            max_pain_pull=max_pain.gravitational_pull,
            oi_change_signal=oi_structure.oi_change_signal,
            dominant_buildup=oi_change.dominant_buildup,
            atm_iv=iv_analysis.atm_iv,
            iv_regime=iv_regime,
            iv_skew=iv_analysis.iv_skew,
            options_vote=options_vote,
            options_confidence=options_confidence,
            warnings=warnings_list,
            gex=adv_gex,
            gex_regime=adv_gex_regime,
            iv_percentile=adv_iv_percentile,
            iv_rank=adv_iv_rank,
            iv_strategy_bias=adv_iv_strategy_bias,
            atm_straddle_price=adv_straddle,
            expected_daily_move_pct=adv_expected_move_pct,
            uoa_direction=adv_uoa_direction,
            uoa_strikes=adv_uoa_strikes,
            nearest_ce_wall=adv_nearest_ce_wall,
            nearest_pe_wall=adv_nearest_pe_wall,
            ce_wall_strikes=adv_ce_walls,
            pe_wall_strikes=adv_pe_walls,
            put_call_skew=adv_put_call_skew,
            skew_regime=adv_skew_regime,
        )
