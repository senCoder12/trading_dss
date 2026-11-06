"""
Options chain analysis: OI interpretation, PCR, and Max Pain calculation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class OIAnalysis:
    """
    Result of an open-interest analysis pass.

    Attributes
    ----------
    pcr_oi:
        Put-Call Ratio based on open interest (total PE OI / total CE OI).
    pcr_volume:
        Put-Call Ratio based on volume.
    max_pain:
        Strike price at which aggregate option buyer loss is maximum.
    atm_strike:
        At-the-money strike closest to current underlying.
    ce_oi_spikes:
        Strikes with unusual CE OI build-up.
    pe_oi_spikes:
        Strikes with unusual PE OI build-up.
    sentiment:
        Derived sentiment: ``"bullish"`` | ``"bearish"`` | ``"neutral"``.
    """

    pcr_oi: float
    pcr_volume: float
    max_pain: float
    atm_strike: float
    ce_oi_spikes: list[float]
    pe_oi_spikes: list[float]
    sentiment: str


def calculate_pcr(chain_df: pd.DataFrame) -> tuple[float, float]:
    """
    Calculate Put-Call Ratio from an options chain DataFrame.

    Parameters
    ----------
    chain_df:
        DataFrame with columns ``ce_oi``, ``pe_oi``, ``ce_volume``, ``pe_volume``.

    Returns
    -------
    tuple[float, float]:
        ``(pcr_oi, pcr_volume)`` — ratios of total PE to total CE.
    """
    total_ce_oi = chain_df["ce_oi"].sum()
    total_pe_oi = chain_df["pe_oi"].sum()
    total_ce_vol = chain_df.get("ce_volume", pd.Series(dtype=float)).sum()
    total_pe_vol = chain_df.get("pe_volume", pd.Series(dtype=float)).sum()

    pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0
    pcr_volume = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0.0
    return round(pcr_oi, 4), round(pcr_volume, 4)


def calculate_max_pain(chain_df: pd.DataFrame) -> float:
    """
    Calculate Max Pain strike.

    Max Pain is the strike at which total option writer profit (= buyer loss)
    is maximised on expiry.

    Parameters
    ----------
    chain_df:
        Options chain with columns ``strike``, ``ce_oi``, ``pe_oi``.

    Returns
    -------
    float:
        Max Pain strike price.
    """
    strikes = chain_df["strike"].sort_values().unique()
    losses: dict[float, float] = {}

    for expiry_strike in strikes:
        ce_loss = sum(
            max(expiry_strike - s, 0) * oi
            for s, oi in zip(chain_df["strike"], chain_df["ce_oi"])
        )
        pe_loss = sum(
            max(s - expiry_strike, 0) * oi
            for s, oi in zip(chain_df["strike"], chain_df["pe_oi"])
        )
        losses[expiry_strike] = ce_loss + pe_loss

    if not losses:
        return 0.0
    return float(min(losses, key=losses.__getitem__))


def find_oi_spikes(
    chain_df: pd.DataFrame,
    column: str,
    multiplier: float | None = None,
) -> list[float]:
    """
    Find strikes with abnormally high OI compared to the mean.

    Parameters
    ----------
    chain_df:
        Options chain DataFrame.
    column:
        ``"ce_oi"`` or ``"pe_oi"``.
    multiplier:
        Flag strike if OI > multiplier × mean. Defaults to
        ``settings.thresholds.oi_spike_threshold``.

    Returns
    -------
    list[float]:
        Strike prices where OI is anomalously high.
    """
    threshold = multiplier or settings.thresholds.oi_spike_threshold
    mean_oi = chain_df[column].mean()
    spikes = chain_df[chain_df[column] > threshold * mean_oi]["strike"]
    return sorted(spikes.tolist())


def analyse_chain(
    chain_df: pd.DataFrame,
    underlying_value: float,
) -> OIAnalysis:
    """
    Run a full options chain analysis.

    Parameters
    ----------
    chain_df:
        Near-expiry options chain DataFrame (output of
        :meth:`~src.data.options_chain.OptionsChainFetcher.get_near_expiry`).
    underlying_value:
        Current spot price of the underlying index.

    Returns
    -------
    OIAnalysis:
        Comprehensive options analysis result.
    """
    if chain_df.empty:
        raise ValueError("Cannot analyse an empty options chain")

    # ATM strike = strike closest to current underlying
    atm = float(chain_df["strike"].sub(underlying_value).abs().idxmin())
    atm_strike = float(chain_df.loc[atm, "strike"])

    pcr_oi, pcr_volume = calculate_pcr(chain_df)
    max_pain = calculate_max_pain(chain_df)
    ce_spikes = find_oi_spikes(chain_df, "ce_oi")
    pe_spikes = find_oi_spikes(chain_df, "pe_oi")

    thresholds = settings.thresholds
    if pcr_oi > thresholds.pcr_extreme_high:
        sentiment = "bearish"
    elif pcr_oi < thresholds.pcr_extreme_low:
        sentiment = "bullish"
    else:
        sentiment = "neutral"

    return OIAnalysis(
        pcr_oi=pcr_oi,
        pcr_volume=pcr_volume,
        max_pain=max_pain,
        atm_strike=atm_strike,
        ce_oi_spikes=ce_spikes,
        pe_oi_spikes=pe_spikes,
        sentiment=sentiment,
    )


@dataclass
class AdvancedOptionsMetrics:
    """
    Advanced options analytics beyond basic PCR/Max Pain.

    These metrics are critical for understanding market structure and
    strategy selection. GEX determines trending vs rangebound character;
    IV percentile determines whether to buy or sell options.
    """
    # Gamma Exposure (GEX)
    gex: float                  # Net dealer gamma exposure (positive=rangebound, negative=trending)
    gex_regime: str             # "RANGEBOUND" | "TRENDING" | "NEUTRAL"
    gex_flip_level: float       # Price level where GEX flips sign (key S/R)

    # IV metrics
    iv_percentile: float        # Current IV vs last 30 days (0–100)
    iv_rank: float              # (current_iv - 30d_low) / (30d_high - 30d_low)
    atm_iv: float               # ATM strike IV (average of CE+PE)
    iv_strategy_bias: str       # "BUY_OPTIONS" | "SELL_PREMIUM" | "NEUTRAL"

    # ATM Straddle
    atm_straddle_price: float   # ATM CE premium + ATM PE premium
    expected_daily_move: float  # Straddle price as expected daily range (±)
    day_type: str               # "DIRECTIONAL" | "RANGEBOUND" | "UNKNOWN"

    # Options Skew
    put_call_skew: float        # PE ATM IV - CE ATM IV (positive = put skew = bearish hedge)
    skew_regime: str            # "PUT_SKEW" | "CALL_SKEW" | "BALANCED"
    otm_skew_25d: float         # 25-delta PE IV - 25-delta CE IV (deeper OTM skew)

    # Unusual Options Activity (UOA)
    uoa_strikes: list[float]    # Strikes with volume > 5x their average
    uoa_direction: str          # "BULLISH" | "BEARISH" | "MIXED" | "NONE"
    deep_otm_activity: bool     # Unusual deep OTM volume detected

    # Wall detection
    ce_wall_strikes: list[float]  # Massive CE OI strikes (resistance walls)
    pe_wall_strikes: list[float]  # Massive PE OI strikes (support walls)
    nearest_ce_wall: float
    nearest_pe_wall: float


def calculate_gex(
    chain_df: pd.DataFrame,
    spot_price: float,
    lot_size: int = 50,
) -> tuple[float, str, float]:
    """
    Calculate Net Dealer Gamma Exposure (GEX).

    GEX determines whether market makers hedge by dampening moves (positive GEX)
    or amplifying moves (negative GEX). Critical for strategy selection:
    - Positive GEX → expect rangebound day, avoid directional options
    - Negative GEX → expect trending/volatile day, directional options work

    Formula: Σ(CE_OI × CE_gamma - PE_OI × PE_gamma) × spot × lot_size
    Since exact Greeks are rarely available, we approximate gamma using
    ATM proximity (gamma is highest ATM and decays with distance).

    Parameters
    ----------
    chain_df:
        Options chain with columns: strike, ce_oi, pe_oi, ce_iv, pe_iv.
    spot_price:
        Current underlying price.
    lot_size:
        Lot size (default 50 for NIFTY).

    Returns
    -------
    tuple[float, str, float]:
        (gex_value, regime, gex_flip_level)
    """
    if chain_df.empty or spot_price <= 0:
        return 0.0, "NEUTRAL", spot_price

    try:
        import math

        def _approx_gamma(strike: float, spot: float, iv: float, dte_years: float = 0.02) -> float:
            """Approximate gamma using Black-Scholes approximation."""
            if iv <= 0 or dte_years <= 0:
                return 0.0
            moneyness = spot / strike if strike > 0 else 1.0
            log_m = math.log(moneyness)
            vol_sqrt_t = iv / 100 * math.sqrt(dte_years)
            if vol_sqrt_t <= 0:
                return 0.0
            d1 = (log_m + 0.5 * (iv / 100) ** 2 * dte_years) / vol_sqrt_t
            # Standard normal PDF approximation
            gamma = math.exp(-0.5 * d1 ** 2) / (math.sqrt(2 * math.pi) * spot * vol_sqrt_t)
            return gamma

        gex_by_strike: list[tuple[float, float]] = []

        for _, row in chain_df.iterrows():
            strike = float(row.get("strike", 0))
            ce_oi = float(row.get("ce_oi", 0))
            pe_oi = float(row.get("pe_oi", 0))
            ce_iv = float(row.get("ce_iv", row.get("ce_ltp", 0)) or 0)
            pe_iv = float(row.get("pe_iv", row.get("pe_ltp", 0)) or 0)

            if strike <= 0:
                continue

            ce_gamma = _approx_gamma(strike, spot_price, ce_iv if ce_iv > 1 else 15.0)
            pe_gamma = _approx_gamma(strike, spot_price, pe_iv if pe_iv > 1 else 15.0)

            # Dealers are short CE (sold to buyers) → long gamma from CE OI
            # Dealers are short PE (sold to buyers) → long gamma from PE OI
            # Net dealer GEX = (CE_OI - PE_OI) × gamma × spot × lot_size
            net_gamma = (ce_oi * ce_gamma - pe_oi * pe_gamma)
            strike_gex = net_gamma * spot_price * lot_size
            gex_by_strike.append((strike, strike_gex))

        if not gex_by_strike:
            return 0.0, "NEUTRAL", spot_price

        total_gex = sum(g for _, g in gex_by_strike)

        # Find GEX flip level (strike where cumulative GEX changes sign)
        gex_flip = spot_price
        sorted_strikes = sorted(gex_by_strike, key=lambda x: x[0])
        cumulative = 0.0
        for strike, sgex in sorted_strikes:
            prev_cum = cumulative
            cumulative += sgex
            if prev_cum * cumulative < 0:  # sign change
                gex_flip = strike
                break

        # Regime classification
        if total_gex > 1e8:
            regime = "RANGEBOUND"
        elif total_gex < -1e8:
            regime = "TRENDING"
        else:
            regime = "NEUTRAL"

        return round(total_gex / 1e8, 4), regime, gex_flip

    except Exception as e:
        logger.warning("GEX calculation failed: %s", e)
        return 0.0, "NEUTRAL", spot_price


# Minimum number of (daily) IV observations for a meaningful percentile.
MIN_IV_HISTORY_OBSERVATIONS: int = 60


def calculate_iv_percentile(
    current_iv: float,
    historical_ivs: list[float],
) -> tuple[float, float, str]:
    """
    Calculate IV Percentile and IV Rank vs historical distribution.

    Raw IV alone is meaningless without context. IVP/IVR answers:
    "Is IV cheap or expensive relative to its own history?"

    Parameters
    ----------
    current_iv:
        Current ATM IV (as percentage, e.g., 15.0 for 15%).
    historical_ivs:
        List of past IV observations (30-day window recommended).

    Returns
    -------
    tuple[float, float, str]:
        (iv_percentile, iv_rank, strategy_bias)
        - iv_percentile: % of days in history where IV was LOWER than today
        - iv_rank: (current - min) / (max - min) normalized to 0-100
        - strategy_bias: "BUY_OPTIONS" | "SELL_PREMIUM" | "NEUTRAL"
    """
    if not historical_ivs or current_iv <= 0:
        return 50.0, 50.0, "NEUTRAL"

    hist = [v for v in historical_ivs if v > 0]
    # A percentile of a handful of observations is noise, and a strategy
    # bias derived from it moved option-buying confidence.  Below the
    # minimum (daily observations), report neutral.
    if len(hist) < MIN_IV_HISTORY_OBSERVATIONS:
        return 50.0, 50.0, "NEUTRAL"

    # IV Percentile: % of days where IV < current
    iv_percentile = sum(1 for h in hist if h < current_iv) / len(hist) * 100

    # IV Rank: normalized position in [min, max]
    hist_min, hist_max = min(hist), max(hist)
    if hist_max > hist_min:
        iv_rank = (current_iv - hist_min) / (hist_max - hist_min) * 100
    else:
        iv_rank = 50.0

    # Strategy bias
    if iv_percentile <= 25:
        bias = "BUY_OPTIONS"    # IV cheap, below 25th percentile
    elif iv_percentile >= 75:
        bias = "SELL_PREMIUM"   # IV expensive, above 75th percentile
    else:
        bias = "NEUTRAL"

    return round(iv_percentile, 1), round(iv_rank, 1), bias


def calculate_atm_straddle(
    chain_df: pd.DataFrame,
    spot_price: float,
    strike_interval: float = 50.0,
) -> tuple[float, float, str]:
    """
    Calculate ATM straddle price and expected daily move.

    The straddle price (CE premium + PE premium at ATM) represents the
    market's implied expected move for the expiry. Used to classify
    whether today is expected to be directional or rangebound.

    Parameters
    ----------
    chain_df:
        Options chain with: strike, ce_ltp, pe_ltp.
    spot_price:
        Current underlying price.
    strike_interval:
        Strike spacing (50 for NIFTY, 100 for BANKNIFTY).

    Returns
    -------
    tuple[float, float, str]:
        (straddle_price, expected_move_pct, day_type)
    """
    if chain_df.empty or spot_price <= 0:
        return 0.0, 0.0, "UNKNOWN"

    try:
        # Find ATM strike
        atm = round(spot_price / strike_interval) * strike_interval
        atm_row = chain_df[chain_df["strike"].sub(atm).abs() < strike_interval / 2]

        if atm_row.empty:
            # Fallback: closest strike
            idx = chain_df["strike"].sub(spot_price).abs().idxmin()
            atm_row = chain_df.iloc[[idx]]

        if atm_row.empty:
            return 0.0, 0.0, "UNKNOWN"

        row = atm_row.iloc[0]
        ce_prem = float(row.get("ce_ltp", 0) or 0)
        pe_prem = float(row.get("pe_ltp", 0) or 0)
        straddle = ce_prem + pe_prem

        if straddle <= 0:
            return 0.0, 0.0, "UNKNOWN"

        # Expected move as % of spot
        expected_move_pct = straddle / spot_price * 100

        # Day type: if price has already moved beyond straddle, it's directional
        # At open (no move yet), use straddle to predict
        day_type = "UNKNOWN"
        if expected_move_pct > 1.5:
            day_type = "DIRECTIONAL"
        elif expected_move_pct < 0.6:
            day_type = "RANGEBOUND"
        else:
            day_type = "MODERATE"

        return round(straddle, 2), round(expected_move_pct, 3), day_type

    except Exception as e:
        logger.warning("ATM straddle calculation failed: %s", e)
        return 0.0, 0.0, "UNKNOWN"


def calculate_options_skew(
    chain_df: pd.DataFrame,
    spot_price: float,
    strike_interval: float = 50.0,
) -> tuple[float, float, str]:
    """
    Calculate put-call IV skew.

    High put skew = institutional bearish hedging (negative for market).
    Rising call skew = unusual bullish positioning.

    Parameters
    ----------
    chain_df:
        Options chain with: strike, ce_iv, pe_iv.
    spot_price:
        Current underlying price.
    strike_interval:
        Strike spacing.

    Returns
    -------
    tuple[float, float, str]:
        (atm_skew, otm_skew_25d, regime)
        - atm_skew: PE ATM IV - CE ATM IV (positive = put skew)
        - otm_skew_25d: OTM put IV - OTM call IV (deeper skew)
        - regime: "PUT_SKEW" | "CALL_SKEW" | "BALANCED"
    """
    if chain_df.empty or spot_price <= 0:
        return 0.0, 0.0, "BALANCED"

    try:
        atm = round(spot_price / strike_interval) * strike_interval

        # Find ATM row
        def _find_iv(target_strike: float) -> tuple[float, float]:
            row_df = chain_df[chain_df["strike"].sub(target_strike).abs() < strike_interval * 1.5]
            if row_df.empty:
                return 0.0, 0.0
            row = row_df.iloc[0]
            ce_iv = float(row.get("ce_iv", 0) or 0)
            pe_iv = float(row.get("pe_iv", 0) or 0)
            return ce_iv, pe_iv

        atm_ce_iv, atm_pe_iv = _find_iv(atm)
        atm_skew = atm_pe_iv - atm_ce_iv

        # OTM skew: ~25 delta approx = 2 strikes OTM
        otm_call_strike = atm + 2 * strike_interval
        otm_put_strike = atm - 2 * strike_interval
        otm_ce_iv, _ = _find_iv(otm_call_strike)
        _, otm_pe_iv = _find_iv(otm_put_strike)
        otm_skew = otm_pe_iv - otm_ce_iv

        # Classify
        if atm_skew > 2.0:
            regime = "PUT_SKEW"
        elif atm_skew < -2.0:
            regime = "CALL_SKEW"
        else:
            regime = "BALANCED"

        return round(atm_skew, 2), round(otm_skew, 2), regime

    except Exception as e:
        logger.warning("Options skew calculation failed: %s", e)
        return 0.0, 0.0, "BALANCED"


def detect_unusual_options_activity(
    chain_df: pd.DataFrame,
    spot_price: float,
    volume_spike_multiplier: float = 5.0,
) -> tuple[list[float], str, bool]:
    """
    Detect unusual options activity — volume spikes vs average.

    Institutional positioning often shows up as unusual OTM volume before
    a large move. A spike to 5x+ average volume at a strike is a signal.

    Parameters
    ----------
    chain_df:
        Options chain with: strike, ce_volume, pe_volume.
    spot_price:
        Current underlying price for OTM classification.
    volume_spike_multiplier:
        Threshold multiplier above average (default 5x).

    Returns
    -------
    tuple[list[float], str, bool]:
        (uoa_strikes, direction, deep_otm_activity)
    """
    if chain_df.empty:
        return [], "NONE", False

    try:
        ce_vol_col = "ce_volume" if "ce_volume" in chain_df.columns else None
        pe_vol_col = "pe_volume" if "pe_volume" in chain_df.columns else None

        if not ce_vol_col and not pe_vol_col:
            return [], "NONE", False

        uoa_strikes = []
        bullish_uoa = 0
        bearish_uoa = 0
        deep_otm_activity = False

        if ce_vol_col:
            avg_ce_vol = chain_df[ce_vol_col].replace(0, np.nan).mean()
            if avg_ce_vol and avg_ce_vol > 0:
                ce_spikes = chain_df[chain_df[ce_vol_col] > volume_spike_multiplier * avg_ce_vol]
                for _, row in ce_spikes.iterrows():
                    strike = float(row["strike"])
                    uoa_strikes.append(strike)
                    if strike > spot_price:
                        bullish_uoa += 1
                        if strike > spot_price * 1.03:
                            deep_otm_activity = True
                    else:
                        bearish_uoa += 1

        if pe_vol_col:
            avg_pe_vol = chain_df[pe_vol_col].replace(0, np.nan).mean()
            if avg_pe_vol and avg_pe_vol > 0:
                pe_spikes = chain_df[chain_df[pe_vol_col] > volume_spike_multiplier * avg_pe_vol]
                for _, row in pe_spikes.iterrows():
                    strike = float(row["strike"])
                    if strike not in uoa_strikes:
                        uoa_strikes.append(strike)
                    if strike < spot_price:
                        bearish_uoa += 1
                        if strike < spot_price * 0.97:
                            deep_otm_activity = True
                    else:
                        bullish_uoa += 1

        if bullish_uoa > bearish_uoa:
            direction = "BULLISH"
        elif bearish_uoa > bullish_uoa:
            direction = "BEARISH"
        elif uoa_strikes:
            direction = "MIXED"
        else:
            direction = "NONE"

        return sorted(uoa_strikes), direction, deep_otm_activity

    except Exception as e:
        logger.warning("UOA detection failed: %s", e)
        return [], "NONE", False


def detect_oi_walls(
    chain_df: pd.DataFrame,
    spot_price: float,
    wall_multiplier: float = 2.5,
) -> tuple[list[float], list[float], float, float]:
    """
    Detect OI walls — strikes with disproportionately large OI.

    A strike with OI > 2.5x the chain average is a significant support/resistance wall.
    Price approaching such a wall with premium compression = wall holding.
    Fast breach with volume spike = wall breaking = strong momentum signal.

    Returns
    -------
    tuple[list[float], list[float], float, float]:
        (ce_wall_strikes, pe_wall_strikes, nearest_ce_wall, nearest_pe_wall)
    """
    if chain_df.empty or spot_price <= 0:
        return [], [], 0.0, 0.0

    try:
        avg_ce_oi = chain_df["ce_oi"].replace(0, np.nan).mean() or 1
        avg_pe_oi = chain_df["pe_oi"].replace(0, np.nan).mean() or 1

        ce_walls = sorted(
            chain_df[chain_df["ce_oi"] > wall_multiplier * avg_ce_oi]["strike"].tolist()
        )
        pe_walls = sorted(
            chain_df[chain_df["pe_oi"] > wall_multiplier * avg_pe_oi]["strike"].tolist()
        )

        # Find nearest walls above/below spot
        ce_above = [s for s in ce_walls if s >= spot_price]
        pe_below = [s for s in pe_walls if s <= spot_price]

        nearest_ce = min(ce_above) if ce_above else 0.0
        nearest_pe = max(pe_below) if pe_below else 0.0

        return ce_walls, pe_walls, nearest_ce, nearest_pe

    except Exception as e:
        logger.warning("OI wall detection failed: %s", e)
        return [], [], 0.0, 0.0


def run_advanced_options_analytics(
    chain_df: pd.DataFrame,
    spot_price: float,
    lot_size: int = 50,
    strike_interval: float = 50.0,
    historical_ivs: Optional[list[float]] = None,
) -> AdvancedOptionsMetrics:
    """
    Run all advanced options analytics in one call.

    Parameters
    ----------
    chain_df:
        Full options chain DataFrame.
    spot_price:
        Current underlying spot price.
    lot_size:
        Lot size for GEX calculation.
    strike_interval:
        Strike interval for ATM identification.
    historical_ivs:
        30-day historical ATM IV list for percentile calculation.

    Returns
    -------
    AdvancedOptionsMetrics
    """
    # GEX
    gex, gex_regime, gex_flip = calculate_gex(chain_df, spot_price, lot_size)

    # ATM straddle
    straddle, expected_move_pct, day_type = calculate_atm_straddle(
        chain_df, spot_price, strike_interval
    )

    # ATM IV
    atm = round(spot_price / strike_interval) * strike_interval
    atm_row_df = chain_df[chain_df["strike"].sub(atm).abs() < strike_interval * 1.5]
    atm_iv = 0.0
    if not atm_row_df.empty:
        r = atm_row_df.iloc[0]
        ce_iv = float(r.get("ce_iv", 0) or 0)
        pe_iv = float(r.get("pe_iv", 0) or 0)
        atm_iv = (ce_iv + pe_iv) / 2 if (ce_iv or pe_iv) else 0.0

    # IV percentile
    ivp, ivr, iv_bias = calculate_iv_percentile(atm_iv, historical_ivs or [])

    # Options skew
    atm_skew, otm_skew, skew_regime = calculate_options_skew(chain_df, spot_price, strike_interval)

    # UOA
    uoa_strikes, uoa_dir, deep_otm = detect_unusual_options_activity(chain_df, spot_price)

    # OI walls
    ce_walls, pe_walls, nearest_ce, nearest_pe = detect_oi_walls(chain_df, spot_price)

    return AdvancedOptionsMetrics(
        gex=gex,
        gex_regime=gex_regime,
        gex_flip_level=gex_flip,
        iv_percentile=ivp,
        iv_rank=ivr,
        atm_iv=atm_iv,
        iv_strategy_bias=iv_bias,
        atm_straddle_price=straddle,
        expected_daily_move=expected_move_pct,
        day_type=day_type,
        put_call_skew=atm_skew,
        skew_regime=skew_regime,
        otm_skew_25d=otm_skew,
        uoa_strikes=uoa_strikes,
        uoa_direction=uoa_dir,
        deep_otm_activity=deep_otm,
        ce_wall_strikes=ce_walls,
        pe_wall_strikes=pe_walls,
        nearest_ce_wall=nearest_ce,
        nearest_pe_wall=nearest_pe,
    )
