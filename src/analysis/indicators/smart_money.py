"""
Custom "Smart Money" indicators for institutional activity detection.

Implements Smart Money Flow Index (SMFI), Volume Shock Detector (VSD),
Breakout Trap Detector (BTD), OI Momentum Index (OIMI), and Liquidity
Absorption Index (LAI). Includes a composite Smart Money Confidence Score.
All methods are pure functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

import numpy as np
import pandas as pd

from src.analysis.indicators.momentum import _detect_divergence
from src.analysis.indicators.volatility import VolatilityIndicators

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SMFIResult:
    """Result of Smart Money Flow Index calculation.
    
    SMFI detects whether institutional players are accumulating or distributing
    by comparing end-of-day positioning (smart money) with open-of-day
    positioning (emotional/retail money).
    """

    smfi: pd.Series               # SMFI values (0-100)
    current_value: float
    signal: str                   # "ACCUMULATION" / "DISTRIBUTION" / "NEUTRAL"
    trend: str                    # "RISING" / "FALLING" / "FLAT"
    divergence: Optional[str]     # "BULLISH_DIV" / "BEARISH_DIV" / None
    days_in_accumulation: int     # Consecutive bars with SMFI > 60
    days_in_distribution: int     # Consecutive bars with SMFI < 40


@dataclass
class VolumeShock:
    """Details of a single detected volume shock."""
    timestamp: datetime
    volume: int
    volume_zscore: float
    price_change_pct: float
    shock_type: str           # "BULLISH_SHOCK" / "BEARISH_SHOCK" / "ABSORPTION"
    significance: str         # "EXTREME" (z>4) / "HIGH" (z>3) / "MODERATE" (z>2.5)


@dataclass
class VSDResult:
    """Result of Volume Shock Detector."""
    shocks: list[VolumeShock]     # List of detected shocks
    recent_shocks_count: int      # Shocks in last 10 bars
    dominant_shock_type: str      # Most common shock type in last 10 bars
    current_bar_shock: bool       # Is latest bar a shock?
    is_climax: bool               # 2+ shocks in 3 bars = potential climax


@dataclass
class BTDResult:
    """Result of Breakout Trap Detector."""
    breakout_detected: bool       # Did a breakout happen in last 5 bars?
    breakout_direction: Optional[str] # "UP" / "DOWN"
    breakout_quality_score: int   # 0-10 score
    is_trap: bool                 # quality_score <= 3
    is_genuine: bool              # quality_score >= 7
    trap_risk: str                # "HIGH" / "MEDIUM" / "LOW"
    recommendation: str           # "AVOID" / "CAUTION_WAIT_FOR_RETEST" / "VALID_BREAKOUT"


@dataclass
class OIMIResult:
    """Result of OI Momentum Index calculation."""
    oimi: float                   # Current OIMI value (-100 to +100)
    oimi_series: pd.Series        # Historical OIMI
    signal: str                   # "STRONG_BULLISH" / "BULLISH" / "NEUTRAL" / "BEARISH" / "STRONG_BEARISH"
    price_oi_alignment: str       # "ALIGNED" / "DIVERGING"
    momentum_shift: bool          # OIMI crossed 0 in last 3 bars


@dataclass
class LAIResult:
    """Result of Liquidity Absorption Index calculation."""
    lai: pd.Series                # LAI values per bar
    current_lai: float
    absorption_detected: bool     # LAI > 2.0 on latest bar
    absorption_type: Optional[str]   # "BULLISH_ABSORPTION" / "BEARISH_ABSORPTION" / None
    streak: int                   # Consecutive bars with absorption > 1.5
    historical_percentile: float  # Where current LAI ranks (0-100)


@dataclass
class SmartMoneyScore:
    """Composite score combining all institutional indicators."""
    score: float                  # -100 to +100
    grade: str                    # "A+" to "F--"
    
    # Component scores
    smfi_component: float         # Weight: 25%
    vsd_component: float          # Weight: 15%
    btd_component: float          # Weight: 10%
    oimi_component: float         # Weight: 30%
    lai_component: float          # Weight: 20%
    
    # Interpretation
    smart_money_bias: str         # "STRONGLY_BULLISH" to "STRONGLY_BEARISH"
    key_finding: str
    actionable_insight: str
    
    # Confidence & reliability
    data_completeness: float      # 0-1
    confidence: float             # 0-1


# ---------------------------------------------------------------------------
# Helper utilities (internal)
# ---------------------------------------------------------------------------


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standard OHLCV preparation."""
    if df.empty:
        return df.copy()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    
    clean = df.copy()
    for col in required:
        if clean[col].isna().any():
            clean[col] = clean[col].ffill()
    return clean


def _detect_slope(series: pd.Series, lookback: int = 5) -> str:
    """Classify the slope of the last *lookback* values."""
    valid = series.dropna()
    if len(valid) < lookback:
        return "FLAT"
    tail = valid.iloc[-lookback:].values.astype(float)
    x = np.arange(len(tail))
    slope = np.polyfit(x, tail, 1)[0]
    
    # Threshold for flatness. 
    # For 0-100 indicators, a slope of 0.5 per bar is significant.
    if slope > 0.5:
        return "RISING"
    if slope < -0.5:
        return "FALLING"
    return "FLAT"


def _rolling_min_max_normalize(series: pd.Series, period: int = 50) -> pd.Series:
    """Normalize a series to 0-100 range using rolling min-max."""
    rolling_min = series.rolling(window=period, min_periods=1).min()
    rolling_max = series.rolling(window=period, min_periods=1).max()
    
    normalized = 100 * (series - rolling_min) / (rolling_max - rolling_min).replace(0, np.nan)
    return normalized.fillna(50)


# ---------------------------------------------------------------------------
# SmartMoneyIndicators
# ---------------------------------------------------------------------------


class SmartMoneyIndicators:
    """Custom institutional activity detection class."""

    def __init__(self):
        self.vol_indicators = VolatilityIndicators()

    def calculate_smfi(self, df: pd.DataFrame, period: int = 50) -> SMFIResult:
        """Calculate Smart Money Flow Index (SMFI)."""
        clean = _prepare_df(df)
        if clean.empty or len(clean) < 2:
            return SMFIResult(
                smfi=pd.Series(dtype=float),
                current_value=50.0,
                signal="NEUTRAL",
                trend="FLAT",
                divergence=None,
                days_in_accumulation=0,
                days_in_distribution=0
            )

        midpoint = (clean["high"] + clean["low"]) / 2
        first_half_move = midpoint - clean["open"]
        second_half_move = clean["close"] - midpoint
        
        avg_vol_20 = clean["volume"].rolling(window=20, min_periods=1).mean()
        volume_factor = clean["volume"] / avg_vol_20.replace(0, np.nan)
        volume_factor = volume_factor.fillna(1.0)
        
        smfi_delta = (second_half_move - first_half_move) * volume_factor
        raw_smfi = smfi_delta.cumsum()
        
        # Trend - detect on raw_smfi before normalization to avoid "flat-topping"
        trend = _detect_slope(raw_smfi, lookback=5)
        
        smfi_series = _rolling_min_max_normalize(raw_smfi, period=period)
        current_value = float(smfi_series.iloc[-1])
        
        # Signal
        if current_value > 70:
            signal = "ACCUMULATION"
        elif current_value < 30:
            signal = "DISTRIBUTION"
        else:
            signal = "NEUTRAL"
        
        # Divergence
        div = _detect_divergence(clean["close"], smfi_series, lookback=14)
        divergence = None
        if div == "BULLISH_DIVERGENCE":
            divergence = "BULLISH_DIV"
        elif div == "BEARISH_DIVERGENCE":
            divergence = "BEARISH_DIV"
            
        # Consecutive days
        acc_mask = smfi_series > 60
        dist_mask = smfi_series < 40
        
        days_in_acc = 0
        for val in reversed(acc_mask.values):
            if val: days_in_acc += 1
            else: break
            
        days_in_dist = 0
        for val in reversed(dist_mask.values):
            if val: days_in_dist += 1
            else: break
            
        return SMFIResult(
            smfi=smfi_series,
            current_value=round(current_value, 2),
            signal=signal,
            trend=trend,
            divergence=divergence,
            days_in_accumulation=days_in_acc,
            days_in_distribution=days_in_dist
        )

    def detect_volume_shocks(self, df: pd.DataFrame, threshold_zscore: float = 2.5) -> VSDResult:
        """Flag abnormal volume spikes indicating institutional activity."""
        clean = _prepare_df(df)
        if clean.empty or len(clean) < 20:
            return VSDResult(shocks=[], recent_shocks_count=0, dominant_shock_type="NONE", current_bar_shock=False, is_climax=False)
            
        vol = clean["volume"].astype(float)
        mean_vol = vol.rolling(window=20).mean()
        std_vol = vol.rolling(window=20).std(ddof=1)
        
        z_scores = (vol - mean_vol) / std_vol.replace(0, np.nan)
        z_scores = z_scores.fillna(0)
        
        shocks = []
        timestamps = clean.index if not "timestamp" in clean.columns else pd.to_datetime(clean["timestamp"])
        
        # Detect all shocks
        for i in range(len(clean)):
            z = float(z_scores.iloc[i])
            if z > threshold_zscore:
                price_change = ((clean["close"].iloc[i] - clean["close"].iloc[i-1]) / clean["close"].iloc[i-1] * 100) if i > 0 else 0.0
                
                # Classify shock type
                # Price movement relative to open is better for single bar shock
                bar_move = (clean["close"].iloc[i] - clean["open"].iloc[i]) / clean["open"].iloc[i] * 100
                if bar_move > 0.3:
                    type_ = "BULLISH_SHOCK"
                elif bar_move < -0.3:
                    type_ = "BEARISH_SHOCK"
                else:
                    type_ = "ABSORPTION"
                    
                # Significance
                if z > 4: sig = "EXTREME"
                elif z > 3: sig = "HIGH"
                else: sig = "MODERATE"
                
                shocks.append(VolumeShock(
                    timestamp=timestamps[i] if hasattr(timestamps[i], "to_pydatetime") else timestamps[i],
                    volume=int(vol.iloc[i]),
                    volume_zscore=round(z, 2),
                    price_change_pct=round(price_change, 2),
                    shock_type=type_,
                    significance=sig
                ))
        
        # Recent analysis
        recent_shocks = [s for s in shocks if s.timestamp in timestamps[-10:].values]
        
        count = len(recent_shocks)
        
        types = [s.shock_type for s in recent_shocks]
        dominant = max(set(types), key=types.count) if types else "NONE"
        
        current_shock = float(z_scores.iloc[-1]) > threshold_zscore
        
        # Climax detection: 2+ shocks in last 3 bars
        climax_window = z_scores.iloc[-3:]
        is_climax = (climax_window > threshold_zscore).sum() >= 2
        
        return VSDResult(
            shocks=shocks,
            recent_shocks_count=count,
            dominant_shock_type=dominant,
            current_bar_shock=current_shock,
            is_climax=is_climax
        )

    def detect_breakout_traps(self, df: pd.DataFrame, support: float, resistance: float, oi_data: Optional[pd.DataFrame] = None, rsi_data: Optional[pd.Series] = None) -> BTDResult:
        """Identify fake breakouts that trap retail traders."""
        clean = _prepare_df(df)
        if clean.empty or len(clean) < 5:
            return BTDResult(breakout_detected=False, breakout_direction=None, breakout_quality_score=0, is_trap=False, is_genuine=False, trap_risk="MEDIUM", recommendation="CAUTION_WAIT_FOR_RETEST")

        # 1. Did a breakout happen in last 5 bars?
        lookback = clean.iloc[-5:]
        break_up = lookback["close"] > resistance
        break_down = lookback["close"] < support
        
        any_up = break_up.any()
        any_down = break_down.any()
        
        if not any_up and not any_down:
            return BTDResult(breakout_detected=False, breakout_direction=None, breakout_quality_score=0, is_trap=False, is_genuine=False, trap_risk="MEDIUM", recommendation="CAUTION_WAIT_FOR_RETEST")
            
        breakout_detected = True
        # Take the most recent breakout if both happened (rare)
        if break_up.iloc[-1]: direction = "UP"
        elif break_down.iloc[-1]: direction = "DOWN"
        else: direction = "UP" if any_up else "DOWN"
        
        # Find the breakout bar (the first one to cross in the last 5)
        if direction == "UP":
            cross_indices = lookback.index[lookback["close"] > resistance]
        else:
            cross_indices = lookback.index[lookback["close"] < support]
        
        breakout_idx = cross_indices[0]
        breakout_bar = clean.loc[breakout_idx]
        
        # Calculate quality score
        score = 0
        
        # Volume check
        avg_vol = clean["volume"].rolling(window=20).mean().loc[breakout_idx]
        vol_ratio = breakout_bar["volume"] / avg_vol if avg_vol > 0 else 1.0
        
        if vol_ratio > 1.5: score += 3
        elif vol_ratio >= 1.2: score += 2
        elif vol_ratio < 1.0: score -= 2
        
        # OI check (if available)
        if oi_data is not None and not oi_data.empty:
            # Check OI change for the breakout bar
            # Logic: Bullish breakout + Rising OI = Strong positioning
            try:
                oi_change = oi_data.loc[breakout_idx, "oi_change"]
                if (direction == "UP" and oi_change > 0) or (direction == "DOWN" and oi_change > 0):
                    score += 2
                elif oi_change < -5: # Massive unwinding
                    score -= 2
            except: pass
            
        # Closed beyond level (not just a wick)
        # We already check 'close' for breakout, let's see how deep it is
        body_size = abs(breakout_bar["close"] - breakout_bar["open"])
        if direction == "UP":
            depth = breakout_bar["close"] - resistance
            confirmed = breakout_bar["close"] > breakout_bar["open"] and depth > 0.5 * body_size
        else:
            depth = support - breakout_bar["close"]
            confirmed = breakout_bar["close"] < breakout_bar["open"] and depth > 0.5 * body_size
            
        if confirmed: score += 2
        
        # RSI check
        if rsi_data is not None:
            last_rsi = rsi_data.loc[breakout_idx]
            if (direction == "UP" and last_rsi > 70) or (direction == "DOWN" and last_rsi < 30):
                score -= 1 # Exhaustion breakout
            else:
                score += 1
                
        # Immediate reversal (only possible if breakout was not the current bar)
        if len(clean) > clean.index.get_loc(breakout_idx) + 1:
            next_idx = clean.index[clean.index.get_loc(breakout_idx) + 1]
            next_close = clean.loc[next_idx, "close"]
            if direction == "UP" and next_close < resistance:
                score -= 2
            elif direction == "DOWN" and next_close > support:
                score -= 2
                
        # Result
        is_trap = score <= 3
        is_genuine = score >= 7
        
        if is_trap:
            risk, reco = "HIGH", "AVOID"
        elif is_genuine:
            risk, reco = "LOW", "VALID_BREAKOUT"
        else:
            risk, reco = "MEDIUM", "CAUTION_WAIT_FOR_RETEST"
            
        return BTDResult(
            breakout_detected=True,
            breakout_direction=direction,
            breakout_quality_score=score,
            is_trap=is_trap,
            is_genuine=is_genuine,
            trap_risk=risk,
            recommendation=reco
        )

    def calculate_oimi(self, price_df: pd.DataFrame, oi_history: list[dict]) -> OIMIResult:
        """Combine price momentum with OI momentum to detect real positioning."""
        if not oi_history or len(oi_history) < 5:
            return OIMIResult(oimi=0.0, oimi_series=pd.Series(dtype=float), signal="NEUTRAL", price_oi_alignment="DIVERGING", momentum_shift=False)
            
        oi_df = pd.DataFrame(oi_history)
        oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"])
        oi_df = oi_df.sort_values("timestamp").set_index("timestamp")
        
        # Aggregate total OI from keys if they exist
        if "total_ce_oi" in oi_df.columns and "total_pe_oi" in oi_df.columns:
            oi_df["total_oi"] = oi_df["total_ce_oi"] + oi_df["total_pe_oi"]
        elif "total_oi" not in oi_df.columns:
             # Fallback
             return OIMIResult(oimi=0.0, oimi_series=pd.Series(dtype=float), signal="NEUTRAL", price_oi_alignment="DIVERGING", momentum_shift=False)

        # Merge with price data to align timestamps
        price_copy = price_df.copy()
        if "timestamp" in price_copy.columns:
            price_copy["timestamp"] = pd.to_datetime(price_copy["timestamp"])
            price_copy = price_copy.set_index("timestamp")
            
        merged = pd.merge_asof(price_copy[["close"]], oi_df[["total_oi"]], left_index=True, right_index=True, direction="backward")
        merged = merged.ffill().dropna()
        
        if len(merged) < 6:
             return OIMIResult(oimi=0.0, oimi_series=pd.Series(dtype=float), signal="NEUTRAL", price_oi_alignment="DIVERGING", momentum_shift=False)
             
        # Rate of Change over 5 bars (%)
        price_roc = merged["close"].pct_change(5) * 100
        oi_roc = merged["total_oi"].pct_change(5) * 100
        
        oimi_series = (price_roc * 0.4) + (oi_roc * 0.6)
        
        # Normalize to -100 to +100
        # ROC can be small, so we scale it.
        # Max OI ROC is typically around 5-10% in a 5-bar move.
        oimi_scaled = oimi_series * 10 # Map ±10 to ±100
        oimi_scaled = np.clip(oimi_scaled, -100, 100)
        
        current_oimi = float(oimi_scaled.iloc[-1])
        
        # Signal
        if current_oimi > 50: signal = "STRONG_BULLISH"
        elif current_oimi > 20: signal = "BULLISH"
        elif current_oimi < -50: signal = "STRONG_BEARISH"
        elif current_oimi < -20: signal = "BEARISH"
        else: signal = "NEUTRAL"
        
        # Alignment
        aligned = (price_roc.iloc[-1] > 0 and oi_roc.iloc[-1] > 0) or (price_roc.iloc[-1] < 0 and oi_roc.iloc[-1] > 0)
        # Wait, prompt says: "ALIGNED (both moving same direction)". 
        # Price Up + OI Up = New Longs (Bullish)
        # Price Down + OI Up = New Shorts (Bearish) -> Moving in opposite price direction but both are momentum?
        # Actually standard definition: Aligned = Price and OI both UP (Bullish) or Price and OI both DOWN (Long Unwinding)
        # Let's stick to: Aligned = both ROC same sign
        p_sign = np.sign(price_roc.iloc[-1])
        o_sign = np.sign(oi_roc.iloc[-1])
        alignment = "ALIGNED" if p_sign == o_sign else "DIVERGING"
        
        # Momentum shift
        if len(oimi_scaled) >= 4:
            crossed = (oimi_scaled.iloc[-4:-1] < 0).any() and oimi_scaled.iloc[-1] > 0
            crossed = crossed or ((oimi_scaled.iloc[-4:-1] > 0).any() and oimi_scaled.iloc[-1] < 0)
        else:
            crossed = False
            
        return OIMIResult(
            oimi=round(current_oimi, 2),
            oimi_series=oimi_scaled,
            signal=signal,
            price_oi_alignment=alignment,
            momentum_shift=crossed
        )

    def calculate_lai(self, df: pd.DataFrame, atr_period: int = 14) -> LAIResult:
        """Detect when large players absorb orders without moving price much."""
        clean = _prepare_df(df)
        if clean.empty or len(clean) < atr_period + 1:
            return LAIResult(lai=pd.Series(dtype=float), current_lai=0.0, absorption_detected=False, absorption_type=None, streak=0, historical_percentile=50.0)

        atr_res = self.vol_indicators.calculate_atr(clean, period=atr_period)
        atr_series = atr_res.atr
        
        actual_move = (clean["close"] - clean["open"]).abs()
        # Ensure actual_move is not 0
        actual_move = np.where(actual_move == 0, 0.0001 * clean["close"], actual_move)
        actual_move_series = pd.Series(actual_move, index=clean.index)
        
        # Expected move is baseline ATR. 
        # If volume is high, expected move should be even higher.
        avg_vol = clean["volume"].rolling(window=20).mean()
        vol_factor = clean["volume"] / avg_vol.replace(0, np.nan)
        vol_factor = vol_factor.fillna(1.0)
        
        expected_move = atr_series * vol_factor
        lai_series = expected_move / actual_move_series
        
        current_lai = float(lai_series.iloc[-1])
        detected = current_lai > 2.0 and vol_factor.iloc[-1] > 1.2
        
        # Absorption type
        # Counter-intuitive: absorption is AGAINST the visible move
        # if close < open slightly despite high volume → bullish absorption (players buying into weakness)
        # if close > open slightly despite high volume → bearish absorption (players selling into strength)
        abs_type = None
        if detected:
            if clean["close"].iloc[-1] < clean["open"].iloc[-1]:
                abs_type = "BULLISH_ABSORPTION"
            else:
                abs_type = "BEARISH_ABSORPTION"
                
        # Streak
        streak = 0
        for val in reversed(lai_series.values):
            if val > 1.5: streak += 1
            else: break
            
        # Historical percentile
        window = lai_series.iloc[-252:] if len(lai_series) >= 252 else lai_series
        rank = (window < current_lai).sum() / len(window) * 100
        
        return LAIResult(
            lai=lai_series,
            current_lai=round(current_lai, 2),
            absorption_detected=detected,
            absorption_type=abs_type,
            streak=streak,
            historical_percentile=round(rank, 2)
        )

    def calculate_smart_money_score(self, df: pd.DataFrame, oi_data: Optional[list[dict]] = None, support: Optional[float] = None, resistance: Optional[float] = None, rsi_data: Optional[pd.Series] = None, fii_futures_net: Optional[float] = None) -> SmartMoneyScore:
        """Combine ALL custom indicators into one actionable score."""
        clean = _prepare_df(df)
        
        # 1. Calculate sub-indicators
        smfi_res = self.calculate_smfi(clean)
        vsd_res = self.detect_volume_shocks(clean)
        
        # BTD (needs support/resistance)
        if support is not None and resistance is not None:
            # We need to bridge oi_data to BTD context if possible
            btd_oi = pd.DataFrame(oi_data) if oi_data else None
            if btd_oi is not None and not btd_oi.empty:
                btd_oi["timestamp"] = pd.to_datetime(btd_oi["timestamp"])
                btd_oi = btd_oi.set_index("timestamp")
                if "total_oi" in btd_oi.columns:
                    btd_oi["oi_change"] = btd_oi["total_oi"].diff()
            
            btd_res = self.detect_breakout_traps(clean, support, resistance, oi_data=btd_oi, rsi_data=rsi_data)
        else:
            btd_res = None
            
        # OIMI
        if oi_data and len(oi_data) >= 5:
            oimi_res = self.calculate_oimi(clean, oi_data)
        else:
            oimi_res = None
            
        # LAI
        lai_res = self.calculate_lai(clean)
        
        # 2. Map components to -100 to +100
        # SMFI
        if smfi_res.signal == "ACCUMULATION": smfi_comp = 80
        elif smfi_res.signal == "DISTRIBUTION": smfi_comp = -80
        else: smfi_comp = 0
        
        # VSD
        if vsd_res.current_bar_shock:
            if vsd_res.shocks[-1].shock_type == "BULLISH_SHOCK": vsd_comp = 60
            elif vsd_res.shocks[-1].shock_type == "BEARISH_SHOCK": vsd_comp = -60
            else: # ABSORPTION
                vsd_comp = 30 if clean["close"].iloc[-1] < clean["open"].iloc[-1] else -30
        else:
            # Look at recent dominant
            if vsd_res.dominant_shock_type == "BULLISH_SHOCK": vsd_comp = 30
            elif vsd_res.dominant_shock_type == "BEARISH_SHOCK": vsd_comp = -30
            else: vsd_comp = 0
            
        # BTD
        if btd_res and btd_res.breakout_detected:
            if btd_res.is_genuine:
                btd_comp = 50 if btd_res.breakout_direction == "UP" else -50
            elif btd_res.is_trap:
                # Trap at top (UP breakout) = Bearish (-50)
                # Trap at bottom (DOWN breakout) = Bullish (+50)
                btd_comp = -50 if btd_res.breakout_direction == "UP" else 50
            else: btd_comp = 0
        else: btd_comp = 0
        
        # OIMI — base score from internal OI/price data
        oimi_comp = oimi_res.oimi if oimi_res else 0.0

        # FII futures net position boost — the most reliable institutional signal
        # FII is the dominant smart money in Indian markets. Net long > 10000 contracts
        # = strong institutional conviction. Adjusts OIMI toward their positioning.
        if fii_futures_net is not None and abs(fii_futures_net) >= 5000:
            # Scale: 5000 contracts = ±15 pts, 20000 contracts = ±40 pts (capped)
            fii_boost = min(40.0, abs(fii_futures_net) / 500.0) * (1 if fii_futures_net > 0 else -1)
            oimi_comp = max(-100.0, min(100.0, oimi_comp + fii_boost))
        
        # LAI
        if lai_res.absorption_detected:
            lai_comp = 40 if lai_res.absorption_type == "BULLISH_ABSORPTION" else -40
        else: lai_comp = 0
        
        # 3. Apply weights
        # Default Weights: SMFI: 25%, VSD: 15%, BTD: 10%, OIMI: 30%, LAI: 20%
        w_smfi, w_vsd, w_btd, w_oimi, w_lai = 0.25, 0.15, 0.10, 0.30, 0.20
        
        data_count = 0
        if not smfi_res.smfi.empty: data_count += 1
        if vsd_res.shocks or vsd_res.current_bar_shock: data_count += 1
        if btd_res and btd_res.breakout_detected: data_count += 1
        if oimi_res and not oimi_res.oimi_series.empty: data_count += 1
        if not lai_res.lai.empty: data_count += 1
        if fii_futures_net is not None: data_count += 1
        
        # Adjust weights if data is missing
        if oimi_res is None or oimi_res.oimi_series.empty:
            # Increase SMFI weight as per requirement
            w_smfi += (w_oimi * 0.5)
            w_lai += (w_oimi * 0.5)
            w_oimi = 0
            
        if btd_res is None or not btd_res.breakout_detected:
            w_smfi += (w_btd * 0.5)
            w_vsd += (w_btd * 0.5)
            w_btd = 0
            
        # Recalculate total weight to ensure it sums to 1.0
        total_w = w_smfi + w_vsd + w_btd + w_oimi + w_lai
        if total_w > 0:
            w_smfi /= total_w
            w_vsd /= total_w
            w_btd /= total_w
            w_oimi /= total_w
            w_lai /= total_w
            
        final_score = (smfi_comp * w_smfi) + (vsd_comp * w_vsd) + (btd_comp * w_btd) + (oimi_comp * w_oimi) + (lai_comp * w_lai)
        final_score = round(final_score, 2)
        
        # Grade mapping
        if final_score > 80: grade = "A+"
        elif final_score > 60: grade = "A"
        elif final_score > 40: grade = "B"
        elif final_score > 20: grade = "C"
        elif final_score > 0: grade = "D"
        elif final_score > -20: grade = "D-"
        elif final_score > -40: grade = "F-"
        elif final_score > -60: grade = "F"
        else: grade = "F--"
        
        # Bias
        if final_score > 40: bias = "STRONGLY_BULLISH"
        elif final_score > 10: bias = "BULLISH"
        elif final_score < -40: bias = "STRONGLY_BEARISH"
        elif final_score < -10: bias = "BEARISH"
        else: bias = "NEUTRAL"
        
        # Findings & Insights
        findings = []
        if smfi_res.signal != "NEUTRAL": findings.append(f"Institutional {smfi_res.signal.lower()}")
        if vsd_res.current_bar_shock: findings.append(f"{vsd_res.shocks[-1].shock_type.replace('_', ' ').lower()}")
        if lai_res.absorption_detected: findings.append(f"{lai_res.absorption_type.replace('_', ' ').lower()}")
        
        key_finding = ", ".join(findings) if findings else "Mixed institutional signals"

        # Append FII positioning context to key finding
        if fii_futures_net is not None and abs(fii_futures_net) >= 5000:
            fii_direction = "net long" if fii_futures_net > 0 else "net short"
            key_finding = (
                f"{key_finding}; FII futures {fii_direction} "
                f"({abs(fii_futures_net):,.0f} contracts)"
            )

        if final_score > 30: insight = "Favor CALL entries"
        elif final_score < -30: insight = "Favor PUT entries"
        else: insight = "Stay cautious"
        
        confidence = 0.8
        if data_count < 3: confidence = 0.3
        
        return SmartMoneyScore(
            score=final_score,
            grade=grade,
            smfi_component=smfi_comp,
            vsd_component=vsd_comp,
            btd_component=btd_comp,
            oimi_component=oimi_comp,
            lai_component=lai_comp,
            smart_money_bias=bias,
            key_finding=key_finding,
            actionable_insight=insight,
            data_completeness=data_count / 6.0,
            confidence=confidence
        )
