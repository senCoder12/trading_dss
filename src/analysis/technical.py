"""
Technical indicators: VWAP, EMA, RSI, Bollinger Bands, support/resistance.

All functions accept and return pandas DataFrames/Series with lowercase column names.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalAnalysisError(Exception):
    """Raised when technical indicators cannot be computed."""


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Compute Exponential Moving Average.

    Parameters
    ----------
    series:
        Price series (typically ``close``).
    period:
        EMA period.
    """
    if len(series) < period:
        raise TechnicalAnalysisError(
            f"EMA({period}): series length {len(series)} < period"
        )
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index.

    Parameters
    ----------
    series:
        Closing price series.
    period:
        RSI lookback period (default 14).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Compute Bollinger Bands.

    Returns a DataFrame with columns: ``bb_mid``, ``bb_upper``, ``bb_lower``,
    ``bb_width``, ``bb_pct_b``.

    Parameters
    ----------
    series:
        Closing price series.
    period:
        Rolling window length (default 20).
    std_dev:
        Number of standard deviations for bands (default 2).
    """
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = (upper - lower) / mid
    pct_b = (series - lower) / (upper - lower)
    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct_b": pct_b,
    })


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute intraday Volume Weighted Average Price.

    Resets at the start of each trading day.  Requires columns:
    ``high``, ``low``, ``close``, ``volume``, with a ``DatetimeIndex``.

    Parameters
    ----------
    df:
        Intraday OHLCV DataFrame.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]
    # Group by date to reset intraday
    date_group = df.index.date  # type: ignore[union-attr]
    cumulative_tp_vol = tp_vol.groupby(date_group).cumsum()
    cumulative_vol = df["volume"].groupby(date_group).cumsum()
    return cumulative_tp_vol / cumulative_vol


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range.

    Parameters
    ----------
    df:
        OHLCV DataFrame with ``high``, ``low``, ``close`` columns.
    period:
        ATR period (default 14).
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute classic floor pivot points (P, R1–R3, S1–S3) from the previous bar.

    Parameters
    ----------
    df:
        Daily OHLCV DataFrame. Uses the last completed bar as "yesterday".

    Returns
    -------
    pd.DataFrame:
        Single-row DataFrame with columns ``P``, ``R1``, ``R2``, ``R3``,
        ``S1``, ``S2``, ``S3``.
    """
    prev = df.iloc[-1]
    h, l, c = prev["high"], prev["low"], prev["close"]
    p = (h + l + c) / 3
    return pd.DataFrame([{
        "P": p,
        "R1": 2 * p - l,
        "R2": p + (h - l),
        "R3": h + 2 * (p - l),
        "S1": 2 * p - h,
        "S2": p - (h - l),
        "S3": l - 2 * (h - p),
    }])


def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Compute Supertrend indicator.

    Effective for Indian markets as a trailing stop and trend indicator.

    Parameters
    ----------
    df:
        OHLCV DataFrame with ``high``, ``low``, ``close`` columns.
    period:
        ATR period (default 10).
    multiplier:
        ATR multiplier for band calculation (default 3.0).

    Returns
    -------
    pd.DataFrame:
        Columns: ``supertrend``, ``supertrend_direction`` (1=bullish, -1=bearish),
        ``supertrend_upper``, ``supertrend_lower``.
    """
    atr_series = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2

    upper_band = hl2 + multiplier * atr_series
    lower_band = hl2 - multiplier * atr_series

    supertrend_vals = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i < period:
            supertrend_vals.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
            continue

        prev_st = supertrend_vals.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]

        # Adjust bands: only move in trend direction
        ub = upper_band.iloc[i]
        lb = lower_band.iloc[i]

        if prev_dir == 1:
            lb = max(lb, supertrend_vals.iloc[i - 1])
        else:
            ub = min(ub, supertrend_vals.iloc[i - 1])

        # Determine direction
        if prev_dir == -1 and close > prev_st:
            direction.iloc[i] = 1
            supertrend_vals.iloc[i] = lb
        elif prev_dir == 1 and close < prev_st:
            direction.iloc[i] = -1
            supertrend_vals.iloc[i] = ub
        else:
            direction.iloc[i] = prev_dir
            supertrend_vals.iloc[i] = lb if prev_dir == 1 else ub

    return pd.DataFrame({
        "supertrend": supertrend_vals,
        "supertrend_direction": direction,
        "supertrend_upper": upper_band,
        "supertrend_lower": lower_band,
    })


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """
    Compute Stochastic Oscillator (%K and %D).

    Complements RSI for overbought/oversold detection in ranging markets.

    Parameters
    ----------
    df:
        OHLCV DataFrame with ``high``, ``low``, ``close`` columns.
    k_period:
        Lookback period for %K (default 14).
    d_period:
        Smoothing period for %D signal line (default 3).
    smooth_k:
        Smoothing period for %K (default 3; set to 1 for fast stochastic).

    Returns
    -------
    pd.DataFrame:
        Columns: ``stoch_k`` (smoothed %K), ``stoch_d`` (%D signal),
        ``stoch_raw_k`` (raw %K before smoothing).
    """
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = high_max - low_min
    raw_k = 100 * (df["close"] - low_min) / denom.replace(0, np.nan)
    smooth_k_series = raw_k.rolling(smooth_k).mean()
    d_series = smooth_k_series.rolling(d_period).mean()
    return pd.DataFrame({
        "stoch_k": smooth_k_series,
        "stoch_d": d_series,
        "stoch_raw_k": raw_k,
    })


def oi_change_velocity(
    oi_series: pd.Series,
    window_minutes: int = 5,
) -> pd.Series:
    """
    Compute OI change velocity over a rolling window.

    Returns rate of OI change per minute — much more informative than
    raw OI change vs day-open baseline.

    Parameters
    ----------
    oi_series:
        Time-indexed OI series (one observation per minute or per snapshot).
    window_minutes:
        Rolling window in number of observations (default 5).

    Returns
    -------
    pd.Series:
        OI velocity (change per period).
    """
    return oi_series.diff(window_minutes)


def support_resistance_levels(
    series: pd.Series,
    window: int = 20,
    min_touches: int = 2,
    tolerance_pct: float = 0.002,
) -> dict[str, list[float]]:
    """
    Detect support and resistance levels by finding price clusters.

    Parameters
    ----------
    series:
        Closing price series.
    window:
        Local high/low detection window (default 20).
    min_touches:
        Minimum times a level must be tested to qualify.
    tolerance_pct:
        Percentage band to cluster nearby levels (default 0.2%).

    Returns
    -------
    dict with keys ``"support"`` and ``"resistance"``, each a sorted list of prices.
    """
    local_highs = series[(series.shift(window) < series) & (series.shift(-window) < series)]
    local_lows = series[(series.shift(window) > series) & (series.shift(-window) > series)]

    def _cluster(levels: pd.Series) -> list[float]:
        if levels.empty:
            return []
        sorted_levels = sorted(levels.values)
        clusters: list[list[float]] = []
        for level in sorted_levels:
            placed = False
            for cluster in clusters:
                if abs(level - cluster[0]) / cluster[0] <= tolerance_pct:
                    cluster.append(level)
                    placed = True
                    break
            if not placed:
                clusters.append([level])
        return sorted([
            float(np.mean(c)) for c in clusters if len(c) >= min_touches
        ])

    return {
        "support": _cluster(local_lows),
        "resistance": _cluster(local_highs),
    }
