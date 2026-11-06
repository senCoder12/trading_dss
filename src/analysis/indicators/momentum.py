"""
Momentum indicators for trading analysis.

Implements RSI, Stochastic Oscillator, CCI, and a composite momentum summary.
All methods are pure functions — no database calls or side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import CCIIndicator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RSIResult:
    """Result of RSI calculation.

    RSI oscillates between 0 and 100:
      > 70 → overbought (potential reversal down)
      < 30 → oversold   (potential reversal up)
      30-70 → neutral momentum zone

    ``divergence`` is a leading reversal signal: price and RSI moving in
    opposite directions warns of a potential trend change.
    """

    rsi: pd.Series                  # RSI values (0-100)
    current_value: float            # Latest RSI reading
    zone: str                       # "OVERBOUGHT" / "OVERSOLD" / "NEUTRAL"
    divergence: Optional[str]       # "BULLISH_DIVERGENCE" / "BEARISH_DIVERGENCE" / None
    entering_overbought: bool       # RSI just crossed above 70 (within last 2 bars)
    leaving_overbought: bool        # RSI just crossed below 70
    entering_oversold: bool         # RSI just crossed below 30
    leaving_oversold: bool          # RSI just crossed above 30 (buy signal)


@dataclass
class StochasticResult:
    """Result of Stochastic Oscillator calculation.

    %K is the fast line (position of close relative to high-low range);
    %D is the smoothed signal line.

    The most reliable signals come from crossovers in extreme zones:
      %K crosses above %D while both < 20 → strong bullish
      %K crosses below %D while both > 80 → strong bearish
    """

    k_line: pd.Series               # %K (fast line, 0-100)
    d_line: pd.Series               # %D (slow / signal line)
    current_k: float
    current_d: float
    zone: str                       # "OVERBOUGHT" (K>80) / "OVERSOLD" (K<20) / "NEUTRAL"
    crossover: Optional[str]        # "BULLISH_CROSS" / "BEARISH_CROSS" / None
    signal_quality: str             # "STRONG" (cross in extreme zone) / "WEAK"


@dataclass
class CCIResult:
    """Result of CCI (Commodity Channel Index) calculation.

    CCI measures how far the typical price deviates from its SMA, scaled by
    mean deviation.  It is unbounded but typically oscillates between −200
    and +200:
      > +100 → overbought / strong upward momentum
      < −100 → oversold / strong downward momentum
    """

    cci: pd.Series                  # CCI values (unbounded)
    current_value: float
    zone: str                       # "OVERBOUGHT" / "OVERSOLD" / "NEUTRAL"
    trend: str                      # "RISING" / "FALLING" / "FLAT" (last 5 bars slope)
    zero_cross: Optional[str]       # "BULLISH_CROSS" / "BEARISH_CROSS" / None


@dataclass
class MomentumSummary:
    """Composite momentum reading from RSI, Stochastic, and CCI.

    ``momentum_vote`` is derived from per-indicator bullish / bearish votes.
    Special signals (``overbought_consensus``, ``oversold_consensus``,
    ``reversal_warning``) flag high-conviction reversal setups.
    """

    timestamp: datetime

    # RSI
    rsi_value: float
    rsi_zone: str
    rsi_divergence: Optional[str]

    # Stochastic
    stochastic_k: float
    stochastic_zone: str
    stochastic_crossover: Optional[str]

    # CCI
    cci_value: float
    cci_zone: str

    # Overall momentum verdict
    momentum_vote: str              # STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG_BEARISH
    momentum_confidence: float      # 0.0 – 1.0

    # Special signals
    overbought_consensus: bool      # 2+ indicators in overbought zone
    oversold_consensus: bool        # 2+ indicators in oversold zone
    divergence_detected: bool       # Any divergence found
    reversal_warning: Optional[str] # "POTENTIAL_TOP" / "POTENTIAL_BOTTOM" / None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _roc_noise_band(df: pd.DataFrame, lookback: int) -> float:
    """Return threshold below which a *lookback*-bar move is noise.

    Half of the typical *lookback*-bar move (median absolute ROC over the
    available history), floored at 0.05 % so flat series never vote.
    """
    closes = df["close"].astype(float)
    rocs = (closes / closes.shift(lookback) - 1.0).abs().dropna()
    if rocs.empty:
        return 0.0005
    return max(float(rocs.median()) * 0.5, 0.0005)


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy of *df* with NaNs forward-filled.

    Logs a warning when NaN values are detected in the input.
    """
    if df.empty:
        return df.copy()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    has_nan = df[list(required)].isna().any().any()
    if has_nan:
        logger.warning("Input DataFrame contains NaN values — forward-filling before calculation.")
    clean = df.copy()
    clean[list(required)] = clean[list(required)].ffill()
    return clean


def _detect_cross(series_a: pd.Series, series_b: pd.Series) -> Optional[str]:
    """Detect whether *series_a* crossed *series_b* at the most recent bar.

    Returns ``"BULLISH_CROSS"`` when A crosses above B, ``"BEARISH_CROSS"``
    when A crosses below B, or ``None``.
    """
    if len(series_a) < 2 or len(series_b) < 2:
        return None

    prev_above = series_a.iloc[-2] > series_b.iloc[-2]
    curr_above = series_a.iloc[-1] > series_b.iloc[-1]

    if not prev_above and curr_above:
        return "BULLISH_CROSS"
    if prev_above and not curr_above:
        return "BEARISH_CROSS"
    return None


def _detect_threshold_cross(
    series: pd.Series, threshold: float
) -> tuple[bool, bool]:
    """Return ``(crossed_above, crossed_below)`` for the most recent bar.

    Examines the last two valid values of *series* relative to *threshold*.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return False, False

    prev = float(valid.iloc[-2])
    curr = float(valid.iloc[-1])

    crossed_above = prev <= threshold < curr
    crossed_below = prev >= threshold > curr
    return crossed_above, crossed_below


def _detect_divergence(
    price_series: pd.Series,
    indicator_series: pd.Series,
    lookback: int = 14,
) -> Optional[str]:
    """Detect bullish or bearish divergence between price and an oscillator.

    Algorithm:
      1. Take the last *lookback* bars.
      2. Find swing lows/highs — a bar whose value is lower/higher than both
         immediate neighbours by at least 0.1 %.
      3. Compare the two most recent swing lows for bullish divergence
         (price makes lower low, indicator makes higher low) and the two most
         recent swing highs for bearish divergence (price makes higher high,
         indicator makes lower high).

    Returns:
        ``"BULLISH_DIVERGENCE"``, ``"BEARISH_DIVERGENCE"``, or ``None``.
    """
    if len(price_series) < lookback or len(indicator_series) < lookback:
        return None

    price = price_series.iloc[-lookback:].values.astype(float)
    indic = indicator_series.iloc[-lookback:].values.astype(float)

    # --- swing lows (bullish divergence check) ---
    swing_lows: list[int] = []
    for i in range(1, len(price) - 1):
        if price[i] < price[i - 1] * 0.999 and price[i] < price[i + 1] * 0.999:
            swing_lows.append(i)

    if len(swing_lows) >= 2:
        i1, i2 = swing_lows[-2], swing_lows[-1]
        if price[i2] < price[i1] and indic[i2] > indic[i1]:
            return "BULLISH_DIVERGENCE"

    # --- swing highs (bearish divergence check) ---
    swing_highs: list[int] = []
    for i in range(1, len(price) - 1):
        if price[i] > price[i - 1] * 1.001 and price[i] > price[i + 1] * 1.001:
            swing_highs.append(i)

    if len(swing_highs) >= 2:
        i1, i2 = swing_highs[-2], swing_highs[-1]
        if price[i2] > price[i1] and indic[i2] < indic[i1]:
            return "BEARISH_DIVERGENCE"

    return None


# ---------------------------------------------------------------------------
# MomentumIndicators
# ---------------------------------------------------------------------------


class MomentumIndicators:
    """Pure-function momentum indicator calculations for OHLCV DataFrames.

    All public methods accept a DataFrame with columns
    ``[timestamp, open, high, low, close, volume]`` and return either a
    ``pd.Series`` or a result dataclass.  No state is stored on the instance.
    """

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> RSIResult:
        """Calculate Relative Strength Index.

        RSI = 100 − 100 / (1 + RS), where RS = avg gain / avg loss over
        *period* bars.  Values above 70 indicate overbought conditions;
        below 30 indicate oversold.

        Args:
            df: OHLCV DataFrame.
            period: Look-back window (default 14).

        Returns:
            RSIResult with full series and latest-bar summary fields.
        """
        clean = _prepare_df(df)

        if clean.empty or len(clean) < period + 1:
            logger.warning(
                "Insufficient data for RSI calculation: need %d rows, got %d.",
                period + 1,
                len(clean) if not clean.empty else 0,
            )
            empty = pd.Series(dtype=float)
            return RSIResult(
                rsi=empty,
                current_value=float("nan"),
                zone="NEUTRAL",
                divergence=None,
                entering_overbought=False,
                leaving_overbought=False,
                entering_oversold=False,
                leaving_oversold=False,
            )

        rsi_ind = RSIIndicator(close=clean["close"], window=period, fillna=False)
        rsi_series = rsi_ind.rsi()

        valid = rsi_series.dropna()
        current_value = float(valid.iloc[-1]) if len(valid) > 0 else float("nan")

        # Zone classification
        if np.isnan(current_value):
            zone = "NEUTRAL"
        elif current_value > 70:
            zone = "OVERBOUGHT"
        elif current_value < 30:
            zone = "OVERSOLD"
        else:
            zone = "NEUTRAL"

        # Threshold crossings (overbought = 70, oversold = 30)
        entering_ob, leaving_ob = _detect_threshold_cross(rsi_series, 70.0)
        # For oversold: crossing *below* 30 = entering, crossing *above* 30 = leaving
        leaving_os, entering_os = _detect_threshold_cross(rsi_series, 30.0)

        # Divergence
        divergence = _detect_divergence(clean["close"], rsi_series, lookback=period)

        return RSIResult(
            rsi=rsi_series,
            current_value=current_value,
            zone=zone,
            divergence=divergence,
            entering_overbought=entering_ob,
            leaving_overbought=leaving_ob,
            entering_oversold=entering_os,
            leaving_oversold=leaving_os,
        )

    # ------------------------------------------------------------------
    # Stochastic Oscillator
    # ------------------------------------------------------------------

    def calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> StochasticResult:
        """Calculate the Stochastic Oscillator (%K and %D).

        %K = SMA( (close − low_N) / (high_N − low_N) × 100, smooth_k )
        %D = SMA(%K, d_period)

        The most reliable signals are crossovers in extreme zones (< 20 or > 80).

        Args:
            df: OHLCV DataFrame.
            k_period: Look-back window for high/low range (default 14).
            d_period: Smoothing period for %D signal line (default 3).
            smooth_k: Smoothing period for %K itself (default 3).

        Returns:
            StochasticResult with all series and latest-bar summary fields.
        """
        clean = _prepare_df(df)

        min_rows = k_period + smooth_k + d_period
        if clean.empty or len(clean) < min_rows:
            logger.warning(
                "Insufficient data for Stochastic calculation: need %d rows, got %d.",
                min_rows,
                len(clean) if not clean.empty else 0,
            )
            empty = pd.Series(dtype=float)
            return StochasticResult(
                k_line=empty,
                d_line=empty,
                current_k=float("nan"),
                current_d=float("nan"),
                zone="NEUTRAL",
                crossover=None,
                signal_quality="WEAK",
            )

        stoch = StochasticOscillator(
            high=clean["high"],
            low=clean["low"],
            close=clean["close"],
            window=k_period,
            smooth_window=smooth_k,
            fillna=False,
        )

        # Slow stochastic, as documented: %K = SMA(raw %K, smooth_k) and
        # %D = SMA(%K, d_period).  ta's stoch() is the RAW (fast) %K and
        # d_period was never used, so crossovers were on the noisiest line.
        k_line = stoch.stoch().rolling(smooth_k).mean()
        d_line = k_line.rolling(d_period).mean()

        valid_k = k_line.dropna()
        valid_d = d_line.dropna()
        current_k = float(valid_k.iloc[-1]) if len(valid_k) > 0 else float("nan")
        current_d = float(valid_d.iloc[-1]) if len(valid_d) > 0 else float("nan")

        # Zone
        if np.isnan(current_k):
            zone = "NEUTRAL"
        elif current_k > 80:
            zone = "OVERBOUGHT"
        elif current_k < 20:
            zone = "OVERSOLD"
        else:
            zone = "NEUTRAL"

        # Crossover
        crossover = _detect_cross(k_line, d_line)

        # Signal quality: cross in extreme zone = STRONG
        if crossover is not None:
            prev_k = float(valid_k.iloc[-2]) if len(valid_k) >= 2 else float("nan")
            if (crossover == "BULLISH_CROSS" and (current_k < 20 or prev_k < 20)):
                signal_quality = "STRONG"
            elif (crossover == "BEARISH_CROSS" and (current_k > 80 or prev_k > 80)):
                signal_quality = "STRONG"
            else:
                signal_quality = "WEAK"
        else:
            signal_quality = "WEAK"

        return StochasticResult(
            k_line=k_line,
            d_line=d_line,
            current_k=current_k,
            current_d=current_d,
            zone=zone,
            crossover=crossover,
            signal_quality=signal_quality,
        )

    # ------------------------------------------------------------------
    # CCI
    # ------------------------------------------------------------------

    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> CCIResult:
        """Calculate the Commodity Channel Index.

        CCI = (Typical Price − SMA(TP, period)) / (0.015 × Mean Deviation)
        where Typical Price = (high + low + close) / 3.

        Values above +100 indicate overbought / strong bullish momentum;
        below −100 indicate oversold / strong bearish momentum.  Zero-line
        crosses are momentum-shift signals.

        Args:
            df: OHLCV DataFrame.
            period: Look-back window (default 20).

        Returns:
            CCIResult with full series and latest-bar summary fields.
        """
        clean = _prepare_df(df)

        if clean.empty or len(clean) < period:
            logger.warning(
                "Insufficient data for CCI calculation: need %d rows, got %d.",
                period,
                len(clean) if not clean.empty else 0,
            )
            empty = pd.Series(dtype=float)
            return CCIResult(
                cci=empty,
                current_value=float("nan"),
                zone="NEUTRAL",
                trend="FLAT",
                zero_cross=None,
            )

        cci_ind = CCIIndicator(
            high=clean["high"],
            low=clean["low"],
            close=clean["close"],
            window=period,
            fillna=False,
        )
        cci_series = cci_ind.cci()

        valid = cci_series.dropna()
        current_value = float(valid.iloc[-1]) if len(valid) > 0 else float("nan")

        # Zone
        if np.isnan(current_value):
            zone = "NEUTRAL"
        elif current_value > 100:
            zone = "OVERBOUGHT"
        elif current_value < -100:
            zone = "OVERSOLD"
        else:
            zone = "NEUTRAL"

        # Trend (slope of last 5 valid bars)
        if len(valid) >= 5:
            tail = valid.iloc[-5:].values
            slope = np.polyfit(range(5), tail, 1)[0]
            if slope > 5:
                trend = "RISING"
            elif slope < -5:
                trend = "FALLING"
            else:
                trend = "FLAT"
        else:
            trend = "FLAT"

        # Zero-line cross
        zero_line = pd.Series(0.0, index=cci_series.index)
        zero_cross = _detect_cross(cci_series, zero_line)

        return CCIResult(
            cci=cci_series,
            current_value=current_value,
            zone=zone,
            trend=trend,
            zero_cross=zero_cross,
        )

    # ------------------------------------------------------------------
    # Momentum Summary
    # ------------------------------------------------------------------

    def get_momentum_summary(self, df: pd.DataFrame) -> MomentumSummary:
        """Produce a composite momentum summary from RSI, Stochastic, and CCI.

        Voting system:

        Bullish votes:
          - RSI leaving oversold (crossed above 30)                → +1
          - RSI zone OVERSOLD (contrarian buy opportunity)         → +1
          - Stochastic BULLISH_CROSS                               → +1 (+2 if STRONG)
          - CCI BULLISH_CROSS (crossed above zero)                 → +1
          - CCI zone OVERSOLD (contrarian)                         → +1

        Bearish votes:
          - RSI entering overbought (crossed above 70)             → +1
          - RSI zone OVERBOUGHT                                    → +1
          - Stochastic BEARISH_CROSS                               → +1 (+2 if STRONG)
          - CCI BEARISH_CROSS (crossed below zero)                 → +1
          - CCI zone OVERBOUGHT                                    → +1

        Score mapping:
          net ≥ +4  → STRONG_BULLISH  (confidence 0.90)
          net ≥ +2  → BULLISH         (confidence 0.70)
          net ≥ −1  → NEUTRAL         (confidence 0.50)
          net ≥ −3  → BEARISH         (confidence 0.70)
          net < −3  → STRONG_BEARISH  (confidence 0.90)

        Divergence detected by RSI overrides: confidence += 0.1.

        Reversal warnings:
          overbought_consensus + bearish divergence → "POTENTIAL_TOP"
          oversold_consensus + bullish divergence   → "POTENTIAL_BOTTOM"

        Args:
            df: OHLCV DataFrame.

        Returns:
            MomentumSummary with all fields populated.
        """
        now = datetime.utcnow()
        empty_summary = MomentumSummary(
            timestamp=now,
            rsi_value=float("nan"),
            rsi_zone="NEUTRAL",
            rsi_divergence=None,
            stochastic_k=float("nan"),
            stochastic_zone="NEUTRAL",
            stochastic_crossover=None,
            cci_value=float("nan"),
            cci_zone="NEUTRAL",
            momentum_vote="NEUTRAL",
            momentum_confidence=0.0,
            overbought_consensus=False,
            oversold_consensus=False,
            divergence_detected=False,
            reversal_warning=None,
        )

        clean = _prepare_df(df)
        if clean.empty or len(clean) < 2:
            logger.warning("DataFrame too short for momentum summary.")
            return empty_summary

        timestamp = (
            pd.Timestamp(clean["timestamp"].iloc[-1]).to_pydatetime()
            if "timestamp" in clean.columns
            else now
        )

        # --- Indicator calculations ----------------------------------------
        rsi_result = self.calculate_rsi(clean)
        stoch_result = self.calculate_stochastic(clean)
        cci_result = self.calculate_cci(clean)

        # --- Zone consensus -------------------------------------------------
        overbought_zones = [
            rsi_result.zone == "OVERBOUGHT",
            stoch_result.zone == "OVERBOUGHT",
            cci_result.zone == "OVERBOUGHT",
        ]
        oversold_zones = [
            rsi_result.zone == "OVERSOLD",
            stoch_result.zone == "OVERSOLD",
            cci_result.zone == "OVERSOLD",
        ]
        overbought_consensus = sum(overbought_zones) >= 2
        oversold_consensus = sum(oversold_zones) >= 2

        # --- Voting: momentum CONTINUATION ----------------------------------
        # An option buyer needs the move to continue.  The old vote was purely
        # contrarian (overbought = bearish, oversold = bullish) and so fought
        # the trend layer on every trending day.  Overbought/oversold are still
        # reported (consensus flags, reversal warnings) but no longer vote.
        #   RSI regime   : > 55 bullish, < 45 bearish; beyond 75/25 the move is
        #                  stretched — no vote rather than a reversal call
        #   10-bar ROC   : direction of the last 10 bars beyond a noise band
        #   slow stoch   : %K/%D crossover (weight 1, either zone)
        votes: list[int] = []
        rsi_now = rsi_result.current_value
        if not np.isnan(rsi_now):
            if 55.0 < rsi_now <= 75.0:
                votes.append(1)
            elif 25.0 <= rsi_now < 45.0:
                votes.append(-1)
            else:
                votes.append(0)

        roc_lookback = 10
        if len(clean) > roc_lookback:
            past = float(clean["close"].iloc[-1 - roc_lookback])
            roc = (float(clean["close"].iloc[-1]) - past) / past if past > 0 else 0.0
            noise = _roc_noise_band(clean, roc_lookback)
            votes.append(1 if roc > noise else (-1 if roc < -noise else 0))

        if stoch_result.crossover == "BULLISH_CROSS":
            votes.append(1)
        elif stoch_result.crossover == "BEARISH_CROSS":
            votes.append(-1)
        elif not np.isnan(stoch_result.current_k):
            votes.append(0)

        bullish_votes = sum(1 for v in votes if v > 0)
        bearish_votes = sum(1 for v in votes if v < 0)
        net = bullish_votes - bearish_votes

        if len(votes) < 2:
            momentum_vote, base_confidence = "NEUTRAL", 0.3
        elif net >= 3:
            momentum_vote, base_confidence = "STRONG_BULLISH", 0.85
        elif net >= 2:
            momentum_vote, base_confidence = "BULLISH", 0.70
        elif net <= -3:
            momentum_vote, base_confidence = "STRONG_BEARISH", 0.85
        elif net <= -2:
            momentum_vote, base_confidence = "BEARISH", 0.70
        else:
            momentum_vote, base_confidence = "NEUTRAL", 0.50

        # --- Divergence -----------------------------------------------------
        divergence_detected = rsi_result.divergence is not None
        if divergence_detected:
            base_confidence = min(1.0, base_confidence + 0.1)

        # --- Reversal warnings ----------------------------------------------
        reversal_warning: Optional[str] = None
        if overbought_consensus and rsi_result.divergence == "BEARISH_DIVERGENCE":
            reversal_warning = "POTENTIAL_TOP"
        elif oversold_consensus and rsi_result.divergence == "BULLISH_DIVERGENCE":
            reversal_warning = "POTENTIAL_BOTTOM"

        momentum_confidence = round(base_confidence, 2)

        return MomentumSummary(
            timestamp=timestamp,
            rsi_value=rsi_result.current_value,
            rsi_zone=rsi_result.zone,
            rsi_divergence=rsi_result.divergence,
            stochastic_k=stoch_result.current_k,
            stochastic_zone=stoch_result.zone,
            stochastic_crossover=stoch_result.crossover,
            cci_value=cci_result.current_value,
            cci_zone=cci_result.zone,
            momentum_vote=momentum_vote,
            momentum_confidence=momentum_confidence,
            overbought_consensus=overbought_consensus,
            oversold_consensus=oversold_consensus,
            divergence_detected=divergence_detected,
            reversal_warning=reversal_warning,
        )
