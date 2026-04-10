"""
Trend indicators for trading analysis.

Implements SMA, EMA, MACD, ADX, and a composite trend summary.
All methods are pure functions — no database calls or side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator

logger = logging.getLogger(__name__)

# Default MA periods used across the module
_DEFAULT_PERIODS = (9, 20, 50, 100, 200)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MACDResult:
    """Result of MACD calculation.

    ``is_bullish`` and ``crossover`` describe the state at the latest bar only.
    ``histogram_increasing`` is True when the latest histogram value is greater
    than the previous one (momentum building).
    """

    macd_line: pd.Series          # fast EMA − slow EMA
    signal_line: pd.Series        # EMA of the MACD line
    histogram: pd.Series          # MACD − signal
    is_bullish: bool               # latest: MACD > signal
    crossover: Optional[str]       # "BULLISH_CROSS" / "BEARISH_CROSS" / None
    histogram_increasing: bool     # histogram growing more positive (or less negative)


@dataclass
class ADXResult:
    """Result of ADX calculation.

    ``trend_strength`` categories by ADX level:
      < 20  → "WEAK"
      20-25 → "EMERGING"
      25-50 → "STRONG"
      50-75 → "VERY_STRONG"
      ≥ 75  → "EXTREME"

    ``trend_direction`` is derived from whichever directional indicator is
    dominant at the latest bar (+DI vs −DI).
    """

    adx: pd.Series                 # 0-100 trend-strength line
    plus_di: pd.Series             # +DI bullish directional
    minus_di: pd.Series            # −DI bearish directional
    trend_strength: str            # WEAK / EMERGING / STRONG / VERY_STRONG / EXTREME
    trend_direction: str           # "BULLISH" (+DI > −DI) or "BEARISH"
    di_crossover: Optional[str]    # "BULLISH_CROSS" / "BEARISH_CROSS" / None


@dataclass
class TrendSummary:
    """Composite trend summary derived from multiple indicators.

    ``trend_vote`` is produced by a 5-signal majority vote:
      - price > EMA20
      - price > EMA50
      - price > EMA200
      - MACD bullish
      - +DI > −DI

    ``trend_confidence`` is scaled from the vote count (0 → 1) with a −0.2
    penalty when ADX < 20 (weak / range-bound market).
    """

    index_id: str
    timeframe: str
    timestamp: datetime

    # Price vs moving averages
    price_vs_ema20: str            # "ABOVE" / "BELOW"
    price_vs_ema50: str
    price_vs_ema200: str

    # MA alignment
    ema_alignment: str             # "BULLISH" / "BEARISH" / "MIXED"
    golden_cross: bool             # EMA50 just crossed above EMA200
    death_cross: bool              # EMA50 just crossed below EMA200

    # MACD state
    macd_signal: str               # "BULLISH" / "BEARISH" / "NEUTRAL"
    macd_crossover: Optional[str]
    macd_histogram_trend: str      # "INCREASING" / "DECREASING"

    # ADX state
    trend_strength: str
    trend_direction: str

    # Composite verdict
    trend_vote: str                # STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG_BEARISH
    trend_confidence: float        # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


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


def _classify_adx_strength(adx_val: float) -> str:
    """Map a scalar ADX value to a human-readable trend-strength label."""
    if adx_val < 20:
        return "WEAK"
    if adx_val < 25:
        return "EMERGING"
    if adx_val < 50:
        return "STRONG"
    if adx_val < 75:
        return "VERY_STRONG"
    return "EXTREME"


def _detect_cross(series_a: pd.Series, series_b: pd.Series) -> Optional[str]:
    """Detect whether *series_a* crossed *series_b* at the most recent bar.

    Returns ``"BULLISH_CROSS"`` when A crosses above B, ``"BEARISH_CROSS"``
    when A crosses below B, or ``None`` when no crossover occurred.
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


# ---------------------------------------------------------------------------
# TrendIndicators
# ---------------------------------------------------------------------------


class TrendIndicators:
    """Pure-function trend indicator calculations for OHLCV DataFrames.

    All public methods accept a DataFrame with columns
    ``[timestamp, open, high, low, close, volume]`` and return either a
    ``pd.Series`` or a result dataclass.  No state is stored on the instance.
    """

    # ------------------------------------------------------------------
    # SMA
    # ------------------------------------------------------------------

    def calculate_sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average for *period* bars.

        Uses the ``ta`` library's ``SMAIndicator`` under the hood.  When the
        DataFrame has fewer rows than *period*, the earlier values are NaN
        (standard rolling-window behaviour).

        Args:
            df: OHLCV DataFrame.
            period: Look-back window length.

        Returns:
            pd.Series of SMA values aligned with *df*'s index.
        """
        clean = _prepare_df(df)
        if clean.empty or len(clean) < 1:
            return pd.Series(dtype=float)

        if len(clean) < period:
            logger.warning(
                "DataFrame has %d rows but SMA period is %d — result will be all NaN.",
                len(clean),
                period,
            )

        indicator = SMAIndicator(close=clean["close"], window=period, fillna=False)
        result = indicator.sma_indicator()
        result.name = f"sma_{period}"
        return result

    def calculate_all_sma(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Calculate SMA for all default periods (9, 20, 50, 100, 200).

        Args:
            df: OHLCV DataFrame.

        Returns:
            Dict mapping ``"sma_<period>"`` to the corresponding pd.Series.
        """
        return {f"sma_{p}": self.calculate_sma(df, p) for p in _DEFAULT_PERIODS}

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average for *period* bars.

        Uses ``ta.EMAIndicator`` with ``fillna=False`` so short DataFrames
        produce NaN values rather than being silently back-filled.

        Args:
            df: OHLCV DataFrame.
            period: Span for the EMA.

        Returns:
            pd.Series of EMA values aligned with *df*'s index.
        """
        clean = _prepare_df(df)
        if clean.empty:
            return pd.Series(dtype=float)

        if len(clean) < period:
            logger.debug(
                "DataFrame has %d rows but EMA period is %d — result will be partially NaN.",
                len(clean),
                period,
            )

        indicator = EMAIndicator(close=clean["close"], window=period, fillna=False)
        result = indicator.ema_indicator()
        result.name = f"ema_{period}"
        return result

    def calculate_all_ema(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Calculate EMA for all default periods (9, 20, 50, 100, 200).

        Args:
            df: OHLCV DataFrame.

        Returns:
            Dict mapping ``"ema_<period>"`` to the corresponding pd.Series.
        """
        return {f"ema_{p}": self.calculate_ema(df, p) for p in _DEFAULT_PERIODS}

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> MACDResult:
        """Calculate MACD (Moving Average Convergence Divergence).

        Formula:
          MACD line   = EMA(close, fast) − EMA(close, slow)
          Signal line = EMA(MACD line, signal)
          Histogram   = MACD line − Signal line

        A positive histogram means bullish momentum; a negative histogram means
        bearish momentum.  A crossover (MACD crossing its signal) is a classic
        entry/exit trigger.

        Args:
            df: OHLCV DataFrame (minimum ~35 bars for meaningful output).
            fast: Fast EMA period (default 12).
            slow: Slow EMA period (default 26).
            signal: Signal EMA period (default 9).

        Returns:
            MACDResult with all series and latest-bar summary fields.
        """
        clean = _prepare_df(df)

        if clean.empty or len(clean) < slow + signal:
            logger.warning("Insufficient data for MACD calculation (%d rows).", len(clean) if not clean.empty else 0)
            empty = pd.Series(dtype=float)
            return MACDResult(
                macd_line=empty,
                signal_line=empty,
                histogram=empty,
                is_bullish=False,
                crossover=None,
                histogram_increasing=False,
            )

        macd_ind = MACD(
            close=clean["close"],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal,
            fillna=False,
        )

        macd_line = macd_ind.macd()
        signal_line = macd_ind.macd_signal()
        histogram = macd_ind.macd_diff()

        # Drop leading NaNs for boolean logic
        valid_macd = macd_line.dropna()
        valid_signal = signal_line.dropna()
        valid_hist = histogram.dropna()

        is_bullish = bool(valid_macd.iloc[-1] > valid_signal.iloc[-1]) if len(valid_macd) > 0 else False
        crossover = _detect_cross(macd_line, signal_line)
        histogram_increasing = (
            bool(valid_hist.iloc[-1] > valid_hist.iloc[-2])
            if len(valid_hist) >= 2
            else False
        )

        return MACDResult(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            is_bullish=is_bullish,
            crossover=crossover,
            histogram_increasing=histogram_increasing,
        )

    # ------------------------------------------------------------------
    # ADX
    # ------------------------------------------------------------------

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> ADXResult:
        """Calculate ADX (Average Directional Index) plus ±DI lines.

        ADX measures *trend strength* (not direction):
          < 20  → weak / ranging
          20-25 → emerging trend
          25-50 → strong trend
          50-75 → very strong trend
          ≥ 75  → extreme trend (rare)

        Direction comes from the Directional Indicators:
          +DI > −DI → bullish trend; −DI > +DI → bearish trend.

        Args:
            df: OHLCV DataFrame (minimum ~28 bars for meaningful ADX).
            period: Smoothing period (default 14).

        Returns:
            ADXResult with all series and latest-bar classification.
        """
        clean = _prepare_df(df)

        min_rows = period * 2
        if clean.empty or len(clean) < min_rows:
            logger.warning(
                "Insufficient data for ADX calculation: need %d rows, got %d.",
                min_rows,
                len(clean) if not clean.empty else 0,
            )
            empty = pd.Series(dtype=float)
            return ADXResult(
                adx=empty,
                plus_di=empty,
                minus_di=empty,
                trend_strength="WEAK",
                trend_direction="BEARISH",
                di_crossover=None,
            )

        adx_ind = ADXIndicator(
            high=clean["high"],
            low=clean["low"],
            close=clean["close"],
            window=period,
            fillna=False,
        )

        adx_series = adx_ind.adx()
        plus_di = adx_ind.adx_pos()
        minus_di = adx_ind.adx_neg()

        valid_adx = adx_series.dropna()
        valid_plus = plus_di.dropna()
        valid_minus = minus_di.dropna()

        latest_adx = float(valid_adx.iloc[-1]) if len(valid_adx) > 0 else 0.0
        latest_plus = float(valid_plus.iloc[-1]) if len(valid_plus) > 0 else 0.0
        latest_minus = float(valid_minus.iloc[-1]) if len(valid_minus) > 0 else 0.0

        trend_strength = _classify_adx_strength(latest_adx)
        trend_direction = "BULLISH" if latest_plus > latest_minus else "BEARISH"
        di_crossover = _detect_cross(plus_di, minus_di)

        return ADXResult(
            adx=adx_series,
            plus_di=plus_di,
            minus_di=minus_di,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            di_crossover=di_crossover,
        )

    # ------------------------------------------------------------------
    # Trend Summary
    # ------------------------------------------------------------------

    def get_trend_summary(
        self,
        df: pd.DataFrame,
        index_id: str = "",
        timeframe: str = "1d",
    ) -> TrendSummary:
        """Produce a composite trend summary for one index/timeframe combination.

        Runs EMA20/50/200, MACD, and ADX; then applies a 5-signal majority vote:

        Bullish signals:
          1. price > EMA20
          2. price > EMA50
          3. price > EMA200
          4. MACD line > signal line
          5. +DI > −DI

        Vote → trend_vote mapping:
          5 bullish  → STRONG_BULLISH  (base confidence 0.95)
          4 bullish  → BULLISH         (base confidence 0.75)
          3 bullish  → NEUTRAL         (base confidence 0.50)
          2 bullish  → BEARISH         (base confidence 0.75)
          1 bullish  → STRONG_BEARISH  (base confidence 0.95)
          0 bullish  → STRONG_BEARISH  (base confidence 0.95)

        A −0.2 confidence penalty is applied when ADX < 20 (weak trend means
        less reliable directional signals).

        Args:
            df: OHLCV DataFrame.
            index_id: Identifier for the index (e.g. "NIFTY50").
            timeframe: Timeframe label (e.g. "1d", "1h").

        Returns:
            TrendSummary dataclass with all fields populated.
        """
        now = datetime.utcnow()
        empty_summary = TrendSummary(
            index_id=index_id,
            timeframe=timeframe,
            timestamp=now,
            price_vs_ema20="BELOW",
            price_vs_ema50="BELOW",
            price_vs_ema200="BELOW",
            ema_alignment="MIXED",
            golden_cross=False,
            death_cross=False,
            macd_signal="NEUTRAL",
            macd_crossover=None,
            macd_histogram_trend="DECREASING",
            trend_strength="WEAK",
            trend_direction="BEARISH",
            trend_vote="NEUTRAL",
            trend_confidence=0.0,
        )

        clean = _prepare_df(df)
        if clean.empty or len(clean) < 2:
            logger.warning("DataFrame too short for trend summary.")
            return empty_summary

        latest_close = float(clean["close"].iloc[-1])
        timestamp = (
            pd.Timestamp(clean["timestamp"].iloc[-1]).to_pydatetime()
            if "timestamp" in clean.columns
            else now
        )

        # --- EMAs ----------------------------------------------------------
        ema20 = self.calculate_ema(clean, 20)
        ema50 = self.calculate_ema(clean, 50)
        ema200 = self.calculate_ema(clean, 200)

        def _latest(series: pd.Series) -> float:
            valid = series.dropna()
            return float(valid.iloc[-1]) if len(valid) > 0 else float("nan")

        e20, e50, e200 = _latest(ema20), _latest(ema50), _latest(ema200)

        # Price vs EMA
        price_vs_ema20 = "ABOVE" if not np.isnan(e20) and latest_close > e20 else "BELOW"
        price_vs_ema50 = "ABOVE" if not np.isnan(e50) and latest_close > e50 else "BELOW"
        price_vs_ema200 = "ABOVE" if not np.isnan(e200) and latest_close > e200 else "BELOW"

        # EMA alignment
        if not (np.isnan(e20) or np.isnan(e50) or np.isnan(e200)):
            if e20 > e50 > e200:
                ema_alignment = "BULLISH"
            elif e200 > e50 > e20:
                ema_alignment = "BEARISH"
            else:
                ema_alignment = "MIXED"
        else:
            ema_alignment = "MIXED"

        # Golden / death cross (EMA50 vs EMA200)
        golden_cross = _detect_cross(ema50, ema200) == "BULLISH_CROSS"
        death_cross = _detect_cross(ema50, ema200) == "BEARISH_CROSS"

        # --- MACD ----------------------------------------------------------
        macd_result = self.calculate_macd(clean)

        if macd_result.macd_line.empty:
            macd_signal_str = "NEUTRAL"
        elif macd_result.is_bullish:
            macd_signal_str = "BULLISH"
        else:
            macd_signal_str = "BEARISH"

        macd_histogram_trend = (
            "INCREASING" if macd_result.histogram_increasing else "DECREASING"
        )

        # --- ADX -----------------------------------------------------------
        adx_result = self.calculate_adx(clean)

        # --- 5-signal vote -------------------------------------------------
        bullish_count = sum([
            price_vs_ema20 == "ABOVE",
            price_vs_ema50 == "ABOVE",
            price_vs_ema200 == "ABOVE",
            macd_signal_str == "BULLISH",
            adx_result.trend_direction == "BULLISH",
        ])

        _vote_map = {
            5: ("STRONG_BULLISH", 0.95),
            4: ("BULLISH", 0.75),
            3: ("NEUTRAL", 0.50),
            2: ("BEARISH", 0.75),
            1: ("STRONG_BEARISH", 0.95),
            0: ("STRONG_BEARISH", 0.95),
        }
        trend_vote, base_confidence = _vote_map[bullish_count]

        # ADX penalty
        adx_valid = adx_result.adx.dropna()
        latest_adx = float(adx_valid.iloc[-1]) if len(adx_valid) > 0 else 0.0
        confidence_penalty = 0.2 if latest_adx < 20 else 0.0
        trend_confidence = max(0.0, round(base_confidence - confidence_penalty, 2))

        return TrendSummary(
            index_id=index_id,
            timeframe=timeframe,
            timestamp=timestamp,
            price_vs_ema20=price_vs_ema20,
            price_vs_ema50=price_vs_ema50,
            price_vs_ema200=price_vs_ema200,
            ema_alignment=ema_alignment,
            golden_cross=golden_cross,
            death_cross=death_cross,
            macd_signal=macd_signal_str,
            macd_crossover=macd_result.crossover,
            macd_histogram_trend=macd_histogram_trend,
            trend_strength=adx_result.trend_strength,
            trend_direction=adx_result.trend_direction,
            trend_vote=trend_vote,
            trend_confidence=trend_confidence,
        )
