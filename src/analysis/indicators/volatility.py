"""
Volatility indicators for trading analysis.

Implements Bollinger Bands, ATR, Standard Deviation, Historical Volatility,
VIX interpretation, and a composite volatility summary.
All methods are pure functions — no database calls or side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BollingerResult:
    """Result of Bollinger Bands calculation.

    ``percent_b`` shows where the close sits within the bands (0 = lower, 1 = upper).
    ``squeeze`` flags a low-volatility compression that often precedes a breakout.
    """

    upper_band: pd.Series           # SMA + (std_dev * σ)
    middle_band: pd.Series          # SMA
    lower_band: pd.Series           # SMA - (std_dev * σ)
    bandwidth: pd.Series            # (upper - lower) / middle * 100
    percent_b: pd.Series            # (close - lower) / (upper - lower)
    current_position: str           # ABOVE_UPPER / UPPER_ZONE / LOWER_ZONE / BELOW_LOWER
    squeeze: bool                   # bandwidth < 20th percentile of last 120 bars
    expansion: bool                 # bandwidth > 80th percentile of last 120 bars
    band_touch: Optional[str]       # UPPER_TOUCH / LOWER_TOUCH / None
    mean_reversion_signal: Optional[str]  # BUY / SELL / None (BB position only)


@dataclass
class ATRResult:
    """Result of ATR (Average True Range) calculation.

    ``atr_pct`` normalises the ATR by current price so that volatility is
    comparable across instruments at different price levels.
    """

    atr: pd.Series                  # ATR values
    current_atr: float              # Latest ATR
    atr_pct: float                  # ATR as % of current price
    volatility_level: str           # LOW / NORMAL / HIGH / EXTREME
    suggested_sl_distance: float    # 1.5 * ATR
    suggested_target_distance: float  # 2.0 * ATR
    expanding: bool                 # ATR increasing over last 5 bars
    contracting: bool               # ATR decreasing over last 5 bars


@dataclass
class StdDevResult:
    """Result of rolling standard deviation calculation."""

    std_dev: pd.Series              # Rolling std dev of close
    current_value: float
    normalized: float               # std_dev / close * 100
    historical_percentile: float    # Rank in last 252 bars (0-100)


@dataclass
class HVResult:
    """Result of Historical Volatility (annualised) calculation.

    ``hv_regime`` thresholds are calibrated for the Indian equity market where
    15-25 % annualised volatility is considered normal.
    """

    hv: pd.Series
    current_hv: float               # Current annualised HV (percentage)
    hv_percentile: float            # Rank in last 252 bars (0-100)
    hv_regime: str                  # LOW_VOL / NORMAL / HIGH_VOL / EXTREME


@dataclass
class VIXInterpretation:
    """Interpretation of the India VIX level and its market implications."""

    value: float
    regime: str                     # LOW / NORMAL / ELEVATED / HIGH / EXTREME
    market_implication: str
    options_implication: str
    vix_trending: str               # RISING / FALLING / STABLE
    contrarian_signal: Optional[str]  # POTENTIAL_BOTTOM / COMPLACENCY_WARNING / None


@dataclass
class VolatilitySummary:
    """Composite volatility assessment from all sub-indicators.

    ``position_size_modifier`` scales the default position size: 1.0 = normal,
    < 1.0 = reduce (high vol), > 1.0 = increase (low vol with trend).
    """

    timestamp: datetime

    # Bollinger
    bb_position: str
    bb_squeeze: bool
    bb_bandwidth_percentile: float

    # ATR
    atr_value: float
    atr_pct: float
    volatility_level: str
    suggested_sl: float
    suggested_target: float

    # Historical Volatility
    hv_current: float
    hv_regime: str

    # VIX (if available)
    vix_regime: Optional[str]

    # Overall
    volatility_vote: str            # LOW / NORMAL / HIGH / EXTREME
    volatility_confidence: float

    # Actionable insights
    position_size_modifier: float
    breakout_alert: bool
    mean_reversion_setup: bool


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


_SESSION_MINUTES = 375  # NSE cash/F&O session 09:15-15:30


def _periods_per_year(df: pd.DataFrame) -> tuple[float, Optional[float]]:
    """Return (bars per year, bar interval in minutes) inferred from timestamps."""
    if "timestamp" not in df.columns or len(df) < 3:
        return 252.0, None
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    diffs = ts.diff().dt.total_seconds().dropna() / 60.0
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 252.0, None
    interval = float(diffs.median())
    if interval >= 24 * 60 * 0.9:
        return 252.0, interval
    return 252.0 * _SESSION_MINUTES / max(interval, 1.0), interval


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


def _percentile_rank(series: pd.Series, lookback: int) -> float:
    """Return the percentile rank (0-100) of the last value within the last *lookback* values."""
    valid = series.dropna()
    if len(valid) < 2:
        return 50.0
    window = valid.iloc[-lookback:] if len(valid) >= lookback else valid
    current = float(window.iloc[-1])
    rank = float((window < current).sum()) / len(window) * 100
    return round(rank, 2)


# ---------------------------------------------------------------------------
# VolatilityIndicators
# ---------------------------------------------------------------------------


class VolatilityIndicators:
    """Pure-function volatility indicator calculations for OHLCV DataFrames.

    All public methods accept a DataFrame with columns
    ``[timestamp, open, high, low, close, volume]`` and return either a
    ``pd.Series`` or a result dataclass.  No state is stored on the instance.
    """

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    def calculate_bollinger(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> BollingerResult:
        """Calculate Bollinger Bands.

        Upper  = SMA(close, period) + std_dev × σ
        Middle = SMA(close, period)
        Lower  = SMA(close, period) − std_dev × σ

        ``bandwidth`` quantifies how wide the bands are relative to the middle
        band, expressed as a percentage.  ``percent_b`` shows where the close
        sits within the bands (0 = lower band, 1 = upper band).

        Args:
            df: OHLCV DataFrame.
            period: SMA look-back window (default 20).
            std_dev: Number of standard deviations for upper/lower bands (default 2).

        Returns:
            BollingerResult with all series and latest-bar summary fields.
        """
        clean = _prepare_df(df)

        if clean.empty or len(clean) < period:
            logger.warning(
                "Insufficient data for Bollinger calculation: need %d rows, got %d.",
                period,
                len(clean) if not clean.empty else 0,
            )
            empty = pd.Series(dtype=float)
            return BollingerResult(
                upper_band=empty,
                middle_band=empty,
                lower_band=empty,
                bandwidth=empty,
                percent_b=empty,
                current_position="LOWER_ZONE",
                squeeze=False,
                expansion=False,
                band_touch=None,
                mean_reversion_signal=None,
            )

        close = clean["close"]
        middle = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std(ddof=0)

        upper = middle + std_dev * rolling_std
        lower = middle - std_dev * rolling_std

        # Bandwidth: (upper - lower) / middle * 100
        bandwidth = (upper - lower) / middle * 100

        # Percent B: (close - lower) / (upper - lower)
        band_range = upper - lower
        percent_b = (close - lower) / band_range.replace(0, np.nan)

        # --- Latest-bar classifications ---
        latest_close = float(close.iloc[-1])
        valid_upper = upper.dropna()
        valid_lower = lower.dropna()
        valid_middle = middle.dropna()

        if len(valid_upper) == 0 or len(valid_lower) == 0:
            current_position = "LOWER_ZONE"
            band_touch = None
            mean_reversion_signal = None
            squeeze = False
            expansion = False
        else:
            latest_upper = float(valid_upper.iloc[-1])
            latest_lower = float(valid_lower.iloc[-1])
            latest_middle = float(valid_middle.iloc[-1])

            # Position classification
            if latest_close > latest_upper:
                current_position = "ABOVE_UPPER"
            elif latest_close > latest_middle:
                current_position = "UPPER_ZONE"
            elif latest_close < latest_lower:
                current_position = "BELOW_LOWER"
            else:
                current_position = "LOWER_ZONE"

            # Band touch: price within 0.1% of band
            touch_threshold = 0.001
            if latest_upper > 0 and abs(latest_close - latest_upper) / latest_upper <= touch_threshold:
                band_touch = "UPPER_TOUCH"
            elif latest_lower > 0 and abs(latest_close - latest_lower) / latest_lower <= touch_threshold:
                band_touch = "LOWER_TOUCH"
            else:
                band_touch = None

            # Mean reversion signal (BB position only; RSI check happens in aggregator)
            if current_position in ("BELOW_LOWER", "LOWER_ZONE") and band_touch == "LOWER_TOUCH":
                mean_reversion_signal = "BUY"
            elif current_position in ("ABOVE_UPPER", "UPPER_ZONE") and band_touch == "UPPER_TOUCH":
                mean_reversion_signal = "SELL"
            elif current_position == "BELOW_LOWER":
                mean_reversion_signal = "BUY"
            elif current_position == "ABOVE_UPPER":
                mean_reversion_signal = "SELL"
            else:
                mean_reversion_signal = None

            # Squeeze / expansion: percentile of bandwidth over last 120 bars
            valid_bw = bandwidth.dropna()
            bw_window = valid_bw.iloc[-120:] if len(valid_bw) >= 120 else valid_bw
            if len(bw_window) >= 2:
                latest_bw = float(bw_window.iloc[-1])
                pct_rank = float((bw_window < latest_bw).sum()) / len(bw_window) * 100
                squeeze = pct_rank < 20
                expansion = pct_rank > 80
            else:
                squeeze = False
                expansion = False

        return BollingerResult(
            upper_band=upper,
            middle_band=middle,
            lower_band=lower,
            bandwidth=bandwidth,
            percent_b=percent_b,
            current_position=current_position,
            squeeze=squeeze,
            expansion=expansion,
            band_touch=band_touch,
            mean_reversion_signal=mean_reversion_signal,
        )

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> ATRResult:
        """Calculate Average True Range.

        True Range = max(high − low, |high − prev_close|, |low − prev_close|)
        ATR = EMA/Wilder smoothing of True Range over *period* bars.

        ``volatility_level`` is derived from the percentile rank of the latest
        ATR within the last 100 bars.

        Args:
            df: OHLCV DataFrame.
            period: Smoothing period (default 14).

        Returns:
            ATRResult with all series and latest-bar summary fields.
        """
        clean = _prepare_df(df)

        if clean.empty or len(clean) < period + 1:
            logger.warning(
                "Insufficient data for ATR calculation: need %d rows, got %d.",
                period + 1,
                len(clean) if not clean.empty else 0,
            )
            empty = pd.Series(dtype=float)
            return ATRResult(
                atr=empty,
                current_atr=float("nan"),
                atr_pct=float("nan"),
                volatility_level="NORMAL",
                suggested_sl_distance=float("nan"),
                suggested_target_distance=float("nan"),
                expanding=False,
                contracting=False,
            )

        high = clean["high"]
        low = clean["low"]
        close = clean["close"]

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder smoothing (equivalent to EMA with alpha=1/period)
        atr_series = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        valid_atr = atr_series.dropna()
        current_atr = float(valid_atr.iloc[-1]) if len(valid_atr) > 0 else float("nan")
        latest_close = float(close.iloc[-1])
        atr_pct = round(current_atr / latest_close * 100, 4) if latest_close > 0 else float("nan")

        # Volatility level from percentile over last 100 bars
        pct = _percentile_rank(atr_series, 100)
        if pct > 95:
            volatility_level = "EXTREME"
        elif pct > 75:
            volatility_level = "HIGH"
        elif pct >= 25:
            volatility_level = "NORMAL"
        else:
            volatility_level = "LOW"

        # Suggested stop-loss / target
        suggested_sl = round(1.5 * current_atr, 2) if not np.isnan(current_atr) else float("nan")
        suggested_target = round(2.0 * current_atr, 2) if not np.isnan(current_atr) else float("nan")

        # Expanding / contracting over last 5 bars
        if len(valid_atr) >= 5:
            tail = valid_atr.iloc[-5:].values
            expanding = bool(tail[-1] > tail[0])
            contracting = bool(tail[-1] < tail[0])
        else:
            expanding = False
            contracting = False

        return ATRResult(
            atr=atr_series,
            current_atr=current_atr,
            atr_pct=atr_pct,
            volatility_level=volatility_level,
            suggested_sl_distance=suggested_sl,
            suggested_target_distance=suggested_target,
            expanding=expanding,
            contracting=contracting,
        )

    # ------------------------------------------------------------------
    # Standard Deviation
    # ------------------------------------------------------------------

    def calculate_std_dev(self, df: pd.DataFrame, period: int = 20) -> StdDevResult:
        """Calculate rolling standard deviation of close prices.

        ``normalized`` expresses the latest std dev as a percentage of the
        current close price, making it comparable across price levels.

        Args:
            df: OHLCV DataFrame.
            period: Rolling window (default 20).

        Returns:
            StdDevResult with full series and latest-bar summary fields.
        """
        clean = _prepare_df(df)

        if clean.empty or len(clean) < period:
            logger.warning(
                "Insufficient data for StdDev calculation: need %d rows, got %d.",
                period,
                len(clean) if not clean.empty else 0,
            )
            empty = pd.Series(dtype=float)
            return StdDevResult(
                std_dev=empty,
                current_value=float("nan"),
                normalized=float("nan"),
                historical_percentile=50.0,
            )

        std_series = clean["close"].rolling(window=period).std(ddof=1)
        valid = std_series.dropna()
        current_value = float(valid.iloc[-1]) if len(valid) > 0 else float("nan")
        latest_close = float(clean["close"].iloc[-1])
        normalized = round(current_value / latest_close * 100, 4) if latest_close > 0 else float("nan")
        historical_percentile = _percentile_rank(std_series, 252)

        return StdDevResult(
            std_dev=std_series,
            current_value=current_value,
            normalized=normalized,
            historical_percentile=historical_percentile,
        )

    # ------------------------------------------------------------------
    # Historical Volatility
    # ------------------------------------------------------------------

    def calculate_historical_volatility(
        self, df: pd.DataFrame, period: int = 20
    ) -> HVResult:
        """Calculate annualised Historical Volatility from log returns.

        HV = std_dev(log_returns, period) × √252 × 100

        ``hv_regime`` thresholds are calibrated for the Indian equity market:
          < 15 %  → LOW_VOL
          15-25 % → NORMAL
          25-35 % → HIGH_VOL
          > 35 %  → EXTREME

        Args:
            df: OHLCV DataFrame.
            period: Rolling window for log-return std dev (default 20).

        Returns:
            HVResult with full series and latest-bar summary fields.
        """
        clean = _prepare_df(df)

        if clean.empty or len(clean) < period + 1:
            logger.warning(
                "Insufficient data for HV calculation: need %d rows, got %d.",
                period + 1,
                len(clean) if not clean.empty else 0,
            )
            empty = pd.Series(dtype=float)
            return HVResult(
                hv=empty,
                current_hv=float("nan"),
                hv_percentile=50.0,
                hv_regime="NORMAL",
            )

        log_returns = np.log(clean["close"] / clean["close"].shift(1))
        # Annualise by the BAR interval: sqrt(252) is only right for daily
        # bars; on 5m bars it understated HV ~9x, pinning the regime to
        # LOW_VOL and inflating position size.  Intraday, the overnight gap
        # return is not a 5-minute return and is excluded.
        periods_per_year, interval_min = _periods_per_year(clean)
        if interval_min is not None and interval_min < 24 * 60 and "timestamp" in clean.columns:
            ts = pd.to_datetime(clean["timestamp"], errors="coerce")
            gap_min = ts.diff().dt.total_seconds() / 60.0
            log_returns = log_returns.where(~(gap_min > 2 * interval_min))
        hv_series = (
            log_returns.rolling(window=period, min_periods=max(2, period // 2)).std(ddof=1)
            * np.sqrt(periods_per_year) * 100
        )

        valid = hv_series.dropna()
        current_hv = round(float(valid.iloc[-1]), 2) if len(valid) > 0 else float("nan")
        hv_percentile = _percentile_rank(hv_series, 252)

        if np.isnan(current_hv):
            hv_regime = "NORMAL"
        elif current_hv < 15:
            hv_regime = "LOW_VOL"
        elif current_hv <= 25:
            hv_regime = "NORMAL"
        elif current_hv <= 35:
            hv_regime = "HIGH_VOL"
        else:
            hv_regime = "EXTREME"

        return HVResult(
            hv=hv_series,
            current_hv=current_hv,
            hv_percentile=hv_percentile,
            hv_regime=hv_regime,
        )

    # ------------------------------------------------------------------
    # VIX Interpretation
    # ------------------------------------------------------------------

    @staticmethod
    def interpret_vix(vix_value: float, vix_change_pct: float = 0.0) -> VIXInterpretation:
        """Interpret the India VIX level and derive market/options implications.

        Regime thresholds (India VIX norms):
          < 13  → LOW         (complacent, options cheap)
          13-18 → NORMAL      (balanced market)
          18-25 → ELEVATED    (increasing uncertainty)
          25-35 → HIGH        (fear rising, options expensive)
          > 35  → EXTREME     (panic, potential capitulation)

        Args:
            vix_value: Current India VIX level.
            vix_change_pct: Percentage change in VIX (positive = rising).

        Returns:
            VIXInterpretation with regime, implications, and contrarian signal.
        """
        # Regime
        if vix_value < 13:
            regime = "LOW"
            market_implication = "Low fear, range-bound market expected"
            options_implication = "Options cheap, good time to buy"
        elif vix_value < 18:
            regime = "NORMAL"
            market_implication = "Normal market conditions, balanced risk"
            options_implication = "Options fairly priced"
        elif vix_value < 25:
            regime = "ELEVATED"
            market_implication = "Increasing uncertainty, wider swings likely"
            options_implication = "Options getting expensive, hedging costs rising"
        elif vix_value < 35:
            regime = "HIGH"
            market_implication = "High fear, significant moves expected"
            options_implication = "Options expensive, consider selling premium"
        else:
            regime = "EXTREME"
            market_implication = "Extreme fear / panic, capitulation possible"
            options_implication = "Options very expensive, selling premium can be lucrative but risky"

        # Trending
        if vix_change_pct > 5:
            vix_trending = "RISING"
        elif vix_change_pct < -5:
            vix_trending = "FALLING"
        else:
            vix_trending = "STABLE"

        # Contrarian signal
        if vix_value > 30:
            contrarian_signal = "POTENTIAL_BOTTOM"
        elif vix_value < 12:
            contrarian_signal = "COMPLACENCY_WARNING"
        else:
            contrarian_signal = None

        return VIXInterpretation(
            value=vix_value,
            regime=regime,
            market_implication=market_implication,
            options_implication=options_implication,
            vix_trending=vix_trending,
            contrarian_signal=contrarian_signal,
        )

    # ------------------------------------------------------------------
    # Volatility Summary
    # ------------------------------------------------------------------

    def get_volatility_summary(
        self,
        df: pd.DataFrame,
        vix_value: Optional[float] = None,
        vix_change_pct: float = 0.0,
    ) -> VolatilitySummary:
        """Produce a composite volatility summary from all sub-indicators.

        Combines Bollinger Bands, ATR, Historical Volatility, and optionally
        India VIX into a single assessment with actionable position-sizing
        guidance and breakout/mean-reversion alerts.

        Position size modifier logic:
          EXTREME volatility → 0.3 (protect capital)
          HIGH volatility    → 0.5-0.7
          NORMAL             → 1.0
          LOW with squeeze   → 0.8 (wait for breakout direction)
          LOW with trend     → 1.2-1.5

        Args:
            df: OHLCV DataFrame.
            vix_value: Current India VIX level (optional).
            vix_change_pct: VIX percentage change (optional).

        Returns:
            VolatilitySummary with all fields populated.
        """
        now = datetime.utcnow()
        empty_summary = VolatilitySummary(
            timestamp=now,
            bb_position="LOWER_ZONE",
            bb_squeeze=False,
            bb_bandwidth_percentile=50.0,
            atr_value=float("nan"),
            atr_pct=float("nan"),
            volatility_level="NORMAL",
            suggested_sl=float("nan"),
            suggested_target=float("nan"),
            hv_current=float("nan"),
            hv_regime="NORMAL",
            vix_regime=None,
            volatility_vote="NORMAL",
            volatility_confidence=0.0,
            position_size_modifier=1.0,
            breakout_alert=False,
            mean_reversion_setup=False,
        )

        clean = _prepare_df(df)
        if clean.empty or len(clean) < 2:
            logger.warning("DataFrame too short for volatility summary.")
            return empty_summary

        timestamp = (
            pd.Timestamp(clean["timestamp"].iloc[-1]).to_pydatetime()
            if "timestamp" in clean.columns
            else now
        )

        # --- Sub-indicator calculations ------------------------------------
        bb = self.calculate_bollinger(clean)
        atr = self.calculate_atr(clean)
        hv = self.calculate_historical_volatility(clean)

        vix_regime: Optional[str] = None
        if vix_value is not None:
            vix_interp = self.interpret_vix(vix_value, vix_change_pct)
            vix_regime = vix_interp.regime

        # BB bandwidth percentile
        valid_bw = bb.bandwidth.dropna()
        if len(valid_bw) >= 2:
            bw_window = valid_bw.iloc[-120:] if len(valid_bw) >= 120 else valid_bw
            latest_bw = float(bw_window.iloc[-1])
            bb_bandwidth_percentile = round(
                float((bw_window < latest_bw).sum()) / len(bw_window) * 100, 2
            )
        else:
            bb_bandwidth_percentile = 50.0

        # --- Volatility vote -----------------------------------------------
        # Collect votes from each sub-indicator
        votes: list[str] = []

        # ATR vote
        votes.append(atr.volatility_level)

        # HV vote
        hv_map = {"LOW_VOL": "LOW", "NORMAL": "NORMAL", "HIGH_VOL": "HIGH", "EXTREME": "EXTREME"}
        votes.append(hv_map.get(hv.hv_regime, "NORMAL"))

        # BB squeeze/expansion → LOW/HIGH proxy
        if bb.squeeze:
            votes.append("LOW")
        elif bb.expansion:
            votes.append("HIGH")
        else:
            votes.append("NORMAL")

        # VIX vote
        if vix_regime is not None:
            vix_vol_map = {
                "LOW": "LOW",
                "NORMAL": "NORMAL",
                "ELEVATED": "HIGH",
                "HIGH": "HIGH",
                "EXTREME": "EXTREME",
            }
            votes.append(vix_vol_map.get(vix_regime, "NORMAL"))

        # Score: LOW=1, NORMAL=2, HIGH=3, EXTREME=4
        score_map = {"LOW": 1, "NORMAL": 2, "HIGH": 3, "EXTREME": 4}
        avg_score = sum(score_map.get(v, 2) for v in votes) / len(votes) if votes else 2.0

        if avg_score >= 3.5:
            volatility_vote = "EXTREME"
        elif avg_score >= 2.5:
            volatility_vote = "HIGH"
        elif avg_score >= 1.5:
            volatility_vote = "NORMAL"
        else:
            volatility_vote = "LOW"

        # Confidence: higher when sub-indicators agree
        vote_scores = [score_map.get(v, 2) for v in votes]
        if len(vote_scores) >= 2:
            score_std = float(np.std(vote_scores))
            volatility_confidence = round(max(0.3, 1.0 - score_std * 0.3), 2)
        else:
            volatility_confidence = 0.5

        # --- Position size modifier ----------------------------------------
        if volatility_vote == "EXTREME":
            position_size_modifier = 0.3
        elif volatility_vote == "HIGH":
            position_size_modifier = 0.6
        elif volatility_vote == "LOW":
            if bb.squeeze:
                position_size_modifier = 0.8  # wait for breakout direction
            else:
                position_size_modifier = 1.3  # low vol with trend = bigger size
        else:
            position_size_modifier = 1.0

        # --- Breakout alert: BB squeeze + ATR contracting ------------------
        breakout_alert = bb.squeeze and atr.contracting

        # --- Mean reversion setup: price at BB extreme + high vol ----------
        mean_reversion_setup = (
            bb.mean_reversion_signal is not None
            and volatility_vote in ("HIGH", "EXTREME")
        )

        return VolatilitySummary(
            timestamp=timestamp,
            bb_position=bb.current_position,
            bb_squeeze=bb.squeeze,
            bb_bandwidth_percentile=bb_bandwidth_percentile,
            atr_value=atr.current_atr,
            atr_pct=atr.atr_pct,
            volatility_level=atr.volatility_level,
            suggested_sl=atr.suggested_sl_distance,
            suggested_target=atr.suggested_target_distance,
            hv_current=hv.current_hv,
            hv_regime=hv.hv_regime,
            vix_regime=vix_regime,
            volatility_vote=volatility_vote,
            volatility_confidence=volatility_confidence,
            position_size_modifier=position_size_modifier,
            breakout_alert=breakout_alert,
            mean_reversion_setup=mean_reversion_setup,
        )
