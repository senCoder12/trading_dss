"""
Liquidity filter for F&O stocks — removes untradeable stocks before scoring.

NSE has ~200 F&O stocks but only ~80–100 are genuinely liquid. Filter is
aggressive: better to miss a good stock than include a stock that can't be
traded. Applies hard thresholds (price, volume, spread, OI, IV) plus
time-adjusted volume filters that scale expectations throughout the day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.data.fo_bulk_fetcher import StockQuote, StockOIData

logger = logging.getLogger(__name__)
_IST = ZoneInfo(IST_TIMEZONE)


@dataclass
class LiquidityThresholds:
    """Configurable liquidity filter thresholds (all from settings)."""

    # Volume filters
    min_avg_daily_volume: int = 500_000
    min_today_volume: int = 100_000
    min_volume_participation_pct: float = 15.0

    # OI filters
    min_total_oi: int = 500_000
    min_oi_change_activity: int = 10_000

    # Spread filters
    max_bid_ask_spread_pct: float = 0.5
    max_atm_options_spread_pct: float = 2.0

    # Price filters
    min_ltp: float = 100.0
    max_ltp: float = 50_000.0
    min_price_change_pct: float = 0.3

    # IV filters
    min_iv: float = 10.0
    max_iv: float = 80.0

    @staticmethod
    def load_from_settings() -> LiquidityThresholds:
        """Load thresholds from settings.filter_liquidity or use defaults."""
        filter_cfg = getattr(settings, "filter_liquidity", None)
        if not filter_cfg:
            return LiquidityThresholds()

        return LiquidityThresholds(
            min_avg_daily_volume=getattr(
                filter_cfg, "min_avg_daily_volume", 500_000
            ),
            min_today_volume=getattr(filter_cfg, "min_today_volume", 100_000),
            min_volume_participation_pct=getattr(
                filter_cfg, "min_volume_participation_pct", 15.0
            ),
            min_total_oi=getattr(filter_cfg, "min_total_oi", 500_000),
            min_oi_change_activity=getattr(
                filter_cfg, "min_oi_change_activity", 10_000
            ),
            max_bid_ask_spread_pct=getattr(
                filter_cfg, "max_bid_ask_spread_pct", 0.5
            ),
            max_atm_options_spread_pct=getattr(
                filter_cfg, "max_atm_options_spread_pct", 2.0
            ),
            min_ltp=getattr(filter_cfg, "min_ltp", 100.0),
            max_ltp=getattr(filter_cfg, "max_ltp", 50_000.0),
            min_price_change_pct=getattr(filter_cfg, "min_price_change_pct", 0.3),
            min_iv=getattr(filter_cfg, "min_iv", 10.0),
            max_iv=getattr(filter_cfg, "max_iv", 80.0),
        )


@dataclass
class FilterResult:
    """Detailed result of liquidity filter pass."""

    total_input: int
    passed: int = 0
    failed_price: int = 0
    failed_volume: int = 0
    failed_spread: int = 0
    failed_oi: int = 0
    failed_iv: int = 0
    failed_no_options: int = 0
    passed_symbols: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(_IST))

    def summary(self) -> str:
        """Return one-line summary for logging."""
        pct = (100 * self.passed / self.total_input) if self.total_input > 0 else 0
        return (
            f"in={self.total_input} passed={self.passed} ({pct:.1f}%) "
            f"[price={self.failed_price} vol={self.failed_volume} "
            f"spread={self.failed_spread} oi={self.failed_oi} "
            f"iv={self.failed_iv} no_options={self.failed_no_options}]"
        )


class LiquidityFilter:
    """Filter removes untradeable stocks before scoring."""

    def __init__(self, thresholds: LiquidityThresholds = None) -> None:
        """
        Initialize liquidity filter.

        Parameters
        ----------
        thresholds : LiquidityThresholds, optional
            Custom thresholds. If None, loads from settings.
        """
        self._thresholds = thresholds or LiquidityThresholds.load_from_settings()
        self._time_of_day_adjustments = True

    # ── Public API ────────────────────────────────────────────────────────────

    def apply_hard_filters(
        self,
        quotes: dict[str, StockQuote],
        oi_data: dict[str, StockOIData],
    ) -> list[str]:
        """
        Apply strict hard filters. Stock must PASS every single filter.

        Parameters
        ----------
        quotes : dict[str, StockQuote]
            Map of symbol → StockQuote
        oi_data : dict[str, StockOIData]
            Map of symbol → StockOIData

        Returns
        -------
        list[str]:
            Symbols that passed all hard filters.
        """
        result = FilterResult(total_input=len(quotes))
        passed = []

        for symbol, quote in quotes.items():
            failure_reason = self._check_hard_filters(symbol, quote, oi_data)
            if failure_reason is None:
                passed.append(symbol)
                result.passed += 1
            else:
                result.__dict__[f"failed_{failure_reason}"] += 1

        result.passed_symbols = passed
        logger.info("hard_filters: %s", result.summary())
        return passed

    def apply_time_adjusted_volume_filter(
        self,
        quotes: dict[str, StockQuote],
        current_time: Optional[time] = None,
    ) -> list[str]:
        """
        Filter by volume participation, adjusted for market time.

        Volume builds throughout the day. Earlier hours expect lower participation.

        Parameters
        ----------
        quotes : dict[str, StockQuote]
            Map of symbol → StockQuote
        current_time : time, optional
            Current time for adjustment. If None, uses system time.

        Returns
        -------
        list[str]:
            Symbols meeting time-adjusted volume participation threshold.
        """
        if current_time is None:
            current_time = datetime.now(_IST).time()

        expected_participation = self._get_time_adjusted_participation(current_time)
        passed = []

        for symbol, quote in quotes.items():
            if quote.avg_volume_20d <= 0:
                continue

            participation = 100.0 * quote.volume / quote.avg_volume_20d
            if participation >= expected_participation:
                passed.append(symbol)

        logger.info(
            "volume_adjustment [time=%s expected_pct=%.1f%%]: "
            "%d/%d passed",
            current_time.strftime("%H:%M"),
            expected_participation,
            len(passed),
            len(quotes),
        )
        return passed

    def get_liquid_symbols(
        self,
        quotes: dict[str, StockQuote],
        oi_data: dict[str, StockOIData],
        current_time: Optional[time] = None,
    ) -> list[str]:
        """
        Master filter: hard filters + time-adjusted volume.

        Parameters
        ----------
        quotes : dict[str, StockQuote]
            Map of symbol → StockQuote
        oi_data : dict[str, StockOIData]
            Map of symbol → StockOIData
        current_time : time, optional
            Current time for volume adjustment. If None, uses system time.

        Returns
        -------
        list[str]:
            Liquid symbols ready for scoring (typically 60–120).
        """
        # Step 1: hard filters
        hard_passed = self.apply_hard_filters(quotes, oi_data)
        hard_passed_quotes = {s: quotes[s] for s in hard_passed}

        # Step 2: time-adjusted volume (only on survivors from step 1)
        if self._time_of_day_adjustments:
            liquid = self.apply_time_adjusted_volume_filter(
                hard_passed_quotes, current_time
            )
        else:
            liquid = hard_passed

        logger.info(
            "get_liquid_symbols: %d input → %d hard_pass → %d final",
            len(quotes),
            len(hard_passed),
            len(liquid),
        )
        return liquid

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_hard_filters(
        self,
        symbol: str,
        quote: StockQuote,
        oi_data: dict[str, StockOIData],
    ) -> Optional[str]:
        """
        Check all hard filters. Return failure reason or None if passes all.

        Order matters for logging/counting — check in this order:
        price, volume, spread, no_options, oi, iv
        """
        # Price filters
        if quote.ltp < self._thresholds.min_ltp or quote.ltp > self._thresholds.max_ltp:
            return "price"

        # Volume filters
        if quote.avg_volume_20d < self._thresholds.min_avg_daily_volume:
            return "volume"

        # Spread filters
        if quote.bid_ask_spread_pct > self._thresholds.max_bid_ask_spread_pct:
            return "spread"

        # OI / IV filters — only applied when oi_data is populated (post-fetch stage)
        if oi_data:
            # Must have options data
            if symbol not in oi_data:
                return "no_options"

            # OI filters
            oi = oi_data[symbol]
            total_oi = oi.total_ce_oi + oi.total_pe_oi
            if total_oi < self._thresholds.min_total_oi:
                return "oi"

            # IV filters
            if oi.avg_iv < self._thresholds.min_iv or oi.avg_iv > self._thresholds.max_iv:
                return "iv"

        # Price movement filter — only applied during active market hours
        # (pre-market selection or weekend runs should not exclude zero-move stocks)
        _now = datetime.now(_IST)
        _market_open = (
            _now.weekday() < 5  # Monday–Friday
            and time(9, 15) <= _now.time() <= time(15, 30)
        )
        if _market_open and abs(quote.price_change_pct) < self._thresholds.min_price_change_pct:
            return "price"

        return None

    def _get_time_adjusted_participation(self, current_time: time) -> float:
        """
        Return expected volume participation % for given time of day.

        Times in IST.
        """
        # 09:15–10:00 AM → 5% (early, low volume normal)
        if current_time < time(10, 0):
            return 5.0
        # 10:00–11:30 AM → 20%
        if current_time < time(11, 30):
            return 20.0
        # 11:30–01:00 PM → 40%
        if current_time < time(13, 0):
            return 40.0
        # 01:00–02:30 PM → 60%
        if current_time < time(14, 30):
            return 60.0
        # 02:30–03:30 PM → 80%
        return 80.0
