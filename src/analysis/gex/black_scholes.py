"""
Pure Black-Scholes pricing engine and Greeks calculator.

No market data, no API calls — only math. Every other GEX module imports
from here. Accuracy is critical: errors propagate into all downstream GEX
calculations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Literal

import numpy as np
from scipy.stats import norm
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Try to load risk-free rate from project settings; fall back to 6.7% T-bill.
# ---------------------------------------------------------------------------
try:
    from config.settings import settings as _settings

    _DEFAULT_RFR: float = getattr(_settings, "risk_free_rate", 0.067)
except Exception:  # settings not available in test environments
    _DEFAULT_RFR = 0.067


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OptionGreeks:
    """Full Greeks result from Black-Scholes calculation."""

    delta: float
    gamma: float
    theta: float        # daily decay, in rupees (negative for long options)
    vega: float         # per 1% IV move, in rupees
    rho: float          # per 1% rate move
    iv: float           # annualized decimal, e.g. 0.18
    option_type: str    # 'CE' or 'PE'
    strike: float
    spot: float
    dte: float          # days to expiry (fractional)
    intrinsic_value: float
    time_value: float
    is_itm: bool


@dataclass
class BSMInputs:
    """Validated inputs for Black-Scholes calculations."""

    spot: float
    strike: float
    dte_days: float     # days to expiry; 0 → same-day expiry
    iv: float           # annualized decimal, e.g. 0.18
    risk_free_rate: float
    option_type: str    # 'CE' or 'PE'

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be > 0, got {self.spot}")
        if self.strike <= 0:
            raise ValueError(f"strike must be > 0, got {self.strike}")
        if self.dte_days < 0:
            raise ValueError(f"dte_days must be >= 0, got {self.dte_days}")
        if not (0 < self.iv < 10):
            raise ValueError(f"iv must be in (0, 10), got {self.iv}")
        if self.option_type not in ("CE", "PE"):
            raise ValueError(f"option_type must be 'CE' or 'PE', got {self.option_type!r}")
        # Floor at 30 minutes equivalent to avoid division by zero
        if self.dte_days == 0:
            object.__setattr__(self, "dte_days", 0.5 / (24 * 60))


# ---------------------------------------------------------------------------
# BlackScholesEngine
# ---------------------------------------------------------------------------


class BlackScholesEngine:
    """Black-Scholes pricing engine for Indian F&O options."""

    MIN_IV = 0.01
    MAX_IV = 5.0
    IV_PRECISION = 0.00001
    MAX_IV_ITERATIONS = 100

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _d1(self, inputs: BSMInputs) -> float:
        T = inputs.dte_days / 365.0
        return (
            math.log(inputs.spot / inputs.strike)
            + (inputs.risk_free_rate + 0.5 * inputs.iv ** 2) * T
        ) / (inputs.iv * math.sqrt(T))

    def _d2(self, inputs: BSMInputs) -> float:
        T = inputs.dte_days / 365.0
        return self._d1(inputs) - inputs.iv * math.sqrt(T)

    # ------------------------------------------------------------------
    # Price
    # ------------------------------------------------------------------

    def calculate_price(self, inputs: BSMInputs) -> float:
        """Black-Scholes theoretical price for a European option."""
        T = inputs.dte_days / 365.0
        d1 = self._d1(inputs)
        d2 = self._d2(inputs)
        disc = math.exp(-inputs.risk_free_rate * T)

        if inputs.option_type == "CE":
            price = inputs.spot * norm.cdf(d1) - inputs.strike * disc * norm.cdf(d2)
        else:
            price = inputs.strike * disc * norm.cdf(-d2) - inputs.spot * norm.cdf(-d1)

        return max(price, 0.0)

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def calculate_greeks(self, inputs: BSMInputs) -> OptionGreeks:
        """Calculate all Greeks for the given inputs."""
        T = inputs.dte_days / 365.0
        sqrt_T = math.sqrt(T)
        d1 = self._d1(inputs)
        d2 = self._d2(inputs)
        disc = math.exp(-inputs.risk_free_rate * T)
        n_prime_d1 = norm.pdf(d1)

        # --- gamma (identical for CE and PE) ---
        gamma = n_prime_d1 / (inputs.spot * inputs.iv * sqrt_T)

        # --- delta ---
        if inputs.option_type == "CE":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1.0

        # --- theta (per calendar day, in rupees) ---
        decay_component = -(inputs.spot * n_prime_d1 * inputs.iv) / (2.0 * sqrt_T)
        if inputs.option_type == "CE":
            theta = (
                decay_component
                - inputs.risk_free_rate * inputs.strike * disc * norm.cdf(d2)
            ) / 365.0
        else:
            theta = (
                decay_component
                + inputs.risk_free_rate * inputs.strike * disc * norm.cdf(-d2)
            ) / 365.0

        # --- vega (per 1% IV move) ---
        vega = inputs.spot * n_prime_d1 * sqrt_T / 100.0

        # --- rho (per 1% rate move) ---
        if inputs.option_type == "CE":
            rho = inputs.strike * T * disc * norm.cdf(d2) / 100.0
        else:
            rho = -inputs.strike * T * disc * norm.cdf(-d2) / 100.0

        # --- intrinsic / time value ---
        if inputs.option_type == "CE":
            intrinsic = max(inputs.spot - inputs.strike, 0.0)
            is_itm = inputs.spot > inputs.strike
        else:
            intrinsic = max(inputs.strike - inputs.spot, 0.0)
            is_itm = inputs.spot < inputs.strike

        option_price = self.calculate_price(inputs)
        time_value = option_price - intrinsic

        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            iv=inputs.iv,
            option_type=inputs.option_type,
            strike=inputs.strike,
            spot=inputs.spot,
            dte=inputs.dte_days,
            intrinsic_value=intrinsic,
            time_value=time_value,
            is_itm=is_itm,
        )

    # ------------------------------------------------------------------
    # IV solver
    # ------------------------------------------------------------------

    def calculate_iv(
        self,
        market_price: float,
        inputs_without_iv: BSMInputs,
        initial_guess: float = 0.20,
    ) -> float:
        """Back-solve implied volatility from a market price via Newton-Raphson."""
        iv = max(self.MIN_IV, min(self.MAX_IV, initial_guess))

        for _ in range(self.MAX_IV_ITERATIONS):
            try:
                test_inputs = replace(inputs_without_iv, iv=iv)
            except Exception:
                break

            theoretical = self.calculate_price(test_inputs)
            greeks = self.calculate_greeks(test_inputs)
            # vega stored per 1% move; convert back to per-unit for NR step
            vega = greeks.vega * 100.0

            price_diff = theoretical - market_price

            if abs(price_diff) < self.IV_PRECISION:
                return iv

            if abs(vega) < 1e-10:
                break

            iv = iv - price_diff / vega
            iv = max(self.MIN_IV, min(self.MAX_IV, iv))

        return self._bisection_iv(market_price, inputs_without_iv)

    def _bisection_iv(
        self,
        market_price: float,
        inputs_without_iv: BSMInputs,
    ) -> float:
        """Fallback IV solver using bisection — slower but guaranteed to converge."""
        low, high = self.MIN_IV, self.MAX_IV

        for _ in range(self.MAX_IV_ITERATIONS * 2):
            mid = (low + high) / 2.0
            try:
                test_inputs = replace(inputs_without_iv, iv=mid)
            except Exception:
                break
            price = self.calculate_price(test_inputs)

            if abs(price - market_price) < self.IV_PRECISION * 10:
                return mid

            if price < market_price:
                low = mid
            else:
                high = mid

        return (low + high) / 2.0

    # ------------------------------------------------------------------
    # GEX helper
    # ------------------------------------------------------------------

    def calculate_gamma_dollars(
        self,
        inputs: BSMInputs,
        lot_size: int,
        oi: int,
    ) -> float:
        """
        Rupee gamma exposure for a strike given its OI.

        Formula: gamma × spot² × 0.01 × lot_size × OI

        Represents how many rupees of delta change per 1% spot move across
        all open contracts at this strike.  Sign (call vs put) is applied by
        the GEX calculator — always positive here.
        """
        greeks = self.calculate_greeks(inputs)
        gamma_dollars = greeks.gamma * (inputs.spot ** 2) * 0.01 * lot_size * oi
        return abs(gamma_dollars)


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def get_risk_free_rate() -> float:
    """India 91-day T-bill rate proxy. Update quarterly."""
    return _DEFAULT_RFR


def get_dte(expiry_date: date, current_datetime: datetime) -> float:
    """
    Fractional days to expiry, measured to 15:30 IST on expiry_date.

    Returns minimum of 30 minutes (expressed in fractional days) to avoid
    division-by-zero in same-session expiry calculations.
    """
    market_close = datetime(
        expiry_date.year,
        expiry_date.month,
        expiry_date.day,
        15, 30, 0,
        tzinfo=_IST,
    )

    # Ensure current_datetime is timezone-aware in IST
    if current_datetime.tzinfo is None:
        current_datetime = current_datetime.replace(tzinfo=_IST)

    remaining_seconds = (market_close - current_datetime).total_seconds()
    dte = remaining_seconds / (365.25 * 24.0 * 3600.0)

    # Minimum: 30 minutes expressed as fractional days
    min_dte = 30.0 / (365.0 * 24.0 * 60.0)
    return max(dte, min_dte)
