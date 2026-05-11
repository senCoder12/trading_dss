"""
Application settings loaded from environment variables.

All configurable parameters are defined here using pydantic BaseSettings.
Values are read from .env file with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, available_timezones

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Project root — two levels up from this file (config/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file explicitly so that nested pydantic-settings models can read them
load_dotenv(PROJECT_ROOT / ".env")


class MarketHoursSettings(BaseSettings):
    """Indian market session timings (all times in IST / Asia/Kolkata)."""

    pre_market_open: str = Field(default="09:00", description="Pre-market session start HH:MM")
    market_open: str = Field(default="09:15", description="Regular session open HH:MM")
    market_close: str = Field(default="15:30", description="Regular session close HH:MM")
    post_market_close: str = Field(default="16:00", description="Post-market session end HH:MM")

    model_config = {"env_prefix": "MARKET_HOURS_", "extra": "ignore"}


class DatabaseSettings(BaseSettings):
    """SQLite database configuration."""

    path: Path = Field(
        default=PROJECT_ROOT / "data" / "db" / "trading_dss.db",
        description="Absolute path to SQLite database file",
    )
    wal_mode: bool = Field(default=True, description="Enable WAL journal mode for concurrency")
    connection_pool_size: int = Field(default=5, ge=1, le=20)
    timeout_seconds: int = Field(default=30, ge=5, le=120)

    model_config = {"env_prefix": "DB_", "extra": "ignore"}


class RateLimitSettings(BaseSettings):
    """Per-source HTTP rate limit configuration."""

    nse_requests_per_minute: int = Field(default=30, ge=1, le=120)
    bse_requests_per_minute: int = Field(default=20, ge=1, le=120)
    yfinance_requests_per_minute: int = Field(default=60, ge=1, le=120)
    news_requests_per_minute: int = Field(default=30, ge=1, le=120)

    model_config = {"env_prefix": "RATE_LIMIT_", "extra": "ignore"}


class PollingSettings(BaseSettings):
    """Data refresh intervals in seconds."""

    price_interval: int = Field(default=45, ge=15, description="Spot price refresh interval (s)")
    options_interval: int = Field(default=180, ge=60, description="Options chain refresh interval (s)")
    news_interval: int = Field(default=180, ge=60, description="News/RSS refresh interval (s)")
    fii_dii_interval: int = Field(default=900, ge=300, description="FII/DII data refresh interval (s)")
    vix_interval: int = Field(default=60, ge=15, description="VIX refresh interval (s)")
    technical_interval: int = Field(default=180, ge=60, description="Technical indicator recompute interval (s)")
    decision_interval: int = Field(default=180, ge=60, description="Decision engine cycle interval (s)")
    fo_universe_refresh_hours: int = Field(default=24, ge=1, description="F&O universe refresh interval (hours)")

    model_config = {"env_prefix": "POLL_", "extra": "ignore"}


ANOMALY_THRESHOLDS: dict[str, dict[str, float | int]] = {
    "intraday": {
        # Volume thresholds
        "volume_zscore_elevated": 2.0,
        "volume_zscore_high": 3.0,
        "volume_zscore_extreme": 4.0,
        "volume_acceleration_multiplier": 3.0,
        "absorption_body_ratio": 0.3,

        # Price thresholds
        "price_gap_threshold_pct": 0.5,
        "large_move_multiplier": 2.0,
        "extreme_move_multiplier": 3.0,
        "reversal_move_pct": 1.0,
        "reversal_retrace_pct": 0.7,
        "range_expansion_multiplier": 1.5,
        "compression_multiplier": 0.3,

        # OI thresholds
        "oi_spike_zscore_medium": 2.5,
        "oi_spike_zscore_high": 4.0,
        "oi_concentration_change_pct": 20,
        "oi_one_sided_ratio": 3.0,

        # PCR thresholds
        "pcr_zscore_extreme": 2.0,
        "pcr_rapid_shift_threshold": 0.15,
        "pcr_absolute_high": 1.5,
        "pcr_absolute_low": 0.5,

        # Max Pain thresholds
        "max_pain_jump_pct": 1.0,
        "spot_max_pain_diverge_pct": 2.0,

        # IV thresholds
        "iv_crush_pct": 20,
        "iv_explosion_pct": 30,
        "iv_skew_extreme_multiplier": 2.0,

        # Cooldown durations (seconds)
        "cooldown_low_seconds": 1800,
        "cooldown_medium_seconds": 900,
        "cooldown_high_seconds": 600,
    },
    "daily": {
        # Volume — wider thresholds for daily bars
        "volume_zscore_elevated": 2.5,
        "volume_zscore_high": 3.5,
        "volume_zscore_extreme": 5.0,
        "volume_acceleration_multiplier": 4.0,
        "absorption_body_ratio": 0.2,

        # Price — daily moves are naturally larger
        "price_gap_threshold_pct": 1.5,
        "large_move_multiplier": 2.5,
        "extreme_move_multiplier": 4.0,
        "reversal_move_pct": 2.0,
        "reversal_retrace_pct": 1.5,
        "range_expansion_multiplier": 2.0,
        "compression_multiplier": 0.4,

        # OI — daily OI changes are bigger
        "oi_spike_zscore_medium": 3.0,
        "oi_spike_zscore_high": 5.0,
        "oi_concentration_change_pct": 30,
        "oi_one_sided_ratio": 4.0,

        # PCR
        "pcr_zscore_extreme": 2.5,
        "pcr_rapid_shift_threshold": 0.25,
        "pcr_absolute_high": 1.7,
        "pcr_absolute_low": 0.4,

        # Max Pain
        "max_pain_jump_pct": 2.0,
        "spot_max_pain_diverge_pct": 3.0,

        # IV
        "iv_crush_pct": 25,
        "iv_explosion_pct": 40,
        "iv_skew_extreme_multiplier": 2.5,

        # Cooldown — one bar IS one day, so cooldowns are much longer
        "cooldown_low_seconds": 86400,
        "cooldown_medium_seconds": 86400,
        "cooldown_high_seconds": 43200,
    },
}


def get_anomaly_thresholds(timeframe: str) -> dict:
    """Return the appropriate anomaly threshold profile for a given timeframe.

    Intraday timeframes (1m, 3m, 5m, 15m, 30m, 1h) use tighter thresholds.
    Daily and above (1d, 1w, 1M) use wider thresholds to avoid false positives.

    Args:
        timeframe: Bar timeframe string, e.g. "5m", "1h", "1d"

    Returns:
        Dict of threshold key-value pairs for the appropriate profile.
    """
    intraday_timeframes = {"1m", "3m", "5m", "15m", "30m", "1h"}
    if timeframe in intraday_timeframes:
        return ANOMALY_THRESHOLDS["intraday"]
    return ANOMALY_THRESHOLDS["daily"]



class SignalThresholds(BaseSettings):
    """Thresholds that trigger trading signal conditions."""

    oi_spike_threshold: float = Field(
        default=10.0, ge=1.0,
        description="OI change % to flag as spike",
    )
    volume_spike_multiplier: float = Field(
        default=1.5, ge=1.0,
        description="Multiple of 20-day avg volume to flag spike",
    )
    pcr_extreme_low: float = Field(
        default=0.7, ge=0.1,
        description="PCR below this is considered extreme bullish",
    )
    pcr_extreme_high: float = Field(
        default=1.3, le=5.0,
        description="PCR above this is considered extreme bearish",
    )
    vix_low_threshold: float = Field(default=12.0, description="VIX below this → complacency")
    vix_high_threshold: float = Field(default=20.0, description="VIX above this → fear zone")
    vix_extreme_threshold: float = Field(default=30.0, description="VIX above this → extreme fear")
    # Regime classification thresholds (used by VIXTracker.get_vix_regime)
    vix_normal_threshold: float = Field(default=13.0, description="VIX below this → LOW_VOL regime")
    vix_elevated_threshold: float = Field(default=18.0, description="VIX above this → ELEVATED regime")
    vix_panic_threshold: float = Field(default=25.0, description="VIX above this → HIGH_VOL regime")
    sentiment_bullish_threshold: float = Field(default=0.3, description="Sentiment score for bullish")
    sentiment_bearish_threshold: float = Field(default=-0.3, description="Sentiment score for bearish")
    rsi_overbought: float = Field(default=70.0, ge=50.0, le=90.0)
    rsi_oversold: float = Field(default=30.0, ge=10.0, le=50.0)

    model_config = {"env_prefix": "THRESHOLD_", "extra": "ignore"}


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""

    bot_token: Optional[str] = Field(default=None, description="Telegram bot API token")
    chat_id: Optional[str] = Field(default=None, description="Target chat/channel ID")
    alert_min_confidence: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum signal confidence to send alert",
    )

    model_config = {"env_prefix": "TELEGRAM_", "extra": "ignore"}


class TimezoneValidator:
    """Validator for timezone configuration."""

    @staticmethod
    def validate_timezone(v: str) -> str:
        """Ensure timezone is valid and matches IST assumption."""
        if v not in available_timezones():
            raise ValueError(f"Timezone '{v}' is not valid")
        if v != "Asia/Kolkata":
            raise ValueError(
                f"Timezone must be 'Asia/Kolkata' for Indian markets; got '{v}'. "
                "Change TIMEZONE env var only if you understand market hours implications."
            )
        try:
            ZoneInfo(v)
        except Exception as e:
            raise ValueError(f"Timezone '{v}' cannot be loaded: {e}")
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Root log level")
    log_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "logs",
        description="Directory where log files are written",
    )
    max_bytes: int = Field(default=10 * 1024 * 1024, description="Max log file size (10 MB)")
    backup_count: int = Field(default=5, description="Number of rotated log backups")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log level must be one of {valid}")
        return v.upper()

    model_config = {"env_prefix": "LOG_", "extra": "ignore"}


class TradingSettings(BaseSettings):
    """F&O universe and trading scope configuration."""

    restrict_to_major_indices: bool = Field(
        default=False,
        description="If True, trade only NIFTY50, BANKNIFTY, FINNIFTY (highest liquidity). If False, trade all F&O indices.",
    )

    model_config = {"env_prefix": "TRADING_", "extra": "ignore"}


class LiquiditySettings(BaseSettings):
    """SMC / liquidity pool detector configuration."""

    equal_high_low_tolerance_pct: float = Field(
        default=0.05,
        ge=0.001, le=1.0,
        description="Two swing highs/lows within this % are considered equal (0.05 = 0.05%)",
    )
    min_level_strength: int = Field(
        default=2, ge=1,
        description="Minimum touch count for a level to be tracked",
    )
    max_levels_per_side: int = Field(
        default=5, ge=1, le=20,
        description="Max BSL and SSL levels kept per symbol",
    )
    level_expiry_hours: int = Field(
        default=48, ge=1,
        description="Levels older than this many hours are pruned",
    )
    round_number_increments: list[int] = Field(
        default_factory=lambda: [50, 100, 500, 1000],
        description="Round number increments to scan (e.g. NIFTY multiples)",
    )
    round_number_range_pct: float = Field(
        default=3.0, ge=0.5, le=10.0,
        description="Scan round numbers within this % above/below current price",
    )
    multi_touch_tolerance_pct: float = Field(
        default=0.1, ge=0.01, le=2.0,
        description="Tolerance for multi-touch level clustering (0.1 = 0.1%)",
    )
    bias_proximity_ratio: float = Field(
        default=0.6, ge=0.1, le=0.9,
        description="If distance_to_ssl < distance_to_bsl * ratio → BEARISH bias (and vice versa)",
    )

    model_config = {"env_prefix": "LIQUIDITY_", "extra": "ignore"}


class SweepSettings(BaseSettings):
    """Liquidity sweep detector configuration."""

    sweep_confirmation_candles: int = Field(
        default=3, ge=1, le=10,
        description="Candles to wait for reversal confirmation after potential sweep",
    )
    min_wick_beyond_pct: float = Field(
        default=0.03, ge=0.001, le=1.0,
        description="Minimum wick beyond level to qualify as sweep (% of price)",
    )
    max_wick_beyond_pct: float = Field(
        default=0.5, ge=0.05, le=5.0,
        description="If wick exceeds this %, likely a real breakout not a sweep",
    )
    reversion_close_threshold_pct: float = Field(
        default=0.1, ge=0.01, le=1.0,
        description="Candle must close back inside level by at least this % to confirm sweep",
    )
    volume_spike_ratio: float = Field(
        default=1.5, ge=1.0, le=10.0,
        description="Sweep candle volume must exceed this multiple of recent average",
    )
    min_reversal_body_pct: float = Field(
        default=0.3, ge=0.1, le=0.9,
        description="Reversal candle body must be >= this fraction of its total range",
    )

    model_config = {"env_prefix": "SWEEP_", "extra": "ignore"}


class StopHuntSettings(BaseSettings):
    """Stop hunt detector configuration."""

    stop_hunt_window_candles: int = Field(
        default=5, ge=1, le=20,
        description="Candles to watch after potential hunt for reversal confirmation",
    )
    min_reversal_strength_pct: float = Field(
        default=0.2, ge=0.01, le=2.0,
        description="Reversal must recover at least this % from the hunt extreme",
    )
    widely_watched_level_min_strength: int = Field(
        default=3, ge=1, le=5,
        description="Level strength threshold — only hunt levels retail is watching",
    )
    futures_basis_change_threshold: float = Field(
        default=2.0, ge=0.1, le=20.0,
        description="Futures basis change in points that signals institutional intent",
    )

    model_config = {"env_prefix": "STOP_HUNT_", "extra": "ignore"}


class ImbalanceSettings(BaseSettings):
    """FVG and Order Block detector configuration."""

    min_fvg_size_pct: float = Field(
        default=0.05, ge=0.001, le=5.0,
        description="FVG must be at least this % of price to be meaningful",
    )
    max_fvg_age_hours: int = Field(
        default=24, ge=1, le=168,
        description="FVGs older than this many hours are ignored",
    )
    min_ob_displacement_pct: float = Field(
        default=0.3, ge=0.05, le=5.0,
        description="Combined displacement after OB candle must be >= this % of price",
    )
    max_ob_age_hours: int = Field(
        default=48, ge=1, le=336,
        description="Order blocks older than this many hours are ignored",
    )
    fvg_fill_threshold_pct: float = Field(
        default=50.0, ge=1.0, le=100.0,
        description="Price entered this % of gap = partial fill",
    )
    ob_mitigation_threshold_pct: float = Field(
        default=100.0, ge=50.0, le=100.0,
        description="Price must trade through this % of OB to mitigate it",
    )

    model_config = {"env_prefix": "IMBALANCE_", "extra": "ignore"}


class VolumeProfileSettings(BaseSettings):
    """Intraday volume profile configuration."""

    tick_size: float = Field(
        default=0.05, ge=0.01, le=1.0,
        description="Minimum price increment (NSE default = 0.05)",
    )
    profile_bucket_size: int = Field(
        default=10, ge=1, le=100,
        description="Price bucket size in points (10 = 10-point buckets for NIFTY)",
    )
    value_area_pct: float = Field(
        default=0.70, ge=0.5, le=0.95,
        description="Fraction of total volume defining the value area (0.70 = 70%)",
    )
    hvn_threshold_pct: float = Field(
        default=1.5, ge=1.0, le=5.0,
        description="Bucket volume > this multiple of average = High Volume Node",
    )
    lvn_threshold_pct: float = Field(
        default=0.4, ge=0.01, le=0.9,
        description="Bucket volume < this multiple of average = Low Volume Node",
    )
    min_candles_for_profile: int = Field(
        default=30, ge=5, le=390,
        description="Minimum 1-min candles before calculating a profile",
    )

    model_config = {"env_prefix": "VP_", "extra": "ignore"}


class GEXSettings(BaseSettings):
    """Gamma Exposure (GEX) calculator configuration."""

    gex_strikes_range: int = Field(
        default=20, ge=5, le=50,
        description="Analyze ±N strikes from ATM (NIFTY 50-pt spacing → ±1000 pts)",
    )
    gex_expiries_to_include: int = Field(
        default=2, ge=1, le=4,
        description="Number of weekly expiries to include (nearest gets full weight)",
    )
    gex_refresh_minutes: int = Field(
        default=5, ge=1, le=30,
        description="GEX recalculation interval during market hours (minutes)",
    )
    near_expiry_weight: float = Field(
        default=1.0, ge=0.1, le=1.0,
        description="Weight applied to current-week expiry GEX",
    )
    next_expiry_weight: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Weight applied to next-week expiry GEX",
    )
    atm_threshold_pct: float = Field(
        default=0.5, ge=0.1, le=2.0,
        description="Strikes within this %% of spot are classified as ATM",
    )
    call_wall_min_gex_cr: float = Field(
        default=50.0, ge=1.0,
        description="Minimum CE GEX (crore rupees) for a strike to qualify as call wall",
    )

    model_config = {"env_prefix": "GEX_", "extra": "ignore"}


class BrokerSettings(BaseSettings):
    """Broker API configuration. Credentials loaded from .env without any prefix."""

    use_broker_api: bool = Field(default=False, description="Master switch — False = use NSE scraper")
    broker_provider: str = Field(default="angel_one", description="angel_one | zerodha | upstox")

    # Angel One SmartAPI credentials
    angel_one_api_key: str = Field(default="", description="SmartAPI key from smartapi.angelbroking.com")
    angel_one_client_id: str = Field(default="", description="Angel One client/login ID")
    angel_one_password: str = Field(default="", description="Angel One login password")
    angel_one_totp_secret: str = Field(default="", description="TOTP secret for 2FA (from Profile → Settings)")

    # Bulk fetcher tuning
    bulk_fetch_batch_size: int = Field(default=50, ge=1, le=100, description="Symbols per bulk quote call")
    bulk_fetch_delay_seconds: float = Field(default=0.4, ge=0.1, le=2.0, description="Delay between batch calls (s)")
    oi_fetch_top_n: int = Field(default=60, ge=10, le=200, description="Top N candidates for OI fetch")

    # WebSocket behaviour
    websocket_subscribe_depth: int = Field(default=10, ge=1, le=50, description="Strikes above/below ATM to stream")
    websocket_reconnect_delay: int = Field(default=5, ge=1, le=60, description="Seconds between reconnect attempts")
    websocket_heartbeat_interval: int = Field(default=30, ge=10, le=120, description="Heartbeat interval (s)")

    model_config = {"env_prefix": "", "extra": "ignore", "env_file": str(PROJECT_ROOT / ".env")}


class Settings(BaseSettings):
    """
    Root application settings.

    Loads from .env file in project root. All nested settings groups
    delegate to their own env prefixes.
    """

    # ── Application identity ────────────────────────────────────────────────
    app_name: str = Field(default="trading_dss", description="Application name")
    environment: str = Field(default="development", description="development | staging | production")
    debug: bool = Field(default=False, description="Enable debug mode")
    timezone: str = Field(default="Asia/Kolkata", description="Market timezone (IST assumed)")

    # ── File paths ──────────────────────────────────────────────────────────
    config_dir: Path = Field(default=PROJECT_ROOT / "config")
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "cache")

    @field_validator("timezone")
    @classmethod
    def validate_tz(cls, v: str) -> str:
        return TimezoneValidator.validate_timezone(v)

    # ── Derived config file paths ───────────────────────────────────────────
    @property
    def indices_config_path(self) -> Path:
        return self.config_dir / "indices.json"

    @property
    def news_mappings_path(self) -> Path:
        return self.config_dir / "news_mappings.json"

    @property
    def rss_feeds_path(self) -> Path:
        return self.config_dir / "rss_feeds.json"

    @property
    def sentiment_keywords_path(self) -> Path:
        return self.config_dir / "sentiment_keywords.json"

    # ── Feature flags ───────────────────────────────────────────────────────
    # News engine fetches RSS/events and drives regime event_modifier, but its
    # sentiment vote is NOT wired into signal weights (adds lag, not edge).
    # Set True only to re-evaluate after validating news signal quality.
    news_signal_enabled: bool = Field(default=False, description="Wire news vote into signal weight aggregation")

    # ── Nested settings ─────────────────────────────────────────────────────
    market_hours: MarketHoursSettings = Field(default_factory=MarketHoursSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    rate_limits: RateLimitSettings = Field(default_factory=RateLimitSettings)
    polling: PollingSettings = Field(default_factory=PollingSettings)
    thresholds: SignalThresholds = Field(default_factory=SignalThresholds)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    liquidity: LiquiditySettings = Field(default_factory=LiquiditySettings)
    sweep: SweepSettings = Field(default_factory=SweepSettings)
    stop_hunt: StopHuntSettings = Field(default_factory=StopHuntSettings)
    imbalance: ImbalanceSettings = Field(default_factory=ImbalanceSettings)
    volume_profile: VolumeProfileSettings = Field(default_factory=VolumeProfileSettings)
    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    gex: GEXSettings = Field(default_factory=GEXSettings)

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "env_nested_delimiter": "__",
    }


# Module-level singleton — import this everywhere instead of constructing your own
settings = Settings()

# ---------------------------------------------------------------------------
# Emergency kill switch
# ---------------------------------------------------------------------------
# If this file EXISTS on disk, all signal generation halts immediately.
# To activate:   touch data/KILL_SWITCH
# To deactivate: rm data/KILL_SWITCH
KILL_SWITCH_FILE = str(PROJECT_ROOT / "data" / "KILL_SWITCH")

# ---------------------------------------------------------------------------
# Strategy Version Tracking
# ---------------------------------------------------------------------------
# Increment whenever strategy parameters change (SL multiplier, target multiplier,
# confidence thresholds, indicator weights, etc.).
# All signals are tagged with this version; performance is tracked separately per version.
STRATEGY_VERSION = "1.0.0"
STRATEGY_VERSION_NOTES = "Initial parameters from optimization"

# Version history — add an entry here whenever you change params:
# 1.0.0 — Initial optimization (2026-04-18): baseline parameters
