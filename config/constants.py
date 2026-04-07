"""
Fixed constants for the trading DSS.

Unlike settings.py (which is environment-configurable), these values are
structural facts about Indian markets that do not change at runtime.
"""

from __future__ import annotations

from datetime import date
from typing import Final

# ── Transaction cost structure ──────────────────────────────────────────────
# All rates are per-unit fractions (multiply by 100 to get percentage)

STT_EQUITY_DELIVERY: Final[float] = 0.001          # 0.10% both sides
STT_EQUITY_INTRADAY: Final[float] = 0.00025        # 0.025% sell side only
STT_FUTURES_SELL: Final[float] = 0.0125 / 100      # 0.0125% sell side
STT_OPTIONS_BUY: Final[float] = 0.0625 / 100       # 0.0625% on premium (buy)
STT_OPTIONS_SELL_ITM: Final[float] = 0.125 / 100   # 0.125% on intrinsic value on exercise

EXCHANGE_TRANSACTION_CHARGE_NSE: Final[float] = 0.00297 / 100   # 0.00297%
EXCHANGE_TRANSACTION_CHARGE_BSE: Final[float] = 0.00375 / 100   # 0.00375%
EXCHANGE_TRANSACTION_CHARGE_NSE_FO: Final[float] = 0.00188 / 100

SEBI_TURNOVER_FEE: Final[float] = 10 / 10_000_000  # ₹10 per crore
GST_RATE: Final[float] = 0.18                       # 18% on brokerage + exchange charges
STAMP_DUTY_DELIVERY: Final[float] = 0.015 / 100    # 0.015% on buy side
STAMP_DUTY_INTRADAY: Final[float] = 0.003 / 100    # 0.003% on buy side
STAMP_DUTY_FO: Final[float] = 0.002 / 100          # 0.002% on buy side

DEFAULT_BROKERAGE_FLAT: Final[float] = 20.0        # ₹20 per order (flat brokerage)
BROKERAGE_CAP_EQUITY: Final[float] = 20.0          # Maximum per-order brokerage

# ── Expiry rules ────────────────────────────────────────────────────────────
# Indices with weekly options expire every Thursday
WEEKLY_EXPIRY_INDICES: Final[frozenset[str]] = frozenset({
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "SENSEX",
})

# All other F&O contracts expire on last Thursday of the month
MONTHLY_EXPIRY_INDICES: Final[frozenset[str]] = frozenset({
    "NIFTY50",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "SENSEX",
})

# Futures expiry → last Thursday of month
FUTURES_EXPIRY_DAY: Final[int] = 3  # Thursday (weekday index 0=Monday)

# ── NSE/BSE HTTP scraping headers ───────────────────────────────────────────
NSE_BASE_URL: Final[str] = "https://www.nseindia.com"
BSE_BASE_URL: Final[str] = "https://www.bseindia.com"

USER_AGENT_ROTATION: Final[list[str]] = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) "
        "Gecko/20100101 Firefox/133.0"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/18.2 Safari/605.1.15"
    ),
]

NSE_HEADERS: Final[dict[str, str]] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Host": "www.nseindia.com",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

BSE_HEADERS: Final[dict[str, str]] = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "api.bseindia.com",
    "Origin": "https://www.bseindia.com",
    "Referer": "https://www.bseindia.com/",
}

# ── Market holidays ─────────────────────────────────────────────────────────
# NSE + BSE trading holidays (both exchanges observe same holidays)
# Source: official NSE/BSE holiday calendar

MARKET_HOLIDAYS: Final[frozenset[date]] = frozenset({
    # 2024
    date(2024, 1, 26),   # Republic Day
    date(2024, 3, 8),    # Mahashivratri
    date(2024, 3, 25),   # Holi
    date(2024, 3, 29),   # Good Friday
    date(2024, 4, 11),   # Id-Ul-Fitr (Ramzan Eid)
    date(2024, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti / Bihu / Bohag
    date(2024, 4, 17),   # Ram Navami
    date(2024, 4, 21),   # Mahavir Jayanti
    date(2024, 5, 23),   # Buddha Pournima
    date(2024, 6, 17),   # Bakri Id (Eid ul-Adha)
    date(2024, 7, 17),   # Muharram
    date(2024, 8, 15),   # Independence Day
    date(2024, 10, 2),   # Mahatma Gandhi Jayanti
    date(2024, 10, 24),  # Dussehra
    date(2024, 11, 1),   # Diwali Laxmi Puja
    date(2024, 11, 15),  # Gurunanak Jayanti
    date(2024, 12, 25),  # Christmas

    # 2025
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr (Ramzan Eid)
    date(2025, 4, 10),   # Shree Ram Navami
    date(2025, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Gandhi Jayanti
    date(2025, 10, 2),   # Dussehra (same day - verify official list)
    date(2025, 10, 21),  # Diwali Laxmi Puja (Muhurat Trading)
    date(2025, 11, 5),   # Gurunanak Jayanti
    date(2025, 12, 25),  # Christmas

    # 2026 (preliminary — verify from NSE official calendar)
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 20),   # Holi
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 8, 15),   # Independence Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 11, 10),  # Diwali (approximate)
    date(2026, 12, 25),  # Christmas
})

# Muhurat trading day (special 1-hour evening trading on Diwali)
MUHURAT_TRADING_DAYS: Final[frozenset[date]] = frozenset({
    date(2024, 11, 1),
    date(2025, 10, 20),
})

# ── Timezone ────────────────────────────────────────────────────────────────
IST_TIMEZONE: Final[str] = "Asia/Kolkata"

# ── Cache TTLs (seconds) ────────────────────────────────────────────────────
CACHE_TTL_SPOT_PRICE: Final[int] = 15
CACHE_TTL_OPTIONS_CHAIN: Final[int] = 60
CACHE_TTL_INDEX_REGISTRY: Final[int] = 3600   # Re-read indices.json hourly
CACHE_TTL_HOLIDAYS: Final[int] = 86400        # Daily

# ── Numeric precision ───────────────────────────────────────────────────────
PRICE_DECIMAL_PLACES: Final[int] = 2
OI_DECIMAL_PLACES: Final[int] = 0
IV_DECIMAL_PLACES: Final[int] = 4

# ── Signal confidence levels ────────────────────────────────────────────────
CONFIDENCE_LOW: Final[float] = 0.4
CONFIDENCE_MEDIUM: Final[float] = 0.6
CONFIDENCE_HIGH: Final[float] = 0.8
CONFIDENCE_VERY_HIGH: Final[float] = 0.9

# ── Max Pain calculation ────────────────────────────────────────────────────
MAX_PAIN_STRIKE_RANGE_PERCENT: Final[float] = 0.10  # ±10% from ATM
