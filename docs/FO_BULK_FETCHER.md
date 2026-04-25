# F&O Bulk Data Fetcher

Efficiently fetches live market data for ~200 F&O stocks via Angel One SmartAPI.

## Architecture

**Layered design:**

1. **SmartAPIClient** (`src/data/smart_api_client.py`)
   - Wrapper around Angel One's SmartAPI
   - Clean interface for broker independence
   - Handles authentication, token management, and API timeouts

2. **FOBulkFetcher** (`src/data/fo_bulk_fetcher.py`)
   - Orchestrates bulk quote, volume, and OI fetches
   - Rate limiting: respects free-tier ~3 calls/second
   - Caching: 20-day volumes cached in DB (daily refresh)
   - OI history: in-memory with daily reset at market open

## Usage

### Quick Start

```python
from src.data.smart_api_client import SmartAPIClient
from src.data.fo_bulk_fetcher import FOBulkFetcher
from src.data.fo_universe_fetcher import FOUniverseFetcher
from src.database.db_manager import DatabaseManager

db = DatabaseManager.instance()
api = SmartAPIClient()
universe = FOUniverseFetcher(db, nse_scraper)

fetcher = FOBulkFetcher(db, api, universe)

# Full snapshot: quotes → liquidity filter → OI
quotes, oi_data = fetcher.fetch_full_snapshot(
    volume_filter_threshold=100000,    # Min 20-day avg volume
    spread_filter_pct=0.5,              # Max bid-ask spread %
)
```

### Dataclasses

**StockQuote** — single stock snapshot, not persistent

```python
@dataclass
class StockQuote:
    symbol: str
    ltp: float                  # Last traded price
    open: float
    high: float
    low: float
    close: float                # Previous close
    volume: int                 # Today's volume
    avg_volume_20d: float       # Cached from DB
    price_change_pct: float
    bid: float
    ask: float
    bid_ask_spread: float       # Calculated: ask - bid
    bid_ask_spread_pct: float   # Calculated: spread % of LTP
    timestamp: datetime
```

**StockOIData** — options chain snapshot, in-memory with history tracking

```python
@dataclass
class StockOIData:
    symbol: str
    expiry: date                # Nearest expiry
    total_ce_oi: int
    total_pe_oi: int
    ce_oi_change: int           # vs previous snapshot
    pe_oi_change: int
    pcr: float                  # PE OI / CE OI
    atm_strike: float
    atm_ce_iv: float
    atm_pe_iv: float
    avg_iv: float               # (ce_iv + pe_iv) / 2
    iv_change: float            # vs previous snapshot
    timestamp: datetime
```

## API Methods

### fetch_all_quotes() → dict[str, StockQuote]

Fetch live quotes for all ~200 F&O stocks.

- **Batching:** 50 symbols per call (configurable)
- **Rate limit:** 0.4s delay between batches (configurable)
- **Failure threshold:** Fails if >30% of symbols lost
- **Time target:** ~12 seconds (4 batches × 50 symbols + delays)

```python
quotes = fetcher.fetch_all_quotes()
# quotes["RELIANCE"] = StockQuote(...)
```

### fetch_avg_volumes(symbols: list[str]) → dict[str, float]

Fetch 20-day average volume from SmartAPI historical data.

- **Caching:** Results stored in `fo_avg_volumes` table (24-hour TTL)
- **Cache hits:** Avoid refetch if cache is fresh
- **Use case:** Liquidity filtering before OI fetch

```python
volumes = fetcher.fetch_avg_volumes(["RELIANCE", "INFY"])
# volumes["RELIANCE"] = 4500000.0
```

### fetch_oi_for_symbols(symbols: list[str]) → dict[str, StockOIData]

Fetch options chain data for filtered subset.

- **Per-symbol call:** Individual SmartAPI call per symbol
- **Change tracking:** Compares vs in-memory `_oi_history`
- **Rate limit:** 0.4s delay between calls
- **Use case:** OI concentration, IV signals, PCR analysis

```python
# Typically called with ~60 candidates (configurable)
oi_data = fetcher.fetch_oi_for_symbols(["RELIANCE", "INFY", ...])
# oi_data["RELIANCE"] = StockOIData(ce_oi_change=+50000, ...)
```

### fetch_full_snapshot() → tuple[dict[str, StockQuote], dict[str, StockOIData]]

**Orchestration method** — full workflow from quotes to OI.

**Step 1:** `fetch_all_quotes()` for all ~200 stocks

**Step 2:** Liquidity filter
- Volume > threshold (default 100,000)
- Spread < spread_pct (default 0.5%)
- Result: ~60 candidates

**Step 3:** `fetch_oi_for_symbols()` for candidates only

**Time target:** < 45 seconds total

```python
quotes, oi_data = fetcher.fetch_full_snapshot(
    volume_filter_threshold=100000,
    spread_filter_pct=0.5,
)
print(f"Quotes: {len(quotes)}, OI: {len(oi_data)}")
```

### reset_oi_history() → None

Clear in-memory OI history.

**When to call:** At market open (9:15 AM IST) to reset change tracking.

```python
# Scheduler at 9:15 AM
fetcher.reset_oi_history()
```

## NSE Bhav Copy (EOD)

**fetch_eod_oi_bhav(date_str: str) → dict[str, int]**

Fetch NSE F&O bhav copy (exchange-official OI snapshot) at end of day.

- **Called at:** 3:30 PM (after market close)
- **Purpose:** Total OI for all F&O stocks (for next-day scoring)
- **Source:** https://www.nseindia.com/api/fo-bhav-copy?date=DD-MM-YYYY
- **Return:** dict[symbol → total_oi]

```python
bhav_oi = fetch_eod_oi_bhav("25-04-2026")  # or None for today
# bhav_oi["RELIANCE"] = 1234567
```

## Configuration

Settings in `config/settings.py` under `BrokerSettings`:

```python
bulk_fetch_batch_size: int = 50         # Symbols per bulk quote call
bulk_fetch_delay_seconds: float = 0.4   # Delay between batches (s)
oi_fetch_top_n: int = 60                # Top candidates for OI fetch
```

Override via `.env`:

```bash
# .env
BULK_FETCH_BATCH_SIZE=75
BULK_FETCH_DELAY_SECONDS=0.3
OI_FETCH_TOP_N=50
```

## Database

**Table: fo_avg_volumes**

Caches 20-day average volume (refreshed daily at 8:45 AM).

| Column | Type | Purpose |
|--------|------|---------|
| symbol | TEXT | PK |
| avg_volume_20d | REAL | 20-day average daily volume |
| last_updated | TEXT | ISO-8601 IST |

**Migration:** Version 14 creates the table.

## Error Handling

**DataFetchError**

Raised when > 30% of symbols fail in `fetch_all_quotes()`.

- Signals systemic issue (API down, token expired, network)
- Caught by caller; triggers re-login or retry with exponential backoff

```python
try:
    quotes = fetcher.fetch_all_quotes()
except DataFetchError as e:
    logger.error("Bulk fetch failed: %s", e)
    # Re-login, retry, or alert
```

**Partial failures**

Successful data is always returned; failed symbols are logged and skipped.

- If 70% of symbols fetch successfully, no exception is raised
- Caller should validate returned dict size

```python
quotes = fetcher.fetch_all_quotes()
assert len(quotes) >= 140  # Expect ~200; alert if significantly lower
```

## Performance

| Operation | Symbols | Calls | Duration | Rate limit |
|-----------|---------|-------|----------|------------|
| fetch_all_quotes() | 200 | 4 | ~12s | 50/call, 0.4s delay |
| fetch_avg_volumes() | 200 | 200 | ~60s | Cached 24h |
| fetch_oi_for_symbols() | 60 | 60 | ~25s | 1/call, 0.4s delay |
| fetch_full_snapshot() | 200 → 60 | 264 total | ~45s | Orchestrated |

- SmartAPI free tier: ~3 calls/second
- Rate limiting via `bulk_fetch_delay_seconds` (0.4s = max 2.5 calls/s, safe margin)

## Integration with Scoring

**Expected workflow:**

1. **8:45 AM** — `refresh_if_stale()` on FOUniverseFetcher
2. **9:15 AM** — Market open
   - `reset_oi_history()` on FOBulkFetcher
   - Start polling loop
3. **Every 3 minutes (during market hours)** — Call `fetch_full_snapshot()`
   - Returns latest quotes + OI for top candidates
   - Feed to scoring engine
4. **3:30 PM** — Market close
   - Call `fetch_eod_oi_bhav()` for daily OI snapshot
   - Store for overnight analysis

## Testing

Run unit tests:

```bash
pytest tests/test_fo_bulk_fetcher.py -v
```

**Coverage:**

- Quote parsing (valid, missing fields, edge cases)
- OI parsing and change calculation
- Batching and rate limiting
- Volume caching (hit/miss)
- Liquidity filtering
- End-to-end orchestration

**15 tests, all passing.**

## Notes

### Token Lookup

Currently, `_get_token_for_symbol()` downloads the master contract on-demand.

**TODO:** Cache master contract at startup to avoid repeated downloads.

### SmartAPI Response Structure

Parsing assumes specific response fields:

- Quotes: `ltp`, `open`, `high`, `low`, `close`, `volume`, `bid`, `ask`
- Historical: `volume` array in OHLCV bars
- Options chain: `atmStrike`, `totalCeOi`, `totalPeOi`, `atmCeIv`, `atmPeIv`, `expiry`

**TODO:** Validate against actual SmartAPI responses; adjust mappings if needed.

### Future Enhancements

1. **Token caching:** Pre-load master contract at 9:00 AM
2. **Token pool:** Batch token lookups to reduce API calls
3. **Fallback broker:** Support Zerodha / Upstox via strategy pattern
4. **Metric collection:** Record fetch times, failure rates, cache hit %
5. **Incremental OI:** Cache full options chain to calculate OI changes per strike
