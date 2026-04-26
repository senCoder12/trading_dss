# F&O Universe Orchestrator — Build Summary

## Overview

Built complete orchestrator for F&O universe feature (Phase 5.4). Single entry point for all F&O stock selection, scoring, and monitoring across DecisionEngine, Telegram bot, and dashboard.

## What Was Built

### 1. FOUniverseManager (`src/analysis/fo_universe_manager.py`)

Central orchestrator tying together:
- **FOUniverseFetcher** — F&O master list management
- **FOBulkFetcher** — Bulk quotes, volumes, OI fetches
- **FOLiquidityFilter** — Aggressive filter (aggressive: better to miss than include illiquid)
- **FOStockScorer** — 6-component scoring engine (0–100)

**Key Methods:**

| Method | Purpose | Schedule |
|--------|---------|----------|
| `run_daily_universe_selection()` | Full pipeline → top 50 + top 15 | 09:00 AM |
| `run_intraday_refresh()` | Lightweight watchlist rerank (top 30 only) | Every 15 min, 9:15–15:30 |
| `run_eod_snapshot()` | EOD persistence & summary | 15:35 PM |
| `get_current_universe()` | Thread-safe snapshot getter | Any time |
| `get_priority_symbols()` | Top 15 for DecisionEngine signals | Any time |
| `get_active_universe_symbols()` | Top 50 for broader monitoring | Any time |
| `get_stock_score(symbol)` | Score details for one stock | Any time |

**Thread Safety:**
- `threading.Lock` protects `_current_snapshot` during updates
- Intraday refresh updates in-place (no snapshot copy overhead)
- Safe for concurrent reads from Telegram bot, API, DecisionEngine

### 2. UniverseSnapshot Dataclass

Atomic daily snapshot containing:
- **active_universe:** list[StockScore] — top 50
- **priority_watchlist:** list[StockScore] — top 15 (used by DecisionEngine)
- **total_liquid:** int — passed liquidity filter
- **total_fo_stocks:** int — full F&O universe size
- **market_breadth:** float — % with significant moves (≥1%)
- **top_sector:** str — sector dominating activity (extensible)
- **filter_stats:** FilterResult — detailed breakdown
- **is_market_hours:** bool — was snapshot during RTH?
- **refresh_count:** int — intraday refreshes so far

### 3. Database Layer

**New Table: `fo_universe_snapshots`**
```sql
CREATE TABLE fo_universe_snapshots (
    id INTEGER PRIMARY KEY,
    date TEXT UNIQUE,                    -- "YYYY-MM-DD"
    timestamp TEXT,                      -- ISO-8601 IST
    active_universe TEXT,                -- JSON: top 50 scores
    priority_watchlist TEXT,             -- JSON: top 15 scores
    total_liquid INTEGER,
    total_fo_stocks INTEGER,
    market_breadth REAL,
    top_sector TEXT,
    filter_stats TEXT,                   -- JSON: FilterResult
    is_market_hours INTEGER,             -- bool
    refresh_count INTEGER                -- count
)
```

**New Queries (queries.py):**
- `UPSERT_FO_UNIVERSE_SNAPSHOT` — idempotent daily upsert
- `GET_FO_UNIVERSE_SNAPSHOT` — fetch by date
- `GET_LATEST_FO_UNIVERSE_SNAPSHOT` — latest snapshot
- `DELETE_FO_UNIVERSE_SNAPSHOT` — cleanup

**New Migration (v15):**
- Automatic schema creation on first run
- Idempotent (safe to re-run)
- Backward compatible

### 4. Scheduler Integration

APScheduler jobs via `start_scheduler()`:

| Time | Job | Function |
|------|-----|----------|
| 08:45 | Pre-market refresh | `_refresh_universe_and_volumes()` |
| 09:00 | Daily selection | `run_daily_universe_selection()` |
| 09:15–15:30 every 15min | Intraday refresh | `run_intraday_refresh()` |
| 15:35 | EOD snapshot | `run_eod_snapshot()` |

**Features:**
- `max_instances=1` prevents overlapping runs
- `coalesce=True` skips missed runs instead of queuing
- Thread-safe snapshot updates
- Graceful error handling (logs, doesn't crash)

### 5. Telegram Bot Commands

Added 3 new commands (wired via `set_fo_universe_manager()`):

#### `/universe`
```
📊 F&O Universe — 25-Apr 10:45
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total F&O: 210 | Liquid: 75
Active Top-50: 50 | Breadth: 42.1%
Top Sector: Banking | Refreshes: 3

🔥 Priority Watchlist (Top 15):
 1. RELIANCE     87.3 OI↑ Vol↑
 2. INFY         84.1 OI↑ IV↑
 3. TCS          82.5
 ...
```

#### `/watchlist`
```
📊 Priority Watchlist
━━━━━━━━━━━━━━━━━━
 1. RELIANCE 87.3
 2. INFY 84.1
 3. TCS 82.5
...
```

#### `/score SYMBOL`
```
💹 RELIANCE Score Details
━━━━━━━━━━━━━━━━━━━━━
Composite: 87.3 (Rank #1)
Tier: PRIORITY

Component Scores (0–100):
OI: 92 | OI Change: 88
Volume: 79 | Price Move: 71
IV Change: 85 | Spread: 94

Raw Metrics:
Total OI: 45,234,000 | Volume Ratio: 2.15x
Price Change: +1.25% | IV Change: +2.3%
Bid-Ask Spread: 0.12%
```

### 6. Data Models

**FOUniverseSnapshot (models.py):**
- Dataclass for in-memory representation
- Serialized to JSON for DB persistence
- Includes all metadata needed for dashboards

## Architecture

```
┌─────────────────────────────────────────────────┐
│           DataCollector (main loop)              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌────────────────────────────────────────┐    │
│  │  FOUniverseManager (orchestrator)      │    │
│  ├────────────────────────────────────────┤    │
│  │ • FOUniverseFetcher (stock list)       │    │
│  │ • FOBulkFetcher (quotes, OI)           │    │
│  │ • FOLiquidityFilter                    │    │
│  │ • FOStockScorer (6 components)         │    │
│  │                                        │    │
│  │ Snapshot: top 50 + priority 15         │    │
│  │ Thread-safe with Lock                  │    │
│  └────────────────────────────────────────┘    │
│                    ↓                            │
│            APScheduler jobs:                   │
│  • 08:45: Refresh universe & volumes           │
│  • 09:00: Daily selection (full pipeline)      │
│  • 09:15–15:30 every 15min: Intraday refresh  │
│  • 15:35: EOD snapshot                         │
│                    ↓                            │
│  ┌─────────────────────────────────────────┐  │
│  │   Consumers (read-only access)          │  │
│  ├─────────────────────────────────────────┤  │
│  │ • TelegramBot (/universe, /watchlist)   │  │
│  │ • DecisionEngine (priority symbols)     │  │
│  │ • Dashboard API (current snapshot)      │  │
│  │ • Database (daily persistence)          │  │
│  └─────────────────────────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Integration Checklist

- [x] FOUniverseManager class built
- [x] UniverseSnapshot dataclass
- [x] Database table + migration (v15)
- [x] Database queries (UPSERT, GET, DELETE)
- [x] APScheduler integration
- [x] Telegram bot commands (/universe, /watchlist, /score)
- [x] Thread-safe snapshot management
- [x] Error handling & logging
- [ ] Wire into DataCollector (see INTEGRATION_FO_UNIVERSE_MANAGER.md)
- [ ] (Optional) Integrate with DecisionEngine for stock signals

## Files Modified/Created

| File | Change | Lines |
|------|--------|-------|
| `src/analysis/fo_universe_manager.py` | Created | 670 |
| `src/database/models.py` | Add UniverseSnapshot + table DDL | +60 |
| `src/database/migrations.py` | Add v15 migration | +9 |
| `src/database/queries.py` | Add snapshot queries | +35 |
| `src/alerts/telegram_bot.py` | Add 3 commands, wire manager | +140 |
| `INTEGRATION_FO_UNIVERSE_MANAGER.md` | Created (integration guide) | — |

## Testing

✅ **Compilation:**
- `fo_universe_manager.py` compiles (imports verified)
- models.py loads without errors
- migrations register (v1–v15)
- queries load

✅ **Database:**
- Schema creates cleanly
- Migration v15 applies
- `fo_universe_snapshots` table created
- All columns present

✅ **Imports:**
- All sub-components importable
- No circular dependencies
- Telegram bot wiring valid

## Known Limitations & Future Work

1. **Sector metadata:** Currently hardcoded "unknown" — enhance with NSE sector classification
2. **DecisionEngine integration:** Optional, user implements based on their needs
3. **Stock-specific signals:** Can be added by implementing `DecisionEngine.run_stocks(symbols)`
4. **FII sector boost:** Currently available in scorer, not yet used in snapshot
5. **Watchlist change alerts:** Telegram notifications when top 15 changes significantly

## Configuration

Customize via `config/settings.py`:

```python
fo_universe = {
    "daily_universe_time": "09:00",      # Full selection start time
    "intraday_refresh_minutes": 15,      # Watchlist refresh interval
    "eod_snapshot_time": "15:35",        # EOD snapshot time
}
```

## Next Steps

1. **Integrate with DataCollector** (see INTEGRATION guide)
2. **Test scheduler** during market hours
3. **Verify Telegram commands** work end-to-end
4. **(Optional) Implement stock signal generation** for DecisionEngine
5. **(Optional) Add sector metadata** to UniverseSnapshot

---

**Phase 5.4 complete.** Orchestrator ready for integration.
