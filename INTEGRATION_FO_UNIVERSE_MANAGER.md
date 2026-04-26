# FOUniverseManager Integration Guide

The F&O Universe Manager (`src/analysis/fo_universe_manager.py`) is the orchestrator for the F&O stock universe feature. It needs to be wired into your application's DataCollector and optionally into the DecisionEngine.

## 1. Initialize in DataCollector

In `src/data/data_collector.py`, add to `__init__`:

```python
from src.analysis.fo_universe_manager import FOUniverseManager

class DataCollector:
    def __init__(self, ...):
        # ... existing init code ...
        
        # F&O Universe Manager
        self.fo_universe_manager = FOUniverseManager(
            db=self.db,
            smart_api=self.data_router.smart_api,  # or directly: SmartAPIClient instance
            nse_scraper=self.nse_scraper,
            fii_fetcher=self.fii_dii_fetcher,
        )
```

## 2. Start Scheduler

After initializing the manager, call:

```python
    def start(self) -> None:
        """Start all background jobs."""
        self.scheduler.start()
        self.fo_universe_manager.start_scheduler()  # Add this
        # ... rest of startup code ...
```

Stop it in shutdown:

```python
    def stop(self) -> None:
        """Shut down gracefully."""
        if self.scheduler.running:
            self.scheduler.shutdown()
        self.fo_universe_manager.stop_scheduler()  # Add this
        # ... rest of shutdown code ...
```

## 3. Wire Telegram Bot

In DataCollector, after creating the TelegramBot:

```python
    self.telegram_bot = TelegramBot(self.db, self.decision_engine)
    self.telegram_bot.set_fo_universe_manager(self.fo_universe_manager)  # Add this
```

Now the bot will respond to `/universe`, `/watchlist`, and `/score SYMBOL` commands.

## 4. (Optional) Integrate with DecisionEngine

To generate signals for priority watchlist stocks:

### 4a. Modify DecisionEngine to accept stocks

In `src/engine/decision_engine.py`, add method:

```python
def run_stocks(self, symbols: Optional[list[str]] = None) -> list[DecisionResult]:
    """Run full cycle for specific stocks instead of indices.
    
    Parameters
    ----------
    symbols:
        Stock symbols to analyze. If None, uses priority watchlist from manager.
    """
    # Implementation similar to run_all_indices but for stocks
    pass
```

### 4b. Call from DataCollector scheduler

In DataCollector._setup_jobs:

```python
    # During market hours, every 5 minutes: analyze priority stocks
    self.scheduler.add_job(
        self._run_stock_signals,
        trigger=IntervalTrigger(seconds=300, timezone=_IST),
        id="stock_signals",
        name="F&O Stock Signals",
        max_instances=1,
    )
```

Implement:

```python
def _run_stock_signals(self) -> None:
    """Generate signals for priority watchlist stocks."""
    symbols = self.fo_universe_manager.get_priority_symbols()
    if not symbols:
        return
    
    results = self.decision_engine.run_stocks(symbols)
    for result in results:
        if result.is_actionable:
            self.telegram_bot.send_alert(
                result.alert_message,
                priority="HIGH"
            )
```

## 5. Settings Configuration

Add to `config/settings.py` or `config/defaults.yaml`:

```python
# F&O Universe configuration
fo_universe = {
    "daily_universe_time": "09:00",      # When to run full selection (IST)
    "intraday_refresh_minutes": 15,      # Refresh watchlist every N minutes
    "eod_snapshot_time": "15:35",        # EOD snapshot time (IST)
}
```

## 6. Database Setup

The schema is automatically created on first run via migrations (version 15).  
If starting fresh, run:

```bash
python scripts/validate_config.py  # Creates schema
```

If joining existing db, the migration runs automatically on next startup.

## 7. Available Methods

### Query current snapshot:

```python
snapshot = fo_universe_manager.get_current_universe()
if snapshot:
    print(f"Active: {len(snapshot.active_universe)}")
    print(f"Priority: {len(snapshot.priority_watchlist)}")
    print(f"Breadth: {snapshot.market_breadth:.1f}%")
```

### Get symbols for decision making:

```python
priority_symbols = fo_universe_manager.get_priority_symbols()  # Top 15
active_symbols = fo_universe_manager.get_active_universe_symbols()  # Top 50

is_priority = fo_universe_manager.is_symbol_in_priority("RELIANCE")
```

### Get stock score:

```python
score = fo_universe_manager.get_stock_score("INFY")
if score:
    print(f"{score.symbol}: {score.composite_score:.1f}")
    print(f"  Tier: {score.tier} (Rank #{score.rank})")
```

## 8. Telegram Commands

Once wired up, users can run:

- `/universe` — Full snapshot with top 50 active, top 15 priority
- `/watchlist` — Compact priority list (top 15 only)
- `/score RELIANCE` — Detailed score breakdown for a stock

## 9. Architecture

```
DataCollector
├─ FOUniverseManager (orchestrator)
│  ├─ FOUniverseFetcher (F&O stock master list)
│  ├─ FOBulkFetcher (quotes, volumes, OI)
│  ├─ FOLiquidityFilter (remove untradeable stocks)
│  └─ FOStockScorer (assign 0–100 composite scores)
├─ Scheduler (APScheduler)
│  ├─ 08:45 — refresh universe & volumes
│  ├─ 09:00 — full daily selection → snapshot
│  ├─ 09:15–15:30 every 15min — intraday refresh
│  └─ 15:35 — EOD snapshot
├─ TelegramBot (optional)
│  └─ /universe, /watchlist, /score commands
└─ DecisionEngine (optional)
   └─ Generate signals for priority symbols
```

## 10. Troubleshooting

**No snapshot after 9:00 AM:**
- Check logs for errors in `run_daily_universe_selection()`
- Verify FOUniverseFetcher has fresh data (refresh manually if needed)
- Ensure SmartAPI is authenticated

**Empty watchlist:**
- Verify liquidity filter isn't too aggressive
- Check `filter_stats.passed` count — if very low, adjust thresholds

**Scheduler not running:**
- Verify `start_scheduler()` was called
- Check for exception in scheduler thread logs
- Ensure APScheduler job IDs are unique

**Telegram commands fail:**
- Verify `set_fo_universe_manager()` was called
- Check bot is initialized with correct Telegram token/chat_id
- Verify manager has at least one snapshot (wait until 9:01 AM)

---

Built as Phase 5.4 of Trading DSS.
