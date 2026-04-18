# Trading DSS — Operational Runbook

> **Audience:** Daily operator of the paper trading system.
> **Scope:** Phase 9 — paper trading, watchdog, automated reporting.
> **Not covered here:** how to extend the strategy, re-train models, or deploy to a server.

---

## Table of Contents

1. [Daily Routine](#daily-routine)
2. [Weekly Routine](#weekly-routine)
3. [Monthly Routine](#monthly-routine)
4. [Emergency Procedures](#emergency-procedures)
5. [Key Commands Reference](#key-commands-reference)
6. [Telegram Commands](#telegram-commands)
7. [Log Files](#log-files)

---

## Daily Routine

### Pre-Market (8:30 – 9:15 AM IST)

1. **Start the system**

   ```bash
   python scripts/run_system.py --paper-trading
   ```

   With pre-launch check (recommended before first session or after changes):

   ```bash
   python scripts/run_system.py --paper-trading --validate-first
   ```

2. **Verify startup** — all components should show ✅ in the console:

   ```
   ✅ Data Collector initialised
   ✅ Decision Engine initialised
   ✅ Paper Trading initialised (₹1,00,000)
   ✅ Watchdog initialised
   ✅ Automated Reporter initialised (daily 15:45, weekly Sat)
   ✅ Telegram Bot initialised
   ✅ API Server will start on port 8000
   ```

3. **Check Telegram** — send `/status`
   Expected: "System ready, market opens in X min"

4. **Check dashboard** — open `http://localhost:8000`
   All panels (prices, signals, portfolio, health) should be loading.

5. **Review overnight news** — send `/news` in Telegram
   Look for CRITICAL events (RBI policy, budget, geopolitical).

6. **Check VIX** — send `/vix`
   - VIX < 12: Very low fear, options may be expensive
   - VIX 12–20: Normal
   - VIX > 20: Elevated volatility — watch for false signals

7. **Check FII** — send `/fii`
   FII selling pressure often leads to index weakness.

---

### Market Hours (9:15 AM – 3:30 PM IST)

The system operates autonomously. Telegram sends alerts for:
- New signal generated
- Position opened (paper trade)
- Stop-loss or target hit (position closed)
- Hourly digest with anomaly summary
- Watchdog alerts if something goes wrong

**Check points (30 seconds each):**

| Time | Action |
|------|--------|
| 10:00 AM | First signals should have generated. Send `/signal` |
| 12:00 PM | Mid-day check. Send `/portfolio` for open positions |
| 2:00 PM | Pre-close check. Expiry day? Any positions need manual attention? |
| 3:20 PM | All positions auto-close at 15:25 IST |

---

### Post-Market (3:30 – 4:00 PM IST)

1. Daily report arrives via Telegram at **3:45 PM** — review it.
2. Check summary: How many trades? Win/loss? Net P&L?
3. Send `/performance` — review rolling 7-day and 30-day metrics.
4. Check the dashboard `Health` panel — any watchdog warnings?
5. Note any unusual behavior for investigation (wrong signal timing, missed exits, etc.).

### Stopping the System

Press `Ctrl+C` in the terminal where run_system.py is running. The system saves paper trading state before exiting, so no trades are lost.

---

## Weekly Routine (Saturday)

1. **Weekly report** arrives automatically at **10:00 AM Saturday**.

2. **Check shadow tracker** — the report shows live vs. backtest comparison:
   - **LOW divergence**: Live is closely tracking backtest — good sign.
   - **MODERATE divergence**: Expected drift — monitor.
   - **HIGH/CRITICAL divergence**: Investigate why live underperforms.

3. **Check edge status** — printed in the weekly report:
   - `STRONG` / `INTACT`: Strategy is working, continue.
   - `WEAKENING`: Reduce trade frequency, consider re-optimisation.
   - `GONE`: Pause trading, re-run optimizer with recent data.
   - `INSUFFICIENT_DATA`: Normal in early weeks (< 20 trades).

4. **Review system logs** for recurring errors:

   ```bash
   grep -i "error\|critical" data/logs/trading_dss.log | tail -50
   ```

5. **Database maintenance** — old intraday data is cleaned automatically. Verify DB size:

   ```bash
   du -sh data/db/trading.db
   ```

6. **Consider re-optimisation** if:
   - More than 4 weeks have elapsed since last optimisation.
   - Edge status is `WEAKENING` or `GONE`.
   - Live win rate has drifted > 10 percentage points from backtest.

---

## Monthly Routine

1. **Re-run the optimizer** with fresh data:

   ```bash
   python scripts/run_optimizer.py \
     --index NIFTY50 \
     --start 2024-01-01 --end 2025-01-01 \
     --auto-approve
   ```

2. **Walk-forward validation** on new data:

   ```bash
   python scripts/run_robustness.py --index NIFTY50
   ```

3. **Compare live vs backtest** — check weekly report's shadow tracker section for cumulative drift.

4. **Review event calendar** — add upcoming known events (budget, RBI policy, elections) to `config/data/events_calendar.json`.

5. **Check disk space** and DB size:

   ```bash
   df -h
   du -sh data/
   ```

6. **Update dependencies** (only if needed, test on dev first):

   ```bash
   pip install -r requirements.txt --upgrade
   ```

---

## Emergency Procedures

### System Not Responding

1. Check if the Python process is running:

   ```bash
   ps aux | grep run_system
   ```

2. Check the last 100 lines of the log:

   ```bash
   tail -100 data/logs/trading_dss.log
   ```

3. Restart:

   ```bash
   python scripts/run_system.py --paper-trading
   ```

4. If the SQLite database is locked (error: "database is locked"):

   ```bash
   sqlite3 data/db/trading.db "PRAGMA wal_checkpoint(TRUNCATE);"
   ```

---

### NSE Blocked Your IP

**Symptom:** Circuit breaker opens; Telegram alert "NSE unreachable". Prices stop updating.

1. The system will auto-detect this (circuit breaker opens).
2. Wait 10–15 minutes — the circuit breaker auto-recovers.
3. If the block persists:
   - Restart your router to get a new IP, or connect through a VPN.
   - Check if you are running multiple instances of the system (double consumption of rate limit).
4. Verify recovery: circuit breaker should return to `CLOSED`.

---

### Kill Switch Activated

**Symptom:** Telegram alert "Kill switch activated". No new paper trades being taken.

1. **Find out why** — check Telegram alerts and the log:

   ```bash
   grep -i "kill switch\|kill_switch" data/logs/trading_dss.log | tail -20
   ```

2. **Fix the root cause** (e.g., excessive drawdown, watchdog detected a crash).

3. **Deactivate** via Telegram:

   ```
   /resume
   ```

   Or manually:

   ```bash
   rm data/KILL_SWITCH
   ```

4. **Monitor closely** for 30 minutes after resuming — watch for the same failure re-occurring.

---

### Edge Appears Gone

**Symptom:** Weekly report shows `edge_status: GONE`; win rate < 45% over the last 14 days.

1. **Don't panic** — a bad week is not a dead strategy. Check at least 2–3 weeks of data.
2. Run the edge tracker manually to inspect the trend:

   ```bash
   python -c "
   from src.database.db_manager import get_db_manager
   from src.paper_trading.edge_tracker import EdgeTracker
   db = get_db_manager(); db.connect()
   et = EdgeTracker(db)
   a = et.assess_current_edge()
   print(a)
   "
   ```

3. Re-run the optimizer with the most recent 6 months of data:

   ```bash
   python scripts/run_optimizer.py --index NIFTY50 --auto-approve
   ```

4. Compare new optimal parameters with the currently approved parameters in `data/approved_params.json`.
   - If significantly different → the market regime may have shifted.
   - If similar → the issue is statistical noise, wait it out.

---

### Data Corruption

1. **Stop the system** (Ctrl+C or kill the process).

2. **Back up the current database**:

   ```bash
   cp data/db/trading.db data/db/trading.db.backup_$(date +%Y%m%d)
   ```

3. **Validate the schema**:

   ```bash
   python scripts/validate_config.py
   ```

4. **If tables are corrupt** — re-seed historical data (this does NOT erase paper trading history if paper_trades table is intact):

   ```bash
   python scripts/seed_historical.py
   ```

5. **Restart the system.**

---

### Watchdog Sends SAFE MODE Alert

**Symptom:** Telegram message "Watchdog entering SAFE MODE — new trades paused".

1. Check what triggered it:

   ```bash
   grep -i "safe mode\|watchdog" data/logs/trading_dss.log | tail -30
   ```

2. Common causes:
   - Repeated data staleness (NSE scraper failing)
   - Memory usage > 2 GB (system overloaded)
   - Error rate spike (bug in signal generation)

3. Fix the underlying issue, then restart the system. Safe mode resets on restart.

---

## Key Commands Reference

| Command | Purpose |
|---------|---------|
| `python scripts/run_system.py --paper-trading` | Start everything |
| `python scripts/run_system.py --validate-first` | Start with pre-launch check |
| `python scripts/validate_system.py` | Run pre-launch validation only (no system start) |
| `python scripts/validate_config.py` | Check configuration files |
| `python scripts/run_optimizer.py --index NIFTY50 --auto-approve` | Re-optimise parameters |
| `python scripts/run_backtest.py --index NIFTY50` | Run backtest |
| `python scripts/run_robustness.py --index NIFTY50` | Robustness / walk-forward check |
| `python scripts/seed_historical.py` | Re-seed historical price data |

---

## Telegram Commands

| Command | Action |
|---------|--------|
| `/status` | System health summary |
| `/signal` | Latest signal for all active indices |
| `/portfolio` | Open paper positions |
| `/performance` | Rolling metrics (7d, 30d) |
| `/vix` | Current VIX level |
| `/fii` | Latest FII/DII flow data |
| `/news` | Recent market news with sentiment |
| `/pause` | Pause new paper trades |
| `/resume` | Resume paper trading |
| `/report` | Generate daily report on demand |
| `/help` | Full command list |

---

## Log Files

| File | Contents |
|------|---------|
| `data/logs/trading_dss.log` | Main unified log (all components) |
| `data/logs/paper_trading.log` | Paper trading events only |
| `data/logs/watchdog.log` | Watchdog health checks |
| `data/reports/daily/` | Daily report files (JSON + text) |
| `data/reports/weekly/` | Weekly report files |
| `data/audit/` | Structured decision audit trail (JSON per signal) |

**Tip:** For live log tailing during market hours:

```bash
tail -f data/logs/trading_dss.log | grep -v DEBUG
```
