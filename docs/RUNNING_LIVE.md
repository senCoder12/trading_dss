# Running trading_dss — Full Live System Guide

All commands run from the **project root**: `/Users/mayukh/projects/trading_dss`

---

## 1. Prerequisites (one-time)

### Python environment

```bash
python3 --version        # must be 3.11+
pip install -r requirements.txt
```

### Frontend build

```bash
cd frontend
npm install
npm run build
cd ..
```

### Environment file

Create `.env` at the project root. Minimum required keys:

```dotenv
# ── Telegram (required for alert delivery) ──────────────────────────────────
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# ── Database (defaults work, override if needed) ────────────────────────────
DB_PATH=data/db/trading_dss.db
DB_WAL_MODE=true

# ── Polling intervals (seconds, defaults shown) ─────────────────────────────
POLL_PRICE_INTERVAL=60
POLL_OPTIONS_INTERVAL=180
POLL_NEWS_INTERVAL=120
POLL_FII_DII_INTERVAL=900
POLL_VIX_INTERVAL=60
POLL_TECHNICAL_INTERVAL=300

# ── Market hours (IST, defaults shown) ─────────────────────────────────────
MARKET_HOURS_MARKET_OPEN=09:15
MARKET_HOURS_MARKET_CLOSE=15:30
```

> **Get your Telegram credentials:**
> 1. Message `@BotFather` → `/newbot` → copy the token
> 2. Message `@userinfobot` → copy your chat ID

### Seed historical data (first time only, ~15 min)

```bash
python scripts/seed_historical.py
```

---

## 2. Validate everything before launch

```bash
python scripts/validate_config.py
```

This checks Python version, `.env` keys, `indices.json`, `news_mappings.json`,
`rss_feeds.json`, `sentiment_keywords.json`, and the SQLite database schema.
Fix **all** errors before proceeding.

---

## 3. Start the full live system

### Recommended: everything on, paper trading active

```bash
python scripts/run_system.py \
  --validate-first \
  --paper-trading \
  --paper-capital 100000 \
  --paper-confidence HIGH \
  --paper-indices NIFTY50 BANKNIFTY
```

What starts up (in order):

| # | Service | What it does |
|---|---------|-------------|
| 1 | **Config validator** | Checks all JSON configs and .env before any thread starts |
| 2 | **Database** | Connects SQLite, applies pending migrations, enables WAL |
| 3 | **Index Registry** | Loads all 48+ indices from `config/indices.json`, syncs to DB |
| 4 | **Data Collector** | APScheduler jobs: prices every 60s, options every 180s, VIX every 60s, news every 120s, FII/DII every 15min, technicals every 5min |
| 5 | **Decision Engine** | Combines technical (40%), options (35%), news (25%) into BUY/SELL/HOLD signals |
| 6 | **Overnight News Engine** | Off-market news pipeline, fires pre-market intelligence report at 08:50 IST |
| 7 | **Telegram Bot** | Polling bot — receives commands, sends alerts |
| 8 | **Paper Trading Engine** | Simulated trades on live signals with warm-up period |
| 9 | **Watchdog** | Health-checks all components every 30s |
| 10 | **Automated Reporter** | Daily P&L report at 15:45 IST, weekly report Saturday 10:00 IST |
| 11 | **FastAPI + Dashboard** | REST API + React dashboard on `http://localhost:8000` |

---

## 4. Access points once running

| URL | What you get |
|-----|-------------|
| `http://localhost:8000` | React dashboard (live signals, portfolio, charts) |
| `http://localhost:8000/docs` | Swagger UI — all API endpoints |
| `http://localhost:8000/redoc` | ReDoc API reference |

---

## 5. API endpoints reference

All prefixed with `/api/v1/`

### Signals
```
GET /signals/current              — latest signal per F&O index
GET /signals/current/{index_id}   — full signal detail for one index
GET /signals/history              — signal history with filters
GET /signals/performance          — win rate, avg return, accuracy stats
```

### Market Data
```
GET /market/prices                — latest prices for all active indices
GET /market/prices/{index_id}     — detailed price for one index
GET /market/options/{index_id}    — options chain summary (PCR, max pain, OI)
GET /market/vix                   — current India VIX
```

### Indices
```
GET /indices/                     — all indices with metadata
GET /indices/{index_id}           — full detail for one index
```

### News
```
GET /news/feed                    — news articles with filtering
GET /news/summary                 — aggregated market news summary
```

### Anomalies
```
GET /anomalies/active             — active anomaly alerts
GET /anomalies/history            — past anomaly alerts
GET /anomalies/dashboard          — anomaly overview across all indices
```

### Portfolio (Paper Trading)
```
GET /portfolio/summary            — portfolio P&L overview
GET /portfolio/positions          — open positions with live P&L
GET /portfolio/history            — daily P&L history (equity curve)
GET /portfolio/trades             — recent closed trades
GET /portfolio/trades/export      — download trade history as CSV
```

### Backtest
```
POST /backtest/run                — run a backtest (JSON body: index, start, end)
```

### System
```
GET /system/health                — component health (collector, DB, engine)
GET /system/status                — overall system status
```

---

## 6. Telegram bot commands

Once the bot is running, send these to your bot:

| Command | Response |
|---------|----------|
| `/start` | Welcome message and command list |
| `/signals` | Current BUY/SELL/HOLD signals for all F&O indices |
| `/signal NIFTY50` | Signal detail for a specific index |
| `/portfolio` | Paper trading portfolio summary |
| `/positions` | Open paper positions with live P&L |
| `/trades` | Recent closed trades |
| `/premarket` | Pre-market intelligence report |
| `/vix` | Current India VIX with interpretation |
| `/news` | Latest market news summary |
| `/anomalies` | Active anomaly alerts |
| `/health` | System health status |

---

## 7. Run modes (flags)

### Live market only, no paper trading

```bash
python scripts/run_system.py --no-paper-trading
```

### API + dashboard only (no data collection, for replay/review)

```bash
python scripts/run_system.py --api-only --api-port 8000
```

### Data collection only (no dashboard, headless server)

```bash
python scripts/run_system.py --collector-only
```

### Skip market-hours gate (run jobs immediately, useful for testing)

```bash
python scripts/run_system.py --force-start --skip-warm-up
```

### Custom paper trading setup

```bash
python scripts/run_system.py \
  --paper-capital 50000 \
  --paper-confidence MEDIUM \
  --paper-indices NIFTY50 BANKNIFTY FINNIFTY MIDCPNIFTY
```

### Debug mode (verbose logging)

```bash
python scripts/run_system.py --debug
```

### Dry run (collect data, no DB writes, no signal execution)

```bash
python scripts/run_system.py --dry-run
```

---

## 8. Run individual services separately

For debugging or running on separate machines/terminals:

### Data Collector only

```bash
python scripts/run_data_collector.py
```

### API Server only

```bash
python scripts/run_api.py
# or directly:
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Backtest

```bash
python scripts/run_backtest.py \
  --index NIFTY50 \
  --start 2024-01-01 \
  --end 2024-12-31
```

### Strategy optimizer

```bash
python scripts/run_optimizer.py \
  --index NIFTY50 \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --auto-approve
```

---

## 9. Production build (clean start)

```bash
bash scripts/build.sh          # builds frontend + validates config
python scripts/run_system.py --validate-first --paper-trading
```

---

## 10. Scheduled automatic reports

The system fires reports automatically — no manual trigger needed:

| Report | When | Delivery |
|--------|------|----------|
| Daily P&L | Every trading day at **15:45 IST** | Telegram + DB |
| Weekly summary | Every **Saturday at 10:00 IST** | Telegram + DB |
| Pre-market intelligence | Every trading day at **08:50 IST** | Telegram |

---

## 11. Troubleshooting

### NSE 403 errors

```
Error: NSE API returned 403 Forbidden
```
Wait 10–15 minutes. NSE WAF blocks automated requests after bursts.
The rate limiter is set to 30 req/min — do not lower it further.
Running from an Indian IP (or VPN) resolves persistent blocks.

### yfinance empty data

```
No data found, symbol may be delisted
```
Check the `yahoo_symbol` field in `config/indices.json`.
Verify the symbol directly on finance.yahoo.com (e.g., BANKNIFTY → `^NSEBANK`).

### Database locked

```
OperationalError: database is locked
```
Only one process should write to the DB at a time. Stop all running
`run_system.py` or `run_data_collector.py` processes first:

```bash
pkill -f run_system.py
pkill -f run_data_collector.py
```

### Telegram bot not responding

1. Check `.env` has both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
2. Run `python scripts/validate_config.py` — it checks Telegram config
3. Ensure only one bot instance is running (duplicate polling conflicts)

### Check logs

```bash
tail -f data/logs/trading_dss.log        # main log
tail -f data/logs/trading_dss_error.log  # errors only
```

---

## 12. Stop all services

Press `Ctrl+C` in the terminal running `run_system.py`.

The shutdown sequence is:
1. Watchdog stops
2. Paper trading state saved to DB
3. Data Collector scheduler stops
4. Telegram Bot polling stops
5. Database connection closed

All services stop gracefully — no data loss.

For background processes:

```bash
pkill -f run_system.py
```
