# trading_dss — Trading Decision Support System

A production-ready Python system for Indian stock market analysis and trading signal generation.
Supports **all** NSE and BSE indices — from NIFTY 50 to sector-specific and thematic indices —
without any code changes (just edit `config/indices.json`).

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         trading_dss                                 │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│  Data Layer  │ Analysis     │   Engine     │   Delivery             │
│              │              │              │                        │
│ nse_scraper  │ technical    │ decision_    │ FastAPI REST           │
│ bse_scraper  │ options_     │ engine       │ Telegram Bot           │
│ options_     │ analysis     │ risk_manager │                        │
│ chain        │ news_engine  │ regime_      │                        │
│ historical_  │ anomaly_     │ detector     │                        │
│ data         │ detector     │              │                        │
│ fii_dii_data │ sector_      │              │                        │
│ vix_data     │ analyzer     │              │                        │
│ index_       │              │              │                        │
│ registry     │              │              │                        │
├──────────────┴──────────────┴──────────────┴────────────────────────┤
│               SQLite (WAL)  │  APScheduler  │  TTL Cache            │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
  NSE/BSE API                              RSS Feeds
      │                                        │
      ▼                                        ▼
 rate_limiter ──► data_validator         news_engine
      │                                        │
      ▼                                        ▼
 index_registry ──► SQLite DB ◄──────── NewsArticle
      │                                        │
      ▼                                        │
 technical.py                                  │
 options_analysis.py                           │
 sector_analyzer.py ──────────────────────────►│
 vix_data.py                                   │
 fii_dii_data.py                               │
      │                                        │
      └──────────────────► decision_engine ────┘
                                  │
                          TradingSignal
                          ┌───────┴──────────┐
                       FastAPI             Telegram
                       /api/v1               Bot
```

---

## Quick Start

### First-Time Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Validate configuration (JSON files, DB, Python version)
python scripts/validate_config.py

# 3. Seed historical data (one-time, ~15 min)
python scripts/seed_historical.py

# 4. Build the React dashboard
cd frontend && npm install && npm run build && cd ..

# 5. (Optional) Configure Telegram alerts
#    Add to .env:
#      TELEGRAM_BOT_TOKEN=your_token
#      TELEGRAM_CHAT_ID=your_chat_id
```

### Run the System

```bash
# Start everything (data collector + API + Telegram bot)
python scripts/run_system.py

# Dashboard:  http://localhost:8000
# API Docs:   http://localhost:8000/docs
# Telegram:   Send /start to your bot
```

**Mode flags:**

| Flag | Effect |
|------|--------|
| `--no-telegram` | Skip Telegram bot |
| `--api-only` | API + dashboard only (no data collection) |
| `--collector-only` | Data collection only (no dashboard) |
| `--api-port 8080` | Use a custom port |
| `--debug` | Verbose logging |
| `--dry-run` | Collect data but suppress signal writes |

### Run a Backtest

```bash
python scripts/run_backtest.py --index NIFTY50 --start 2024-01-01 --end 2024-12-31
```

### Optimize Strategy Parameters

```bash
python scripts/run_optimizer.py --index NIFTY50 --start 2024-01-01 --end 2024-12-31 --auto-approve
```

### Production Build (single command)

```bash
bash scripts/build.sh       # builds frontend + validates config
python scripts/run_system.py
```

---

## Configuration Guide

The platform separates execution logic from target data dynamically utilizing configuration layers.

* `indices.json` — The master dictionary of ALL Indian indices tracking Yahoo symbols, F&O availabilities, and exchange designations.
* `news_mappings.json` — Routes news keyword triggers directly to the listed affected index IDs.
* `rss_feeds.json` — Outlines active RSS news endpoints alongside a preset `credibility_score (0-1)`. 
* `sentiment_keywords.json` — Tracks bullish, bearish, and neutral textual NLP keywords ensuring sets remain completely distinct.

### Adding a New Index
Trading DSS is fully config-driven. To add a new index tracking routine, **do not touch python code**. Simply edit `indices_json`!

```json
{
  "id": "NIFTY_MIDCAP150",
  "display_name": "NIFTY MIDCAP 150",
  "nse_symbol": "NIFTY MIDCAP 150",
  "yahoo_symbol": "^NIFTYMIDCAP150",
  "exchange": "NSE",
  "lot_size": null,
  "has_options": false,
  "option_symbol": null,
  "sector_category": "broad_market",
  "is_active": true,
  "description": ""
}
```

---

## Troubleshooting Common Issues

* **NSE blocking IP (403 Forbidden)**
  * *Reason*: The NSE applies strict WAF geo-blocking and automated bot limitations.
  * *Fix*: Wait 10-15 minutes for the connection timeout to reset. Modify the `User-Agent` strings inside `NSEScraper` or execute the script exclusively from an Indian IP space.

* **yfinance returning empty data (`No data found, symbol may be delisted`)**
  * *Reason*: The specified `yahoo_symbol` inside `indices.json` is malformed.
  * *Fix*: Verify the symbol on Yahoo Finance directly. For example, `BANKNIFTY` should equal `^NSEBANK` on Yahoo endpoints.

* **Database locked errors (`OperationalError: database is locked`)**
  * *Reason*: SQLite is fighting for write-access across thread scopes without WAL engagement. 
  * *Fix*: Ensure WAL mode is instantiated `PRAGMA journal_mode=WAL;` which `DatabaseManager` handles natively natively. Do not circumvent `DatabaseManager.execute()` context blocks.

---

## Performance Notes
* **Database Size**: Due to UPSERT mechanics compressing duplication, 2 years of history for 50 indices ranges around ~15 MB. 
* **RAM footprint**: Running the `data_collector` headless averages ~80MB footprint executing natively via `APScheduler`.

---

## Phase Roadmap

### Phase 1 — Foundation (Current)
- [x] Project structure and config system
- [x] Index registry (all 48+ indices, config-driven)
- [x] NSE/BSE scrapers with rate limiting
- [x] Options chain analysis (PCR, Max Pain, OI spikes)
- [x] Historical data via yfinance
- [x] Integration Scripts & End to End Orchestrators

### Phase 2 — Enhancement
- [ ] Full options chain polling for all F&O indices
- [ ] Intraday VWAP and 5-min signals
- [ ] Multi-timeframe confluence

### Phase 3 — Production
- [ ] PostgreSQL migration path
- [ ] Docker + docker-compose
- [ ] CI/CD pipeline