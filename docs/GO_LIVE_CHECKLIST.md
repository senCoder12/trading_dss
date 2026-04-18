# Paper Trading Go-Live Checklist

> Use this checklist before starting paper trading for the first time, and revisit the
> graduation criteria after 4+ weeks to assess readiness for real-money trading.
>
> **Important:** "Go live" in this document means *paper trading*, not real money.
> The real-money section is clearly marked and is outside the scope of this project.

---

## Before First Paper Trading Day

### System Build

- [ ] All Phases 1–8 code is built and committed
- [ ] `python scripts/validate_system.py` passes all checks (0 critical failures)
- [ ] No unresolved `TODO(BLOCKING)` or `FIXME` comments in core modules

### Data

- [ ] Historical data seeded: 2+ years daily, 5+ days intraday
  ```bash
  python scripts/seed_historical.py
  ```
- [ ] NIFTY50 and BANKNIFTY have 200+ daily price bars in the database
- [ ] Options snapshots exist (at least the most recent chain)
- [ ] VIX data present in `daily_prices` table

### Strategy

- [ ] Strategy optimised for primary indices (NIFTY50, BANKNIFTY)
  ```bash
  python scripts/run_optimizer.py --index NIFTY50 --auto-approve
  python scripts/run_optimizer.py --index BANKNIFTY --auto-approve
  ```
- [ ] Walk-forward validation shows > 50% profitable test windows
- [ ] Robustness score > 0.5 for approved parameters
- [ ] `data/approved_params.json` exists and contains NIFTY50 + BANKNIFTY entries

### Alerts & Monitoring

- [ ] Telegram bot configured and responds to `/status`
- [ ] Dashboard loading at `http://localhost:8000` and displaying real data
- [ ] WebSocket notifications working (browser tab receives events)
- [ ] Watchdog initialises without errors; shown as ✅ in startup output
- [ ] Kill switch tested: create `data/KILL_SWITCH`, verify trades are paused; delete and verify trading resumes

### Paper Engine

- [ ] Paper trading engine tested with a mock signal (manual test or `test_paper_engine.py`)
- [ ] State saves and restores correctly (restart system, verify capital is preserved)
- [ ] Forced EOD exit confirmed (paper positions close at 15:25 IST)

---

## Week 1 Goals (Conservative)

Focus: **stability**. One index, high confidence only, low trade count.

- [ ] System runs all 5 market days without unexpected crashes
- [ ] Only **HIGH** confidence signals traded (`--paper-confidence HIGH`)
- [ ] Only **NIFTY50** active (`--paper-indices NIFTY50`)
- [ ] Respect implicit limit of ~2 trades per day (position cap = 1 for NIFTY50)
- [ ] Daily report arrives every trading day at 3:45 PM via Telegram
- [ ] All paper trades logged in `paper_trades` table

**End-of-week review:**
- [ ] How many signals were generated vs executed?
- [ ] Are signal timings reasonable (not too close to open/close)?
- [ ] Any system errors in the log that need fixing?
- [ ] Win/loss makes intuitive sense given the week's market movement?

---

## Week 2 Goals (Expanding)

Focus: **multi-index**. Add BANKNIFTY; still conservative.

- [ ] Add BANKNIFTY to active indices (`--paper-indices NIFTY50 BANKNIFTY`)
- [ ] Still **HIGH** confidence only
- [ ] Track: simultaneous positions in both indices on the same day
- [ ] Shadow tracker: first weekly comparison in Saturday's report
- [ ] Edge tracker: first assessment (expect `INSUFFICIENT_DATA` — < 20 trades)
- [ ] Fix any bugs discovered during Week 1

**End-of-week review:**
- [ ] Does BANKNIFTY behave differently from NIFTY50? (higher volatility expected)
- [ ] Are risk limits (2% per trade, 5% per day) being respected?
- [ ] Any spurious signals on volatile days?

---

## Week 3 Goals (Testing)

Focus: **confidence calibration**. Evaluate whether MEDIUM signals add value.

- [ ] Review Week 1–2 data: what is the HIGH-confidence win rate?
  - Win rate > 55% → consider adding MEDIUM
  - Win rate < 50% → **do not expand** — investigate why first
- [ ] If expanding to MEDIUM:
  - [ ] Run 3 days with MEDIUM included and track separately
  - [ ] Compare MEDIUM vs HIGH win rates in the daily summary
- [ ] Run optimizer with 2 weeks of paper trading data included:
  ```bash
  python scripts/run_optimizer.py --index NIFTY50 --include-live-data
  ```
- [ ] Edge tracker first real assessment (should reach 20+ trades)
- [ ] Verify shadow comparison divergence is < HIGH

---

## Week 4 Goals (Validation)

Focus: **full validation**. Collect enough data for a statistical edge assessment.

- [ ] Full system running at target settings (final config for graduation)
- [ ] Minimum **20 paper trades** completed (edge tracker needs this)
- [ ] Calculate live metrics from the weekly report:
  - [ ] Win rate: _____%
  - [ ] Profit factor: _____
  - [ ] Max drawdown: _____%
  - [ ] Sharpe ratio: _____
- [ ] Compare with backtest expectations (from `data/approved_params.json`)
- [ ] Edge tracker status: `INTACT` or better (not `WEAKENING` or `GONE`)
- [ ] Shadow comparison divergence: `LOW` or `MODERATE`

---

## Paper Trading Graduation Criteria

> The system is **READY for consideration of real money** ONLY when **ALL** of these are met.
> If ANY criterion is not met: continue paper trading and investigate before proceeding.

### Performance

- [ ] **4+ weeks** of continuous paper trading completed
- [ ] **30+ paper trades** executed
- [ ] Live **win rate > 48%** (after all simulated costs)
- [ ] Live **profit factor > 1.1**
- [ ] **Maximum drawdown < 10%** of starting capital
- [ ] Live **Sharpe ratio > 0.5** (annualised, per weekly report)

### System Health

- [ ] **System uptime > 95%** of market hours (watchdog log confirms)
- [ ] No unresolved **critical bugs** (crashes, data loss, wrong P&L)
- [ ] Daily/weekly reports arriving **consistently** (no missed days)
- [ ] **Watchdog recovery** tested and confirmed working (simulated failure → auto-recovery)

### Strategy Validation

- [ ] Shadow comparison: divergence < **HIGH** (live closely tracks backtest)
- [ ] Edge tracker: status **INTACT** or **STRONG**
- [ ] Kill switch tested: activate + deactivate with no state corruption

---

## IF Graduating to Real Money (Future — Not Part of This Project)

> This section is informational only. No real-money trading is implemented in this codebase.

- Start with **10% of intended capital** (not the full amount)
- Use the **exact same parameters** as paper trading (do not change strategy at go-live)
- **First month**: verify live execution matches paper expectations trade-by-trade
- Gradually increase capital over **3 months** only if performance matches
- **ALWAYS** keep the kill switch accessible and monitor daily
- **NEVER** trade with money you cannot afford to lose
- Consult a registered investment advisor if you are unsure

---

## Quick Reference: Run Commands

```bash
# Pre-launch validation only
python scripts/validate_system.py

# Week 1: conservative (NIFTY50, HIGH only)
python scripts/run_system.py \
  --paper-trading \
  --paper-indices NIFTY50 \
  --paper-confidence HIGH \
  --validate-first

# Week 2+: both indices
python scripts/run_system.py \
  --paper-trading \
  --paper-indices NIFTY50 BANKNIFTY \
  --paper-confidence HIGH

# Full config (Week 4+)
python scripts/run_system.py \
  --paper-trading \
  --paper-capital 100000 \
  --paper-confidence HIGH \
  --paper-indices NIFTY50 BANKNIFTY

# Development (skip warm-up, no validation)
python scripts/run_system.py \
  --paper-trading \
  --skip-warm-up \
  --skip-validation \
  --force-start \
  --debug
```
