# Backtest Methodology

How the decision model is validated, what each test does and does not
show, and the current results.

> **Status (2026-11-06): the model has no demonstrated edge.** Out of
> sample, the calibrated win rate is about the random-walk base rate.
> Without a validated calibration the engine does not trade, by design.

The previous version of this document reported out-of-sample Sharpe
ratios of 1.6–1.8 for Sep 2024 – Mar 2025. None of those results can be
reproduced from this codebase or its data:

- Option and VIX history start in 2026.
- No code path fitted parameters per window.
- The files the old document cited do not exist.

That version has been replaced by this one.

---

## 1. What is tested

The backtest replays the **live decision engine**, not a simplified copy of it.

| Stage | Code | Notes |
|---|---|---|
| Replay loop | `src/backtest/engine_replay.py`, `scripts/replay_decision_engine.py` | Calls `DecisionEngine.run_full_cycle(index, as_of=t)` once per stored 5-minute bar. That is the live 5-minute cadence. |
| Signal model | `src/engine/signal_generator.py` | Confidence is a calibrated P(win) (§3). |
| Risk and execution | `src/engine/risk_manager.py` | Option-native. It selects a contract by delta, sets premium SL/target, sizes on ₹ premium risk, and enforces the daily limit, caps, cooldown and loss-streak halt. |
| Costs | `src/engine/option_costs.py` | Brokerage ₹20 per order; STT 0.1 % on the sell side; exchange (NSE 0.03503 % / BSE 0.0325 %); SEBI ₹10 per crore; stamp 0.003 % on the buy side; GST 18 %. |

**Point-in-time rules.**
- Every DB read is bounded by `as_of`.
- Decision time is the end of bar *t*, so the decision never sees bar *t+1*.
- News is skipped in replay, because there is no point-in-time news archive.
- The anomaly store and the kill switch, which describe production today, are disabled.

**Fills.**
- Entry is filled at bar *t+1*'s open. The selected contract is re-priced to that spot, and half the spread is added.
- Exits are checked on every bar:
  - At the open: gaps through a trigger, and time rules.
  - Intrabar: high/low against the spot and premium triggers. If both stop and target are touched in one bar, the stop is assumed first.
  - At the close: time stop, 15:15 forced exit, trailing stop.
- Exit premium is the model re-price minus half the spread.

**DB safety.** The replay copies the database file to a temporary
location and never opens the source.

### Assumptions you must keep in mind

- **Synthetic option chain.** Historical chains are essentially absent, so
  premiums are Black-Scholes prices.
  - IV is VIX at `as_of` when available, otherwise 1.1 × realised vol of the last 75 bars, with a 10 % floor.
  - IV is flat across strikes and over the life of a trade: no smile and no IV crush.
  - The half spread is the larger of 0.5 % of premium and ₹0.05.
- **Expiries follow rules.** NIFTY expires weekly on Tuesday; BANKNIFTY on the last Tuesday of the month. Holidays are ignored.
- **Lot sizes are today's** (NIFTY 65, BANKNIFTY 30), which is correct for 2026 data.
- **Market orders at the next open.** There is no limit-order queue and no slippage beyond the half spread.
- **The data window is short.** About 115 sessions of 5m bars (Mar–Nov 2026). Index volume is zero, so volume-based components are unavailable.

## 2. Labels: what counts as a win

`scripts/calibrate_signal_model.py` replays the live **scoring** function
(`SignalGenerator.model_score`) on every tradeable bar:

- Tradeable means 09:30–14:25, a safe regime, and a score outside the dead zone.
- Entry is the next bar's open.
- **Win:** price reaches entry ± 1.5 × ATR before entry ∓ 1.0 × ATR, within 12 bars (60 min) of the same session.
- **Loss:** the stop is hit first, both are hit in one bar, or neither is hit by the horizon. A timeout bleeds theta for an option buyer, so it counts as a loss.

The live engine uses **the same multiples** for its stop and target
(read from the calibration file), so the calibrated probability
describes the trade actually taken.

Break-even win rate at 1.5 : 1 is 40 % before costs. Theta and costs
push the practical threshold to about 45 %.

## 3. Calibration and out-of-sample validation

- **Fitting.** Bins of |score|, shrunk with a Beta(2, 2) prior and made monotone with pooled-adjacent-violators. Fitted on the chronologically first 70 % of samples.
- **Validation.** The last 30 % is out-of-sample. The model is `validated` only if all of the following hold out of sample:
  - at least 30 samples fall in tradeable bins (P ≥ 0.50);
  - their realised win rate is ≥ 0.45;
  - the Brier score beats the constant base-rate forecast.
- **Runtime.** `SignalCalibration` maps the score to P(win) and applies only penalties for conditions it did not see (anomaly risk, regime transition, extreme volatility, expensive IV). If there is no file, the file is older than 120 days, or the model is not validated, there is **no trade**, and the NO_TRADE reason says why.

Options/OI and anomaly inputs have no point-in-time history. They are
not part of the score; they can only veto a trade or reduce its probability.

## 4. Statistics

| Tool | Code | Purpose |
|---|---|---|
| Daily-return metrics | `src/backtest/metrics.py` | Sharpe, Sortino and Calmar on daily close-to-close returns, with zero-P&L days counted, annualised by √252. The risk-free rate is converted per day. Undefined ratios are `inf` and flagged (no 999 sentinels). |
| Bootstrap | `src/backtest/optimizer/robustness.py` | i.i.d. and stationary/block bootstrap of daily returns. Gives distributions of final equity, max drawdown and Sharpe, and P(loss). |
| Nested walk-forward | `src/backtest/walk_forward.py::nested_walk_forward` | Chronological folds with an embargo. `fit_fn` sees only the train slice. An untouched holdout is evaluated exactly once. |
| Multiple testing | `src/backtest/overfitting.py` | Probabilistic and Deflated Sharpe (Bailey & López de Prado), expected max Sharpe for N trials, minimum track-record length, and PBO via CSCV. |
| Optimiser | `src/backtest/optimizer/optimization_engine.py` | Splits the data into a selection period and a holdout. Reports DSR with N = all trials. A config is approved only if DSR ≥ 0.95 and it passes the holdout. |

The legacy `WalkForwardValidator.run_walk_forward` runs a **fixed**
configuration on every window. It is a stability check, not an
out-of-sample test, and it now warns that it is not out-of-sample.

## 5. Current results (2026-11-06)

### Calibration (NIFTY50 + BANKNIFTY, ~4,900 labelled samples, 5m bars Mar–Nov 2026)

Trade definition: stop = max(1.0 × ATR, 0.1 % of spot), target = 1.5 × stop, 60-minute horizon.

| | Base win rate | OOS win rate by \|score\| bin (n ≥ 100) | Strongest signals (\|score\| ≥ 1.2), OOS | Validated |
|---|---|---|---|---|
| NIFTY50 | 25 % | 0.36–0.42 | 0.38 | no |
| BANKNIFTY | 29 % | 0.31–0.38 | 0.27 | no |
| Pooled | 27 % | 0.33–0.40 | 0.32 | no |

- Win rate does not rise reliably with score strength.
- The strongest signals are among the worst: entries come after the move.
- The 0.40–0.42 OOS bins on NIFTY do not appear in the training period (train fit ≈ 0.27). That points to a regime difference between the periods, not a stable edge.

An earlier label with a 0.3 % stop floor gave a 2.5–6.5 % base rate. NIFTY rarely moves 0.45 % within 60 minutes, so that definition labelled almost everything a loss. The floor is now 0.1 %, in both the calibration and `RiskConfig.min_sl_pct`.

### Engine replay

NIFTY50, 2026-03-25 → 2026-11-05 (116 sessions, 8,578 decisions, ~76 ms per cycle):

| Mode | Trades | Win rate | Gross | Costs | Net | Max DD |
|---|---|---|---|---|---|---|
| **Live calibration** | 0 | – | – | – | ₹0 | – |
| Diagnostic (all bins P = 0.60) | 41 | 43.9 % | +₹5,899 | ₹3,761 | +₹2,138 | ₹29,573 (5.9 %) |
| Diagnostic, BANKNIFTY | 27 | 40.7 % | +₹1,027 | ₹2,434 | −₹1,407 | ₹32,611 (6.5 %) |

- **Live calibration.** 65 % of decisions stop at "calibration failed out-of-sample validation" and 20 % at the score dead zone. The rest are time and regime rules: too close to close, VOLATILE_CHOPPY, opening auction, expiry after 14:30.
- **Diagnostic statistics (on days with trades):**
  - NIFTY: PSR(SR > 0) = 0.52; stationary-bootstrap P(loss) = 0.48; minimum track record ≈ 16,500 trading days.
  - BANKNIFTY: PSR = 0.48; P(loss) = 0.505.

  Both are statistically indistinguishable from zero. Costs consume most or all of the gross.
- **What binds in diagnostic mode:**
  - the premium R:R check after theta (~23 % of decisions);
  - repeat-signal suppression after a risk rejection (~39 %);
  - the score dead zone (~20 %).

Diagnostic mode only exercises the trade path. **It is not evidence of edge.**

## 6. Reproducing

```bash
# 1. calibrate (writes config/signal_calibration.json; --report-only to just print)
python scripts/calibrate_signal_model.py --db data/db/trading_dss.db --stride 2

# 2. replay the live engine with the real calibration
python scripts/replay_decision_engine.py --db data/db/trading_dss.db \
    --index NIFTY50 --start 2026-03-25 --end 2026-11-05 --out /tmp/replay_live

# 3. DIAGNOSTIC: permissive calibration, to exercise the trade path (NOT evidence of edge)
python scripts/replay_decision_engine.py --db data/db/trading_dss.db \
    --index NIFTY50 --start 2026-03-25 --end 2026-11-05 --diagnostic --out /tmp/replay_diag
```

Each replay writes `replay_result.json`, `trades.csv`, `daily_pnl.csv`
and `decision_counts.csv`.

## 7. What would change the verdict

- **Features with information the price series lacks:**
  - near-month futures volume, OI and order-flow imbalance (`futures_depth_1m`, now being collected);
  - real option-chain history collected during market hours (positioning, IV, skew).
- **More history:** at least one year of point-in-time data, to reach the minimum track-record length and to cover different volatility regimes.
- **Once a candidate passes calibration:**
  - nested walk-forward of calibrate-then-replay;
  - the DSR computed with every variant ever tried;
  - 4–6 weeks of premium-native paper trading (`src/paper_trading/adapter.py`) before real capital.
