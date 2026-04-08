"""
Validation script for RiskManager (Phase 5.3).

Runs the full Phase 2 → 5.1 → 5.2 → 5.3 pipeline:
  1. Downloads real NIFTY50 history via yfinance
  2. Runs TechnicalAggregator, RegimeDetector, SignalGenerator
  3. Builds a realistic mock options chain around the live spot price
  4. Feeds the signal into RiskManager.validate_and_refine_signal()
  5. Prints the full RefinedSignal and portfolio summary

If the live signal is NO_TRADE (common in neutral markets), a synthetic
BUY_CALL signal is also run to demonstrate the full risk-management output.
"""

import logging
import os
import uuid
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.WARNING)
_IST = ZoneInfo("Asia/Kolkata")

SEP  = "=" * 65
SEP2 = "-" * 65

# ── DB ────────────────────────────────────────────────────────────────────────
from src.database.db_manager import DatabaseManager
from src.data.index_registry import get_registry

DB_PATH = "data/db/validate_risk_manager.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

db = DatabaseManager(DB_PATH)
db.connect()
db.initialise_schema()
registry = get_registry()
registry.sync_to_db(db)
print("✓ DB + schema ready")

# ── Real NIFTY50 data ─────────────────────────────────────────────────────────
from src.data.historical_data import HistoricalDataManager

hm = HistoricalDataManager(registry=registry, db=db)
print("Downloading NIFTY50 history (1y daily)…")
df = hm.download_index_history("NIFTY50", period="1y", interval="1d")
spot_price = float(df["close"].iloc[-1])
print(f"✓ {len(df)} bars  |  latest close: ₹{spot_price:,.2f}\n")
df_indexed = df.set_index("timestamp")

# ── Phase 2: Technical Analysis ───────────────────────────────────────────────
from src.analysis.technical_aggregator import TechnicalAggregator

tech_agg    = TechnicalAggregator()
tech_result = tech_agg.analyze("NIFTY50", df_indexed, vix_value=15.0, timeframe="1d")
atr = tech_result.volatility.atr_value
print(f"✓ Technical: {tech_result.overall_signal}  conf={tech_result.overall_confidence:.2f}")
print(f"  ATR={atr:.1f}  RSI={tech_result.momentum.rsi_value:.1f}")
print(f"  Support=₹{tech_result.immediate_support:,.0f}  Resistance=₹{tech_result.immediate_resistance:,.0f}")

# ── Phase 5.1: Regime Detection ───────────────────────────────────────────────
from src.engine.regime_detector import RegimeDetector

regime_detector = RegimeDetector(db)
regime = regime_detector.detect_regime("NIFTY50", df_indexed, tech_result, vix_value=15.0)
print(f"\n✓ Regime: {regime.regime}  conf={regime.regime_confidence:.2f}")
print(f"  Pos. multiplier={regime.position_size_multiplier:.2f}  Max trades={regime.max_trades_today}")

# ── Mock Phase 3 & 4 votes ────────────────────────────────────────────────────
from src.analysis.news.news_engine import NewsVote
from src.analysis.anomaly.anomaly_aggregator import AnomalyVote

news_vote = NewsVote(
    index_id="NIFTY50",
    timestamp=datetime.now(tz=_IST),
    vote="BULLISH",
    confidence=0.60,
    active_article_count=7,
    weighted_sentiment=0.32,
    top_headline="India markets steady ahead of RBI policy decision",
    event_regime="NORMAL",
    reasoning="7 articles with mild positive sentiment.",
)
anomaly_vote = AnomalyVote(
    index_id="NIFTY50",
    vote="NEUTRAL",
    confidence=0.45,
    risk_level="NORMAL",
    position_size_modifier=1.0,
    active_alerts=0,
    primary_alert_message=None,
    institutional_activity=False,
    reasoning="No significant anomalies detected.",
)
print(f"\n✓ News: {news_vote.vote} ({news_vote.confidence:.2f})")
print(f"✓ Anomaly: {anomaly_vote.vote}  risk={anomaly_vote.risk_level}")

# ── Phase 5.2: Signal Generation ─────────────────────────────────────────────
from src.engine.signal_generator import SignalGenerator
from unittest.mock import patch

generator = SignalGenerator(db)
print("\nRunning Phase 5.2 — SignalGenerator…")
with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
    signal = generator.generate_signal(
        index_id="NIFTY50",
        technical=tech_result,
        news=news_vote,
        anomaly=anomaly_vote,
        regime=regime,
        current_spot_price=spot_price,
    )

print(f"✓ Live signal: {signal.signal_type}  conf={signal.confidence_level}  score={signal.weighted_score:+.4f}")

# ── Build realistic mock options chain ────────────────────────────────────────
from src.data.options_chain import OptionsChainData, OptionStrike

def _build_mock_chain(spot: float, index_id: str = "NIFTY50") -> OptionsChainData:
    """
    Realistic mock chain centred on spot.
    PE OI peak at ATM-200 (support), CE OI peak at ATM+300 (resistance).
    Premiums via Black-Scholes at 15% IV, ~1 week to expiry.
    """
    import math
    from statistics import NormalDist

    S, T, r, sigma = spot, 7 / 365, 0.065, 0.15
    nd = NormalDist()

    def bs_call(K):
        if T <= 0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return round(max(0.1, S * nd.cdf(d1) - K * math.exp(-r * T) * nd.cdf(d2)), 2)

    def bs_put(K):
        return round(max(0.1, bs_call(K) - S + K * math.exp(-r * T)), 2)

    atm = round(spot / 50) * 50
    support_strike    = atm - 200
    resistance_strike = atm + 300

    # Next Thursday expiry
    today = date.today()
    days_to_thu = (3 - today.weekday()) % 7 or 7
    expiry      = today + timedelta(days=days_to_thu)
    next_expiry = expiry + timedelta(days=7)

    strike_objects = []
    for k in range(int(atm - 1000), int(atm + 1050), 50):
        dist = abs(k - atm)
        ce_oi = 1_500_000 if k == resistance_strike else max(50_000, 400_000 - dist * 500)
        pe_oi = 1_200_000 if k == support_strike    else max(50_000, 400_000 - dist * 500)
        strike_objects.append(OptionStrike(
            strike_price=float(k),
            ce_oi=ce_oi,    ce_oi_change=int(ce_oi * 0.02),
            ce_volume=max(1000, int(ce_oi * 0.05)),
            ce_ltp=bs_call(k), ce_iv=15.0,
            pe_oi=pe_oi,    pe_oi_change=int(pe_oi * 0.02),
            pe_volume=max(1000, int(pe_oi * 0.05)),
            pe_ltp=bs_put(k),  pe_iv=15.0,
        ))

    return OptionsChainData(
        index_id=index_id,
        spot_price=spot,
        timestamp=datetime.now(tz=_IST),
        expiry_date=expiry,
        strikes=tuple(strike_objects),
        available_expiries=(expiry, next_expiry),
    )

chain = _build_mock_chain(spot_price)
atm_strike = round(spot_price / 50) * 50
print(f"\n✓ Mock options chain: {len(chain.strikes)} strikes")
print(f"  ATM=₹{atm_strike:,}  |  Support(PE peak)=₹{atm_strike-200:,}  |  Resistance(CE peak)=₹{atm_strike+300:,}")
print(f"  Expiry={chain.expiry_date}  Next={chain.available_expiries[1]}")

# ── Phase 5.3: Risk Manager ───────────────────────────────────────────────────
from src.engine.risk_manager import RiskConfig, RiskManager
from src.engine.signal_generator import TradingSignal

config = RiskConfig(
    total_capital=500_000.0,        # ₹5L — realistic retail F&O capital for NIFTY
    max_risk_per_trade_pct=2.0,     # ₹10,000 max per trade
    max_risk_per_day_pct=5.0,       # ₹25,000 daily limit
    max_open_positions=3,
    max_lots_per_trade=5,
)
risk_mgr = RiskManager(db, config=config)
sim_now  = datetime.now(_IST).replace(hour=10, minute=30, second=0, microsecond=0)

# Decide which signal(s) to run through the risk manager
signals_to_run: list[tuple[str, TradingSignal]] = []

if signal.signal_type != "NO_TRADE":
    signals_to_run.append(("LIVE SIGNAL", signal))
else:
    # Live market is neutral today — show result but also run a synthetic signal
    signals_to_run.append(("LIVE SIGNAL (NO_TRADE — market neutral today)", signal))

# Always add a synthetic BUY_CALL to demonstrate full risk-manager output
sl_dist  = max(80.0, atr * 0.5)           # ~half ATR as stop distance
tgt_dist = sl_dist * 1.8                   # 1.8× RR
synth = TradingSignal(
    signal_id=str(uuid.uuid4()),
    index_id="NIFTY50",
    generated_at=sim_now,
    signal_type="BUY_CALL",
    confidence_level="HIGH",        # HIGH bypasses OI resistance cap on target
    confidence_score=0.75,
    entry_price=spot_price,
    target_price=round(spot_price + tgt_dist, 2),
    stop_loss=round(spot_price - sl_dist, 2),
    risk_reward_ratio=round(tgt_dist / sl_dist, 2),
    regime=regime.regime,
    weighted_score=0.82,
    vote_breakdown={},
    risk_level="NORMAL",
    position_size_modifier=1.0,
    suggested_lot_count=2,
    estimated_max_loss=sl_dist * 75 * 2,
    estimated_max_profit=tgt_dist * 75 * 2,
    reasoning=f"Synthetic HIGH BUY_CALL for validation. SL={sl_dist:.0f}pts, Target={tgt_dist:.0f}pts, ATR={atr:.0f}",
    warnings=[],
    data_completeness=1.0,
    signals_generated_today=1,
)
signals_to_run.append((
    f"SYNTHETIC HIGH BUY_CALL (demo — SL={sl_dist:.0f}pts / Target={tgt_dist:.0f}pts / ATR={atr:.0f})",
    synth,
))


def _print_refined(label: str, refined) -> None:
    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)

    status = "✅  EXECUTABLE" if refined.is_valid else "❌  REJECTED"
    print(f"  STATUS       : {status}")
    if refined.rejection_reasons:
        for r in refined.rejection_reasons:
            print(f"    Reason: {r}")
        print()

    print(f"  Signal type  : {refined.signal_type}")
    print(f"  Confidence   : {refined.confidence_level} ({refined.confidence_score:.4f})")

    print(f"\n{SEP2}")
    print("  PRICE LEVELS")
    print(SEP2)
    orig_e = refined.entry_price
    orig_t = refined.target_price
    orig_s = refined.stop_loss
    print(f"  Entry       : ₹{refined.refined_entry:>10,.2f}  (original: ₹{orig_e:,.2f})")
    print(f"  Target      : ₹{refined.refined_target:>10,.2f}  (original: ₹{orig_t:,.2f})")
    print(f"  Stop Loss   : ₹{refined.refined_stop_loss:>10,.2f}  (original: ₹{orig_s:,.2f})")
    print(f"  Risk:Reward : 1:{refined.risk_reward_ratio:.2f}")

    if refined.is_valid:
        print(f"\n{SEP2}")
        print("  POSITION SIZING")
        print(SEP2)
        print(f"  Lots        : {refined.lots}")
        print(f"  Risk amount : ₹{refined.risk_amount:>10,.0f}  ({refined.risk_pct_of_capital:.2f}% of capital)")
        print(f"  Max loss    : ₹{refined.max_loss_amount:>10,.0f}  (incl. transaction costs)")
        print(f"  Max profit  : ₹{refined.max_profit_amount:>10,.0f}")
        print(f"  Tx cost     : ₹{refined.transaction_cost_total:>10,.0f}  (round-trip)")
        print(f"  Margin req. : ₹{refined.total_margin_required:>10,.0f}")
        print(f"  Breakeven   : {refined.breakeven_move:>6.1f} pts  to cover costs")
        print(f"  Daily remain: ₹{refined.daily_loss_remaining:>10,.0f}")

        if refined.recommended_strike:
            print(f"\n{SEP2}")
            print("  OPTION DETAILS")
            print(SEP2)
            print(f"  Strike      : ₹{refined.recommended_strike:,.0f}")
            print(f"  Expiry      : {refined.recommended_expiry}")
            if refined.option_premium:
                print(f"  Premium     : ₹{refined.option_premium:,.2f}  per unit")
                print(f"  Cost/lot    : ₹{refined.option_premium * 75:,.0f}  (75 units)")
            if refined.option_greeks:
                g = refined.option_greeks
                if g.get("delta") is not None:
                    print(f"  Delta       : {g['delta']:+.4f}")
                if g.get("theta") is not None:
                    print(f"  Theta       : {g['theta']:+.4f} pts/day")

        print(f"\n{SEP2}")
        print("  EXECUTION INSTRUCTIONS")
        print(SEP2)
        print(f"  Order type  : {refined.execution_type}")
        if refined.limit_price:
            print(f"  Limit price : ₹{refined.limit_price:,.2f}")
        print(f"  Validity    : {refined.validity}")

    if refined.adjustments_made:
        print(f"\n{SEP2}")
        print("  ADJUSTMENTS MADE")
        print(SEP2)
        for adj in refined.adjustments_made:
            print(f"  ✎  {adj}")

    if refined.warnings:
        print(f"\n{SEP2}")
        print("  WARNINGS")
        print(SEP2)
        for w in refined.warnings:
            print(f"  ⚠  {w}")


# ── Run signals through Risk Manager ─────────────────────────────────────────
print(f"\n\nRunning Phase 5.3 — RiskManager  (simulating {sim_now.strftime('%H:%M IST')})")
for label, sig in signals_to_run:
    refined = risk_mgr.validate_and_refine_signal(sig, current_chain=chain, _now=sim_now)
    _print_refined(label, refined)

# ── Portfolio summary ─────────────────────────────────────────────────────────
portfolio = risk_mgr.get_portfolio_summary()
daily     = risk_mgr.get_daily_pnl_summary()

print(f"\n{SEP}")
print("  PORTFOLIO SUMMARY")
print(SEP)
print(f"  Total capital        : ₹{portfolio.total_capital:>10,.0f}")
print(f"  Current exposure     : ₹{portfolio.current_exposure:>10,.0f}  (margin in open positions)")
print(f"  Available capital    : ₹{portfolio.available_capital:>10,.0f}")
print(f"  Open positions       : {portfolio.total_trades_today} (today)  |  {len(portfolio.open_positions)} open now")
print(f"  Today P&L            : ₹{portfolio.today_pnl:>+10,.0f}")
print(f"  Daily limit remaining: {portfolio.daily_limit_remaining_pct:>5.0f}%")

print(f"\n{SEP2}")
print("  DAILY P&L DETAIL")
print(SEP2)
print(f"  Date                 : {daily.date}")
print(f"  Realised P&L         : ₹{daily.total_pnl:>+10,.0f}  ({daily.total_pnl_pct:+.2f}% of capital)")
print(f"  Trades               : {daily.total_trades}  |  W={daily.winning_trades}  L={daily.losing_trades}  Win%={daily.win_rate:.0%}")
print(f"  Largest win          : ₹{daily.largest_win:>+10,.0f}")
print(f"  Largest loss         : ₹{daily.largest_loss:>+10,.0f}")
print(f"  Risk used today      : {daily.risk_used_pct:.1f}%  |  Remaining: ₹{daily.remaining_risk:,.0f}")
print(f"  Open positions       : {daily.open_positions}")

print(f"\n{SEP}")
print("  Phase 5.3 validation complete.")
print(SEP)
