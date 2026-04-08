"""
Validation script for SignalGenerator (Phase 5.2).

Downloads real NIFTY50 history via yfinance, runs the full Phase 2→5.1→5.2
pipeline, and prints the generated TradingSignal.
"""

import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.WARNING)

# ── DB setup ──────────────────────────────────────────────────────────────────
from src.database.db_manager import DatabaseManager
from src.data.index_registry import get_registry

DB_PATH = "data/db/validate_signal_gen.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

db = DatabaseManager(DB_PATH)
db.connect()
db.initialise_schema()

# Seed index_master (required for FK in trading_signals)
registry = get_registry()
registry.sync_to_db(db)
print("✓ DB + schema ready\n")

# ── Download real NIFTY50 data ─────────────────────────────────────────────────
from src.data.historical_data import HistoricalDataManager

hm = HistoricalDataManager(registry=registry, db=db)
print("Downloading NIFTY50 history (1y daily)…")
df = hm.download_index_history("NIFTY50", period="1y", interval="1d")
print(f"✓ {len(df)} bars  |  columns: {list(df.columns)}")
print(f"  Date range: {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")
spot_price = float(df["close"].iloc[-1])
print(f"  Latest close: ₹{spot_price:,.2f}\n")

# yfinance returns a 'timestamp' column after our reset_index; TechnicalAggregator
# expects the DataFrame indexed by timestamp or with open/high/low/close/volume cols.
# Set index back to timestamp for indicator calculations.
df_indexed = df.set_index("timestamp")

# ── Phase 2: Technical Analysis ───────────────────────────────────────────────
from src.analysis.technical_aggregator import TechnicalAggregator

tech_agg = TechnicalAggregator()
print("Running Phase 2 — TechnicalAggregator…")
tech_result = tech_agg.analyze(
    "NIFTY50",
    df_indexed,
    vix_value=15.0,        # reasonable VIX for current market
    timeframe="1d",
)
print(f"✓ Technical signal: {tech_result.overall_signal}  (conf: {tech_result.overall_confidence:.2f})")
print(f"  Trend: {tech_result.trend.trend_vote}  |  Momentum: {tech_result.momentum.momentum_vote}")
print(f"  Volume: {tech_result.volume.volume_vote}  |  Quant: {tech_result.quant.quant_vote}")
print(f"  Immediate support: {tech_result.immediate_support:,.0f}  |  resistance: {tech_result.immediate_resistance:,.0f}")
print(f"  ATR: {tech_result.volatility.atr_value:.1f}  |  RSI: {tech_result.momentum.rsi_value:.1f}")
if tech_result.smart_money:
    print(f"  Smart money: {tech_result.smart_money.smart_money_bias} (grade {tech_result.smart_money.grade})")
print()

# ── Phase 5.1: Regime Detection ───────────────────────────────────────────────
from src.engine.regime_detector import RegimeDetector

regime_detector = RegimeDetector(db)
print("Running Phase 5.1 — RegimeDetector…")
regime = regime_detector.detect_regime(
    "NIFTY50",
    df_indexed,
    tech_result,
    vix_value=15.0,
)
print(f"✓ Regime: {regime.regime}  (conf: {regime.regime_confidence:.2f})")
print(f"  Volatility: {regime.volatility_regime}  |  Event: {regime.event_regime}")
print(f"  Position multiplier: {regime.position_size_multiplier:.2f}  |  Max trades: {regime.max_trades_today}")
print(f"  Description: {regime.description}")
is_safe, reason = RegimeDetector.is_safe_to_trade(regime)
print(f"  Safe to trade: {is_safe}  ({reason})")
print()

# ── Mock Phase 3 NewsVote ─────────────────────────────────────────────────────
from src.analysis.news.news_engine import NewsVote

news_vote = NewsVote(
    index_id="NIFTY50",
    timestamp=datetime.now(tz=ZoneInfo("Asia/Kolkata")),
    vote="BULLISH",
    confidence=0.60,
    active_article_count=7,
    weighted_sentiment=0.32,
    top_headline="India markets steady ahead of RBI policy decision",
    event_regime="NORMAL",
    reasoning="7 articles with mild positive sentiment; no critical events.",
)
print(f"Mock Phase 3 NewsVote: {news_vote.vote} (conf: {news_vote.confidence:.2f}, {news_vote.active_article_count} articles)")

# ── Mock Phase 4 AnomalyVote ──────────────────────────────────────────────────
from src.analysis.anomaly.anomaly_aggregator import AnomalyVote

anomaly_vote = AnomalyVote(
    index_id="NIFTY50",
    vote="NEUTRAL",
    confidence=0.45,
    risk_level="NORMAL",
    position_size_modifier=1.0,
    active_alerts=0,
    primary_alert_message=None,
    institutional_activity=False,
    reasoning="No significant anomalies detected in volume, price, or OI.",
)
print(f"Mock Phase 4 AnomalyVote: {anomaly_vote.vote}  risk={anomaly_vote.risk_level}\n")

# ── Phase 5.2: Signal Generation ─────────────────────────────────────────────
from src.engine.signal_generator import SignalGenerator
from unittest.mock import patch

generator = SignalGenerator(db)
print("Running Phase 5.2 — SignalGenerator…")
print("  (market-close guard bypassed for validation — simulating 10:30 AM IST)\n")
# Pre-compute votes for diagnostic display (even on NO_TRADE)
from src.engine.signal_generator import _VOTE_NUMERIC

def _num(v):
    return _VOTE_NUMERIC.get((v or "NEUTRAL").upper(), 0.0)

from src.engine.regime_detector import SignalWeights
available = {"trend", "momentum", "volume"}
if tech_result.options:      available.add("options")
if tech_result.smart_money:  available.add("smart_money")
available.add("news")
available.add("anomaly")

raw_votes_for_display = {
    "trend":       (_num(tech_result.trend.trend_vote),            tech_result.trend.trend_confidence),
    "momentum":    (_num(tech_result.momentum.momentum_vote),      tech_result.momentum.momentum_confidence),
    "volume":      (_num(tech_result.volume.volume_vote),          tech_result.volume.volume_confidence),
    "smart_money": (_num(tech_result.smart_money.smart_money_bias), tech_result.smart_money.confidence) if tech_result.smart_money else None,
    "news":        (_num(news_vote.vote),                          news_vote.confidence),
    "anomaly":     (_num(anomaly_vote.vote),                       anomaly_vote.confidence),
}

with patch.object(SignalGenerator, "_is_near_market_close", return_value=False):
    signal = generator.generate_signal(
        index_id="NIFTY50",
        technical=tech_result,
        news=news_vote,
        anomaly=anomaly_vote,
        regime=regime,
        current_spot_price=spot_price,
    )

# ── Output ────────────────────────────────────────────────────────────────────
SEP = "=" * 65

print(f"\n{SEP}")
print("  TRADING SIGNAL OUTPUT")
print(SEP)
print(signal.reasoning)

print(f"\n{SEP}")
print("  SIGNAL SUMMARY")
print(SEP)
print(f"  Signal ID   : {signal.signal_id[:12]}…")
print(f"  Type        : {signal.signal_type}")
print(f"  Confidence  : {signal.confidence_level}  ({signal.confidence_score:.4f})")
print(f"  Entry       : ₹{signal.entry_price:,.2f}")
print(f"  Target      : ₹{signal.target_price:,.2f}")
print(f"  Stop Loss   : ₹{signal.stop_loss:,.2f}")
print(f"  R:R         : 1:{signal.risk_reward_ratio:.2f}")
print(f"  Regime      : {signal.regime}")
print(f"  Score       : {signal.weighted_score:+.4f}")
print(f"  Risk level  : {signal.risk_level}")
print(f"  Pos. modifier: {signal.position_size_modifier:.4f}  → {signal.suggested_lot_count} lot(s)")
print(f"  Max loss    : {signal.estimated_max_loss:,.2f} pts  (if SL hit)")
print(f"  Max profit  : {signal.estimated_max_profit:,.2f} pts  (if target hit)")
print(f"  Data quality: {signal.data_completeness:.0%}")
print(f"  Today count : {signal.signals_generated_today}")

print(f"\n{SEP}")
print("  VOTE BREAKDOWN")
print(SEP)
for comp, detail in signal.vote_breakdown.items():
    bar_score = detail['score']
    direction = "↑" if bar_score > 0 else ("↓" if bar_score < 0 else "→")
    print(
        f"  {comp:<14} score={bar_score:+.1f} {direction}  "
        f"conf={detail['confidence']:.2f}  "
        f"wt={detail['weight']:.3f}  "
        f"contrib={detail['weighted_contribution']:+.4f}"
    )

if signal.warnings:
    print(f"\n{SEP}")
    print("  WARNINGS")
    print(SEP)
    for w in signal.warnings:
        print(f"  ⚠  {w}")

# Show raw votes even for NO_TRADE (diagnostic only)
if signal.signal_type == "NO_TRADE":
    print(f"\n{SEP}")
    print("  RAW VOTES (diagnostic — why no edge?)")
    print(SEP)
    for comp, val in raw_votes_for_display.items():
        if val is None:
            continue
        score, conf = val
        arrow = "↑" if score > 0 else ("↓" if score < 0 else "→")
        print(f"  {comp:<14} score={score:+.1f} {arrow}  conf={conf:.2f}")
    print(f"\n  Weighted score: {signal.weighted_score:+.4f}  (dead zone = ±0.40)")
    print(f"  → No trade: score not convincing enough in either direction.")

# Verify DB write
if signal.signal_type != "NO_TRADE":
    row = db.fetch_one("SELECT COUNT(*) as c FROM trading_signals")
    print(f"\n✓ DB: {row['c']} signal(s) persisted to trading_signals")
