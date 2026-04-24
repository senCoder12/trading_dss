
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.append(os.getcwd())

from src.database.db_manager import DatabaseManager
from src.engine.decision_engine import DecisionEngine
from config.settings import settings

def run_test():
    db = DatabaseManager(settings.database.path)
    engine = DecisionEngine(db)
    
    # Get the latest price point for context, but we will simulate current time as 11:30 AM
    row = db.fetch_one("SELECT timestamp FROM price_data WHERE index_id = 'BANKNIFTY' ORDER BY timestamp DESC LIMIT 1")
    if not row:
        print("No data found for BANKNIFTY")
        return
        
    # Create a simulated "mid-market" timestamp: 2026-04-20 11:30:00 IST
    simulated_now = datetime(2026, 4, 20, 11, 30, 0, tzinfo=ZoneInfo('Asia/Kolkata'))
    
    print(f"SIMULATION: Evaluating BANKNIFTY logic as if it were 11:30 AM IST on April 20th.")
    print("Skipping live news fetch for speed; using technical and anomaly data only.\n")
    
    # We'll use the technical, news, and anomaly steps directly
    (cb, dr, ir, oc, oi, vix, bmr) = engine._step_gather_data("BANKNIFTY")
    pdf = engine._build_price_df(dr, ir)
    cp = float(cb["close"]) if cb else 0.0
    
    tech = engine._step_technical("BANKNIFTY", pdf, oc, vix, bmr)
    news = None  # Skip news fetch to ensure immediate result
    anom = engine._step_anomaly("BANKNIFTY", cb, ir, oc, oi)
    regime = engine._step_regime("BANKNIFTY", pdf, tech, None, anom, vix)
    anom_vote = engine._anomaly_result_to_vote(anom, "BANKNIFTY")
    
    # Generate the signal using the simulated timestamp
    signal = engine.signal_generator.generate_signal(
        "BANKNIFTY", tech, news, anom_vote, regime, cp, 
        timestamp=simulated_now,
        mode="TECHNICAL_ONLY"
    )
    
    print("="*60)
    print(f" SIMULATED SIGNAL (TECHNICAL): {signal.signal_type}")
    print("="*60)
    print(f"Confidence:    {signal.confidence_level} ({signal.confidence_score:.4f})")
    print(f"Weighted Score: {signal.weighted_score:.4f}")
    print(f"Market Regime: {signal.regime}")
    print("-"*60)
    print(f"Entry:         {signal.entry_price}")
    print(f"Target:        {signal.target_price}")
    print(f"Stop Loss:     {signal.stop_loss}")
    print(f"Risk/Reward:   {signal.risk_reward_ratio}")
    print("-"*60)
    print(f"REASONING:\n{signal.reasoning}")
    
    if signal.warnings:
        print(f"\nWarnings: {', '.join(signal.warnings)}")
    print("="*60)

if __name__ == "__main__":
    run_test()
