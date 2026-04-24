
import sys
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import DatabaseManager
from src.engine.decision_engine import DecisionEngine
from src.engine.signal_generator import SignalGenerator
from config.settings import settings

def run_analysis():
    db = DatabaseManager.instance()
    db.connect()
    engine = DecisionEngine(db)
    
    # We want to see what a signal would look like at peak confluence today
    simulated_now = datetime(2026, 4, 21, 14, 30, 0, tzinfo=ZoneInfo('Asia/Kolkata'))
    
    index_id = "BANKNIFTY"
    print(f"\n--- EXHUMING HIDDEN SIGNAL FOR {index_id} ---")
    
    try:
        (
            current_bar,
            daily_rows,
            intraday_rows,
            options_chain,
            oi_summary,
            vix_value,
            benchmark_rows,
        ) = engine._step_gather_data(index_id)
        
        price_df = engine._build_price_df(daily_rows, intraday_rows)
        current_price = float(current_bar["close"]) if current_bar else 0.0
        
        tech = engine._step_technical(index_id, price_df, options_chain, vix_value, benchmark_rows)
        anom_res = engine._step_anomaly(index_id, current_bar, daily_rows, options_chain, oi_summary)
        
        # Convert result to vote like DecisionEngine does
        anom_vote = engine._anomaly_result_to_vote(anom_res, index_id)
        
        regime = engine._step_regime(index_id, price_df, tech, None, anom_res, vix_value)
        
        # Signal Generation (Simulated Time)
        signal = engine.signal_generator.generate_signal(
            index_id=index_id,
            technical=tech,
            news=None,
            anomaly=anom_vote,
            regime=regime,
            current_spot_price=current_price,
            timestamp=simulated_now,
            mode="TECHNICAL_OPTIONS"
        )
        
        print("="*64)
        print(f"  POTENTIAL SIGNAL: {signal.signal_type}")
        print("="*64)
        if signal.signal_type != "NO_TRADE":
            print(f"  Confidence  : {signal.confidence_level} ({signal.confidence_score:.2f})")
            print(f"  Entry Price : {signal.entry_price}")
            print(f"  Target      : {signal.target_price}")
            print(f"  Stop Loss   : {signal.stop_loss}")
            print(f"  RR Ratio    : {signal.risk_reward_ratio:.2f}")
            print(f"  Regime      : {signal.regime}")
            print("-" * 64)
            print(f"  REASONING:\n{signal.reasoning}")
        else:
            print(f"  Result: {signal.signal_type}")
            print(f"  Reason: {signal.reasoning}")

    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    run_analysis()
