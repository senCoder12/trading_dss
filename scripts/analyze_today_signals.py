
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import DatabaseManager
from src.engine.decision_engine import DecisionEngine
from config.settings import settings

def run_analysis():
    db = DatabaseManager.instance()
    db.connect()
    engine = DecisionEngine(db)
    
    # Fool the news engine into thinking a cycle just ran
    engine._last_news_cycle = datetime.now(tz=ZoneInfo('Asia/Kolkata'))
    
    index_id = "NIFTY50"
    
    print(f"\n--- Decision Engine Analysis (Post-Market) ---")
    
    try:
        results = engine.run_all_indices()
        
        for res in results:
            if res.index_id not in ["NIFTY50", "BANKNIFTY"]:
                continue
                
            sig = res.signal
            print("\n" + "="*64)
            print(f"  SIGNAL FOR {res.index_id}: {getattr(sig, 'signal_type', 'N/A')}")
            print("="*64)
            print(f"  Confidence  : {getattr(sig, 'confidence_level', 'N/A')} ({getattr(sig, 'confidence_score', 0.0):.2f})")
            print(f"  Regime      : {res.regime.regime if res.regime else 'UNKNOWN'}")
            print("-" * 64)
            print("  VOTES:")
            if res.technical_result:
                print(f"    Technical : {res.technical_result.overall_signal}")
                if "options" in res.technical_result.votes:
                    print(f"    Options   : {res.technical_result.votes['options']}")
            if res.news_vote:
                print(f"    News      : {res.news_vote.vote}")
            if res.anomaly_vote:
                print(f"    Anomaly   : {res.anomaly_vote.vote}")
            print("-" * 64)
            print(f"  REASONING: {getattr(sig, 'reasoning', 'N/A')}")
            
            if res.warnings:
                print(f"  WARNINGS: {', '.join(res.warnings[:3])}")

    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    run_analysis()
