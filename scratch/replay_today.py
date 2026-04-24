import sys
import os
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
import logging
from unittest.mock import patch

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.database.db_manager import DatabaseManager
from src.engine.decision_engine import DecisionEngine
from src.data.index_registry import get_registry

logging.basicConfig(level=logging.ERROR) # Only show errors
logger = logging.getLogger("Replay")

_IST = ZoneInfo("Asia/Kolkata")

def replay():
    db = DatabaseManager()
    engine = DecisionEngine(db)
    registry = get_registry()
    
    start_time = datetime(2026, 4, 22, 10, 25, tzinfo=_IST)
    end_time = datetime(2026, 4, 22, 15, 30, tzinfo=_IST)
    
    current = start_time
    interval = timedelta(minutes=5)
    
    fo_indices = ["NIFTY50", "BANKNIFTY"]
    
    print(f"{'Time':<10} | {'Index':<10} | {'Signal':<10} | {'Conf':<6} | {'Reason'}")
    print("-" * 80)
    
    while current <= end_time:
        with patch('src.engine.decision_engine.datetime') as mock_datetime:
            mock_datetime.now.return_value = current
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            with patch('src.analysis.news.news_engine.NewsEngine.run_news_cycle', return_value=[]):
                with patch('src.analysis.news.rss_fetcher.RSSFetcher.fetch_all_feeds', return_value=[]):
                    with patch('src.engine.signal_generator.datetime') as mock_dt_sig:
                        mock_dt_sig.now.return_value = current
                        
                        for index_id in fo_indices:
                            try:
                                with patch.object(engine.tracker, 'record_signal') as mock_record:
                                    result = engine.run_full_cycle(index_id, as_of=current)
                                    
                                    sig = result.signal
                                    is_actionable = result.is_actionable
                                    
                                    if sig.signal_type != "NO_TRADE" or current.minute % 30 == 0:
                                        regime_name = result.regime.regime if result.regime else "N/A"
                                        score = sig.weighted_score
                                        
                                        # Check if it was a hero zero candidate
                                        is_exp = (result.technical_result.options and result.technical_result.options.days_to_expiry == 0) if result.technical_result else False
                                        is_aft = current.hour >= 13 and (current.hour > 13 or current.minute >= 30)
                                        is_hz = is_exp and is_aft
                                        
                                        print(f"{current.strftime('%H:%M'):<10} | {index_id:<10} | {sig.signal_type:<10} | {sig.confidence_level:<6} | Score: {score:>5.2f} | Regime: {regime_name:<15} | HZ: {is_hz}")
                                        if sig.signal_type != "NO_TRADE":
                                             print(f"  Reason: {result.alert_message}")
                                             # print(f"  Breakdown: {sig.vote_breakdown}")
                            except Exception as e:
                                print(f"Error at {current} for {index_id}: {e}")
                                # import traceback
                                # traceback.print_exc()
        
        current += interval

if __name__ == "__main__":
    replay()
