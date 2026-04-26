"""
Analyze generated signals against historical price data.
Calculates success ratio based on entry, target, and stop loss prices.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import IST_TIMEZONE
from src.database.db_manager import DatabaseManager

_IST = ZoneInfo(IST_TIMEZONE)

def analyze_signal_performance(db):
    """Analyze each generated signal against post-signal price data."""

    # Fetch historical signals (generated at past dates, not today)
    signals = db.fetch_all("""
        SELECT id, index_id, generated_at, signal_type, entry_price,
               target_price, stop_loss, confidence_level, regime
        FROM trading_signals
        WHERE signal_type IN ('BUY_CALL', 'BUY_PUT')
          AND DATE(generated_at) < DATE('now')
        ORDER BY generated_at ASC
        LIMIT 200
    """)

    if not signals:
        print("No signals found to analyze.")
        return

    print(f"\n📊 Analyzing {len(signals)} signals against price data...\n")
    print(f"{'Index':<15} {'Signal':<12} {'Entry':<12} {'Target':<12} {'SL':<12} {'Result':<15} {'Bars Used':<12}")
    print("─" * 98)

    winning = 0
    losing = 0
    inconclusive = 0

    for sig in signals:
        index_id = sig['index_id']
        sig_time = sig['generated_at']
        entry = sig['entry_price']
        target = sig['target_price']
        stop_loss = sig['stop_loss']
        signal_type = sig['signal_type']

        # Fetch price data AFTER signal generation (next 20 bars = 20 days)
        # Using the same timeframe the signal was based on
        bars = db.fetch_all("""
            SELECT timestamp, open, high, low, close
            FROM price_data
            WHERE index_id = ? AND timeframe = '1d' AND timestamp > ?
            ORDER BY timestamp ASC
            LIMIT 20
        """, (index_id, sig_time))

        if not bars:
            result = "NO_DATA"
            bars_used = 0
        else:
            result, bars_used = evaluate_signal(signal_type, entry, target, stop_loss, bars)
            if result == "WIN":
                winning += 1
            elif result == "LOSE":
                losing += 1
            else:
                inconclusive += 1

        print(f"{index_id:<15} {signal_type:<12} {entry:<12.2f} {target:<12.2f} {stop_loss:<12.2f} {result:<15} {bars_used:<12}")

    total = winning + losing + inconclusive

    print("\n" + "─" * 98)
    print(f"\n✅ Results Summary:")
    print(f"   Winning signals:      {winning}/{total} ({winning/total*100:.1f}%) - Target hit before SL")
    print(f"   Losing signals:       {losing}/{total} ({losing/total*100:.1f}%) - SL hit before target")
    print(f"   Inconclusive:         {inconclusive}/{total} ({inconclusive/total*100:.1f}%) - No clear outcome in 20 bars")

    if total > 0:
        success_ratio = (winning / total) * 100
        print(f"\n🎯 SUCCESS RATIO: {success_ratio:.1f}%")
        print(f"📈 Win/Loss Ratio: {winning}W : {losing}L")

        if winning > 0 and losing > 0:
            avg_win = winning / losing if losing > 0 else 0
            print(f"💰 Profit Factor: {avg_win:.2f}x")


def evaluate_signal(signal_type: str, entry: float, target: float, stop_loss: float, bars: list) -> tuple:
    """
    Evaluate if signal hit target or stop loss first.

    For BUY_CALL: price needs to go UP
    For BUY_PUT: price needs to go DOWN

    Returns: (result, bars_used)
    """

    for i, bar in enumerate(bars):
        high = bar['high']
        low = bar['low']

        if signal_type == 'BUY_CALL':
            # Check if target was hit (high >= target)
            if high >= target:
                return ("WIN", i + 1)
            # Check if stop loss was hit (low <= stop_loss)
            if low <= stop_loss:
                return ("LOSE", i + 1)

        elif signal_type == 'BUY_PUT':
            # Check if target was hit (low <= target)
            if low <= target:
                return ("WIN", i + 1)
            # Check if stop loss was hit (high >= stop_loss)
            if high >= stop_loss:
                return ("LOSE", i + 1)

    # Neither target nor SL hit in available bars
    return ("INCONCLUSIVE", len(bars))


def main():
    db = DatabaseManager()
    db.connect()

    analyze_signal_performance(db)

    db.close()

if __name__ == "__main__":
    main()
