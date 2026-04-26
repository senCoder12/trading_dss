"""
Comprehensive signal performance report with statistics and breakdown.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import DatabaseManager

def generate_report(db):
    """Generate comprehensive signal performance report."""

    # Get historical signals
    signals = db.fetch_all("""
        SELECT id, index_id, generated_at, signal_type, entry_price,
               target_price, stop_loss, confidence_level, regime
        FROM trading_signals
        WHERE signal_type IN ('BUY_CALL', 'BUY_PUT')
          AND DATE(generated_at) < DATE('now')
        ORDER BY generated_at ASC
    """)

    if not signals:
        print("No historical signals to analyze.")
        return

    # Group signals by status
    winning = []
    losing = []
    inconclusive = []

    # Analyze each signal
    for sig in signals:
        index_id = sig['index_id']
        sig_time = sig['generated_at']
        entry = sig['entry_price']
        target = sig['target_price']
        stop_loss = sig['stop_loss']
        signal_type = sig['signal_type']
        confidence = sig['confidence_level']
        regime = sig['regime']

        # Get 20 bars after signal
        bars = db.fetch_all("""
            SELECT timestamp, open, high, low, close
            FROM price_data
            WHERE index_id = ? AND timeframe = '1d' AND timestamp > ?
            ORDER BY timestamp ASC
            LIMIT 20
        """, (index_id, sig_time))

        result = evaluate_signal(signal_type, entry, target, stop_loss, bars)

        sig_data = {
            'index': index_id,
            'type': signal_type,
            'entry': entry,
            'target': target,
            'sl': stop_loss,
            'confidence': confidence,
            'regime': regime,
            'date': sig_time,
            'bars_available': len(bars),
            'result': result
        }

        if result == "WIN":
            winning.append(sig_data)
        elif result == "LOSE":
            losing.append(sig_data)
        else:
            inconclusive.append(sig_data)

    total = len(winning) + len(losing) + len(inconclusive)

    print("\n" + "="*100)
    print("TRADING SIGNAL PERFORMANCE ANALYSIS REPORT")
    print("="*100)

    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total signals analyzed:  {total}")
    print(f"   Winning signals:         {len(winning)} ({len(winning)/total*100:.1f}%)")
    print(f"   Losing signals:          {len(losing)} ({len(losing)/total*100:.1f}%)")
    print(f"   Inconclusive:            {len(inconclusive)} ({len(inconclusive)/total*100:.1f}%)")

    if total > 0:
        success_ratio = (len(winning) / total) * 100
        print(f"\n🎯 SUCCESS RATIO: {success_ratio:.1f}%")

        if len(losing) > 0:
            print(f"📈 Win/Loss Ratio: {len(winning)}/{len(losing)} = {len(winning)/len(losing) if len(losing) > 0 else float('inf'):.2f}x")

    print(f"\n" + "─"*100)
    print("\n📈 BREAKDOWN BY SIGNAL TYPE")

    call_win = sum(1 for s in winning if s['type'] == 'BUY_CALL')
    call_lose = sum(1 for s in losing if s['type'] == 'BUY_CALL')
    put_win = sum(1 for s in winning if s['type'] == 'BUY_PUT')
    put_lose = sum(1 for s in losing if s['type'] == 'BUY_PUT')

    total_calls = call_win + call_lose
    total_puts = put_win + put_lose

    print(f"   BUY_CALL:  {call_win}/{total_calls if total_calls > 0 else 1} ({call_win/total_calls*100 if total_calls > 0 else 0:.1f}% success)")
    print(f"   BUY_PUT:   {put_win}/{total_puts if total_puts > 0 else 1} ({put_win/total_puts*100 if total_puts > 0 else 0:.1f}% success)")

    print(f"\n" + "─"*100)
    print("\n📊 BREAKDOWN BY REGIME")

    regimes = defaultdict(lambda: {"win": 0, "lose": 0, "inconc": 0})
    for s in winning:
        regimes[s['regime']]["win"] += 1
    for s in losing:
        regimes[s['regime']]["lose"] += 1
    for s in inconclusive:
        regimes[s['regime']]["inconc"] += 1

    for regime in sorted(regimes.keys()):
        data = regimes[regime]
        total_regime = data["win"] + data["lose"] + data["inconc"]
        win_pct = (data["win"] / total_regime * 100) if total_regime > 0 else 0
        print(f"   {regime:<20}: {data['win']}/{total_regime} ({win_pct:.1f}% success)")

    print(f"\n" + "─"*100)
    print("\n💰 BREAKDOWN BY CONFIDENCE LEVEL")

    confidence_levels = defaultdict(lambda: {"win": 0, "lose": 0, "inconc": 0})
    for s in winning:
        confidence_levels[s['confidence']]["win"] += 1
    for s in losing:
        confidence_levels[s['confidence']]["lose"] += 1
    for s in inconclusive:
        confidence_levels[s['confidence']]["inconc"] += 1

    for conf in sorted(confidence_levels.keys(), reverse=True):
        data = confidence_levels[conf]
        total_conf = data["win"] + data["lose"] + data["inconc"]
        win_pct = (data["win"] / total_conf * 100) if total_conf > 0 else 0
        print(f"   Confidence {conf}: {data['win']}/{total_conf} ({win_pct:.1f}% success)")

    print(f"\n" + "─"*100)
    print("\n🏆 TOP PERFORMING INDICES")

    index_stats = defaultdict(lambda: {"win": 0, "lose": 0})
    for s in winning:
        index_stats[s['index']]["win"] += 1
    for s in losing:
        index_stats[s['index']]["lose"] += 1

    ranked = sorted(index_stats.items(), key=lambda x: x[1]["win"]/(x[1]["win"]+x[1]["lose"])*100 if (x[1]["win"]+x[1]["lose"]) > 0 else 0, reverse=True)

    for idx, data in ranked[:10]:
        total_idx = data["win"] + data["lose"]
        if total_idx > 0:
            win_pct = data["win"] / total_idx * 100
            print(f"   {idx:<20}: {data['win']}/{total_idx} ({win_pct:.1f}%)")

    print(f"\n" + "="*100 + "\n")


def evaluate_signal(signal_type: str, entry: float, target: float, stop_loss: float, bars: list) -> str:
    """Evaluate if signal hit target or stop loss first."""
    for bar in bars:
        high = bar['high']
        low = bar['low']

        if signal_type == 'BUY_CALL':
            if high >= target:
                return "WIN"
            if low <= stop_loss:
                return "LOSE"
        elif signal_type == 'BUY_PUT':
            if low <= target:
                return "WIN"
            if high >= stop_loss:
                return "LOSE"

    return "INCONCLUSIVE"


if __name__ == "__main__":
    db = DatabaseManager()
    db.connect()
    generate_report(db)
    db.close()
