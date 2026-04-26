"""
Signal performance analysis with 5% return threshold as success criteria.
Success = 5% profit before hitting stop loss (not target price).
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import DatabaseManager

def generate_report(db):
    """Generate signal performance report with 5% return threshold."""

    # Get all historical signals
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

    winning = []
    losing = []
    inconclusive = []

    for sig in signals:
        index_id = sig['index_id']
        sig_time = sig['generated_at']
        entry = sig['entry_price']
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

        result = evaluate_signal_5pct(signal_type, entry, stop_loss, bars)

        sig_data = {
            'index': index_id,
            'type': signal_type,
            'entry': entry,
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

    print("\n" + "="*110)
    print("TRADING SIGNAL PERFORMANCE ANALYSIS — 5% RETURN THRESHOLD")
    print("="*110)
    print("✓ Success = Signal achieves 5% profit before hitting stop loss")
    print("✗ Failure = Stop loss hit before 5% profit reached")
    print("⏳ Inconclusive = Neither target nor SL hit within 20 bars")
    print("="*110)

    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total signals analyzed:  {total}")
    print(f"   Winning signals (5%+):   {len(winning)} ({len(winning)/total*100:.1f}%)")
    print(f"   Losing signals (SL hit): {len(losing)} ({len(losing)/total*100:.1f}%)")
    print(f"   Inconclusive:            {len(inconclusive)} ({len(inconclusive)/total*100:.1f}%)")

    if total > 0:
        success_ratio = (len(winning) / total) * 100
        print(f"\n🎯 SUCCESS RATIO: {success_ratio:.2f}%")

        if len(losing) > 0:
            ratio = len(winning)/len(losing) if len(losing) > 0 else float('inf')
            print(f"📈 Win/Loss Ratio: {len(winning)}/{len(losing)} = {ratio:.2f}x")
            if ratio < 1:
                print(f"   ⚠️  System losing 1:{1/ratio:.2f} - needs tuning")
            else:
                print(f"   ✓ System profitable with {ratio:.2f}x win/loss ratio")

    print(f"\n" + "─"*110)
    print("\n📈 BREAKDOWN BY SIGNAL TYPE")

    call_win = sum(1 for s in winning if s['type'] == 'BUY_CALL')
    call_lose = sum(1 for s in losing if s['type'] == 'BUY_CALL')
    put_win = sum(1 for s in winning if s['type'] == 'BUY_PUT')
    put_lose = sum(1 for s in losing if s['type'] == 'BUY_PUT')

    total_calls = call_win + call_lose
    total_puts = put_win + put_lose

    if total_calls > 0:
        print(f"   BUY_CALL:  {call_win}/{total_calls} ({call_win/total_calls*100:.1f}% success)")
    if total_puts > 0:
        print(f"   BUY_PUT:   {put_win}/{total_puts} ({put_win/total_puts*100:.1f}% success)")

    if call_win + put_win > 0:
        better = "PUT" if put_win/total_puts > call_win/total_calls else "CALL"
        print(f"   → BUY_{better} performs better")

    print(f"\n" + "─"*110)
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

    print(f"\n" + "─"*110)
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
        print(f"   {conf:<10}: {data['win']}/{total_conf} ({win_pct:.1f}% success)")

    print(f"\n" + "─"*110)
    print("\n🏆 TOP PERFORMING INDICES (min 50 signals)")

    index_stats = defaultdict(lambda: {"win": 0, "lose": 0})
    for s in winning:
        index_stats[s['index']]["win"] += 1
    for s in losing:
        index_stats[s['index']]["lose"] += 1

    ranked = sorted(
        [(idx, data) for idx, data in index_stats.items() if (data["win"]+data["lose"]) >= 50],
        key=lambda x: x[1]["win"]/(x[1]["win"]+x[1]["lose"])*100 if (x[1]["win"]+x[1]["lose"]) > 0 else 0,
        reverse=True
    )

    for idx, data in ranked[:15]:
        total_idx = data["win"] + data["lose"]
        if total_idx > 0:
            win_pct = data["win"] / total_idx * 100
            status = "✓" if win_pct >= 50 else "✗"
            print(f"   {status} {idx:<20}: {data['win']}/{total_idx} ({win_pct:.1f}%)")

    print(f"\n" + "="*110)
    print("\n💡 KEY INSIGHTS")

    success = len(winning) / total * 100
    if success >= 50:
        print(f"   ✓ System is PROFITABLE with {success:.1f}% success ratio")
    elif success >= 40:
        print(f"   ⚠️  System is MARGINAL with {success:.1f}% success - needs optimization")
    else:
        print(f"   ✗ System is UNPROFITABLE with {success:.1f}% success - requires rework")

    # Recommendation
    best_type = "BUY_PUT" if (total_puts > 0 and put_win/total_puts > call_win/total_calls) else "BUY_CALL"
    best_regime = max(regimes.items(), key=lambda x: x[1]["win"]/(x[1]["win"]+x[1]["lose"]) if (x[1]["win"]+x[1]["lose"]) > 0 else 0)

    print(f"\n   Recommendation:")
    print(f"   → Focus on {best_type} signals")
    print(f"   → Best performance in {best_regime[0]} regime")
    print(f"   → Filter by MEDIUM/HIGH confidence only")

    print(f"\n" + "="*110 + "\n")


def evaluate_signal_5pct(signal_type: str, entry: float, stop_loss: float, bars: list) -> str:
    """
    Evaluate if signal reaches 5% return before hitting stop loss.

    For BUY_CALL: 5% profit = entry * 1.05
    For BUY_PUT: 5% profit = entry * 0.95
    """

    if signal_type == 'BUY_CALL':
        profit_target = entry * 1.05
        for bar in bars:
            if bar['high'] >= profit_target:
                return "WIN"
            if bar['low'] <= stop_loss:
                return "LOSE"

    elif signal_type == 'BUY_PUT':
        profit_target = entry * 0.95
        for bar in bars:
            if bar['low'] <= profit_target:
                return "WIN"
            if bar['high'] >= stop_loss:
                return "LOSE"

    return "INCONCLUSIVE"


if __name__ == "__main__":
    db = DatabaseManager()
    db.connect()
    generate_report(db)
    db.close()
