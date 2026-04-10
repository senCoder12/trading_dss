import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from zoneinfo import ZoneInfo
from src.database.db_manager import DatabaseManager
from src.backtest.strategy_runner import StrategyRunner, BacktestConfig

ist = ZoneInfo("Asia/Kolkata")

db = DatabaseManager(db_path=Path("data/db/trading.db"))
db.connect()

runner = StrategyRunner(db)
config = BacktestConfig(
    index_id="NIFTY50",
    start_date=datetime(2024, 7, 1, tzinfo=ist),
    end_date=datetime(2024, 12, 31, tzinfo=ist),
    timeframe="1d",
    mode="TECHNICAL_ONLY",
    show_progress=False
)

from src.backtest.metrics import MetricsCalculator

calc = MetricsCalculator()

# Use the result from Step 6.3 validation
result = runner.run(config)

metrics = calc.calculate_all(
    trade_history=result.trade_history,
    equity_curve=result.equity_curve,
    initial_capital=result.initial_capital
)

print(f"=== Performance Metrics ===")
print(f"Return: {metrics.total_return_pct:+.2f}% (₹{metrics.total_return_amount:+,.0f})")
print(f"Trades: {metrics.total_trades} | Win Rate: {metrics.win_rate:.1f}%")
print(f"Profit Factor: {metrics.profit_factor:.2f}")
print(f"Expected Value: ₹{metrics.expected_value_per_trade:+,.0f}/trade")
print(f"Max Drawdown: -{metrics.max_drawdown_pct:.1f}%")
print(f"Sharpe: {metrics.sharpe_ratio:.2f} | Sortino: {metrics.sortino_ratio:.2f}")
print(f"Grade: {metrics.strategy_grade}")
print(f"\n{metrics.assessment}")

db.close()
