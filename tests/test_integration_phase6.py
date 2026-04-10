import pytest
import os
import time
from datetime import date, timedelta
import subprocess

from src.database.db_manager import DatabaseManager
from src.backtest.strategy_runner import StrategyRunner, BacktestConfig
from src.backtest.metrics import MetricsCalculator
from src.backtest.trade_simulator import SimulatorConfig
from src.backtest.walk_forward import WalkForwardValidator, WalkForwardConfig
from src.backtest.report_generator import ReportGenerator

# Make sure tests ONLY run if database actually has data for testing. 
# We can mock, but integration tests usually work on test db or mock db.
# We'll use a fixture to skip if no real db, or seed it. For now, assume test data exists.

@pytest.fixture(scope="module")
def db_manager():
    # Setup test DB or connection
    db = DatabaseManager()
    db.connect()
    
    # Check if table exists
    try:
        if not db.table_exists("market_data"):
            pytest.skip("No market_data table in DB for integration tests.")
        res = db.fetch_one("SELECT COUNT(*) as c FROM market_data WHERE symbol = 'NIFTY50'")
        if not res or res['c'] == 0:
            pytest.skip("No NIFTY50 data in DB for integration tests.")
    except Exception as e:
        pytest.skip(f"DB not ready: {e}")
        
    yield db
    db.close()

def test_full_backtest_6_months(db_manager):
    runner = StrategyRunner(db_manager)
    start = date(2023, 1, 1)
    end = date(2023, 7, 1)
    
    config = BacktestConfig(
        index_id="NIFTY50",
        start_date=start,
        end_date=end,
        timeframe="1d",
        mode="TECHNICAL_ONLY",
        simulator_config=SimulatorConfig(initial_capital=100000, max_open_positions=3)
    )
    
    t0 = time.time()
    result = runner.run(config)
    dt = time.time() - t0
    
    assert dt < 60, f"Backtest took {dt}s, expected < 60s"
    # Result has trades
    assert isinstance(result.trade_history, list)
    
    calc = MetricsCalculator()
    metrics = calc.calculate_all(result.trade_history, result.equity_curve, 100000)
    
    assert metrics is not None
    assert metrics.total_trades == len(result.trade_history)
    
    report = ReportGenerator.generate_backtest_report(result, metrics)
    assert "BACKTEST REPORT \u2014 NIFTY50" in report
    
    # Save report
    filepath = ReportGenerator.save_report(report, "data/reports/test_backtest_report.txt")
    assert os.path.exists(filepath)
    os.remove(filepath)

def test_walk_forward_integration(db_manager):
    validator = WalkForwardValidator(db_manager)
    
    start = date(2021, 1, 1)
    end = date(2023, 1, 1) # 2 years
    
    # Check if we have 2 years of data
    days = validator._get_trading_days("NIFTY50", start, end)
    if len(days) < 400:
        pytest.skip("Not enough data to run full WF integration test.")
        
    config = WalkForwardConfig(
        index_id="NIFTY50",
        full_start_date=start,
        full_end_date=end,
        train_window_days=252, # 1 yr
        test_window_days=63,   # 3m
        step_days=63
    )
    
    result = validator.run_walk_forward(config)
    
    # 2 years = ~504 trading days. Train = 252. Left = 252. Step = 63. Expected windows = 4
    assert result.total_windows == 4
    assert len(result.windows) == 4
    
    # Overfitting assessment
    assert result.overfitting_assessment in ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK", "SEVERE"]
    
    report = ReportGenerator.generate_walk_forward_report(result)
    assert "WALK-FORWARD VALIDATION REPORT" in report

def test_cli_script_no_crash():
    import sys
    # Test simple help command to ensure imports and script are okay
    res = subprocess.run([sys.executable, "scripts/run_backtest.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Trading DSS \u2014 Backtesting Engine v1.0" in res.stdout
