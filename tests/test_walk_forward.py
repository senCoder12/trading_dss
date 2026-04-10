import pytest
from datetime import date, timedelta
from src.database.db_manager import DatabaseManager
from src.backtest.walk_forward import WalkForwardValidator, WalkForwardConfig, WalkForwardResult
from src.backtest.trade_simulator import SimulatorConfig

class DummyMetrics:
    def __init__(self, ret=5.0, wr=55.0, sharpe=1.0):
        self.total_return_pct = ret
        self.win_rate = wr
        self.sharpe_ratio = sharpe
        self.total_return_amount = ret * 1000
        self.max_drawdown_pct = 5.0
        self.positive_months = 3
        self.monthly_returns = [1,2,3]
        self.total_trades = 10
        self.winning_trades = 5
        self.losing_trades = 5
        self.profit_factor = 1.5

from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_db_manager():
    return MagicMock(spec=DatabaseManager)

def test_window_calculation(mock_db_manager):
    validator = WalkForwardValidator(mock_db_manager)
    start_date = date(2022, 1, 1)
    # Generate 2 years of trading days (using simple daily increment for mock)
    trading_days = [start_date + timedelta(days=i) for i in range(504)] # approx 2 years 252*2
    
    config = WalkForwardConfig(
        index_id="NIFTY50",
        full_start_date=start_date,
        full_end_date=start_date + timedelta(days=504),
        train_window_days=252,
        test_window_days=63,
        step_days=63
    )
    
    windows = validator._calculate_windows(config, trading_days)
    # math: 504 - 252 (train) = 252 days left. Step size 63. 252/63 = 4 windows with no partials.
    assert len(windows) == 4
    
    # window 1
    assert windows[0]['window_id'] == 1
    # 0 to 251 (252 days)
    assert windows[0]['train_end'] == trading_days[251]
    # 252 to 314
    assert windows[0]['test_start'] == trading_days[252]
    assert windows[0]['test_end'] == trading_days[314]

def test_insufficient_data(mock_db_manager):
    validator = WalkForwardValidator(mock_db_manager)
    start_date = date(2022, 1, 1)
    trading_days = [start_date + timedelta(days=i) for i in range(126)] # 6 months
    
    config = WalkForwardConfig(
        index_id="NIFTY50",
        full_start_date=trading_days[0],
        full_end_date=trading_days[-1],
        train_window_days=252,
        test_window_days=63,
        step_days=63
    )
    
    windows = validator._calculate_windows(config, trading_days)
    assert len(windows) == 0

def test_overfitting_score(mock_db_manager):
    validator = WalkForwardValidator(mock_db_manager)
    start_date = date(2022, 1, 1)
    trading_days = [start_date + timedelta(days=i) for i in range(500)]
    
    config = WalkForwardConfig(
        index_id="NIFTY50",
        full_start_date=trading_days[0],
        full_end_date=trading_days[-1],
        train_window_days=100,
        test_window_days=50,
        step_days=50
    )
    
    validator._get_trading_days = MagicMock(return_value=trading_days)
    
    # Mocking StrategyRunner and MetricsCalculator
    validator.strategy_runner = MagicMock()
    validator.strategy_runner.run.return_value = MagicMock(trades=[], equity_curve=[])
    
    validator.metrics_calculator = MagicMock()
    
    test_metrics_1 = DummyMetrics(ret=-1.0, wr=40, sharpe=0.2)
    test_metrics_2 = DummyMetrics(ret=-2.0, wr=35, sharpe=0.1)
    train_metrics = DummyMetrics(ret=10.0, wr=60, sharpe=1.5)
    
    # sequence of returns for [train, test, train, test...]
    validator.metrics_calculator.calculate_all.side_effect = [
        train_metrics, test_metrics_1,
        train_metrics, test_metrics_2,
        train_metrics, test_metrics_1,
        train_metrics, test_metrics_2,
        train_metrics, test_metrics_1,
        train_metrics, test_metrics_2,
        train_metrics, test_metrics_1,
        train_metrics, test_metrics_2,
        train_metrics, test_metrics_1,
        train_metrics, test_metrics_2,
    ][:30] # up to 15 windows
    
    result = validator.run_walk_forward(config)
    
    # Let's see. Degradation: 10 - (-1.5) = 11.5% avg. Score += 0.2
    # Profitability rate: 0. Score += 0.3
    # WR degradation: 60 - ~37.5 = ~22.5%. Score += 0.2
    # Sharpe degradation: 1.5 - ~0.15 = ~1.35. Score += 0.2
    # Total = ~0.9
    
    assert result.total_windows == 8
    assert result.overfitting_assessment in ["SEVERE", "HIGH_RISK"]
    assert result.is_robust == False
