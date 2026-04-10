from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from src.database.db_manager import DatabaseManager
from src.backtest.trade_simulator import SimulatorConfig, ClosedTrade
from src.backtest.strategy_runner import StrategyRunner, BacktestConfig, BacktestResult
from src.backtest.metrics import MetricsCalculator, BacktestMetrics

@dataclass
class WalkForwardConfig:
    index_id: str
    full_start_date: date
    full_end_date: date
    timeframe: str = "1d"
    
    # Window configuration
    train_window_days: int = 252           # Trading days for training. Default: 252 (1 year)
    test_window_days: int = 63             # Trading days for testing. Default: 63 (3 months)
    step_days: int = 63                    # How many days to step forward. Default: 63 (= test_window)
    
    # Strategy config (applied to all windows)
    simulator_config: SimulatorConfig = field(default_factory=SimulatorConfig)
    mode: str = "TECHNICAL_ONLY"
    min_confidence: str = "MEDIUM"
    
    # Robustness
    parameter_sensitivity: bool = False

@dataclass
class WindowResult:
    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    return_degradation_pct: float   # train_return - test_return
    win_rate_degradation: float     # train_wr - test_wr
    is_test_profitable: bool

@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    windows: list[WindowResult]
    
    total_windows: int
    profitable_test_windows: int
    test_profitability_rate: float
    
    avg_train_return: float
    avg_test_return: float
    avg_degradation: float
    
    avg_train_win_rate: float
    avg_test_win_rate: float
    
    avg_train_sharpe: float
    avg_test_sharpe: float
    
    max_test_drawdown: float
    
    combined_test_return: float
    combined_test_trades: int
    combined_test_win_rate: float
    
    overfitting_score: float
    overfitting_assessment: str
    
    is_robust: bool
    verdict: str

class WalkForwardValidator:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.strategy_runner = StrategyRunner(db)
        self.metrics_calculator = MetricsCalculator()
        
    def _get_trading_days(self, index_id: str, start: date, end: date) -> list[date]:
        """Fetch all valid trading dates from the database for the given index."""
        # For simplicity, we just fetch from the DB
        # This prevents weekends and holidays from skewing window sizes.
        query = """
            SELECT DISTINCT date(timestamp) as dt 
            FROM market_data 
            WHERE symbol = ? AND date(timestamp) >= ? AND date(timestamp) <= ?
            ORDER BY timestamp
        """
        rows = self.db.fetch_all(query, (index_id, start.isoformat(), end.isoformat()))
        if not rows:
            # Fallback to simple days if db is empty (for testing)
            res = []
            curr = start
            while curr <= end:
                if curr.weekday() < 5:
                    res.append(curr)
                curr += timedelta(days=1)
            return res
        return [datetime.strptime(row['dt'], "%Y-%m-%d").date() for row in rows]

    def _calculate_windows(self, config: WalkForwardConfig, trading_days: list[date]) -> list[dict]:
        windows = []
        if not trading_days:
            return windows
            
        current_idx = 0
        window_id = 1
        
        while True:
            # train end index
            train_end_idx = current_idx + config.train_window_days - 1
            if train_end_idx >= len(trading_days):
                break
                
            # test start and end index
            test_start_idx = train_end_idx + 1
            test_end_idx = test_start_idx + config.test_window_days - 1
            
            if test_end_idx >= len(trading_days):
                # We could run a partial window, but the prompt says 
                # "if test_end > full_end_date: break # Not enough data for another window"
                break
                
            windows.append({
                'train_start': trading_days[current_idx],
                'train_end': trading_days[train_end_idx],
                'test_start': trading_days[test_start_idx],
                'test_end': trading_days[test_end_idx],
                'window_id': window_id
            })
            
            # Step forward
            current_idx += config.step_days
            window_id += 1
            
        return windows

    def run_walk_forward(self, config: WalkForwardConfig) -> WalkForwardResult:
        trading_days = self._get_trading_days(config.index_id, config.full_start_date, config.full_end_date)
        windows = self._calculate_windows(config, trading_days)
        
        if len(windows) < 2:
            raise ValueError(f"Not enough data to calculate at least 2 windows. Got {len(windows)}")
            
        window_results = []
        combined_test_trades = []
        combined_test_return_amount = 0.0
        max_test_drawdown = 0.0
        
        for w in windows:
            # 2a. Run backtest on TRAIN period
            train_config = BacktestConfig(
                index_id=config.index_id,
                start_date=w['train_start'],
                end_date=w['train_end'],
                timeframe=config.timeframe,
                mode=config.mode,
                simulator_config=config.simulator_config,
                min_confidence=config.min_confidence
            )
            train_result = self.strategy_runner.run(train_config)
            train_metrics = self.metrics_calculator.calculate_all(
                train_result.trade_history,
                train_result.equity_curve,
                config.simulator_config.initial_capital
            )
            
            # 2b. Run backtest on TEST period
            test_config = BacktestConfig(
                index_id=config.index_id,
                start_date=w['test_start'],
                end_date=w['test_end'],
                timeframe=config.timeframe,
                mode=config.mode,
                simulator_config=config.simulator_config,
                min_confidence=config.min_confidence
            )
            test_result = self.strategy_runner.run(test_config)
            test_metrics = self.metrics_calculator.calculate_all(
                test_result.trade_history,
                test_result.equity_curve,
                config.simulator_config.initial_capital
            )
            
            # 2c. Record results
            ret_deg = train_metrics.total_return_pct - test_metrics.total_return_pct
            wr_deg = train_metrics.win_rate - test_metrics.win_rate
            is_prof = test_metrics.total_return_pct > 0
            
            w_res = WindowResult(
                window_id=w['window_id'],
                train_start=w['train_start'],
                train_end=w['train_end'],
                test_start=w['test_start'],
                test_end=w['test_end'],
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                return_degradation_pct=ret_deg,
                win_rate_degradation=wr_deg,
                is_test_profitable=is_prof
            )
            window_results.append(w_res)
            
            combined_test_trades.extend(test_result.trade_history)
            combined_test_return_amount += test_metrics.total_return_pct  # Simple sum for combined return pct
            
            if test_metrics.max_drawdown_pct > max_test_drawdown:
                max_test_drawdown = test_metrics.max_drawdown_pct
            
            # 2d. Parameter sensitivity (optional stub)
            if config.parameter_sensitivity:
                pass # Full optimization is Phase 7
                
            # Print progress
            print(f"Window {w['window_id']}/{len(windows)}: Train {w['train_start']} \u2192 {w['train_end']} | Test {w['test_start']} \u2192 {w['test_end']}")
            print(f"  Train: {train_metrics.total_trades} trades, {train_metrics.win_rate:.1f}% win rate, +{train_metrics.total_return_pct:.1f}%")
            print(f"  Test:  {test_metrics.total_trades} trades, {test_metrics.win_rate:.1f}% win rate, +{test_metrics.total_return_pct:.1f}%")
            indicator = '\u2190 Within acceptable range' if ret_deg < 10 else ''
            print(f"  Degradation: {ret_deg:.1f} percentage points {indicator}")
            print()
            
        # 3. Aggregate results
        total_windows = len(windows)
        profitable_test_windows = sum(1 for w in window_results if w.is_test_profitable)
        test_profitability_rate = profitable_test_windows / total_windows
        
        avg_train_ret = sum(w.train_metrics.total_return_pct for w in window_results) / total_windows
        avg_test_ret = sum(w.test_metrics.total_return_pct for w in window_results) / total_windows
        avg_deg = sum(w.return_degradation_pct for w in window_results) / total_windows
        
        avg_train_wr = sum(w.train_metrics.win_rate for w in window_results) / total_windows
        avg_test_wr = sum(w.test_metrics.win_rate for w in window_results) / total_windows
        
        avg_train_sharpe = sum(w.train_metrics.sharpe_ratio for w in window_results) / total_windows
        avg_test_sharpe = sum(w.test_metrics.sharpe_ratio for w in window_results) / total_windows
        
        combined_trades_count = len(combined_test_trades)
        combined_wins = sum(1 for t in combined_test_trades if t.outcome == 'WIN')
        combined_test_wr = (combined_wins / combined_trades_count * 100) if combined_trades_count > 0 else 0.0
        
        # Overfitting score calculation
        overfitting_score = 0.0
        
        if avg_deg > 20: overfitting_score += 0.3
        elif avg_deg > 10: overfitting_score += 0.2
        elif avg_deg > 5: overfitting_score += 0.1
        
        if test_profitability_rate < 0.25: overfitting_score += 0.3
        elif test_profitability_rate < 0.50: overfitting_score += 0.2
        elif test_profitability_rate < 0.75: overfitting_score += 0.1
        
        wr_deg = avg_train_wr - avg_test_wr
        if wr_deg > 15: overfitting_score += 0.2
        elif wr_deg > 10: overfitting_score += 0.15
        elif wr_deg > 5: overfitting_score += 0.1
        
        sharpe_deg = avg_train_sharpe - avg_test_sharpe
        if sharpe_deg > 1.0: overfitting_score += 0.2
        elif sharpe_deg > 0.5: overfitting_score += 0.1
        
        # Classify
        if overfitting_score < 0.25:
            overfitting_assessment = "LOW_RISK"
        elif overfitting_score < 0.5:
            overfitting_assessment = "MODERATE_RISK"
        elif overfitting_score < 0.75:
            overfitting_assessment = "HIGH_RISK"
        else:
            overfitting_assessment = "SEVERE"
            
        is_robust = overfitting_score < 0.5 and test_profitability_rate >= 0.5
        
        verdict = ""
        if is_robust:
            verdict = "\u2705 STRATEGY IS ROBUST ENOUGH FOR PAPER TRADING\n\n  The strategy shows consistent positive returns across test windows.\n  Expected live performance: +1-3% per quarter (lower than backtest suggests).\n  Proceed with caution \u2014 monitor for 1 month of paper trading before any live risk."
        else:
            verdict = "\u274c STRATEGY IS NOT ROBUST\n\n  The strategy exhibits significant overfitting or fails to remain profitable out-of-sample.\n  Do not trade live."

        return WalkForwardResult(
            config=config,
            windows=window_results,
            total_windows=total_windows,
            profitable_test_windows=profitable_test_windows,
            test_profitability_rate=test_profitability_rate,
            avg_train_return=avg_train_ret,
            avg_test_return=avg_test_ret,
            avg_degradation=avg_deg,
            avg_train_win_rate=avg_train_wr,
            avg_test_win_rate=avg_test_wr,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            max_test_drawdown=max_test_drawdown,
            combined_test_return=combined_test_return_amount,
            combined_test_trades=combined_trades_count,
            combined_test_win_rate=combined_test_wr,
            overfitting_score=overfitting_score,
            overfitting_assessment=overfitting_assessment,
            is_robust=is_robust,
            verdict=verdict
        )
