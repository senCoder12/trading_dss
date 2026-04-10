import os
from datetime import datetime
from src.backtest.strategy_runner import BacktestResult
from src.backtest.metrics import BacktestMetrics
from src.backtest.walk_forward import WalkForwardResult
from src.backtest.trade_simulator import ClosedTrade

class ReportGenerator:
    @staticmethod
    def generate_backtest_report(result: BacktestResult, metrics: BacktestMetrics) -> str:
        cfg = result.config
        scfg = cfg.simulator_config
        
        capital = scfg.initial_capital
        risk_per_trade = scfg.max_risk_per_trade_pct * 100
        risk_per_day = scfg.max_risk_per_day_pct * 100
        
        # Grading extraction
        grade_desc = {
            "A": "A (Excellent)",
            "B": "B (Good)",
            "C": "C (Average)",
            "D": "D (Below Average)",
            "F": "F (Failing)"
        }.get(metrics.strategy_grade, f"{metrics.strategy_grade} (Unknown)")
        
        sign = "+" if metrics.total_return_amount > 0 else ""
        
        report = f"""╔══════════════════════════════════════════════════════╗
║         BACKTEST REPORT — {cfg.index_id:<26} ║
║         Period: {cfg.start_date!s} → {cfg.end_date!s:<15} ║
║         Mode: {cfg.mode:<14} | Timeframe: {cfg.timeframe:<10} ║
╠══════════════════════════════════════════════════════╣

CONFIGURATION:
  Capital: ₹{capital:,.0f} | Risk/Trade: {risk_per_trade:.1f}% | Risk/Day: {risk_per_day:.1f}%
  Max Positions: {scfg.max_open_positions} | Min Confidence: {cfg.min_confidence}
  Slippage: {scfg.slippage_points} pts | Costs: Realistic (STT+brokerage+GST)

RETURNS:
  Total Return: {sign}{metrics.total_return_pct:.1f}% (₹{metrics.total_return_amount:,.0f})
  Annualized: ~{metrics.annualized_return_pct:.1f}%
  Best Month: +{metrics.best_month_pct:.1f}%
  Worst Month: {metrics.worst_month_pct:.1f}%
  Monthly Win Rate: {metrics.monthly_win_rate:.1f}% ({metrics.positive_months}/{len(metrics.monthly_returns)} months profitable)

TRADES:
  Total: {metrics.total_trades} | Wins: {metrics.winning_trades} | Losses: {metrics.losing_trades}
  Win Rate: {metrics.win_rate:.1f}%
  Avg Win: ₹{metrics.avg_win_amount:,.0f} | Avg Loss: ₹{metrics.avg_loss_amount:,.0f}
  Largest Win: ₹{metrics.largest_win:,.0f} | Largest Loss: ₹{metrics.largest_loss:,.0f}
  Profit Factor: {metrics.profit_factor:.2f}
  Expected Value: {sign}₹{metrics.expected_value_per_trade:,.0f}/trade

RISK:
  Max Drawdown: -{metrics.max_drawdown_pct:.1f}% (₹{metrics.max_drawdown_amount:,.0f})
  Max DD Duration: {metrics.max_drawdown_duration_bars} bars
  Max Consecutive Losses: {metrics.max_consecutive_losses}
  Sharpe Ratio: {metrics.sharpe_ratio:.2f}
  Sortino Ratio: {metrics.sortino_ratio:.2f}
  Calmar Ratio: {metrics.calmar_ratio:.2f}

BY CONFIDENCE:
  HIGH:   {metrics.high_confidence_trades:<2} trades | {metrics.high_confidence_win_rate:>4.1f}% WR | Avg {metrics.high_confidence_avg_pnl:>+6.0f} | Total {metrics.high_confidence_trades * metrics.high_confidence_avg_pnl:>+7.0f}
  MEDIUM: {metrics.medium_confidence_trades:<2} trades | {metrics.medium_confidence_win_rate:>4.1f}% WR | Avg {metrics.medium_confidence_avg_pnl:>+6.0f} | Total {metrics.medium_confidence_trades * metrics.medium_confidence_avg_pnl:>+7.0f}
  LOW:    {metrics.low_confidence_trades:<2} trades | {metrics.low_confidence_win_rate:>4.1f}% WR | Avg {metrics.low_confidence_avg_pnl:>+6.0f} | Total {metrics.low_confidence_trades * metrics.low_confidence_avg_pnl:>+7.0f}

BY SIGNAL TYPE:
  CALL: {metrics.call_trades:<2} trades | {metrics.call_win_rate:>4.1f}% WR | +₹{metrics.call_total_pnl:,.0f}
  PUT:  {metrics.put_trades:<2} trades | {metrics.put_win_rate:>4.1f}% WR | +₹{metrics.put_total_pnl:,.0f}

BY EXIT REASON:
  Target Hit:    {metrics.target_hit_count:<2} ({metrics.target_hit_pct:>4.1f}%) | Avg +₹{(metrics.total_return_amount / max(1, metrics.target_hit_count)) if metrics.target_hit_count else 0:,.0f}
  Stop Loss:     {metrics.sl_hit_count:<2} ({metrics.sl_hit_pct:>4.1f}%) | Avg -₹{(metrics.total_return_amount / max(1, metrics.sl_hit_count)) if metrics.sl_hit_count else 0:,.0f}
  Trailing SL:   {metrics.trailing_sl_count:<2} ({metrics.trailing_sl_pct:>4.1f}%) | Avg +₹{(metrics.total_return_amount / max(1, metrics.trailing_sl_count)) if metrics.trailing_sl_count else 0:,.0f}
  Forced EOD:    {metrics.forced_eod_count:<2} ({(metrics.forced_eod_count/metrics.total_trades*100) if metrics.total_trades else 0:>4.1f}%) | Avg ₹{metrics.forced_eod_avg_pnl:,.0f}

GRADE: {grade_desc}

RECOMMENDATIONS:"""
        
        recommendations = []
        if metrics.has_edge:
            recommendations.append("• Positive edge detected — proceed to walk-forward validation")
        if metrics.low_confidence_avg_pnl < 0:
            recommendations.append("• Consider removing LOW confidence signals (negative EV)")
        if metrics.call_total_pnl > metrics.put_total_pnl and metrics.put_total_pnl < 0:
            recommendations.append("• CALL signals stronger than PUT — review PUT entry criteria")
        elif metrics.put_total_pnl > metrics.call_total_pnl and metrics.call_total_pnl < 0:
            recommendations.append("• PUT signals stronger than CALL — review CALL entry criteria")
        if metrics.forced_eod_avg_pnl < 0:
            recommendations.append("• Forced EOD exits are slightly negative — review timing rules")
        
        if not recommendations:
            recommendations.append("• No specific warnings at this time.")
            
        report += "\n" + "\n".join(recommendations)
        report += "\n\n╚══════════════════════════════════════════════════════╝"
        return report

    @staticmethod
    def generate_walk_forward_report(wf_result: WalkForwardResult) -> str:
        cfg = wf_result.config
        
        report = f"""╔══════════════════════════════════════════════════════╗
║     WALK-FORWARD VALIDATION REPORT — {cfg.index_id:<16} ║
║     Full Period: {cfg.full_start_date!s} → {cfg.full_end_date!s:<16} ║
╠══════════════════════════════════════════════════════╣

WINDOWS:
┌──────┬────────────────────┬─────────────────────┬────────┬────────┐
│  #   │ Train Period       │ Test Period         │ Train% │ Test%  │
├──────┼────────────────────┼─────────────────────┼────────┼────────┤"""

        for w in wf_result.windows:
            t_start = w.train_start.strftime("%Y-%m")
            t_end = w.train_end.strftime("%Y-%m")
            te_start = w.test_start.strftime("%Y-%m")
            te_end = w.test_end.strftime("%Y-%m")
            
            tr_pct = f"{w.train_metrics.total_return_pct:>+5.1f}%"
            te_pct = f"{w.test_metrics.total_return_pct:>+5.1f}%"
            
            row = f"\n│ {w.window_id:^4} │ {t_start} → {t_end:<7} │ {te_start} → {te_end:<8} │ {tr_pct:<6} │ {te_pct:<6} │"
            report += row
            
        report += "\n└──────┴────────────────────┴─────────────────────┴────────┴────────┘\n"

        report += f"""
AGGREGATE:
  Avg Train Return: {wf_result.avg_train_return:>+5.2f}%  | Avg Test Return: {wf_result.avg_test_return:>+5.2f}%
  Avg Degradation: {wf_result.avg_degradation:.1f} percentage points
  Test Profitability: {wf_result.test_profitability_rate*100:.0f}% ({wf_result.profitable_test_windows}/{wf_result.total_windows} windows profitable)
  Combined Test Return: {wf_result.combined_test_return:>+5.1f}%
  Combined Test Win Rate: {wf_result.combined_test_win_rate:.1f}%
  Worst Test Drawdown: -{wf_result.max_test_drawdown:.1f}%

OVERFITTING ASSESSMENT:
  Score: {wf_result.overfitting_score:.2f} / 1.0 — {wf_result.overfitting_assessment}
  
  Return degradation is {wf_result.avg_degradation:.1f} points. Test profitability is {wf_result.test_profitability_rate*100:.0f}%.
  
VERDICT:\n{wf_result.verdict}

╚══════════════════════════════════════════════════════╝"""

        return report

    @staticmethod
    def save_report(report: str, filepath: str = None, index_id: str = "BACKTEST") -> str:
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"data/reports/backtest_{index_id}_{timestamp}.txt"
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(report)
        return filepath

    @staticmethod
    def generate_trade_log(trade_history: list[ClosedTrade]) -> str:
        log = """#  | Date       | Type     | Entry   | Exit    | SL      | TGT     | PnL     | Conf  | Exit Reason    | Duration
---|------------|----------|---------|---------|---------|---------|---------|-------|----------------|----------
"""
        for i, t in enumerate(trade_history, 1):
            date_str = t.entry_timestamp.strftime("%Y-%m-%d")
            log += f"{i:<3}| {date_str:<11}| {t.trade_type:<9}| {t.entry_price:<8.0f}| {t.exit_price:<8.0f}| {t.stop_loss:<8.0f}| {t.target:<8.0f}| {t.net_pnl:>+8.0f}| {t.confidence_level[:3]:<6}| {t.exit_reason:<15}| {t.duration_bars} bars\n"
        return log
