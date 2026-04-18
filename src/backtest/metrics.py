from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import math
import statistics
from collections import defaultdict

from src.backtest.trade_simulator import ClosedTrade, EquityPoint

@dataclass
class BacktestMetrics:
    # === RETURN METRICS ===
    total_return_pct: float
    total_return_amount: float
    annualized_return_pct: float
    monthly_returns: list[dict]
    best_month_pct: float
    worst_month_pct: float
    positive_months: int
    negative_months: int
    monthly_win_rate: float
    
    # === TRADE METRICS ===
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    
    avg_win_amount: float
    avg_loss_amount: float
    avg_win_pct: float
    avg_loss_pct: float
    
    largest_win: float
    largest_loss: float
    largest_win_pct: float
    largest_loss_pct: float
    
    avg_trade_duration_bars: float
    avg_winning_trade_duration: float
    avg_losing_trade_duration: float
    
    profit_factor: float
    payoff_ratio: float
    expected_value_per_trade: float
    expected_value_pct: float
    
    # === RISK METRICS ===
    max_drawdown_pct: float
    max_drawdown_amount: float
    max_drawdown_duration_bars: int
    max_drawdown_start: Optional[datetime]
    max_drawdown_end: Optional[datetime]
    avg_drawdown_pct: float
    
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int
    
    max_recovery_time_bars: int
    
    # === RISK-ADJUSTED RETURNS ===
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # === CONFIDENCE LEVEL ANALYSIS ===
    high_confidence_trades: int
    high_confidence_win_rate: float
    high_confidence_avg_pnl: float
    
    medium_confidence_trades: int
    medium_confidence_win_rate: float
    medium_confidence_avg_pnl: float
    
    low_confidence_trades: int
    low_confidence_win_rate: float
    low_confidence_avg_pnl: float
    
    # === SIGNAL TYPE ANALYSIS ===
    call_trades: int
    call_win_rate: float
    call_total_pnl: float
    
    put_trades: int
    put_win_rate: float
    put_total_pnl: float
    
    # === EXIT ANALYSIS ===
    target_hit_count: int
    target_hit_pct: float
    sl_hit_count: int
    sl_hit_pct: float
    trailing_sl_count: int
    trailing_sl_pct: float
    forced_eod_count: int
    forced_eod_avg_pnl: float
    
    # === REGIME ANALYSIS ===
    trades_by_regime: dict[str, dict]
    best_regime: str
    worst_regime: str
    
    # === TIMING ANALYSIS ===
    trades_by_day_of_week: dict[str, dict]
    trades_by_hour: dict[int, dict]
    best_day_of_week: str
    worst_day_of_week: str
    
    # === OVERALL ASSESSMENT ===
    is_profitable: bool
    has_edge: bool
    strategy_grade: str
    assessment: str

    # === CONFIG VERSIONING ===
    config_hash: Optional[str] = None

    # === BENCHMARK COMPARISON ===
    benchmark_return_pct: Optional[float] = None     # Buy-and-hold return for same period
    benchmark_index: Optional[str] = None            # Which index used as benchmark
    alpha: Optional[float] = None                    # strategy_return - benchmark_return
    alpha_annualized: Optional[float] = None
    information_ratio: Optional[float] = None        # alpha / tracking_error (annualised)
    beats_benchmark: Optional[bool] = None
    benchmark_comparison_note: Optional[str] = None  # Human-readable verdict

@dataclass
class ComparisonReport:
    headers: list[str]
    metrics_table: list[dict]
    best_overall: str
    ranking: list[tuple[str, float]]

class MetricsCalculator:
    @staticmethod
    def calculate_all(
        trade_history: list[ClosedTrade],
        equity_curve: list[EquityPoint],
        initial_capital: float,
        risk_free_rate: float = 0.065,
        benchmark_prices: Optional[Any] = None,  # pandas DataFrame with 'close' column
        benchmark_name: str = "NIFTY50",
    ) -> BacktestMetrics:
        
        # Guard clause: No trades
        if not trade_history:
            return MetricsCalculator._zero_metrics(initial_capital)
        
        final_capital = equity_curve[-1].capital if equity_curve else initial_capital
        total_return_amount = final_capital - initial_capital
        total_return_pct = (total_return_amount / initial_capital) * 100 if initial_capital > 0 else 0.0
        
        # Daily Returns & Sharpe/Sortino
        daily_returns = []
        if len(equity_curve) > 1:
            for i in range(1, len(equity_curve)):
                prev_cap = equity_curve[i-1].capital
                curr_cap = equity_curve[i].capital
                if prev_cap > 0:
                    daily_returns.append((curr_cap / prev_cap) - 1.0)
                else:
                    daily_returns.append(0.0)
                    
        annualized_return_pct = 0.0
        sharpe = 0.0
        sortino = 0.0
        
        if daily_returns:
            avg_daily_return = sum(daily_returns) / len(daily_returns)
            # Use population std dev or sample std dev? Let's use statistics.pstdev for population if length > 0
            if len(daily_returns) >= 2:
                std_daily_return = statistics.stdev(daily_returns)
            else:
                std_daily_return = 0.0
                
            annualized_return = avg_daily_return * 252
            annualized_return_pct = annualized_return * 100
            
            if std_daily_return > 0:
                annualized_std = std_daily_return * math.sqrt(252)
                sharpe = (annualized_return - risk_free_rate) / annualized_std
            
            downside_returns = [r for r in daily_returns if r < 0]
            if not downside_returns:
                sortino = 999.0
            else:
                # Semi-deviation: use all N observations in denominator (standard Sortino formula)
                downside_variance = sum(r**2 for r in downside_returns) / len(daily_returns)
                downside_std = math.sqrt(downside_variance) * math.sqrt(252)
                if downside_std > 0:
                    sortino = (annualized_return - risk_free_rate) / downside_std
                else:
                    sortino = 999.0
                    
        # Drawdowns
        max_drawdown_amount = 0.0
        max_drawdown_pct = 0.0
        max_drawdown_start = None
        max_drawdown_end = None
        max_drawdown_duration_bars = 0
        max_recovery_time_bars = 0
        
        peak = initial_capital
        drawdowns_pct = []
        
        current_dd_start = None
        current_dd_duration = 0
        
        if equity_curve:
            peak = equity_curve[0].capital
            for point in equity_curve:
                if point.capital >= peak:
                    # New peak reached, previous drawdown ends
                    if current_dd_start is not None:
                        max_recovery_time_bars = max(max_recovery_time_bars, current_dd_duration)
                    peak = point.capital
                    current_dd_start = None
                    current_dd_duration = 0
                else:
                    dd_amt = peak - point.capital
                    dd_pct = dd_amt / peak
                    drawdowns_pct.append(dd_pct)
                    
                    if current_dd_start is None:
                        current_dd_start = point.timestamp
                    current_dd_duration += 1
                    
                    if dd_pct > (max_drawdown_pct / 100):
                        max_drawdown_pct = dd_pct * 100
                        max_drawdown_amount = dd_amt
                        max_drawdown_start = current_dd_start
                        max_drawdown_duration_bars = current_dd_duration
                        # max_drawdown_end would be when it recovers, but till then it's None. We can set it to point.timestamp temporarily.
                        max_drawdown_end = point.timestamp
            
            # If we end in a drawdown
            if current_dd_start is not None:
                max_recovery_time_bars = max(max_recovery_time_bars, current_dd_duration)
                
        avg_drawdown_pct = (sum(drawdowns_pct) / len(drawdowns_pct) * 100) if drawdowns_pct else 0.0
        
        calmar_ratio = 0.0
        if max_drawdown_pct > 0:
            calmar_ratio = (annualized_return_pct / 100) / (max_drawdown_pct / 100)
            
        # Monthly Returns
        month_groups = defaultdict(list)
        if equity_curve:
            for pt in equity_curve:
                m_key = pt.timestamp.strftime("%Y-%m")
                month_groups[m_key].append(pt)
        
        monthly_returns = []
        positive_months = 0
        negative_months = 0
        best_month_pct = -999.0
        worst_month_pct = 999.0
        
        for m_key, pts in sorted(month_groups.items()):
            first_cap = pts[0].capital
            last_cap = pts[-1].capital
            if first_cap > 0:
                m_ret = (last_cap - first_cap) / first_cap * 100
            else:
                m_ret = 0.0
            
            pnl = last_cap - first_cap
            monthly_returns.append({"month": m_key, "return_pct": round(m_ret, 2), "pnl": round(pnl, 2)})
            
            best_month_pct = max(best_month_pct, m_ret)
            worst_month_pct = min(worst_month_pct, m_ret)
            
            if m_ret > 0:
                positive_months += 1
            elif m_ret < 0:
                negative_months += 1
                
        if best_month_pct == -999.0: best_month_pct = 0.0
        if worst_month_pct == 999.0: worst_month_pct = 0.0
            
        monthly_win_rate = (positive_months / len(monthly_returns) * 100) if monthly_returns else 0.0
        
        # Trade Metrics
        total_trades = len(trade_history)
        winning_trades = sum(1 for t in trade_history if t.outcome == "WIN")
        losing_trades = sum(1 for t in trade_history if t.outcome == "LOSS")
        breakeven_trades = sum(1 for t in trade_history if t.outcome == "BREAKEVEN")
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        wins = [t for t in trade_history if t.outcome == "WIN"]
        losses = [t for t in trade_history if t.outcome == "LOSS"]
        
        avg_win_amount = sum(t.net_pnl for t in wins) / max(len(wins), 1)
        avg_loss_amount = abs(sum(t.net_pnl for t in losses) / max(len(losses), 1))
        
        avg_win_pct = sum(t.net_pnl_pct for t in wins) / max(len(wins), 1)
        avg_loss_pct = abs(sum(t.net_pnl_pct for t in losses) / max(len(losses), 1))
        
        largest_win = max([t.net_pnl for t in wins] + [0.0])
        largest_loss = abs(min([t.net_pnl for t in losses] + [0.0]))
        
        largest_win_pct = max([t.net_pnl_pct for t in wins] + [0.0])
        largest_loss_pct = abs(min([t.net_pnl_pct for t in losses] + [0.0]))
        
        avg_trade_duration_bars = sum(t.duration_bars for t in trade_history) / total_trades if total_trades else 0.0
        avg_winning_trade_duration = sum(t.duration_bars for t in wins) / max(len(wins), 1)
        avg_losing_trade_duration = sum(t.duration_bars for t in losses) / max(len(losses), 1)
        
        gross_wins = sum(t.net_pnl for t in wins)
        gross_losses = abs(sum(t.net_pnl for t in losses))
        
        if gross_losses == 0:
            profit_factor = 999.0 if gross_wins > 0 else 0.0
        else:
            profit_factor = gross_wins / gross_losses
            
        if avg_loss_amount == 0:
            payoff_ratio = 999.0 if avg_win_amount > 0 else 0.0
        else:
            payoff_ratio = avg_win_amount / avg_loss_amount
            
        expected_value_per_trade = (win_rate/100 * avg_win_amount) - ((losing_trades/max(total_trades,1)) * avg_loss_amount)
        expected_value_pct = (win_rate/100 * avg_win_pct) - ((losing_trades/max(total_trades,1)) * avg_loss_pct)
        
        # Consecutives
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        curr_win_streak = 0
        curr_loss_streak = 0
        current_streak = 0
        
        for t in trade_history:
            if t.outcome == "WIN":
                curr_win_streak += 1
                curr_loss_streak = 0
                max_consecutive_wins = max(max_consecutive_wins, curr_win_streak)
                current_streak = curr_win_streak
            elif t.outcome == "LOSS":
                curr_loss_streak += 1
                curr_win_streak = 0
                max_consecutive_losses = max(max_consecutive_losses, curr_loss_streak)
                current_streak = -curr_loss_streak
            else:
                curr_win_streak = 0
                curr_loss_streak = 0
                current_streak = 0
        
        # Breakdowns
        def analyze_subset(trades):
            cnt = len(trades)
            if not cnt: return 0.0, 0.0
            wr = sum(1 for tr in trades if tr.outcome == "WIN") / cnt * 100
            pnl = sum(tr.net_pnl for tr in trades) / cnt
            return wr, pnl
            
        high_trades = [t for t in trade_history if t.confidence_level == "HIGH"]
        high_win_rate, high_avg_pnl = analyze_subset(high_trades)
        
        med_trades = [t for t in trade_history if t.confidence_level == "MEDIUM"]
        med_win_rate, med_avg_pnl = analyze_subset(med_trades)
        
        low_trades = [t for t in trade_history if t.confidence_level == "LOW"]
        low_win_rate, low_avg_pnl = analyze_subset(low_trades)
        
        call_trades = [t for t in trade_history if t.trade_type == "BUY_CALL"]
        put_trades = [t for t in trade_history if t.trade_type == "BUY_PUT"]
        
        call_win_rate = (sum(1 for t in call_trades if t.outcome == "WIN") / len(call_trades) * 100) if call_trades else 0.0
        call_total_pnl = sum(t.net_pnl for t in call_trades)
        
        put_win_rate = (sum(1 for t in put_trades if t.outcome == "WIN") / len(put_trades) * 100) if put_trades else 0.0
        put_total_pnl = sum(t.net_pnl for t in put_trades)
        
        # Exit Analysis
        target_hits = sum(1 for t in trade_history if t.exit_reason == "TARGET_HIT")
        sl_hits = sum(1 for t in trade_history if t.exit_reason == "STOP_LOSS_HIT")
        trailing_hits = sum(1 for t in trade_history if t.exit_reason == "TRAILING_SL_HIT")
        forced_eod = [t for t in trade_history if t.exit_reason == "FORCED_EOD"]
        
        target_hit_pct = target_hits / total_trades * 100 if total_trades else 0.0
        sl_hit_pct = sl_hits / total_trades * 100 if total_trades else 0.0
        trailing_sl_pct = trailing_hits / total_trades * 100 if total_trades else 0.0
        
        forced_eod_avg_pnl = sum(t.net_pnl for t in forced_eod) / len(forced_eod) if forced_eod else 0.0
        
        # Regime Analysis
        trades_by_regime = defaultdict(list)
        for t in trade_history:
            regime = t.entry_bar.get("regime", "UNKNOWN")
            trades_by_regime[regime].append(t)
            
        regime_stats = {}
        for reg, trs in trades_by_regime.items():
            wr, apnl = analyze_subset(trs)
            regime_stats[reg] = {"count": len(trs), "win_rate": round(wr, 2), "avg_pnl": round(apnl, 2)}
            
        best_regime = max(regime_stats.keys(), key=lambda k: regime_stats[k]["avg_pnl"]) if regime_stats else ""
        worst_regime = min(regime_stats.keys(), key=lambda k: regime_stats[k]["avg_pnl"]) if regime_stats else ""
        
        # Timing Analysis
        dow = defaultdict(list)
        hod = defaultdict(list)
        for t in trade_history:
            day = t.entry_timestamp.strftime("%A")
            hour = t.entry_timestamp.hour
            dow[day].append(t)
            hod[hour].append(t)
            
        dow_stats = {}
        for d, trs in dow.items():
            wr, apnl = analyze_subset(trs)
            dow_stats[d] = {"count": len(trs), "win_rate": round(wr, 2), "avg_pnl": round(apnl, 2)}
            
        best_dow = max(dow_stats.keys(), key=lambda k: dow_stats[k]["avg_pnl"]) if dow_stats else ""
        worst_dow = min(dow_stats.keys(), key=lambda k: dow_stats[k]["avg_pnl"]) if dow_stats else ""
        
        hod_stats = {}
        for h, trs in hod.items():
            wr, _ = analyze_subset(trs)
            hod_stats[h] = {"count": len(trs), "win_rate": round(wr, 2)}
            
        is_profitable = total_return_amount > 0
        has_edge = expected_value_per_trade > 0 and profit_factor > 1.1

        grade = MetricsCalculator._grade_strategy(win_rate, profit_factor, sharpe, max_drawdown_pct)
            
        metrics = BacktestMetrics(
            total_return_pct=round(total_return_pct, 4),
            total_return_amount=round(total_return_amount, 2),
            annualized_return_pct=round(annualized_return_pct, 4),
            monthly_returns=monthly_returns,
            best_month_pct=round(best_month_pct, 2),
            worst_month_pct=round(worst_month_pct, 2),
            positive_months=positive_months,
            negative_months=negative_months,
            monthly_win_rate=round(monthly_win_rate, 2),
            
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            win_rate=round(win_rate, 2),
            
            avg_win_amount=round(avg_win_amount, 2),
            avg_loss_amount=round(avg_loss_amount, 2),
            avg_win_pct=round(avg_win_pct, 4),
            avg_loss_pct=round(avg_loss_pct, 4),
            
            largest_win=round(largest_win, 2),
            largest_loss=round(largest_loss, 2),
            largest_win_pct=round(largest_win_pct, 4),
            largest_loss_pct=round(largest_loss_pct, 4),
            
            avg_trade_duration_bars=round(avg_trade_duration_bars, 2),
            avg_winning_trade_duration=round(avg_winning_trade_duration, 2),
            avg_losing_trade_duration=round(avg_losing_trade_duration, 2),
            
            profit_factor=round(profit_factor, 4),
            payoff_ratio=round(payoff_ratio, 4),
            expected_value_per_trade=round(expected_value_per_trade, 2),
            expected_value_pct=round(expected_value_pct, 4),
            
            max_drawdown_pct=round(max_drawdown_pct, 2),
            max_drawdown_amount=round(max_drawdown_amount, 2),
            max_drawdown_duration_bars=max_drawdown_duration_bars,
            max_drawdown_start=max_drawdown_start,
            max_drawdown_end=max_drawdown_end,
            avg_drawdown_pct=round(avg_drawdown_pct, 2),
            
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            current_streak=current_streak,
            
            max_recovery_time_bars=max_recovery_time_bars,
            
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar_ratio, 4),
            
            high_confidence_trades=len(high_trades),
            high_confidence_win_rate=round(high_win_rate, 2),
            high_confidence_avg_pnl=round(high_avg_pnl, 2),
            
            medium_confidence_trades=len(med_trades),
            medium_confidence_win_rate=round(med_win_rate, 2),
            medium_confidence_avg_pnl=round(med_avg_pnl, 2),
            
            low_confidence_trades=len(low_trades),
            low_confidence_win_rate=round(low_win_rate, 2),
            low_confidence_avg_pnl=round(low_avg_pnl, 2),
            
            call_trades=len(call_trades),
            call_win_rate=round(call_win_rate, 2),
            call_total_pnl=round(call_total_pnl, 2),
            
            put_trades=len(put_trades),
            put_win_rate=round(put_win_rate, 2),
            put_total_pnl=round(put_total_pnl, 2),
            
            target_hit_count=target_hits,
            target_hit_pct=round(target_hit_pct, 2),
            sl_hit_count=sl_hits,
            sl_hit_pct=round(sl_hit_pct, 2),
            trailing_sl_count=trailing_hits,
            trailing_sl_pct=round(trailing_sl_pct, 2),
            forced_eod_count=len(forced_eod),
            forced_eod_avg_pnl=round(forced_eod_avg_pnl, 2),
            
            trades_by_regime=regime_stats,
            best_regime=best_regime,
            worst_regime=worst_regime,
            
            trades_by_day_of_week=dow_stats,
            trades_by_hour=hod_stats,
            best_day_of_week=best_dow,
            worst_day_of_week=worst_dow,
            
            is_profitable=is_profitable,
            has_edge=has_edge,
            strategy_grade=grade,
            assessment="",
        )
        # === Benchmark comparison ===
        if benchmark_prices is not None and len(benchmark_prices) > 1:
            try:
                bm_start = float(benchmark_prices.iloc[0]['close'])
                bm_end = float(benchmark_prices.iloc[-1]['close'])
                benchmark_return = (bm_end - bm_start) / bm_start * 100 if bm_start > 0 else 0.0

                alpha = metrics.total_return_pct - benchmark_return

                trading_days = max(len(equity_curve), 1)
                alpha_annualized = alpha * (252 / trading_days)

                information_ratio = 0.0
                if len(equity_curve) > 1:
                    strategy_returns = [
                        equity_curve[i].capital / equity_curve[i - 1].capital - 1
                        for i in range(1, len(equity_curve))
                        if equity_curve[i - 1].capital > 0
                    ]
                    bm_returns = [
                        float(benchmark_prices.iloc[i]['close']) / float(benchmark_prices.iloc[i - 1]['close']) - 1
                        for i in range(1, len(benchmark_prices))
                        if float(benchmark_prices.iloc[i - 1]['close']) > 0
                    ]
                    min_len = min(len(strategy_returns), len(bm_returns))
                    if min_len > 1:
                        excess = [s - b for s, b in zip(strategy_returns[:min_len], bm_returns[:min_len])]
                        mean_ex = sum(excess) / len(excess)
                        variance = sum((r - mean_ex) ** 2 for r in excess) / len(excess)
                        tracking_error = math.sqrt(variance) * math.sqrt(252)
                        if tracking_error > 0:
                            information_ratio = (mean_ex * 252) / tracking_error

                metrics.benchmark_return_pct = round(benchmark_return, 2)
                metrics.benchmark_index = benchmark_name
                metrics.alpha = round(alpha, 2)
                metrics.alpha_annualized = round(alpha_annualized, 2)
                metrics.information_ratio = round(information_ratio, 4)
                metrics.beats_benchmark = alpha > 0

                if alpha > 0:
                    metrics.benchmark_comparison_note = (
                        f"Strategy OUTPERFORMED {benchmark_name} buy-and-hold by {alpha:+.1f}%"
                    )
                elif alpha > -2:
                    metrics.benchmark_comparison_note = (
                        f"Strategy roughly matched {benchmark_name} ({alpha:+.1f}% vs benchmark)"
                    )
                else:
                    metrics.benchmark_comparison_note = (
                        f"Strategy UNDERPERFORMED {benchmark_name} buy-and-hold by {abs(alpha):.1f}%"
                    )
            except Exception:
                pass  # Benchmark data malformed — skip silently

        metrics.assessment = MetricsCalculator.generate_assessment(metrics)
        return metrics

    @staticmethod
    def _grade_strategy(win_rate: float, profit_factor: float, sharpe: float, max_drawdown_pct: float) -> str:
        if win_rate > 55 and profit_factor > 1.5 and sharpe > 1.5 and max_drawdown_pct < 15:
            return "A"
        if win_rate > 50 and profit_factor > 1.2 and sharpe > 1.0 and max_drawdown_pct < 20:
            return "B"
        if win_rate > 45 and profit_factor > 1.0 and sharpe > 0.5:
            return "C"
        if profit_factor > 0.8:
            return "D"
        return "F"

    @staticmethod
    def _zero_metrics(initial_capital: float) -> BacktestMetrics:
        m = BacktestMetrics(
            total_return_pct=0.0, total_return_amount=0.0, annualized_return_pct=0.0,
            monthly_returns=[], best_month_pct=0.0, worst_month_pct=0.0,
            positive_months=0, negative_months=0, monthly_win_rate=0.0,
            total_trades=0, winning_trades=0, losing_trades=0, breakeven_trades=0, win_rate=0.0,
            avg_win_amount=0.0, avg_loss_amount=0.0, avg_win_pct=0.0, avg_loss_pct=0.0,
            largest_win=0.0, largest_loss=0.0, largest_win_pct=0.0, largest_loss_pct=0.0,
            avg_trade_duration_bars=0.0, avg_winning_trade_duration=0.0, avg_losing_trade_duration=0.0,
            profit_factor=0.0, payoff_ratio=0.0, expected_value_per_trade=0.0, expected_value_pct=0.0,
            max_drawdown_pct=0.0, max_drawdown_amount=0.0, max_drawdown_duration_bars=0,
            max_drawdown_start=None, max_drawdown_end=None, avg_drawdown_pct=0.0,
            max_consecutive_wins=0, max_consecutive_losses=0, current_streak=0,
            max_recovery_time_bars=0, sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            high_confidence_trades=0, high_confidence_win_rate=0.0, high_confidence_avg_pnl=0.0,
            medium_confidence_trades=0, medium_confidence_win_rate=0.0, medium_confidence_avg_pnl=0.0,
            low_confidence_trades=0, low_confidence_win_rate=0.0, low_confidence_avg_pnl=0.0,
            call_trades=0, call_win_rate=0.0, call_total_pnl=0.0,
            put_trades=0, put_win_rate=0.0, put_total_pnl=0.0,
            target_hit_count=0, target_hit_pct=0.0, sl_hit_count=0, sl_hit_pct=0.0,
            trailing_sl_count=0, trailing_sl_pct=0.0, forced_eod_count=0, forced_eod_avg_pnl=0.0,
            trades_by_regime={}, best_regime="", worst_regime="",
            trades_by_day_of_week={}, trades_by_hour={}, best_day_of_week="", worst_day_of_week="",
            is_profitable=False, has_edge=False, strategy_grade="F", assessment="No trades generated"
        )
        return m

    @staticmethod
    def generate_assessment(m: BacktestMetrics) -> str:
        if m.total_trades == 0:
            return "No trades generated"
            
        grade_desc = {
            "A": "A (Excellent)",
            "B": "B (Good)",
            "C": "C (Average)",
            "D": "D (Below Average)",
            "F": "F (Failing)"
        }.get(m.strategy_grade, "Unknown")
        
        capital = m.total_return_amount / (m.total_return_pct / 100) if m.total_return_pct != 0 else 0
        
        sign = "+" if m.total_return_amount > 0 else ""
        
        lines = [
            "BACKTEST ASSESSMENT — STRATEGY SUMMARY",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"GRADE: {grade_desc}",
            "",
            "RETURNS:",
            f"Total: {sign}{m.total_return_pct:.1f}% (₹{m.total_return_amount:,.0f})",
            f"Annualized: ~{m.annualized_return_pct:.0f}% | Best Month: +{m.best_month_pct:.1f}% | Worst: {m.worst_month_pct:.1f}%",
            "",
            "TRADE QUALITY:",
            f"{m.total_trades} trades | Win Rate: {m.win_rate:.1f}% | Profit Factor: {m.profit_factor:.2f}",
            f"Avg Win: ₹{m.avg_win_amount:.0f} | Avg Loss: ₹{m.avg_loss_amount:.0f} | Payoff: {m.payoff_ratio:.2f}",
            f"Expected Value: {sign}₹{m.expected_value_per_trade:.0f} per trade",
            "",
            "RISK:",
            f"Max Drawdown: -{m.max_drawdown_pct:.1f}% (₹{m.max_drawdown_amount:,.0f}) lasting {m.max_drawdown_duration_bars} bars",
            f"Sharpe: {m.sharpe_ratio:.2f} | Sortino: {m.sortino_ratio:.2f} | Calmar: {m.calmar_ratio:.2f}",
            f"Max Consecutive Losses: {m.max_consecutive_losses}",
            "",
            "CONFIDENCE BREAKDOWN:",
            f"HIGH confidence: {m.high_confidence_trades} trades, {m.high_confidence_win_rate:.1f}% win rate, avg +₹{m.high_confidence_avg_pnl:.0f}",
            f"MEDIUM confidence: {m.medium_confidence_trades} trades, {m.medium_confidence_win_rate:.1f}% win rate, avg +₹{m.medium_confidence_avg_pnl:.0f}",
            f"LOW confidence: {m.low_confidence_trades} trades, {m.low_confidence_win_rate:.1f}% win rate, avg {m.low_confidence_avg_pnl:+.0f}",
            "",
            "SIGNAL TYPE:",
            f"CALL: {m.call_trades} trades, {m.call_win_rate:.1f}% win rate, +₹{m.call_total_pnl:,.0f}",
            f"PUT: {m.put_trades} trades, {m.put_win_rate:.1f}% win rate, +₹{m.put_total_pnl:,.0f}",
            "",
            "RECOMMENDATIONS:"
        ]
        
        if m.has_edge:
            lines.append("• Strategy shows a positive edge — suitable for paper trading")
        else:
            lines.append("• Strategy lacks clear statistical edge in this run")
            
        if m.low_confidence_avg_pnl < 0:
            lines.append("• Consider removing LOW confidence signals (negative expected value)")
            
        if m.call_total_pnl > m.put_total_pnl and m.put_total_pnl < 0:
            lines.append("• CALL signals outperform PUT — strategy is better at detecting bullish setups")
        elif m.put_total_pnl > m.call_total_pnl and m.call_total_pnl < 0:
            lines.append("• PUT signals outperform CALL — strategy is better at detecting bearish setups")
            
        if m.max_drawdown_pct < 10:
            lines.append(f"• Max drawdown of {m.max_drawdown_pct:.1f}% is well within acceptable limits")
        elif m.max_drawdown_pct > 20:
            lines.append(f"• Warning: Max drawdown is high ({m.max_drawdown_pct:.1f}%) — review risk parameters")
            
        if m.best_regime:
            lines.append(f"• Best performance in {m.best_regime} regimes")

        if m.benchmark_comparison_note:
            lines += [
                "",
                "BENCHMARK vs BUY-AND-HOLD:",
                f"  {m.benchmark_comparison_note}",
                f"  Alpha: {m.alpha:+.1f}% | Info Ratio: {m.information_ratio:.2f}",
            ]
            if not m.beats_benchmark:
                lines.append("  ⚠️  Active trading did not add value vs simply holding the index.")

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    @staticmethod
    def compare_backtests(results: list[BacktestMetrics], labels: list[str] = None) -> ComparisonReport:
        if not labels or len(labels) != len(results):
            labels = [f"Run {i+1}" for i in range(len(results))]
            
        metrics_table = []
        metrics_to_show = {
            "Total Return %": lambda m: f"{m.total_return_pct:.2f}%",
            "Win Rate %": lambda m: f"{m.win_rate:.1f}%",
            "Profit Factor": lambda m: f"{m.profit_factor:.2f}",
            "Sharpe Ratio": lambda m: f"{m.sharpe_ratio:.2f}",
            "Sortino Ratio": lambda m: f"{m.sortino_ratio:.2f}",
            "Max Drawdown %": lambda m: f"-{m.max_drawdown_pct:.2f}%",
            "Total Trades": lambda m: str(m.total_trades),
            "Expected Value": lambda m: f"₹{m.expected_value_per_trade:.0f}"
        }
        
        for name, extractor in metrics_to_show.items():
            metrics_table.append({
                "metric_name": name,
                "values": [extractor(r) for r in results]
            })
            
        scores = []
        for i, m in enumerate(results):
            score = (m.sharpe_ratio * 0.3) + (max(min(m.win_rate / 100, 1.0), 0.0) * 0.2) + (max(min(m.profit_factor / 3.0, 1.0), 0.0) * 0.2) + (max(1.0 - m.max_drawdown_pct/100, 0.0) * 0.3)
            # Alternative requested composite score formula: sharpe * 0.3 + win_rate * 0.2 + profit_factor * 0.2 + (1 - max_dd/100) * 0.3
            score = m.sharpe_ratio * 0.3 + m.win_rate * 0.2 + m.profit_factor * 0.2 + (1 - m.max_drawdown_pct/100) * 0.3
            scores.append((labels[i], score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        best_overall = scores[0][0] if scores else ""
        
        return ComparisonReport(
            headers=labels,
            metrics_table=metrics_table,
            best_overall=best_overall,
            ranking=scores
        )
