import { formatPercentage, formatCurrency } from '../../utils/formatters';

const STATS = [
  { key: 'win_rate', label: 'Win Rate', fmt: (v) => formatPercentage(v), positive: (v) => v >= 50 },
  { key: 'profit_factor', label: 'Profit Factor', fmt: (v) => v?.toFixed(2) ?? '--', positive: (v) => v >= 1 },
  { key: 'sharpe_ratio', label: 'Sharpe Ratio', fmt: (v) => v?.toFixed(2) ?? '--', positive: (v) => v >= 1 },
  { key: 'max_drawdown', label: 'Max Drawdown', fmt: (v) => formatPercentage(v), positive: () => false, invert: true },
  { key: 'total_pnl', label: 'Total P&L', fmt: (v) => formatCurrency(v), positive: (v) => v >= 0 },
  { key: 'avg_pnl_per_trade', label: 'Avg P&L/Trade', fmt: (v) => formatCurrency(v), positive: (v) => v >= 0 },
  { key: 'total_trades', label: 'Total Trades', fmt: (v) => v ?? '--', positive: null },
  { key: 'wins', label: 'Wins', fmt: (v) => v ?? '--', positive: null },
  { key: 'losses', label: 'Losses', fmt: (v) => v ?? '--', positive: null, invert: true },
  { key: 'largest_win', label: 'Largest Win', fmt: (v) => formatCurrency(v), positive: () => true },
  { key: 'largest_loss', label: 'Largest Loss', fmt: (v) => formatCurrency(v), positive: () => false, invert: true },
];

export function PerformanceStats({ stats }) {
  if (!stats) return null;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {STATS.map(({ key, label, fmt, positive, invert }) => {
        const value = stats[key];
        let color = 'text-slate-200';
        if (positive !== null) {
          const isPositive = positive ? positive(value) : false;
          color = invert
            ? isPositive ? 'text-slate-200' : 'text-slate-200'
            : isPositive ? 'text-green-400' : (value != null && value < 0 ? 'text-red-400' : 'text-slate-200');
        }
        return (
          <div key={key} className="bg-slate-800 rounded-lg border border-slate-700 p-3">
            <div className="text-slate-500 text-[11px] mb-1">{label}</div>
            <div className={`font-mono font-bold text-base ${color}`}>
              {value != null ? fmt(value) : '--'}
            </div>
          </div>
        );
      })}
    </div>
  );
}
