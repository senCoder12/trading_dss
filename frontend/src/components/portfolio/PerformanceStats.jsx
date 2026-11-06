import { formatPercentage, formatCurrency, formatPnL } from '../../utils/formatters';

const num = (v, d = 2) => (v || v === 0 ? Number(v).toFixed(d) : '—');
const signCls = (v) => (v > 0 ? 'text-up' : v < 0 ? 'text-down' : 'text-ink');

// Grouped so the page reads as three short lists instead of an 11-tile wall.
const GROUPS = [
  {
    title: 'Returns',
    rows: [
      ['Total P&L', (s) => formatPnL(s.total_pnl), (s) => signCls(s.total_pnl)],
      ['Average per trade', (s) => formatPnL(s.avg_pnl_per_trade), (s) => signCls(s.avg_pnl_per_trade)],
      ['Largest win', (s) => formatCurrency(s.largest_win), () => 'text-ink'],
      ['Largest loss', (s) => formatCurrency(s.largest_loss), (s) => (s.largest_loss ? 'text-down' : 'text-ink')],
    ],
  },
  {
    title: 'Quality',
    rows: [
      ['Win rate', (s) => formatPercentage(s.win_rate, false), () => 'text-ink'],
      ['Profit factor', (s) => (s.profit_factor ? num(s.profit_factor) : '—'), (s) => (s.profit_factor ? (s.profit_factor >= 1 ? 'text-up' : 'text-down') : 'text-ink')],
      ['Sharpe ratio', (s) => (s.sharpe_ratio ? num(s.sharpe_ratio) : '—'), () => 'text-ink'],
      ['Max drawdown', (s) => formatPercentage(s.max_drawdown, false), (s) => (s.max_drawdown ? 'text-down' : 'text-ink')],
    ],
  },
  {
    title: 'Activity',
    rows: [
      ['Closed trades', (s) => s.total_trades ?? '—', () => 'text-ink'],
      ['Wins', (s) => s.wins ?? '—', () => 'text-ink'],
      ['Losses', (s) => s.losses ?? '—', () => 'text-ink'],
      ['Actionable signals', (s) => s.actionable_signals ?? '—', () => 'text-ink'],
    ],
  },
];

export function PerformanceStats({ stats }) {
  if (!stats) return null;
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
      {GROUPS.map((g) => (
        <section key={g.title} className="card">
          <header className="card-header"><h3 className="card-title">{g.title}</h3></header>
          <dl className="divide-y divide-line">
            {g.rows.map(([label, fmt, cls]) => (
              <div key={label} className="flex items-center justify-between px-5 py-3 text-[13px]">
                <dt className="text-ink-muted">{label}</dt>
                <dd className={`font-medium num ${cls(stats)}`}>{fmt(stats)}</dd>
              </div>
            ))}
          </dl>
        </section>
      ))}
    </div>
  );
}
