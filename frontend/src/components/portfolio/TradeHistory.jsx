import { Badge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';
import { formatPrice, formatPnL, formatISTDateTime } from '../../utils/formatters';
import { SIGNAL_TYPE_LABELS } from '../../utils/constants';
import { BarChart2, Download } from 'lucide-react';

export function TradeHistory({ trades, days = 7 }) {
  if (!trades?.length) return (
    <EmptyState icon={BarChart2} title="No closed trades" message="Completed trades will appear here" />
  );

  return (
    <div className="overflow-x-auto">
      <div className="flex justify-end px-3 py-2">
        <button
          onClick={() => {
            window.open(`/api/portfolio/trades/export?days=${days}`, '_blank');
          }}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-xs text-slate-300 transition-colors"
        >
          <Download size={13} />
          Export CSV
        </button>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-400 border-b border-slate-700">
            <th className="text-left py-2 px-3 font-medium">Index</th>
            <th className="text-left py-2 px-3 font-medium">Signal</th>
            <th className="text-left py-2 px-3 font-medium">Conf.</th>
            <th className="text-right py-2 px-3 font-medium">Entry</th>
            <th className="text-right py-2 px-3 font-medium">Exit</th>
            <th className="text-left py-2 px-3 font-medium">Result</th>
            <th className="text-right py-2 px-3 font-medium">P&L</th>
            <th className="text-right py-2 px-3 font-medium">Closed</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => {
            const pnlColor = t.pnl > 0 ? 'text-green-400' : t.pnl < 0 ? 'text-red-400' : 'text-slate-400';
            return (
              <tr key={t.id} className="border-b border-slate-700/50 table-row-hover">
                <td className="py-2 px-3 text-slate-200 font-semibold">{t.index_id}</td>
                <td className="py-2 px-3">
                  <Badge variant={t.signal_type} size="xs">
                    {SIGNAL_TYPE_LABELS[t.signal_type] ?? t.signal_type}
                  </Badge>
                </td>
                <td className="py-2 px-3">
                  <Badge variant={t.confidence_level} size="xs">{t.confidence_level}</Badge>
                </td>
                <td className="py-2 px-3 text-right font-mono text-slate-300">
                  {formatPrice(t.entry_price)}
                </td>
                <td className="py-2 px-3 text-right font-mono text-slate-300">
                  {t.exit_price ? formatPrice(t.exit_price) : '--'}
                </td>
                <td className="py-2 px-3">
                  {t.outcome ? <Badge variant={t.outcome} size="xs">{t.outcome}</Badge> : '--'}
                </td>
                <td className={`py-2 px-3 text-right font-mono font-semibold ${pnlColor}`}>
                  {t.pnl != null ? formatPnL(t.pnl) : '--'}
                </td>
                <td className="py-2 px-3 text-right text-slate-500">
                  {formatISTDateTime(t.closed_at)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
