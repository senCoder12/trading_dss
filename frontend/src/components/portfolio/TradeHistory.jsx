import { Receipt } from 'lucide-react';
import { Badge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';
import { formatPrice, formatPnL, formatISTDateTime } from '../../utils/formatters';
import { SIGNAL_TYPE_LABELS } from '../../utils/constants';

export function TradeHistory({ trades, days = 7 }) {
  if (!trades?.length) {
    return (
      <EmptyState
        icon={Receipt}
        title="No closed trades"
        message={`Nothing closed in the last ${days} days. Completed trades will appear here.`}
      />
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="tbl">
        <thead>
          <tr>
            <th>Index</th>
            <th>Signal</th>
            <th>Confidence</th>
            <th className="r">Entry</th>
            <th className="r">Exit</th>
            <th>Result</th>
            <th className="r">P&L</th>
            <th className="r">Closed</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => {
            const pnlCls = t.pnl > 0 ? 'text-up' : t.pnl < 0 ? 'text-down' : '';
            return (
              <tr key={t.id}>
                <td className="text-ink font-medium">{t.index_id}</td>
                <td><Badge variant={t.signal_type} size="xs">{SIGNAL_TYPE_LABELS[t.signal_type] ?? t.signal_type}</Badge></td>
                <td><Badge variant={t.confidence_level} size="xs">{t.confidence_level}</Badge></td>
                <td className="r num">{formatPrice(t.entry_price)}</td>
                <td className="r num">{t.exit_price ? formatPrice(t.exit_price) : '—'}</td>
                <td>{t.outcome ? <Badge variant={t.outcome} size="xs">{t.outcome}</Badge> : <span className="text-ink-ghost">—</span>}</td>
                <td className={`r num font-medium ${pnlCls}`}>{t.pnl != null ? formatPnL(t.pnl) : '—'}</td>
                <td className="r text-ink-faint">{formatISTDateTime(t.closed_at)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
