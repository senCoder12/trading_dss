import { History } from 'lucide-react';
import { Badge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';
import { RowSkeleton } from '../common/Skeleton';
import { formatPrice, formatISTDateTime, formatPnL } from '../../utils/formatters';
import { SIGNAL_TYPE_LABELS } from '../../utils/constants';

export function SignalHistory({ signals, loading, days }) {
  if (loading) return <RowSkeleton rows={5} />;
  if (!signals?.length) {
    return (
      <EmptyState
        icon={History}
        title="No signals in this period"
        message={`Nothing was recorded in the last ${days} days. Try a longer window.`}
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
            <th className="r">Target</th>
            <th className="r">Stop</th>
            <th>Outcome</th>
            <th className="r">P&L</th>
            <th className="r">Generated</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s) => (
            <tr key={s.id}>
              <td className="text-ink font-medium">{s.index_id}</td>
              <td><Badge variant={s.signal_type} size="xs">{SIGNAL_TYPE_LABELS[s.signal_type] ?? s.signal_type}</Badge></td>
              <td><Badge variant={s.confidence_level} size="xs">{s.confidence_level}</Badge></td>
              <td className="r num">{s.entry_price ? formatPrice(s.entry_price) : '—'}</td>
              <td className="r num text-up">{s.target_price ? formatPrice(s.target_price) : '—'}</td>
              <td className="r num text-down">{s.stop_loss ? formatPrice(s.stop_loss) : '—'}</td>
              <td>{s.outcome ? <Badge variant={s.outcome} size="xs">{s.outcome}</Badge> : <span className="text-ink-ghost">—</span>}</td>
              <td className={`r num font-medium ${s.actual_pnl > 0 ? 'text-up' : s.actual_pnl < 0 ? 'text-down' : ''}`}>
                {s.actual_pnl != null ? formatPnL(s.actual_pnl) : '—'}
              </td>
              <td className="r text-ink-faint">{formatISTDateTime(s.generated_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
