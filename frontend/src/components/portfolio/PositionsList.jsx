import { Briefcase } from 'lucide-react';
import { Badge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';
import { formatPrice, formatPnL, formatPercentage, timeAgo } from '../../utils/formatters';
import { SIGNAL_TYPE_LABELS } from '../../utils/constants';

export function PositionsList({ positions }) {
  if (!positions?.length) {
    return <EmptyState icon={Briefcase} title="No open positions" message="Positions appear here when a signal is taken." />;
  }

  return (
    <div className="overflow-x-auto">
      <table className="tbl">
        <thead>
          <tr>
            <th>Index</th>
            <th>Direction</th>
            <th>Confidence</th>
            <th className="r">Entry</th>
            <th className="r">Current</th>
            <th className="r">Target</th>
            <th className="r">Stop</th>
            <th className="r">Unrealised P&L</th>
            <th className="r">Age</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p) => {
            const pnlCls = p.unrealized_pnl > 0 ? 'text-up' : p.unrealized_pnl < 0 ? 'text-down' : '';
            return (
              <tr key={p.id}>
                <td className="text-ink font-medium">{p.index_id}</td>
                <td><Badge variant={p.signal_type} size="xs">{SIGNAL_TYPE_LABELS[p.signal_type] ?? p.signal_type}</Badge></td>
                <td><Badge variant={p.confidence_level} size="xs">{p.confidence_level}</Badge></td>
                <td className="r num">{formatPrice(p.entry_price)}</td>
                <td className="r num text-ink">{p.current_price ? formatPrice(p.current_price) : '—'}</td>
                <td className="r num text-up">{p.target_price ? formatPrice(p.target_price) : '—'}</td>
                <td className="r num text-down">{p.stop_loss ? formatPrice(p.stop_loss) : '—'}</td>
                <td className={`r num font-medium ${pnlCls}`}>
                  {p.unrealized_pnl != null ? (
                    <>
                      {formatPnL(p.unrealized_pnl)}
                      <span className="text-2xs ml-1 opacity-80">{formatPercentage(p.unrealized_pnl_pct)}</span>
                    </>
                  ) : '—'}
                </td>
                <td className="r text-ink-faint">{timeAgo(p.generated_at)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
