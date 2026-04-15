import { Badge } from '../common/Badge';
import { Spinner } from '../common/Spinner';
import { EmptyState } from '../common/EmptyState';
import { formatPrice, formatISTDateTime, formatPnL } from '../../utils/formatters';
import { SIGNAL_TYPE_LABELS } from '../../utils/constants';
import { Clock } from 'lucide-react';

export function SignalHistory({ signals, loading }) {
  if (loading) return <div className="flex justify-center p-8"><Spinner /></div>;
  if (!signals?.length) return <EmptyState icon={Clock} title="No signal history" />;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-400 border-b border-slate-700">
            <th className="text-left py-2 px-3 font-medium">Index</th>
            <th className="text-left py-2 px-3 font-medium">Signal</th>
            <th className="text-left py-2 px-3 font-medium">Conf.</th>
            <th className="text-right py-2 px-3 font-medium">Entry</th>
            <th className="text-right py-2 px-3 font-medium">Target</th>
            <th className="text-right py-2 px-3 font-medium">SL</th>
            <th className="text-left py-2 px-3 font-medium">Outcome</th>
            <th className="text-right py-2 px-3 font-medium">P&L</th>
            <th className="text-right py-2 px-3 font-medium">Generated</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s) => (
            <tr key={s.id} className="border-b border-slate-700/50 table-row-hover">
              <td className="py-2 px-3 text-slate-200 font-medium">{s.index_id}</td>
              <td className="py-2 px-3">
                <Badge variant={s.signal_type} size="xs">
                  {SIGNAL_TYPE_LABELS[s.signal_type] ?? s.signal_type}
                </Badge>
              </td>
              <td className="py-2 px-3">
                <Badge variant={s.confidence_level} size="xs">{s.confidence_level}</Badge>
              </td>
              <td className="py-2 px-3 text-right font-mono text-slate-300">
                {s.entry_price ? formatPrice(s.entry_price) : '--'}
              </td>
              <td className="py-2 px-3 text-right font-mono text-green-400">
                {s.target_price ? formatPrice(s.target_price) : '--'}
              </td>
              <td className="py-2 px-3 text-right font-mono text-red-400">
                {s.stop_loss ? formatPrice(s.stop_loss) : '--'}
              </td>
              <td className="py-2 px-3">
                {s.outcome ? (
                  <Badge variant={s.outcome} size="xs">{s.outcome}</Badge>
                ) : (
                  <span className="text-slate-600">--</span>
                )}
              </td>
              <td className={`py-2 px-3 text-right font-mono ${
                s.actual_pnl > 0 ? 'text-green-400' : s.actual_pnl < 0 ? 'text-red-400' : 'text-slate-500'
              }`}>
                {s.actual_pnl != null ? formatPnL(s.actual_pnl) : '--'}
              </td>
              <td className="py-2 px-3 text-right text-slate-500">
                {formatISTDateTime(s.generated_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
