import { Badge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';
import { formatPrice, formatPnL, formatPercentage, timeAgo } from '../../utils/formatters';
import { SIGNAL_TYPE_LABELS } from '../../utils/constants';
import { Briefcase } from 'lucide-react';

export function PositionsList({ positions }) {
  if (!positions?.length) return (
    <EmptyState icon={Briefcase} title="No open positions" message="No trades currently open" />
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-400 border-b border-slate-700">
            <th className="text-left py-2 px-3 font-medium">Index</th>
            <th className="text-left py-2 px-3 font-medium">Direction</th>
            <th className="text-left py-2 px-3 font-medium">Conf.</th>
            <th className="text-right py-2 px-3 font-medium">Entry</th>
            <th className="text-right py-2 px-3 font-medium">Current</th>
            <th className="text-right py-2 px-3 font-medium">Target</th>
            <th className="text-right py-2 px-3 font-medium">SL</th>
            <th className="text-right py-2 px-3 font-medium">Unr. P&L</th>
            <th className="text-right py-2 px-3 font-medium">Age</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p) => {
            const pnlColor = p.unrealized_pnl > 0 ? 'text-green-400' : p.unrealized_pnl < 0 ? 'text-red-400' : 'text-slate-400';
            return (
              <tr key={p.id} className="border-b border-slate-700/50 table-row-hover">
                <td className="py-2 px-3 text-slate-200 font-semibold">{p.index_id}</td>
                <td className="py-2 px-3">
                  <Badge variant={p.signal_type} size="xs">
                    {SIGNAL_TYPE_LABELS[p.signal_type] ?? p.signal_type}
                  </Badge>
                </td>
                <td className="py-2 px-3">
                  <Badge variant={p.confidence_level} size="xs">{p.confidence_level}</Badge>
                </td>
                <td className="py-2 px-3 text-right font-mono text-slate-300">
                  {formatPrice(p.entry_price)}
                </td>
                <td className="py-2 px-3 text-right font-mono text-slate-200">
                  {p.current_price ? formatPrice(p.current_price) : '--'}
                </td>
                <td className="py-2 px-3 text-right font-mono text-green-400">
                  {p.target_price ? formatPrice(p.target_price) : '--'}
                </td>
                <td className="py-2 px-3 text-right font-mono text-red-400">
                  {p.stop_loss ? formatPrice(p.stop_loss) : '--'}
                </td>
                <td className={`py-2 px-3 text-right font-mono font-semibold ${pnlColor}`}>
                  {p.unrealized_pnl != null ? (
                    <>
                      {formatPnL(p.unrealized_pnl)}{' '}
                      <span className="text-[10px]">({formatPercentage(p.unrealized_pnl_pct)})</span>
                    </>
                  ) : '--'}
                </td>
                <td className="py-2 px-3 text-right text-slate-500">
                  {timeAgo(p.generated_at)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
