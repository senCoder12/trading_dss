import { ArrowUpRight, ArrowDownRight, Minus, Crosshair } from 'lucide-react';
import { useDataStore } from '../../hooks/useDataStore';
import { useIndexMeta, indexLabel } from '../../hooks/useIndexMeta';
import { Badge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';
import { RowSkeleton } from '../common/Skeleton';
import { formatPrice, formatConfidence, timeAgo } from '../../utils/formatters';
import { SIGNAL_TYPE_LABELS, SHORT_NAMES } from '../../utils/constants';

const ICON = {
  BUY_CALL: { Icon: ArrowUpRight, cls: 'bg-up/10 text-up' },
  BUY_PUT: { Icon: ArrowDownRight, cls: 'bg-down/10 text-down' },
  NO_TRADE: { Icon: Minus, cls: 'bg-surface-3 text-ink-faint' },
};

/** Compact signal rows for the dashboard side panel. Actionable signals first. */
export default function ActiveSignals({ limit = 6 }) {
  const { signals: { data, loading } } = useDataStore();
  const meta = useIndexMeta();
  const signals = [...(data?.signals ?? [])].sort(
    (a, b) => (a.signal_type === 'NO_TRADE') - (b.signal_type === 'NO_TRADE'),
  );

  if (loading && !data) return <RowSkeleton rows={4} />;
  if (!signals.length) {
    return <EmptyState icon={Crosshair} title="No signals yet" message="The signal engine has not produced output for this session." />;
  }

  return (
    <ul className="divide-y divide-line">
      {signals.slice(0, limit).map((sig) => {
        const { Icon, cls } = ICON[sig.signal_type] ?? ICON.NO_TRADE;
        const actionable = sig.signal_type !== 'NO_TRADE';
        return (
          <li key={sig.index_id} className="flex items-center gap-3 px-5 py-3 hover:bg-surface-2/60 transition-colors">
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${cls}`}>
              <Icon className="w-4 h-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-medium text-ink truncate">
                  {SHORT_NAMES[sig.index_id] ?? indexLabel(meta, sig.index_id)}
                </span>
                {sig.generated_at && <span className="text-2xs text-ink-ghost">{timeAgo(sig.generated_at)}</span>}
              </div>
              <div className="text-xs text-ink-faint truncate mt-0.5">
                {actionable && sig.entry
                  ? <span className="num">Entry {formatPrice(sig.entry)} · Tgt <span className="text-up">{formatPrice(sig.target)}</span> · SL <span className="text-down">{formatPrice(sig.sl)}</span></span>
                  : (sig.reasoning_summary?.replace(/^NO_TRADE:\s*/, '') ?? sig.regime?.replace(/_/g, ' ').toLowerCase() ?? '—')}
              </div>
            </div>
            <div className="flex flex-col items-end gap-1 flex-shrink-0">
              <Badge variant={sig.signal_type} size="xs">{SIGNAL_TYPE_LABELS[sig.signal_type] ?? sig.signal_type}</Badge>
              {sig.confidence_score != null && (
                <span className="text-2xs text-ink-faint num">{formatConfidence(sig.confidence_score)} conf.</span>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
