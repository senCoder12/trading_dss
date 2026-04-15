import { ArrowUpCircle, ArrowDownCircle, MinusCircle } from 'lucide-react';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { Spinner } from '../common/Spinner';
import { EmptyState } from '../common/EmptyState';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { formatPrice, formatConfidence, timeAgo } from '../../utils/formatters';
import { REFRESH_INTERVALS, SIGNAL_TYPE_LABELS } from '../../utils/constants';
import { TrendingUp } from 'lucide-react';

const SIGNAL_ICONS = {
  BUY_CALL: <ArrowUpCircle className="w-4 h-4 text-green-400" />,
  BUY_PUT: <ArrowDownCircle className="w-4 h-4 text-red-400" />,
  NO_TRADE: <MinusCircle className="w-4 h-4 text-slate-500" />,
};

export default function ActiveSignals() {
  const { data, loading, error, lastUpdated } = usePolling(api.getCurrentSignals, REFRESH_INTERVALS.signals);
  const signals = data?.signals ?? [];

  return (
    <Card
      title="Active Signals"
      padding={false}
      actions={<RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />}
    >
      {loading && !data ? (
        <div className="flex justify-center p-8"><Spinner /></div>
      ) : !signals.length ? (
        <EmptyState icon={TrendingUp} title="No signals" message="Signal engine may be offline" />
      ) : (
        <div className="divide-y divide-slate-700/50">
          {signals.map((sig) => {
            const isNoTrade = sig.signal_type === 'NO_TRADE';
            return (
              <div
                key={sig.index_id}
                className={`p-3 hover:bg-slate-700/30 transition-colors ${isNoTrade ? 'opacity-60' : ''}`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    {SIGNAL_ICONS[sig.signal_type] ?? SIGNAL_ICONS.NO_TRADE}
                    <span className="text-slate-100 font-semibold text-sm">{sig.index_id}</span>
                    <Badge variant={sig.signal_type}>
                      {SIGNAL_TYPE_LABELS[sig.signal_type] ?? sig.signal_type}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={sig.confidence_level}>{sig.confidence_level}</Badge>
                    {sig.confidence_score != null && (
                      <span className="text-slate-400 text-xs font-mono">
                        {formatConfidence(sig.confidence_score)}
                      </span>
                    )}
                  </div>
                </div>

                {!isNoTrade && (sig.entry || sig.target || sig.sl) && (
                  <div className="grid grid-cols-3 gap-1 text-[11px]">
                    {sig.entry != null && (
                      <div>
                        <span className="text-slate-500">Entry </span>
                        <span className="text-slate-300 font-mono">{formatPrice(sig.entry)}</span>
                      </div>
                    )}
                    {sig.target != null && (
                      <div>
                        <span className="text-slate-500">Target </span>
                        <span className="text-green-400 font-mono">{formatPrice(sig.target)}</span>
                      </div>
                    )}
                    {sig.sl != null && (
                      <div>
                        <span className="text-slate-500">SL </span>
                        <span className="text-red-400 font-mono">{formatPrice(sig.sl)}</span>
                      </div>
                    )}
                  </div>
                )}

                {sig.reasoning_summary && (
                  <p className="text-[11px] text-slate-500 mt-1 line-clamp-1">{sig.reasoning_summary}</p>
                )}

                {sig.generated_at && (
                  <div className="text-[10px] text-slate-600 mt-1">{timeAgo(sig.generated_at)}</div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}
