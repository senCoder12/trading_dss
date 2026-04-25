import { useState } from 'react';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Spinner } from '../common/Spinner';
import { EmptyState } from '../common/EmptyState';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { AnomalyCard } from './AnomalyCard';
import { REFRESH_INTERVALS } from '../../utils/constants';
import { CheckCircle } from 'lucide-react';

const SEV_OPTS = [null, 'HIGH', 'MEDIUM', 'LOW'];
const SEV_ORDER = { HIGH: 0, MEDIUM: 1, LOW: 2 };

export default function AnomalyPage() {
  const [minSev, setMinSev] = useState(null);

  const { data: dashboard } = usePolling(
    api.getAnomalyDashboard, REFRESH_INTERVALS.anomalies,
  );
  const { data, loading, error, lastUpdated } = usePolling(
    api.getActiveAnomalies,
    REFRESH_INTERVALS.anomalies,
  );

  const allAnomalies = data?.anomalies ?? [];
  const anomalies = minSev
    ? allAnomalies.filter(a => (SEV_ORDER[a.severity] ?? 2) <= (SEV_ORDER[minSev] ?? 2))
    : allAnomalies;

  return (
    <div className="p-4 space-y-4">
      {/* Dashboard summary */}
      {dashboard && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: 'Total Active', value: dashboard.total_active, color: dashboard.total_active > 0 ? 'text-red-400' : 'text-green-400' },
            { label: 'High Severity', value: dashboard.high_severity, color: dashboard.high_severity > 0 ? 'text-red-400' : 'text-slate-200' },
            { label: 'Medium', value: dashboard.medium_severity, color: dashboard.medium_severity > 0 ? 'text-yellow-400' : 'text-slate-200' },
            { label: 'Low', value: dashboard.low_severity, color: dashboard.low_severity > 0 ? 'text-blue-400' : 'text-slate-200' },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-slate-800 border border-slate-700 rounded-lg p-3 text-center">
              <div className={`font-bold text-2xl ${color}`}>{value}</div>
              <div className="text-slate-500 text-xs mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* By category + by index */}
      {dashboard && (Object.keys(dashboard.by_category ?? {}).length > 0 || Object.keys(dashboard.by_index ?? {}).length > 0) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {Object.keys(dashboard.by_category ?? {}).length > 0 && (
            <Card title="By Category" className="">
              <div className="space-y-2">
                {Object.entries(dashboard.by_category).map(([cat, cnt]) => (
                  <div key={cat} className="flex items-center justify-between text-sm">
                    <span className="text-slate-400">{cat}</span>
                    <span className="text-slate-100 font-semibold">{cnt}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
          {Object.keys(dashboard.by_index ?? {}).length > 0 && (
            <Card title="By Index" className="">
              <div className="space-y-2">
                {Object.entries(dashboard.by_index).map(([idx, cnt]) => (
                  <div key={idx} className="flex items-center justify-between text-sm">
                    <span className="text-slate-400">{idx}</span>
                    <span className="text-slate-100 font-semibold">{cnt}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* Active alerts list */}
      <Card
        title={`Active Alerts${anomalies.length > 0 ? ` (${anomalies.length})` : ''}`}
        padding={false}
        actions={<RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />}
      >
        <div className="px-4 py-2.5 border-b border-slate-700 flex items-center gap-2">
          <span className="text-slate-500 text-xs">Min Severity:</span>
          {SEV_OPTS.map((s) => (
            <button
              key={String(s)}
              onClick={() => setMinSev(s)}
              className={`px-2.5 py-0.5 rounded text-xs font-medium transition-colors ${
                minSev === s
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
              }`}
            >
              {s ?? 'All'}
            </button>
          ))}
        </div>

        <div className="p-4">
          {loading ? (
            <div className="flex justify-center p-8"><Spinner /></div>
          ) : !anomalies.length ? (
            <EmptyState
              icon={CheckCircle}
              title="No active alerts"
              message="All clear — no anomalies detected"
            />
          ) : (
            <div className="space-y-3">
              {anomalies.map((a) => (
                <AnomalyCard key={a.id ?? `${a.index_id}-${a.timestamp}`} anomaly={a} />
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
