import { AlertTriangle, Clock } from 'lucide-react';
import { Badge } from '../common/Badge';
import { timeAgo } from '../../utils/formatters';
import { SEVERITY_COLORS } from '../../utils/constants';

export function AnomalyCard({ anomaly }) {
  const {
    index_id, anomaly_type, severity, category, message, details, timestamp, is_active,
  } = anomaly;

  const sevColor = SEVERITY_COLORS[severity] ?? 'text-slate-400';

  return (
    <div className={`bg-slate-800 border rounded-lg p-3 space-y-2 ${
      severity === 'HIGH' ? 'border-red-700/50' : severity === 'MEDIUM' ? 'border-yellow-700/40' : 'border-slate-700'
    }`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-start gap-2">
          <AlertTriangle className={`w-4 h-4 flex-shrink-0 mt-0.5 ${sevColor}`} />
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-slate-100 text-sm font-semibold">{index_id}</span>
              <Badge variant={severity} size="xs">{severity}</Badge>
              <Badge variant={category} size="xs">{category}</Badge>
              {!is_active && <Badge variant="EXPIRED" size="xs">RESOLVED</Badge>}
            </div>
            <div className="text-slate-400 text-xs mt-0.5 font-mono">{anomaly_type}</div>
          </div>
        </div>
        <div className="flex items-center gap-1 text-slate-500 text-[10px] flex-shrink-0">
          <Clock className="w-3 h-3" />
          {timeAgo(timestamp)}
        </div>
      </div>

      {message && (
        <p className="text-slate-300 text-xs leading-snug">{message}</p>
      )}

      {details && Object.keys(details).length > 0 && (
        <div className="bg-slate-900/50 rounded p-2 space-y-0.5">
          {Object.entries(details).slice(0, 6).map(([k, v]) => (
            <div key={k} className="flex items-center justify-between text-[11px]">
              <span className="text-slate-500 capitalize">{k.replace(/_/g, ' ')}</span>
              <span className="text-slate-300 font-mono">{String(v)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
