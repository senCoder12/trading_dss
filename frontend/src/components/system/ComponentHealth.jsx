import { CheckCircle, AlertTriangle, XCircle, HelpCircle, Clock } from 'lucide-react';
import { timeAgo } from '../../utils/formatters';

const STATUS_ICONS = {
  OK: <CheckCircle className="w-4 h-4 text-green-400" />,
  RUNNING: <CheckCircle className="w-4 h-4 text-green-400" />,
  HEALTHY: <CheckCircle className="w-4 h-4 text-green-400" />,
  WARNING: <AlertTriangle className="w-4 h-4 text-yellow-400" />,
  DEGRADED: <AlertTriangle className="w-4 h-4 text-yellow-400" />,
  ERROR: <XCircle className="w-4 h-4 text-red-400" />,
  UNHEALTHY: <XCircle className="w-4 h-4 text-red-400" />,
  UNKNOWN: <HelpCircle className="w-4 h-4 text-slate-500" />,
};

const STATUS_BG = {
  OK: 'border-green-700/30',
  RUNNING: 'border-green-700/30',
  HEALTHY: 'border-green-700/30',
  WARNING: 'border-yellow-700/40',
  DEGRADED: 'border-yellow-700/40',
  ERROR: 'border-red-700/50',
  UNHEALTHY: 'border-red-700/50',
  UNKNOWN: 'border-slate-700',
};

export function ComponentHealth({ component }) {
  const { component: name, status, last_seen, message } = component;
  const icon = STATUS_ICONS[status?.toUpperCase()] ?? STATUS_ICONS.UNKNOWN;
  const border = STATUS_BG[status?.toUpperCase()] ?? 'border-slate-700';

  return (
    <div className={`bg-slate-800 border ${border} rounded-lg p-3`}>
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-slate-200 text-sm font-semibold capitalize">
            {name.replace(/_/g, ' ')}
          </span>
        </div>
        <span className={`text-xs font-medium ${
          status === 'OK' || status === 'RUNNING' || status === 'HEALTHY'
            ? 'text-green-400'
            : status === 'WARNING' || status === 'DEGRADED'
            ? 'text-yellow-400'
            : status === 'ERROR' || status === 'UNHEALTHY'
            ? 'text-red-400'
            : 'text-slate-400'
        }`}>
          {status}
        </span>
      </div>
      {message && (
        <p className="text-slate-400 text-xs mt-1">{message}</p>
      )}
      {last_seen && (
        <div className="flex items-center gap-1 text-slate-600 text-[10px] mt-1.5">
          <Clock className="w-3 h-3" />
          Last seen {timeAgo(last_seen)}
        </div>
      )}
    </div>
  );
}
