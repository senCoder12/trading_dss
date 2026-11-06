import { CheckCircle2, AlertTriangle, XCircle, HelpCircle, Clock } from 'lucide-react';
import { Badge } from '../common/Badge';
import { timeAgo } from '../../utils/formatters';

const STATUS = {
  OK: { Icon: CheckCircle2, tint: 'bg-up/10 text-up' },
  RUNNING: { Icon: CheckCircle2, tint: 'bg-up/10 text-up' },
  HEALTHY: { Icon: CheckCircle2, tint: 'bg-up/10 text-up' },
  WARNING: { Icon: AlertTriangle, tint: 'bg-warn/10 text-warn' },
  DEGRADED: { Icon: AlertTriangle, tint: 'bg-warn/10 text-warn' },
  ERROR: { Icon: XCircle, tint: 'bg-down/10 text-down' },
  UNHEALTHY: { Icon: XCircle, tint: 'bg-down/10 text-down' },
  UNKNOWN: { Icon: HelpCircle, tint: 'bg-surface-3 text-ink-faint' },
};

const BAD = new Set(['ERROR', 'UNHEALTHY']);
const WARN = new Set(['WARNING', 'DEGRADED']);

export function ComponentHealth({ component }) {
  const { component: name, status, last_seen, message } = component;
  const key = status?.toUpperCase() ?? 'UNKNOWN';
  const { Icon, tint } = STATUS[key] ?? STATUS.UNKNOWN;
  const border = BAD.has(key) ? 'border-down/30' : WARN.has(key) ? 'border-warn/30' : '';
  const pretty = name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className={`card p-4 flex items-start gap-3 min-w-0 ${border}`}>
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${tint}`}>
        <Icon className="w-4 h-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[13px] font-medium text-ink truncate" title={name}>{pretty}</span>
          <Badge variant={key} size="xs">{status ?? 'UNKNOWN'}</Badge>
        </div>
        {message && <p className="text-xs text-ink-faint mt-1 leading-snug line-clamp-2" title={message}>{message}</p>}
        {last_seen && (
          <div className="flex items-center gap-1 text-2xs text-ink-ghost mt-1.5">
            <Clock className="w-3 h-3" />
            Last seen {timeAgo(last_seen)}
          </div>
        )}
      </div>
    </div>
  );
}
