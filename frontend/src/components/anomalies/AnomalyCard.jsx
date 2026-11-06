import { useState } from 'react';
import { AlertTriangle, ChevronDown } from 'lucide-react';
import { Badge } from '../common/Badge';
import { timeAgo, formatISTDateTime } from '../../utils/formatters';

const SEV_TINT = {
  CRITICAL: 'bg-down/10 text-down',
  HIGH: 'bg-down/10 text-down',
  MEDIUM: 'bg-warn/10 text-warn',
  LOW: 'bg-info/10 text-info',
};

/** One anomaly row inside a divide-y list. Details collapse behind a toggle. */
export function AnomalyCard({ anomaly }) {
  const { index_id, anomaly_type, severity, category, message, details, timestamp, is_active } = anomaly;
  const [open, setOpen] = useState(false);
  const hasDetails = details && Object.keys(details).length > 0;
  const tint = SEV_TINT[severity?.toUpperCase()] ?? 'bg-surface-3 text-ink-faint';

  return (
    <li className="px-5 py-4 hover:bg-surface-2/50 transition-colors">
      <div className="flex items-start gap-3.5">
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${tint}`}>
          <AlertTriangle className="w-4 h-4" />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[13px] font-semibold text-ink">{index_id}</span>
            <span className="text-2xs text-ink-faint font-mono">{anomaly_type?.replace(/_/g, ' ').toLowerCase()}</span>
            {severity && <Badge variant={severity} kind="severity" size="xs">{severity}</Badge>}
            {category && <Badge tone="neutral" size="xs">{category}</Badge>}
            {is_active === false && <Badge tone="neutral" size="xs">Resolved</Badge>}
          </div>

          {message && <p className="text-[13px] text-ink-muted leading-snug mt-1">{message}</p>}

          {hasDetails && (
            <button
              onClick={() => setOpen((o) => !o)}
              className="mt-2 inline-flex items-center gap-1 text-2xs font-medium text-ink-faint hover:text-ink-muted transition-colors"
              aria-expanded={open}
            >
              Details
              <ChevronDown className={`w-3.5 h-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
            </button>
          )}

          {open && hasDetails && (
            <dl className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1.5 rounded-lg bg-canvas border border-line px-3.5 py-3 animate-fade-in">
              {Object.entries(details).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between gap-3 text-xs min-w-0">
                  <dt className="text-ink-faint capitalize truncate">{k.replace(/_/g, ' ')}</dt>
                  <dd className="text-ink-muted font-mono num truncate">
                    {typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : String(v)}
                  </dd>
                </div>
              ))}
            </dl>
          )}
        </div>

        <span className="text-2xs text-ink-faint whitespace-nowrap flex-shrink-0" title={formatISTDateTime(timestamp)}>
          {timeAgo(timestamp)}
        </span>
      </div>
    </li>
  );
}
