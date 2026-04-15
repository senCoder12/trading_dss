import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { timeAgo } from '../../utils/formatters';

export default function StatusBar() {
  const { data: health, error, lastUpdated } = usePolling(api.getSystemHealth, 30_000);

  const overall = health?.overall_status ?? (error ? 'ERROR' : 'UNKNOWN');
  const dot = {
    OK: 'bg-green-400',
    WARNING: 'bg-yellow-400',
    ERROR: 'bg-red-400',
    UNKNOWN: 'bg-slate-500',
  }[overall] ?? 'bg-slate-500';

  return (
    <footer className="bg-slate-950 border-t border-slate-800 px-4 py-1 flex items-center justify-between text-[11px] text-slate-600 flex-shrink-0">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
          <span>{error ? 'API unreachable' : `API · ${overall}`}</span>
        </div>
        {lastUpdated && <span>Refreshed {timeAgo(lastUpdated)}</span>}
        {health?.db_size && <span>DB: {health.db_size}</span>}
      </div>

      <div className="hidden sm:flex items-center gap-4">
        {health?.components?.slice(0, 3).map((c) => (
          <span key={c.component}>
            {c.component}:{' '}
            <span
              className={
                c.status === 'OK' || c.status === 'RUNNING'
                  ? 'text-green-500'
                  : c.status === 'WARNING'
                  ? 'text-yellow-500'
                  : 'text-red-500'
              }
            >
              {c.status}
            </span>
          </span>
        ))}
      </div>
    </footer>
  );
}
