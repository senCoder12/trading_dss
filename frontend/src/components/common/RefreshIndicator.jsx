import { timeAgo } from '../../utils/formatters';

export function RefreshIndicator({ lastUpdated, error, loading, className = '' }) {
  let dot = null;
  let text = null;
  if (loading && !lastUpdated) { dot = 'bg-warn animate-pulse'; text = 'Loading'; }
  else if (error) { dot = 'bg-down'; text = 'Update failed'; }
  else if (lastUpdated) { dot = 'bg-up'; text = timeAgo(lastUpdated); }
  if (!dot) return null;

  return (
    <div
      className={`flex items-center gap-1.5 text-2xs text-ink-faint whitespace-nowrap ${className}`}
      title={error ? String(error) : lastUpdated ? `Updated ${new Date(lastUpdated).toLocaleTimeString('en-IN')}` : undefined}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
      <span className={error ? 'text-down' : ''}>{text}</span>
    </div>
  );
}
