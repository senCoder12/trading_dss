import { timeAgo } from '../../utils/formatters';

export function RefreshIndicator({ lastUpdated, error, loading, className = '' }) {
  return (
    <div className={`flex items-center gap-1.5 text-xs ${className}`}>
      {loading && !lastUpdated ? (
        <><span className="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse" /><span className="text-slate-500">loading</span></>
      ) : error ? (
        <><span className="w-1.5 h-1.5 rounded-full bg-red-400" /><span className="text-red-400">error</span></>
      ) : lastUpdated ? (
        <><span className="w-1.5 h-1.5 rounded-full bg-green-400" /><span className="text-slate-500">{timeAgo(lastUpdated)}</span></>
      ) : null}
    </div>
  );
}
