export function StaleDataBanner({ isStale, lastUpdated }) {
  if (!isStale) return null;

  const age = lastUpdated ? Math.floor((Date.now() - lastUpdated) / 60_000) : '?';

  return (
    <div className="bg-yellow-900/50 border border-yellow-700/60 rounded px-3 py-1.5 text-yellow-300 text-xs flex items-center gap-2 mx-4 mt-2">
      <span className="flex-shrink-0">&#x26A0;</span>
      <span>Connection lost. Showing data from {age} min ago. Reconnecting...</span>
    </div>
  );
}
