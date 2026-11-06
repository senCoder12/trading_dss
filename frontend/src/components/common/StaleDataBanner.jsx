import { WifiOff } from 'lucide-react';

export function StaleDataBanner({ isStale, lastUpdated }) {
  if (!isStale) return null;
  const age = lastUpdated ? Math.floor((Date.now() - lastUpdated) / 60_000) : '?';

  return (
    <div className="flex items-center gap-2.5 rounded-lg border border-warn/30 bg-warn/10 px-4 py-2.5 text-xs text-warn">
      <WifiOff className="w-4 h-4 flex-shrink-0" />
      <span>Connection lost — showing data from {age} min ago. Reconnecting…</span>
    </div>
  );
}
