import { AlertTriangle, Newspaper, CheckCircle2 } from 'lucide-react';
import { usePolling } from '../../hooks/usePolling';
import { useDataStore } from '../../hooks/useDataStore';
import { api } from '../../api/client';
import { Badge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';
import { timeAgo } from '../../utils/formatters';
import { REFRESH_INTERVALS } from '../../utils/constants';

/** Merged anomalies + high-impact news, newest first. */
export function useQuickAlerts(limit = 6) {
  const { anomalies: { data: anomData } } = useDataStore();
  const { data: newsData } = usePolling(() => api.getNewsFeed(5, 'HIGH'), REFRESH_INTERVALS.news);

  const items = [
    ...(anomData?.anomalies ?? []).map((a) => ({
      kind: 'anomaly', id: `a-${a.id ?? a.timestamp}`, title: a.message ?? a.anomaly_type,
      meta: a.index_id, severity: a.severity, ts: a.timestamp,
    })),
    ...(newsData?.articles ?? []).map((n) => ({
      kind: 'news', id: `n-${n.id ?? n.title}`, title: n.title, url: n.url,
      meta: n.source, severity: n.impact_category, ts: n.published_at,
    })),
  ].sort((a, b) => new Date(b.ts) - new Date(a.ts)).slice(0, limit);

  return items;
}

export default function QuickAlerts({ items }) {
  if (!items.length) {
    return <EmptyState icon={CheckCircle2} title="All quiet" message="No active anomalies or high-impact news." />;
  }

  return (
    <ul className="divide-y divide-line">
      {items.map((it) => {
        const isAnomaly = it.kind === 'anomaly';
        const Icon = isAnomaly ? AlertTriangle : Newspaper;
        const Wrapper = it.url ? 'a' : 'div';
        return (
          <li key={it.id}>
            <Wrapper
              {...(it.url ? { href: it.url, target: '_blank', rel: 'noreferrer' } : {})}
              className="flex items-start gap-3 px-5 py-3 hover:bg-surface-2/60 transition-colors"
            >
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${isAnomaly ? 'bg-warn/10 text-warn' : 'bg-info/10 text-info'}`}>
                <Icon className="w-4 h-4" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-[13px] text-ink leading-snug line-clamp-2">{it.title}</p>
                <div className="flex items-center gap-2 mt-1 text-2xs text-ink-faint">
                  {it.severity && <Badge variant={it.severity} kind="severity" size="xs">{it.severity}</Badge>}
                  {it.meta && <span className="truncate">{it.meta}</span>}
                  <span className="text-ink-ghost">· {timeAgo(it.ts)}</span>
                </div>
              </div>
            </Wrapper>
          </li>
        );
      })}
    </ul>
  );
}
