import { AlertTriangle, Newspaper } from 'lucide-react';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { EmptyState } from '../common/EmptyState';
import { timeAgo } from '../../utils/formatters';
import { REFRESH_INTERVALS, SEVERITY_COLORS } from '../../utils/constants';

export default function QuickAlerts() {
  const { data: anomData, error: anomErr, lastUpdated: anomTs } = usePolling(
    api.getActiveAnomalies, REFRESH_INTERVALS.anomalies,
  );
  const { data: newsData, error: newsErr, lastUpdated: newsTs } = usePolling(
    () => api.getNewsFeed(5, 'HIGH'), REFRESH_INTERVALS.news,
  );

  const anomalies = anomData?.anomalies ?? [];
  const articles = newsData?.articles ?? [];
  const total = anomalies.length + articles.length;

  return (
    <Card
      title={`Alerts & News ${total > 0 ? `(${total})` : ''}`}
      padding={false}
      actions={
        <RefreshIndicator
          lastUpdated={anomTs ?? newsTs}
          error={anomErr ?? newsErr}
        />
      }
    >
      {total === 0 ? (
        <EmptyState icon={AlertTriangle} title="No active alerts" className="py-6" />
      ) : (
        <div className="divide-y divide-slate-700/50 max-h-56 overflow-y-auto">
          {anomalies.slice(0, 5).map((a) => (
            <div key={a.id} className="flex items-start gap-2.5 px-3 py-2 hover:bg-slate-700/30">
              <AlertTriangle
                className={`w-3.5 h-3.5 flex-shrink-0 mt-0.5 ${SEVERITY_COLORS[a.severity] ?? 'text-slate-400'}`}
              />
              <div className="min-w-0">
                <div className="text-slate-200 text-xs leading-snug truncate">
                  {a.message ?? a.anomaly_type}
                </div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className="text-slate-600 text-[10px]">{a.index_id}</span>
                  <Badge variant={a.severity} size="xs">{a.severity}</Badge>
                  <span className="text-slate-600 text-[10px]">{timeAgo(a.timestamp)}</span>
                </div>
              </div>
            </div>
          ))}

          {articles.slice(0, 5).map((n) => (
            <div key={n.id} className="flex items-start gap-2.5 px-3 py-2 hover:bg-slate-700/30">
              <Newspaper className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-blue-400" />
              <div className="min-w-0">
                <div className="text-slate-200 text-xs leading-snug line-clamp-1">{n.title}</div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  {n.source && <span className="text-slate-600 text-[10px]">{n.source}</span>}
                  {n.impact_category && (
                    <Badge variant={n.impact_category} size="xs">{n.impact_category}</Badge>
                  )}
                  <span className="text-slate-600 text-[10px]">{timeAgo(n.published_at)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
