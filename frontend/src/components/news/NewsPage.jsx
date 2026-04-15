import { useState } from 'react';
import { useApi } from '../../hooks/useApi';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Spinner } from '../common/Spinner';
import { EmptyState } from '../common/EmptyState';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { NewsCard } from './NewsCard';
import { Badge } from '../common/Badge';
import { REFRESH_INTERVALS, SENTIMENT_LABELS } from '../../utils/constants';
import { Newspaper } from 'lucide-react';

const SEVERITY_OPTIONS = [null, 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];
const LIMIT_OPTIONS = [10, 20, 50];

export default function NewsPage() {
  const [severity, setSeverity] = useState(null);
  const [limit, setLimit] = useState(20);

  const { data: summary, lastUpdated: sumTs } = usePolling(
    api.getNewsSummary, REFRESH_INTERVALS.news,
  );
  const { data: feed, loading, error, lastUpdated: feedTs, refetch } = useApi(
    () => api.getNewsFeed(limit, severity), [limit, severity],
  );

  const articles = feed?.articles ?? [];
  const sentLabel = summary?.overall_sentiment_label ?? 'NEUTRAL';
  const sentInfo = SENTIMENT_LABELS[sentLabel] ?? SENTIMENT_LABELS.NEUTRAL;

  return (
    <div className="p-4 space-y-4">
      {/* Summary bar */}
      {summary && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 text-center">
            <div className="text-slate-100 font-bold text-lg">{summary.total_articles}</div>
            <div className="text-slate-500 text-xs">Articles Today</div>
          </div>
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 text-center">
            <div className={`font-bold text-lg ${sentInfo.color}`}>{sentInfo.label}</div>
            <div className="text-slate-500 text-xs">Overall Sentiment</div>
          </div>
          {summary.critical_count > 0 && (
            <div className="bg-slate-800 border border-red-700/40 rounded-lg p-3 text-center">
              <div className="text-red-400 font-bold text-lg">{summary.critical_count}</div>
              <div className="text-slate-500 text-xs">Critical</div>
            </div>
          )}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-3">
            <div className="text-slate-400 text-xs mb-1">By Severity</div>
            <div className="flex flex-wrap gap-1">
              {Object.entries(summary.by_severity ?? {}).map(([sev, cnt]) => (
                <span key={sev} className="text-xs">
                  <Badge variant={sev} size="xs">{sev}</Badge>
                  <span className="text-slate-400 ml-1">{cnt}</span>
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Feed controls */}
      <Card title="News Feed" padding={false}
        actions={<RefreshIndicator lastUpdated={feedTs ?? sumTs} error={error} loading={loading} />}
      >
        <div className="px-4 py-2.5 border-b border-slate-700 flex flex-wrap items-center gap-3">
          {/* Severity filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 text-xs">Min:</span>
            {SEVERITY_OPTIONS.map((s) => (
              <button
                key={String(s)}
                onClick={() => setSeverity(s)}
                className={`px-2.5 py-0.5 rounded text-xs font-medium transition-colors ${
                  severity === s
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
                }`}
              >
                {s ?? 'All'}
              </button>
            ))}
          </div>

          {/* Limit */}
          <div className="flex items-center gap-1.5 ml-auto">
            <span className="text-slate-500 text-xs">Show:</span>
            {LIMIT_OPTIONS.map((l) => (
              <button
                key={l}
                onClick={() => setLimit(l)}
                className={`px-2.5 py-0.5 rounded text-xs font-medium transition-colors ${
                  limit === l
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
                }`}
              >
                {l}
              </button>
            ))}
          </div>
        </div>

        <div className="p-4">
          {loading ? (
            <div className="flex justify-center p-8"><Spinner /></div>
          ) : !articles.length ? (
            <EmptyState icon={Newspaper} title="No articles found" message="Try changing the filter or check back later" />
          ) : (
            <div className="space-y-3">
              {articles.map((a) => (
                <NewsCard key={a.id ?? a.title} article={a} />
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
