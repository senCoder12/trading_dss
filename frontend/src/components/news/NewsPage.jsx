import { useState, useEffect } from 'react';
import { Newspaper, Smile, AlertOctagon, Flame, BarChart3 } from 'lucide-react';
import { useApi } from '../../hooks/useApi';
import { usePolling } from '../../hooks/usePolling';
import { useIndexMeta, indexLabel } from '../../hooks/useIndexMeta';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { StatTile } from '../common/StatTile';
import { Tabs, useTabParam } from '../common/Tabs';
import { Segmented } from '../common/Segmented';
import { EmptyState } from '../common/EmptyState';
import { RowSkeleton } from '../common/Skeleton';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { Page, Toolbar } from '../layout/Page';
import { NewsCard } from './NewsCard';
import { REFRESH_INTERVALS, SENTIMENT_LABELS } from '../../utils/constants';

// Minimum-severity filter, most inclusive first. Default hides LOW/NOISE so the
// feed opens on articles that can actually move a position.
const SEVERITY_OPTIONS = [
  { value: 'ALL', label: 'All', title: 'Everything, including noise' },
  { value: 'LOW', label: 'Low+', title: 'Hide noise' },
  { value: 'MEDIUM', label: 'Medium+' },
  { value: 'HIGH', label: 'High+' },
  { value: 'CRITICAL', label: 'Critical' },
];
const LIMIT_OPTIONS = [10, 20, 50];

/** Diverging bar for a −1..+1 score, centred at zero. */
function SentimentBar({ value }) {
  const v = Math.max(-1, Math.min(1, value ?? 0));
  const width = Math.abs(v) * 50;
  const pos = v >= 0;
  return (
    <div className="flex items-center gap-3 justify-end">
      <div className="relative w-40 h-1.5 rounded-full bg-surface-3">
        <span className="absolute left-1/2 -top-0.5 -bottom-0.5 w-px bg-line-strong" />
        <span
          className={`absolute top-0 bottom-0 rounded-full ${pos ? 'bg-up' : 'bg-down'}`}
          style={pos ? { left: '50%', width: `${width}%` } : { right: '50%', width: `${width}%` }}
        />
      </div>
      <span className={`w-12 text-right num text-[13px] ${v > 0.05 ? 'text-up' : v < -0.05 ? 'text-down' : 'text-ink-muted'}`}>
        {v > 0 ? '+' : ''}{v.toFixed(2)}
      </span>
    </div>
  );
}

function FeedTab() {
  const [severity, setSeverity] = useState('MEDIUM');
  const [limit, setLimit] = useState(20);
  const { data: feed, loading, error, refetch } = useApi(
    () => api.getNewsFeed(limit, severity === 'ALL' ? null : severity), [limit, severity],
  );
  const [lastUpdated, setLastUpdated] = useState(null);
  const articles = feed?.articles ?? [];

  // Track when the feed last arrived for the refresh indicator
  useEffect(() => { if (feed) setLastUpdated(Date.now()); }, [feed]);

  return (
    <div className="space-y-4">
      <Toolbar
        right={
          <>
            <RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />
            <button onClick={refetch} className="btn btn-ghost h-7 text-xs">Refresh</button>
          </>
        }
      >
        <Segmented size="xs" ariaLabel="Minimum severity" options={SEVERITY_OPTIONS} value={severity} onChange={setSeverity} />
        <div className="flex items-center gap-2">
          <span className="label">Show</span>
          <Segmented size="xs" ariaLabel="Articles to show" options={LIMIT_OPTIONS} value={limit} onChange={setLimit} />
        </div>
      </Toolbar>

      <Card padding={false}>
        {loading && !feed ? (
          <RowSkeleton rows={6} />
        ) : !articles.length ? (
          <EmptyState icon={Newspaper} title="No articles found" message="Try a lower severity filter or check back later." />
        ) : (
          <ul className={`divide-y divide-line ${loading ? 'opacity-60 transition-opacity' : ''}`}>
            {articles.map((a) => <NewsCard key={a.id ?? a.title} article={a} />)}
          </ul>
        )}
      </Card>
    </div>
  );
}

function ByIndexTab({ summary }) {
  const meta = useIndexMeta();
  const rows = summary?.most_impacted_indices ?? [];
  const max = Math.max(1, ...rows.map((r) => r.article_count ?? 0));

  return (
    <Card title="Most impacted indices" subtitle="Article volume and average sentiment per index" padding={false}>
      {!rows.length ? (
        <EmptyState icon={BarChart3} title="No index coverage yet" message="Index mappings appear once articles are classified." />
      ) : (
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th>Index</th>
                <th>Articles</th>
                <th className="r">Avg. sentiment</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.index_id}>
                  <td>
                    <div className="text-ink font-medium">{indexLabel(meta, r.index_id)}</div>
                    <div className="text-2xs text-ink-ghost font-mono">{r.index_id}</div>
                  </td>
                  <td>
                    <div className="flex items-center gap-3">
                      <span className="w-10 num text-ink">{r.article_count}</span>
                      <div className="w-32 h-1.5 rounded-full bg-surface-3 overflow-hidden">
                        <div className="h-full rounded-full bg-brand/70" style={{ width: `${(r.article_count / max) * 100}%` }} />
                      </div>
                    </div>
                  </td>
                  <td className="r"><SentimentBar value={r.avg_sentiment} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

export default function NewsPage() {
  const [tab, setTab] = useTabParam('feed');
  const { data: summary } = usePolling(api.getNewsSummary, REFRESH_INTERVALS.news);

  const sentLabel = summary?.overall_sentiment_label ?? 'NEUTRAL';
  const sentInfo = SENTIMENT_LABELS[sentLabel] ?? SENTIMENT_LABELS.NEUTRAL;
  const score = summary?.overall_sentiment;
  const sentTone = sentLabel.includes('BULL') ? 'up' : sentLabel.includes('BEAR') ? 'down' : 'default';
  const critical = summary?.by_severity?.CRITICAL ?? summary?.critical_count ?? 0;
  const high = summary?.by_severity?.HIGH ?? 0;

  return (
    <Page>
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <StatTile label="Articles today" icon={Newspaper} value={summary ? summary.total_articles ?? 0 : '—'} sub={summary ? 'Across all sources' : null} />
        <StatTile
          label="Overall sentiment"
          icon={Smile}
          value={summary ? sentInfo.label : '—'}
          tone={sentTone}
          sub={score != null ? `${score > 0 ? '+' : ''}${score.toFixed(3)} score` : null}
        />
        <StatTile
          label="Critical"
          icon={AlertOctagon}
          value={summary ? critical : '—'}
          tone={critical > 0 ? 'down' : 'default'}
          sub={summary ? (critical > 0 ? 'Needs attention' : 'None today') : null}
        />
        <StatTile
          label="High impact"
          icon={Flame}
          value={summary ? high : '—'}
          tone={high > 0 ? 'warn' : 'default'}
          sub={summary ? 'High-severity articles' : null}
        />
      </div>

      <Tabs
        value={tab}
        onChange={setTab}
        tabs={[
          { id: 'feed', label: 'Feed' },
          { id: 'by-index', label: 'By index', count: summary?.most_impacted_indices?.length || undefined },
        ]}
      />

      {tab === 'by-index' ? <ByIndexTab summary={summary} /> : <FeedTab />}
    </Page>
  );
}
