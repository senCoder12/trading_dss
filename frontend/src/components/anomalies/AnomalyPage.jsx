import { useState } from 'react';
import { BellRing, AlertOctagon, AlertTriangle, Info, ShieldCheck, PieChart } from 'lucide-react';
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
import { AnomalyCard } from './AnomalyCard';
import { REFRESH_INTERVALS } from '../../utils/constants';

const SEV_OPTS = [
  { value: 'ALL', label: 'All' },
  { value: 'HIGH', label: 'High' },
  { value: 'MEDIUM', label: 'Medium+' },
  { value: 'LOW', label: 'Low+' },
];
const SEV_ORDER = { CRITICAL: -1, HIGH: 0, MEDIUM: 1, LOW: 2 };

/** Horizontal bar list: label, count, proportional bar. */
function BarList({ entries, labelFor = (k) => k, barClass = 'bg-brand/70' }) {
  const max = Math.max(1, ...entries.map(([, v]) => v));
  return (
    <ul className="space-y-3.5">
      {entries.map(([k, v]) => (
        <li key={k}>
          <div className="flex items-center justify-between text-[13px] mb-1.5">
            <span className="text-ink-muted truncate">{labelFor(k)}</span>
            <span className="text-ink font-medium num">{v}</span>
          </div>
          <div className="h-1.5 rounded-full bg-surface-3 overflow-hidden">
            <div className={`h-full rounded-full ${barClass}`} style={{ width: `${(v / max) * 100}%` }} />
          </div>
        </li>
      ))}
    </ul>
  );
}

function ActiveTab({ data, loading, error, lastUpdated }) {
  const [minSev, setMinSev] = useState('ALL');
  const all = data?.anomalies ?? [];
  const anomalies = minSev === 'ALL'
    ? all
    : all.filter((a) => (SEV_ORDER[a.severity] ?? 2) <= (SEV_ORDER[minSev] ?? 2));

  return (
    <div className="space-y-4">
      <Toolbar right={<RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />}>
        <Segmented size="xs" ariaLabel="Minimum severity" options={SEV_OPTS} value={minSev} onChange={setMinSev} />
      </Toolbar>

      <Card padding={false}>
        {loading && !data ? (
          <RowSkeleton rows={4} />
        ) : !anomalies.length ? (
          <EmptyState
            icon={ShieldCheck}
            title="All clear — no anomalies detected"
            message={all.length
              ? 'Nothing matches this severity filter.'
              : 'Volume spikes, OI shifts and price dislocations will show up here as soon as the detector flags them.'}
          />
        ) : (
          <ul className="divide-y divide-line">
            {anomalies.map((a) => (
              <AnomalyCard key={a.id ?? `${a.index_id}-${a.timestamp}`} anomaly={a} />
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

function BreakdownTab({ dashboard }) {
  const meta = useIndexMeta();
  const byCat = Object.entries(dashboard?.by_category ?? {}).sort((a, b) => b[1] - a[1]);
  const byIdx = Object.entries(dashboard?.by_index ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <Card title="By category" subtitle="Active anomalies per detector category">
        {byCat.length
          ? <BarList entries={byCat} labelFor={(k) => k.replace(/_/g, ' ')} barClass="bg-warn/70" />
          : <EmptyState icon={PieChart} title="No categories to show" message="Breakdown fills in when anomalies are active." className="py-8" />}
      </Card>
      <Card title="By index" subtitle="Where anomalies are concentrated">
        {byIdx.length
          ? <BarList entries={byIdx} labelFor={(k) => indexLabel(meta, k)} />
          : <EmptyState icon={PieChart} title="No indices affected" message="Every tracked index is behaving normally." className="py-8" />}
      </Card>
    </div>
  );
}

export default function AnomalyPage() {
  const [tab, setTab] = useTabParam('active');
  const { data: dashboard } = usePolling(api.getAnomalyDashboard, REFRESH_INTERVALS.anomalies);
  const { data, loading, error, lastUpdated } = usePolling(api.getActiveAnomalies, REFRESH_INTERVALS.anomalies);

  const d = dashboard ?? {};
  const has = dashboard != null;

  return (
    <Page>
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <StatTile
          label="Active"
          icon={BellRing}
          value={has ? d.total_active ?? 0 : '—'}
          tone={d.total_active > 0 ? 'brand' : 'default'}
          sub={has ? (d.total_active > 0 ? 'Open anomalies' : 'Nothing flagged') : null}
        />
        <StatTile label="High severity" icon={AlertOctagon} value={has ? d.high_severity ?? 0 : '—'} tone={d.high_severity > 0 ? 'down' : 'default'} />
        <StatTile label="Medium" icon={AlertTriangle} value={has ? d.medium_severity ?? 0 : '—'} tone={d.medium_severity > 0 ? 'warn' : 'default'} />
        <StatTile label="Low" icon={Info} value={has ? d.low_severity ?? 0 : '—'} tone={d.low_severity > 0 ? 'info' : 'default'} />
      </div>

      <Tabs
        value={tab}
        onChange={setTab}
        tabs={[
          { id: 'active', label: 'Active', count: (data?.anomalies?.length) || undefined },
          { id: 'breakdown', label: 'Breakdown' },
        ]}
      />

      {tab === 'breakdown'
        ? <BreakdownTab dashboard={dashboard} />
        : <ActiveTab data={data} loading={loading} error={error} lastUpdated={lastUpdated} />}
    </Page>
  );
}
