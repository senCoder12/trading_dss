import { useState } from 'react';
import { Power, PowerOff, AlertTriangle, Store, CalendarClock, Timer, Clock3, Server, Database } from 'lucide-react';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { StatTile } from '../common/StatTile';
import { Tabs, useTabParam } from '../common/Tabs';
import { EmptyState } from '../common/EmptyState';
import { Skeleton } from '../common/Skeleton';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { Page, Toolbar } from '../layout/Page';
import { ComponentHealth } from './ComponentHealth';
import { REFRESH_INTERVALS } from '../../utils/constants';
import { formatISTDateTime } from '../../utils/formatters';

function formatAge(sec) {
  if (sec == null) return '—';
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
  return `${Math.floor(sec / 86400)}d ${Math.floor((sec % 86400) / 3600)}h`;
}

function freshness(sec, marketOpen) {
  if (sec == null) return { label: 'Unknown', tone: 'neutral', cls: 'text-ink-faint' };
  if (sec < 60) return { label: 'Fresh', tone: 'success', cls: 'text-up' };
  if (sec < 300) return { label: 'Lagging', tone: 'warning', cls: 'text-warn' };
  // Outside market hours old data is expected — don't paint it as an error.
  return marketOpen
    ? { label: 'Stale', tone: 'danger', cls: 'text-down' }
    : { label: 'Idle', tone: 'neutral', cls: 'text-ink-muted' };
}

const titleCase = (s) => s?.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

function KillSwitchCard({ status, onChanged }) {
  const [reason, setReason] = useState('');
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const active = status?.kill_switch_active ?? false;

  async function run(fn) {
    setBusy(true);
    setErr(null);
    try {
      await fn();
      onChanged?.();
    } catch (e) {
      setErr(e.message || 'Request failed');
    } finally {
      setBusy(false);
    }
  }

  const activate = () => run(async () => {
    await api.activateKillSwitch(reason.trim() || 'Manual activation');
    setReason('');
    setConfirming(false);
  });
  const deactivate = () => run(async () => {
    await api.deactivateKillSwitch();
    setConfirming(false);
  });

  return (
    <section className={`card p-5 ${active ? 'border-down/40 bg-down/[0.05]' : ''}`}>
      <div className="flex flex-wrap items-center gap-4">
        <div className={`w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 ${active ? 'bg-down/15 text-down' : 'bg-up/10 text-up'}`}>
          {active ? <PowerOff className="w-5 h-5" /> : <Power className="w-5 h-5" />}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-ink">Kill switch</h3>
            <Badge tone={active ? 'danger' : 'success'} size="xs" dot>{active ? 'Active' : 'Inactive'}</Badge>
          </div>
          <p className={`text-[13px] mt-0.5 ${active ? 'text-down' : 'text-ink-faint'}`}>
            {active
              ? `All trading halted — ${status?.kill_switch_reason ?? 'no reason given'}`
              : 'Trading allowed. Activate to halt all new signals and entries immediately.'}
          </p>
          {err && <p className="text-xs text-down mt-1">{err}</p>}
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {active ? (
            <button onClick={deactivate} disabled={busy} className="btn btn-success">
              <Power className="w-4 h-4" />
              {busy ? 'Working…' : 'Deactivate'}
            </button>
          ) : confirming ? (
            <>
              <input
                autoFocus
                className="input h-8 w-56"
                placeholder="Reason (required)"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && reason.trim()) activate();
                  if (e.key === 'Escape') { setConfirming(false); setReason(''); }
                }}
              />
              <button onClick={activate} disabled={busy || !reason.trim()} className="btn btn-danger">
                {busy ? 'Activating…' : 'Confirm'}
              </button>
              <button onClick={() => { setConfirming(false); setReason(''); }} className="btn btn-ghost">
                Cancel
              </button>
            </>
          ) : (
            <button onClick={() => setConfirming(true)} className="btn btn-danger">
              <AlertTriangle className="w-4 h-4" />
              Activate kill switch
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

function ComponentsTab({ health, loading, error, lastUpdated }) {
  const components = health?.components ?? [];
  return (
    <div className="space-y-4">
      <Toolbar right={<RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />}>
        {health?.overall_status && (
          <div className="flex items-center gap-2 text-[13px]">
            <span className="text-ink-faint">Overall</span>
            <Badge variant={health.overall_status} dot>{health.overall_status}</Badge>
          </div>
        )}
        {health?.db_size && (
          <div className="flex items-center gap-1.5 text-[13px] text-ink-faint">
            <Database className="w-3.5 h-3.5" />
            Database <span className="text-ink-muted num">{health.db_size}</span>
          </div>
        )}
      </Toolbar>

      {loading && !health ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }, (_, i) => <Skeleton key={i} height="84px" className="rounded-xl" />)}
        </div>
      ) : components.length ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {components.map((c) => <ComponentHealth key={c.component} component={c} />)}
        </div>
      ) : (
        <Card><EmptyState icon={Server} title="No component data" message="Components report in once the engine is running." /></Card>
      )}
    </div>
  );
}

function FreshnessTab({ rows, marketOpen }) {
  const stale = (rows ?? []).filter((r) => r.age_seconds >= 300).length;
  const subtitle = rows?.length
    ? `${stale} of ${rows.length} tables older than 5 min${!marketOpen && stale ? ' — expected while the market is closed' : ''}`
    : 'Age of the newest row in each table';
  return (
    <Card
      title="Data freshness"
      subtitle={subtitle}
      padding={false}
      actions={
        <div className="hidden sm:flex items-center gap-3 text-2xs text-ink-faint">
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-up" />&lt; 1m</span>
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-warn" />&lt; 5m</span>
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-down" />older</span>
        </div>
      }
    >
      {!rows?.length ? (
        <EmptyState icon={Database} title="No freshness data" />
      ) : (
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th>Table</th>
                <th>Latest row</th>
                <th className="r">Age</th>
                <th className="r">Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const f = freshness(r.age_seconds, marketOpen);
                return (
                  <tr key={r.table}>
                    <td className="font-mono text-xs text-ink">{r.table}</td>
                    <td className="num">{r.latest_timestamp ? formatISTDateTime(r.latest_timestamp) : '—'}</td>
                    <td className={`r num font-medium ${f.cls}`}>{formatAge(r.age_seconds)}</td>
                    <td className="r"><Badge tone={f.tone} size="xs">{f.label}</Badge></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

export default function SystemPage() {
  const [tab, setTab] = useTabParam('components');
  const { data: health, loading: healthLoading, lastUpdated: healthTs, error: healthErr } = usePolling(
    api.getSystemHealth, REFRESH_INTERVALS.system,
  );
  const { data: status, refetch: refetchStatus } = usePolling(api.getSystemStatus, REFRESH_INTERVALS.system);

  const up = status?.uptime_seconds;
  const isOpen = status?.market_status === 'OPEN';

  return (
    <Page>
      <KillSwitchCard status={status} onChanged={refetchStatus} />

      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <StatTile
          label="Market"
          icon={Store}
          value={status ? (isOpen ? 'Open' : titleCase(status.market_status?.toLowerCase()) ?? '—') : '—'}
          tone={isOpen ? 'up' : 'default'}
          sub={status ? 'NSE / BSE cash & F&O' : null}
        />
        <StatTile
          label="Session"
          icon={CalendarClock}
          value={status?.market_session ? titleCase(status.market_session) : '—'}
        />
        <StatTile
          label="Uptime"
          icon={Timer}
          value={up != null ? `${Math.floor(up / 3600)}h ${Math.floor((up % 3600) / 60)}m` : '—'}
          sub={up != null ? 'Since last engine start' : null}
        />
        <StatTile
          label="Next event"
          icon={Clock3}
          value={status?.time_remaining ?? '—'}
          sub={status?.next_event ? `until ${status.next_event.replace(/_/g, ' ')}` : null}
        />
      </div>

      <Tabs
        value={tab}
        onChange={setTab}
        tabs={[
          { id: 'components', label: 'Components', count: health?.components?.length || undefined },
          { id: 'freshness', label: 'Data freshness' },
        ]}
      />

      {tab === 'freshness'
        ? <FreshnessTab rows={status?.data_freshness} marketOpen={isOpen} />
        : <ComponentsTab health={health} loading={healthLoading} error={healthErr} lastUpdated={healthTs} />}
    </Page>
  );
}
