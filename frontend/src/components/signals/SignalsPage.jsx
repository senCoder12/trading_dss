import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';
import { Crosshair, ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';
import { usePolling } from '../../hooks/usePolling';
import { useApi } from '../../hooks/useApi';
import { useIndexMeta, indexLabel } from '../../hooks/useIndexMeta';
import { api } from '../../api/client';
import { SignalCard, ConfidenceMeter } from './SignalCard';
import { SignalHistory } from './SignalHistory';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { Tabs, useTabParam } from '../common/Tabs';
import { Segmented } from '../common/Segmented';
import { StatTile } from '../common/StatTile';
import { EmptyState } from '../common/EmptyState';
import { RowSkeleton, Skeleton } from '../common/Skeleton';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { Page, Toolbar } from '../layout/Page';
import { formatPercentage, formatPnL, timeAgo } from '../../utils/formatters';
import { REFRESH_INTERVALS, CHART_COLORS, TOOLTIP_STYLE, SIGNAL_TYPE_LABELS, SHORT_NAMES } from '../../utils/constants';

const PERIODS = [7, 14, 30, 90].map((d) => ({ value: d, label: `${d}D` }));

const TYPE_ICON = {
  BUY_CALL: { Icon: ArrowUpRight, cls: 'bg-up/10 text-up' },
  BUY_PUT: { Icon: ArrowDownRight, cls: 'bg-down/10 text-down' },
  NO_TRADE: { Icon: Minus, cls: 'bg-surface-3 text-ink-faint' },
};

// ── Live ─────────────────────────────────────────────────────────────────────

function LiveSignals() {
  const meta = useIndexMeta();
  const [filter, setFilter] = useState('all');
  const [selected, setSelected] = useState(null);
  const { data, loading, error, lastUpdated } = usePolling(api.getCurrentSignals, REFRESH_INTERVALS.signals);

  const all = [...(data?.signals ?? [])].sort(
    (a, b) => (a.signal_type === 'NO_TRADE') - (b.signal_type === 'NO_TRADE'),
  );
  const actionable = all.filter((s) => s.signal_type !== 'NO_TRADE');
  const rows = filter === 'actionable' ? actionable : all;
  const active = selected ?? rows[0]?.index_id;

  // Detail payload carries model votes + full reasoning; refresh it with the list.
  const { data: detail } = useApi(
    () => (active ? api.getSignalDetail(active) : Promise.resolve(null)),
    [active, lastUpdated],
  );
  const listItem = rows.find((s) => s.index_id === active);
  const shown = detail?.index_id === active ? { ...listItem, ...detail } : listItem;
  const name = (id) => SHORT_NAMES[id] ?? indexLabel(meta, id);

  return (
    <div className="space-y-4">
      <Toolbar right={<RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />}>
        <Segmented
          size="xs"
          value={filter}
          onChange={(v) => { setFilter(v); setSelected(null); }}
          options={[
            { value: 'all', label: `All (${all.length})` },
            { value: 'actionable', label: `Actionable (${actionable.length})` },
          ]}
        />
      </Toolbar>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_400px] gap-5 items-start">
        <Card padding={false}>
          {loading && !data ? (
            <RowSkeleton rows={5} />
          ) : !rows.length ? (
            <EmptyState
              icon={Crosshair}
              title={filter === 'actionable' ? 'No actionable setups' : 'No signals yet'}
              message={filter === 'actionable'
                ? 'Every index is currently a no-trade. Signals appear here when the model finds an edge.'
                : 'The signal engine has not produced output for this session.'}
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Index</th>
                    <th>Signal</th>
                    <th className="w-44">Confidence</th>
                    <th>Regime</th>
                    <th className="r">Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((s) => {
                    const { Icon, cls } = TYPE_ICON[s.signal_type] ?? TYPE_ICON.NO_TRADE;
                    const isSel = s.index_id === active;
                    return (
                      <tr
                        key={s.index_id}
                        onClick={() => setSelected(s.index_id)}
                        className={`cursor-pointer ${isSel ? '!bg-brand/[0.06]' : ''}`}
                      >
                        <td>
                          <div className="flex items-center gap-3">
                            <span className={`w-7 h-7 rounded-lg flex items-center justify-center ${cls}`}>
                              <Icon className="w-3.5 h-3.5" />
                            </span>
                            <span className={`font-medium ${isSel ? 'text-brand' : 'text-ink'}`}>{name(s.index_id)}</span>
                          </div>
                        </td>
                        <td><Badge variant={s.signal_type} size="xs">{SIGNAL_TYPE_LABELS[s.signal_type] ?? s.signal_type}</Badge></td>
                        <td><ConfidenceMeter score={s.confidence_score} /></td>
                        <td className="capitalize">{s.regime ? s.regime.replace(/_/g, ' ').toLowerCase() : '—'}</td>
                        <td className="r text-ink-faint">{timeAgo(s.generated_at)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card padding={false} className="lg:sticky lg:top-0">
          {shown ? (
            <SignalCard signal={shown} title={name(shown.index_id)} />
          ) : (
            <EmptyState icon={Crosshair} title="Select a signal" message="Pick a row to see levels, votes and reasoning." />
          )}
        </Card>
      </div>
    </div>
  );
}

// ── History ──────────────────────────────────────────────────────────────────

function HistoryTab() {
  const [days, setDays] = useState(7);
  const { data, loading } = useApi(() => api.getSignalHistory(days), [days]);
  const signals = data?.signals ?? [];

  return (
    <div className="space-y-4">
      <Toolbar right={data?.total != null && <span className="text-xs text-ink-faint">{data.total} signals</span>}>
        <Segmented size="xs" options={PERIODS} value={days} onChange={setDays} ariaLabel="Period" />
      </Toolbar>
      <Card padding={false}>
        <SignalHistory signals={signals} loading={loading} days={days} />
      </Card>
    </div>
  );
}

// ── Performance ──────────────────────────────────────────────────────────────

function PerformanceTab() {
  const [days, setDays] = useState(30);
  const { data: perf, loading } = useApi(() => api.getPerformance(days), [days]);

  const confData = perf ? [
    { name: 'High', wr: perf.high_confidence_win_rate },
    { name: 'Medium', wr: perf.medium_confidence_win_rate },
    { name: 'Low', wr: perf.low_confidence_win_rate },
  ] : [];

  return (
    <div className="space-y-5">
      <Toolbar>
        <Segmented size="xs" options={PERIODS} value={days} onChange={setDays} ariaLabel="Period" />
      </Toolbar>

      {loading || !perf ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
          {Array.from({ length: 4 }, (_, i) => <Skeleton key={i} height="92px" className="rounded-xl" />)}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
            <StatTile
              label="Win rate"
              value={formatPercentage(perf.win_rate, false)}
              tone={perf.total_trades ? (perf.win_rate >= 50 ? 'up' : 'down') : 'default'}
              sub={`${perf.wins ?? 0} wins · ${perf.losses ?? 0} losses`}
            />
            <StatTile
              label="Trades"
              value={perf.total_trades}
              sub={`${perf.actionable_signals ?? 0} actionable of ${perf.total_signals ?? 0} signals`}
            />
            <StatTile
              label="Profit factor"
              value={perf.profit_factor ? perf.profit_factor.toFixed(2) : '—'}
              tone={perf.profit_factor ? (perf.profit_factor >= 1 ? 'up' : 'down') : 'default'}
              sub={`Total ${formatPnL(perf.total_pnl)}`}
            />
            <StatTile
              label="Max drawdown"
              value={formatPercentage(perf.max_drawdown, false)}
              tone={perf.max_drawdown ? 'down' : 'default'}
              sub={perf.sharpe_ratio ? `Sharpe ${perf.sharpe_ratio.toFixed(2)}` : 'Sharpe —'}
            />
          </div>

          <Card title="Win rate by confidence" subtitle="Does higher model confidence translate into better outcomes?">
            {perf.total_trades ? (
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={confData} margin={{ top: 8, right: 8, bottom: 0, left: -16 }} barCategoryGap="35%">
                    <CartesianGrid vertical={false} stroke={CHART_COLORS.grid} />
                    <XAxis dataKey="name" tick={{ fontSize: 12, fill: CHART_COLORS.text }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 11, fill: CHART_COLORS.text }} domain={[0, 100]} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                    <Tooltip {...TOOLTIP_STYLE} formatter={(v) => [`${v?.toFixed(1)}%`, 'Win rate']} />
                    <Bar dataKey="wr" radius={[6, 6, 0, 0]} isAnimationActive={false}>
                      {confData.map((_, i) => <Cell key={i} fill={[CHART_COLORS.green, CHART_COLORS.yellow, CHART_COLORS.blue][i]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState title="Not enough trades" message={perf.edge_comment ?? 'Statistics appear once trades have closed.'} className="py-8" />
            )}
          </Card>

          {perf.total_trades > 0 && perf.edge_comment && (
            <p className="text-xs text-ink-faint">{perf.edge_comment}</p>
          )}
        </>
      )}
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function SignalsPage() {
  const [tab, setTab] = useTabParam('live');

  // Keep scroll position sane when switching tabs
  useEffect(() => { document.querySelector('main')?.scrollTo({ top: 0 }); }, [tab]);

  return (
    <Page>
      <Tabs
        value={tab}
        onChange={setTab}
        tabs={[
          { id: 'live', label: 'Live' },
          { id: 'history', label: 'History' },
          { id: 'performance', label: 'Performance' },
        ]}
      />
      {tab === 'live' && <LiveSignals />}
      {tab === 'history' && <HistoryTab />}
      {tab === 'performance' && <PerformanceTab />}
    </Page>
  );
}
