import { useState } from 'react';
import { Wallet, TrendingUp, Briefcase, Target, Download } from 'lucide-react';
import { usePolling } from '../../hooks/usePolling';
import { useApi } from '../../hooks/useApi';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { ErrorBoundary } from '../common/ErrorBoundary';
import { StatTile, signTone } from '../common/StatTile';
import { Tabs, useTabParam } from '../common/Tabs';
import { Segmented } from '../common/Segmented';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { RowSkeleton, Skeleton } from '../common/Skeleton';
import { Page, Toolbar } from '../layout/Page';
import { EquityCurve } from './EquityCurve';
import { PositionsList } from './PositionsList';
import { TradeHistory } from './TradeHistory';
import { PerformanceStats } from './PerformanceStats';
import { formatCurrency, formatPercentage, formatPnL } from '../../utils/formatters';
import { REFRESH_INTERVALS } from '../../utils/constants';

const PERIODS = [7, 14, 30, 90].map((d) => ({ value: d, label: `${d}D` }));
const EQ_PERIODS = [7, 30, 90].map((d) => ({ value: d, label: `${d}D` }));

function OverviewTab({ portfolio }) {
  const [days, setDays] = useState(30);
  const { data, loading } = useApi(() => api.getEquityHistory(days), [days]);

  return (
    <Card
      title="Equity curve"
      subtitle={portfolio?.initial_capital ? `Dashed line = starting capital ${formatCurrency(portfolio.initial_capital, 0)}` : undefined}
      actions={<Segmented size="xs" options={EQ_PERIODS} value={days} onChange={setDays} ariaLabel="Period" />}
    >
      {loading ? <Skeleton height="260px" /> : (
        <EquityCurve history={data?.history ?? []} initial={portfolio?.initial_capital ?? 100_000} />
      )}
    </Card>
  );
}

function TradesTab() {
  const [days, setDays] = useState(7);
  const { data, loading } = useApi(() => api.getTrades(days), [days]);
  const trades = data?.trades ?? [];

  return (
    <div className="space-y-4">
      <Toolbar
        right={
          <>
            {data?.total != null && <span className="text-xs text-ink-faint">{data.total} trades</span>}
            <button
              onClick={() => window.open(`/api/portfolio/trades/export?days=${days}`, '_blank')}
              disabled={!trades.length}
              className="btn btn-secondary h-7 text-xs"
            >
              <Download className="w-3.5 h-3.5" /> Export CSV
            </button>
          </>
        }
      >
        <Segmented size="xs" options={PERIODS} value={days} onChange={setDays} ariaLabel="Period" />
      </Toolbar>
      <Card padding={false}>
        {loading ? <RowSkeleton rows={5} /> : <TradeHistory trades={trades} days={days} />}
      </Card>
    </div>
  );
}

function StatsTab() {
  const [days, setDays] = useState(30);
  const { data, loading } = useApi(() => api.getPerformance(days), [days]);
  return (
    <div className="space-y-4">
      <Toolbar>
        <Segmented size="xs" options={PERIODS} value={days} onChange={setDays} ariaLabel="Period" />
      </Toolbar>
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {Array.from({ length: 3 }, (_, i) => <Skeleton key={i} height="230px" className="rounded-xl" />)}
        </div>
      ) : (
        <PerformanceStats stats={data} />
      )}
    </div>
  );
}

export default function PortfolioPage() {
  const [tab, setTab] = useTabParam('overview');
  const { data: portfolio, loading, error, lastUpdated } = usePolling(api.getPortfolio, REFRESH_INTERVALS.portfolio);
  const positions = portfolio?.open_positions ?? [];

  return (
    <Page>
      {/* KPIs — always visible, one number each */}
      {loading && !portfolio ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
          {Array.from({ length: 4 }, (_, i) => <Skeleton key={i} height="92px" className="rounded-xl" />)}
        </div>
      ) : (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
          <StatTile
            label="Capital"
            icon={Wallet}
            value={formatCurrency(portfolio?.capital, 0)}
            sub={`${formatPercentage(portfolio?.total_return_pct)} all-time`}
            subTone={signTone(portfolio?.total_return_pct)}
          />
          <StatTile
            label="Today's P&L"
            icon={TrendingUp}
            value={formatPnL(portfolio?.today_pnl)}
            tone={signTone(portfolio?.today_pnl)}
            sub={`${formatPercentage(portfolio?.today_pnl_pct)} · ${portfolio?.today_trades ?? 0} trades`}
          />
          <StatTile label="Open positions" icon={Briefcase} value={positions.length} sub="Live exposure" />
          <StatTile
            label="Win rate"
            icon={Target}
            value={portfolio?.overall_win_rate != null ? `${portfolio.overall_win_rate.toFixed(0)}%` : '—'}
            sub={`${portfolio?.total_closed_trades ?? 0} closed trades`}
          />
        </div>
      )}

      <Tabs
        value={tab}
        onChange={setTab}
        right={<RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />}
        tabs={[
          { id: 'overview', label: 'Equity' },
          { id: 'positions', label: 'Positions', count: positions.length || undefined },
          { id: 'trades', label: 'Trade log' },
          { id: 'stats', label: 'Statistics' },
        ]}
      />

      <ErrorBoundary>
        {tab === 'overview' && <OverviewTab portfolio={portfolio} />}
        {tab === 'positions' && (
          <Card padding={false}><PositionsList positions={positions} /></Card>
        )}
        {tab === 'trades' && <TradesTab />}
        {tab === 'stats' && <StatsTab />}
      </ErrorBoundary>
    </Page>
  );
}
