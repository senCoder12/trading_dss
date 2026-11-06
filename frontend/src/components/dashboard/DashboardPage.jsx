import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Wallet, TrendingUp, Crosshair, Briefcase, ArrowRight } from 'lucide-react';
import { useDataStore } from '../../hooks/useDataStore';
import { useIndexMeta, indexLabel } from '../../hooks/useIndexMeta';
import { ErrorBoundary } from '../common/ErrorBoundary';
import { StaleDataBanner } from '../common/StaleDataBanner';
import { StatTile, signTone } from '../common/StatTile';
import { Tabs } from '../common/Tabs';
import { FlashPrice } from '../common/FlashPrice';
import { Delta } from '../common/Delta';
import { Skeleton } from '../common/Skeleton';
import { Page } from '../layout/Page';
import ChartPanel from '../chart/ChartPanel';
import ActiveSignals from './ActiveSignals';
import QuickAlerts, { useQuickAlerts } from './QuickAlerts';
import ShadowModelPanel from './ShadowModelPanel';
import { formatCurrency, formatPercentage, formatPnL, formatPrice } from '../../utils/formatters';
import { WATCHLIST, SHORT_NAMES } from '../../utils/constants';

function WatchTile({ item, label, selected, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`card text-left px-4 py-3 transition-colors hover:border-line-strong focus:outline-none focus-visible:ring-2 focus-visible:ring-brand/40 ${
        selected ? 'border-brand/50 bg-brand/[0.04]' : ''
      }`}
    >
      <div className={`text-xs font-medium truncate ${selected ? 'text-brand' : 'text-ink-muted'}`}>{label}</div>
      <div className="mt-1 text-[17px] font-semibold num text-ink truncate">
        {item?.ltp != null ? <FlashPrice value={item.ltp} formatter={formatPrice} /> : '—'}
      </div>
      <div className="flex items-center gap-1.5 mt-0.5 text-2xs num whitespace-nowrap overflow-hidden">
        <Delta value={item?.change_pct} flash showIcon={false} className="font-medium" />
        {item?.change != null && (
          <span className="text-ink-faint truncate">{item.change > 0 ? '+' : ''}{formatPrice(item.change)}</span>
        )}
      </div>
    </button>
  );
}

export default function DashboardPage() {
  const [selected, setSelected] = useState('NIFTY50');
  const [sideTab, setSideTab] = useState('signals');
  const meta = useIndexMeta();
  const {
    prices: { data: priceData, lastUpdated, isStale },
    portfolio: { data: pf },
    signals: { data: sigData },
  } = useDataStore();
  const alerts = useQuickAlerts(6);

  const byId = Object.fromEntries((priceData?.indices ?? []).map((i) => [i.index_id, i]));
  const signals = sigData?.signals ?? [];
  const actionable = signals.filter((s) => s.signal_type !== 'NO_TRADE').length;

  return (
    <Page>
      <StaleDataBanner isStale={isStale} lastUpdated={lastUpdated} />

      {/* KPIs */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <StatTile
          label="Capital"
          icon={Wallet}
          value={formatCurrency(pf?.capital, 0)}
          sub={pf ? `${formatPercentage(pf.total_return_pct)} all-time` : null}
          subTone={signTone(pf?.total_return_pct)}
        />
        <StatTile
          label="Today's P&L"
          icon={TrendingUp}
          value={formatPnL(pf?.today_pnl)}
          tone={signTone(pf?.today_pnl)}
          sub={pf ? `${pf.today_trades ?? 0} trades today` : null}
        />
        <StatTile
          label="Active signals"
          icon={Crosshair}
          value={sigData ? actionable : '—'}
          tone={actionable > 0 ? 'brand' : 'default'}
          sub={sigData ? (actionable ? `of ${signals.length} indices scanned` : `No setups across ${signals.length} indices`) : null}
        />
        <StatTile
          label="Open positions"
          icon={Briefcase}
          value={pf ? pf.open_positions?.length ?? 0 : '—'}
          sub={pf?.overall_win_rate != null ? `${pf.overall_win_rate.toFixed(0)}% win rate` : null}
        />
      </div>

      {/* Watchlist */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-ink">Watchlist</h2>
          <Link to="/market" className="text-xs text-ink-faint hover:text-brand inline-flex items-center gap-1 transition-colors">
            All {priceData?.indices?.length ?? ''} indices <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {priceData
            ? WATCHLIST.map((id) => (
                <WatchTile
                  key={id}
                  item={byId[id]}
                  label={SHORT_NAMES[id] ?? indexLabel(meta, id)}
                  selected={selected === id}
                  onClick={() => setSelected(id)}
                />
              ))
            : WATCHLIST.map((id) => <Skeleton key={id} height="86px" className="rounded-xl" />)}
        </div>
      </div>

      {/* Chart + side panel */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <ChartPanel indexId={selected} onIndexChange={setSelected} height={400} className="xl:col-span-2" />

        <ErrorBoundary>
          {/* Wrapper stretches to the chart's height; card fills it and scrolls inside */}
          <div className="relative min-w-0 min-h-[420px]">
          <section className="card flex flex-col min-w-0 xl:absolute xl:inset-0">
            <Tabs
              className="px-3 pt-3"
              value={sideTab}
              onChange={setSideTab}
              tabs={[
                { id: 'signals', label: 'Signals', count: actionable || undefined },
                { id: 'alerts', label: 'Alerts', count: alerts.length || undefined },
              ]}
            />
            <div className="flex-1 overflow-y-auto">
              {sideTab === 'signals' ? <ActiveSignals /> : <QuickAlerts items={alerts} />}
            </div>
            <Link
              to={sideTab === 'signals' ? '/signals' : '/anomalies'}
              className="flex items-center justify-center gap-1.5 h-11 border-t border-line text-xs font-medium text-ink-faint hover:text-brand transition-colors"
            >
              {sideTab === 'signals' ? 'Open signals' : 'Open alerts'} <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </section>
          </div>
        </ErrorBoundary>
      </div>

      {/* Shadow model: evaluation only, never traded */}
      <ErrorBoundary><ShadowModelPanel /></ErrorBoundary>
    </Page>
  );
}
