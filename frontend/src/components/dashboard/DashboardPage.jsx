import { useState } from 'react';
import { TrendingUp, TrendingDown, Minus, ChevronDown } from 'lucide-react';
import { useDataStore } from '../../hooks/useDataStore';
import { Card } from '../common/Card';
import { ErrorBoundary } from '../common/ErrorBoundary';
import { StaleDataBanner } from '../common/StaleDataBanner';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { formatPrice, formatPercentage } from '../../utils/formatters';
import PriceChart from '../chart/PriceChart';
import ChartControls from '../chart/ChartControls';
import ActiveSignals from './ActiveSignals';
import PortfolioSummary from './PortfolioSummary';
import QuickAlerts from './QuickAlerts';
import MarketSentiment from './MarketSentiment';

/** Compact pill that shows price + change for one index */
function IndexPill({ item, selected, onClick }) {
  const chg = item.change_pct ?? 0;
  const pos  = chg > 0;
  const neg  = chg < 0;

  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs transition-colors whitespace-nowrap ${
        selected
          ? 'bg-blue-500/15 border border-blue-500/40 text-blue-300'
          : 'bg-slate-800 border border-slate-700 text-slate-300 hover:border-slate-600'
      }`}
    >
      <span className="font-medium">{item.index_id}</span>
      <span className="font-mono text-[11px]">{formatPrice(item.ltp)}</span>
      <span className={`flex items-center gap-0.5 ${pos ? 'text-green-400' : neg ? 'text-red-400' : 'text-slate-500'}`}>
        {pos ? <TrendingUp className="w-3 h-3" /> : neg ? <TrendingDown className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
        {formatPercentage(chg)}
      </span>
    </button>
  );
}

export default function DashboardPage() {
  // Selected index for the primary chart (default to first returned index)
  const [selectedIndex, setSelectedIndex] = useState('NIFTY50');
  const [timeframe,     setTimeframe]     = useState('1d');
  const [indicators,    setIndicators]    = useState(['ema20', 'ema50']);

  // Live index price strip from shared data store (no duplicate polling)
  const { prices: { data, loading, error, lastUpdated, isStale } } = useDataStore();
  const indices = data?.indices ?? [];

  // Once data arrives, default selection to first available index if still on placeholder
  // (selectedIndex='NIFTY50' is a good default; keep it unless user explicitly changes it)

  const chartTitle = selectedIndex
    ? `${selectedIndex} · Price Chart`
    : 'Price Chart';

  return (
    <div className="p-4 space-y-4">
      <StaleDataBanner isStale={isStale} lastUpdated={lastUpdated} />

      {/* ── Primary chart section ─────────────────────────────────────────── */}
      <div className="space-y-0">
        {/* Index selector strip */}
        <div className="flex items-center gap-2 mb-2 flex-wrap">
          <span className="text-slate-500 text-xs shrink-0">Chart</span>
          <div className="flex gap-1.5 flex-wrap">
            {indices.length > 0
              ? indices.map(item => (
                  <IndexPill
                    key={item.index_id}
                    item={item}
                    selected={selectedIndex === item.index_id}
                    onClick={() => setSelectedIndex(item.index_id)}
                  />
                ))
              : /* skeleton while loading */
                ['NIFTY50', 'BANKNIFTY', 'FINNIFTY'].map(id => (
                  <div key={id} className="h-7 w-28 bg-slate-800 rounded-md animate-pulse" />
                ))
            }
          </div>
          <div className="ml-auto">
            <RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />
          </div>
        </div>

        {/* Chart card */}
        <ErrorBoundary>
          <Card title={chartTitle} padding={false}>
            <ChartControls
              timeframe={timeframe}
              onTimeframeChange={setTimeframe}
              indicators={indicators}
              onIndicatorsChange={setIndicators}
            />
            <div className="p-2">
              <PriceChart
                indexId={selectedIndex}
                height={420}
                timeframe={timeframe}
                indicators={indicators}
                showVolume
                showSignalMarkers
              />
            </div>
          </Card>
        </ErrorBoundary>
      </div>

      {/* ── Secondary row: Signals · Portfolio · Alerts ───────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ErrorBoundary><ActiveSignals /></ErrorBoundary>
        <ErrorBoundary><PortfolioSummary /></ErrorBoundary>
        <div className="space-y-4">
          <ErrorBoundary><MarketSentiment /></ErrorBoundary>
          <ErrorBoundary><QuickAlerts /></ErrorBoundary>
        </div>
      </div>

    </div>
  );
}
