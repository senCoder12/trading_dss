import { useState } from 'react';
import { usePolling } from '../../hooks/usePolling';
import { useDataStore } from '../../hooks/useDataStore';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Spinner } from '../common/Spinner';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { FlashPrice } from '../common/FlashPrice';
import { IndexDetail } from './IndexDetail';
import { OptionsHeatmap } from './OptionsHeatmap';
import PriceChart from '../chart/PriceChart';
import ChartControls from '../chart/ChartControls';
import { formatPrice, formatPercentage } from '../../utils/formatters';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

export default function MarketPage() {
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [tf, setTf] = useState('1d');
  const [indicators, setIndicators] = useState(['ema20', 'ema50']);

  // Live prices from shared store (fed by WS ticks + 30s polling fallback)
  const { prices: { data, loading, error, lastUpdated } } = useDataStore();
  const indices = data?.indices ?? [];
  const active = selectedIndex ?? indices[0]?.index_id;

  // Options chain polling is per-index so it stays local
  const { data: optData, loading: optLoading } = usePolling(
    () => active ? api.getOptionsChain(active) : Promise.resolve(null),
    60_000,
  );

  return (
    <div className="p-4 space-y-4">
      {/* Index price cards */}
      <Card
        title="Market Prices"
        padding={false}
        actions={<RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />}
      >
        {loading && !data ? (
          <div className="flex justify-center p-8"><Spinner /></div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 divide-x divide-slate-700/50">
            {indices.map((item) => {
              const chg = item.change_pct ?? 0;
              const pos = chg > 0;
              const neg = chg < 0;
              const isSelected = (selectedIndex ?? indices[0]?.index_id) === item.index_id;
              return (
                <button
                  key={item.index_id}
                  onClick={() => setSelectedIndex(item.index_id)}
                  className={`p-3 text-left hover:bg-slate-700/40 transition-colors ${
                    isSelected ? 'bg-blue-500/10 border-b-2 border-blue-500' : ''
                  }`}
                >
                  <div className="text-slate-400 text-xs font-medium">{item.index_id}</div>
                  <div className="font-mono font-semibold text-slate-100 text-sm">
                    <FlashPrice value={item.ltp} formatter={formatPrice} />
                  </div>
                  <div className={`flex items-center gap-1 text-xs ${pos ? 'text-green-400' : neg ? 'text-red-400' : 'text-slate-500'}`}>
                    {pos ? <TrendingUp className="w-3 h-3" /> : neg ? <TrendingDown className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
                    <FlashPrice value={chg} formatter={formatPercentage} />
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </Card>

      {active && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card title={`${active} · Detail`}>
            <IndexDetail key={active} indexId={active} />
          </Card>

          <Card title={`${active} · Price Chart`} padding={false}>
            <ChartControls
              timeframe={tf}
              onTimeframeChange={setTf}
              indicators={indicators}
              onIndicatorsChange={setIndicators}
            />
            <div className="p-2">
              <PriceChart
                indexId={active}
                height={360}
                timeframe={tf}
                indicators={indicators}
                showVolume
                showSignalMarkers
              />
            </div>
          </Card>
        </div>
      )}

      {active && (
        <Card title={`${active} · Options OI Heatmap`} padding={false}>
          <div className="p-4">
            {optLoading ? (
              <div className="flex justify-center p-6"><Spinner /></div>
            ) : (
              <OptionsHeatmap data={optData} />
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
