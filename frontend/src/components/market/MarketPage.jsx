import { useState } from 'react';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Spinner } from '../common/Spinner';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { IndexDetail } from './IndexDetail';
import { OptionsHeatmap } from './OptionsHeatmap';
import { TechnicalView } from './TechnicalView';
import { formatPrice, formatPercentage } from '../../utils/formatters';
import { REFRESH_INTERVALS } from '../../utils/constants';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

const TF_OPTIONS = ['1m', '5m', '15m', '1h', '1d'];
const HIST_DAYS = { '1m': 1, '5m': 3, '15m': 7, '1h': 14, '1d': 30 };

export default function MarketPage() {
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [tf, setTf] = useState('1d');

  const { data, loading, error, lastUpdated } = usePolling(
    api.getMarketPrices, REFRESH_INTERVALS.marketPrices,
  );

  const indices = data?.indices ?? [];
  const active = selectedIndex ?? indices[0]?.index_id;

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
                    {formatPrice(item.ltp)}
                  </div>
                  <div className={`flex items-center gap-1 text-xs ${pos ? 'text-green-400' : neg ? 'text-red-400' : 'text-slate-500'}`}>
                    {pos ? <TrendingUp className="w-3 h-3" /> : neg ? <TrendingDown className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
                    {formatPercentage(chg)}
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </Card>

      {active && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Index detail */}
          <Card title={`${active} · Detail`}>
            <IndexDetail key={active} indexId={active} />
          </Card>

          {/* Price chart with timeframe selector */}
          <Card title={`${active} · Price Chart`} padding={false}>
            <div className="px-4 py-2 border-b border-slate-700 flex items-center gap-2">
              {TF_OPTIONS.map((t) => (
                <button
                  key={t}
                  onClick={() => setTf(t)}
                  className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                    tf === t
                      ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
            <div className="p-4">
              <TechnicalView key={`${active}-${tf}`} indexId={active} days={HIST_DAYS[tf]} timeframe={tf} />
            </div>
          </Card>
        </div>
      )}

      {/* Options heatmap */}
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
