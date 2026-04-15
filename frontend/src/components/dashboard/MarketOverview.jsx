import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Spinner } from '../common/Spinner';
import { EmptyState } from '../common/EmptyState';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { formatPrice, formatPercentage } from '../../utils/formatters';
import { REFRESH_INTERVALS } from '../../utils/constants';
import { BarChart2 } from 'lucide-react';

export default function MarketOverview() {
  const { data, loading, error, lastUpdated } = usePolling(api.getMarketPrices, REFRESH_INTERVALS.marketPrices);
  const indices = data?.indices ?? [];

  return (
    <Card
      title="Market Prices"
      padding={false}
      actions={<RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />}
    >
      {loading && !data ? (
        <div className="flex justify-center p-8"><Spinner /></div>
      ) : !indices.length ? (
        <EmptyState icon={BarChart2} title="No price data" message="Waiting for market data feed" />
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 divide-y divide-slate-700/50">
          {indices.map((item) => {
            const chg = item.change_pct ?? 0;
            const positive = chg > 0;
            const negative = chg < 0;
            return (
              <div
                key={item.index_id}
                className="p-3 hover:bg-slate-700/30 transition-colors"
              >
                <div className="text-slate-400 text-xs font-medium mb-1">{item.index_id}</div>
                <div className="font-mono font-semibold text-slate-100 text-sm">
                  {formatPrice(item.ltp)}
                </div>
                <div
                  className={`flex items-center gap-1 text-xs mt-0.5 ${
                    positive ? 'text-green-400' : negative ? 'text-red-400' : 'text-slate-500'
                  }`}
                >
                  {positive ? (
                    <TrendingUp className="w-3 h-3" />
                  ) : negative ? (
                    <TrendingDown className="w-3 h-3" />
                  ) : (
                    <Minus className="w-3 h-3" />
                  )}
                  <span>{formatPercentage(chg)}</span>
                  {item.change != null && (
                    <span className="text-slate-500">({positive ? '+' : ''}{item.change?.toFixed(2)})</span>
                  )}
                </div>
                {item.high != null && item.low != null && (
                  <div className="text-[10px] text-slate-600 mt-1">
                    H: {formatPrice(item.high)} · L: {formatPrice(item.low)}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}
