import {
  ComposedChart, Line, Bar, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer,
} from 'recharts';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { Spinner } from '../common/Spinner';
import { formatPrice } from '../../utils/formatters';
import { EmptyState } from '../common/EmptyState';
import { TrendingUp } from 'lucide-react';

export function TechnicalView({ indexId, days = 30, timeframe = '1d' }) {
  const intervalMs = timeframe === '1d' ? 60_000 : 15_000;
  const { data, loading } = usePolling(
    () => api.getPriceHistory(indexId, days, timeframe),
    intervalMs,
  );

  if (loading) return <div className="flex justify-center p-8"><Spinner /></div>;

  const bars = data?.bars ?? [];
  if (!bars.length) return <EmptyState icon={TrendingUp} title="No price history" />;

  // Simple SMA-20 calculation
  const chartData = bars.map((b, i) => {
    const slice = bars.slice(Math.max(0, i - 19), i + 1);
    const sma20 = slice.reduce((s, x) => s + x.close, 0) / slice.length;
    return {
      date: b.timestamp.split('T')[0],
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
      volume: b.volume,
      sma20: parseFloat(sma20.toFixed(2)),
    };
  });

  return (
    <div className="space-y-4">
      {/* Price chart */}
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 9, fill: '#94a3b8' }}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fontSize: 9, fill: '#94a3b8' }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => formatPrice(v)}
              width={60}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, fontSize: 11 }}
              formatter={(v, n) => [formatPrice(v), n === 'sma20' ? 'SMA 20' : n]}
              labelStyle={{ color: '#94a3b8' }}
            />
            <Bar dataKey="close" fill="#3b82f6" fillOpacity={0.5} radius={[2, 2, 0, 0]} />
            <Line dataKey="sma20" stroke="#eab308" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Latest bar stats */}
      {chartData.length > 0 && (() => {
        const last = chartData[chartData.length - 1];
        const prev = chartData[chartData.length - 2];
        const chg = prev ? last.close - prev.close : 0;
        const chgPct = prev ? (chg / prev.close) * 100 : 0;
        const pos = chg >= 0;
        return (
          <div className="grid grid-cols-4 gap-2 text-xs">
            {[
              { label: 'Close', value: formatPrice(last.close), color: pos ? 'text-green-400' : 'text-red-400' },
              { label: 'Change', value: `${pos ? '+' : ''}${chg.toFixed(2)} (${chgPct.toFixed(2)}%)`, color: pos ? 'text-green-400' : 'text-red-400' },
              { label: 'SMA 20', value: formatPrice(last.sma20), color: last.close > last.sma20 ? 'text-green-400' : 'text-red-400' },
              { label: 'High/Low', value: `${formatPrice(last.high)} / ${formatPrice(last.low)}`, color: 'text-slate-300' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-slate-900/50 rounded p-2 text-center">
                <div className="text-slate-500 text-[10px]">{label}</div>
                <div className={`font-mono font-semibold text-sm mt-0.5 ${color}`}>{value}</div>
              </div>
            ))}
          </div>
        );
      })()}
    </div>
  );
}
