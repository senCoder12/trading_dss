import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { Spinner } from '../common/Spinner';
import { formatPrice, formatPercentage, formatLargeNumber } from '../../utils/formatters';
import { REFRESH_INTERVALS } from '../../utils/constants';

export function IndexDetail({ indexId }) {
  const { data, loading } = usePolling(() => api.getIndexPrice(indexId), REFRESH_INTERVALS.marketPrices);

  if (loading) return <div className="flex justify-center p-4"><Spinner /></div>;
  if (!data) return null;

  const chgColor = (data.change_pct ?? 0) >= 0 ? 'text-green-400' : 'text-red-400';

  const rows = [
    { label: 'LTP', value: formatPrice(data.ltp), bold: true },
    { label: 'Change', value: `${data.change > 0 ? '+' : ''}${data.change?.toFixed(2)} (${formatPercentage(data.change_pct)})`, color: chgColor },
    { label: 'Open', value: formatPrice(data.open) },
    { label: 'High', value: formatPrice(data.high) },
    { label: 'Low', value: formatPrice(data.low) },
    { label: 'Prev Close', value: formatPrice(data.previous_close) },
    { label: 'VWAP', value: formatPrice(data.vwap) },
    { label: 'Volume', value: data.volume ? formatLargeNumber(data.volume) : '--' },
    { label: 'Timeframe', value: data.timeframe ?? '--' },
  ];

  return (
    <div className="space-y-1">
      {rows.map(({ label, value, bold, color }) => (
        <div key={label} className="flex items-center justify-between py-1.5 border-b border-slate-700/30 text-sm">
          <span className="text-slate-400 text-xs">{label}</span>
          <span className={`font-mono ${bold ? 'font-bold text-slate-100' : 'text-slate-300'} ${color ?? ''}`}>
            {value ?? '--'}
          </span>
        </div>
      ))}
    </div>
  );
}
