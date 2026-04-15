import { formatPrice } from '../../utils/formatters';

/**
 * Compact OHLC overlay shown at the top of the chart.
 * `values` is the object set by subscribeCrosshairMove:
 *   { open, high, low, close, time }
 */
export default function ChartLegend({ values }) {
  if (!values) {
    return (
      <div className="px-3 h-7 flex items-center border-b border-slate-800/60">
        <span className="text-slate-600 text-[11px] font-mono select-none">
          Hover chart to see OHLC values
        </span>
      </div>
    );
  }

  const isUp = values.close >= values.open;

  const items = [
    { label: 'O', value: formatPrice(values.open),  cls: 'text-slate-300' },
    { label: 'H', value: formatPrice(values.high),  cls: 'text-green-400' },
    { label: 'L', value: formatPrice(values.low),   cls: 'text-red-400'   },
    { label: 'C', value: formatPrice(values.close), cls: isUp ? 'text-green-400' : 'text-red-400' },
  ];

  return (
    <div className="px-3 h-7 flex items-center gap-4 border-b border-slate-800/60">
      {items.map(({ label, value, cls }) => (
        <span key={label} className="flex items-center gap-1 text-[11px] font-mono">
          <span className="text-slate-500">{label}</span>
          <span className={cls}>{value}</span>
        </span>
      ))}
    </div>
  );
}
