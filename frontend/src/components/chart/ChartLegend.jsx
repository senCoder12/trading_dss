import { formatPrice } from '../../utils/formatters';

/**
 * OHLC readout above the chart. `values` comes from subscribeCrosshairMove:
 *   { open, high, low, close, time }
 */
export default function ChartLegend({ values }) {
  if (!values) {
    return (
      <div className="px-1 h-8 flex items-center">
        <span className="text-ink-ghost text-2xs select-none">Hover the chart for OHLC</span>
      </div>
    );
  }

  const isUp = values.close >= values.open;
  const items = [
    { label: 'O', value: formatPrice(values.open), cls: 'text-ink-muted' },
    { label: 'H', value: formatPrice(values.high), cls: 'text-up' },
    { label: 'L', value: formatPrice(values.low), cls: 'text-down' },
    { label: 'C', value: formatPrice(values.close), cls: isUp ? 'text-up' : 'text-down' },
  ];

  return (
    <div className="px-1 h-8 flex items-center gap-4">
      {items.map(({ label, value, cls }) => (
        <span key={label} className="flex items-center gap-1.5 text-xs num">
          <span className="text-ink-ghost">{label}</span>
          <span className={cls}>{value}</span>
        </span>
      ))}
    </div>
  );
}
