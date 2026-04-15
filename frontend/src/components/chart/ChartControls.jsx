/**
 * Toolbar rendered above the price chart.
 * Timeframe selector + indicator toggle buttons.
 */

const TIMEFRAMES = [
  { label: '5m',  value: '5m'  },
  { label: '15m', value: '15m' },
  { label: '1H',  value: '1h'  },
  { label: '1D',  value: '1d'  },
];

export const AVAILABLE_INDICATORS = [
  { id: 'ema20',  label: 'EMA 20',    color: '#3b82f6' },
  { id: 'ema50',  label: 'EMA 50',    color: '#a855f7' },
  { id: 'ema200', label: 'EMA 200',   color: '#ec4899' },
  { id: 'bb',     label: 'Bollinger', color: '#6366f1' },
  { id: 'vwap',   label: 'VWAP',      color: '#f97316' },
];

export default function ChartControls({ timeframe, onTimeframeChange, indicators, onIndicatorsChange }) {
  function toggleIndicator(id) {
    const next = indicators.includes(id)
      ? indicators.filter(i => i !== id)
      : [...indicators, id];
    onIndicatorsChange(next);
  }

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 border-b border-slate-800/60">
      {/* Timeframe buttons */}
      <div className="flex gap-1">
        {TIMEFRAMES.map(tf => (
          <button
            key={tf.value}
            onClick={() => onTimeframeChange(tf.value)}
            className={`px-3 py-0.5 rounded text-xs font-medium transition-colors ${
              timeframe === tf.value
                ? 'bg-blue-500/20 text-blue-400 border border-blue-500/40'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/60'
            }`}
          >
            {tf.label}
          </button>
        ))}
      </div>

      {/* Divider */}
      <span className="h-4 border-l border-slate-700" />

      {/* Indicator toggles */}
      <div className="flex flex-wrap gap-1.5">
        {AVAILABLE_INDICATORS.map(ind => {
          const active = indicators.includes(ind.id);
          return (
            <button
              key={ind.id}
              onClick={() => toggleIndicator(ind.id)}
              title={active ? `Hide ${ind.label}` : `Show ${ind.label}`}
              className={`px-2 py-0.5 rounded text-xs border transition-all ${
                active
                  ? 'border-current opacity-100'
                  : 'border-slate-700 text-slate-500 opacity-60 hover:opacity-80'
              }`}
              style={{ color: active ? ind.color : undefined }}
            >
              {ind.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
