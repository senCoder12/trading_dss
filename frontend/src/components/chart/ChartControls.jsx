import { Segmented } from '../common/Segmented';

export const TIMEFRAMES = [
  { label: '5m', value: '5m' },
  { label: '15m', value: '15m' },
  { label: '1H', value: '1h' },
  { label: '1D', value: '1d' },
];

export const AVAILABLE_INDICATORS = [
  { id: 'ema20', label: 'EMA 20', color: '#3b82f6' },
  { id: 'ema50', label: 'EMA 50', color: '#a855f7' },
  { id: 'ema200', label: 'EMA 200', color: '#ec4899' },
  { id: 'bb', label: 'Bollinger', color: '#6366f1' },
  { id: 'vwap', label: 'VWAP', color: '#f97316' },
];

/** Timeframe selector on its own — fits in a card header. */
export function TimeframeControl({ timeframe, onTimeframeChange }) {
  return (
    <Segmented
      size="xs"
      ariaLabel="Timeframe"
      options={TIMEFRAMES}
      value={timeframe}
      onChange={onTimeframeChange}
    />
  );
}

/** Indicator toggle chips. */
export function IndicatorToggles({ indicators, onIndicatorsChange }) {
  function toggle(id) {
    onIndicatorsChange(indicators.includes(id) ? indicators.filter((i) => i !== id) : [...indicators, id]);
  }
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="label mr-1">Overlays</span>
      {AVAILABLE_INDICATORS.map((ind) => {
        const active = indicators.includes(ind.id);
        return (
          <button
            key={ind.id}
            onClick={() => toggle(ind.id)}
            aria-pressed={active}
            className={`inline-flex items-center gap-1.5 h-6 px-2 rounded-md text-2xs font-medium border transition-colors ${
              active
                ? 'bg-surface-3 border-line-strong text-ink'
                : 'border-line text-ink-faint hover:text-ink-muted hover:border-line-strong'
            }`}
          >
            <span className="w-2 h-2 rounded-full" style={{ background: active ? ind.color : 'transparent', boxShadow: `inset 0 0 0 1.5px ${ind.color}` }} />
            {ind.label}
          </button>
        );
      })}
    </div>
  );
}

/** Back-compat: full toolbar (timeframe + overlays) in one row. */
export default function ChartControls({ timeframe, onTimeframeChange, indicators, onIndicatorsChange }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <IndicatorToggles indicators={indicators} onIndicatorsChange={onIndicatorsChange} />
      <TimeframeControl timeframe={timeframe} onTimeframeChange={onTimeframeChange} />
    </div>
  );
}
