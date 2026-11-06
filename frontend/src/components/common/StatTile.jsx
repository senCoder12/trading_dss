const TONE_TEXT = {
  up: 'text-up',
  down: 'text-down',
  warn: 'text-warn',
  info: 'text-info',
  brand: 'text-brand',
  default: 'text-ink',
  muted: 'text-ink-muted',
};

/** Signed tone helper: positive → up, negative → down, zero/null → default */
export function signTone(v) {
  if (v == null || Number.isNaN(v) || v === 0) return 'default';
  return v > 0 ? 'up' : 'down';
}

/**
 * KPI tile: label on top, big value, optional sub-line.
 * Keep to one number per tile — that is the whole point.
 */
export function StatTile({ label, value, sub, subTone = 'muted', tone = 'default', icon: Icon, className = '', children }) {
  return (
    <div className={`card px-4 py-3.5 min-w-0 ${className}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="label truncate">{label}</span>
        {Icon && <Icon className="w-4 h-4 text-ink-ghost flex-shrink-0" />}
      </div>
      <div className={`mt-1.5 text-[22px] leading-7 font-semibold tracking-tight num truncate ${TONE_TEXT[tone] ?? TONE_TEXT.default}`}>
        {value ?? '—'}
      </div>
      {sub != null && (
        <div className={`mt-0.5 text-xs num truncate ${subTone === 'muted' ? 'text-ink-faint' : TONE_TEXT[subTone]}`}>
          {sub}
        </div>
      )}
      {children}
    </div>
  );
}
