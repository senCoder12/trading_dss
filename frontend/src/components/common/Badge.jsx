const TONES = {
  success: 'bg-up/10 text-up ring-up/25',
  danger: 'bg-down/10 text-down ring-down/25',
  warning: 'bg-warn/10 text-warn ring-warn/25',
  info: 'bg-info/10 text-info ring-info/25',
  brand: 'bg-brand/10 text-brand ring-brand/25',
  violet: 'bg-violet/10 text-violet ring-violet/25',
  neutral: 'bg-ink/5 text-ink-muted ring-ink/10',
};

// Domain value → tone. Lets callers pass raw API enums as `variant`.
const VARIANT_TONE = {
  BUY_CALL: 'success', BUY_PUT: 'danger', NO_TRADE: 'neutral',
  BUY: 'success', SELL: 'danger', HOLD: 'warning',
  HIGH: 'success', MEDIUM: 'warning', LOW: 'neutral',
  BULLISH: 'success', BEARISH: 'danger', NEUTRAL: 'neutral',
  SLIGHTLY_BULLISH: 'success', SLIGHTLY_BEARISH: 'danger',
  WIN: 'success', LOSS: 'danger', OPEN: 'info', EXPIRED: 'neutral',
  OK: 'success', HEALTHY: 'success', RUNNING: 'success',
  WARNING: 'warning', DEGRADED: 'warning',
  ERROR: 'danger', UNHEALTHY: 'danger', UNKNOWN: 'neutral',
  CRITICAL: 'danger', NOISE: 'neutral',
  PRIORITY: 'success', ACTIVE: 'brand', WATCHLIST: 'neutral',
  // Decision source: LIVE engine vs SHADOW model must never look alike
  LIVE: 'success', SHADOW: 'violet',
};

// Severity reads differently from confidence: HIGH severity is bad.
const SEVERITY_TONE = { CRITICAL: 'danger', HIGH: 'danger', MEDIUM: 'warning', LOW: 'info', NOISE: 'neutral' };

export function toneFor(variant, kind) {
  const key = variant?.toUpperCase?.();
  if (kind === 'severity') return SEVERITY_TONE[key] ?? 'neutral';
  return VARIANT_TONE[key] ?? 'neutral';
}

/**
 * Small status pill.
 *   <Badge variant="BUY_CALL">Buy Call</Badge>          — tone from domain enum
 *   <Badge variant="HIGH" kind="severity">High</Badge>   — severity semantics
 *   <Badge tone="info">Live</Badge>                      — explicit tone
 */
export function Badge({ variant, kind, tone, children, className = '', size = 'sm', dot = false }) {
  const t = tone ?? toneFor(variant, kind);
  const sz = size === 'xs' ? 'px-1.5 h-5 text-[10.5px]' : 'px-2 h-[22px] text-2xs';
  return (
    <span
      className={`inline-flex items-center gap-1 ${sz} rounded-md font-medium ring-1 ring-inset whitespace-nowrap ${TONES[t]} ${className}`}
    >
      {dot && <span className="w-1.5 h-1.5 rounded-full bg-current" />}
      {children}
    </span>
  );
}
