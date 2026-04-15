const styles = {
  // Signal types
  BUY_CALL:   'bg-green-500/20 text-green-400 border-green-500/40',
  BUY_PUT:    'bg-red-500/20 text-red-400 border-red-500/40',
  NO_TRADE:   'bg-slate-500/20 text-slate-400 border-slate-500/40',
  // Directions
  BUY:        'bg-green-500/20 text-green-400 border-green-500/40',
  SELL:       'bg-red-500/20 text-red-400 border-red-500/40',
  HOLD:       'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
  // Confidence
  HIGH:       'bg-green-500/20 text-green-400 border-green-500/40',
  MEDIUM:     'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
  LOW:        'bg-blue-500/20 text-blue-400 border-blue-500/40',
  // Sentiment
  BULLISH:    'bg-green-500/20 text-green-400 border-green-500/40',
  BEARISH:    'bg-red-500/20 text-red-400 border-red-500/40',
  NEUTRAL:    'bg-slate-500/20 text-slate-400 border-slate-500/40',
  SLIGHTLY_BULLISH: 'bg-green-500/10 text-green-300 border-green-500/30',
  SLIGHTLY_BEARISH: 'bg-red-500/10 text-red-300 border-red-500/30',
  // Outcomes
  WIN:        'bg-green-500/20 text-green-400 border-green-500/40',
  LOSS:       'bg-red-500/20 text-red-400 border-red-500/40',
  OPEN:       'bg-blue-500/20 text-blue-400 border-blue-500/40',
  EXPIRED:    'bg-slate-500/20 text-slate-400 border-slate-500/40',
  // Health
  OK:         'bg-green-500/20 text-green-400 border-green-500/40',
  HEALTHY:    'bg-green-500/20 text-green-400 border-green-500/40',
  WARNING:    'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
  DEGRADED:   'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
  ERROR:      'bg-red-500/20 text-red-400 border-red-500/40',
  UNHEALTHY:  'bg-red-500/20 text-red-400 border-red-500/40',
  UNKNOWN:    'bg-slate-500/20 text-slate-400 border-slate-500/40',
  // Severity
  CRITICAL:   'bg-red-600/30 text-red-300 border-red-600/50',
  NOISE:      'bg-slate-600/20 text-slate-500 border-slate-600/30',
};

export function Badge({ variant, children, className = '', size = 'sm' }) {
  const style = styles[variant?.toUpperCase()] ?? 'bg-slate-500/20 text-slate-400 border-slate-500/40';
  const sz = size === 'xs' ? 'px-1.5 py-0 text-[10px]' : 'px-2 py-0.5 text-xs';
  return (
    <span className={`inline-flex items-center ${sz} rounded-full font-medium border ${style} ${className}`}>
      {children}
    </span>
  );
}
