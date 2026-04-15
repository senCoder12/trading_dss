export const COLORS = {
  background: '#0f172a',
  card: '#1e293b',
  textPrimary: '#f1f5f9',
  textSecondary: '#94a3b8',
  green: '#22c55e',
  red: '#ef4444',
  yellow: '#eab308',
  blue: '#3b82f6',
  border: '#334155',
};

export const CHART_COLORS = {
  green: '#22c55e',
  red: '#ef4444',
  yellow: '#eab308',
  blue: '#3b82f6',
  purple: '#a855f7',
  grid: '#1e293b',
  text: '#94a3b8',
};

export const REFRESH_INTERVALS = {
  marketPrices: 10_000,
  signals: 15_000,
  portfolio: 15_000,
  anomalies: 30_000,
  vix: 30_000,
  news: 60_000,
  system: 30_000,
};

export const SIGNAL_TYPE_LABELS = {
  BUY_CALL: 'BUY CALL',
  BUY_PUT: 'BUY PUT',
  NO_TRADE: 'NO TRADE',
};

export const SIGNAL_TYPE_COLORS = {
  BUY_CALL: 'text-green-400',
  BUY_PUT: 'text-red-400',
  NO_TRADE: 'text-slate-400',
};

export const SEVERITY_COLORS = {
  HIGH: 'text-red-400',
  MEDIUM: 'text-yellow-400',
  LOW: 'text-blue-400',
  CRITICAL: 'text-red-500',
  NOISE: 'text-slate-500',
};

export const SENTIMENT_LABELS = {
  BULLISH: { label: 'Bullish', color: 'text-green-400' },
  SLIGHTLY_BULLISH: { label: 'Slight Bull', color: 'text-green-300' },
  NEUTRAL: { label: 'Neutral', color: 'text-slate-400' },
  SLIGHTLY_BEARISH: { label: 'Slight Bear', color: 'text-red-300' },
  BEARISH: { label: 'Bearish', color: 'text-red-400' },
};

export const VIX_REGIMES = {
  LOW: { label: 'Low', color: 'text-green-400', bg: 'bg-green-500/20' },
  NORMAL: { label: 'Normal', color: 'text-blue-400', bg: 'bg-blue-500/20' },
  ELEVATED: { label: 'Elevated', color: 'text-yellow-400', bg: 'bg-yellow-500/20' },
  HIGH: { label: 'High', color: 'text-red-400', bg: 'bg-red-500/20' },
  EXTREME: { label: 'Extreme', color: 'text-red-500', bg: 'bg-red-500/30' },
};
