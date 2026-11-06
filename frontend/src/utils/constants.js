// Palette mirrors tailwind.config.js tokens — used where JS needs raw colours
// (lightweight-charts, recharts).
export const COLORS = {
  canvas: '#080b11',
  surface: '#10151d',
  surface2: '#151b25',
  line: '#1d2430',
  lineStrong: '#29323f',
  ink: '#e7ebf1',
  inkMuted: '#a0a9b8',
  inkFaint: '#6b7585',
  brand: '#5b8cff',
  up: '#22c38e',
  down: '#f0525a',
  warn: '#f4b03e',
  info: '#4cb8ff',
  violet: '#a78bfa',
};

export const CHART_COLORS = {
  green: COLORS.up,
  red: COLORS.down,
  yellow: COLORS.warn,
  blue: COLORS.brand,
  purple: COLORS.violet,
  grid: '#161c26',
  text: COLORS.inkFaint,
};

/** Shared recharts tooltip styling */
export const TOOLTIP_STYLE = {
  contentStyle: {
    background: COLORS.surface2,
    border: `1px solid ${COLORS.lineStrong}`,
    borderRadius: 8,
    fontSize: 12,
    boxShadow: '0 8px 24px -8px rgb(0 0 0 / 0.6)',
  },
  labelStyle: { color: COLORS.inkMuted, marginBottom: 4 },
  itemStyle: { color: COLORS.ink },
  cursor: { fill: 'rgb(255 255 255 / 0.03)' },
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
  BUY_CALL: 'Buy Call',
  BUY_PUT: 'Buy Put',
  NO_TRADE: 'No Trade',
};

export const SIGNAL_TYPE_COLORS = {
  BUY_CALL: 'text-up',
  BUY_PUT: 'text-down',
  NO_TRADE: 'text-ink-faint',
};

export const SEVERITY_COLORS = {
  HIGH: 'text-down',
  MEDIUM: 'text-warn',
  LOW: 'text-info',
  CRITICAL: 'text-down',
  NOISE: 'text-ink-faint',
};

export const SENTIMENT_LABELS = {
  BULLISH: { label: 'Bullish', color: 'text-up' },
  SLIGHTLY_BULLISH: { label: 'Mildly bullish', color: 'text-up' },
  NEUTRAL: { label: 'Neutral', color: 'text-ink-muted' },
  SLIGHTLY_BEARISH: { label: 'Mildly bearish', color: 'text-down' },
  BEARISH: { label: 'Bearish', color: 'text-down' },
};

export const VIX_REGIMES = {
  LOW: { label: 'Low', color: 'text-up', tone: 'success' },
  NORMAL: { label: 'Normal', color: 'text-info', tone: 'info' },
  ELEVATED: { label: 'Elevated', color: 'text-warn', tone: 'warning' },
  HIGH: { label: 'High', color: 'text-down', tone: 'danger' },
  EXTREME: { label: 'Extreme', color: 'text-down', tone: 'danger' },
};

/** Headline indices shown on the dashboard watchlist (F&O-tradable first). */
export const WATCHLIST = ['NIFTY50', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX'];

export const INDEX_CATEGORY_LABELS = {
  broad_market: 'Broad market',
  sectoral: 'Sectoral',
  thematic: 'Thematic',
  strategy: 'Strategy',
  volatility: 'Volatility',
};

/** Compact names for headline indices (registry names are long, e.g. "NIFTY FINANCIAL SERVICES"). */
export const SHORT_NAMES = {
  NIFTY50: 'Nifty 50',
  BANKNIFTY: 'Bank Nifty',
  FINNIFTY: 'Fin Nifty',
  MIDCPNIFTY: 'Midcap Select',
  SENSEX: 'Sensex',
  INDIA_VIX: 'India VIX',
};
