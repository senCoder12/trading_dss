const BASE = '/api';

async function get(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

async function post(path, body) {
  const res = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

async function del(path) {
  const res = await fetch(BASE + path, { method: 'DELETE' });
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

export const api = {
  // Market
  getMarketPrices: () => get('/market/prices'),
  getIndexPrice: (id) => get(`/market/prices/${id}`),
  getPriceHistory: (id, days = 30, timeframe = '1d') =>
    get(`/market/prices/${id}/history?days=${days}&timeframe=${timeframe}`),
  getIndicatorValues: (id, timeframe = '1d', indicators = []) =>
    get(`/market/prices/${id}/indicators?timeframe=${timeframe}&indicators=${indicators.join(',')}`),
  getOptionsChain: (id) => get(`/market/options/${id}`),
  getVix: () => get('/market/vix'),

  // Signals
  getCurrentSignals: () => get('/signals/current'),
  getSignalDetail: (id) => get(`/signals/current/${id}`),
  getSignalHistory: (days = 7, indexId) =>
    get(`/signals/history?days=${days}${indexId ? `&index_id=${indexId}` : ''}`),
  getPerformance: (days = 30) => get(`/signals/performance?days=${days}`),

  // Portfolio
  getPortfolio: () => get('/portfolio/summary'),
  getPositions: () => get('/portfolio/positions'),
  getEquityHistory: (days = 30) => get(`/portfolio/history?days=${days}`),
  getTrades: (days = 7, indexId) =>
    get(`/portfolio/trades?days=${days}${indexId ? `&index_id=${indexId}` : ''}`),

  // News
  getNewsFeed: (limit = 20, minSeverity, indexId) => {
    const params = new URLSearchParams({ limit });
    if (minSeverity) params.set('min_severity', minSeverity);
    if (indexId) params.set('index_id', indexId);
    return get(`/news/feed?${params}`);
  },
  getNewsSummary: () => get('/news/summary'),
  getNewsSentiment: (indexId) => get(`/news/sentiment/${indexId}`),

  // Anomalies
  getActiveAnomalies: (indexId, minSeverity) => {
    const params = new URLSearchParams();
    if (indexId) params.set('index_id', indexId);
    if (minSeverity) params.set('min_severity', minSeverity);
    const qs = params.toString();
    return get(`/anomalies/active${qs ? `?${qs}` : ''}`);
  },
  getAnomalyDashboard: () => get('/anomalies/dashboard'),

  // System
  getSystemHealth: () => get('/system/health'),
  getSystemStatus: () => get('/system/status'),
  activateKillSwitch: (reason) => post('/system/kill-switch', { reason }),
  deactivateKillSwitch: () => del('/system/kill-switch'),
};
