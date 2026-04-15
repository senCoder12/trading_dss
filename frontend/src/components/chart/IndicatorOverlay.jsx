/**
 * IndicatorOverlay — manages indicator line series on a lightweight-charts instance.
 *
 * Call `applyIndicators(chart, candleSeries, opts)` inside a chart useEffect.
 * It returns a cleanup function that removes the added series and price lines.
 *
 * No React component is exported; this module is a pure utility layer so that
 * PriceChart.jsx stays readable.
 */

export const EMA_COLORS = {
  ema9:   '#f59e0b',  // amber
  ema20:  '#3b82f6',  // blue
  ema50:  '#a855f7',  // purple
  ema200: '#ec4899',  // pink
};

/** Convert API timestamp → lightweight-charts time value */
export function toChartTime(t, isDaily) {
  if (!t) return null;
  if (isDaily) {
    // Accept both 'YYYY-MM-DD' and 'YYYY-MM-DDTHH:mm:ss'
    return typeof t === 'string' ? t.split('T')[0] : t;
  }
  // Intraday: Unix seconds
  if (typeof t === 'number') return t;
  return Math.floor(new Date(t).getTime() / 1000);
}

/** Map [{time, value}] → chart-ready array, dropping nulls */
export function toLineSeries(arr, isDaily) {
  return (arr ?? [])
    .filter(p => p?.value != null)
    .map(p => ({ time: toChartTime(p.time, isDaily), value: p.value }))
    .filter(p => p.time != null);
}

/**
 * Add all active indicator series to `chart`.
 *
 * @param {object} chart         — lightweight-charts IChartApi instance
 * @param {object} candleSeries  — the ISeriesApi for the main candlestick series
 * @param {object} opts
 *   indicators      — string[] of active indicator IDs
 *   indicatorData   — data map returned by /indicators API
 *   timeframe       — '5m' | '15m' | '1h' | '1d'
 *   supportLevels   — number[] of support price levels
 *   resistanceLevels — number[] of resistance price levels
 *   showSignalMarkers — boolean
 *
 * @returns {() => void} cleanup — removes all added series / price lines
 */
export function applyIndicators(chart, candleSeries, opts) {
  const {
    indicators = [],
    indicatorData = {},
    timeframe = '1d',
    supportLevels = [],
    resistanceLevels = [],
    showSignalMarkers = true,
  } = opts;

  const isDaily = timeframe === '1d';
  const addedSeries = [];
  const addedLines = [];

  // ── EMA lines ──────────────────────────────────────────────────────────────
  for (const ind of indicators) {
    if (!ind.startsWith('ema')) continue;
    const data = toLineSeries(indicatorData[ind], isDaily);
    if (!data.length) continue;

    const s = chart.addLineSeries({
      color: EMA_COLORS[ind] || '#94a3b8',
      lineWidth: 1,
      title: ind.toUpperCase(),
      lastValueVisible: true,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });
    s.setData(data);
    addedSeries.push(s);
  }

  // ── Bollinger Bands ────────────────────────────────────────────────────────
  if (indicators.includes('bb')) {
    const bbDefs = [
      { key: 'bb_upper',  color: '#6366f1',   style: 2 },  // dashed
      { key: 'bb_lower',  color: '#6366f1',   style: 2 },
      { key: 'bb_middle', color: '#6366f160', style: 0 },  // solid, semi-transparent
    ];
    for (const { key, color, style } of bbDefs) {
      const data = toLineSeries(indicatorData[key], isDaily);
      if (!data.length) continue;
      const s = chart.addLineSeries({
        color,
        lineWidth: 1,
        lineStyle: style,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      });
      s.setData(data);
      addedSeries.push(s);
    }
  }

  // ── VWAP ──────────────────────────────────────────────────────────────────
  if (indicators.includes('vwap')) {
    const data = toLineSeries(indicatorData.vwap, isDaily);
    if (data.length) {
      const s = chart.addLineSeries({
        color: '#f97316',
        lineWidth: 2,
        title: 'VWAP',
        lastValueVisible: true,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      });
      s.setData(data);
      addedSeries.push(s);
    }
  }

  // ── Support / resistance horizontal lines ──────────────────────────────────
  for (const level of supportLevels) {
    if (typeof level !== 'number') continue;
    addedLines.push(candleSeries.createPriceLine({
      price: level,
      color: '#22c55e80',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: `S ${level}`,
    }));
  }
  for (const level of resistanceLevels) {
    if (typeof level !== 'number') continue;
    addedLines.push(candleSeries.createPriceLine({
      price: level,
      color: '#ef444480',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: `R ${level}`,
    }));
  }

  // Max Pain (from options data injected into indicatorData)
  if (typeof indicatorData.max_pain === 'number') {
    addedLines.push(candleSeries.createPriceLine({
      price: indicatorData.max_pain,
      color: '#eab30880',
      lineWidth: 2,
      lineStyle: 1,  // dotted
      axisLabelVisible: true,
      title: 'Max Pain',
    }));
  }

  // ── Signal markers (arrows at entry/exit points) ───────────────────────────
  if (showSignalMarkers && indicatorData.signals?.length) {
    const markers = indicatorData.signals
      .map(sig => ({
        time: toChartTime(sig.time, isDaily),
        position: sig.type === 'BUY_CALL' ? 'belowBar' : 'aboveBar',
        color: sig.type === 'BUY_CALL' ? '#22c55e' : '#ef4444',
        shape: sig.type === 'BUY_CALL' ? 'arrowUp' : 'arrowDown',
        text: sig.type === 'BUY_CALL' ? 'CALL' : 'PUT',
      }))
      .filter(m => m.time != null)
      .sort((a, b) =>
        typeof a.time === 'string' ? a.time.localeCompare(b.time) : a.time - b.time,
      );
    try { candleSeries.setMarkers(markers); } catch { /* ignore if series gone */ }
  }

  // ── Cleanup closure ────────────────────────────────────────────────────────
  return function cleanupIndicators() {
    for (const s of addedSeries) {
      try { chart.removeSeries(s); } catch { /* chart may already be removed */ }
    }
    for (const pl of addedLines) {
      try { candleSeries.removePriceLine(pl); } catch {}
    }
    try { candleSeries.setMarkers([]); } catch {}
  };
}
