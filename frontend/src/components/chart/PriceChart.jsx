import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import { useRef, useEffect, useState, useCallback } from 'react';
import { api } from '../../api/client';
import { applyIndicators, toChartTime } from './IndicatorOverlay';
import ChartLegend from './ChartLegend';
import { Spinner } from '../common/Spinner';
import { COLORS, CHART_COLORS } from '../../utils/constants';

// All indicators fetched at once so toggling never re-fetches
const ALL_INDICATOR_IDS = ['ema9', 'ema20', 'ema50', 'ema200', 'bb', 'vwap'];

// Days of history per timeframe
const DAYS_FOR_TF = { '5m': 7, '15m': 14, '1h': 30, '1d': 120 };

// Polling interval for intraday live update (ms)
const LIVE_UPDATE_MS = 10_000;

/** Transform API price history → lightweight-charts OHLCV array */
function transformPriceData(data, timeframe) {
  const isDaily = timeframe === '1d';
  const seen = new Map();
  for (const b of (data?.bars ?? [])) {
    if (b.open == null || b.close == null) continue;
    const time = toChartTime(b.timestamp, isDaily);
    if (time == null) continue;
    seen.set(time, { time, open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume ?? 0 });
  }
  return [...seen.values()].sort((a, b) =>
    typeof a.time === 'string' ? a.time.localeCompare(b.time) : a.time - b.time,
  );
}

/** Round a Unix-second timestamp down to the nearest timeframe boundary */
function snapToInterval(tsSeconds, timeframe) {
  const mins = timeframe === '5m' ? 5 : timeframe === '15m' ? 15 : 60;
  const secs = mins * 60;
  return Math.floor(tsSeconds / secs) * secs;
}

/**
 * PriceChart — interactive candlestick chart powered by lightweight-charts v4.
 *
 * Props:
 *   indexId            — e.g. "NIFTY50"
 *   height             — px height of chart canvas (default 400)
 *   timeframe          — '5m' | '15m' | '1h' | '1d'
 *   indicators         — string[] of active indicator IDs (e.g. ['ema20', 'bb'])
 *   showVolume         — boolean (default true)
 *   showSignalMarkers  — boolean (default true)
 *   supportLevels      — number[] horizontal support lines
 *   resistanceLevels   — number[] horizontal resistance lines
 */
export default function PriceChart({
  indexId,
  height = 400,
  timeframe = '1d',
  indicators = [],
  showVolume = true,
  showSignalMarkers = true,
  supportLevels = [],
  resistanceLevels = [],
}) {
  const containerRef = useRef(null);
  const chartRef      = useRef(null);
  const candleRef     = useRef(null);
  const cleanupIndRef = useRef(null);  // indicator cleanup fn

  const [priceData,     setPriceData]     = useState([]);
  const [indicatorData, setIndicatorData] = useState({});
  const [loading,       setLoading]       = useState(false);
  const [legendValues,  setLegendValues]  = useState(null);

  // ── 1. Fetch price history ─────────────────────────────────────────────────
  useEffect(() => {
    if (!indexId) return;
    setLoading(true);
    setPriceData([]);
    const days = DAYS_FOR_TF[timeframe] ?? 30;
    api.getPriceHistory(indexId, days, timeframe)
      .then(data => setPriceData(transformPriceData(data, timeframe)))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [indexId, timeframe]);

  // ── 2. Fetch ALL indicator data (no re-fetch on toggle) ────────────────────
  useEffect(() => {
    if (!indexId) return;
    setIndicatorData({});
    api.getIndicatorValues(indexId, timeframe, ALL_INDICATOR_IDS)
      .then(setIndicatorData)
      .catch(() => setIndicatorData({}));
  }, [indexId, timeframe]);

  // ── 3. Build / rebuild the chart whenever price data changes ───────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Remove previous chart instance
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
      candleRef.current = null;
    }

    if (priceData.length === 0) return;

    const isDaily = timeframe === '1d';

    // ── Chart ────────────────────────────────────────────────────────────────
    const chart = createChart(el, {
      width:  el.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: COLORS.surface },
        textColor: COLORS.inkFaint,
        fontSize: 11,
        fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
      },
      grid: {
        vertLines: { color: CHART_COLORS.grid },
        horzLines: { color: CHART_COLORS.grid },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: {
        borderColor: COLORS.line,
        scaleMargins: { top: 0.05, bottom: 0.2 },
      },
      timeScale: {
        borderColor: COLORS.line,
        timeVisible: !isDaily,
        secondsVisible: false,
        rightOffset: 8,
      },
      handleScroll:  true,
      handleScale:   true,
    });

    // ── Candlestick series ───────────────────────────────────────────────────
    const candleSeries = chart.addCandlestickSeries({
      upColor:        COLORS.up,
      downColor:      COLORS.down,
      borderUpColor:  COLORS.up,
      borderDownColor:COLORS.down,
      wickUpColor:    COLORS.up,
      wickDownColor:  COLORS.down,
    });
    candleSeries.setData(priceData);

    // ── Volume histogram ─────────────────────────────────────────────────────
    if (showVolume) {
      const volSeries = chart.addHistogramSeries({
        priceFormat:  { type: 'volume' },
        priceScaleId: 'volume',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      try {
        volSeries.priceScale().applyOptions({
          scaleMargins: { top: 0.85, bottom: 0 },
        });
      } catch { /* ignore if API not present */ }

      volSeries.setData(
        priceData.map(d => ({
          time:  d.time,
          value: d.volume,
          color: d.close >= d.open ? `${COLORS.up}30` : `${COLORS.down}30`,
        })),
      );
    }

    // ── Crosshair → legend ───────────────────────────────────────────────────
    chart.subscribeCrosshairMove(param => {
      if (!param.time || !param.seriesData) {
        setLegendValues(null);
        return;
      }
      const candle = param.seriesData.get(candleSeries);
      setLegendValues(candle ? { ...candle, time: param.time } : null);
    });

    chart.timeScale().fitContent();

    // ── Responsive resize ────────────────────────────────────────────────────
    const ro = new ResizeObserver(entries => {
      const w = entries[0]?.contentRect?.width;
      if (w && chart) chart.applyOptions({ width: w });
    });
    ro.observe(el);

    chartRef.current  = chart;
    candleRef.current = candleSeries;

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current  = null;
      candleRef.current = null;
    };
  }, [priceData, showVolume, height, timeframe]);

  // ── 4. Apply / update indicator series (no chart recreation) ──────────────
  //    Runs after effect 3, and again whenever indicators/data toggle.
  useEffect(() => {
    const chart       = chartRef.current;
    const candleSeries = candleRef.current;
    if (!chart || !candleSeries) return;

    // Remove previous indicator series before adding new ones
    if (cleanupIndRef.current) {
      cleanupIndRef.current();
      cleanupIndRef.current = null;
    }

    cleanupIndRef.current = applyIndicators(chart, candleSeries, {
      indicators,
      indicatorData,
      timeframe,
      supportLevels,
      resistanceLevels,
      showSignalMarkers,
    });

    return () => {
      if (cleanupIndRef.current) {
        cleanupIndRef.current();
        cleanupIndRef.current = null;
      }
    };
  }, [indicators, indicatorData, timeframe, supportLevels, resistanceLevels, showSignalMarkers]);

  // ── 5. Live tick update for intraday ──────────────────────────────────────
  useEffect(() => {
    if (timeframe === '1d' || !indexId) return;

    const id = setInterval(async () => {
      if (!candleRef.current) return;
      try {
        const latest = await api.getIndexPrice(indexId);
        if (!latest?.ltp) return;
        const now = snapToInterval(Math.floor(Date.now() / 1000), timeframe);
        candleRef.current.update({
          time:  now,
          open:  latest.open  ?? latest.ltp,
          high:  latest.high  ?? latest.ltp,
          low:   latest.low   ?? latest.ltp,
          close: latest.ltp,
        });
      } catch { /* ignore transient fetch errors */ }
    }, LIVE_UPDATE_MS);

    return () => clearInterval(id);
  }, [indexId, timeframe]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="relative overflow-hidden">
      <ChartLegend values={legendValues} />

      {/* Loading spinner */}
      {loading && (
        <div
          className="absolute inset-0 top-8 flex items-center justify-center z-10 bg-surface/70"
        >
          <Spinner />
        </div>
      )}

      {/* Empty state */}
      {!loading && priceData.length === 0 && indexId && (
        <div
          className="flex items-center justify-center text-ink-faint text-sm"
          style={{ height }}
        >
          No price data for {indexId} on the {timeframe} timeframe
        </div>
      )}

      {!indexId && (
        <div
          className="flex items-center justify-center text-ink-faint text-sm"
          style={{ height }}
        >
          Select an index to view the price chart
        </div>
      )}

      {/* Chart canvas — always mounted with fixed height */}
      {priceData.length > 0 && <div ref={containerRef} style={{ height }} />}
    </div>
  );
}
