import { createContext, useContext, useState, useCallback, useMemo, useEffect } from 'react';
import { usePolling } from './usePolling';
import { useWebSocket } from './useWebSocket';
import { api } from '../api/client';
import { REFRESH_INTERVALS } from '../utils/constants';

const DataStoreContext = createContext(null);

export function DataStoreProvider({ children, onExternalEvent }) {
  const [priceData, setPriceData] = useState(null);
  const [priceLoading, setPriceLoading] = useState(true);
  const [priceError, setPriceError] = useState(null);
  const [priceLastUpdated, setPriceLastUpdated] = useState(null);

  const handleWsMessage = useCallback((message) => {
    if (message.type === 'price_tick') {
      const { index_id, ...fields } = message.data;
      setPriceData(prev => {
        if (!prev?.indices) return prev;
        const updated = prev.indices.map(idx => {
          if (idx.index_id !== index_id) return idx;
          const pc = idx.prev_close;
          const change = pc ? Math.round((fields.ltp - pc) * 100) / 100 : idx.change;
          const change_pct = pc ? Math.round(((fields.ltp - pc) / pc) * 10000) / 100 : idx.change_pct;
          return {
            ...idx,
            ...fields,
            change,
            change_pct,
            updated_at: fields.timestamp,
          };
        });
        return { ...prev, indices: updated };
      });
      setPriceLastUpdated(Date.now());
      return;
    }
    onExternalEvent?.(message);
  }, [onExternalEvent]);

  const { connected } = useWebSocket(handleWsMessage);

  // Poll prices every 30s — covers WS dead/disconnected scenarios as fallback
  const pricesPoll = usePolling(api.getMarketPrices, 30_000);

  useEffect(() => {
    if (pricesPoll.data != null) {
      setPriceData(pricesPoll.data);
      setPriceLastUpdated(Date.now());
      setPriceError(null);
      setPriceLoading(false);
    }
  }, [pricesPoll.data]);

  useEffect(() => {
    if (pricesPoll.error) {
      setPriceError(pricesPoll.error);
      setPriceLoading(false);
    }
  }, [pricesPoll.error]);

  // Other data sources still use polling
  const signals = usePolling(api.getCurrentSignals, REFRESH_INTERVALS.signals);
  const portfolio = usePolling(api.getPortfolio, REFRESH_INTERVALS.portfolio);
  const vix = usePolling(api.getVix, REFRESH_INTERVALS.vix);
  const systemHealth = usePolling(api.getSystemStatus, 30_000);
  const anomalies = usePolling(api.getActiveAnomalies, REFRESH_INTERVALS.anomalies);

  const prices = useMemo(() => ({
    data: priceData,
    loading: priceLoading,
    error: priceError,
    lastUpdated: priceLastUpdated,
    isStale: false,
  }), [priceData, priceLoading, priceError, priceLastUpdated]);

  const value = { prices, signals, portfolio, vix, systemHealth, anomalies, connected };

  return (
    <DataStoreContext.Provider value={value}>
      {children}
    </DataStoreContext.Provider>
  );
}

export function useDataStore() {
  const ctx = useContext(DataStoreContext);
  if (!ctx) throw new Error('useDataStore must be used inside DataStoreProvider');
  return ctx;
}
