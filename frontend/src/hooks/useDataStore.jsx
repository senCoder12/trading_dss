import { createContext, useContext } from 'react';
import { usePolling } from './usePolling';
import { api } from '../api/client';
import { REFRESH_INTERVALS } from '../utils/constants';

const DataStoreContext = createContext(null);

export function DataStoreProvider({ children }) {
  // Core data — fetched ONCE, shared across all components
  const prices = usePolling(api.getMarketPrices, REFRESH_INTERVALS.marketPrices);
  const signals = usePolling(api.getCurrentSignals, REFRESH_INTERVALS.signals);
  const portfolio = usePolling(api.getPortfolio, REFRESH_INTERVALS.portfolio);
  const vix = usePolling(api.getVix, REFRESH_INTERVALS.vix);
  const systemHealth = usePolling(api.getSystemStatus, 30_000);
  const anomalies = usePolling(api.getActiveAnomalies, REFRESH_INTERVALS.anomalies);

  const value = { prices, signals, portfolio, vix, systemHealth, anomalies };

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
