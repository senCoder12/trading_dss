import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useApi } from '../../hooks/useApi';
import { useWebSocket } from '../../hooks/useWebSocket';
import { api } from '../../api/client';
import { Spinner } from '../common/Spinner';
import LeftPanel from './LeftPanel';
import RightPanel from './RightPanel';

export default function UniversePage() {
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [selectedExpiry, setSelectedExpiry] = useState(null);
  const [activeFilter, setActiveFilter] = useState('all');
  const expirySetRef = useRef(false);

  // Fetch universe snapshot
  const { data: universeData, loading: universeLoading, refetch: refetchUniverse } = useApi(
    api.getUniverseSnapshot,
    [],
  );

  // Fetch options chain for selected symbol
  const { data: chainData, loading: chainLoading } = useApi(
    selectedSymbol ? () => api.getUniverseOptionsChain(selectedSymbol, selectedExpiry) : null,
    [selectedSymbol, selectedExpiry],
  );

  // Set default expiry when symbol is first selected
  useEffect(() => {
    if (selectedSymbol) {
      expirySetRef.current = false; // Reset when symbol changes
    }
  }, [selectedSymbol]);

  useEffect(() => {
    if (selectedSymbol && !expirySetRef.current && chainData?.available_expiries) {
      setSelectedExpiry(chainData.available_expiries[0]);
      expirySetRef.current = true;
    }
  }, [chainData?.available_expiries]);

  // Filter universe data based on active filter
  const filteredUniverse = useMemo(() => {
    if (!universeData?.active_universe) return [];

    const stocks = universeData.active_universe;
    switch (activeFilter) {
      case 'priority':
        return stocks.filter(s => s.tier === 'PRIORITY');
      case 'oi_up':
        return stocks.filter(s => s.oi_change_score > 75);
      case 'iv_up':
        return stocks.filter(s => s.iv_change_score > 75);
      default:
        return stocks;
    }
  }, [universeData?.active_universe, activeFilter]);

  // WebSocket handler for universe updates
  const handleWsMessage = useCallback((message) => {
    if (message.type === 'universe_update') {
      // Smooth update of universe data
      refetchUniverse();
    }
  }, [refetchUniverse]);

  useWebSocket(handleWsMessage);

  if (universeLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spinner />
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-slate-900 text-slate-100 overflow-hidden">
      {/* Left panel — fixed width, scrollable stock list */}
      <LeftPanel
        universeData={universeData}
        filteredUniverse={filteredUniverse}
        activeFilter={activeFilter}
        onFilterChange={setActiveFilter}
        selectedSymbol={selectedSymbol}
        onSelectSymbol={setSelectedSymbol}
      />

      {/* Right panel — options chain */}
      {selectedSymbol ? (
        <RightPanel
          symbol={selectedSymbol}
          universeData={universeData}
          chainData={chainData}
          chainLoading={chainLoading}
          selectedExpiry={selectedExpiry}
          onExpiryChange={setSelectedExpiry}
        />
      ) : (
        <div className="flex-1 flex items-center justify-center text-slate-400">
          Select a stock to view options chain
        </div>
      )}
    </div>
  );
}
