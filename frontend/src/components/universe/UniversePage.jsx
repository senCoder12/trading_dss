import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Sparkles, AlertTriangle, RefreshCw } from 'lucide-react';
import { useApi } from '../../hooks/useApi';
import { useWebSocket } from '../../hooks/useWebSocket';
import { api } from '../../api/client';
import { FullPage } from '../layout/Page';
import { Spinner } from '../common/Spinner';
import { EmptyState } from '../common/EmptyState';
import LeftPanel from './LeftPanel';
import RightPanel from './RightPanel';

function formatRefresh(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString('en-IN', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'Asia/Kolkata' });
}

export default function UniversePage() {
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [selectedExpiry, setSelectedExpiry] = useState(null);
  const [activeFilter, setActiveFilter] = useState('all');
  const [search, setSearch] = useState('');
  const expirySetRef = useRef(false);

  // Universe snapshot
  const {
    data: universeData, loading: universeLoading, error: universeError, refetch: refetchUniverse,
  } = useApi(api.getUniverseSnapshot, []);

  // Option chain for the selected symbol
  const { data: chainData, loading: chainLoading } = useApi(
    () => (selectedSymbol ? api.getUniverseOptionsChain(selectedSymbol, selectedExpiry) : Promise.resolve(null)),
    [selectedSymbol, selectedExpiry],
  );

  const stocks = universeData?.active_universe ?? [];

  // Auto-select the top-ranked stock once data arrives
  useEffect(() => {
    if (!selectedSymbol && stocks.length) {
      const top = [...stocks].sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity))[0];
      setSelectedSymbol(top.symbol);
    }
  }, [stocks, selectedSymbol]);

  // New symbol → pick its first expiry once the chain arrives
  useEffect(() => {
    if (selectedSymbol) expirySetRef.current = false;
  }, [selectedSymbol]);

  useEffect(() => {
    if (selectedSymbol && !expirySetRef.current && chainData?.available_expiries?.length) {
      setSelectedExpiry(chainData.available_expiries[0]);
      expirySetRef.current = true;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chainData?.available_expiries]);

  const selectSymbol = useCallback((sym) => {
    setSelectedSymbol(sym);
    setSelectedExpiry(null);
  }, []);

  // Filter + search
  const filteredUniverse = useMemo(() => {
    let list = stocks;
    switch (activeFilter) {
      case 'priority': list = list.filter((s) => s.tier === 'PRIORITY'); break;
      case 'oi_up': list = list.filter((s) => (s.oi_change_score ?? 0) > 75); break;
      case 'iv_up': list = list.filter((s) => (s.iv_change_score ?? 0) > 75); break;
      default:
    }
    const q = search.trim().toLowerCase();
    if (q) {
      list = list.filter((s) =>
        s.symbol?.toLowerCase().includes(q)
        || s.sector?.toLowerCase().includes(q)
        || s.company_name?.toLowerCase().includes(q));
    }
    return list;
  }, [stocks, activeFilter, search]);

  // Live universe refresh
  const handleWsMessage = useCallback((message) => {
    if (message.type === 'universe_update') refetchUniverse();
  }, [refetchUniverse]);
  useWebSocket(handleWsMessage);

  // ── States ─────────────────────────────────────────────────────────────────
  if (universeLoading && !universeData) {
    return (
      <FullPage className="items-center justify-center gap-3 text-ink-faint">
        <Spinner />
        <span className="text-xs">Loading F&amp;O universe…</span>
      </FullPage>
    );
  }

  if (universeError && !universeData) {
    return (
      <FullPage className="justify-center">
        <EmptyState
          icon={AlertTriangle}
          title="Couldn't load the F&O universe"
          message={String(universeError)}
          action={<button onClick={refetchUniverse} className="btn btn-secondary"><RefreshCw className="w-3.5 h-3.5" />Retry</button>}
        />
      </FullPage>
    );
  }

  if (!stocks.length) {
    const scanned = universeData?.total_fo_stocks ?? 0;
    const liquid = universeData?.total_liquid ?? 0;
    return (
      <FullPage className="justify-center">
        <EmptyState
          icon={Sparkles}
          title="No tradable stocks in the universe yet"
          message={
            <>
              <span className="block num">{scanned} F&amp;O stocks scanned · {liquid} passed liquidity filters</span>
              <span className="block mt-1">
                The universe scorer ranks stocks once live volume and OI data clear the liquidity gates.
                Last refresh {formatRefresh(universeData?.timestamp)}.
              </span>
            </>
          }
          action={<button onClick={refetchUniverse} className="btn btn-secondary"><RefreshCw className="w-3.5 h-3.5" />Refresh</button>}
        />
      </FullPage>
    );
  }

  return (
    <FullPage>
      <div className="flex flex-1 min-h-0">
        <LeftPanel
          universeData={universeData}
          filteredUniverse={filteredUniverse}
          activeFilter={activeFilter}
          onFilterChange={setActiveFilter}
          search={search}
          onSearchChange={setSearch}
          selectedSymbol={selectedSymbol}
          onSelectSymbol={selectSymbol}
          refreshedAt={formatRefresh(universeData?.timestamp)}
        />

        {selectedSymbol ? (
          <RightPanel
            key={selectedSymbol}
            symbol={selectedSymbol}
            universeData={universeData}
            chainData={chainData}
            chainLoading={chainLoading}
            selectedExpiry={selectedExpiry}
            onExpiryChange={setSelectedExpiry}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <EmptyState icon={Sparkles} title="Select a stock" message="Pick a stock on the left to see its score breakdown and option chain." />
          </div>
        )}
      </div>
    </FullPage>
  );
}
