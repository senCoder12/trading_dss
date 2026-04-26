import { useMemo, useRef } from 'react';

const TIER_COLORS = {
  PRIORITY: 'bg-green-500',
  ACTIVE: 'bg-blue-500',
  WATCHLIST: 'bg-slate-500',
};

const TIER_LABELS = {
  PRIORITY: 'Priority (1–15)',
  ACTIVE: 'Active (16–50)',
  WATCHLIST: 'Watchlist',
};

function StockRow({ stock, selected, onSelect }) {
  const scoreColor = stock.tier === 'PRIORITY'
    ? 'text-green-400'
    : stock.tier === 'ACTIVE'
    ? 'text-blue-400'
    : 'text-slate-400';

  const barColor = stock.tier === 'PRIORITY'
    ? 'bg-green-500/30'
    : stock.tier === 'ACTIVE'
    ? 'bg-blue-500/30'
    : 'bg-slate-500/30';

  return (
    <div
      onClick={onSelect}
      className={`px-3 py-2.5 border-b border-slate-700 cursor-pointer transition-colors ${
        selected ? 'bg-blue-500/20' : 'hover:bg-slate-800'
      }`}
    >
      <div className="flex items-center gap-2">
        {/* Tier dot */}
        <div className={`w-2 h-2 rounded-full ${TIER_COLORS[stock.tier]}`} />

        {/* Symbol + sector + lot size */}
        <div className="flex-1 min-w-0">
          <div className="font-bold text-sm text-slate-100">{stock.symbol}</div>
          <div className="text-xs text-slate-500">
            {stock.sector} • {stock.lot_size.toLocaleString()}L
          </div>
        </div>

        {/* Score on right */}
        <div className="text-right">
          <div className={`font-bold text-sm ${scoreColor}`}>
            {stock.composite_score.toFixed(1)}
          </div>
          <div className={`h-1 w-10 rounded mt-1 ${barColor}`} />
        </div>

        {/* Rank (muted) */}
        <div className="text-xs text-slate-600 w-6 text-right">#{stock.rank}</div>
      </div>
    </div>
  );
}

export default function LeftPanel({
  universeData,
  filteredUniverse,
  activeFilter,
  onFilterChange,
  selectedSymbol,
  onSelectSymbol,
}) {
  const scrollRef = useRef(null);

  // Group by tier for visual separators
  const groupedStocks = useMemo(() => {
    const groups = {};
    filteredUniverse.forEach(stock => {
      if (!groups[stock.tier]) groups[stock.tier] = [];
      groups[stock.tier].push(stock);
    });
    return groups;
  }, [filteredUniverse]);

  // Summary metrics
  const metrics = useMemo(() => {
    if (!universeData?.active_universe) {
      return { active: 0, priority: 0, liquid: 0, sector: '' };
    }
    const priority = universeData.active_universe.filter(s => s.tier === 'PRIORITY').length;
    const active = universeData.active_universe.length;
    return {
      active,
      priority,
      liquid: universeData.total_liquid || 0,
      sector: universeData.top_sector || '—',
    };
  }, [universeData]);

  const lastRefreshTime = useMemo(() => {
    if (!universeData?.timestamp) return '—';
    const date = new Date(universeData.timestamp);
    return date.toLocaleTimeString('en-IN');
  }, [universeData?.timestamp]);

  return (
    <div className="w-[280px] border-r border-slate-700 flex flex-col bg-slate-900 overflow-hidden">
      {/* Header */}
      <div className="border-b border-slate-700 p-3">
        <h2 className="font-bold text-sm text-slate-100">F&O Universe</h2>
        <div className="text-xs text-slate-500 mt-1">Last refresh: {lastRefreshTime}</div>
      </div>

      {/* Summary metrics grid */}
      <div className="grid grid-cols-2 gap-2 p-3 border-b border-slate-700 bg-slate-800/50">
        <div>
          <div className="text-2xl font-bold text-slate-100">{metrics.active}</div>
          <div className="text-xs text-slate-500">Active Universe</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-green-400">{metrics.priority}</div>
          <div className="text-xs text-slate-500">Priority List</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-blue-400">{metrics.liquid}</div>
          <div className="text-xs text-slate-500">Liquidity Passed</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-slate-100">{metrics.sector}</div>
          <div className="text-xs text-slate-500">Top Sector</div>
        </div>
      </div>

      {/* Filter pills */}
      <div className="flex gap-2 p-3 border-b border-slate-700 overflow-x-auto">
        {[
          { key: 'all', label: `All ${metrics.active}` },
          { key: 'priority', label: 'Priority' },
          { key: 'oi_up', label: 'OI↑' },
          { key: 'iv_up', label: 'IV↑' },
        ].map(pill => (
          <button
            key={pill.key}
            onClick={() => onFilterChange(pill.key)}
            className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap ${
              activeFilter === pill.key
                ? 'bg-blue-500/30 text-blue-400 border border-blue-500/50'
                : 'text-slate-400 hover:text-slate-200 bg-slate-700 hover:bg-slate-600'
            }`}
          >
            {pill.label}
          </button>
        ))}
      </div>

      {/* Stock list */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {['PRIORITY', 'ACTIVE', 'WATCHLIST'].map(tier => {
          const stocks = groupedStocks[tier] || [];
          if (stocks.length === 0) return null;

          return (
            <div key={tier}>
              {/* Tier separator */}
              <div className="sticky top-0 bg-slate-800 border-b border-slate-700 px-3 py-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wide">
                {TIER_LABELS[tier]}
              </div>

              {/* Stocks in this tier */}
              {stocks.map(stock => (
                <StockRow
                  key={stock.symbol}
                  stock={stock}
                  selected={selectedSymbol === stock.symbol}
                  onSelect={() => onSelectSymbol(stock.symbol)}
                />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
