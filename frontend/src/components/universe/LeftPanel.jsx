import { useMemo } from 'react';
import { Search, X } from 'lucide-react';
import { Segmented } from '../common/Segmented';

const TIERS = ['PRIORITY', 'ACTIVE', 'WATCHLIST'];

const TIER_META = {
  PRIORITY: { label: 'Priority', range: 'Rank 1–15', dot: 'bg-up', text: 'text-up', bar: 'bg-up' },
  ACTIVE: { label: 'Active', range: 'Rank 16–50', dot: 'bg-brand', text: 'text-brand', bar: 'bg-brand' },
  WATCHLIST: { label: 'Watchlist', range: 'Rank 50+', dot: 'bg-ink-ghost', text: 'text-ink-muted', bar: 'bg-ink-faint' },
};

const FILTERS = [
  { value: 'all', label: 'All' },
  { value: 'priority', label: 'Priority' },
  { value: 'oi_up', label: 'OI ↑', title: 'OI change score above 75' },
  { value: 'iv_up', label: 'IV ↑', title: 'IV change score above 75' },
];

function Metric({ label, value, tone = 'text-ink' }) {
  return (
    <div className="rounded-lg bg-surface border border-line px-3 py-2 min-w-0">
      <div className="text-[10.5px] text-ink-faint truncate">{label}</div>
      <div className={`text-[15px] font-semibold num truncate capitalize ${tone}`}>{value}</div>
    </div>
  );
}

function StockRow({ stock, selected, onSelect }) {
  const meta = TIER_META[stock.tier] ?? TIER_META.WATCHLIST;
  const score = stock.composite_score;
  const lot = stock.lot_size != null ? Number(stock.lot_size).toLocaleString('en-IN') : '—';

  return (
    <button
      onClick={onSelect}
      className={`relative w-full text-left flex items-center gap-3 px-4 py-2.5 border-b border-line/60 transition-colors ${
        selected ? 'bg-surface-3' : 'hover:bg-surface-2'
      }`}
    >
      {selected && <span className="absolute left-0 top-2 bottom-2 w-[3px] rounded-r bg-brand" />}
      <div className="min-w-0 flex-1">
        <div className={`text-[13px] font-medium truncate ${selected ? 'text-ink' : 'text-ink'}`}>{stock.symbol}</div>
        <div className="text-2xs text-ink-faint truncate">
          {stock.sector ?? '—'} · Lot {lot}
        </div>
      </div>
      <div className="flex flex-col items-end gap-1 flex-shrink-0">
        <span className={`text-[13px] font-semibold num ${meta.text}`}>{score != null ? score.toFixed(1) : '—'}</span>
        <span className="w-12 h-1 rounded-full bg-surface-3 overflow-hidden">
          <span className={`block h-full rounded-full ${meta.bar}`} style={{ width: `${Math.max(0, Math.min(100, score ?? 0))}%` }} />
        </span>
      </div>
      <span className="w-8 text-right text-2xs text-ink-ghost num flex-shrink-0">#{stock.rank ?? '—'}</span>
    </button>
  );
}

export default function LeftPanel({
  universeData,
  filteredUniverse,
  activeFilter,
  onFilterChange,
  search,
  onSearchChange,
  selectedSymbol,
  onSelectSymbol,
  refreshedAt,
}) {
  const grouped = useMemo(() => {
    const groups = {};
    filteredUniverse.forEach((s) => { (groups[s.tier] ??= []).push(s); });
    Object.values(groups).forEach((g) => g.sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity)));
    return groups;
  }, [filteredUniverse]);

  const all = universeData?.active_universe ?? [];
  const priority = all.filter((s) => s.tier === 'PRIORITY').length;
  const topSector = universeData?.top_sector && universeData.top_sector !== 'unknown' ? universeData.top_sector : '—';

  return (
    <aside className="w-[300px] flex-shrink-0 border-r border-line bg-panel/50 flex flex-col min-h-0">
      {/* Header */}
      <div className="px-4 pt-4 pb-3 border-b border-line space-y-3 flex-shrink-0">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-sm font-semibold text-ink">Universe</span>
          <span className="text-2xs text-ink-faint num truncate">{all.length} stocks · {refreshedAt}</span>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Metric label="Active" value={all.length} />
          <Metric label="Priority" value={priority} tone="text-up" />
          <Metric label="Liquid" value={universeData?.total_liquid ?? '—'} tone="text-brand" />
          <Metric label="Top sector" value={topSector} />
        </div>

        <div className="relative">
          <Search className="w-4 h-4 text-ink-ghost absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search symbol or sector"
            className="input h-8 pl-9 pr-8 text-[13px]"
            aria-label="Search stocks"
          />
          {search && (
            <button onClick={() => onSearchChange('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-ink-faint hover:text-ink" aria-label="Clear search">
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        <Segmented size="xs" ariaLabel="Filter" options={FILTERS} value={activeFilter} onChange={onFilterChange} />
      </div>

      {/* List */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {filteredUniverse.length === 0 ? (
          <div className="px-6 py-10 text-center text-xs text-ink-faint">No stocks match this filter.</div>
        ) : (
          TIERS.map((tier) => {
            const list = grouped[tier] ?? [];
            if (!list.length) return null;
            const meta = TIER_META[tier];
            return (
              <div key={tier}>
                <div className="sticky top-0 z-10 flex items-center justify-between px-4 h-8 bg-panel/95 backdrop-blur border-b border-line">
                  <span className="flex items-center gap-2 label">
                    <span className={`w-1.5 h-1.5 rounded-full ${meta.dot}`} />
                    {meta.label}
                    <span className="normal-case tracking-normal text-ink-ghost font-normal">{meta.range}</span>
                  </span>
                  <span className="text-2xs text-ink-faint num">{list.length}</span>
                </div>
                {list.map((stock) => (
                  <StockRow
                    key={stock.symbol}
                    stock={stock}
                    selected={selectedSymbol === stock.symbol}
                    onSelect={() => onSelectSymbol(stock.symbol)}
                  />
                ))}
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
}
