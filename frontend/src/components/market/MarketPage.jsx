import { useEffect, useMemo, useState } from 'react';
import { Search, CandlestickChart, Layers } from 'lucide-react';
import { usePolling } from '../../hooks/usePolling';
import { useDataStore } from '../../hooks/useDataStore';
import { useIndexMeta, indexLabel } from '../../hooks/useIndexMeta';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { Tabs, useTabParam } from '../common/Tabs';
import { Segmented } from '../common/Segmented';
import { FlashPrice } from '../common/FlashPrice';
import { Delta } from '../common/Delta';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { EmptyState } from '../common/EmptyState';
import { RowSkeleton, Skeleton } from '../common/Skeleton';
import { Page } from '../layout/Page';
import ChartPanel from '../chart/ChartPanel';
import { OptionsHeatmap } from './OptionsHeatmap';
import { formatPrice, formatLargeNumber } from '../../utils/formatters';
import { INDEX_CATEGORY_LABELS } from '../../utils/constants';

const CATEGORY_FILTERS = [
  { value: 'all', label: 'All' },
  { value: 'broad_market', label: 'Broad' },
  { value: 'sectoral', label: 'Sector' },
  { value: 'thematic', label: 'Theme' },
  { value: 'strategy', label: 'Strategy' },
];

const SORTS = [
  { value: 'default', label: 'Default' },
  { value: 'gainers', label: 'Top gainers' },
  { value: 'losers', label: 'Top losers' },
  { value: 'name', label: 'Name A–Z' },
];

function Breadth({ indices }) {
  const up = indices.filter((i) => i.change_pct > 0).length;
  const down = indices.filter((i) => i.change_pct < 0).length;
  const flat = indices.length - up - down;
  const total = indices.length || 1;
  return (
    <div>
      <div className="flex items-center justify-between text-2xs num mb-1.5">
        <span className="text-up">{up} advancing</span>
        {flat > 0 && <span className="text-ink-faint">{flat} flat</span>}
        <span className="text-down">{down} declining</span>
      </div>
      <div className="flex h-1.5 rounded-full overflow-hidden bg-surface-3 gap-px">
        <div className="bg-up" style={{ width: `${(up / total) * 100}%` }} />
        <div className="bg-ink-ghost" style={{ width: `${(flat / total) * 100}%` }} />
        <div className="bg-down" style={{ width: `${(down / total) * 100}%` }} />
      </div>
    </div>
  );
}

function IndexList({ indices, meta, active, onSelect, loading }) {
  const [query, setQuery] = useState('');
  const [cat, setCat] = useState('all');
  const [sort, setSort] = useState('default');

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    let r = indices.filter((i) => {
      if (cat !== 'all' && meta[i.index_id]?.sector_category !== cat) return false;
      if (!q) return true;
      return i.index_id.toLowerCase().includes(q) || indexLabel(meta, i.index_id).toLowerCase().includes(q);
    });
    if (sort === 'gainers') r = [...r].sort((a, b) => (b.change_pct ?? 0) - (a.change_pct ?? 0));
    if (sort === 'losers') r = [...r].sort((a, b) => (a.change_pct ?? 0) - (b.change_pct ?? 0));
    if (sort === 'name') r = [...r].sort((a, b) => indexLabel(meta, a.index_id).localeCompare(indexLabel(meta, b.index_id)));
    return r;
  }, [indices, meta, query, cat, sort]);

  return (
    <section className="card flex flex-col min-w-0 min-h-0 lg:sticky lg:top-6 lg:h-[calc(100vh-56px-48px)]">
      <div className="p-4 space-y-3 border-b border-line">
        <Breadth indices={indices} />
        <div className="relative">
          <Search className="w-4 h-4 text-ink-faint absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={`Search ${indices.length} indices`}
            className="input pl-9 h-8 text-[13px]"
          />
        </div>
        <div className="flex items-center justify-between gap-2">
          <Segmented size="xs" options={CATEGORY_FILTERS} value={cat} onChange={setCat} className="overflow-x-auto" />
        </div>
        <div className="flex items-center justify-between text-2xs text-ink-faint">
          <span>{rows.length} shown</span>
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value)}
            className="bg-transparent text-ink-muted hover:text-ink focus:outline-none cursor-pointer text-2xs"
            aria-label="Sort"
          >
            {SORTS.map((s) => <option key={s.value} value={s.value} className="bg-surface-2">{s.label}</option>)}
          </select>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto min-h-[240px]">
        {loading ? <RowSkeleton rows={8} /> : !rows.length ? (
          <EmptyState icon={Search} title="No match" message="Try a different search or category." />
        ) : (
          <ul className="py-1">
            {rows.map((i) => {
              const sel = i.index_id === active;
              return (
                <li key={i.index_id}>
                  <button
                    onClick={() => onSelect(i.index_id)}
                    className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors border-l-2 ${
                      sel ? 'bg-brand/[0.07] border-brand' : 'border-transparent hover:bg-surface-2/70'
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div className={`text-[13px] font-medium truncate ${sel ? 'text-ink' : 'text-ink-muted'}`}>
                        {indexLabel(meta, i.index_id)}
                      </div>
                      <div className="text-2xs text-ink-ghost truncate">
                        {INDEX_CATEGORY_LABELS[meta[i.index_id]?.sector_category] ?? i.index_id}
                        {meta[i.index_id]?.has_options && <span className="text-brand/80"> · F&O</span>}
                      </div>
                    </div>
                    <div className="text-right flex-shrink-0">
                      <div className="text-[13px] num text-ink">
                        {i.ltp != null ? <FlashPrice value={i.ltp} formatter={formatPrice} /> : '—'}
                      </div>
                      <Delta value={i.change_pct} showIcon={false} className="text-2xs" />
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}

function QuoteHeader({ quote, info }) {
  const stats = [
    ['Open', quote?.open],
    ['High', quote?.high],
    ['Low', quote?.low],
    ['Prev close', quote?.prev_close],
  ];
  return (
    <section className="card px-5 py-4">
      <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold text-ink tracking-tight truncate">{info?.display_name ?? quote?.index_id}</h2>
            {info?.exchange && <Badge tone="neutral" size="xs">{info.exchange}</Badge>}
            {info?.has_options && <Badge tone="brand" size="xs">F&O</Badge>}
          </div>
          {info?.description && <p className="text-xs text-ink-faint mt-0.5 truncate max-w-md">{info.description}</p>}
          <div className="flex items-baseline gap-3 mt-2">
            <span className="text-[28px] leading-8 font-semibold num text-ink">
              {quote?.ltp != null ? <FlashPrice value={quote.ltp} formatter={formatPrice} /> : '—'}
            </span>
            <Delta value={quote?.change_pct} flash className="text-sm font-medium" />
            {quote?.change != null && (
              <span className="text-sm num text-ink-faint">{quote.change > 0 ? '+' : ''}{formatPrice(quote.change)}</span>
            )}
          </div>
        </div>
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-8 gap-y-3">
          {stats.map(([l, v]) => (
            <div key={l}>
              <dt className="label">{l}</dt>
              <dd className="text-[13px] font-medium num text-ink mt-1">{v != null ? formatPrice(v) : '—'}</dd>
            </div>
          ))}
          {quote?.volume ? (
            <div><dt className="label">Volume</dt><dd className="text-[13px] font-medium num text-ink mt-1">{formatLargeNumber(quote.volume)}</dd></div>
          ) : null}
        </dl>
      </div>
    </section>
  );
}

function OpenInterestTab({ indexId }) {
  const { data, loading, error, lastUpdated, refetch } = usePolling(
    () => api.getOptionsChain(indexId), 60_000,
  );
  // usePolling reads the latest fn via ref — refetch immediately on index change
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { refetch(); }, [indexId]);

  return (
    <Card
      title="Open interest"
      subtitle="Where option writers are positioned"
      icon={Layers}
      actions={<RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />}
    >
      {loading && !data ? <Skeleton height="340px" /> : <OptionsHeatmap data={data?.index_id === indexId ? data : null} />}
    </Card>
  );
}

export default function MarketPage() {
  const meta = useIndexMeta();
  const { prices: { data, loading } } = useDataStore();
  const indices = data?.indices ?? [];
  const [active, setActive] = useTabParam('NIFTY50', 'index');
  const [tab, setTab] = useTabParam('chart');

  const quote = indices.find((i) => i.index_id === active);
  const info = meta[active];
  const hasOptions = !!info?.has_options;
  const shownTab = tab === 'oi' && !hasOptions ? 'chart' : tab;

  return (
    <Page>
      <div className="grid grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)] gap-5 items-start">
        <IndexList indices={indices} meta={meta} active={active} onSelect={setActive} loading={loading && !data} />

        <div className="space-y-5 min-w-0">
          <QuoteHeader quote={quote} info={info} />

          {hasOptions && (
            <Tabs
              value={shownTab}
              onChange={setTab}
              tabs={[
                { id: 'chart', label: 'Price chart', icon: CandlestickChart },
                { id: 'oi', label: 'Open interest', icon: Layers },
              ]}
            />
          )}

          {shownTab === 'chart' && <ChartPanel indexId={active} title="Price chart" height={440} />}
          {shownTab === 'oi' && <OpenInterestTab indexId={active} />}
        </div>
      </div>
    </Page>
  );
}
