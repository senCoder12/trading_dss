import { useMemo, useState } from 'react';
import { Layers } from 'lucide-react';
import { Tabs } from '../common/Tabs';
import { Badge } from '../common/Badge';
import { Delta } from '../common/Delta';
import { Segmented } from '../common/Segmented';
import { Spinner } from '../common/Spinner';
import { EmptyState } from '../common/EmptyState';
import { formatNumber } from '../../utils/formatters';

// ── Helpers ──────────────────────────────────────────────────────────────────

const isNum = (v) => typeof v === 'number' && !Number.isNaN(v);
const num = (v, dec = 2) => (isNum(v) ? formatNumber(v, dec) : '—');
const fmtK = (v) => {
  if (!isNum(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e7) return `${(v / 1e7).toFixed(2)}Cr`;
  if (a >= 1e5) return `${(v / 1e5).toFixed(2)}L`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return String(v);
};
const signColour = (v) => (!isNum(v) || v === 0 ? 'text-ink-faint' : v > 0 ? 'text-up' : 'text-down');

const scoreTone = (v) => (!isNum(v) ? 'neutral' : v >= 70 ? 'up' : v >= 50 ? 'warn' : 'down');
const TONE_TEXT = { up: 'text-up', warn: 'text-warn', down: 'text-down', neutral: 'text-ink-faint' };
const TONE_BAR = { up: 'bg-up', warn: 'bg-warn', down: 'bg-down', neutral: 'bg-ink-ghost' };
const TONE_BADGE = { up: 'success', warn: 'warning', down: 'danger', neutral: 'neutral' };

const SCORE_COMPONENTS = [
  { key: 'oi_score', label: 'Open interest', hint: 'Depth of outstanding option positions' },
  { key: 'oi_change_score', label: 'OI change', hint: 'Fresh positioning since previous session' },
  { key: 'volume_score', label: 'Volume', hint: 'Traded option volume vs. average' },
  { key: 'price_move_score', label: 'Price move', hint: 'Underlying momentum / range expansion' },
  { key: 'iv_change_score', label: 'IV change', hint: 'Shift in implied volatility' },
  { key: 'spread_score', label: 'Spread', hint: 'Bid-ask tightness — higher is cheaper to trade' },
];

function formatExpiry(exp) {
  const d = new Date(exp);
  if (Number.isNaN(d.getTime())) return exp;
  return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
}

// ── Score breakdown ──────────────────────────────────────────────────────────

function ScoreBreakdown({ stock }) {
  return (
    <div className="card divide-y divide-line">
      {SCORE_COMPONENTS.map(({ key, label, hint }) => {
        const v = stock[key];
        const tone = scoreTone(v);
        return (
          <div key={key} className="flex items-center gap-4 px-5 py-3.5">
            <div className="w-44 min-w-0 flex-shrink-0">
              <div className="text-[13px] font-medium text-ink">{label}</div>
              <div className="text-2xs text-ink-faint truncate" title={hint}>{hint}</div>
            </div>
            <div className="flex-1 h-1.5 rounded-full bg-surface-3 overflow-hidden">
              <div className={`h-full rounded-full ${TONE_BAR[tone]}`} style={{ width: `${isNum(v) ? Math.max(0, Math.min(100, v)) : 0}%` }} />
            </div>
            <span className={`w-10 text-right text-[15px] font-semibold num ${TONE_TEXT[tone]}`}>
              {isNum(v) ? v.toFixed(0) : '—'}
            </span>
          </div>
        );
      })}
      <div className="flex items-center gap-4 px-5 py-2.5 text-2xs text-ink-faint">
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-up" />≥ 70 strong</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-warn" />50–69 fair</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-down" />&lt; 50 weak</span>
      </div>
    </div>
  );
}

// ── Option chain ─────────────────────────────────────────────────────────────

function MiniStat({ label, value, sub, tone = 'text-ink' }) {
  return (
    <div className="card px-4 py-3 min-w-0">
      <div className="label truncate">{label}</div>
      <div className={`mt-1 text-[17px] font-semibold num truncate ${tone}`}>{value}</div>
      {sub && <div className="text-2xs text-ink-faint truncate">{sub}</div>}
    </div>
  );
}

const DEFAULT_ROWS = 11;

function nearestIndex(strikes, target) {
  if (!isNum(target)) return Math.floor(strikes.length / 2);
  let best = 0;
  let bestDist = Infinity;
  strikes.forEach((s, i) => {
    const d = Math.abs((s.strike ?? 0) - target);
    if (d < bestDist) { bestDist = d; best = i; }
  });
  return best;
}

function ChainTable({ chainData }) {
  const [moreAbove, setMoreAbove] = useState(0);
  const [moreBelow, setMoreBelow] = useState(0);
  const strikes = chainData?.strikes ?? [];

  const maxCe = useMemo(() => Math.max(...strikes.map((s) => s.ce_oi || 0), 1), [strikes]);
  const maxPe = useMemo(() => Math.max(...strikes.map((s) => s.pe_oi || 0), 1), [strikes]);

  if (!strikes.length) {
    return <EmptyState icon={Layers} title="No option chain" message="No strikes returned for this expiry." />;
  }

  const atmTarget = chainData.atm_strike ?? chainData.spot_price;
  const atmIdx = nearestIndex(strikes, atmTarget);
  const half = Math.floor(DEFAULT_ROWS / 2);
  const start = Math.max(0, atmIdx - half - moreAbove);
  const end = Math.min(strikes.length, atmIdx + half + 1 + moreBelow);
  const visible = strikes.slice(start, end);
  const spot = chainData.spot_price;
  const maxPain = chainData.max_pain_strike;

  const th = 'px-3 h-8 text-2xs font-medium uppercase tracking-wider text-ink-faint bg-surface border-b border-line whitespace-nowrap';
  const td = 'px-3 py-2 border-b border-line/60';

  return (
    <div className="card overflow-hidden">
      {start > 0 && (
        <button onClick={() => setMoreAbove((p) => p + 5)} className="w-full h-8 text-2xs font-medium text-ink-faint hover:text-brand hover:bg-surface-2 border-b border-line transition-colors">
          Show 5 more strikes above
        </button>
      )}
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-0 text-[12.5px] num min-w-[720px]">
          <thead>
            <tr>
              <th className={`${th} text-right text-down/80`}>OI</th>
              <th className={`${th} text-right`}>OI Chg</th>
              <th className={`${th} text-right`}>Volume</th>
              <th className={`${th} text-right`}>IV</th>
              <th className={`${th} text-right text-down/80`}>Call LTP</th>
              <th className={`${th} text-center bg-surface-2 border-x`}>Strike</th>
              <th className={`${th} text-left text-up/80`}>Put LTP</th>
              <th className={`${th} text-left`}>IV</th>
              <th className={`${th} text-left`}>Volume</th>
              <th className={`${th} text-left`}>OI Chg</th>
              <th className={`${th} text-left text-up/80`}>OI</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((s) => {
              const idx = strikes.indexOf(s);
              const isAtm = idx === atmIdx;
              const isPain = isNum(maxPain) && Math.abs(s.strike - maxPain) < 0.01;
              const ceItm = isNum(spot) && s.strike < spot && !isAtm;
              const peItm = isNum(spot) && s.strike > spot && !isAtm;
              const ceBar = ((s.ce_oi || 0) / maxCe) * 100;
              const peBar = ((s.pe_oi || 0) / maxPe) * 100;
              const ce = ceItm ? 'bg-down/[0.04]' : '';
              const pe = peItm ? 'bg-up/[0.04]' : '';
              return (
                <tr key={s.strike} className={`hover:bg-surface-2/70 transition-colors ${isAtm ? 'bg-warn/[0.06]' : ''}`}>
                  <td className={`${td} ${ce} relative text-right text-ink-muted`}>
                    <span aria-hidden className="absolute bottom-1 right-3 h-[3px] rounded-full bg-down/45" style={{ width: `calc((100% - 1.5rem) * ${ceBar / 100})` }} />
                    {fmtK(s.ce_oi)}
                  </td>
                  <td className={`${td} ${ce} text-right ${signColour(s.ce_oi_change)}`}>{isNum(s.ce_oi_change) ? `${s.ce_oi_change > 0 ? '+' : ''}${fmtK(s.ce_oi_change)}` : '—'}</td>
                  <td className={`${td} ${ce} text-right text-ink-faint`}>{fmtK(s.ce_volume)}</td>
                  <td className={`${td} ${ce} text-right text-violet`}>{isNum(s.ce_iv) && s.ce_iv ? `${s.ce_iv.toFixed(1)}%` : '—'}</td>
                  <td className={`${td} ${ce} text-right text-ink font-medium`}>{num(s.ce_ltp)}</td>
                  <td className={`${td} text-center font-semibold border-x border-line ${isAtm ? 'text-warn bg-warn/[0.08]' : 'text-ink bg-surface-2/60'}`}>
                    <span className="inline-flex items-center gap-1.5">
                      {isAtm && <span className="w-1.5 h-1.5 rounded-full bg-warn" />}
                      {num(s.strike, 0)}
                      {isPain && <span className="text-[10px] font-semibold text-warn/80" title="Max pain">MP</span>}
                    </span>
                  </td>
                  <td className={`${td} ${pe} text-left text-ink font-medium`}>{num(s.pe_ltp)}</td>
                  <td className={`${td} ${pe} text-left text-violet`}>{isNum(s.pe_iv) && s.pe_iv ? `${s.pe_iv.toFixed(1)}%` : '—'}</td>
                  <td className={`${td} ${pe} text-left text-ink-faint`}>{fmtK(s.pe_volume)}</td>
                  <td className={`${td} ${pe} text-left ${signColour(s.pe_oi_change)}`}>{isNum(s.pe_oi_change) ? `${s.pe_oi_change > 0 ? '+' : ''}${fmtK(s.pe_oi_change)}` : '—'}</td>
                  <td className={`${td} ${pe} relative text-left text-ink-muted`}>
                    <span aria-hidden className="absolute bottom-1 left-3 h-[3px] rounded-full bg-up/45" style={{ width: `calc((100% - 1.5rem) * ${peBar / 100})` }} />
                    {fmtK(s.pe_oi)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {end < strikes.length && (
        <button onClick={() => setMoreBelow((p) => p + 5)} className="w-full h-8 text-2xs font-medium text-ink-faint hover:text-brand hover:bg-surface-2 border-t border-line transition-colors">
          Show 5 more strikes below
        </button>
      )}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 px-4 py-2.5 border-t border-line text-2xs text-ink-faint">
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-warn/40" />ATM</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-down/25" />Call ITM</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-up/25" />Put ITM</span>
        <span className="flex items-center gap-1.5"><span className="font-semibold text-warn/80">MP</span>Max pain</span>
        <span className="ml-auto num">{visible.length} of {strikes.length} strikes</span>
      </div>
    </div>
  );
}

function OptionChainTab({ chainData, chainLoading, selectedExpiry, onExpiryChange }) {
  const expiries = chainData?.available_expiries ?? [];
  const pcr = chainData?.pcr;
  const pcrHint = !isNum(pcr) ? null : pcr > 1.3 ? 'Put-heavy' : pcr < 0.7 ? 'Call-heavy' : 'Balanced';
  const pcrTone = !isNum(pcr) ? 'text-ink' : pcr > 1.3 ? 'text-up' : pcr < 0.7 ? 'text-down' : 'text-ink';

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        {expiries.length > 0 ? (
          <Segmented
            size="xs"
            ariaLabel="Expiry"
            options={expiries.map((e) => ({ value: e, label: formatExpiry(e) }))}
            value={selectedExpiry}
            onChange={onExpiryChange}
          />
        ) : <span />}
        <div className="flex items-center gap-3 text-xs text-ink-faint num">
          {chainLoading && chainData && <Spinner size="xs" />}
          <span>Spot <span className="text-ink font-medium">{num(chainData?.spot_price)}</span></span>
          <span className="h-3 border-l border-line" />
          <span>Lot <span className="text-ink font-medium">{chainData?.lot_size ?? '—'}</span></span>
        </div>
      </div>

      {chainLoading && !chainData ? (
        <div className="flex justify-center py-16"><Spinner /></div>
      ) : !chainData ? (
        <EmptyState icon={Layers} title="No option chain available" message="The chain for this stock could not be fetched." />
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <MiniStat label="Call OI" value={fmtK(chainData.total_ce_oi)} tone="text-down" />
            <MiniStat label="Put OI" value={fmtK(chainData.total_pe_oi)} tone="text-up" />
            <MiniStat label="Put / call ratio" value={isNum(pcr) ? pcr.toFixed(2) : '—'} sub={pcrHint} tone={pcrTone} />
            <MiniStat
              label="Max pain"
              value={num(chainData.max_pain_strike, 0)}
              tone="text-warn"
              sub={isNum(chainData.max_pain_strike) && isNum(chainData.spot_price) && chainData.spot_price
                ? `${(((chainData.max_pain_strike - chainData.spot_price) / chainData.spot_price) * 100).toFixed(2)}% from spot`
                : null}
            />
          </div>
          <ChainTable key={selectedExpiry ?? 'default'} chainData={chainData} />
        </>
      )}
    </div>
  );
}

// ── Panel ────────────────────────────────────────────────────────────────────

export default function RightPanel({
  symbol,
  universeData,
  chainData,
  chainLoading,
  selectedExpiry,
  onExpiryChange,
}) {
  const [tab, setTab] = useState('score');
  const stock = useMemo(
    () => universeData?.active_universe?.find((s) => s.symbol === symbol) ?? null,
    [universeData?.active_universe, symbol],
  );

  if (!stock) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <EmptyState title={`${symbol} is no longer in the universe`} message="It may have dropped out on the latest refresh." />
      </div>
    );
  }

  const score = stock.composite_score;
  const tierTone = stock.tier === 'PRIORITY' ? 'success' : stock.tier === 'ACTIVE' ? 'brand' : 'neutral';

  return (
    <div className="flex-1 min-w-0 flex flex-col min-h-0">
      {/* Header */}
      <div className="px-6 pt-5 flex-shrink-0">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2.5">
              <h2 className="text-xl font-semibold text-ink tracking-tight">{symbol}</h2>
              {stock.tier && <Badge tone={tierTone}>{stock.tier.charAt(0) + stock.tier.slice(1).toLowerCase()}</Badge>}
            </div>
            <div className="text-[13px] text-ink-faint mt-0.5 truncate">
              {[stock.company_name, stock.sector].filter(Boolean).join(' · ') || '—'}
            </div>
          </div>
          <div className="text-right">
            <div className="text-xl font-semibold num text-ink">{isNum(stock.ltp) ? `₹${formatNumber(stock.ltp, 2)}` : '—'}</div>
            {isNum(stock.price_change_pct) && <Delta value={stock.price_change_pct} className="text-[13px] font-medium justify-end" />}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 mt-3">
          <Badge tone="neutral">Rank #{stock.rank ?? '—'}</Badge>
          <Badge tone={TONE_BADGE[scoreTone(score)]}>Score {isNum(score) ? score.toFixed(1) : '—'}</Badge>
          {stock.lot_size != null && <Badge tone="neutral">Lot {Number(stock.lot_size).toLocaleString('en-IN')}</Badge>}
        </div>

        <Tabs
          className="mt-4 -mx-6 px-6"
          value={tab}
          onChange={setTab}
          tabs={[
            { id: 'score', label: 'Score breakdown' },
            { id: 'chain', label: 'Option chain' },
          ]}
        />
      </div>

      {/* Body */}
      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-5">
        {tab === 'score' ? (
          <ScoreBreakdown stock={stock} />
        ) : (
          <OptionChainTab
            chainData={chainData}
            chainLoading={chainLoading}
            selectedExpiry={selectedExpiry}
            onExpiryChange={onExpiryChange}
          />
        )}
      </div>
    </div>
  );
}
