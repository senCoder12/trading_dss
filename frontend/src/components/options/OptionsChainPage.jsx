import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { BarChart2, AlertTriangle } from 'lucide-react';
import { usePolling } from '../../hooks/usePolling';
import { useWebSocket } from '../../hooks/useWebSocket';
import { api } from '../../api/client';
import { FullPage } from '../layout/Page';
import { Segmented } from '../common/Segmented';
import { Spinner } from '../common/Spinner';
import { EmptyState } from '../common/EmptyState';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { FlashPrice } from '../common/FlashPrice';
import { formatNumber, formatPrice } from '../../utils/formatters';

// ── Formatters ───────────────────────────────────────────────────────────────

const fmtK = (v) => {
  if (v == null || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e7) return `${(v / 1e7).toFixed(2)}Cr`;
  if (a >= 1e5) return `${(v / 1e5).toFixed(2)}L`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return String(v);
};
const fmtSignedK = (v) => (v == null ? '—' : `${v > 0 ? '+' : ''}${fmtK(v)}`);
const fmtP = (v, dec = 1) => (v == null || Number.isNaN(v) ? '—' : `${v.toFixed(dec)}%`);
const fmtSignedP = (v, dec = 1) => (v == null || Number.isNaN(v) ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(dec)}%`);
const fmtLtp = (v) => (v == null ? '—' : formatNumber(v, 2));

const signColour = (v) => (v == null || v === 0 ? 'text-ink-faint' : v > 0 ? 'text-up' : 'text-down');

// ── Buildup classification (price change × OI change) ───────────────────────

const BUILDUP = {
  LB: { label: 'LB', full: 'Long buildup', cls: 'bg-up/15 text-up' },
  SB: { label: 'SB', full: 'Short buildup', cls: 'bg-down/15 text-down' },
  LU: { label: 'LU', full: 'Long unwinding', cls: 'bg-warn/15 text-warn' },
  SC: { label: 'SC', full: 'Short covering', cls: 'bg-info/15 text-info' },
};

const getBuildup = (pChg, oiChg) => {
  if (!pChg || !oiChg) return null;
  if (pChg > 0 && oiChg > 0) return BUILDUP.LB;
  if (pChg < 0 && oiChg > 0) return BUILDUP.SB;
  if (pChg < 0 && oiChg < 0) return BUILDUP.LU;
  if (pChg > 0 && oiChg < 0) return BUILDUP.SC;
  return null;
};

function BuildupTag({ b }) {
  if (!b) return null;
  return (
    <span title={b.full} className={`inline-flex items-center h-4 px-1 rounded text-[10px] font-semibold leading-none ${b.cls}`}>
      {b.label}
    </span>
  );
}

// ── Column definitions ───────────────────────────────────────────────────────
// kind: 'oi' renders the depth bar, 'ltp' renders the buildup tag.

const CE_DETAILED = [
  { key: 'ce_oi', label: 'OI', fmt: fmtK, kind: 'oi' },
  { key: 'ce_oi_change', label: 'OI Chg', fmt: fmtSignedK, colour: signColour },
  { key: 'ce_oi_p_change', label: 'OI %', fmt: fmtSignedP, colour: signColour },
  { key: 'ce_volume', label: 'Volume', fmt: fmtK, cls: 'text-ink-faint' },
  { key: 'ce_iv', label: 'IV', fmt: (v) => (v ? fmtP(v) : '—'), cls: 'text-violet' },
  { key: 'ce_ltp', label: 'LTP', fmt: fmtLtp, cls: 'text-ink font-medium', kind: 'ltp' },
  { key: 'ce_ltp_p_change', label: 'Chg %', fmt: fmtSignedP, colour: signColour },
];

const PE_DETAILED = [
  { key: 'pe_ltp_p_change', label: 'Chg %', fmt: fmtSignedP, colour: signColour },
  { key: 'pe_ltp', label: 'LTP', fmt: fmtLtp, cls: 'text-ink font-medium', kind: 'ltp' },
  { key: 'pe_iv', label: 'IV', fmt: (v) => (v ? fmtP(v) : '—'), cls: 'text-violet' },
  { key: 'pe_volume', label: 'Volume', fmt: fmtK, cls: 'text-ink-faint' },
  { key: 'pe_oi_p_change', label: 'OI %', fmt: fmtSignedP, colour: signColour },
  { key: 'pe_oi_change', label: 'OI Chg', fmt: fmtSignedK, colour: signColour },
  { key: 'pe_oi', label: 'OI', fmt: fmtK, kind: 'oi' },
];

const COMPACT_KEYS = new Set(['ce_oi', 'ce_oi_change', 'ce_ltp', 'pe_ltp', 'pe_oi_change', 'pe_oi']);
const CE_COMPACT = CE_DETAILED.filter((c) => COMPACT_KEYS.has(c.key));
const PE_COMPACT = PE_DETAILED.filter((c) => COMPACT_KEYS.has(c.key));

// ── F&O index list (those with options) ──────────────────────────────────────

const FO_INDICES = [
  { value: 'NIFTY50', label: 'Nifty' },
  { value: 'BANKNIFTY', label: 'Bank Nifty' },
  { value: 'FINNIFTY', label: 'Fin Nifty' },
  { value: 'MIDCPNIFTY', label: 'Midcap' },
  { value: 'SENSEX', label: 'Sensex' },
];

const RANGE_OPTIONS = [
  { value: 10, label: '±10', title: '10 strikes either side of ATM' },
  { value: 20, label: '±20', title: '20 strikes either side of ATM' },
  { value: 'all', label: 'All strikes' },
];

const VIEW_OPTIONS = [
  { value: 'compact', label: 'Compact' },
  { value: 'detailed', label: 'Detailed' },
];

// ── Helpers ──────────────────────────────────────────────────────────────────

function atmIndexOf(strikes, spot) {
  const i = strikes.findIndex((s) => s.is_atm);
  if (i >= 0 || spot == null) return i;
  let best = -1;
  let bestDist = Infinity;
  strikes.forEach((s, idx) => {
    const d = Math.abs(s.strike - spot);
    if (d < bestDist) { bestDist = d; best = idx; }
  });
  return best;
}

/** Heuristic bias from PCR, max-pain distance and buildup near ATM (unchanged scoring). */
function computeBias(data, strikes) {
  if (!data || !strikes.length) return null;
  const pcr = data.pcr || 1;
  const spot = data.spot || 0;
  const maxPain = data.max_pain || 0;
  const atmIndex = strikes.findIndex((s) => s.is_atm);
  const window = strikes.slice(Math.max(0, atmIndex - 3), Math.min(strikes.length, atmIndex + 4));
  let ceLB = 0, ceSB = 0, peLB = 0, peSB = 0;
  window.forEach((s) => {
    if (s.ce_ltp_p_change > 0 && s.ce_oi_change > 0) ceLB++;
    if (s.ce_ltp_p_change < 0 && s.ce_oi_change > 0) ceSB++;
    if (s.pe_ltp_p_change > 0 && s.pe_oi_change > 0) peLB++;
    if (s.pe_ltp_p_change < 0 && s.pe_oi_change > 0) peSB++;
  });

  let score = 50;
  const reasons = [];
  if (pcr > 1.4) { score += 15; reasons.push('PCR reversal'); }
  else if (pcr < 0.6) { score -= 15; reasons.push('Overbought'); }
  if (spot > 0 && maxPain > 0) {
    const painDiff = ((spot - maxPain) / spot) * 100;
    if (painDiff > 0.5) { score -= 10; reasons.push('Above max pain'); }
    else if (painDiff < -0.5) { score += 10; reasons.push('Below max pain'); }
  }
  if (ceLB > ceSB) { score += 5; } else if (ceSB > ceLB) { score -= 10; reasons.push('Call writing'); }
  if (peLB > peSB) { score += 10; reasons.push('Put buying'); } else if (peSB > peLB) { score -= 5; reasons.push('Put writing'); }

  const verdict = score > 75 ? 'Strongly bullish' : score > 60 ? 'Bullish' : score < 25 ? 'Strongly bearish' : score < 40 ? 'Bearish' : 'Neutral';
  const tone = score > 60 ? 'up' : score < 40 ? 'down' : 'neutral';
  return { score: Math.max(0, Math.min(100, score)), verdict, tone, reasons };
}

function formatExpiry(exp) {
  if (!exp) return '—';
  const d = new Date(`${exp}T15:30:00+05:30`);
  if (Number.isNaN(d.getTime())) return exp;
  const label = d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', timeZone: 'Asia/Kolkata' });
  const days = Math.ceil((d.getTime() - Date.now()) / 86_400_000);
  return days >= 0 ? `${label} · ${days}d` : label;
}

const TONE_TEXT = { up: 'text-up', down: 'text-down', neutral: 'text-ink', warn: 'text-warn' };
const TONE_BG = { up: 'bg-up', down: 'bg-down', neutral: 'bg-ink-faint', warn: 'bg-warn' };

// ── Stat tile (local: needs custom body for bars) ────────────────────────────

function Tile({ label, right, value, valueClass = 'text-ink', sub, children }) {
  return (
    <div className="card px-4 py-3 min-w-0">
      <div className="flex items-center justify-between gap-2">
        <span className="label truncate">{label}</span>
        {right}
      </div>
      {value != null && (
        <div className={`mt-1 text-lg xl:text-xl leading-7 font-semibold tracking-tight num truncate ${valueClass}`}>{value}</div>
      )}
      {children}
      {sub != null && <div className="mt-1 text-xs text-ink-faint num truncate">{sub}</div>}
    </div>
  );
}

function StatRow({ data, liveSpot, strikes, bias }) {
  const spot = liveSpot ?? data.spot;
  const atm = strikes[atmIndexOf(strikes, spot)]?.strike;
  const painDist = spot && data.max_pain ? ((data.max_pain - spot) / spot) * 100 : null;

  const pcr = data.pcr;
  const pcrTone = pcr == null ? 'neutral' : pcr > 1.3 ? 'up' : pcr < 0.7 ? 'down' : 'neutral';
  const pcrHint = pcr == null ? '—' : pcr > 1.3 ? 'Put-heavy · supportive' : pcr < 0.7 ? 'Call-heavy · capped' : 'Balanced positioning';
  const pcrPct = pcr == null ? 0 : ((Math.min(Math.max(pcr, 0.2), 2.5) - 0.2) / 2.3) * 100;

  const ce = data.total_ce_oi || 0;
  const pe = data.total_pe_oi || 0;
  const cePct = ce + pe ? (ce / (ce + pe)) * 100 : 50;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
      <Tile
        label="Spot"
        right={liveSpot != null && (
          <span className="flex items-center gap-1 text-2xs text-up"><span className="w-1.5 h-1.5 rounded-full bg-up animate-pulse" />Live</span>
        )}
        value={spot != null ? <FlashPrice value={spot} formatter={formatPrice} /> : '—'}
        sub={`ATM ${atm != null ? formatNumber(atm, 0) : '—'}${data.iv_atm ? ` · IV ${data.iv_atm.toFixed(2)}%` : ''}`}
      />
      <Tile
        label="Max pain"
        value={data.max_pain != null ? formatNumber(data.max_pain, 0) : '—'}
        valueClass="text-warn"
        sub={painDist != null ? `${painDist > 0 ? '+' : ''}${painDist.toFixed(2)}% from spot` : null}
      />
      <Tile label="Put / call ratio" value={pcr != null ? pcr.toFixed(2) : '—'} valueClass={TONE_TEXT[pcrTone]} sub={pcrHint}>
        <div className="mt-2 h-1 rounded-full bg-surface-3 overflow-hidden">
          <div className={`h-full rounded-full ${TONE_BG[pcrTone]}`} style={{ width: `${pcrPct}%` }} />
        </div>
      </Tile>
      <Tile label="Call vs put OI">
        <div className="mt-2 flex items-baseline justify-between text-[13px] num">
          <span title="Total call OI"><span className="text-ink-faint text-xs mr-1 hidden xl:inline">Calls</span><span className="text-down font-semibold">{fmtK(ce)}</span></span>
          <span title="Total put OI"><span className="text-up font-semibold">{fmtK(pe)}</span><span className="text-ink-faint text-xs ml-1 hidden xl:inline">Puts</span></span>
        </div>
        <div className="mt-2 flex h-1.5 rounded-full overflow-hidden bg-surface-3">
          <div className="h-full bg-down/70" style={{ width: `${cePct}%` }} />
          <div className="h-full bg-up/70 flex-1" />
        </div>
      </Tile>
      <Tile
        label="Bias"
        right={bias && <span className="text-2xs text-ink-faint num">{bias.score}/100</span>}
        value={bias?.verdict ?? '—'}
        valueClass={TONE_TEXT[bias?.tone ?? 'neutral']}
        sub={bias?.reasons?.length ? bias.reasons.slice(0, 3).join(' · ') : 'No strong OI signal'}
      >
        {bias && (
          <div className="mt-1.5 h-1 rounded-full bg-surface-3 overflow-hidden">
            <div className={`h-full rounded-full ${TONE_BG[bias.tone]}`} style={{ width: `${bias.score}%` }} />
          </div>
        )}
      </Tile>
    </div>
  );
}

// ── Table cells ──────────────────────────────────────────────────────────────

function ChainCell({ col, row, side, max, shade }) {
  const val = row[col.key];
  const isZero = val == null || val === 0;
  const colour = isZero ? 'text-ink-ghost' : col.colour ? col.colour(val) : (col.cls ?? 'text-ink-muted');
  const buildup = col.kind === 'ltp'
    ? getBuildup(row[`${side}_ltp_p_change`], row[`${side}_oi_change`])
    : null;
  const barPct = col.kind === 'oi' && val && max ? Math.min((val / max) * 100, 100) : 0;
  const align = side === 'ce' ? 'text-right' : 'text-left';

  return (
    <td className={`relative px-3 py-2 border-b border-line/60 ${align} ${shade}`}>
      {barPct > 0 && (
        <span
          aria-hidden
          className={`absolute bottom-1 h-[3px] rounded-full ${side === 'ce' ? 'right-3 bg-down/45' : 'left-3 bg-up/45'}`}
          style={{ width: `calc((100% - 1.5rem) * ${barPct / 100})` }}
        />
      )}
      <span className={`relative inline-flex items-center gap-1.5 ${side === 'ce' ? 'flex-row' : 'flex-row-reverse'} ${colour}`}>
        {buildup && <BuildupTag b={buildup} />}
        <span>{isZero && col.kind !== 'oi' ? '—' : col.fmt(val)}</span>
      </span>
    </td>
  );
}

const GROUP_BG = {
  ce: 'linear-gradient(rgb(240 82 90 / 0.08), rgb(240 82 90 / 0.08)), #10151d',
  pe: 'linear-gradient(rgb(34 195 142 / 0.08), rgb(34 195 142 / 0.08)), #10151d',
};

function ChainTable({ rows, ceCols, peCols, spot, maxCeOI, maxPeOI, atmRef }) {
  return (
    <table className="w-full border-separate border-spacing-0 text-[12.5px] num min-w-[640px]">
      <thead className="sticky top-0 z-10">
        <tr>
          <th colSpan={ceCols.length} style={{ background: GROUP_BG.ce }} className="h-8 text-center text-2xs font-semibold uppercase tracking-wider text-down border-b border-line">
            Calls
          </th>
          <th className="h-8 w-28 text-center text-2xs font-semibold uppercase tracking-wider text-ink-muted bg-surface-2 border-x border-b border-line">
            Strike
          </th>
          <th colSpan={peCols.length} style={{ background: GROUP_BG.pe }} className="h-8 text-center text-2xs font-semibold uppercase tracking-wider text-up border-b border-line">
            Puts
          </th>
        </tr>
        <tr>
          {ceCols.map((c) => (
            <th key={c.key} className="px-3 h-8 text-right text-2xs font-medium uppercase tracking-wider text-ink-faint bg-surface border-b border-line whitespace-nowrap">
              {c.label}
            </th>
          ))}
          <th className="bg-surface-2 border-x border-b border-line" />
          {peCols.map((c) => (
            <th key={c.key} className="px-3 h-8 text-left text-2xs font-medium uppercase tracking-wider text-ink-faint bg-surface border-b border-line whitespace-nowrap">
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const isAtm = row.is_atm;
          const ceItm = spot != null && row.strike < spot;
          const peItm = spot != null && row.strike > spot;
          const ceShade = isAtm ? '' : ceItm ? 'bg-down/[0.04]' : '';
          const peShade = isAtm ? '' : peItm ? 'bg-up/[0.04]' : '';
          return (
            <tr
              key={row.strike}
              ref={isAtm ? atmRef : null}
              className={`transition-colors hover:bg-surface-2/70 ${isAtm ? 'bg-warn/[0.06]' : ''}`}
            >
              {ceCols.map((col) => (
                <ChainCell key={col.key} col={col} row={row} side="ce" max={maxCeOI} shade={ceShade} />
              ))}
              <td className={`px-3 py-2 text-center font-semibold border-x border-b border-line ${isAtm ? 'text-warn bg-warn/[0.08]' : 'text-ink bg-surface-2/60'}`}>
                <span className="inline-flex items-center gap-1.5">
                  {isAtm && <span className="w-1.5 h-1.5 rounded-full bg-warn" />}
                  {formatNumber(row.strike, 0)}
                </span>
              </td>
              {peCols.map((col) => (
                <ChainCell key={col.key} col={col} row={row} side="pe" max={maxPeOI} shade={peShade} />
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Legend({ updatedAt }) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-2.5 border-t border-line text-2xs text-ink-faint">
      <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-warn/40" />ATM</span>
      <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-down/25" />Call ITM</span>
      <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-up/25" />Put ITM</span>
      <span className="hidden md:flex items-center gap-3 pl-4 border-l border-line">
        {Object.values(BUILDUP).map((b) => (
          <span key={b.label} className="flex items-center gap-1.5"><BuildupTag b={b} />{b.full}</span>
        ))}
      </span>
      <span className="ml-auto num">
        Updated {updatedAt ? new Date(updatedAt).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : '—'}
      </span>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function OptionsChainPage() {
  const [selectedIndex, setSelectedIndex] = useState(FO_INDICES[0].value);
  const [selectedExpiry, setSelectedExpiry] = useState(null);
  const [liveSpot, setLiveSpot] = useState(null);
  const [range, setRange] = useState(10);
  const [view, setView] = useState('compact');

  const atmRef = useRef(null);
  const scrollRef = useRef(null);
  const scrolledKeyRef = useRef(null);

  const fetchChain = useCallback(
    () => api.getOptionsChainFull(selectedIndex, selectedExpiry),
    [selectedIndex, selectedExpiry],
  );

  const { data, loading, error, lastUpdated, refetch } = usePolling(fetchChain, 30_000);

  useEffect(() => { setLiveSpot(null); }, [selectedIndex]);

  // usePolling reads fetchFn via ref, so selection changes need an explicit refetch.
  useEffect(() => {
    refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIndex, selectedExpiry]);

  useEffect(() => {
    if (data?.expiries?.length > 0) {
      if (!selectedExpiry || !data.expiries.includes(selectedExpiry)) {
        setSelectedExpiry(data.expiries[0]);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data?.expiries, selectedIndex]);

  // ── WebSocket ──────────────────────────────────────────────────────────────
  const handleWsMessage = useCallback((msg) => {
    if (msg.type === 'options_update') {
      const { index_id, expiry_date } = msg.data;
      if (index_id === selectedIndex && (!selectedExpiry || expiry_date === selectedExpiry)) {
        refetch();
      }
    } else if (msg.type === 'price_tick' && msg.data.index_id === selectedIndex) {
      setLiveSpot(msg.data.ltp);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIndex, selectedExpiry]);

  const { connected } = useWebSocket(handleWsMessage);

  // ── Derived ────────────────────────────────────────────────────────────────
  const strikes = data?.strikes ?? [];
  const spot = liveSpot ?? data?.spot;
  const bias = useMemo(() => computeBias(data, strikes), [data, strikes]);
  const maxCeOI = useMemo(() => Math.max(...strikes.map((s) => s.ce_oi || 0), 1), [strikes]);
  const maxPeOI = useMemo(() => Math.max(...strikes.map((s) => s.pe_oi || 0), 1), [strikes]);

  const rows = useMemo(() => {
    if (range === 'all' || !strikes.length) return strikes;
    const atm = atmIndexOf(strikes, data?.spot);
    if (atm < 0) return strikes;
    return strikes.slice(Math.max(0, atm - range), Math.min(strikes.length, atm + range + 1));
  }, [strikes, range, data?.spot]);

  const ceCols = view === 'compact' ? CE_COMPACT : CE_DETAILED;
  const peCols = view === 'compact' ? PE_COMPACT : PE_DETAILED;

  // ── Centre the ATM row inside the table scroller (once per selection) ──────
  useEffect(() => {
    const key = `${selectedIndex}|${selectedExpiry}|${range}|${view}`;
    if (!rows.length || scrolledKeyRef.current === key) return;
    const t = setTimeout(() => {
      const box = scrollRef.current;
      const atm = atmRef.current;
      if (!box || !atm) return;
      const delta = atm.getBoundingClientRect().top - box.getBoundingClientRect().top;
      box.scrollTop = box.scrollTop + delta - box.clientHeight / 2 + atm.offsetHeight / 2;
      scrolledKeyRef.current = key;
    }, 80);
    return () => clearTimeout(t);
  }, [rows, selectedIndex, selectedExpiry, range, view]);

  const changeIndex = (id) => {
    setSelectedIndex(id);
    setSelectedExpiry(null);
  };

  const expiries = data?.expiries ?? [];
  const currentExpiry = selectedExpiry ?? data?.expiry ?? '';

  // ── Body ───────────────────────────────────────────────────────────────────
  let body;
  if (loading && !data) {
    body = (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-ink-faint">
        <Spinner />
        <span className="text-xs">Loading option chain…</span>
      </div>
    );
  } else if (error && !data) {
    body = (
      <EmptyState
        className="flex-1"
        icon={AlertTriangle}
        title="Couldn't load the option chain"
        message={String(error)}
        action={<button onClick={refetch} className="btn btn-secondary">Retry</button>}
      />
    );
  } else if (!strikes.length) {
    body = (
      <EmptyState
        className="flex-1"
        icon={BarChart2}
        title={`No chain data for ${selectedIndex}`}
        message="Option chain data may be unavailable post-market or for this expiry."
      />
    );
  } else {
    body = (
      <div className="flex-1 min-h-0 flex flex-col gap-4 px-4 lg:px-6 py-4">
        <StatRow data={data} liveSpot={liveSpot} strikes={strikes} bias={bias} />
        <section className="card flex-1 min-h-[280px] flex flex-col overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 border-b border-line">
            <div className="flex items-center gap-2">
              <Segmented size="xs" ariaLabel="Strike range" options={RANGE_OPTIONS} value={range} onChange={setRange} />
              <Segmented size="xs" ariaLabel="Columns" options={VIEW_OPTIONS} value={view} onChange={setView} />
            </div>
            <span className="text-2xs text-ink-faint num">
              {rows.length === strikes.length ? `${strikes.length} strikes` : `${rows.length} of ${strikes.length} strikes around ATM`}
            </span>
          </div>
          <div ref={scrollRef} className="flex-1 min-h-0 overflow-auto">
            <ChainTable
              rows={rows}
              ceCols={ceCols}
              peCols={peCols}
              spot={spot}
              maxCeOI={maxCeOI}
              maxPeOI={maxPeOI}
              atmRef={atmRef}
            />
          </div>
          <Legend updatedAt={data?.updated_at} />
        </section>
      </div>
    );
  }

  return (
    <FullPage>
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 lg:px-6 py-3 border-b border-line bg-panel/40 flex-shrink-0">
        <Segmented size="xs" ariaLabel="Index" options={FO_INDICES} value={selectedIndex} onChange={changeIndex} />

        <div className="relative">
          <select
            className="select h-8 text-[13px] w-40 bg-surface-2"
            value={currentExpiry}
            onChange={(e) => setSelectedExpiry(e.target.value)}
            disabled={!expiries.length}
            aria-label="Expiry"
          >
            {!expiries.length && <option value="">Expiry</option>}
            {expiries.map((exp) => (
              <option key={exp} value={exp}>{formatExpiry(exp)}</option>
            ))}
          </select>
          <svg className="w-4 h-4 text-ink-faint absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m6 9 6 6 6-6" /></svg>
        </div>


        <div className="ml-auto flex items-center gap-3">
          {loading && data && <Spinner size="xs" />}
          <span
            className="flex items-center gap-1.5 text-2xs text-ink-faint"
            title={connected ? 'Real-time updates via WebSocket' : 'WebSocket disconnected — polling every 30s'}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-up animate-pulse' : 'bg-ink-ghost'}`} />
            {connected ? 'Live' : 'Polling'}
          </span>
          <RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />
        </div>
      </div>

      {body}
    </FullPage>
  );
}
