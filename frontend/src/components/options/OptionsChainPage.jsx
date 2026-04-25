import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { usePolling } from '../../hooks/usePolling';
import { useWebSocket } from '../../hooks/useWebSocket';
import { api } from '../../api/client';
import { Spinner } from '../common/Spinner';
import { RefreshIndicator } from '../common/RefreshIndicator';
import {
  TrendingUp, TrendingDown, Activity, ChevronDown,
  BarChart2, Target, Shield, Zap, Wifi, WifiOff
} from 'lucide-react';

// ── Formatters ───────────────────────────────────────────────────────────────

const fmt  = (v, dec = 2) => v == null ? '—' : Number(v).toFixed(dec);
const fmtK = (v) => {
  if (v == null) return '—';
  if (Math.abs(v) >= 1e7) return `${(v / 1e7).toFixed(2)}Cr`;
  if (Math.abs(v) >= 1e5) return `${(v / 1e5).toFixed(2)}L`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return String(v);
};
const fmtP = (v, dec = 2) => v == null ? '—' : `${v.toFixed(dec)}%`;
const fmtPr = (v) => v == null ? '—' : `₹${Number(v).toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;

// ── Colour helpers ───────────────────────────────────────────────────────────
const oiChangeColour = (v) => {
  if (v == null) return 'text-slate-500';
  if (v > 0) return 'text-emerald-400';
  if (v < 0) return 'text-rose-400';
  return 'text-slate-500';
};

// ── OI bar (horizontal mini bar) ─────────────────────────────────────────────
function OIBar({ value, max, side }) {
  if (!value || !max) return null;
  const pct = Math.min((value / max) * 100, 100);
  const color = side === 'ce' ? 'bg-red-500/60' : 'bg-emerald-500/60';
  return (
    <div className={`absolute bottom-0 ${side === 'ce' ? 'right-0' : 'left-0'} h-[3px] ${color} rounded-full`}
      style={{ width: `${pct}%` }} />
  );
}

// ── Stat pill (Ultra-Compact) ────────────────────────────────────────────────
function StatPill({ label, value, colour = 'text-slate-100' }) {
  return (
    <div className="flex flex-col px-3 border-r border-slate-800 last:border-r-0">
      <span className="text-[9px] uppercase font-bold text-slate-500 leading-tight mb-0.5 whitespace-nowrap">{label}</span>
      <span className={`text-[11px] font-mono font-black ${colour} whitespace-nowrap`}>{value}</span>
    </div>
  );
}

// ── PCR Gauge (Micro) ─────────────────────────────────────────────────────────
function PCRGauge({ pcr }) {
  if (pcr == null) return null;
  const capped = Math.min(Math.max(pcr, 0.2), 2.5);
  const pct = ((capped - 0.2) / (2.5 - 0.2)) * 100;
  const col = pcr > 1.3 ? '#10b981' : pcr < 0.7 ? '#ef4444' : '#64748b';

  return (
    <div className="flex flex-col px-3 border-r border-slate-800 min-w-[100px]">
      <div className="flex items-center justify-between gap-2 mb-0.5">
        <span className="text-[9px] text-slate-500 uppercase font-bold">PCR</span>
        <span className="text-[10px] font-mono font-black" style={{ color: col }}>{pcr.toFixed(2)}</span>
      </div>
      <div className="relative h-1 w-full rounded-full bg-slate-800">
        <div
          className="absolute left-0 h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: col }}
        />
      </div>
    </div>
  );
}

// ── OI Balance (Micro) ───────────────────────────────────────────────────────
function OIBalance({ totalCe, totalPe }) {
  if (!totalCe && !totalPe) return null;
  const total = (totalCe || 0) + (totalPe || 0);
  const cePct = total ? (totalCe / total) * 100 : 50;
  return (
    <div className="flex flex-col px-3 border-r border-slate-800 min-w-[120px]">
      <div className="flex justify-between text-[8px] font-bold uppercase mb-0.5">
        <span className="text-rose-400">CE {fmtK(totalCe)}</span>
        <span className="text-emerald-400">PE {fmtK(totalPe)}</span>
      </div>
      <div className="flex h-1 rounded-full overflow-hidden bg-slate-800">
        <div className="h-full bg-rose-500/60" style={{ width: `${cePct}%` }} />
        <div className="h-full bg-emerald-500/60" style={{ width: `${100 - cePct}%` }} />
      </div>
    </div>
  );
}

// ── Table columns config ─────────────────────────────────────────────────────
const CE_COLS = [
  { key: 'ce_oi',          label: 'OI',      fmt: fmtK,  cls: 'text-slate-300' },
  { key: 'ce_oi_change',   label: 'OI Chg',  fmt: fmtK,  cls: (v) => oiChangeColour(v) },
  { key: 'ce_oi_p_change', label: 'OI%',     fmt: (v) => fmtP(v, 1), cls: (v) => oiChangeColour(v) },
  { key: 'ce_volume',      label: 'Vol',      fmt: fmtK,  cls: 'text-slate-400' },
  { key: 'ce_iv',          label: 'IV%',      fmt: (v) => fmtP(v, 1), cls: 'text-violet-300' },
  { key: 'ce_ltp',         label: 'LTP',      fmt: fmtPr, cls: 'text-slate-100 font-semibold' },
  { key: 'ce_ltp_p_change',label: 'Chg%',     fmt: (v) => fmtP(v, 1), cls: (v) => oiChangeColour(v) },
];

const PE_COLS = [
  { key: 'pe_ltp_p_change',label: 'Chg%',     fmt: (v) => fmtP(v, 1), cls: (v) => oiChangeColour(v) },
  { key: 'pe_ltp',         label: 'LTP',      fmt: fmtPr, cls: 'text-slate-100 font-semibold' },
  { key: 'pe_iv',          label: 'IV%',      fmt: (v) => fmtP(v, 1), cls: 'text-violet-300' },
  { key: 'pe_volume',      label: 'Vol',      fmt: fmtK,  cls: 'text-slate-400' },
  { key: 'pe_oi_p_change', label: 'OI%',     fmt: (v) => fmtP(v, 1), cls: (v) => oiChangeColour(v) },
  { key: 'pe_oi_change',   label: 'OI Chg',  fmt: fmtK,  cls: (v) => oiChangeColour(v) },
  { key: 'pe_oi',          label: 'OI',      fmt: fmtK,  cls: 'text-slate-300' },
];

// ── Buildup Trend Helper ──────────────────────────────────────────────────────
const getBuildup = (pChg, oiChg) => {
  if (!pChg || !oiChg) return null;
  if (pChg > 0 && oiChg > 0) return { label: 'LB', color: 'bg-emerald-500/20 text-emerald-400', full: 'Long Buildup' };
  if (pChg < 0 && oiChg > 0) return { label: 'SB', color: 'bg-red-500/20 text-red-400', full: 'Short Buildup' };
  if (pChg < 0 && oiChg < 0) return { label: 'LU', color: 'bg-orange-500/20 text-orange-400', full: 'Long Unwinding' };
  if (pChg > 0 && oiChg < 0) return { label: 'SC', color: 'bg-sky-500/20 text-sky-400', full: 'Short Covering' };
  return null;
};

// ── F&O index list (those with options) ──────────────────────────────────────
const FO_INDICES = [
  { id: 'NIFTY50', label: 'NIFTY 50' },
  { id: 'BANKNIFTY', label: 'BANKNIFTY' },
  { id: 'FINNIFTY', label: 'FINNIFTY' },
  { id: 'MIDCPNIFTY', label: 'MIDCP' },
  { id: 'SENSEX', label: 'SENSEX' },
];

// ── Main page ─────────────────────────────────────────────────────────────────
export default function OptionsChainPage() {
  const [selectedIndex, setSelectedIndex] = useState(FO_INDICES[0].id);
  const [selectedExpiry, setSelectedExpiry] = useState(null);
  const [showExpiries, setShowExpiries] = useState(false);
  const [highlight, setHighlight] = useState(null); // hovered strike
  const [liveSpot, setLiveSpot] = useState(null);

  const atmRef = useRef(null);
  const containerRef = useRef(null);

  const fetchChain = useCallback(
    () => api.getOptionsChainFull(selectedIndex, selectedExpiry),
    [selectedIndex, selectedExpiry],
  );

  const { data, loading, error, lastUpdated, refetch } = usePolling(fetchChain, 30_000);

  // ── Auto-scroll to ATM ─────────────────────────────────────────────────────
  useEffect(() => {
    if (data && data.strikes?.length > 0) {
      const timer = setTimeout(() => {
        if (atmRef.current) {
          atmRef.current.scrollIntoView({
            behavior: 'smooth',
            block: 'center',
          });
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [data, selectedIndex, selectedExpiry]);

  useEffect(() => { setLiveSpot(null); }, [selectedIndex]);

  useEffect(() => {
    refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIndex]);

  useEffect(() => {
    if (data?.expiries?.length > 0) {
      if (!selectedExpiry || !data.expiries.includes(selectedExpiry)) {
        setSelectedExpiry(data.expiries[0]);
      }
    }
  }, [data?.expiries, selectedIndex]);

  // ── WebSocket Integration ──────────────────────────────────────────────────
  const handleWsMessage = useCallback((msg) => {
    if (msg.type === 'options_update') {
      const { index_id, expiry_date } = msg.data;
      if (index_id === selectedIndex) {
        if (!selectedExpiry || expiry_date === selectedExpiry) {
          refetch();
        }
      }
    } else if (msg.type === 'price_tick' && msg.data.index_id === selectedIndex) {
      setLiveSpot(msg.data.ltp);
    }
  }, [selectedIndex, selectedExpiry, refetch]);

  const { connected } = useWebSocket(handleWsMessage);

  const strikes = data?.strikes ?? [];
  const maxCeOI = useMemo(() => Math.max(...strikes.map((s) => s.ce_oi || 0), 1), [strikes]);
  const maxPeOI = useMemo(() => Math.max(...strikes.map((s) => s.pe_oi || 0), 1), [strikes]);

  const handleExpiry = (exp) => {
    setSelectedExpiry(exp);
    setShowExpiries(false);
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-100 select-none">
      {/* ── Unified Command Header ───────────────────────────────── */}
      {data && (
        <div className="px-5 py-2 border-b border-slate-800 bg-slate-900/60 flex items-center justify-between gap-4 flex-shrink-0">
          {/* Group 1: Controls */}
          <div className="flex items-center gap-3 pr-4 border-r border-slate-800">
            <div className="flex items-center bg-slate-950/50 rounded-lg p-0.5 border border-slate-800/50">
              {FO_INDICES.map(idx => (
                <button
                  key={idx.id}
                  onClick={() => { setSelectedIndex(idx.id); setSelectedExpiry(null); }}
                  className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${selectedIndex === idx.id ? 'bg-violet-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}
                >
                  {idx.label}
                </button>
              ))}
            </div>

            <div className="relative">
              <button
                onClick={() => setShowExpiries((p) => !p)}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-950 border border-slate-800 rounded-lg text-[10px] font-bold text-slate-300 hover:bg-slate-800 transition-colors"
              >
                <span>{selectedExpiry ?? data?.expiry ?? 'Expiry'}</span>
                <ChevronDown className="w-3 h-3" />
              </button>
              {showExpiries && data?.expiries?.length > 0 && (
                <div className="absolute left-0 top-full mt-1 z-50 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl py-1 min-w-[120px]">
                  {data.expiries.map((exp) => (
                    <button
                      key={exp}
                      onClick={() => { handleExpiry(exp); setShowExpiries(false); }}
                      className={`w-full text-left px-3 py-1.5 text-[10px] hover:bg-slate-800 transition-colors ${
                        (selectedExpiry ?? data.expiry) === exp ? 'text-violet-400 font-bold' : 'text-slate-400'
                      }`}
                    >
                      {exp}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Group 2: Live Market Stats */}
          <div className="flex items-center flex-1">
             <StatPill
               label={liveSpot ? 'Spot ●' : 'Spot'}
               value={(liveSpot ?? data.spot) ? `₹${Number(liveSpot ?? data.spot).toLocaleString('en-IN')}` : '—'}
               colour={liveSpot ? 'text-emerald-300' : 'text-slate-100'}
             />
             <StatPill label="Max Pain" value={data.max_pain ? `₹${Number(data.max_pain).toLocaleString('en-IN')}` : '—'} colour="text-yellow-400" />
             <PCRGauge pcr={data.pcr} />
             <OIBalance totalCe={data.total_ce_oi} totalPe={data.total_pe_oi} />
          </div>

          {/* Group 3: Intelligence Dashboard */}
          <div className="flex items-center gap-4 pl-4 border-l border-slate-800">
             {(() => {
                const pcr = data.pcr || 1;
                const spot = data.spot || 0;
                const maxPain = data.max_pain || 0;
                const atmIndex = strikes.findIndex(s => s.is_atm);
                const window = strikes.slice(Math.max(0, atmIndex - 3), Math.min(strikes.length, atmIndex + 4));
                let ceLB = 0, ceSB = 0, peLB = 0, peSB = 0;
                window.forEach(s => {
                  if (s.ce_ltp_p_change > 0 && s.ce_oi_change > 0) ceLB++;
                  if (s.ce_ltp_p_change < 0 && s.ce_oi_change > 0) ceSB++;
                  if (s.pe_ltp_p_change > 0 && s.pe_oi_change > 0) peLB++;
                  if (s.pe_ltp_p_change < 0 && s.pe_oi_change > 0) peSB++;
                });

                let score = 50; 
                const reasons = [];
                if (pcr > 1.4) { score += 15; reasons.push("PCR Reversal"); }
                else if (pcr < 0.6) { score -= 15; reasons.push("Overbought"); }
                const painDiff = ((spot - maxPain) / spot) * 100;
                if (painDiff > 0.5) { score -= 10; reasons.push("Above Pain"); }
                else if (painDiff < -0.5) { score += 10; reasons.push("Below Pain"); }
                if (ceLB > ceSB) { score += 5; } else if (ceSB > ceLB) { score -= 10; reasons.push("Call Writing"); }
                if (peLB > peSB) { score += 10; reasons.push("Put Buying"); } else if (peSB > peLB) { score -= 5; reasons.push("Put Writing"); }

                const verdict = score > 75 ? "STR. BULLISH" : score > 60 ? "BULLISH" : score < 25 ? "STR. BEARISH" : score < 40 ? "BEARISH" : "NEUTRAL";
                const vColor = score > 60 ? "text-emerald-400" : score < 40 ? "text-rose-400" : "text-slate-400";
                const vBg = score > 60 ? "bg-emerald-500/10 border-emerald-500/20" : score < 40 ? "bg-rose-500/10 border-rose-500/20" : "bg-slate-800/50 border-slate-700/50";

                return (
                  <div className="flex items-center gap-3">
                    <div className={`flex flex-col justify-center px-3 py-1 rounded-lg border ${vBg} min-w-[100px]`}>
                      <div className="text-[8px] uppercase font-bold text-slate-500 mb-0.5 leading-none">Verdict</div>
                      <div className={`text-[10px] font-black tracking-tighter ${vColor}`}>{verdict}</div>
                      <div className="flex items-center gap-1 mt-0.5">
                        <div className="flex-1 h-0.5 bg-slate-800 rounded-full overflow-hidden">
                           <div className={`h-full ${score > 60 ? 'bg-emerald-500' : score < 40 ? 'bg-rose-500' : 'bg-slate-500'}`} style={{ width: `${score}%` }} />
                        </div>
                      </div>
                    </div>
                    
                    <div className="hidden xl:flex flex-wrap gap-x-2 gap-y-0 max-w-[150px]">
                       {reasons.slice(0, 3).map((r, i) => (
                         <div key={i} className="text-[8px] text-slate-500 flex items-center gap-0.5">
                           <div className="w-0.5 h-0.5 rounded-full bg-violet-600" />
                           {r}
                         </div>
                       ))}
                    </div>
                  </div>
                );
             })()}

             <div className="flex flex-col items-end gap-0.5 pl-3 border-l border-slate-800">
                <div className="flex items-center gap-1.5">
                   <div className={`w-1 h-1 rounded-full ${connected ? 'bg-emerald-500 animate-pulse' : 'bg-slate-600'}`} />
                   <span className={`text-[8px] font-bold ${connected ? 'text-emerald-500' : 'text-slate-500'}`}>
                     {connected ? 'LIVE' : 'OFF'}
                   </span>
                </div>
                <RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />
             </div>
          </div>
        </div>
      )}
      {/* ── Table (Maximized) ─────────────────────────────────────────── */}
      {loading && !data ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <Spinner />
            <span className="text-sm text-slate-500">Loading options chain…</span>
          </div>
        </div>
      ) : error ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <BarChart2 className="w-12 h-12 text-slate-700 mx-auto mb-3" />
            <p className="text-slate-400 text-sm">No options data available</p>
            <p className="text-slate-600 text-xs mt-1">{error?.message ?? String(error)}</p>
          </div>
        </div>
      ) : !strikes.length ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md px-6">
            <BarChart2 className="w-12 h-12 text-slate-700 mx-auto mb-3" />
            <p className="text-slate-400 text-sm font-semibold">No chain data for {selectedIndex}</p>
            <p className="text-slate-600 text-xs mt-1">Options chain data may not be available post-market or for this specific expiry.</p>
          </div>
        </div>
      ) : (
        <div ref={containerRef} className="flex-1 overflow-auto">
          <table className="w-full text-xs border-collapse min-w-[900px]">
            <thead className="sticky top-0 z-10">
              <tr className="bg-slate-900 border-b border-slate-700">
                <th colSpan={CE_COLS.length} className="py-2 text-center text-red-400 font-bold tracking-widest uppercase text-[10px] border-r border-slate-700/80 bg-red-950/20">
                  CALLS (CE)
                </th>
                <th className="py-2 px-3 text-center text-slate-300 font-bold tracking-widest uppercase text-[10px] bg-slate-800/60 border-x border-slate-600">
                  STRIKE
                </th>
                <th colSpan={PE_COLS.length} className="py-2 text-center text-emerald-400 font-bold tracking-widest uppercase text-[10px] border-l border-slate-700/80 bg-emerald-950/20">
                  PUTS (PE)
                </th>
              </tr>
              <tr className="bg-slate-900/90 border-b border-slate-700/60">
                {CE_COLS.map((c) => (
                  <th key={c.key} className="py-1.5 px-2 text-right text-[9px] text-slate-500 font-semibold uppercase tracking-wider border-r border-slate-800 last:border-slate-600">
                    {c.label}
                  </th>
                ))}
                <th className="py-1.5 px-3 text-center text-[9px] text-slate-400 font-bold uppercase tracking-wider bg-slate-800/40 border-x border-slate-600" />
                {PE_COLS.map((c) => (
                  <th key={c.key} className="py-1.5 px-2 text-left text-[9px] text-slate-500 font-semibold uppercase tracking-wider border-l border-slate-800 first:border-slate-600">
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {strikes.map((row) => {
                const isAtm = row.is_atm;
                const isHovered = highlight === row.strike;
                const itm_ce = data?.spot != null && row.strike < data.spot;
                const itm_pe = data?.spot != null && row.strike > data.spot;

                return (
                  <tr
                    key={row.strike}
                    ref={isAtm ? atmRef : null}
                    onMouseEnter={() => setHighlight(row.strike)}
                    onMouseLeave={() => setHighlight(null)}
                    className={`
                      border-b transition-colors duration-75
                      ${isAtm
                        ? 'border-yellow-500/40 bg-yellow-500/5'
                        : isHovered
                          ? 'border-slate-700 bg-slate-800/50'
                          : 'border-slate-800/60 hover:bg-slate-800/30'}
                    `}
                  >
                    {CE_COLS.map((col, ci) => {
                      const val = row[col.key];
                      const cls = typeof col.cls === 'function' ? col.cls(val) : col.cls;
                      const isLast = ci === CE_COLS.length - 1;
                      const buildup = col.key === 'ce_ltp' ? getBuildup(row.ce_ltp_p_change, row.ce_oi_change) : null;

                      return (
                        <td
                          key={col.key}
                          className={`
                            relative py-1.5 px-2 text-right font-mono
                            ${itm_ce ? 'bg-red-950/20' : ''}
                            ${isLast ? 'border-r border-slate-600' : 'border-r border-slate-800/40'}
                            ${cls}
                          `}
                        >
                          {col.key === 'ce_oi' && (
                            <OIBar value={row.ce_oi} max={maxCeOI} side="ce" />
                          )}
                          <div className="flex items-center justify-end gap-1.5">
                            {buildup && (
                              <span className={`text-[8px] px-1 rounded font-bold ${buildup.color}`} title={buildup.full}>
                                {buildup.label}
                              </span>
                            )}
                            <span>
                              {val != null ? (
                                typeof col.fmt === 'function' ? col.fmt(val) : val
                              ) : (
                                <span className="text-slate-700">—</span>
                              )}
                            </span>
                          </div>
                        </td>
                      );
                    })}

                    <td className={`
                      py-1.5 px-3 text-center font-mono font-bold text-sm border-x border-slate-600
                      ${isAtm
                        ? 'text-yellow-300 bg-yellow-500/10 shadow-inner'
                        : 'text-slate-200 bg-slate-800/40'}
                    `}>
                      {isAtm && (
                        <span className="inline-block w-1.5 h-1.5 rounded-full bg-yellow-400 mr-1.5 mb-0.5" />
                      )}
                      {Number(row.strike).toLocaleString('en-IN')}
                    </td>

                    {PE_COLS.map((col, ci) => {
                      const val = row[col.key];
                      const cls = typeof col.cls === 'function' ? col.cls(val) : col.cls;
                      const isFirst = ci === 0;
                      const buildup = col.key === 'pe_ltp' ? getBuildup(row.pe_ltp_p_change, row.pe_oi_change) : null;

                      return (
                        <td
                          key={col.key}
                          className={`
                            relative py-1.5 px-2 text-left font-mono
                            ${itm_pe ? 'bg-emerald-950/20' : ''}
                            ${isFirst ? 'border-l border-slate-600' : 'border-l border-slate-800/40'}
                            ${cls}
                          `}
                        >
                          {col.key === 'pe_oi' && (
                            <OIBar value={row.pe_oi} max={maxPeOI} side="pe" />
                          )}
                          <div className="flex items-center gap-1.5">
                            <span>
                              {val != null ? (
                                typeof col.fmt === 'function' ? col.fmt(val) : val
                              ) : (
                                <span className="text-slate-700">—</span>
                              )}
                            </span>
                            {buildup && (
                              <span className={`text-[8px] px-1 rounded font-bold ${buildup.color}`} title={buildup.full}>
                                {buildup.label}
                              </span>
                            )}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Footer legend ─────────────────────────────────────────────── */}
      {strikes.length > 0 && (
        <div className="flex flex-wrap items-center gap-y-2 gap-x-5 px-5 py-2.5 border-t border-slate-800 bg-slate-900/70 flex-shrink-0 text-[10px] text-slate-500">
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-1.5 rounded-full bg-yellow-400/70" />
            <span>ATM Strike</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-1.5 rounded-full bg-red-500/50" />
            <span>CE ITM zone</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-1.5 rounded-full bg-emerald-500/50" />
            <span>PE ITM zone</span>
          </div>
          <div className="flex items-center gap-3 ml-4 border-l border-slate-800 pl-4">
             <span className="text-emerald-400 font-bold">LB: Long Buildup</span>
             <span className="text-red-400 font-bold">SB: Short Buildup</span>
             <span className="text-orange-400 font-bold">LU: Long Unwinding</span>
             <span className="text-sky-400 font-bold">SC: Short Covering</span>
          </div>
          <span className="ml-auto flex items-center gap-2">
             <div className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-emerald-500' : 'bg-rose-500'}`} />
             {connected ? 'Real-time WebSocket active' : 'WebSocket disconnected (Polling mode)'}
             <span className="mx-2">|</span>
             {strikes.length} strikes · updated {data?.updated_at ? new Date(data.updated_at).toLocaleTimeString('en-IN') : '—'}
          </span>
        </div>
      )}
    </div>
  );
}
