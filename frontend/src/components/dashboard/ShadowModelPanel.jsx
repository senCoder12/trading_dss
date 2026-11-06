import { useEffect, useState } from 'react';
import { FlaskConical, AlertTriangle, Check, X } from 'lucide-react';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';

const DECISION_STYLE = {
  BUY: 'text-green-400',
  SELL: 'text-red-400',
  NEUTRAL: 'text-slate-400',
};

function time(iso) {
  return iso ? iso.slice(11, 16) : '—';
}

// Numeric contributions when the model scored; per-component votes (BULLISH/…)
// when it short-circuited (e.g. not tradeable).
function topComponents(components) {
  if (!components || typeof components !== 'object') return [];
  return Object.entries(components)
    .map(([name, c]) => {
      if (c && typeof c === 'object') {
        const num = c.weighted_contribution ?? c.score;
        if (typeof num === 'number') return [name, num, c.vote ?? null];
        if (c.vote) return [name, null, c.vote];
      }
      return [name, typeof c === 'number' ? c : null, null];
    })
    .filter(([, v, vote]) => v !== null || vote)
    .sort((a, b) => Math.abs(b[1] ?? 0) - Math.abs(a[1] ?? 0))
    .slice(0, 6);
}

const VOTE_STYLE = { BULLISH: 'text-green-400', BEARISH: 'text-red-400' };

function Row({ d }) {
  const [open, setOpen] = useState(false);
  const warnings = d.warnings ?? [];
  const reason = (d.reasoning || '').split('\n')[0];
  return (
    <div className="px-4 py-2.5 hover:bg-fuchsia-500/5 cursor-pointer" onClick={() => setOpen(!open)}>
      <div className="grid grid-cols-12 items-center gap-2 text-xs">
        <span className="col-span-2 text-slate-100 font-semibold">{d.index_id}</span>
        <span className="col-span-2 flex items-center gap-1.5">
          <Badge variant="LIVE" size="xs">LIVE</Badge>
          <span className={`font-mono ${DECISION_STYLE[d.live_decision] ?? 'text-slate-500'}`}>
            {d.live_decision ?? '—'}
          </span>
          <span className="text-slate-500 font-mono">{time(d.live_generated_at)}</span>
        </span>
        <span className="col-span-3 flex items-center gap-1.5">
          <Badge variant="SHADOW" size="xs">SHADOW</Badge>
          <span className={`font-mono ${DECISION_STYLE[d.decision]}`}>
            {d.decision}
            {d.forced_neutral && <span title="Forced NEUTRAL: stale or missing inputs">*</span>}
          </span>
          <span className="text-slate-400 font-mono" title="Calibrated P(win)">
            P={d.confidence_score != null ? d.confidence_score.toFixed(2) : '—'}
          </span>
        </span>
        <span className="col-span-2 text-slate-400 truncate" title={d.regime}>{d.regime ?? '—'}</span>
        <span className="col-span-1">
          {d.agree === null ? (
            <span className="text-slate-500">—</span>
          ) : d.agree ? (
            <Check className="w-3.5 h-3.5 text-emerald-400" aria-label="agrees with live" />
          ) : (
            <X className="w-3.5 h-3.5 text-amber-400" aria-label="differs from live" />
          )}
        </span>
        <span className="col-span-2 flex items-center gap-1 justify-end text-amber-400">
          {warnings.length > 0 && (
            <>
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>{warnings.length}</span>
            </>
          )}
        </span>
      </div>
      <div className="text-[11px] text-slate-400 mt-1 truncate" title={reason}>{reason}</div>
      {open && (
        <div className="mt-2 space-y-1.5 text-[11px]">
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-slate-300 font-mono">
            {topComponents(d.components).map(([n, v, vote]) => (
              <span key={n}>
                {n}:{' '}
                {vote && <span className={VOTE_STYLE[vote] ?? 'text-slate-400'}>{vote}</span>}
                {v !== null && (
                  <span className={v > 0 ? 'text-green-400' : v < 0 ? 'text-red-400' : ''}>
                    {vote ? ` (${v.toFixed(1)})` : v.toFixed(3)}
                  </span>
                )}
              </span>
            ))}
          </div>
          {warnings.map((w, i) => (
            <div key={i} className="text-amber-300/90">⚠ {w}</div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ShadowModelPanel() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api.getShadowLatest()
        .then((d) => alive && (setData(d), setError(null)))
        .catch((e) => alive && setError(e.message));
    load();
    const t = setInterval(load, 60_000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const run = data?.run;
  const decisions = data?.decisions ?? [];
  return (
    <Card
      className="border-dashed border-fuchsia-500/50"
      padding={false}
      title={
        <span className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-fuchsia-300" />
          Shadow model — evaluation only, NOT traded
        </span>
      }
      actions={
        <span className="flex items-center gap-2 text-[11px] text-slate-400">
          <Badge variant="SHADOW" size="xs">SHADOW</Badge>
          {run && (
            <span className="font-mono">
              {run.model_branch ?? ''} @ {run.model_commit} · run {time(run.as_of)} · {run.status}
            </span>
          )}
        </span>
      }
    >
      {error ? (
        <div className="p-4 text-xs text-red-400">Shadow API error: {error}</div>
      ) : !run ? (
        <EmptyState icon={FlaskConical} title="No shadow runs yet" message="The shadow runner records a decision every 5 minutes during market hours." />
      ) : (
        <div className="divide-y divide-slate-700/50">
          <div className="grid grid-cols-12 gap-2 px-4 py-1.5 text-[10px] uppercase tracking-wide text-slate-500">
            <span className="col-span-2">Index</span>
            <span className="col-span-2">Live decision</span>
            <span className="col-span-3">Shadow decision</span>
            <span className="col-span-2">Regime</span>
            <span className="col-span-1">Agree</span>
            <span className="col-span-2 text-right">Warnings</span>
          </div>
          {decisions.map((d) => <Row key={d.index_id} d={d} />)}
          <div className="px-4 py-2 text-[10px] text-slate-500">
            * forced NEUTRAL (stale or missing inputs). Shadow decisions are stored for analysis and never reach live signals or trades.
          </div>
        </div>
      )}
    </Card>
  );
}
