import { ArrowUpRight, ArrowDownRight, Minus, Ticket } from 'lucide-react';
import { Badge } from '../common/Badge';
import { formatPrice, timeAgo, formatISTDateTime } from '../../utils/formatters';
import { SIGNAL_TYPE_LABELS } from '../../utils/constants';

const TYPE_META = {
  BUY_CALL: { Icon: ArrowUpRight, cls: 'bg-up/10 text-up' },
  BUY_PUT: { Icon: ArrowDownRight, cls: 'bg-down/10 text-down' },
  NO_TRADE: { Icon: Minus, cls: 'bg-surface-3 text-ink-faint' },
};

const VOTES = [
  ['technical_vote', 'Technical'],
  ['options_vote', 'Options / OI'],
  ['news_vote', 'News'],
  ['anomaly_vote', 'Anomaly'],
];

function voteTone(v) {
  const k = v?.toUpperCase();
  if (k === 'BUY' || k === 'BUY_CALL' || k === 'BULLISH') return 'success';
  if (k === 'SELL' || k === 'BUY_PUT' || k === 'BEARISH') return 'danger';
  return 'neutral';
}

/** Small section heading inside the detail panel. */
function Section({ title, children }) {
  return (
    <div>
      <div className="label mb-2.5">{title}</div>
      {children}
    </div>
  );
}

export function ConfidenceMeter({ score, className = '' }) {
  const pct = score != null ? Math.round(score * 100) : null;
  const color = pct == null ? 'bg-ink-ghost' : pct >= 70 ? 'bg-up' : pct >= 45 ? 'bg-warn' : 'bg-ink-faint';
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <div className="flex-1 h-1.5 rounded-full bg-surface-3 overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all duration-500`} style={{ width: `${pct ?? 0}%` }} />
      </div>
      <span className="text-xs num text-ink-muted w-9 text-right">{pct == null ? '—' : `${pct}%`}</span>
    </div>
  );
}

function TradeTicket({ trade }) {
  if (!trade?.strike || !trade?.expiry) return null;
  const isCall = trade.option_type === 'CE';
  return (
    <div className="rounded-lg border border-line bg-canvas/60 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-ink-muted">
          <Ticket className="w-4 h-4" />
          <span className="text-[13px] font-medium">Option contract</span>
        </div>
        <Badge tone={isCall ? 'success' : 'danger'} className="font-mono">
          {Math.round(trade.strike)} {trade.option_type}
        </Badge>
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 mt-4 text-[13px]">
        <div><dt className="text-2xs text-ink-faint">Expiry</dt><dd className="num text-ink mt-0.5">{trade.expiry}</dd></div>
        <div><dt className="text-2xs text-ink-faint">Premium</dt><dd className="num text-ink mt-0.5">{trade.premium != null ? `₹${trade.premium.toFixed(2)}` : '—'}</dd></div>
        <div><dt className="text-2xs text-ink-faint">Lots</dt><dd className="num text-ink mt-0.5">{trade.lots ?? '—'}</dd></div>
        <div>
          <dt className="text-2xs text-ink-faint">Max loss</dt>
          <dd className="num text-down mt-0.5">
            {trade.max_loss_amount != null ? `₹${Math.round(trade.max_loss_amount).toLocaleString('en-IN')}` : '—'}
            {trade.risk_pct_of_capital != null && (
              <span className="text-ink-faint ml-1">({trade.risk_pct_of_capital.toFixed(2)}%)</span>
            )}
          </dd>
        </div>
      </dl>
    </div>
  );
}

/**
 * Detail panel for one signal. Accepts both the list payload (entry/target/sl)
 * and the detail payload (entry_price/target_price/stop_loss + votes).
 */
export function SignalCard({ signal, title }) {
  const {
    index_id, signal_type, confidence_level, confidence_score,
    entry_price, target_price, stop_loss, risk_reward_ratio,
    entry, target, sl, rr_ratio, regime, reasoning, reasoning_summary, generated_at, option_trade,
  } = signal;

  const entryVal = entry_price ?? entry;
  const targetVal = target_price ?? target;
  const slVal = stop_loss ?? sl;
  const rrVal = risk_reward_ratio ?? rr_ratio;
  const isNoTrade = signal_type === 'NO_TRADE';
  const { Icon, cls } = TYPE_META[signal_type] ?? TYPE_META.NO_TRADE;
  const hasVotes = VOTES.some(([k]) => signal[k] != null);
  const text = reasoning?.text ?? reasoning_summary;

  return (
    <div className="divide-y divide-line">
      {/* Header */}
      <div className="p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${cls}`}>
              <Icon className="w-5 h-5" />
            </div>
            <div className="min-w-0">
              <div className="text-base font-semibold text-ink truncate">{title ?? index_id}</div>
              <div className="text-xs text-ink-faint mt-0.5" title={generated_at ? formatISTDateTime(generated_at) : undefined}>
                {generated_at ? `Generated ${timeAgo(generated_at)}` : '—'}
              </div>
            </div>
          </div>
          <Badge variant={signal_type}>{SIGNAL_TYPE_LABELS[signal_type] ?? signal_type}</Badge>
        </div>

        <div className="mt-5">
          <div className="flex items-center justify-between mb-2">
            <span className="label">Confidence</span>
            {confidence_level && <Badge variant={confidence_level} size="xs">{confidence_level}</Badge>}
          </div>
          <ConfidenceMeter score={confidence_score} />
        </div>
      </div>

      {/* Levels */}
      {!isNoTrade && (entryVal || targetVal || slVal) ? (
        <div className="p-5 space-y-4">
          <Section title="Levels">
            <div className="grid grid-cols-3 gap-2">
              {[
                ['Entry', entryVal, 'text-ink'],
                ['Target', targetVal, 'text-up'],
                ['Stop loss', slVal, 'text-down'],
              ].map(([l, v, c]) => (
                <div key={l} className="rounded-lg bg-canvas/60 border border-line px-3 py-2.5">
                  <div className="text-2xs text-ink-faint">{l}</div>
                  <div className={`text-[13px] font-semibold num mt-0.5 ${c}`}>{v ? formatPrice(v) : '—'}</div>
                </div>
              ))}
            </div>
          </Section>
          <TradeTicket trade={option_trade} />
        </div>
      ) : null}

      {/* Context */}
      <div className="p-5">
        <dl className="grid grid-cols-2 gap-4 text-[13px]">
          <div>
            <dt className="label">Regime</dt>
            <dd className="text-ink mt-1 capitalize">{regime ? regime.replace(/_/g, ' ').toLowerCase() : '—'}</dd>
          </div>
          <div>
            <dt className="label">Risk : reward</dt>
            <dd className="text-ink mt-1 num">{rrVal ? rrVal.toFixed(2) : '—'}</dd>
          </div>
        </dl>
      </div>

      {hasVotes && (
        <div className="p-5">
          <Section title="Model votes">
            <div className="space-y-2">
              {VOTES.map(([key, label]) => (
                <div key={key} className="flex items-center justify-between text-[13px]">
                  <span className="text-ink-muted">{label}</span>
                  <Badge tone={voteTone(signal[key])} size="xs">{signal[key] ?? '—'}</Badge>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {text && (
        <div className="p-5">
          <Section title="Reasoning">
            <p className="text-[13px] text-ink-muted leading-relaxed">{text.replace(/^NO_TRADE:\s*/, '')}</p>
          </Section>
        </div>
      )}
    </div>
  );
}

