import { ArrowUpCircle, ArrowDownCircle, MinusCircle, Target, Shield } from 'lucide-react';
import { Badge } from '../common/Badge';
import { formatPrice, formatConfidence, formatPercentage, timeAgo } from '../../utils/formatters';
import { SIGNAL_TYPE_LABELS } from '../../utils/constants';

const VOTES = ['technical_vote', 'options_vote', 'news_vote', 'anomaly_vote'];
const VOTE_LABELS = {
  technical_vote: 'Technical',
  options_vote: 'Options/OI',
  news_vote: 'News',
  anomaly_vote: 'Anomaly',
};

function VoteBar({ label, vote }) {
  const color =
    vote === 'BUY' || vote === 'BUY_CALL'
      ? 'bg-green-500'
      : vote === 'SELL' || vote === 'BUY_PUT'
      ? 'bg-red-500'
      : 'bg-slate-600';
  const textColor =
    vote === 'BUY' || vote === 'BUY_CALL'
      ? 'text-green-400'
      : vote === 'SELL' || vote === 'BUY_PUT'
      ? 'text-red-400'
      : 'text-slate-500';
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-slate-500 w-20 flex-shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div className={`h-full w-full ${color} opacity-70 rounded-full`} />
      </div>
      <span className={`w-16 text-right font-medium ${textColor}`}>{vote ?? '--'}</span>
    </div>
  );
}

function ConfidenceMeter({ score }) {
  const pct = score != null ? Math.round(score * 100) : 0;
  const color = pct >= 70 ? 'bg-green-500' : pct >= 50 ? 'bg-yellow-500' : 'bg-blue-500';
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400">Confidence</span>
        <span className="text-slate-200 font-mono font-semibold">{pct}%</span>
      </div>
      <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function SignalCard({ signal }) {
  const {
    index_id, signal_type, confidence_level, confidence_score,
    entry_price, target_price, stop_loss, risk_reward_ratio,
    technical_vote, options_vote, news_vote, anomaly_vote,
    regime, reasoning, generated_at,
  } = signal;

  const isNoTrade = signal_type === 'NO_TRADE';

  const Icon = signal_type === 'BUY_CALL'
    ? ArrowUpCircle
    : signal_type === 'BUY_PUT'
    ? ArrowDownCircle
    : MinusCircle;

  const iconColor = signal_type === 'BUY_CALL'
    ? 'text-green-400'
    : signal_type === 'BUY_PUT'
    ? 'text-red-400'
    : 'text-slate-500';

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Icon className={`w-6 h-6 ${iconColor}`} />
          <div>
            <div className="text-slate-100 font-bold text-base">{index_id}</div>
            <Badge variant={signal_type} className="mt-0.5">
              {SIGNAL_TYPE_LABELS[signal_type] ?? signal_type}
            </Badge>
          </div>
        </div>
        <div className="text-right">
          <Badge variant={confidence_level}>{confidence_level}</Badge>
          {generated_at && (
            <div className="text-[11px] text-slate-500 mt-1">{timeAgo(generated_at)}</div>
          )}
        </div>
      </div>

      {/* Confidence meter */}
      <ConfidenceMeter score={confidence_score} />

      {/* Key levels */}
      {!isNoTrade && (entry_price || target_price || stop_loss) && (
        <div className="grid grid-cols-3 gap-2 py-3 border-y border-slate-700/50">
          <div className="text-center">
            <div className="text-slate-500 text-[11px] mb-0.5">Entry</div>
            <div className="text-slate-200 font-mono text-sm font-semibold">
              {entry_price ? formatPrice(entry_price) : '--'}
            </div>
          </div>
          <div className="text-center">
            <div className="flex items-center justify-center gap-1 text-slate-500 text-[11px] mb-0.5">
              <Target className="w-3 h-3" /> Target
            </div>
            <div className="text-green-400 font-mono text-sm font-semibold">
              {target_price ? formatPrice(target_price) : '--'}
            </div>
          </div>
          <div className="text-center">
            <div className="flex items-center justify-center gap-1 text-slate-500 text-[11px] mb-0.5">
              <Shield className="w-3 h-3" /> Stop Loss
            </div>
            <div className="text-red-400 font-mono text-sm font-semibold">
              {stop_loss ? formatPrice(stop_loss) : '--'}
            </div>
          </div>
        </div>
      )}

      {/* R:R ratio and regime */}
      {(risk_reward_ratio != null || regime) && (
        <div className="flex items-center gap-3 text-xs">
          {risk_reward_ratio != null && (
            <span className="text-slate-400">
              R:R <span className="text-slate-200 font-semibold">{risk_reward_ratio.toFixed(2)}</span>
            </span>
          )}
          {regime && (
            <span className="text-slate-400">
              Regime <span className="text-slate-200 font-semibold">{regime}</span>
            </span>
          )}
        </div>
      )}

      {/* Votes */}
      <div className="space-y-2">
        <div className="text-xs text-slate-500 font-medium">Vote Breakdown</div>
        {VOTES.map((key) => (
          <VoteBar key={key} label={VOTE_LABELS[key]} vote={signal[key]} />
        ))}
      </div>

      {/* Reasoning */}
      {reasoning?.text && (
        <div className="bg-slate-900/50 rounded p-2.5 text-[11px] text-slate-400 leading-relaxed">
          {reasoning.text}
        </div>
      )}
    </div>
  );
}
