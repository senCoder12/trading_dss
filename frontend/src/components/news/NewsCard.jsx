import { ExternalLink } from 'lucide-react';
import { Badge } from '../common/Badge';
import { timeAgo } from '../../utils/formatters';
import { SENTIMENT_LABELS } from '../../utils/constants';

function sentimentBar(score) {
  // score: -1 to 1
  const pct = ((score + 1) / 2) * 100;
  const color = score > 0.1 ? 'bg-green-500' : score < -0.1 ? 'bg-red-500' : 'bg-slate-500';
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-slate-500 text-[10px]">Bear</span>
      <div className="flex-1 h-1 bg-slate-700 rounded-full overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${Math.max(2, pct)}%` }} />
      </div>
      <span className="text-slate-500 text-[10px]">Bull</span>
    </div>
  );
}

export function NewsCard({ article }) {
  const { title, summary, source, url, published_at, sentiment, impact_category, related_indices } = article;

  const sentLabel = sentiment != null
    ? (sentiment > 0.2 ? 'BULLISH' : sentiment > 0.05 ? 'SLIGHTLY_BULLISH' : sentiment < -0.2 ? 'BEARISH' : sentiment < -0.05 ? 'SLIGHTLY_BEARISH' : 'NEUTRAL')
    : 'NEUTRAL';
  const sentInfo = SENTIMENT_LABELS[sentLabel] ?? SENTIMENT_LABELS.NEUTRAL;

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 space-y-2 hover:border-slate-600 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-start gap-2 flex-1 min-w-0">
          {impact_category && (
            <Badge variant={impact_category} size="xs" className="flex-shrink-0 mt-0.5">
              {impact_category}
            </Badge>
          )}
          <h4 className="text-slate-200 text-sm leading-snug">{title}</h4>
        </div>
        {url && (
          <a href={url} target="_blank" rel="noreferrer" className="text-slate-500 hover:text-blue-400 flex-shrink-0">
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        )}
      </div>

      {/* Summary */}
      {summary && (
        <p className="text-slate-400 text-xs leading-relaxed line-clamp-2">{summary}</p>
      )}

      {/* Sentiment bar */}
      {sentiment != null && (
        <div className="space-y-1">
          {sentimentBar(sentiment)}
          <div className="flex items-center justify-between text-[10px]">
            <span className={sentInfo.color}>{sentInfo.label}</span>
            <span className="text-slate-600 font-mono">{sentiment.toFixed(3)}</span>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center gap-2 flex-wrap text-[10px] text-slate-500">
        {source && <span>{source}</span>}
        <span>·</span>
        <span>{timeAgo(published_at)}</span>
        {related_indices?.length > 0 && (
          <>
            <span>·</span>
            <span className="text-slate-400">{related_indices.join(', ')}</span>
          </>
        )}
      </div>
    </div>
  );
}
