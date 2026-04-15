import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { REFRESH_INTERVALS, SENTIMENT_LABELS, VIX_REGIMES } from '../../utils/constants';
import { Gauge } from 'lucide-react';

function VixBar({ value, regime }) {
  // 0-40+ scale, clamp to 40
  const pct = Math.min(100, ((value ?? 0) / 40) * 100);
  const colors = {
    LOW: 'bg-green-400',
    NORMAL: 'bg-blue-400',
    ELEVATED: 'bg-yellow-400',
    HIGH: 'bg-red-400',
    EXTREME: 'bg-red-600',
  };
  return (
    <div className="w-full">
      <div className="flex items-center justify-between text-xs mb-1">
        <span className="text-slate-500">0</span>
        <span className="text-slate-500">40+</span>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${colors[regime] ?? 'bg-blue-400'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export default function MarketSentiment() {
  const { data: vix, error: vixErr, lastUpdated: vixTs } = usePolling(api.getVix, REFRESH_INTERVALS.vix);
  const { data: news, lastUpdated: newsTs } = usePolling(api.getNewsSummary, REFRESH_INTERVALS.news);

  const regime = vix?.regime ?? 'UNKNOWN';
  const vixInfo = VIX_REGIMES[regime] ?? { label: regime, color: 'text-slate-400', bg: 'bg-slate-700' };
  const sentLabel = news?.overall_sentiment_label ?? 'NEUTRAL';
  const sentInfo = SENTIMENT_LABELS[sentLabel] ?? SENTIMENT_LABELS.NEUTRAL;

  return (
    <Card
      title="Market Sentiment"
      padding={false}
      actions={<RefreshIndicator lastUpdated={vixTs ?? newsTs} error={vixErr} />}
    >
      <div className="p-4 space-y-4">
        {/* VIX */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <Gauge className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400 text-xs font-medium">India VIX</span>
            </div>
            {vix?.value != null && (
              <div className="flex items-center gap-2">
                <span className={`font-mono font-bold text-lg ${vixInfo.color}`}>
                  {vix.value.toFixed(2)}
                </span>
                <span className={`text-xs px-2 py-0.5 rounded ${vixInfo.bg} ${vixInfo.color}`}>
                  {vixInfo.label}
                </span>
              </div>
            )}
          </div>
          {vix?.value != null && <VixBar value={vix.value} regime={regime} />}
        </div>

        {/* News Sentiment */}
        <div className="border-t border-slate-700/50 pt-3">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 text-xs font-medium">News Sentiment</span>
            <div className="flex items-center gap-2">
              <span className={`font-semibold text-sm ${sentInfo.color}`}>{sentInfo.label}</span>
              {news?.overall_sentiment != null && (
                <span className="text-slate-600 text-xs font-mono">
                  ({news.overall_sentiment >= 0 ? '+' : ''}{news.overall_sentiment.toFixed(3)})
                </span>
              )}
            </div>
          </div>
          {news && (
            <div className="flex items-center gap-3 mt-2 text-[11px] text-slate-500">
              <span>{news.total_articles ?? 0} articles today</span>
              {news.critical_count > 0 && (
                <span className="text-red-400">{news.critical_count} critical</span>
              )}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
