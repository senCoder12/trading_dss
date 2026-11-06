import { ExternalLink } from 'lucide-react';
import { Badge } from '../common/Badge';
import { timeAgo } from '../../utils/formatters';

function sentimentInfo(score) {
  if (score == null) return null;
  if (score > 0.2) return { label: 'Bullish', cls: 'bg-up/10 text-up ring-up/25' };
  if (score > 0.05) return { label: 'Mild bull', cls: 'bg-up/10 text-up ring-up/20' };
  if (score < -0.2) return { label: 'Bearish', cls: 'bg-down/10 text-down ring-down/25' };
  if (score < -0.05) return { label: 'Mild bear', cls: 'bg-down/10 text-down ring-down/20' };
  return { label: 'Neutral', cls: 'bg-ink/5 text-ink-muted ring-ink/10' };
}

/** Feeds often repeat the headline as the summary — strip it so rows don't say everything twice. */
function cleanSummary(title, summary) {
  if (!summary) return null;
  const norm = (x) => x.toLowerCase().replace(/\s+/g, ' ').trim();
  const t = norm(title ?? '');
  let s = summary.trim();
  if (t && norm(s).startsWith(t)) s = s.slice(title.trim().length).replace(/^[\s:.\-–—|]+/, '');
  return s.length >= 24 ? s : null;
}

/** One news row inside a divide-y list — no own border. */
export function NewsCard({ article }) {
  const { title, source, url, published_at, sentiment, impact_category, related_indices } = article;
  const summary = cleanSummary(title, article.summary);
  const sent = sentimentInfo(sentiment);
  const related = related_indices ?? [];
  const shown = related.slice(0, 3);
  const extra = related.length - shown.length;

  return (
    <li className="flex items-start gap-4 px-5 py-4 hover:bg-surface-2/50 transition-colors">
      <div className="min-w-0 flex-1">
        <div className="flex items-start gap-2.5">
          {impact_category && (
            <Badge variant={impact_category} kind="severity" size="xs" className="mt-0.5 flex-shrink-0">
              {impact_category}
            </Badge>
          )}
          {url ? (
            <a
              href={url}
              target="_blank"
              rel="noreferrer"
              className="group text-[13.5px] font-medium text-ink leading-snug hover:text-brand transition-colors"
            >
              {title}
              <ExternalLink className="inline w-3 h-3 ml-1.5 -mt-0.5 text-ink-ghost group-hover:text-brand" />
            </a>
          ) : (
            <h4 className="text-[13.5px] font-medium text-ink leading-snug">{title}</h4>
          )}
        </div>

        {summary && (
          <p className="text-[13px] text-ink-faint leading-relaxed line-clamp-2 mt-1.5">{summary}</p>
        )}

        <div className="flex items-center gap-2 flex-wrap mt-2 text-2xs text-ink-faint">
          {source && <span className="text-ink-muted">{source}</span>}
          {source && <span className="text-ink-ghost">·</span>}
          <span>{timeAgo(published_at)}</span>
          {shown.length > 0 && <span className="text-ink-ghost">·</span>}
          {shown.map((id) => (
            <Badge key={id} tone="neutral" size="xs">{id.replace(/_/g, ' ')}</Badge>
          ))}
          {extra > 0 && <span className="text-ink-ghost">+{extra}</span>}
        </div>
      </div>

      {sent && (
        <div className="flex flex-col items-end gap-1 flex-shrink-0 pt-0.5">
          <span className={`inline-flex items-center h-[22px] px-2 rounded-md text-2xs font-medium ring-1 ring-inset ${sent.cls}`}>
            {sent.label}
          </span>
          <span className="text-2xs text-ink-ghost num">
            {sentiment > 0 ? '+' : ''}{sentiment.toFixed(2)}
          </span>
        </div>
      )}
    </li>
  );
}
