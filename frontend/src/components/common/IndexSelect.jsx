import { ChevronDown } from 'lucide-react';
import { useIndexMeta, indexLabel } from '../../hooks/useIndexMeta';
import { INDEX_CATEGORY_LABELS } from '../../utils/constants';

const CATEGORY_ORDER = ['broad_market', 'sectoral', 'thematic', 'strategy', 'volatility'];

/**
 * Compact index picker grouped by registry category.
 *   ids: index ids to offer (defaults to every id in the registry)
 */
export function IndexSelect({ value, onChange, ids, className = '' }) {
  const meta = useIndexMeta();
  const all = ids ?? Object.keys(meta);

  const groups = {};
  for (const id of all) {
    const cat = meta[id]?.sector_category ?? 'other';
    (groups[cat] ??= []).push(id);
  }
  const cats = [...CATEGORY_ORDER.filter((c) => groups[c]), ...Object.keys(groups).filter((c) => !CATEGORY_ORDER.includes(c))];

  return (
    <div className={`relative ${className}`}>
      <select
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        className="select h-8 text-[13px] font-medium pl-3 bg-surface-2"
        aria-label="Select index"
      >
        {!all.includes(value) && value && <option value={value}>{indexLabel(meta, value)}</option>}
        {cats.map((cat) => (
          <optgroup key={cat} label={INDEX_CATEGORY_LABELS[cat] ?? 'Other'}>
            {groups[cat].map((id) => (
              <option key={id} value={id}>{indexLabel(meta, id)}</option>
            ))}
          </optgroup>
        ))}
      </select>
      <ChevronDown className="w-4 h-4 text-ink-faint absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
    </div>
  );
}
