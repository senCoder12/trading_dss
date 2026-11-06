import { useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';

/**
 * Tab state kept in the URL (?tab=…) so tabs are linkable and survive reloads.
 */
export function useTabParam(defaultTab, param = 'tab') {
  const [params, setParams] = useSearchParams();
  const value = params.get(param) ?? defaultTab;
  const setValue = useCallback((next) => {
    setParams((prev) => {
      const p = new URLSearchParams(prev);
      if (next === defaultTab) p.delete(param);
      else p.set(param, next);
      return p;
    }, { replace: true });
  }, [setParams, defaultTab, param]);
  return [value, setValue];
}

/**
 * Underline tabs.
 *   tabs: [{ id, label, count?, icon? }]
 */
export function Tabs({ tabs, value, onChange, className = '', right }) {
  return (
    <div className={`flex items-end justify-between gap-4 border-b border-line ${className}`}>
      <div role="tablist" className="flex items-end gap-1 overflow-x-auto -mb-px">
        {tabs.map(({ id, label, count, icon: Icon }) => {
          const active = id === value;
          return (
            <button
              key={id}
              role="tab"
              aria-selected={active}
              onClick={() => onChange(id)}
              className={`relative flex items-center gap-2 px-3 pb-2.5 pt-1 text-[13px] font-medium whitespace-nowrap transition-colors
                border-b-2 ${active
                  ? 'text-ink border-brand'
                  : 'text-ink-faint border-transparent hover:text-ink-muted'}`}
            >
              {Icon && <Icon className="w-4 h-4" />}
              {label}
              {count != null && (
                <span className={`min-w-[20px] h-5 px-1.5 rounded-full text-[10.5px] font-semibold num inline-flex items-center justify-center
                  ${active ? 'bg-brand/15 text-brand' : 'bg-surface-3 text-ink-faint'}`}>
                  {count}
                </span>
              )}
            </button>
          );
        })}
      </div>
      {right && <div className="flex items-center gap-2 pb-2 flex-shrink-0">{right}</div>}
    </div>
  );
}
