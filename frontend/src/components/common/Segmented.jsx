/**
 * Segmented control for small mutually-exclusive choices (periods, views…).
 *   options: Array<{ value, label } | primitive>
 */
export function Segmented({ options, value, onChange, size = 'sm', className = '', ariaLabel }) {
  const opts = options.map((o) => (typeof o === 'object' && o !== null ? o : { value: o, label: String(o) }));
  const h = size === 'xs' ? 'h-7 text-xs px-2.5' : 'h-8 text-[13px] px-3';
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={`inline-flex items-center p-0.5 rounded-lg bg-canvas border border-line ${className}`}
    >
      {opts.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={String(o.value)}
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.value)}
            title={o.title}
            className={`${h} rounded-md font-medium whitespace-nowrap transition-colors ${
              active ? 'bg-surface-3 text-ink shadow-sm' : 'text-ink-faint hover:text-ink-muted'
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
