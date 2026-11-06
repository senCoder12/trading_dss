/**
 * Surface container. Title row is optional; `actions` sit right-aligned in it.
 * `padding` controls the body padding (false for tables / edge-to-edge content).
 */
export function Card({
  children, className = '', bodyClassName = '', title, subtitle, icon: Icon, actions, padding = true, footer,
}) {
  return (
    <section className={`card flex flex-col min-w-0 ${className}`}>
      {(title || actions) && (
        <header className="card-header">
          <div className="flex items-center gap-2.5 min-w-0">
            {Icon && <Icon className="w-4 h-4 text-ink-faint flex-shrink-0" />}
            <div className="min-w-0">
              {title && <h3 className="card-title truncate">{title}</h3>}
              {subtitle && <p className="text-xs text-ink-faint mt-0.5 truncate">{subtitle}</p>}
            </div>
          </div>
          {actions && <div className="flex items-center gap-2 flex-shrink-0">{actions}</div>}
        </header>
      )}
      <div className={`${padding ? 'p-5' : ''} flex-1 min-h-0 ${bodyClassName}`}>{children}</div>
      {footer && <footer className="px-5 py-3 border-t border-line">{footer}</footer>}
    </section>
  );
}
