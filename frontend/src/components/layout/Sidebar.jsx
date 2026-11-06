import { NavLink } from 'react-router-dom';
import { X, PanelLeftClose, PanelLeftOpen, Keyboard } from 'lucide-react';
import { NAV_GROUPS } from '../../routes';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';

const HEALTH_DOT = {
  OK: 'bg-up', HEALTHY: 'bg-up', WARNING: 'bg-warn', DEGRADED: 'bg-warn', ERROR: 'bg-down', UNKNOWN: 'bg-ink-ghost',
};

function BrandMark() {
  return (
    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand to-violet flex items-center justify-center flex-shrink-0 shadow-[0_0_0_1px_rgb(255_255_255/0.08)_inset]">
      <svg viewBox="0 0 24 24" className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 16l5-5 4 4 7-8" />
      </svg>
    </div>
  );
}

function ApiHealth({ collapsed }) {
  const { data: health, error } = usePolling(api.getSystemHealth, 30_000);
  const overall = error ? 'ERROR' : health?.overall_status ?? 'UNKNOWN';
  const label = error ? 'API unreachable' : overall === 'OK' || overall === 'HEALTHY' ? 'All systems normal' : `System ${overall.toLowerCase()}`;

  return (
    <NavLink
      to="/system"
      title={label}
      className={`flex items-center gap-2.5 rounded-lg text-xs text-ink-faint hover:text-ink-muted hover:bg-surface-2 transition-colors
        ${collapsed ? 'justify-center h-9' : 'px-3 h-9'}`}
    >
      <span className="relative flex w-2 h-2 flex-shrink-0">
        {overall === 'OK' && <span className="absolute inline-flex h-full w-full rounded-full bg-up opacity-40 animate-ping" />}
        <span className={`relative inline-flex w-2 h-2 rounded-full ${HEALTH_DOT[overall] ?? HEALTH_DOT.UNKNOWN}`} />
      </span>
      {!collapsed && <span className="truncate">{label}</span>}
    </NavLink>
  );
}

export default function Sidebar({ open, onClose, collapsed, onToggleCollapsed, onShowShortcuts }) {
  return (
    <>
      {/* Mobile backdrop */}
      {open && <div className="fixed inset-0 bg-black/60 z-30 lg:hidden" onClick={onClose} />}

      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-40 flex flex-col flex-shrink-0
          bg-panel border-r border-line transition-[transform,width] duration-200 ease-out
          ${collapsed ? 'lg:w-[68px]' : 'lg:w-60'} w-60
          ${open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Brand */}
        <div className={`flex items-center h-14 border-b border-line flex-shrink-0 ${collapsed ? 'lg:justify-center lg:px-0 px-4' : 'px-4'} gap-2.5`}>
          <BrandMark />
          <div className={`min-w-0 ${collapsed ? 'lg:hidden' : ''}`}>
            <div className="text-sm font-semibold text-ink leading-tight tracking-tight">Trading DSS</div>
            <div className="text-2xs text-ink-faint leading-tight">Decision support</div>
          </div>
          <button onClick={onClose} className="icon-btn ml-auto lg:hidden" aria-label="Close menu">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-5">
          {NAV_GROUPS.map((group) => (
            <div key={group.label}>
              <div className={`px-3 mb-1.5 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-ink-ghost ${collapsed ? 'lg:hidden' : ''}`}>
                {group.label}
              </div>
              {collapsed && <div className="hidden lg:block mx-3 mb-2 border-t border-line first:hidden" />}
              <div className="space-y-0.5">
                {group.items.map(({ to, icon: Icon, label, end, key }) => (
                  <NavLink
                    key={to}
                    to={to}
                    end={end}
                    onClick={onClose}
                    title={collapsed ? `${label} (${key})` : undefined}
                    className={({ isActive }) =>
                      `group relative flex items-center gap-3 h-9 rounded-lg text-[13px] font-medium transition-colors
                      ${collapsed ? 'lg:justify-center lg:px-0 px-3' : 'px-3'}
                      ${isActive
                        ? 'bg-surface-3 text-ink'
                        : 'text-ink-faint hover:text-ink hover:bg-surface-2'}`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        {isActive && <span className="absolute left-0 top-2 bottom-2 w-[3px] rounded-r bg-brand -ml-3" />}
                        <Icon className={`w-[18px] h-[18px] flex-shrink-0 ${isActive ? 'text-brand' : ''}`} />
                        <span className={`truncate ${collapsed ? 'lg:hidden' : ''}`}>{label}</span>
                        <kbd className={`ml-auto text-[10px] font-mono text-ink-ghost opacity-0 group-hover:opacity-100 transition-opacity ${collapsed ? 'lg:hidden' : ''}`}>
                          {key}
                        </kbd>
                      </>
                    )}
                  </NavLink>
                ))}
              </div>
            </div>
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t border-line p-3 space-y-0.5 flex-shrink-0">
          <ApiHealth collapsed={collapsed} />
          <button
            onClick={onShowShortcuts}
            title="Keyboard shortcuts (?)"
            className={`w-full flex items-center gap-2.5 h-9 rounded-lg text-xs text-ink-faint hover:text-ink-muted hover:bg-surface-2 transition-colors
              ${collapsed ? 'lg:justify-center lg:px-0 px-3' : 'px-3'}`}
          >
            <Keyboard className="w-4 h-4 flex-shrink-0" />
            <span className={collapsed ? 'lg:hidden' : ''}>Shortcuts</span>
            <kbd className={`ml-auto text-[10px] font-mono text-ink-ghost ${collapsed ? 'lg:hidden' : ''}`}>?</kbd>
          </button>
          <button
            onClick={onToggleCollapsed}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className={`hidden lg:flex w-full items-center gap-2.5 h-9 rounded-lg text-xs text-ink-faint hover:text-ink-muted hover:bg-surface-2 transition-colors
              ${collapsed ? 'justify-center' : 'px-3'}`}
          >
            {collapsed ? <PanelLeftOpen className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
            {!collapsed && <span>Collapse</span>}
          </button>
        </div>
      </aside>
    </>
  );
}
