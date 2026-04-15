import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, TrendingUp, Briefcase, BarChart2,
  Newspaper, AlertTriangle, Activity, X,
} from 'lucide-react';

const NAV = [
  { to: '/',          icon: LayoutDashboard, label: 'Dashboard',  end: true },
  { to: '/signals',   icon: TrendingUp,      label: 'Signals' },
  { to: '/portfolio', icon: Briefcase,        label: 'Portfolio' },
  { to: '/market',    icon: BarChart2,        label: 'Market' },
  { to: '/news',      icon: Newspaper,        label: 'News' },
  { to: '/anomalies', icon: AlertTriangle,    label: 'Anomalies' },
  { to: '/system',    icon: Activity,         label: 'System' },
];

export default function Sidebar({ open, onClose }) {
  return (
    <>
      {/* Mobile backdrop */}
      {open && (
        <div
          className="fixed inset-0 bg-black/60 z-20 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-30
          w-52 bg-slate-900 border-r border-slate-700
          flex flex-col flex-shrink-0
          transition-transform duration-200 ease-in-out
          ${open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Sidebar header */}
        <div className="flex items-center justify-between px-4 h-12 border-b border-slate-700 flex-shrink-0">
          <span className="text-slate-400 text-xs font-semibold tracking-widest uppercase">Navigation</span>
          <button
            onClick={onClose}
            className="lg:hidden p-1 text-slate-500 hover:text-slate-100"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Nav links */}
        <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
          {NAV.map(({ to, icon: Icon, label, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-500/15 text-blue-400 border border-blue-500/25'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800'
                }`
              }
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-slate-700 text-[10px] text-slate-600">
          v0.1.0 · Trading DSS
        </div>
      </aside>
    </>
  );
}
