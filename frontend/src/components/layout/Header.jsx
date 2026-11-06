import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { Menu, Settings, ShieldAlert } from 'lucide-react';
import { useDataStore } from '../../hooks/useDataStore';
import { formatIST, getMarketTimeInfo } from '../../utils/formatters';
import { VIX_REGIMES } from '../../utils/constants';
import { routeFor } from '../../routes';

function Pill({ children, className = '', title }) {
  return (
    <div title={title} className={`hidden md:flex items-center gap-2 h-8 px-3 rounded-lg bg-surface border border-line text-xs ${className}`}>
      {children}
    </div>
  );
}

export default function Header({ onMenuClick, onSettingsClick }) {
  const { pathname } = useLocation();
  const route = routeFor(pathname);
  const [now, setNow] = useState(new Date());
  const { vix: { data: vix }, systemHealth: { data: status }, connected } = useDataStore();

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const market = getMarketTimeInfo();
  const vixRegime = vix?.regime ? VIX_REGIMES[vix.regime] : null;
  const killActive = status?.kill_switch_active;

  return (
    <header className="h-14 flex-shrink-0 border-b border-line bg-panel/80 backdrop-blur supports-[backdrop-filter]:bg-panel/70 px-4 lg:px-6 flex items-center gap-4 z-20">
      <button onClick={onMenuClick} className="icon-btn lg:hidden -ml-1" aria-label="Open menu">
        <Menu className="w-5 h-5" />
      </button>

      {/* Page title */}
      <div className="min-w-0 flex-1">
        <h1 className="text-[15px] font-semibold text-ink tracking-tight leading-tight truncate">{route.label}</h1>
        <p className="text-xs text-ink-faint leading-tight truncate hidden sm:block">{route.desc}</p>
      </div>

      {/* Status cluster */}
      <div className="flex items-center gap-2">
        {killActive && (
          <div className="flex items-center gap-1.5 h-8 px-3 rounded-lg bg-down/10 border border-down/30 text-down text-xs font-semibold">
            <ShieldAlert className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Kill switch on</span>
          </div>
        )}

        <Pill title={market.label}>
          <span className={`w-1.5 h-1.5 rounded-full ${market.open ? 'bg-up animate-pulse' : 'bg-ink-ghost'}`} />
          <span className={market.open ? 'text-up font-medium' : 'text-ink-muted'}>
            {market.open ? 'Market open' : 'Market closed'}
          </span>
          <span className="text-ink-faint hidden xl:inline">· {market.detail}</span>
        </Pill>

        {vix?.value != null && (
          <Pill title={`India VIX · ${vixRegime?.label ?? ''} regime`}>
            <span className="text-ink-faint">VIX</span>
            <span className={`font-semibold num ${vixRegime?.color ?? 'text-ink'}`}>{vix.value.toFixed(2)}</span>
          </Pill>
        )}

        <Pill title={connected ? 'Live feed connected' : 'Live feed disconnected — reconnecting'}>
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-up' : 'bg-down animate-pulse'}`} />
          <span className={connected ? 'text-ink-muted' : 'text-down'}>{connected ? 'Live' : 'Offline'}</span>
        </Pill>

        <div className="hidden sm:flex items-baseline gap-1 px-2 text-ink-muted" title="Indian Standard Time">
          <span className="font-mono text-xs num">{formatIST(now)}</span>
          <span className="text-[10px] text-ink-ghost">IST</span>
        </div>

        <button onClick={onSettingsClick} className="icon-btn" title="Quick settings (S)" aria-label="Quick settings">
          <Settings className="w-[18px] h-[18px]" />
        </button>
      </div>
    </header>
  );
}
