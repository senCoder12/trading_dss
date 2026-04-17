import { useEffect, useState } from 'react';
import { Menu, Clock, Activity, AlertTriangle, Wifi, WifiOff, Settings } from 'lucide-react';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { formatIST, getMarketTimeInfo } from '../../utils/formatters';
import { VIX_REGIMES } from '../../utils/constants';

export default function Header({ onMenuClick, wsConnected = false, onSettingsClick }) {
  const [now, setNow] = useState(new Date());
  const { data: vix } = usePolling(api.getVix, 30_000);
  const { data: status } = usePolling(api.getSystemStatus, 30_000);

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const market = getMarketTimeInfo();
  const vixRegime = vix?.regime ? VIX_REGIMES[vix.regime] : null;
  const killActive = status?.kill_switch_active;

  return (
    <header className="bg-slate-900 border-b border-slate-700 px-4 py-2.5 flex items-center justify-between flex-shrink-0 z-10">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="lg:hidden p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800"
          aria-label="Open menu"
        >
          <Menu className="w-5 h-5" />
        </button>
        <span className="text-slate-100 font-bold text-base tracking-tight select-none">
          Trading <span className="text-blue-400">DSS</span>
        </span>
      </div>

      <div className="flex items-center gap-3 sm:gap-5 text-sm">
        {/* Kill switch warning */}
        {killActive && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-red-500/20 border border-red-500/40">
            <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
            <span className="text-red-400 font-semibold text-xs">KILL SWITCH ON</span>
          </div>
        )}

        {/* WebSocket connection status */}
        <div className="flex items-center gap-1" title={wsConnected ? 'Live connection' : 'Disconnected — reconnecting'}>
          {wsConnected ? (
            <Wifi className="w-3.5 h-3.5 text-green-400" />
          ) : (
            <WifiOff className="w-3.5 h-3.5 text-red-400 animate-pulse" />
          )}
          <span className={`text-[10px] font-medium hidden sm:block ${wsConnected ? 'text-green-400' : 'text-red-400'}`}>
            {wsConnected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>

        {/* Market status */}
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full flex-shrink-0 ${
              market.open ? 'bg-green-400 animate-pulse' : 'bg-slate-600'
            }`}
          />
          <span
            className={`font-medium hidden sm:block ${
              market.open ? 'text-green-400' : 'text-slate-400'
            }`}
          >
            {market.label}
          </span>
        </div>

        {/* VIX */}
        {vix?.value != null && (
          <div className="hidden sm:flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-slate-400 text-xs">VIX</span>
            <span className={`font-mono font-semibold ${vixRegime?.color ?? 'text-slate-300'}`}>
              {vix.value.toFixed(2)}
            </span>
            {vixRegime && (
              <span className={`text-[10px] px-1.5 py-0.5 rounded ${vixRegime.bg} ${vixRegime.color}`}>
                {vixRegime.label}
              </span>
            )}
          </div>
        )}

        {/* Settings */}
        <button
          onClick={onSettingsClick}
          className="p-1.5 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          title="Quick Settings (S)"
        >
          <Settings className="w-4 h-4" />
        </button>

        {/* IST clock */}
        <div className="flex items-center gap-1.5 text-slate-500">
          <Clock className="w-3.5 h-3.5" />
          <span className="font-mono text-xs tracking-wide">{formatIST(now)}</span>
          <span className="text-[10px] text-slate-600">IST</span>
        </div>
      </div>
    </header>
  );
}
