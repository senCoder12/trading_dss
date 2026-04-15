import { useState } from 'react';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Spinner } from '../common/Spinner';
import { Badge } from '../common/Badge';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { ComponentHealth } from './ComponentHealth';
import { REFRESH_INTERVALS } from '../../utils/constants';
import { AlertTriangle, Power, PowerOff, Database, Activity } from 'lucide-react';
import { timeAgo } from '../../utils/formatters';

function DataFreshnessRow({ row }) {
  const age = row.age_seconds;
  const stale = age != null && age > 300;
  const fresh = age != null && age < 60;
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-slate-700/30 text-xs">
      <div className="flex items-center gap-2">
        <Database className="w-3.5 h-3.5 text-slate-500" />
        <span className="text-slate-400 font-mono">{row.table}</span>
      </div>
      <div className={`font-mono ${fresh ? 'text-green-400' : stale ? 'text-red-400' : 'text-yellow-400'}`}>
        {age != null ? (age < 60 ? `${age}s` : `${Math.floor(age / 60)}m ${age % 60}s`) : '--'}
      </div>
    </div>
  );
}

export default function SystemPage() {
  const [killReason, setKillReason] = useState('');
  const [confirming, setConfirming] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  const { data: health, loading: healthLoading, lastUpdated: healthTs, error: healthErr } = usePolling(
    api.getSystemHealth, REFRESH_INTERVALS.system,
  );
  const { data: status, loading: statusLoading, lastUpdated: statusTs } = usePolling(
    api.getSystemStatus, REFRESH_INTERVALS.system,
  );

  const killActive = status?.kill_switch_active ?? false;

  async function handleKillSwitch() {
    if (!killActive && !killReason.trim()) return;
    setActionLoading(true);
    try {
      if (killActive) {
        await api.deactivateKillSwitch();
        setConfirming(false);
      } else {
        await api.activateKillSwitch(killReason.trim() || 'Manual activation');
        setKillReason('');
        setConfirming(false);
      }
    } finally {
      setActionLoading(false);
    }
  }

  return (
    <div className="p-4 space-y-4">
      {/* Kill switch — MOST PROMINENT */}
      <div className={`rounded-lg border p-4 ${killActive
        ? 'bg-red-500/10 border-red-500/50'
        : 'bg-slate-800 border-slate-700'}`}
      >
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            {killActive ? (
              <PowerOff className="w-6 h-6 text-red-400" />
            ) : (
              <Power className="w-6 h-6 text-green-400" />
            )}
            <div>
              <div className="text-slate-100 font-bold">Kill Switch</div>
              <div className={`text-sm ${killActive ? 'text-red-400' : 'text-green-400'}`}>
                {killActive ? `ACTIVE — ${status.kill_switch_reason ?? 'All trading halted'}` : 'INACTIVE — Trading allowed'}
              </div>
            </div>
          </div>

          {killActive ? (
            <button
              onClick={() => { setConfirming(false); handleKillSwitch(); }}
              disabled={actionLoading}
              className="px-4 py-2 bg-green-500/20 text-green-400 border border-green-500/40 rounded text-sm font-medium hover:bg-green-500/30 transition-colors disabled:opacity-50"
            >
              {actionLoading ? 'Working...' : 'Deactivate'}
            </button>
          ) : confirming ? (
            <div className="flex items-center gap-2 flex-wrap">
              <input
                className="bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-slate-200 w-48 focus:outline-none focus:border-red-500"
                placeholder="Reason..."
                value={killReason}
                onChange={(e) => setKillReason(e.target.value)}
              />
              <button
                onClick={handleKillSwitch}
                disabled={actionLoading || !killReason.trim()}
                className="px-3 py-1.5 bg-red-500/20 text-red-400 border border-red-500/40 rounded text-sm font-medium hover:bg-red-500/30 transition-colors disabled:opacity-50"
              >
                {actionLoading ? 'Activating...' : 'Confirm'}
              </button>
              <button
                onClick={() => { setConfirming(false); setKillReason(''); }}
                className="px-3 py-1.5 text-slate-400 hover:text-slate-200 text-sm"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirming(true)}
              className="px-4 py-2 bg-red-500/20 text-red-400 border border-red-500/40 rounded text-sm font-medium hover:bg-red-500/30 transition-colors"
            >
              <div className="flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4" /> Activate Kill Switch
              </div>
            </button>
          )}
        </div>
      </div>

      {/* Market + system status */}
      {status && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: 'Market', value: status.market_status, color: status.market_status === 'OPEN' ? 'text-green-400' : 'text-slate-400' },
            { label: 'Uptime', value: status.uptime_seconds != null ? `${Math.floor(status.uptime_seconds / 3600)}h ${Math.floor((status.uptime_seconds % 3600) / 60)}m` : '--', color: 'text-slate-200' },
            { label: 'Session', value: status.market_session ?? '--', color: 'text-slate-300' },
            { label: 'Time Remaining', value: status.time_remaining ?? '--', color: 'text-slate-300' },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-slate-800 border border-slate-700 rounded-lg p-3 text-center">
              <div className={`font-bold text-base ${color}`}>{value}</div>
              <div className="text-slate-500 text-xs mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Component health grid */}
      <Card
        title="Component Health"
        padding={false}
        actions={<RefreshIndicator lastUpdated={healthTs} error={healthErr} loading={healthLoading} />}
      >
        <div className="p-4">
          {healthLoading && !health ? (
            <div className="flex justify-center p-6"><Spinner /></div>
          ) : (
            <>
              {health?.overall_status && (
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-slate-400 text-xs">Overall:</span>
                  <Badge variant={health.overall_status}>{health.overall_status}</Badge>
                  {health.db_size && (
                    <span className="text-slate-500 text-xs ml-auto">DB size: {health.db_size}</span>
                  )}
                </div>
              )}
              {health?.components?.length ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {health.components.map((c) => (
                    <ComponentHealth key={c.component} component={c} />
                  ))}
                </div>
              ) : (
                <div className="text-slate-500 text-sm text-center py-4">No component data</div>
              )}
            </>
          )}
        </div>
      </Card>

      {/* Data freshness */}
      {status?.data_freshness?.length > 0 && (
        <Card title="Data Freshness" padding={false}>
          <div className="px-4 py-2">
            <div className="flex items-center gap-3 text-[10px] text-slate-500 mb-2">
              <span className="text-green-400">● &lt; 1m</span>
              <span className="text-yellow-400">● 1-5m</span>
              <span className="text-red-400">● &gt; 5m</span>
            </div>
            {status.data_freshness.map((row) => (
              <DataFreshnessRow key={row.table} row={row} />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
