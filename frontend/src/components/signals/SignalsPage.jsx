import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { usePolling } from '../../hooks/usePolling';
import { useApi } from '../../hooks/useApi';
import { api } from '../../api/client';
import { SignalCard } from './SignalCard';
import { SignalHistory } from './SignalHistory';
import { Card } from '../common/Card';
import { Spinner } from '../common/Spinner';
import { EmptyState } from '../common/EmptyState';
import { formatPercentage } from '../../utils/formatters';
import { REFRESH_INTERVALS, CHART_COLORS } from '../../utils/constants';
import { TrendingUp } from 'lucide-react';

const PERIODS = [7, 14, 30, 90];

function PerformanceOverview({ perf }) {
  if (!perf) return null;

  const confData = [
    { name: 'High', wr: perf.high_confidence_win_rate },
    { name: 'Medium', wr: perf.medium_confidence_win_rate },
    { name: 'Low', wr: perf.low_confidence_win_rate },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
      {[
        { label: 'Win Rate', value: formatPercentage(perf.win_rate), color: perf.win_rate >= 50 ? 'text-green-400' : 'text-red-400' },
        { label: 'Total Trades', value: perf.total_trades, color: 'text-slate-200' },
        { label: 'Profit Factor', value: perf.profit_factor?.toFixed(2) ?? '--', color: perf.profit_factor >= 1 ? 'text-green-400' : 'text-red-400' },
        { label: 'Max Drawdown', value: formatPercentage(perf.max_drawdown), color: 'text-red-400' },
      ].map(({ label, value, color }) => (
        <div key={label} className="bg-slate-800 rounded-lg border border-slate-700 p-3 text-center">
          <div className={`text-lg font-bold font-mono ${color}`}>{value}</div>
          <div className="text-slate-500 text-xs mt-0.5">{label}</div>
        </div>
      ))}

      <div className="col-span-2 sm:col-span-4 bg-slate-800 rounded-lg border border-slate-700 p-3">
        <div className="text-slate-400 text-xs font-medium mb-2">Win Rate by Confidence Level</div>
        <div className="h-24">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={confData} margin={{ top: 2, right: 8, bottom: 0, left: -20 }}>
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#94a3b8' }} />
              <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, fontSize: 11 }}
                formatter={(v) => [`${v?.toFixed(1)}%`, 'Win Rate']}
              />
              <Bar dataKey="wr" radius={[3, 3, 0, 0]}>
                {confData.map((_, i) => (
                  <Cell key={i} fill={[CHART_COLORS.green, CHART_COLORS.yellow, CHART_COLORS.blue][i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default function SignalsPage() {
  const [histDays, setHistDays] = useState(7);
  const [perfDays, setPerfDays] = useState(30);

  const { data: current, loading: currentLoading } = usePolling(
    api.getCurrentSignals, REFRESH_INTERVALS.signals,
  );
  const { data: history, loading: histLoading } = useApi(
    () => api.getSignalHistory(histDays), [histDays],
  );
  const { data: perf, loading: perfLoading } = useApi(
    () => api.getPerformance(perfDays), [perfDays],
  );

  const signals = current?.signals ?? [];
  const histSignals = history?.signals ?? [];

  return (
    <div className="p-4 space-y-4">
      {/* Current signals */}
      <div>
        <h2 className="text-slate-100 font-semibold text-base mb-3">Current Signals</h2>
        {currentLoading && !current ? (
          <div className="flex justify-center p-8"><Spinner /></div>
        ) : !signals.length ? (
          <EmptyState icon={TrendingUp} title="No current signals" />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-2 gap-4">
            {signals.map((sig) => (
              <SignalCard key={sig.index_id} signal={sig} />
            ))}
          </div>
        )}
      </div>

      {/* Performance */}
      <Card title="Signal Performance" padding={false}>
        <div className="p-4 space-y-4">
          <div className="flex items-center gap-2">
            {PERIODS.map((d) => (
              <button
                key={d}
                onClick={() => setPerfDays(d)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  perfDays === d
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
                }`}
              >
                {d}d
              </button>
            ))}
          </div>
          {perfLoading ? (
            <div className="flex justify-center p-6"><Spinner /></div>
          ) : (
            <PerformanceOverview perf={perf} />
          )}
          {perf?.edge_comment && (
            <p className="text-xs text-slate-500 italic">{perf.edge_comment}</p>
          )}
        </div>
      </Card>

      {/* Signal history */}
      <Card title="Signal History" padding={false}>
        <div className="px-4 py-3 border-b border-slate-700 flex items-center gap-2">
          {PERIODS.map((d) => (
            <button
              key={d}
              onClick={() => setHistDays(d)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                histDays === d
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
              }`}
            >
              {d}d
            </button>
          ))}
          {history?.total != null && (
            <span className="ml-auto text-slate-500 text-xs">{history.total} total</span>
          )}
        </div>
        <SignalHistory signals={histSignals} loading={histLoading} />
      </Card>
    </div>
  );
}
