import { AreaChart, Area, ResponsiveContainer, Tooltip } from 'recharts';
import { usePolling } from '../../hooks/usePolling';
import { useApi } from '../../hooks/useApi';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Spinner } from '../common/Spinner';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { formatCurrency, formatPercentage, formatPnL } from '../../utils/formatters';
import { REFRESH_INTERVALS } from '../../utils/constants';

function MiniEquityCurve({ history }) {
  if (!history?.length) return null;
  return (
    <div className="h-16 mt-2">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={history} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, fontSize: 11 }}
            formatter={(v) => ['₹' + v.toLocaleString('en-IN'), 'Equity']}
            labelFormatter={(l) => l}
          />
          <Area
            type="monotone"
            dataKey="capital"
            stroke="#3b82f6"
            strokeWidth={1.5}
            fill="url(#eqGrad)"
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function PortfolioSummary() {
  const { data, loading, error, lastUpdated } = usePolling(api.getPortfolio, REFRESH_INTERVALS.portfolio);
  const { data: histData } = useApi(api.getEquityHistory);
  const history = histData?.history ?? [];

  const todayPosColor = (data?.today_pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400';
  const totalRetColor = (data?.total_return_pct ?? 0) >= 0 ? 'text-green-400' : 'text-red-400';

  return (
    <Card
      title="Portfolio"
      padding={false}
      actions={<RefreshIndicator lastUpdated={lastUpdated} error={error} loading={loading} />}
    >
      {loading && !data ? (
        <div className="flex justify-center p-8"><Spinner /></div>
      ) : (
        <div className="p-4">
          {/* Capital */}
          <div className="flex items-end justify-between mb-3">
            <div>
              <div className="text-slate-400 text-xs mb-0.5">Capital</div>
              <div className="text-slate-100 font-mono font-bold text-xl">
                {formatCurrency(data?.capital)}
              </div>
              <div className={`text-xs font-medium ${totalRetColor}`}>
                {formatPercentage(data?.total_return_pct)} overall return
              </div>
            </div>
            <div className="text-right">
              <div className="text-slate-400 text-xs mb-0.5">Today</div>
              <div className={`font-mono font-semibold text-sm ${todayPosColor}`}>
                {formatPnL(data?.today_pnl)}
              </div>
              <div className={`text-xs ${todayPosColor}`}>
                {formatPercentage(data?.today_pnl_pct)}
              </div>
            </div>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-3 gap-3 py-3 border-y border-slate-700/50 text-center">
            <div>
              <div className="text-slate-100 font-semibold text-sm">{data?.open_positions?.length ?? 0}</div>
              <div className="text-slate-500 text-[11px]">Open</div>
            </div>
            <div>
              <div className="text-slate-100 font-semibold text-sm">{data?.today_trades ?? 0}</div>
              <div className="text-slate-500 text-[11px]">Trades Today</div>
            </div>
            <div>
              <div className="text-slate-100 font-semibold text-sm">
                {data?.overall_win_rate != null ? `${data.overall_win_rate.toFixed(0)}%` : '--'}
              </div>
              <div className="text-slate-500 text-[11px]">Win Rate</div>
            </div>
          </div>

          {/* Mini equity curve */}
          <MiniEquityCurve history={history} />
          {!history.length && (
            <div className="text-[11px] text-slate-600 text-center mt-2">No equity history yet</div>
          )}
        </div>
      )}
    </Card>
  );
}
