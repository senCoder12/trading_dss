import { useState } from 'react';
import { usePolling } from '../../hooks/usePolling';
import { useApi } from '../../hooks/useApi';
import { api } from '../../api/client';
import { Card } from '../common/Card';
import { Spinner } from '../common/Spinner';
import { RefreshIndicator } from '../common/RefreshIndicator';
import { EquityCurve } from './EquityCurve';
import { PositionsList } from './PositionsList';
import { TradeHistory } from './TradeHistory';
import { PerformanceStats } from './PerformanceStats';
import { formatCurrency, formatPercentage, formatPnL } from '../../utils/formatters';
import { REFRESH_INTERVALS } from '../../utils/constants';

const PERIODS = [7, 14, 30, 90];
const EQ_PERIODS = [7, 30, 90];

export default function PortfolioPage() {
  const [tradeDays, setTradeDays] = useState(7);
  const [eqDays, setEqDays] = useState(30);
  const [perfDays, setPerfDays] = useState(30);

  const { data: portfolio, loading: portLoading, lastUpdated } = usePolling(
    api.getPortfolio, REFRESH_INTERVALS.portfolio,
  );
  const { data: histData, loading: histLoading } = useApi(
    () => api.getEquityHistory(eqDays), [eqDays],
  );
  const { data: tradesData, loading: tradesLoading } = useApi(
    () => api.getTrades(tradeDays), [tradeDays],
  );
  const { data: perf, loading: perfLoading } = useApi(
    () => api.getPerformance(perfDays), [perfDays],
  );

  const history = histData?.history ?? [];
  const trades = tradesData?.trades ?? [];
  const positions = portfolio?.open_positions ?? [];
  const todayPnlColor = (portfolio?.today_pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400';
  const totalRetColor = (portfolio?.total_return_pct ?? 0) >= 0 ? 'text-green-400' : 'text-red-400';

  return (
    <div className="p-4 space-y-4">
      {/* Portfolio header stats */}
      <Card padding={false}>
        <div className="p-4">
          {portLoading && !portfolio ? (
            <div className="flex justify-center p-6"><Spinner /></div>
          ) : (
            <div className="flex flex-wrap items-center gap-6 justify-between">
              <div>
                <div className="text-slate-400 text-xs mb-0.5">Total Capital</div>
                <div className="text-slate-100 font-mono font-bold text-2xl">
                  {formatCurrency(portfolio?.capital)}
                </div>
                <div className={`text-sm font-medium ${totalRetColor}`}>
                  {formatPercentage(portfolio?.total_return_pct)} overall
                </div>
              </div>
              <div className="text-right">
                <div className="text-slate-400 text-xs mb-0.5">Today's P&L</div>
                <div className={`font-mono font-bold text-xl ${todayPnlColor}`}>
                  {formatPnL(portfolio?.today_pnl)}
                </div>
                <div className={`text-xs ${todayPnlColor}`}>
                  {formatPercentage(portfolio?.today_pnl_pct)}
                </div>
              </div>
              <div className="flex gap-6">
                {[
                  { label: 'Open', value: positions.length },
                  { label: 'Win Rate', value: portfolio?.overall_win_rate != null ? `${portfolio.overall_win_rate.toFixed(0)}%` : '--' },
                  { label: 'Trades (All)', value: portfolio?.total_closed_trades ?? '--' },
                ].map(({ label, value }) => (
                  <div key={label} className="text-center">
                    <div className="text-slate-100 font-semibold text-sm">{value}</div>
                    <div className="text-slate-500 text-xs">{label}</div>
                  </div>
                ))}
              </div>
              <RefreshIndicator lastUpdated={lastUpdated} />
            </div>
          )}
        </div>
      </Card>

      {/* Equity curve */}
      <Card title="Equity Curve" padding={false}>
        <div className="px-4 py-2 border-b border-slate-700 flex items-center gap-2">
          {EQ_PERIODS.map((d) => (
            <button
              key={d}
              onClick={() => setEqDays(d)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                eqDays === d
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
              }`}
            >
              {d}d
            </button>
          ))}
        </div>
        <div className="px-4 py-3">
          {histLoading ? <div className="flex justify-center p-6"><Spinner /></div>
            : <EquityCurve history={history} />}
        </div>
      </Card>

      {/* Open positions */}
      <Card title={`Open Positions (${positions.length})`} padding={false}>
        <PositionsList positions={positions} />
      </Card>

      {/* Performance stats */}
      <Card title="Performance Statistics" padding={false}>
        <div className="px-4 py-3 border-b border-slate-700 flex items-center gap-2">
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
        <div className="p-4">
          {perfLoading ? <div className="flex justify-center p-4"><Spinner /></div>
            : <PerformanceStats stats={perf} />}
        </div>
      </Card>

      {/* Trade history */}
      <Card title="Trade History" padding={false}>
        <div className="px-4 py-3 border-b border-slate-700 flex items-center gap-2">
          {PERIODS.map((d) => (
            <button
              key={d}
              onClick={() => setTradeDays(d)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                tradeDays === d
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
              }`}
            >
              {d}d
            </button>
          ))}
          {tradesData?.total != null && (
            <span className="ml-auto text-slate-500 text-xs">{tradesData.total} total</span>
          )}
        </div>
        {tradesLoading ? <div className="flex justify-center p-6"><Spinner /></div>
          : <TradeHistory trades={trades} />}
      </Card>
    </div>
  );
}
