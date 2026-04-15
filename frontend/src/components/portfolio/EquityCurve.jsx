import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  CartesianGrid, ReferenceLine, ResponsiveContainer,
} from 'recharts';
import { formatCurrency, formatPnL } from '../../utils/formatters';

const INITIAL = 100_000;

export function EquityCurve({ history }) {
  if (!history?.length) return (
    <div className="flex items-center justify-center h-48 text-slate-500 text-sm">
      No equity history available
    </div>
  );

  const max = Math.max(...history.map((d) => d.capital));
  const min = Math.min(...history.map((d) => d.capital));
  const domain = [Math.min(min * 0.998, INITIAL * 0.995), max * 1.002];

  const isProfit = (history[history.length - 1]?.capital ?? INITIAL) >= INITIAL;
  const stroke = isProfit ? '#22c55e' : '#ef4444';
  const gradId = isProfit ? 'eqGreenGrad' : 'eqRedGrad';

  return (
    <div className="h-52">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={history} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={stroke} stopOpacity={0.25} />
              <stop offset="95%" stopColor={stroke} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={domain}
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => '₹' + (v / 1000).toFixed(0) + 'K'}
            width={52}
          />
          <ReferenceLine y={INITIAL} stroke="#334155" strokeDasharray="4 4" />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, fontSize: 11 }}
            formatter={(v, n) => [formatCurrency(v), 'Equity']}
            labelStyle={{ color: '#94a3b8', marginBottom: 2 }}
          />
          <Area
            type="monotone"
            dataKey="capital"
            stroke={stroke}
            strokeWidth={2}
            fill={`url(#${gradId})`}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
