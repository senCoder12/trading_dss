import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  CartesianGrid, ReferenceLine, ResponsiveContainer,
} from 'recharts';
import { LineChart } from 'lucide-react';
import { EmptyState } from '../common/EmptyState';
import { formatCurrency } from '../../utils/formatters';
import { COLORS, CHART_COLORS, TOOLTIP_STYLE } from '../../utils/constants';

export function EquityCurve({ history, initial = 100_000, height = 260 }) {
  if (!history?.length) {
    return (
      <EmptyState
        icon={LineChart}
        title="No equity history yet"
        message="The curve starts plotting once the first trade closes."
        className="py-14"
      />
    );
  }

  const values = history.map((d) => d.capital);
  const domain = [Math.min(...values, initial) * 0.998, Math.max(...values, initial) * 1.002];
  const isProfit = (history[history.length - 1]?.capital ?? initial) >= initial;
  const stroke = isProfit ? COLORS.up : COLORS.down;
  const gradId = isProfit ? 'eqUp' : 'eqDown';

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={history} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={stroke} stopOpacity={0.22} />
              <stop offset="100%" stopColor={stroke} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke={CHART_COLORS.grid} />
          <XAxis dataKey="date" tick={{ fontSize: 11, fill: CHART_COLORS.text }} tickLine={false} axisLine={false} interval="preserveStartEnd" minTickGap={24} />
          <YAxis
            domain={domain}
            tick={{ fontSize: 11, fill: CHART_COLORS.text }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}K`}
            width={52}
          />
          <ReferenceLine y={initial} stroke={COLORS.lineStrong} strokeDasharray="4 4" />
          <Tooltip {...TOOLTIP_STYLE} formatter={(v) => [formatCurrency(v), 'Equity']} />
          <Area type="monotone" dataKey="capital" stroke={stroke} strokeWidth={2} fill={`url(#${gradId})`} dot={false} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
