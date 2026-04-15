import {
  ComposedChart, Bar, XAxis, YAxis, Tooltip,
  ReferenceLine, CartesianGrid, ResponsiveContainer, Cell,
} from 'recharts';
import { EmptyState } from '../common/EmptyState';
import { formatLargeNumber, formatPrice } from '../../utils/formatters';
import { BarChart2 } from 'lucide-react';

export function OptionsHeatmap({ data }) {
  if (!data) return <EmptyState icon={BarChart2} title="No options data" />;

  const { top_ce_strikes = [], top_pe_strikes = [], spot, max_pain, pcr, oi_support, oi_resistance } = data;

  if (!top_ce_strikes.length && !top_pe_strikes.length) {
    return <EmptyState icon={BarChart2} title="No OI data available" />;
  }

  // Merge CE and PE strikes into a single set
  const strikeMap = new Map();
  for (const s of top_ce_strikes) {
    strikeMap.set(s.strike, { strike: s.strike, ce_oi: s.oi, pe_oi: 0 });
  }
  for (const s of top_pe_strikes) {
    if (strikeMap.has(s.strike)) {
      strikeMap.get(s.strike).pe_oi = s.oi;
    } else {
      strikeMap.set(s.strike, { strike: s.strike, ce_oi: 0, pe_oi: s.oi });
    }
  }

  const chartData = Array.from(strikeMap.values())
    .sort((a, b) => a.strike - b.strike)
    .map((d) => ({
      ...d,
      ce_neg: -d.ce_oi, // Negative for downward bars
    }));

  const maxOI = Math.max(
    ...chartData.flatMap((d) => [d.pe_oi, d.ce_oi]),
  );

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const d = chartData.find((x) => x.strike === label);
    return (
      <div className="bg-slate-800 border border-slate-600 rounded-lg p-2.5 text-xs shadow-lg">
        <div className="text-slate-200 font-semibold mb-1">Strike: {label}</div>
        <div className="text-red-400">CE OI: {formatLargeNumber(d?.ce_oi)}</div>
        <div className="text-green-400">PE OI: {formatLargeNumber(d?.pe_oi)}</div>
        {d?.ce_oi && d?.pe_oi && (
          <div className="text-slate-400 mt-1">
            P/C: {(d.pe_oi / d.ce_oi).toFixed(2)}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-3">
      {/* Summary stats */}
      <div className="flex flex-wrap gap-4 text-xs">
        {spot && (
          <span className="text-slate-400">Spot: <span className="text-slate-100 font-mono font-semibold">{formatPrice(spot)}</span></span>
        )}
        {max_pain && (
          <span className="text-slate-400">Max Pain: <span className="text-yellow-400 font-mono font-semibold">{formatPrice(max_pain)}</span></span>
        )}
        {pcr != null && (
          <span className="text-slate-400">PCR: <span className={`font-mono font-semibold ${pcr > 1 ? 'text-green-400' : 'text-red-400'}`}>{pcr.toFixed(2)}</span></span>
        )}
        {oi_support && (
          <span className="text-slate-400">Support: <span className="text-green-400 font-mono">{formatPrice(oi_support)}</span></span>
        )}
        {oi_resistance && (
          <span className="text-slate-400">Resistance: <span className="text-red-400 font-mono">{formatPrice(oi_resistance)}</span></span>
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 bg-red-500/70 rounded-sm" />
          <span className="text-slate-400">CE OI (resistance)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 bg-green-500/70 rounded-sm" />
          <span className="text-slate-400">PE OI (support)</span>
        </div>
      </div>

      {/* Mirror chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="strike"
              tick={{ fontSize: 9, fill: '#94a3b8' }}
              tickLine={false}
              axisLine={false}
              interval={0}
            />
            <YAxis
              tick={{ fontSize: 9, fill: '#94a3b8' }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => formatLargeNumber(Math.abs(v))}
              domain={[-maxOI * 1.1, maxOI * 1.1]}
              width={50}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={0} stroke="#334155" strokeWidth={1} />
            {spot && (
              <ReferenceLine
                x={spot}
                stroke="#3b82f6"
                strokeDasharray="4 4"
                label={{ value: 'Spot', position: 'top', fontSize: 10, fill: '#3b82f6' }}
              />
            )}
            {max_pain && (
              <ReferenceLine
                x={max_pain}
                stroke="#eab308"
                strokeDasharray="4 4"
                label={{ value: 'Max Pain', position: 'insideTop', fontSize: 10, fill: '#eab308' }}
              />
            )}
            {/* PE OI — upward (positive) */}
            <Bar dataKey="pe_oi" name="PE OI" fill="#22c55e" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
            {/* CE OI — downward (negative) */}
            <Bar dataKey="ce_neg" name="CE OI" fill="#ef4444" fillOpacity={0.7} radius={[0, 0, 2, 2]} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
