import {
  ComposedChart, Bar, XAxis, YAxis, Tooltip,
  ReferenceLine, CartesianGrid, ResponsiveContainer,
} from 'recharts';
import { Layers } from 'lucide-react';
import { EmptyState } from '../common/EmptyState';
import { formatLargeNumber, formatPrice } from '../../utils/formatters';
import { COLORS, CHART_COLORS } from '../../utils/constants';

function niceStep(x) {
  const p = 10 ** Math.floor(Math.log10(x || 1));
  const m = x / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}

function Stat({ label, value, cls = 'text-ink' }) {
  return (
    <div className="min-w-0">
      <div className="label">{label}</div>
      <div className={`text-[15px] font-semibold num mt-1 truncate ${cls}`}>{value ?? '—'}</div>
    </div>
  );
}

/** Top CE/PE open-interest strikes as a mirrored bar chart + key levels. */
export function OptionsHeatmap({ data }) {
  if (!data) return <EmptyState icon={Layers} title="No options data" message="Open interest is only available for F&O indices." />;

  const { top_ce_strikes = [], top_pe_strikes = [], spot, max_pain, pcr, oi_support, oi_resistance, expiry, iv_atm } = data;
  if (!top_ce_strikes.length && !top_pe_strikes.length) {
    return <EmptyState icon={Layers} title="No open interest yet" message="The latest chain snapshot has no OI data." />;
  }

  const strikeMap = new Map();
  for (const s of top_ce_strikes) strikeMap.set(s.strike, { strike: s.strike, ce_oi: s.oi, pe_oi: 0 });
  for (const s of top_pe_strikes) {
    if (strikeMap.has(s.strike)) strikeMap.get(s.strike).pe_oi = s.oi;
    else strikeMap.set(s.strike, { strike: s.strike, ce_oi: 0, pe_oi: s.oi });
  }
  const chartData = [...strikeMap.values()].sort((a, b) => a.strike - b.strike).map((d) => ({ ...d, ce_neg: -d.ce_oi }));
  const maxOI = Math.max(...chartData.flatMap((d) => [d.pe_oi, d.ce_oi]), 1);
  // Symmetric, readable ticks around zero (puts above, calls below)
  const step = niceStep(maxOI / 2);
  const top = step * Math.ceil(maxOI / step);
  const ticks = [-top, -top / 2, 0, top / 2, top];

  const CustomTooltip = ({ active, label }) => {
    if (!active) return null;
    const d = chartData.find((x) => x.strike === label);
    return (
      <div className="card shadow-pop px-3 py-2.5 text-xs space-y-1">
        <div className="text-ink font-semibold num">Strike {formatPrice(label)}</div>
        <div className="text-down num">Call OI {formatLargeNumber(d?.ce_oi)}</div>
        <div className="text-up num">Put OI {formatLargeNumber(d?.pe_oi)}</div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-5">
        <Stat label="Expiry" value={expiry} />
        <Stat label="PCR" value={pcr?.toFixed(2)} cls={pcr > 1 ? 'text-up' : pcr < 0.8 ? 'text-down' : 'text-ink'} />
        <Stat label="Max pain" value={max_pain ? formatPrice(max_pain) : null} cls="text-warn" />
        <Stat label="OI support" value={oi_support ? formatPrice(oi_support) : null} cls="text-up" />
        <Stat label="OI resistance" value={oi_resistance ? formatPrice(oi_resistance) : null} cls="text-down" />
        <Stat label="ATM IV" value={iv_atm != null ? `${iv_atm.toFixed(2)}%` : null} cls="text-violet" />
      </div>

      <div>
        <div className="flex items-center gap-4 text-xs text-ink-faint mb-3">
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-up/80" /> Put OI (support)</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-down/80" /> Call OI (resistance)</span>
          <span className="flex items-center gap-1.5"><span className="w-3 border-t border-dashed border-brand" /> Spot</span>
          <span className="flex items-center gap-1.5"><span className="w-3 border-t border-dashed border-warn" /> Max pain</span>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 16, right: 8, bottom: 0, left: 0 }} barCategoryGap="25%">
              <CartesianGrid vertical={false} stroke={CHART_COLORS.grid} />
              <XAxis dataKey="strike" tick={{ fontSize: 11, fill: CHART_COLORS.text }} tickLine={false} axisLine={false} interval={0} />
              <YAxis
                tick={{ fontSize: 11, fill: CHART_COLORS.text }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => formatLargeNumber(Math.abs(v))}
                domain={[-top, top]}
                ticks={ticks}
                width={52}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgb(255 255 255 / 0.03)' }} />
              <ReferenceLine y={0} stroke={COLORS.lineStrong} />
              {spot && <ReferenceLine x={spot} stroke={COLORS.brand} strokeDasharray="4 4" />}
              {max_pain && <ReferenceLine x={max_pain} stroke={COLORS.warn} strokeDasharray="4 4" />}
              <Bar dataKey="pe_oi" name="Put OI" fill={COLORS.up} fillOpacity={0.8} radius={[4, 4, 0, 0]} isAnimationActive={false} />
              <Bar dataKey="ce_neg" name="Call OI" fill={COLORS.down} fillOpacity={0.8} radius={[0, 0, 4, 4]} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="text-2xs text-ink-ghost mt-2">Top 5 strikes by open interest on each side. Spot {spot ? formatPrice(spot) : '—'}.</p>
      </div>
    </div>
  );
}
