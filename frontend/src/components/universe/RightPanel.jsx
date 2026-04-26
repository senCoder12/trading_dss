import { useMemo, useState, useCallback } from 'react';
import { Spinner } from '../common/Spinner';

function ScoreCell({ label, value }) {
  let color = 'text-red-400';
  if (value >= 70) color = 'text-green-400';
  else if (value >= 50) color = 'text-amber-400';

  return (
    <div className="text-center">
      <div className="text-xs text-slate-500 font-medium">{label}</div>
      <div className={`text-lg font-bold ${color}`}>{value.toFixed(0)}</div>
    </div>
  );
}

function OptionsChainTable({ chainData, maxVisibleRows = 11 }) {
  const [showMoreAbove, setShowMoreAbove] = useState(0);
  const [showMoreBelow, setShowMoreBelow] = useState(0);

  if (!chainData?.strikes || chainData.strikes.length === 0) {
    return <div className="p-4 text-slate-400">No options chain data available</div>;
  }

  // Find ATM strike
  const atmStrike = chainData.atm_strike || chainData.spot_price;

  // Compute OI bar widths
  const allOI = chainData.strikes.reduce((acc, s) => {
    acc.ce.push(s.ce_oi);
    acc.pe.push(s.pe_oi);
    return acc;
  }, { ce: [], pe: [] });

  const maxCEOI = Math.max(...allOI.ce, 1);
  const maxPEOI = Math.max(...allOI.pe, 1);

  // Find ATM index and visible range
  const atmIndex = chainData.strikes.findIndex(s => Math.abs(s.strike - atmStrike) < 0.01);
  const defaultRows = 11;
  const aboveATM = Math.floor(defaultRows / 2);
  const belowATM = Math.floor(defaultRows / 2);

  const startIdx = Math.max(0, atmIndex - aboveATM - showMoreAbove);
  const endIdx = Math.min(chainData.strikes.length, atmIndex + belowATM + 1 + showMoreBelow);

  const visibleStrikes = chainData.strikes.slice(startIdx, endIdx);

  return (
    <div className="space-y-3">
      {/* OI Summary */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Total CE OI', value: chainData.total_ce_oi.toLocaleString() },
          { label: 'Total PE OI', value: chainData.total_pe_oi.toLocaleString() },
          {
            label: 'PCR',
            value: `${chainData.pcr.toFixed(2)} ${chainData.pcr > 1 ? '(bullish)' : '(bearish)'}`,
          },
          { label: 'Max Pain', value: chainData.max_pain_strike.toFixed(0) },
        ].map(card => (
          <div key={card.label} className="bg-slate-800 rounded border border-slate-700 p-2">
            <div className="text-xs text-slate-500 font-medium">{card.label}</div>
            <div className="text-sm font-bold text-slate-100 mt-1">{card.value}</div>
          </div>
        ))}
      </div>

      {/* Options Chain Table */}
      <div className="border border-slate-700 rounded overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-800 border-b border-slate-700">
                <th className="px-2 py-1.5 text-right text-slate-400 font-medium">CE OI</th>
                <th className="px-2 py-1.5 text-right text-slate-400 font-medium">CE Chg</th>
                <th className="px-2 py-1.5 text-right text-slate-400 font-medium">CE Vol</th>
                <th className="px-2 py-1.5 text-right text-slate-400 font-medium">CE IV%</th>
                <th className="px-2 py-1.5 text-right text-slate-400 font-medium">CE LTP</th>
                <th className="px-2 py-1.5 text-center text-slate-400 font-medium">Strike</th>
                <th className="px-2 py-1.5 text-left text-slate-400 font-medium">PE LTP</th>
                <th className="px-2 py-1.5 text-left text-slate-400 font-medium">PE IV%</th>
                <th className="px-2 py-1.5 text-left text-slate-400 font-medium">PE Vol</th>
                <th className="px-2 py-1.5 text-left text-slate-400 font-medium">PE Chg</th>
                <th className="px-2 py-1.5 text-left text-slate-400 font-medium">PE OI</th>
              </tr>
            </thead>
            <tbody>
              {visibleStrikes.map((strike, idx) => {
                const isATM = Math.abs(strike.strike - atmStrike) < 0.01;
                const isMaxPain = Math.abs(strike.strike - chainData.max_pain_strike) < 0.01;

                const bgClass = isATM
                  ? 'bg-green-900/20'
                  : isMaxPain
                  ? 'bg-green-900/10'
                  : 'hover:bg-slate-800';

                const ceBgWidth = (strike.ce_oi / maxCEOI) * 100;
                const peBgWidth = (strike.pe_oi / maxPEOI) * 100;

                return (
                  <tr key={idx} className={`border-b border-slate-700 ${bgClass}`}>
                    <td className="px-2 py-1.5 text-right text-green-400 font-mono relative">
                      <div
                        className="absolute inset-y-0 left-0 bg-green-500/20"
                        style={{ width: `${ceBgWidth}%` }}
                      />
                      <span className="relative">{strike.ce_oi.toLocaleString()}</span>
                    </td>
                    <td className="px-2 py-1.5 text-right text-green-400 font-mono">
                      {strike.ce_oi_change > 0 ? '+' : ''}
                      {strike.ce_oi_change.toLocaleString()}
                    </td>
                    <td className="px-2 py-1.5 text-right text-green-400 font-mono">
                      {strike.ce_volume.toLocaleString()}
                    </td>
                    <td className="px-2 py-1.5 text-right text-green-400 font-mono">
                      {strike.ce_iv.toFixed(1)}
                    </td>
                    <td className="px-2 py-1.5 text-right text-green-400 font-mono">
                      {strike.ce_ltp.toFixed(2)}
                    </td>

                    <td className="px-2 py-1.5 text-center text-slate-100 font-bold bg-slate-800">
                      {strike.strike.toFixed(0)}
                      {isATM && <span className="text-slate-400 ml-1">◀</span>}
                    </td>

                    <td className="px-2 py-1.5 text-left text-red-400 font-mono">
                      {strike.pe_ltp.toFixed(2)}
                    </td>
                    <td className="px-2 py-1.5 text-left text-red-400 font-mono">
                      {strike.pe_iv.toFixed(1)}
                    </td>
                    <td className="px-2 py-1.5 text-left text-red-400 font-mono">
                      {strike.pe_volume.toLocaleString()}
                    </td>
                    <td className="px-2 py-1.5 text-left text-red-400 font-mono">
                      {strike.pe_oi_change > 0 ? '+' : ''}
                      {strike.pe_oi_change.toLocaleString()}
                    </td>
                    <td className="px-2 py-1.5 text-left text-red-400 font-mono relative">
                      <div
                        className="absolute inset-y-0 right-0 bg-red-500/20"
                        style={{ width: `${peBgWidth}%` }}
                      />
                      <span className="relative">{strike.pe_oi.toLocaleString()}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Show more button */}
        {endIdx < chainData.strikes.length || startIdx > 0 ? (
          <div className="flex justify-center gap-3 p-2 bg-slate-800 border-t border-slate-700">
            {startIdx > 0 && (
              <button
                onClick={() => setShowMoreAbove(prev => prev + 5)}
                className="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded text-slate-300"
              >
                Show more above
              </button>
            )}
            {endIdx < chainData.strikes.length && (
              <button
                onClick={() => setShowMoreBelow(prev => prev + 5)}
                className="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded text-slate-300"
              >
                Show more below
              </button>
            )}
          </div>
        ) : null}
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-xs text-slate-400 pt-2">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-green-500/30" />
          <span>CE OI</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-red-500/30" />
          <span>PE OI</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="font-bold">◀</span>
          <span>ATM</span>
        </div>
      </div>
    </div>
  );
}

export default function RightPanel({
  symbol,
  universeData,
  chainData,
  chainLoading,
  selectedExpiry,
  onExpiryChange,
}) {
  // Get stock details from universe data
  const stockDetail = useMemo(() => {
    if (!universeData?.active_universe) return null;
    return universeData.active_universe.find(s => s.symbol === symbol);
  }, [universeData?.active_universe, symbol]);

  if (!stockDetail) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner />
      </div>
    );
  }

  const rankColor = stockDetail.tier === 'PRIORITY'
    ? 'bg-green-500/20 text-green-400 border-green-500/30'
    : stockDetail.tier === 'ACTIVE'
    ? 'bg-blue-500/20 text-blue-400 border-blue-500/30'
    : 'bg-slate-700 text-slate-400 border-slate-600';

  const scoreColor = stockDetail.composite_score >= 70
    ? 'bg-green-500/20 text-green-400 border-green-500/30'
    : stockDetail.composite_score >= 50
    ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
    : 'bg-red-500/20 text-red-400 border-red-500/30';

  return (
    <div className="flex-1 flex flex-col bg-slate-900 overflow-hidden">
      {/* Header — fixed */}
      <div className="border-b border-slate-700 p-4 bg-slate-800">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-2xl font-bold text-slate-100">{symbol}</h2>
            <div className="text-sm text-slate-500">{stockDetail.company_name}</div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-slate-100">
              ₹{stockDetail.ltp.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </div>
            <div
              className={`text-sm font-semibold ${
                stockDetail.price_change_pct >= 0 ? 'text-green-400' : 'text-red-400'
              }`}
            >
              {stockDetail.price_change_pct >= 0 ? '+' : ''}
              {stockDetail.price_change_pct.toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Badges */}
        <div className="flex gap-2">
          <div
            className={`px-2 py-1 rounded text-xs font-semibold border ${rankColor}`}
          >
            Rank #{stockDetail.rank}
          </div>
          <div
            className={`px-2 py-1 rounded text-xs font-semibold border ${scoreColor}`}
          >
            Score {stockDetail.composite_score.toFixed(1)}
          </div>
        </div>
      </div>

      {/* Score breakdown — fixed */}
      <div className="border-b border-slate-700 p-4 grid grid-cols-6 gap-3">
        <ScoreCell label="OI" value={stockDetail.oi_score} />
        <ScoreCell label="OI Change" value={stockDetail.oi_change_score} />
        <ScoreCell label="Volume" value={stockDetail.volume_score} />
        <ScoreCell label="Price Move" value={stockDetail.price_move_score} />
        <ScoreCell label="IV Change" value={stockDetail.iv_change_score} />
        <ScoreCell label="Spread" value={stockDetail.spread_score} />
      </div>

      {/* Options chain — scrollable */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* Chain header and expiry pills */}
        <div className="mb-4">
          <h3 className="font-semibold text-slate-100 mb-2">Options Chain</h3>
          <div className="flex items-center gap-2">
            <div className="flex gap-2">
              {chainData?.available_expiries?.map(expiry => (
                <button
                  key={expiry}
                  onClick={() => onExpiryChange(expiry)}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                    selectedExpiry === expiry
                      ? 'bg-blue-500 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {new Date(expiry).toLocaleDateString('en-IN', {
                    month: 'short',
                    day: 'numeric',
                  })}
                </button>
              ))}
            </div>
            <div className="ml-auto text-xs text-slate-400">
              Spot: ₹{chainData?.spot_price?.toFixed(2)} | Lot: {chainData?.lot_size}
            </div>
          </div>
        </div>

        {/* Chain table */}
        {chainLoading ? (
          <div className="flex justify-center p-8">
            <Spinner />
          </div>
        ) : chainData ? (
          <OptionsChainTable chainData={chainData} />
        ) : (
          <div className="text-slate-400 text-sm">No options chain data available</div>
        )}
      </div>
    </div>
  );
}
