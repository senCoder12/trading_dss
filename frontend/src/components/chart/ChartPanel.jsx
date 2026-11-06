import { useState } from 'react';
import { useDataStore } from '../../hooks/useDataStore';
import { useIndexMeta, indexLabel } from '../../hooks/useIndexMeta';
import { ErrorBoundary } from '../common/ErrorBoundary';
import { FlashPrice } from '../common/FlashPrice';
import { Delta } from '../common/Delta';
import { IndexSelect } from '../common/IndexSelect';
import { formatPrice } from '../../utils/formatters';
import PriceChart from './PriceChart';
import { TimeframeControl, IndicatorToggles } from './ChartControls';

/**
 * Card wrapping the price chart: index name + live price in the header,
 * overlays + timeframe in a single toolbar row.
 * Pass `onIndexChange` to show an index picker in the header.
 */
export default function ChartPanel({ indexId, onIndexChange, title, height = 400, className = '' }) {
  const [timeframe, setTimeframe] = useState('1d');
  const [indicators, setIndicators] = useState(['ema20', 'ema50']);
  const meta = useIndexMeta();
  const { prices: { data } } = useDataStore();
  const quote = data?.indices?.find((i) => i.index_id === indexId);

  return (
    <section className={`card flex flex-col min-w-0 ${className}`}>
      <header className="card-header flex-wrap">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 min-w-0">
          {onIndexChange ? (
            <IndexSelect value={indexId} onChange={onIndexChange} className="w-full sm:w-56" />
          ) : (
            <h3 className="card-title truncate">{title ?? indexLabel(meta, indexId)}</h3>
          )}
          {/* When a title is given the page already shows the quote — don't repeat it */}
          {!title && quote?.ltp != null && (
            <div className="flex items-baseline gap-2.5">
              <span className="text-lg font-semibold num text-ink">
                <FlashPrice value={quote.ltp} formatter={formatPrice} />
              </span>
              <Delta value={quote.change_pct} flash className="text-[13px] font-medium" />
            </div>
          )}
        </div>
        <TimeframeControl timeframe={timeframe} onTimeframeChange={setTimeframe} />
      </header>

      <div className="px-5 pt-3">
        <IndicatorToggles indicators={indicators} onIndicatorsChange={setIndicators} />
      </div>

      <div className="px-4 pb-4 pt-1">
        <ErrorBoundary>
          <PriceChart
            indexId={indexId}
            height={height}
            timeframe={timeframe}
            indicators={indicators}
            showVolume
            showSignalMarkers
          />
        </ErrorBoundary>
      </div>
    </section>
  );
}
