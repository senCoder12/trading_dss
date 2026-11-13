"""Intraday refresh of FOUniverseManager (all collaborators mocked)."""

import threading
from datetime import datetime
from unittest.mock import Mock

from src.analysis.fo_universe_manager import FOUniverseManager
from src.data.fo_bulk_fetcher import StockQuote


def _quote(symbol: str) -> StockQuote:
    return StockQuote(
        symbol=symbol, ltp=100.0, open=99.0, high=101.0, low=98.0, close=99.0,
        volume=50_000, avg_volume_20d=0.0, price_change_pct=1.0,
        bid=99.9, ask=100.1, bid_ask_spread=0.2, bid_ask_spread_pct=0.2,
        timestamp=datetime.now(),
    )


def _manager(top_symbols: list[str]) -> FOUniverseManager:
    mgr = FOUniverseManager.__new__(FOUniverseManager)
    mgr._snapshot_lock = threading.Lock()
    mgr.market_hours = Mock()
    mgr.market_hours.is_market_open.return_value = True
    mgr._current_snapshot = Mock(
        active_universe=[Mock(symbol=s) for s in top_symbols],
        priority_watchlist=[],
        refresh_count=0,
    )
    mgr.fo_bulk_fetcher = Mock()
    mgr.scorer = Mock()
    mgr.scorer.score_universe.return_value = []
    return mgr


def test_intraday_refresh_quotes_only_top_30_and_injects_avg_volume():
    symbols = [f"S{i}" for i in range(40)]
    mgr = _manager(symbols)
    top_30 = symbols[:30]
    mgr.fo_bulk_fetcher.fetch_all_quotes.return_value = {s: _quote(s) for s in top_30}
    mgr.fo_bulk_fetcher.fetch_avg_volumes.return_value = {s: 100_000.0 for s in top_30}
    mgr.fo_bulk_fetcher.fetch_oi_for_symbols.return_value = {}

    mgr.run_intraday_refresh()

    mgr.fo_bulk_fetcher.fetch_all_quotes.assert_called_once_with(top_30)
    scored_quotes = mgr.scorer.score_universe.call_args.kwargs["quotes"]
    assert set(scored_quotes) == set(top_30)
    # Without the 20-day average every volume ratio would be 0.
    assert all(q.avg_volume_20d == 100_000.0 for q in scored_quotes.values())
