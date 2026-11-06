"""
Paper-trading adapter — the only integration surface between the live
decision loop and the premium-native paper ledger.

Before this module nothing ever called ``PaperTradingEngine.on_signal`` or
``update_positions``: ``run_system.py`` built the engine and never fed it,
so paper trading had never run.  The adapter needs two hooks, both on jobs
that already exist in ``DataCollector``:

* :meth:`PaperTradingAdapter.on_decisions` — after every decision cycle.
  Only **actionable** results (a risk-approved option ``RefinedSignal``)
  become paper orders; nothing is re-sized, the approved lots are used.
* :meth:`PaperTradingAdapter.on_monitor_tick` — every 60 s monitor tick.
  Fills pending orders from the option quotes, marks open positions
  (quote, else flagged model mark) and applies the live exit rules.

Neither method ever raises into the scheduler: failures are logged.

Wiring (three one-liners; nothing else changes)
-----------------------------------------------
``scripts/run_system.py`` — in the "Paper Trading Engine" block, right after
``components["paper_engine"] = paper_engine``::

    if "collector" in components:
        from src.paper_trading.adapter import PaperTradingAdapter
        components["collector"].paper_adapter = PaperTradingAdapter(paper_engine)

``src/data/data_collector.py`` — ``DataCollector._run_decision_cycle``, right
after ``results = self.decision_engine.run_all_indices()``::

    if getattr(self, "paper_adapter", None) is not None: self.paper_adapter.on_decisions(results)

``src/data/data_collector.py`` — ``DataCollector._monitor_positions``, as the
last statement of the method (after the ``try/except`` around
``monitor_open_positions()`` so a live-monitor error does not skip paper
marks)::

    if getattr(self, "paper_adapter", None) is not None: self.paper_adapter.on_monitor_tick()

The adapter reads quotes from ``options_chain_snapshot`` and spot from
``price_data`` through the engine's own ``DatabaseManager``; no new tables.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, Optional

from src.paper_trading.paper_engine import PaperOrder, PaperTrade, PaperTradingEngine

logger = logging.getLogger(__name__)


class PaperTradingAdapter:
    """Feeds ``DecisionResult`` objects and monitor ticks into a
    :class:`PaperTradingEngine`.

    Parameters
    ----------
    engine:
        The paper engine (``components["paper_engine"]`` in ``run_system``).
    """

    def __init__(self, engine: PaperTradingEngine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------
    # Decision cycle
    # ------------------------------------------------------------------

    def on_decision(self, result) -> Optional[PaperOrder]:
        """Turn one ``DecisionResult`` into a paper order.

        Returns the order (``PENDING`` until a quote arrives, ``FILLED``, or
        ``NO_FILL``), or ``None`` when the result is not actionable or the
        paper engine declined it (filter, cap, duplicate).
        """
        if result is None or not getattr(result, "is_actionable", False):
            return None
        signal = getattr(result, "signal", None)
        if signal is None or getattr(signal, "signal_type", None) not in ("BUY_CALL", "BUY_PUT"):
            return None
        if not getattr(signal, "option_type", None) or getattr(signal, "is_valid", True) is False:
            logger.warning(
                "PaperTradingAdapter: actionable %s %s has no approved option contract — skipped",
                getattr(result, "index_id", "?"), getattr(signal, "signal_type", "?"),
            )
            return None
        decision_time = getattr(result, "timestamp", None) or getattr(signal, "generated_at", None)
        try:
            return self.engine.submit_order(signal, decision_time=decision_time)
        except Exception:
            logger.exception(
                "PaperTradingAdapter.on_decision failed for %s", getattr(result, "index_id", "?"),
            )
            return None

    def on_decisions(self, results: Optional[Iterable]) -> list[PaperOrder]:
        """:meth:`on_decision` for every result of ``run_all_indices()``."""
        orders: list[PaperOrder] = []
        for result in results or ():
            order = self.on_decision(result)
            if order is not None:
                orders.append(order)
        return orders

    # ------------------------------------------------------------------
    # Monitor tick
    # ------------------------------------------------------------------

    def on_monitor_tick(self, now: Optional[datetime] = None) -> list[PaperTrade]:
        """Fill pending orders, mark and exit open paper positions.

        Returns the paper trades closed on this tick.
        """
        try:
            return self.engine.on_tick(now=now)
        except Exception:
            logger.exception("PaperTradingAdapter.on_monitor_tick failed")
            return []
