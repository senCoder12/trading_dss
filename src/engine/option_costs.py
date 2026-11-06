"""
Transaction costs for exchange-traded index/stock options in India.

All statutory charges on options are levied on **premium turnover**, not on
the underlying's notional — the old model charged a flat ₹120/lot and used
index-point distances as "turnover", which is wrong in both directions.

Schedule (per side unless noted), as of 2025-26.  Every rate is a field, so
a circular revision is a config change, not a code change:

* Brokerage         ₹20 per executed order (typical discount broker)
* STT               0.1 % of premium, **sell side only** (since Oct 2024)
* Exchange txn      NSE 0.03503 % / BSE 0.0325 % of premium turnover
* SEBI fee          ₹10 per crore of premium turnover
* Stamp duty        0.003 % of premium, **buy side only**
* GST               18 % on (brokerage + exchange txn + SEBI fee)

STT on an option *exercised* at expiry is charged on intrinsic value at a
higher rate; the system always exits before expiry, so that case is not
modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

_CRORE = 1e7


@dataclass(frozen=True)
class OptionCostSchedule:
    brokerage_per_order: float = 20.0
    stt_sell_pct: float = 0.001
    exchange_txn_pct: float = 0.0003503
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.00003
    gst_pct: float = 0.18


NSE_OPTIONS = OptionCostSchedule()
BSE_OPTIONS = OptionCostSchedule(exchange_txn_pct=0.000325)

_SCHEDULES: dict[str, OptionCostSchedule] = {"NSE": NSE_OPTIONS, "BSE": BSE_OPTIONS}


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst


def schedule_for(exchange: str) -> OptionCostSchedule:
    return _SCHEDULES.get((exchange or "NSE").upper(), NSE_OPTIONS)


def round_trip_cost(
    buy_premium: float,
    sell_premium: float,
    quantity: int,
    exchange: str = "NSE",
    schedule: OptionCostSchedule | None = None,
) -> CostBreakdown:
    """All-in cost of buying *quantity* units at *buy_premium* and selling
    them at *sell_premium* (both per unit, ₹).  Two orders are assumed."""
    s = schedule or schedule_for(exchange)
    qty = max(int(quantity), 0)
    buy_turnover = max(buy_premium, 0.0) * qty
    sell_turnover = max(sell_premium, 0.0) * qty
    turnover = buy_turnover + sell_turnover

    brokerage = 2 * s.brokerage_per_order if qty > 0 else 0.0
    stt = sell_turnover * s.stt_sell_pct
    exchange_fee = turnover * s.exchange_txn_pct
    sebi = turnover * s.sebi_per_crore / _CRORE
    stamp = buy_turnover * s.stamp_buy_pct
    gst = (brokerage + exchange_fee + sebi) * s.gst_pct
    return CostBreakdown(
        brokerage=round(brokerage, 2),
        stt=round(stt, 2),
        exchange=round(exchange_fee, 2),
        sebi=round(sebi, 4),
        stamp=round(stamp, 2),
        gst=round(gst, 2),
    )
