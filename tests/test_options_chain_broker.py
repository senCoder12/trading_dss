"""Offline tests for the Angel One broker path of OptionsChainFetcher."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.analysis.gex.black_scholes import BlackScholesEngine, BSMInputs, get_dte, get_risk_free_rate
from src.data.options_chain import OptionsChainFetcher, _implied_vol_pct, _quote_time

_IST = ZoneInfo("Asia/Kolkata")


class _FakeScraper:
    """Never touches the network."""


class _FakeBroker:
    def __init__(self, master: pd.DataFrame, quotes: dict[str, dict], spot: float) -> None:
        self.master = master
        self.quotes = quotes
        self.spot = spot
        self.batches: list[list[str]] = []

    def get_master_contract(self, exchange: str):
        return self.master[self.master["exch_seg"] == exchange].copy()

    def get_market_data(self, mode: str, exchange_tokens: dict):
        if mode == "LTP":
            return {"fetched": [{"ltp": self.spot}]}
        tokens = next(iter(exchange_tokens.values()))
        self.batches.append(tokens)
        return {"fetched": [self.quotes[t] for t in tokens if t in self.quotes]}


class _FakeDB:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.queries: list[tuple] = []

    def fetch_all(self, query: str, params: tuple):
        self.queries.append(params)
        return self.rows


def _bs_price(spot, strike, expiry, as_of, iv, opt_type):
    inputs = BSMInputs(spot=spot, strike=strike, dte_days=get_dte(expiry, as_of) * 365.0,
                       iv=iv, risk_free_rate=get_risk_free_rate(), option_type=opt_type)
    return BlackScholesEngine().calculate_price(inputs)


@pytest.fixture()
def nifty_setup():
    as_of = datetime.now(tz=_IST).replace(hour=15, minute=30, second=0, microsecond=0)
    expiry = (as_of + timedelta(days=7)).date()
    exp_code = expiry.strftime("%d%b%Y").upper()
    spot = 24500.0
    rows, quotes = [], {}
    token = 1000
    for strike in range(23000, 26050, 50):  # 61 strikes → 122 tokens, forces batching
        for opt in ("CE", "PE"):
            token += 1
            t = str(token)
            rows.append({
                "token": t, "symbol": f"NIFTY{exp_code}{strike}{opt}", "name": "NIFTY",
                "expiry": exp_code, "strike": f"{strike * 100}.000000", "lotsize": "65",
                "instrumenttype": "OPTIDX", "exch_seg": "NFO",
            })
            quotes[t] = {
                "symbolToken": t,
                "ltp": round(_bs_price(spot, strike, expiry, as_of, 0.15, opt), 2),
                "opnInterest": 65 * 1000,          # 1000 contracts
                "tradeVolume": 65 * 50,            # 50 contracts
                "netChange": 1.0, "percentChange": 2.0,
                "totBuyQuan": 10, "totSellQuan": 20,
                "exchFeedTime": as_of.strftime("%d-%b-%Y %H:%M:%S"),
                "depth": {"buy": [{"price": 9.5, "quantity": 65}], "sell": [{"price": 10.5, "quantity": 130}]},
            }
    master = pd.DataFrame(rows)
    return master, quotes, spot, expiry, as_of


def test_quote_time_picks_latest_valid():
    t = _quote_time([{"exchFeedTime": "05-Nov-2026 15:39:59"}, {"exchFeedTime": "05-Nov-2026 15:40:00"}, {}])
    assert t == datetime(2026, 11, 5, 15, 40, tzinfo=_IST)


def test_implied_vol_round_trips_black_scholes():
    as_of = datetime(2026, 11, 5, 15, 30, tzinfo=_IST)
    expiry = (as_of + timedelta(days=4)).date()
    price = _bs_price(24500, 24550, expiry, as_of, 0.18, "CE")
    assert _implied_vol_pct(price, 24500, 24550, "CE", expiry, as_of) == pytest.approx(18.0, abs=0.05)


def test_implied_vol_zero_when_unsolvable():
    as_of = datetime(2026, 11, 5, 15, 30, tzinfo=_IST)
    expiry = (as_of + timedelta(days=4)).date()
    assert _implied_vol_pct(0.0, 24500, 24550, "CE", expiry, as_of) == 0.0
    # Below intrinsic (deep ITM call quoted under spot - strike)
    assert _implied_vol_pct(400.0, 24500, 24000, "CE", expiry, as_of) == 0.0


def test_broker_chain_units_iv_and_batching(nifty_setup):
    master, quotes, spot, expiry, _ = nifty_setup
    broker = _FakeBroker(master, quotes, spot)
    f = OptionsChainFetcher(scraper=_FakeScraper(), broker_client=broker)

    raw = f._fetch_from_broker("NIFTY")

    assert raw is not None
    data = raw["records"]["data"]
    assert len(data) == 50                          # _BROKER_MAX_STRIKES nearest to spot
    assert all(len(b) <= 50 for b in broker.batches) and len(broker.batches) == 2
    assert raw["records"]["expiryDates"] == [expiry.strftime("%d-%b-%Y")]
    atm = next(d for d in data if d["strikePrice"] == 24500.0)
    ce = atm["CE"]
    assert ce["openInterest"] == 1000               # units / lot size
    assert ce["totalTradedVolume"] == 50
    assert ce["impliedVolatility"] == pytest.approx(15.0, abs=0.1)
    assert ce["bidprice"] == 9.5 and ce["askQty"] == 130
    assert ce["changeinOpenInterest"] == 0          # no db → no baseline


def test_broker_chain_oi_change_from_prior_session(nifty_setup):
    master, quotes, spot, expiry, as_of = nifty_setup
    db = _FakeDB([{"strike_price": 24500.0, "option_type": "CE", "open_interest": 800}])
    f = OptionsChainFetcher(scraper=_FakeScraper(), broker_client=_FakeBroker(master, quotes, spot), db=db)

    raw = f._fetch_from_broker("NIFTY")

    atm = next(d for d in raw["records"]["data"] if d["strikePrice"] == 24500.0)
    assert atm["CE"]["changeinOpenInterest"] == 200
    assert atm["CE"]["pchangeinOpenInterest"] == 25.0
    assert atm["PE"]["changeinOpenInterest"] == 0   # no baseline for this leg
    # Baseline must come from before the quotes' session opened
    assert db.queries[0][-1] == f"{as_of.date().isoformat()}T09:15"
