"""
Tests for Angel One SnapQuote depth parsing (WebSocketFeed) and near-month
index futures subscription / expiry selection (DataRouter).
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import threading

import src.data.data_router as dr
from src.data.broker.websocket_feed import LiveTick, WebSocketFeed
from src.data.data_router import DataRouter, _best_level_ofi, _nearest_expiry


# ── Helpers ───────────────────────────────────────────────────────────────────


def _level(price_rupees: float, qty: int, orders: int = 1) -> dict:
    """One SmartWebSocketV2 depth packet (price in paise)."""
    return {"flag": 0, "quantity": qty, "price": int(round(price_rupees * 100)), "no of orders": orders}


def _snapquote(**overrides) -> dict:
    msg = {
        "token": "35003",
        "exchange_type": 2,
        "last_traded_price": 2450050,
        "volume_trade_for_the_day": 120000,
        "open_interest": 1500000,
        "open_interest_change_percentage": 1.5,
        "open_price_of_the_day": 2440000,
        "high_price_of_the_day": 2460000,
        "low_price_of_the_day": 2435000,
        "closed_price": 2438000,
        "total_buy_quantity": 600000.0,
        "total_sell_quantity": 400000.0,
        "best_5_buy_data": [
            _level(24499.0, 300),
            _level(24500.0, 150),   # best bid, deliberately out of order
            _level(24498.0, 75),
            _level(0.0, 0),         # empty level — must be dropped
            _level(24497.0, 75),
        ],
        "best_5_sell_data": [
            _level(24501.0, 75),    # best ask
            _level(24502.0, 150),
            _level(24503.0, 75),
            _level(24504.0, 0),     # zero qty — must be dropped
            _level(24505.0, 75),
        ],
    }
    msg.update(overrides)
    return msg


@pytest.fixture()
def feed() -> WebSocketFeed:
    return WebSocketFeed(client=MagicMock())


# ── WebSocketFeed depth parsing ───────────────────────────────────────────────


def test_snapquote_depth_parsed(feed: WebSocketFeed) -> None:
    depth_ticks: list[LiveTick] = []
    feed.on_depth_update = depth_ticks.append

    feed._on_data(None, _snapquote())

    tick = feed.get_latest("35003")
    assert tick is not None
    assert tick.bid_price == pytest.approx(24500.0)
    assert tick.ask_price == pytest.approx(24501.0)
    assert tick.bid_qty == 150
    assert tick.ask_qty == 75
    assert tick.bid_qty_5 == 300 + 150 + 75 + 75
    assert tick.ask_qty_5 == 75 + 150 + 75 + 75
    assert tick.total_buy_qty == pytest.approx(600000.0)
    assert tick.total_sell_qty == pytest.approx(400000.0)
    assert tick.spread == pytest.approx(1.0)
    assert tick.depth_imbalance == pytest.approx((600 - 375) / 975)
    assert tick.book_imbalance == pytest.approx(0.2)
    assert depth_ticks == [tick]
    assert feed.crossed_depth_ticks == 0


def test_index_tick_without_depth(feed: WebSocketFeed) -> None:
    depth_ticks: list[LiveTick] = []
    feed.on_depth_update = depth_ticks.append

    feed._on_data(None, _snapquote(
        token="26000", exchange_type=1, open_interest=0,
        best_5_buy_data=[], best_5_sell_data=[],
        total_buy_quantity=0.0, total_sell_quantity=0.0,
    ))

    tick = feed.get_latest("26000")
    assert tick.bid_price is None and tick.ask_price is None
    assert not tick.has_depth
    assert tick.depth_imbalance is None
    assert tick.book_imbalance is None
    assert tick.spread is None
    assert depth_ticks == []


def test_crossed_book_counted_not_mutated(feed: WebSocketFeed) -> None:
    feed._on_data(None, _snapquote(
        best_5_buy_data=[_level(24510.0, 75)],
        best_5_sell_data=[_level(24500.0, 75)],
    ))

    tick = feed.get_latest("35003")
    assert tick.bid_price == pytest.approx(24510.0)
    assert tick.ask_price == pytest.approx(24500.0)
    assert feed.depth_ticks == 1
    assert feed.crossed_depth_ticks == 1


# ── Expiry selection ──────────────────────────────────────────────────────────


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D401
        return datetime(2026, 11, 6, 10, 0, tzinfo=tz)


@pytest.fixture()
def fixed_today(monkeypatch):
    monkeypatch.setattr(dr, "datetime", _FixedDatetime)


def test_nearest_expiry_is_chronological(fixed_today) -> None:
    # Text sort would pick "01DEC2026"; "03NOV2026" is already expired.
    df = pd.DataFrame({"expiry": ["01DEC2026", "24NOV2026", "03NOV2026", "10NOV2026"]})
    assert _nearest_expiry(df) == "10NOV2026"


def test_nearest_expiry_includes_today_and_handles_empty(fixed_today) -> None:
    assert _nearest_expiry(pd.DataFrame({"expiry": ["06NOV2026", "13NOV2026"]})) == "06NOV2026"
    assert _nearest_expiry(pd.DataFrame({"expiry": ["01NOV2026"]})) is None
    assert _nearest_expiry(pd.DataFrame({"expiry": []})) is None


# ── Futures token lookup ──────────────────────────────────────────────────────


def _master() -> pd.DataFrame:
    rows = [
        # NIFTY futures (NFO) — Dec sorts first as text, Nov is nearest
        ("50001", "NIFTY29DEC26FUT", "NIFTY", "29DEC2026", "FUTIDX", "NFO"),
        ("50000", "NIFTY24NOV26FUT", "NIFTY", "24NOV2026", "FUTIDX", "NFO"),
        ("50002", "NIFTY27OCT26FUT", "NIFTY", "27OCT2026", "FUTIDX", "NFO"),  # expired
        # NIFTY weekly option on an earlier date — must not be picked
        ("60000", "NIFTY10NOV2624000CE", "NIFTY", "10NOV2026", "OPTIDX", "NFO"),
        # SENSEX futures on BFO
        ("70000", "SENSEX26NOV26FUT", "SENSEX", "26NOV2026", "FUTIDX", "BFO"),
    ]
    return pd.DataFrame(
        rows, columns=["token", "symbol", "name", "expiry", "instrumenttype", "exch_seg"]
    )


def _bare_router(db=None, dry_run: bool = False) -> DataRouter:
    """DataRouter with only the depth state initialised (no broker/scraper)."""
    router = DataRouter.__new__(DataRouter)
    router.db = db
    router.dry_run = dry_run
    router._fut_token_to_index = {}
    router._latest_fut_depth = {}
    router._depth_bars = {}
    router._prev_best = {}
    router._prev_fut_volume = {}
    router._depth_lock = threading.Lock()
    return router


def _router(master: pd.DataFrame, indices: list) -> DataRouter:
    router = _bare_router()
    router._master_contract = master
    router.registry = SimpleNamespace(get_active_indices=lambda: indices)
    return router


def test_near_futures_tokens(fixed_today) -> None:
    indices = [
        SimpleNamespace(id="NIFTY50", has_options=True, option_symbol="NIFTY"),
        SimpleNamespace(id="SENSEX", has_options=True, option_symbol="SENSEX"),
        SimpleNamespace(id="NIFTYIT", has_options=False, option_symbol=None),
        SimpleNamespace(id="BANKNIFTY", has_options=True, option_symbol="BANKNIFTY"),  # no rows
    ]
    tokens_by_exchange, token_to_index = _router(_master(), indices)._get_near_futures_tokens()

    assert tokens_by_exchange == {2: ["50000"], 4: ["70000"]}
    assert token_to_index == {"50000": "NIFTY50", "70000": "SENSEX"}


def test_futures_depth_cached_and_exposed() -> None:
    router = _bare_router(db=MagicMock())
    router._fut_token_to_index = {"50000": "NIFTY50"}

    feed = WebSocketFeed(client=MagicMock())
    feed.on_depth_update = router._handle_depth_update
    feed._on_data(None, _snapquote(token="50000"))
    feed._on_data(None, _snapquote(token="99999"))  # unrelated token ignored

    depth = router.get_futures_depth("NIFTY50")
    assert depth is not None
    assert depth["token"] == "50000"
    assert depth["bid"] == pytest.approx(24500.0)
    assert depth["ask"] == pytest.approx(24501.0)
    assert depth["depth_imbalance"] == pytest.approx((600 - 375) / 975)
    assert list(router._latest_fut_depth) == ["NIFTY50"]
    assert router.get_futures_depth("BANKNIFTY") is None


# ── Order flow imbalance ──────────────────────────────────────────────────────


def test_ofi_first_snapshot_is_zero() -> None:
    assert _best_level_ofi(None, (100.0, 10, 101.0, 10)) == 0.0


def test_ofi_unchanged_prices_tracks_queue_changes() -> None:
    # Same prices: bid queue +5, ask queue −3 → e = (15 − 10) − (7 − 10) = +8
    assert _best_level_ofi((100.0, 10, 101.0, 10), (100.0, 15, 101.0, 7)) == 8.0


def test_ofi_bid_up_is_buying_pressure() -> None:
    # Bid ticks up: +new bid qty; ask unchanged with same qty → 0 net on ask side
    assert _best_level_ofi((100.0, 10, 101.0, 10), (100.5, 4, 101.0, 10)) == 4.0


def test_ofi_ask_down_is_selling_pressure() -> None:
    # Ask ticks down: −new ask qty; bid unchanged with same qty → 0 net on bid side
    assert _best_level_ofi((100.0, 10, 101.0, 10), (100.0, 10, 100.5, 6)) == -6.0


# ── 1-minute depth aggregation + persistence ──────────────────────────────────


def _depth_tick(minute: int, second: int, bid: float, bid_qty: int, ask: float, ask_qty: int,
                bid5: int, ask5: int, ltp: float = 24500.0, oi: int = 1_000_000,
                vol: int = 0) -> LiveTick:
    return LiveTick(
        token="50000", exchange_type=2, ltp=ltp, volume=vol, open_interest=oi,
        oi_change_pct=0.0, bid_price=bid, ask_price=ask,
        open=0.0, high=0.0, low=0.0, close=0.0,
        timestamp=datetime(2026, 11, 6, 10, minute, second, tzinfo=dr._IST),
        bid_qty=bid_qty, ask_qty=ask_qty, bid_qty_5=bid5, ask_qty_5=ask5,
        total_buy_qty=600.0, total_sell_qty=400.0,
    )


@pytest.fixture()
def depth_db(tmp_path):
    from src.database.db_manager import DatabaseManager

    db = DatabaseManager(db_path=tmp_path / "depth.db")
    db.connect()
    db.initialise_schema()
    yield db
    db.close()


def test_depth_minute_bar_flushed_on_rollover(depth_db) -> None:
    router = _bare_router(db=depth_db)
    router._fut_token_to_index = {"50000": "NIFTY50"}

    # Minute 10:00 — three snapshots (cumulative day volume 500_000 → 500_300)
    router._handle_depth_update(_depth_tick(0, 5, 24500.0, 10, 24501.0, 10, 300, 100,
                                            vol=500_000))                                # baseline
    router._handle_depth_update(_depth_tick(0, 20, 24500.0, 15, 24501.0, 7, 200, 200,   # ofi +8
                                            vol=500_075))
    router._handle_depth_update(_depth_tick(0, 40, 24501.0, 4, 24502.0, 9, 100, 300,    # bid & ask both tick up
                                            ltp=24501.5, oi=1_000_500, vol=500_300))
    assert depth_db.fetch_all("SELECT * FROM futures_depth_1m") == []  # not flushed yet

    # First tick of 10:01 flushes 10:00
    router._handle_depth_update(_depth_tick(1, 2, 24501.0, 4, 24502.0, 9, 100, 100, vol=500_450))

    rows = router.get_futures_depth_history(
        "NIFTY50", "2026-11-06T10:00:00+05:30", "2026-11-06T10:01:00+05:30"
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["timestamp"] == "2026-11-06T10:00:00+05:30"
    assert row["fut_token"] == "50000"
    assert row["fut_close"] == pytest.approx(24501.5)
    assert row["open_interest"] == 1_000_500
    assert row["volume"] == 300  # first tick is baseline only
    assert row["samples"] == 3
    assert row["crossed_samples"] == 0
    assert row["spread_avg"] == pytest.approx(1.0)
    assert row["bid_qty_5_avg"] == pytest.approx(200.0)
    assert row["ask_qty_5_avg"] == pytest.approx(200.0)
    assert row["depth_imb_avg"] == pytest.approx((0.5 + 0.0 - 0.5) / 3)
    assert row["depth_imb_last"] == pytest.approx(-0.5)
    assert row["book_imb_avg"] == pytest.approx(0.2)
    # OFI: snapshot 2 → +8; snapshot 3: bid up (+4) & bid<=prev false; ask up → +7 (prev ask qty)
    assert row["ofi"] == pytest.approx(8.0 + 4.0 + 7.0)


def test_depth_bar_crossed_counted_and_stop_flushes(depth_db) -> None:
    router = _bare_router(db=depth_db)
    router._fut_token_to_index = {"50000": "NIFTY50"}
    router.ws_feed = MagicMock()
    router._current_bars = {}
    router._bars_lock = threading.Lock()

    router._handle_depth_update(_depth_tick(5, 0, 24505.0, 10, 24500.0, 10, 100, 100))  # crossed
    router.stop()

    rows = depth_db.fetch_all("SELECT samples, crossed_samples FROM futures_depth_1m")
    assert rows == [{"samples": 1, "crossed_samples": 1}]
    assert router._depth_bars == {}


def test_depth_dry_run_writes_nothing(depth_db) -> None:
    router = _bare_router(db=depth_db, dry_run=True)
    router._fut_token_to_index = {"50000": "NIFTY50"}

    router._handle_depth_update(_depth_tick(0, 0, 24500.0, 10, 24501.0, 10, 100, 100))
    router._handle_depth_update(_depth_tick(1, 0, 24500.0, 10, 24501.0, 10, 100, 100))

    assert depth_db.fetch_all("SELECT * FROM futures_depth_1m") == []


def test_migration_v19_creates_table(tmp_path) -> None:
    from src.database.db_manager import DatabaseManager
    from src.database.migrations import MigrationRunner
    from src.database.models import TABLE_DDL

    db = DatabaseManager(db_path=tmp_path / "mig.db")
    db.connect()
    conn = db._conn()
    with conn:
        for stmt in TABLE_DDL["schema_version"]:
            conn.execute(stmt)
    MigrationRunner(db).run_pending()

    assert db.table_exists("futures_depth_1m")
    versions = {r["version"] for r in db.fetch_all("SELECT version FROM schema_version")}
    assert 19 in versions
    db.close()


def test_depth_volume_counts_ticks_without_depth_and_clamps_reset(depth_db) -> None:
    router = _bare_router(db=depth_db)
    router._fut_token_to_index = {"50000": "NIFTY50"}

    router._handle_depth_update(_depth_tick(0, 0, 24500.0, 10, 24501.0, 10, 100, 100, vol=1_000))
    one_sided = _depth_tick(0, 10, 24500.0, 10, 24501.0, 10, 100, 100, vol=1_200)
    one_sided.ask_price = None  # no depth → not aggregated, but its volume must not vanish
    router._handle_depth_update(one_sided)
    router._handle_depth_update(_depth_tick(0, 20, 24500.0, 10, 24501.0, 10, 100, 100, vol=1_250))
    # Cumulative counter reset (e.g. new session) → clamp to 0, re-baseline
    router._handle_depth_update(_depth_tick(1, 0, 24500.0, 10, 24501.0, 10, 100, 100, vol=40))
    router._handle_depth_update(_depth_tick(1, 30, 24500.0, 10, 24501.0, 10, 100, 100, vol=90))
    router._handle_depth_update(_depth_tick(2, 0, 24500.0, 10, 24501.0, 10, 100, 100, vol=90))

    rows = depth_db.fetch_all("SELECT timestamp, volume FROM futures_depth_1m ORDER BY timestamp")
    assert [r["volume"] for r in rows] == [250, 50]
