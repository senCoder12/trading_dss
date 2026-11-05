"""Tests for src/data/contract_master.py — lot sizes and expiry calendar."""

from __future__ import annotations

import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from src.data.contract_master import (
    ContractMaster,
    parse_scrip_master,
    sync_registry_lot_sizes,
)
from src.data.index_registry import Index, IndexRegistry

_IST = ZoneInfo("Asia/Kolkata")

_RAW_ROWS = [
    # NIFTY weekly — note 09NOV2026 is a Monday (holiday-shifted)
    {"name": "NIFTY", "expiry": "09NOV2026", "lotsize": "65", "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
    {"name": "NIFTY", "expiry": "17NOV2026", "lotsize": "65", "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
    # SENSEX: lot size changes from the Jan 2027 contract
    {"name": "SENSEX", "expiry": "26NOV2026", "lotsize": "20", "instrumenttype": "OPTIDX", "exch_seg": "BFO"},
    {"name": "SENSEX", "expiry": "28JAN2027", "lotsize": "25", "instrumenttype": "OPTIDX", "exch_seg": "BFO"},
    # Non-option and malformed rows must be ignored
    {"name": "NIFTY", "expiry": "23NOV2026", "lotsize": "65", "instrumenttype": "FUTIDX", "exch_seg": "NFO"},
    {"name": "NIFTY", "expiry": "garbage", "lotsize": "65", "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
    {"name": "NIFTY", "expiry": "24NOV2026", "lotsize": "0", "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
    "not-a-dict",
]


@pytest.fixture(autouse=True)
def _enable_master(monkeypatch):
    # The suite disables the master globally; these tests exercise it directly.
    monkeypatch.delenv("TRADING_DSS_CONTRACT_MASTER_DISABLED", raising=False)


def _master(tmp_path, rows=_RAW_ROWS, allow_download=True):
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return rows

    return ContractMaster(cache_dir=tmp_path, fetcher=fetch, allow_download=allow_download), calls


def test_parse_keeps_only_valid_option_rows():
    contracts = parse_scrip_master(_RAW_ROWS)
    assert contracts["NIFTY"] == {"2026-11-09": 65, "2026-11-17": 65}
    assert contracts["SENSEX"] == {"2026-11-26": 20, "2027-01-28": 25}


def test_lot_size_is_per_expiry(tmp_path):
    m, _ = _master(tmp_path)
    assert m.refresh(today=date(2026, 11, 6))
    assert m.lot_size("SENSEX", expiry=date(2026, 11, 26)) == 20
    assert m.lot_size("SENSEX", expiry=date(2027, 1, 28)) == 25


def test_alias_resolves_registry_id(tmp_path):
    m, _ = _master(tmp_path)
    m.refresh(today=date(2026, 11, 6))
    assert m.lot_size("NIFTY50", as_of=date(2026, 11, 6)) == 65


def test_nearest_expiry_rolls_after_close(tmp_path):
    m, _ = _master(tmp_path)
    m.refresh(today=date(2026, 11, 9))
    before_close = datetime(2026, 11, 9, 15, 0, tzinfo=_IST)
    after_close = datetime(2026, 11, 9, 15, 31, tzinfo=_IST)
    assert m.nearest_expiry("NIFTY", before_close) == date(2026, 11, 9)
    assert m.nearest_expiry("NIFTY", after_close) == date(2026, 11, 17)


def test_is_expiry_day_uses_listed_dates_not_weekday(tmp_path):
    m, _ = _master(tmp_path)
    m.refresh(today=date(2026, 11, 6))
    assert m.is_expiry_day("NIFTY", date(2026, 11, 9)) is True      # Monday
    assert m.is_expiry_day("NIFTY", date(2026, 11, 10)) is False    # Tuesday
    assert m.is_expiry_day("UNKNOWN", date(2026, 11, 9)) is None


def test_cache_avoids_second_download(tmp_path):
    m1, calls1 = _master(tmp_path)
    m1.refresh(today=date(2026, 11, 6))
    assert calls1["n"] == 1
    m2, calls2 = _master(tmp_path)
    assert m2.refresh(today=date(2026, 11, 6))
    assert calls2["n"] == 0
    assert m2.source == "cache"


def test_falls_back_to_stale_cache_when_download_fails(tmp_path):
    m1, _ = _master(tmp_path)
    m1.refresh(today=date(2026, 11, 5))

    def boom():
        raise ConnectionError("offline")

    m2 = ContractMaster(cache_dir=tmp_path, fetcher=boom, allow_download=True)
    assert m2.refresh(today=date(2026, 11, 6))
    assert m2.source == "stale_cache:2026-11-05"
    assert m2.lot_size("NIFTY", expiry=date(2026, 11, 9)) == 65


def test_no_data_returns_none(tmp_path):
    m, calls = _master(tmp_path, allow_download=False)
    assert m.refresh(today=date(2026, 11, 6)) is False
    assert calls["n"] == 0
    assert m.lot_size("NIFTY") is None


def test_corrupt_cache_is_ignored(tmp_path):
    (tmp_path / "contract_master_2026-11-06.json").write_text("{not json")
    m, calls = _master(tmp_path)
    assert m.refresh(today=date(2026, 11, 6))
    assert calls["n"] == 1  # re-downloaded


def test_disabled_switch(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_DSS_CONTRACT_MASTER_DISABLED", "1")
    m, calls = _master(tmp_path)
    assert m.refresh(today=date(2026, 11, 6)) is False
    assert m.lot_size("NIFTY") is None
    assert calls["n"] == 0


def test_sync_registry_lot_sizes(tmp_path):
    reg = IndexRegistry([
        Index(id="NIFTY50", display_name="NIFTY 50", exchange="NSE", lot_size=25,
              has_options=True, option_symbol="NIFTY"),
        Index(id="NIFTY_IT", display_name="NIFTY IT", exchange="NSE"),
    ])
    m, _ = _master(tmp_path)
    m.refresh()
    changed = sync_registry_lot_sizes(reg, m, as_of=date(2026, 11, 6))
    assert changed == {"NIFTY50": 65}
    assert reg.get_index("NIFTY50").lot_size == 65
    assert reg.get_index("NIFTY_IT").lot_size is None
