"""
Tests for RiskManager — option-native (premium) risk.

The risk manager turns a spot-level TradingSignal into an executable option
trade.  Covered here:

- Spot trigger validation (SL side / noise floor / max width / OI tighten-only;
  target side + OI cap only on the profitable side)
- Contract selection on the chain (delta-targeted strike, expiry-day roll)
- Premium levels and the edge check (theta over the hold, spread, costs)
- Sizing on ₹ premium risk (never forced to 1 lot), daily limit incl. open
  risk, premium outlay cap, confidence scaling
- Entry gates: time window, position caps, same-direction cap, loss
  cooldown, consecutive-loss halt
- Position lifecycle on the option mark: SL / target / spot triggers,
  trailing to breakeven, time stop, force exit, model mark fallback
- close_position net P&L, persistence and restart restore, startup cleanup
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pytest

from src.data.options_chain import OptionsChainData, OptionStrike
from src.database.db_manager import DatabaseManager
from src.engine.option_contract import bs_price_greeks, time_to_expiry_years
from src.engine.option_costs import round_trip_cost
from src.engine.risk_manager import RefinedSignal, RiskConfig, RiskManager, _OpenPosition
from src.engine.signal_generator import TradingSignal
from src.engine.signal_tracker import SignalTracker

_IST = ZoneInfo("Asia/Kolkata")

SPOT = 24_500.0
LOT = 65
NOW = datetime(2026, 11, 5, 10, 30, tzinfo=_IST)      # Thursday
EXPIRY = date(2026, 11, 10)                             # next Tuesday
NEXT_EXPIRY = date(2026, 11, 17)
_STRIKE_FIELDS = {f.name for f in dataclasses.fields(OptionStrike)}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _strike(k: float, expiry: date, now: datetime, spot: float, iv: float = 0.14) -> OptionStrike:
    t = time_to_expiry_years(expiry, now)
    ce = bs_price_greeks(spot, k, t, iv, "CE").price
    pe = bs_price_greeks(spot, k, t, iv, "PE").price
    kw = {n: 0 for n in _STRIKE_FIELDS}
    kw.update(
        strike_price=float(k), ce_ltp=round(ce, 2), pe_ltp=round(pe, 2),
        ce_iv=iv * 100, pe_iv=iv * 100,
        # OI walls: PE (support) at 24_300, CE (resistance) at 24_700
        ce_oi=300_000 if k == 24_700 else 80_000,
        pe_oi=300_000 if k == 24_300 else 80_000,
        ce_bid_price=round(ce * 0.995, 2), ce_ask_price=round(ce * 1.005, 2),
        pe_bid_price=round(pe * 0.995, 2), pe_ask_price=round(pe * 1.005, 2),
    )
    return OptionStrike(**kw)


def _chain(
    expiry: date = EXPIRY, now: datetime = NOW, spot: float = SPOT, iv: float = 0.14,
) -> OptionsChainData:
    return OptionsChainData(
        index_id="NIFTY50", spot_price=spot, timestamp=now, expiry_date=expiry,
        strikes=tuple(_strike(k, expiry, now, spot, iv) for k in range(24_000, 25_050, 50)),
        available_expiries=(expiry, expiry + timedelta(days=7)),
    )


def _signal(
    signal_type: str = "BUY_CALL",
    confidence: str = "HIGH",
    entry: float = SPOT,
    sl_pts: float = 90.0,
    tgt_pts: float = 180.0,
    index_id: str = "NIFTY50",
    modifier: float = 1.0,
    warnings: Optional[list] = None,
    generated_at: datetime = NOW,
) -> TradingSignal:
    call = signal_type == "BUY_CALL"
    return TradingSignal(
        signal_id=str(uuid.uuid4()), index_id=index_id, generated_at=generated_at,
        signal_type=signal_type, confidence_level=confidence, confidence_score=0.7,
        entry_price=entry,
        target_price=entry + tgt_pts if call else entry - tgt_pts,
        stop_loss=entry - sl_pts if call else entry + sl_pts,
        risk_reward_ratio=round(tgt_pts / sl_pts, 2), regime="TREND_UP",
        weighted_score=1.2, vote_breakdown={}, risk_level="NORMAL",
        position_size_modifier=modifier, suggested_lot_count=1,
        estimated_max_loss=0.0, estimated_max_profit=0.0, reasoning="test",
        warnings=list(warnings or []),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    d = DatabaseManager(db_path=tmp_path / "rm.db")
    d.connect()
    d.initialise_schema()
    ts = NOW.isoformat()
    for idx, sym in (("NIFTY50", "NIFTY"), ("BANKNIFTY", "BANKNIFTY"), ("FINNIFTY", "FINNIFTY")):
        d.execute(
            """INSERT OR IGNORE INTO index_master
               (id, display_name, nse_symbol, yahoo_symbol, exchange, lot_size,
                has_options, option_symbol, sector_category, is_active, created_at, updated_at)
               VALUES (?,?,?,?, 'NSE', ?, 1, ?, 'broad_market', 1, ?, ?)""",
            (idx, idx, idx, f"^{idx}", LOT, sym, ts, ts),
        )
    return d


def _rm(db: DatabaseManager, **cfg) -> RiskManager:
    defaults = dict(total_capital=500_000.0, max_risk_per_trade_pct=2.0, max_risk_per_day_pct=10.0)
    defaults.update(cfg)
    rm = RiskManager(db, config=RiskConfig(**defaults))
    rm._get_lot_size = lambda index_id: LOT  # type: ignore[method-assign]
    rm._contract_lot_size = lambda index_id, expiry: LOT  # type: ignore[method-assign]
    rm._exchange_for = lambda index_id: "NSE"  # type: ignore[method-assign]
    return rm


@pytest.fixture()
def rm(db) -> RiskManager:
    return _rm(db)


def _open(rm: RiskManager, sig: Optional[TradingSignal] = None, chain=None, now=NOW) -> RefinedSignal:
    sig = sig or _signal()
    refined = rm.validate_and_refine_signal(sig, chain or _chain(now=now), _now=now)
    assert refined.is_valid, refined.rejection_reasons
    rm.track_open_position(refined)
    return refined


# ===========================================================================
# 1. Spot trigger validation
# ===========================================================================


class TestSpotTriggers:
    def test_sl_wrong_side_for_call_is_corrected(self, rm):
        sig = _signal(sl_pts=-50)  # SL above entry
        r = rm.validate_and_refine_signal(sig, _chain(), _now=NOW)
        assert r.refined_stop_loss < r.refined_entry

    def test_sl_wrong_side_for_put_is_corrected(self, rm):
        sig = _signal("BUY_PUT", sl_pts=-50)
        r = rm.validate_and_refine_signal(sig, _chain(), _now=NOW)
        assert r.refined_stop_loss > r.refined_entry

    def test_sl_too_tight_widened_to_noise_floor(self, rm):
        r = rm.validate_and_refine_signal(_signal(sl_pts=10), _chain(), _now=NOW)
        assert (r.refined_entry - r.refined_stop_loss) / r.refined_entry >= rm.config.min_sl_pct - 1e-9

    def test_sl_too_wide_tightened(self, rm):
        r = rm.validate_and_refine_signal(_signal(sl_pts=900, tgt_pts=1800), _chain(), _now=NOW)
        assert (r.refined_entry - r.refined_stop_loss) / r.refined_entry <= rm.config.max_sl_pct + 1e-9

    def test_oi_support_tightens_but_never_widens(self, rm):
        # PE wall at 24_300; a 300-pt stop (24_200) is tightened to just below the wall
        r = rm.validate_and_refine_signal(_signal(sl_pts=300, tgt_pts=700), _chain(), _now=NOW)
        assert r.refined_stop_loss == pytest.approx(24_300 * (1 - rm.config.sl_sr_buffer_pct), abs=0.01)
        # a 90-pt stop (24_410) is already tighter than the wall → unchanged
        r2 = rm.validate_and_refine_signal(_signal(sl_pts=90), _chain(), _now=NOW)
        assert r2.refined_stop_loss == pytest.approx(SPOT - 90)

    def test_target_capped_at_oi_resistance_for_medium(self, rm):
        r = rm.validate_and_refine_signal(
            _signal(confidence="MEDIUM", sl_pts=90, tgt_pts=400), _chain(), _now=NOW,
        )
        assert r.refined_target <= 24_700

    def test_resistance_below_entry_does_not_drag_target(self, rm):
        # spot has rallied above the CE wall: the wall is not overhead resistance
        spot = 24_800.0
        chain = _chain(spot=spot)
        r = rm.validate_and_refine_signal(
            _signal(confidence="MEDIUM", entry=spot, sl_pts=90, tgt_pts=200), chain, _now=NOW,
        )
        assert r.refined_target > spot

    def test_spot_rr_below_one_rejected(self, rm):
        r = rm.validate_and_refine_signal(_signal(sl_pts=100, tgt_pts=80), _chain(), _now=NOW)
        assert not r.is_valid


# ===========================================================================
# 2. Contract selection
# ===========================================================================


class TestContractSelection:
    def test_high_confidence_near_atm_delta(self, rm):
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        assert r.option_type == "CE"
        assert abs(r.option_greeks["delta"] - 0.50) < 0.08
        assert r.recommended_expiry == EXPIRY.isoformat()

    def test_medium_confidence_further_otm_than_high(self, rm):
        hi = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        med = rm.validate_and_refine_signal(_signal(confidence="MEDIUM", tgt_pts=190), _chain(), _now=NOW)
        assert med.recommended_strike > hi.recommended_strike

    def test_put_selects_pe(self, rm):
        r = rm.validate_and_refine_signal(_signal("BUY_PUT"), _chain(), _now=NOW)
        assert r.option_type == "PE" and r.option_greeks["delta"] < 0

    def test_no_chain_rejected(self, rm):
        r = rm.validate_and_refine_signal(_signal(), None, _now=NOW)
        assert not r.is_valid
        assert "No executable option contract" in r.rejection_reasons[0]

    def test_expiry_day_requires_next_expiry_chain(self, rm):
        expiry_day = datetime(2026, 11, 10, 11, 0, tzinfo=_IST)
        today_chain = _chain(expiry=EXPIRY, now=expiry_day)
        r = rm.validate_and_refine_signal(_signal(generated_at=expiry_day), today_chain, _now=expiry_day)
        assert not r.is_valid and "next-expiry chain is unavailable" in r.rejection_reasons[0]

    def test_expiry_day_uses_next_expiry_contract(self, rm):
        expiry_day = datetime(2026, 11, 10, 11, 0, tzinfo=_IST)
        r = rm.validate_and_refine_signal(
            _signal(generated_at=expiry_day),
            _chain(expiry=EXPIRY, now=expiry_day),
            _now=expiry_day,
            next_chain=_chain(expiry=NEXT_EXPIRY, now=expiry_day),
        )
        assert r.is_valid, r.rejection_reasons
        assert r.recommended_expiry == NEXT_EXPIRY.isoformat()


# ===========================================================================
# 3. Premium levels and edge
# ===========================================================================


class TestPremiumLevels:
    def test_levels_ordered_and_stop_capped(self, rm):
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        assert r.premium_stop < r.entry_premium < r.premium_target
        assert r.premium_stop >= r.premium_mid * (1 - rm.config.max_premium_sl_pct) - 0.01
        assert r.entry_premium == pytest.approx(r.premium_mid + r.half_spread, abs=0.01)

    def test_wide_spot_stop_capped_at_35pct_of_premium(self, rm):
        r = rm.validate_and_refine_signal(_signal(sl_pts=600, tgt_pts=1300), _chain(), _now=NOW)
        assert r.premium_stop == pytest.approx(r.premium_mid * 0.65, abs=0.02)
        assert any("capped" in a for a in r.adjustments_made)

    def test_rr_is_premium_rr_net_of_costs(self, rm):
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        assert r.is_valid
        assert r.risk_reward_ratio == r.premium_rr
        # R:R is measured on ₹ per lot after spread and statutory costs.
        # (It can exceed the spot R:R: a long option's gamma makes the
        # upside move in premium larger than the equal-sized downside.)
        exit_t = r.premium_target - r.half_spread
        cost_t = round_trip_cost(r.entry_premium, exit_t, LOT).total
        reward = (exit_t - r.entry_premium) * LOT - cost_t
        assert r.premium_rr == pytest.approx(reward / r.risk_per_lot, abs=0.01)

    def test_target_that_cannot_beat_theta_is_rejected(self, db):
        # Spot R:R is acceptable, but over a 3-day expected hold theta eats
        # the move: the premium at the spot target is below what was paid.
        rm = _rm(db, expected_hold_minutes=3 * 1440)
        r = rm.validate_and_refine_signal(_signal(sl_pts=90, tgt_pts=120), _chain(), _now=NOW)
        assert not r.is_valid
        assert any("No edge" in x for x in r.rejection_reasons)

    def test_premium_rr_below_minimum_rejected(self, db):
        rm = _rm(db, expected_hold_minutes=3 * 1440)
        r = rm.validate_and_refine_signal(_signal(sl_pts=90, tgt_pts=180), _chain(), _now=NOW)
        assert not r.is_valid
        assert any("Premium R:R" in x and "spot R:R was 2.00" in x for x in r.rejection_reasons)

    def test_expensive_iv_lowers_premium_rr(self, rm):
        cheap = rm.validate_and_refine_signal(_signal(), _chain(iv=0.12), _now=NOW)
        rich = rm.validate_and_refine_signal(_signal(), _chain(iv=0.30), _now=NOW)
        assert rich.premium_rr < cheap.premium_rr


# ===========================================================================
# 4. Sizing
# ===========================================================================


class TestSizing:
    def test_small_capital_skips_instead_of_forcing_one_lot(self, db):
        rm = _rm(db, total_capital=100_000.0)
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        assert not r.is_valid and r.lots == 0
        assert "skipped rather than forced" in r.rejection_reasons[0]

    def test_lots_from_premium_risk(self, rm):
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        budget = rm.config.total_capital * rm.config.max_risk_per_trade_pct / 100
        assert r.lots == min(int(budget // r.risk_per_lot), rm.config.max_lots_per_trade)
        assert r.max_loss_amount <= budget
        assert r.quantity == r.lots * LOT

    def test_max_loss_includes_spread_and_costs(self, rm):
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        exit_stop = r.premium_stop - r.half_spread
        costs = round_trip_cost(r.entry_premium, exit_stop, r.quantity).total
        assert r.max_loss_amount == pytest.approx((r.entry_premium - exit_stop) * r.quantity + costs, abs=0.05)

    def test_medium_confidence_sizes_smaller(self, db):
        rm = _rm(db, total_capital=2_000_000.0, max_lots_per_trade=50)
        hi = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        med = rm.validate_and_refine_signal(_signal(confidence="MEDIUM", tgt_pts=190), _chain(), _now=NOW)
        assert hi.is_valid and med.is_valid
        assert med.risk_amount < hi.risk_amount

    def test_position_modifier_scales_budget(self, rm):
        full = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        half = rm.validate_and_refine_signal(_signal(modifier=0.5), _chain(), _now=NOW)
        assert half.risk_amount == pytest.approx(full.risk_amount * 0.5)

    def test_max_lots_cap(self, db):
        rm = _rm(db, total_capital=10_000_000.0, max_lots_per_trade=4)
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        assert r.lots == 4

    def test_outlay_cap(self, db):
        rm = _rm(db, total_capital=500_000.0, max_premium_outlay_pct=1.0)  # ₹5k < 1 lot
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        assert not r.is_valid and "outlay" in r.rejection_reasons[0].lower()

    def test_kelly_negative_edge_rejects_when_enabled(self, db):
        rm = _rm(db, use_kelly=True, kelly_min_trades=1)
        rm._fetch_historical_stats = lambda idx: {  # type: ignore[method-assign]
            "total_trades": 40, "win_rate": 0.3, "avg_win": 100.0, "avg_loss": 100.0,
        }
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        assert not r.is_valid


# ===========================================================================
# 5. Daily limit and entry gates
# ===========================================================================


class TestDailyLimitAndGates:
    def test_open_position_risk_is_reserved(self, rm):
        before = rm._compute_daily_remaining()
        first = _open(rm)
        assert rm._compute_daily_remaining() == pytest.approx(before - first.max_loss_amount, abs=1.0)

    def test_lots_reduced_to_fit_daily_limit(self, db):
        rm = _rm(db, max_risk_per_day_pct=2.5)
        _open(rm)  # reserves ~2 % → ~0.5 % left
        r = rm.validate_and_refine_signal(_signal("BUY_PUT", index_id="BANKNIFTY"), _chain(), _now=NOW)
        assert not r.is_valid or r.max_loss_amount <= rm._compute_daily_remaining() + 1

    def test_realised_loss_reduces_remaining(self, rm):
        rm._daily_realised_pnl = -20_000.0
        assert rm._compute_daily_remaining() == pytest.approx(50_000 - 20_000)

    def test_daily_limit_exhausted_rejects(self, rm):
        rm._daily_realised_pnl = -60_000.0
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        assert not r.is_valid and "Daily loss limit" in r.rejection_reasons[0]

    def test_no_entry_before_0930(self, rm):
        early = NOW.replace(hour=9, minute=20)
        r = rm.validate_and_refine_signal(_signal(), _chain(now=early), _now=early)
        assert not r.is_valid and "09:30" in r.rejection_reasons[0]

    def test_no_entry_near_close(self, rm):
        late = NOW.replace(hour=15, minute=5)
        r = rm.validate_and_refine_signal(_signal(), _chain(now=late), _now=late)
        assert not r.is_valid and "market close" in r.rejection_reasons[0]

    def test_one_position_per_index(self, rm):
        _open(rm)
        r = rm.validate_and_refine_signal(_signal("BUY_PUT"), _chain(), _now=NOW)
        assert not r.is_valid and "NIFTY50" in r.rejection_reasons[0]

    def test_same_direction_cap(self, db):
        rm = _rm(db, max_same_direction_positions=2, max_open_positions=5, max_risk_per_day_pct=50)
        _open(rm, _signal(index_id="NIFTY50"))
        _open(rm, _signal(index_id="BANKNIFTY"))
        third_call = rm.validate_and_refine_signal(_signal(index_id="FINNIFTY"), _chain(), _now=NOW)
        assert not third_call.is_valid and "same-direction" in third_call.rejection_reasons[0]
        put = rm.validate_and_refine_signal(_signal("BUY_PUT", index_id="FINNIFTY"), _chain(), _now=NOW)
        assert put.is_valid, put.rejection_reasons

    def _closed(self, db, index_id: str, outcome: str, minutes_ago: int) -> None:
        ts = (NOW - timedelta(minutes=minutes_ago)).isoformat()
        db.execute(
            """INSERT INTO trading_signals (index_id, generated_at, signal_type, confidence_level,
               entry_price, target_price, stop_loss, risk_reward_ratio, regime, technical_vote,
               options_vote, news_vote, anomaly_vote, reasoning, outcome, actual_exit_price,
               actual_pnl, closed_at)
               VALUES (?, ?, 'BUY_CALL', 'HIGH', 1, 2, 0.5, 2, 'TRENDING', 'N', 'N', 'N', 'N',
                       '{}', ?, 1, ?, ?)""",
            (index_id, ts, outcome, -1000.0 if outcome == "LOSS" else 1000.0, ts),
        )

    def test_cooldown_after_loss_on_same_index(self, db, rm):
        self._closed(db, "NIFTY50", "LOSS", minutes_ago=10)
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        assert not r.is_valid and "Cooldown" in r.rejection_reasons[0]
        other = rm.validate_and_refine_signal(_signal(index_id="BANKNIFTY"), _chain(), _now=NOW)
        assert other.is_valid, other.rejection_reasons

    def test_cooldown_expires(self, db, rm):
        self._closed(db, "NIFTY50", "LOSS", minutes_ago=45)
        assert rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW).is_valid

    def test_consecutive_losses_halt_the_day(self, db, rm):
        for i, idx in enumerate(("BANKNIFTY", "FINNIFTY", "BANKNIFTY")):
            self._closed(db, idx, "LOSS", minutes_ago=120 - i * 30)
        r = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        assert not r.is_valid and "consecutive losses" in r.rejection_reasons[0]

    def test_a_win_resets_the_streak(self, db, rm):
        self._closed(db, "BANKNIFTY", "LOSS", 150)
        self._closed(db, "FINNIFTY", "LOSS", 120)
        self._closed(db, "BANKNIFTY", "WIN", 90)
        self._closed(db, "FINNIFTY", "LOSS", 60)
        assert rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW).is_valid

    def test_no_trade_rejected(self, rm):
        sig = _signal()
        sig.signal_type = "NO_TRADE"
        assert not rm.validate_and_refine_signal(sig, _chain(), _now=NOW).is_valid


# ===========================================================================
# 6. Position lifecycle (option mark)
# ===========================================================================


class TestLifecycle:
    def test_premium_stop_exit(self, rm):
        r = _open(rm)
        u = rm.update_position(r.signal_id, SPOT, _now=NOW + timedelta(minutes=5),
                               option_ltp=r.premium_stop - 1)
        assert u.action == "EXIT_SL" and u.premium_source == "quote" and u.current_pnl < 0

    def test_premium_target_exit(self, rm):
        r = _open(rm)
        u = rm.update_position(r.signal_id, SPOT, _now=NOW + timedelta(minutes=5),
                               option_ltp=r.premium_target + 1)
        assert u.action == "EXIT_TARGET" and u.current_pnl > 0

    def test_spot_stop_trigger_exits_even_if_premium_above_stop(self, rm):
        r = _open(rm)
        u = rm.update_position(r.signal_id, r.refined_stop_loss - 1, _now=NOW + timedelta(minutes=5),
                               option_ltp=r.entry_premium)
        assert u.action == "EXIT_SL"

    def test_trailing_to_breakeven_then_trailing_exit(self, rm):
        r = _open(rm)
        half_way = r.entry_premium + 0.55 * (r.premium_target - r.entry_premium)
        u = rm.update_position(r.signal_id, SPOT + 80, _now=NOW + timedelta(minutes=10), option_ltp=half_way)
        assert u.action == "HOLD"
        pos = rm._open_positions[r.signal_id]
        assert pos.premium_stop > pos.entry_premium  # breakeven incl. spread + costs
        u2 = rm.update_position(r.signal_id, SPOT + 10, _now=NOW + timedelta(minutes=20),
                                option_ltp=pos.premium_stop - 0.5)
        assert u2.action == "EXIT_TRAILING_SL"

    def test_stop_never_loosens(self, rm):
        r = _open(rm)
        pos = rm._open_positions[r.signal_id]
        rm.update_position(r.signal_id, SPOT + 120, _now=NOW + timedelta(minutes=10),
                           option_ltp=r.entry_premium + 0.8 * (r.premium_target - r.entry_premium))
        locked = pos.premium_stop
        rm.update_position(r.signal_id, SPOT + 60, _now=NOW + timedelta(minutes=15),
                           option_ltp=r.entry_premium + 0.6 * (r.premium_target - r.entry_premium))
        assert pos.premium_stop == locked

    def test_time_stop_when_trade_not_working(self, rm):
        r = _open(rm)
        u = rm.update_position(r.signal_id, SPOT, _now=NOW + timedelta(minutes=65),
                               option_ltp=r.entry_premium + 1)
        assert u.action == "EXIT_TIME_STOP"

    def test_no_time_stop_when_working(self, rm):
        r = _open(rm)
        working = r.entry_premium + 0.45 * (r.entry_premium - r.premium_stop)
        u = rm.update_position(r.signal_id, SPOT + 40, _now=NOW + timedelta(minutes=65), option_ltp=working)
        assert u.action == "HOLD"

    def test_force_exit_at_1515(self, rm):
        r = _open(rm)
        u = rm.update_position(r.signal_id, SPOT, _now=NOW.replace(hour=15, minute=15),
                               option_ltp=r.entry_premium)
        assert u.action == "EXIT_TIME"

    def test_model_mark_when_no_quote(self, rm):
        r = _open(rm)
        u = rm.update_position(r.signal_id, SPOT + 50, _now=NOW + timedelta(minutes=5))
        assert u.premium_source == "model"
        assert u.mark_premium > r.premium_mid  # call gained on +50 spot

    def test_flat_spot_loses_to_theta(self, rm):
        r = _open(rm)
        u = rm.update_position(r.signal_id, SPOT, _now=NOW + timedelta(minutes=55))
        assert u.mark_premium < r.premium_mid and u.current_pnl < 0

    def test_hero_zero_exits_at_1455(self, rm):
        expiry_day = datetime(2026, 11, 10, 13, 0, tzinfo=_IST)
        r = _open(rm, _signal(warnings=["HERO_ZERO: gamma squeeze"], generated_at=expiry_day,
                              sl_pts=50, tgt_pts=150),
                  chain=_chain(expiry=EXPIRY, now=expiry_day), now=expiry_day)
        assert r.is_hero_zero and r.recommended_expiry == EXPIRY.isoformat()
        assert r.max_loss_amount <= rm.config.total_capital * rm.config.hero_zero_risk_pct / 100
        u = rm.update_position(r.signal_id, SPOT, _now=expiry_day.replace(hour=14, minute=55),
                               option_ltp=r.entry_premium)
        assert u.action == "EXIT_TIME"

    def test_legacy_spot_position_still_managed(self, rm):
        rm._open_positions["legacy"] = _OpenPosition(
            db_id=-1, signal_id="legacy", index_id="NIFTY50", signal_type="BUY_CALL",
            entry_price=SPOT, current_sl=SPOT - 100, target_price=SPOT + 200, lots=1,
            lot_size=LOT, entry_time=NOW, confidence_level="HIGH",
        )
        assert rm.update_position("legacy", SPOT - 101, _now=NOW + timedelta(minutes=5)).action == "EXIT_SL"


# ===========================================================================
# 7. close_position, persistence, restart
# ===========================================================================


class TestCloseAndPersistence:
    def _record_and_open(self, db, rm) -> RefinedSignal:
        refined = rm.validate_and_refine_signal(_signal(), _chain(), _now=NOW)
        assert refined.is_valid, refined.rejection_reasons
        SignalTracker(db, capital=rm.config.total_capital).record_signal(refined, is_actionable=True)
        rm.track_open_position(refined)
        return refined

    def test_close_returns_net_pnl_and_updates_db(self, db, rm):
        r = self._record_and_open(db, rm)
        exit_mark = r.entry_premium + 20
        pnl = rm.close_position(r.signal_id, SPOT + 50, "TARGET", exit_premium=exit_mark)
        exit_px = exit_mark - r.half_spread
        costs = round_trip_cost(r.entry_premium, exit_px, r.quantity).total
        assert pnl == pytest.approx((exit_px - r.entry_premium) * r.quantity - costs, abs=0.05)
        row = db.fetch_one("SELECT outcome, actual_pnl, actual_exit_price, reasoning FROM trading_signals", ())
        assert row["outcome"] == "WIN" and row["actual_pnl"] == pytest.approx(pnl, abs=0.05)
        assert row["actual_exit_price"] == pytest.approx(exit_px, abs=0.01)
        assert json.loads(row["reasoning"])["exit_details"]["reason"] == "TARGET"
        assert rm._daily_realised_pnl == pytest.approx(pnl, abs=0.05)

    def test_tiny_gain_after_costs_is_a_loss(self, db, rm):
        r = self._record_and_open(db, rm)
        pnl = rm.close_position(r.signal_id, SPOT, "TIME", exit_premium=r.entry_premium + r.half_spread)
        assert pnl < 0
        assert db.fetch_one("SELECT outcome FROM trading_signals", ())["outcome"] == "LOSS"

    def test_close_unknown_returns_none(self, rm):
        assert rm.close_position("nope", SPOT, "SL") is None

    def test_restart_restores_contract_and_trailed_stop(self, db):
        # Restart recovery loads TODAY's open rows, so this test runs on the
        # real date (a fixed past date would be expired by startup cleanup).
        now = datetime.now(tz=_IST).replace(hour=10, minute=30, second=0, microsecond=0)
        expiry = (now + timedelta(days=5)).date()
        rm = _rm(db)
        refined = rm.validate_and_refine_signal(
            _signal(generated_at=now), _chain(expiry=expiry, now=now), _now=now,
        )
        assert refined.is_valid, refined.rejection_reasons
        SignalTracker(db, capital=rm.config.total_capital).record_signal(refined, is_actionable=True)
        rm.track_open_position(refined)
        rm.update_position(
            refined.signal_id, SPOT + 100, _now=now + timedelta(minutes=10),
            option_ltp=refined.entry_premium + 0.8 * (refined.premium_target - refined.entry_premium),
        )
        trailed = rm._open_positions[refined.signal_id].premium_stop
        assert trailed > refined.premium_stop

        restarted = _rm(db)  # fresh instance on the same DB
        pos = restarted._open_positions[refined.signal_id]
        assert pos.is_option
        assert (pos.option_type, pos.strike, pos.expiry) == ("CE", refined.recommended_strike, expiry)
        assert pos.premium_stop == pytest.approx(trailed)
        assert pos.initial_premium_stop == pytest.approx(refined.premium_stop)
        assert pos.lots == refined.lots

    def test_startup_cleanup_marks_orphans_expired_without_pnl(self, db):
        yesterday = (datetime.now(tz=_IST) - timedelta(days=1)).isoformat()
        db.execute(
            """INSERT INTO trading_signals (index_id, generated_at, signal_type, confidence_level,
               entry_price, target_price, stop_loss, risk_reward_ratio, regime, technical_vote,
               options_vote, news_vote, anomaly_vote, reasoning, outcome)
               VALUES ('NIFTY50', ?, 'BUY_CALL', 'HIGH', 24500, 24700, 24400, 2, 'TRENDING',
                       'N','N','N','N','{}','OPEN')""",
            (yesterday,),
        )
        _rm(db)
        row = db.fetch_one("SELECT outcome, actual_pnl FROM trading_signals", ())
        assert row["outcome"] == "EXPIRED" and row["actual_pnl"] is None


class TestPortfolioSummary:
    def test_option_position_outlay_is_premium(self, rm):
        r = _open(rm)
        summary = rm.get_portfolio_summary()
        assert summary.current_exposure == pytest.approx(r.entry_premium * r.quantity, abs=0.05)
        assert summary.open_positions[0]["contract"].endswith("CE")
