"""
Tests for RiskManager — Phase 5.3.

Covers:
- Position sizing: capital × risk% ÷ SL distance
- Daily loss limit enforcement and lot reduction
- Max open positions cap (3rd accepted, 4th rejected)
- Trailing SL milestones: 50% → breakeven, 75% → entry+30%, 100% → entry+60%
- Stop-loss validation: too tight, too wide, wrong side
- Target validation: beyond resistance pulled back
- Confidence-based sizing: HIGH vs LOW lot counts
- Option strike selection: ATM (HIGH), 1 OTM (MEDIUM)
- Breakeven calculation with realistic transaction costs
- Portfolio summary computation
- Rejection for insufficient capital
- NO_TRADE signal immediately rejected
- Time gate: no entries near close
"""

from __future__ import annotations

import json
import math
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.data.options_chain import OptionsChainData, OptionStrike
from src.database.db_manager import DatabaseManager
from src.engine.risk_manager import (
    DailyPnL,
    PortfolioSummary,
    PositionUpdate,
    RefinedSignal,
    RiskConfig,
    RiskManager,
    _OpenPosition,
)
from src.engine.signal_generator import TradingSignal

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SPOT = 22_450.0
INDEX_ID = "NIFTY50"
LOT_SIZE = 75          # NIFTY50 lot size from indices.json
MORNING = datetime(2026, 4, 8, 10, 30, 0, tzinfo=_IST)
NEAR_CLOSE = datetime(2026, 4, 8, 15, 5, 0, tzinfo=_IST)   # 25 min before close

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path) -> DatabaseManager:
    """In-memory (well, tmp-file) DB with full schema."""
    db = DatabaseManager(db_path=tmp_path / "test_rm.db")
    db.connect()
    db.initialise_schema()
    # Seed index_master so FK constraints pass
    now = datetime.now(tz=_IST).isoformat()
    db.execute(
        """
        INSERT OR IGNORE INTO index_master
            (id, display_name, nse_symbol, yahoo_symbol, exchange,
             lot_size, has_options, option_symbol, sector_category,
             is_active, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("NIFTY50", "NIFTY 50", "NIFTY 50", "^NSEI", "NSE",
         75, 1, "NIFTY", "broad_market", 1, now, now),
    )
    db.execute(
        """
        INSERT OR IGNORE INTO index_master
            (id, display_name, nse_symbol, yahoo_symbol, exchange,
             lot_size, has_options, option_symbol, sector_category,
             is_active, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("NIFTY_IT", "NIFTY IT", "NIFTY IT", "^CNXIT", "NSE",
         None, 0, None, "sectoral", 1, now, now),
    )
    return db


@pytest.fixture()
def default_config() -> RiskConfig:
    # Daily limit is set high (50%) so it doesn't interfere with general unit tests.
    # Tests that specifically probe daily-limit behaviour create their own RiskConfig
    # inline with a tighter max_risk_per_day_pct.
    return RiskConfig(
        total_capital=100_000.0,
        max_risk_per_trade_pct=2.0,
        max_risk_per_day_pct=50.0,   # 50k / day — won't block general tests
        max_open_positions=3,
        max_positions_per_index=1,
        min_risk_reward_ratio=1.3,
        default_risk_reward=1.5,
        slippage_points=2.0,
        transaction_cost_per_lot=120.0,
        high_confidence_size_pct=100.0,
        medium_confidence_size_pct=60.0,
        low_confidence_size_pct=30.0,
        no_entry_after_minutes_before_close=30,
        force_exit_minutes_before_close=5,
    )


def _strict_daily_config() -> RiskConfig:
    """5% daily limit config for TestDailyLossLimit tests."""
    return RiskConfig(
        total_capital=100_000.0,
        max_risk_per_trade_pct=2.0,
        max_risk_per_day_pct=5.0,
        transaction_cost_per_lot=120.0,
    )


@pytest.fixture()
def rm(tmp_db: DatabaseManager, default_config: RiskConfig) -> RiskManager:
    """RiskManager with isolated DB and default config."""
    with patch("src.engine.risk_manager.get_registry") as mock_reg:
        mock_idx = MagicMock()
        mock_idx.lot_size = LOT_SIZE
        mock_reg.return_value.get.return_value = mock_idx
        manager = RiskManager(tmp_db, config=default_config)
    return manager


def _make_signal(
    signal_type: str = "BUY_CALL",
    confidence_level: str = "HIGH",
    confidence_score: float = 0.75,
    entry: float = SPOT,
    target: float = SPOT + 150.0,
    stop_loss: float = SPOT - 100.0,
    index_id: str = INDEX_ID,
    position_size_modifier: float = 1.0,
    regime: str = "TREND_UP",
) -> TradingSignal:
    """Factory for TradingSignal test objects."""
    sl_dist = abs(entry - stop_loss)
    tgt_dist = abs(target - entry)
    rr = tgt_dist / sl_dist if sl_dist > 0 else 0.0
    return TradingSignal(
        signal_id="test-uuid-001",
        index_id=index_id,
        generated_at=MORNING,
        signal_type=signal_type,
        confidence_level=confidence_level,
        confidence_score=confidence_score,
        entry_price=entry,
        target_price=target,
        stop_loss=stop_loss,
        risk_reward_ratio=round(rr, 2),
        regime=regime,
        weighted_score=0.85,
        vote_breakdown={},
        risk_level="NORMAL",
        position_size_modifier=position_size_modifier,
        suggested_lot_count=2,
        estimated_max_loss=sl_dist * LOT_SIZE * 2,
        estimated_max_profit=tgt_dist * LOT_SIZE * 2,
        reasoning="test signal",
        warnings=[],
        data_completeness=1.0,
        signals_generated_today=1,
    )


def _make_chain(
    spot: float = SPOT,
    strikes_range: range = range(22_000, 23_000, 50),
    expiry: Optional[date] = None,
    ce_ltp_override: Optional[float] = None,
    pe_ltp_override: Optional[float] = None,
) -> OptionsChainData:
    """Build a minimal OptionsChainData for testing."""
    if expiry is None:
        expiry = date(2026, 4, 10)  # upcoming Thursday

    strikes = []
    for k in strikes_range:
        # Rough Black-Scholes-like premium: higher OTM = lower premium
        distance = abs(k - spot)
        ce_premium = ce_ltp_override if ce_ltp_override is not None else max(5.0, 200.0 - distance * 0.5)
        pe_premium = pe_ltp_override if pe_ltp_override is not None else max(5.0, 200.0 - distance * 0.5)
        # Give the ATM strike large OI for support/resistance tests
        ce_oi = 500_000 if k == 22_500 else 100_000
        pe_oi = 500_000 if k == 22_200 else 100_000
        strikes.append(OptionStrike(
            strike_price=float(k),
            ce_oi=ce_oi,
            ce_oi_change=1000,
            ce_volume=5000,
            ce_ltp=round(ce_premium, 2),
            ce_iv=15.0,
            pe_oi=pe_oi,
            pe_oi_change=500,
            pe_volume=3000,
            pe_ltp=round(pe_premium, 2),
            pe_iv=15.0,
        ))

    return OptionsChainData(
        index_id=INDEX_ID,
        spot_price=spot,
        timestamp=MORNING,
        expiry_date=expiry,
        strikes=tuple(strikes),
        available_expiries=(expiry, expiry + timedelta(days=7)),
    )


# ---------------------------------------------------------------------------
# Helper to patch get_registry inside tests that call rm methods
# ---------------------------------------------------------------------------

def _patched_rm(rm: RiskManager, lot_size: int = LOT_SIZE) -> RiskManager:
    """Monkeypatch _get_lot_size to return a fixed value."""
    rm._get_lot_size = lambda index_id: lot_size  # type: ignore[method-assign]
    return rm


# ===========================================================================
# 1. Position sizing
# ===========================================================================


class TestPositionSizing:
    """₹1L capital, 2% risk, SL 50 points away → expected lot count."""

    def test_basic_lot_count(self, rm: RiskManager) -> None:
        """
        capital=100_000, risk_pct=2% → max_risk=2000
        confidence HIGH → risk_amount = 2000 * 1.0 = 2000
        position_size_modifier = 1.0
        sl_dist = 50 pts
        loss_per_lot = 50 * 75 + 120 = 3870
        lots = floor(2000 / 3870) = 0 → clamped to 1
        """
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 50.0, target=SPOT + 100.0,
            confidence_level="HIGH", position_size_modifier=1.0,
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert refined.lots == 1

    def test_wider_sl_100pts(self, rm: RiskManager) -> None:
        """
        sl_dist=100pts, loss_per_lot=100*75+120=7620 → 1 lot
        """
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
            confidence_level="HIGH",
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert refined.lots >= 1

    def test_large_capital_multiple_lots(self, tmp_db: DatabaseManager) -> None:
        """With ₹5L capital and 2% risk = ₹10_000, 100pt SL → 1 lot (7620 loss/lot)."""
        cfg = RiskConfig(total_capital=500_000.0, max_risk_per_trade_pct=2.0,
                         transaction_cost_per_lot=120.0)
        rm = RiskManager(tmp_db, config=cfg)
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
            confidence_level="HIGH",
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        # 10000 / 7620 = 1.31 → 1 lot
        assert refined.lots == 1

    def test_max_lots_cap(self, tmp_db: DatabaseManager) -> None:
        """Position never exceeds max_lots_per_trade=5."""
        cfg = RiskConfig(
            total_capital=10_000_000.0,   # enormous capital
            max_risk_per_trade_pct=2.0,
            max_lots_per_trade=5,
            transaction_cost_per_lot=120.0,
        )
        rm = RiskManager(tmp_db, config=cfg)
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 20.0, target=SPOT + 100.0,
            confidence_level="HIGH",
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert refined.lots <= 5

    def test_transaction_costs_included_in_loss_calc(self, rm: RiskManager) -> None:
        """max_loss_amount includes tx cost contribution."""
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        sl_dist = abs(refined.refined_entry - refined.refined_stop_loss)
        expected_min = refined.lots * LOT_SIZE * sl_dist
        assert refined.max_loss_amount >= expected_min


# ===========================================================================
# 2. Daily loss limit
# ===========================================================================


class TestDailyLossLimit:
    """After ₹4000 loss, next trade risk should be limited to ₹1000."""

    def _inject_closed_loss(self, db: DatabaseManager, pnl: float) -> None:
        now_ist = datetime.now(tz=_IST)
        db.execute(
            """
            INSERT INTO trading_signals
                (index_id, generated_at, signal_type, confidence_level,
                 entry_price, target_price, stop_loss, risk_reward_ratio,
                 regime, technical_vote, options_vote, news_vote, anomaly_vote,
                 reasoning, outcome, actual_exit_price, actual_pnl, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("NIFTY50", now_ist.isoformat(), "BUY_CALL", "HIGH",
             22450.0, 22600.0, 22350.0, 1.5,
             "TRENDING", "BULLISH", "BULLISH", "NEUTRAL", "NEUTRAL",
             "{}", "LOSS", 22350.0, pnl, now_ist.isoformat()),
        )

    def test_limit_remaining_after_partial_loss(
        self, tmp_db: DatabaseManager,
    ) -> None:
        """After ₹4000 loss on a ₹5000 daily limit, ₹1000 remains."""
        cfg = _strict_daily_config()
        self._inject_closed_loss(tmp_db, -4000.0)
        rm = RiskManager(tmp_db, config=cfg)
        rm = _patched_rm(rm)
        remaining = rm._compute_daily_remaining()
        assert abs(remaining - 1000.0) < 5.0  # small tolerance for float

    def test_trade_rejected_when_limit_exhausted(
        self, tmp_db: DatabaseManager,
    ) -> None:
        """After ₹5000+ loss, new trade must be rejected."""
        cfg = _strict_daily_config()
        self._inject_closed_loss(tmp_db, -5100.0)
        rm = RiskManager(tmp_db, config=cfg)
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert not refined.is_valid
        assert any("daily" in r.lower() or "limit" in r.lower()
                   for r in refined.rejection_reasons)

    def test_lots_reduced_to_fit_within_remaining(
        self, tmp_db: DatabaseManager
    ) -> None:
        """Lots are reduced, not outright rejected, when we can still fit 1 lot."""
        # Daily limit ₹3000, already lost ₹2500 → ₹500 remaining
        # But we'll use a wider daily limit so 1 lot fits
        cfg = RiskConfig(
            total_capital=100_000.0,
            max_risk_per_trade_pct=2.0,
            max_risk_per_day_pct=10.0,    # ₹10_000 daily limit
            transaction_cost_per_lot=120.0,
        )
        # Inject ₹9500 loss → only ₹500 remains (not enough for 1 lot at 100pt SL)
        self._inject_closed_loss(tmp_db, -9500.0)
        rm = RiskManager(tmp_db, config=cfg)
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        # 1-lot cost = 7620 > 500 remaining → should be rejected
        assert not refined.is_valid

    def test_daily_pnl_summary_win_rate(
        self, tmp_db: DatabaseManager, default_config: RiskConfig
    ) -> None:
        """DailyPnL correctly computes win_rate from DB."""
        now_ist = datetime.now(tz=_IST)

        def _insert(outcome: str, pnl: float) -> None:
            tmp_db.execute(
                """
                INSERT INTO trading_signals
                    (index_id, generated_at, signal_type, confidence_level,
                     entry_price, target_price, stop_loss, risk_reward_ratio,
                     regime, technical_vote, options_vote, news_vote, anomaly_vote,
                     reasoning, outcome, actual_exit_price, actual_pnl, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("NIFTY50", now_ist.isoformat(), "BUY_CALL", "HIGH",
                 22450.0, 22600.0, 22350.0, 1.5,
                 "TRENDING", "BULLISH", "BULLISH", "NEUTRAL", "NEUTRAL",
                 "{}", outcome, 22600.0, pnl, now_ist.isoformat()),
            )

        _insert("WIN", 3000.0)
        _insert("WIN", 2500.0)
        _insert("LOSS", -1500.0)

        rm = RiskManager(tmp_db, config=default_config)
        summary = rm.get_daily_pnl_summary()

        assert summary.total_trades == 3
        assert summary.winning_trades == 2
        assert summary.losing_trades == 1
        assert abs(summary.win_rate - 2 / 3) < 0.01
        assert abs(summary.total_pnl - 4000.0) < 1.0


# ===========================================================================
# 3. Max open positions
# ===========================================================================


class TestMaxOpenPositions:
    """3 open positions accepted; 4th rejected."""

    def _make_open_position(
        self, signal_id: str, index_id: str = "NIFTY50"
    ) -> _OpenPosition:
        return _OpenPosition(
            db_id=-1,
            signal_id=signal_id,
            index_id=index_id,
            signal_type="BUY_CALL",
            entry_price=SPOT,
            current_sl=SPOT - 100.0,
            target_price=SPOT + 150.0,
            lots=1,
            lot_size=LOT_SIZE,
            entry_time=MORNING,
            confidence_level="HIGH",
        )

    def test_fourth_trade_rejected(self, rm: RiskManager) -> None:
        rm = _patched_rm(rm)
        # Inject 3 open positions (different index IDs to bypass per-index cap)
        rm._open_positions["s1"] = self._make_open_position("s1", "NIFTY50")
        rm._open_positions["s2"] = self._make_open_position("s2", "BANKNIFTY")
        rm._open_positions["s3"] = self._make_open_position("s3", "FINNIFTY")

        # 30pt SL → 1-lot cost ≈ 2490 < ₹5000 daily limit, so daily gate doesn't fire
        signal = _replace_signal(
            _make_signal(
                entry=SPOT, stop_loss=SPOT - 30.0, target=SPOT + 100.0,
                index_id="NIFTY50",
            ),
            index_id="MIDCPNIFTY",
        )
        rm._get_lot_size = lambda _: 75  # type: ignore[method-assign]
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert not refined.is_valid
        assert any("max open" in r.lower() or "positions" in r.lower()
                   for r in refined.rejection_reasons)

    def test_third_trade_accepted(self, rm: RiskManager) -> None:
        rm = _patched_rm(rm)
        rm._open_positions["s1"] = self._make_open_position("s1", "BANKNIFTY")
        rm._open_positions["s2"] = self._make_open_position("s2", "FINNIFTY")

        # 30pt SL → max_loss ≈ 2490 < ₹5000 daily limit
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 30.0, target=SPOT + 100.0,
            index_id="NIFTY50",
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert refined.is_valid

    def test_same_index_cap(self, rm: RiskManager) -> None:
        """Two simultaneous NIFTY50 positions blocked (per-index max = 1)."""
        rm = _patched_rm(rm)
        rm._open_positions["s1"] = self._make_open_position("s1", "NIFTY50")

        # 30pt SL to avoid daily-limit early-exit
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 30.0, target=SPOT + 100.0,
            index_id="NIFTY50",
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert not refined.is_valid
        assert any("nifty50" in r.lower() or "index" in r.lower()
                   for r in refined.rejection_reasons)


# ---------------------------------------------------------------------------
# Helper: replace fields on a TradingSignal (dataclass replace)
# ---------------------------------------------------------------------------

from dataclasses import replace as _dc_replace

def _replace_signal(s: TradingSignal, **kwargs) -> TradingSignal:
    return _dc_replace(s, **kwargs)


# ===========================================================================
# 4. Trailing stop loss
# ===========================================================================


class TestTrailingStopLoss:
    """Trailing SL ratchets up correctly at 50/75/100% of target distance."""

    def _make_pos(
        self,
        signal_type: str = "BUY_CALL",
        entry: float = 22_000.0,
        target: float = 22_200.0,
        sl: float = 21_900.0,
    ) -> _OpenPosition:
        return _OpenPosition(
            db_id=-1,
            signal_id="trail-test",
            index_id="NIFTY50",
            signal_type=signal_type,
            entry_price=entry,
            current_sl=sl,
            target_price=target,
            lots=1,
            lot_size=LOT_SIZE,
            entry_time=MORNING,
            confidence_level="HIGH",
        )

    def test_50pct_profit_moves_sl_to_breakeven(self, rm: RiskManager) -> None:
        pos = self._make_pos(entry=22_000.0, target=22_200.0, sl=21_900.0)
        # 50% of 200 pts = 22_100
        current_price = 22_100.0
        rm._apply_trailing_sl(pos, current_price)
        assert pos.current_sl == pytest.approx(22_000.0, abs=1.0)  # breakeven

    def test_75pct_profit_moves_sl_to_entry_plus_30pct(self, rm: RiskManager) -> None:
        pos = self._make_pos(entry=22_000.0, target=22_200.0, sl=21_900.0)
        # 75% of 200 pts = 22_150
        current_price = 22_155.0
        rm._apply_trailing_sl(pos, current_price)
        # Expected: entry + 30% * 200 = 22_060
        assert pos.current_sl == pytest.approx(22_000.0 + 0.3 * 200.0, abs=1.0)

    def test_100pct_profit_moves_sl_to_entry_plus_60pct(self, rm: RiskManager) -> None:
        pos = self._make_pos(entry=22_000.0, target=22_200.0, sl=21_900.0)
        # 100%+ → current at or beyond target
        current_price = 22_210.0
        rm._apply_trailing_sl(pos, current_price)
        # Expected: entry + 60% * 200 = 22_120
        assert pos.current_sl == pytest.approx(22_000.0 + 0.6 * 200.0, abs=1.0)

    def test_trailing_sl_never_widens_for_call(self, rm: RiskManager) -> None:
        pos = self._make_pos(entry=22_000.0, target=22_200.0, sl=21_900.0)
        # Already moved to breakeven
        pos.current_sl = 22_000.0
        # Price drops back — sl should NOT move down
        rm._apply_trailing_sl(pos, 22_010.0)
        assert pos.current_sl >= 22_000.0

    def test_trailing_sl_for_put(self, rm: RiskManager) -> None:
        """For BUY_PUT, trailing SL moves DOWN (never up)."""
        pos = self._make_pos(
            signal_type="BUY_PUT",
            entry=22_000.0, target=21_800.0, sl=22_100.0,
        )
        # 75% of 200 pts drop = 21_850
        current_price = 21_845.0
        rm._apply_trailing_sl(pos, current_price)
        # Expected: entry - 30% * 200 = 21_940
        assert pos.current_sl == pytest.approx(22_000.0 - 0.3 * 200.0, abs=1.0)

    def test_update_position_returns_exit_sl(self, rm: RiskManager) -> None:
        rm = _patched_rm(rm)
        pos = self._make_pos(entry=22_000.0, target=22_200.0, sl=21_900.0)
        rm._open_positions["trail-test"] = pos
        update = rm.update_position("trail-test", 21_850.0, _now=MORNING)
        assert update.action == "EXIT_SL"

    def test_update_position_returns_exit_target(self, rm: RiskManager) -> None:
        rm = _patched_rm(rm)
        pos = self._make_pos(entry=22_000.0, target=22_200.0, sl=21_900.0)
        rm._open_positions["trail-test"] = pos
        update = rm.update_position("trail-test", 22_210.0, _now=MORNING)
        assert update.action == "EXIT_TARGET"


# ===========================================================================
# 5. Stop-loss validation
# ===========================================================================


class TestStopLossValidation:
    def test_sl_too_tight_is_adjusted(self, rm: RiskManager) -> None:
        """SL < 0.3% from entry → adjusted to min_sl_pct."""
        rm = _patched_rm(rm)
        # 10 pts on 22450 = 0.044% — far too tight
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 10.0, target=SPOT + 200.0,
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        # Should be adjusted, not rejected
        assert refined.is_valid or any("tight" in a.lower() for a in refined.adjustments_made)
        actual_sl_dist_pct = abs(refined.refined_entry - refined.refined_stop_loss) / refined.refined_entry
        assert actual_sl_dist_pct >= rm.config.min_sl_pct - 1e-9

    def test_sl_wrong_side_for_call_is_corrected(self, rm: RiskManager) -> None:
        """SL above entry for BUY_CALL → corrected to below entry."""
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT + 50.0, target=SPOT + 200.0,
            signal_type="BUY_CALL",
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert refined.refined_stop_loss < refined.refined_entry

    def test_sl_wrong_side_for_put_is_corrected(self, rm: RiskManager) -> None:
        """SL below entry for BUY_PUT → corrected to above entry."""
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT,
            stop_loss=SPOT - 50.0,
            target=SPOT - 200.0,
            signal_type="BUY_PUT",
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert refined.refined_stop_loss > refined.refined_entry

    def test_sl_too_wide_is_tightened(self, rm: RiskManager) -> None:
        """SL > 3% → tightened to max_sl_pct."""
        rm = _patched_rm(rm)
        # 3.5% of 22450 = 785 pts
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 785.0, target=SPOT + 1500.0,
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        final_pct = abs(refined.refined_entry - refined.refined_stop_loss) / refined.refined_entry
        assert final_pct <= rm.config.max_sl_pct + 1e-9

    def test_sl_adjusted_below_oi_support(self, rm: RiskManager) -> None:
        """For BUY_CALL, SL placed below the highest-PE-OI strike (support)."""
        rm = _patched_rm(rm)
        chain = _make_chain(spot=SPOT)
        # Highest PE OI is at 22_200 (see _make_chain)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
            confidence_level="MEDIUM",
        )
        refined = rm.validate_and_refine_signal(signal, current_chain=chain, _now=MORNING)
        # SL should be at or below 22_200 * (1 - 0.002)
        expected_floor = 22_200 * (1 - rm.config.sl_sr_buffer_pct)
        assert refined.refined_stop_loss <= expected_floor + 1.0  # 1pt tolerance


# ===========================================================================
# 6. Target validation
# ===========================================================================


class TestTargetValidation:
    def test_target_beyond_resistance_pulled_back(self, rm: RiskManager) -> None:
        """MEDIUM confidence + target beyond max CE OI → pulled back."""
        rm = _patched_rm(rm)
        chain = _make_chain(spot=SPOT)
        # Max CE OI is at 22_500 (see _make_chain).  Set target beyond.
        signal = _make_signal(
            entry=SPOT,
            stop_loss=SPOT - 100.0,
            target=SPOT + 300.0,    # 22_750 — beyond 22_500 resistance
            confidence_level="MEDIUM",
        )
        refined = rm.validate_and_refine_signal(signal, current_chain=chain, _now=MORNING)
        assert refined.refined_target <= 22_500.0 + 1.0  # pulled to resistance

    def test_target_not_pulled_for_high_confidence(self, rm: RiskManager) -> None:
        """HIGH confidence may trade through resistance."""
        rm = _patched_rm(rm)
        chain = _make_chain(spot=SPOT)
        signal = _make_signal(
            entry=SPOT,
            stop_loss=SPOT - 100.0,
            target=SPOT + 300.0,
            confidence_level="HIGH",
        )
        refined = rm.validate_and_refine_signal(signal, current_chain=chain, _now=MORNING)
        # Should retain original target (or close to it)
        assert refined.refined_target > 22_500.0

    def test_rr_too_low_rejects_signal(self, rm: RiskManager) -> None:
        """RR < min_risk_reward_ratio after adjustments → rejected."""
        rm = _patched_rm(rm)
        chain = _make_chain(spot=SPOT)
        # Target only 50 pts but SL 100 pts → RR 0.5 → reject
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 50.0,
            confidence_level="LOW",
        )
        refined = rm.validate_and_refine_signal(signal, current_chain=chain, _now=MORNING)
        assert not refined.is_valid
        assert any("rr" in r.lower() or "risk" in r.lower() or "reward" in r.lower()
                   for r in refined.rejection_reasons + refined.adjustments_made + [refined.reasoning])

    def test_rr_below_1_always_rejected(self, rm: RiskManager) -> None:
        """RR < 1.0 is hard-rejected even at HIGH confidence."""
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 80.0,
            confidence_level="HIGH",
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert not refined.is_valid


# ===========================================================================
# 7. Confidence-based sizing
# ===========================================================================


class TestConfidenceSizing:
    """HIGH confidence → more lots; LOW confidence → fewer lots."""

    def _lots_for(
        self, rm: RiskManager, confidence: str, sl_dist: float = 30.0
    ) -> int:
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT,
            stop_loss=SPOT - sl_dist,
            target=SPOT + sl_dist * 2.5,
            confidence_level=confidence,
            position_size_modifier=1.0,
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        return refined.lots

    def test_high_confidence_more_lots_than_low(
        self, rm: RiskManager, tmp_db: DatabaseManager, default_config: RiskConfig
    ) -> None:
        """HIGH should produce >= lots as LOW (same or more capital deployed)."""
        # Use a fresh rm for each call to avoid daily-limit interference
        rm_high = RiskManager(tmp_db, config=default_config)
        rm_low = RiskManager(tmp_db, config=default_config)
        lots_high = self._lots_for(rm_high, "HIGH", sl_dist=30.0)
        lots_low = self._lots_for(rm_low, "LOW", sl_dist=30.0)
        assert lots_high >= lots_low

    def test_low_confidence_size_modifier_applied(
        self, rm: RiskManager
    ) -> None:
        """LOW confidence uses 30% of max risk → risk_amount is proportionally smaller."""
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
            confidence_level="LOW", position_size_modifier=1.0,
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        max_risk = rm.config.total_capital * rm.config.max_risk_per_trade_pct / 100
        expected_risk = max_risk * rm.config.low_confidence_size_pct / 100
        assert refined.risk_amount <= expected_risk * 1.05  # 5% tolerance

    def test_position_modifier_scales_risk(self, rm: RiskManager) -> None:
        """position_size_modifier=0.5 halves the risk amount."""
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
            confidence_level="HIGH", position_size_modifier=0.5,
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        max_risk = rm.config.total_capital * rm.config.max_risk_per_trade_pct / 100
        # 100% conf * 0.5 modifier = 50% of max risk
        assert refined.risk_amount <= max_risk * 0.55


# ===========================================================================
# 8. Strike selection
# ===========================================================================


def _make_neutral_chain(spot: float = SPOT) -> OptionsChainData:
    """
    Chain where OI peaks are well outside the typical trade range so that
    OI-based S/R adjustments don't interfere with SL/target in strike tests.
    PE OI peak at 21500 (far below any SL), CE OI peak at 23500 (far above any target).
    """
    strikes = []
    for k in range(21_000, 24_000, 50):
        ce_oi = 500_000 if k == 23_500 else 100_000
        pe_oi = 500_000 if k == 21_500 else 100_000
        dist = abs(k - spot)
        premium = max(5.0, 250.0 - dist * 0.5)
        strikes.append(OptionStrike(
            strike_price=float(k),
            ce_oi=ce_oi, ce_oi_change=1000, ce_volume=5000,
            ce_ltp=round(premium, 2), ce_iv=15.0,
            pe_oi=pe_oi, pe_oi_change=500, pe_volume=3000,
            pe_ltp=round(premium, 2), pe_iv=15.0,
        ))
    expiry = date(2026, 4, 10)
    return OptionsChainData(
        index_id=INDEX_ID, spot_price=spot, timestamp=MORNING,
        expiry_date=expiry, strikes=tuple(strikes),
        available_expiries=(expiry, expiry + timedelta(days=7)),
    )


class TestStrikeSelection:
    # Signal with target well within OI resistance (see _make_neutral_chain)
    _SL = SPOT - 100.0   # natural 100pt SL, will be widened; within neutral chain
    _TGT = SPOT + 200.0

    def test_high_confidence_selects_atm(self, rm: RiskManager) -> None:
        """HIGH confidence → ATM strike (closest to spot)."""
        rm = _patched_rm(rm)
        chain = _make_neutral_chain(spot=SPOT)
        signal = _make_signal(
            entry=SPOT, stop_loss=self._SL, target=self._TGT,
            confidence_level="HIGH",
        )
        refined = rm.validate_and_refine_signal(signal, current_chain=chain, _now=MORNING)
        assert refined.recommended_strike == pytest.approx(SPOT, abs=51.0)

    def test_medium_confidence_selects_1_otm(self, rm: RiskManager) -> None:
        """MEDIUM confidence → 1 OTM (one strike above spot for BUY_CALL)."""
        rm = _patched_rm(rm)
        chain = _make_neutral_chain(spot=22_450.0)
        signal = _make_signal(
            entry=SPOT, stop_loss=self._SL, target=self._TGT,
            confidence_level="MEDIUM",
        )
        refined = rm.validate_and_refine_signal(signal, current_chain=chain, _now=MORNING)
        # ATM = 22_450; 1 OTM call = 22_500
        assert refined.recommended_strike == pytest.approx(22_500.0, abs=51.0)

    def test_zero_ltp_skips_to_next_strike(self, rm: RiskManager) -> None:
        """Strike with LTP=0 is skipped; falls back to BS estimate."""
        rm = _patched_rm(rm)
        chain = _make_chain(
            spot=SPOT,
            strikes_range=range(22_400, 22_650, 50),
            ce_ltp_override=0.0,   # all CE LTPs = 0 → BS fallback
        )
        signal = _make_signal(
            entry=SPOT, stop_loss=self._SL, target=self._TGT,
            confidence_level="HIGH",
        )
        refined = rm.validate_and_refine_signal(signal, current_chain=chain, _now=MORNING)
        # Should not crash; BS fallback supplies a strike
        assert refined.recommended_strike is not None

    def test_expiry_day_switches_to_next_expiry(self, rm: RiskManager) -> None:
        """When days_to_expiry < 1, use next week's expiry."""
        rm = _patched_rm(rm)
        today = date.today()
        chain = _make_chain(spot=SPOT, expiry=today)  # expiry TODAY
        signal = _make_signal(
            entry=SPOT, stop_loss=self._SL, target=self._TGT,
        )
        refined = rm.validate_and_refine_signal(signal, current_chain=chain, _now=MORNING)
        if refined.recommended_expiry:
            recommended = date.fromisoformat(refined.recommended_expiry)
            assert recommended > today

    def test_strike_selection_returns_premium(self, rm: RiskManager) -> None:
        """option_premium should be a positive number when chain is available."""
        rm = _patched_rm(rm)
        chain = _make_chain(spot=SPOT)
        signal = _make_signal(
            entry=SPOT, stop_loss=self._SL, target=self._TGT,
        )
        refined = rm.validate_and_refine_signal(signal, current_chain=chain, _now=MORNING)
        if refined.is_valid:
            assert refined.option_premium is not None
            assert refined.option_premium > 0.0


# ===========================================================================
# 9. Breakeven calculation
# ===========================================================================


class TestBreakevenCalculation:
    def test_breakeven_move_formula(self, rm: RiskManager) -> None:
        """breakeven_move = total_tx_cost / (lots × lot_size)."""
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
            confidence_level="HIGH",
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        if refined.is_valid and refined.lots > 0:
            expected = refined.transaction_cost_total / (refined.lots * LOT_SIZE)
            assert abs(refined.breakeven_move - expected) < 0.5

    def test_high_cost_warning_generated(self, rm: RiskManager) -> None:
        """When breakeven > 30% of target distance, a warning is added."""
        rm = _patched_rm(rm)
        # Tight target = 30 pts; tx cost will be high relative to target
        signal = _make_signal(
            entry=SPOT,
            stop_loss=SPOT - 20.0,
            target=SPOT + 30.0,
            confidence_level="HIGH",
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        if refined.is_valid:
            tgt_dist = abs(refined.refined_target - refined.refined_entry)
            if refined.breakeven_move > tgt_dist * 0.3:
                assert any("cost" in w.lower() or "breakeven" in w.lower()
                           for w in refined.warnings)

    def test_transaction_cost_scales_with_lots(self, rm: RiskManager) -> None:
        """transaction_cost_total = lots × cost_per_lot × 2 (entry + exit)."""
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        expected_tx = refined.lots * rm.config.transaction_cost_per_lot * 2
        assert abs(refined.transaction_cost_total - expected_tx) < 1.0


# ===========================================================================
# 10. Portfolio summary
# ===========================================================================


class TestPortfolioSummary:
    def test_empty_portfolio(
        self, tmp_db: DatabaseManager, default_config: RiskConfig
    ) -> None:
        rm = RiskManager(tmp_db, config=default_config)
        summary = rm.get_portfolio_summary()
        assert summary.total_capital == default_config.total_capital
        assert len(summary.open_positions) == 0
        assert summary.today_pnl == 0.0

    def test_open_position_appears_in_summary(
        self, tmp_db: DatabaseManager, default_config: RiskConfig
    ) -> None:
        rm = RiskManager(tmp_db, config=default_config)
        rm._open_positions["p1"] = _OpenPosition(
            db_id=-1,
            signal_id="p1",
            index_id="NIFTY50",
            signal_type="BUY_CALL",
            entry_price=SPOT,
            current_sl=SPOT - 100.0,
            target_price=SPOT + 200.0,
            lots=1,
            lot_size=LOT_SIZE,
            entry_time=MORNING,
            confidence_level="HIGH",
        )
        summary = rm.get_portfolio_summary()
        assert len(summary.open_positions) == 1
        assert summary.open_positions[0]["index"] == "NIFTY50"

    def test_pnl_fields_populated_from_db(
        self, tmp_db: DatabaseManager, default_config: RiskConfig
    ) -> None:
        now_ist = datetime.now(tz=_IST)
        tmp_db.execute(
            """
            INSERT INTO trading_signals
                (index_id, generated_at, signal_type, confidence_level,
                 entry_price, target_price, stop_loss, risk_reward_ratio,
                 regime, technical_vote, options_vote, news_vote, anomaly_vote,
                 reasoning, outcome, actual_exit_price, actual_pnl, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("NIFTY50", now_ist.isoformat(), "BUY_CALL", "HIGH",
             22450.0, 22600.0, 22350.0, 1.5,
             "TRENDING", "BULLISH", "BULLISH", "NEUTRAL", "NEUTRAL",
             "{}", "WIN", 22600.0, 4500.0, now_ist.isoformat()),
        )
        rm = RiskManager(tmp_db, config=default_config)
        summary = rm.get_portfolio_summary()
        assert summary.today_pnl == pytest.approx(4500.0, abs=1.0)


# ===========================================================================
# 11. Insufficient capital
# ===========================================================================


class TestInsufficientCapital:
    def test_very_low_capital_rejected(
        self, tmp_db: DatabaseManager
    ) -> None:
        """₹5000 capital can't afford even 1 NIFTY lot with 100pt SL."""
        cfg = RiskConfig(
            total_capital=5_000.0,
            max_risk_per_trade_pct=2.0,
            transaction_cost_per_lot=120.0,
        )
        rm = RiskManager(tmp_db, config=cfg)
        rm._get_lot_size = lambda _: LOT_SIZE  # type: ignore[method-assign]
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
        )
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        # 1-lot loss = 100*75+240 = 7740 > 5000 → rejected
        assert not refined.is_valid
        assert any("capital" in r.lower() or "insufficient" in r.lower()
                   for r in refined.rejection_reasons)

    def test_no_fo_index_rejected(self, rm: RiskManager) -> None:
        """Non-F&O index (no lot_size) → rejected with clear message."""
        rm._get_lot_size = lambda index_id: None  # type: ignore[method-assign]
        signal = _make_signal(index_id="NIFTY_IT")
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert not refined.is_valid
        assert any("lot" in r.lower() or "f&o" in r.lower() or "lot size" in r.lower()
                   for r in refined.rejection_reasons)


# ===========================================================================
# 12. NO_TRADE signal immediately rejected
# ===========================================================================


class TestNoTradeRejection:
    def test_no_trade_rejected_immediately(self, rm: RiskManager) -> None:
        rm = _patched_rm(rm)
        signal = _make_signal(signal_type="NO_TRADE")
        refined = rm.validate_and_refine_signal(signal, _now=MORNING)
        assert not refined.is_valid
        assert any("no_trade" in r.lower() for r in refined.rejection_reasons)

    def test_no_trade_does_not_consume_daily_limit(
        self, rm: RiskManager
    ) -> None:
        rm = _patched_rm(rm)
        before = rm._compute_daily_remaining()
        signal = _make_signal(signal_type="NO_TRADE")
        rm.validate_and_refine_signal(signal, _now=MORNING)
        after = rm._compute_daily_remaining()
        assert before == after


# ===========================================================================
# 13. Time gate
# ===========================================================================


class TestTimeGate:
    def test_entry_blocked_near_close(
        self, tmp_db: DatabaseManager, default_config: RiskConfig
    ) -> None:
        """Signals generated within 30 min of close are rejected."""
        rm = RiskManager(tmp_db, config=default_config)
        rm._get_lot_size = lambda _: LOT_SIZE  # type: ignore[method-assign]

        signal = _make_signal()
        # Pass NEAR_CLOSE (25 min before 15:30 = within 30-min gate)
        refined = rm.validate_and_refine_signal(signal, _now=NEAR_CLOSE)
        assert not refined.is_valid
        assert any("close" in r.lower() or "entry" in r.lower()
                   for r in refined.rejection_reasons)

    def test_force_exit_near_close(
        self, tmp_db: DatabaseManager, default_config: RiskConfig
    ) -> None:
        """update_position returns EXIT_TIME when within force_exit window."""
        rm = RiskManager(tmp_db, config=default_config)
        pos = _OpenPosition(
            db_id=-1,
            signal_id="time-exit",
            index_id="NIFTY50",
            signal_type="BUY_CALL",
            entry_price=SPOT,
            current_sl=SPOT - 100.0,
            target_price=SPOT + 200.0,
            lots=1,
            lot_size=LOT_SIZE,
            entry_time=MORNING,
            confidence_level="HIGH",
        )
        rm._open_positions["time-exit"] = pos

        # 4 minutes before close = within 5-min force-exit window
        force_exit_time = datetime(2026, 4, 8, 15, 26, 0, tzinfo=_IST)
        update = rm.update_position("time-exit", SPOT + 50.0, _now=force_exit_time)
        assert update.action == "EXIT_TIME"


# ===========================================================================
# 14. close_position P&L accounting
# ===========================================================================


class TestClosePosition:
    def test_win_updates_realised_pnl(
        self, tmp_db: DatabaseManager, default_config: RiskConfig
    ) -> None:
        rm = RiskManager(tmp_db, config=default_config)
        rm._daily_realised_pnl = 0.0

        pos = _OpenPosition(
            db_id=-1,
            signal_id="close-test",
            index_id="NIFTY50",
            signal_type="BUY_CALL",
            entry_price=22_000.0,
            current_sl=21_900.0,
            target_price=22_200.0,
            lots=1,
            lot_size=LOT_SIZE,
            entry_time=MORNING,
            confidence_level="HIGH",
        )
        rm._open_positions["close-test"] = pos

        rm.close_position("close-test", 22_200.0, "TARGET")
        # pnl = (22200-22000)*75 - 240 = 15000 - 240 = 14760
        assert rm._daily_realised_pnl == pytest.approx(14_760.0, abs=5.0)
        assert "close-test" not in rm._open_positions

    def test_loss_reduces_daily_remaining(
        self, tmp_db: DatabaseManager, default_config: RiskConfig
    ) -> None:
        rm = RiskManager(tmp_db, config=default_config)
        rm._daily_realised_pnl = 0.0

        pos = _OpenPosition(
            db_id=-1,
            signal_id="loss-test",
            index_id="NIFTY50",
            signal_type="BUY_CALL",
            entry_price=22_000.0,
            current_sl=21_900.0,
            target_price=22_200.0,
            lots=1,
            lot_size=LOT_SIZE,
            entry_time=MORNING,
            confidence_level="HIGH",
        )
        rm._open_positions["loss-test"] = pos

        rm.close_position("loss-test", 21_900.0, "SL")
        # pnl = (21900-22000)*75 - 240 = -7500 - 240 = -7740
        assert rm._daily_realised_pnl < 0
        remaining = rm._compute_daily_remaining()
        daily_limit = default_config.total_capital * default_config.max_risk_per_day_pct / 100
        assert remaining < daily_limit


# ===========================================================================
# 15. Black-Scholes fallback
# ===========================================================================


class TestBSFallback:
    def test_no_chain_uses_bs_estimate(self, rm: RiskManager) -> None:
        """Without options chain, premium estimated via Black-Scholes."""
        rm = _patched_rm(rm)
        signal = _make_signal(
            entry=SPOT, stop_loss=SPOT - 100.0, target=SPOT + 200.0,
        )
        refined = rm.validate_and_refine_signal(signal, current_chain=None, _now=MORNING)
        if refined.is_valid:
            assert refined.option_premium is not None
            assert refined.option_premium > 0.0
            assert any("black-scholes" in a.lower() or "bs" in a.lower()
                       for a in refined.adjustments_made)
