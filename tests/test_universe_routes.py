"""Unit tests for pure helpers in src/api/routes/universe.py."""

from __future__ import annotations

from src.api.routes.universe import OptionStrike, _max_pain


def _strike(k: float, ce_oi: int = 0, pe_oi: int = 0) -> OptionStrike:
    return OptionStrike(
        strike=k, ce_oi=ce_oi, ce_oi_change=0, ce_volume=0, ce_iv=0.0, ce_ltp=0.0,
        pe_oi=pe_oi, pe_oi_change=0, pe_volume=0, pe_iv=0.0, pe_ltp=0.0,
    )


def test_max_pain_minimises_writer_payout() -> None:
    # Heavy CE OI at 110, heavy PE OI at 90 -> pain lowest at 100
    strikes = [_strike(90, pe_oi=1000), _strike(100, 100, 100), _strike(110, ce_oi=1000)]
    assert _max_pain(strikes) == 100


def test_max_pain_skewed_put_writing() -> None:
    # Only puts written at 120 -> expiry at/above 120 pays nothing
    strikes = [_strike(100), _strike(110), _strike(120, pe_oi=500)]
    assert _max_pain(strikes) == 120


def test_max_pain_no_oi_returns_none() -> None:
    assert _max_pain([_strike(100), _strike(110)]) is None
    assert _max_pain([]) is None
