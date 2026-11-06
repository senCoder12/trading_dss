"""
Calibrated signal confidence.

Why
---
The old "confidence" was ``|score| / 2`` plus a dozen hand-set boosts and
penalties.  It was never compared with outcomes, could not reach its own
0.45 gate (live median 0.18, max 0.23 — 625 signals, zero trades), and its
numbers meant nothing as probabilities.

What this module does
---------------------
``scripts/calibrate_signal_model.py`` replays the live scoring function over
historical bars, labels each bar with what happened next (did price reach a
target before a stop within the horizon?), and fits a **monotone** mapping
from |score| to the empirical win probability:

* bins of |score| with Beta(2, 2) shrinkage toward 50 % (small bins cannot
  claim extreme probabilities),
* pooled-adjacent-violators so a stronger score never maps to a lower
  probability,
* a chronological train/test split — the out-of-sample (OOS) results are
  stored alongside and decide whether the model is ``validated``.

At runtime :class:`SignalCalibration` turns a score into P(win).  If the
file is missing, stale, or the model failed out of sample, the probability
is ``None`` and the signal generator does not trade — no measured edge, no
risk.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "signal_calibration.json"
CALIBRATION_VERSION = 1
POOLED_KEY = "ALL"

# Beta prior pseudo-counts for shrinkage toward 50 %.
_PRIOR_WINS = 2.0
_PRIOR_LOSSES = 2.0


# ---------------------------------------------------------------------------
# Fitting helpers (used by the calibration script)
# ---------------------------------------------------------------------------


@dataclass
class Bin:
    lo: float
    hi: float
    n: int
    wins: int
    p_win: float  # shrunk + monotone


def _pav(values: list[float], weights: list[float]) -> list[float]:
    """Pool-adjacent-violators: non-decreasing fit, weighted."""
    blocks: list[list[float]] = []  # [value, weight, count]
    for v, w in zip(values, weights):
        blocks.append([v, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            wt = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / wt if wt else (v1 + v2) / 2, wt, c1 + c2])
    out: list[float] = []
    for v, _, c in blocks:
        out.extend([v] * c)
    return out


def fit_bins(
    scores: Sequence[float],
    wins: Sequence[int],
    edges: Sequence[float],
) -> list[Bin]:
    """Fit monotone, shrunk win probabilities over |score| bins."""
    raw: list[tuple[float, float, int, int]] = []
    for lo, hi in zip(edges, edges[1:]):
        idx = [i for i, s in enumerate(scores) if lo <= s < hi]
        n = len(idx)
        w = sum(int(wins[i]) for i in idx)
        raw.append((lo, hi, n, w))
    shrunk = [(w + _PRIOR_WINS) / (n + _PRIOR_WINS + _PRIOR_LOSSES) for _, _, n, w in raw]
    weights = [n + _PRIOR_WINS + _PRIOR_LOSSES for _, _, n, _ in raw]
    mono = _pav(shrunk, weights)
    return [Bin(lo, hi, n, w, round(p, 4)) for (lo, hi, n, w), p in zip(raw, mono)]


def brier(probs: Sequence[float], wins: Sequence[int]) -> float:
    if not probs:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probs, wins)) / len(probs)


# ---------------------------------------------------------------------------
# Runtime lookup
# ---------------------------------------------------------------------------


@dataclass
class _Model:
    bins: list[Bin]
    validated: bool
    meta: dict = field(default_factory=dict)

    def probability(self, abs_score: float) -> Optional[float]:
        for b in self.bins:
            if b.lo <= abs_score < b.hi:
                return b.p_win
        if self.bins and abs_score >= self.bins[-1].hi:
            return self.bins[-1].p_win
        return None


class SignalCalibration:
    """Loads ``config/signal_calibration.json`` and maps score → P(win)."""

    def __init__(self, path: Optional[Path] = None, max_age_days: int = 120) -> None:
        self._path = Path(path) if path else DEFAULT_PATH
        self._max_age_days = max_age_days
        self._lock = threading.Lock()
        self._models: dict[str, _Model] = {}
        self._status = "not loaded"
        self._mtime: Optional[float] = None
        self._loaded = False

    @property
    def status(self) -> str:
        self._ensure_loaded()
        return self._status

    def _ensure_loaded(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            mtime = None
        with self._lock:
            if self._loaded and mtime == self._mtime:
                return
            self._mtime = mtime
            self._loaded = True
            self._models = {}
            if mtime is None:
                self._status = f"no calibration file ({self._path.name}) — run scripts/calibrate_signal_model.py"
                return
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                if payload.get("version") != CALIBRATION_VERSION:
                    self._status = "calibration file version mismatch — recalibrate"
                    return
                generated = datetime.fromisoformat(payload["generated_at"])
                age = (datetime.now(generated.tzinfo) - generated).days
                if age > self._max_age_days:
                    self._status = f"calibration is {age} days old — recalibrate"
                    return
                for key, m in payload.get("models", {}).items():
                    bins = [Bin(**b) for b in m["bins"]]
                    self._models[key.upper()] = _Model(bins, bool(m.get("validated")), m.get("meta", {}))
                self._status = "ok" if self._models else "calibration file has no models"
            except Exception as exc:  # malformed file must never crash the engine
                logger.exception("SignalCalibration: could not load %s", self._path)
                self._status = f"calibration file unreadable: {exc}"
                self._models = {}

    def lookup(self, index_id: str, abs_score: float) -> tuple[Optional[float], str]:
        """Return ``(P(win) | None, reason)`` for a directional score."""
        self._ensure_loaded()
        if not self._models:
            return None, self._status
        model = self._models.get(index_id.upper()) or self._models.get(POOLED_KEY)
        if model is None:
            return None, f"no calibration for {index_id}"
        if not model.validated:
            return None, (
                f"calibration for {index_id} failed out-of-sample validation "
                f"({model.meta.get('validation_note', 'no edge')}) — not trading"
            )
        p = model.probability(abs_score)
        if p is None:
            return None, f"score {abs_score:.2f} outside calibrated range"
        return p, "ok"


_singleton: Optional[SignalCalibration] = None
_singleton_lock = threading.Lock()


def get_signal_calibration() -> SignalCalibration:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = SignalCalibration()
        return _singleton


def reset_signal_calibration(path: Optional[Path] = None) -> SignalCalibration:
    """Replace the singleton (tests / after recalibration)."""
    global _singleton
    with _singleton_lock:
        _singleton = SignalCalibration(path)
        return _singleton
