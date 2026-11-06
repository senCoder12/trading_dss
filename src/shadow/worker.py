"""
Shadow worker: runs INSIDE the candidate model's worktree process.

It is launched by ShadowRunner with:
    cwd=<model worktree>, PYTHONPATH=<model worktree>, DB_PATH=<snapshot copy>
so every ``src.*`` import resolves to the candidate's code and every DB access
goes to the snapshot. It must therefore import nothing from the live
worktree, only the stdlib and the candidate's own ``src``.

    python -c "import runpy,sys; sys.argv=['w', OUT, AS_OF, INDICES]; runpy.run_path(WORKER, run_name='__main__')"
"""

import json
import os
import sys
import traceback
from datetime import datetime


def _g(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default


def _serialize(res) -> dict:
    s = res.signal
    tech = res.technical_result
    refined = None
    if res.is_actionable:
        refined = {k: _g(s, k) for k in (
            "recommended_strike", "recommended_expiry", "option_type", "entry_premium",
            "premium_stop", "premium_target", "lots", "max_loss_amount", "rejection_reasons",
        )}
    return {
        "index_id": res.index_id,
        "signal_type": _g(s, "signal_type"),
        "is_actionable": bool(res.is_actionable),
        "confidence_score": _g(s, "confidence_score"),
        "confidence_level": _g(s, "confidence_level"),
        "weighted_score": _g(s, "weighted_score"),
        "vote_breakdown": _g(s, "vote_breakdown", {}) or {},
        "reasoning": _g(s, "reasoning", "") or "",
        "signal_warnings": list(_g(s, "warnings", []) or []),
        "result_warnings": list(res.warnings or []),
        "regime": _g(res.regime, "regime"),
        "regime_warnings": list(_g(res.regime, "warnings", []) or []),
        "unavailable_components": list(_g(tech, "unavailable_components", []) or []),
        "volume_available": _g(tech, "volume_available"),
        "refined": refined,
    }


def main(out_path: str, as_of_iso: str, indices_csv: str) -> int:
    as_of = datetime.fromisoformat(as_of_iso)
    payload = {"results": [], "db_path": os.environ.get("DB_PATH"), "cwd": os.getcwd()}
    try:
        from config.settings import settings
        from src.database.db_manager import DatabaseManager
        from src.engine.decision_engine import DecisionEngine

        from pathlib import Path

        if Path(settings.database.path).resolve() != Path(os.environ.get("DB_PATH", "")).resolve():
            raise RuntimeError(f"refusing to run: DB is {settings.database.path}, expected snapshot {os.environ.get('DB_PATH')}")
        db = DatabaseManager()
        db.connect()
        eng = DecisionEngine(db)
        indices = [i for i in indices_csv.split(",") if i] or [
            i.id for i in eng.registry.get_trading_universe(
                major_only=settings.trading.restrict_to_major_indices)
        ]
        for idx in indices:
            try:
                payload["results"].append(_serialize(eng.run_full_cycle(idx, as_of=as_of)))
            except Exception as exc:  # one index failing must not hide the others
                payload["results"].append({
                    "index_id": idx, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-3000:],
                })
    except Exception as exc:
        payload["fatal"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()[-3000:]
    with open(out_path, "w") as fh:
        json.dump(payload, fh, default=str)
    return 0 if "fatal" not in payload else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))
