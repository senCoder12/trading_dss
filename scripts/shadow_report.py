"""
Print shadow-model decisions next to the live engine's decisions.

    .venv/bin/python scripts/shadow_report.py [--shadow-db PATH] [--runs 3] [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shadow.store import DEFAULT_SHADOW_DB, open_readonly  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-db", type=Path, default=DEFAULT_SHADOW_DB)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    conn = open_readonly(args.shadow_db)
    if conn is None:
        print(f"No shadow DB at {args.shadow_db}")
        return 1

    runs = conn.execute(
        "SELECT * FROM shadow_runs WHERE (? IS NULL OR substr(as_of,1,10) = ?) ORDER BY run_id DESC LIMIT ?",
        (args.date, args.date, args.runs)).fetchall()
    for run in reversed(runs):
        print(f"\n=== SHADOW run {run['run_id']} @ {run['as_of'][:19]} | model {run['model_commit']} "
              f"({run['model_branch']}) | live {run['live_commit']} | {run['status']} {run['duration_ms']} ms"
              + (f" | ERROR {run['error'][:200]}" if run["error"] else ""))
        rows = conn.execute("SELECT * FROM shadow_decisions WHERE run_id = ? ORDER BY index_id",
                            (run["run_id"],)).fetchall()
        print(f"  {'index':<11} {'LIVE':<8} {'live@':<6} {'SHADOW':<8} {'P(win)':>6} {'conf':<6} "
              f"{'regime':<16} {'agree':<5} reason")
        for r in rows:
            reason = (r["reasoning"] or "").strip().splitlines()[0][:90] if r["reasoning"] else ""
            live_at = (r["live_generated_at"] or "")[11:16]
            agree = {1: "yes", 0: "NO", None: "-"}[r["agree"]]
            p = f"{r['confidence_score']:.2f}" if r["confidence_score"] is not None else "-"
            print(f"  {r['index_id']:<11} {r['live_decision'] or '-':<8} {live_at:<6} "
                  f"{r['decision'] + ('*' if r['forced_neutral'] else ''):<8} {p:>6} "
                  f"{r['confidence_level'] or '-':<6} {r['regime'] or '-':<16} {agree:<5} {reason}")
            comps = json.loads(r["components_json"] or "{}")
            if comps:
                parts = []
                for name, c in comps.items():
                    if isinstance(c, dict):
                        contrib = c.get("weighted_contribution", c.get("score", c.get("vote")))
                        if isinstance(contrib, (int, float)):
                            parts.append(f"{name}={contrib:+.2f}")
                        else:
                            vote = c.get("vote")
                            score = c.get("score")
                            parts.append(f"{name}={vote or contrib}" + (f"({score:+.1f})" if isinstance(score, (int, float)) else ""))
                print(f"  {'':<11} components: {', '.join(parts)}")
            for w in json.loads(r["warnings_json"] or "[]")[:4]:
                print(f"  {'':<11} ⚠ {w[:140]}")
    stats = conn.execute(
        "SELECT COUNT(*) n, SUM(agree) agree, SUM(agree IS NOT NULL) compared, "
        "SUM(decision != 'NEUTRAL') shadow_dir, SUM(live_decision != 'NEUTRAL') live_dir, "
        "SUM(forced_neutral) forced FROM shadow_decisions WHERE (? IS NULL OR substr(as_of,1,10) = ?)",
        (args.date, args.date)).fetchone()
    print(f"\nTotals: {stats['n']} shadow decisions, compared {stats['compared']}, agree {stats['agree']}, "
          f"shadow directional {stats['shadow_dir']}, live directional {stats['live_dir']}, forced neutral {stats['forced']}")
    print("(* = forced NEUTRAL because of stale/missing inputs. SHADOW decisions are never traded.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
