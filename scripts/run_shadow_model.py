"""
Run a candidate decision model in SHADOW mode on production data.

Each cycle snapshots the production DB and runs the candidate (a pinned git
worktree) against the snapshot in a subprocess. Results go to the shadow DB,
together with the live engine's decision. Nothing is written to production,
no Telegram message is sent, and no trade is placed.

    .venv/bin/python scripts/run_shadow_model.py --model-root ../shadow-phase2 \
        --model-branch feat/model-phase2 --prod-db /path/to/trading_dss.db [--once] [--interval 300]

By default it loops during market hours only (09:15–15:30 IST, trading days).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_IST = ZoneInfo("Asia/Kolkata")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-root", type=Path, required=True, help="worktree of the candidate model commit")
    ap.add_argument("--model-branch", default=None)
    ap.add_argument("--prod-db", type=Path, default=None, help="production DB (default: DB_PATH / settings)")
    ap.add_argument("--shadow-db", type=Path, default=None)
    ap.add_argument("--interval", type=int, default=300, help="seconds between cycles")
    ap.add_argument("--indices", default="", help="comma list (default: model's trading universe)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--always", action="store_true", help="ignore market hours (testing)")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    if args.prod_db:
        os.environ["DB_PATH"] = str(args.prod_db.resolve())
    from config.settings import settings
    from src.shadow.runner import ShadowRunner
    from src.shadow.store import DEFAULT_SHADOW_DB, ShadowStore
    from src.utils.market_hours import is_market_open

    prod_db = Path(settings.database.path).resolve()
    shadow_db = (args.shadow_db or DEFAULT_SHADOW_DB).resolve()
    shadow_dir = shadow_db.parent
    shadow_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(shadow_dir / "runner.log"), logging.StreamHandler()],
    )
    log = logging.getLogger("shadow")
    if not prod_db.exists():
        log.error("production DB not found: %s", prod_db)
        return 2

    runner = ShadowRunner(
        prod_db=prod_db, model_root=args.model_root, store=ShadowStore(shadow_db),
        work_dir=shadow_dir / "work", model_branch=args.model_branch, live_root=ROOT,
        python=args.python, indices=[i for i in args.indices.split(",") if i],
    )
    log.info("Shadow model %s (%s) | prod DB %s (read-only snapshots) | shadow DB %s",
             runner.model_commit, args.model_branch, prod_db, shadow_db)
    if args.once:
        runner.run_once()
        return 0

    (shadow_dir / "runner.pid").write_text(str(os.getpid()))
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    try:
        while not stop["flag"]:
            now = datetime.now(tz=_IST)
            if args.always or is_market_open(now):
                runner.run_once()
            # sleep to the next interval boundary, waking early on stop
            nxt = now + timedelta(seconds=args.interval - (now.timestamp() % args.interval))
            while not stop["flag"] and datetime.now(tz=_IST) < nxt:
                time.sleep(1)
    finally:
        (shadow_dir / "runner.pid").unlink(missing_ok=True)
        log.info("Shadow runner stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
