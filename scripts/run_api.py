"""
Start the FastAPI server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from config.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the trading_dss FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default 8000)")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only)",
    )
    args = parser.parse_args()

    # ── PID Lock ─────────────────────────────────────────────────────────────
    import os
    pid_path = Path("data/trading_dss.pid")
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
            os.kill(old_pid, 0)
            print(f"\n  ❌ ERROR: Another instance is already running (PID {old_pid})")
            sys.exit(1)
        except (OSError, ValueError):
            pass
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))

    try:
        uvicorn.run(
            "src.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=settings.logging.level.lower(),
        )
    finally:
        if pid_path.exists():
            pid_path.unlink()


if __name__ == "__main__":
    main()
