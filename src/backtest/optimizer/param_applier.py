"""
Approved Parameter Manager — Step 7.4 of the Trading Decision Support System.

After optimisation and robustness testing approve a parameter set, this module
saves the approved parameters and makes them available to the live Decision
Engine.  It also provides expiry tracking and the ability to apply approved
parameters directly to a ``BacktestConfig``.

Usage
-----
::

    mgr = ApprovedParameterManager()
    mgr.save_approved_params("NIFTY50", params, metrics, 0.68, "a3f2b1c8")
    params = mgr.get_approved_params("NIFTY50")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.backtest.metrics import BacktestMetrics
from src.backtest.optimizer.param_space import ParameterApplicator
from src.backtest.strategy_runner import BacktestConfig
from src.utils.date_utils import get_ist_now

logger = logging.getLogger(__name__)

_DEFAULT_APPROVED_PARAMS_FILE = Path("config/approved_params.json")


class ApprovedParameterManager:
    """Manages approved parameters and applies them to the live system configuration.

    After optimization and robustness testing approve a parameter set,
    this class saves them and makes them available to the live Decision Engine.
    """

    def __init__(self, filepath: str | Path | None = None) -> None:
        self.filepath = Path(filepath) if filepath else _DEFAULT_APPROVED_PARAMS_FILE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_approved_params(
        self,
        index_id: str,
        params: dict[str, Any],
        metrics: BacktestMetrics,
        robustness_score: float,
        config_hash: str,
        notes: str = "",
    ) -> None:
        """Save approved parameters for an index.

        Stores the parameters, approval timestamp, justification metrics,
        robustness score, and config hash for traceability.
        """
        existing = self._load_all()

        existing[index_id] = {
            "params": params,
            "approved_at": get_ist_now().isoformat(),
            "metrics_at_approval": {
                "return_pct": metrics.total_return_pct,
                "sharpe": metrics.sharpe_ratio,
                "win_rate": metrics.win_rate,
                "max_drawdown": metrics.max_drawdown_pct,
                "profit_factor": metrics.profit_factor,
                "total_trades": metrics.total_trades,
            },
            "robustness_score": robustness_score,
            "config_hash": config_hash,
            "notes": notes,
            "status": "ACTIVE",
        }

        self._save_all(existing)
        logger.info("Approved params saved for %s: %s", index_id, params)

    def get_approved_params(self, index_id: str) -> dict[str, Any] | None:
        """Get the currently approved parameters for an index.

        Returns ``None`` if no active params exist. Logs a warning if the
        approved params are older than 90 days.
        """
        all_params = self._load_all()
        entry = all_params.get(index_id)

        if entry is None:
            return None

        if entry.get("status") != "ACTIVE":
            return None

        # Check age — params older than 90 days should be re-optimized
        try:
            approved_at = datetime.fromisoformat(entry["approved_at"])
            age_days = (get_ist_now() - approved_at).days
            if age_days > 90:
                logger.warning(
                    "Approved params for %s are %d days old. "
                    "Re-optimization recommended (> 90 days).",
                    index_id,
                    age_days,
                )
        except (KeyError, ValueError):
            pass  # If timestamp is missing or invalid, just return the params

        return entry["params"]

    def expire_params(self, index_id: str, reason: str) -> None:
        """Mark parameters as expired.

        Typically called when a shadow tracker detects edge decay.
        """
        all_params = self._load_all()
        if index_id in all_params:
            all_params[index_id]["status"] = "EXPIRED"
            all_params[index_id]["expired_at"] = get_ist_now().isoformat()
            all_params[index_id]["expire_reason"] = reason
            self._save_all(all_params)
            logger.info(
                "Params for %s expired: %s", index_id, reason,
            )
        else:
            logger.warning(
                "Cannot expire params for %s: no entry found.", index_id,
            )

    def list_all_approved(self) -> dict[str, Any]:
        """List all approved parameter sets with their status."""
        return self._load_all()

    def apply_to_backtest_config(
        self,
        index_id: str,
        base_config: BacktestConfig,
    ) -> BacktestConfig | None:
        """Apply approved params to a backtest config.

        Returns ``None`` if no approved params exist for the index or if the
        resulting configuration is invalid.
        """
        params = self.get_approved_params(index_id)
        if params is None:
            return None

        applicator = ParameterApplicator()
        return applicator.apply(base_config, params)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> dict[str, Any]:
        """Load all entries from the approved-params JSON file."""
        if not self.filepath.exists():
            return {}
        try:
            text = self.filepath.read_text(encoding="utf-8")
            if not text.strip():
                return {}
            return json.loads(text)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load approved params from %s: %s", self.filepath, exc)
            return {}

    def _save_all(self, data: dict[str, Any]) -> None:
        """Persist all entries to the approved-params JSON file."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.filepath.write_text(
            json.dumps(data, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
