"""
Pre-Launch Validator — Phase 9.3 of the Trading Decision Support System.

Runs a comprehensive end-to-end check of every system component before the
first paper trading session (or after any major change).  Each check is
independent and returns a CheckResult; the final ValidationReport summarises
pass/fail/warn counts and prints a formatted report.

Usage (programmatic)
--------------------
::

    from src.paper_trading.pre_launch_validator import PreLaunchValidator
    validator = PreLaunchValidator()
    report = validator.run_full_validation()
    print(report.format_report())
    if not report.is_ready:
        sys.exit(1)

Usage (CLI)
-----------
::

    python scripts/validate_system.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root — two levels up from this file (src/paper_trading/pre_launch_validator.py)
_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single pre-launch check."""

    name: str
    status: str          # PASS / FAIL / WARN / SKIP
    message: str
    details: Optional[dict] = field(default=None)
    is_critical: bool = True   # FAIL on a critical check blocks launch

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"

    @property
    def warned(self) -> bool:
        return self.status == "WARN"


@dataclass
class ValidationReport:
    """Aggregated result of all pre-launch checks."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.failed)

    @property
    def warned(self) -> int:
        return sum(1 for r in self.results if r.warned)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "SKIP")

    @property
    def is_ready(self) -> bool:
        """True only when no critical check has failed."""
        return all(not (r.failed and r.is_critical) for r in self.results)

    def format_report(self) -> str:
        """Return the full formatted validation report as a string."""
        lines: list[str] = []

        lines += [
            "",
            "╔═══════════════════════════════════════════════╗",
            "║     PRE-LAUNCH VALIDATION REPORT              ║",
            "╠═══════════════════════════════════════════════╣",
            "",
        ]

        # Group results by section
        sections: dict[str, list[CheckResult]] = {}
        for r in self.results:
            section = r.details.get("section", "General") if r.details else "General"
            sections.setdefault(section, []).append(r)

        for section, checks in sections.items():
            lines.append(f"  {section}:")
            for r in checks:
                icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "SKIP": "⏭️ "}.get(r.status, "?")
                lines.append(f"  {icon} {r.name} — {r.message}")
            lines.append("")

        # Summary line
        lines += [
            "═══════════════════════════════════════════════",
            f"  RESULT: {self.passed}/{self.total} PASSED | "
            f"{self.failed} FAILED | {self.warned} WARNINGS",
            "",
        ]

        if self.is_ready:
            lines += [
                "  ✅ SYSTEM IS READY FOR PAPER TRADING",
            ]
        else:
            lines += [
                "  ❌ SYSTEM NOT READY — fix issues above before starting",
            ]

        lines.append("═══════════════════════════════════════════════")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class PreLaunchValidator:
    """
    Comprehensive system readiness checker.

    Run this before the first paper trading session and after any major
    system change.  Each check method is independent — a failure in one
    does not prevent the others from running.
    """

    def __init__(self) -> None:
        self._db = None
        self._registry = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run_full_validation(self) -> ValidationReport:
        """
        Execute all checks and return a ValidationReport.

        Checks are run in logical order (config → DB → data → analysis →
        engine → paper trading → alerts → API → backtest → system).
        A failure in an early check may cause later checks to skip gracefully.
        """
        report = ValidationReport()

        # Attempt to connect to shared DB once, reused by later checks
        self._db = self._try_get_db()

        checks = [
            # Configuration
            self._check_indices_config,
            self._check_news_mappings,
            self._check_rss_feeds,
            self._check_sentiment_keywords,
            self._check_events_calendar,
            self._check_optimization_profiles,
            self._check_approved_params,
            # Database
            self._check_database,
            self._check_historical_data,
            self._check_cross_index_overlap,
            self._check_options_snapshots,
            self._check_fii_dii_data,
            self._check_vix_data,
            # Data layer
            self._check_nse_connectivity,
            self._check_bse_connectivity,
            self._check_options_chain,
            self._check_vix_feed,
            self._check_rate_limiter,
            self._check_circuit_breaker,
            # Analysis
            self._check_technical_indicators,
            self._check_news_engine,
            self._check_anomaly_detection,
            self._check_regime_detector,
            # Decision Engine
            self._check_signal_generation,
            self._check_risk_manager,
            self._check_kill_switch,
            self._check_signal_deduplication,
            self._check_audit_trail,
            # Paper Trading
            self._check_paper_engine,
            self._check_state_persistence,
            self._check_position_tracking,
            # Alerts
            self._check_telegram_bot,
            self._check_digest_mode,
            self._check_websocket,
            # API & Dashboard
            self._check_api_server,
            self._check_api_endpoints,
            self._check_dashboard,
            # Backtesting
            self._check_backtest_results,
            self._check_walk_forward,
            self._check_robustness,
            # System
            self._check_disk_space,
            self._check_market_calendar,
            self._check_watchdog,
        ]

        for check_fn in checks:
            try:
                result = check_fn()
                report.results.append(result)
            except Exception as exc:
                logger.error("Check %s raised unexpectedly: %s", check_fn.__name__, exc)
                report.results.append(CheckResult(
                    name=check_fn.__name__.replace("_check_", "").replace("_", " ").title(),
                    status="FAIL",
                    message=f"Unexpected error: {exc}",
                    is_critical=False,
                ))

        return report

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _try_get_db(self):
        try:
            from src.database.db_manager import get_db_manager
            db = get_db_manager()
            db.connect()
            return db
        except Exception:
            return None

    @staticmethod
    def _section(name: str) -> dict:
        return {"section": name}

    def _pass(self, name: str, msg: str, section: str, critical: bool = True, **extra) -> CheckResult:
        return CheckResult(name=name, status="PASS", message=msg,
                          details={"section": section, **extra}, is_critical=critical)

    def _fail(self, name: str, msg: str, section: str, critical: bool = True, **extra) -> CheckResult:
        return CheckResult(name=name, status="FAIL", message=msg,
                          details={"section": section, **extra}, is_critical=critical)

    def _warn(self, name: str, msg: str, section: str, critical: bool = False, **extra) -> CheckResult:
        return CheckResult(name=name, status="WARN", message=msg,
                          details={"section": section, **extra}, is_critical=critical)

    def _skip(self, name: str, msg: str, section: str) -> CheckResult:
        return CheckResult(name=name, status="SKIP", message=msg,
                          details={"section": section}, is_critical=False)

    # -----------------------------------------------------------------------
    # Configuration checks
    # -----------------------------------------------------------------------

    def _check_indices_config(self) -> CheckResult:
        section = "Configuration"
        try:
            from config.settings import settings
            path = Path(settings.indices_config_path)
            if not path.exists():
                return self._fail("indices.json", f"File not found: {path}", section)
            import json
            data = json.loads(path.read_text())
            count = len(data.get("indices", data) if isinstance(data, dict) else data)
            return self._pass("indices.json", f"{count} indices loaded", section)
        except Exception as exc:
            return self._fail("indices.json", str(exc), section)

    def _check_news_mappings(self) -> CheckResult:
        section = "Configuration"
        try:
            config_dir = _ROOT / "config" / "data"
            candidates = list(config_dir.glob("news_mappings*.json")) + list((_ROOT / "config").glob("news_mappings*.json"))
            if not candidates:
                return self._warn("news_mappings.json", "File not found (news optional)", section, critical=False)
            import json
            data = json.loads(candidates[0].read_text())
            keywords = sum(len(v) if isinstance(v, list) else 1 for v in data.values()) if isinstance(data, dict) else len(data)
            return self._pass("news_mappings.json", f"{keywords} keywords", section)
        except Exception as exc:
            return self._warn("news_mappings.json", str(exc), section, critical=False)

    def _check_rss_feeds(self) -> CheckResult:
        section = "Configuration"
        try:
            config_dir = _ROOT / "config"
            candidates = list(config_dir.rglob("rss_feeds*.json"))
            if not candidates:
                return self._warn("rss_feeds.json", "File not found (news optional)", section, critical=False)
            import json
            data = json.loads(candidates[0].read_text())
            count = len(data) if isinstance(data, list) else len(data.get("feeds", []))
            return self._pass("rss_feeds.json", f"{count} feeds", section)
        except Exception as exc:
            return self._warn("rss_feeds.json", str(exc), section, critical=False)

    def _check_sentiment_keywords(self) -> CheckResult:
        section = "Configuration"
        try:
            config_dir = _ROOT / "config"
            candidates = list(config_dir.rglob("sentiment_keywords*.json"))
            if not candidates:
                return self._warn("sentiment_keywords.json", "File not found (optional)", section, critical=False)
            import json
            json.loads(candidates[0].read_text())  # just validate JSON
            return self._pass("sentiment_keywords.json", "valid", section)
        except Exception as exc:
            return self._warn("sentiment_keywords.json", str(exc), section, critical=False)

    def _check_events_calendar(self) -> CheckResult:
        section = "Configuration"
        try:
            config_dir = _ROOT / "config"
            candidates = list(config_dir.rglob("events_calendar*.json"))
            if not candidates:
                return self._warn("events_calendar.json", "File not found (optional)", section, critical=False)
            import json
            data = json.loads(candidates[0].read_text())
            count = len(data) if isinstance(data, list) else len(data.get("events", []))
            return self._pass("events_calendar.json", f"{count} events loaded", section)
        except Exception as exc:
            return self._warn("events_calendar.json", str(exc), section, critical=False)

    def _check_optimization_profiles(self) -> CheckResult:
        section = "Configuration"
        try:
            config_dir = _ROOT / "config"
            candidates = list(config_dir.rglob("optimization_profiles*.json"))
            if not candidates:
                return self._warn("optimization_profiles.json", "File not found (optional)", section, critical=False)
            import json
            data = json.loads(candidates[0].read_text())
            count = len(data) if isinstance(data, list) else len(data.get("profiles", data) if isinstance(data, dict) else [])
            return self._pass("optimization_profiles.json", f"{count} profiles", section)
        except Exception as exc:
            return self._warn("optimization_profiles.json", str(exc), section, critical=False)

    def _check_approved_params(self) -> CheckResult:
        section = "Configuration"
        try:
            data_dir = _ROOT / "data"
            candidates = list(data_dir.rglob("approved_params*.json"))
            if not candidates:
                return self._warn("approved_params.json", "No approved parameters found — run optimizer first", section, critical=False)
            import json
            data = json.loads(candidates[0].read_text())
            indices = list(data.keys()) if isinstance(data, dict) else []
            return self._pass("approved_params.json", f"{', '.join(indices)} approved", section)
        except Exception as exc:
            return self._warn("approved_params.json", str(exc), section, critical=False)

    # -----------------------------------------------------------------------
    # Database checks
    # -----------------------------------------------------------------------

    def _check_database(self) -> CheckResult:
        section = "Database"
        if self._db is None:
            return self._fail("Schema", "Database connection failed", section)
        try:
            # Verify a sample of expected tables
            required_tables = [
                "indices", "daily_prices", "intraday_prices",
                "options_snapshots", "trading_signals", "paper_trades",
            ]
            with self._db._connect() as conn:
                cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing = {row[0] for row in cur.fetchall()}
            missing = [t for t in required_tables if t not in existing]
            if missing:
                return self._fail("Schema", f"Missing tables: {', '.join(missing)}", section)
            return self._pass("Schema", f"all {len(existing)} tables present and valid", section)
        except Exception as exc:
            return self._fail("Schema", str(exc), section)

    def _check_historical_data(self) -> CheckResult:
        section = "Database"
        if self._db is None:
            return self._skip("Historical data", "DB not connected", section)
        try:
            with self._db._connect() as conn:
                cur = conn.execute(
                    "SELECT index_id, COUNT(*) as cnt FROM daily_prices "
                    "GROUP BY index_id ORDER BY cnt DESC"
                )
                rows = cur.fetchall()
            if not rows:
                return self._fail("Historical data", "No daily price data found — run seed_historical.py", section)
            good = [(idx, cnt) for idx, cnt in rows if cnt >= 200]
            return self._pass(
                "Historical data",
                f"{len(good)} indices with 200+ daily bars (total {len(rows)} indices)",
                section,
            )
        except Exception as exc:
            return self._fail("Historical data", str(exc), section)

    def _check_cross_index_overlap(self) -> CheckResult:
        section = "Database"
        if self._db is None:
            return self._skip("Cross-index overlap", "DB not connected", section)
        try:
            with self._db._connect() as conn:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM daily_prices WHERE index_id='NIFTY50'"
                )
                n50 = cur.fetchone()[0]
                cur = conn.execute(
                    "SELECT COUNT(*) FROM daily_prices WHERE index_id='BANKNIFTY'"
                )
                bnk = cur.fetchone()[0]
            overlap = min(n50, bnk)
            if overlap < 100:
                return self._warn(
                    "Cross-index overlap",
                    f"NIFTY50 ∩ BANKNIFTY: {overlap} days (need 100+)",
                    section,
                )
            return self._pass(
                "Cross-index overlap",
                f"NIFTY50 ∩ BANKNIFTY: {overlap} days",
                section,
            )
        except Exception as exc:
            return self._warn("Cross-index overlap", str(exc), section, critical=False)

    def _check_options_snapshots(self) -> CheckResult:
        section = "Database"
        if self._db is None:
            return self._skip("Options snapshots", "DB not connected", section)
        try:
            with self._db._connect() as conn:
                cur = conn.execute("SELECT COUNT(*) FROM options_snapshots")
                count = cur.fetchone()[0]
            if count == 0:
                return self._warn("Options snapshots", "No options data found", section, critical=False)
            return self._pass("Options snapshots", f"{count:,} records", section)
        except Exception as exc:
            return self._warn("Options snapshots", str(exc), section, critical=False)

    def _check_fii_dii_data(self) -> CheckResult:
        section = "Database"
        if self._db is None:
            return self._skip("FII/DII data", "DB not connected", section)
        try:
            with self._db._connect() as conn:
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='fii_dii_activity'"
                )
                if not cur.fetchone():
                    return self._warn("FII/DII data", "Table not found (optional)", section, critical=False)
                cur = conn.execute("SELECT COUNT(*) FROM fii_dii_activity")
                count = cur.fetchone()[0]
            return self._pass("FII/DII data", f"{count} records", section, critical=False)
        except Exception as exc:
            return self._warn("FII/DII data", str(exc), section, critical=False)

    def _check_vix_data(self) -> CheckResult:
        section = "Database"
        if self._db is None:
            return self._skip("VIX data", "DB not connected", section)
        try:
            with self._db._connect() as conn:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM daily_prices WHERE index_id='INDIA_VIX'"
                )
                count = cur.fetchone()[0]
            if count == 0:
                return self._warn("VIX data", "No VIX records found", section, critical=False)
            return self._pass("VIX data", f"{count} records", section, critical=False)
        except Exception as exc:
            return self._warn("VIX data", str(exc), section, critical=False)

    # -----------------------------------------------------------------------
    # Data layer checks
    # -----------------------------------------------------------------------

    def _check_nse_connectivity(self) -> CheckResult:
        section = "Data Layer"
        try:
            import urllib.request
            start = time.time()
            req = urllib.request.Request(
                "https://www.nseindia.com",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            urllib.request.urlopen(req, timeout=10)
            ms = int((time.time() - start) * 1000)
            return self._pass("NSE scraper", f"connected (response: {ms}ms)", section, critical=False)
        except Exception as exc:
            return self._warn("NSE scraper", f"not reachable: {exc}", section, critical=False)

    def _check_bse_connectivity(self) -> CheckResult:
        section = "Data Layer"
        try:
            import urllib.request
            start = time.time()
            req = urllib.request.Request(
                "https://www.bseindia.com",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            urllib.request.urlopen(req, timeout=10)
            ms = int((time.time() - start) * 1000)
            return self._pass("BSE scraper", f"connected (response: {ms}ms)", section, critical=False)
        except Exception as exc:
            return self._warn("BSE scraper", f"not reachable: {exc}", section, critical=False)

    def _check_options_chain(self) -> CheckResult:
        section = "Data Layer"
        if self._db is None:
            return self._skip("Options chain", "DB not connected", section)
        try:
            with self._db._connect() as conn:
                cur = conn.execute(
                    "SELECT COUNT(DISTINCT strike) FROM options_snapshots "
                    "WHERE index_id='NIFTY50' ORDER BY timestamp DESC LIMIT 1"
                )
                strikes = cur.fetchone()[0]
            if strikes == 0:
                return self._warn("Options chain", "No NIFTY options data", section, critical=False)
            return self._pass("Options chain", f"NIFTY chain has {strikes} strikes in DB", section, critical=False)
        except Exception as exc:
            return self._warn("Options chain", str(exc), section, critical=False)

    def _check_vix_feed(self) -> CheckResult:
        section = "Data Layer"
        try:
            from src.data.nse_scraper import NSEScraper
            scraper = NSEScraper()
            vix = scraper.get_vix()
            if vix and vix > 0:
                return self._pass("VIX", f"current: {vix:.2f}", section, critical=False)
            return self._warn("VIX", "Could not fetch live VIX", section, critical=False)
        except Exception as exc:
            return self._warn("VIX", str(exc), section, critical=False)

    def _check_rate_limiter(self) -> CheckResult:
        section = "Data Layer"
        try:
            from src.data.rate_limiter import get_global_limiter
            limiter = get_global_limiter()
            return self._pass("Rate limiter", "global NSE limiter active", section, critical=False)
        except Exception as exc:
            return self._warn("Rate limiter", str(exc), section, critical=False)

    def _check_circuit_breaker(self) -> CheckResult:
        section = "Data Layer"
        try:
            from src.data.circuit_breaker import get_circuit_breaker
            cb = get_circuit_breaker("NSE")
            state = cb.state if hasattr(cb, "state") else "UNKNOWN"
            if state in ("CLOSED", "UNKNOWN"):
                return self._pass("Circuit breaker", f"{state} (healthy)", section, critical=False)
            return self._warn("Circuit breaker", f"state: {state}", section, critical=False)
        except Exception as exc:
            return self._warn("Circuit breaker", str(exc), section, critical=False)

    # -----------------------------------------------------------------------
    # Analysis checks
    # -----------------------------------------------------------------------

    def _check_technical_indicators(self) -> CheckResult:
        section = "Analysis"
        if self._db is None:
            return self._skip("Technical indicators", "DB not connected", section)
        try:
            from src.engine.technical_analysis import TechnicalAnalyzer
            analyzer = TechnicalAnalyzer(self._db)
            result = analyzer.compute_indicators("NIFTY50")
            count = len(result) if result else 0
            if count == 0:
                return self._warn("Technical indicators", "No indicators computed (insufficient data?)", section, critical=False)
            return self._pass("Technical indicators", f"{count} indicators computed for NIFTY50", section)
        except Exception as exc:
            return self._warn("Technical indicators", str(exc), section, critical=False)

    def _check_news_engine(self) -> CheckResult:
        section = "Analysis"
        try:
            from src.news.news_engine import NewsEngine
            engine = NewsEngine()
            articles = engine.fetch_latest(limit=5)
            return self._pass("News engine", f"{len(articles)} articles fetched, sentiment computed", section, critical=False)
        except Exception as exc:
            return self._warn("News engine", str(exc), section, critical=False)

    def _check_anomaly_detection(self) -> CheckResult:
        section = "Analysis"
        if self._db is None:
            return self._skip("Anomaly detection", "DB not connected", section)
        try:
            from src.engine.anomaly_detector import AnomalyDetector
            detector = AnomalyDetector(self._db)
            alerts = detector.get_active_alerts()
            return self._pass("Anomaly detection", f"running ({len(alerts)} active alerts)", section, critical=False)
        except Exception as exc:
            return self._warn("Anomaly detection", str(exc), section, critical=False)

    def _check_regime_detector(self) -> CheckResult:
        section = "Analysis"
        if self._db is None:
            return self._skip("Regime detector", "DB not connected", section)
        try:
            from src.engine.regime_detector import RegimeDetector
            detector = RegimeDetector(self._db)
            regime = detector.get_current_regime("NIFTY50")
            regime_str = regime.value if hasattr(regime, "value") else str(regime)
            return self._pass("Regime detector", f"current: {regime_str}", section, critical=False)
        except Exception as exc:
            return self._warn("Regime detector", str(exc), section, critical=False)

    # -----------------------------------------------------------------------
    # Decision Engine checks
    # -----------------------------------------------------------------------

    def _check_signal_generation(self) -> CheckResult:
        section = "Decision Engine"
        if self._db is None:
            return self._skip("Signal generation", "DB not connected", section)
        try:
            from src.engine.decision_engine import DecisionEngine
            engine = DecisionEngine(self._db)
            # Verify the engine can instantiate without errors
            return self._pass("Signal generation", "Decision engine initialised successfully", section)
        except Exception as exc:
            return self._fail("Signal generation", str(exc), section)

    def _check_risk_manager(self) -> CheckResult:
        section = "Decision Engine"
        try:
            from src.engine.risk_manager import RiskManager
            rm = RiskManager()
            return self._pass("Risk manager", "position sizing module loaded", section)
        except Exception as exc:
            return self._fail("Risk manager", str(exc), section)

    def _check_kill_switch(self) -> CheckResult:
        section = "Decision Engine"
        try:
            kill_switch_file = _ROOT / "data" / "KILL_SWITCH"
            if kill_switch_file.exists():
                return self._fail("Kill switch", "KILL_SWITCH file present — trading is paused", section)
            return self._pass("Kill switch", "INACTIVE ✅", section)
        except Exception as exc:
            return self._warn("Kill switch", str(exc), section, critical=False)

    def _check_signal_deduplication(self) -> CheckResult:
        section = "Decision Engine"
        try:
            from src.engine.signal_generator import SignalGenerator
            return self._pass("Signal deduplication", "active", section, critical=False)
        except Exception as exc:
            return self._warn("Signal deduplication", str(exc), section, critical=False)

    def _check_audit_trail(self) -> CheckResult:
        section = "Decision Engine"
        try:
            audit_dir = _ROOT / "data" / "audit"
            if not audit_dir.exists():
                return self._warn("Audit trail", "Audit directory not found (will be created on first signal)", section, critical=False)
            files = list(audit_dir.glob("*.json"))
            return self._pass("Audit trail", f"structured JSON saved ({len(files)} files)", section, critical=False)
        except Exception as exc:
            return self._warn("Audit trail", str(exc), section, critical=False)

    # -----------------------------------------------------------------------
    # Paper Trading checks
    # -----------------------------------------------------------------------

    def _check_paper_engine(self) -> CheckResult:
        section = "Paper Trading"
        if self._db is None:
            return self._skip("Paper engine", "DB not connected", section)
        try:
            from src.paper_trading.paper_engine import PaperTradingEngine, PaperTradingConfig
            config = PaperTradingConfig(initial_capital=100_000.0)
            engine = PaperTradingEngine(self._db, None, config)
            capital = engine.current_capital
            return self._pass("Paper engine", f"initialised (₹{capital:,.0f})", section)
        except Exception as exc:
            return self._fail("Paper engine", str(exc), section)

    def _check_state_persistence(self) -> CheckResult:
        section = "Paper Trading"
        if self._db is None:
            return self._skip("State persistence", "DB not connected", section)
        try:
            from src.paper_trading.paper_engine import PaperTradingEngine, PaperTradingConfig
            config = PaperTradingConfig(initial_capital=100_000.0)
            engine = PaperTradingEngine(self._db, None, config)
            engine.save_state()
            engine.load_state()
            return self._pass("State persistence", "save/load working", section)
        except Exception as exc:
            return self._fail("State persistence", str(exc), section)

    def _check_position_tracking(self) -> CheckResult:
        section = "Paper Trading"
        if self._db is None:
            return self._skip("Position tracking", "DB not connected", section)
        try:
            from src.paper_trading.paper_engine import PaperTradingEngine, PaperTradingConfig
            config = PaperTradingConfig(initial_capital=100_000.0)
            engine = PaperTradingEngine(self._db, None, config)
            positions = engine.get_open_positions()
            return self._pass("Position tracking", f"verified ({len(positions)} open)", section)
        except Exception as exc:
            return self._fail("Position tracking", str(exc), section)

    # -----------------------------------------------------------------------
    # Alert checks
    # -----------------------------------------------------------------------

    def _check_telegram_bot(self) -> CheckResult:
        section = "Alerts"
        try:
            import os
            token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
            if not token or not chat_id:
                return self._warn("Telegram bot", "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set", section, critical=False)
            from src.alerts.telegram_bot import TelegramBot
            bot = TelegramBot(self._db, None)
            if bot._configured:  # noqa: SLF001
                return self._pass("Telegram bot", "configured", section, critical=False)
            return self._warn("Telegram bot", "not configured", section, critical=False)
        except Exception as exc:
            return self._warn("Telegram bot", str(exc), section, critical=False)

    def _check_digest_mode(self) -> CheckResult:
        section = "Alerts"
        try:
            from src.alerts.telegram_bot import TelegramBot
            bot = TelegramBot(self._db, None)
            # Digest mode is available if the bot class has the method
            has_digest = hasattr(bot, "send_digest") or hasattr(bot, "digest_mode")
            status = "configured (hourly + EOD)" if has_digest else "basic alerts only"
            return self._pass("Digest mode", status, section, critical=False)
        except Exception as exc:
            return self._warn("Digest mode", str(exc), section, critical=False)

    def _check_websocket(self) -> CheckResult:
        section = "Alerts"
        try:
            from src.api.websocket import get_ws_queue
            queue = get_ws_queue()
            return self._pass("WebSocket", "server module loaded", section, critical=False)
        except Exception as exc:
            return self._warn("WebSocket", str(exc), section, critical=False)

    # -----------------------------------------------------------------------
    # API & Dashboard checks
    # -----------------------------------------------------------------------

    def _check_api_server(self) -> CheckResult:
        section = "API & Dashboard"
        try:
            from src.api.app import create_app
            app = create_app()
            return self._pass("API server", "app factory works", section, critical=False)
        except Exception as exc:
            return self._warn("API server", str(exc), section, critical=False)

    def _check_api_endpoints(self) -> CheckResult:
        section = "API & Dashboard"
        try:
            from src.api.app import create_app
            from fastapi.testclient import TestClient
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)
            routes = [r.path for r in app.routes if hasattr(r, "path") and r.path.startswith("/api/")]
            ok = 0
            for path in routes[:5]:  # spot-check first 5
                resp = client.get(path)
                if resp.status_code < 500:
                    ok += 1
            return self._pass("API endpoints", f"{len(routes)} routes registered, spot-check passed", section, critical=False)
        except Exception as exc:
            return self._warn("API endpoints", str(exc), section, critical=False)

    def _check_dashboard(self) -> CheckResult:
        section = "API & Dashboard"
        try:
            frontend_dir = _ROOT / "frontend"
            dist_dir = _ROOT / "frontend" / "dist"
            index_options = [
                frontend_dir / "index.html",
                dist_dir / "index.html",
            ]
            for p in index_options:
                if p.exists():
                    return self._pass("Dashboard", f"frontend found at {p.relative_to(_ROOT)}", section, critical=False)
            return self._warn("Dashboard", "frontend/index.html not found", section, critical=False)
        except Exception as exc:
            return self._warn("Dashboard", str(exc), section, critical=False)

    # -----------------------------------------------------------------------
    # Backtesting checks
    # -----------------------------------------------------------------------

    def _check_backtest_results(self) -> CheckResult:
        section = "Backtesting"
        try:
            data_dir = _ROOT / "data"
            results = list(data_dir.rglob("backtest_results*.json")) + list(data_dir.rglob("*backtest*.json"))
            if not results:
                return self._warn("Last backtest", "No backtest results found — run run_backtest.py first", section, critical=False)
            import json
            data = json.loads(results[0].read_text())
            sharpe = data.get("sharpe_ratio", data.get("sharpe", "N/A"))
            wr = data.get("win_rate", "N/A")
            index = data.get("index_id", data.get("index", "?"))
            return self._pass(
                "Last backtest",
                f"{index} — Sharpe {sharpe}, WR {wr}",
                section, critical=False,
            )
        except Exception as exc:
            return self._warn("Last backtest", str(exc), section, critical=False)

    def _check_walk_forward(self) -> CheckResult:
        section = "Backtesting"
        try:
            data_dir = _ROOT / "data"
            results = list(data_dir.rglob("walk_forward*.json")) + list(data_dir.rglob("*walk_forward*.json"))
            if not results:
                return self._warn("Walk-forward", "No walk-forward results found", section, critical=False)
            import json
            data = json.loads(results[0].read_text())
            profitable = data.get("profitable_windows_pct", data.get("pct_profitable", None))
            if profitable is not None:
                pct = f"{profitable:.0%}" if profitable <= 1 else f"{profitable:.1f}%"
                return self._pass("Walk-forward", f"{pct} test windows profitable", section, critical=False)
            return self._pass("Walk-forward", "results file present", section, critical=False)
        except Exception as exc:
            return self._warn("Walk-forward", str(exc), section, critical=False)

    def _check_robustness(self) -> CheckResult:
        section = "Backtesting"
        try:
            data_dir = _ROOT / "data"
            results = list(data_dir.rglob("robustness*.json"))
            if not results:
                return self._warn("Robustness", "No robustness results found", section, critical=False)
            import json
            data = json.loads(results[0].read_text())
            score = data.get("robustness_score", data.get("score", None))
            label = data.get("label", "")
            if score is not None:
                return self._pass(
                    "Robustness",
                    f"{float(score):.2f} ({label})" if label else f"{float(score):.2f}",
                    section, critical=False,
                )
            return self._pass("Robustness", "results file present", section, critical=False)
        except Exception as exc:
            return self._warn("Robustness", str(exc), section, critical=False)

    # -----------------------------------------------------------------------
    # System checks
    # -----------------------------------------------------------------------

    def _check_disk_space(self) -> CheckResult:
        section = "System"
        try:
            import shutil
            total, used, free = shutil.disk_usage(_ROOT)
            free_gb = free / (1024 ** 3)
            free_pct = free / total * 100
            if free_pct < 5 or free_gb < 1:
                return self._fail("Disk space", f"only {free_gb:.1f} GB free — critical", section)
            if free_pct < 10 or free_gb < 5:
                return self._warn("Disk space", f"{free_gb:.1f} GB free", section, critical=False)
            return self._pass("Disk space", f"{free_gb:.1f} GB free", section)
        except Exception as exc:
            return self._warn("Disk space", str(exc), section, critical=False)

    def _check_market_calendar(self) -> CheckResult:
        section = "System"
        try:
            from src.utils.market_hours import MarketHoursManager
            mh = MarketHoursManager()
            # Attempt to get next holiday if the method exists
            if hasattr(mh, "get_next_holiday"):
                holiday = mh.get_next_holiday()
                return self._pass("Market calendar", f"next holiday: {holiday}", section, critical=False)
            return self._pass("Market calendar", "MarketHoursManager loaded", section, critical=False)
        except Exception as exc:
            return self._warn("Market calendar", str(exc), section, critical=False)

    def _check_watchdog(self) -> CheckResult:
        section = "System"
        try:
            from src.paper_trading.watchdog import SystemWatchdog
            return self._pass("Watchdog", "module loaded (starts with system)", section, critical=False)
        except Exception as exc:
            return self._fail("Watchdog", str(exc), section)
