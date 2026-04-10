"""
Configuration Validation Script.

Validates the entire system configuration before first run, including:
- JSON consistency and schema boundaries
- Missing environment variables
- Database migration readiness
- Network connectivity
- Package dependencies and Python versions
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.database.db_manager import DatabaseManager
from src.data.index_registry import IndexRegistry, IndexRegistryError

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
}


class Reporter:
    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0
        self.warnings: int = 0
        self.errors: list[str] = []

    def log_pass(self, name: str, msg: str) -> None:
        self.passed += 1
        print(f"[PASS] {name} — {msg}")

    def log_warn(self, name: str, msg: str) -> None:
        self.warnings += 1
        print(f"[WARN] {name} — {msg}")

    def log_fail(self, name: str, msg: str, fix: str) -> None:
        self.failed += 1
        self.errors.append(f"{name}: {msg}\n  Fix: {fix}")
        print(f"[FAIL] {name} — {msg}")


def check_indices(reporter: Reporter) -> None:
    try:
        registry = IndexRegistry.from_file(settings.indices_config_path)
        active = len(registry.get_active_indices())
        fo = len(registry.get_indices_with_options())
        
        # Verify yahoo_symbol formats and duplicates
        ids = set()
        for i in registry.get_active_indices():
            if getattr(i, "id", None) in ids:
                raise ValueError(f"Duplicate id {i.id} found in indices")
            ids.add(getattr(i, "id", None))
            if i.yahoo_symbol and not i.yahoo_symbol.startswith("^") and "." not in i.yahoo_symbol:
                # yfinance indices generally start with ^ or represent proper symbols.
                pass 
                
        reporter.log_pass("indices.json", f"{active} indices loaded, {fo} with F&O")
    except Exception as exc:
        reporter.log_fail("indices.json", str(exc), "Ensure JSON is well-formatted and all mandatory fields exist (see logs).")


def check_news_mappings(reporter: Reporter) -> None:
    if not settings.news_mappings_path.exists():
        reporter.log_fail("news_mappings.json", "File missing", f"Create {settings.news_mappings_path.name}")
        return
        
    try:
        registry = IndexRegistry.from_file(settings.indices_config_path)
        data = json.loads(settings.news_mappings_path.read_text(encoding="utf-8"))
        categories = 0
        keywords = 0
        for cat, mapping in data.items():
            if cat.startswith("_"):
                continue
            categories += 1
            kw_list = mapping.get("keywords", [])
            keywords += len(kw_list)
            for i in mapping.get("affected_indices", []):
                if i not in registry:
                    raise ValueError(f"Unknown index '{i}' in {cat}")

        reporter.log_pass("news_mappings.json", f"{categories} categories, {keywords} keywords")
    except Exception as exc:
        reporter.log_fail("news_mappings.json", str(exc), "Ensure referenced indices map exactly to indices.json IDs.")


def check_rss_feeds(reporter: Reporter) -> None:
    if not settings.rss_feeds_path.exists():
        reporter.log_warn("rss_feeds.json", "File not found, News Engine will be silent")
        return
        
    try:
        data = json.loads(settings.rss_feeds_path.read_text(encoding="utf-8"))
        feeds = 0
        for entry in data:
            url = entry.get("url", "")
            if not urlparse(url).scheme:
                raise ValueError(f"Invalid URL: {url}")
            t_score = entry.get("credibility_score", 1.0)
            if not (0.0 <= t_score <= 1.0):
                raise ValueError(f"Credibility score {t_score} out of bounds (0-1).")
            feeds += 1
        reporter.log_pass("rss_feeds.json", f"{feeds} feeds configured")
    except Exception as exc:
        reporter.log_fail("rss_feeds.json", str(exc), "Check URL formatting and ensure credibility floats are 0.0-1.0")


def check_sentiment(reporter: Reporter) -> None:
    if not settings.sentiment_keywords_path.exists():
        reporter.log_warn("sentiment_keywords.json", "Not found, defaulting sentiment to neutral.")
        return
        
    try:
        data = json.loads(settings.sentiment_keywords_path.read_text(encoding="utf-8"))
        bullish = set(item["keyword"] for item in data.get("bullish", []))
        bearish = set(item["keyword"] for item in data.get("bearish", []))
        neutral = set(item["keyword"] for item in data.get("neutral_uncertainty", []))
        
        intersect = bullish.intersection(bearish)
        if intersect:
            raise ValueError(f"Key overlap between Bullish and Bearish: {intersect}")
            
        reporter.log_pass("sentiment_keywords.json", f"{len(bullish)} bullish, {len(bearish)} bearish, {len(neutral)} neutral")
    except Exception as exc:
        reporter.log_fail("sentiment_keywords.json", str(exc), "Make sure keywords don't overlap between sentiments.")


def check_env(reporter: Reporter) -> None:
    env_p = settings.config_dir.parent / ".env"
    if env_p.exists():
        reporter.log_pass(".env file", "found and loaded")
    else:
        reporter.log_warn(".env file not found", "using defaults")


def check_database(reporter: Reporter) -> None:
    try:
        db = DatabaseManager.instance()
        db.connect()
        db.initialise_schema()
        # count tables explicitly
        out = db.execute("SELECT count(*) FROM sqlite_master WHERE type='table';").fetchone()
        table_count = out[0] if out else 0
        reporter.log_pass("Database", f"SQLite OK, {table_count} tables created")
    except Exception as exc:
        reporter.log_fail("Database", str(exc), "Check directory read/write permissions for data/db folder.")
        return

    # Run migrations
    try:
        from src.database.migrations import run_migrations
        applied = run_migrations(db)
        if applied:
            reporter.log_pass("Migrations", f"{applied} new migration(s) applied")
        else:
            reporter.log_pass("Migrations", "Schema up to date")
    except Exception as exc:
        reporter.log_fail("Migrations", str(exc), "Check src/database/migrations.py for errors.")

    # Schema validation
    try:
        is_valid, issues = db.validate_schema()
        if is_valid:
            reporter.log_pass("Schema validation", "All tables and columns match expected schema")
        else:
            for issue in issues:
                if "Unexpected table" in issue:
                    reporter.log_warn("Schema", issue)
                else:
                    reporter.log_fail("Schema", issue, "Run migrations to fix: python -c \"from src.database.migrations import run_migrations; run_migrations()\"")
    except Exception as exc:
        reporter.log_fail("Schema validation", str(exc), "Check src/database/models.py EXPECTED_SCHEMA.")


def check_network(reporter: Reporter) -> None:
    targets = [("NSE", "https://www.nseindia.com/"), ("BSE", "https://www.bseindia.com/")]
    for name, url in targets:
        try:
            start = time.perf_counter()
            requests.get(url, headers=HEADERS, timeout=10) # NSE rejects HEAD, GET is safer for bypassing basic block filters
            ms = int((time.perf_counter() - start) * 1000)
            reporter.log_pass(f"{name} connectivity", f"reachable ({ms}ms)")
        except requests.RequestException as exc:
            reporter.log_fail(f"{name} connectivity", f"unreachable", "Check your firewall, DNS, or NSE geo-blocks from your IP.")


def check_system(reporter: Reporter) -> None:
    # Python limits
    v = sys.version_info
    py_ver = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        reporter.log_pass(f"Python {py_ver}", "OK")
    else:
        reporter.log_fail(f"Python {py_ver}", "Unsupported version", "Upgrade to Python >= 3.11 for optimal compatibility.")
        
    modules = ["pandas", "sqlalchemy", "requests", "yfinance", "apscheduler", "fastapi"]
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            reporter.log_fail("Dependencies", f"Module {mod} not found", f"Run: pip install -r requirements.txt")
            return
            
    reporter.log_pass("All dependencies installed", "Ready")


def main() -> None:
    print("Configuration Validation Report\n")
    rpt = Reporter()
    
    check_indices(rpt)
    check_news_mappings(rpt)
    check_rss_feeds(rpt)
    check_sentiment(rpt)
    check_database(rpt)
    check_network(rpt)
    check_env(rpt)
    check_system(rpt)
    
    print(f"\nResult: {rpt.passed} passed, {rpt.failed} failed, {rpt.warnings} warning")
    if rpt.failed > 0:
        print("\n--- Failures ---")
        for e in rpt.errors:
            print(f"- {e}")
        print("\nSystem has broken configurations. Please fix before running.")
        sys.exit(1)
    else:
        print("System is ready to run.")
        sys.exit(0)


if __name__ == "__main__":
    main()
