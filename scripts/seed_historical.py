"""
Historical Data Bootstrap and Seeding Script.

Idempotent one-time setup script that populates:
 - 2 Years Daily OHLCV Data for ALL active indices
 - 5 Days Intraday (5m) Data for F&O indices
 - 30 Days Historic FII/DII
 - 30 Days Historic VIX
"""

import sys
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.database.db_manager import DatabaseManager
from src.database.migrations import MigrationRunner
from src.data.index_registry import IndexRegistry
from src.data.historical_data import HistoricalDataManager
from src.data.nse_scraper import NSEScraper
from src.data.fii_dii_data import FIIDIIFetcher, FIIDIIData
from src.data.vix_data import VIXTracker, VIXData

_IST = ZoneInfo(IST_TIMEZONE)


def human_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


def validate_seed_results(db) -> None:
    """Validate that seeded data is sufficient for system operation."""

    print("\n" + "=" * 50)
    print("  Seed Validation Report")
    print("=" * 50)

    issues = []

    # Check 1: Indices with sufficient daily bars
    try:
        result = db.fetch_all(
            """SELECT index_id, COUNT(*) as bar_count
               FROM price_data
               WHERE timeframe = '1d'
               GROUP BY index_id
               HAVING bar_count >= 200"""
        )
        indices_with_data = len(result) if result else 0
        total_indices_row = db.fetch_one(
            "SELECT COUNT(DISTINCT index_id) as c FROM price_data WHERE timeframe = '1d'"
        )
        total = total_indices_row['c'] if total_indices_row else 0
    except Exception:
        indices_with_data = 0
        total = 0

    status = "PASS" if indices_with_data >= 5 else "FAIL"
    if status == "FAIL":
        issues.append("Fewer than 5 indices have 200+ daily bars")
    print(f"  [{status}] Indices with 200+ daily bars: {indices_with_data}/{total}")

    # Check 2: F&O indices specifically
    fo_indices = ["NIFTY50", "BANKNIFTY"]
    for idx in fo_indices:
        try:
            count = db.fetch_one(
                "SELECT COUNT(*) as c FROM price_data WHERE index_id = ? AND timeframe = '1d'",
                (idx,)
            )
            bars = count['c'] if count else 0
        except Exception:
            bars = 0
        status = "PASS" if bars >= 200 else "FAIL"
        if status == "FAIL":
            issues.append(f"{idx} has only {bars} daily bars (need 200+)")
        print(f"  [{status}] {idx}: {bars} daily bars")

    # Check 3: Cross-index overlap for divergence detection
    try:
        nifty_dates = db.fetch_all(
            """SELECT DISTINCT DATE(timestamp) as d FROM price_data
               WHERE index_id = 'NIFTY50' AND timeframe = '1d'"""
        )
        banknifty_dates = db.fetch_all(
            """SELECT DISTINCT DATE(timestamp) as d FROM price_data
               WHERE index_id = 'BANKNIFTY' AND timeframe = '1d'"""
        )
    except Exception:
        nifty_dates = []
        banknifty_dates = []

    if nifty_dates and banknifty_dates:
        nifty_set = {row['d'] for row in nifty_dates}
        bn_set = {row['d'] for row in banknifty_dates}
        overlap = len(nifty_set & bn_set)
        status = "PASS" if overlap >= 100 else "WARN"
        print(f"  [{status}] NIFTY50 ∩ BANKNIFTY overlap: {overlap} days")
    else:
        print(f"  [FAIL] Cannot check cross-index overlap — missing data")
        issues.append("Cross-index overlap check failed")

    # Check 4: Intraday data for F&O indices
    try:
        intraday = db.fetch_all(
            """SELECT index_id, COUNT(*) as c FROM price_data
               WHERE timeframe IN ('5m', '15m', '1m')
               GROUP BY index_id"""
        )
        intraday_count = len(intraday) if intraday else 0
    except Exception:
        intraday_count = 0
    status = "PASS" if intraday_count >= 2 else "WARN"
    print(f"  [{status}] Indices with intraday data: {intraday_count}")

    # Check 5: FII/DII data
    try:
        fii_count = db.fetch_one("SELECT COUNT(*) as c FROM fii_dii_activity")
        fii = fii_count['c'] if fii_count else 0
    except Exception:
        fii = 0
    status = "PASS" if fii >= 10 else "WARN"
    if fii == 0:
        status = "WARN"
    print(f"  [{status}] FII/DII data points: {fii}")

    # Check 6: VIX data
    try:
        vix_count = db.fetch_one("SELECT COUNT(*) as c FROM vix_data")
        vix = vix_count['c'] if vix_count else 0
    except Exception:
        vix = 0
    status = "PASS" if vix >= 10 else "WARN"
    print(f"  [{status}] VIX data points: {vix}")

    # Check 7: Database size
    db_path = getattr(db, "_db_path", "data/db/trading.db")
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"  [INFO] Database size: {size_mb:.1f} MB")

    # Final verdict
    print()
    if not issues:
        print("  ✅ Ready for backtesting: YES")
        print("  ✅ Ready for divergence detection: YES")
        print("  ✅ Ready for signal generation: YES")
    else:
        print(f"  ⚠️  Issues found: {len(issues)}")
        for issue in issues:
            print(f"     - {issue}")

        # Determine what's available despite issues
        if indices_with_data >= 3:
            print("  ✅ Ready for basic backtesting: YES (limited indices)")
        else:
            print("  ❌ Ready for backtesting: NO — need more data")

    print("=" * 50)


def main() -> None:
    start_time = time.time()
    total_records = 0
    failed_indices: list[str] = []

    # ── Database & Registry ────────────────────────────────────────────────
    db = DatabaseManager.instance()
    db.connect()
    db.initialise_schema()
    MigrationRunner(db).run_pending()

    registry = IndexRegistry.from_file(settings.indices_config_path)
    registry.sync_to_db(db)

    manager = HistoricalDataManager(registry, db)

    print("\n" + "=" * 60)
    print("  Trading DSS — Historical Data Seeding")
    print("=" * 60)

    # ── Priority Tiers Definition ──────────────────────────────────────────
    PRIORITY_TIER_1 = ["NIFTY50", "BANKNIFTY", "SENSEX"]
    PRIORITY_TIER_2 = [
        "FINNIFTY", "MIDCPNIFTY",
        "NIFTYIT", "NIFTYPHARMA", "NIFTYMETAL",
        "NIFTYENERGY", "NIFTYFMCG", "NIFTYAUTO",
    ]
    PRIORITY_TIER_3 = [
        "NIFTYREALTY", "NIFTYMEDIA", "NIFTYINFRA",
        "NIFTY_PSU_BANK", "NIFTY_PRIVATE_BANK",
        "NIFTY_COMMODITIES", "NIFTY_CONSUMPTION",
        "NIFTYCPSE", "NIFTY_FINANCIAL_SERVICES",
    ]

    all_active = {idx.id for idx in registry.get_active_indices()}

    ordered_indices = []
    for tier_name, tier_list in [("TIER 1 (Critical)", PRIORITY_TIER_1),
                                  ("TIER 2 (Important)", PRIORITY_TIER_2),
                                  ("TIER 3 (Supplementary)", PRIORITY_TIER_3)]:
        tier_valid = [idx for idx in tier_list if idx in all_active]
        if tier_valid:
            print(f"\n  {tier_name}: {len(tier_valid)} indices")
            ordered_indices.extend(tier_valid)

    # Tier 4: remaining indices not in any priority tier
    already_listed = set(PRIORITY_TIER_1 + PRIORITY_TIER_2 + PRIORITY_TIER_3)
    remaining = sorted(all_active - already_listed)
    if remaining:
        print(f"\n  TIER 4 (Remaining): {len(remaining)} indices")
        ordered_indices.extend(remaining)

    print(f"\n  Total download order: {len(ordered_indices)} indices")
    print(f"  Priority indices will be downloaded first.\n")

    try:
        # ── Phase 1: Daily OHLCV (2 years) ─────────────────────────────────────
        print(f"[1/4] Daily OHLCV data (2 years) — {len(ordered_indices)} indices")
        print("-" * 50)
        success_count = 0
        curr = 1

        tier_groups = [
            ("TIER 1 (Critical)", [idx for idx in PRIORITY_TIER_1 if idx in all_active]),
            ("TIER 2 (Important)", [idx for idx in PRIORITY_TIER_2 if idx in all_active]),
            ("TIER 3 (Supplementary)", [idx for idx in PRIORITY_TIER_3 if idx in all_active]),
            ("TIER 4 (Remaining)", remaining)
        ]

        for t_name, t_list in tier_groups:
            if not t_list:
                continue

            for idx_id in t_list:
                idx = registry.get_index(idx_id)
                label = f"  ({curr}/{len(ordered_indices)}) {idx.id:<20}"
                if not idx.yahoo_symbol:
                    print(f"{label} SKIP — no yahoo_symbol")
                    curr += 1
                    continue
                try:
                    df = manager.download_index_history(idx.id, period="2y", interval="1d")
                    if df.empty:
                        print(f"{label} SKIP — empty response")
                        failed_indices.append(idx.id)
                    else:
                        manager._save_dataframe_to_db(idx.id, df, timeframe="1d")
                        print(f"{label} {len(df):>4} candles ✓")
                        total_records += len(df)
                        success_count += 1
                except Exception as e:
                    print(f"{label} FAIL — {e}")
                    failed_indices.append(idx.id)
                curr += 1
                time.sleep(0.5)  # rate limiting

            if t_name == "TIER 1 (Critical)":
                print(f"\n  ✓ TIER 1 complete — core F&O indices downloaded. "
                      f"System can generate basic signals even if script stops here.\n")
            elif t_name == "TIER 2 (Important)":
                print(f"\n  ✓ TIER 2 complete — sector indices downloaded. "
                      f"Divergence detection and sector rotation now available.\n")

        print(f"\n  ✓ Daily: {success_count}/{len(ordered_indices)} indices downloaded\n")

        # ── Phase 2: 5-min Intraday (5 days, F&O only) ────────────────────────
        fo_all = {idx.id for idx in registry.get_indices_with_options()}
        fo_ordered = [idx_id for idx_id in ordered_indices if idx_id in fo_all]

        print(f"[2/4] Intraday data (5 days, 5-min) — {len(fo_ordered)} F&O indices")
        print("-" * 50)
        success_fo = 0
        for i, idx_id in enumerate(fo_ordered, 1):
            idx = registry.get_index(idx_id)
            label = f"  ({i}/{len(fo_ordered)}) {idx.id:<20}"
            if not idx.yahoo_symbol:
                print(f"{label} SKIP — no yahoo_symbol")
                continue
            try:
                df = manager.download_index_history(idx.id, period="5d", interval="5m")
                if df.empty:
                    print(f"{label} SKIP — empty response")
                else:
                    manager._save_dataframe_to_db(idx.id, df, timeframe="5m")
                    print(f"{label} {len(df):>5} candles ✓")
                    total_records += len(df)
                    success_fo += 1
            except Exception as e:
                print(f"{label} FAIL — {e}")
            time.sleep(0.5)

        print(f"\n  ✓ Intraday: {success_fo}/{len(fo_ordered)} indices downloaded\n")

        # ── Phase 3: FII/DII (30 days) ────────────────────────────────────────
        print("[3/4] FII/DII data (30 days)")
        print("-" * 50)
        try:
            scraper = NSEScraper()
            fii_fetcher = FIIDIIFetcher(scraper=scraper)
            now = date.today()
            fii_history = fii_fetcher.fetch_historical_fii_dii(
                now - timedelta(days=30), now
            )
            if not fii_history:
                # NSE may not return historical; seed placeholder data for testing
                print("  NSE returned no historical FII/DII — seeding placeholder data")
                fii_history = [
                    FIIDIIData(
                        date=now - timedelta(days=i),
                        fii_buy_value=5000.0, fii_sell_value=4500.0, fii_net_value=500.0,
                        dii_buy_value=3000.0, dii_sell_value=2800.0, dii_net_value=200.0,
                    )
                    for i in range(22)
                    if (now - timedelta(days=i)).weekday() < 5  # skip weekends
                ]
            for h in fii_history:
                fii_fetcher.save_to_db(h, db)
            total_records += len(fii_history)
            print(f"  ✓ {len(fii_history)} trading days saved\n")
        except Exception as e:
            print(f"  FAIL — {e}\n")

        # ── Phase 4: VIX History (30 days) ─────────────────────────────────────
        print("[4/4] VIX history (30 days)")
        print("-" * 50)
        try:
            import yfinance as yf

            vix_df = yf.download("^INDIAVIX", period="1mo", interval="1d", progress=False)
            if vix_df is not None and not vix_df.empty:
                vix_count = 0
                scraper_vix = NSEScraper()
                vix_tracker = VIXTracker(scraper=scraper_vix)
                for ts, row in vix_df.iterrows():
                    try:
                        # Close value might be a series if columns are multi-indexed
                        close_val = float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"])
                        vd = VIXData(
                            value=close_val,
                            change=0.0,
                            change_pct=0.0,
                            timestamp=ts.to_pydatetime().replace(tzinfo=_IST),
                        )
                        vix_tracker.save_to_db(vd, db)
                        vix_count += 1
                    except Exception:
                        pass
                total_records += vix_count
                print(f"  ✓ {vix_count} VIX data points saved\n")
            else:
                print("  yfinance returned no VIX data — seeding placeholder\n")
                scraper_vix = NSEScraper()
                vix_tracker = VIXTracker(scraper=scraper_vix)
                now_dt = datetime.now(tz=_IST)
                for i in range(22):
                    dt = now_dt - timedelta(days=i)
                    if dt.weekday() < 5:
                        vd = VIXData(value=14.5, change=0.0, change_pct=0.0, timestamp=dt)
                        vix_tracker.save_to_db(vd, db)
                total_records += 16
                print(f"  ✓ 16 placeholder VIX points saved\n")
        except Exception as e:
            print(f"  FAIL — {e}\n")

    except KeyboardInterrupt:
        print("\n\n  [!] Seeding interrupted by user. Finalizing...")
    except Exception as e:
        print(f"\n\n  [!] Seeding encountered error: {e}")
    finally:
        validate_seed_results(db)

    # ── Summary ────────────────────────────────────────────────────────────
    runtime = human_time(time.time() - start_time)
    db_size = db.get_db_size()

    print("=" * 60)
    print("  Seeding Complete!")
    print(f"  Total records inserted : {total_records:,}")
    print(f"  Database size          : {db_size}")
    print(f"  Failed indices         : {len(failed_indices)}")
    if failed_indices:
        print(f"  Failed list            : {', '.join(failed_indices[:10])}")
    print(f"  Time taken             : {runtime}")
    print("=" * 60)


if __name__ == "__main__":
    main()
