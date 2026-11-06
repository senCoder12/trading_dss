"""
Recover missing FII/DII provisional *cash* records (₹ crore) from saved
third-party mirrors of NSE's combined provisional series. NSE itself serves
only the latest day, so there is no exchange archive to recover from.

Sources (raw files saved beforehand, no fetching here):
  * Groww JSON API windows (≤ 30 days each → daily rows; longer = weekly sums, rejected)
  * StockEzee page (__NEXT_DATA__: FII/DII net, 2 dp)
  * Trendlyne monthly pages (data-jsondata table: all six fields, 1 dp)

A date is VERIFIED only if all three sources have it, and:
  * Groww buy − sell = net for FII and DII        (±0.011)
  * StockEzee FII and DII net == Groww net         (±0.005, i.e. exact at 2 dp)
  * Trendlyne's six fields == Groww's six fields  (±0.051, Trendlyne rounds to 1 dp)
Anything else is reported as REJECTED or NO_DATA and is never written.

Writes use the existing upsert into fii_dii_activity (segment CASH). Dates that
already have CASH rows are never touched. A verified backup is taken first, and
a provenance manifest (every verified row plus all three source values) is
written next to the backup. Default is a dry run.

Usage:
    .venv/bin/python scripts/recover_fii_cash.py --raw DIR --start 2026-06-08 --end 2026-11-04 [--db PATH] [--apply]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from src.database import queries as Q  # noqa: E402
from src.database.db_manager import DatabaseManager  # noqa: E402

_IST = ZoneInfo("Asia/Kolkata")
FIELDS = ("fii_buy", "fii_sell", "fii_net", "dii_buy", "dii_sell", "dii_net")


# ── parsers ───────────────────────────────────────────────────────────────────


def parse_groww(raw_dir: Path) -> dict[date, dict]:
    out: dict[date, dict] = {}
    for f in sorted(raw_dir.glob("groww_*_*.json")):
        m = re.match(r"groww_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.json", f.name)
        if not m:
            continue
        span = (date.fromisoformat(m.group(2)) - date.fromisoformat(m.group(1))).days
        if span > 30:  # Groww silently switches to weekly aggregates
            print(f"  skip {f.name}: window {span} d > 30 d (weekly aggregates)")
            continue
        for r in json.loads(f.read_text())["data"]["data"]:
            out[date.fromisoformat(r["date"][:10])] = {
                "fii_buy": float(r["fii"]["grossBuy"]), "fii_sell": float(r["fii"]["grossSell"]),
                "fii_net": float(r["fii"]["netBuySell"]),
                "dii_buy": float(r["dii"]["grossBuy"]), "dii_sell": float(r["dii"]["grossSell"]),
                "dii_net": float(r["dii"]["netBuySell"]),
            }
    return out


def parse_stockezee(page: Path) -> dict[date, dict]:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page.read_text(), re.S)
    rows = json.loads(m.group(1))["props"]["pageProps"]["initialData"]
    out: dict[date, dict] = {}
    for r in rows:
        nets = {c["category"]: float(c["net_value"]) for c in r.get("fii_cash", [])}
        if "FII/FPI" in nets and "DII" in nets:
            out[date.fromisoformat(r["date"][:10])] = {"fii_net": nets["FII/FPI"], "dii_net": nets["DII"]}
    return out


def parse_trendlyne(raw_dir: Path) -> dict[date, dict]:
    out: dict[date, dict] = {}
    for f in sorted(raw_dir.glob("tl_*.html")):
        for blob in re.findall(r'data-jsondata="([^"]*)"', f.read_text()):
            try:
                t = json.loads(html.unescape(blob))
            except json.JSONDecodeError:
                continue  # empty/non-table attributes
            if not isinstance(t, dict):
                continue
            h = t.get("headers", [])
            if h[:7] != ["date", "FII Gross Purchase", "FII Gross Sales", "FII Net Purchase / Sales",
                         "DII Net Purchase / Sales", "DII Gross Sales", "DII Gross Purchase"]:
                continue
            for r in t["data"]:
                try:
                    d = date.fromisoformat(str(r[0]))
                except ValueError:
                    continue  # "Last 30 Days" etc.
                out[d] = {"fii_buy": r[1], "fii_sell": r[2], "fii_net": r[3],
                          "dii_net": r[4], "dii_sell": r[5], "dii_buy": r[6]}
    return out


# ── verification ──────────────────────────────────────────────────────────────


@dataclass
class Verdict:
    day: date
    status: str                       # VERIFIED | REJECTED | NO_DATA
    values: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def verify(day: date, g: dict | None, s: dict | None, t: dict | None) -> Verdict:
    if g is None and s is None and t is None:
        return Verdict(day, "NO_DATA", reasons=["no source has this date (holiday/weekend)"])
    v = Verdict(day, "VERIFIED", values=dict(g or {}))
    for name, src in (("groww", g), ("stockezee", s), ("trendlyne", t)):
        if src is None:
            v.reasons.append(f"missing in {name}")
    if g:
        for side in ("fii", "dii"):
            if abs(g[f"{side}_buy"] - g[f"{side}_sell"] - g[f"{side}_net"]) > 0.011:
                v.reasons.append(f"groww {side} buy−sell≠net")
    if g and s:
        for k in ("fii_net", "dii_net"):
            if abs(g[k] - s[k]) > 0.005:
                v.reasons.append(f"stockezee {k} {s[k]} ≠ groww {g[k]}")
    if g and t:
        for k in FIELDS:
            if abs(g[k] - float(t[k])) > 0.051:
                v.reasons.append(f"trendlyne {k} {t[k]} ≠ groww {g[k]}")
    if day.weekday() >= 5:
        v.reasons.append("weekend date")
    if v.reasons:
        v.status = "REJECTED"
    return v


def _digest(conn: sqlite3.Connection, sql: str, params=()) -> str:
    rows = conn.execute(sql, params).fetchall()
    return hashlib.sha256(repr([tuple(r) for r in rows]).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="dir with groww/, trendlyne/, stockezee.html")
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--apply", action="store_true", help="write VERIFIED rows (default: dry run)")
    args = ap.parse_args()

    g = parse_groww(args.raw / "groww")
    s = parse_stockezee(args.raw / "stockezee.html")
    t = parse_trendlyne(args.raw / "trendlyne")

    verdicts, d = [], args.start
    while d <= args.end:
        verdicts.append(verify(d, g.get(d), s.get(d), t.get(d)))
        d += timedelta(days=1)
    weekday = [v for v in verdicts if v.day.weekday() < 5]
    ok = [v for v in verdicts if v.status == "VERIFIED"]
    bad = [v for v in verdicts if v.status == "REJECTED"]
    none_wd = [v for v in weekday if v.status == "NO_DATA"]
    print(f"Range {args.start} → {args.end}: VERIFIED {len(ok)}, REJECTED {len(bad)}, "
          f"weekdays with no data {len(none_wd)}: {[str(v.day) for v in none_wd]}")
    for v in bad:
        print(f"  REJECTED {v.day}: {'; '.join(v.reasons)}")

    db_path = (args.db or settings.database.path).resolve()
    db = DatabaseManager(db_path=db_path)
    db.connect()
    have = {r["date"] for r in db.fetch_all(
        "SELECT DISTINCT date FROM fii_dii_activity WHERE segment = 'CASH' AND date BETWEEN ? AND ?",
        (args.start.isoformat(), args.end.isoformat()))}
    to_write = [v for v in ok if v.day.isoformat() not in have]
    print(f"Already present (left untouched): {sorted(have)}; to write: {len(to_write)} dates "
          f"({2 * len(to_write)} rows: FII + DII, segment CASH)")
    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply.")
        return 0 if not bad else 1
    if bad:
        print("Refusing to apply while any date in range is REJECTED.")
        return 1

    stamp = datetime.now(tz=_IST).strftime("%Y%m%d_%H%M%S")
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    bak = backup_dir / f"{db_path.stem}.pre_fii_cash_recovery_{stamp}.db"
    with sqlite3.connect(str(db_path)) as src, sqlite3.connect(str(bak)) as dst:
        src.backup(dst)
        assert dst.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    print(f"Backup: {bak}")

    conn = db._conn()
    other_before = _digest(conn, "SELECT * FROM fii_dii_activity WHERE NOT (segment='CASH' AND date BETWEEN ? AND ?) ORDER BY id",
                           (args.start.isoformat(), args.end.isoformat()))
    existing_before = _digest(conn, "SELECT * FROM fii_dii_activity ORDER BY id")
    rows = []
    for v in to_write:
        x = v.values
        rows.append((v.day.isoformat(), "FII", x["fii_buy"], x["fii_sell"], x["fii_net"], "CASH"))
        rows.append((v.day.isoformat(), "DII", x["dii_buy"], x["dii_sell"], x["dii_net"], "CASH"))
    db.execute_many(Q.INSERT_FII_DII_ACTIVITY, rows)

    other_after = _digest(conn, "SELECT * FROM fii_dii_activity WHERE NOT (segment='CASH' AND date BETWEEN ? AND ?) ORDER BY id",
                          (args.start.isoformat(), args.end.isoformat()))
    pre_existing_after = _digest(
        conn, "SELECT * FROM fii_dii_activity WHERE id <= (SELECT MAX(id) FROM fii_dii_activity) - ? ORDER BY id",
        (len(rows),))
    print(f"rows outside the recovered range unchanged: {other_before == other_after}")
    print(f"all pre-existing rows unchanged: {existing_before == pre_existing_after}")

    manifest = backup_dir / f"fii_cash_recovery_manifest_{args.start}_{args.end}_{stamp}.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", *FIELDS, "stockezee_fii_net", "stockezee_dii_net",
                    *[f"trendlyne_{k}" for k in FIELDS], "sources"])
        for v in to_write:
            sv, tv = s[v.day], t[v.day]
            w.writerow([v.day, *[v.values[k] for k in FIELDS], sv["fii_net"], sv["dii_net"],
                        *[tv[k] for k in FIELDS], "groww+stockezee+trendlyne"])
    print(f"Provenance manifest: {manifest}")
    print(f"Wrote {len(rows)} rows for {len(to_write)} dates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
