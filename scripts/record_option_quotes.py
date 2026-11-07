#!/usr/bin/env python3
"""
Record real option quotes (bid / ask / LTP / OI) around the money, once a minute.

Why: every option P&L figure so far assumes costs (0.10 R per trade) and a
Black-Scholes premium path.  Spread, theta and fill quality are measurable,
but only from real quotes — and the production DB has no market-hours option
history.  This recorder builds that history in its OWN SQLite file, so it
touches neither the production DB nor its migrations.

For each index it records the nearest tradeable expiry (rolls after 15:30 on
expiry day), the strikes ATM ± N and both CE and PE, from Angel One
getMarketData(FULL): last price, best bid/ask with quantities, open interest,
volume, plus the index spot.  Two REST calls per minute in total.

    python scripts/record_option_quotes.py --once          # one sample now (smoke test)
    python scripts/record_option_quotes.py --daemon        # all trading days 09:15-15:30 IST

It opens its own Angel REST session (no WebSocket).  Stop with SIGTERM / Ctrl-C.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sqlite3
import sys
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.constants import MARKET_HOLIDAYS  # noqa: E402
from src.data.broker.angel_one import INDEX_REST_TOKENS, AngelOneClient  # noqa: E402

logger = logging.getLogger("record_option_quotes")
_IST = ZoneInfo("Asia/Kolkata")
DEFAULT_DB = ROOT / "data" / "forward_test" / "option_quotes.db"
OPEN, CLOSE = dtime(9, 15), dtime(15, 30)
MAX_TOKENS = 50
SEGMENT = {"NIFTY50": "NFO", "BANKNIFTY": "NFO", "SENSEX": "BFO"}
MASTER_NAME = {"NIFTY50": "NIFTY", "BANKNIFTY": "BANKNIFTY", "SENSEX": "SENSEX"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS option_quotes (
    ts          TEXT    NOT NULL,       -- sample time, IST ISO-8601
    index_id    TEXT    NOT NULL,
    expiry      TEXT    NOT NULL,
    strike      REAL    NOT NULL,
    opt_type    TEXT    NOT NULL,
    token       TEXT    NOT NULL,
    lot_size    INTEGER,
    spot        REAL,
    ltp         REAL,
    bid         REAL,
    ask         REAL,
    bid_qty     INTEGER,
    ask_qty     INTEGER,
    oi          INTEGER,                -- contracts (units / lot size)
    volume      INTEGER,                -- contracts
    PRIMARY KEY (ts, index_id, expiry, strike, opt_type)
);
CREATE INDEX IF NOT EXISTS idx_oq_idx_ts ON option_quotes (index_id, ts);
"""


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in MARKET_HOLIDAYS


def next_open(now: datetime) -> datetime:
    d = now.date()
    cand = datetime.combine(d, dtime(9, 14), tzinfo=_IST)
    if now >= cand:
        d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return datetime.combine(d, dtime(9, 14), tzinfo=_IST)


class Recorder:
    def __init__(self, db: Path, indices: list[str], strikes_each_side: int) -> None:
        self.db_path = db
        self.indices = indices
        self.n = strikes_each_side
        self.client = AngelOneClient()
        self._master: dict[str, object] = {}
        self._master_day: date | None = None
        db.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(db)
        self.con.executescript(SCHEMA)

    # ── instrument master ──────────────────────────────────────────────
    def _options(self, index_id: str, now: datetime):
        today = now.date()
        if self._master_day != today:
            self._master = {}
            self._master_day = today
        seg = SEGMENT[index_id]
        if seg not in self._master:
            df = self.client.get_master_contract(seg)
            if df is None:
                return None
            df = df[df["instrumenttype"] == "OPTIDX"].copy()
            df["expiry_dt"] = df["expiry"].map(lambda s: datetime.strptime(s, "%d%b%Y").date())
            df["strike_val"] = df["strike"].astype(float) / 100.0
            self._master[seg] = df
        df = self._master[seg]
        return df[df["name"] == MASTER_NAME[index_id]]

    def _pick(self, index_id: str, spot: float, now: datetime):
        opts = self._options(index_id, now)
        if opts is None or opts.empty:
            return None
        first_day = now.date() + (timedelta(days=1) if now.time() >= CLOSE else timedelta(0))
        live = opts[opts["expiry_dt"] >= first_day]
        if live.empty:
            return None
        expiry = live["expiry_dt"].min()
        chain = live[live["expiry_dt"] == expiry]
        strikes = sorted(chain["strike_val"].unique())
        atm_i = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
        keep = strikes[max(0, atm_i - self.n): atm_i + self.n + 1]
        return expiry, chain[chain["strike_val"].isin(keep)]

    # ── one sample ─────────────────────────────────────────────────────
    def sample(self, now: datetime | None = None) -> int:
        now = now or datetime.now(tz=_IST)
        ts = now.replace(second=0, microsecond=0).isoformat()
        spots = {}
        for idx in self.indices:
            exch, tok = INDEX_REST_TOKENS[idx]
            data = self.client.get_market_data("LTP", {exch: [tok]})
            fetched = (data or {}).get("fetched") or []
            if fetched:
                spots[idx] = float(fetched[0].get("ltp") or 0)
        plan = {}      # segment -> tokens ; token -> meta
        meta: dict[str, tuple] = {}
        for idx, spot in spots.items():
            if spot <= 0:
                continue
            picked = self._pick(idx, spot, now)
            if picked is None:
                continue
            expiry, rows = picked
            for r in rows.itertuples(index=False):
                opt = str(r.symbol)[-2:]
                if opt not in ("CE", "PE"):
                    continue
                plan.setdefault(SEGMENT[idx], []).append(str(r.token))
                meta[str(r.token)] = (idx, expiry.isoformat(), float(r.strike_val), opt,
                                      int(float(r.lotsize or 1)) or 1, spot)
        written = 0
        for seg, tokens in plan.items():
            for i in range(0, len(tokens), MAX_TOKENS):
                if i:
                    time.sleep(0.4)
                batch = self.client.get_market_data("FULL", {seg: tokens[i:i + MAX_TOKENS]})
                for q in (batch or {}).get("fetched") or []:
                    m = meta.get(str(q.get("symbolToken")))
                    if not m:
                        continue
                    idx, expiry, strike, opt, lot, spot = m
                    depth = q.get("depth") or {}
                    bid = (depth.get("buy") or [{}])[0]
                    ask = (depth.get("sell") or [{}])[0]
                    self.con.execute(
                        "INSERT OR REPLACE INTO option_quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (ts, idx, expiry, strike, opt, str(q.get("symbolToken")), lot, spot,
                         float(q.get("ltp") or 0), float(bid.get("price") or 0), float(ask.get("price") or 0),
                         int(bid.get("quantity") or 0), int(ask.get("quantity") or 0),
                         int(float(q.get("opnInterest") or 0) / lot),
                         int(float(q.get("tradeVolume") or 0) / lot)),
                    )
                    written += 1
        self.con.commit()
        return written

    def login(self) -> bool:
        return bool(self.client.login())


def run_daemon(rec: Recorder, interval: int, stop: dict) -> None:
    while not stop["flag"]:
        now = datetime.now(tz=_IST)
        if is_trading_day(now.date()) and OPEN <= now.time() < CLOSE:
            t0 = time.monotonic()
            try:
                n = rec.sample(now)
                logger.info("%s sampled %d quotes", now.strftime("%H:%M"), n)
            except Exception:
                logger.exception("sample failed; re-login next cycle")
                try:
                    rec.login()
                except Exception:
                    logger.exception("re-login failed")
            sleep_for = max(1.0, interval - (time.monotonic() - t0))
        else:
            wake = next_open(now)
            logger.info("market closed; sleeping until %s", wake.strftime("%a %d-%b %H:%M"))
            sleep_for = min(900.0, max(30.0, (wake - now).total_seconds()))
            if now.time() >= CLOSE or now.time() < OPEN or not is_trading_day(now.date()):
                try:
                    rec.con.commit()
                except sqlite3.Error:
                    pass
        end = time.monotonic() + sleep_for
        while time.monotonic() < end and not stop["flag"]:
            time.sleep(1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--indices", default="NIFTY50,BANKNIFTY,SENSEX")
    ap.add_argument("--strikes", type=int, default=3, help="strikes each side of ATM")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--daemon", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    indices = [i.strip().upper() for i in args.indices.split(",") if i.strip()]
    rec = Recorder(args.db, indices, args.strikes)
    if not rec.login():
        logger.error("Angel One login failed")
        return 1
    if args.once:
        n = rec.sample()
        print(f"wrote {n} quotes to {args.db}")
        return 0
    if args.daemon:
        stop = {"flag": False}
        signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
        signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
        run_daemon(rec, args.interval, stop)
        rec.con.commit()
        return 0
    ap.error("use --once or --daemon")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
