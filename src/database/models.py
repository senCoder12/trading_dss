"""
Database table definitions.

Each dataclass maps one-to-one to a SQLite table.  The companion
``TABLE_DDL`` dict (keyed by table name) contains the authoritative
CREATE TABLE statement for that model — ``migrations.py`` uses it to
build the initial schema so the definitions are never duplicated.

All timestamps are stored as ISO-8601 strings in IST
(``Asia/Kolkata``).  Boolean columns are stored as INTEGER (0/1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Dataclass models
# ---------------------------------------------------------------------------


@dataclass
class IndexMaster:
    """
    Static metadata for one tradeable Indian index.

    Populated once from ``indices.json`` and refreshed on config reload.
    """

    id: str                          # e.g. "NIFTY50"  — PRIMARY KEY
    display_name: str
    nse_symbol: Optional[str]
    yahoo_symbol: Optional[str]
    exchange: str                    # "NSE" | "BSE"
    lot_size: Optional[int]
    has_options: bool
    option_symbol: Optional[str]
    sector_category: str
    is_active: bool
    created_at: str                  # ISO-8601 IST
    updated_at: str                  # ISO-8601 IST


@dataclass
class PriceData:
    """
    OHLCV bar for any timeframe — live intraday tick through daily EOD.

    ``UNIQUE(index_id, timestamp, timeframe)`` prevents duplicate ingestion.
    """

    id: Optional[int]                # AUTOINCREMENT PK
    index_id: str                    # FK → index_master.id
    timestamp: str                   # ISO-8601 IST bar-open time
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float]
    source: str                      # "nse_live" | "bse_live" | "yfinance"
    timeframe: str                   # "1m" | "5m" | "15m" | "1h" | "1d"


@dataclass
class OptionsChainSnapshot:
    """
    Single strike-level options chain row.

    One snapshot = one (index, timestamp, expiry, strike, type) combination.
    The ``UNIQUE`` constraint ensures idempotent inserts via ``INSERT OR REPLACE``.
    """

    id: Optional[int]
    index_id: str
    timestamp: str
    expiry_date: str                 # "YYYY-MM-DD"
    strike_price: float
    option_type: str                 # "CE" | "PE"
    open_interest: int
    oi_change: int
    volume: int
    ltp: float
    iv: Optional[float]              # implied volatility (%)
    bid_price: Optional[float]
    ask_price: Optional[float]


@dataclass
class OIAggregated:
    """
    Per-expiry OI summary computed from ``options_chain_snapshot``.

    Stored separately so the dashboard can read aggregates without
    re-scanning the full chain table on every request.
    """

    id: Optional[int]
    index_id: str
    timestamp: str
    expiry_date: str
    total_ce_oi: int
    total_pe_oi: int
    total_ce_oi_change: int
    total_pe_oi_change: int
    pcr: float                       # total_pe_oi / total_ce_oi
    max_pain_strike: float
    highest_ce_oi_strike: float
    highest_pe_oi_strike: float


@dataclass
class TechnicalIndicators:
    """
    Pre-computed technical indicator values for one (index, timeframe, timestamp).

    Stored so the API can serve indicator values without re-computing on
    every request.
    """

    id: Optional[int]
    index_id: str
    timestamp: str
    timeframe: str
    vwap: Optional[float]
    ema_20: Optional[float]
    ema_50: Optional[float]
    rsi_14: Optional[float]
    support_1: Optional[float]
    resistance_1: Optional[float]
    support_2: Optional[float]
    resistance_2: Optional[float]
    avg_volume_20: Optional[float]
    technical_signal: str            # "BULLISH" | "BEARISH" | "NEUTRAL"
    signal_strength: float           # 0.0 – 1.0


@dataclass
class NewsArticle:
    """
    Ingested news article with raw and adjusted sentiment.
    """

    id: Optional[int]
    title: str
    summary: Optional[str]
    source: str
    url: str                         # UNIQUE — deduplication key
    published_at: str
    fetched_at: str
    raw_sentiment_score: float       # VADER compound: −1 to +1
    adjusted_sentiment: float        # after domain keyword boosting
    impact_category: str             # "CRITICAL"|"HIGH"|"MEDIUM"|"LOW"|"NOISE"
    source_credibility: float        # 0.0 – 1.0
    is_processed: bool               # mapped to index_impact yet?


@dataclass
class NewsIndexImpact:
    """
    Many-to-many mapping of news articles to affected indices.
    """

    id: Optional[int]
    news_id: int                     # FK → news_articles.id
    index_id: str                    # FK → index_master.id
    relevance_score: float           # 0.0 – 1.0
    mapped_via: str                  # "keyword" | "sector" | "direct"


@dataclass
class AnomalyEvent:
    """
    Detected market anomaly — volume/price anomalies, OI spikes, breakouts, etc.

    Phase 4 extended fields:
    - ``category``: broad bucket for filtering ("VOLUME" | "PRICE" | "OI" | "OTHER")
    - ``message``: human-readable one-liner for dashboards and alerts
    """

    id: Optional[int]
    index_id: str
    timestamp: str
    anomaly_type: str                # e.g. "VOLUME_SPIKE" | "ABSORPTION" | "GAP_UP" | …
    severity: str                    # "HIGH" | "MEDIUM" | "LOW"
    category: str                    # "VOLUME" | "PRICE" | "OI" | "OTHER"
    details: str                     # JSON string with detector-specific numbers
    message: str                     # Human-readable alert text
    cooldown_key: str                # Dedup key for cooldown logic
    is_active: bool


@dataclass
class TradingSignal:
    """
    Composite trading signal with per-component votes and outcome tracking.
    """

    id: Optional[int]
    index_id: str
    generated_at: str
    signal_type: str                 # "BUY_CALL" | "BUY_PUT" | "NO_TRADE"
    confidence_level: str            # "HIGH" | "MEDIUM" | "LOW"
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    regime: str                      # "TRENDING" | "EVENT" | "RANGE_BOUND"
    technical_vote: str
    options_vote: str
    news_vote: str
    anomaly_vote: str
    reasoning: str                   # JSON with explanation dict
    outcome: Optional[str]           # "WIN" | "LOSS" | "OPEN" | "EXPIRED"
    actual_exit_price: Optional[float]
    actual_pnl: Optional[float]
    closed_at: Optional[str]


@dataclass
class FIIDIIActivity:
    """
    Daily FII / DII buy-sell activity by segment.
    """

    id: Optional[int]
    date: str                        # "YYYY-MM-DD"
    category: str                    # "FII" | "DII"
    buy_value: float                 # ₹ crore
    sell_value: float
    net_value: float
    segment: str                     # "CASH" | "FO"


@dataclass
class VIXData:
    """India VIX readings."""

    id: Optional[int]
    timestamp: str
    vix_value: float
    vix_change: float
    vix_change_pct: float


@dataclass
class SystemHealth:
    """
    Per-component health check record.

    Written after every data-collection cycle so dashboards can surface
    stale-data warnings.
    """

    id: Optional[int]
    timestamp: str
    component: str                   # "nse_scraper" | "bse_scraper" | "news_engine" …
    status: str                      # "OK" | "WARNING" | "ERROR"
    message: Optional[str]
    response_time_ms: Optional[int]


@dataclass
class SchemaVersion:
    """Migration version tracking."""

    version: int
    applied_at: str
    description: str


# ---------------------------------------------------------------------------
# CREATE TABLE statements  (single source of truth for DDL)
# ---------------------------------------------------------------------------

TABLE_DDL: dict[str, list[str]] = {
    # ── schema version ───────────────────────────────────────────────────────
    "schema_version": [
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version     INTEGER PRIMARY KEY,
            applied_at  TEXT    NOT NULL,
            description TEXT    NOT NULL DEFAULT ''
        )
        """,
    ],

    # ── index master ─────────────────────────────────────────────────────────
    "index_master": [
        """
        CREATE TABLE IF NOT EXISTS index_master (
            id               TEXT    PRIMARY KEY,
            display_name     TEXT    NOT NULL,
            nse_symbol       TEXT,
            yahoo_symbol     TEXT,
            exchange         TEXT    NOT NULL CHECK (exchange IN ('NSE','BSE')),
            lot_size         INTEGER,
            has_options      INTEGER NOT NULL DEFAULT 0,
            option_symbol    TEXT,
            sector_category  TEXT    NOT NULL DEFAULT 'unknown',
            is_active        INTEGER NOT NULL DEFAULT 1,
            created_at       TEXT    NOT NULL,
            updated_at       TEXT    NOT NULL
        )
        """,
    ],

    # ── price data ────────────────────────────────────────────────────────────
    "price_data": [
        """
        CREATE TABLE IF NOT EXISTS price_data (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id   TEXT    NOT NULL REFERENCES index_master(id),
            timestamp  TEXT    NOT NULL,
            open       REAL    NOT NULL,
            high       REAL    NOT NULL,
            low        REAL    NOT NULL,
            close      REAL    NOT NULL,
            volume     REAL    NOT NULL DEFAULT 0,
            vwap       REAL,
            source     TEXT    NOT NULL DEFAULT 'nse_live',
            timeframe  TEXT    NOT NULL DEFAULT '1d',
            UNIQUE (index_id, timestamp, timeframe)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_price_id_ts     ON price_data (index_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_price_id_tf_ts  ON price_data (index_id, timeframe, timestamp)",
    ],

    # ── options chain snapshot ────────────────────────────────────────────────
    "options_chain_snapshot": [
        """
        CREATE TABLE IF NOT EXISTS options_chain_snapshot (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id       TEXT    NOT NULL REFERENCES index_master(id),
            timestamp      TEXT    NOT NULL,
            expiry_date    TEXT    NOT NULL,
            strike_price   REAL    NOT NULL,
            option_type    TEXT    NOT NULL CHECK (option_type IN ('CE','PE')),
            open_interest  INTEGER NOT NULL DEFAULT 0,
            oi_change      INTEGER NOT NULL DEFAULT 0,
            volume         INTEGER NOT NULL DEFAULT 0,
            ltp            REAL    NOT NULL DEFAULT 0,
            iv             REAL,
            bid_price      REAL,
            ask_price      REAL,
            UNIQUE (index_id, timestamp, expiry_date, strike_price, option_type)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ocs_id_exp_ts ON options_chain_snapshot (index_id, expiry_date, timestamp)",
    ],

    # ── OI aggregated ─────────────────────────────────────────────────────────
    "oi_aggregated": [
        """
        CREATE TABLE IF NOT EXISTS oi_aggregated (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id              TEXT    NOT NULL REFERENCES index_master(id),
            timestamp             TEXT    NOT NULL,
            expiry_date           TEXT    NOT NULL,
            total_ce_oi           INTEGER NOT NULL DEFAULT 0,
            total_pe_oi           INTEGER NOT NULL DEFAULT 0,
            total_ce_oi_change    INTEGER NOT NULL DEFAULT 0,
            total_pe_oi_change    INTEGER NOT NULL DEFAULT 0,
            pcr                   REAL    NOT NULL DEFAULT 0,
            max_pain_strike       REAL    NOT NULL DEFAULT 0,
            highest_ce_oi_strike  REAL    NOT NULL DEFAULT 0,
            highest_pe_oi_strike  REAL    NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_oia_id_ts ON oi_aggregated (index_id, timestamp)",
    ],

    # ── technical indicators ──────────────────────────────────────────────────
    "technical_indicators": [
        """
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id         TEXT    NOT NULL REFERENCES index_master(id),
            timestamp        TEXT    NOT NULL,
            timeframe        TEXT    NOT NULL,
            vwap             REAL,
            ema_20           REAL,
            ema_50           REAL,
            rsi_14           REAL,
            support_1        REAL,
            resistance_1     REAL,
            support_2        REAL,
            resistance_2     REAL,
            avg_volume_20    REAL,
            technical_signal TEXT    NOT NULL DEFAULT 'NEUTRAL'
                             CHECK (technical_signal IN ('BULLISH','BEARISH','NEUTRAL')),
            signal_strength  REAL    NOT NULL DEFAULT 0
                             CHECK (signal_strength BETWEEN 0 AND 1),
            UNIQUE (index_id, timestamp, timeframe)
        )
        """,
    ],

    # ── news articles ─────────────────────────────────────────────────────────
    "news_articles": [
        """
        CREATE TABLE IF NOT EXISTS news_articles (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            title                TEXT    NOT NULL,
            summary              TEXT,
            source               TEXT    NOT NULL,
            url                  TEXT    NOT NULL UNIQUE,
            published_at         TEXT    NOT NULL,
            fetched_at           TEXT    NOT NULL,
            raw_sentiment_score  REAL    NOT NULL DEFAULT 0,
            adjusted_sentiment   REAL    NOT NULL DEFAULT 0,
            impact_category      TEXT    NOT NULL DEFAULT 'NOISE'
                                 CHECK (impact_category IN ('CRITICAL','HIGH','MEDIUM','LOW','NOISE')),
            source_credibility   REAL    NOT NULL DEFAULT 0.7,
            is_processed         INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles (published_at)",
        "CREATE INDEX IF NOT EXISTS idx_news_impact    ON news_articles (impact_category)",
    ],

    # ── news index impact ─────────────────────────────────────────────────────
    "news_index_impact": [
        """
        CREATE TABLE IF NOT EXISTS news_index_impact (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            news_id         INTEGER NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
            index_id        TEXT    NOT NULL REFERENCES index_master(id),
            relevance_score REAL    NOT NULL DEFAULT 0,
            mapped_via      TEXT    NOT NULL DEFAULT 'keyword'
                            CHECK (mapped_via IN ('keyword','sector','direct')),
            UNIQUE (news_id, index_id)
        )
        """,
    ],

    # ── anomaly events ────────────────────────────────────────────────────────
    # Phase 4: anomaly_type is open-ended (no CHECK) to allow new detector types
    # without further migrations.  category groups them for fast dashboard filtering.
    "anomaly_events": [
        """
        CREATE TABLE IF NOT EXISTS anomaly_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id     TEXT    NOT NULL REFERENCES index_master(id),
            timestamp    TEXT    NOT NULL,
            anomaly_type TEXT    NOT NULL,
            severity     TEXT    NOT NULL DEFAULT 'MEDIUM'
                         CHECK (severity IN ('LOW','MEDIUM','HIGH','EXTREME')),
            category     TEXT    NOT NULL DEFAULT 'OTHER'
                         CHECK (category IN ('VOLUME','PRICE','OI','FII_DII','DIVERGENCE','OTHER')),
            details      TEXT    NOT NULL DEFAULT '{}',
            message      TEXT    NOT NULL DEFAULT '',
            cooldown_key TEXT    NOT NULL DEFAULT '',
            is_active    INTEGER NOT NULL DEFAULT 1
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_anomaly_id_ts       ON anomaly_events (index_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_anomaly_type_active ON anomaly_events (anomaly_type, is_active)",
        "CREATE INDEX IF NOT EXISTS idx_anomaly_cat_active  ON anomaly_events (category, is_active)",
    ],

    # ── trading signals ───────────────────────────────────────────────────────
    "trading_signals": [
        """
        CREATE TABLE IF NOT EXISTS trading_signals (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id           TEXT    NOT NULL REFERENCES index_master(id),
            generated_at       TEXT    NOT NULL,
            signal_type        TEXT    NOT NULL
                               CHECK (signal_type IN ('BUY_CALL','BUY_PUT','NO_TRADE')),
            confidence_level   TEXT    NOT NULL
                               CHECK (confidence_level IN ('HIGH','MEDIUM','LOW')),
            entry_price        REAL    NOT NULL DEFAULT 0,
            target_price       REAL    NOT NULL DEFAULT 0,
            stop_loss          REAL    NOT NULL DEFAULT 0,
            risk_reward_ratio  REAL    NOT NULL DEFAULT 0,
            regime             TEXT    NOT NULL DEFAULT 'RANGE_BOUND'
                               CHECK (regime IN ('TRENDING','EVENT','RANGE_BOUND')),
            technical_vote     TEXT    NOT NULL DEFAULT 'NEUTRAL',
            options_vote       TEXT    NOT NULL DEFAULT 'NEUTRAL',
            news_vote          TEXT    NOT NULL DEFAULT 'NEUTRAL',
            anomaly_vote       TEXT    NOT NULL DEFAULT 'NEUTRAL',
            reasoning          TEXT    NOT NULL DEFAULT '{}',
            outcome            TEXT
                               CHECK (outcome IS NULL OR outcome IN ('WIN','LOSS','OPEN','EXPIRED')),
            actual_exit_price  REAL,
            actual_pnl         REAL,
            closed_at          TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_sig_id_ts     ON trading_signals (index_id, generated_at)",
        "CREATE INDEX IF NOT EXISTS idx_sig_confidence ON trading_signals (confidence_level)",
    ],

    # ── FII/DII activity ──────────────────────────────────────────────────────
    "fii_dii_activity": [
        """
        CREATE TABLE IF NOT EXISTS fii_dii_activity (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT    NOT NULL,
            category    TEXT    NOT NULL CHECK (category IN ('FII','DII')),
            buy_value   REAL    NOT NULL DEFAULT 0,
            sell_value  REAL    NOT NULL DEFAULT 0,
            net_value   REAL    NOT NULL DEFAULT 0,
            segment     TEXT    NOT NULL DEFAULT 'CASH'
                        CHECK (segment IN ('CASH','FO')),
            UNIQUE (date, category, segment)
        )
        """,
    ],

    # ── VIX data ──────────────────────────────────────────────────────────────
    "vix_data": [
        """
        CREATE TABLE IF NOT EXISTS vix_data (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            vix_value       REAL    NOT NULL,
            vix_change      REAL    NOT NULL DEFAULT 0,
            vix_change_pct  REAL    NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_vix_ts ON vix_data (timestamp)",
    ],

    # ── system health ─────────────────────────────────────────────────────────
    "system_health": [
        """
        CREATE TABLE IF NOT EXISTS system_health (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            component        TEXT    NOT NULL,
            status           TEXT    NOT NULL DEFAULT 'OK'
                             CHECK (status IN ('OK','WARNING','ERROR')),
            message          TEXT,
            response_time_ms INTEGER
        )
        """,
    ],
}

# Ordered list of table names to ensure FK-safe creation order
TABLE_CREATION_ORDER: list[str] = [
    "schema_version",
    "index_master",
    "price_data",
    "options_chain_snapshot",
    "oi_aggregated",
    "technical_indicators",
    "news_articles",
    "news_index_impact",
    "anomaly_events",
    "trading_signals",
    "fii_dii_activity",
    "vix_data",
    "system_health",
]


# ================================================================
# EXPECTED SCHEMA — Source of truth for schema validation
# Maps table_name → {column_name: column_type}
# Update this whenever adding a migration that changes schema.
# ================================================================

EXPECTED_SCHEMA: dict[str, dict[str, str]] = {
    "index_master": {
        "id": "TEXT", "display_name": "TEXT", "nse_symbol": "TEXT",
        "yahoo_symbol": "TEXT", "exchange": "TEXT", "lot_size": "INTEGER",
        "has_options": "INTEGER", "option_symbol": "TEXT",
        "sector_category": "TEXT", "is_active": "INTEGER",
        "created_at": "TEXT", "updated_at": "TEXT",
    },
    "price_data": {
        "id": "INTEGER", "index_id": "TEXT", "timestamp": "TEXT",
        "open": "REAL", "high": "REAL", "low": "REAL", "close": "REAL",
        "volume": "REAL", "vwap": "REAL", "source": "TEXT", "timeframe": "TEXT",
    },
    "options_chain_snapshot": {
        "id": "INTEGER", "index_id": "TEXT", "timestamp": "TEXT",
        "expiry_date": "TEXT", "strike_price": "REAL", "option_type": "TEXT",
        "open_interest": "INTEGER", "oi_change": "INTEGER", "volume": "INTEGER",
        "ltp": "REAL", "iv": "REAL", "bid_price": "REAL", "ask_price": "REAL",
    },
    "oi_aggregated": {
        "id": "INTEGER", "index_id": "TEXT", "timestamp": "TEXT",
        "expiry_date": "TEXT", "total_ce_oi": "INTEGER", "total_pe_oi": "INTEGER",
        "total_ce_oi_change": "INTEGER", "total_pe_oi_change": "INTEGER",
        "pcr": "REAL", "max_pain_strike": "REAL",
        "highest_ce_oi_strike": "REAL", "highest_pe_oi_strike": "REAL",
    },
    "technical_indicators": {
        "id": "INTEGER", "index_id": "TEXT", "timestamp": "TEXT",
        "timeframe": "TEXT", "vwap": "REAL", "ema_20": "REAL", "ema_50": "REAL",
        "rsi_14": "REAL", "support_1": "REAL", "resistance_1": "REAL",
        "support_2": "REAL", "resistance_2": "REAL", "avg_volume_20": "REAL",
        "technical_signal": "TEXT", "signal_strength": "REAL",
    },
    "news_articles": {
        "id": "INTEGER", "title": "TEXT", "summary": "TEXT",
        "source": "TEXT", "url": "TEXT", "published_at": "TEXT",
        "fetched_at": "TEXT", "raw_sentiment_score": "REAL",
        "adjusted_sentiment": "REAL", "impact_category": "TEXT",
        "source_credibility": "REAL", "is_processed": "INTEGER",
    },
    "news_index_impact": {
        "id": "INTEGER", "news_id": "INTEGER", "index_id": "TEXT",
        "relevance_score": "REAL", "mapped_via": "TEXT",
    },
    "anomaly_events": {
        "id": "INTEGER", "index_id": "TEXT", "timestamp": "TEXT",
        "anomaly_type": "TEXT", "severity": "TEXT", "category": "TEXT",
        "details": "TEXT", "message": "TEXT", "cooldown_key": "TEXT",
        "is_active": "INTEGER",
    },
    "trading_signals": {
        "id": "INTEGER", "index_id": "TEXT", "generated_at": "TEXT",
        "signal_type": "TEXT", "confidence_level": "TEXT",
        "entry_price": "REAL", "target_price": "REAL",
        "stop_loss": "REAL", "risk_reward_ratio": "REAL",
        "regime": "TEXT", "technical_vote": "TEXT", "options_vote": "TEXT",
        "news_vote": "TEXT", "anomaly_vote": "TEXT",
        "reasoning": "TEXT", "outcome": "TEXT",
        "actual_exit_price": "REAL", "actual_pnl": "REAL", "closed_at": "TEXT",
    },
    "fii_dii_activity": {
        "id": "INTEGER", "date": "TEXT", "category": "TEXT",
        "buy_value": "REAL", "sell_value": "REAL", "net_value": "REAL",
        "segment": "TEXT",
    },
    "vix_data": {
        "id": "INTEGER", "timestamp": "TEXT", "vix_value": "REAL",
        "vix_change": "REAL", "vix_change_pct": "REAL",
    },
    "system_health": {
        "id": "INTEGER", "timestamp": "TEXT", "component": "TEXT",
        "status": "TEXT", "message": "TEXT", "response_time_ms": "INTEGER",
    },
}
