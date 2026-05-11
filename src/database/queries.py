"""
Named SQL query constants, organised by table.

Every SQL statement used anywhere in the codebase lives here.
No ad-hoc SQL in business logic — always import from this module.

Naming convention
-----------------
- ``CREATE_<TABLE>``          — DDL (tables / indexes)
- ``INSERT_<TABLE>``          — INSERT OR REPLACE / INSERT OR IGNORE
- ``UPSERT_<TABLE>``          — INSERT … ON CONFLICT DO UPDATE
- ``UPDATE_<TABLE>_<field>``  — targeted UPDATE
- ``GET_<description>``       — single-row SELECT (fetch_one)
- ``LIST_<description>``      — multi-row SELECT (fetch_all)
- ``AGG_<description>``       — aggregation query
- ``DELETE_<TABLE>_<cond>``   — DELETE statements
"""

from __future__ import annotations

# ============================================================================
# index_master
# ============================================================================

INSERT_INDEX_MASTER = """
INSERT INTO index_master
    (id, display_name, nse_symbol, yahoo_symbol, exchange,
     lot_size, has_options, option_symbol, sector_category,
     is_active, created_at, updated_at)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    display_name    = excluded.display_name,
    nse_symbol      = excluded.nse_symbol,
    yahoo_symbol    = excluded.yahoo_symbol,
    exchange        = excluded.exchange,
    lot_size        = excluded.lot_size,
    has_options     = excluded.has_options,
    option_symbol   = excluded.option_symbol,
    sector_category = excluded.sector_category,
    is_active       = excluded.is_active,
    updated_at      = excluded.updated_at
"""

GET_INDEX_MASTER = """
SELECT * FROM index_master WHERE id = ?
"""

LIST_ALL_INDICES = """
SELECT * FROM index_master ORDER BY exchange, sector_category, id
"""

LIST_ACTIVE_INDICES = """
SELECT * FROM index_master WHERE is_active = 1 ORDER BY exchange, sector_category, id
"""

LIST_FO_INDICES = """
SELECT * FROM index_master WHERE has_options = 1 AND is_active = 1 ORDER BY id
"""

LIST_INDICES_BY_EXCHANGE = """
SELECT * FROM index_master WHERE exchange = ? AND is_active = 1 ORDER BY id
"""

LIST_INDICES_BY_SECTOR = """
SELECT * FROM index_master WHERE sector_category = ? AND is_active = 1 ORDER BY id
"""

# ============================================================================
# price_data
# ============================================================================

INSERT_PRICE_DATA = """
INSERT OR IGNORE INTO price_data
    (index_id, timestamp, open, high, low, close, volume, vwap, source, timeframe)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPSERT_PRICE_DATA = """
INSERT INTO price_data
    (index_id, timestamp, open, high, low, close, volume, vwap, source, timeframe)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(index_id, timestamp, timeframe) DO UPDATE SET
    open      = excluded.open,
    high      = excluded.high,
    low       = excluded.low,
    close     = excluded.close,
    volume    = excluded.volume,
    vwap      = excluded.vwap,
    source    = excluded.source
"""

GET_LATEST_PRICE = """
SELECT * FROM price_data
WHERE index_id = ? AND timeframe = ? AND timestamp <= ?
ORDER BY timestamp DESC
LIMIT 1
"""

GET_LATEST_PRICE_ANY_TF = """
SELECT * FROM price_data
WHERE index_id = ?
ORDER BY timestamp DESC
LIMIT 1
"""

LIST_PRICE_HISTORY = """
SELECT * FROM price_data
WHERE index_id = ?
  AND timeframe = ?
  AND timestamp BETWEEN ? AND ?
ORDER BY timestamp ASC
"""

LIST_PRICE_HISTORY_LIMIT = """
SELECT * FROM price_data
WHERE index_id = ? AND timeframe = ? AND timestamp <= ?
ORDER BY timestamp DESC
LIMIT ?
"""

AGG_DAILY_OHLCV_FROM_MINUTES = """
SELECT
    index_id,
    DATE(timestamp)     AS date,
    MIN(timestamp)      AS bar_open_time,
    FIRST_VALUE(open)   OVER (PARTITION BY index_id, DATE(timestamp) ORDER BY timestamp)
                        AS open,
    MAX(high)           AS high,
    MIN(low)            AS low,
    LAST_VALUE(close)   OVER (PARTITION BY index_id, DATE(timestamp)
                              ORDER BY timestamp
                              ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
                        AS close,
    SUM(volume)         AS volume,
    '1d'                AS timeframe
FROM price_data
WHERE index_id = ?
  AND timeframe = '1m'
  AND timestamp BETWEEN ? AND ?
GROUP BY index_id, DATE(timestamp)
ORDER BY date ASC
"""

DELETE_PRICE_DATA_OLD_INTRADAY = """
DELETE FROM price_data
WHERE timestamp < ? AND timeframe != '1d'
"""

# ============================================================================
# options_chain_snapshot
# ============================================================================

INSERT_OPTIONS_CHAIN = """
INSERT INTO options_chain_snapshot
    (index_id, timestamp, expiry_date, strike_price, option_type,
     open_interest, oi_change, oi_change_pct, volume, 
     ltp, ltp_change, ltp_change_pct, iv, 
     bid_price, bid_qty, ask_price, ask_qty)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(index_id, timestamp, expiry_date, strike_price, option_type)
DO UPDATE SET
    open_interest  = excluded.open_interest,
    oi_change      = excluded.oi_change,
    oi_change_pct  = excluded.oi_change_pct,
    volume         = excluded.volume,
    ltp            = excluded.ltp,
    ltp_change     = excluded.ltp_change,
    ltp_change_pct = excluded.ltp_change_pct,
    iv             = excluded.iv,
    bid_price      = excluded.bid_price,
    bid_qty        = excluded.bid_qty,
    ask_price      = excluded.ask_price,
    ask_qty        = excluded.ask_qty
"""

LIST_OPTIONS_CHAIN_FOR_EXPIRY = """
SELECT * FROM options_chain_snapshot
WHERE index_id = ?
  AND expiry_date = ?
  AND timestamp <= ?
  AND timestamp = (
      SELECT MAX(timestamp) FROM options_chain_snapshot
      WHERE index_id = ? AND expiry_date = ? AND timestamp <= ?
  )
ORDER BY strike_price ASC, option_type ASC
"""

LIST_OPTIONS_CHAIN_AT_TIME = """
SELECT * FROM options_chain_snapshot
WHERE index_id = ?
  AND expiry_date = ?
  AND timestamp BETWEEN ? AND ?
ORDER BY timestamp ASC, strike_price ASC
"""

LIST_EXPIRY_DATES = """
SELECT DISTINCT expiry_date
FROM options_chain_snapshot
WHERE index_id = ?
ORDER BY expiry_date ASC
"""

GET_OPTIONS_CHAIN_LATEST_TS = """
SELECT MAX(timestamp) AS latest_ts
FROM options_chain_snapshot
WHERE index_id = ? AND expiry_date = ?
"""

DELETE_OPTIONS_CHAIN_OLD = """
DELETE FROM options_chain_snapshot WHERE timestamp < ?
"""

# ============================================================================
# oi_aggregated
# ============================================================================

INSERT_OI_AGGREGATED = """
INSERT INTO oi_aggregated
    (index_id, timestamp, expiry_date,
     total_ce_oi, total_pe_oi, total_ce_oi_change, total_pe_oi_change,
     pcr, max_pain_strike, highest_ce_oi_strike, highest_pe_oi_strike)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

GET_LATEST_OI_AGGREGATED = """
SELECT * FROM oi_aggregated
WHERE index_id = ? AND expiry_date = ? AND timestamp <= ?
ORDER BY timestamp DESC
LIMIT 1
"""

LIST_OI_AGGREGATED_HISTORY = """
SELECT * FROM oi_aggregated
WHERE index_id = ?
  AND expiry_date = ?
  AND timestamp BETWEEN ? AND ?
ORDER BY timestamp ASC
"""

AGG_OI_PCR_TREND = """
SELECT
    index_id,
    expiry_date,
    timestamp,
    pcr,
    total_ce_oi,
    total_pe_oi,
    AVG(pcr) OVER (
        PARTITION BY index_id, expiry_date
        ORDER BY timestamp
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS pcr_5bar_avg
FROM oi_aggregated
WHERE index_id = ? AND expiry_date = ?
ORDER BY timestamp ASC
"""

# ============================================================================
# technical_indicators
# ============================================================================

INSERT_TECHNICAL_INDICATORS = """
INSERT INTO technical_indicators
    (index_id, timestamp, timeframe,
     vwap, ema_20, ema_50, rsi_14,
     support_1, resistance_1, support_2, resistance_2,
     avg_volume_20, technical_signal, signal_strength)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(index_id, timestamp, timeframe) DO UPDATE SET
    vwap             = excluded.vwap,
    ema_20           = excluded.ema_20,
    ema_50           = excluded.ema_50,
    rsi_14           = excluded.rsi_14,
    support_1        = excluded.support_1,
    resistance_1     = excluded.resistance_1,
    support_2        = excluded.support_2,
    resistance_2     = excluded.resistance_2,
    avg_volume_20    = excluded.avg_volume_20,
    technical_signal = excluded.technical_signal,
    signal_strength  = excluded.signal_strength
"""

GET_LATEST_TECHNICALS = """
SELECT * FROM technical_indicators
WHERE index_id = ? AND timeframe = ?
ORDER BY timestamp DESC
LIMIT 1
"""

LIST_TECHNICALS_HISTORY = """
SELECT * FROM technical_indicators
WHERE index_id = ?
  AND timeframe = ?
  AND timestamp BETWEEN ? AND ?
ORDER BY timestamp ASC
"""

# ============================================================================
# news_articles
# ============================================================================

INSERT_NEWS_ARTICLE = """
INSERT OR IGNORE INTO news_articles
    (title, summary, source, url, published_at, fetched_at,
     raw_sentiment_score, adjusted_sentiment, impact_category,
     source_credibility, is_processed)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPDATE_NEWS_PROCESSED = """
UPDATE news_articles SET is_processed = 1 WHERE id = ?
"""

UPDATE_NEWS_SENTIMENT = """
UPDATE news_articles
SET raw_sentiment_score = ?,
    adjusted_sentiment  = ?,
    impact_category     = ?
WHERE id = ?
"""

GET_NEWS_BY_URL = """
SELECT * FROM news_articles WHERE url = ?
"""

LIST_RECENT_NEWS = """
SELECT * FROM news_articles
WHERE published_at >= ?
ORDER BY published_at DESC
LIMIT ?
"""

LIST_UNPROCESSED_NEWS = """
SELECT * FROM news_articles
WHERE is_processed = 0
ORDER BY published_at DESC
LIMIT ?
"""

LIST_NEWS_BY_IMPACT = """
SELECT * FROM news_articles
WHERE impact_category = ?
  AND published_at >= ?
ORDER BY published_at DESC
LIMIT ?
"""

DELETE_NEWS_OLD = """
DELETE FROM news_articles WHERE published_at < ?
"""

# ============================================================================
# news_index_impact
# ============================================================================

INSERT_NEWS_INDEX_IMPACT = """
INSERT OR IGNORE INTO news_index_impact
    (news_id, index_id, relevance_score, mapped_via)
VALUES
    (?, ?, ?, ?)
"""

LIST_NEWS_FOR_INDEX = """
SELECT na.*, nii.relevance_score, nii.mapped_via
FROM news_articles na
JOIN news_index_impact nii ON na.id = nii.news_id
WHERE nii.index_id = ?
  AND na.published_at >= ?
ORDER BY na.published_at DESC
LIMIT ?
"""

LIST_INDICES_FOR_NEWS = """
SELECT index_id, relevance_score, mapped_via
FROM news_index_impact
WHERE news_id = ?
ORDER BY relevance_score DESC
"""

AGG_NEWS_SENTIMENT_FOR_INDEX = """
SELECT
    nii.index_id,
    COUNT(na.id)                                        AS article_count,
    AVG(na.adjusted_sentiment * na.source_credibility)  AS weighted_sentiment,
    AVG(na.adjusted_sentiment)                          AS raw_avg_sentiment,
    SUM(CASE WHEN na.adjusted_sentiment > 0.1 THEN 1 ELSE 0 END) AS bullish_count,
    SUM(CASE WHEN na.adjusted_sentiment < -0.1 THEN 1 ELSE 0 END) AS bearish_count
FROM news_articles na
JOIN news_index_impact nii ON na.id = nii.news_id
WHERE nii.index_id = ?
  AND na.published_at >= ?
GROUP BY nii.index_id
"""

# ============================================================================
# anomaly_events
# ============================================================================

INSERT_ANOMALY_EVENT = """
INSERT INTO anomaly_events
    (index_id, timestamp, anomaly_type, severity, category, details, message, is_active)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?)
"""

UPDATE_ANOMALY_DEACTIVATE = """
UPDATE anomaly_events SET is_active = 0 WHERE id = ?
"""

UPDATE_ANOMALY_RESOLVE = """
UPDATE anomaly_events
SET    is_active = 0,
       details   = ?
WHERE  id = ?
"""

LIST_ACTIVE_ANOMALIES_FILTERED = """
SELECT * FROM anomaly_events
WHERE is_active = 1
ORDER BY
    CASE severity WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END,
    timestamp DESC
"""

LIST_ANOMALY_HISTORY = """
SELECT * FROM anomaly_events
WHERE index_id = ?
  AND timestamp >= ?
ORDER BY timestamp DESC
"""

AGG_ANOMALY_STATS = """
SELECT
    COUNT(*)                                AS total,
    category,
    severity,
    index_id,
    anomaly_type
FROM anomaly_events
WHERE timestamp >= ?
GROUP BY category, severity, index_id, anomaly_type
"""

LIST_SHORT_LIVED_ALERTS = """
SELECT id, timestamp, anomaly_type, is_active, details
FROM anomaly_events
WHERE timestamp >= ?
  AND is_active = 0
"""

LIST_ACTIVE_ANOMALIES = """
SELECT * FROM anomaly_events
WHERE is_active = 1
ORDER BY timestamp DESC
"""

LIST_ANOMALIES_FOR_INDEX = """
SELECT * FROM anomaly_events
WHERE index_id = ?
  AND timestamp >= ?
ORDER BY timestamp DESC
LIMIT ?
"""

LIST_ANOMALIES_BY_TYPE = """
SELECT * FROM anomaly_events
WHERE anomaly_type = ? AND is_active = 1
ORDER BY timestamp DESC
"""

LIST_ANOMALIES_BY_CATEGORY = """
SELECT * FROM anomaly_events
WHERE category = ?
  AND is_active = 1
  AND timestamp >= ?
ORDER BY timestamp DESC
LIMIT ?
"""

# Phase 4 — cooldown dedup: fetch the single most recent event for
# a given (index_id, anomaly_type) pair to check whether we are
# within the suppression window.
GET_LATEST_ANOMALY_BY_KEY = """
SELECT id, timestamp, severity
FROM   anomaly_events
WHERE  index_id     = ?
  AND  anomaly_type = ?
ORDER  BY timestamp DESC
LIMIT  1
"""

# ============================================================================
# trading_signals
# ============================================================================

INSERT_TRADING_SIGNAL = """
INSERT INTO trading_signals
    (index_id, generated_at, signal_type, confidence_level,
     entry_price, target_price, stop_loss, risk_reward_ratio,
     regime, technical_vote, options_vote, news_vote, anomaly_vote,
     reasoning, outcome, actual_exit_price, actual_pnl, closed_at)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPDATE_SIGNAL_OUTCOME = """
UPDATE trading_signals
SET outcome            = ?,
    actual_exit_price  = ?,
    actual_pnl         = ?,
    closed_at          = ?
WHERE id = ?
"""

GET_LATEST_SIGNAL = """
SELECT * FROM trading_signals
WHERE index_id = ?
ORDER BY generated_at DESC
LIMIT 1
"""

LIST_SIGNALS_FOR_INDEX = """
SELECT * FROM trading_signals
WHERE index_id = ?
  AND generated_at >= ?
ORDER BY generated_at DESC
LIMIT ?
"""

LIST_SIGNALS_BY_CONFIDENCE = """
SELECT * FROM trading_signals
WHERE confidence_level = ?
  AND generated_at >= ?
ORDER BY generated_at DESC
LIMIT ?
"""

LIST_OPEN_SIGNALS = """
SELECT * FROM trading_signals
WHERE outcome IS NULL OR outcome = 'OPEN'
ORDER BY generated_at DESC
"""

AGG_SIGNAL_PERFORMANCE = """
SELECT
    index_id,
    signal_type,
    confidence_level,
    COUNT(*)                                        AS total,
    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) AS losses,
    AVG(actual_pnl)                                 AS avg_pnl,
    SUM(actual_pnl)                                 AS total_pnl
FROM trading_signals
WHERE outcome IS NOT NULL
  AND index_id = ?
GROUP BY index_id, signal_type, confidence_level
ORDER BY total DESC
"""

COUNT_SIGNALS_TODAY = """
SELECT COUNT(*) AS count
FROM trading_signals
WHERE index_id = ?
  AND DATE(generated_at) = DATE(?)
  AND signal_type IN ('BUY_CALL', 'BUY_PUT')
"""

# ============================================================================
# fii_dii_activity
# ============================================================================

INSERT_FII_DII_ACTIVITY = """
INSERT INTO fii_dii_activity
    (date, category, buy_value, sell_value, net_value, segment)
VALUES
    (?, ?, ?, ?, ?, ?)
ON CONFLICT(date, category, segment) DO UPDATE SET
    buy_value  = excluded.buy_value,
    sell_value = excluded.sell_value,
    net_value  = excluded.net_value
"""

LIST_FII_DII_RECENT = """
SELECT * FROM fii_dii_activity
WHERE date >= ?
ORDER BY date DESC, category ASC
"""

AGG_FII_DII_NET_BY_DATE = """
SELECT
    date,
    SUM(CASE WHEN category = 'FII' THEN net_value ELSE 0 END) AS fii_net,
    SUM(CASE WHEN category = 'DII' THEN net_value ELSE 0 END) AS dii_net,
    SUM(net_value)                                             AS total_net
FROM fii_dii_activity
WHERE segment = 'CASH'
  AND date >= ?
GROUP BY date
ORDER BY date DESC
"""

GET_LATEST_FII_DII_DATE = """
SELECT MAX(date) AS latest_date FROM fii_dii_activity
"""

# ============================================================================
# vix_data
# ============================================================================

INSERT_VIX_DATA = """
INSERT INTO vix_data
    (timestamp, vix_value, vix_change, vix_change_pct)
VALUES
    (?, ?, ?, ?)
"""

GET_LATEST_VIX = """
SELECT * FROM vix_data ORDER BY timestamp DESC LIMIT 1
"""

LIST_VIX_HISTORY = """
SELECT * FROM vix_data
WHERE timestamp BETWEEN ? AND ?
ORDER BY timestamp ASC
"""

AGG_VIX_DAILY_AVG = """
SELECT
    DATE(timestamp) AS date,
    AVG(vix_value)  AS avg_vix,
    MIN(vix_value)  AS min_vix,
    MAX(vix_value)  AS max_vix
FROM vix_data
WHERE timestamp BETWEEN ? AND ?
GROUP BY DATE(timestamp)
ORDER BY date ASC
"""

# ============================================================================
# system_health
# ============================================================================

INSERT_SYSTEM_HEALTH = """
INSERT INTO system_health
    (timestamp, component, status, message, response_time_ms)
VALUES
    (?, ?, ?, ?, ?)
"""

LIST_SYSTEM_HEALTH_RECENT = """
SELECT * FROM system_health
ORDER BY timestamp DESC
LIMIT ?
"""

LIST_SYSTEM_HEALTH_FOR_COMPONENT = """
SELECT * FROM system_health
WHERE component = ?
ORDER BY timestamp DESC
LIMIT ?
"""

GET_LATEST_HEALTH_BY_COMPONENT = """
SELECT component, MAX(timestamp) AS last_seen, status, message
FROM system_health
GROUP BY component
ORDER BY component ASC
"""

DELETE_SYSTEM_HEALTH_OLD = """
DELETE FROM system_health WHERE timestamp < ?
"""

# ============================================================================
# schema_version
# ============================================================================

INSERT_SCHEMA_VERSION = """
INSERT INTO schema_version (version, applied_at, description)
VALUES (?, ?, ?)
"""

GET_SCHEMA_VERSION = """
SELECT version FROM schema_version ORDER BY version DESC LIMIT 1
"""

LIST_ALL_MIGRATIONS = """
SELECT * FROM schema_version ORDER BY version ASC
"""

# ============================================================================
# Backtest / data-replay helpers
# ============================================================================

# Date range available for an index + timeframe
GET_PRICE_DATE_RANGE = """
SELECT MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts,
       COUNT(*) AS bar_count
FROM price_data
WHERE index_id = ? AND timeframe = ?
"""

# All available timeframes for an index
LIST_AVAILABLE_TIMEFRAMES = """
SELECT DISTINCT timeframe FROM price_data
WHERE index_id = ? ORDER BY timeframe
"""

# All index_ids that have price data
LIST_INDICES_WITH_DATA = """
SELECT DISTINCT index_id FROM price_data ORDER BY index_id
"""

# Options chain snapshot nearest to (and <=) a given timestamp
GET_OPTIONS_SNAPSHOT_AT_TIME = """
SELECT * FROM options_chain_snapshot
WHERE index_id = ? AND timestamp <= ?
ORDER BY timestamp DESC
LIMIT 200
"""

# OI aggregated nearest to (and <=) a given timestamp
GET_OI_AGGREGATED_AT_TIME = """
SELECT * FROM oi_aggregated
WHERE index_id = ? AND timestamp <= ?
ORDER BY timestamp DESC
LIMIT 1
"""

# FII/DII for the most recent date strictly BEFORE the given date
GET_FII_DII_BEFORE_DATE = """
SELECT * FROM fii_dii_activity
WHERE date < ? ORDER BY date DESC LIMIT 10
"""

# VIX reading nearest to (and <=) a given timestamp
GET_VIX_AT_TIME = """
SELECT * FROM vix_data
WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1
"""

# Check whether any options data exists in a date range
AGG_OPTIONS_DATA_EXISTS = """
SELECT COUNT(*) AS cnt FROM options_chain_snapshot
WHERE index_id = ? AND timestamp BETWEEN ? AND ?
"""

# Check whether any FII/DII data exists in a date range
AGG_FII_DII_DATA_EXISTS = """
SELECT COUNT(*) AS cnt FROM fii_dii_activity
WHERE date BETWEEN ? AND ?
"""

# Check whether any VIX data exists in a date range
AGG_VIX_DATA_EXISTS = """
SELECT COUNT(*) AS cnt FROM vix_data
WHERE timestamp BETWEEN ? AND ?
"""

# ============================================================================
# fo_universe
# ============================================================================

UPSERT_FO_UNIVERSE = """
INSERT INTO fo_universe
    (symbol, company_name, series, isin, lot_size, underlying_value, sector,
     is_active, last_updated)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol) DO UPDATE SET
    company_name     = excluded.company_name,
    series           = excluded.series,
    isin             = excluded.isin,
    lot_size         = excluded.lot_size,
    underlying_value = excluded.underlying_value,
    sector           = excluded.sector,
    is_active        = excluded.is_active,
    last_updated     = excluded.last_updated
"""

LIST_FO_UNIVERSE_ACTIVE = """
SELECT * FROM fo_universe WHERE is_active = 1 ORDER BY symbol
"""

GET_FO_UNIVERSE_STOCK = """
SELECT * FROM fo_universe WHERE symbol = ?
"""

INSERT_FO_UNIVERSE_HISTORY = """
INSERT INTO fo_universe_history (symbol, event, event_date, lot_size)
VALUES (?, ?, ?, ?)
"""

LIST_FO_UNIVERSE_HISTORY = """
SELECT * FROM fo_universe_history WHERE symbol = ? ORDER BY event_date DESC
"""

# ============================================================================
# fo_avg_volumes
# ============================================================================

UPSERT_FO_AVG_VOLUME = """
INSERT INTO fo_avg_volumes (symbol, avg_volume_20d, last_updated)
VALUES (?, ?, ?)
ON CONFLICT(symbol) DO UPDATE SET
    avg_volume_20d = excluded.avg_volume_20d,
    last_updated   = excluded.last_updated
"""

GET_FO_AVG_VOLUME = """
SELECT avg_volume_20d FROM fo_avg_volumes WHERE symbol = ?
"""

LIST_FO_AVG_VOLUMES = """
SELECT * FROM fo_avg_volumes ORDER BY symbol
"""

DELETE_FO_AVG_VOLUMES = """
DELETE FROM fo_avg_volumes
"""

# ============================================================================
# fo_daily_scores
# ============================================================================

UPSERT_FO_DAILY_SCORE = """
INSERT INTO fo_daily_scores
    (date, symbol, rank, tier, composite_score,
     oi_score, oi_change_score, volume_score,
     price_move_score, iv_change_score, spread_score,
     total_oi, volume_ratio, timestamp)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(date, symbol) DO UPDATE SET
    rank = excluded.rank,
    tier = excluded.tier,
    composite_score = excluded.composite_score,
    oi_score = excluded.oi_score,
    oi_change_score = excluded.oi_change_score,
    volume_score = excluded.volume_score,
    price_move_score = excluded.price_move_score,
    iv_change_score = excluded.iv_change_score,
    spread_score = excluded.spread_score,
    total_oi = excluded.total_oi,
    volume_ratio = excluded.volume_ratio,
    timestamp = excluded.timestamp
"""

LIST_FO_SCORES_BY_DATE = """
SELECT * FROM fo_daily_scores WHERE date = ? ORDER BY rank
"""

LIST_FO_SCORES_BY_DATE_AND_TIER = """
SELECT * FROM fo_daily_scores WHERE date = ? AND tier = ? ORDER BY rank
"""

GET_FO_SCORE = """
SELECT * FROM fo_daily_scores WHERE date = ? AND symbol = ?
"""

LIST_TOP_FO_SCORES = """
SELECT * FROM fo_daily_scores WHERE date = ? AND rank <= ? ORDER BY rank
"""

DELETE_FO_SCORES_BY_DATE = """
DELETE FROM fo_daily_scores WHERE date = ?
"""

# ============================================================================
# fo_universe_snapshots
# ============================================================================

UPSERT_FO_UNIVERSE_SNAPSHOT = """
INSERT INTO fo_universe_snapshots
    (date, timestamp, active_universe, priority_watchlist,
     total_liquid, total_fo_stocks, market_breadth, top_sector,
     filter_stats, is_market_hours, refresh_count)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(date) DO UPDATE SET
    timestamp = excluded.timestamp,
    active_universe = excluded.active_universe,
    priority_watchlist = excluded.priority_watchlist,
    total_liquid = excluded.total_liquid,
    total_fo_stocks = excluded.total_fo_stocks,
    market_breadth = excluded.market_breadth,
    top_sector = excluded.top_sector,
    filter_stats = excluded.filter_stats,
    is_market_hours = excluded.is_market_hours,
    refresh_count = excluded.refresh_count
"""

GET_FO_UNIVERSE_SNAPSHOT = """
SELECT * FROM fo_universe_snapshots WHERE date = ?
"""

GET_LATEST_FO_UNIVERSE_SNAPSHOT = """
SELECT * FROM fo_universe_snapshots ORDER BY date DESC LIMIT 1
"""

DELETE_FO_UNIVERSE_SNAPSHOT = """
DELETE FROM fo_universe_snapshots WHERE date = ?
"""

# ============================================================================
# futures_basis — NIFTY futures LTP vs spot basis
# ============================================================================

CREATE_FUTURES_BASIS = """
CREATE TABLE IF NOT EXISTS futures_basis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    index_id TEXT NOT NULL,
    spot_price REAL NOT NULL,
    futures_ltp REAL NOT NULL,
    basis REAL NOT NULL,
    basis_pct REAL NOT NULL,
    fair_value_basis REAL DEFAULT 0.0,
    expiry_date TEXT,
    days_to_expiry INTEGER DEFAULT 0,
    basis_signal TEXT DEFAULT 'NEUTRAL',
    UNIQUE(timestamp, index_id)
)
"""

UPSERT_FUTURES_BASIS = """
INSERT INTO futures_basis (timestamp, index_id, spot_price, futures_ltp, basis, basis_pct, fair_value_basis, expiry_date, days_to_expiry, basis_signal)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(timestamp, index_id) DO UPDATE SET
    spot_price = excluded.spot_price,
    futures_ltp = excluded.futures_ltp,
    basis = excluded.basis,
    basis_pct = excluded.basis_pct,
    fair_value_basis = excluded.fair_value_basis,
    basis_signal = excluded.basis_signal
"""

GET_LATEST_FUTURES_BASIS = """
SELECT * FROM futures_basis WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1
"""

# ============================================================================
# iv_history — ATM IV snapshots for IV percentile calculation
# ============================================================================

CREATE_IV_HISTORY = """
CREATE TABLE IF NOT EXISTS iv_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    index_id TEXT NOT NULL,
    atm_iv REAL NOT NULL,
    atm_ce_iv REAL DEFAULT 0.0,
    atm_pe_iv REAL DEFAULT 0.0,
    pcr REAL DEFAULT 1.0,
    put_call_skew REAL DEFAULT 0.0,
    atm_straddle_price REAL DEFAULT 0.0,
    UNIQUE(timestamp, index_id)
)
"""

UPSERT_IV_HISTORY = """
INSERT INTO iv_history (timestamp, index_id, atm_iv, atm_ce_iv, atm_pe_iv, pcr, put_call_skew, atm_straddle_price)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(timestamp, index_id) DO UPDATE SET
    atm_iv = excluded.atm_iv,
    atm_ce_iv = excluded.atm_ce_iv,
    atm_pe_iv = excluded.atm_pe_iv,
    pcr = excluded.pcr,
    put_call_skew = excluded.put_call_skew,
    atm_straddle_price = excluded.atm_straddle_price
"""

LIST_IV_HISTORY_30D = """
SELECT atm_iv, timestamp FROM iv_history
WHERE index_id = ? AND timestamp >= datetime('now', '-30 days')
ORDER BY timestamp DESC
"""

# ============================================================================
# market_breadth — Advance/Decline ratio snapshots
# ============================================================================

CREATE_MARKET_BREADTH = """
CREATE TABLE IF NOT EXISTS market_breadth (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    index_id TEXT NOT NULL DEFAULT 'NIFTY50',
    advances INTEGER DEFAULT 0,
    declines INTEGER DEFAULT 0,
    unchanged INTEGER DEFAULT 0,
    ad_ratio REAL DEFAULT 1.0,
    breadth_signal TEXT DEFAULT 'NEUTRAL',
    UNIQUE(timestamp, index_id)
)
"""

UPSERT_MARKET_BREADTH = """
INSERT INTO market_breadth (timestamp, index_id, advances, declines, unchanged, ad_ratio, breadth_signal)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(timestamp, index_id) DO UPDATE SET
    advances = excluded.advances,
    declines = excluded.declines,
    unchanged = excluded.unchanged,
    ad_ratio = excluded.ad_ratio,
    breadth_signal = excluded.breadth_signal
"""

GET_LATEST_MARKET_BREADTH = """
SELECT * FROM market_breadth WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1
"""

# ============================================================================
# gift_nifty — pre-market GIFT NIFTY snapshots
# ============================================================================

CREATE_GIFT_NIFTY = """
CREATE TABLE IF NOT EXISTS gift_nifty (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    ltp REAL NOT NULL,
    change_pts REAL DEFAULT 0.0,
    change_pct REAL DEFAULT 0.0,
    prev_nifty_close REAL DEFAULT 0.0,
    implied_gap_pts REAL DEFAULT 0.0,
    implied_gap_pct REAL DEFAULT 0.0,
    gap_classification TEXT DEFAULT 'FLAT_OPEN',
    wait_15min INTEGER DEFAULT 0,
    UNIQUE(timestamp)
)
"""

UPSERT_GIFT_NIFTY = """
INSERT INTO gift_nifty (timestamp, ltp, change_pts, change_pct, prev_nifty_close, implied_gap_pts, implied_gap_pct, gap_classification, wait_15min)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(timestamp) DO UPDATE SET
    ltp = excluded.ltp,
    change_pts = excluded.change_pts,
    implied_gap_pts = excluded.implied_gap_pts,
    gap_classification = excluded.gap_classification
"""

GET_LATEST_GIFT_NIFTY = """
SELECT * FROM gift_nifty ORDER BY timestamp DESC LIMIT 1
"""

# ── candle_cache (liquidity module) ─────────────────────────────────────────

CREATE_CANDLE_CACHE = """
CREATE TABLE IF NOT EXISTS candle_cache (
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      INTEGER NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp)
)
"""

UPSERT_CANDLE_CACHE = """
INSERT INTO candle_cache (symbol, timeframe, timestamp, open, high, low, close, volume)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol, timeframe, timestamp) DO UPDATE SET
    high   = MAX(candle_cache.high,   excluded.high),
    low    = MIN(candle_cache.low,    excluded.low),
    close  = excluded.close,
    volume = excluded.volume
"""

LIST_CANDLE_CACHE = """
SELECT symbol, timeframe, timestamp, open, high, low, close, volume
FROM candle_cache
WHERE symbol = ?
  AND timeframe = ?
  AND timestamp >= ?
ORDER BY timestamp ASC
LIMIT ?
"""

DELETE_CANDLE_CACHE_OLD = """
DELETE FROM candle_cache WHERE timestamp < ?
"""

# ── liquidity_levels ─────────────────────────────────────────────────────────

CREATE_LIQUIDITY_LEVELS = """
CREATE TABLE IF NOT EXISTS liquidity_levels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    level_type      TEXT    NOT NULL,
    price           REAL    NOT NULL,
    price_key       INTEGER NOT NULL,
    strength        INTEGER NOT NULL DEFAULT 1,
    timeframe       TEXT    NOT NULL,
    first_formed_at TEXT    NOT NULL,
    last_touched_at TEXT    NOT NULL,
    is_swept        INTEGER NOT NULL DEFAULT 0,
    swept_at        TEXT,
    wick_beyond     REAL    NOT NULL DEFAULT 0,
    is_active       INTEGER NOT NULL DEFAULT 1,
    date            TEXT    NOT NULL,
    UNIQUE(symbol, level_type, price_key, date)
)
"""

CREATE_LIQUIDITY_LEVELS_IDX = """
CREATE INDEX IF NOT EXISTS idx_liq_levels_symbol_date
ON liquidity_levels (symbol, date, is_active)
"""

UPSERT_LIQUIDITY_LEVEL = """
INSERT INTO liquidity_levels
    (symbol, level_type, price, price_key, strength, timeframe,
     first_formed_at, last_touched_at, is_swept, swept_at, wick_beyond, is_active, date)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol, level_type, price_key, date) DO UPDATE SET
    price           = excluded.price,
    strength        = MAX(liquidity_levels.strength, excluded.strength),
    last_touched_at = excluded.last_touched_at,
    is_active       = excluded.is_active
"""

UPDATE_LIQUIDITY_LEVEL_SWEPT = """
UPDATE liquidity_levels
SET is_swept  = 1,
    swept_at  = ?,
    wick_beyond = ?
WHERE id = ?
"""

UPDATE_LIQUIDITY_LEVEL_INACTIVE = """
UPDATE liquidity_levels SET is_active = 0
WHERE symbol = ? AND date = ? AND is_swept = 0
"""

LIST_ACTIVE_LIQUIDITY_LEVELS = """
SELECT id, symbol, level_type, price, price_key, strength, timeframe,
       first_formed_at, last_touched_at, is_swept, swept_at, wick_beyond, is_active, date
FROM liquidity_levels
WHERE symbol = ? AND date = ? AND is_active = 1
ORDER BY price ASC
"""

# ── level_touches ─────────────────────────────────────────────────────────────

CREATE_LEVEL_TOUCHES = """
CREATE TABLE IF NOT EXISTS level_touches (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    level_id   INTEGER NOT NULL REFERENCES liquidity_levels(id),
    touch_price REAL   NOT NULL,
    touch_time  TEXT   NOT NULL,
    held        INTEGER NOT NULL DEFAULT 1
)
"""

CREATE_LEVEL_TOUCHES_IDX = """
CREATE INDEX IF NOT EXISTS idx_level_touches_level_id
ON level_touches (level_id)
"""

INSERT_LEVEL_TOUCH = """
INSERT INTO level_touches (level_id, touch_price, touch_time, held)
VALUES (?, ?, ?, ?)
"""

LIST_LEVEL_TOUCHES = """
SELECT id, level_id, touch_price, touch_time, held
FROM level_touches
WHERE level_id = ?
ORDER BY touch_time ASC
"""

DELETE_LEVEL_TOUCHES_OLD = """
DELETE FROM level_touches
WHERE touch_time < ?
"""

# ── liquidity_sweeps ───────────────────────────────────────────────────────────

CREATE_LIQUIDITY_SWEEPS = """
CREATE TABLE IF NOT EXISTS liquidity_sweeps (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol               TEXT    NOT NULL,
    timeframe            TEXT    NOT NULL,
    sweep_type           TEXT    NOT NULL,
    level_type           TEXT    NOT NULL,
    level_price          REAL    NOT NULL,
    sweep_candle_time    TEXT    NOT NULL,
    sweep_high           REAL    NOT NULL,
    sweep_low            REAL    NOT NULL,
    wick_beyond_pct      REAL    NOT NULL,
    signal_quality       TEXT    NOT NULL DEFAULT 'LOW',
    reversion_confirmed  INTEGER NOT NULL DEFAULT 0,
    reversion_candle_time TEXT,
    oi_confirms          INTEGER NOT NULL DEFAULT 0,
    futures_confirms     INTEGER NOT NULL DEFAULT 0,
    date                 TEXT    NOT NULL
)
"""

CREATE_LIQUIDITY_SWEEPS_IDX = """
CREATE INDEX IF NOT EXISTS idx_liq_sweeps_symbol_date
ON liquidity_sweeps (symbol, date, sweep_candle_time)
"""

INSERT_LIQUIDITY_SWEEP = """
INSERT INTO liquidity_sweeps
    (symbol, timeframe, sweep_type, level_type, level_price,
     sweep_candle_time, sweep_high, sweep_low, wick_beyond_pct,
     signal_quality, reversion_confirmed, reversion_candle_time,
     oi_confirms, futures_confirms, date)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

LIST_RECENT_SWEEPS = """
SELECT id, symbol, timeframe, sweep_type, level_type, level_price,
       sweep_candle_time, sweep_high, sweep_low, wick_beyond_pct,
       signal_quality, reversion_confirmed, reversion_candle_time,
       oi_confirms, futures_confirms, date
FROM liquidity_sweeps
WHERE symbol = ? AND sweep_candle_time >= ?
ORDER BY sweep_candle_time DESC
"""

# ── fair_value_gaps ───────────────────────────────────────────────────────────

CREATE_FAIR_VALUE_GAPS = """
CREATE TABLE IF NOT EXISTS fair_value_gaps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    gap_type        TEXT    NOT NULL,
    gap_top         REAL    NOT NULL,
    gap_bottom      REAL    NOT NULL,
    gap_midpoint    REAL    NOT NULL,
    gap_size_pct    REAL    NOT NULL,
    formed_at       TEXT    NOT NULL,
    is_filled       INTEGER NOT NULL DEFAULT 0,
    fill_pct        REAL    NOT NULL DEFAULT 0.0,
    is_mitigated    INTEGER NOT NULL DEFAULT 0,
    date            TEXT    NOT NULL
)
"""

CREATE_FAIR_VALUE_GAPS_IDX = """
CREATE INDEX IF NOT EXISTS idx_fvg_symbol_date
ON fair_value_gaps (symbol, date, is_mitigated)
"""

INSERT_FAIR_VALUE_GAP = """
INSERT INTO fair_value_gaps
    (symbol, timeframe, gap_type, gap_top, gap_bottom, gap_midpoint,
     gap_size_pct, formed_at, is_filled, fill_pct, is_mitigated, date)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPDATE_FAIR_VALUE_GAP_FILL = """
UPDATE fair_value_gaps
SET fill_pct     = ?,
    is_filled    = ?,
    is_mitigated = ?
WHERE id = ?
"""

LIST_ACTIVE_FAIR_VALUE_GAPS = """
SELECT id, symbol, timeframe, gap_type, gap_top, gap_bottom, gap_midpoint,
       gap_size_pct, formed_at, is_filled, fill_pct, is_mitigated, date
FROM fair_value_gaps
WHERE symbol = ? AND date = ? AND is_mitigated = 0
ORDER BY formed_at DESC
"""

# ── order_blocks ──────────────────────────────────────────────────────────────

CREATE_ORDER_BLOCKS = """
CREATE TABLE IF NOT EXISTS order_blocks (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol                TEXT    NOT NULL,
    timeframe             TEXT    NOT NULL,
    ob_type               TEXT    NOT NULL,
    ob_high               REAL    NOT NULL,
    ob_low                REAL    NOT NULL,
    displacement_size_pct REAL    NOT NULL,
    strength              TEXT    NOT NULL,
    formed_at             TEXT    NOT NULL,
    touch_count           INTEGER NOT NULL DEFAULT 0,
    is_mitigated          INTEGER NOT NULL DEFAULT 0,
    mitigated_at          TEXT,
    date                  TEXT    NOT NULL
)
"""

CREATE_ORDER_BLOCKS_IDX = """
CREATE INDEX IF NOT EXISTS idx_ob_symbol_date
ON order_blocks (symbol, date, is_mitigated)
"""

INSERT_ORDER_BLOCK = """
INSERT INTO order_blocks
    (symbol, timeframe, ob_type, ob_high, ob_low, displacement_size_pct,
     strength, formed_at, touch_count, is_mitigated, mitigated_at, date)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPDATE_ORDER_BLOCK_MITIGATED = """
UPDATE order_blocks
SET is_mitigated = 1,
    mitigated_at = ?
WHERE id = ?
"""

UPDATE_ORDER_BLOCK_TOUCH = """
UPDATE order_blocks
SET touch_count = touch_count + 1
WHERE id = ?
"""

LIST_ACTIVE_ORDER_BLOCKS = """
SELECT id, symbol, timeframe, ob_type, ob_high, ob_low, displacement_size_pct,
       strength, formed_at, touch_count, is_mitigated, mitigated_at, date
FROM order_blocks
WHERE symbol = ? AND date = ? AND is_mitigated = 0
ORDER BY formed_at DESC
"""


# ── volume_profiles ───────────────────────────────────────────────────────────

CREATE_VOLUME_PROFILES = """
CREATE TABLE IF NOT EXISTS volume_profiles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT    NOT NULL,
    date          TEXT    NOT NULL,
    poc           REAL    NOT NULL,
    vah           REAL    NOT NULL,
    val           REAL    NOT NULL,
    total_volume  INTEGER NOT NULL,
    calculated_at TEXT    NOT NULL,
    UNIQUE (symbol, date)
)
"""

CREATE_VOLUME_PROFILES_IDX = """
CREATE INDEX IF NOT EXISTS idx_vp_symbol_date
ON volume_profiles (symbol, date)
"""

CREATE_VOLUME_PROFILE_LEVELS = """
CREATE TABLE IF NOT EXISTS volume_profile_levels (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id   INTEGER NOT NULL REFERENCES volume_profiles(id),
    price        REAL    NOT NULL,
    volume       INTEGER NOT NULL,
    volume_pct   REAL    NOT NULL,
    level_type   TEXT    NOT NULL,
    is_value_area INTEGER NOT NULL DEFAULT 0
)
"""

CREATE_VOLUME_PROFILE_LEVELS_IDX = """
CREATE INDEX IF NOT EXISTS idx_vpl_profile_id
ON volume_profile_levels (profile_id)
"""

UPSERT_VOLUME_PROFILE = """
INSERT OR REPLACE INTO volume_profiles
    (symbol, date, poc, vah, val, total_volume, calculated_at)
VALUES
    (?, ?, ?, ?, ?, ?, ?)
"""

SELECT_VOLUME_PROFILE = """
SELECT id, symbol, date, poc, vah, val, total_volume, calculated_at
FROM volume_profiles
WHERE symbol = ? AND date = ?
"""

DELETE_VOLUME_PROFILE_LEVELS = """
DELETE FROM volume_profile_levels WHERE profile_id = ?
"""

INSERT_VOLUME_PROFILE_LEVEL = """
INSERT INTO volume_profile_levels
    (profile_id, price, volume, volume_pct, level_type, is_value_area)
VALUES
    (?, ?, ?, ?, ?, ?)
"""

LIST_VOLUME_PROFILE_LEVELS = """
SELECT price, volume, volume_pct, level_type, is_value_area
FROM volume_profile_levels
WHERE profile_id = ?
ORDER BY price ASC
"""

# ── stop_hunts ────────────────────────────────────────────────────────────────

CREATE_STOP_HUNTS = """
CREATE TABLE IF NOT EXISTS stop_hunts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT    NOT NULL,
    hunt_type           TEXT    NOT NULL,
    level_type          TEXT    NOT NULL,
    level_price         REAL    NOT NULL,
    hunt_price          REAL    NOT NULL,
    candle_time         TEXT    NOT NULL,
    reversal_direction  TEXT    NOT NULL,
    confidence          REAL    NOT NULL,
    confidence_tier     TEXT    NOT NULL DEFAULT 'LOW',
    oi_confirms         INTEGER NOT NULL DEFAULT 0,
    futures_confirms    INTEGER NOT NULL DEFAULT 0,
    suggested_signal    TEXT    NOT NULL,
    entry_zone_low      REAL    NOT NULL,
    entry_zone_high     REAL    NOT NULL,
    target              REAL    NOT NULL,
    stop                REAL    NOT NULL,
    date                TEXT    NOT NULL
)
"""

CREATE_STOP_HUNTS_IDX = """
CREATE INDEX IF NOT EXISTS idx_stop_hunts_symbol_date
ON stop_hunts (symbol, date, candle_time)
"""

INSERT_STOP_HUNT = """
INSERT INTO stop_hunts
    (symbol, hunt_type, level_type, level_price, hunt_price, candle_time,
     reversal_direction, confidence, confidence_tier,
     oi_confirms, futures_confirms, suggested_signal,
     entry_zone_low, entry_zone_high, target, stop, date)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

LIST_RECENT_STOP_HUNTS = """
SELECT id, symbol, hunt_type, level_type, level_price, hunt_price, candle_time,
       reversal_direction, confidence, confidence_tier,
       oi_confirms, futures_confirms, suggested_signal,
       entry_zone_low, entry_zone_high, target, stop, date
FROM stop_hunts
WHERE symbol = ? AND candle_time >= ?
ORDER BY candle_time DESC
"""
