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
WHERE index_id = ? AND timeframe = ?
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
WHERE index_id = ? AND timeframe = ?
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
     open_interest, oi_change, volume, ltp, iv, bid_price, ask_price)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(index_id, timestamp, expiry_date, strike_price, option_type)
DO UPDATE SET
    open_interest = excluded.open_interest,
    oi_change     = excluded.oi_change,
    volume        = excluded.volume,
    ltp           = excluded.ltp,
    iv            = excluded.iv,
    bid_price     = excluded.bid_price,
    ask_price     = excluded.ask_price
"""

LIST_OPTIONS_CHAIN_FOR_EXPIRY = """
SELECT * FROM options_chain_snapshot
WHERE index_id = ?
  AND expiry_date = ?
  AND timestamp = (
      SELECT MAX(timestamp) FROM options_chain_snapshot
      WHERE index_id = ? AND expiry_date = ?
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
WHERE index_id = ? AND expiry_date = ?
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
