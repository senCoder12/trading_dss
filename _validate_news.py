import logging
logging.basicConfig(level=logging.WARNING)

from src.analysis.news import NewsEngine
from src.database.db_manager import DatabaseManager
from src.data.index_registry import get_registry

# Fresh DB for this validation run
import os
db_path = "data/db/trading_validate.db"
if os.path.exists(db_path):
    os.remove(db_path)

db = DatabaseManager(db_path)
db.connect()
db.initialise_schema()

# Seed index_master — required for news_index_impact FK
registry = get_registry()
registry.sync_to_db(db)

engine = NewsEngine(db)

# 1. Run a full news cycle
result = engine.run_news_cycle()
print(f"=== News Cycle Complete ===")
print(f"Fetched: {result.articles_fetched}, New: {result.articles_new}")
print(f"Severity: {result.by_severity}")
print(f"Cycle time: {result.cycle_duration_ms}ms")

# 2. Get news vote for specific indices
for index_id in ["NIFTY50", "BANKNIFTY", "NIFTYIT"]:
    vote = engine.get_news_vote(index_id)
    if vote:
        print(f"\n{index_id}: {vote.vote} (conf: {vote.confidence:.2f})")
        print(f"  Articles: {vote.active_article_count}")
        print(f"  Event regime: {vote.event_regime}")
        print(f"  Reasoning: {vote.reasoning}")

# 3. Check alerts
alerts = engine.get_critical_alerts()
print(f"\nCritical alerts: {len(alerts)}")
for a in alerts:
    print(f"  [{a.severity}] {a.message}")

# 4. News feed for dashboard
feed = engine.get_news_feed(limit=10)
print(f"\nLatest news ({len(feed)} items):")
for item in feed:
    print(f"  [{item['severity']:>8}] [{item['sentiment_label']:>12}] {item['title'][:60]}...")

# 5. Verify database
article_count = db.fetch_one("SELECT COUNT(*) as c FROM news_articles")
impact_count = db.fetch_one("SELECT COUNT(*) as c FROM news_index_impact")
index_count = db.fetch_one("SELECT COUNT(*) as c FROM index_master")
print(f"\nDB: {article_count['c']} articles, {impact_count['c']} index impacts, {index_count['c']} indices in master")

# 6. Run second cycle (should mostly skip duplicates)
result2 = engine.run_news_cycle()
print(f"\nSecond cycle: {result2.articles_new} new (should be low)")
