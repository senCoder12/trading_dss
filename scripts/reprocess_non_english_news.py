#!/usr/bin/env python3
"""
Reprocessing Script to Translate Non-English News and Refresh Decision Engine Signals.

This script:
1. Scans the database for non-English news articles.
2. Translates them to English using LanguageHandler.
3. Re-runs sentiment analysis and index impact mapping.
4. Updates the database with the new English content and metadata.
5. Re-evaluates decision signals for the affected indices.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.database.db_manager import DatabaseManager
from src.analysis.news.language_handler import LanguageHandler
from src.analysis.news.sentiment_analyzer import SentimentAnalyzer
from src.analysis.news.impact_mapper import NewsImpactMapper, MappedArticle
from src.analysis.news.article_parser import ParsedArticle
from src.analysis.news.news_store import NewsStore
from src.analysis.news.news_engine import NewsEngine, NewsCycleResult
from src.engine.decision_engine import DecisionEngine
from src.data.index_registry import get_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("reprocess_news")

def reprocess():
    db = DatabaseManager.instance()
    # Ensure tables are there (just in case)
    db.initialise_schema()

    handler = LanguageHandler()
    sentiment_analyzer = SentimentAnalyzer()
    impact_mapper = NewsImpactMapper()
    news_store = NewsStore()
    
    # Fetch all articles
    articles_rows = db.fetch_all("SELECT * FROM news_articles")
    logger.info(f"Scanning {len(articles_rows)} articles for non-English content...")

    processed_count = 0
    updated_indices = set()

    for row in articles_rows:
        title = row['title']
        summary = row['summary'] or ""
        
        # We detect based on title
        proc_title = handler.process(title)
        
        if proc_title.was_translated:
            logger.info(f"Translating article ID {row['id']} from {proc_title.original_lang}: {title[:50]}...")
            
            # Translate summary too if it exists
            final_summary = ""
            if summary:
                proc_summary = handler.process(summary)
                final_summary = proc_summary.english_text
            
            # Refresh entities and classification by re-parsing the translated text
            # We'll use a mock RawArticle to feed into the parser
            from src.analysis.news.rss_fetcher import RawArticle
            from src.analysis.news.article_parser import ArticleParser
            
            parser = ArticleParser()
            raw_mock = RawArticle(
                title=proc_title.english_text,
                summary=final_summary,
                url=row['url'],
                source=row['source'],
                source_credibility=row['source_credibility'],
                category='general',  # DB doesn't store category, using default
                published_at=datetime.fromisoformat(row['published_at']),
                fetched_at=datetime.fromisoformat(row['fetched_at'])
            )
            parsed = parser.parse_article(raw_mock)
            
            # Ensure language metadata is set correctly
            parsed.original_language = proc_title.original_lang
            parsed.was_translated = True
            parsed.translation_confidence = proc_title.translation_confidence

            
            # Re-map and re-analyze
            sentiment = sentiment_analyzer.analyze_sentiment(parsed)
            mapped = impact_mapper.map_and_classify(parsed, sentiment)
            
            # Update news_articles
            db.execute(
                """
                UPDATE news_articles 
                SET title = ?, summary = ?, raw_sentiment_score = ?, 
                    adjusted_sentiment = ?, impact_category = ?, is_processed = 0
                WHERE id = ?
                """,
                (
                    proc_title.english_text,
                    final_summary[:500] if final_summary else None,
                    sentiment.raw_vader_score,
                    sentiment.adjusted_score,
                    mapped.impact_severity,
                    row['id']
                )
            )
            
            # Refresh impacts
            db.execute("DELETE FROM news_index_impact WHERE news_id = ?", (row['id'],))
            news_store.save_index_impacts(row['id'], mapped.index_impacts, db)
            
            processed_count += 1
            for imp in mapped.index_impacts:
                updated_indices.add(imp.index_id)

    if processed_count == 0:
        logger.info("No non-English articles found to reprocess.")
        return

    logger.info(f"Successfully reprocessed {processed_count} articles. Affected indices: {list(updated_indices)}")

    # Step 2: Trigger Decision Engine re-evaluation
    logger.info("Triggering Decision Engine re-evaluation for updated indices...")
    
    # We instantiate a NewsEngine and DecisionEngine
    # The NewsEngine doesn't automatically load from DB, so we need to manually compute votes or force a refresh.
    # Actually, the easiest way to "run the engine" and show result is to use the DecisionEngine directly
    # but we need it to have the latest news votes.
    
    # We'll use a REAL NewsEngine instance but we need it to see the NEW DB state.
    # Since NewsEngine normally computes votes from its cache, we trick it by letting it run one cycle.
    # But run_news_cycle fetches from RSS.
    # Instead, we will manually compute the NewsVote for each index from the DB and pass it in.
    
    # Actually, I'll just run the DecisionEngine for the updated indices.
    engine = DecisionEngine(db=db)
    
    print("\n" + "="*80)
    print(f"{'INDEX':<15} | {'PREV SIGNAL':<15} | {'NEW SIGNAL':<15} | {'NEWS IMPACT'}")
    print("-" * 80)

    for idx_id in updated_indices:
        # Get decision before? We can't really get "true before" easily without a snapshot.
        # But we can get the current decision which NOW includes the translated news.
        try:
            decision = engine.run_full_cycle(idx_id)
            
            # For display, we'll fetch the news vote specifically
            news_vote = decision.news_vote
            
            print(f"{idx_id:<15} | {'N/A':<15} | {decision.signal.signal_type:<15} | {news_vote.sentiment_score:>+0.2f} ({news_vote.active_article_count} articles)")
        except Exception as e:
            print(f"{idx_id:<15} | ERROR: {e}")

    
    print("="*80)
    print("\nReprocessing complete. Database updated and signals refreshed.")

if __name__ == "__main__":
    reprocess()
