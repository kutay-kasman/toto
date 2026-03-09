"""
Quick test for the improved scraper
"""
import logging
from src.database import MatchDatabase
from src.scraper import NesineScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("Testing improved scraper with new button selectors")
    logger.info("=" * 60)
    
    db = MatchDatabase()
    scraper = NesineScraper(db, headless=True, use_undetected=False)
    
    try:
        logger.info("Starting scrape...")
        matches = scraper.scrape_upcoming_matches()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"✓ SUCCESS! Scraped {len(matches)} matches")
        logger.info("=" * 60)
        
        for i, (home, away) in enumerate(matches, 1):
            logger.info(f"{i:2d}. {home:30s} - {away}")
        
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 60)
        logger.error(f"✗ FAILED: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
