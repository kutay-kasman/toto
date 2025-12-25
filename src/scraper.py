"""
Web scraper module for nesine.com
Includes caching logic to avoid redundant scraping.
"""

from typing import List, Tuple, Optional
import logging

from src.database import MatchDatabase
from src.scraper_utils import (
    cleanup_driver,
    scrape_historical,
    scrape_latest_week,
    scrape_upcoming,
    setup_driver,
)

logger = logging.getLogger(__name__)


class NesineScraper:
    """Scraper for nesine.com with database caching."""
    
    def __init__(self, db: MatchDatabase):
        """
        Initialize scraper.
        
        Args:
            db: MatchDatabase instance for caching
        """
        self.db = db
        self.driver = None
    
    def scrape_upcoming_matches(self, url: str = "https://www.nesine.com/sportoto") -> List[Tuple[str, str]]:
        """
        Scrape upcoming match fixtures.
        Thin wrapper around `scrape_upcoming` that also caches to DB.
        
        Args:
            url: URL to scrape from
            
        Returns:
            List of (home_team, away_team) tuples
        """
        try:
            self.driver = setup_driver()
            matches = scrape_upcoming(self.driver, url)

            if not matches:
                return []

            # Store in database
            self.db.clear_upcoming_matches()
            for home_team, away_team in matches:
                self.db.insert_upcoming_match(home_team, away_team)
            
            logger.info(f"Successfully scraped {len(matches)} upcoming matches")
            return matches
            
        except Exception as e:
            logger.error(f"Error scraping upcoming matches: {e}")
            return []
        finally:
            cleanup_driver(self.driver)
            self.driver = None
    
    def scrape_latest_week_results(self,
                                   url: str = "https://www.nesine.com/sportoto/mac-sonuclari") -> List[Tuple]:
        """
        Scrape only the most recent week's match results.
        Wrapper around `scrape_latest_week`; handles DB write.
        
        Args:
            url: URL to scrape from (can include pNo parameter, e.g., ?pNo=316)
            
        Returns:
            List of (home_team, away_team, result, week_range, match_date, home_goals, away_goals) tuples
        """
        try:
            self.driver = setup_driver()
            week_results = scrape_latest_week(self.driver, url)

            if week_results:
                logger.info(f"Scraped {len(week_results)} matches from latest week")
                self.db.insert_historical_matches_batch(week_results)
                logger.info(f"Stored {len(week_results)} matches from latest week")

            return week_results
            
        except Exception as e:
            logger.error(f"Error scraping latest week results: {e}")
            return []
        finally:
            cleanup_driver(self.driver)
            self.driver = None
    
    def scrape_historical_results(self, 
                                  url: str = "https://www.nesine.com/sportoto/mac-sonuclari",
                                  max_weeks: Optional[int] = None) -> List[Tuple]:
        """
        Scrape historical match results.
        Wrapper around `scrape_historical`; handles DB write.
        
        Args:
            url: URL to scrape from
            max_weeks: Maximum number of weeks to scrape (None for all)
            
        Returns:
            List of (home_team, away_team, result, week_range, match_date) tuples
        """
        all_results = []
        
        try:
            self.driver = setup_driver()
            all_results = scrape_historical(self.driver, url, max_weeks)

            if all_results:
                self.db.insert_historical_matches_batch(all_results)
                logger.info(f"Cached {len(all_results)} matches in database")

            return all_results
            
        except Exception as e:
            logger.error(f"Error scraping historical results: {e}")
            return []
        finally:
            cleanup_driver(self.driver)
            self.driver = None
    
    def scrape_with_cache(self, force_refresh: bool = False) -> List[Tuple]:
        """
        Scrape historical results with caching.
        Only scrapes new data if not already in database.
        
        Args:
            force_refresh: If True, scrape even if data exists
            
        Returns:
            List of scraped matches (7-tuple with goals)
        """
        if not force_refresh:
            # Check if we have recent data
            existing_matches = self.db.get_historical_matches()
            if len(existing_matches) > 0:
                logger.info(f"Found {len(existing_matches)} existing matches in database")
                # Return existing data with goals
                return [
                    (row['Ev Sahibi Takım'], row['Deplasman Takımı'], 
                     row['Mac Sonucu'], None, None,
                     row.get('Home_Goals'), row.get('Away_Goals'))
                    for _, row in existing_matches.iterrows()
                ]
        
        # Scrape new data
        matches = self.scrape_historical_results()
        
        # Store in database
        if matches:
            self.db.insert_historical_matches_batch(matches)
            logger.info(f"Cached {len(matches)} matches in database")
        
        return matches

