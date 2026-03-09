"""
Web scraper module for nesine.com
Includes caching logic to avoid redundant scraping.
"""

from typing import List, Tuple, Optional
import logging

from src.database import MatchDatabase
from src.scraper_utils import (
    cleanup_driver,
    fetch_match_elements_with_retries,
    parse_score_from_status_text,
    scrape_historical,
    scrape_latest_week,
    scrape_upcoming,
    scroll_to_load_matches,
    setup_driver,
    wait_for_min_matches,
    wait_for_week_selector,
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

    def scrape_recent_weeks(self,
                            n_weeks: int = 3,
                            url: str = "https://www.nesine.com/sportoto/mac-sonuclari") -> List[Tuple]:
        """
        Scrape the N most recent weeks of results using a single browser session.
        Returns combined list of all tuples across all scraped weeks.
        """
        import time as _time
        from selenium.webdriver.support import expected_conditions as _EC
        from selenium.webdriver.common.by import By as _By
        from selenium.webdriver.support.ui import WebDriverWait as _WDW

        all_results: List[Tuple] = []

        try:
            self.driver = setup_driver()
            self.driver.get(url)

            select = wait_for_week_selector(self.driver, timeout=20)
            week_values = [opt.get_attribute("value") for opt in select.options]
            real_weeks = week_values[1:]  # skip placeholder option at index 0

            weeks_to_scrape = real_weeks[:n_weeks]
            logger.info(f"Scraping {len(weeks_to_scrape)} recent weeks: {weeks_to_scrape}")

            for week_value in weeks_to_scrape:
                try:
                    select = wait_for_week_selector(self.driver, timeout=10)
                    select.select_by_value(week_value)
                    _time.sleep(1)

                    try:
                        _WDW(self.driver, 10).until(
                            _EC.presence_of_element_located(
                                (_By.XPATH, "//td[@data-test-id='programResult-result']")
                            )
                        )
                    except Exception:
                        logger.warning(f"No results found for week {week_value}, skipping")
                        continue

                    week_label = select.first_selected_option.text.strip()
                    logger.info(f"Scraping week {week_value}: {week_label}")

                    wait_for_min_matches(self.driver, expected_matches=14, max_wait_time=10)
                    _time.sleep(1.5)
                    scroll_to_load_matches(self.driver)

                    name_els, result_els, status_els = fetch_match_elements_with_retries(
                        self.driver, attempts=2, sleep_between=1.5
                    )

                    week_count = 0
                    for name_el, result_el, status_el in zip(name_els, result_els, status_els):
                        match_name   = (name_el.get_attribute("textContent") or "").strip()
                        match_result = (result_el.get_attribute("textContent") or "").strip()
                        status_text  = (status_el.get_attribute("textContent") or "").strip()

                        if match_name and match_result in ("1", "2", "X") and "-" in match_name:
                            teams = match_name.split("-", 1)
                            home  = teams[0].strip()
                            away  = teams[1].strip()
                            if home and away:
                                hg, ag = parse_score_from_status_text(status_text)
                                all_results.append((home, away, match_result, week_label, None, hg, ag))
                                week_count += 1

                    logger.info(f"Week {week_value} ({week_label}): collected {week_count} matches")

                except Exception as e:
                    logger.error(f"Error scraping week {week_value}: {e}")
                    continue

            if all_results:
                self.db.insert_historical_matches_batch(all_results)
                logger.info(f"Stored total {len(all_results)} matches from {n_weeks} recent weeks")

            return all_results

        except Exception as e:
            logger.error(f"Error in scrape_recent_weeks: {e}")
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

