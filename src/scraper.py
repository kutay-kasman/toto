"""
Web scraper module for nesine.com
Includes caching logic to avoid redundant scraping.
"""

import time
import re
from typing import List, Tuple, Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import logging

from src.database import MatchDatabase

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
    
    def _setup_driver(self):
        """Setup Selenium Chrome driver."""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
    
    def _cleanup_driver(self):
        """Close browser driver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def scrape_upcoming_matches(self, url: str = "https://www.nesine.com/sportoto") -> List[Tuple[str, str]]:
        """
        Scrape upcoming match fixtures.
        
        Args:
            url: URL to scrape from
            
        Returns:
            List of (home_team, away_team) tuples
        """
        try:
            self._setup_driver()
            logger.info(f"Scraping upcoming matches from {url}")
            
            self.driver.get(url)
            
            # Wait for match elements to load
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//td[@data-test-id='name']/button[@data-testid='nsn-button']")
                )
            )
            
            time.sleep(2)  # Allow page to fully load
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            match_elements = soup.find_all('button', attrs={'data-testid': 'nsn-button'})
            
            if not match_elements:
                logger.warning("No match elements found. Website structure may have changed.")
                return []
            
            logger.info(f"Found {len(match_elements)} potential match elements")
            
            matches = []
            for element in match_elements:
                full_match_name = element.get_text().strip()
                
                # Extract team names (before newline/date)
                match_teams_only = full_match_name.split('\n')[0].strip()
                
                # Remove date/time pattern
                match_teams_only = re.sub(
                    r'\s*\d{2}\.\d{2}\.\d{4}\s*\d{2}:\d{2}', 
                    '', 
                    match_teams_only
                ).strip()
                
                # Split by dash
                if '-' in match_teams_only:
                    teams = match_teams_only.split('-')
                    home_team = teams[0].strip()
                    away_team = teams[1].strip()
                    
                    if home_team and away_team:
                        matches.append((home_team, away_team))
            
            # Take last 15 matches (Spor Toto format)
            matches = matches[-15:] if len(matches) >= 15 else matches
            
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
            self._cleanup_driver()
    
    def _parse_score_from_status_element(self, status_text: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse goal scores from status element text using regex.
        
        The status element contains text like:
        - "Bitti\n / \n0-3" (multi-line format)
        - " Bitti / 0-3 " (with whitespace/newlines)
        - " Bitti / 0 - 3 " (with spaces around dash)
        - " 10' / 1-1 " (live match)
        - " IY " (no score, match hasn't started)
        
        Args:
            status_text: Raw text from status element (may contain newlines/whitespace)
            
        Returns:
            Tuple of (home_goals, away_goals) or (None, None) if not found
        """
        # Normalize text: Replace all newlines with spaces, then strip
        # Example: "Bitti\n / \n0-3" -> "Bitti / 0-3"
        normalized_text = status_text.replace('\n', ' ').replace('\r', ' ').strip()
        
        # Robust regex to find score pattern: digits-whitespace-dash-whitespace-digits
        # This handles "0-3", "0 - 3", "0- 3", "0 -3", etc.
        # re.search searches through the entire string, not just the start
        score_pattern = r'(\d+)\s*-\s*(\d+)'
        
        match = re.search(score_pattern, normalized_text)
        if match:
            home_goals = int(match.group(1))
            away_goals = int(match.group(2))
            return home_goals, away_goals
        
        # No score found (match hasn't started)
        return None, None
    
    def scrape_historical_results(self, 
                                  url: str = "https://www.nesine.com/sportoto/mac-sonuclari",
                                  max_weeks: Optional[int] = None) -> List[Tuple]:
        """
        Scrape historical match results.
        
        Args:
            url: URL to scrape from
            max_weeks: Maximum number of weeks to scrape (None for all)
            
        Returns:
            List of (home_team, away_team, result, week_range, match_date) tuples
        """
        all_results = []
        
        try:
            self._setup_driver()
            logger.info(f"Scraping historical results from {url}")
            
            self.driver.get(url)
            
            # Wait for week selector
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.ID, "filter-week-select"))
            )
            
            week_select_element = self.driver.find_element(By.ID, "filter-week-select")
            select = Select(week_select_element)
            
            week_values = [option.get_attribute('value') for option in select.options]
            
            if max_weeks:
                week_values = week_values[1:max_weeks+1]
            else:
                week_values = week_values[1:]  # Skip first empty option
            
            logger.info(f"Scraping {len(week_values)} weeks of historical data")
            
            for week_value in week_values:
                try:
                    # Re-find element to avoid stale reference
                    week_select_element = self.driver.find_element(By.ID, "filter-week-select")
                    select = Select(week_select_element)
                    select.select_by_value(week_value)
                    
                    time.sleep(1)  # Wait for page update
                    
                    # Wait for results table
                    try:
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located(
                                (By.XPATH, "//td[@data-test-id='programResult-result']")
                            )
                        )
                    except:
                        logger.warning(f"No results found for week {week_value}")
                        continue
                    
                    selected_option_text = select.first_selected_option.text.strip()
                    logger.info(f"Processing week {week_value} ({selected_option_text})")
                    
                    # Wait a bit more for dynamic content to load
                    time.sleep(1.5)
                    
                    # Use Selenium to find elements directly (more reliable for dynamic content)
                    # This ensures we get the actual rendered text, including newlines
                    result_elements_selenium = self.driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id="programResult-result"]')
                    name_elements_selenium = self.driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id="programResult-name"]')
                    status_elements_selenium = self.driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id="programResult-status"]')
                    
                    # Verify we have matching counts
                    if len(name_elements_selenium) != len(result_elements_selenium) or len(name_elements_selenium) != len(status_elements_selenium):
                        logger.warning(f"Mismatch in element counts: names={len(name_elements_selenium)}, "
                                     f"results={len(result_elements_selenium)}, status={len(status_elements_selenium)}")
                    
                    week_results = []
                    for idx, (name_elem, result_elem, status_elem) in enumerate(
                        zip(name_elements_selenium, result_elements_selenium, status_elements_selenium), 1
                    ):
                        match_name = name_elem.text.strip()
                        match_result = result_elem.text.strip()
                        
                        # Get raw text from status element using .text property
                        # This captures the actual rendered text including newlines
                        status_raw_text = status_elem.text
                        
                        if match_name and match_result in ['1', '2', 'X']:
                            # Normalize text: Replace newlines with spaces for consistent parsing
                            # Example: "Bitti\n / \n0-3" -> "Bitti / 0-3"
                            status_normalized = status_raw_text.replace('\n', ' ').replace('\r', ' ').strip()
                            
                            # Extract score from normalized status text using regex
                            home_goals, away_goals = self._parse_score_from_status_element(status_raw_text)
                            
                            # Detailed debug logging - print to console for immediate visibility
                            score_match = f"{home_goals}-{away_goals}" if (home_goals is not None and away_goals is not None) else "None"
                            print(f"DEBUG ROW {idx}: Raw Text='{repr(status_raw_text)}' | Normalized='{status_normalized}' -> Extracted Score: {score_match}")
                            logger.info(f"DEBUG ROW {idx}: Match='{match_name}', Result='{match_result}', "
                                      f"Raw Text='{repr(status_raw_text)}', Normalized='{status_normalized}' -> Extracted Score: {score_match}")
                            
                            # Parse team names (clean, no score extraction needed)
                            if '-' in match_name:
                                teams = match_name.split('-')
                                home_team = teams[0].strip()
                                away_team = teams[1].strip()
                                
                                if home_team and away_team:
                                    # Return tuple: (home_team, away_team, result, week_range, match_date, home_goals, away_goals)
                                    week_results.append((
                                        home_team,
                                        away_team,
                                        match_result,
                                        selected_option_text,
                                        None,  # match_date not available from this source
                                        home_goals,
                                        away_goals
                                    ))
                    
                    all_results.extend(week_results)
                    logger.info(f"Added {len(week_results)} matches from week {week_value}")
                    
                except Exception as e:
                    logger.error(f"Error processing week {week_value}: {e}")
                    continue
            
            logger.info(f"Total {len(all_results)} historical matches scraped")
            return all_results
            
        except Exception as e:
            logger.error(f"Error scraping historical results: {e}")
            return []
        finally:
            self._cleanup_driver()
    
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

