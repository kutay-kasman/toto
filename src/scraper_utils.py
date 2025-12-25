"""
Utility helpers for the Nesine scraper.

This module centralizes selenium setup/teardown, scrolling, waiting, element
collection, and score parsing so `scraper.py` can stay focused on the high-level
scraping flow.
"""

import logging
import re
import time
from typing import List, Optional, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Driver lifecycle
# -----------------------------------------------------------------------------

def setup_driver() -> webdriver.Chrome:
    """Configure and return a headless Chrome driver suitable for scraping."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")  # desktop viewport for full rendering
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def cleanup_driver(driver: Optional[webdriver.Chrome]) -> None:
    """Safely close and dispose of the webdriver."""
    if driver:
        driver.quit()


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

def parse_score_from_status_text(status_text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract (home_goals, away_goals) from raw status text using a forgiving regex.

    Handles variants such as:
    - "Bitti\\n / \\n0-3"
    - "10' / 1 - 1"
    Returns (None, None) when no score is present.
    """
    normalized_text = status_text.replace("\n", " ").replace("\r", " ").strip()
    match = re.search(r"(\d+)\s*-\s*(\d+)", normalized_text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


# -----------------------------------------------------------------------------
# Waiting / scrolling helpers
# -----------------------------------------------------------------------------

def wait_for_week_selector(driver: webdriver.Chrome, timeout: int = 20) -> Select:
    """Wait for the week dropdown to appear and return a Select wrapper."""
    WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.ID, "filter-week-select")))
    week_select_element = driver.find_element(By.ID, "filter-week-select")
    return Select(week_select_element)


def wait_for_min_matches(
    driver: webdriver.Chrome,
    expected_matches: int = 15,
    max_wait_time: float = 15.0,
    wait_interval: float = 0.5,
) -> None:
    """Poll until at least `expected_matches` result cells exist or timeout."""
    waited_time = 0.0
    logger.info("Waiting for all matches to load...")
    while waited_time < max_wait_time:
        current_matches = len(driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id="programResult-result"]'))
        if current_matches >= expected_matches:
            logger.info(f"Found all {current_matches} matches after {waited_time:.1f} seconds")
            return
        time.sleep(wait_interval)
        waited_time += wait_interval
        if int(waited_time) % 2 == 0 and waited_time > 0:
            logger.info(f"Waiting... Found {current_matches} matches so far (waited {waited_time:.1f}s)")
    final_count = len(driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id="programResult-result"]'))
    logger.info(f"Finished waiting. Final match count: {final_count}")


def scroll_to_load_matches(driver: webdriver.Chrome, max_scroll_attempts: int = 5, pause_seconds: float = 1.0) -> None:
    """Scroll page bottom-up to trigger lazy-loaded match rows."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scroll_attempts):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_seconds)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.5)


def fetch_match_elements_with_retries(
    driver: webdriver.Chrome,
    attempts: int = 3,
    sleep_between: float = 2.0,
) -> Tuple[List, List, List]:
    """
    Collect name/result/status elements, retrying to handle progressive loading.
    Returns three lists of selenium elements.
    """
    name_elements: List = []
    result_elements: List = []
    status_elements: List = []

    for attempt in range(attempts):
        result_elements = driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id="programResult-result"]')
        name_elements = driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id="programResult-name"]')
        status_elements = driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id="programResult-status"]')
        logger.info(f"Attempt {attempt + 1}: Found {len(name_elements)} match elements")
        if len(name_elements) >= 15:
            logger.info("Found all expected matches!")
            break
        if attempt < attempts - 1:
            time.sleep(sleep_between)

    return name_elements, result_elements, status_elements


def wait_additional_for_matches(
    driver: webdriver.Chrome,
    min_matches: int = 10,
    cycles: int = 3,
    pause_seconds: float = 2.0,
) -> Tuple[List, List, List]:
    """Extra wait cycles plus incremental scrolling if fewer than `min_matches` were found."""
    name_elements: List = []
    result_elements: List = []
    status_elements: List = []

    for wait_cycle in range(cycles):
        time.sleep(pause_seconds)
        result_elements = driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id="programResult-result"]')
        name_elements = driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id="programResult-name"]')
        status_elements = driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id="programResult-status"]')
        logger.info(f"After additional wait cycle {wait_cycle + 1}, found {len(name_elements)} match elements")
        if len(name_elements) >= 15:
            logger.info("Found all expected matches!")
            break

    # Targeted scrolling near the table if still sparse
    if len(name_elements) < min_matches:
        try:
            table_container = driver.find_element(By.CSS_SELECTOR, 'table, [class*=\"table\"], [class*=\"result\"]')
            driver.execute_script("arguments[0].scrollIntoView(true);", table_container)
            time.sleep(2)
            for i in range(3):
                driver.execute_script(f"window.scrollBy(0, {500 * (i + 1)});")
                time.sleep(1)
            result_elements = driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id=\"programResult-result\"]')
            name_elements = driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id=\"programResult-name\"]')
            status_elements = driver.find_elements(By.CSS_SELECTOR, 'td[data-test-id=\"programResult-status\"]')
            logger.info(f"After additional scrolling, found {len(name_elements)} match elements")
        except Exception as exc:
            logger.debug(f"Could not find table container for additional scrolling: {exc}")

    return name_elements, result_elements, status_elements


# -----------------------------------------------------------------------------
# High-level scraping helpers (return raw scraped data)
# -----------------------------------------------------------------------------

def scrape_upcoming(driver: webdriver.Chrome, url: str) -> List[Tuple[str, str]]:
    """
    Scrape upcoming fixtures (home_team, away_team).
    Driver lifecycle is managed by the caller.
    """
    logger.info(f"Scraping upcoming matches from {url}")
    driver.get(url)

    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located(
            (By.XPATH, "//td[@data-test-id='name']/button[@data-testid='nsn-button']")
        )
    )

    time.sleep(2)  # Allow page to fully load
    soup = BeautifulSoup(driver.page_source, "html.parser")
    match_elements = soup.find_all("button", attrs={"data-testid": "nsn-button"})

    if not match_elements:
        logger.warning("No match elements found. Website structure may have changed.")
        return []

    logger.info(f"Found {len(match_elements)} potential match elements")

    matches: List[Tuple[str, str]] = []
    for element in match_elements:
        full_match_name = element.get_text().strip()
        match_teams_only = full_match_name.split("\n")[0].strip()
        match_teams_only = re.sub(r"\s*\d{2}\.\d{2}\.\d{4}\s*\d{2}:\d{2}", "", match_teams_only).strip()
        if "-" in match_teams_only:
            teams = match_teams_only.split("-")
            home_team = teams[0].strip()
            away_team = teams[1].strip()
            if home_team and away_team:
                matches.append((home_team, away_team))

    return matches[-15:] if len(matches) >= 15 else matches


def scrape_latest_week(driver: webdriver.Chrome, url: str) -> List[Tuple]:
    """
    Scrape the most recent week's results.
    Returns tuples: (home_team, away_team, result, week_range, match_date, home_goals, away_goals)
    """
    logger.info(f"Scraping latest week results from {url}")

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    pno_value = query_params["pNo"][0] if "pNo" in query_params else None
    if pno_value:
        logger.info(f"Found pNo parameter in URL: {pno_value}")

    driver.get(url)
    select = wait_for_week_selector(driver, timeout=20)
    week_values = [option.get_attribute("value") for option in select.options]

    if len(week_values) < 2:
        logger.warning("No weeks available to scrape")
        return []

    if pno_value and pno_value in week_values:
        selected_week_value = pno_value
        logger.info(f"Selecting week from pNo parameter: {selected_week_value}")
    else:
        if pno_value:
            logger.warning(f"pNo value {pno_value} not found in week options, falling back to latest week")
        selected_week_value = week_values[1]
        logger.info(f"No pNo in URL, selecting latest week: {selected_week_value}")

    select.select_by_value(selected_week_value)
    time.sleep(1)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//td[@data-test-id='programResult-result']"))
        )
    except Exception:
        logger.warning(f"No results found for week {selected_week_value}")
        return []

    selected_option_text = select.first_selected_option.text.strip()
    logger.info(f"Processing week: {selected_option_text}")

    wait_for_min_matches(driver, expected_matches=15, max_wait_time=15, wait_interval=0.5)
    time.sleep(2)
    scroll_to_load_matches(driver)

    name_elements_selenium, result_elements_selenium, status_elements_selenium = fetch_match_elements_with_retries(
        driver, attempts=3, sleep_between=2
    )

    if len(name_elements_selenium) < 10:
        logger.warning(f"Only found {len(name_elements_selenium)} matches, waiting longer for all matches to load...")
        name_elements_selenium, result_elements_selenium, status_elements_selenium = wait_additional_for_matches(
            driver, min_matches=10, cycles=3, pause_seconds=2
        )

    if (
        len(name_elements_selenium) != len(result_elements_selenium)
        or len(name_elements_selenium) != len(status_elements_selenium)
    ):
        logger.warning(
            f"Mismatch in element counts: names={len(name_elements_selenium)}, "
            f"results={len(result_elements_selenium)}, status={len(status_elements_selenium)}"
        )

    week_results: List[Tuple] = []
    for idx, (name_elem, result_elem, status_elem) in enumerate(
        zip(name_elements_selenium, result_elements_selenium, status_elements_selenium), 1
    ):
        match_name = name_elem.get_attribute("textContent").strip() if name_elem.get_attribute("textContent") else ""
        match_result_raw = result_elem.get_attribute("textContent") if result_elem.get_attribute("textContent") else ""
        match_result = match_result_raw.strip() if match_result_raw else ""
        status_raw_text = status_elem.get_attribute("textContent") if status_elem.get_attribute("textContent") else ""
        
        if match_name and match_result in ["1", "2", "X"]:
            home_goals, away_goals = parse_score_from_status_text(status_raw_text)
            if "-" in match_name:
                teams = match_name.split("-")
                home_team = teams[0].strip()
                away_team = teams[1].strip()
                if home_team and away_team:
                    week_results.append(
                        (home_team, away_team, match_result, selected_option_text, None, home_goals, away_goals)
                    )
        else:
            if match_name:
                reason = f"Result was '{match_result_raw}' (normalized: '{match_result}')"
                if not match_result:
                    reason = "Result was empty or None"
                logger.warning(f"Skipping match: {match_name}, Reason: {reason}")
            else:
                logger.warning(f"Skipping match at index {idx}: Match name was empty")

    logger.info(f"Scraped {len(week_results)} matches from week {selected_option_text}")
    return week_results


def scrape_historical(driver: webdriver.Chrome, url: str, max_weeks: Optional[int]) -> List[Tuple]:
    """
    Scrape historical results across weeks.
    Returns tuples: (home_team, away_team, result, week_range, match_date, home_goals, away_goals)
    """
    logger.info(f"Scraping historical results from {url}")
    driver.get(url)

    select = wait_for_week_selector(driver, timeout=20)
    week_values = [option.get_attribute("value") for option in select.options]
    week_values = week_values[1 : max_weeks + 1] if max_weeks else week_values[1:]

    logger.info(f"Scraping {len(week_values)} weeks of historical data")
    all_results: List[Tuple] = []

    for week_value in week_values:
        try:
            select = wait_for_week_selector(driver, timeout=10)
            select.select_by_value(week_value)
            time.sleep(1)

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//td[@data-test-id='programResult-result']"))
                )
            except Exception:
                logger.warning(f"No results found for week {week_value}")
                continue

            selected_option_text = select.first_selected_option.text.strip()
            logger.info(f"Processing week {week_value} ({selected_option_text})")
            time.sleep(1.5)

            scroll_to_load_matches(driver)
            name_elements_selenium, result_elements_selenium, status_elements_selenium = (
                fetch_match_elements_with_retries(driver, attempts=3, sleep_between=2)
            )

            if len(name_elements_selenium) < 10:
                logger.info("Found fewer matches than expected, trying additional scrolling...")
                name_elements_selenium, result_elements_selenium, status_elements_selenium = wait_additional_for_matches(
                    driver, min_matches=10, cycles=3, pause_seconds=2
                )

            if (
                len(name_elements_selenium) != len(result_elements_selenium)
                or len(name_elements_selenium) != len(status_elements_selenium)
            ):
                logger.warning(
                    f"Mismatch in element counts: names={len(name_elements_selenium)}, "
                    f"results={len(result_elements_selenium)}, status={len(status_elements_selenium)}"
                )

            week_results: List[Tuple] = []
            for idx, (name_elem, result_elem, status_elem) in enumerate(
                zip(name_elements_selenium, result_elements_selenium, status_elements_selenium), 1
            ):
                match_name = name_elem.get_attribute("textContent").strip() if name_elem.get_attribute("textContent") else ""
                match_result = result_elem.get_attribute("textContent").strip() if result_elem.get_attribute("textContent") else ""
                status_raw_text = status_elem.get_attribute("textContent") if status_elem.get_attribute("textContent") else ""

                if match_name and match_result in ["1", "2", "X"]:
                    home_goals, away_goals = parse_score_from_status_text(status_raw_text)
                    if "-" in match_name:
                        teams = match_name.split("-")
                        home_team = teams[0].strip()
                        away_team = teams[1].strip()
                        if home_team and away_team:
                            week_results.append(
                                (home_team, away_team, match_result, selected_option_text, None, home_goals, away_goals)
                            )
            all_results.extend(week_results)
            logger.info(f"Added {len(week_results)} matches from week {week_value}")
        except Exception as exc:
            logger.error(f"Error processing week {week_value}: {exc}")
            continue

    logger.info(f"Total {len(all_results)} historical matches scraped")
    return all_results


