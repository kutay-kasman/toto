import time
import re
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# --- 1. Sürücü Hazırlığı (Basitleştirilmiş) ---
def setup_driver():
    options = Options()
    options.add_argument("--headless")  # Arka planda çalışsın istersen bunu aç
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # Chrome driver'ı otomatik indirip kurar
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

# --- 2. Ana Scraper Mantığı ---
def scrape_upcoming(driver, url):
    print(f"Bağlanılıyor: {url}...")
    driver.get(url)

    # Elementlerin yüklenmesini bekle (Timeout: 15 sn)
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.XPATH, "//td[@data-test-id='name']/button[@data-testid='nsn-button']")
            )
        )
    except Exception:
        print("❌ Hata: Sayfa geç yüklendi veya element bulunamadı.")
        return []

    time.sleep(2)  # Tam yükleme için kısa bir bekleme
    soup = BeautifulSoup(driver.page_source, "html.parser")
    match_elements = soup.find_all("button", attrs={"data-testid": "nsn-button"})

    if not match_elements:
        print("⚠️ Uyarı: Maç elementi bulunamadı.")
        return []

    print(f"✅ Toplam {len(match_elements)} ham veri bulundu. İşleniyor...")

    matches = []
    for element in match_elements:
        full_match_name = element.get_text().strip()
        # Satır sonlarını temizle
        match_teams_only = full_match_name.split("\n")[0].strip()
        # Tarih ve saat bilgisini regex ile uçur
        match_teams_only = re.sub(r"\s*\d{2}\.\d{2}\.\d{4}\s*\d{2}:\d{2}", "", match_teams_only).strip()
        
        if "-" in match_teams_only:
            teams = match_teams_only.split("-")
            if len(teams) >= 2:
                home_team = teams[0].strip()
                away_team = teams[1].strip()
                matches.append((home_team, away_team))

    # Sadece son 15 maçı döndür
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

# --- 3. Çalıştırma Alanı (Trigger) ---
if __name__ == "__main__":
    driver = setup_driver()
    try:
        url = "https://www.nesine.com/sportoto"
        results = scrape_upcoming(driver, url)
        
        print("\n" + "="*40)
        print("📢 SON 15 MAÇ LİSTESİ")
        print("="*40)
        
        if results:
            for i, (home, away) in enumerate(results, 1):
                print(f"{i:02d}. 🏠 {home} - 🚌 {away}")
        else:
            print("Maç bulunamadı.")
            
        print("="*40 + "\n")
        
    finally:
        driver.quit()
        print("👋 Driver kapatıldı, işlem tamam.")