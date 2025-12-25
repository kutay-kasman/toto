import time
import re
from urllib.parse import urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- 1. Yardımcı Araçlar (Senin importların yerine) ---

def setup_driver():
    options = Options()
    # options.add_argument("--headless") 
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized") # Tabloyu tam görmek için
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def parse_score_from_status_text(text):
    # Örnek metin: "MS 2-1" veya "İY 0-0" -> Buradan 2 ve 1'i alır
    match = re.search(r'(\d+)\s*-\s*(\d+)', text)
    if match:
        return match.group(1), match.group(2)
    return None, None

# --- 2. Ana Mantık ---

def scrape_latest_week(driver, url):
    print(f"🕵️  Bağlanılıyor: {url}")
    
    # URL Analizi
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    pno_value = query_params["pNo"][0] if "pNo" in query_params else None
    
    driver.get(url)

    # 1. Hafta Seçiciyi Bekle
    # ... driver.get(url) satırından sonra ...

    print("Dropdown aranıyor...")
    select_element = None
    
    # YÖNTEM 1: Sayfadaki TÜM <select> etiketlerini çek
    try:
        # Sayfanın yüklenmesi için biraz bekle
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "select")))
        
        all_selects = driver.find_elements(By.TAG_NAME, "select")
        print(f"🔎 Sayfada toplam {len(all_selects)} adet 'select' nesnesi bulundu.")

        # Hepsini kontrol et, içinde "Hafta" mantığına uyanı bul
        for index, sel in enumerate(all_selects):
            try:
                temp_select = Select(sel)
                option_count = len(temp_select.options)
                first_text = temp_select.options[0].text
                print(f"   ➡️ Select #{index}: {option_count} seçenek var. İlk seçenek: '{first_text}'")
                
                # Kural: Hafta listesi genelde 10'dan fazla seçenek içerir
                if option_count > 10:
                    select_element = sel
                    select = temp_select
                    print("   ✅ İşte bu! Hafta seçicisi bulundu.")
                    break
            except:
                continue
                
    except Exception as e:
        print(f"❌ Select bulma hatası: {e}")

    # Eğer Yöntem 1 çalışmazsa, manuel XPATH denemesi (Yedek)
    if select_element is None:
        print("⚠️ Standart arama başarısız, manuel XPATH deneniyor...")
        try:
            # Bazen ID verilir, örneğin 'week-select' vb. (Bunu tahmin ediyoruz)
            select_element = driver.find_element(By.XPATH, "//div[contains(@class, 'filter')]//select")
            select = Select(select_element)
        except:
            print("❌ Hata: Dropdown kesinlikle bulunamadı.")
            return []

    # ... Buradan sonra week_values = ... diye devam eden kod gelecek

    week_values = [option.get_attribute("value") for option in select.options]
    
    if len(week_values) < 2:
        return []

    # 2. Hafta Seçimi
    target_value = week_values[1] # Varsayılan: Son sonuçlanan hafta
    if pno_value and pno_value in week_values:
        target_value = pno_value
    
    print(f"📅 Seçilen Hafta ID: {target_value}")
    select.select_by_value(target_value)
    time.sleep(2) 

    # 3. Sonuçların Yüklenmesini Bekle (GÜNCELLENDİ: programResult-result)
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//td[@data-test-id='programResult-result']"))
        )
    except:
        print("⚠️ Uyarı: Sonuç tablosu yüklenmedi.")
        return []

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    # --- KRİTİK DÜZELTME BURADA ---
    # Ekran görüntüsündeki yeni ID'lere göre güncellendi:
    name_elems = driver.find_elements(By.XPATH, "//td[@data-test-id='programResult-name']")
    result_elems = driver.find_elements(By.XPATH, "//td[@data-test-id='programResult-result']")
    status_elems = driver.find_elements(By.XPATH, "//td[@data-test-id='programResult-status']") 

    print(f"📊 Bulunan Veri: İsimler={len(name_elems)}, Skorlar={len(status_elems)}")

    week_results = []
    
    for name_el, result_el, status_el in zip(name_elems, result_elems, status_elems):
        # HTML içindeki texti alırken .text yerine get_attribute("innerText") bazen daha temizdir
        name = name_el.get_attribute("innerText").strip()
        result = result_el.get_attribute("innerText").strip()
        status_text = status_el.get_attribute("innerText").strip() # Örn: "Bitti / 3-2"
        
        # Sadece geçerli sonuçları al (1, 0, 2)
        if name and result in ["1", "0", "2", "X"]:
            # Regex "Bitti / 3-2" içinden 3 ve 2'yi çekecek
            home_goals, away_goals = parse_score_from_status_text(status_text)
            
            if "-" in name:
                # İsim bazen "Galatasaray-Samsunspor" bazen "Galatasaray - Samsunspor" olabilir
                # Tireden bölmek garanti olsun
                splitter = "-" if "-" in name else "–" # Farklı tire ihtimaline karşı
                parts = name.split(splitter)
                
                if len(parts) >= 2:
                    home = parts[0].strip()
                    away = parts[1].strip()
                    
                    week_results.append((home, away, result, target_value, None, home_goals, away_goals))

    return week_results

# --- 3. Çalıştırma ---
if __name__ == "__main__":
    driver = setup_driver()
    try:
        # İstersen sonuna ?pNo=316 gibi parametre ekleyip deneyebilirsin
        url = "https://www.nesine.com/sportoto/mac-sonuclari" 
        
        results = scrape_latest_week(driver, url)
        
        print("\n" + "="*50)
        print(f"📋 MAÇ SONUÇLARI RAPORU ({len(results)} Maç)")
        print("="*50)
        
        for i, res in enumerate(results, 1):
            # res: (home, away, result, week, date, h_goal, a_goal)
            print(f"{i:02d}. {res[0]} {res[5]}-{res[6]} {res[1]} | Sonuç: {res[2]}")
            
        print("="*50 + "\n")
        
    finally:
        driver.quit()
        print("👋 Driver kapatıldı.")