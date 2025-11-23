# Spor Toto Football Match Prediction Workflow Guide

## Overview
This system predicts weekly football match results using machine learning. Each week, new match results are collected, the model is updated, and predictions are generated for the upcoming week.

## File Purpose Summary

### 1. Data Collection & Loading
- **`nesine_cekici.py`**: Scrapes historical match results from nesine.com
  - Output: `spor_toto_sonuclari_YYYYMMDD_HHMMSS.csv` (timestamped CSV files)
  - Collects all weeks' results from the website

### 2. Data Preprocessing
- **`clear_data.py`**: Cleans raw CSV data
  - Input: `spor_toto_sonuclari_*.csv` (update FILE_NAME variable)
  - Output: `temizlenmis_spor_toto_verisi.csv`
  - Separates team names, removes invalid rows

### 3. Feature Engineering
- **`encoding.py`**: Basic one-hot encoding (DEPRECATED - not used in main workflow)
  - Output: `ml_hazir_veri.npz`
  
- **`feature_engineering.py`**: Adds form scores based on last 5 matches (ALTERNATIVE)
  - Output: `ml_hazir_veri_form_ekli.npz`
  
- **`takim_analiz.py`** (or `takim_takim.py` - they are identical): **PRIMARY FEATURE ENGINEERING**
  - Input: `temizlenmis_spor_toto_verisi.csv`
  - Output: `ml_hazir_veri_ISTATISTIKSEL.npz`
  - Creates statistical features: Win/Draw/Loss ratios for last 10 matches
  - This is the file used by the prediction script

### 4. Model Training
- **`model_training.py`**: Random Forest with form features (ALTERNATIVE)
  - Input: `ml_hazir_veri_form_ekli.npz`
  
- **`model_training_xgboost.py`**: XGBoost with form features (ALTERNATIVE)
  - Input: `ml_hazir_veri_form_ekli.npz`
  
- **`model_training_nn.py`**: Neural Network (MLP) with statistical features
  - Input: `ml_hazir_veri_ISTATISTIKSEL.npz`
  - **NOTE**: This trains but doesn't save the model. The prediction script retrains instead.

### 5. Prediction
- **`gercek_tahmin.py`**: **MAIN PREDICTION SCRIPT**
  - Input: `ml_hazir_veri_ISTATISTIKSEL.npz`, `temizlenmis_spor_toto_verisi.csv`
  - Uses XGBoost (not Neural Network) - retrains model each run
  - Fetches upcoming matches from nesine.com
  - Output: Predictions for 15 matches in coupon format

### 6. Match Program Scraping
- **`mac_programi_cek.py`**: Scrapes upcoming match fixtures from nesine.com
  - Function: `cek_mac_programi()` (also aliased as `cek_atlanan_hafta_maclari()`)
  - Used by `gercek_tahmin.py`

### 7. Analysis Tools (Optional)
- **`derbi.py`**: Analyzes head-to-head statistics between two teams
- **`nesine_cekici.py`**: Historical data scraper (already covered above)

---

## Complete Workflow: Step-by-Step Execution Order

### Phase 1: Collect New Week's Results (After matches are played)

**Step 1.1: Scrape Historical Results**
```bash
python nesine_cekici.py
```
- This scrapes ALL historical results from nesine.com
- Creates: `spor_toto_sonuclari_YYYYMMDD_HHMMSS.csv`
- **Note**: If you only want the latest week, you may need to manually add it to your existing CSV

**Step 1.2: Clean the Data**
1. Open `clear_data.py`
2. Update `FILE_NAME` variable with your new CSV filename (line 5)
3. Run:
```bash
python clear_data.py
```
- Creates/updates: `temizlenmis_spor_toto_verisi.csv`
- This file accumulates all historical data

### Phase 2: Prepare Training Data

**Step 2.1: Generate Statistical Features**
```bash
python takim_analiz.py
```
- Reads: `temizlenmis_spor_toto_verisi.csv`
- Creates: `ml_hazir_veri_ISTATISTIKSEL.npz`
- Calculates Win/Draw/Loss ratios for last 10 matches per team
- **This is the primary feature engineering step**

### Phase 3: Generate Predictions for Next Week

**Step 3.1: Run Prediction Script**
```bash
python gercek_tahmin.py
```
- Loads: `ml_hazir_veri_ISTATISTIKSEL.npz` and `temizlenmis_spor_toto_verisi.csv`
- Retrains XGBoost model (trains fresh each time)
- Fetches upcoming matches from nesine.com
- Generates predictions for 15 matches
- Displays results in coupon format

---

## Complete Workflow Summary (Quick Reference)

### For Training/Updating Model with New Week's Data:
```
1. python nesine_cekici.py                    # Collect new results
2. Edit clear_data.py → Update FILE_NAME      # Set new CSV filename
3. python clear_data.py                       # Clean data
4. python takim_analiz.py                     # Generate features
```

### For Generating Predictions:
```
5. python gercek_tahmin.py                    # Generate predictions
```

---

## File Dependencies

### Data Flow:
```
nesine_cekici.py
    ↓ (spor_toto_sonuclari_*.csv)
clear_data.py
    ↓ (temizlenmis_spor_toto_verisi.csv)
takim_analiz.py
    ↓ (ml_hazir_veri_ISTATISTIKSEL.npz)
gercek_tahmin.py
    ↓ (uses both files above)
    → Predictions Output
```

### Key Variables Shared Between Scripts:

1. **`FILE_NAME` in `clear_data.py`** (line 5):
   - Must match the CSV file from `nesine_cekici.py`
   - Example: `"spor_toto_sonuclari_20251110_004409.csv"`

2. **`FILE_NAME` in `takim_analiz.py`** (line 8):
   - Must be: `"temizlenmis_spor_toto_verisi.csv"`
   - This is the cleaned data file

3. **`FILE_HISTORICAL` in `gercek_tahmin.py`** (line 9):
   - Must be: `"temizlenmis_spor_toto_verisi.csv"`
   - Used to calculate team statistics for predictions

4. **`FILE_MODEL_DATA` in `gercek_tahmin.py`** (line 10):
   - Must be: `"ml_hazir_veri_ISTATISTIKSEL.npz"`
   - Contains preprocessed training data

5. **`LOOKBACK_GAMES`**:
   - `takim_analiz.py`: 10 (last 10 matches for statistics)
   - `gercek_tahmin.py`: 10 (same, for consistency)

---

## Important Notes

### Model Training Approach:
- **`gercek_tahmin.py` retrains the model every time it runs**
- It does NOT load a saved model - it trains XGBoost from scratch
- The Neural Network in `model_training_nn.py` is trained but not used for predictions
- The prediction script uses XGBoost, not the Neural Network

### Data Updates:
- Each week, you need to:
  1. Run `nesine_cekici.py` to get latest results (or manually update CSV)
  2. Update `clear_data.py` with new filename
  3. Run cleaning and feature engineering
  4. The model will automatically include new data when `gercek_tahmin.py` runs

### Team Name Consistency:
- Team names must match exactly between:
  - Historical data (`temizlenmis_spor_toto_verisi.csv`)
  - Upcoming matches (scraped from nesine.com)
- Check for variations like "A.Ş." suffix, spaces, etc.

### Fix Applied:
- Fixed import issue in `mac_programi_cek.py`: Added alias `cek_atlanan_hafta_maclari = cek_mac_programi` for compatibility with `gercek_tahmin.py`

---

## Alternative Workflows

### Using Form Features (instead of statistical ratios):
1. `feature_engineering.py` → `ml_hazir_veri_form_ekli.npz`
2. `model_training_xgboost.py` or `model_training.py` → Train model
3. Modify `gercek_tahmin.py` to use form features instead

### Using Neural Network:
1. Run `takim_analiz.py` → `ml_hazir_veri_ISTATISTIKSEL.npz`
2. Run `model_training_nn.py` → Trains but doesn't save model
3. Modify `gercek_tahmin.py` to use the Neural Network model (currently uses XGBoost)

---

## Troubleshooting

### Common Issues:

1. **Import Error**: `cek_atlanan_hafta_maclari not found`
   - ✅ Fixed: Added alias in `mac_programi_cek.py`

2. **File Not Found**: Check FILE_NAME variables match actual filenames

3. **Team Name Mismatch**: Verify team names are consistent across all files

4. **Empty Predictions**: Ensure `mac_programi_cek.py` can access nesine.com (requires Chrome/ChromeDriver)

5. **Model Accuracy**: The model retrains each time, so accuracy may vary. Consider saving/loading models for consistency.

