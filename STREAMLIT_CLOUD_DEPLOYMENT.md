# Streamlit Cloud Deployment Guide

## Overview

This guide explains how to deploy the dashboard to Streamlit Cloud. The dashboard has been refactored to be **viewer-only** - it reads directly from the database without requiring Selenium or ML model dependencies.

## Key Changes for Cloud Deployment

### 1. Dashboard Refactoring (`dashboard.py`)

- ✅ **Removed dependencies** on `src.scraper` and `src.ml_model`
- ✅ **Direct database access** using `sqlite3` and `pandas`
- ✅ **No Selenium required** - reads pre-scraped data from database
- ✅ **No ML model loading** - displays predictions already saved in database

### 2. Requirements (`requirements.txt`)

- ✅ **Minimal dependencies** for Streamlit Cloud
- ✅ **Selenium excluded** (commented out, optional for local use)
- ✅ **XGBoost optional** (only needed for local training)

### 3. Git Configuration (`.gitignore`)

- ✅ **Database file allowed** - `data/matches.db` is NOT ignored
- ✅ **Must be committed** to GitHub for Streamlit Cloud to access it

## Deployment Steps

### Step 1: Prepare Your Repository

1. **Ensure database exists:**
   ```bash
   # Run locally to populate database
   python main.py --mode scrape --force-refresh
   python main.py --mode train --retrain
   python main.py --mode predict
   ```

2. **Verify database is tracked:**
   ```bash
   git status
   # Should show data/matches.db as tracked (not ignored)
   ```

3. **Commit database to Git:**
   ```bash
   git add data/matches.db
   git commit -m "Add database for Streamlit Cloud deployment"
   git push
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
2. **Connect your GitHub repository**
3. **Configure deployment:**
   - **Main file path:** `dashboard.py`
   - **Python version:** 3.9+ (recommended)
   - **Branch:** `main` or `master`

4. **Deploy!**

### Step 3: Update Database (Periodic)

Since the dashboard is viewer-only, you need to update the database locally and push it:

```bash
# 1. Update data locally
python main.py --mode full

# 2. Commit and push updated database
git add data/matches.db
git commit -m "Update match predictions"
git push
```

Streamlit Cloud will automatically redeploy when you push changes.

## Local Testing

Test the dashboard locally without Chrome:

```bash
# Ensure you have the database
ls data/matches.db

# Run dashboard (no Chrome needed!)
streamlit run dashboard.py
```

## File Structure for Cloud

```
.
├── dashboard.py          # Main Streamlit app (viewer-only)
├── requirements.txt     # Minimal dependencies
├── data/
│   └── matches.db       # Database (MUST be committed)
├── src/                 # Source code (not used by dashboard)
└── README.md
```

## Troubleshooting

### Database Not Found

**Error:** "Database not found at data/matches.db"

**Solution:**
- Ensure `data/matches.db` exists and is committed to Git
- Check that `.gitignore` doesn't ignore `data/matches.db`

### Missing Dependencies

**Error:** Import errors in Streamlit Cloud

**Solution:**
- Check `requirements.txt` includes all needed packages
- Ensure versions are compatible with Streamlit Cloud

### No Predictions Showing

**Solution:**
- Run prediction pipeline locally: `python main.py --mode predict`
- Commit updated database: `git add data/matches.db && git commit -m "Update" && git push`

## Workflow

### Development (Local)
```bash
# Scrape, train, predict
python main.py --mode full

# Test dashboard locally
streamlit run dashboard.py
```

### Deployment (Cloud)
```bash
# Update database
python main.py --mode full

# Commit and push
git add data/matches.db
git commit -m "Update predictions"
git push
```

## Notes

- **Database size:** Keep `data/matches.db` under 100MB for GitHub
- **Update frequency:** Update database weekly or as needed
- **No real-time scraping:** Dashboard shows pre-computed predictions
- **No model training:** Model must be trained locally, predictions saved to DB

