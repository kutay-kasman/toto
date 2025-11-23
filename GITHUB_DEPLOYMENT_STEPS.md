# Step-by-Step GitHub Deployment Guide

## Repository: https://github.com/kutay-kasman/toto

## Files to Commit to GitHub

### ✅ REQUIRED Files (Must Commit)

These files are **essential** for Streamlit Cloud:

1. **`dashboard.py`** - Main Streamlit app
2. **`requirements.txt`** - Python dependencies
3. **`data/matches.db`** - Database with predictions (CRITICAL!)
4. **`src/` directory** - All source code:
   - `src/__init__.py`
   - `src/database.py`
   - `src/data_processing.py`
   - `src/ml_model.py`
   - `src/scraper.py` (optional, but good to include)
5. **`README.md`** - Project documentation
6. **`.gitignore`** - Git ignore rules

### ✅ RECOMMENDED Files (Optional but Good)

7. **`main.py`** - Main pipeline (for reference)
8. **`STREAMLIT_CLOUD_DEPLOYMENT.md`** - Deployment guide
9. **`setup.py`** - Setup script (if exists)

### ❌ DO NOT Commit (Already Ignored)

These are automatically ignored by `.gitignore`:
- `__pycache__/` folders
- `*.pyc` files
- `venv/` or `env/` folders
- `logs/` folder
- `*.log` files
- `data/*.csv` files
- `models/*.pkl` files
- `*.npz` files
- Old script files (optional - you can keep them or remove)

## Step-by-Step Commands

### Step 1: Check Current Status

```bash
git status
```

This shows what files are modified/untracked.

### Step 2: Ensure Database is Up-to-Date

```bash
# Make sure you have latest predictions
python main.py --mode full
```

### Step 3: Add Essential Files

```bash
# Add dashboard and requirements
git add dashboard.py
git add requirements.txt

# Add database (CRITICAL!)
git add data/matches.db

# Add source code
git add src/

# Add documentation
git add README.md
git add .gitignore
git add STREAMLIT_CLOUD_DEPLOYMENT.md
```

### Step 4: Add Optional Files (Your Choice)

```bash
# Main pipeline (recommended)
git add main.py

# Setup script (if you want)
git add setup.py

# Other documentation (optional)
git add *.md
```

### Step 5: Verify What Will Be Committed

```bash
git status
```

**IMPORTANT:** Make sure you see:
- ✅ `data/matches.db` in the list
- ✅ `dashboard.py` in the list
- ✅ `requirements.txt` in the list
- ✅ `src/` files in the list

### Step 6: Commit Changes

```bash
git commit -m "Prepare for Streamlit Cloud deployment

- Add viewer-only dashboard
- Include database with predictions
- Update requirements for cloud deployment
- Add deployment documentation"
```

### Step 7: Push to GitHub

```bash
git push origin main
```

(Or `git push origin master` if your default branch is `master`)

## Quick One-Liner (After Step 2)

If you want to add everything at once (be careful):

```bash
# Add all tracked changes and new files
git add dashboard.py requirements.txt data/matches.db src/ README.md .gitignore STREAMLIT_CLOUD_DEPLOYMENT.md main.py

# Commit
git commit -m "Prepare for Streamlit Cloud deployment"

# Push
git push origin main
```

## Verify Database is Tracked

After pushing, verify the database is on GitHub:

1. Go to: https://github.com/kutay-kasman/toto
2. Navigate to `data/` folder
3. You should see `matches.db` listed
4. Click on it - it should show file size (not "binary file")

## What NOT to Commit

These should NOT appear in `git status` (they're ignored):
- ❌ `__pycache__/`
- ❌ `logs/app.log`
- ❌ `models/xgboost_model.pkl`
- ❌ `*.csv` files in `data/`
- ❌ `*.npz` files
- ❌ `venv/` or `env/`

## Troubleshooting

### Database Not Showing in Git Status

If `data/matches.db` doesn't appear:

```bash
# Force add (if it was previously ignored)
git add -f data/matches.db

# Check if it's now tracked
git status
```

### Database Too Large

If GitHub complains about file size (>100MB):

```bash
# Check database size
ls -lh data/matches.db

# If too large, you may need to:
# 1. Clean old data
# 2. Or use Git LFS (Large File Storage)
```

### Verify .gitignore

Make sure `.gitignore` has this line (commented out):
```
# data/matches.db
```

This means the database is NOT ignored and WILL be committed.

## After Pushing

1. **Go to Streamlit Cloud**: https://share.streamlit.io/
2. **Connect Repository**: https://github.com/kutay-kasman/toto
3. **Configure**:
   - Main file: `dashboard.py`
   - Python version: 3.9+
4. **Deploy!**

## Summary Checklist

Before pushing, verify:
- [ ] `data/matches.db` exists and has data
- [ ] `dashboard.py` is updated (viewer-only version)
- [ ] `requirements.txt` doesn't include Selenium
- [ ] `.gitignore` allows `data/matches.db`
- [ ] All `src/` files are added
- [ ] `git status` shows the right files
- [ ] Database is in the commit list

