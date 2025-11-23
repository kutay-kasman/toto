# Goal Score Scraping Update

## Overview

Updated the scraper to extract **actual goal scores** from nesine.com instead of estimating them. This provides more accurate data for the ML model.

## Changes Made

### 1. Scraper (`src/scraper.py`)

#### Added `_parse_score_from_match_text()` Method
- Extracts goal scores from match text using regex pattern matching
- Handles multiple patterns:
  - **Finished matches**: `"Team1-Team2Bitti / 0-3"` → extracts `(0, 3)`
  - **Live matches**: `"Team1-Team210' / 0-0"` → extracts `(0, 0)`
  - **No score**: `"Team1-Team2"` → returns `(None, None)`

#### Updated `scrape_historical_results()`
- Now extracts goal scores from match name text
- Returns 7-tuple: `(home_team, away_team, result, week_range, match_date, home_goals, away_goals)`
- Cleans team names by removing score patterns before parsing

#### Updated `scrape_with_cache()`
- Returns matches with goal data from database
- Handles both new scraped data and cached data with goals

### 2. Data Processing (`src/data_processing.py`)

#### Removed Estimation Logic
- **Before**: Estimated goals based on match results (e.g., home win = 2-1)
- **After**: Only uses matches with actual goal data
- **Behavior**: Skips matches without goal data (doesn't estimate)

#### Updated `calculate_goals_statistics()`
- Only counts matches with real goal data (`pd.notna()` check)
- Returns `(0.0, 0.0)` if no valid matches found
- Ensures data quality by using only actual scores

## Score Parsing Patterns

The regex pattern `r'(?:Bitti\s*/\s*|\d+\'\s*/\s*)(\d+)-(\d+)'` matches:

| Pattern | Example | Extracted |
|---------|---------|-----------|
| Finished match | `"Team1-Team2Bitti / 0-3"` | `(0, 3)` |
| Finished match | `"Team1-Team2Bitti / 3-2"` | `(3, 2)` |
| Live match | `"Team1-Team210' / 0-0"` | `(0, 0)` |
| No score | `"Team1-Team2"` | `(None, None)` |

## Database Schema

The database already supports goal storage:
- `home_goals INTEGER` - Home team goals (nullable)
- `away_goals INTEGER` - Away team goals (nullable)

## Handling None Scores

The system gracefully handles cases where scores are `None`:

1. **During Scraping**: Future matches without scores are stored with `NULL` goals
2. **During Feature Calculation**: Matches without goals are skipped (not counted)
3. **During Training**: Only matches with goal data contribute to goal statistics

## Impact

### Benefits
- ✅ **More Accurate**: Uses real goal data instead of estimates
- ✅ **Better Features**: Goal-based features reflect actual team performance
- ✅ **Data Quality**: Only uses verified data, no assumptions

### Considerations
- ⚠️ **Historical Data**: Old matches without goal data won't contribute to goal statistics
- ⚠️ **Retraining Required**: Model should be retrained after scraping new data with goals
- ⚠️ **Future Matches**: Matches without scores (not yet played) are handled gracefully

## Usage

### Scrape New Data with Goals

```bash
# Force refresh to get goal data
python main.py --mode scrape --force-refresh
```

### Retrain Model

```bash
# Retrain with new goal-based features
python main.py --mode train --retrain
```

### Make Predictions

```bash
# Predictions will use goal statistics
python main.py --mode predict
```

## Testing

To verify the scraper is working:

1. Check database for goal data:
   ```python
   from src.database import MatchDatabase
   db = MatchDatabase()
   df = db.get_historical_matches()
   print(df[['Ev Sahibi Takım', 'Deplasman Takımı', 'Home_Goals', 'Away_Goals']].head(10))
   ```

2. Verify score parsing:
   - Finished matches should have integer goals
   - Future matches may have `None` goals (expected)

## Next Steps

1. **Scrape Historical Data**: Run scraper to populate goal data
2. **Retrain Model**: Train new model with goal-based features
3. **Monitor Performance**: Compare accuracy with/without goal features

