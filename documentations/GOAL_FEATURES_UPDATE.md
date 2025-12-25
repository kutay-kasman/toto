# Goal-Based Features Update

## Overview

Added granular statistical features to improve prediction accuracy by including goal-based metrics for both home and away teams.

## New Features Added

### 1. Average Goals Scored (Attacking Strength)
- **Home_Avg_Goals_Scored**: Average goals scored by home team in last `LOOKBACK_GAMES` matches
- **Away_Avg_Goals_Scored**: Average goals scored by away team in last `LOOKBACK_GAMES` matches

### 2. Average Goals Conceded (Defensive Weakness)
- **Home_Avg_Goals_Conceded**: Average goals conceded by home team in last `LOOKBACK_GAMES` matches
- **Away_Avg_Goals_Conceded**: Average goals conceded by away team in last `LOOKBACK_GAMES` matches

## Changes Made

### 1. Database Schema (`src/database.py`)
- Added `home_goals` and `away_goals` columns to `historical_matches` table
- Updated `insert_historical_match()` and `insert_historical_matches_batch()` to support goal data
- Updated `get_historical_matches()` to return goal columns

### 2. Data Processing (`src/data_processing.py`)
- Added `calculate_goals_statistics()` method to calculate average goals scored/conceded
- Updated `create_features()` to include goal-based features
- Updated `prepare_training_data()` to include goal features in training data
- **Fallback Logic**: If goal data is not available, estimates goals based on match results:
  - Home win: 2-1
  - Draw: 1-1
  - Home loss: 0-2
  - (Similar logic for away team)

### 3. ML Model (`src/ml_model.py`)
- Updated `predict_matches()` to calculate goal statistics for predictions
- Goal features are now included in prediction feature matrix
- Results DataFrame includes goal statistics for analysis

## Feature Set

The model now uses **10 statistical features** (plus one-hot encoded team names):

**Win/Loss Ratios (6 features):**
- Home_Win_Ratio
- Home_Draw_Ratio
- Home_Loss_Ratio
- Away_Win_Ratio
- Away_Draw_Ratio
- Away_Loss_Ratio

**Goal Statistics (4 features):**
- Home_Avg_Goals_Scored
- Home_Avg_Goals_Conceded
- Away_Avg_Goals_Scored
- Away_Avg_Goals_Conceded

## Usage

### Retraining with New Features

Since the feature set has changed, you need to retrain the model:

```bash
# Retrain model with new features
python main.py --mode train --retrain
```

### Making Predictions

The goal features are automatically calculated when making predictions:

```bash
python main.py --mode predict
```

## Notes

1. **Goal Data Availability**: Currently, the scraper doesn't extract goal scores from nesine.com. The system uses estimated goals based on match results. To get actual goal data:
   - Update the scraper to extract goal scores from the website
   - Or manually add goal data to the database

2. **Backward Compatibility**: The system works with or without goal data:
   - If goal data is available: Uses actual goals
   - If goal data is missing: Estimates based on results

3. **Model Retraining Required**: Existing models were trained without goal features. You must retrain to use the new features.

## Expected Impact

These features should improve prediction accuracy by:
- Capturing attacking strength (goals scored)
- Capturing defensive weakness (goals conceded)
- Providing more granular team performance metrics
- Better distinguishing between teams with similar win/loss ratios but different goal patterns

## Next Steps (Optional)

1. **Scrape Goal Data**: Update `src/scraper.py` to extract actual goal scores from nesine.com
2. **Additional Features**: Consider adding:
   - Goal difference (scored - conceded)
   - Goals per game ratio
   - Clean sheet statistics
   - Head-to-head goal statistics

