# Prediction Accuracy Update Guide

## Overview

The system now includes functionality to automatically measure prediction accuracy by matching your predictions with actual match results from the website.

## How It Works

1. **Predictions are stored** when you run `python main.py --mode predict`
2. **Actual results are scraped** from `nesine.com` and stored in the `historical_matches` table
3. **The system matches** predictions with historical matches by team names
4. **Accuracy is calculated** and stored in the `predictions` table

## Usage

### Step 1: Make Predictions (if not already done)

```bash
python main.py --mode predict
```

This creates predictions for upcoming matches and stores them in the database.

### Step 2: Wait for Matches to Finish

Wait until the matches you predicted have been played and results are available on nesine.com.

### Step 3: Update Accuracy

**Recommended:** The system automatically scrapes **only the latest week's results** (not all weeks) for efficiency:

```bash
python main.py --mode update-accuracy
```

This will:
1. **Scrape only the latest week's results** from nesine.com (fast and efficient)
2. Store them in the database
3. Match your predictions with the actual results
4. Update the `actual_result` and `is_correct` columns
5. Display accuracy summary

**Note:** The system only scrapes the most recent week, not all historical data. This is much faster and exactly what you need to compare with your predictions.

### Step 4: View Accuracy in Dashboard

Open the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

Navigate to **"Historical Accuracy"** page to see:
- Overall accuracy percentage
- Accuracy over time chart
- Confusion matrix
- Recent predictions with results

## Workflow Example

```bash
# 1. Make predictions for upcoming matches
python main.py --mode predict

# 2. Wait for matches to finish...

# 3. Update accuracy (automatically scrapes only latest week)
python main.py --mode update-accuracy

# 4. View results in dashboard
streamlit run dashboard.py
```

## Efficiency

The `update-accuracy` mode is optimized to:
- **Scrape only the latest week** (not all historical data)
- **Match only predictions without results** (skips already updated predictions)
- **Fast execution** - typically completes in under a minute

## How Matching Works

The system matches predictions with historical matches by:
1. **Exact match**: Same home team and away team
2. **Reverse match**: If exact match fails, tries swapped teams (and reverses the result: 1→2, 2→1, X→X)

## Database Schema

The `predictions` table includes:
- `predicted_result`: Your prediction ('1', 'X', or '2')
- `actual_result`: Actual match result (NULL until updated)
- `is_correct`: 1 if prediction matches actual, 0 otherwise
- `probability_1`, `probability_x`, `probability_2`: Prediction probabilities

## Accuracy Metrics

The system calculates:
- **Total Predictions**: Number of predictions with actual results
- **Correct Predictions**: Number of correct predictions
- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: Shows prediction vs actual breakdown

## Troubleshooting

### No predictions found to update
- Make sure you've run predictions first: `python main.py --mode predict`

### No historical matches found
- The `update-accuracy` mode automatically scrapes the latest week
- If you still have issues, try: `python main.py --mode scrape --max-weeks 1`

### Predictions not matching
- Check team name spelling (must match exactly)
- The system automatically scrapes the latest week when you run `update-accuracy`
- Make sure matches have finished and results are available on nesine.com

## Integration with Full Pipeline

You can also include accuracy update in the full pipeline:

```bash
# This will scrape, train, predict, and update accuracy
python main.py --mode full --force-refresh
```

Note: The `--mode full` doesn't automatically update accuracy. Run `update-accuracy` separately after matches finish.

