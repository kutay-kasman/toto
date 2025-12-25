# Football Match Prediction System

A production-ready machine learning pipeline for predicting football match outcomes from nesine.com data.

## Features

- 🔄 **Automated ETL Pipeline**: Scrape → Process → Train → Predict
- 💾 **Database Caching**: SQLite database prevents redundant scraping
- 🤖 **ML Optimization**: XGBoost with hyperparameter tuning and cross-validation
- 📊 **Web Dashboard**: Streamlit interface for predictions and analytics
- 🛡️ **Error Handling**: Robust error handling prevents pipeline crashes
- 📈 **Performance Tracking**: Historical accuracy tracking and model metrics

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── database.py          # SQLite database management
│   ├── scraper.py           # Web scraper with caching
│   ├── data_processing.py   # Data cleaning and feature engineering
│   └── ml_model.py          # ML model with optimization
├── main.py                  # Main pipeline orchestrator
├── dashboard.py             # Streamlit web dashboard
├── requirements.txt         # Python dependencies
├── data/                    # Database and data files
├── models/                  # Saved ML models
└── logs/                    # Application logs
```

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories:**
```bash
mkdir -p data models logs
```

## Usage

### Command Line Interface

#### Full Pipeline (Scrape → Train → Predict)
```bash
python main.py --mode full
```

#### Individual Steps

**Scrape historical data:**
```bash
python main.py --mode scrape
```

**Train model:**
```bash
python main.py --mode train --retrain
```

**Make predictions:**
```bash
python main.py --mode predict
```

#### Options

- `--mode`: Choose `scrape`, `train`, `predict`, or `full`
- `--force-refresh`: Force re-scraping even if data exists
- `--max-weeks N`: Limit number of weeks to scrape
- `--retrain`: Force model retraining

### Web Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

The dashboard provides:
- **This Week's Predictions**: View upcoming match predictions with probabilities
- **Historical Accuracy**: Track prediction performance over time
- **Model Statistics**: Feature importance and model metrics
- **Data Overview**: Explore historical match data

## Workflow

### Initial Setup

1. **Scrape historical data:**
```bash
python main.py --mode scrape --force-refresh
```

2. **Train the model:**
```bash
python main.py --mode train --retrain
```

3. **Make predictions:**
```bash
python main.py --mode predict
```

4. **View dashboard:**
```bash
streamlit run dashboard.py
```

### Weekly Workflow

After matches are played:

1. **Update historical data** (optional, if new results available):
```bash
python main.py --mode scrape
```

2. **Retrain model with new data:**
```bash
python main.py --mode train --retrain
```

3. **Generate new predictions:**
```bash
python main.py --mode predict
```

## Database Schema

The SQLite database (`data/matches.db`) contains:

- **historical_matches**: Past match results
- **upcoming_matches**: Upcoming fixtures
- **predictions**: Model predictions with probabilities
- **model_performance**: Training metrics and model versions

## Machine Learning

### Model Details

- **Algorithm**: XGBoost Classifier
- **Features**: 
  - One-hot encoded team names
  - Win/Draw/Loss ratios (last 10 matches)
- **Classes**: 1 (Home Win), X (Draw), 2 (Away Win)
- **Optimization**: 
  - Class weighting for imbalanced data
  - Cross-validation for validation
  - Hyperparameter tuning support

### Model Persistence

Models are automatically saved to `models/xgboost_model.pkl` after training. The pipeline loads existing models to avoid retraining.

## Configuration

Key parameters can be adjusted in the source code:

- `LOOKBACK_GAMES` (default: 10): Number of previous matches for statistics
- `TEST_SIZE` (default: 0.2): Train/test split ratio

# Football Match Prediction System

A production-ready machine learning pipeline for predicting football match outcomes from nesine.com data.

## Features

- 🔄 **Automated ETL Pipeline**: Scrape → Process → Train → Predict
- 💾 **Database Caching**: SQLite database prevents redundant scraping
- 🤖 **ML Optimization**: XGBoost with hyperparameter tuning and cross-validation
- 📊 **Web Dashboard**: Streamlit interface for predictions and analytics
- 🛡️ **Error Handling**: Robust error handling prevents pipeline crashes
- 📈 **Performance Tracking**: Historical accuracy tracking and model metrics

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── database.py          # SQLite database management
│   ├── scraper.py           # Web scraper with caching
│   ├── data_processing.py   # Data cleaning and feature engineering
│   └── ml_model.py          # ML model with optimization
├── main.py                  # Main pipeline orchestrator
├── dashboard.py             # Streamlit web dashboard
├── requirements.txt         # Python dependencies
├── data/                    # Database and data files
├── models/                  # Saved ML models
└── logs/                    # Application logs
```

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories:**
```bash
mkdir -p data models logs
```

## Usage

### Command Line Interface

#### Full Pipeline (Scrape → Train → Predict)
```bash
python main.py --mode full
```

#### Individual Steps

**Scrape historical data:**
```bash
python main.py --mode scrape
```

**Train model:**
```bash
python main.py --mode train --retrain
```

**Make predictions:**
```bash
python main.py --mode predict
```

#### Options

- `--mode`: Choose `scrape`, `train`, `predict`, or `full`
- `--force-refresh`: Force re-scraping even if data exists
- `--max-weeks N`: Limit number of weeks to scrape
- `--retrain`: Force model retraining

### Web Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

The dashboard provides:
- **This Week's Predictions**: View upcoming match predictions with probabilities
- **Historical Accuracy**: Track prediction performance over time
- **Model Statistics**: Feature importance and model metrics
- **Data Overview**: Explore historical match data

## Workflow

### Initial Setup

1. **Scrape historical data:**
```bash
python main.py --mode scrape --force-refresh
```

2. **Train the model:**
```bash
python main.py --mode train --retrain
```

3. **Make predictions:**
```bash
python main.py --mode predict
```

4. **View dashboard:**
```bash
streamlit run dashboard.py
```

### Weekly Workflow

After matches are played:

1. **Update historical data** (optional, if new results available):
```bash
python main.py --mode scrape
```

2. **Retrain model with new data:**
```bash
python main.py --mode train --retrain
```

3. **Generate new predictions:**
```bash
python main.py --mode predict
```

## Database Schema

The SQLite database (`data/matches.db`) contains:

- **historical_matches**: Past match results
- **upcoming_matches**: Upcoming fixtures
- **predictions**: Model predictions with probabilities
- **model_performance**: Training metrics and model versions

## Machine Learning

### Model Details

- **Algorithm**: XGBoost Classifier
- **Features**: 
  - One-hot encoded team names
  - Win/Draw/Loss ratios (last 10 matches)
- **Classes**: 1 (Home Win), X (Draw), 2 (Away Win)
- **Optimization**: 
  - Class weighting for imbalanced data
  - Cross-validation for validation
  - Hyperparameter tuning support

### Model Persistence

Models are automatically saved to `models/xgboost_model.pkl` after training. The pipeline loads existing models to avoid retraining.

## Configuration

Key parameters can be adjusted in the source code:

- `LOOKBACK_GAMES` (default: 10): Number of previous matches for statistics
- `TEST_SIZE` (default: 0.2): Train/test split ratio
- Database path: `data/matches.db`

## Troubleshooting

### Common Issues

1. **ChromeDriver errors**: Ensure Chrome browser is installed. webdriver-manager will handle driver installation.

2. **No matches found**: Check internet connection and nesine.com accessibility.

3. **Model not found**: Run training first: `python main.py --mode train --retrain`

4. **Database errors**: Delete `data/matches.db` to reset the database.

### Logs

Check `logs/app.log` for detailed error messages and debugging information.

## Code Quality

- Follows PEP 8 style guidelines
- Comprehensive error handling
- Type hints where applicable
- Modular, maintainable architecture
- Extensive logging

## License

This project is for educational purposes.

## Notes

- The scraper uses Selenium with headless Chrome
- Database caching significantly reduces scraping time
- Model performance is tracked in the database
- Predictions can be updated with actual results for accuracy tracking


```

#### Individual Steps

**Scrape historical data:**
```bash
python main.py --mode scrape
```

**Train model:**
```bash
python main.py --mode train --retrain
```

**Make predictions:**
```bash
python main.py --mode predict
```

#### Options

- `--mode`: Choose `scrape`, `train`, `predict`, or `full`
- `--force-refresh`: Force re-scraping even if data exists
- `--max-weeks N`: Limit number of weeks to scrape
- `--retrain`: Force model retraining

### Web Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

The dashboard provides:
- **This Week's Predictions**: View upcoming match predictions with probabilities
- **Historical Accuracy**: Track prediction performance over time
- **Model Statistics**: Feature importance and model metrics
- **Data Overview**: Explore historical match data

## Workflow

### Initial Setup

1. **Scrape historical data:**
```bash
python main.py --mode scrape --force-refresh
```

2. **Train the model:**
```bash
python main.py --mode train --retrain
```

3. **Make predictions:**
```bash
python main.py --mode predict
```

4. **View dashboard:**
```bash
streamlit run dashboard.py
```

### Weekly Workflow

After matches are played:

1. **Update historical data** (optional, if new results available):
```bash
python main.py --mode scrape
```

2. **Retrain model with new data:**
```bash
python main.py --mode train --retrain
```

3. **Generate new predictions:**
```bash
python main.py --mode predict
```

## Database Schema

The SQLite database (`data/matches.db`) contains:

- **historical_matches**: Past match results
- **upcoming_matches**: Upcoming fixtures
- **predictions**: Model predictions with probabilities
- **model_performance**: Training metrics and model versions

## Machine Learning

### Model Details

- **Algorithm**: XGBoost Classifier
- **Features**: 
  - One-hot encoded team names
  - Win/Draw/Loss ratios (last 10 matches)
- **Classes**: 1 (Home Win), X (Draw), 2 (Away Win)
- **Optimization**: 
  - Class weighting for imbalanced data
  - Cross-validation for validation
  - Hyperparameter tuning support

### Model Persistence

Models are automatically saved to `models/xgboost_model.pkl` after training. The pipeline loads existing models to avoid retraining.

## Configuration

Key parameters can be adjusted in the source code:

- `LOOKBACK_GAMES` (default: 10): Number of previous matches for statistics
- `TEST_SIZE` (default: 0.2): Train/test split ratio
- Database path: `data/matches.db`

## Troubleshooting

### Common Issues

1. **ChromeDriver errors**: Ensure Chrome browser is installed. webdriver-manager will handle driver installation.

2. **No matches found**: Check internet connection and nesine.com accessibility.

3. **Model not found**: Run training first: `python main.py --mode train --retrain`

4. **Database errors**: Delete `data/matches.db` to reset the database.

### Logs

Check `logs/app.log` for detailed error messages and debugging information.

## Code Quality

- Follows PEP 8 style guidelines
- Comprehensive error handling
- Type hints where applicable
- Modular, maintainable architecture
- Extensive logging

## License

This project is for educational purposes.

## Notes

- The scraper uses Selenium with headless Chrome
- Database caching significantly reduces scraping time
- Model performance is tracked in the database
- Predictions can be updated with actual results for accuracy tracking

