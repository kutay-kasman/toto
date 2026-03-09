"""
Main entry point for the Football Match Prediction System.
Orchestrates the ETL pipeline: Scrape -> Process -> Train/Predict.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

from src.database import MatchDatabase
from src.scraper import NesineScraper
from src.data_processing import DataProcessor
from src.ml_model import MLModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, db_path: str = "data/matches.db"):
        """Initialize pipeline components."""
        self.db = MatchDatabase(db_path)
        self.scraper = NesineScraper(self.db)
        self.processor = DataProcessor()
        self.model = MLModel()
    
    def scrape_historical_data(self, force_refresh: bool = False, max_weeks: int = None):
        """
        Scrape historical match results.
        
        Args:
            force_refresh: Force re-scraping even if data exists
            max_weeks: Maximum number of weeks to scrape
        """
        logger.info("Starting historical data scraping")
        
        try:
            matches = self.scraper.scrape_with_cache(force_refresh=force_refresh)
            
            if not matches and force_refresh:
                # If cache was empty, do a fresh scrape
                matches = self.scraper.scrape_historical_results(max_weeks=max_weeks)
                if matches:
                    self.db.insert_historical_matches_batch(matches)
            
            logger.info(f"Historical data collection complete: {len(matches)} matches")
            return len(matches) > 0
            
        except Exception as e:
            logger.error(f"Error in historical data scraping: {e}", exc_info=True)
            return False
    
    def process_data(self):
        """Process and prepare data for training."""
        logger.info("Starting data processing")
        
        try:
            # Get historical data from database
            df = self.db.get_historical_matches()
            
            if df.empty:
                logger.error("No historical data found. Please scrape data first.")
                return False
            
            # Clean data
            df_cleaned = self.processor.clean_data(df)
            
            # Create features
            df_features = self.processor.create_features(df_cleaned)
            
            # Prepare training data
            X, y, feature_columns = self.processor.prepare_training_data(df_features)
            
            # Store feature columns in model
            self.model.set_feature_columns(feature_columns)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X.values, y.values, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info("Data processing complete")
            return (X_train, X_test, y_train, y_test, df_cleaned)
            
        except Exception as e:
            logger.error(f"Error in data processing: {e}", exc_info=True)
            return False
    
    def train_model(self, retrain: bool = True, optimize: bool = False, n_trials: int = 20):
        """
        Train the ML model.
        
        Args:
            retrain: If True, train new model. If False, try to load existing.
        """
        logger.info("Starting model training")
        
        try:
            # Try to load existing model if not retraining
            if not retrain:
                if self.model.load():
                    logger.info("Loaded existing model")
                    return True
            
            # Process data
            data = self.process_data()
            if not data:
                return False
            
            X_train, X_test, y_train, y_test, df_cleaned = data
            
            # Train model
            metrics = self.model.train(
                X_train, y_train, X_test, y_test,
                use_cross_validation=True,
                optimize=optimize,
                n_trials=n_trials
            )
            
            # Save model
            model_path = self.model.save()
            
            # Save performance metrics to database
            if 'test_accuracy' in metrics:
                report = metrics.get('classification_report', {})
                precision = report.get('weighted avg', {}).get('precision', 0)
                recall = report.get('weighted avg', {}).get('recall', 0)
                f1 = report.get('weighted avg', {}).get('f1-score', 0)
                
                self.db.save_model_performance(
                    model_name='XGBoost',
                    accuracy=metrics['test_accuracy'],
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    model_path=model_path
                )
            
            logger.info("Model training complete")
            return True
            
        except Exception as e:
            logger.error(f"Error in model training: {e}", exc_info=True)
            return False
    
    def predict_upcoming_matches(self):
        """Scrape upcoming matches and make predictions."""
        logger.info("Starting prediction pipeline")
        
        try:
            # Load model
            if not self.model.load():
                logger.warning("No saved model found. Training new model...")
                if not self.train_model(retrain=True):
                    logger.error("Failed to train model")
                    return False
            
            # Scrape upcoming matches
            matches = self.scraper.scrape_upcoming_matches()
            
            if not matches:
                logger.error("No upcoming matches found")
                return False
            
            matches_df = pd.DataFrame(matches, columns=['Ev Sahibi Takım', 'Deplasman Takımı'])
            
            # Get historical data for feature calculation
            historical_df = self.db.get_historical_matches()
            
            # Make predictions
            predictions_df = self.model.predict_matches(
                matches_df, historical_df, self.model.feature_columns
            )
            
            # Generate batch_id first (e.g., "2026-W05")
            from datetime import datetime
            batch_id = f"{datetime.now().year}-W{datetime.now().strftime('%W')}"
            
            # Store predictions in database with batch_id
            for _, pred in predictions_df.iterrows():
                self.db.insert_prediction(
                    home_team=pred['Ev Sahibi Takım'],
                    away_team=pred['Deplasman Takımı'],
                    predicted_result=pred['Predicted_Result'],
                    probability_1=pred['Probability_1'] / 100,
                    probability_x=pred['Probability_X'] / 100,
                    probability_2=pred['Probability_2'] / 100,
                    batch_id=batch_id
                )
            
            # Generate and save prediction combinations
            from dashboard import generate_prediction_combinations
            
            # Get predictions with batch_id
            predictions_for_combos = self.db.get_predictions_by_batch(batch_id)
            
            # Generate 7 different combinations
            combinations = generate_prediction_combinations(predictions_for_combos, n_combinations=7)
            
            # Save combinations to database
            if combinations:
                self.db.save_prediction_combinations(batch_id, combinations)
                logger.info(f"Saved {len(combinations)} combinations to database with batch_id: {batch_id}")
            
            # Print results
            self._print_predictions(predictions_df)
            
            logger.info("Prediction pipeline complete")
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}", exc_info=True)
            return False
    
    def update_prediction_accuracy(self, scrape_latest: bool = True, weeks: int = 1):
        """
        Update predictions with actual results from historical matches.

        Args:
            scrape_latest: If True, scrape recent weeks first.
            weeks: How many recent weeks to scrape (default 1 = latest only).
        """
        import difflib
        logger.info("Starting prediction accuracy update")
        
        try:
            # Scrape recent weeks
            if scrape_latest:
                if weeks > 1:
                    logger.info(f"Scraping last {weeks} weeks of results...")
                    latest_matches = self.scraper.scrape_recent_weeks(n_weeks=weeks)
                else:
                    logger.info("Scraping latest week's results...")
                    latest_matches = self.scraper.scrape_latest_week_results()
                
                if latest_matches:
                    self.db.insert_historical_matches_batch(latest_matches)
                    logger.info(f"Stored {len(latest_matches)} matches from {weeks} week(s)")
                else:
                    logger.warning("No matches found")
            
            # Get all predictions without actual results
            predictions = self.db.get_predictions(limit=10000)
            predictions_without_results = predictions[predictions['actual_result'].isna()]
            
            if predictions_without_results.empty:
                logger.info("No predictions need updating")
                return True
            
            logger.info(f"Found {len(predictions_without_results)} predictions without results")
            
            # Debug: Print all predictions that need updating
            print("\n" + "="*60)
            print("           PREDICTIONS WAITING FOR RESULTS")
            print("="*60)
            for idx, (_, pred) in enumerate(predictions_without_results.iterrows(), 1):
                print(f"{idx:02}. {pred['home_team']:<30} vs {pred['away_team']:<30} (Predicted: {pred['predicted_result']})")
            print("="*60 + "\n")
            
            # Get historical matches with results (prioritize latest week if scraped)
            historical_df = self.db.get_historical_matches()
            historical_with_results = historical_df[
                historical_df['Mac Sonucu'].isin(['1', 'X', '2'])
            ]
            
            if historical_with_results.empty:
                logger.warning("No historical matches with results found")
                return False
            
            # Debug: Print all scraped matches with results
            print("\n" + "="*60)
            print("           SCRAPED MATCHES WITH RESULTS")
            print("="*60)
            # Show only the most recent matches (last 20)
            recent_matches = historical_with_results.tail(20)
            for idx, (_, match) in enumerate(recent_matches.iterrows(), 1):
                result = match['Mac Sonucu']
                home_goals = match.get('Home_Goals', 'N/A')
                away_goals = match.get('Away_Goals', 'N/A')
                print(f"{idx:02}. {match['Ev Sahibi Takım']:<30} vs {match['Deplasman Takımı']:<30} "
                      f"Result: {result} ({home_goals}-{away_goals})")
            print(f"\nTotal matches with results in database: {len(historical_with_results)}")
            print("="*60 + "\n")
            
            # Match predictions with historical matches
            updated_count = 0
            for _, pred in predictions_without_results.iterrows():
                home_team = pred['home_team']
                away_team = pred['away_team']
                
                # Find matching historical match
                # Try exact match first
                match = historical_with_results[
                    (historical_with_results['Ev Sahibi Takım'] == home_team) &
                    (historical_with_results['Deplasman Takımı'] == away_team)
                ]
                
                if match.empty:
                    # Fuzzy fallback: find closest team name pair
                    all_home = historical_with_results['Ev Sahibi Takım'].tolist()
                    all_away = historical_with_results['Deplasman Takımı'].tolist()
                    home_close = difflib.get_close_matches(home_team, all_home, n=1, cutoff=0.75)
                    away_close = difflib.get_close_matches(away_team, all_away, n=1, cutoff=0.75)
                    if home_close and away_close:
                        match = historical_with_results[
                            (historical_with_results['Ev Sahibi Takım'] == home_close[0]) &
                            (historical_with_results['Deplasman Takımı'] == away_close[0])
                        ]
                        if not match.empty:
                            logger.info(f"Fuzzy match: '{home_team}' -> '{home_close[0]}', '{away_team}' -> '{away_close[0]}'")

                if match.empty:
                    # Try reverse match (teams might be swapped)
                    match = historical_with_results[
                        (historical_with_results['Ev Sahibi Takım'] == away_team) &
                        (historical_with_results['Deplasman Takımı'] == home_team)
                    ]
                    # If found reverse match, need to reverse the result
                    if not match.empty:
                        actual_result = match.iloc[0]['Mac Sonucu']
                        # Reverse: 1->2, 2->1, X->X
                        if actual_result == '1':
                            actual_result = '2'
                        elif actual_result == '2':
                            actual_result = '1'
                        # X stays X
                        logger.info(f"Found reverse match: {home_team} vs {away_team} -> {actual_result}")
                    else:
                        logger.debug(f"No match found for: {home_team} vs {away_team}")
                        continue
                else:
                    actual_result = match.iloc[0]['Mac Sonucu']
                    logger.info(f"Found exact match: {home_team} vs {away_team} -> {actual_result}")
                
                # Update prediction with actual result
                self.db.update_prediction_result(home_team, away_team, actual_result)
                updated_count += 1
                logger.info(f"[OK] Updated prediction: {home_team} vs {away_team} -> {actual_result}")
            
            logger.info(f"Updated {updated_count} predictions with actual results")
            
            # Print accuracy summary
            accuracy = self.db.get_prediction_accuracy()
            if accuracy['total'] > 0:
                print("\n" + "="*60)
                print("           PREDICTION ACCURACY SUMMARY")
                print("="*60)
                print(f"Total Predictions: {accuracy['total']}")
                print(f"Correct Predictions: {accuracy['correct']}")
                print(f"Accuracy: {accuracy['accuracy']:.2f}%")
                print("="*60 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating prediction accuracy: {e}", exc_info=True)
            return False
    
    def _print_predictions(self, predictions_df: pd.DataFrame):
        """Print predictions in formatted output."""
        print("\n" + "="*60)
        print("           SPOR TOTO TAHMİNLERİ")
        print("="*60)
        
        for i, (_, pred) in enumerate(predictions_df.iterrows(), 1):
            print(f"\n{i:02}. {pred['Ev Sahibi Takım']:<25} - {pred['Deplasman Takımı']:<25}")
            print(f"    TAHMİN: {pred['Predicted_Result']}")
            print(f"    OLASILIKLAR: 1: {pred['Probability_1']:.1f}%, "
                  f"X: {pred['Probability_X']:.1f}%, 2: {pred['Probability_2']:.1f}%")
        
        print("\n" + "-"*60)
        print("Sıralı Kupon Tahminleri (15 Maç):")
        coupon = " ".join(predictions_df['Predicted_Result'].tolist())
        print(coupon)
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Football Match Prediction System'
    )
    parser.add_argument(
        '--mode',
        choices=['scrape', 'train', 'predict', 'update-accuracy', 'full'],
        default='full',
        help='Pipeline mode: scrape (data only), train (model only), '
             'predict (predictions only), update-accuracy (update predictions with results), '
             'full (all steps)'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force re-scraping of historical data'
    )
    parser.add_argument(
        '--max-weeks',
        type=int,
        default=None,
        help='Maximum number of weeks to scrape'
    )
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force retraining of model'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run hyperparameter optimization using Optuna'
    )
    parser.add_argument(
        '--weeks',
        type=int,
        default=1,
        help='Number of recent result weeks to scrape for update-accuracy (default: 1)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of trials for hyperparameter optimization'
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    pipeline = PredictionPipeline()
    
    success = True
    
    if args.mode in ['scrape', 'full']:
        success = pipeline.scrape_historical_data(
            force_refresh=args.force_refresh,
            max_weeks=args.max_weeks
        )
        if not success:
            logger.error("Scraping failed")
            return 1
    
    if args.mode in ['train', 'full']:
        success = pipeline.train_model(
            retrain=args.retrain,
            optimize=args.optimize,
            n_trials=args.n_trials
        )
        if not success:
            logger.error("Training failed")
            return 1
    
    if args.mode in ['predict', 'full']:
        result = pipeline.predict_upcoming_matches()
        # predict_upcoming_matches returns DataFrame on success, False on failure
        if result is False or (isinstance(result, pd.DataFrame) and len(result) == 0):
            logger.error("Prediction failed")
            return 1
    
    if args.mode == 'update-accuracy':
        # Scrape latest week's results and update accuracy
        # scrape_latest=True means it will scrape only the latest week (efficient)
        success = pipeline.update_prediction_accuracy(scrape_latest=True, weeks=args.weeks)
        if not success:
            logger.error("Accuracy update failed")
            return 1
    
    logger.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

