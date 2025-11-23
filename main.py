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
    
    def train_model(self, retrain: bool = True):
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
                use_cross_validation=True
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
            
            # Store predictions in database
            for _, pred in predictions_df.iterrows():
                self.db.insert_prediction(
                    home_team=pred['Ev Sahibi Takım'],
                    away_team=pred['Deplasman Takımı'],
                    predicted_result=pred['Predicted_Result'],
                    probability_1=pred['Probability_1'] / 100,
                    probability_x=pred['Probability_X'] / 100,
                    probability_2=pred['Probability_2'] / 100
                )
            
            # Print results
            self._print_predictions(predictions_df)
            
            logger.info("Prediction pipeline complete")
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}", exc_info=True)
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
        choices=['scrape', 'train', 'predict', 'full'],
        default='full',
        help='Pipeline mode: scrape (data only), train (model only), '
             'predict (predictions only), full (all steps)'
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
        success = pipeline.train_model(retrain=args.retrain)
        if not success:
            logger.error("Training failed")
            return 1
    
    if args.mode in ['predict', 'full']:
        result = pipeline.predict_upcoming_matches()
        # predict_upcoming_matches returns DataFrame on success, False on failure
        if result is False or (isinstance(result, pd.DataFrame) and len(result) == 0):
            logger.error("Prediction failed")
            return 1
    
    logger.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

