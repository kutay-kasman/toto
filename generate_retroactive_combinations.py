"""
Utility script to generate combinations for existing predictions.
Run this to retroactively create combinations for predictions made before the combination tracking system was added.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import MatchDatabase
from dashboard import generate_prediction_combinations
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_combinations_for_existing_predictions():
    """
    Generate combinations for existing predictions that don't have combinations yet.
    Groups predictions by batch_id (or creates one if missing).
    """
    db = MatchDatabase()
    
    # Get all predictions
    all_predictions = db.get_predictions(limit=10000)
    
    if all_predictions.empty:
        logger.warning("No predictions found in database.")
        return
    
    logger.info(f"Found {len(all_predictions)} total predictions")
    
    # Group predictions by batch_id (or prediction_date if batch_id is null)
    # First, let's see which predictions already have combinations
    existing_batches = set()
    try:
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT batch_id FROM prediction_combinations")
        existing_batches = {row[0] for row in cursor.fetchall()}
        conn.close()
        logger.info(f"Found {len(existing_batches)} existing batches with combinations")
    except Exception as e:
        logger.warning(f"Could not load existing batches: {e}")
    
    # Group predictions
    # If batch_id is null, create one based on prediction_date
    predictions_by_batch = {}
    
    for _, pred in all_predictions.iterrows():
        batch_id = pred.get('batch_id')
        
        # If no batch_id, create one from prediction_date
        if pd.isna(batch_id) or batch_id is None:
            pred_date = pd.to_datetime(pred['prediction_date'])
            batch_id = f"{pred_date.year}-W{pred_date.strftime('%W')}"
            logger.info(f"Created batch_id {batch_id} from prediction_date {pred_date}")
        
        # Skip if already has combinations
        if batch_id in existing_batches:
            continue
        
        if batch_id not in predictions_by_batch:
            predictions_by_batch[batch_id] = []
        
        predictions_by_batch[batch_id].append(pred)
    
    if not predictions_by_batch:
        logger.info("✅ All predictions already have combinations!")
        return
    
    logger.info(f"Found {len(predictions_by_batch)} batches needing combinations")
    
    # Generate combinations for each batch
    for batch_id, predictions in predictions_by_batch.items():
        logger.info(f"\nProcessing batch: {batch_id} ({len(predictions)} predictions)")
        
        # Convert to DataFrame
        batch_df = pd.DataFrame(predictions)
        
        # Skip if less than 3 matches (too few for meaningful combinations)
        if len(batch_df) < 3:
            logger.warning(f"Skipping {batch_id}: only {len(batch_df)} predictions")
            continue
        
        try:
            # Generate combinations
            combinations = generate_prediction_combinations(batch_df, n_combinations=7)
            
            if combinations:
                # Save to database
                db.save_prediction_combinations(batch_id, combinations)
                logger.info(f"✅ Saved {len(combinations)} combinations for batch {batch_id}")
                
                # Also update the batch_id in predictions table if it was null
                import sqlite3
                conn = sqlite3.connect(db.db_path)
                cursor = conn.cursor()
                
                for pred in predictions:
                    if pd.isna(pred.get('batch_id')):
                        cursor.execute("""
                            UPDATE predictions 
                            SET batch_id = ?
                            WHERE home_team = ? AND away_team = ? AND prediction_date = ?
                        """, (batch_id, pred['home_team'], pred['away_team'], pred['prediction_date']))
                
                conn.commit()
                conn.close()
                logger.info(f"✅ Updated batch_id for predictions in {batch_id}")
            else:
                logger.warning(f"No combinations generated for {batch_id}")
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
    
    logger.info("\n🎉 Done! All existing predictions now have combinations.")


if __name__ == "__main__":
    print("=" * 60)
    print("  Generate Combinations for Existing Predictions")
    print("=" * 60)
    print()
    
    generate_combinations_for_existing_predictions()
    
    print()
    print("=" * 60)
    print("You can now view combinations in the dashboard!")
    print("Run: streamlit run dashboard.py")
    print("=" * 60)
