"""
Clean duplicate predictions from the database.
Keeps only the most recent prediction for each (home_team, away_team, batch_id) combination.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import MatchDatabase
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_duplicate_predictions():
    """
    Remove duplicate predictions from the database.
    Keeps only the most recent prediction for each match in each batch.
    """
    db = MatchDatabase()
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    # Find duplicates
    cursor.execute("""
        SELECT home_team, away_team, batch_id, COUNT(*) as count
        FROM predictions
        GROUP BY home_team, away_team, batch_id
        HAVING count > 1
    """)
    
    duplicates = cursor.fetchall()
    
    if not duplicates:
        logger.info("✅ No duplicates found!")
        conn.close()
        return
    
    logger.info(f"Found {len(duplicates)} sets of duplicate predictions")
    
    total_deleted = 0
    
    for home, away, batch_id, count in duplicates:
        logger.info(f"Processing: {home} vs {away} (batch: {batch_id}) - {count} duplicates")
        
        # Get all predictions for this match in this batch, ordered by id (newest last)
        cursor.execute("""
            SELECT id FROM predictions
            WHERE home_team = ? AND away_team = ? AND (batch_id = ? OR (batch_id IS NULL AND ? IS NULL))
            ORDER BY id DESC
        """, (home, away, batch_id, batch_id))
        
        ids = [row[0] for row in cursor.fetchall()]
        
        if len(ids) > 1:
            # Keep the newest (last in list after DESC), delete the rest
            keep_id = ids[0]
            delete_ids = ids[1:]
            
            logger.info(f"  Keeping ID {keep_id}, deleting {len(delete_ids)} older predictions")
            
            for del_id in delete_ids:
                cursor.execute("DELETE FROM predictions WHERE id = ?", (del_id,))
                total_deleted += 1
    
    conn.commit()
    conn.close()
    
    logger.info(f"\n✅ Cleanup complete! Deleted {total_deleted} duplicate predictions.")


if __name__ == "__main__":
    print("=" * 60)
    print("  Clean Duplicate Predictions")
    print("=" * 60)
    print()
    
    clean_duplicate_predictions()
    
    print()
    print("=" * 60)
    print("Database cleaned! No more duplicates.")
    print("=" * 60)
