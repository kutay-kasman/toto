"""
Database module for storing and retrieving match data.
Uses SQLite for local storage with caching capabilities.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MatchDatabase:
    """Manages SQLite database for match data storage and retrieval."""
    
    def __init__(self, db_path: str = "data/matches.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Historical matches table
        # Note: SQLite allows multiple NULL values in UNIQUE constraints (they are considered distinct)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                result TEXT NOT NULL,
                home_goals INTEGER,
                away_goals INTEGER,
                week_range TEXT,
                match_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(home_team, away_team, match_date)
            )
        """)
        
        # Add goal columns if they don't exist (for existing databases)
        try:
            cursor.execute("ALTER TABLE historical_matches ADD COLUMN home_goals INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE historical_matches ADD COLUMN away_goals INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Upcoming matches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS upcoming_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                match_date TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(home_team, away_team, match_date)
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                predicted_result TEXT NOT NULL,
                probability_1 REAL,
                probability_x REAL,
                probability_2 REAL,
                batch_id TEXT,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                actual_result TEXT,
                is_correct INTEGER,
                UNIQUE(home_team, away_team, prediction_date)
            )
        """)
        
        # Add batch_id column if it doesn't exist (for existing databases)
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN batch_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Unique index on (home_team, away_team, batch_id) – prevents duplicate
        # rows when the prediction pipeline is run more than once for the same week.
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_batch_match
            ON predictions (home_team, away_team, batch_id)
            WHERE batch_id IS NOT NULL
        """)
        
        # Model performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_path TEXT
            )
        """)
        
        # Prediction combinations table - stores all generated combinations per batch
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_combinations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                combination_rank INTEGER NOT NULL,
                predictions TEXT NOT NULL,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Combination results table - tracks which combination was correct
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS combination_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                winning_rank INTEGER,
                total_correct INTEGER,
                total_matches INTEGER,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(batch_id)
            )
        """)
        
        # Monte Carlo results table - stores simulation results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS monte_carlo_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                attempts_to_perfect INTEGER NOT NULL,
                theoretical_probability REAL NOT NULL,
                expected_attempts INTEGER NOT NULL,
                simulation_time REAL,
                is_success INTEGER DEFAULT 0,
                best_correct INTEGER,
                best_attempt INTEGER,
                total_matches INTEGER,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(batch_id)
            )
        """)
        
        # Add newer columns for existing databases
        for col, definition in [
            ("is_success",  "INTEGER DEFAULT 0"),
            ("best_correct", "INTEGER"),
            ("best_attempt", "INTEGER"),
            ("total_matches", "INTEGER"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE monte_carlo_results ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass  # column already exists
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def insert_historical_match(self, home_team: str, away_team: str, 
                                result: str, week_range: Optional[str] = None,
                                match_date: Optional[str] = None,
                                home_goals: Optional[int] = None,
                                away_goals: Optional[int] = None) -> bool:
        """
        Insert a historical match result.
        
        Returns:
            True if inserted, False if already exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO historical_matches 
                (home_team, away_team, result, week_range, match_date, home_goals, away_goals)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (home_team, away_team, result, week_range, match_date, home_goals, away_goals))
            conn.commit()
            inserted = cursor.rowcount > 0
            conn.close()
            return inserted
        except sqlite3.Error as e:
            logger.error(f"Error inserting historical match: {e}")
            conn.close()
            return False
    
    def insert_historical_matches_batch(self, matches: List[Tuple]):
        """
        Insert multiple historical matches in batch.
        Updates existing records if they have NULL goals and new data has goals.
        
        Args:
            matches: List of tuples. Can be:
                - (home_team, away_team, result, week_range, match_date) - 5 elements
                - (home_team, away_team, result, week_range, match_date, home_goals, away_goals) - 7 elements
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check tuple length to determine if goals are included
            if matches and len(matches[0]) == 7:
                # Update existing records with goals, or insert new ones
                # This ensures that if a match exists with NULL goals, it gets updated with the new goals
                updated_count = 0
                inserted_count = 0
                
                for match in matches:
                    home_team, away_team, result, week_range, match_date, home_goals, away_goals = match
                    
                    # Check if match exists (match by home_team + away_team, since match_date is often NULL)
                    if match_date:
                        cursor.execute("""
                            SELECT id, home_goals, away_goals FROM historical_matches
                            WHERE home_team = ? AND away_team = ? AND match_date = ?
                        """, (home_team, away_team, match_date))
                    else:
                        cursor.execute("""
                            SELECT id, home_goals, away_goals FROM historical_matches
                            WHERE home_team = ? AND away_team = ? AND match_date IS NULL
                            ORDER BY id DESC LIMIT 1
                        """, (home_team, away_team))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing record if it has NULL goals and we have goals
                        existing_id, existing_home_goals, existing_away_goals = existing
                        if (existing_home_goals is None or existing_away_goals is None) and (home_goals is not None or away_goals is not None):
                            cursor.execute("""
                                UPDATE historical_matches
                                SET home_goals = COALESCE(?, home_goals),
                                    away_goals = COALESCE(?, away_goals),
                                    result = ?,
                                    week_range = ?
                                WHERE id = ?
                            """, (home_goals, away_goals, result, week_range, existing_id))
                            updated_count += 1
                            logger.debug(f"Updated match {home_team} vs {away_team} with goals {home_goals}-{away_goals}")
                    else:
                        # Insert new record
                        try:
                            cursor.execute("""
                                INSERT INTO historical_matches 
                                (home_team, away_team, result, week_range, match_date, home_goals, away_goals)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, match)
                            inserted_count += 1
                        except sqlite3.IntegrityError:
                            # Match already exists (unique constraint), try to update it
                            logger.debug(f"Match {home_team} vs {away_team} already exists, attempting update")
                            cursor.execute("""
                                UPDATE historical_matches
                                SET home_goals = COALESCE(?, home_goals),
                                    away_goals = COALESCE(?, away_goals),
                                    result = ?,
                                    week_range = ?
                                WHERE home_team = ? AND away_team = ? AND (match_date = ? OR (match_date IS NULL AND ? IS NULL))
                            """, (home_goals, away_goals, result, week_range, home_team, away_team, match_date, match_date))
                            updated_count += 1
                
                logger.info(f"Processed {len(matches)} matches: {inserted_count} inserted, {updated_count} updated")
            else:
                # Legacy format without goals - use INSERT OR IGNORE
                cursor.executemany("""
                    INSERT OR IGNORE INTO historical_matches 
                    (home_team, away_team, result, week_range, match_date, home_goals, away_goals)
                    VALUES (?, ?, ?, ?, ?, NULL, NULL)
                """, matches)
            conn.commit()
            logger.info(f"Processed {len(matches)} historical matches (inserted/updated)")
        except sqlite3.Error as e:
            logger.error(f"Error inserting batch matches: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_historical_matches(self) -> pd.DataFrame:
        """Retrieve all historical matches as DataFrame."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT home_team as 'Ev Sahibi Takım',
                   away_team as 'Deplasman Takımı',
                   result as 'Mac Sonucu',
                   home_goals as 'Home_Goals',
                   away_goals as 'Away_Goals'
            FROM historical_matches
            ORDER BY match_date, id
        """, conn)
        conn.close()
        return df
    
    def match_exists(self, home_team: str, away_team: str, 
                    match_date: Optional[str] = None) -> bool:
        """Check if a match already exists in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if match_date:
            cursor.execute("""
                SELECT COUNT(*) FROM historical_matches
                WHERE home_team = ? AND away_team = ? AND match_date = ?
            """, (home_team, away_team, match_date))
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM historical_matches
                WHERE home_team = ? AND away_team = ?
            """, (home_team, away_team))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    
    def insert_upcoming_match(self, home_team: str, away_team: str,
                             match_date: Optional[str] = None) -> bool:
        """Insert or update upcoming match."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO upcoming_matches 
                (home_team, away_team, match_date, scraped_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (home_team, away_team, match_date))
            conn.commit()
            conn.close()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error inserting upcoming match: {e}")
            conn.close()
            return False
    
    def get_upcoming_matches(self) -> pd.DataFrame:
        """Get all upcoming matches."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT home_team as 'Ev Sahibi Takım',
                   away_team as 'Deplasman Takımı',
                   match_date
            FROM upcoming_matches
            ORDER BY match_date, id
        """, conn)
        conn.close()
        return df
    
    def clear_upcoming_matches(self):
        """Clear all upcoming matches (before new scrape)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM upcoming_matches")
        conn.commit()
        conn.close()
        logger.info("Upcoming matches cleared")
    
    def insert_prediction(self, home_team: str, away_team: str,
                          predicted_result: str, probability_1: float,
                          probability_x: float, probability_2: float,
                          batch_id: Optional[str] = None):
        """Insert a prediction with optional batch_id.
        
        When batch_id is given, uses upsert semantics keyed on
        (home_team, away_team, batch_id) so re-running the pipeline
        for the same week overwrites rather than duplicates the row.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if batch_id is not None:
                cursor.execute("""
                    INSERT INTO predictions
                    (home_team, away_team, predicted_result, probability_1,
                     probability_x, probability_2, batch_id, prediction_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(home_team, away_team, batch_id)
                    DO UPDATE SET
                        predicted_result = excluded.predicted_result,
                        probability_1    = excluded.probability_1,
                        probability_x    = excluded.probability_x,
                        probability_2    = excluded.probability_2,
                        prediction_date  = excluded.prediction_date
                """, (home_team, away_team, predicted_result, probability_1,
                      probability_x, probability_2, batch_id))
            else:
                # No batch_id: fall back to timestamp-keyed insert
                cursor.execute("""
                    INSERT OR REPLACE INTO predictions
                    (home_team, away_team, predicted_result, probability_1,
                     probability_x, probability_2, batch_id, prediction_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (home_team, away_team, predicted_result, probability_1,
                      probability_x, probability_2, batch_id))
            conn.commit()
            conn.close()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error inserting prediction: {e}")
            conn.close()
            return False
    
    def update_prediction_result(self, home_team: str, away_team: str,
                                actual_result: str):
        """Update prediction with actual result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE predictions
            SET actual_result = ?,
                is_correct = CASE 
                    WHEN predicted_result = ? THEN 1 
                    ELSE 0 
                END
            WHERE home_team = ? AND away_team = ?
            AND actual_result IS NULL
        """, (actual_result, actual_result, home_team, away_team))
        
        conn.commit()
        conn.close()
    
    def get_predictions_by_batch(self, batch_id: str) -> pd.DataFrame:
        """Get predictions for a specific batch."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT home_team, away_team, predicted_result,
                   probability_1, probability_x, probability_2,
                   actual_result, is_correct, prediction_date, batch_id
            FROM predictions
            WHERE batch_id = ?
            ORDER BY id
        """, conn, params=(batch_id,))
        conn.close()
        return df
    
    def get_predictions(self, limit: int = 100) -> pd.DataFrame:
        """Get recent predictions with accuracy."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT home_team, away_team, predicted_result,
                   probability_1, probability_x, probability_2,
                   actual_result, is_correct, prediction_date
            FROM predictions
            ORDER BY prediction_date DESC
            LIMIT ?
        """, conn, params=(limit,))
        conn.close()
        return df
    
    def get_prediction_accuracy(self) -> dict:
        """Calculate overall prediction accuracy."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(is_correct) as correct,
                AVG(is_correct) as accuracy
            FROM predictions
            WHERE actual_result IS NOT NULL
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] > 0:
            return {
                'total': result[0],
                'correct': result[1] or 0,
                'accuracy': (result[2] or 0) * 100
            }
        return {'total': 0, 'correct': 0, 'accuracy': 0.0}
    
    def save_model_performance(self, model_name: str, accuracy: float,
                              precision: float, recall: float, f1_score: float,
                              model_path: Optional[str] = None):
        """Save model performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance
            (model_name, accuracy, precision, recall, f1_score, model_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (model_name, accuracy, precision, recall, f1_score, model_path))
        
        conn.commit()
        conn.close()
    
    def save_prediction_combinations(self, batch_id: str, combinations: List[dict]):
        """
        Save all prediction combinations for a batch.
        
        Args:
            batch_id: Unique identifier for this prediction batch (e.g., "2026-W05")
            combinations: List of dicts with 'predictions' (list) and 'score' (float)
        """
        import json
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Clear existing combinations for this batch
            cursor.execute("DELETE FROM prediction_combinations WHERE batch_id = ?", (batch_id,))
            
            # Insert all combinations
            for rank, combo in enumerate(combinations, 1):
                predictions_json = json.dumps(combo['predictions'])
                cursor.execute("""
                    INSERT INTO prediction_combinations
                    (batch_id, combination_rank, predictions, confidence_score)
                    VALUES (?, ?, ?, ?)
                """, (batch_id, rank, predictions_json, combo['score']))
            
            conn.commit()
            logger.info(f"Saved {len(combinations)} combinations for batch {batch_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error saving combinations: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_combinations_for_batch(self, batch_id: str) -> List[dict]:
        """Get all combinations for a specific batch."""
        import json
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT combination_rank, predictions, confidence_score
            FROM prediction_combinations
            WHERE batch_id = ?
            ORDER BY combination_rank
        """, (batch_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'rank': row[0],
                'predictions': json.loads(row[1]),
                'score': row[2]
            })
        
        conn.close()
        return results
    
    def save_combination_result(self, batch_id: str, winning_rank: Optional[int], 
                                total_correct: int, total_matches: int):
        """
        Save the evaluation result for a batch.
        
        Args:
            batch_id: Batch identifier
            winning_rank: Which combination was correct (None if none matched perfectly)
            total_correct: Number of correct matches in winning combination
            total_matches: Total number of matches
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO combination_results
                (batch_id, winning_rank, total_correct, total_matches)
                VALUES (?, ?, ?, ?)
            """, (batch_id, winning_rank, total_correct, total_matches))
            
            conn.commit()
            logger.info(f"Saved result for batch {batch_id}: rank {winning_rank}, {total_correct}/{total_matches}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error saving combination result: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_combination_statistics(self) -> pd.DataFrame:
        """Get statistics on which combination ranks win most often."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT batch_id, winning_rank, total_correct, total_matches,
                   CAST(total_correct AS REAL) / total_matches * 100 as accuracy,
                   evaluated_at
            FROM combination_results
            ORDER BY evaluated_at DESC
        """, conn)
        conn.close()
        return df
    
    def get_latest_batch_id(self) -> Optional[str]:
        """Get the most recent batch_id."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT batch_id FROM prediction_combinations
            ORDER BY created_at DESC LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def save_monte_carlo_result(self, batch_id: str, attempts: int,
                                theoretical_prob: float, expected_attempts: int,
                                simulation_time: float, is_success: bool = False,
                                best_correct: Optional[int] = None,
                                best_attempt: Optional[int] = None,
                                total_matches: Optional[int] = None):
        """Save Monte Carlo simulation result (always saved, success or not)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO monte_carlo_results
                (batch_id, attempts_to_perfect, theoretical_probability,
                 expected_attempts, simulation_time,
                 is_success, best_correct, best_attempt, total_matches)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (batch_id, attempts, theoretical_prob, expected_attempts,
                  simulation_time, 1 if is_success else 0,
                  best_correct, best_attempt, total_matches))
            
            conn.commit()
            logger.info(f"Saved Monte Carlo result for {batch_id}: {attempts} attempts (success={is_success})")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error saving Monte Carlo result: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_monte_carlo_statistics(self) -> pd.DataFrame:
        """Get all Monte Carlo simulation results."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT batch_id, attempts_to_perfect, theoretical_probability,
                   expected_attempts, simulation_time,
                   is_success, best_correct, best_attempt, total_matches,
                   evaluated_at
            FROM monte_carlo_results
            ORDER BY evaluated_at DESC
        """, conn)
        conn.close()
        return df

