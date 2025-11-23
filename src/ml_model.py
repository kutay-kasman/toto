"""
Machine Learning model module with optimization and persistence.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MLModel:
    """Machine Learning model wrapper with optimization and persistence."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize ML model.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        self.model = None
        self.feature_columns = None
        self.label_encoder = None
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: Optional[np.ndarray] = None,
              y_test: Optional[np.ndarray] = None,
              hyperparameters: Optional[Dict] = None,
              use_cross_validation: bool = True) -> Dict:
        """
        Train XGBoost model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Optional test set
            y_test: Optional test labels
            hyperparameters: Optional hyperparameter dict
            use_cross_validation: Whether to use cross-validation
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training XGBoost model")
        
        # Calculate class weights for imbalanced data
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        max_count = max(class_counts.values())
        scale_pos_weight_map = {
            cls: max_count / count 
            for cls, count in class_counts.items()
        }
        sample_weights = np.array([scale_pos_weight_map[label] for label in y_train])
        
        # Default hyperparameters
        default_params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        if hyperparameters:
            default_params.update(hyperparameters)
        
        # Create model
        self.model = xgb.XGBClassifier(**default_params)
        
        # Cross-validation if requested
        cv_scores = None
        if use_cross_validation:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=cv, scoring='accuracy', n_jobs=-1
            )
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train model
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluate on test set if provided
        metrics = {}
        if X_test is not None and y_test is not None:
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            metrics['test_accuracy'] = accuracy
            metrics['classification_report'] = classification_report(
                y_test, y_pred, 
                target_names=['1 (Ev Sahibi)', '2 (Deplasman)', 'X (Berabere)'],
                output_dict=True
            )
            logger.info(f"Test accuracy: {accuracy:.4f}")
        
        if cv_scores is not None:
            metrics['cv_accuracy_mean'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_matches(self, matches_df: pd.DataFrame, 
                       historical_df: pd.DataFrame,
                       feature_columns: list) -> pd.DataFrame:
        """
        Predict outcomes for upcoming matches.
        
        Args:
            matches_df: DataFrame with upcoming matches
            historical_df: Historical match data for feature calculation
            feature_columns: List of feature column names from training
            
        Returns:
            DataFrame with predictions
        """
        from src.data_processing import DataProcessor
        
        processor = DataProcessor()
        
        # Calculate features for each match
        prediction_features = []
        for _, row in matches_df.iterrows():
            home_team = row['Ev Sahibi Takım']
            away_team = row['Deplasman Takımı']
            
            # Get team statistics
            h_win, h_draw, h_loss = processor.calculate_performance_ratios(
                historical_df, home_team, len(historical_df)
            )
            a_win, a_draw, a_loss = processor.calculate_performance_ratios(
                historical_df, away_team, len(historical_df)
            )
            
            # Calculate goal statistics
            h_goals_scored, h_goals_conceded = processor.calculate_goals_statistics(
                historical_df, home_team, len(historical_df)
            )
            a_goals_scored, a_goals_conceded = processor.calculate_goals_statistics(
                historical_df, away_team, len(historical_df)
            )
            
            prediction_features.append({
                'Home_Win_Ratio': h_win,
                'Home_Draw_Ratio': h_draw,
                'Home_Loss_Ratio': h_loss,
                'Away_Win_Ratio': a_win,
                'Away_Draw_Ratio': a_draw,
                'Away_Loss_Ratio': a_loss,
                'Home_Avg_Goals_Scored': h_goals_scored,
                'Home_Avg_Goals_Conceded': h_goals_conceded,
                'Away_Avg_Goals_Scored': a_goals_scored,
                'Away_Avg_Goals_Conceded': a_goals_conceded
            })
        
        # Create feature matrix
        X_predict_ohe = pd.DataFrame(0, index=matches_df.index, 
                                    columns=[col for col in feature_columns 
                                            if col.startswith(('Ev_', 'Dep_'))])
        X_predict_stat = pd.DataFrame(prediction_features)
        
        # Set one-hot encoding
        for i, row in matches_df.iterrows():
            home_team = row['Ev Sahibi Takım']
            away_team = row['Deplasman Takımı']
            
            col_home = f'Ev_{home_team}'
            col_away = f'Dep_{away_team}'
            
            if col_home in X_predict_ohe.columns:
                X_predict_ohe.loc[i, col_home] = 1
            if col_away in X_predict_ohe.columns:
                X_predict_ohe.loc[i, col_away] = 1
        
        # Combine features
        X_predict_final = pd.concat([X_predict_ohe, X_predict_stat], axis=1)
        
        # Ensure column order matches training
        X_predict = X_predict_final.reindex(columns=feature_columns, fill_value=0)
        
        # Make predictions
        predictions, probabilities = self.predict(X_predict.values)
        
        # Create results DataFrame
        # XGBoost predict_proba returns probabilities in class order
        # LabelEncoder typically encodes in sorted order: '1'=0, '2'=1, 'X'=2
        # But we'll map based on the actual prediction class
        result_map = {0: '1', 1: '2', 2: 'X'}  # Standard encoding
        
        results = []
        for i, (_, match) in enumerate(matches_df.iterrows()):
            pred_class = int(predictions[i])
            proba = probabilities[i]
            
            # XGBoost returns probabilities in class order: [class_0, class_1, class_2]
            # With LabelEncoder: '1'=0, '2'=1, 'X'=2 (alphabetical order)
            results.append({
                'Ev Sahibi Takım': match['Ev Sahibi Takım'],
                'Deplasman Takımı': match['Deplasman Takımı'],
                'Predicted_Result': result_map.get(pred_class, '?'),
                'Probability_1': proba[0] * 100,  # Class 0 = '1'
                'Probability_2': proba[1] * 100,  # Class 1 = '2'
                'Probability_X': proba[2] * 100,  # Class 2 = 'X'
                'Home_Win_Ratio': prediction_features[i]['Home_Win_Ratio'],
                'Home_Draw_Ratio': prediction_features[i]['Home_Draw_Ratio'],
                'Home_Loss_Ratio': prediction_features[i]['Home_Loss_Ratio'],
                'Away_Win_Ratio': prediction_features[i]['Away_Win_Ratio'],
                'Away_Draw_Ratio': prediction_features[i]['Away_Draw_Ratio'],
                'Away_Loss_Ratio': prediction_features[i]['Away_Loss_Ratio'],
                'Home_Avg_Goals_Scored': prediction_features[i]['Home_Avg_Goals_Scored'],
                'Home_Avg_Goals_Conceded': prediction_features[i]['Home_Avg_Goals_Conceded'],
                'Away_Avg_Goals_Scored': prediction_features[i]['Away_Avg_Goals_Scored'],
                'Away_Avg_Goals_Conceded': prediction_features[i]['Away_Avg_Goals_Conceded']
            })
        
        return pd.DataFrame(results)
    
    def save(self, filename: str = "xgboost_model.pkl"):
        """Save model to disk."""
        model_path = os.path.join(self.model_dir, filename)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, model_path)
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load(self, filename: str = "xgboost_model.pkl") -> bool:
        """
        Load model from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        model_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        try:
            data = joblib.load(model_path)
            self.model = data['model']
            self.feature_columns = data['feature_columns']
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def set_feature_columns(self, columns: list):
        """Set feature column names."""
        self.feature_columns = columns

