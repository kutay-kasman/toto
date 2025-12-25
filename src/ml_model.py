"""
Machine Learning model module with Optuna optimization and persistence.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna # <--- YENİ EKLENTİ
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from typing import Tuple, Optional, Dict, Any
import logging

# Optuna loglarını temiz tutalım
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

class MLModel:
    """Machine Learning model wrapper with Optuna optimization and persistence."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.feature_columns = None
        # En iyi parametreleri saklamak için bir değişken
        self.best_params = None 
        os.makedirs(model_dir, exist_ok=True)

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int = 20) -> Dict[str, Any]:
        """
        Optuna kullanarak en iyi hiperparametreleri bulur.
        
        Args:
            X: Training features
            y: Training labels
            n_trials: Deneme sayısı (Optuna kaç farklı kombinasyon denesin?)
            
        Returns:
            En iyi parametre sözlüğü (dictionary)
        """
        logger.info(f"Hyperparameter optimization started with {n_trials} trials...")

        # Class Imbalance için ağırlık hesabı (Optimization sırasında da önemli)
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        max_count = max(class_counts.values())
        # sample_weights dizisini oluşturuyoruz
        # Not: CV sırasında split edildiğinde bu ağırlıklar korunmalı,
        # XGBoost fit içine sample_weight vererek bunu sağlarız.
        weights = np.array([max_count / class_counts[cls] for cls in y])

        def objective(trial):
            # 1. PARAMETRE UZAYI (Search Space)
            # Futbol verisi genelde gürültülüdür, bu yüzden regülarizasyona önem veriyoruz.
            param = {
                'objective': 'multi:softmax',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'tree_method': 'hist', # Hızlandırma için (büyük veride gpu_hist yapılabilir)
                'verbosity': 0,
                
                # Kritik Parametreler
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                
                # Overfitting Engelleyiciler
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True), # L1 Reg
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True), # L2 Reg
            }

            # 2. MODEL KURULUMU
            model = xgb.XGBClassifier(**param)

            # 3. DEĞERLENDİRME (Stratified K-Fold CV)
            # Normal cross_val_score yerine, fit_params ile sample_weight geçebileceğimiz yapı
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                w_tr = weights[train_idx] # Ağırlıkları da bölüyoruz

                model.fit(X_tr, y_tr, sample_weight=w_tr)
                preds = model.predict(X_val)
                scores.append(accuracy_score(y_val, preds))

            # Ortalama başarıyı döndür
            return np.mean(scores)

        # Optuna Çalıştır
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Optimization finished. Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        # En iyi parametreleri sınıfın içine kaydet
        self.best_params = study.best_params
        return study.best_params

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: Optional[np.ndarray] = None,
              y_test: Optional[np.ndarray] = None,
              hyperparameters: Optional[Dict] = None,
              use_cross_validation: bool = True,
              optimize: bool = False, # <--- YENİ FLAG
              n_trials: int = 20) -> Dict:
        """
        Train XGBoost model.
        Args:
            optimize: True ise önce Optuna çalıştırır, sonra en iyi ayarlarla eğitir.
            n_trials: Optuna deneme sayısı.
        """
        logger.info("Training process started...")
        
        # Calculate class weights for imbalanced data (Global calculation)
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        max_count = max(class_counts.values())
        scale_pos_weight_map = {cls: max_count / count for cls, count in class_counts.items()}
        sample_weights = np.array([scale_pos_weight_map[label] for label in y_train])
        
        # 1. PARAMETRE BELİRLEME AŞAMASI
        final_params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42
        }

        if optimize:
            # Optuna ile en iyileri bul
            logger.info("Optimization mode: ON. Finding best hyperparameters...")
            best_optuna_params = self.optimize_hyperparameters(X_train, y_train, n_trials=n_trials)
            final_params.update(best_optuna_params)
        elif hyperparameters:
            # Kullanıcı manuel parametre verdiyse
            final_params.update(hyperparameters)
        elif self.best_params:
            # Daha önce optimize edildiyse hafızadakini kullan
            logger.info("Using previously optimized parameters.")
            final_params.update(self.best_params)
        else:
            # Hiçbir şey yoksa varsayılanlar
            default_params = {
                'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6,
                'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8
            }
            final_params.update(default_params)
        
        # 2. FİNAL EĞİTİM
        logger.info(f"Training Final Model with params: {final_params}")
        self.model = xgb.XGBClassifier(**final_params)
        
        # Cross-validation for final report
        cv_scores = None
        if use_cross_validation:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # Not: cross_val_score sample_weight'i doğrudan desteklemez, 
            # basit bir metrik için burada weightsiz bakıyoruz veya fit_params ile uğraşmak gerek.
            # Raporlama için standart Accuracy yeterli.
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            logger.info(f"Final Model CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Eğitimi başlat (Weights ile)
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Test seti değerlendirmesi
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
            logger.info(f"Test Set Accuracy: {accuracy:.4f}")
        
        if cv_scores is not None:
            metrics['cv_accuracy_mean'] = cv_scores.mean()
        
        return metrics

    # ... (predict, predict_matches, save, load, set_feature_columns metodları AYNI KALACAK)
    # Onları buraya tekrar yazarak kodu uzatmıyorum, eski kodundaki kısımları aynen koru.
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
       # ... Eski kodun aynısı ...
       if self.model is None:
           raise ValueError("Model not trained. Call train() first.")
       predictions = self.model.predict(X)
       probabilities = self.model.predict_proba(X)
       return predictions, probabilities

    def predict_matches(self, matches_df: pd.DataFrame, historical_df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        # ... Eski kodun aynısı ...
        # (Bu kısım modelden ziyade veri işleme logic'i içeriyor ama şimdilik burada kalsın)
        # Buradaki logic çok uzun olduğu için yukarıdaki orjinal kodundan kopyala-yapıştır yapabilirsin.
        # Sadece import ve DataProcessor çağırma kısımlarının çalıştığından emin ol.
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
        result_map = {0: '1', 1: '2', 2: 'X'}  # Standard encoding
        
        results = []
        for i, (_, match) in enumerate(matches_df.iterrows()):
            pred_class = int(predictions[i])
            proba = probabilities[i]
            
            results.append({
                'Ev Sahibi Takım': match['Ev Sahibi Takım'],
                'Deplasman Takımı': match['Deplasman Takımı'],
                'Predicted_Result': result_map.get(pred_class, '?'),
                'Probability_1': proba[0] * 100,
                'Probability_2': proba[1] * 100,
                'Probability_X': proba[2] * 100,
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
        # ... Eski kodun aynısı ...
        model_path = os.path.join(self.model_dir, filename)
        joblib.dump({'model': self.model, 'feature_columns': self.feature_columns, 'best_params': self.best_params}, model_path)
        logger.info(f"Model saved to {model_path}")
        return model_path

    def load(self, filename: str = "xgboost_model.pkl") -> bool:
        # ... Eski kodun aynısı ...
        model_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(model_path): return False
        try:
            data = joblib.load(model_path)
            self.model = data['model']
            self.feature_columns = data['feature_columns']
            self.best_params = data.get('best_params', None) # Eğer eski model dosyasında yoksa hata vermesin
            return True
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def set_feature_columns(self, columns: list):
        self.feature_columns = columns