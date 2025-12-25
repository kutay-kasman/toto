import time
import pandas as pd
import xgboost as xgb
import optuna
import warnings
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Uyarıları kapatalım (Temiz ekran)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

print("⏳ Veri hazırlanıyor ve 'Arena' kuruluyor...")
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Başlıyoruz!\n")

# --- YARIŞMACI 1: GRID SEARCH ---
print("Search (Kaba Kuvvet) Çalışıyor...")
start_grid = time.time()

param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200]
}

xgb_grid = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb_grid, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

end_grid = time.time()
best_params_grid = grid_search.best_params_ # <--- İŞTE REÇETE BURADA
acc_grid = grid_search.best_score_


# --- YARIŞMACI 2: OPTUNA ---
print("Optuna (Akıllı Avcı) Çalışıyor...")
start_optuna = time.time()

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

end_optuna = time.time()
best_params_optuna = study.best_trial.params # <--- İŞTE REÇETE BURADA
acc_optuna = study.best_value


# --- BÜYÜK FİNAL VE REÇETELER ---
print("\n" + "="*60)
print(f"{'SONUÇLAR':^60}")
print("="*60)

# 1. Grid Search Sonuçları
print(f"\n🏗️  GRID SEARCH SONUCU:")
print(f"    ⏱️  Süre: {end_grid - start_grid:.2f} saniye")
print(f"    🎯 Skor: {acc_grid:.4f}")
print(f"    📜 KAZANAN AYARLAR (REÇETE):")
for key, value in best_params_grid.items():
    print(f"       • {key}: {value}")

# 2. Optuna Sonuçları
print(f"\n🚀 OPTUNA SONUCU:")
print(f"    ⏱️  Süre: {end_optuna - start_optuna:.2f} saniye")
print(f"    🎯 Skor: {acc_optuna:.4f}")
print(f"    📜 KAZANAN AYARLAR (REÇETE):")
for key, value in best_params_optuna.items():
    # Optuna bazen sayıları çok uzun float verir, onları yuvarlayalım
    if isinstance(value, float):
        print(f"       • {key}: {value:.4f}")
    else:
        print(f"       • {key}: {value}")

print("\n" + "="*60)