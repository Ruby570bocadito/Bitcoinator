import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_data
from src.data.cleaner import clean_data
from src.features.technical import add_technical_indicators
from src.features.temporal import add_temporal_features
from src.features.lags import add_lag_features, add_rolling_features


print("=" * 70)
print("BTC PRICE PREDICTION - INFINITE TRAINING")
print("Press Ctrl+C to stop when satisfied")
print("=" * 70)

start_time = time.time()

print("\n[1/5] Loading data...")
df = load_data('archive/btcusd_1-min_data.csv')

df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.set_index('Timestamp').resample('1D').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna().reset_index()

df = df[df['Timestamp'] >= '2020-01-01']
df = clean_data(df)
print(f"Data: {len(df)} rows")

print("\n[2/5] Creating features...")
df = add_technical_indicators(df)
df = add_temporal_features(df)
df = add_lag_features(df, lags=list(range(1, 61)))
df = add_rolling_features(df, windows=[3, 5, 7, 14, 21, 30, 60, 90])
df = df.dropna()
print(f"Features: {len(df.columns)}")

print("\n[3/5] Splitting...")
n = len(df)
train = df.iloc[:int(n*0.70)]
val = df.iloc[int(n*0.70):int(n*0.85)]
test = df.iloc[int(n*0.85):]

feature_cols = [c for c in df.columns if c not in ['Close', 'Timestamp']]
X_train = train[feature_cols].values
y_train = train['Close'].values
X_val = val[feature_cols].values
y_val = val['Close'].values
X_test = test[feature_cols].values
y_test = test['Close'].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\n[4/5] INFINITE TRAINING LOOP (Ctrl+C to stop)")
print("-" * 70)

all_results = []
best_rmse = float('inf')
best_model = None
iteration = 0
np.random.seed(42)

while True:
    iteration += 1
    
    params = {
        'n_estimators': np.random.choice([500, 1000, 1500, 2000]),
        'max_depth': np.random.choice([4, 5, 6, 7, 8]),
        'learning_rate': np.random.choice([0.01, 0.02, 0.05, 0.1]),
        'subsample': np.random.choice([0.6, 0.7, 0.8, 0.9]),
        'min_child_weight': np.random.choice([1, 3, 5, 10]),
        'reg_lambda': np.random.choice([1, 2, 3, 5, 10]),
    }
    
    try:
        import xgboost as xgb
        
        model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=0.7,
            min_child_weight=params['min_child_weight'],
            reg_alpha=0.1,
            reg_lambda=params['reg_lambda'],
            random_state=42,
            tree_method='hist',
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        preds = model.predict(X_test_scaled)
        
        rmse = np.sqrt(np.mean((preds - y_test) ** 2))
        mae = np.mean(np.abs(preds - y_test))
        
        true_dir = np.diff(y_test) > 0
        pred_dir = np.diff(preds) > 0
        dir_acc = np.mean(true_dir == pred_dir) * 100
        
        result = {**params, 'rmse': rmse, 'mae': mae, 'dir_acc': dir_acc}
        all_results.append(result)
        
        is_best = False
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            is_best = True
        
        elapsed = time.time() - start_time
        
        print(f"[{iteration}] RMSE: ${rmse:,.0f} | MAE: ${mae:,.0f} | Dir: {dir_acc:.1f}% | Best: ${best_rmse:,.0f} | Time: {elapsed/60:.1f}min" + (" ***" if is_best else ""))
        
        if iteration % 50 == 0 and best_model is not None:
            best_model.save_model('models/xgboost_infinite.json')
            import joblib
            joblib.dump(scaler, 'models/scaler_infinite.pkl')
            with open('models/infinite_results.json', 'w') as f:
                json.dump(all_results[-100:], f, indent=2, default=str)
            print(f"  >>> Auto-saved at iteration {iteration}")
        
        if iteration >= 500:
            break
            
    except KeyboardInterrupt:
        print("\n\nStopping training...")
        break
    except Exception as e:
        print(f"Error: {str(e)[:50]}")
        continue

print("\n[5/5] Final saving...")

if best_model is not None:
    best_model.save_model('models/xgboost_infinite.json')
    import joblib
    joblib.dump(scaler, 'models/scaler_infinite.pkl')

results_sorted = sorted(all_results, key=lambda x: x['rmse'])

print("\n" + "=" * 70)
print("TOP 10 MODELS")
print("=" * 70)
for i, res in enumerate(results_sorted[:10]):
    print(f"#{i+1}: RMSE=${res['rmse']:,.0f} | Dir={res['dir_acc']:.1f}%")

elapsed = time.time() - start_time
print(f"\nTotal iterations: {iteration}")
print(f"Total time: {elapsed/60:.1f} minutes")
print(f"Best RMSE: ${best_rmse:,.0f}")
print("Done!")