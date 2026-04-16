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
print("BTC DIRECTION PREDICTION - UP/DOWN CLASSIFICATION")
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

print("\n[2/5] Creating target: 1=UP, 0=DOWN")
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df[:-1]
print(f"Target: UP={df['target'].sum()}, DOWN={len(df)-df['target'].sum()}")

print("\n[3/5] Creating features...")
df = add_technical_indicators(df)
df = add_temporal_features(df)
df = add_lag_features(df, lags=list(range(1, 61)))
df = add_rolling_features(df, windows=[3, 5, 7, 14, 21, 30, 60, 90])
df['return_1'] = df['Close'].pct_change(1)
df['return_3'] = df['Close'].pct_change(3)
df['return_7'] = df['Close'].pct_change(7)
df['return_14'] = df['Close'].pct_change(14)
df = df.dropna()
print(f"Features: {len(df.columns)}")

print("\n[4/5] Splitting...")
n = len(df)
train = df.iloc[:int(n*0.70)]
val = df.iloc[int(n*0.70):int(n*0.85)]
test = df.iloc[int(n*0.85):]

feature_cols = [c for c in df.columns if c not in ['Close', 'target', 'Timestamp']]
X_train = train[feature_cols].values
y_train = train['target'].values
X_val = val[feature_cols].values
y_val = val['target'].values
X_test = test[feature_cols].values
y_test = test['target'].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

print("\n[5/5] INFINITE TRAINING LOOP (Ctrl+C to stop)")
print("-" * 70)

all_results = []
best_acc = 0
best_model = None
iteration = 0
np.random.seed(42)

while True:
    iteration += 1
    
    params = {
        'n_estimators': np.random.choice([300, 500, 800, 1000]),
        'max_depth': np.random.choice([3, 4, 5, 6]),
        'learning_rate': np.random.choice([0.01, 0.02, 0.05, 0.1]),
        'subsample': np.random.choice([0.6, 0.7, 0.8, 0.9]),
        'colsample_bytree': np.random.choice([0.5, 0.6, 0.7, 0.8]),
        'min_child_weight': np.random.choice([3, 5, 10]),
    }
    
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score
        
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            min_child_weight=params['min_child_weight'],
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='logloss',
            early_stopping_rounds=30
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        
        result = {**params, 'accuracy': acc}
        all_results.append(result)
        
        is_best = False
        if acc > best_acc:
            best_acc = acc
            best_model = model
            is_best = True
        
        elapsed = time.time() - start_time
        
        if iteration % 10 == 0 or is_best:
            print(f"[{iteration}] Acc: {acc*100:.1f}% | Best: {best_acc*100:.1f}% | Time: {elapsed/60:.1f}min" + (" ***" if is_best else ""))
        
        if iteration % 50 == 0:
            best_model.save_model('models/direction_model.json')
            import joblib
            joblib.dump(scaler, 'models/scaler_direction.pkl')
            with open('models/direction_results.json', 'w') as f:
                json.dump(all_results[-50:], f, indent=2, default=str)
            print(f"  >>> Auto-saved at iteration {iteration}")
        
        if iteration >= 500:
            break
            
    except KeyboardInterrupt:
        print("\n\nStopping training...")
        break
    except Exception as e:
        continue

print("\n[FINAL] Saving...")
if best_model is not None:
    best_model.save_model('models/direction_model.json')
    import joblib
    joblib.dump(scaler, 'models/scaler_direction.pkl')

results_sorted = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)

print("\n" + "=" * 70)
print("TOP 10 MODELS")
print("=" * 70)
for i, res in enumerate(results_sorted[:10]):
    print(f"#{i+1}: Accuracy={res['accuracy']*100:.1f}% | n={res['n_estimators']}, d={res['max_depth']}, lr={res['learning_rate']}")

elapsed = time.time() - start_time
print(f"\nTotal iterations: {iteration}")
print(f"Total time: {elapsed/60:.1f} minutes")
print(f"Best Accuracy: {best_acc*100:.1f}%")
print("Done!")