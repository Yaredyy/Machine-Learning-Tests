# bitcoinTrainer.py
# Predict: Will BTC price go up in the next 3 hours?

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import yfinance as yf
import ta
import joblib

# --- 1. Download Data ---
print("📥 Downloading BTC-USD data...")
data = yf.download("BTC-USD", period="1y", interval="1h")  # 1 year now
if data.empty:
    raise Exception("No data downloaded. Check internet or ticker.")

fold = input("Enter Model Folder")
fold = "sklearning/BitcoinTrader/"+fold+"/"

# --- 2. Add Features ---
print("🔧 Adding technical indicators...")
close = data['Close'].squeeze()  # Fix: Ensure 1D array

data['rsi'] = ta.momentum.RSIIndicator(close).rsi()
data['macd'] = ta.trend.MACD(close).macd()
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()
data['price_change'] = data['Close'].pct_change(1)
data['volatility'] = data['Close'].rolling(10).std()
data['ema_12'] = data['Close'].ewm(span=12).mean()
data['ema_26'] = data['Close'].ewm(span=26).mean()

# --- 3. Target: +3h price move ---
data['target'] = (data['Close'].shift(-3) > data['Close']).astype(int)

# --- 4. Drop NaN & Select Features ---
data.dropna(inplace=True)
features = ['rsi', 'macd', 'sma_20', 'sma_50', 'price_change', 'volatility', 'Volume', 'ema_12', 'ema_26']
X = data[features].values
Y = data['target'].values

# --- 5. Fixed Time-Based Split (80/20) ---
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

print(f"📊 Dataset: {len(X)} samples | Train: {len(X_train)} | Test: {len(X_test)}")

# --- 6. Train Final Model ---
print("🚀 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, Y_train)

# --- 7. Evaluate Properly ---
preds = model.predict(X_test)
acc = accuracy_score(Y_test, preds)
print(f"\n🎯 Test Accuracy: {acc:.4f}")

print("\n📝 Classification Report:")
print(classification_report(Y_test, preds, target_names=['Up', 'Down']))

# --- 8. Save Model & Features ---
joblib.dump(model, fold+"bitcoin_model.pkl")
joblib.dump(features, fold+"model_features.pkl")
print("💾 Model saved: 'bitcoin_model.pkl' and 'model_features.pkl'")

# --- 9. Feature Importance ---
importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📈 Top Features:")
print(importances.head(10))

# Optional: Save to CSV
importances.to_csv(fold+"feature_importance.csv", index=False)
