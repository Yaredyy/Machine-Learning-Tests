# bitcoinTrainer.py
# Train a model to predict: Will BTC go up in the next 3 hours?

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import yfinance as yf
import ta
import joblib
import os

# Get stock
symbol = input("Enter Ticker Symbol (e.g., BTC-USD, AAPL): ").upper()

print("📥 Downloading BTC-USD data...")
data = yf.download(symbol, period="1y", interval="1h", auto_adjust=False)
if data.empty:
    raise Exception("No data downloaded.")

# Get save folder
folder = input("Enter Model Folder: ")
folder = f"sklearning/StockTrader/{symbol}/{folder}/"
os.makedirs(folder, exist_ok=True)

# Add features
print("🔧 Computing technical indicators...")
close = data['Close'].squeeze()

data['rsi'] = ta.momentum.RSIIndicator(close).rsi()
data['macd'] = ta.trend.MACD(close).macd()
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()
data['price_change'] = data['Close'].pct_change(1)
data['volatility'] = data['Close'].rolling(10).std()
data['ema_12'] = data['Close'].ewm(span=12).mean()
data['ema_26'] = data['Close'].ewm(span=26).mean()

# Target: price higher in 3 hours?
data['target'] = (data['Close'].shift(-3) > data['Close']).astype(int)

# Prepare dataset
data.dropna(inplace=True)
features = ['rsi', 'macd', 'sma_20', 'sma_50', 'price_change', 'volatility', 'Volume', 'ema_12', 'ema_26']
X = data[features].values
Y = data['target'].values

# Time-based split: old → train, new → test
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

print(f"📊 Dataset: {len(X)} samples | Train: {len(X_train)} | Test: {len(X_test)}")

# Train model
print("🚀 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, Y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(Y_test, preds)
print(f"\n🎯 Test Accuracy: {acc:.4f}")
print("\n📝 Classification Report:")
print(classification_report(Y_test, preds, target_names=['Up', 'Down']))

# Save model
joblib.dump(model, folder + "bitcoin_model.pkl")
joblib.dump(features, folder + "model_features.pkl")
print("💾 Model saved.")

# Feature importance
importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📈 Top Features:")
print(importances.head(10))
importances.to_csv(folder + "feature_importance.csv", index=False)