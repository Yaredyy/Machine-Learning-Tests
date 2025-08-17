# bitcoinPredictor.py
# Predict: Should I buy or short BTC in the next 3 hours?

import joblib
import pandas as pd
import yfinance as yf
import ta
import numpy as np

# --- 1. Load Model & Features ---
folder = input("Enter Model Folder: ")
folder = "sklearning/BitcoinTrader/" + folder + "/"

model = joblib.load(folder + "bitcoin_model.pkl")
features = joblib.load(folder + "model_features.pkl")

print(f"✅ Loaded model from {folder}")

# --- 2. Get Latest Data (last 5 days to compute indicators) ---
print("📥 Fetching live BTC data...")
data = yf.download("BTC-USD", period="5d", interval="1h")
if data.empty:
    raise Exception("Failed to download data.")

# --- 3. Add Same Features as Training ---
close = data['Close'].squeeze()

data['rsi'] = ta.momentum.RSIIndicator(close).rsi()
data['macd'] = ta.trend.MACD(close).macd()
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()
data['price_change'] = data['Close'].pct_change(1)
data['volatility'] = data['Close'].rolling(10).std()
data['ema_12'] = data['Close'].ewm(span=12).mean()
data['ema_26'] = data['Close'].ewm(span=26).mean()

# --- 4. Get the Most Recent Row (Latest Hour) ---
latest = data[features].dropna().tail(1)  # Only keep fully computed row

if latest.empty:
    raise Exception("Not enough data to make prediction.")

print(f"📊 Using latest data from: {latest.index[0]}")

# --- 5. Predict Probability (KEY: Use confidence!) ---
prob = model.predict_proba(latest)[0]  # [P(down), P(up)]
pred = model.predict(latest)[0]

up_prob = prob[1]
down_prob = prob[0]

print(f"\n🔮 Prediction Confidence:")
print(f"   Up (Buy) Probability: {up_prob:.2%}")
print(f" Down (Short) Probability: {down_prob:.2%}")

# --- 6. Make Decision ---
threshold = 0.55  # Only act if >55% confidence

if up_prob > threshold:
    print("✅ RECOMMENDATION: BUY (Long) 🟢")
elif down_prob > threshold:
    print("✅ RECOMMENDATION: SHORT (Sell) 🔴")
else:
    print("🟰 RECOMMENDATION: HOLD (Low Confidence) ⚪")

# Bonus: Show feature importance impact (simple version)
print(f"\n📌 Model based on {len(features)} features like RSI, MACD, etc.")