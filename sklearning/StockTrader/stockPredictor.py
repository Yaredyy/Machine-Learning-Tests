# bitcoinPredictor.py
# Predict: Buy, Short, or Hold BTC in the next 3 hours?

import joblib
import yfinance as yf
import ta

# Load model
folder = input("Enter Model Folder: ")
folder = f"sklearning/StockTrader/{folder}/"

model = joblib.load(folder + "bitcoin_model.pkl")
features = joblib.load(folder + "model_features.pkl")
print(f"✅ Loaded model from {folder}")

# Fetch live data
print("📥 Fetching live BTC data...")
data = yf.download("BTC-USD", period="5d", interval="1h")
if data.empty:
    raise Exception("Download failed.")

# Recompute features
close = data['Close'].squeeze()
data['rsi'] = ta.momentum.RSIIndicator(close).rsi()
data['macd'] = ta.trend.MACD(close).macd()
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()
data['price_change'] = data['Close'].pct_change(1)
data['volatility'] = data['Close'].rolling(10).std()
data['ema_12'] = data['Close'].ewm(span=12).mean()
data['ema_26'] = data['Close'].ewm(span=26).mean()

# Get latest complete row
latest = data[features].dropna().tail(1)
if latest.empty:
    raise Exception("Not enough data to predict.")

print(f"📊 Using data from: {latest.index[0]}")

# Predict probabilities
prob = model.predict_proba(latest)[0]
up_prob, down_prob = prob[1], prob[0]

print(f"\n🔮 Prediction Confidence:")
print(f"   Up (Buy):  {up_prob:.2%}")
print(f" Down (Short): {down_prob:.2%}")

# Decision engine
threshold = 0.58  # Only act on high confidence

if up_prob >= threshold:
    print("✅ STRONG BUY SIGNAL 🟢🔥")
elif down_prob >= threshold:
    print("✅ STRONG SHORT SIGNAL 🔴🔥")
elif up_prob > 0.52:
    print("🟨 WEAK BUY (Caution) 🟢")
elif down_prob > 0.52:
    print("🟨 WEAK SHORT (Caution) 🔴")
else:
    print("⚪ HOLD — No Clear Signal")

# Bonus info
print(f"\n📌 Model uses {len(features)} features.")
print("Top: Volatility, RSI, MACD → all market momentum signals.")