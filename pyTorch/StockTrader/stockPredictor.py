# stockPredictorTorch.py
    
import torch
import torch.nn as nn
import yfinance as yf
import ta
import joblib
import pandas as pd

# User input
symbol = input("Enter Ticker Symbol (e.g., BTC-USD, AAPL): ").upper()
folder_name = input("Enter Model Folder: ")
folder = f"pyTorch/StockTrader/{symbol}/{folder_name}/"

# Load features and model
features = joblib.load(folder + "model_features.pkl")

class StockClassifier(nn.Module):
    def __init__(self, input_size):
        super(StockClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# Instantiate and load model
model = StockClassifier(input_size=len(features))
model.load_state_dict(torch.load(folder + "torch_model.pt"))
model.eval()

# Download latest data
print(f"📥 Fetching live {symbol} data...")
data = yf.download(symbol, period="5d", interval="1h", auto_adjust=False)
if data.empty:
    raise Exception("Download failed.")

# Compute indicators
close = data['Close'].squeeze()
data['rsi'] = ta.momentum.RSIIndicator(close).rsi()
data['macd'] = ta.trend.MACD(close).macd()
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()
data['price_change'] = data['Close'].pct_change(1)
data['volatility'] = data['Close'].rolling(10).std()
data['ema_12'] = data['Close'].ewm(span=12).mean()
data['ema_26'] = data['Close'].ewm(span=26).mean()

# Use latest row
latest = data[features].dropna().tail(1)
if latest.empty:
    raise Exception("Not enough data to predict.")
print(f"📊 Using data from: {latest.index[0]}")

# Predict
input_tensor = torch.tensor(latest.values.astype('float32'))
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).numpy()[0]

up_prob, down_prob = probs[1], probs[0]

# Show prediction
print(f"\n🔮 Prediction Confidence:")
print(f"   Up (Buy):  {up_prob:.2%}")
print(f" Down (Short): {down_prob:.2%}")

threshold = 0.58

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

# Info
print(f"\n📌 Model uses {len(features)} features.")
