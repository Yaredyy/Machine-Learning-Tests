# stockPredictorTorch.py

import torch
import torch.nn as nn
import yfinance as yf
import ta
import joblib
import json
import sys
import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

if len(sys.argv) >= 3:
    symbol = sys.argv[1].upper()
    folder_name = sys.argv[2]
else:
    symbol = input("Enter Ticker Symbol (e.g., BTC-USD, AAPL): ").upper()
    folder_name = input("Enter Model Folder: ")
    
folder = f"pyTorch/StockTrader/{symbol}/{folder_name}/"

features_path_json = folder + "checkpoint_features.json"
with open(features_path_json, "r") as f:
    features = json.load(f)

# Define model (match trainer architecture)
class StockTransformer(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=2, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 2)
        self.position_embeddings = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embedding(x) + self.position_embeddings
        x = self.transformer_encoder(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

# Instantiate and load model
model = StockTransformer(input_size=len(features)).to(device)
model.load_state_dict(torch.load(folder + "checkpoint_model.pt", map_location=device))
model.eval()

# Download latest data
print(f"Fetching live {symbol} data...")
data = yf.download(symbol, period="1y", interval="1d", auto_adjust=False)

if data.empty:
    raise Exception("Download failed.")

# Compute indicators
close = data['Close'].astype(float).squeeze()
volume = data['Volume'].astype(float).squeeze()

data['rsi'] = ta.momentum.RSIIndicator(close).rsi()
data['macd'] = ta.trend.MACD(close).macd()
data['sma_20'] = close.rolling(20).mean()
data['sma_50'] = close.rolling(50).mean()
data['price_change'] = close.pct_change(1)
data['volatility'] = close.rolling(10).std()
data['ema_12'] = close.ewm(span=12).mean()
data['ema_26'] = close.ewm(span=26).mean()
data['return_1'] = close.pct_change(1)
data['return_5'] = close.pct_change(5)
data['rolling_max'] = close.rolling(5).max()
data['rolling_min'] = close.rolling(5).min()
data['momentum'] = close - close.shift(5)
data['volume_change'] = volume.pct_change(1)
data['Volume'] = volume
data['price_position'] = (close - data['rolling_min']) / (data['rolling_max'] - data['rolling_min'])

# Drop NaNs
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Prepare input sequence (window_size=20 to match trainer)
window_size = 20
latest_seq = data[features].values[-window_size:]
if latest_seq.shape[0] < window_size:
    raise Exception("Not enough data to predict.")

# Scale
scaler_path = folder + "checkpoint_scaler.pkl"
scaler = joblib.load(scaler_path)
latest_scaled = scaler.transform(latest_seq)

# Tensor
input_tensor = torch.tensor(latest_scaled.astype('float32')).unsqueeze(0).to(device)  # shape (1, window, features)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]

up_prob, down_prob = probs[1], probs[0]

# Show prediction
print(f"\nPrediction Confidence:")
print(f"   Up (Buy):  {up_prob:.2%}")
print(f" Down (Short): {down_prob:.2%}")

threshold = 0.58
if up_prob >= threshold:
    print(" STRONG BUY SIGNAL ")
elif down_prob >= threshold:
    print(" STRONG SHORT SIGNAL ")
elif up_prob > 0.52:
    print(" WEAK BUY (Caution) ")
elif down_prob > 0.52:
    print(" WEAK SHORT (Caution) ")
else:
    print(" HOLD — No Clear Signal")

# Info
print(f"\n Model uses {len(features)} features and window_size={window_size}.",flush=True)
