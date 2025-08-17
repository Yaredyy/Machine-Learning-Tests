# stockTrainerTorch.py

import os
import numpy as np
import yfinance as yf
import ta
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# User input
import sys

if len(sys.argv) >= 3:
    symbol = sys.argv[1].upper()
    folder_name = sys.argv[2]
else:
    symbol = input("Enter Ticker Symbol (e.g., BTC-USD, AAPL): ").upper()
    folder_name = input("Enter Model Folder: ")

folder = f"pyTorch/StockTrader/{symbol}/{folder_name}/"
os.makedirs(folder, exist_ok=True)

print(f"Downloading {symbol} data...")
data = yf.download(symbol, period="1y", interval="1h", auto_adjust=False)
if data.empty:
    raise Exception("No data downloaded.")

# Feature engineering
print("Computing technical indicators...")
close = data['Close'].squeeze()
data['rsi'] = ta.momentum.RSIIndicator(close).rsi()
data['macd'] = ta.trend.MACD(close).macd()
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()
data['price_change'] = data['Close'].pct_change(1)
data['volatility'] = data['Close'].rolling(10).std()
data['ema_12'] = data['Close'].ewm(span=12).mean()
data['ema_26'] = data['Close'].ewm(span=26).mean()

# Target: will the price go up in 3 hours?
data['target'] = (data['Close'].shift(-3) > data['Close']).astype(int)
data.dropna(inplace=True)

# Prepare data
features = ['rsi', 'macd', 'sma_20', 'sma_50', 'price_change', 'volatility', 'Volume', 'ema_12', 'ema_26']
X = data[features].values.astype(np.float32)
Y = data['target'].values.astype(np.int64)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

X_train_tensor = torch.tensor(X_train)
Y_train_tensor = torch.tensor(Y_train)
X_test_tensor = torch.tensor(X_test)
Y_test_tensor = torch.tensor(Y_test)

print(f"Dataset: {len(X)} samples | Train: {len(X_train)} | Test: {len(X_test)}")

# Define model
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

# Training
model = StockClassifier(input_size=len(features))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
print("Training PyTorch model...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, Y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)
    predicted_classes = torch.argmax(preds, dim=1)
    acc = (predicted_classes == Y_test_tensor).float().mean().item()
print(f"\nTest Accuracy: {acc:.4f}")

# Save model and metadata
torch.save(model.state_dict(), folder + "torch_model.pt")
joblib.dump(features, folder + "model_features.pkl")
print("Model saved.")
