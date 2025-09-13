# stockTrainerTorch.py

import os
import numpy as np
import yfinance as yf
import ta
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Normalize all features

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

X_train_tensor = torch.tensor(X_train)
Y_train_tensor = torch.tensor(Y_train)
X_test_tensor = torch.tensor(X_test)
Y_test_tensor = torch.tensor(Y_test)
X_train_tensor = X_train_tensor.unsqueeze(1)  
X_test_tensor = X_test_tensor.unsqueeze(1)  

print(f"Dataset: {len(X)} samples | Train: {len(X_train)} | Test: {len(X_test)}")

# Define model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        output, _ = self.lstm(x)
        out = self.dropout(output[:, -1, :])  # Use the last output in the sequence
        out = self.fc(out)
        return out

# Training
model = StockLSTM(input_size=len(features))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10000
print("Training PyTorch model...")
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

temp = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct=0
    total=0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(output, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(train_loader):.4f}")
        temp.append({
        "Epoch": epoch+1,
        "Avg Loss": (total_loss/len(train_loader)),
        "Accuracy": correct/total,
        "Stock": symbol,
        "Features": features,
        "Train Size": len(X_train),
        "Test Size": len(X_test),
        "Scaler": "StandardScaler",
        "Model": "StockLSTM",
        "Interval": "1h",
        "Period": "1y"})

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
joblib.dump(scaler, folder + "scaler.pkl")

df = pd.DataFrame(temp)
df.to_csv(folder+"ModelSummary.csv", index=False)

print("Model saved.")
