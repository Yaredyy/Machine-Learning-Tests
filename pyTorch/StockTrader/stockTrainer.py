import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import joblib
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import signal
import random

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("CUDA version (PyTorch built with):", torch.version.cuda)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


stop_training = False

def signal_handler(sig, frame):
    global stop_training
    print(f"\nReceived signal {sig}. Will stop after current epoch and save model...")
    stop_training = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler) # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler) # Termination


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


features = ['rsi', 'macd', 'sma_20', 'sma_50', 'price_change', 'volatility', 'Volume', 'ema_12', 'ema_26']
X = data[features].values.astype(np.float32)
Y = data['target'].values.astype(np.int64)

scaler = StandardScaler()
X = scaler.fit_transform(X) 

# Sequence windowing
def create_sequences(X, Y, window_size):
    X_seq, Y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        Y_seq.append(Y[i+window_size])
    return np.array(X_seq), np.array(Y_seq)

window_size = 24  # 24 hours of lookback
X_seq, Y_seq = create_sequences(X, Y, window_size)

X_train_full, X_test, Y_train_full, Y_test = train_test_split(X_seq, Y_seq, test_size=0.2, shuffle=False)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.2, shuffle=False)

X_train_tensor = torch.tensor(X_train)
Y_train_tensor = torch.tensor(Y_train)
X_val_tensor = torch.tensor(X_val)
Y_val_tensor = torch.tensor(Y_val)
X_test_tensor = torch.tensor(X_test)
Y_test_tensor = torch.tensor(Y_test)

X_val_tensor = X_val_tensor.to(device)
Y_val_tensor = Y_val_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
Y_test_tensor = Y_test_tensor.to(device)


print(f"Dataset: {len(X_seq)} samples | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
unique, counts = np.unique(Y_seq, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))



# Define model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        output, _ = self.lstm(x)
        out = self.dropout(output[:, -1, :])
        return self.fc(out)


model = StockLSTM(input_size=len(features))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)


def safe_save(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Failed to save with {func.__name__}: {e}")

csv_path = folder + "ModelSummary.csv"
features_path = folder + "model_features.json"
scaler_path = folder + "scaler.pkl"



epochs = 80000
patience = 250
best_metric = float('-inf')
counter = 0
train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=256, shuffle=True,pin_memory=True,num_workers=0)

temp_path = folder + "checkpoint_model_temp.pt"
final_path = folder + "checkpoint_model.pt"

temp = []


print("Training PyTorch model...")

try:
    for epoch in range(epochs):
        if stop_training:
                print("Stopping training early due to signal.")
                break
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)


        train_acc = correct / total

    
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_preds = torch.argmax(val_output, dim=1)
            val_acc = (val_preds == Y_val_tensor).float().mean().item()

        scheduler.step(val_acc)



        if val_acc > best_metric:
            best_metric = val_acc
            torch.save(model.state_dict(), temp_path)
            os.replace(temp_path, final_path)
            print(f"Epoch {epoch+1}: New best val accuracy: {val_acc:.4f} — Checkpoint saved!")
            counter = 0
        else:
            counter += 1
            if counter > patience:
                print("Early stopping triggered.")
                break

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            temp.append({
                "Epoch": epoch + 1,
                "Avg Loss": avg_loss,
                "Train Accuracy": train_acc,
                "Val Accuracy": val_acc,
                "Stock": symbol,
                "Features": features,
                "Train Size": len(X_train),
                "Val Size": len(X_val),
                "Test Size": len(X_test),
                "Scaler": "StandardScaler",
                "Model": "StockLSTM",
                "Interval": "1h",
                "Period": "1y",
                "Window Size": window_size
            })

except Exception as e:
    print(f"Exception caught: {e}")
    print("Saving current model state before exiting...")
    df = pd.DataFrame(temp)
    safe_save(df.to_csv, csv_path)
    safe_save(lambda: json.dump(features, open(features_path, "w"), indent=4))
    safe_save(joblib.dump, scaler, scaler_path)
    torch.save(model.state_dict(), folder + "final_model.pt")
    print("Model saved!")
    sys.exit(1)

except KeyboardInterrupt:
    print("Ctrl+C detected! Saving model before exit...")
    df = pd.DataFrame(temp)
    safe_save(df.to_csv, csv_path)
    safe_save(lambda: json.dump(features, open(features_path, "w"), indent=4))
    safe_save(joblib.dump, scaler, scaler_path)
    torch.save(model.state_dict(), folder + "final_model.pt")
    print("Model saved!")
    sys.exit(0)



df = pd.DataFrame(temp)

model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_classes = torch.argmax(test_preds, dim=1)
    final_acc = (test_classes == Y_test_tensor).float().mean().item()

safe_save(df.to_csv, csv_path)
safe_save(lambda: json.dump(features, open(features_path, "w"), indent=4))
safe_save(joblib.dump, scaler, scaler_path)
torch.save(model.state_dict(), folder + "final_model.pt")

print(" Model saved successfully.")

model.load_state_dict(torch.load(folder + "checkpoint_model.pt"))
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_classes = torch.argmax(test_preds, dim=1)
    best_acc = (test_classes == Y_test_tensor).float().mean().item()


print(f"\n Final Val Accuracy: {final_acc:.4f} | Best Val Accuracy: {best_acc:.4f} | Best_metic: {best_metric:.4f}")


print("Training loop exited normally or by User based signal.")
