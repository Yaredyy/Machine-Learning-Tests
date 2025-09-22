# Libraries  *make sure to have them installed*
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
from sklearn.utils.class_weight import compute_class_weight
import signal
import random
from datetime import datetime

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device information
print("────────────────────────────────────────────────────────────")
print("CUDA Available:", torch.cuda.is_available())
print("CUDA version (PyTorch built with):", torch.version.cuda)

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using Apple Silicon GPU.")
    device = torch.device("mps")
else:
    print("Using CPU.")
    device = torch.device("cpu")
print("────────────────────────────────────────────────────────────\n")

# Signal handling for graceful exit
global stop_training
stop_training = False
def signal_handler(sig, frame):
    global stop_training
    print(f"\nSignal {sig} received. Stopping after current epoch...")
    stop_training = True
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Input
if len(sys.argv) >= 3:
    symbol = sys.argv[1].upper()
    folder_name = sys.argv[2]
else:
    symbol = input("Enter Ticker Symbol (e.g., BTC-USD, AAPL): ").upper()
    folder_name = input("Enter Model Folder: ")
folder = f"pyTorch/StockTrader/{symbol}/{folder_name}/"
os.makedirs(folder, exist_ok=True)

# Features
features = [
    'rsi', 'macd', 'sma_20', 'sma_50',
    'price_change', 'volatility', 'Volume',
    'ema_12', 'ema_26',
    'return_1', 'return_5', 'rolling_max', 'rolling_min',
    'momentum', 'volume_change', 'price_position'
]

# Sequence windowing
def create_sequences(X, Y, window_size=1):
    X_seq, Y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        Y_seq.append(Y[i+window_size])
    return np.array(X_seq), np.array(Y_seq)

# Transformer Model
class StockTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=3, dropout=0.1):
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

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

# Safe save
def safe_save(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Failed to save with {func.__name__}: {e}")

# Date controls
year = 2025
wind = 1
c = 0
def new_start(): return datetime(year - (wind * (c + 1)), 1, 1)
def new_end(): return datetime(year - (wind * (c)), 1, 1)

# Preprocess data
def load_and_format(symbol):
    global class_weights, scaler, X_val_tensor, Y_val_tensor, X_test_tensor, Y_test_tensor, train_loader
    global df, newdf
    print(f"Downloading {symbol} data: {new_start().date()} → {new_end().date()}")
    try:
        newdf = yf.download(symbol, start=new_start(), end=new_end(), auto_adjust=False)
    except:
        print("Error downloading, continuing.")

    if newdf.empty:
        if df.empty:
            raise ValueError("No data downloaded")
        else:
            print("No data downloaded, using same dataset")
            return X_val_tensor, Y_val_tensor, X_test_tensor, Y_test_tensor
    else:
        df=newdf


    close = df['Close'].squeeze()
    df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    df['macd'] = ta.trend.MACD(close).macd()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['price_change'] = df['Close'].pct_change(1)
    df['volatility'] = df['Close'].rolling(10).std()
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    df['return_1'] = df['Close'].pct_change(1)
    df['return_5'] = df['Close'].pct_change(5)
    df['rolling_max'] = df['Close'].rolling(window=5).max()
    df['rolling_min'] = df['Close'].rolling(window=5).min()
    df['momentum'] = df['Close'] - df['Close'].shift(5)
    df['volume_change'] = df['Volume'].pct_change(1)
    df['price_position'] = (df['Close'] - df['rolling_min']) / (df['rolling_max'] - df['rolling_min'])

    df['target'] = (df['Close'].shift(-24) > df['Close']).astype(int)
    df.dropna(inplace=True)

    X = df[features].values.astype(np.float32)
    Y = df['target'].values.astype(np.int64)
    X, Y = create_sequences(X, Y, window_size=1)

    X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.3, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X.shape[-1])).reshape(X_test.shape)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long).to(device)

    if torch.cuda.is_available():
        train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=256, shuffle=True, pin_memory=True)
    else:
        train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=256, shuffle=True)

    unique, counts = np.unique(Y_train, return_counts=True)
    if len(unique) < 2:
        class_weights = None
    else:
        weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=Y_train)
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    return X_val_tensor, Y_val_tensor, X_test_tensor, Y_test_tensor

# Load data
try:
    X_val_tensor, Y_val_tensor, X_test_tensor, Y_test_tensor = load_and_format(symbol)
except ValueError:
    print("Couldn't download stocks, removing save folder.")
    os.rmdir(folder)
    sys.exit(1)

model = StockTransformer(input_size=len(features)).to(device)
criterion = FocalLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=0.01, patience=50)

# Paths
csv_path = folder + "ModelSummary.csv"
features_path = folder + "features.json"
scaler_path = folder + "scaler.pkl"
model_path = folder + "final_model.pt"
temp_path = folder + "temp_checkpoint_model.pt"
final_path = folder + "checkpoint_model.pt"
checkpoint_features_path = folder + "checkpoint_features.json"
checkpoint_scaler_path = folder + "checkpoint_scaler.pkl"

# Training loop
epochs = 100000
patien = 50010
best_metric = 0
counter = 0
temp = []

print("\nTraining begins...\n")

try:
    for epoch in range(epochs):
        if stop_training:
            print("Stopping due to signal.")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            df = pd.DataFrame(temp)
            safe_save(df.to_csv, csv_path)
            safe_save(lambda: json.dump(features, open(checkpoint_features_path, "w"), indent=4))
            safe_save(joblib.dump, scaler, checkpoint_scaler_path)
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} — New best val acc: {val_acc:.4f} — Model saved at Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            counter = 0
        else:
            counter += 1
            if counter > patien:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            temp.append({
                "Epoch": epoch + 1,
                "Avg Loss": avg_loss,
                "Train Accuracy": train_acc,
                "Val Accuracy": val_acc,
                "Stock": symbol,
                "Features": features,
                "Train Size": len(train_loader.dataset),
                "Scaler": "StandardScaler",
                "Model": "StockTransformer",
                "Interval": "24h",
                "Period": "1y"
            })

        if (epoch + 1) % 100 == 0:
            c += 1
            X_val_tensor, Y_val_tensor, X_test_tensor, Y_test_tensor = load_and_format(symbol)

except Exception as e:
    print(f"Exception caught: {e}")
    print("Saving current model state before exiting...")
    df = pd.DataFrame(temp)
    safe_save(df.to_csv, csv_path)
    safe_save(lambda: json.dump(features, open(features_path, "w"), indent=4))
    safe_save(joblib.dump, scaler, scaler_path)
    torch.save(model.state_dict(), model_path)
    print("Model saved!")
    sys.exit(1)

except KeyboardInterrupt:
    print("Ctrl+C detected! Saving model before exit...")
    df = pd.DataFrame(temp)
    safe_save(df.to_csv, csv_path)
    safe_save(lambda: json.dump(features, open(features_path, "w"), indent=4))
    safe_save(joblib.dump, scaler, scaler_path)
    torch.save(model.state_dict(), model_path)
    print("Model saved!")
    sys.exit(0)

safe_save(lambda: pd.DataFrame(temp).to_csv(csv_path))
safe_save(lambda: json.dump(features, open(features_path, "w"), indent=4))
safe_save(joblib.dump, scaler, scaler_path)
torch.save(model.state_dict(), model_path)
print(" Model saved successfully.")


model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_classes = torch.argmax(test_preds, dim=1)
    final_acc = (test_classes == Y_test_tensor).float().mean().item()


model.load_state_dict(torch.load(final_path))
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_classes = torch.argmax(test_preds, dim=1)
    best_acc = (test_classes == Y_test_tensor).float().mean().item()

print(f"\n Final Val Accuracy: {final_acc:.4f} | Best Val Accuracy: {best_acc:.4f} | Best_metic: {best_metric:.4f}")
print("────────────────────────────────────────────────────────────")
print("Training loop completed, training loop exited "+ ("by user or error." if stop_training else "normally.")+" Model saved successfully.")
print("────────────────────────────────────────────────────────────")
