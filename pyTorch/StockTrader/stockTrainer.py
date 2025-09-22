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

print("CUDA Available:", torch.cuda.is_available())
print("CUDA version (PyTorch built with):", torch.version.cuda)


if torch.cuda.is_available():
    print("Using GPU. Name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")



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


# Sequence windowing
# def create_sequences(X, Y, window_size):
#     X_seq, Y_seq = [], []
#     for i in range(len(X) - window_size):
#         X_seq.append(X[i:i+window_size])
#         Y_seq.append(Y[i+window_size])
#     return np.array(X_seq), np.array(Y_seq)

year = 2025
wind=1
c=0

def new_start():
    return datetime(year-(wind*(c+1)),1,1)
def new_end():
    return datetime(year-(wind*(c)),1,1)
 
print(f"Downloading {symbol} data...")
data = yf.download(symbol, start= new_start(), end= new_end(), auto_adjust=False)
c+=1
if data.empty:
    raise Exception("No data downloaded.")



def formatNewData():
    print("Computing new technical indicators...")
    close = data['Close'].squeeze()
    data['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    data['macd'] = ta.trend.MACD(close).macd()
    data['sma_20'] = data['Close'].rolling(20).mean()
    data['sma_50'] = data['Close'].rolling(50).mean()
    data['price_change'] = data['Close'].pct_change(1)
    data['volatility'] = data['Close'].rolling(10).std()
    data['ema_12'] = data['Close'].ewm(span=12).mean()
    data['ema_26'] = data['Close'].ewm(span=26).mean()

    # Target: will the price go up in 24 hours?
    data['target'] = (data['Close'].shift(-24) > data['Close']).astype(int)
    data.dropna(inplace=True)


    X = data[features].values.astype(np.float32)
    Y = data['target'].values.astype(np.int64)


    X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False,train_size=.7)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.3, shuffle=False,train_size=.7)


    scaler = StandardScaler()

    # Fit on train, transform all sets
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long).to(device)
    print(f"New Dataset: {len(X)} samples | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    unique, counts = np.unique(Y_test, return_counts=True)
    print("New Class distribution:", dict(zip(unique, counts)))
    if len(unique) < 2:
        print("Only one class found in dataset, disabling class weights.")
        class_weights = None
    else:
        weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=Y_test)
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = FocalLoss(weight=class_weights)
    if torch.cuda.is_available():
        train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=256, shuffle=True,pin_memory=True)
    else:
        train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=256, shuffle=True,pin_memory=False)



features = ['rsi', 'macd', 'sma_20', 'sma_50', 'price_change', 'volatility', 'Volume', 'ema_12', 'ema_26']

print("Computing new technical indicators...")
close = data['Close'].squeeze()
data['rsi'] = ta.momentum.RSIIndicator(close).rsi()
data['macd'] = ta.trend.MACD(close).macd()
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()
data['price_change'] = data['Close'].pct_change(1)
data['volatility'] = data['Close'].rolling(10).std()
data['ema_12'] = data['Close'].ewm(span=12).mean()
data['ema_26'] = data['Close'].ewm(span=26).mean()

# Target: will the price go up in 24 hours?
data['target'] = (data['Close'].shift(-24) > data['Close']).astype(int)
data.dropna(inplace=True)


X = data[features].values.astype(np.float32)
Y = data['target'].values.astype(np.int64)


X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False,train_size=.7)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.3, shuffle=False,train_size=.7)


scaler = StandardScaler()

# Fit on train, transform all sets
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.long).to(device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long).to(device)




print(f"Dataset: {len(X)} samples | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
unique, counts = np.unique(Y_test, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))



# Define model
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
            x = x.unsqueeze(1)  # [B, 1, F]

        # Positional embedding (can be learned or sinusoidal)
        x = self.embedding(x) + self.position_embeddings
        x = self.transformer_encoder(x)
        x = self.dropout(x[:, -1, :])  # last time step
        return self.fc(x)


model = StockTransformer(input_size=len(features))
model = model.to(device)
if len(unique) < 2:
    print("Only one class found in dataset, disabling class weights.")
    class_weights = None
else:
    weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=Y_test)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

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
    
criterion = FocalLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=0.01, patience=50)


def safe_save(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Failed to save with {func.__name__}: {e}")

csv_path = folder + "ModelSummary.csv"
features_path = folder + "features.json"
scaler_path = folder + "scaler.pkl"



epochs = 100000
patien = 50010
best_metric = float(0)
counter = 0

if torch.cuda.is_available():
    train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=256, shuffle=True,pin_memory=True)
else:
    train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=256, shuffle=True,pin_memory=False)
    

temp_path = folder + "checkpoint_model_temp.pt"
final_path = folder + "checkpoint_model.pt"

checkpoint_features_path = folder + "checkpoint_features.json"
checkpoint_scaler_path = folder + "checkpoint_scaler.pkl"
Last_path = folder + "final_model.pt"


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
            df = pd.DataFrame(temp)
            safe_save(df.to_csv, csv_path)
            safe_save(lambda: json.dump(features, open(checkpoint_features_path, "w"), indent=4))
            safe_save(joblib.dump, scaler, checkpoint_scaler_path)
            print(f"Epoch {epoch+1}: New best val accuracy: {val_acc:.4f} — Checkpoint saved!")
            counter = 0
        else:
            counter += 1
            if counter > patien:
                print(f"Early stopping triggered at Epoch:{epoch+1}, after a pactience of {patien}.")
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
                "Interval": "24h",
                "Period": "1y"
                # "Window Size": window_size
            })
        
        if (epoch +1)%100 == 0:
            print(f"Downloading new {symbol} data...")

            try:
                newdata = yf.download(symbol, start= new_start(), end= new_end(), auto_adjust=False)
            except Exception:
                print(f"download failed, caught:{e}")
            
            c+=1
    
            if newdata.empty:
                print("No data downloaded.")
            else:
                data = newdata
                formatNewData()
            




except Exception as e:
    print(f"Exception caught: {e}")
    print("Saving current model state before exiting...")
    df = pd.DataFrame(temp)
    safe_save(df.to_csv, csv_path)
    safe_save(lambda: json.dump(features, open(features_path, "w"), indent=4))
    safe_save(joblib.dump, scaler, scaler_path)
    torch.save(model.state_dict(), Last_path)
    print("Model saved!")
    sys.exit(1)

except KeyboardInterrupt:
    print("Ctrl+C detected! Saving model before exit...")
    df = pd.DataFrame(temp)
    safe_save(df.to_csv, csv_path)
    safe_save(lambda: json.dump(features, open(features_path, "w"), indent=4))
    safe_save(joblib.dump, scaler, scaler_path)
    torch.save(model.state_dict(), Last_path)
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
torch.save(model.state_dict(), Last_path)

print(" Model saved successfully.")

model.load_state_dict(torch.load(final_path))

model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_classes = torch.argmax(test_preds, dim=1)
    best_acc = (test_classes == Y_test_tensor).float().mean().item()


print(f"\n Final Val Accuracy: {final_acc:.4f} | Best Val Accuracy: {best_acc:.4f} | Best_metic: {best_metric:.4f}")

if stop_training:
    print("Training loop exited by user or error")
else:
    print("Training loop exited normally")
