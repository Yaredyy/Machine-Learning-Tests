# stockEvaluatorTorch.py

import torch
import torch.nn as nn
import yfinance as yf
import ta
import json
import joblib
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Get symbol and folder from args
if len(sys.argv) >= 3:
    symbol = sys.argv[1].upper()
    folder_name = sys.argv[2]
else:
    symbol = input("Enter Ticker Symbol (e.g., BTC-USD, AAPL): ").upper()
    folder_name = input("Enter Model Folder: ")

folder = f"pyTorch/StockTrader/{symbol}/{folder_name}/"

# Load features and model
features_path = folder + "features.json"
scaler_path = folder + "scaler.pkl"
scaler = joblib.load(scaler_path)
with open(features_path, "r") as f:
    features = json.load(f)

# Define model
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
model = StockTransformer(input_size=len(features))
model.load_state_dict(torch.load(folder + "final_model.pt"))
model.eval()

# Download data
print(f"Evaluating {symbol} model...")
data = yf.download(symbol, period="max", auto_adjust=False)
if data.empty:
    print("No data available.")
    sys.exit()

# Feature engineering
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

data['target'] = (data['Close'].shift(-3) > data['Close']).astype(int)
X = data[features].values.astype(np.float32)
Y = data['target'].values.astype(np.int64)
X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Convert to tensors
X_test_tensor = torch.tensor(X_test).unsqueeze(1)
Y_test_tensor = torch.tensor(Y_test)

# Evaluate
weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=Y_test)
class_weights = torch.tensor(weights, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights)

with torch.no_grad():
    outputs = model(X_test_tensor)
    loss = criterion(outputs, Y_test_tensor).item()
    predicted = torch.argmax(outputs, dim=1)
    flipped = 1 - predicted

    # Accuracy
    original_acc = (predicted == Y_test_tensor).float().mean().item()
    flipped_acc = (flipped == Y_test_tensor).float().mean().item()

    # Loss
    criterion = nn.CrossEntropyLoss()
    original_loss = criterion(outputs, Y_test_tensor).item()

    # For flipped loss, flip targets instead (not predictions)
    flipped_targets = 1 - Y_test_tensor
    flipped_loss = criterion(outputs, flipped_targets).item()


print(f"Evaluation Results for {symbol} final model:")
print(f"   Original Loss:     {original_loss:.4f}")
print(f"   Original Accuracy: {original_acc:.2%}")
print()
print(f"   Flipped Loss:      {flipped_loss:.4f}")
print(f"   Flipped Accuracy:  {flipped_acc:.2%}")

interpretation = ""

if abs(original_acc - 0.5) < 0.03 and abs(flipped_acc - 0.5) < 0.03:
    interpretation = "Model performs like random guessing."
elif flipped_acc > original_acc and flipped_acc > 0.53:
    interpretation = "Model may be learning the **opposite** signal."
elif original_acc > flipped_acc and original_acc > 0.53:
    interpretation = "Model shows useful predictive skill."
elif original_acc > flipped_acc:
    interpretation = "Original slightly better, but weak overall."
elif flipped_acc > original_acc:
    interpretation = "Flipped slightly better, but weak overall."
else:
    interpretation = "Unexpected pattern — check labels or model."

print(f"\nInterpretation: {interpretation}")

print("="*50)
# Load features and model
features_path = folder + "checkpoint_features.json"
scaler_path = folder + "checkpoint_scaler.pkl"
scaler = joblib.load(scaler_path)
with open(features_path, "r") as f:
    features = json.load(f)
model.load_state_dict(torch.load(folder + "checkpoint_model.pt"))
model.eval()
X = data[features].values.astype(np.float32)
Y = data['target'].values.astype(np.int64)
X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Convert to tensors
X_test_tensor = torch.tensor(X_test).unsqueeze(1)
Y_test_tensor = torch.tensor(Y_test)

# Evaluate
weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=Y_test)
class_weights = torch.tensor(weights, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights)

with torch.no_grad():
    outputs = model(X_test_tensor)
    loss = criterion(outputs, Y_test_tensor).item()
    predicted = torch.argmax(outputs, dim=1)
    flipped = 1 - predicted

    # Accuracy
    original_acc = (predicted == Y_test_tensor).float().mean().item()
    flipped_acc = (flipped == Y_test_tensor).float().mean().item()

    # Loss
    criterion = nn.CrossEntropyLoss()
    original_loss = criterion(outputs, Y_test_tensor).item()

    # For flipped loss, flip targets instead (not predictions)
    flipped_targets = 1 - Y_test_tensor
    flipped_loss = criterion(outputs, flipped_targets).item()


print(f"Evaluation Results for {symbol} checkpoint model:")
print(f"   Original Loss:     {original_loss:.4f}")
print(f"   Original Accuracy: {original_acc:.2%}")
print()
print(f"   Flipped Loss:      {flipped_loss:.4f}")
print(f"   Flipped Accuracy:  {flipped_acc:.2%}")

interpretation = ""

if abs(original_acc - 0.5) < 0.03 and abs(flipped_acc - 0.5) < 0.03:
    interpretation = "Model performs like random guessing."
elif flipped_acc > original_acc and flipped_acc > 0.53:
    interpretation = "Model may be learning the **opposite** signal."
elif original_acc > flipped_acc and original_acc > 0.53:
    interpretation = "Model shows useful predictive skill."
elif original_acc > flipped_acc:
    interpretation = "Original slightly better, but weak overall."
elif flipped_acc > original_acc:
    interpretation = "Flipped slightly better, but weak overall."
else:
    interpretation = "Unexpected pattern — check labels or model."

print(f"\nInterpretation: {interpretation}")
print("=" * 50 + "\n")