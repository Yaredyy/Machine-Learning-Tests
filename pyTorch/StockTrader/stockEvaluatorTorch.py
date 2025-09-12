# stockEvaluatorTorch.py

import torch
import torch.nn as nn
import yfinance as yf
import ta
import joblib
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
features = joblib.load(folder + "model_features.pkl")
scaler = joblib.load(folder + "scaler.pkl")

# Define model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        output, _ = self.lstm(x)
        out = self.dropout(output[:, -1, :])  # Use last output in sequence
        return self.fc(out)

# Instantiate and load model
model = StockLSTM(input_size=len(features))
model.load_state_dict(torch.load(folder + "torch_model.pt"))
model.eval()

# Download data
print(f"Evaluating {symbol} model...")
data = yf.download(symbol, period="1y", interval="1h", auto_adjust=False)
if data.empty:
    print("No data available.")
    sys.exit()

# Feature engineering
close = data['Close'].squeeze()
data['rsi'] = ta.momentum.RSIIndicator(close).rsi()
data['macd'] = ta.trend.MACD(close).macd()
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()
data['price_change'] = data['Close'].pct_change(1)
data['volatility'] = data['Close'].rolling(10).std()
data['ema_12'] = data['Close'].ewm(span=12).mean()
data['ema_26'] = data['Close'].ewm(span=26).mean()
data['target'] = (data['Close'].shift(-3) > data['Close']).astype(int)
data.dropna(inplace=True)

X = data[features].values.astype(np.float32)
Y = data['target'].values.astype(np.int64)
X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Convert to tensors
X_test_tensor = torch.tensor(X_test).unsqueeze(1)
Y_test_tensor = torch.tensor(Y_test)

# Evaluate
criterion = nn.CrossEntropyLoss()
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


print(f"Evaluation Results for {symbol}:")
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