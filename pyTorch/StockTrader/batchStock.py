import subprocess
import os

# Path to your Python executable inside your env
python_path = r"C:/Users/yared/Documents/GitHub/Machine-Learning-Tests/myEnv/Scripts/python.exe"

# List of tickers to process
tickers = [
    "BTC-USD", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
   "ETH-USD", "SPY", "GC=F"
]

folder_name = input("Enter Model Folder: ")  # same folder for all or customize if you want

for symbol in tickers:
    print("\n" + "="*50)
    print(f"Running training for {symbol}...")

    # Run trainer script
    proc_train = subprocess.Popen(
        [python_path, r"pyTorch/StockTrader/stockTrainer.py", symbol, folder_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Send inputs: symbol and folder_name
    outs, _ = proc_train.communicate()
    print(outs)

    if "No data downloaded." in outs:
        print(f"Skipping prediction for {symbol} due to training data error.")
        continue

    model_path = f"pyTorch/StockTrader/{symbol}/{folder_name}/model_features.pkl"
    if not os.path.exists(model_path):
        print(f"Model files not found for {symbol}, skipping prediction.")
        continue

    print(f"Running prediction for {symbol}...")

    # Run predictor script
    proc_pred = subprocess.Popen(
        [python_path, r"pyTorch/StockTrader/stockPredictor.py", symbol, folder_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    # Send inputs: symbol and folder_name
    outs_pred, _ = proc_pred.communicate()
    print(outs_pred)

