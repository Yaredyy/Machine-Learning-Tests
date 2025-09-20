import subprocess
import os


# List of tickers to process
tickers = [
    "BTC-USD", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
   "ETH-USD", "SPY", "GC=F"
]

# Path to your Python executable inside env
python_path = r"myEnv/Scripts/python.exe"
python_path1 = f"myEnv/bin/python3"

if not os.path.exists(python_path):
    python_path = python_path1

if not os.path.exists(python_path):
    print("Error with python, please reset paths or reinstall python/env")
    sys.exit(0)

folder_name = input("Enter Model Folder: ")  # same folder for all or customize if you want

for symbol in tickers:
    print("\n" + "="*50)
    print(f"Running training for {symbol}...")

    proc_train = subprocess.Popen(
        [python_path, "-u", r"pyTorch/StockTrader/stockTrainer.py", symbol, folder_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Stream output line by line
    for line in proc_train.stdout:
        print(line, end='')

    proc_train.wait()
    if proc_train.returncode != 0:
        print(f"Training process for {symbol} exited with code {proc_train.returncode}")
        continue  # skip prediction if training failed

    # Check if training output indicated no data
    # (Optional: implement a smarter check if needed)

    model_path = f"pyTorch/StockTrader/{symbol}/{folder_name}/checkpoint_model.pt"
    if not os.path.exists(model_path):
        print(f"Model files not found for {symbol}, skipping prediction.")
        continue

    print(f"Running prediction for {symbol}...")

    proc_pred = subprocess.Popen(
        [python_path, "-u", r"pyTorch/StockTrader/stockPredictor.py", symbol, folder_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    for line in proc_pred.stdout:
        print(line, end='')

    proc_pred.wait()
    if proc_pred.returncode != 0:
        print(f"Prediction process for {symbol} exited with code {proc_pred.returncode}")
