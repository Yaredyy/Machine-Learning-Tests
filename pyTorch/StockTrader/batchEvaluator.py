import subprocess
import os


# Path to your Python executable inside env
python_path = r"myEnv/Scripts/python.exe"
python_path1 = f"myEnv/bin/python3"

if not os.path.exists(python_path):
    python_path = python_path1

if not os.path.exists(python_path):
    print("Error with python")
    sys.exit(0)


# List of tickers
tickers = [
    "AAPL", "AMZN","BTC-USD", "ETH-USD",
    "GC=F", "GOOGL", "MSFT","SPY", "TSLA"
]

folder_name = input("Enter Model Folder: ")

for symbol in tickers:
    print("\n" + "="*50)
    print(f"Evaluating model for {symbol}...")

    model_path = f"pyTorch/StockTrader/{symbol}/{folder_name}/checkpoint_model.pt"
    if not os.path.exists(model_path):
        print(f"Model not found for {symbol}, skipping...")
        continue

    # Run evaluator
    proc_eval = subprocess.Popen(
        [python_path, "pyTorch/StockTrader/stockEvaluatorTorch.py", symbol, folder_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    outs_eval, _ = proc_eval.communicate()
    print(outs_eval)
