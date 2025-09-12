import subprocess
import os

# Path to Python executable in your environment
python_path = r"C:/Users/yared/Documents/GitHub/Machine-Learning-Tests/myEnv/Scripts/python.exe"

# List of tickers
tickers = [
    "BTC-USD", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "ETH-USD", "SPY", "GC=F"
]

folder_name = input("Enter Model Folder: ")

for symbol in tickers:
    print("\n" + "="*50)
    print(f"Evaluating model for {symbol}...")

    model_path = f"pyTorch/StockTrader/{symbol}/{folder_name}/torch_model.pt"
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
