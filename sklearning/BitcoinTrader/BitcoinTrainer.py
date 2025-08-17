# trading_model.py
# Predict: Will BTC price go up in the next 3 hours?

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split as splitter
from sklearn.metrics import accuracy_score
import pandas as pd
import yfinance as yf
import ta
import joblib

# --- 1. Download Data ---
print("Downloading BTC-USD data...")
data = yf.download("BTC-USD", period="6mo", interval="1h")
if data.empty:
    raise Exception("No data downloaded. Check internet or ticker.")

# --- 2. Add Features (Technical Indicators) ---
print("Adding technical indicators...")
data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
data['macd'] = ta.trend.MACD(data['Close']).macd()
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()
data['price_change'] = data['Close'].pct_change(1)
data['volatility'] = data['Close'].rolling(10).std()

# --- 3. Target: Will price be higher in 3 hours?
data['target'] = (data['Close'].shift(-3) > data['Close']).astype(int)

# --- 4. Drop NaN and select features
data.dropna(inplace=True)
features = ['rsi', 'macd', 'sma_20', 'sma_50', 'price_change', 'volatility', 'Volume']
X = data[features].values
Y = data['target'].values

# --- 5. Test Different Splits & Train Model ---
print("Training and testing across different train/test splits...")
results = []

for i in range(1, 21):
    test_size = 0.05 * i  # From 5% to 100% step 5%
    
    # ⚠️ Time-series split: older = train, newer = test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # Train model
    model = RFC(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, Y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)

    results.append({
        "Split_Index": i,
        "Test_Size": round(test_size, 3),
        "Samples_Train": len(X_train),
        "Samples_Test": len(X_test),
        "Accuracy": round(accuracy, 4)
    })

    print(f"Split {i}: Test={test_size:.2f} | "
          f"Train={len(X_train)} | Test={len(X_test)} | "
          f"Accuracy={accuracy:.4f}")

# --- 6. Save Results to CSV ---
df_results = pd.DataFrame(results)
df_results.to_csv("trading_model_results.csv", index=False)
print(f"\n✅ Results saved to 'trading_model_results.csv'")

# --- 7. Retrain on Best Split (Highest Accuracy)
best_row = df_results.loc[df_results['Accuracy'].idxmax()]
best_test_size = best_row['Test_Size']
print(f"\n🔍 Best accuracy: {best_row['Accuracy']:.4f} at test_size={best_test_size}")

# Use best split to train final model
final_split_idx = int(len(X) * (1 - best_test_size))
X_train_final, X_test_final = X[:final_split_idx], X[final_split_idx:]
Y_train_final, Y_test_final = Y[:final_split_idx], Y[final_split_idx:]

final_model = RFC(n_estimators=100, random_state=42)
final_model.fit(X_train_final, Y_train_final)

# Final test accuracy
final_pred = final_model.predict(X_test_final)
final_acc = accuracy_score(Y_test_final, final_pred)
print(f"🎯 Final model accuracy: {final_acc:.4f}")

# --- 8. Save the Trained Model ---
joblib.dump(final_model, "trading_model.pkl")
joblib.dump(features, "model_features.pkl")  # Save feature names for later
print("💾 Model and features saved: 'trading_model.pkl' and 'model_features.pkl'")

# --- 9. Feature Importance (Bonus)
importances = final_model.feature_importances_
feat_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n📈 Feature Importance:")
print(feat_importance)