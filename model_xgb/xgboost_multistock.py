"""
XGBoost Multi-Stock Evaluation using Best Params
Author: Haoyun Deng (CS3244 Team)
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== 1️⃣ Load Dataset =====
DATA_PATH = "../data/combined_cleaned_extended.csv"
df = pd.read_csv(DATA_PATH)

tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
features = ["lag1", "lag3", "lag5", "MA5", "MA20", "vol_5", "rel_vol", "momentum", "volatility"]

# ===== 2️⃣ Insert Your Best Params (from xgb_tuning_results.csv) =====
best_params = {
    "max_depth": 7,
    "learning_rate": 0.01,
    "n_estimators": 100,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
}

print("✅ Using best parameters:", best_params)

# ===== 3️⃣ TimeSeries CV Function =====
def evaluate_stock(stock_df, params):
    X = stock_df[features].values
    y = stock_df["label"].values
    tscv = TimeSeriesSplit(n_splits=5)

    accs, f1s, aucs = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            **params,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))

    return np.mean(accs), np.mean(f1s), np.mean(aucs)

# ===== 4️⃣ Evaluate Across Stocks =====
results = []
for ticker in tqdm(tickers, desc="Evaluating Stocks"):
    stock_df = df[df["Ticker"] == ticker].reset_index(drop=True)
    acc, f1, auc = evaluate_stock(stock_df, best_params)
    results.append([ticker, acc, f1, auc])
    print(f"{ticker}: ACC={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

results_df = pd.DataFrame(results, columns=["Ticker", "Accuracy", "F1", "ROC_AUC"])
results_df.to_csv("xgb_multistock_results.csv", index=False)

print("\n📊 Final Multi-Stock Results:")
print(results_df)
print("\nAverage Performance:")
print(results_df.mean(numeric_only=True))

# ===== 5️⃣ Plot Results =====
plt.figure(figsize=(7,5))
plt.bar(results_df["Ticker"], results_df["ROC_AUC"], color="steelblue")
plt.ylabel("Mean ROC AUC (5-fold)")
plt.title("XGBoost Performance Across Stocks")
plt.tight_layout()
plt.savefig("xgb_multistock_performance.png", dpi=150)
plt.close()
print("\n✅ Plot saved as xgb_multistock_performance.png")