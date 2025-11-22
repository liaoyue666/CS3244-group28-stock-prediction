"""
LightGBM Multi-Stock Evaluation
Author: Deng Haoyun
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== Load Data =====
DATA_PATH = "../data/combined_cleaned_extended.csv"
df = pd.read_csv(DATA_PATH)
tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
features = ["lag1", "lag3", "lag5", "MA5", "MA20", "vol_5", "rel_vol", "momentum", "volatility"]

# ===== Insert Best Params (from tuning CSV) =====
best_params = {
    "num_leaves": 31,
    "learning_rate": 0.01,
    "n_estimators": 200,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}
print("✅ Using best parameters:", best_params)

# ===== Evaluation Function =====
def evaluate_stock(stock_df, params):
    X = stock_df[features].values
    y = stock_df["label"].values
    tscv = TimeSeriesSplit(n_splits=5)

    accs, f1s, aucs = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LGBMClassifier(objective="binary", random_state=42, verbose=-1, **params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))

    return np.mean(accs), np.mean(f1s), np.mean(aucs)

# ===== Evaluate Across Stocks =====
results = []
for ticker in tqdm(tickers, desc="Evaluating Stocks"):
    stock_df = df[df["Ticker"] == ticker].reset_index(drop=True)
    acc, f1, auc = evaluate_stock(stock_df, best_params)
    results.append([ticker, acc, f1, auc])
    print(f"{ticker}: ACC={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")

results_df = pd.DataFrame(results, columns=["Ticker", "Accuracy", "F1", "ROC_AUC"])
results_df.to_csv("lgbm_multistock_results.csv", index=False)
print("\n📊 Multi-Stock Results:\n", results_df)

# ===== Plot =====
plt.figure(figsize=(7,5))
plt.bar(results_df["Ticker"], results_df["ROC_AUC"], color="mediumseagreen")
plt.ylabel("Mean ROC AUC (5-fold)")
plt.title("LightGBM Performance Across Stocks")
plt.tight_layout()
plt.savefig("lgbm_multistock_performance.png", dpi=150)
plt.close()
print("✅ Saved lgbm_multistock_performance.png")