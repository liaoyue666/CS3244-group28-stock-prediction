"""
LightGBM Hyperparameter Tuning – TimeSeriesSplit
Author: Deng Haoyun
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===== Load Data =====
DATA_PATH = "../data/combined_cleaned.csv"
df = pd.read_csv(DATA_PATH)
stock_df = df[df["Ticker"] == "AAPL"].reset_index(drop=True)
features = ["lag1", "lag3", "lag5", "MA5", "MA20", "vol_5", "rel_vol"]
X = stock_df[features].values
y = stock_df["label"].values

# ===== Parameter Grid =====
param_grid = {
    "num_leaves": [15, 31, 63], #spans different model complexities (low-medium-high)
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 400, 600], #More trees = more model capacity but higher compute and potential overfitting
    "subsample": [0.8, 1.0], # fraction of data to sample for each tree
    "colsample_bytree": [0.8, 1.0], #fraction of features to sample for each tree
}
grid = list(ParameterGrid(param_grid))
tscv = TimeSeriesSplit(n_splits=3)

results = []

for params in tqdm(grid, desc="Tuning grid"):
    auc_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LGBMClassifier(
            objective="binary",
            random_state=42,
            verbose=-1,
            **params
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        auc_scores.append(auc)

    results.append({**params, "mean_auc": np.mean(auc_scores)})

results_df = pd.DataFrame(results).sort_values("mean_auc", ascending=False)
results_df.to_csv("lgbm_tuning_results.csv", index=False)
print("\n🔍 Top 5 Parameter Sets:\n", results_df.head())

# Plot
plt.figure(figsize=(7,5))
plt.plot(range(len(results_df)), results_df["mean_auc"], marker="o")
plt.xlabel("Parameter Combination")
plt.ylabel("Mean AUC (3-fold)")
plt.title("LightGBM Tuning Results (AAPL)")
plt.tight_layout()
plt.savefig("lgbm_tuning_auc.png", dpi=150)
print("✅ Saved lgbm_tuning_auc.png")