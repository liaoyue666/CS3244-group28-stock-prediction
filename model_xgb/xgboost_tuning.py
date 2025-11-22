"""
XGBoost Hyperparameter Tuning – TimeSeriesSplit
Author: Haoyun Deng (CS3244 Team)
"""

# ===== 1️⃣ Imports =====
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== 2️⃣ Load Data =====
DATA_PATH = "../data/combined_cleaned.csv"
df = pd.read_csv(DATA_PATH)

# Focus on one representative stock first (AAPL)
stock_df = df[df["Ticker"] == "AAPL"].copy().reset_index(drop=True)
print(f"✅ Loaded AAPL data — {stock_df.shape[0]} rows")

# ===== 3️⃣ Define Features & Target =====
features = ["lag1", "lag3", "lag5", "MA5", "MA20", "vol_5", "rel_vol"]
X = stock_df[features].values
y = stock_df["label"].values

# ===== 4️⃣ Parameter Grid =====
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
grid = list(ParameterGrid(param_grid))
print(f"🔍 Total combinations: {len(grid)}")

# ===== 5️⃣ TimeSeriesSplit =====
tscv = TimeSeriesSplit(n_splits=3)

# ===== 6️⃣ Run Tuning Loop =====
results = []
for params in tqdm(grid, desc="Tuning grid"):
    auc_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            **params
        )
        model.fit(X_train, y_train, verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        auc_scores.append(auc)

    mean_auc = np.mean(auc_scores)
    results.append({**params, "mean_auc": mean_auc})

# ===== 7️⃣ Analyse Results =====
results_df = pd.DataFrame(results).sort_values("mean_auc", ascending=False)
results_df.to_csv("xgb_tuning_results.csv", index=False)
print("\n📊 Top 5 Results:")
print(results_df.head())

best_params = results_df.iloc[0].to_dict()
print(f"\n✅ Best Params:\n{best_params}")

# ===== 8️⃣ Plot Top Results =====
plt.figure(figsize=(8, 5))
plt.plot(range(len(results_df)), results_df["mean_auc"], marker="o")
plt.xlabel("Parameter Combination Index")
plt.ylabel("Mean AUC (3-fold)")
plt.title("XGBoost Tuning Results (AAPL)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("xgb_tuning_auc.png", dpi=150)
plt.close()
print("\n📈 Plot saved as xgb_tuning_auc.png")