"""
1. XGBoost Baseline with Progress Bar & ROC Visualization
Author: Deng Haoyun
"""

# ===== Imports =====
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== Load Dataset =====
DATA_PATH = "../data/combined_cleaned.csv"
df = pd.read_csv(DATA_PATH)
stock_df = df[df["Ticker"] == "AAPL"].copy().reset_index(drop=True)
print(f"Loaded AAPL data — {stock_df.shape[0]} rows")

# ===== Features & Target =====
features = ["lag1", "lag3", "lag5", "MA5", "MA20", "vol_5", "rel_vol"]
X = stock_df[features].values
y = stock_df["label"].values

# ===== TimeSeriesSplit =====
tscv = TimeSeriesSplit(n_splits=5)
fold_results = []
tqdm_bar = tqdm(enumerate(tscv.split(X), 1), total=5, desc="Running folds")

# ===== Run Cross-Validation with Progress =====
for fold, (train_idx, test_idx) in tqdm_bar:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    fold_results.append((fold, acc, f1, auc))
    tqdm_bar.set_postfix({"ACC": f"{acc:.3f}", "F1": f"{f1:.3f}", "AUC": f"{auc:.3f}"})

# ===== Summarize Results =====
results_df = pd.DataFrame(fold_results, columns=["Fold", "Accuracy", "F1", "ROC_AUC"])
print("\n Fold Results:\n", results_df)
print("\n Average Performance:")
print(results_df.mean(numeric_only=True))

results_df.to_csv("xgb_baseline_results.csv", index=False)

# ===== ROC Curve (last fold for visualization) =====
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – XGBoost Baseline (AAPL)")
plt.legend()
plt.tight_layout()
plt.savefig("xgb_roc_curve.png", dpi=150)
plt.close()
print("\n ROC curve saved as xgb_roc_curve.png")

"""
2. XGBoost Hyperparameter Tuning – TimeSeriesSplit
Author: Deng Haoyun
"""

# ===== Imports =====
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== Load Data =====
DATA_PATH = "../data/combined_cleaned.csv"
df = pd.read_csv(DATA_PATH)

# Focus on one representative stock first (AAPL)
stock_df = df[df["Ticker"] == "AAPL"].copy().reset_index(drop=True)
print(f"Loaded AAPL data — {stock_df.shape[0]} rows")

# ===== Define Features & Target =====
features = ["lag1", "lag3", "lag5", "MA5", "MA20", "vol_5", "rel_vol"]
X = stock_df[features].values
y = stock_df["label"].values

# ===== Parameter Grid =====
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
grid = list(ParameterGrid(param_grid))
print(f"Total combinations: {len(grid)}")

# ===== TimeSeriesSplit =====
tscv = TimeSeriesSplit(n_splits=3)

# ===== Run Tuning Loop =====
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

# ===== Analyse Results =====
results_df = pd.DataFrame(results).sort_values("mean_auc", ascending=False)
results_df.to_csv("xgb_tuning_results.csv", index=False)
print("\n Top 5 Results:")
print(results_df.head())

best_params = results_df.iloc[0].to_dict()
print(f"\n Best Params:\n{best_params}")

# ===== Plot Top Results =====
plt.figure(figsize=(8, 5))
plt.plot(range(len(results_df)), results_df["mean_auc"], marker="o")
plt.xlabel("Parameter Combination Index")
plt.ylabel("Mean AUC (3-fold)")
plt.title("XGBoost Tuning Results (AAPL)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("xgb_tuning_auc.png", dpi=150)
plt.close()
print("\n Plot saved as xgb_tuning_auc.png")


"""
3. XGBoost Multi-Stock Evaluation using Best Params
Author: Deng Haoyun
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== Load Dataset =====
DATA_PATH = "../data/combined_cleaned_extended.csv"
df = pd.read_csv(DATA_PATH)

tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
features = ["lag1", "lag3", "lag5", "MA5", "MA20", "vol_5", "rel_vol", "momentum", "volatility"]


best_params = {
    "max_depth": 7,
    "learning_rate": 0.01,
    "n_estimators": 100,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
}

print("Using best parameters:", best_params)

# ===== TimeSeries CV Function =====
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

# ===== Evaluate Across Stocks =====
results = []
for ticker in tqdm(tickers, desc="Evaluating Stocks"):
    stock_df = df[df["Ticker"] == ticker].reset_index(drop=True)
    acc, f1, auc = evaluate_stock(stock_df, best_params)
    results.append([ticker, acc, f1, auc])
    print(f"{ticker}: ACC={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

results_df = pd.DataFrame(results, columns=["Ticker", "Accuracy", "F1", "ROC_AUC"])
results_df.to_csv("xgb_multistock_results.csv", index=False)

print("\n Final Multi-Stock Results:")
print(results_df)
print("\nAverage Performance:")
print(results_df.mean(numeric_only=True))

# ===== Plot Results =====
plt.figure(figsize=(7,5))
plt.bar(results_df["Ticker"], results_df["ROC_AUC"], color="steelblue")
plt.ylabel("Mean ROC AUC (5-fold)")
plt.title("XGBoost Performance Across Stocks")
plt.tight_layout()
plt.savefig("xgb_multistock_performance.png", dpi=150)
plt.close()
print("\n Plot saved as xgb_multistock_performance.png")