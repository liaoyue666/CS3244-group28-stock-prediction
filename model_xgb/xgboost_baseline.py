"""
XGBoost Baseline with Progress Bar & ROC Visualization
Author: Haoyun Deng (CS3244 Team)
"""

# ===== 1️⃣ Imports =====
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== 2️⃣ Load Dataset =====
DATA_PATH = "../data/combined_cleaned.csv"
df = pd.read_csv(DATA_PATH)
stock_df = df[df["Ticker"] == "AAPL"].copy().reset_index(drop=True)
print(f"✅ Loaded AAPL data — {stock_df.shape[0]} rows")

# ===== 3️⃣ Features & Target =====
features = ["lag1", "lag3", "lag5", "MA5", "MA20", "vol_5", "rel_vol"]
X = stock_df[features].values
y = stock_df["label"].values

# ===== 4️⃣ TimeSeriesSplit =====
tscv = TimeSeriesSplit(n_splits=5)
fold_results = []
tqdm_bar = tqdm(enumerate(tscv.split(X), 1), total=5, desc="Running folds")

# ===== 5️⃣ Run Cross-Validation with Progress =====
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

# ===== 6️⃣ Summarize Results =====
results_df = pd.DataFrame(fold_results, columns=["Fold", "Accuracy", "F1", "ROC_AUC"])
print("\n📊 Fold Results:\n", results_df)
print("\n📈 Average Performance:")
print(results_df.mean(numeric_only=True))

results_df.to_csv("xgb_baseline_results.csv", index=False)

# ===== 7️⃣ ROC Curve (last fold for visualization) =====
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
print("\n✅ ROC curve saved as xgb_roc_curve.png")