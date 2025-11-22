"""
LightGBM Baseline – Single Stock Evaluation (AAPL)
Author: Deng Haoyun
"""

# ===== 1️⃣ Imports =====
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# ===== 2️⃣ Load Dataset =====
DATA_PATH = "../data/combined_cleaned.csv"
df = pd.read_csv(DATA_PATH)

# Focus on AAPL
stock_df = df[df["Ticker"] == "AAPL"].reset_index(drop=True)
print(f"✅ Loaded AAPL data: {stock_df.shape}")

# ===== 3️⃣ Define Features =====
features = ["lag1", "lag3", "lag5", "MA5", "MA20", "vol_5", "rel_vol"]
X = stock_df[features].values
y = stock_df["label"].values

# ===== 4️⃣ Cross-Validation =====
tscv = TimeSeriesSplit(n_splits=5)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(tqdm(tscv.split(X), total=5, desc="CV Folds"), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    fold_results.append([fold, acc, f1, auc])
    print(f"Fold {fold}: ACC={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

# ===== 5️⃣ Summary =====
results_df = pd.DataFrame(fold_results, columns=["Fold", "Accuracy", "F1", "ROC_AUC"])
results_df.to_csv("lgbm_baseline_results.csv", index=False)
print("\n📊 Mean Performance:")
print(results_df.mean(numeric_only=True))