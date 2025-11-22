"""
LightGBM SHAP Interpretability – Multi-Stock Analysis
Author: Deng Haoyun

Generates (for each stock):
 - lgbm_feature_importance_<TICKER>.png
 - lgbm_shap_summary_<TICKER>.png
 - lgbm_shap_dependence_<TICKER>_<feature>.png
"""

# ===== 1️⃣ Imports =====
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import os

# ===== 2️⃣ Config =====
DATA_PATH = "../data/combined_cleaned.csv"
OUTPUT_DIR = "./"
tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]

# Tuned parameters from your tuning results
best_params = {
    "num_leaves": 31,
    "learning_rate": 0.01,
    "n_estimators": 200,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}

# ===== 3️⃣ Load Dataset =====
df = pd.read_csv(DATA_PATH)
features = ["lag1", "lag3", "lag5", "MA5", "MA20", "vol_5", "rel_vol"]

# ===== 4️⃣ SHAP Analysis Loop =====
for ticker in tickers:
    stock_df = df[df["Ticker"] == ticker].reset_index(drop=True)
    if stock_df.empty:
        print(f"⚠️ No data found for {ticker}, skipping.")
        continue

    print(f"\n=== 🔍 SHAP Analysis for {ticker} ({stock_df.shape[0]} rows) ===")
    X = stock_df[features].values
    y = stock_df["label"].values

    # Train model
    model = LGBMClassifier(objective="binary", random_state=42, verbose=-1, **best_params)
    model.fit(X, y)

    # ===== Feature Importance =====
    importances = model.feature_importances_
    plt.figure(figsize=(6,4))
    plt.barh(features, importances, color="teal")
    plt.xlabel("Feature Importance (Gain)")
    plt.title(f"LightGBM Feature Importance ({ticker})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"lgbm_feature_importance_{ticker}.png"), dpi=150)
    plt.close()
    print(f"✅ Saved lgbm_feature_importance_{ticker}.png")

    # ===== SHAP Explainer =====
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --- Summary Plot ---
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    plt.tight_layout()
    plt.title(f"SHAP Summary Plot ({ticker})")
    plt.savefig(os.path.join(OUTPUT_DIR, f"lgbm_shap_summary_{ticker}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved lgbm_shap_summary_{ticker}.png")

    # --- Dependence Plots (Top 3) ---
    top_features_idx = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:3]
    for idx in top_features_idx:
        fname = features[idx]
        shap.dependence_plot(
            fname, shap_values, X, feature_names=features, show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"lgbm_shap_dependence_{ticker}_{fname}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved lgbm_shap_dependence_{ticker}_{fname}.png")

print("\n🎯 Multi-stock SHAP analysis complete for all tickers.")