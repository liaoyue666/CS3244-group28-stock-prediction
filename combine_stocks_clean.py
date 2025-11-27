"""
Combine and clean multiple stock datasets 
Author: Deng Haoyun
"""

import os
import pandas as pd
import numpy as np

# ========== Configuration ==========
DATA_DIR = "../data"   
OUTPUT_FILE = "combined_cleaned.csv"

TICKERS = ["aapl", "amzn", "goog", "googl", "msft", "tsla"]

# ========== Helper Functions ==========

def clean_one_stock(file_path, ticker):
    df = pd.read_csv(file_path)
    
    # Basic cleanup
    df.columns = [col.strip().capitalize() for col in df.columns]
    df = df.dropna(subset=["Close"])  # remove empty rows
    
    # Ensure chronological order
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Compute log returns
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["target"] = df["log_return"].shift(-1)  # predict next-day log return
    df["label"] = (df["target"] > 0).astype(int)  # classification target (1 = up, 0 = down)
    
    # Lag features
    for lag in [1, 3, 5]:
        df[f"lag{lag}"] = df["log_return"].shift(lag)
    
    # Moving averages
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    
    # Volume-based features
    df["vol_5"] = df["Volume"].rolling(window=5).mean()
    df["rel_vol"] = df["Volume"] / df["vol_5"]
    
    # Drop rows with NaNs caused by shifting
    df = df.dropna().reset_index(drop=True)
    
    # Add ticker column
    df["Ticker"] = ticker.upper()
    
    # Reorder columns
    df = df[[
        "Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
        "log_return", "lag1", "lag3", "lag5",
        "MA5", "MA20", "vol_5", "rel_vol",
        "target", "label"
    ]]
    
    print(f" Cleaned {ticker.upper()} | Shape: {df.shape}")
    return df


# ==========  Apply to All Stocks ==========
all_stocks = []

for ticker in TICKERS:
    file_path = os.path.join(DATA_DIR, f"{ticker}.us.txt")
    if os.path.exists(file_path):
        cleaned = clean_one_stock(file_path, ticker)
        all_stocks.append(cleaned)
    else:
        print(f" Missing file: {file_path}")

# Combine all
combined_df = pd.concat(all_stocks, ignore_index=True)

# ==========  Save Combined Dataset ==========
combined_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n Combined dataset saved to {OUTPUT_FILE}")
print(f"Total rows: {combined_df.shape[0]}, Columns: {combined_df.shape[1]}")

# ==========  Quick Sanity Check ==========
print("\nSample:")
print(combined_df.head())