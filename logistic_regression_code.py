import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def preprocess_data(df):
    df = df.copy()
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date'])

    df['dist_MA5'] = (df['Close'] - df['MA5']) / df['MA5']
    df['dist_MA20'] = (df['Close'] - df['MA20']) / df['MA20']

    df.dropna(subset=['dist_MA5', 'dist_MA20'], inplace=True)
    return df

FEATURES = ['lag1', 'lag3', 'lag5', 'dist_MA5', 'dist_MA20', 'vol_5', 'rel_vol']
TARGET = 'label'

# Experiment 1: Intra-Stock TimeSeries Cross-Validation
def run_experiment_1(df):
    print("Experiment 1: Intra-Stock TimeSeries Cross-Validation")
    
    results = []
    
    for Ticker in df['Ticker'].unique():
        Ticker_data = df[df['Ticker'] == Ticker].copy()
        X = Ticker_data[FEATURES]
        y = Ticker_data[TARGET]
        
        tscv = TimeSeriesSplit(n_splits=5)
        fold_metrics = {'acc': [], 'f1': [], 'auc': []}
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(class_weight='balanced', random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            fold_metrics['acc'].append(accuracy_score(y_test, y_pred))
            fold_metrics['f1'].append(f1_score(y_test, y_pred))
            try:
                fold_metrics['auc'].append(roc_auc_score(y_test, y_proba))
            except:
                fold_metrics['auc'].append(0.5) 

        results.append({
            'Ticker': Ticker,
            'Mean_Acc': np.mean(fold_metrics['acc']),
            'Mean_F1': np.mean(fold_metrics['f1']),
            'Mean_AUC': np.mean(fold_metrics['auc'])
        })
    
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df

# Experiment 2: Cross-Stock Generalisation
def run_experiment_2(df):
    print("Experiment 2: Cross-Stock Generalisation (Train 4, Test 1)")
    
    Tickers = df['Ticker'].unique()
    results = []
    
    for test_Ticker in Tickers:

        train_df = df[df['Ticker'] != test_Ticker]
        test_df = df[df['Ticker'] == test_Ticker]
        
        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]
        X_test = test_df[FEATURES]
        y_test = test_df[TARGET]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        results.append({
            'Test_Ticker': test_Ticker,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1_Score': f1_score(y_test, y_pred),
            'ROC_AUC': roc_auc_score(y_test, y_proba)
        })
        
    results_df = pd.DataFrame(results)
    print(results_df)
    print(f"\nAverage Generalisation Accuracy: {results_df['Accuracy'].mean():.4f}")
    return results_df

# Experiment 3: Feature & Behaviour Analysis
def run_experiment_3(df):
    print("Experiment 3: Feature Analysis & Coefficient Stability")
    
    coef_data = []
    
    for Ticker in df['Ticker'].unique():
        Ticker_data = df[df['Ticker'] == Ticker]
        X = Ticker_data[FEATURES]
        y = Ticker_data[TARGET]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_scaled, y)
        
        for feat, coef in zip(FEATURES, model.coef_[0]):
            coef_data.append({'Ticker': Ticker, 'Feature': feat, 'Coefficient': coef})
            
    coef_df = pd.DataFrame(coef_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=coef_df, x='Feature', y='Coefficient', hue='Ticker', palette='viridis')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Stability of Feature Coefficients Across Tickers (Experiment 3)")
    plt.ylabel("Coefficient (Log Odds)")
    plt.xlabel("Feature")
    plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    avg_importance = coef_df.groupby('Feature')['Coefficient'].mean().sort_values(ascending=False)
    print("\nAverage Feature Coefficients (Across All Tickers):")
    print(avg_importance)

# Main entrance
if __name__ == "__main__":
    df_raw =  pd.read_csv('stock_data.csv')

    df_clean = preprocess_data(df_raw)
    
    res1 = run_experiment_1(df_clean)
    res2 = run_experiment_2(df_clean)
    run_experiment_3(df_clean)