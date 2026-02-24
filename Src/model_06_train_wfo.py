"""
Executes Combinatorial Walk-Forward Optimization (WFO) using TimeSeriesSplit.
Ensures zero Look-Ahead bias by strictly rolling the training window forward over time.
"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import os

INPUT_FILE = "Data/ml_master_matrix.csv"
OUTPUT_CSV = "Data/ai_buy_signals_wfo.csv"
HORIZONS = ['1M', '2M', '6M']
FEATURES = ['Role_Weight', 'Consensus_Score', 'Portfolio_Pct', 'Pct_Volume_Absorbed', 'Value', 'Close', 'Volume', 'ATR_14']

def main():
    print(f"⏳ Booting Institutional Walk-Forward Optimization Engine...")
    df = pd.read_csv(INPUT_FILE, parse_dates=['Trade_Date']).sort_values('Trade_Date')
    df.dropna(subset=[f'Target_{h}' for h in HORIZONS] + [f'Ret_{h}' for h in HORIZONS] + FEATURES, inplace=True)
    df.reset_index(drop=True, inplace=True)

    master_wfo_df = df.copy()

    for horizon in HORIZONS:
        print(f"\n🔄 Running WFO for Horizon: {horizon}")
        target_col = f'Target_{horizon}'
        X = df[FEATURES]
        y = df[target_col]
        
        # gap=126 ensures a full 6 months of separation between train and test sets!
        tscv = TimeSeriesSplit(n_splits=5, gap=126)
        
        out_of_sample_preds = pd.Series(index=df.index, dtype=float)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            
            model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
            model.fit(X_train, y_train)
            
            out_of_sample_preds.iloc[test_idx] = model.predict_proba(X_test)[:, 1]
            
        master_wfo_df[f'AI_Confidence_{horizon}'] = out_of_sample_preds

    # Drop the rows that were exclusively used for the very first training fold (they have no predictions)
    master_wfo_df.dropna(subset=[f'AI_Confidence_{h}' for h in HORIZONS], inplace=True)
    master_wfo_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ WFO Out-Of-Sample Signals saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()