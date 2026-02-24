"""
General description: 
Executes Combinatorial Walk-Forward Optimization (WFO) for 1M, 2M, and 6M horizons.
Uses TimeSeriesSplit to ensure zero Look-Ahead bias by strictly rolling the training 
window forward over time. Fuses out-of-sample predictions into a single master file.
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# --- CONFIGURATION & HYPERPARAMETERS ---
# ==========================================

INPUT_FILE = "Data/ml_master_matrix.csv"
OUTPUT_CSV = "Data/ai_buy_signals_wfo.csv"

# --- THE HORIZONS TO TRAIN ---
HORIZONS = ['1M', '2M', '6M']

# --- EXACT ORIGINAL FEATURE SPACE ---
FEATURES = [
    'Role_Weight', 
    'Consensus_Score', 
    'Portfolio_Pct', 
    'Pct_Volume_Absorbed', 
    'Value', 
    'Close', 
    'Volume', 
    'ATR_14'
]

def main():
    print(f"🤖 Booting Institutional Walk-Forward Optimization Engine...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Check your path!")
        return

    # 1. LOAD & PREP MASTER DATA
    print(f"📦 Loading Master Matrix...")
    df = pd.read_csv(INPUT_FILE, parse_dates=['Trade_Date'])
    
    # Sort chronologically to prevent "Look-Ahead Bias"
    df.sort_values('Trade_Date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    master_wfo_df = df.copy()

    # 2. LOOP THROUGH EACH HORIZON
    for horizon in HORIZONS:
        print(f"\n====================================")
        print(f"🔄 RUNNING WFO HORIZON: {horizon}")
        print(f"====================================")
        
        target_col = f'Target_{horizon}'
        ret_col = f'Ret_{horizon}'
        confidence_col = f'AI_Confidence_{horizon}'
        
        # Check if the columns actually exist in the DB
        if target_col not in df.columns or ret_col not in df.columns:
            print(f"⚠️ Skipping {horizon}: Targets not found in database.")
            continue
            
        # Isolate data for this specific horizon and drop NaNs
        horizon_df = df.dropna(subset=[target_col, ret_col] + FEATURES).copy()
        
        X = horizon_df[FEATURES]
        y = horizon_df[target_col]
        
        # Setup the rolling window (gap=126 ensures strict separation between train and test)
        tscv = TimeSeriesSplit(n_splits=5, gap=126)
        out_of_sample_preds = pd.Series(index=horizon_df.index, dtype=float)
        
        # 3. TRAIN THE ROLLING FOLDS
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            
            model = xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.05, 
                max_depth=4, 
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predict only on the unseen future
            out_of_sample_preds.iloc[test_idx] = model.predict_proba(X_test)[:, 1]
            
        # 4. INJECT PREDICTIONS
        # Map predictions back to the master dataframe
        master_wfo_df.loc[horizon_df.index, confidence_col] = out_of_sample_preds

    # 5. DYNAMIC EXPORT PATHS
    # Drop rows without predictions and save unified signals
    master_wfo_df.dropna(subset=[f'AI_Confidence_{h}' for h in HORIZONS], inplace=True)
    master_wfo_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ WFO Out-Of-Sample Signals saved to: {OUTPUT_CSV}")
    print("🎉 ALL WALK-FORWARD HORIZONS COMPLETED!")

if __name__ == "__main__":
    main()