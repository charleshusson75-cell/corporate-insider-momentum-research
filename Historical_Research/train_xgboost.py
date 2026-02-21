"""
General description: 
Trains three separate XGBoost classifiers for 1-Month, 2-Month, and 6-Month horizons.
Exports the AI's official "BUY" signals and trained models for each specific timeframe,
using the exact original feature set.
"""

import pandas as pd
import xgboost as xgb
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# --- CONFIGURATION & HYPERPARAMETERS ---
# ==========================================

INPUT_FILE = "Data/ml_master_matrix.csv"
CONFIDENCE_THRESHOLD = 0.00  

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
    print(f"🤖 Booting Multi-Horizon AI Trading Engine...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Check your path!")
        return

    # 1. LOAD & PREP MASTER DATA
    print(f"📦 Loading Master Matrix...")
    df = pd.read_csv(INPUT_FILE, parse_dates=['Trade_Date'])
    
    # Sort chronologically to prevent "Look-Ahead Bias"
    df.sort_values('Trade_Date', ascending=True, inplace=True)
    SPLIT_DATE = '2020-01-01'

    # 2. LOOP THROUGH EACH HORIZON
    for horizon in HORIZONS:
        print(f"\n====================================")
        print(f"⏳ TRAINING MODEL HORIZON: {horizon}")
        print(f"====================================")
        
        target_col = f'Target_{horizon}'
        ret_col = f'Ret_{horizon}'
        
        # Check if the columns actually exist in the DB
        if target_col not in df.columns or ret_col not in df.columns:
            print(f"⚠️ Skipping {horizon}: Targets not found in database.")
            continue
            
        # Isolate data for this specific horizon and drop NaNs
        horizon_df = df.dropna(subset=[target_col, ret_col] + FEATURES).copy()
        
        train_df = horizon_df[horizon_df['Trade_Date'] < SPLIT_DATE].copy()
        test_df = horizon_df[horizon_df['Trade_Date'] >= SPLIT_DATE].copy()
        
        X_train, y_train = train_df[FEATURES], train_df[target_col]
        X_test = test_df[FEATURES]
        
        # 3. TRAIN THE SPECIFIC BRAIN
        model = xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=4, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 4. PREDICT & FILTER
        test_df['AI_Confidence'] = model.predict_proba(X_test)[:, 1]
        buy_signals = test_df[test_df['AI_Confidence'] >= CONFIDENCE_THRESHOLD]
        
        # 5. DYNAMIC EXPORT PATHS
        output_csv = f"Data/ai_buy_signals_{horizon}.csv"
        output_model = f"Models/xgboost_production_{horizon}.pkl"
        
        os.makedirs(os.path.dirname(output_model), exist_ok=True)
        
        # Save Signals
        buy_signals.to_csv(output_csv, index=False)
        print(f"✅ Generated {len(buy_signals)} high-confidence BUY signals.")
        print(f"💾 Saved Signals to: {output_csv}")
        
        # Save Model
        joblib.dump(model, output_model) 
        print(f"💾 Saved Brain to: {output_model}")

    print("\n🎉 ALL HORIZONS TRAINED SUCCESSFULLY!")

if __name__ == "__main__":
    main()