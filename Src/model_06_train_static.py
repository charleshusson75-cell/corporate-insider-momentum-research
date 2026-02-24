"""
General description: 
Trains three separate XGBoost classifiers for 1-Month, 2-Month, and 6-Month horizons.
Fuses the AI's official "BUY" signals into a single master CSV file, and exports 
trained models for each specific timeframe using the exact original feature set.
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
OUTPUT_CSV = "Data/ai_buy_signals.csv"
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
    
    # Create the master test dataframe that will hold all fused columns
    master_test_df = df[df['Trade_Date'] >= SPLIT_DATE].copy()

    # 2. LOOP THROUGH EACH HORIZON
    for horizon in HORIZONS:
        print(f"\n====================================")
        print(f"⏳ TRAINING MODEL HORIZON: {horizon}")
        print(f"====================================")
        
        target_col = f'Target_{horizon}'
        ret_col = f'Ret_{horizon}'
        confidence_col = f'AI_Confidence_{horizon}' # New dynamic column name
        
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
        
        # 4. PREDICT 
        # Inject predictions directly into the unified master_test_df using index matching
        master_test_df.loc[test_df.index, confidence_col] = model.predict_proba(X_test)[:, 1]
        
        # 5. DYNAMIC EXPORT PATHS
        output_model = f"Models/xgboost_production_{horizon}.pkl"
        os.makedirs(os.path.dirname(output_model), exist_ok=True)
        
        # Save Model
        joblib.dump(model, output_model) 
        print(f"💾 Saved Brain to: {output_model}")

    # Save Unified Signals at the very end
    master_test_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Fused Master Signals saved to: {OUTPUT_CSV}")
    print("🎉 ALL HORIZONS TRAINED SUCCESSFULLY!")

if __name__ == "__main__":
    main()