"""
General description: 
Trains the XGBoost classifier, applies a Confidence Threshold (e.g., 65%), 
and exports the AI's official "BUY" signals to a CSV for portfolio backtesting.
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

# --- FILE PATHS ---
INPUT_FILE = "Data/ml_master_matrix.csv"
OUTPUT_FILE = "Data/ai_buy_signals.csv"
OUTPUT_MODEL = "Models/xgboost_production_v1.pkl"

# --- ML TARGETS & THRESHOLDS ---
TARGET_COL = 'Target_1M'  
CONFIDENCE_THRESHOLD = 0.00  

# --- FEATURE SPACE (The "Lego Blocks") ---
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
    print(f"🤖 Booting AI Trading Engine (Confidence Target: {CONFIDENCE_THRESHOLD*100}%)...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Check your path!")
        return

    # 1. LOAD & PREP DATA
    df = pd.read_csv(INPUT_FILE, parse_dates=['Trade_Date'])
    df.dropna(subset=[TARGET_COL, 'Ret_1M'] + FEATURES, inplace=True)
    
    # Sort chronologically (Oldest to Newest) to prevent "Look-Ahead Bias"
    df.sort_values('Trade_Date', ascending=True, inplace=True)
    
    # 2. THE TIME SPLIT (Hard Date Split for 2020 Stress Test)
    SPLIT_DATE = '2020-01-01'
    
    train_df = df[df['Trade_Date'] < SPLIT_DATE].copy()
    test_df = df[df['Trade_Date'] >= SPLIT_DATE].copy()
    
    X_train, y_train = train_df[FEATURES], train_df[TARGET_COL]
    X_test = test_df[FEATURES]
    
    # 3. TRAIN THE MODEL
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 4. PREDICT PROBABILITIES & FILTER
    # Extract the raw probability (0.0 to 1.0) that the trade is a winner
    test_df['AI_Confidence'] = model.predict_proba(X_test)[:, 1]
    
    # Filter only High-Conviction BUYS
    buy_signals = test_df[test_df['AI_Confidence'] >= CONFIDENCE_THRESHOLD]
    
    # 5. EXPORT FOR BACKTESTING
    buy_signals.to_csv(OUTPUT_FILE, index=False)
    
    print("\n====================================")
    print("📈 AI SIGNAL GENERATION COMPLETE")
    print("====================================")
    print(f"✅ Generated {len(buy_signals)} high-confidence BUY signals.")
    print(f"💾 Saved to: {OUTPUT_FILE}")
    print("====================================\n")

    # Save the trained model for live production
    print("💾 Saving the trained model for live production...")
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    
    # Replace 'model' with whatever variable name your XGBoost classifier uses!
    joblib.dump(model, OUTPUT_MODEL) 
    print(f"✅ SUCCESS: Brain frozen and saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    main()