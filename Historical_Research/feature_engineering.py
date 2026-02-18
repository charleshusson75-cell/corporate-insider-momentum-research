"""
General description: 
Merges raw SEC Form 4 insider trades with a local historical price database. 
Applies strict liquidity filters, engineers custom quantitative features 
(e.g., Weighted Consensus, Liquidity Absorption), and calculates forward-looking 
binary targets (1-month, 2-month, 6-month) to create a final, ML-ready matrix.

Args:
    TRADES_FILE (str): Filepath to the cleaned SEC insider trades CSV.
    PRICES_FILE (str): Filepath to the local historical OHLCV price database.

Returns:
    OUTPUT_FILE (csv): Saves the final numerical feature matrix to disk, 
                       ready for XGBoost ingestion.
"""

import pandas as pd
import numpy as np
import os
import warnings

# Suppress annoying pandas warnings
warnings.filterwarnings('ignore')

# ==========================================
# --- CONFIGURATION & HYPERPARAMETERS ---
# ==========================================

# --- FILE PATHS ---
TRADES_FILE = "Data/clean_insider_signals.csv"
PRICES_FILE = "Data/historical_prices_db.csv"
OUTPUT_FILE = "Data/ml_master_matrix.csv"

# --- 1. THE BOUNCER (Liquidity Filters) ---
MIN_PRICE = 2.0             # Penny stock ban: Stock must cost at least $2.00
MIN_VOLUME = 50000          # Liquidity ban: Must trade at least 50k shares a day

# --- 2. TIME HORIZONS (Trading Days for Answer Keys) ---
TARGET_1M = 21              # Days to hold for 1-Month target
TARGET_2M = 42              # Days to hold for 2-Month target
TARGET_6M = 126             # Days to hold for 6-Month target

# --- 3. THE WOLFPACK (Consensus Scoring) ---
CONSENSUS_WINDOW = '7D'     # How many days to look back for other insiders buying
WEIGHT_C_SUITE = 3          # Points for CEO/CFO buying
WEIGHT_VP = 2               # Points for President/COO buying
WEIGHT_DIRECTOR = 1         # Points for standard board members


def main():
    print("🚀 Starting Phase 2: Feature Engineering & Labeling")
    
    if not os.path.exists(TRADES_FILE) or not os.path.exists(PRICES_FILE):
        print("❌ Error: Could not find the input CSV files. Check your paths!")
        return

    # 1. LOAD DATA WITH M1 MEMORY PROTECTION
    print("📦 Loading datasets (Optimizing RAM footprint)...")
    
    # Downcasting: Forcing Pandas to use smaller data types cuts RAM usage massively
    dtypes = {
        'Ticker': 'category', 
        'Close': 'float32', 
        'ATR_14': 'float32', 
        'Volume': 'float32'
    }
    
    # The '# type: ignore' tells Pylance to stop throwing a false-positive error here
    prices_df = pd.read_csv(PRICES_FILE, dtype=dtypes, parse_dates=['Trade_Date'])  # type: ignore
    trades_df = pd.read_csv(TRADES_FILE, parse_dates=['Trade_Date', 'Report_Date'])
    
    # 2. CALCULATE THE "ANSWER KEYS" (Future Returns)
    print(f"🔮 Calculating {TARGET_1M}, {TARGET_2M}, and {TARGET_6M}-Day Future Targets...")
    
    prices_df.sort_values(by=['Ticker', 'Trade_Date'], inplace=True)
    
    # Shift prices forward to look into the future
    prices_df['Close_1M'] = prices_df.groupby('Ticker')['Close'].shift(-TARGET_1M)
    prices_df['Close_2M'] = prices_df.groupby('Ticker')['Close'].shift(-TARGET_2M)
    prices_df['Close_6M'] = prices_df.groupby('Ticker')['Close'].shift(-TARGET_6M)
    
    # Calculate % Returns
    prices_df['Ret_1M'] = (prices_df['Close_1M'] - prices_df['Close']) / prices_df['Close']
    prices_df['Ret_2M'] = (prices_df['Close_2M'] - prices_df['Close']) / prices_df['Close']
    prices_df['Ret_6M'] = (prices_df['Close_6M'] - prices_df['Close']) / prices_df['Close']
    
    # Binary Answer Keys: 1 if profitable, 0 if loss
    prices_df['Target_1M'] = (prices_df['Ret_1M'] > 0).astype(float)
    prices_df['Target_2M'] = (prices_df['Ret_2M'] > 0).astype(float)
    prices_df['Target_6M'] = (prices_df['Ret_6M'] > 0).astype(float)

    print("🗓️ Aligning Date formats...")
    # Force both to exact datetime and strip the hours/minutes (normalize)
    trades_df['Trade_Date'] = pd.to_datetime(trades_df['Trade_Date'], errors='coerce').dt.normalize()
    prices_df['Trade_Date'] = pd.to_datetime(prices_df['Trade_Date'], errors='coerce').dt.normalize()
    
    # Drop any rows where the date was completely corrupted
    trades_df.dropna(subset=['Trade_Date'], inplace=True)
    prices_df.dropna(subset=['Trade_Date'], inplace=True)
    
    # ---------------------------------------------------------
    # --- 🚨 SQUASH THE "DRIP-FEED" BUG (Daily Aggregation) ---
    # ---------------------------------------------------------
    print("🗜️ Aggregating intraday micro-trades...")
    
    # We group by the exact Person, on the exact Day, for the exact Stock
    agg_funcs = {
        'Price': 'mean',          # Calculate their average fill price
        'Shares': 'sum',          # Add up all shares bought that day
        'Value': 'sum',           # Add up total dollars spent that day
        'Portfolio_Pct': 'max',   # Keep their max portfolio increase
        'Raw_Role': 'first'       # Keep their job title
    }
    
    # Keep any other existing columns intact by taking the first value
    for col in trades_df.columns:
        if col not in ['Trade_Date', 'Ticker', 'Insider Name'] and col not in agg_funcs:
            agg_funcs[col] = 'first'
            
    trades_df = trades_df.groupby(['Trade_Date', 'Ticker', 'Insider Name'], as_index=False).agg(agg_funcs)
    print(f"   📉 Trades compressed to {len(trades_df)} unique daily decisions.")
    # ---------------------------------------------------------

    # 3. MERGE THE DATASETS
    print("🔗 Merging SEC Trades with Market Data...")
    df = pd.merge(trades_df, prices_df, on=['Ticker', 'Trade_Date'], how='inner')
    
    # 4. THE BOUNCER (Filter Garbage)
    print(f"🗑️ Filtering out stocks under ${MIN_PRICE} and volume under {MIN_VOLUME}...")
    original_len = len(df)
    df = df[(df['Close'] >= MIN_PRICE) & (df['Volume'] >= MIN_VOLUME)].copy()
    print(f"   Deleted {original_len - len(df)} garbage/untradable rows.")

    # 5. BUILD THE LEGO BLOCKS (Features)
    print("🏗️ Engineering Machine Learning Features...")
    
    # Feature A: Liquidity Absorption
    df['Daily_Dollar_Volume'] = df['Close'] * df['Volume']
    df['Pct_Volume_Absorbed'] = df['Value'] / df['Daily_Dollar_Volume']
    
    # Feature B: The Wolfpack (Weighted Consensus)
    def score_role(role):
        r = str(role).upper()
        if 'CEO' in r or 'CHIEF EXECUTIVE' in r: return WEIGHT_C_SUITE
        if 'CFO' in r or 'FINANCIAL' in r or 'ACCOUNTING' in r: return WEIGHT_C_SUITE
        if 'CHIEF' in r or 'PRESIDENT' in r or 'COO' in r: return WEIGHT_VP
        return WEIGHT_DIRECTOR 
        
    df['Role_Weight'] = df['Raw_Role'].apply(score_role)
    
    # Calculate rolling cluster weight per ticker
    df.sort_values(by=['Ticker', 'Trade_Date'], inplace=True)
    df.set_index('Trade_Date', inplace=True)
    
    df['Consensus_Score'] = df.groupby('Ticker')['Role_Weight'].transform(lambda x: x.rolling(CONSENSUS_WINDOW).sum())
    df.reset_index(inplace=True)
    
    # Clean up Phase 1 features
    df['Portfolio_Pct'] = df['Portfolio_Pct'].fillna(0)
    
    # 6. EXPORT THE FINAL MATRIX
    final_features = [
        'Trade_Date', 'Ticker', 'Insider Name', 'Raw_Role', 'Role_Weight', 
        'Consensus_Score', 'Portfolio_Pct', 'Pct_Volume_Absorbed', 'Value', 
        'Close', 'Volume', 'ATR_14', 
        'Ret_1M', 'Ret_2M', 'Ret_6M',
        'Target_1M', 'Target_2M', 'Target_6M'
    ]
    
    # Ensure all required columns exist before filtering
    available_cols = [col for col in final_features if col in df.columns]
    ml_matrix = df[available_cols].copy()
    
    # Drop rows where we don't have 1-month future targets yet
    if 'Target_1M' in ml_matrix.columns:
        ml_matrix.dropna(subset=['Target_1M'], inplace=True)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # --- CLEAN FORMATTING FOR HUMAN READABILITY ---
    # Round standard dollar/percentage amounts to 2 decimals
    ml_matrix['Close'] = ml_matrix['Close'].round(2)
    ml_matrix['Value'] = ml_matrix['Value'].round(2)
    ml_matrix['Portfolio_Pct'] = ml_matrix['Portfolio_Pct'].round(2)
    ml_matrix['ATR_14'] = ml_matrix['ATR_14'].round(2)
    ml_matrix['Ret_1M'] = ml_matrix['Ret_1M'].round(2)
    ml_matrix['Ret_2M'] = ml_matrix['Ret_2M'].round(2)
    ml_matrix['Ret_6M'] = ml_matrix['Ret_6M'].round(2)
    
    # Convert Volume to clean whole integers
    ml_matrix['Volume'] = ml_matrix['Volume'].fillna(0).astype(int)
    
    # Liquidity absorption is usually tiny (e.g., 0.0001%), so we give it 6 decimals
    ml_matrix['Pct_Volume_Absorbed'] = ml_matrix['Pct_Volume_Absorbed'].round(6)
    
    # Save (and prevent Pandas from bringing back scientific notation)
    ml_matrix.to_csv(OUTPUT_FILE, index=False, float_format='%.6f')
    
    print(f"\n🎉 SUCCESS! Matrix built and saved to: {OUTPUT_FILE}")
    print(f"   🤖 Final Tradable Rows for XGBoost: {len(ml_matrix):,}")

if __name__ == "__main__":
    main()