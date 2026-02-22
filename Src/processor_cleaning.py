"""
General description: 
Ingests the raw, messy SEC master bulk dataset and performs rigorous ETL (Extract, 
Transform, Load). Removes exact duplicates, filters for pure open-market purchases, 
drops "fat-finger" future dates, and enforces a minimum dollar value threshold to 
remove noise. Also calculates the percentage by which the insider increased their portfolio.

Args:
    INPUT_FILE (str): Filepath to the raw, uncleaned SEC bulk CSV.

Returns:
    OUTPUT_FILE (csv): Saves the cleaned, chronological, ML-ready SEC trades.
"""

import pandas as pd
import os

# ==========================================
# --- CONFIGURATION & HYPERPARAMETERS ---
# ==========================================

# --- FILE PATHS ---
INPUT_FILE = "Data/sec_master_bulk_data.csv" 
OUTPUT_FILE = "Data/clean_insider_signals.csv"  # Fixed path to save inside Data/

# --- TRADING FILTERS ---
MIN_VALUE = 20000           # Ignore trades smaller than this dollar amount
TARGET_CODE = 'P'           # SEC Code 'P' = Open Market Purchase. Reject all others.

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Could not find {INPUT_FILE}.")
        return

    print("Load Raw Data...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"   📉 Original Raw Rows: {len(df)}")

    # 1. REMOVE EXACT DUPLICATES (User Request)
    df.drop_duplicates(inplace=True)

    # 2. DROP MISSING CRITICAL DATA (Sanity Check)
    df.dropna(subset=['Ticker', 'Trade_Date', 'Price', 'Shares'], inplace=True)
    
    # 3. THE ONE GOLDEN RULE: Open Market Buys Only
    if 'Code' in df.columns:
        df = df[df['Code'] == TARGET_CODE].copy()
    else:
        print("   ⚠️ WARNING: 'Code' column missing! Falling back to 'Type'.")
        df = df[df['Type'] == 'Buy'].copy()

    # 4. FILTER: Minimum Value Constraint
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df[df['Value'] >= MIN_VALUE].copy()
    
    # 5. SANITY CHECK: Dates & Time Travelers
    df['Trade_Date'] = pd.to_datetime(df['Trade_Date'], errors='coerce')
    today = pd.Timestamp.today()
    df = df[df['Trade_Date'] <= today] # Drop "fat finger" future dates
    
    # 6. ENHANCE: Calculate Portfolio %
    df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce')
    df['Shares_Owned_After'] = pd.to_numeric(df['Shares_Owned_After'], errors='coerce')
    
    def calc_portfolio_pct(row):
        bought = row['Shares']
        owned_after = row['Shares_Owned_After']
        if pd.isna(owned_after) or owned_after <= 0 or pd.isna(bought):
            return 0.0
        pct = (bought / owned_after) * 100
        return min(round(pct, 2), 100.0)

    df['Portfolio_Pct'] = df.apply(calc_portfolio_pct, axis=1)

    # 7. SORT: Chronological Order (Newest Trades at the Top)
    df.sort_values('Trade_Date', ascending=False, inplace=True)

    # 8. SAVE
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n🎉 SUCCESS! Data Cleaned (Lightweight Version).")
    print(f"   💾 Saved to: {OUTPUT_FILE}")
    print(f"   🔥 Final Surviving Trades: {len(df)}")

if __name__ == "__main__":
    main()