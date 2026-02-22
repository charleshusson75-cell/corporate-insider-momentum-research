"""
General description: 
Reads a list of unique stock tickers from the cleaned SEC dataset and downloads 
their maximum available historical daily price data from Yahoo Finance. Calculates 
the Average True Range (ATR) for volatility measurement and compiles everything 
into a single, highly compressed offline database.

Args:
    INPUT_FILE (str): Filepath to the cleaned SEC insider trades CSV.

Returns:
    OUTPUT_FILE (csv): Saves the master historical price database to disk.
"""

import pandas as pd
import yfinance as yf
import numpy as np
import os
from tqdm import tqdm
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

# ==========================================
# --- CONFIGURATION & HYPERPARAMETERS ---
# ==========================================

# --- FILE PATHS ---
INPUT_FILE = "Data/clean_insider_signals.csv"
OUTPUT_FILE = "Data/historical_prices_db.csv"

# --- TECHNICAL INDICATORS ---
ATR_WINDOW = 14             # Number of days to calculate the Average True Range

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Run the cleaner first!")
        return

    print("Load Clean SEC Data to extract unique tickers...")
    df = pd.read_csv(INPUT_FILE)
    
    unique_tickers = df['Ticker'].dropna().unique()
    print(f"🔍 Found {len(unique_tickers)} unique companies to fetch market data for...")

    db_chunks = []

    for ticker in tqdm(unique_tickers, desc="Building Local Database"):
        try:
            # Download history (we need High, Low, Close for ATR)
            stock = yf.Ticker(ticker)
            hist = stock.history(period="max")
            
            if hist.empty:
                continue
                
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            
            # --- CALCULATE ATR ---
            hist['Prev_Close'] = hist['Close'].shift(1)
            hist['TR1'] = hist['High'] - hist['Low']
            hist['TR2'] = abs(hist['High'] - hist['Prev_Close'])
            hist['TR3'] = abs(hist['Low'] - hist['Prev_Close'])
            
            # True Range is the maximum of those three values
            hist['True_Range'] = hist[['TR1', 'TR2', 'TR3']].max(axis=1)
            
            # ATR is the rolling average of the True Range
            hist['ATR_14'] = hist['True_Range'].rolling(window=ATR_WINDOW).mean()
            
            # --- FILTER THE DATABASE ---
            clean_hist = hist[['Close', 'Volume', 'ATR_14']].copy()
            clean_hist['Ticker'] = ticker
            clean_hist.reset_index(inplace=True)
            clean_hist.rename(columns={'Date': 'Trade_Date'}, inplace=True)
            
            # Drop the initial days where ATR couldn't be calculated yet
            clean_hist.dropna(subset=['ATR_14'], inplace=True)
            
            db_chunks.append(clean_hist)
            
        except Exception as e:
            # Skip delisted or dead tickers
            pass

    if not db_chunks:
        print("❌ Error: Failed to download any market data.")
        return

    print("💾 Merging database chunks...")
    master_db = pd.concat(db_chunks, ignore_index=True)
    master_db.sort_values(by=['Ticker', 'Trade_Date'], ascending=[True, True], inplace=True)
    master_db.to_csv(OUTPUT_FILE, index=False)
    
    print("\n🎉 SUCCESS! Quant Data Warehouse Built.")
    print(f"   💾 Saved to: {OUTPUT_FILE}")
    print(f"   📊 Total Daily Price Records: {len(master_db):,}")

if __name__ == "__main__":
    main()