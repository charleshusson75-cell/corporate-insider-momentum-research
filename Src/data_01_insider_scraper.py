"""
General description: 
Automates the bulk extraction of SEC Form 4 insider trading data directly from 
the SEC EDGAR database. Downloads quarterly ZIP files, extracts reporting owner 
and transaction data in-memory, dynamically maps shifting SEC column headers 
using fuzzy matching, and compiles a raw master dataset.

Args:
    None (Pulls directly from SEC EDGAR quarterly ZIP URLs based on config).

Returns:
    OUTPUT_FILE (csv): Saves the raw, uncleaned master SEC dataset to disk.
"""

import requests
import pandas as pd
import io
import zipfile
import os
import time

# ==========================================
# --- CONFIGURATION & HYPERPARAMETERS ---
# ==========================================

# --- FILE PATHS ---
OUTPUT_DIR = "Data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sec_master_bulk_data.csv")

# --- SEC API SETTINGS ---
YEARS = range(2010, 2026)   # Target years to scrape
QUARTERS = range(1, 5)      # Target quarters (1 through 4)
HEADERS = {
    "User-Agent": "StudentProject charleshusson75@gmail.com",
    "Host": "www.sec.gov"
}

# --- SCRAPING THRESHOLDS ---
TARGET_CODES = ['A', 'P', 'D']  # Only keep Grants (A), Open Market Buys (P), and Returns (D)
MAX_RETRIES = 3                 # How many times to retry a failed SEC download
TIMEOUT = 60                    # Max seconds to wait for SEC server response

# --- BULLETPROOF COLUMN MAPPING ---
COLUMN_MAP = {
    'ACCESSION_NUMBER': ['ACCESSION_NUMBER', 'ACCESSION_NUM'],
    'FILING_DATE': ['FILING_DATE', 'DATE_ACCEPTED', 'DATE'],
    'TICKER': ['ISSUERTRADINGSYMBOL', 'ISSUER_TRADING_SYMBOL', 'TICKER', 'SYMBOL'],
    'OWNER_NAME': ['RPTOWNERNAME', 'REPORTING_OWNER_NAME', 'OWNER_NAME', 'NAME'],
    'IS_DIRECTOR': ['IS_DIRECTOR', 'DIRECTOR'],
    'IS_OFFICER': ['IS_OFFICER', 'OFFICER'],
    'OFFICER_TITLE': ['OFFICER_TITLE', 'TITLE'],
    'TRANS_DATE': ['TRANS_DATE', 'TRANSACTION_DATE', 'DATE'],
    'TRANS_SHARES': ['TRANS_SHARES', 'SHARES', 'TRANSACTION_SHARES', 'SHRS'],
    'TRANS_PRICE': ['TRANS_PRICEPPS', 'PRICE_PER_SHARE', 'TRANS_PRICE_PER_SHARE', 'TRANS_PRICE', 'PRICE', 'PRICEPPS', 'PPS', 'TRANS_PRIC'],
    'TRANS_CODE': ['TRANS_ACQUIRED_DISP_CODE', 'ACQUIRED_DISP_CODE', 'TRANS_CODE', 'CODE', 'ACQ_DISP_CODE'],
    'SHARES_OWNED_AFTER': ['SHRS_OWND_FOLWNG_TRANS', 'SHARES_OWNED_FOLLOWING_TRANSACTION', 'SHARES_OWNED_AFTER', 'SHARES_OWNED', 'OWND_FOLWNG'],
    'OWNERSHIP_TYPE': ['DIRECT_INDIRECT_OWNERSHIP', 'DIRECT_OR_INDIRECT_OWNERSHIP', 'OWNERSHIP_TYPE', 'OWNERSHIP'],
    'SIGNATURE_NAME': ['SIGNATURENAME', 'SIGNATURE_NAME', 'SIGNATURE']
}

def find_file_smart(z, keyword):
    keyword = keyword.lower()
    for filename in z.namelist():
        if keyword in os.path.basename(filename).lower():
            return filename
    return None

def load_with_adaptive_columns(zip_file, filename, expected_fields):
    """Reads header, fuzzy matches names, and prints debug info if missing."""
    with zip_file.open(filename) as f:
        header_df = pd.read_csv(f, sep="\t", nrows=0, on_bad_lines='skip')
        raw_cols = list(header_df.columns)
        clean_to_raw = {str(c).strip().upper(): c for c in raw_cols}
    
    use_cols = []
    rename_map = {}
    
    for standard_name in expected_fields:
        possible_names = COLUMN_MAP.get(standard_name, [])
        found = False
        
        # 1. Exact Match
        for name in possible_names:
            if name in clean_to_raw:
                raw_name = clean_to_raw[name]
                use_cols.append(raw_name)
                rename_map[raw_name] = standard_name
                found = True
                break
        
        # 2. Fuzzy Match
        if not found:
            for raw_c in raw_cols:
                raw_upper = str(raw_c).upper()
                if any(kw in raw_upper for kw in possible_names if len(kw) > 3): 
                    use_cols.append(raw_c)
                    rename_map[raw_c] = standard_name
                    found = True
                    break
        
        # 3. Critical Failure check
        if not found and standard_name == 'ACCESSION_NUMBER':
            return None, f"Missing critical ID column. Found: {list(raw_cols)}"

    with zip_file.open(filename) as f:
        df = pd.read_csv(f, sep="\t", usecols=use_cols, on_bad_lines='skip', low_memory=False)
    
    df.rename(columns=rename_map, inplace=True)
    return df, None

def download_and_process_quarter(year, quarter):
    url = f"https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{year}q{quarter}_form345.zip"
    print(f"📥 Downloading {year} Q{quarter} ...")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if response.status_code == 404:
            print(f"   ⚠️ No data found for {year} Q{quarter}")
            return None
        
        try:
            z = zipfile.ZipFile(io.BytesIO(response.content))
        except zipfile.BadZipFile:
            print(f"   ❌ Error: Bad Zip File")
            return None
        
        # FIND FILES
        sub_file = find_file_smart(z, "SUBMISSION")
        own_file = find_file_smart(z, "REPORTINGOWNER")
        trans_file = find_file_smart(z, "NONDERIV_TRANS")
        sig_file = find_file_smart(z, "SIGNATURE")
        
        if not (sub_file and own_file and trans_file):
            print(f"   ❌ Missing primary files in zip.")
            return None

        # LOAD DATA
        df_sub, err = load_with_adaptive_columns(z, sub_file, ['ACCESSION_NUMBER', 'FILING_DATE', 'TICKER'])
        if df_sub is None: return None

        df_own, err = load_with_adaptive_columns(z, own_file, ['ACCESSION_NUMBER', 'OWNER_NAME', 'IS_DIRECTOR', 'IS_OFFICER', 'OFFICER_TITLE'])
        if df_own is None: return None
        df_own = df_own.drop_duplicates(subset=['ACCESSION_NUMBER'])

        df_trans, err = load_with_adaptive_columns(z, trans_file, ['ACCESSION_NUMBER', 'TRANS_DATE', 'TRANS_SHARES', 'TRANS_PRICE', 'TRANS_CODE', 'SHARES_OWNED_AFTER', 'OWNERSHIP_TYPE'])
        if df_trans is None: return None

        # MERGE CORE DATA
        merged = pd.merge(df_trans, df_sub, on="ACCESSION_NUMBER", how="left")
        merged = pd.merge(merged, df_own, on="ACCESSION_NUMBER", how="left")

        # LOAD & MERGE SIGNATURE DATA
        if sig_file:
            df_sig, err = load_with_adaptive_columns(z, sig_file, ['ACCESSION_NUMBER', 'SIGNATURE_NAME'])
            if df_sig is not None:  # <-- Changed this line!
                df_sig = df_sig.drop_duplicates(subset=['ACCESSION_NUMBER'])
                merged = pd.merge(merged, df_sig, on="ACCESSION_NUMBER", how="left")
        
        if 'SIGNATURE_NAME' not in merged.columns:
            merged['SIGNATURE_NAME'] = "Unknown"

        # FILTER: Keep only recognized codes
        if 'TRANS_CODE' in merged.columns:
            merged = merged[merged['TRANS_CODE'].isin(TARGET_CODES)]
        
        # NUMERICS & VALUES
        if 'TRANS_PRICE' in merged.columns:
            merged['TRANS_PRICE'] = pd.to_numeric(merged['TRANS_PRICE'], errors='coerce')
        else:
            return None
            
        if 'TRANS_SHARES' in merged.columns:
            merged['TRANS_SHARES'] = pd.to_numeric(merged['TRANS_SHARES'], errors='coerce')
        
        merged = merged[merged['TRANS_PRICE'] > 0]
        merged['Value'] = round(merged['TRANS_SHARES'] * merged['TRANS_PRICE'], 2)

        # CHECK IF EXECUTOR IS DIFFERENT
        def check_diff_executor(row):
            owner = str(row.get('OWNER_NAME', '')).strip().upper()
            executor = str(row.get('SIGNATURE_NAME', '')).strip().upper()
            
            if executor == 'NAN' or executor == 'UNKNOWN' or executor == '':
                return "Unknown"
            
            if owner in executor or executor in owner:
                return False
            return True

        merged['Is_Different_Executor'] = merged.apply(check_diff_executor, axis=1)

        # RENAME TO CLEAN COLUMNS
        merged.rename(columns={
            'TRANS_DATE': 'Trade_Date',
            'FILING_DATE': 'Report_Date',
            'TICKER': 'Ticker',
            'OWNER_NAME': 'Insider Name',
            'SIGNATURE_NAME': 'Executor_Name',
            'OFFICER_TITLE': 'Raw_Role',           
            'TRANS_PRICE': 'Price',
            'TRANS_SHARES': 'Shares',
            'TRANS_CODE': 'Code',
            'SHARES_OWNED_AFTER': 'Shares_Owned_After',
            'OWNERSHIP_TYPE': 'Ownership_Type'
        }, inplace=True)

        if 'Code' in merged.columns:
            merged['Type'] = merged['Code'].apply(lambda x: 'Buy' if x in ['P', 'A'] else 'Sell')
        else:
            merged['Type'] = 'Unknown'
            
        if 'Ownership_Type' in merged.columns:
            merged['Ownership_Type'] = merged['Ownership_Type'].apply(
                lambda x: 'Direct' if str(x).upper() == 'D' else ('Indirect' if str(x).upper() == 'I' else 'Unknown')
            )
        
        def make_role(row):
            roles = []
            if str(row.get('IS_DIRECTOR')) == '1': roles.append('Director')
            if str(row.get('IS_OFFICER')) == '1': roles.append(str(row.get('Raw_Role')))
            return ", ".join(roles) if roles else "Other"
        
        merged['Role'] = merged.apply(make_role, axis=1)

        # FINAL OUTPUT COLUMNS
        final_cols = ['Trade_Date', 'Report_Date', 'Ticker', 'Insider Name', 'Executor_Name', 'Is_Different_Executor', 'Raw_Role', 'Code', 'Type', 'Ownership_Type', 'Price', 'Shares', 'Value', 'Shares_Owned_After']
        final_cols = [c for c in final_cols if c in merged.columns]
        
        final_df = merged[final_cols]
        print(f"   ✅ Processed: {len(final_df)} trades")
        return final_df

    except Exception as e:
        print(f"   ❌ Error processing {year} Q{quarter}: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_data = []
    print("🚀 Starting SEC Bulk Pipeline (2010-2025)...")
    
    for year in YEARS:
        for q in QUARTERS:
            for attempt in range(MAX_RETRIES):
                try:
                    df = download_and_process_quarter(year, q)
                    if df is not None:
                        all_data.append(df)
                    break 
                except Exception as e:
                    print(f"   ⚠️ Retry ({attempt+1}/{MAX_RETRIES})...")
                    time.sleep(2)
            time.sleep(0.5) 
    
    if all_data:
        print("💾 Merging all years...")
        master_df = pd.concat(all_data, ignore_index=True)
        master_df.sort_values('Trade_Date', ascending=False, inplace=True)
        master_df.to_csv(OUTPUT_FILE, index=False)
        print(f"🎉 SUCCESS! Database saved to: {OUTPUT_FILE}")
        print(f"   Total Trades: {len(master_df)}")
    else:
        print("❌ No data collected.")

if __name__ == "__main__":
    main()