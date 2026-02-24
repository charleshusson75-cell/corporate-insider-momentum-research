"""
General description:
Unified Visualization Engine.
Reads the optimized equity curve CSVs outputted by the backtesters (both Static and WFO).
Downloads the S&P 500 baseline from Yahoo Finance and plots the comparative equity curves.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# --- CONFIGURATION & HYPERPARAMETERS ---
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTING_CAPITAL = 100000.00
HORIZONS = ['1M', '2M', '6M']

def plot_curve(csv_path, graph_path, horizon, is_wfo):
    if not os.path.exists(csv_path): return
    
    eq_df = pd.read_csv(csv_path, parse_dates=['Date']).set_index('Date')
    
    # Fetch SPY benchmark just for this timeframe
    spy = yf.download('SPY', start=eq_df.index.min().strftime('%Y-%m-%d'), end=eq_df.index.max().strftime('%Y-%m-%d'), progress=False)
    spy['SPY_Normalized'] = (spy['Close'] / spy['Close'].iloc[0]) * STARTING_CAPITAL
    
    trading_days = np.busday_count(eq_df.index.min().date(), eq_df.index.max().date())
    cagr = ((eq_df['Strategy_Equity'].iloc[-1] / STARTING_CAPITAL) ** (252 / trading_days)) - 1
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    color = 'purple' if is_wfo else 'blue'
    title_prefix = "Walk-Forward" if is_wfo else "Static"
    
    ax1.plot(eq_df.index, eq_df['Strategy_Equity'], label=f"{title_prefix} AI Strategy ({cagr*100:.1f}% CAGR)", color=color, linewidth=2)
    ax1.plot(spy.index, spy['SPY_Normalized'], label="S&P 500 Baseline", color='gray', linestyle='--')
    ax1.set_title(f"{title_prefix} AI Insider Momentum vs S&P 500 ({horizon} Horizon)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.fill_between(eq_df.index, eq_df['Pct_Invested'] * 100, color=color, alpha=0.2)
    ax2.plot(eq_df.index, eq_df['Pct_Invested'] * 100, color=color, linewidth=1)
    ax2.set_ylabel("% Capital Invested")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close(fig) 
    print(f"✅ Generated {title_prefix} Graph: {os.path.basename(graph_path)}")

def main():
    print("📉 Booting Unified Visualization Engine...")
    
    for horizon in HORIZONS:
        # 1. Plot Static Backtests if they exist
        static_csv = os.path.join(BASE_DIR, "Data", f"optimal_equity_curve_{horizon}.csv")
        static_png = os.path.join(BASE_DIR, "Data", f"strategy_vs_spy_{horizon}.png")
        plot_curve(static_csv, static_png, horizon, is_wfo=False)
        
        # 2. Plot WFO Backtests if they exist
        wfo_csv = os.path.join(BASE_DIR, "Data", f"optimal_equity_curve_{horizon}_wfo.csv")
        wfo_png = os.path.join(BASE_DIR, "Data", f"strategy_vs_spy_{horizon}_wfo.png")
        plot_curve(wfo_csv, wfo_png, horizon, is_wfo=True)
        
    print("🎉 ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()