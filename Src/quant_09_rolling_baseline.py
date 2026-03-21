"""
quant_09_random_baseline.py
Monkey Dartboard Baseline (Multi-Horizon).
Randomly selects 3,300 trades from the 2020-2025 period and allocates a strict, 
fixed $5,000 to each trade. Outputs the equity curve CSV and a performance graph.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "Data", "ai_buy_signals.csv")

STARTING_CAPITAL = 100000.00
FIXED_BET_SIZE = 5000.00
NUM_TRADES_TO_TAKE = 3300

HORIZON_CONFIGS = {
    '1M': {'hold_days': 21,  'ret_col': 'Ret_1M'},
    '2M': {'hold_days': 42,  'ret_col': 'Ret_2M'},
    '6M': {'hold_days': 126, 'ret_col': 'Ret_6M'}
}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    # Load master data and filter for 2020-2025
    df_full = pd.read_csv(INPUT_FILE, parse_dates=['Trade_Date'])
    df_full = df_full[(df_full['Trade_Date'] >= '2020-01-01') & (df_full['Trade_Date'] <= '2025-12-31')]

    for horizon, config in HORIZON_CONFIGS.items():
        print(f"\n💼 Running Random 3,300 Trade Baseline for {horizon}...")
        
        output_csv = os.path.join(BASE_DIR, "Data", f"random_baseline_metrics_{horizon}.csv")
        output_png = os.path.join(BASE_DIR, "Data", f"random_baseline_graph_{horizon}.png")
        
        holding_period = config['hold_days']
        ret_col = config['ret_col']
        
        if ret_col not in df_full.columns: continue

        # 1. Isolate horizon and take 3300 random trades
        df_horizon = df_full.dropna(subset=[ret_col]).copy()
        n_samples = min(NUM_TRADES_TO_TAKE, len(df_horizon))
        if n_samples == 0: continue
        
        # random_state=42 ensures the "randomness" is the exact same every time you run it
        df_sampled = df_horizon.sample(n=n_samples, random_state=42)
        df_sampled.sort_values('Trade_Date', inplace=True)

        # 2. Setup the Timeline
        current_cash = STARTING_CAPITAL
        active_trades = []
        daily_equity_log = []
        winning_trades = 0

        # Create a continuous calendar to map equity daily (even on days with no trades)
        start_date = df_sampled['Trade_Date'].min()
        end_date = df_sampled['Trade_Date'].max() + pd.Timedelta(days=holding_period)
        all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        trading_days = len(all_dates)

        # 3. Simulate Time
        for current_date in all_dates:
            # A. Close matured trades
            still_active = []
            for exit_date, ret, bet_size in active_trades:
                if current_date >= exit_date:
                    current_cash += bet_size * (1 + ret)
                    if ret > 0: winning_trades += 1
                else:
                    still_active.append((exit_date, ret, bet_size))
            active_trades = still_active

            # B. Open new trades
            todays_signals = df_sampled[df_sampled['Trade_Date'] == current_date]
            for _, trade in todays_signals.iterrows():
                current_cash -= FIXED_BET_SIZE
                active_trades.append((current_date + pd.Timedelta(days=holding_period), trade[ret_col], FIXED_BET_SIZE))

            # C. Log Daily Wallet (Cash + Invested Capital)
            invested_capital = sum([t[2] for t in active_trades])
            current_wallet = current_cash + invested_capital
            daily_equity_log.append({'Date': current_date, 'Total_Wallet': current_wallet})

        # 4. Calculate Final Metrics
        ending_capital = daily_equity_log[-1]['Total_Wallet']
        win_rate = (winning_trades / n_samples) * 100 if n_samples > 0 else 0
        cagr = ((ending_capital / STARTING_CAPITAL) ** (252 / trading_days)) - 1 if trading_days > 0 else 0

        # Save CSV
        eq_df = pd.DataFrame(daily_equity_log)
        eq_df.to_csv(output_csv, index=False)
        
        # 5. Generate Graph
        plt.figure(figsize=(12, 6))
        plt.plot(eq_df['Date'], eq_df['Total_Wallet'], label=f'Random $5k Baseline ({cagr*100:.1f}% CAGR)', color='orange', linewidth=2)
        plt.axhline(STARTING_CAPITAL, color='gray', linestyle='--', alpha=0.7)
        plt.title(f"Monkey Dartboard Baseline: 3,300 Random Trades ($5k Each) - {horizon} Horizon")
        plt.ylabel("Total Portfolio Value ($)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_png)
        plt.close()

        print(f"✅ Random Baseline {horizon} | Trades: {n_samples} | Win Rate: {win_rate:.1f}% | CAGR: {cagr*100:.1f}%")
        print(f"✅ Saved Graph: {os.path.basename(output_png)}")

if __name__ == "__main__":
    main()