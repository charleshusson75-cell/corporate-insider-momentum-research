"""
General description: 
Fully Invested Rolling Baseline.
Allocates capital based on the user's formula: 
Bet Size = Total Current Wallet * (1 / Number of trades in the upcoming 28 days).
Forces the portfolio to be nearly 100% invested at all times.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "Data", "ai_buy_signals.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "Data", "rolling_baseline_metrics.csv")

STARTING_CAPITAL = 100000.00
HOLDING_PERIOD = 21
ROLLING_WINDOW_DAYS = 28

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE, parse_dates=['Trade_Date'])
    df.sort_values('Trade_Date', inplace=True)

    total_db_trades = len(df)
    if total_db_trades == 0:
        return

    # 1. PRE-CALCULATE THE 28-DAY FORWARD WINDOW FOR EVERY DATE
    # Count how many trades happen on each specific day
    date_counts = df.groupby('Trade_Date').size().reset_index(name='count')
    date_counts.set_index('Trade_Date', inplace=True)
    
    # Create a continuous calendar to avoid skipping days with 0 trades
    full_cal = pd.date_range(start=df['Trade_Date'].min(), 
                             end=df['Trade_Date'].max() + pd.Timedelta(days=ROLLING_WINDOW_DAYS))
    date_counts = date_counts.reindex(full_cal, fill_value=0)
    
    # Calculate the rolling sum looking FORWARD 28 days
    date_counts['forward_28d_trades'] = date_counts['count'].iloc[::-1].rolling(window=ROLLING_WINDOW_DAYS, min_periods=1).sum().iloc[::-1]

    print(f"💼 Running Fully Invested Rolling Baseline...")
    
    # 2. CHRONOLOGICAL BACKTEST
    current_cash = STARTING_CAPITAL
    active_trades = []
    daily_equity_log = []
    
    trades_taken = 0
    winning_trades = 0

    unique_dates = sorted(df['Trade_Date'].unique())
    first_trade = unique_dates[0]
    last_trade = unique_dates[-1]
    trading_days = np.busday_count(first_trade.date(), last_trade.date())

    for current_date in unique_dates:
        # A. Close matured trades and free up cash
        still_active = []
        for exit_date, ret, bet_size in active_trades:
            if current_date >= exit_date:
                current_cash += bet_size * (1 + ret)
                if ret > 0: winning_trades += 1
            else:
                still_active.append((exit_date, ret, bet_size))
        active_trades = still_active

        # B. Calculate Current Total Wallet (Cash + Invested)
        invested_capital = sum([trade[2] for trade in active_trades])
        current_wallet = current_cash + invested_capital
        daily_equity_log.append({'Date': current_date, 'Total_Wallet': current_wallet})

        # C. Apply Your Formula: Wallet * (1 / Trades in next 28 days)
        trades_in_next_28d = date_counts.loc[current_date, 'forward_28d_trades']
        
        if trades_in_next_28d > 0:
            target_bet_size = current_wallet / trades_in_next_28d
        else:
            target_bet_size = 0

        # D. Process new signals for today
        todays_signals = df[df['Trade_Date'] == current_date]
        
        for _, trade in todays_signals.iterrows():
            # CRITICAL: We cap the bet at our actual available cash so we never go negative
            actual_bet = min(target_bet_size, current_cash)
            
            # If we have at least $1 left to trade, take it
            if actual_bet > 1.0:
                current_cash -= actual_bet
                trades_taken += 1
                
                exit_date = current_date + pd.Timedelta(days=HOLDING_PERIOD)
                active_trades.append((exit_date, trade['Ret_1M'], actual_bet))
    
    # End of Backtest Liquidation
    for _, ret, bet_size in active_trades:
        current_cash += bet_size * (1 + ret)
        if ret > 0: winning_trades += 1

    # 3. CALCULATE FINAL METRICS
    ending_capital = current_cash
    win_rate = (winning_trades / trades_taken) * 100 if trades_taken > 0 else 0
    cagr = ((ending_capital / STARTING_CAPITAL) ** (252 / trading_days)) - 1 if trading_days > 0 else 0

    # Save to CSV as requested
    pd.DataFrame(daily_equity_log).to_csv(OUTPUT_FILE, index=False)

    print("==========================================================")
    print("📈 FULLY INVESTED ROLLING BASELINE (28-DAY WINDOW)")
    print("==========================================================")
    print(f"Total Trades Taken:  {trades_taken} (Out of {total_db_trades})")
    print(f"Simulated Win Rate:  {win_rate:.2f}%")
    print(f"Starting Capital:    ${STARTING_CAPITAL:,.2f}")
    print(f"Ending Capital:      ${ending_capital:,.2f}")
    print(f"Resulting True CAGR: {cagr*100:.2f}%")
    print("==========================================================")
    print(f"✅ Saved daily wallet tracking to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()