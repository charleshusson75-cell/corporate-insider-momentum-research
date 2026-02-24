"""
Fully Invested Rolling Baseline (Multi-Horizon).
Allocates capital based on the formula: Bet Size = Total Wallet * (1 / trades in upcoming window).
Loops through 1M, 2M, and 6M horizons using the unified signals database.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "Data", "ai_buy_signals.csv")

STARTING_CAPITAL = 100000.00

HORIZON_CONFIGS = {
    '1M': {'hold_days': 21,  'ret_col': 'Ret_1M'},
    '2M': {'hold_days': 42,  'ret_col': 'Ret_2M'},
    '6M': {'hold_days': 126, 'ret_col': 'Ret_6M'}
}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE, parse_dates=['Trade_Date'])
    df.sort_values('Trade_Date', inplace=True)
    total_db_trades = len(df)

    for horizon, config in HORIZON_CONFIGS.items():
        print(f"\n💼 Running Fully Invested Baseline for {horizon}...")
        
        output_file = os.path.join(BASE_DIR, "Data", f"rolling_baseline_metrics_{horizon}.csv")
        holding_period = config['hold_days']
        ret_col = config['ret_col']
        
        if ret_col not in df.columns: continue

        date_counts = df.groupby('Trade_Date').size().reset_index(name='count')
        date_counts.set_index('Trade_Date', inplace=True)
        
        full_cal = pd.date_range(start=df['Trade_Date'].min(), end=df['Trade_Date'].max() + pd.Timedelta(days=holding_period))
        date_counts = date_counts.reindex(full_cal, fill_value=0)
        date_counts['forward_trades'] = date_counts['count'].iloc[::-1].rolling(window=holding_period, min_periods=1).sum().iloc[::-1]

        current_cash = STARTING_CAPITAL
        active_trades = []
        daily_equity_log = []
        trades_taken = 0
        winning_trades = 0

        unique_dates = sorted(df['Trade_Date'].unique())
        trading_days = np.busday_count(unique_dates[0].date(), unique_dates[-1].date())

        for current_date in unique_dates:
            still_active = []
            for exit_date, ret, bet_size in active_trades:
                if current_date >= exit_date:
                    current_cash += bet_size * (1 + ret)
                    if ret > 0: winning_trades += 1
                else:
                    still_active.append((exit_date, ret, bet_size))
            active_trades = still_active

            invested_capital = sum([t[2] for t in active_trades])
            current_wallet = current_cash + invested_capital
            daily_equity_log.append({'Date': current_date, 'Total_Wallet': current_wallet})

            trades_in_next_window = date_counts.loc[current_date, 'forward_trades']
            target_bet_size = current_wallet / trades_in_next_window if trades_in_next_window > 0 else 0

            todays_signals = df[df['Trade_Date'] == current_date]
            for _, trade in todays_signals.iterrows():
                actual_bet = min(target_bet_size, current_cash)
                if actual_bet > 1.0:
                    current_cash -= actual_bet
                    trades_taken += 1
                    active_trades.append((current_date + pd.Timedelta(days=holding_period), trade[ret_col], actual_bet))
        
        for _, ret, bet_size in active_trades:
            current_cash += bet_size * (1 + ret)
            if ret > 0: winning_trades += 1

        ending_capital = current_cash
        win_rate = (winning_trades / trades_taken) * 100 if trades_taken > 0 else 0
        cagr = ((ending_capital / STARTING_CAPITAL) ** (252 / trading_days)) - 1 if trading_days > 0 else 0

        pd.DataFrame(daily_equity_log).to_csv(output_file, index=False)
        print(f"✅ Baseline {horizon} | Trades: {trades_taken} | Win Rate: {win_rate:.1f}% | CAGR: {cagr*100:.1f}%")

if __name__ == "__main__":
    main()