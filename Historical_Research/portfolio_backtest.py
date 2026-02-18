"""
General description: 
Runs a realistic Grid Search Optimization over the AI's predictions using the 
Kelly Criterion. Dynamically calculates the mathematically optimal Position Size 
(Half-Kelly) for each AI Confidence Threshold, enforces capital constraints, 
and finds the optimal True CAGR.
"""

import pandas as pd
import numpy as np
import os

# ==========================================
# --- CONFIGURATION & HYPERPARAMETERS ---
# ==========================================
INPUT_FILE = "Data/ai_buy_signals.csv"
OUTPUT_FILE = "Data/optimal_equity_curve.csv"

STARTING_CAPITAL = 100000.00
STOP_LOSS_PCT = -0.10        # Hard stop loss at -10% drawdown
HOLDING_PERIOD = 21          # Number of trading days a trade is held
TRADING_FRICTION = 0.000     # Zero fees on Alpaca

# Let's test from 50% to 80%
THRESHOLDS_TO_TEST = np.arange(0.50, 0.81, 0.01)

def main():
    if not os.path.exists(INPUT_FILE):
        print("❌ Error: Run XGBoost first to generate signals.")
        return
        
    df = pd.read_csv(INPUT_FILE, parse_dates=['Trade_Date'])
    df.sort_values('Trade_Date', inplace=True)
    
    first_trade = df['Trade_Date'].min()
    last_trade = df['Trade_Date'].max()
    trading_days = np.busday_count(first_trade.date(), last_trade.date())

    print(f"💼 Running Kelly-Optimized Equity Backtest...")
    print(f"   Test Period: {first_trade.date()} to {last_trade.date()} ({trading_days} Days)")
    print(f"   Fees: {TRADING_FRICTION*100}% | Stop-Loss: {STOP_LOSS_PCT*100}%\n")
    
    print(f"{'Thresh':<6} | {'Half-Kelly':<10} | {'Trades':<6} | {'Win Rate':<8} | {'End Capital':<12} | {'True CAGR'}")
    print("-" * 70)

    results = []

    for threshold in THRESHOLDS_TO_TEST:
        loop_df = df[df['AI_Confidence'] >= threshold].copy()
        
        if len(loop_df) < 20: # Skip if too few trades to be statistically valid
            continue

        # --- 1. KELLY CRITERION MATH ---
        wins = loop_df[loop_df['Ret_1M'] > 0]['Ret_1M']
        losses = loop_df[loop_df['Ret_1M'] <= 0]['Ret_1M']
        
        win_rate = len(wins) / len(loop_df)
        loss_rate = 1.0 - win_rate
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0001
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        if win_loss_ratio > 0:
            full_kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        else:
            full_kelly = 0
            
        half_kelly = full_kelly / 2.0
        
        # If math says we have no edge, skip this threshold
        if half_kelly <= 0:
            continue
            
        # SAFETY CAP: Never risk more than 5% on a single trade, even if Kelly says to.
        dynamic_risk_pct = min(half_kelly, 0.05)

        # --- 2. REALISTIC DYNAMIC BACKTEST ---
        current_cash = STARTING_CAPITAL
        active_trades = [] # (exit_date, return_pct, bet_size)
        trades_taken_count = 0
        winning_trades_count = 0

        daily_equity_log = []
        
        for current_date in sorted(loop_df['Trade_Date'].unique()):
            
            # Close matured trades
            still_active = []
            for exit_date, ret, bet_size in active_trades:
                if current_date >= exit_date:
                    current_cash += bet_size * (1 + ret)
                    if ret > 0: winning_trades_count += 1
                else:
                    still_active.append((exit_date, ret, bet_size))
            active_trades = still_active

            # Calculate Current Total Equity
            invested_capital = sum([trade[2] for trade in active_trades])
            current_equity = current_cash + invested_capital
            daily_equity_log.append({'Date': current_date, 'Strategy_Equity': current_equity})

            # Process new signals
            todays_signals = loop_df[loop_df['Trade_Date'] == current_date]
            
            for _, trade in todays_signals.iterrows():
                # 🛑 THE MAGIC: Size bet based on Kelly Criterion limits!
                max_bet_dollars = current_equity * dynamic_risk_pct 
                bet_size = max_bet_dollars * trade['AI_Confidence']
                
                # Take the trade if we have enough raw cash
                if current_cash >= bet_size:
                    current_cash -= bet_size
                    trades_taken_count += 1
                    
                    exit_date = current_date + pd.Timedelta(days=HOLDING_PERIOD)
                    realized_ret = max(trade['Ret_1M'], STOP_LOSS_PCT) - TRADING_FRICTION
                    active_trades.append((exit_date, realized_ret, bet_size))
        
        # End of Backtest Liquidation
        for _, ret, bet_size in active_trades:
            current_cash += bet_size * (1 + ret)
            if ret > 0: winning_trades_count += 1

        if trades_taken_count == 0: continue
            
        final_win_rate = winning_trades_count / trades_taken_count
        
        if trading_days > 0 and current_cash > 0:
            cagr = ((current_cash / STARTING_CAPITAL) ** (252 / trading_days)) - 1
        else:
            cagr = 0.0
            
        results.append({
            'Threshold': threshold,
            'Optimal_Risk_Pct': dynamic_risk_pct,
            'Trades': trades_taken_count,
            'Win_Rate': final_win_rate,
            'Ending_Capital': current_cash,
            'CAGR': cagr,
            'Equity_Curve': daily_equity_log
        })
        
        print(f"{threshold*100:>4.0f}% | {dynamic_risk_pct*100:>7.2f}% | {trades_taken_count:<6} | {final_win_rate*100:>5.2f}% | ${current_cash:,.2f} | {cagr*100:>6.2f}%")

    if results:
        best = max(results, key=lambda x: x['CAGR'])

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        pd.DataFrame(best['Equity_Curve']).to_csv(OUTPUT_FILE, index=False)
        
        print("=" * 70)
        print("🏆 REALISTIC OPTIMAL STRATEGY DISCOVERED")
        print("=" * 70)
        print(f"Best Confidence Threshold: {best['Threshold']*100:.0f}%")
        print(f"Optimal Position Size:     {best['Optimal_Risk_Pct']*100:.2f}% (Half-Kelly)")
        print(f"Resulting CAGR:            {best['CAGR']*100:.2f}%")
        print(f"Total Trades Taken:        {best['Trades']}")
        print(f"Win Rate:                  {best['Win_Rate']*100:.2f}%")
        print("=" * 70)

if __name__ == "__main__":
    main()