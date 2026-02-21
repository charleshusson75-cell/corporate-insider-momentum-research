"""
General description: 
Institutional-Grade Backtester.
Runs Kelly Optimization, tracks ML metrics (AUC, F1), calculates Quant 
Metrics (Sharpe, Max Drawdown), exports a detailed trade log, and generates 
a comparative equity curve graph vs the S&P 500.
"""

import pandas as pd
import numpy as np
import os
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "Data", "ai_buy_signals.csv")
OUTPUT_EQUITY = os.path.join(BASE_DIR, "Data", "optimal_equity_curve.csv")
OUTPUT_TRADES = os.path.join(BASE_DIR, "Data", "trade_log.csv")
OUTPUT_GRAPH = os.path.join(BASE_DIR, "Data", "strategy_vs_spy.png")

STARTING_CAPITAL = 100000.00
STOP_LOSS_PCT = -0.10
HOLDING_PERIOD = 21
TRADING_FRICTION = 0.000
THRESHOLDS_TO_TEST = np.arange(0.30, 0.81, 0.02)

def main():
    if not os.path.exists(INPUT_FILE):
        print("❌ Error: Run XGBoost first to generate signals.")
        return
        
    df = pd.read_csv(INPUT_FILE, parse_dates=['Trade_Date'])
    df.sort_values('Trade_Date', inplace=True)
    
    first_trade = df['Trade_Date'].min()
    last_trade = df['Trade_Date'].max()
    trading_days = np.busday_count(first_trade.date(), last_trade.date())

    # --- GLOBAL ML METRICS ---
    # AUC evaluates the AI's pure ranking ability regardless of threshold
    actual_wins = (df['Ret_1M'] > 0).astype(int)
    global_auc = roc_auc_score(actual_wins, df['AI_Confidence'])

    print(f"💼 Running Institutional Kelly Backtest...")
    print(f"   Period: {first_trade.date()} to {last_trade.date()} | Model AUC: {global_auc:.3f}")
    print(f"{'Thresh':<6} | {'Risk/Trd':<8} | {'Trades':<6} | {'Select%':<7} | {'Win%':<6} | {'F1':<5} | {'Max DD':<7} | {'Sharpe':<6} | {'True CAGR'}")
    print("-" * 85)

    results = []

    for threshold in THRESHOLDS_TO_TEST:
        loop_df = df[df['AI_Confidence'] >= threshold].copy()
        
        if len(loop_df) < 20: continue

        # ML Metrics specific to this threshold
        selection_pct = len(loop_df) / len(df)
        predicted_binary = (df['AI_Confidence'] >= threshold).astype(int)
        f1 = f1_score(actual_wins, predicted_binary)

        # Kelly Math
        wins = loop_df[loop_df['Ret_1M'] > 0]['Ret_1M']
        losses = loop_df[loop_df['Ret_1M'] <= 0]['Ret_1M']
        win_rate = len(wins) / len(loop_df)
        loss_rate = 1.0 - win_rate
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0001
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        full_kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio if win_loss_ratio > 0 else 0
        dynamic_risk_pct = min(full_kelly / 2.0, 0.05)
        
        if dynamic_risk_pct <= 0: continue

        # --- DYNAMIC BACKTEST ---
        current_cash = STARTING_CAPITAL
        active_trades = [] 
        trades_taken_count = 0
        winning_trades_count = 0
        
        daily_equity_log = []
        historical_trade_log = []

        for current_date in sorted(loop_df['Trade_Date'].unique()):
            # Close matured trades
            still_active = []
            for t in active_trades:
                if current_date >= t['exit_date']:
                    profit_dollars = t['bet_size'] * t['ret']
                    current_cash += t['bet_size'] + profit_dollars
                    if t['ret'] > 0: winning_trades_count += 1
                    
                    # Log the completed trade
                    t['exit_date_actual'] = current_date
                    t['profit_dollars'] = profit_dollars
                    historical_trade_log.append(t)
                else:
                    still_active.append(t)
            active_trades = still_active

            # Track Equity & Invested Percentage
            invested_capital = sum([t['bet_size'] for t in active_trades])
            current_equity = current_cash + invested_capital
            pct_invested = invested_capital / current_equity if current_equity > 0 else 0
            daily_equity_log.append({'Date': current_date, 'Strategy_Equity': current_equity, 'Pct_Invested': pct_invested})

            # Open new trades
            todays_signals = loop_df[loop_df['Trade_Date'] == current_date]
            for _, trade in todays_signals.iterrows():
                max_bet_dollars = current_equity * dynamic_risk_pct 
                bet_size = max_bet_dollars * trade['AI_Confidence']
                
                if current_cash >= bet_size:
                    current_cash -= bet_size
                    trades_taken_count += 1
                    exit_date = current_date + timedelta(days=HOLDING_PERIOD)
                    realized_ret = max(trade['Ret_1M'], STOP_LOSS_PCT) - TRADING_FRICTION
                    
                    active_trades.append({
                        'ticker': trade.get('Ticker', 'Unknown'),
                        'buy_date': current_date,
                        'exit_date': exit_date,
                        'ret': realized_ret,
                        'bet_size': bet_size
                    })
        
        # End of Backtest Liquidation
        for t in active_trades:
            profit_dollars = t['bet_size'] * t['ret']
            current_cash += t['bet_size'] + profit_dollars
            if t['ret'] > 0: winning_trades_count += 1
            t['exit_date_actual'] = last_trade
            t['profit_dollars'] = profit_dollars
            historical_trade_log.append(t)

        if trades_taken_count == 0: continue
            
        final_win_rate = winning_trades_count / trades_taken_count
        cagr = ((current_cash / STARTING_CAPITAL) ** (252 / trading_days)) - 1 if trading_days > 0 else 0

        # --- Quant Metrics (Sharpe & Max DD) ---
        eq_df = pd.DataFrame(daily_equity_log)
        eq_df['Daily_Ret'] = eq_df['Strategy_Equity'].pct_change().fillna(0)
        
        # Max Drawdown
        eq_df['Cum_Max'] = eq_df['Strategy_Equity'].cummax()
        eq_df['Drawdown'] = (eq_df['Strategy_Equity'] - eq_df['Cum_Max']) / eq_df['Cum_Max']
        max_dd = eq_df['Drawdown'].min()

        # Sharpe Ratio (Assuming 0% risk free rate for simplicity)
        daily_mean = eq_df['Daily_Ret'].mean()
        daily_std = eq_df['Daily_Ret'].std()
        sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0

        results.append({
            'Threshold': threshold,
            'Risk_Pct': dynamic_risk_pct,
            'Trades': trades_taken_count,
            'Selection_Pct': selection_pct,
            'Win_Rate': final_win_rate,
            'F1': f1,
            'Max_DD': max_dd,
            'Sharpe': sharpe,
            'Ending_Capital': current_cash,
            'CAGR': cagr,
            'Equity_Curve': eq_df,
            'Trade_Log': historical_trade_log
        })
        
        print(f"{threshold*100:>4.0f}% | {dynamic_risk_pct*100:>7.2f}% | {trades_taken_count:<6} | {selection_pct*100:>6.1f}% | {final_win_rate*100:>5.1f}% | {f1:.3f} | {max_dd*100:>6.1f}% | {sharpe:>5.2f} | {cagr*100:>6.2f}%")

    # --- GENERATE OUTPUTS FOR THE BEST STRATEGY ---
    if results:
        best = max(results, key=lambda x: x['CAGR'])

        # 1. Save Equity Curve & Trade Log
        best['Equity_Curve'].to_csv(OUTPUT_EQUITY, index=False)
        pd.DataFrame(best['Trade_Log']).to_csv(OUTPUT_TRADES, index=False)
        
        print("=" * 85)
        print("🏆 OPTIMAL STRATEGY EXPORTED")
        print(f"Metrics: Sharpe {best['Sharpe']:.2f} | Max DD {best['Max_DD']*100:.1f}% | Trades: {best['Trades']}")
        print(f"Exported detailed trade log to: {OUTPUT_TRADES}")
        
        # 2. Download S&P 500 Benchmark and Plot
        print("📉 Generating comparative graph vs S&P 500...")
        spy = yf.download('SPY', start=first_trade.strftime('%Y-%m-%d'), end=last_trade.strftime('%Y-%m-%d'), progress=False)
        spy['SPY_Normalized'] = (spy['Close'] / spy['Close'].iloc[0]) * STARTING_CAPITAL
        
        eq_df = best['Equity_Curve'].set_index('Date')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Top chart: Equity Curves
        ax1.plot(eq_df.index, eq_df['Strategy_Equity'], label=f"AI Strategy ({best['CAGR']*100:.1f}% CAGR)", color='blue', linewidth=2)
        ax1.plot(spy.index, spy['SPY_Normalized'], label="S&P 500 Baseline", color='gray', linestyle='--')
        ax1.set_title(f"AI Insider Momentum vs S&P 500 (Starting Capital: ${STARTING_CAPITAL:,})")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Bottom chart: Invested Percentage
        ax2.fill_between(eq_df.index, eq_df['Pct_Invested'] * 100, color='blue', alpha=0.2)
        ax2.plot(eq_df.index, eq_df['Pct_Invested'] * 100, color='blue', linewidth=1)
        ax2.set_ylabel("% Capital Invested")
        ax2.set_ylim(0, 100)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_GRAPH)
        print(f"✅ Saved visually rich Graph to: {OUTPUT_GRAPH}")
        print("=" * 85)

if __name__ == "__main__":
    main()