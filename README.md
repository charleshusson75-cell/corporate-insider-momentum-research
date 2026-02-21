# 📈 Corporate Insider Momentum: Replicating Alpha via Machine Learning

**Status:** Completed (Historical Backtest) | **Language:** Python | **Model:** XGBoost

## Overview
This repository contains the quantitative research, data engineering pipeline, and historical backtesting engine for an algorithmic trading strategy based on SEC Form 4 corporate insider disclosures.

While retail traders operate on delayed public news, corporate insiders (CEOs, CFOs) possess material, non-public information. By algorithmically filtering the noise of routine insider trading, this project isolates opportunistic, high-conviction "Wolfpack" buying behavior to generate market-beating returns.

## 📂 Repository Structure
```text
corporate-insider-momentum-research/
├── Historical_Research/
│   ├── insider_scraper.py       # Scrapes SEC EDGAR (2010-2025)
│   ├── market_data_fetcher.py   # Builds local price DB via Yahoo Fin
│   ├── feature_engineering.py   # Calculates ATR-14, Volume Abs, Wolfpack Score
│   ├── train_xgboost.py         # Trains ML model & outputs probabilities
│   └── portfolio_backtest.py    # Kelly Criterion simulated execution
├── Data/                        # (Excluded from Git) CSV datasets & Figures
├── Decision making/             # Strategy diagrams and flowcharts
├── Some docs/                   # Academic papers and personal paper on the project
├── requirements.txt             # Python dependencies
└── README.md                    # This file

---

## Methodology

The pipeline is designed to process unstructured government data into a strict financial framework:

1.  **Data Ingestion:** Scrapes and parses bulk SEC EDGAR Form 4 filings (2010–2025). Excludes 'Sell' transactions to eliminate liquidity-driven noise.
2.  **Feature Engineering:** Calculates 14-day rolling volatility (`ATR_14`), market liquidity absorption, and executive hierarchical weighting. Aggregates intraday "drip-feed" institutional orders into unified daily signals.
3.  **Machine Learning:** An XGBoost classifier is trained to predict positive 21-day momentum, outputting a continuous probability score (AI Confidence) rather than a binary signal.
4.  **Dynamic Sizing & Risk Management:** Signals are passed through a capital-constrained backtester using Kelly Criterion mathematics to optimize position sizing net of trading friction.

---

## 🧪 How to Run / Test

To replicate the backtest results:

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Backtest:**
    Execute the portfolio simulation. This script will load the pre-computed signals and apply the Kelly Criterion logic.
    ```bash
    python 01_Historical_Research/portfolio_backtest.py
    ```

3.  **Verify Output:**
    The script will generate `Data/optimal_equity_curve.csv` and print the optimized Sharpe Ratio and CAGR tables to the console.

---

## Backtest Results (Kelly Optimized)

To maximize geometric portfolio growth while strictly managing drawdown, dynamic position sizing was calculated using the Kelly Criterion. The backtest simulated a **$100,000 portfolio** from **Jan 2020 to Dec 2025** using a strict **5% max risk cap** per trade.