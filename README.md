# 📈 Corporate Insider Momentum: Replicating Alpha via Machine Learning

**Status:** Completed (Historical Backtest) | **Language:** Python | **Model:** XGBoost

## Overview
This repository contains the quantitative research, data engineering pipeline, and historical backtesting engine for an algorithmic trading strategy based on SEC Form 4 corporate insider disclosures.

While retail traders operate on delayed public news, corporate insiders (CEOs, CFOs, board members) possess material, non-public information. By algorithmically filtering the noise of routine and liquidity-driven insider trading, this project isolates opportunistic, high-conviction "Wolfpack" buying behavior to generate market-beating returns. 

📄 **Read the full academic research paper:** [`Replicating_corporate_insider_alpha_via_ML.pdf`](./Replicating_corporate_insider_alpha_via_ML.pdf)

---

## 📂 Project Structure

The architecture is strictly modular to prevent data leakage between ingestion, model training, and historical backtesting. The directory has been cleaned and optimized for data efficiency.

```text
corporate-insider-momentum-research/
├── .github/                                       # CI/CD and Cloud Automation workflows
├── Replicating_corporate_insider_alpha_via_ML.pdf # Full Academic Whitepaper
├── README.md                                      # This file
├── requirements.txt                               # Python dependencies
│
├── Decision making/                               # Strategy rules and flowcharts
│   ├── Criterion.txt
│   ├── Decision.txt
│   └── variables.txt
│
├── Data/                                          # Local Data Warehouse (Ignored in Git)
│   ├── ai_brain_weights_1M.png                    # SHAP visualizations
│   ├── strategy_vs_spy_1M.png                     # Equity Curve outputs
│   ├── sec_master_bulk_data.csv                   # Raw EDGAR filings
│   ├── ml_master_matrix.csv                       # Final ML-ready dataset
│   ├── ai_buy_signals_master.csv                  # Consolidated AI probabilities (1M, 2M, 6M)
│   └── optimal_equity_curve_1M.csv                # Backtester trade logs and equity tracking
│
└── Src/                                           # Core Research Environment
    ├─ insider_scraper.py                          # 1. SEC EDGAR ingestion (2010-2025)
    ├─ market_data_fetcher.py                      # 2. Yahoo Finance OHLCV mapping
    ├─ generate_full_db_baseline.py                # 3. Database compilation
    ├─ processor_cleaning.py                       # 4. Data sanitization & filtering
    ├─ feature_engineering.py                      # 5. Technicals (ATR) & Consensus weighting
    ├─ train_xgboost.py                            # 6. Multi-Horizon ML Model Training
    ├─ explain_model.py                            # 7. SHAP Game-Theory Interpretation
    ├─ portfolio_backtest.py                       # 8. Capital-Constrained Kelly Backtester
    ├─ plot_performance.py                         # 9. Matplotlib Equity Visualizations
    └─ trading_agent_groq.py                       # 10. LLM Reasoning Interface (Groq)

```

---

## ⚙️ Methodology

The pipeline is designed to process unstructured government data into a strict financial framework:

1. **Data Ingestion & Fidelity:** Scrapes and parses bulk SEC EDGAR Form 4 filings. Excludes 'Sell' transactions to eliminate liquidity-driven noise and enforces strict tradability filters (Price > $2.00, Volume > 50,000).
2. **Feature Engineering:** Calculates 14-day rolling volatility (`ATR_14`), market liquidity absorption, and executive hierarchical weighting. Aggregates intraday "drip-feed" institutional orders into unified daily signals.
3. **Machine Learning:** An XGBoost classifier is trained to predict positive momentum across multiple horizons (1M, 2M, 6M), outputting a continuous probability score (AI Confidence) rather than a rigid binary signal.
4. **Dynamic Sizing & Risk Management:** Signals are passed through a capital-constrained backtester using Half-Kelly Criterion mathematics and a strict 5% max risk cap to optimize position sizing net of trading friction.

---

## 📊 Backtest Results & Visual Analysis

To maximize geometric portfolio growth while strictly managing drawdown, the backtest simulated a **$100,000 portfolio** from **Jan 2020 to Dec 2025**.

The system was evaluated across three distinct holding horizons, revealing a massive regime shift in the AI's logic:

| Horizon | Optimal Threshold | True CAGR | Max Drawdown | Sharpe Ratio | Win Rate |
| --- | --- | --- | --- | --- | --- |
| **1-Month (Momentum)** | 60% | **+144.7%** | -16.7% | 5.51 | 59.1% |
| **2-Month (Balanced)** | 34% | **+93.1%** | -10.1% | 3.39 | 55.1% |
| **6-Month (Fundamental)** | 60% | **+58.3%** | **-2.8%** | 2.42 | **60.6%** |

*Note: The 1-Month model was selected as the primary strategy for live deployment due to its hyper-aggressive geometric compounding.*

### Test Examples

**1. Output Equity Curve vs. S&P 500 (1-Month Strategy)** The strategy heavily outperforms the benchmark by capturing rapid, volatility-adjusted momentum following insider disclosures.

<img width="1000" alt="Strategy vs SPY" src="Data/strategy_vs_spy_1M.png">

**2. SHAP Feature Importance (1-Month Strategy)** Opening the "Black Box" reveals that for short-term holds, the AI prioritizes market microstructure (`Close`, `Volume`, `ATR_14`) to capture immediate momentum, using insider signals (`Consensus_Score`) as secondary filters.

<img width="800" alt="SHAP Values" src="Data/ai_brain_weights_1M.png">

---

## 🧪 How to Run / Reproduce

To replicate the backtest results and generate the visual charts:

**1. Install Dependencies:**

```bash
pip install -r requirements.txt

```

**2. Generate the Master Database:**
Execute the data processing scripts sequentially to scrape the SEC, fetch Yahoo Finance data, and build the engineered features.

```bash
python Src/feature_engineering.py

```

**3. Train the XGBoost Models:**
Train the multi-horizon AI brains. This will output the `.pkl` models and the consolidated AI signals CSV file.

```bash
python Src/train_xgboost.py

```

**4. Run the Institutional Backtest:**
Execute the portfolio simulation. This script applies the Kelly Criterion logic and outputs the optimal performance metrics to the console.

```bash
python Src/portfolio_backtest.py

```

**5. (Optional) Audit the AI Brain:**
Generate the SOTA SHAP visualizations to verify feature importance.

```bash
python Src/explain_model.py

```

---

*Disclaimer: This repository is for academic and quantitative research purposes only. It does not constitute financial advice. The live execution engine (API keys, order routing, and cloud automation) is maintained in a separate, private repository to protect proprietary infrastructure.*

---

© 2026 Charles Husson. All Rights Reserved. This repository is provided for academic review and portfolio demonstration purposes only. No license is granted for commercial or personal use.