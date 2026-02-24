import subprocess
import sys
import argparse

def run_script(script_path):
    print(f"\n[{script_path}]")
    print("=" * 50)
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        print(f"\n❌ CRITICAL FAILURE: {script_path} crashed. Halting pipeline.")
        sys.exit(1)
    print(f"✅ SUCCESS: {script_path} completed.\n")

def main():
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Corporate Insider ML Pipeline Orchestrator")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=['data_only', 'static_run', 'wfo_run', 'full'], 
        default='full',
        help="Choose which part of the pipeline to run."
    )
    args = parser.parse_args()

    print(f"🚀 BOOTING PIPELINE | MODE: {args.mode.upper()} 🚀")

    # 2. Define the exact script sequences
    data_scripts = [
        "Src/data_01_insider_scraper.py",
        "Src/data_02_market_fetcher.py",
        "Src/data_03_db_baseline.py",
        "Src/feat_04_cleaning.py",
        "Src/feat_05_engineering.py"
    ]
    
    static_scripts = [
        "Src/model_06_train_static.py",
        "Src/model_07_explain_shap.py",
        "Src/quant_08_backtest_static.py",
        "Src/quant_09_rolling_baseline.py",
        "Src/viz_10_performance.py"
    ]

    wfo_scripts = [
        # "Src/model_06_train_wfo.py",
        # "Src/quant_08_backtest_wfo.py",
        # "Src/viz_10_performance.py"
    ]

    # 3. Execute based on the user's terminal choice
    if args.mode in ['data_only', 'full']:
        print("\n" + "#"*50 + "\n Executing Phase 1: Data & Features\n" + "#"*50)
        for script in data_scripts:
            run_script(script)

    if args.mode in ['static_run', 'full']:
        print("\n" + "#"*50 + "\n Executing Phase 2: Static AI & Backtest\n" + "#"*50)
        for script in static_scripts:
            run_script(script)

    if args.mode == 'wfo_run':
        print("\n" + "#"*50 + "\n Executing Phase 3: Walk-Forward Optimization\n" + "#"*50)
        for script in wfo_scripts:
            run_script(script)

    print("\n🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()