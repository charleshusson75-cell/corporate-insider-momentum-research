"""
General description: 
Multi-Horizon SHAP Explainer.
Opens the "Black Box" of the 1M, 2M, and 6M XGBoost models.
Generates a Mean Feature Importance (Bar Chart) for each horizon to prove 
how the AI's decision-making shifts from momentum to fundamental insider data over time.
"""

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- BULLETPROOF PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_MATRIX = os.path.join(BASE_DIR, "Data", "ml_master_matrix.csv")

HORIZONS = ['1M', '2M', '6M']

# --- EXACT FEATURE SPACE FROM TRAINING ---
FEATURES = [
    'Role_Weight', 
    'Consensus_Score', 
    'Portfolio_Pct', 
    'Pct_Volume_Absorbed', 
    'Value', 
    'Close', 
    'Volume', 
    'ATR_14'
]

def main():
    if not os.path.exists(INPUT_MATRIX):
        print("❌ Error: Cannot find master matrix. Run feature engineering first.")
        return

    print("🧠 Loading Historical Features for SHAP Analysis...")
    df = pd.read_csv(INPUT_MATRIX)
    df.dropna(subset=FEATURES, inplace=True)
    features_df = df[FEATURES]
    
    # Sample to keep computation fast
    if len(features_df) > 2000:
        features_sample = features_df.sample(n=2000, random_state=42)
    else:
        features_sample = features_df

    for horizon in HORIZONS:
        model_path = os.path.join(BASE_DIR, "Models", f"xgboost_production_{horizon}.pkl")
        output_graph = os.path.join(BASE_DIR, "Data", f"ai_brain_weights_{horizon}.png")
        
        if not os.path.exists(model_path):
            print(f"⚠️ Skipping {horizon}: Model not found at {model_path}")
            continue

        print(f"\n🔬 Calculating SHAP values for {horizon} Horizon...")
        model = joblib.load(model_path)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_sample)
        
        # Draw the Bar Graph
        plt.figure(figsize=(10, 6))
        
        # plot_type="bar" forces the clean Mean Absolute Importance chart
        shap.summary_plot(shap_values, features_sample, plot_type="bar", show=False)
        
        plt.title(f"XGBoost Feature Importance ({horizon} Horizon)", fontsize=14, pad=20)
        plt.xlabel("Mean |SHAP value| (Average impact on model output)")
        plt.tight_layout()
        
        # Save and close memory
        plt.savefig(output_graph, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ SUCCESS: {horizon} Brain Weights saved to: {output_graph}")

    print("\n🎉 All SHAP visualizations complete. Check your Data folder!")

if __name__ == "__main__":
    main()