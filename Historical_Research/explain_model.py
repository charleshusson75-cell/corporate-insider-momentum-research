"""
General description: 
Opens the "Black Box" of the XGBoost model using SHAP (SHapley Additive exPlanations).
Generates a State-of-the-Art visual Summary Plot showing exactly how much each feature 
(e.g., Wolfpack Score, Volume) influenced the AI's buying decisions.
"""

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# --- BULLETPROOF PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_MATRIX = os.path.join(BASE_DIR, "Data", "ml_master_matrix.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Models", "xgboost_production_v1.pkl")
OUTPUT_GRAPH = os.path.join(BASE_DIR, "Data", "ai_brain_weights.png")

# --- EXACT FEATURE SPACE FROM TRAINING ---
# This guarantees we only feed the AI the numbers it knows how to read
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
    if not os.path.exists(MODEL_PATH) or not os.path.exists(INPUT_MATRIX):
        print("❌ Error: Cannot find model or master matrix. Run train_xgboost.py first.")
        return

    print("🧠 Loading AI Model and Historical Features...")
    
    # 1. Load the exact model used for production
    model = joblib.load(MODEL_PATH)
    
    # 2. Load the data the model was trained on
    df = pd.read_csv(INPUT_MATRIX)
    
    # Drop rows with missing values in our features (same as training)
    df.dropna(subset=FEATURES, inplace=True)
    
    # Explicitly isolate ONLY the numerical features the model expects
    features_df = df[FEATURES]
    
    # To save time and memory, sample 2,000 random trades to calculate SHAP
    if len(features_df) > 2000:
        features_sample = features_df.sample(n=2000, random_state=42)
    else:
        features_sample = features_df

    print("🔬 Calculating SHAP values (This opens the Black Box)...")
    
    # 3. Generate the SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_sample)
    
    # 4. Draw and Save the SOTA Graph (Clean Bar Chart)
    plt.figure(figsize=(10, 6))
    
    # ADD plot_type="bar" HERE:
    shap.summary_plot(shap_values, features_sample, plot_type="bar", show=False)
    
    plt.title("XGBoost Insider Trading: Mean Feature Importance", fontsize=14, pad=20)
    plt.xlabel("Mean |SHAP value| (Average impact on model output magnitude)")
    plt.tight_layout()
    
    # Save the new clean graph
    os.makedirs(os.path.dirname(OUTPUT_GRAPH), exist_ok=True)
    plt.savefig(OUTPUT_GRAPH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-" * 60)
    print(f"✅ SUCCESS: AI Brain Weights saved to: {OUTPUT_GRAPH}")
    print("-" * 60)
    print("How to read the graph:")
    print(" - Red dots = High value of that feature")
    print(" - Blue dots = Low value of that feature")
    print(" - Dots pushed to the RIGHT = Increased confidence to BUY")
    print(" - Dots pushed to the LEFT = Decreased confidence (Avoid)")

if __name__ == "__main__":
    main()