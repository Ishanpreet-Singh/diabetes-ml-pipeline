import sys
import joblib
import pandas as pd
import os

# Check if model and scaler exist
MODEL_PATH = "models/diabetes_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("Error: Trained model or scaler not found in 'models/' folder. Please run train.py first.")
    sys.exit(1)

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict(input_csv):
    # Read input data
    data = pd.read_csv(input_csv)
    data_scaled = scaler.transform(data)
    predictions = model.predict(data_scaled)
    
    # Combine input and predictions
    results = data.copy()
    results["Prediction"] = predictions
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <input_csv>")
    else:
        input_csv = sys.argv[1]
        if not os.path.exists(input_csv):
            print(f"Error: File '{input_csv}' not found.")
            sys.exit(1)

        results = predict(input_csv)
        print("\nPredictions:")
        print(results)

        # Save predictions to predictions.csv
        results.to_csv("predictions.csv", index=False)
        print("\nPredictions saved to predictions.csv")
