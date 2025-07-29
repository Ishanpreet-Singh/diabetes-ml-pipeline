import joblib
import optuna
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load data
def load_data():
    data = load_diabetes(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    return df

# Preprocess data
def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler

# Objective function for Optuna
def objective(trial, X, y):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    score = cross_val_score(model, X, y, cv=3, scoring="neg_root_mean_squared_error")
    return -score.mean()

# Main training function
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_data()
    (X_train, X_test, y_train, y_test), scaler = preprocess_data(df)

    print("Tuning hyperparameters with Optuna (20 trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)

    best_params = study.best_params
    print("Best parameters:", best_params)

    print("Training final model...")
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE:", rmse)
    print("R²:", r2_score(y_test, preds))

    # Ensure models/ directory exists
    os.makedirs("models", exist_ok=True)

    # Save model and scaler
    joblib.dump(model, "models/diabetes_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Model and scaler saved in models/ folder.")
