import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def train_and_save_model():
    # Load dataset
    df = pd.read_csv("data/energy_data.csv")
    X = df.drop("energy_consumption", axis=1)
    y = df["energy_consumption"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_model.pkl")
    print("✅ Model trained and saved.")

if __name__ == "__main__":
    train_and_save_model()
