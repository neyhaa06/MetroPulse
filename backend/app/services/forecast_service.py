
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd


# Locate the MetroPulse project directory
ROOT_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    ROOT_DIR
    / "ml"
    / "artifacts"
    / "best_demand_forecasting_model.joblib"
)


@lru_cache(maxsize=1)
def load_forecasting_model():
    """Load the saved forecasting model only once."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Forecasting model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    features = artifact["features"]

    return model, features


def predict_demand(input_features: dict):
    """Generate an hourly taxi demand prediction."""

    model, feature_columns = load_forecasting_model()

    # Validate that all required model features are supplied
    missing_features = [
        feature
        for feature in feature_columns
        if feature not in input_features
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Arrange inputs in the exact training feature order
    input_data = pd.DataFrame(
        [[input_features[feature] for feature in feature_columns]],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]

    return {
        "predicted_demand": round(float(prediction), 2),
        "unit": "trips per hour",
        "model": "Random Forest"
    }
