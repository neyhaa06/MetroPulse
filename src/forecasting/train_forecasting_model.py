
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demand_features_2024_01.csv"
)

ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"
EVALUATION_DIR = PROJECT_ROOT / "ml" / "evaluation"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

for folder in [
    ARTIFACT_DIR,
    EVALUATION_DIR,
    REPORT_DIR,
    FIGURE_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

TEST_SIZE = 0.20
RANDOM_STATE = 42

TARGET = "total_trips"

FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "is_rush_hour",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168"
]


# --------------------------------------------------
# 3. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("Starting MetroPulse demand forecasting...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    # --------------------------------------------------
    # 4. LOAD AND VALIDATE DATA
    # --------------------------------------------------

    data = pd.read_csv(INPUT_FILE)

    data["pickup_hour"] = pd.to_datetime(
        data["pickup_hour"]
    )

    data = data.sort_values(
        "pickup_hour"
    ).reset_index(drop=True)

    required_columns = [
        "pickup_hour",
        TARGET
    ] + FEATURES

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in the forecasting dataset."
        )

    if not np.isfinite(
        data[FEATURES + [TARGET]].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite or invalid numerical values found."
        )

    if len(data) < 20:
        raise ValueError(
            "Insufficient records for model training."
        )

    # --------------------------------------------------
    # 5. CHRONOLOGICAL TRAIN-TEST SPLIT
    # --------------------------------------------------

    split_index = int(
        len(data) * (1 - TEST_SIZE)
    )

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    print("\nDataset information:")
    print("Total records:", len(data))
    print("Training records:", len(train_data))
    print("Testing records:", len(test_data))

    print(
        "Training period:",
        train_data["pickup_hour"].min(),
        "to",
        train_data["pickup_hour"].max()
    )

    print(
        "Testing period:",
        test_data["pickup_hour"].min(),
        "to",
        test_data["pickup_hour"].max()
    )

    # --------------------------------------------------
    # 6. DEFINE FORECASTING MODELS
    # --------------------------------------------------

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            loss="huber",
            random_state=RANDOM_STATE
        )
    }

    # --------------------------------------------------
    # 7. TRAIN AND EVALUATE MODELS
    # --------------------------------------------------

    results = []
    predictions = {}
    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train
        )

        predicted = model.predict(X_test)

        predicted = np.maximum(
            predicted,
            0
        )

        mae = mean_absolute_error(
            y_test,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predicted
            )
        )

        r2 = r2_score(
            y_test,
            predicted
        )

        results.append({
            "model": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4)
        })

        predictions[name] = predicted
        trained_models[name] = model

        print("MAE:", round(mae, 3))
        print("RMSE:", round(rmse, 3))
        print("R2:", round(r2, 4))

    # --------------------------------------------------
    # 8. SEASONAL NAIVE BASELINE
    # --------------------------------------------------

    baseline_predictions = test_data["lag_168"].to_numpy()

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_predictions
        )
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_predictions
    )

    results.append({
        "model": "Seasonal Naive (168h)",
        "MAE": round(baseline_mae, 3),
        "RMSE": round(baseline_rmse, 3),
        "R2": round(baseline_r2, 4)
    })

    predictions["Seasonal Naive (168h)"] = (
        baseline_predictions
    )

    # --------------------------------------------------
    # 9. COMPARE MODEL PERFORMANCE
    # --------------------------------------------------

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        "MAE"
    ).reset_index(drop=True)

    comparison_file = (
        EVALUATION_DIR / "forecast_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

    best_model_name = comparison.iloc[0]["model"]

    if best_model_name == "Seasonal Naive (168h)":
        print(
            "\nThe seasonal naive baseline performed best. "
            "No machine-learning model will be selected "
            "as the best model."
        )

        best_model = None

    else:
        best_model = trained_models[best_model_name]

        model_artifact = {
            "model": best_model,
            "features": FEATURES,
            "target": TARGET,
            "model_name": best_model_name
        }

        joblib.dump(
            model_artifact,
            ARTIFACT_DIR / "best_demand_forecasting_model.joblib"
        )

    # --------------------------------------------------
    # 10. SAVE TEST PREDICTIONS
    # --------------------------------------------------

    prediction_data = pd.DataFrame({
        "pickup_hour": test_data["pickup_hour"].to_numpy(),
        "actual_demand": y_test.to_numpy()
    })

    for name, predicted in predictions.items():

        column_name = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

        prediction_data[column_name] = predicted

    prediction_file = (
        EVALUATION_DIR / "forecast_test_predictions.csv"
    )

    prediction_data.to_csv(
        prediction_file,
        index=False
    )

    # --------------------------------------------------
    # 11. ACTUAL VS PREDICTED CHART
    # --------------------------------------------------

    plt.figure(figsize=(15, 7))

    plt.plot(
        prediction_data["pickup_hour"],
        prediction_data["actual_demand"],
        label="Actual Demand",
        linewidth=2
    )

    if best_model_name != "Seasonal Naive (168h)":

        best_column = (
            best_model_name.lower()
            .replace(" ", "_")
        )

        plt.plot(
            prediction_data["pickup_hour"],
            prediction_data[best_column],
            label=f"{best_model_name} Prediction",
            linewidth=1.8
        )

    else:

        plt.plot(
            prediction_data["pickup_hour"],
            prediction_data["seasonal_naive_168h"],
            label="Seasonal Naive Prediction",
            linewidth=1.8
        )

    plt.title(
        "MetroPulse - Actual vs Predicted Taxi Demand"
    )

    plt.xlabel("Pickup Time")
    plt.ylabel("Number of Taxi Trips")

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "demand_forecast_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 12. MODEL COMPARISON CHART
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison["model"],
        comparison["MAE"]
    )

    plt.title(
        "Forecasting Model Comparison - MAE"
    )

    plt.xlabel("Model")
    plt.ylabel("Mean Absolute Error")

    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "forecast_model_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 13. SAVE EVALUATION RESULTS
    # --------------------------------------------------

    evaluation = {
        "dataset": "January 2024 NYC Yellow Taxi Trips",
        "total_records": len(data),
        "training_records": len(train_data),
        "testing_records": len(test_data),
        "training_start": str(
            train_data["pickup_hour"].min()
        ),
        "training_end": str(
            train_data["pickup_hour"].max()
        ),
        "testing_start": str(
            test_data["pickup_hour"].min()
        ),
        "testing_end": str(
            test_data["pickup_hour"].max()
        ),
        "best_model": best_model_name,
        "model_metrics": comparison.to_dict(
            orient="records"
        )
    }

    evaluation_file = (
        EVALUATION_DIR / "forecast_evaluation.json"
    )

    with open(
        evaluation_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=4
        )

    # --------------------------------------------------
    # 14. GENERATE REPORT
    # --------------------------------------------------

    report = f"""
METROPULSE - DEMAND FORECASTING MODEL EVALUATION
=================================================

Dataset: January 2024 NYC Yellow Taxi Trips

1. DATASET
----------
Total records: {len(data)}
Training records: {len(train_data)}
Testing records: {len(test_data)}

Training period:
{train_data['pickup_hour'].min()} to
{train_data['pickup_hour'].max()}

Testing period:
{test_data['pickup_hour'].min()} to
{test_data['pickup_hour'].max()}

2. MODEL PERFORMANCE
--------------------
{comparison.to_string(index=False)}

3. BEST PERFORMING MODEL
------------------------
{best_model_name}

Selection metric: Lowest MAE

4. EVALUATION METRICS
---------------------
MAE: Mean Absolute Error
RMSE: Root Mean Squared Error
R2: Coefficient of Determination

5. EVALUATION APPROACH
----------------------
Chronological train-test split.
No random shuffling.

Forecasting evaluation uses observed historical
lag and rolling features for each test hour.
This represents rolling one-hour-ahead prediction,
not a multi-day forecast made all at once.

6. OUTPUT FILES
---------------
ml/evaluation/forecast_model_comparison.csv
ml/evaluation/forecast_test_predictions.csv
ml/evaluation/forecast_evaluation.json
ml/artifacts/best_demand_forecasting_model.joblib
reports/figures/demand_forecast_comparison.png
reports/figures/forecast_model_comparison.png

=================================================
END OF REPORT
"""

    report_file = (
        REPORT_DIR / "forecasting_model_report.txt"
    )

    report_file.write_text(
        report,
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 15. DISPLAY FINAL RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("METROPULSE FORECASTING RESULTS")
    print("=" * 55)

    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    print("\nBest performing model:", best_model_name)

    print("\nGenerated files:")
    print(comparison_file)
    print(prediction_file)
    print(evaluation_file)
    print(report_file)

    print("\nForecasting model training completed!")


if __name__ == "__main__":
    main()
