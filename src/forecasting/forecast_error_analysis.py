
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "ml"
    / "evaluation"
    / "forecast_test_predictions.csv"
)

EVALUATION_FILE = (
    PROJECT_ROOT
    / "ml"
    / "evaluation"
    / "forecast_evaluation.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
    / "best_demand_forecasting_model.joblib"
)

EVALUATION_DIR = PROJECT_ROOT / "ml" / "evaluation"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

for folder in [EVALUATION_DIR, REPORT_DIR, FIGURE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("Starting MetroPulse forecast error analysis...")

    # --------------------------------------------------
    # 3. LOAD PREVIOUS RESULTS
    # --------------------------------------------------

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(PREDICTIONS_FILE)

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(EVALUATION_FILE)

    predictions = pd.read_csv(PREDICTIONS_FILE)

    predictions["pickup_hour"] = pd.to_datetime(
        predictions["pickup_hour"]
    )

    with open(EVALUATION_FILE, encoding="utf-8") as file:
        evaluation = json.load(file)

    best_model_name = evaluation["best_model"]

    prediction_column = (
        best_model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )

    if prediction_column not in predictions.columns:
        raise ValueError(
            f"Prediction column not found: {prediction_column}"
        )

    predictions["prediction"] = predictions[prediction_column]

    # --------------------------------------------------
    # 4. CALCULATE FORECAST ERRORS
    # --------------------------------------------------

    predictions["error"] = (
        predictions["actual_demand"]
        - predictions["prediction"]
    )

    predictions["absolute_error"] = (
        predictions["error"].abs()
    )

    predictions["squared_error"] = (
        predictions["error"] ** 2
    )

    predictions["hour"] = (
        predictions["pickup_hour"].dt.hour
    )

    predictions["day_of_week"] = (
        predictions["pickup_hour"].dt.day_of_week
    )

    mae = predictions["absolute_error"].mean()

    rmse = (
        predictions["squared_error"].mean() ** 0.5
    )

    mean_error = predictions["error"].mean()

    actual_mean = predictions["actual_demand"].mean()

    prediction_mean = predictions["prediction"].mean()

    # Positive error means underprediction.
    # Negative error means overprediction.

    # --------------------------------------------------
    # 5. ERROR ANALYSIS BY HOUR
    # --------------------------------------------------

    hourly_error = (
        predictions.groupby("hour")
        .agg(
            average_actual_demand=("actual_demand", "mean"),
            average_predicted_demand=("prediction", "mean"),
            average_absolute_error=("absolute_error", "mean"),
            number_of_observations=("error", "count")
        )
        .reset_index()
    )

    hourly_error.to_csv(
        EVALUATION_DIR / "forecast_error_by_hour.csv",
        index=False
    )

    # --------------------------------------------------
    # 6. SAVE DETAILED ERROR DATA
    # --------------------------------------------------

    predictions.to_csv(
        EVALUATION_DIR / "forecast_error_details.csv",
        index=False
    )

    # --------------------------------------------------
    # 7. RESIDUAL DISTRIBUTION CHART
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        predictions["error"],
        bins=25,
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
        label="Zero error"
    )

    plt.title("MetroPulse - Forecast Residual Distribution")
    plt.xlabel("Actual Demand - Predicted Demand")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "forecast_residual_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 8. ERROR BY HOUR CHART
    # --------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.bar(
        hourly_error["hour"],
        hourly_error["average_absolute_error"]
    )

    plt.title("MetroPulse - Forecast Error by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Absolute Error")

    plt.xticks(range(0, 24))
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "forecast_error_by_hour.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 9. RANDOM FOREST FEATURE IMPORTANCE
    # --------------------------------------------------

    importance_saved = False

    if MODEL_FILE.exists():

        artifact = joblib.load(MODEL_FILE)

        model = artifact["model"]
        features = artifact["features"]

        if hasattr(model, "feature_importances_"):

            importance = pd.DataFrame({
                "feature": features,
                "importance": model.feature_importances_
            })

            importance = importance.sort_values(
                "importance",
                ascending=False
            )

            importance.to_csv(
                EVALUATION_DIR / "forecast_feature_importance.csv",
                index=False
            )

            plt.figure(figsize=(10, 7))

            plt.barh(
                importance["feature"][::-1],
                importance["importance"][::-1]
            )

            plt.title("MetroPulse - Random Forest Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()

            plt.savefig(
                FIGURE_DIR / "forecast_feature_importance.png",
                dpi=300
            )

            plt.close()

            importance_saved = True

        else:
            print(
                "Selected model does not provide "
                "tree-based feature importance."
            )

    else:
        print(
            "Saved Random Forest model not found. "
            "Skipping feature importance."
        )

    # --------------------------------------------------
    # 10. GENERATE REPORT
    # --------------------------------------------------

    report = f"""
METROPULSE - FORECAST ERROR ANALYSIS
=====================================

Dataset: January 2024 NYC Yellow Taxi Trips

1. SELECTED MODEL
-----------------
{best_model_name}

2. FORECAST PERFORMANCE
-----------------------
Mean Absolute Error: {mae:.3f}
Root Mean Squared Error: {rmse:.3f}
Mean Error (Actual - Predicted): {mean_error:.3f}

Average Actual Demand: {actual_mean:.2f}
Average Predicted Demand: {prediction_mean:.2f}

3. ERROR INTERPRETATION
-----------------------
Positive mean error indicates average underprediction.
Negative mean error indicates average overprediction.

4. HOURLY ERROR ANALYSIS
------------------------
{hourly_error.to_string(index=False)}

5. FEATURE IMPORTANCE
---------------------
{"Generated successfully." if importance_saved else "Not generated for the selected model."}

6. OUTPUT FILES
---------------
ml/evaluation/forecast_error_details.csv
ml/evaluation/forecast_error_by_hour.csv
reports/figures/forecast_residual_distribution.png
reports/figures/forecast_error_by_hour.png

{"ml/evaluation/forecast_feature_importance.csv" if importance_saved else ""}
{"reports/figures/forecast_feature_importance.png" if importance_saved else ""}

=====================================
END OF REPORT
"""

    report_file = (
        REPORT_DIR / "forecast_error_analysis_report.txt"
    )

    report_file.write_text(
        report,
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 11. DISPLAY RESULTS
    # --------------------------------------------------

    print("\nForecast error analysis completed!")

    print("Selected model:", best_model_name)
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("Mean error:", round(mean_error, 3))

    print("\nHourly error analysis:")
    print(hourly_error.to_string(index=False))

    if importance_saved:
        print("\nFeature importance:")
        print(importance.to_string(index=False))

    print("\nReport saved:", report_file)


if __name__ == "__main__":
    main()
