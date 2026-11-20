
from pathlib import Path
import json
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    ROOT_DIR / "data" / "processed" / "demand_features_2024_01.csv"
)

MODEL_PATH = (
    ROOT_DIR / "ml" / "artifacts" /
    "best_demand_forecasting_model.joblib"
)

OUTPUT_DIR = ROOT_DIR / "ml" / "evaluation"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"
REPORT_PATH = ROOT_DIR / "reports" / "shap_analysis_report.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("Starting MetroPulse SHAP explainability analysis...")

# Load feature data and trained model
data = pd.read_csv(FEATURES_PATH)
model_artifact = joblib.load(MODEL_PATH)
model = model_artifact["model"]
feature_columns = model_artifact["features"]

# Use the same input features as the forecasting model
feature_columns = [
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
    "rolling_mean_168",
]

X = data[feature_columns].copy()

# Sample data to keep SHAP computation efficient
sample_size = min(200, len(X))
X_sample = X.sample(n=sample_size, random_state=42)

print(f"Total feature records: {len(X)}")
print(f"SHAP sample records: {len(X_sample)}")

# Explain the Random Forest model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Handle SHAP output formats
if isinstance(shap_values, list):
    shap_values = shap_values[0]

if getattr(shap_values, "ndim", 0) == 3:
    shap_values = shap_values[:, :, 0]

# Calculate mean absolute SHAP importance
importance = pd.DataFrame({
    "feature": feature_columns,
    "mean_absolute_shap_value": abs(shap_values).mean(axis=0),
})

importance = importance.sort_values(
    "mean_absolute_shap_value",
    ascending=False
)

importance_path = OUTPUT_DIR / "shap_feature_importance.csv"
importance.to_csv(importance_path, index=False)

# SHAP summary plot
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)
plt.tight_layout()

summary_path = FIGURE_DIR / "shap_summary_plot.png"
plt.savefig(summary_path, dpi=300, bbox_inches="tight")
plt.close()

# SHAP bar plot
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)
plt.tight_layout()

bar_path = FIGURE_DIR / "shap_feature_importance.png"
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()

# Create a readable report
top_features = importance.head(5)

with open(REPORT_PATH, "w", encoding="utf-8") as report:
    report.write("METROPULSE - SHAP EXPLAINABILITY REPORT\n")
    report.write("=" * 50 + "\n\n")

    report.write("Model: Random Forest Demand Forecasting\n")
    report.write(f"Records analyzed: {sample_size}\n\n")

    report.write("TOP 5 INFLUENTIAL FEATURES\n")
    report.write("-" * 50 + "\n")

    for _, row in top_features.iterrows():
        report.write(
            f"{row['feature']}: "
            f"{row['mean_absolute_shap_value']:.4f}\n"
        )

    report.write("\n")
    report.write(
        "SHAP values describe each feature's contribution "
        "to individual model predictions.\n"
    )

print("\nSHAP explainability analysis completed!")

print("\nTop 5 influential features:")
print(top_features.to_string(index=False))

print("\nGenerated files:")
print(f"Feature importance: {importance_path}")
print(f"Summary plot: {summary_path}")
print(f"Bar plot: {bar_path}")
print(f"Report: {REPORT_PATH}")
