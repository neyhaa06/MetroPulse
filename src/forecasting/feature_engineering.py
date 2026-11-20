
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_demand_2024_01.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demand_features_2024_01.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "feature_engineering_report.txt"
)


# --------------------------------------------------
# 2. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("Starting MetroPulse feature engineering...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Hourly demand file not found: {INPUT_FILE}"
        )

    # --------------------------------------------------
    # 3. LOAD AND VALIDATE DATA
    # --------------------------------------------------

    data = pd.read_csv(INPUT_FILE)

    data["pickup_hour"] = pd.to_datetime(
        data["pickup_hour"]
    )

    data = data.sort_values(
        "pickup_hour"
    ).reset_index(drop=True)

    if data["pickup_hour"].duplicated().any():
        raise ValueError("Duplicate hourly timestamps found.")

    expected_hours = pd.date_range(
        start=data["pickup_hour"].min(),
        end=data["pickup_hour"].max(),
        freq="h"
    )

    if not data["pickup_hour"].reset_index(drop=True).equals(
        pd.Series(expected_hours, name="pickup_hour")
    ):
        raise ValueError(
            "Hourly timestamps contain missing or irregular hours."
        )

    # --------------------------------------------------
    # 4. CALENDAR FEATURES
    # --------------------------------------------------

    data["hour"] = data["pickup_hour"].dt.hour

    data["day_of_week"] = (
        data["pickup_hour"].dt.day_of_week
    )

    data["day_of_month"] = (
        data["pickup_hour"].dt.day
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Morning: 7 AM to 10 AM
    # Evening: 4 PM to 8 PM

    data["is_rush_hour"] = (
        data["hour"].between(7, 10)
        | data["hour"].between(16, 20)
    ).astype(int)

    # --------------------------------------------------
    # 5. CYCLICAL TIME FEATURES
    # --------------------------------------------------

    import numpy as np

    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    data["day_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["day_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    # --------------------------------------------------
    # 6. LAG FEATURES
    # --------------------------------------------------

    data["lag_1"] = (
        data["total_trips"].shift(1)
    )

    data["lag_24"] = (
        data["total_trips"].shift(24)
    )

    data["lag_168"] = (
        data["total_trips"].shift(168)
    )

    # --------------------------------------------------
    # 7. ROLLING AVERAGE FEATURES
    # --------------------------------------------------

    previous_demand = data["total_trips"].shift(1)

    data["rolling_mean_24"] = (
        previous_demand
        .rolling(window=24)
        .mean()
    )

    data["rolling_mean_168"] = (
        previous_demand
        .rolling(window=168)
        .mean()
    )

    # --------------------------------------------------
    # 8. REMOVE ROWS WITHOUT COMPLETE FEATURES
    # --------------------------------------------------

    original_rows = len(data)

    data = data.dropna().reset_index(drop=True)

    removed_rows = original_rows - len(data)

    if data.empty:
        raise ValueError(
            "No records remain after feature engineering."
        )

    # --------------------------------------------------
    # 9. SAVE FEATURE DATASET
    # --------------------------------------------------

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # 10. GENERATE REPORT
    # --------------------------------------------------

    feature_columns = [
        column
        for column in data.columns
        if column not in [
            "pickup_hour",
            "total_trips"
        ]
    ]

    report = f"""
METROPULSE - FEATURE ENGINEERING REPORT
========================================

Dataset: January 2024 Yellow Taxi Trips

1. INPUT DATA
-------------
Original hourly records: {original_rows}

2. FEATURE ENGINEERING
----------------------
Calendar features:
- Hour
- Day of week
- Day of month
- Weekend indicator
- Rush-hour indicator

Cyclical features:
- Hour sine and cosine
- Day sine and cosine

Lag features:
- Previous 1-hour demand
- Previous 24-hour demand
- Previous 168-hour demand

Rolling features:
- Previous 24-hour average demand
- Previous 168-hour average demand

3. DATA CLEANING
----------------
Rows removed due to unavailable lag/rolling values: {removed_rows}
Final feature records: {len(data)}

4. FEATURE COLUMNS
------------------
{chr(10).join(feature_columns)}

5. OUTPUT
---------
{OUTPUT_FILE}

========================================
END OF REPORT
"""

    REPORT_FILE.write_text(
        report,
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 11. DISPLAY RESULTS
    # --------------------------------------------------

    print("\nFeature engineering completed!")

    print("Original records:", original_rows)
    print("Removed records:", removed_rows)
    print("Final feature records:", len(data))

    print("\nFeatures created:")
    for column in feature_columns:
        print("-", column)

    print("\nFiles saved:")
    print(OUTPUT_FILE)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()
