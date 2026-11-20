
from pathlib import Path

import duckdb
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRIPS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yellow_trips_clean_2024_01.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("Starting MetroPulse hourly demand aggregation...")

    if not TRIPS_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {TRIPS_FILE}"
        )

    parquet_path = (
        TRIPS_FILE.as_posix().replace("'", "''")
    )

    # --------------------------------------------------
    # 3. AGGREGATE HOURLY PICKUP DEMAND
    # --------------------------------------------------

    connection = duckdb.connect()

    hourly_demand = connection.execute(
        f"""
        SELECT
            date_trunc('hour', tpep_pickup_datetime)
                AS pickup_hour,
            COUNT(*) AS total_trips
        FROM read_parquet('{parquet_path}')
        WHERE tpep_pickup_datetime IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()

    connection.close()

    if hourly_demand.empty:
        raise ValueError("No hourly demand records were generated.")

    hourly_demand["pickup_hour"] = pd.to_datetime(
        hourly_demand["pickup_hour"]
    )

    hourly_demand["total_trips"] = (
        hourly_demand["total_trips"].astype(int)
    )

    # --------------------------------------------------
    # 4. CHECK HOURLY DATA
    # --------------------------------------------------

    busiest_hour = hourly_demand.loc[
        hourly_demand["total_trips"].idxmax()
    ]

    average_demand = hourly_demand["total_trips"].mean()

    minimum_demand = hourly_demand["total_trips"].min()

    maximum_demand = hourly_demand["total_trips"].max()

    # --------------------------------------------------
    # 5. SAVE PROCESSED DATA
    # --------------------------------------------------

    output_file = (
        OUTPUT_DIR / "hourly_demand_2024_01.csv"
    )

    hourly_demand.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------
    # 6. GENERATE SUMMARY REPORT
    # --------------------------------------------------

    report = f"""
METROPULSE - HOURLY TAXI DEMAND AGGREGATION
============================================

Dataset: January 2024 Yellow Taxi Trips

1. HOURLY DEMAND SUMMARY
------------------------
Total hourly records: {len(hourly_demand)}
Average hourly demand: {average_demand:.2f}
Minimum hourly demand: {minimum_demand}
Maximum hourly demand: {maximum_demand}

2. BUSIEST HOUR
---------------
Date and hour: {busiest_hour['pickup_hour']}
Total trips: {busiest_hour['total_trips']}

3. DATA OUTPUT
--------------
{output_file}

============================================
END OF REPORT
"""

    report_file = (
        REPORT_DIR / "hourly_demand_report.txt"
    )

    report_file.write_text(
        report,
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 7. DISPLAY RESULTS
    # --------------------------------------------------

    print("\nHourly aggregation completed!")

    print("Total hourly records:", len(hourly_demand))
    print("Average hourly demand:", round(average_demand, 2))
    print("Busiest hour:", busiest_hour["pickup_hour"])
    print("Trips:", busiest_hour["total_trips"])

    print("\nFiles saved:")
    print(output_file)
    print(report_file)


if __name__ == "__main__":
    main()
