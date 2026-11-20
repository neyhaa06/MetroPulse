
from pathlib import Path

import duckdb
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yellow_trips_clean_2024_01.parquet"
)

ZONE_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "taxi_zone_lookup.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main():

    if not CLEAN_FILE.exists():
        print("Cleaned dataset not found.")
        return

    if not ZONE_FILE.exists():
        print("Taxi zone lookup file not found.")
        return

    print("Starting MetroPulse exploratory data analysis...")

    parquet_path = CLEAN_FILE.as_posix().replace("'", "''")

    connection = duckdb.connect()

    # --------------------------------------------------
    # 1. HOURLY DEMAND
    # --------------------------------------------------

    hourly = connection.execute(
        f"""
        SELECT
            pickup_hour,
            COUNT(*) AS total_trips
        FROM read_parquet('{parquet_path}')
        GROUP BY pickup_hour
        ORDER BY pickup_hour
        """
    ).fetchdf()

    hourly.to_csv(
        REPORT_DIR / "hourly_demand.csv",
        index=False
    )

    # --------------------------------------------------
    # 2. DAILY DEMAND
    # --------------------------------------------------

    daily = connection.execute(
        f"""
        SELECT
            pickup_date,
            COUNT(*) AS total_trips
        FROM read_parquet('{parquet_path}')
        GROUP BY pickup_date
        ORDER BY pickup_date
        """
    ).fetchdf()

    daily.to_csv(
        REPORT_DIR / "daily_demand.csv",
        index=False
    )

    # --------------------------------------------------
    # 3. WEEKDAY DEMAND
    # --------------------------------------------------

    weekday = connection.execute(
        f"""
        SELECT
            EXTRACT(DOW FROM pickup_date) AS day_number,
            strftime(pickup_date, '%A') AS day_name,
            COUNT(*) AS total_trips
        FROM read_parquet('{parquet_path}')
        GROUP BY day_number, day_name
        ORDER BY day_number
        """
    ).fetchdf()

    weekday.to_csv(
        REPORT_DIR / "weekday_demand.csv",
        index=False
    )

    # --------------------------------------------------
    # 4. TOP PICKUP ZONES
    # --------------------------------------------------

    zones = connection.execute(
        f"""
        SELECT
            PULocationID,
            COUNT(*) AS total_trips
        FROM read_parquet('{parquet_path}')
        GROUP BY PULocationID
        ORDER BY total_trips DESC
        LIMIT 10
        """
    ).fetchdf()

    zone_lookup = pd.read_csv(ZONE_FILE)

    top_zones = zones.merge(
        zone_lookup,
        left_on="PULocationID",
        right_on="LocationID",
        how="left"
    )

    top_zones["Zone"] = top_zones["Zone"].fillna("Unknown")
    top_zones["Borough"] = top_zones["Borough"].fillna("Unknown")

    top_zones = top_zones[
        ["PULocationID", "Zone", "Borough", "total_trips"]
    ]

    top_zones.to_csv(
        REPORT_DIR / "top_pickup_zones.csv",
        index=False
    )

    # --------------------------------------------------
    # 5. SUMMARY
    # --------------------------------------------------

    total_trips = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{parquet_path}')
        """
    ).fetchone()[0]

    busiest_hour = hourly.loc[hourly["total_trips"].idxmax()]
    busiest_day = daily.loc[daily["total_trips"].idxmax()]
    busiest_weekday = weekday.loc[weekday["total_trips"].idxmax()]
    busiest_zone = top_zones.iloc[0]

    summary = f"""
METROPULSE - EXPLORATORY DATA ANALYSIS
======================================

Dataset: January 2024 Yellow Taxi Trips

1. TOTAL TRIPS
--------------
{total_trips:,}

2. HOURLY DEMAND
----------------
Busiest pickup hour: {int(busiest_hour['pickup_hour'])}:00
Trips: {int(busiest_hour['total_trips']):,}

3. DAILY DEMAND
---------------
Busiest date: {busiest_day['pickup_date']}
Trips: {int(busiest_day['total_trips']):,}

4. WEEKDAY DEMAND
-----------------
Busiest weekday: {busiest_weekday['day_name']}
Trips: {int(busiest_weekday['total_trips']):,}

5. TOP PICKUP ZONE
------------------
Zone: {busiest_zone['Zone']}
Borough: {busiest_zone['Borough']}
Trips: {int(busiest_zone['total_trips']):,}

======================================
END OF REPORT
"""

    (REPORT_DIR / "eda_summary.txt").write_text(
        summary,
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 6. VISUALIZATION - HOURLY DEMAND
    # --------------------------------------------------

    plt.figure(figsize=(11, 5))

    plt.bar(
        hourly["pickup_hour"],
        hourly["total_trips"]
    )

    plt.title("NYC Taxi Demand by Hour - January 2024")
    plt.xlabel("Pickup Hour")
    plt.ylabel("Number of Trips")
    plt.xticks(range(24))
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "hourly_demand.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 7. VISUALIZATION - DAILY DEMAND
    # --------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        daily["pickup_date"],
        daily["total_trips"],
        marker="o"
    )

    plt.title("Daily Taxi Demand - January 2024")
    plt.xlabel("Date")
    plt.ylabel("Number of Trips")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "daily_demand.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 8. VISUALIZATION - WEEKDAY DEMAND
    # --------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.bar(
        weekday["day_name"],
        weekday["total_trips"]
    )

    plt.title("Taxi Demand by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Number of Trips")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "weekday_demand.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 9. VISUALIZATION - TOP PICKUP ZONES
    # --------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.barh(
        top_zones["Zone"].iloc[::-1],
        top_zones["total_trips"].iloc[::-1]
    )

    plt.title("Top 10 NYC Taxi Pickup Zones")
    plt.xlabel("Number of Trips")
    plt.ylabel("Pickup Zone")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "top_pickup_zones.png",
        dpi=300
    )

    plt.close()

    connection.close()

    print("\nEDA completed successfully!")
    print(f"Total trips analyzed: {total_trips:,}")
    print(f"Busiest hour: {int(busiest_hour['pickup_hour'])}:00")
    print(f"Busiest weekday: {busiest_weekday['day_name']}")
    print(f"EDA reports saved in: {REPORT_DIR}")
    print(f"Charts saved in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
