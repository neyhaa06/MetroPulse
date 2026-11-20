
from pathlib import Path
import duckdb

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "yellow_tripdata_2024-01.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_FILE = OUTPUT_DIR / "yellow_trips_clean_2024_01.parquet"
REPORT_FILE = PROJECT_ROOT / "reports" / "cleaning_report_jan2024.txt"


def main():
    if not RAW_FILE.exists():
        print("Raw dataset not found:", RAW_FILE)
        return

    if CLEAN_FILE.exists():
        print("A cleaned dataset already exists.")
        print("No existing file has been overwritten.")
        print("Review or rename it before running this script again.")
        return

    raw_path = RAW_FILE.as_posix().replace("'", "''")
    clean_path = CLEAN_FILE.as_posix().replace("'", "''")

    connection = duckdb.connect()

    print("Starting MetroPulse data cleaning...")

    # Original row count
    total_rows = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{raw_path}')
        """
    ).fetchone()[0]

    # Count records outside the expected pickup month
    outside_month = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{raw_path}')
        WHERE tpep_pickup_datetime < TIMESTAMP '2024-01-01'
           OR tpep_pickup_datetime >= TIMESTAMP '2024-02-01'
        """
    ).fetchone()[0]

    # Count invalid chronological records
    invalid_time = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{raw_path}')
        WHERE tpep_dropoff_datetime <= tpep_pickup_datetime
        """
    ).fetchone()[0]

    # Create a separate cleaned dataset.
    # Negative financial values and missing optional values are preserved.
    connection.execute(
        f"""
        COPY (
            SELECT
                *,
                date_diff(
                    'second',
                    tpep_pickup_datetime,
                    tpep_dropoff_datetime
                ) / 60.0 AS trip_duration_minutes,

                EXTRACT(
                    HOUR FROM tpep_pickup_datetime
                ) AS pickup_hour,

                CAST(
                    tpep_pickup_datetime AS DATE
                ) AS pickup_date,

                EXTRACT(
                    DOW FROM tpep_pickup_datetime
                ) AS pickup_day_of_week,

                fare_amount < 0 AS has_negative_fare,
                total_amount < 0 AS has_negative_total

            FROM read_parquet('{raw_path}')

            WHERE tpep_pickup_datetime >= TIMESTAMP '2024-01-01'
              AND tpep_pickup_datetime < TIMESTAMP '2024-02-01'
              AND tpep_dropoff_datetime > tpep_pickup_datetime

        )
        TO '{clean_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    # Final row count
    clean_rows = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{clean_path}')
        """
    ).fetchone()[0]

    # Check retained negative values
    negative_fares = connection.execute(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE has_negative_fare),
            COUNT(*) FILTER (WHERE has_negative_total)
        FROM read_parquet('{clean_path}')
        """
    ).fetchone()

    # Final date range
    date_range = connection.execute(
        f"""
        SELECT
            MIN(tpep_pickup_datetime),
            MAX(tpep_pickup_datetime)
        FROM read_parquet('{clean_path}')
        """
    ).fetchone()

    report = f"""
METROPULSE - DATA CLEANING REPORT
=================================

Original dataset: {RAW_FILE.name}
Cleaned dataset: {CLEAN_FILE.name}

1. ROW COUNTS
-------------
Original rows: {total_rows:,}
Pickups outside January 2024: {outside_month:,}
Invalid or zero-duration records: {invalid_time:,}
Final cleaned rows: {clean_rows:,}
Rows removed: {total_rows - clean_rows:,}

2. CLEANING RULES
-----------------
- Keep pickup timestamps within January 2024.
- Remove trips with drop-off time less than or equal to pickup time.
- Preserve missing optional values.
- Preserve negative fares and total amounts.
- Add trip duration and pickup time features.
- Add flags for negative financial values.
- Keep the original raw dataset unchanged.

3. FINAL DATE RANGE
-------------------
First pickup: {date_range[0]}
Last pickup: {date_range[1]}

4. RETAINED NEGATIVE VALUES
---------------------------
Negative fares: {negative_fares[0]:,}
Negative total amounts: {negative_fares[1]:,}

=================================
END OF REPORT
"""

    REPORT_FILE.write_text(report, encoding="utf-8")

    print("\nData cleaning completed successfully!")
    print(f"Original rows: {total_rows:,}")
    print(f"Cleaned rows: {clean_rows:,}")
    print(f"Rows removed: {total_rows - clean_rows:,}")
    print(f"\nCleaned dataset: {CLEAN_FILE}")
    print(f"Cleaning report: {REPORT_FILE}")

    connection.close()


if __name__ == "__main__":
    main()
