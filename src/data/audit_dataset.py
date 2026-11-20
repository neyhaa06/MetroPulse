
from pathlib import Path
import duckdb
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "yellow_tripdata_2024-01.parquet"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

REPORT_FILE = REPORT_DIR / "data_audit_jan2024.txt"
SUMMARY_FILE = REPORT_DIR / "data_quality_summary.csv"


def main():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    print("Starting MetroPulse dataset audit...")
    print("Dataset:", DATA_FILE.name)

    connection = duckdb.connect()

    # Inspect dataset structure
    schema = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{DATA_FILE.as_posix()}')"
    ).fetchdf()

    columns = schema["column_name"].tolist()

    # Total rows
    total_rows = connection.execute(
        f"SELECT COUNT(*) FROM read_parquet('{DATA_FILE.as_posix()}')"
    ).fetchone()[0]

    # Missing values
    null_expressions = ", ".join(
        f'COUNT(*) FILTER (WHERE "{column}" IS NULL) AS "{column}"'
        for column in columns
    )

    null_result = connection.execute(
        f"SELECT {null_expressions} "
        f"FROM read_parquet('{DATA_FILE.as_posix()}')"
    ).fetchdf()

    missing_values = null_result.iloc[0]

    # Exact duplicate rows
    distinct_rows = connection.execute(
        f"SELECT COUNT(*) FROM "
        f"(SELECT DISTINCT * FROM read_parquet('{DATA_FILE.as_posix()}'))"
    ).fetchone()[0]

    duplicate_rows = total_rows - distinct_rows

    # Date range
    date_result = connection.execute(
        f"""
        SELECT
            MIN(tpep_pickup_datetime),
            MAX(tpep_pickup_datetime),
            MIN(tpep_dropoff_datetime),
            MAX(tpep_dropoff_datetime)
        FROM read_parquet('{DATA_FILE.as_posix()}')
        """
    ).fetchone()

    # Invalid trip durations
    duration_result = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{DATA_FILE.as_posix()}')
        WHERE tpep_dropoff_datetime < tpep_pickup_datetime
        """
    ).fetchone()[0]

    # Negative fares and distances
    invalid_result = connection.execute(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE fare_amount < 0),
            COUNT(*) FILTER (WHERE trip_distance < 0),
            COUNT(*) FILTER (WHERE total_amount < 0)
        FROM read_parquet('{DATA_FILE.as_posix()}')
        """
    ).fetchone()

    # Basic statistics
    statistics = connection.execute(
        f"""
        SELECT
            AVG(trip_distance),
            AVG(fare_amount),
            AVG(total_amount),
            MIN(passenger_count),
            MAX(passenger_count)
        FROM read_parquet('{DATA_FILE.as_posix()}')
        """
    ).fetchone()

    # Save column-level quality summary
    summary = pd.DataFrame({
        "column": columns,
        "missing_values": [
            int(missing_values[column]) for column in columns
        ]
    })

    summary["missing_percentage"] = (
        summary["missing_values"] / total_rows * 100
    ).round(2)

    summary.to_csv(SUMMARY_FILE, index=False)

    # Build audit report
    report = f"""
METROPULSE - DATASET AUDIT REPORT
=================================

Dataset: {DATA_FILE.name}

1. DATASET OVERVIEW
-------------------
Total rows: {total_rows:,}
Total columns: {len(columns)}
Exact duplicate rows: {duplicate_rows:,}

2. DATASET COLUMNS
------------------
{chr(10).join(columns)}

3. DATE RANGE
-------------
Pickup start: {date_result[0]}
Pickup end: {date_result[1]}
Drop-off start: {date_result[2]}
Drop-off end: {date_result[3]}

4. MISSING VALUES
-----------------
{summary.to_string(index=False)}

5. DATA QUALITY CHECKS
----------------------
Drop-off before pickup: {duration_result:,}
Negative fare amounts: {invalid_result[0]:,}
Negative trip distances: {invalid_result[1]:,}
Negative total amounts: {invalid_result[2]:,}

6. BASIC STATISTICS
-------------------
Average trip distance: {statistics[0]:.2f}
Average fare amount: {statistics[1]:.2f}
Average total amount: {statistics[2]:.2f}
Minimum passenger count: {statistics[3]}
Maximum passenger count: {statistics[4]}

=================================
END OF REPORT
"""

    REPORT_FILE.write_text(report, encoding="utf-8")

    print("\nDataset audit completed successfully!")
    print(f"Total rows: {total_rows:,}")
    print(f"Total columns: {len(columns)}")
    print(f"Exact duplicate rows: {duplicate_rows:,}")
    print(f"Audit report: {REPORT_FILE}")
    print(f"Quality summary: {SUMMARY_FILE}")

    connection.close()


if __name__ == "__main__":
    main()
