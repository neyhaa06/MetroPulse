
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TRIPS_FILE = (
    ROOT / "data" / "processed"
    / "yellow_trips_clean_2024_01.parquet"
)

SHAPEFILE = (
    ROOT / "data" / "external"
    / "taxi_zones" / "taxi_zones"
    / "taxi_zones.shp"
)


def main():

    print("Starting taxi-zone demand validation...")

    # 1. Load shapefile
    zones = gpd.read_file(SHAPEFILE)

    print("\nShapefile columns:")
    print(zones.columns.tolist())

    print("\nTotal geographical zones:", len(zones))

    # 2. Standardize column names
    zones = zones.rename(
        columns={
            "LocationID": "LocationID",
            "zone": "Zone",
            "borough": "Borough"
        }
    )

    zones["LocationID"] = pd.to_numeric(
        zones["LocationID"],
        errors="coerce"
    )

    # 3. Read pickup demand
    parquet_path = (
        TRIPS_FILE.as_posix().replace("'", "''")
    )

    con = duckdb.connect()

    total = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{parquet_path}')
        """
    ).fetchone()[0]

    demand = con.execute(
        f"""
        SELECT
            PULocationID AS LocationID,
            COUNT(*) AS pickup_trips
        FROM read_parquet('{parquet_path}')
        GROUP BY PULocationID
        """
    ).fetchdf()

    con.close()

    demand["LocationID"] = pd.to_numeric(
        demand["LocationID"],
        errors="coerce"
    )

    # 4. Compare dataset and shapefile
    print("\nTotal cleaned trips:", f"{total:,}")

    print(
        "Distinct pickup location IDs:",
        demand["LocationID"].nunique()
    )

    print(
        "Total pickup demand:",
        f"{demand['pickup_trips'].sum():,}"
    )

    matched = demand.merge(
        zones[["LocationID"]],
        on="LocationID",
        how="inner"
    )

    unmatched = demand[
        ~demand["LocationID"].isin(
            zones["LocationID"]
        )
    ]

    matched_total = int(
        matched["pickup_trips"].sum()
    )

    unmatched_total = int(
        unmatched["pickup_trips"].sum()
    )

    print(
        "\nMatched pickup demand:",
        f"{matched_total:,}"
    )

    print(
        "Unmatched pickup demand:",
        f"{unmatched_total:,}"
    )

    print("\nUnmatched location IDs:")
    print(unmatched.to_string(index=False))

    # 5. Check JFK Airport
    jfk = zones[
        zones["Zone"].astype(str).str.contains(
            "JFK Airport",
            case=False,
            na=False
        )
    ]

    print("\nJFK Airport shapefile records:")

    if jfk.empty:
        print("JFK Airport was not found.")
    else:
        print(
            jfk[
                ["LocationID", "Zone", "Borough"]
            ].to_string(index=False)
        )

        jfk_demand = demand.merge(
            jfk[["LocationID"]],
            on="LocationID",
            how="inner"
        )

        print("\nJFK pickup trips:")
        print(
            f"{jfk_demand['pickup_trips'].sum():,}"
        )

    # 6. Save validation report
    report = f"""
METROPULSE - TAXI ZONE VALIDATION
==================================

Total cleaned trips: {total:,}
Total geographical zones: {len(zones)}
Distinct pickup location IDs: {demand['LocationID'].nunique()}

Matched pickup demand: {matched_total:,}
Unmatched pickup demand: {unmatched_total:,}

Unmatched location IDs:
{unmatched.to_string(index=False)}

JFK Airport pickup demand:
{
    f"{jfk_demand['pickup_trips'].sum():,}"
    if not jfk.empty
    else "Not available"
}

==================================
END OF REPORT
"""

    output = ROOT / "reports" / "zone_validation_report.txt"

    output.write_text(
        report,
        encoding="utf-8"
    )

    print("\nValidation completed successfully!")
    print("Report saved:", output)


if __name__ == "__main__":
    main()
