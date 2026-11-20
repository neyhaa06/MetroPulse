
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd

from sklearn.cluster import DBSCAN


# Project paths
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

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def main():

    print("Starting DBSCAN parameter comparison...")

    # Load taxi zones
    zones = gpd.read_file(SHAPEFILE)

    zones = zones.to_crs(epsg=26918)

    zones = zones.rename(
        columns={"LocationID": "LocationID"}
    )

    # Load pickup demand
    parquet_path = TRIPS_FILE.as_posix().replace("'", "''")

    connection = duckdb.connect()

    demand = connection.execute(
        f"""
        SELECT
            PULocationID AS LocationID,
            COUNT(*) AS pickup_trips
        FROM read_parquet('{parquet_path}')
        GROUP BY PULocationID
        """
    ).fetchdf()

    connection.close()

    # Join demand with zones
    zones["LocationID"] = pd.to_numeric(
        zones["LocationID"],
        errors="coerce"
    )

    demand["LocationID"] = pd.to_numeric(
        demand["LocationID"],
        errors="coerce"
    )

    data = zones.merge(
        demand,
        on="LocationID",
        how="inner"
    )

    data = data[data["pickup_trips"] > 0].copy()

    # Calculate centroids in projected coordinates
    centroids = data.geometry.centroid

    coordinates = pd.DataFrame({
        "x": centroids.x,
        "y": centroids.y
    }).to_numpy()

    eps_values = [500, 750, 1000, 1250, 1500, 2000]

    min_samples = 3

    results = []

    for eps in eps_values:

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean"
        )

        labels = model.fit_predict(coordinates)

        data["cluster_id"] = labels

        number_of_clusters = len(
            set(labels) - {-1}
        )

        noise_mask = labels == -1

        clustered_mask = labels != -1

        noise_zones = int(noise_mask.sum())
        clustered_zones = int(clustered_mask.sum())

        noise_trips = int(
            data.loc[noise_mask, "pickup_trips"].sum()
        )

        clustered_trips = int(
            data.loc[clustered_mask, "pickup_trips"].sum()
        )

        results.append({
            "eps_metres": eps,
            "min_samples": min_samples,
            "number_of_clusters": number_of_clusters,
            "clustered_zones": clustered_zones,
            "noise_zones": noise_zones,
            "clustered_pickup_trips": clustered_trips,
            "noise_pickup_trips": noise_trips,
            "clustered_trip_percentage": round(
                clustered_trips
                / data["pickup_trips"].sum() * 100,
                2
            )
        })

        print(
            f"EPS={eps}m | "
            f"Clusters={number_of_clusters} | "
            f"Clustered zones={clustered_zones} | "
            f"Noise zones={noise_zones}"
        )

    # Save comparison
    comparison = pd.DataFrame(results)

    output_file = (
        REPORT_DIR / "dbscan_parameter_comparison.csv"
    )

    comparison.to_csv(output_file, index=False)

    print("\nParameter comparison completed!")
    print(comparison.to_string(index=False))
    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
