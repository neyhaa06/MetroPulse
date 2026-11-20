
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from matplotlib import colormaps
import folium


# --------------------------------------------------
# 1. PROJECT PATHS AND PARAMETERS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRIPS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yellow_trips_clean_2024_01.parquet"
)

SHAPEFILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "taxi_zones"
    / "taxi_zones"
    / "taxi_zones.shp"
)

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EPS_METERS = 1500
MIN_SAMPLES = 3


# --------------------------------------------------
# 2. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("Starting MetroPulse DBSCAN clustering...")

    if not TRIPS_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {TRIPS_FILE}"
        )

    if not SHAPEFILE.exists():
        raise FileNotFoundError(
            f"Taxi zone shapefile not found: {SHAPEFILE}"
        )

    # --------------------------------------------------
    # 3. LOAD TAXI ZONE GEOGRAPHY
    # --------------------------------------------------

    zones = gpd.read_file(SHAPEFILE)

    if zones.crs is None:
        raise ValueError("Taxi zone shapefile has no CRS.")

    zones = zones.to_crs(epsg=4326)

    # Identify important columns
    columns = {
        column.lower(): column
        for column in zones.columns
    }

    required = ["locationid", "zone", "borough"]

    for column in required:
        if column not in columns:
            raise ValueError(
                f"Required shapefile column missing: {column}"
            )

    zones = zones.rename(
        columns={
            columns["locationid"]: "LocationID",
            columns["zone"]: "Zone",
            columns["borough"]: "Borough"
        }
    )

    zones["LocationID"] = pd.to_numeric(
        zones["LocationID"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 4. AGGREGATE PICKUP DEMAND
    # --------------------------------------------------

    parquet_path = (
        TRIPS_FILE.as_posix().replace("'", "''")
    )

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

    demand["LocationID"] = pd.to_numeric(
        demand["LocationID"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 5. JOIN DEMAND WITH TAXI ZONES
    # --------------------------------------------------

    geo_data = zones.merge(
        demand,
        on="LocationID",
        how="left"
    )

    geo_data["pickup_trips"] = (
        geo_data["pickup_trips"]
        .fillna(0)
        .astype(int)
    )

    # Cluster only zones with observed pickup demand
    active_zones = geo_data[
        geo_data["pickup_trips"] > 0
    ].copy()

    if len(active_zones) < MIN_SAMPLES:
        raise ValueError(
            "Not enough active zones for DBSCAN."
        )

    # --------------------------------------------------
    # 6. CALCULATE GEOGRAPHICAL CENTROIDS
    # --------------------------------------------------

    # EPSG:26918 uses metres for New York coordinates.
    projected = active_zones.to_crs(epsg=26918)

    centroids = projected.geometry.centroid

    coordinates = pd.DataFrame({
        "x": centroids.x,
        "y": centroids.y
    }).to_numpy()

    # --------------------------------------------------
    # 7. APPLY DBSCAN
    # --------------------------------------------------

    model = DBSCAN(
        eps=EPS_METERS,
        min_samples=MIN_SAMPLES,
        metric="euclidean"
    )

    cluster_labels = model.fit_predict(coordinates)

    active_zones["cluster_id"] = cluster_labels

    # --------------------------------------------------
    # 8. ADD CLUSTER RESULTS TO ALL ZONES
    # --------------------------------------------------

    geo_data["cluster_id"] = -1

    geo_data.loc[
        active_zones.index,
        "cluster_id"
    ] = cluster_labels

    geo_data["cluster_type"] = geo_data[
        "cluster_id"
    ].apply(
        lambda value: (
            "Noise / Unclustered"
            if value == -1
            else f"Cluster {value}"
        )
    )

    # --------------------------------------------------
    # 9. GENERATE CLUSTER SUMMARY
    # --------------------------------------------------

    cluster_summary = (
        active_zones[
            active_zones["cluster_id"] != -1
        ]
        .groupby("cluster_id")
        .agg(
            number_of_zones=("LocationID", "count"),
            total_pickup_trips=("pickup_trips", "sum"),
            average_pickup_trips=("pickup_trips", "mean"),
            maximum_zone_demand=("pickup_trips", "max")
        )
        .reset_index()
        .sort_values(
            "total_pickup_trips",
            ascending=False
        )
    )

    cluster_summary["average_pickup_trips"] = (
        cluster_summary["average_pickup_trips"].round(2)
    )

    cluster_summary.to_csv(
        REPORT_DIR / "dbscan_cluster_summary.csv",
        index=False
    )

    # --------------------------------------------------
    # 10. SAVE ZONE-LEVEL CLUSTER ASSIGNMENTS
    # --------------------------------------------------

    zone_results = pd.DataFrame({
        "LocationID": geo_data["LocationID"],
        "Zone": geo_data["Zone"],
        "Borough": geo_data["Borough"],
        "pickup_trips": geo_data["pickup_trips"],
        "cluster_id": geo_data["cluster_id"],
        "cluster_type": geo_data["cluster_type"]
    })

    zone_results.to_csv(
        REPORT_DIR / "dbscan_zone_assignments.csv",
        index=False
    )

    # --------------------------------------------------
    # 11. PREPARE MAP COLORS
    # --------------------------------------------------

    cluster_ids = sorted(
        value
        for value in geo_data["cluster_id"].unique()
        if value != -1
    )

    palette = colormaps["tab20"]

    cluster_colors = {
        cluster_id: (
            "#"
            + "".join(
                f"{int(channel * 255):02x}"
                for channel in palette(
                    index % 20
                )[:3]
            )
        )
        for index, cluster_id in enumerate(cluster_ids)
    }

    def get_cluster_color(cluster_id):
        if cluster_id == -1:
            return "#BDBDBD"

        return cluster_colors[cluster_id]

    geo_data["cluster_color"] = (
        geo_data["cluster_id"].apply(get_cluster_color)
    )

    # --------------------------------------------------
    # 12. STATIC CLUSTER MAP
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(13, 10))

    geo_data.plot(
        color=geo_data["cluster_color"],
        edgecolor="white",
        linewidth=0.3,
        ax=ax
    )

    ax.set_title(
        "NYC Taxi Zone Clusters - DBSCAN - January 2024",
        fontsize=15,
        pad=15
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "dbscan_clusters.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # 13. INTERACTIVE CLUSTER MAP
    # --------------------------------------------------

    taxi_map = folium.Map(
        location=[40.75, -73.90],
        zoom_start=10,
        tiles=None,
        control_scale=True
    )

    geojson_data = geo_data[
        [
            "Zone",
            "Borough",
            "pickup_trips",
            "cluster_id",
            "cluster_type",
            "cluster_color",
            "geometry"
        ]
    ].to_json()

    def style_function(feature):

        color = feature["properties"]["cluster_color"]

        return {
            "fillColor": color,
            "color": "#555555",
            "weight": 0.5,
            "fillOpacity": 0.75
        }

    folium.GeoJson(
        geojson_data,
        name="DBSCAN Clusters",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "Zone",
                "Borough",
                "pickup_trips",
                "cluster_type"
            ],
            aliases=[
                "Zone:",
                "Borough:",
                "Pickup Trips:",
                "Cluster:"
            ],
            localize=True
        )
    ).add_to(taxi_map)

    folium.LayerControl().add_to(taxi_map)

    taxi_map.save(
        str(REPORT_DIR / "dbscan_clusters.html")
    )

    # --------------------------------------------------
    # 14. CLUSTER PERFORMANCE METRICS
    # --------------------------------------------------

    total_zones = len(active_zones)

    clustered_zones = int(
        (active_zones["cluster_id"] != -1).sum()
    )

    noise_zones = int(
        (active_zones["cluster_id"] == -1).sum()
    )

    number_of_clusters = len(cluster_ids)

    # --------------------------------------------------
    # 15. GENERATE REPORT
    # --------------------------------------------------

    report = f"""
METROPULSE - DBSCAN SPATIAL CLUSTERING
=======================================

Dataset: January 2024 Yellow Taxi Trips

1. MODEL CONFIGURATION
----------------------
Algorithm: DBSCAN
Distance metric: Euclidean
Epsilon: {EPS_METERS} metres
Minimum samples: {MIN_SAMPLES}
Input: Active taxi-zone centroids
Coordinate system: EPSG:26918

2. CLUSTERING RESULTS
---------------------
Active taxi zones: {total_zones}
Number of clusters: {number_of_clusters}
Zones assigned to clusters: {clustered_zones}
Noise / unclustered zones: {noise_zones}

3. CLUSTER SUMMARY
------------------
{cluster_summary.to_string(index=False)}

4. OUTPUT FILES
---------------
reports/dbscan_cluster_summary.csv
reports/dbscan_zone_assignments.csv
reports/dbscan_clusters.html
reports/figures/dbscan_clusters.png

=======================================
END OF REPORT
"""

    (
        REPORT_DIR / "dbscan_clustering_report.txt"
    ).write_text(
        report,
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 16. DISPLAY RESULTS
    # --------------------------------------------------

    print("\nDBSCAN clustering completed!")

    print("\nModel configuration:")
    print("Epsilon:", EPS_METERS, "metres")
    print("Minimum samples:", MIN_SAMPLES)

    print("\nNumber of clusters:", number_of_clusters)
    print("Clustered zones:", clustered_zones)
    print("Noise zones:", noise_zones)

    print("\nCluster summary:")
    print(cluster_summary.to_string(index=False))

    print("\nReports saved in:", REPORT_DIR)

    print("\nMetroPulse DBSCAN analysis completed successfully!")


if __name__ == "__main__":
    main()
