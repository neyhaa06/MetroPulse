
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import folium
import branca.colormap as cm


# --------------------------------------------------
# PROJECT PATHS
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


def main():

    # --------------------------------------------------
    # 1. VERIFY INPUT FILES
    # --------------------------------------------------

    if not TRIPS_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned trips file not found: {TRIPS_FILE}"
        )

    if not SHAPEFILE.exists():
        raise FileNotFoundError(
            f"Taxi zone shapefile not found: {SHAPEFILE}"
        )

    print("Starting MetroPulse geospatial analysis...")

    # --------------------------------------------------
    # 2. LOAD TAXI ZONE GEOGRAPHY
    # --------------------------------------------------

    zones = gpd.read_file(SHAPEFILE)

    print("\nTaxi zone shapefile loaded.")
    print("Number of geographical zones:", len(zones))
    print("Available columns:", zones.columns.tolist())
    print("Original CRS:", zones.crs)

    if zones.crs is None:
        raise ValueError(
            "The shapefile has no CRS information."
        )

    # Convert to latitude/longitude coordinates
    zones = zones.to_crs(epsg=4326)

    # Identify the zone ID column
    location_column = next(
        (
            column for column in zones.columns
            if column.lower() == "locationid"
        ),
        None
    )

    if location_column is None:
        raise ValueError(
            "LocationID column not found in shapefile."
        )

    # --------------------------------------------------
    # 3. AGGREGATE PICKUP DEMAND
    # --------------------------------------------------

    parquet_path = (
        TRIPS_FILE.as_posix().replace("'", "''")
    )

    connection = duckdb.connect()

    pickup_demand = connection.execute(
        f"""
        SELECT
            PULocationID AS LocationID,
            COUNT(*) AS pickup_trips
        FROM read_parquet('{parquet_path}')
        GROUP BY PULocationID
        """
    ).fetchdf()

    connection.close()

    print("\nPickup demand aggregation completed.")

    # --------------------------------------------------
    # 4. JOIN DEMAND WITH GEOGRAPHICAL ZONES
    # --------------------------------------------------

    zones["LocationID"] = pd.to_numeric(
        zones[location_column],
        errors="coerce"
    )

    pickup_demand["LocationID"] = pd.to_numeric(
        pickup_demand["LocationID"],
        errors="coerce"
    )

    map_zones = zones.merge(
        pickup_demand,
        on="LocationID",
        how="left"
    )

    map_zones["pickup_trips"] = (
        map_zones["pickup_trips"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------
    # 5. IDENTIFY TOP HOTSPOTS
    # --------------------------------------------------

    top_zones = (
        map_zones[
            ["LocationID", "zone", "borough", "pickup_trips"]
        ]
        .sort_values(
            "pickup_trips",
            ascending=False
        )
        .head(10)
        .copy()
    )

    top_zones = top_zones.rename(
        columns={
            "zone": "Zone",
            "borough": "Borough"
        }
    )

    top_zones.to_csv(
        REPORT_DIR / "geospatial_hotspots.csv",
        index=False
    )

    # --------------------------------------------------
    # 6. CREATE STATIC HOTSPOT MAP
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(13, 10))

    map_zones.plot(
        column="pickup_trips",
        cmap="YlOrRd",
        linewidth=0.25,
        edgecolor="white",
        legend=True,
        legend_kwds={
            "label": "Number of Pickup Trips",
            "shrink": 0.65
        },
        ax=ax
    )

    ax.set_title(
        "NYC Taxi Pickup Demand Hotspots - January 2024",
        fontsize=15,
        pad=15
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "nyc_pickup_hotspots.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # 7. CREATE INTERACTIVE FOLIUM MAP
    # --------------------------------------------------

    center = [40.75, -73.90]
    taxi_map = folium.Map(
    location=center,
    zoom_start=10,
    tiles=None,
    control_scale=True
)
    maximum = max(
        int(map_zones["pickup_trips"].max()),
        1
    )

    color_scale = cm.linear.YlOrRd_09.scale(
        0,
        maximum
    )

    color_scale.caption = (
        "January 2024 Taxi Pickup Trips"
    )

    def style_function(feature):

        trips = feature["properties"]["pickup_trips"]

        return {
            "fillColor": color_scale(trips),
            "color": "#555555",
            "weight": 0.4,
            "fillOpacity": 0.7
        }

    geojson_data = map_zones[
        ["zone", "borough", "pickup_trips", "geometry"]
    ].to_json()

    folium.GeoJson(
        geojson_data,
        name="Taxi Pickup Demand",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "zone",
                "borough",
                "pickup_trips"
            ],
            aliases=[
                "Zone:",
                "Borough:",
                "Pickup Trips:"
            ],
            localize=True
        )
    ).add_to(taxi_map)

    color_scale.add_to(taxi_map)

    taxi_map.save(
        str(REPORT_DIR / "nyc_pickup_hotspots.html")
    )

    # --------------------------------------------------
    # 8. GENERATE SUMMARY REPORT
    # --------------------------------------------------

    total_trips = int(
        pickup_demand["pickup_trips"].sum()
    )

    top_zone = top_zones.iloc[0]

    report = f"""
METROPULSE - GEOSPATIAL HOTSPOT ANALYSIS
=========================================

Dataset: January 2024 Yellow Taxi Trips

1. GEOGRAPHICAL COVERAGE
------------------------
Taxi zones in shapefile: {len(zones)}
Zones with pickup demand: {
    int((map_zones["pickup_trips"] > 0).sum())
}

2. TOTAL PICKUP TRIPS
---------------------
{total_trips:,}

3. TOP PICKUP HOTSPOT
---------------------
Zone: {top_zone['Zone']}
Borough: {top_zone['Borough']}
Location ID: {top_zone['LocationID']}
Pickup trips: {int(top_zone['pickup_trips']):,}

4. TOP 10 PICKUP HOTSPOTS
-------------------------
{top_zones.to_string(index=False)}

5. GENERATED VISUALIZATIONS
---------------------------
Static map: reports/figures/nyc_pickup_hotspots.png
Interactive map: reports/nyc_pickup_hotspots.html

=========================================
END OF REPORT
"""

    (
        REPORT_DIR / "geospatial_summary.txt"
    ).write_text(
        report,
        encoding="utf-8"
    )

    print("\nGeospatial analysis completed successfully!")

    print("\nTop 10 pickup hotspots:")
    print(top_zones.to_string(index=False))

    print("\nGenerated files:")
    print("1. reports/geospatial_summary.txt")
    print("2. reports/geospatial_hotspots.csv")
    print("3. reports/figures/nyc_pickup_hotspots.png")
    print("4. reports/nyc_pickup_hotspots.html")


if __name__ == "__main__":
    main()
