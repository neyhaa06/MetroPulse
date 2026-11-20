
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT_DIR / "reports" / "fleet_allocation_results.csv"

OUTPUT_DIR = ROOT_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD OPTIMIZATION RESULTS
# --------------------------------------------------

print("Starting fleet allocation visualization...")

zones = pd.read_csv(INPUT_PATH)

required_columns = [
    "Zone",
    "Borough",
    "average_hourly_demand",
    "allocated_vehicles",
    "served_hourly_demand",
    "unmet_hourly_demand",
    "demand_coverage_percentage"
]

missing_columns = [
    column for column in required_columns
    if column not in zones.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print(f"Zones loaded: {len(zones)}")


# --------------------------------------------------
# 1. TOP 15 ZONES BY VEHICLE ALLOCATION
# --------------------------------------------------

top_zones = zones.nlargest(
    15,
    "allocated_vehicles"
).sort_values("allocated_vehicles")

plt.figure(figsize=(12, 7))

plt.barh(
    top_zones["Zone"],
    top_zones["allocated_vehicles"]
)

plt.xlabel("Number of Vehicles")
plt.ylabel("Taxi Zone")
plt.title("Top 15 Taxi Zones by Fleet Allocation")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fleet_allocation_by_zone.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 2. DEMAND VS SERVED DEMAND
# --------------------------------------------------

top_demand = zones.nlargest(
    15,
    "average_hourly_demand"
).sort_values("average_hourly_demand")

plt.figure(figsize=(13, 7))

plt.bar(
    top_demand["Zone"],
    top_demand["average_hourly_demand"],
    label="Average Hourly Demand"
)

plt.bar(
    top_demand["Zone"],
    top_demand["served_hourly_demand"],
    label="Served Demand"
)

plt.xlabel("Taxi Zone")
plt.ylabel("Trips per Hour")
plt.title("Demand vs Served Demand – Top 15 Zones")

plt.xticks(rotation=65, ha="right")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fleet_demand_vs_served.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 3. DEMAND COVERAGE BY ZONE
# --------------------------------------------------

coverage_zones = zones.nlargest(
    20,
    "average_hourly_demand"
).sort_values("demand_coverage_percentage")

plt.figure(figsize=(12, 8))

plt.barh(
    coverage_zones["Zone"],
    coverage_zones["demand_coverage_percentage"]
)

plt.xlabel("Demand Coverage (%)")
plt.ylabel("Taxi Zone")
plt.title("Demand Coverage – Top 20 High-Demand Zones")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fleet_demand_coverage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 4. FLEET DISTRIBUTION BY BOROUGH
# --------------------------------------------------

borough_fleet = (
    zones.groupby("Borough")["allocated_vehicles"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))

plt.bar(
    borough_fleet.index,
    borough_fleet.values
)

plt.xlabel("Borough")
plt.ylabel("Allocated Vehicles")
plt.title("Fleet Distribution Across NYC Boroughs")

plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fleet_allocation_by_borough.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 5. DEMAND COVERAGE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    zones["demand_coverage_percentage"],
    bins=15,
    edgecolor="black"
)

plt.xlabel("Demand Coverage (%)")
plt.ylabel("Number of Taxi Zones")
plt.title("Distribution of Demand Coverage Across Zones")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fleet_coverage_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nFleet allocation visualizations completed!")

print("\nGenerated charts:")

for filename in [
    "fleet_allocation_by_zone.png",
    "fleet_demand_vs_served.png",
    "fleet_demand_coverage.png",
    "fleet_allocation_by_borough.png",
    "fleet_coverage_distribution.png"
]:
    print(OUTPUT_DIR / filename)

print("\nAll charts saved successfully.")
