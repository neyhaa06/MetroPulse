
from pathlib import Path
import json

import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[3]

FLEET_RESULTS_PATH = (
    ROOT_DIR
    / "reports"
    / "fleet_allocation_results.csv"
)


# --------------------------------------------------
# LOAD FLEET ALLOCATION RESULTS
# --------------------------------------------------

def load_fleet_results():

    if not FLEET_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Fleet allocation results were not found. "
            "Please run the fleet allocation optimization first."
        )

    results = pd.read_csv(FLEET_RESULTS_PATH)

    required_columns = [
        "LocationID",
        "Zone",
        "Borough",
        "average_hourly_demand",
        "allocated_vehicles",
        "served_hourly_demand",
        "unmet_hourly_demand",
        "demand_coverage_percentage"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if results.empty:
        raise ValueError(
            "Fleet allocation results contain no records."
        )

    return results


# --------------------------------------------------
# FLEET SUMMARY
# --------------------------------------------------

def get_fleet_summary():

    results = load_fleet_results()

    total_vehicles = int(
        results["allocated_vehicles"].sum()
    )

    total_zones = int(len(results))

    zones_with_vehicles = int(
        (results["allocated_vehicles"] > 0).sum()
    )

    total_demand = float(
        results["average_hourly_demand"].sum()
    )

    total_served = float(
        results["served_hourly_demand"].sum()
    )

    total_unmet = float(
        results["unmet_hourly_demand"].sum()
    )

    overall_coverage = (
        total_served / total_demand * 100
        if total_demand > 0
        else 0
    )

    minimum_zone_coverage = float(
        results["demand_coverage_percentage"].min()
    )

    average_zone_coverage = float(
        results["demand_coverage_percentage"].mean()
    )

    return {
        "total_vehicles_allocated": total_vehicles,
        "total_zones": total_zones,
        "zones_with_vehicles": zones_with_vehicles,
        "hourly_demand": round(total_demand, 2),
        "hourly_demand_served": round(total_served, 2),
        "hourly_unmet_demand": round(total_unmet, 2),
        "overall_demand_coverage_percentage": round(
            overall_coverage, 2
        ),
        "minimum_zone_coverage_percentage": round(
            minimum_zone_coverage, 2
        ),
        "average_zone_coverage_percentage": round(
            average_zone_coverage, 2
        )
    }


# --------------------------------------------------
# ALL ZONE ALLOCATIONS
# --------------------------------------------------

def get_zone_allocations():

    results = load_fleet_results()

    results = results.sort_values(
        "allocated_vehicles",
        ascending=False
    )

    return json.loads(
        results.to_json(orient="records")
    )


# --------------------------------------------------
# INDIVIDUAL ZONE ALLOCATION
# --------------------------------------------------

def get_zone_allocation(location_id: int):

    results = load_fleet_results()

    zone = results[
        results["LocationID"] == location_id
    ]

    if zone.empty:
        return None

    return json.loads(
        zone.to_json(orient="records")
    )[0]
