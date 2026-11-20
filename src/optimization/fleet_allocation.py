
from pathlib import Path
import math

import pandas as pd
import pulp


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT_DIR / "reports" / "dbscan_zone_assignments.csv"

OUTPUT_DIR = ROOT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUTPUT_DIR / "fleet_allocation_results.csv"
REPORT_PATH = OUTPUT_DIR / "fleet_allocation_report.txt"


# --------------------------------------------------
# SIMULATION ASSUMPTIONS
# --------------------------------------------------

TOTAL_FLEET = 1000
TRIPS_PER_VEHICLE_PER_HOUR = 2
HOURS_IN_JANUARY = 31 * 24


# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

print("Starting MetroPulse fair fleet allocation optimization...")

zones = pd.read_csv(INPUT_PATH)

zones = zones[
    zones["LocationID"].between(1, 263)
    & (zones["pickup_trips"] > 0)
].copy()

zones["average_hourly_demand"] = (
    zones["pickup_trips"] / HOURS_IN_JANUARY
)

zones["maximum_required_vehicles"] = zones[
    "average_hourly_demand"
].apply(
    lambda demand: math.ceil(
        demand / TRIPS_PER_VEHICLE_PER_HOUR
    )
)

zones = zones.reset_index(drop=True)

print(f"Geographical zones considered: {len(zones)}")
print(f"Simulated fleet available: {TOTAL_FLEET}")


# --------------------------------------------------
# CREATE OPTIMIZATION MODEL
# --------------------------------------------------

model = pulp.LpProblem(
    "MetroPulse_Fair_Fleet_Allocation",
    pulp.LpMaximize
)


# --------------------------------------------------
# DECISION VARIABLES
# --------------------------------------------------

# Integer number of vehicles assigned to each zone
vehicle_vars = {
    i: model.add_variable(
        f"vehicles_zone_{i}",
        lowBound=1,
        upBound=int(zones.loc[i, "maximum_required_vehicles"]),
        cat="Integer"
    )
    for i in zones.index
}

# Hourly demand served in each zone
served_demand_vars = {
    i: model.add_variable(
        f"served_demand_zone_{i}",
        lowBound=0,
        upBound=float(zones.loc[i, "average_hourly_demand"]),
        cat="Continuous"
    )
    for i in zones.index
}

# Minimum percentage of demand covered across all zones
minimum_coverage = model.add_variable(
    "minimum_demand_coverage",
    lowBound=0,
    upBound=1,
    cat="Continuous"
)


# --------------------------------------------------
# OBJECTIVE FUNCTION
# Maximize minimum demand coverage
# --------------------------------------------------

model += minimum_coverage


# --------------------------------------------------
# CONSTRAINTS
# --------------------------------------------------

# Total fleet availability
model += (
    pulp.lpSum(vehicle_vars[i] for i in zones.index)
    <= TOTAL_FLEET,
    "Total_Fleet_Limit"
)

for i in zones.index:

    demand = float(zones.loc[i, "average_hourly_demand"])

    # Vehicle service capacity
    model += (
        served_demand_vars[i]
        <= TRIPS_PER_VEHICLE_PER_HOUR * vehicle_vars[i],
        f"Zone_Capacity_{i}"
    )

    # Every zone must achieve at least the optimized
    # minimum percentage of demand coverage
    model += (
        served_demand_vars[i]
        >= minimum_coverage * demand,
        f"Minimum_Demand_Coverage_{i}"
    )


# --------------------------------------------------
# SOLVE USING HiGHS
# --------------------------------------------------

print("\nSolving fair fleet allocation model...")

solver = pulp.HiGHS(msg=False)

stats = model.solve(solver)

status = stats.status_str

print(f"Optimization status: {status}")

if not stats.has_solution or status != "Optimal":
    raise RuntimeError(
        f"Optimization failed. Solver status: {status}"
    )


# --------------------------------------------------
# EXTRACT RESULTS
# --------------------------------------------------

zones["allocated_vehicles"] = [
    int(round(pulp.value(vehicle_vars[i])))
    for i in zones.index
]

# Preserve full precision for all calculations
zones["served_hourly_demand"] = [
    float(pulp.value(served_demand_vars[i]))
    for i in zones.index
]

# Calculate unmet demand using unrounded values
zones["unmet_hourly_demand"] = (
    zones["average_hourly_demand"]
    - zones["served_hourly_demand"]
).clip(lower=0)

# Calculate coverage using unrounded values
zones["demand_coverage_percentage"] = (
    zones["served_hourly_demand"]
    / zones["average_hourly_demand"]
    * 100
)

# Round only after all calculations are complete
zones["served_hourly_demand"] = (
    zones["served_hourly_demand"].round(4)
)

zones["unmet_hourly_demand"] = (
    zones["unmet_hourly_demand"].round(4)
)

zones["demand_coverage_percentage"] = (
    zones["demand_coverage_percentage"].round(2)
)

zones["allocation_status"] = zones[
    "allocated_vehicles"
].apply(
    lambda count: (
        "Vehicles Allocated"
        if count > 0
        else "No Vehicles Allocated"
    )
)

zones = zones.sort_values(
    "average_hourly_demand",
    ascending=False
)

zones.to_csv(RESULT_PATH, index=False)



# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

total_allocated = int(zones["allocated_vehicles"].sum())

total_demand = zones["average_hourly_demand"].sum()

total_served = zones["served_hourly_demand"].sum()

total_unmet = zones["unmet_hourly_demand"].sum()

overall_coverage = (
    total_served / total_demand * 100
    if total_demand > 0
    else 0
)

minimum_coverage_result = (
    float(pulp.value(minimum_coverage)) * 100
)


# --------------------------------------------------
# GENERATE REPORT
# --------------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as report:

    report.write("METROPULSE - FAIR FLEET ALLOCATION REPORT\n")
    report.write("=" * 55 + "\n\n")

    report.write("OPTIMIZATION CONFIGURATION\n")
    report.write("-" * 55 + "\n")
    report.write("Solver: HiGHS\n")
    report.write("Method: Integer Linear Programming\n")
    report.write("Objective: Maximize minimum zone demand coverage\n")
    report.write("Fleet availability: SIMULATED\n")
    report.write(f"Total simulated fleet: {TOTAL_FLEET}\n")
    report.write(
        f"Trips per vehicle per hour: "
        f"{TRIPS_PER_VEHICLE_PER_HOUR}\n\n"
    )

    report.write("OPTIMIZATION RESULTS\n")
    report.write("-" * 55 + "\n")
    report.write(f"Solver status: {status}\n")
    report.write(f"Zones considered: {len(zones)}\n")
    report.write(f"Vehicles allocated: {total_allocated}\n")
    report.write(
        f"Minimum zone coverage: "
        f"{minimum_coverage_result:.2f}%\n"
    )
    report.write(
        f"Average demand coverage: {overall_coverage:.2f}%\n"
    )
    report.write(
        f"Average hourly demand: {total_demand:.2f}\n"
    )
    report.write(
        f"Demand served per hour: {total_served:.2f}\n"
    )
    report.write(
        f"Unmet demand per hour: {total_unmet:.2f}\n\n"
    )

    report.write("TOP 10 ZONE ALLOCATIONS\n")
    report.write("-" * 55 + "\n")

    report.write(
        zones[
            [
                "LocationID",
                "Zone",
                "Borough",
                "average_hourly_demand",
                "allocated_vehicles",
                "demand_coverage_percentage"
            ]
        ].head(10).to_string(index=False)
    )

    report.write("\n\n")
    report.write(
        "NOTE: Fleet availability and vehicle service capacity "
        "are simulation assumptions, not observed real-world "
        "fleet measurements.\n"
    )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\nFair fleet allocation completed!")

print(f"Solver status: {status}")
print(f"Vehicles allocated: {total_allocated}/{TOTAL_FLEET}")

print(
    f"Minimum zone coverage: "
    f"{minimum_coverage_result:.2f}%"
)

print(f"Average demand coverage: {overall_coverage:.2f}%")
print(f"Demand served per hour: {total_served:.2f}")
print(f"Unmet demand per hour: {total_unmet:.2f}")

print(
    "\nZones receiving vehicles:",
    (zones["allocated_vehicles"] > 0).sum()
)

print("\nTop 10 zone allocations:")

print(
    zones[
        [
            "Zone",
            "Borough",
            "average_hourly_demand",
            "allocated_vehicles",
            "demand_coverage_percentage"
        ]
    ].head(10).to_string(index=False)
)

print(f"\nResults saved: {RESULT_PATH}")
print(f"Report saved: {REPORT_PATH}")
