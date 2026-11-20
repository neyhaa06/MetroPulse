
import sys
from pathlib import Path

from fastapi.testclient import TestClient


# Make the backend application importable during testing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_PATH))

from app import main


client = TestClient(main.app)


# --------------------------------------------------
# TEST DATA
# --------------------------------------------------

VALID_FORECAST_INPUT = {
    "hour": 18,
    "day_of_week": 2,
    "day_of_month": 15,
    "is_weekend": 0,
    "is_rush_hour": 1,
    "hour_sin": -1.0,
    "hour_cos": 0.0,
    "day_sin": 0.97,
    "day_cos": -0.22,
    "lag_1": 5000.0,
    "lag_24": 4500.0,
    "lag_168": 4000.0,
    "rolling_mean_24": 4200.0,
    "rolling_mean_168": 3900.0,
}


SAMPLE_FLEET_SUMMARY = {
    "total_vehicles_allocated": 1000,
    "total_zones": 258,
    "zones_with_vehicles": 258,
    "hourly_demand": 3967.57,
    "hourly_demand_served": 1625.97,
    "hourly_unmet_demand": 2341.59,
    "overall_demand_coverage_percentage": 40.98,
    "minimum_zone_coverage_percentage": 39.97,
    "average_zone_coverage_percentage": 84.64,
}


SAMPLE_ZONE = {
    "LocationID": 132,
    "Zone": "JFK Airport",
    "Borough": "Queens",
    "allocated_vehicles": 39,
    "demand_coverage_percentage": 39.97,
}


# --------------------------------------------------
# HOME AND HEALTH TESTS
# --------------------------------------------------

def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["project"] == "MetroPulse"


# --------------------------------------------------
# DEMAND FORECASTING TESTS
# --------------------------------------------------

def test_forecast_success(monkeypatch):

    def mock_predict_demand(data):
        return 7335.43

    monkeypatch.setattr(
        main,
        "predict_demand",
        mock_predict_demand
    )

    response = client.post(
        "/forecast",
        json=VALID_FORECAST_INPUT
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["forecast"] == 7335.43


def test_forecast_invalid_input():

    invalid_data = VALID_FORECAST_INPUT.copy()
    invalid_data["hour"] = 25

    response = client.post(
        "/forecast",
        json=invalid_data
    )

    assert response.status_code == 422


def test_forecast_service_error(monkeypatch):

    def mock_predict_demand(data):
        raise ValueError("Invalid forecast input")

    monkeypatch.setattr(
        main,
        "predict_demand",
        mock_predict_demand
    )

    response = client.post(
        "/forecast",
        json=VALID_FORECAST_INPUT
    )

    assert response.status_code == 400


# --------------------------------------------------
# FLEET SUMMARY TESTS
# --------------------------------------------------

def test_fleet_summary_success(monkeypatch):

    def mock_get_fleet_summary():
        return SAMPLE_FLEET_SUMMARY

    monkeypatch.setattr(
        main,
        "get_fleet_summary",
        mock_get_fleet_summary
    )

    response = client.get("/fleet/summary")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert (
        response.json()["fleet_summary"]["total_vehicles_allocated"]
        == 1000
    )


# --------------------------------------------------
# FLEET ALLOCATION TESTS
# --------------------------------------------------

def test_all_zone_allocations_success(monkeypatch):

    sample_allocations = [SAMPLE_ZONE]

    def mock_get_zone_allocations():
        return sample_allocations

    monkeypatch.setattr(
        main,
        "get_zone_allocations",
        mock_get_zone_allocations
    )

    response = client.get("/fleet/allocations")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["total_zones"] == 1
    assert len(response.json()["allocations"]) == 1


def test_individual_zone_success(monkeypatch):

    def mock_get_zone_allocation(location_id):

        if location_id == 132:
            return SAMPLE_ZONE

        return None

    monkeypatch.setattr(
        main,
        "get_zone_allocation",
        mock_get_zone_allocation
    )

    response = client.get("/fleet/allocations/132")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["allocation"]["Zone"] == "JFK Airport"


def test_individual_zone_not_found(monkeypatch):

    def mock_get_zone_allocation(location_id):
        return None

    monkeypatch.setattr(
        main,
        "get_zone_allocation",
        mock_get_zone_allocation
    )

    response = client.get("/fleet/allocations/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Taxi zone not found."
