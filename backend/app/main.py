
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.forecast_service import predict_demand

from app.services.fleet_service import (
    get_fleet_summary,
    get_zone_allocations,
    get_zone_allocation
)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="MetroPulse API",
    description=(
        "Urban Mobility Intelligence, "
        "Demand Forecasting and Fleet Optimization"
    ),
    version="1.1.0"
)


# --------------------------------------------------
# FORECAST INPUT MODEL
# --------------------------------------------------

class ForecastInput(BaseModel):

    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    day_of_month: int = Field(ge=1, le=31)
    is_weekend: int = Field(ge=0, le=1)
    is_rush_hour: int = Field(ge=0, le=1)

    hour_sin: float
    hour_cos: float
    day_sin: float
    day_cos: float

    lag_1: float = Field(ge=0)
    lag_24: float = Field(ge=0)
    lag_168: float = Field(ge=0)

    rolling_mean_24: float = Field(ge=0)
    rolling_mean_168: float = Field(ge=0)


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to MetroPulse API",
        "status": "running"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "project": "MetroPulse"
    }


# --------------------------------------------------
# DEMAND FORECASTING ENDPOINT
# --------------------------------------------------

@app.post("/forecast")
def forecast_demand(data: ForecastInput):

    try:

        result = predict_demand(data.model_dump())

        return {
            "status": "success",
            "forecast": result
        }

    except (ValueError, FileNotFoundError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail="Forecast generation failed."
        ) from error


# --------------------------------------------------
# FLEET OPTIMIZATION SUMMARY ENDPOINT
# --------------------------------------------------

@app.get("/fleet/summary")
def fleet_summary():

    try:

        result = get_fleet_summary()

        return {
            "status": "success",
            "fleet_summary": result
        }

    except (ValueError, FileNotFoundError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail="Fleet summary retrieval failed."
        ) from error


# --------------------------------------------------
# ALL ZONE ALLOCATIONS ENDPOINT
# --------------------------------------------------

@app.get("/fleet/allocations")
def fleet_allocations():

    try:

        results = get_zone_allocations()

        return {
            "status": "success",
            "total_zones": len(results),
            "allocations": results
        }

    except (ValueError, FileNotFoundError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail="Fleet allocation retrieval failed."
        ) from error


# --------------------------------------------------
# INDIVIDUAL ZONE ALLOCATION ENDPOINT
# --------------------------------------------------

@app.get("/fleet/allocations/{location_id}")
def individual_zone_allocation(location_id: int):

    try:

        result = get_zone_allocation(location_id)

        if result is None:

            raise HTTPException(
                status_code=404,
                detail="Taxi zone not found."
            )

        return {
            "status": "success",
            "allocation": result
        }

    except HTTPException:
        raise

    except (ValueError, FileNotFoundError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail="Zone allocation retrieval failed."
        ) from error
