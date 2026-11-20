
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit.components.v1 as components


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
EVALUATION_DIR = ROOT / "ml" / "evaluation"

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="MetroPulse | Urban Mobility Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #101a35;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 0;
    }

    .subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #172554;
        font-weight: 700;
        font-size: 21px;
        margin-top: 15px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5eaf3;
        padding: 18px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_hourly():
    df = load_csv(
        DATA_DIR / "hourly_demand_2024_01.csv"
    )
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    return df


@st.cache_data
def load_daily():
    df = load_csv(REPORT_DIR / "daily_demand.csv")
    df["pickup_date"] = pd.to_datetime(df["pickup_date"])
    return df


hourly = load_hourly()
daily = load_daily()

hotspots = load_csv(
    REPORT_DIR / "geospatial_hotspots.csv"
)

zone_assignments = load_csv(
    REPORT_DIR / "dbscan_zone_assignments.csv"
)

cluster_summary = load_csv(
    REPORT_DIR / "dbscan_cluster_summary.csv"
)

predictions = load_csv(
    EVALUATION_DIR / "forecast_test_predictions.csv"
)

shap_importance = load_csv(
    EVALUATION_DIR / "shap_feature_importance.csv"
)

model_comparison = load_csv(
    EVALUATION_DIR / "forecast_model_comparison.csv"
)

features = load_csv(
    DATA_DIR / "demand_features_2024_01.csv"
)

# --------------------------------------------------
# FLEET API DATA LOADING
# --------------------------------------------------

def load_fleet_summary():
    response = requests.get(
        f"{API_URL}/fleet/summary",
        timeout=15
    )

    response.raise_for_status()

    return response.json()["fleet_summary"]


def load_fleet_allocations():
    response = requests.get(
        f"{API_URL}/fleet/allocations",
        timeout=15
    )

    response.raise_for_status()

    allocation_data = response.json()["allocations"]

    return pd.DataFrame(allocation_data)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("# 🚕 MetroPulse")
    st.caption("URBAN MOBILITY INTELLIGENCE")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Demand Analytics",
            "Demand Forecasting",
            "Geospatial Insights",
            "Model Explainability",
            "Model Performance",
            "Fleet Optimization"
        ]
    )

    st.divider()

    st.markdown("### DATASET")
    st.write("NYC Yellow Taxi Trips")
    st.write("January 2024")

    st.caption("Data-driven urban transportation analytics")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<p class="main-title">MetroPulse</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Urban Mobility Intelligence, Demand Forecasting '
    '& Fleet Optimization'
    '</p>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    total_trips = int(hourly["total_trips"].sum())

    busiest_hour = (
        hourly.assign(hour=hourly["pickup_hour"].dt.hour)
        .groupby("hour")["total_trips"]
        .sum()
        .idxmax()
    )

    busiest_day = daily.loc[
        daily["total_trips"].idxmax()
    ]

    matched_demand = int(
        zone_assignments["pickup_trips"].sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Trips",
        f"{total_trips:,}"
    )

    col2.metric(
        "Busiest Hour",
        f"{busiest_hour:02d}:00"
    )

    col3.metric(
        "Busiest Day",
        busiest_day["pickup_date"].strftime("%d Jan")
    )

    col4.metric(
        "Geospatial Zones",
        f"{zone_assignments['LocationID'].nunique()}"
    )

    st.markdown(
        '<p class="section-title">Demand Overview</p>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        daily_chart = px.area(
            daily,
            x="pickup_date",
            y="total_trips",
            title="Daily Taxi Demand",
            labels={
                "pickup_date": "Date",
                "total_trips": "Trips"
            }
        )

        daily_chart.update_layout(
            template="plotly_white",
            height=350
        )

        st.plotly_chart(
            daily_chart,
            use_container_width=True
        )

    with right:
        hourly_profile = (
            hourly.assign(
                hour=hourly["pickup_hour"].dt.hour
            )
            .groupby("hour", as_index=False)["total_trips"]
            .sum()
        )

        hourly_chart = px.bar(
            hourly_profile,
            x="hour",
            y="total_trips",
            title="Demand by Hour of Day",
            labels={
                "hour": "Hour",
                "total_trips": "Total Trips"
            }
        )

        hourly_chart.update_layout(
            template="plotly_white",
            height=350
        )

        st.plotly_chart(
            hourly_chart,
            use_container_width=True
        )

    st.markdown(
        '<p class="section-title">Top Pickup Locations</p>',
        unsafe_allow_html=True
    )

    top_zones = hotspots.head(10).sort_values(
        "pickup_trips"
    )

    zone_chart = px.bar(
        top_zones,
        x="pickup_trips",
        y="Zone",
        color="Borough",
        orientation="h",
        title="Top 10 Pickup Zones",
        labels={
            "pickup_trips": "Pickup Trips",
            "Zone": "Taxi Zone"
        }
    )

    zone_chart.update_layout(
        template="plotly_white",
        height=430
    )

    st.plotly_chart(
        zone_chart,
        use_container_width=True
    )


# --------------------------------------------------
# DEMAND ANALYTICS
# --------------------------------------------------

elif page == "Demand Analytics":

    st.subheader("Demand Analytics")

    st.write(
        "Explore hourly, daily and weekday taxi demand patterns."
    )

    hourly_profile = (
        hourly.assign(
            hour=hourly["pickup_hour"].dt.hour
        )
        .groupby("hour", as_index=False)["total_trips"]
        .sum()
    )

    fig = px.line(
        hourly_profile,
        x="hour",
        y="total_trips",
        markers=True,
        title="Hourly Demand Distribution",
        labels={
            "hour": "Hour of Day",
            "total_trips": "Total Trips"
        }
    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    daily_fig = px.bar(
        daily,
        x="pickup_date",
        y="total_trips",
        title="Daily Demand Distribution",
        labels={
            "pickup_date": "Date",
            "total_trips": "Trips"
        }
    )

    daily_fig.update_layout(template="plotly_white")

    st.plotly_chart(
        daily_fig,
        use_container_width=True
    )

    st.markdown("### Hourly Demand Dataset")

    st.dataframe(
        hourly,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# DEMAND FORECASTING
# --------------------------------------------------

elif page == "Demand Forecasting":

    st.subheader("Random Forest Demand Forecasting")

    st.write(
        "Generate a prediction using the trained model "
        "through the MetroPulse FastAPI backend."
    )

    st.info(
        "Select an existing historical feature record. "
        "This demonstrates the API using recorded January "
        "2024 inputs, not live taxi conditions."
    )

    if "pickup_hour" in features.columns:

        features["pickup_hour"] = pd.to_datetime(
            features["pickup_hour"]
        )

        selected_index = st.selectbox(
            "Select a historical hour",
            options=features.index.tolist(),
            format_func=lambda i: features.loc[
                i, "pickup_hour"
            ].strftime("%d %b %Y, %H:%M")
        )

    else:
        selected_index = st.selectbox(
            "Select a feature record",
            options=features.index.tolist()
        )

    feature_columns = [
        "hour",
        "day_of_week",
        "day_of_month",
        "is_weekend",
        "is_rush_hour",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_24",
        "rolling_mean_168"
    ]

    selected_row = features.loc[selected_index]

    st.markdown("### Selected Model Inputs")

    st.dataframe(
        selected_row[feature_columns].to_frame(
            name="Input Value"
        ),
        use_container_width=True
    )

    if st.button(
        "Generate Demand Forecast",
        type="primary",
        use_container_width=True
    ):

        payload = {}

        for column in feature_columns:
            value = float(selected_row[column])

            if column in [
                "hour",
                "day_of_week",
                "day_of_month",
                "is_weekend",
                "is_rush_hour"
            ]:
                payload[column] = int(value)
            else:
                payload[column] = value

        try:
            response = requests.post(
                f"{API_URL}/forecast",
                json=payload,
                timeout=30
            )

            response.raise_for_status()

            result = response.json()["forecast"]

            st.success("Forecast generated successfully!")

            st.metric(
                "Predicted Hourly Demand",
                f"{result['predicted_demand']:,.2f} trips"
            )

            st.write(
                f"Model: {result['model']}"
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the forecasting API. "
                "Start the FastAPI server in another terminal."
            )

        except requests.exceptions.RequestException as error:
            st.error(f"Forecast request failed: {error}")

        except (KeyError, ValueError) as error:
            st.error(f"Invalid forecast response: {error}")


# --------------------------------------------------
# GEOSPATIAL INSIGHTS
# --------------------------------------------------

elif page == "Geospatial Insights":

    st.subheader("Geospatial Demand Intelligence")

    st.write(
        "Explore pickup hotspots and spatial clusters "
        "across NYC taxi zones."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mapped Zones",
        f"{zone_assignments['LocationID'].nunique()}"
    )

    col2.metric(
        "DBSCAN Clusters",
        f"{cluster_summary['cluster_id'].nunique()}"
    )

    col3.metric(
        "Clustered Demand",
        f"{cluster_summary['total_pickup_trips'].sum():,}"
    )

    st.markdown("### Top Pickup Hotspots")

    top_hotspots = hotspots.head(15)

    fig = px.bar(
        top_hotspots.sort_values("pickup_trips"),
        x="pickup_trips",
        y="Zone",
        color="Borough",
        orientation="h",
        title="Highest Demand Taxi Zones",
        labels={
            "pickup_trips": "Pickup Trips",
            "Zone": "Taxi Zone"
        }
    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### DBSCAN Cluster Summary")

    st.dataframe(
        cluster_summary,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Existing Hotspot Map")

    map_path = REPORT_DIR / "nyc_pickup_hotspots.html"

    if map_path.exists():
        components.html(
            map_path.read_text(encoding="utf-8"),
            height=600,
            scrolling=True
        )
    else:
        st.warning("Hotspot map file not found.")


# --------------------------------------------------
# MODEL EXPLAINABILITY
# --------------------------------------------------

elif page == "Model Explainability":

    st.subheader("Explainable AI – SHAP")

    st.write(
        "Understand which features influence the "
        "Random Forest demand predictions."
    )

    shap_chart = px.bar(
        shap_importance.sort_values(
            "mean_absolute_shap_value"
        ),
        x="mean_absolute_shap_value",
        y="feature",
        orientation="h",
        title="SHAP Feature Importance",
        labels={
            "mean_absolute_shap_value": "Mean Absolute SHAP Value",
            "feature": "Feature"
        }
    )

    shap_chart.update_layout(
        template="plotly_white",
        height=550
    )

    st.plotly_chart(
        shap_chart,
        use_container_width=True
    )

    summary_path = FIGURE_DIR / "shap_summary_plot.png"

    if summary_path.exists():
        st.markdown("### SHAP Summary Plot")
        st.image(
            str(summary_path),
            use_container_width=True
        )

    st.caption(
        "SHAP values describe feature contributions to "
        "individual model predictions. They do not establish "
        "causation."
    )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.subheader("Forecast Model Evaluation")

    st.write(
        "Compare the forecasting models using "
        "the saved January 2024 test results."
    )

    st.markdown("### Model Comparison")

    st.dataframe(
        model_comparison,
        use_container_width=True,
        hide_index=True
    )

    comparison_fig = px.bar(
        model_comparison,
        x="model",
        y="MAE",
        title="Model Comparison – Mean Absolute Error",
        labels={
            "model": "Model",
            "MAE": "Mean Absolute Error"
        }
    )

    comparison_fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        comparison_fig,
        use_container_width=True
    )

    st.markdown("### Actual vs Predicted Demand")

    prediction_fig = go.Figure()

    prediction_fig.add_trace(
        go.Scatter(
            x=predictions["pickup_hour"],
            y=predictions["actual_demand"],
            name="Actual Demand",
            mode="lines"
        )
    )

    prediction_fig.add_trace(
        go.Scatter(
            x=predictions["pickup_hour"],
            y=predictions["random_forest"],
            name="Random Forest",
            mode="lines"
        )
    )

    prediction_fig.update_layout(
        title="Actual vs Random Forest Predictions",
        xaxis_title="Pickup Hour",
        yaxis_title="Trips",
        template="plotly_white"
    )

    st.plotly_chart(
        prediction_fig,
        use_container_width=True
    )

    error_image = FIGURE_DIR / "forecast_residual_distribution.png"

    if error_image.exists():
        st.markdown("### Forecast Residual Distribution")
        st.image(
            str(error_image),
            use_container_width=True
        )

    st.markdown("### Forecast Test Predictions")

    st.dataframe(
        predictions,
        use_container_width=True,
        hide_index=True
    )




# --------------------------------------------------

# --------------------------------------------------
# FLEET OPTIMIZATION
# --------------------------------------------------

elif page == "Fleet Optimization":

    st.subheader("Fleet Allocation Optimization")

    st.write(
        "Explore how the optimization model distributes a limited "
        "vehicle fleet across NYC pickup zones to improve minimum "
        "zone-level demand coverage."
    )

    try:
        # Retrieve fleet data through FastAPI
        fleet_summary = load_fleet_summary()
        fleet_results = load_fleet_allocations()

        if fleet_results.empty:
            st.warning("No fleet allocation records were returned.")

        else:
            total_vehicles = fleet_summary[
                "total_vehicles_allocated"
            ]

            zones_with_vehicles = fleet_summary[
                "zones_with_vehicles"
            ]

            total_zones = fleet_summary["total_zones"]

            total_hourly_demand = fleet_summary[
                "hourly_demand"
            ]

            served_hourly_demand = fleet_summary[
                "hourly_demand_served"
            ]

            unmet_hourly_demand = fleet_summary[
                "hourly_unmet_demand"
            ]

            overall_coverage = fleet_summary[
                "overall_demand_coverage_percentage"
            ]

            minimum_zone_coverage = fleet_summary[
                "minimum_zone_coverage_percentage"
            ]

            st.success(
                "Fleet allocation data successfully retrieved "
                "from the MetroPulse FastAPI backend."
            )

            st.info(
                "Simulation assumptions: 1,000 vehicles in the available "
                "fleet and a service capacity of 2 trips per vehicle per hour. "
                "These are model inputs, not observed real-world fleet data. "
                "The maximin optimization objective seeks to maximize the "
                "minimum demand coverage across zones."
            )

            # ------------------------------------------
            # PERFORMANCE METRICS
            # ------------------------------------------

            metric1, metric2, metric3, metric4 = st.columns(4)

            metric1.metric(
                "Vehicles Allocated",
                f"{total_vehicles:,}"
            )

            metric2.metric(
                "Zones Served",
                f"{zones_with_vehicles:,} / {total_zones:,}"
            )

            metric3.metric(
                "Overall Demand Coverage",
                f"{overall_coverage:.2f}%"
            )

            metric4.metric(
                "Minimum Zone Coverage",
                f"{minimum_zone_coverage:.2f}%"
            )

            metric5, metric6, metric7 = st.columns(3)

            metric5.metric(
                "Hourly Demand",
                f"{total_hourly_demand:,.2f} trips"
            )

            metric6.metric(
                "Hourly Demand Served",
                f"{served_hourly_demand:,.2f} trips"
            )

            metric7.metric(
                "Hourly Unmet Demand",
                f"{unmet_hourly_demand:,.2f} trips"
            )

            # ------------------------------------------
            # VISUALIZATIONS
            # ------------------------------------------

            st.markdown("### Fleet Optimization Visualizations")

            chart_left, chart_right = st.columns(2)

            chart_definitions = [
                (
                    "fleet_allocation_by_zone.png",
                    "Vehicle allocation across the highest-demand zones",
                    chart_left
                ),
                (
                    "fleet_demand_vs_served.png",
                    "Hourly demand compared with demand served",
                    chart_right
                ),
                (
                    "fleet_demand_coverage.png",
                    "Demand coverage across high-demand zones",
                    chart_left
                ),
                (
                    "fleet_allocation_by_borough.png",
                    "Fleet distribution by borough",
                    chart_right
                )
            ]

            for image_name, chart_title, chart_column in chart_definitions:

                image_path = FIGURE_DIR / image_name

                with chart_column:

                    st.markdown(f"**{chart_title}**")

                    if image_path.exists():
                        st.image(
                            str(image_path),
                            use_container_width=True
                        )
                    else:
                        st.warning(
                            f"Chart not found: {image_name}"
                        )

            coverage_image = (
                FIGURE_DIR / "fleet_coverage_distribution.png"
            )

            st.markdown(
                "**Distribution of zone-level demand coverage**"
            )

            if coverage_image.exists():
                st.image(
                    str(coverage_image),
                    use_container_width=True
                )
            else:
                st.warning(
                    "Chart not found: fleet_coverage_distribution.png"
                )

            # ------------------------------------------
            # TOP ZONE ALLOCATIONS
            # ------------------------------------------

            st.markdown("### Top Zones by Vehicle Allocation")

            display_columns = [
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
                column for column in display_columns
                if column not in fleet_results.columns
            ]

            if missing_columns:
                st.error(
                    "API allocation response is missing columns: "
                    + ", ".join(missing_columns)
                )

            else:
                top_allocations = (
                    fleet_results
                    .sort_values(
                        "allocated_vehicles",
                        ascending=False
                    )
                    [display_columns]
                    .head(15)
                    .copy()
                )

                top_allocations = top_allocations.rename(
                    columns={
                        "LocationID": "Zone ID",
                        "Zone": "Taxi Zone",
                        "Borough": "Borough",
                        "average_hourly_demand": "Hourly Demand",
                        "allocated_vehicles": "Vehicles Allocated",
                        "served_hourly_demand": "Hourly Demand Served",
                        "unmet_hourly_demand": "Hourly Unmet Demand",
                        "demand_coverage_percentage": "Coverage (%)"
                    }
                )

                st.dataframe(
                    top_allocations.round(2),
                    use_container_width=True,
                    hide_index=True
                )

                with st.expander(
                    "View complete fleet allocation results"
                ):
                    st.dataframe(
                        fleet_results,
                        use_container_width=True,
                        hide_index=True
                    )

            st.caption(
                "Results are based on January 2024 average hourly pickup "
                "demand and the stated fleet simulation assumptions. "
                "They represent an optimization scenario, not live vehicle "
                "dispatch or real-time demand."
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the MetroPulse API. "
            "Please start the FastAPI server in another terminal."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Fleet API request failed: {error}")

    except (KeyError, ValueError) as error:
        st.error(f"Invalid fleet API response: {error}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "MetroPulse | Urban Mobility Intelligence | "
    "NYC Yellow Taxi Dataset – January 2024"
)
