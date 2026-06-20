from __future__ import annotations
import json
import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.predict import generate_predictions
from src.traffic_service import (
    TomTomTrafficClient,
    calculate_proxy_congestion_scores,
)
from src.utils import (
    AGGREGATED_DATA_PATH,
    CATBOOST_MODEL_PATH,
    DATA_FILE_NAME,
    METADATA_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    RAW_DATA_PATH,
    clean_stringified_list,
    normalize_junction_name,
    safe_datetime,
)


app = FastAPI(
    title="Parking Hotspot Intelligence API",
    description="FastAPI backend for React frontend and ML hotspot predictions",
    version="1.0.0",
)

cors_origins = os.getenv(
    "CORS_ORIGINS",
    "https://parking-frontend-uv08.onrender.com,http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_OUTPUT_DIR = PREDICTIONS_PATH.parent

DASHBOARD_SUMMARY_PATH = DASHBOARD_OUTPUT_DIR / "dashboard_summary.json"
DASHBOARD_TOP_STATIONS_PATH = DASHBOARD_OUTPUT_DIR / "dashboard_top_stations.csv"
DASHBOARD_TOP_JUNCTIONS_PATH = DASHBOARD_OUTPUT_DIR / "dashboard_top_junctions.csv"
DASHBOARD_TOP_JUNCTIONS_ALL_PATH = DASHBOARD_OUTPUT_DIR / "dashboard_top_junctions_all.csv"
DASHBOARD_HOURLY_TREND_PATH = DASHBOARD_OUTPUT_DIR / "dashboard_hourly_trend.csv"
DASHBOARD_VEHICLE_TYPES_PATH = DASHBOARD_OUTPUT_DIR / "dashboard_vehicle_types.csv"


def read_dashboard_csv(path, limit: int | None = None):
    df = pd.read_csv(path)

    if limit is not None:
        df = df.head(limit)

    return {"data": df.fillna("").to_dict(orient="records")}

def read_predictions() -> pd.DataFrame:
    """
    Runtime prediction loader.

    On Render, the large raw dataset is not uploaded.
    So the deployed app should mainly use:
    backend/outputs/hotspot_predictions.csv
    """
    if not PREDICTIONS_PATH.exists():
        if not METADATA_PATH.exists() or not CATBOOST_MODEL_PATH.exists():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Predictions/model not found. First run locally: "
                    "python src/train_model.py and python src/predict.py"
                ),
            )

        return generate_predictions()

    df = pd.read_csv(PREDICTIONS_PATH)

    if (
        "congestion_impact_score" not in df.columns
        or "final_enforcement_score" not in df.columns
    ):
        df = calculate_proxy_congestion_scores(df)
        df.to_csv(PREDICTIONS_PATH, index=False)

    return df


def load_clean_raw_data() -> pd.DataFrame:
    """
    Loads raw dataset locally if available.

    On Render, the raw CSV is not present because it is larger than GitHub's
    100 MB limit. In that case, fallback to hotspot_predictions.csv.
    """
    if RAW_DATA_PATH.exists():
        df = pd.read_csv(RAW_DATA_PATH)
        df = df.dropna(subset=["latitude", "longitude", "created_datetime"]).copy()

        df["created_datetime"] = safe_datetime(df["created_datetime"])
        df = df.dropna(subset=["created_datetime"]).copy()

        df["hour"] = df["created_datetime"].dt.hour
        df["day_of_week"] = df["created_datetime"].dt.day_name()
        df["month"] = df["created_datetime"].dt.month

        if "violation_type" in df.columns:
            df["violation_type"] = df["violation_type"].apply(clean_stringified_list)

        for col in ["police_station", "junction_name", "vehicle_type", "violation_type"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()

        if "junction_name" in df.columns:
            df["junction_name"] = df["junction_name"].apply(normalize_junction_name)

        return df

    # Render fallback: use prediction output.
    df = read_predictions().copy()

    if "police_station" not in df.columns:
        df["police_station"] = "Unknown"

    if "junction_name" not in df.columns:
        df["junction_name"] = "Unknown"

    if "hour" not in df.columns:
        df["hour"] = 0

    if "vehicle_type" not in df.columns:
        if "common_vehicle_type" in df.columns:
            df["vehicle_type"] = df["common_vehicle_type"]
        else:
            df["vehicle_type"] = "Unknown"

    if "violation_type" not in df.columns:
        if "common_violation_type" in df.columns:
            df["violation_type"] = df["common_violation_type"]
        else:
            df["violation_type"] = "Unknown"

    for col in ["police_station", "junction_name", "vehicle_type", "violation_type"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)

    return df


def aggregate_violations(df: pd.DataFrame, group_col: str, limit: int) -> pd.DataFrame:
    """
    If raw data is available, count rows.
    If prediction output is used, sum predicted_violation_count.
    """
    if group_col not in df.columns:
        return pd.DataFrame(columns=["name", "violations"])

    if "predicted_violation_count" in df.columns:
        temp = df.copy()
        temp["predicted_violation_count"] = pd.to_numeric(
            temp["predicted_violation_count"], errors="coerce"
        ).fillna(0)

        data = (
            temp.groupby(group_col, as_index=False)["predicted_violation_count"]
            .sum()
            .sort_values("predicted_violation_count", ascending=False)
            .head(limit)
        )
        data.columns = ["name", "violations"]
    else:
        data = df[group_col].value_counts().head(limit).reset_index()
        data.columns = ["name", "violations"]

    data["violations"] = pd.to_numeric(data["violations"], errors="coerce").fillna(0)
    data["violations"] = data["violations"].round(2)

    return data


@app.get("/")
def home():
    return {
        "message": "Parking Hotspot Intelligence API is running",
        "docs": "/docs",
        "health": "/health",
        "react_frontend_should_call": "/api/hotspots",
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "ml-backend"}


@app.get("/api/status")
def get_status():
    return {
        "dataset_exists": RAW_DATA_PATH.exists(),
        "dataset_file": DATA_FILE_NAME,
        "aggregated_data_exists": AGGREGATED_DATA_PATH.exists(),
        "model_exists": CATBOOST_MODEL_PATH.exists(),
        "metadata_exists": METADATA_PATH.exists(),
        "predictions_exist": PREDICTIONS_PATH.exists(),
        "metrics_exist": METRICS_PATH.exists(),
    }


@app.get("/api/summary")
def get_summary():
    if DASHBOARD_SUMMARY_PATH.exists():
        return json.loads(DASHBOARD_SUMMARY_PATH.read_text(encoding="utf-8"))

    df = load_clean_raw_data()

    if "predicted_violation_count" in df.columns:
        total_violations = int(
            pd.to_numeric(df["predicted_violation_count"], errors="coerce")
            .fillna(0)
            .sum()
        )
    else:
        total_violations = int(len(df))

    return {
        "total_violations": total_violations,
        "police_stations": int(df["police_station"].nunique()),
        "junctions": int(df["junction_name"].nunique()),
        "vehicle_types": int(df["vehicle_type"].nunique()),
    }


@app.get("/api/top-stations")
def get_top_stations(limit: int = Query(default=15, ge=1, le=100)):
    if DASHBOARD_TOP_STATIONS_PATH.exists():
        return read_dashboard_csv(DASHBOARD_TOP_STATIONS_PATH, limit)

    df = load_clean_raw_data()
    data = aggregate_violations(df, "police_station", limit)

    return {"data": data.to_dict(orient="records")}


@app.get("/api/top-junctions")
def get_top_junctions(
    limit: int = Query(default=15, ge=1, le=100),
    include_unmapped: bool = Query(default=False),
):
    if include_unmapped and DASHBOARD_TOP_JUNCTIONS_ALL_PATH.exists():
        return read_dashboard_csv(DASHBOARD_TOP_JUNCTIONS_ALL_PATH, limit)

    if DASHBOARD_TOP_JUNCTIONS_PATH.exists():
        return read_dashboard_csv(DASHBOARD_TOP_JUNCTIONS_PATH, limit)

    df = load_clean_raw_data()

    if not include_unmapped and "junction_name" in df.columns:
        df = df[df["junction_name"].astype(str) != "Unmapped Location"]

    data = aggregate_violations(df, "junction_name", limit)

    return {"data": data.to_dict(orient="records")}


@app.get("/api/hourly-trend")
def get_hourly_trend():
    if DASHBOARD_HOURLY_TREND_PATH.exists():
        return read_dashboard_csv(DASHBOARD_HOURLY_TREND_PATH)

    df = load_clean_raw_data()

    if "predicted_violation_count" in df.columns:
        temp = df.copy()
        temp["predicted_violation_count"] = pd.to_numeric(
            temp["predicted_violation_count"], errors="coerce"
        ).fillna(0)

        data = (
            temp.groupby("hour", as_index=False)["predicted_violation_count"]
            .sum()
            .sort_values("hour")
        )
        data.columns = ["hour", "violations"]
    else:
        data = df.groupby("hour", as_index=False).size()
        data.columns = ["hour", "violations"]

    data["violations"] = pd.to_numeric(data["violations"], errors="coerce").fillna(0)
    data["violations"] = data["violations"].round(2)

    return {"data": data.to_dict(orient="records")}


@app.get("/api/vehicle-types")
def get_vehicle_types(limit: int = Query(default=12, ge=1, le=100)):
    if DASHBOARD_VEHICLE_TYPES_PATH.exists():
        return read_dashboard_csv(DASHBOARD_VEHICLE_TYPES_PATH, limit)

    df = load_clean_raw_data()
    data = aggregate_violations(df, "vehicle_type", limit)

    return {"data": data.to_dict(orient="records")}


@app.get("/api/hotspots")
def get_hotspots(
    police_station: Optional[str] = Query(default=None),
    junction_name: Optional[str] = Query(default=None),
    risk_level: Optional[str] = Query(default=None),
    hour: Optional[int] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    live_traffic: bool = Query(default=False),
):
    df = read_predictions()

    # For submission/prototype, external live traffic is disabled.
    # Always use proxy congestion from dataset/model output.
    df = calculate_proxy_congestion_scores(df)

    if police_station:
        df = df[df["police_station"].astype(str) == police_station]

    if junction_name:
        df = df[df["junction_name"].astype(str) == junction_name]

    if risk_level:
        df = df[df["risk_level"].astype(str) == risk_level]

    if hour is not None:
        df = df[df["hour"].astype(int) == int(hour)]

    default_columns = {
        "traffic_data_source": "proxy",
        "traffic_delay_ratio": 1.0,
        "live_traffic_score": 0.0,
        "normal_duration_seconds": 0.0,
        "traffic_duration_seconds": 0.0,
        "current_speed_kmph": 0.0,
        "free_flow_speed_kmph": 0.0,
        "traffic_confidence": 0.0,
        "road_closure": False,
        "road_latitude": 0.0,
        "road_longitude": 0.0,
    }

    for col, default_value in default_columns.items():
        if col not in df.columns:
            df[col] = default_value

    df["traffic_data_source"] = "proxy"

    if "final_enforcement_score" in df.columns:
        df = df.sort_values("final_enforcement_score", ascending=False)

    df = df.head(limit)

    columns = [
        "hotspot_id",
        "junction_name",
        "police_station",
        "latitude_center",
        "longitude_center",
        "road_latitude",
        "road_longitude",
        "predicted_violation_count",
        "risk_level",
        "priority_score",
        "congestion_impact_score",
        "final_enforcement_score",
        "recommended_enforcement_time",
        "hour",
        "traffic_data_source",
        "traffic_delay_ratio",
        "live_traffic_score",
        "normal_duration_seconds",
        "traffic_duration_seconds",
        "current_speed_kmph",
        "free_flow_speed_kmph",
        "traffic_confidence",
        "road_closure",
    ]

    available_columns = [col for col in columns if col in df.columns]

    return {
        "count": int(len(df)),
        "live_traffic_requested": False,
        "live_traffic_used": False,
        "compliance_mode": "Only provided dataset/model output used. External live traffic API disabled.",
        "data": df[available_columns].fillna("").to_dict(orient="records"),
    }


@app.post("/api/predict/regenerate")
def regenerate_predictions():
    try:
        df = generate_predictions()
        df = calculate_proxy_congestion_scores(df)
        df.to_csv(PREDICTIONS_PATH, index=False)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Train model first. Missing file: {exc}",
        ) from exc

    return {
        "message": "Predictions regenerated successfully",
        "count": int(len(df)),
    }


@app.get("/api/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        return {
            "message": (
                "Metrics file not found. Train model first using "
                "python src/train_model.py"
            )
        }

    return {"metrics": METRICS_PATH.read_text(encoding="utf-8")}


@app.get("/api/filter-options")
def get_filter_options():
    df = read_predictions()

    required_defaults = {
        "police_station": "Unknown",
        "junction_name": "Unknown",
        "risk_level": "Unknown",
        "hour": 0,
    }

    for col, default_value in required_defaults.items():
        if col not in df.columns:
            df[col] = default_value

    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)

    return {
        "police_stations": sorted(
            df["police_station"].dropna().astype(str).unique().tolist()
        ),
        "junctions": sorted(
            df["junction_name"].dropna().astype(str).unique().tolist()
        ),
        "risk_levels": sorted(
            df["risk_level"].dropna().astype(str).unique().tolist()
        ),
        "hours": sorted(df["hour"].dropna().astype(int).unique().tolist()),
    }


@app.get("/api/debug/tomtom")
def debug_tomtom(
    lat: float = 12.9716,
    lon: float = 77.5946,
):
    client = TomTomTrafficClient()
    live = client.get_live_traffic_impact(lat, lon)

    return {
        "api_key_loaded": client.enabled(),
        "api_key_length": len(client.api_key),
        "base_url": client.base_url,
        "test_result": live,
    }