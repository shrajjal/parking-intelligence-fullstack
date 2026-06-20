from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Makes imports work both when running:
# python src/predict.py
# and when FastAPI imports src.predict
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.preprocess import preprocess_raw_data
from src.traffic_service import calculate_proxy_congestion_scores
from src.utils import (
    AGGREGATED_DATA_PATH,
    CATBOOST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    METADATA_PATH,
    NUMERIC_FEATURES,
    PREDICTIONS_PATH,
    RF_MODEL_PATH,
    fill_missing_categoricals,
    load_joblib,
    make_risk_levels,
    recommended_time_from_hour,
)


def load_training_panel() -> pd.DataFrame:
    if not AGGREGATED_DATA_PATH.exists():
        preprocess_raw_data()

    df = pd.read_csv(AGGREGATED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df.dropna(subset=["date"]).copy()


def build_future_prediction_grid(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    latest_date = panel["date"].max()
    next_date = latest_date + pd.Timedelta(days=1)

    profile_cols = [
        "hotspot_id",
        "latitude_center",
        "longitude_center",
        "junction_name",
        "police_station",
        "dominant_vehicle_type",
        "dominant_violation_type",
        "dominant_offence_code",
    ]

    missing_profile_cols = [col for col in profile_cols if col not in panel.columns]
    if missing_profile_cols:
        raise ValueError(
            f"Missing required columns in aggregated data: {missing_profile_cols}"
        )

    hotspot_profile = panel[profile_cols].drop_duplicates("hotspot_id").copy()
    rows = []

    for _, profile in hotspot_profile.iterrows():
        hotspot_id = profile["hotspot_id"]
        police_station = profile["police_station"]

        hotspot_history = panel[panel["hotspot_id"] == hotspot_id].copy()

        for hour in range(24):
            same_hour = hotspot_history[
                hotspot_history["hour"].astype(int) == hour
            ].sort_values("date")

            lag_1 = same_hour.loc[
                same_hour["date"] == latest_date,
                "violation_count",
            ]

            lag_7_date = next_date - pd.Timedelta(days=7)
            lag_7 = same_hour.loc[
                same_hour["date"] == lag_7_date,
                "violation_count",
            ]

            last_7_start = next_date - pd.Timedelta(days=7)
            last_7 = same_hour[
                (same_hour["date"] >= last_7_start)
                & (same_hour["date"] < next_date)
            ]

            station_hour_history = panel[
                (panel["police_station"] == police_station)
                & (panel["hour"].astype(int) == hour)
            ]

            rows.append(
                {
                    **profile.to_dict(),
                    "date": next_date,
                    "prediction_date": next_date.date(),
                    "hour": hour,
                    "day_of_week": next_date.day_name(),
                    "month": int(next_date.month),
                    "is_weekend": int(
                        next_date.day_name() in ["Saturday", "Sunday"]
                    ),
                    "hotspot_avg_count": (
                        float(hotspot_history["violation_count"].mean())
                        if not hotspot_history.empty
                        else 0.0
                    ),
                    "station_hour_avg_count": (
                        float(station_hour_history["violation_count"].mean())
                        if not station_hour_history.empty
                        else 0.0
                    ),
                    "lag_1_day_count": (
                        float(lag_1.iloc[0]) if not lag_1.empty else 0.0
                    ),
                    "lag_7_day_count": (
                        float(lag_7.iloc[0]) if not lag_7.empty else 0.0
                    ),
                    "rolling_7_day_avg": (
                        float(last_7["violation_count"].mean())
                        if not last_7.empty
                        else 0.0
                    ),
                    "historical_total": (
                        float(hotspot_history["violation_count"].sum())
                        if not hotspot_history.empty
                        else 0.0
                    ),
                    "historical_avg": (
                        float(hotspot_history["violation_count"].mean())
                        if not hotspot_history.empty
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


def prepare_prediction_features(grid_df: pd.DataFrame) -> pd.DataFrame:
    missing_features = [col for col in FEATURE_COLUMNS if col not in grid_df.columns]
    if missing_features:
        raise ValueError(f"Prediction grid missing required features: {missing_features}")

    X = grid_df[FEATURE_COLUMNS].copy()
    X = fill_missing_categoricals(X, CATEGORICAL_FEATURES)

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X


def predict_with_model(X: pd.DataFrame, metadata: Dict) -> np.ndarray:
    model_type = metadata.get("model_type", "")

    if model_type == "CatBoostRegressor":
        model = load_joblib(CATBOOST_MODEL_PATH)
        preds = model.predict(X)
        return np.maximum(preds, 0)

    model = load_joblib(RF_MODEL_PATH)

    X_encoded = X.copy()
    encoders = metadata.get("fallback_encoders", {})

    for col in CATEGORICAL_FEATURES:
        if col in X_encoded.columns:
            mapping = encoders.get(col, {})
            X_encoded[col] = (
                X_encoded[col]
                .fillna("Unknown")
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )

    preds = model.predict(X_encoded)
    return np.maximum(preds, 0)


def generate_predictions() -> pd.DataFrame:
    metadata = load_joblib(METADATA_PATH)
    panel = load_training_panel()

    grid = build_future_prediction_grid(panel)

    X = prepare_prediction_features(grid)
    grid["predicted_violation_count"] = predict_with_model(X, metadata)

    best = (
        grid.sort_values(
            ["hotspot_id", "predicted_violation_count"],
            ascending=[True, False],
        )
        .groupby("hotspot_id", as_index=False)
        .head(1)
        .copy()
    )

    best["risk_level"] = make_risk_levels(best["predicted_violation_count"])

    if best["predicted_violation_count"].nunique() > 1:
        best["priority_score"] = (
            best["predicted_violation_count"].rank(method="dense", pct=True) * 100
        ).round(2)
    else:
        best["priority_score"] = 100.0

    best["recommended_enforcement_time"] = best["hour"].apply(
        recommended_time_from_hour
    )

    best = calculate_proxy_congestion_scores(best)

    output_cols = [
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
        "prediction_date",
        "traffic_data_source",
        "live_traffic_score",
        "traffic_delay_ratio",
        "normal_duration_seconds",
        "traffic_duration_seconds",
        "current_speed_kmph",
        "free_flow_speed_kmph",
        "traffic_confidence",
        "road_closure",
        "dominant_vehicle_type",
        "dominant_violation_type",
        "dominant_offence_code",
        "historical_total",
        "historical_avg",
        "lag_1_day_count",
        "lag_7_day_count",
        "rolling_7_day_avg",
        "hotspot_avg_count",
        "station_hour_avg_count",
    ]

    existing_output_cols = [col for col in output_cols if col in best.columns]

    best = best[existing_output_cols].sort_values(
        ["final_enforcement_score", "priority_score", "predicted_violation_count"],
        ascending=False,
    )

    # Do NOT include latitude_center / longitude_center here.
    # They must stay precise for map and TomTom.
    round_cols = [
        "predicted_violation_count",
        "priority_score",
        "congestion_impact_score",
        "final_enforcement_score",
        "historical_total",
        "historical_avg",
        "rolling_7_day_avg",
        "hotspot_avg_count",
        "station_hour_avg_count",
        "live_traffic_score",
        "traffic_delay_ratio",
        "normal_duration_seconds",
        "traffic_duration_seconds",
        "current_speed_kmph",
        "free_flow_speed_kmph",
        "traffic_confidence",
    ]

    for col in round_cols:
        if col in best.columns:
            best[col] = pd.to_numeric(best[col], errors="coerce").fillna(0).round(2)

    # Keep coordinates accurate up to 6 decimals.
    best["latitude_center"] = pd.to_numeric(
        best["latitude_center"], errors="coerce"
    ).fillna(0).round(6)

    best["longitude_center"] = pd.to_numeric(
        best["longitude_center"], errors="coerce"
    ).fillna(0).round(6)

    if "road_latitude" in best.columns:
        best["road_latitude"] = pd.to_numeric(
            best["road_latitude"], errors="coerce"
        ).fillna(0).round(6)

    if "road_longitude" in best.columns:
        best["road_longitude"] = pd.to_numeric(
            best["road_longitude"], errors="coerce"
        ).fillna(0).round(6)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(PREDICTIONS_PATH, index=False)

    return best


if __name__ == "__main__":
    print("Generating hotspot predictions...")
    predictions = generate_predictions()

    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print("\nTop 10 enforcement priorities:")
    print(predictions.head(10).to_string(index=False))