from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

try:
    from .utils import (
        AGGREGATED_DATA_PATH,
        DBSCAN_MODEL_PATH,
        DROP_COLUMNS,
        HOTSPOT_SUMMARY_PATH,
        RAW_DATA_PATH,
        add_time_features,
        check_dataset_exists,
        clean_stringified_list,
        ensure_directories,
        fill_missing_categoricals,
        mode_or_unknown,
        normalize_junction_name,
        save_joblib,
    )
except ImportError:
    from utils import (
        AGGREGATED_DATA_PATH,
        DBSCAN_MODEL_PATH,
        DROP_COLUMNS,
        HOTSPOT_SUMMARY_PATH,
        RAW_DATA_PATH,
        add_time_features,
        check_dataset_exists,
        clean_stringified_list,
        ensure_directories,
        fill_missing_categoricals,
        mode_or_unknown,
        normalize_junction_name,
        save_joblib,
    )


def load_raw_data(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    check_dataset_exists(csv_path)
    return pd.read_csv(csv_path)


def project_lat_lon_to_meters(lat: pd.Series, lon: pd.Series, mean_lat: float) -> Tuple[pd.Series, pd.Series]:
    """Convert latitude/longitude into approximate local meter coordinates."""
    lat_m = lat.astype(float) * 110_540.0
    lon_m = lon.astype(float) * 111_320.0 * np.cos(np.radians(mean_lat))
    return lat_m, lon_m


def apply_dbscan_clustering(
    df: pd.DataFrame,
    eps_meters: float = 150.0,
    min_samples: int = 15,
    grid_size_meters: float = 25.0,
) -> Tuple[pd.DataFrame, DBSCAN]:
    """
    Fast location clustering using DBSCAN.

    It converts latitude/longitude into meter coordinates, aggregates points into
    small grid cells, then runs DBSCAN with sample_weight. This is much faster
    than DBSCAN over all raw violation rows.
    """
    df = df.copy()
    mean_lat = float(df["latitude"].mean())
    x_m, y_m = project_lat_lon_to_meters(df["latitude"], df["longitude"], mean_lat)

    df["grid_x"] = np.floor(x_m / grid_size_meters).astype(int)
    df["grid_y"] = np.floor(y_m / grid_size_meters).astype(int)

    grid = (
        df.groupby(["grid_x", "grid_y"], as_index=False)
        .agg(
            x_center=("grid_x", lambda s: (float(s.iloc[0]) + 0.5) * grid_size_meters),
            y_center=("grid_y", lambda s: (float(s.iloc[0]) + 0.5) * grid_size_meters),
            point_count=("grid_x", "size"),
        )
    )

    dbscan = DBSCAN(
        eps=eps_meters,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    )

    labels = dbscan.fit_predict(
        grid[["x_center", "y_center"]].to_numpy(),
        sample_weight=grid["point_count"].to_numpy(),
    )
    grid["cluster_label"] = labels

    df = df.merge(grid[["grid_x", "grid_y", "cluster_label"]], on=["grid_x", "grid_y"], how="left")
    df["cluster_label"] = df["cluster_label"].fillna(-1).astype(int)
    df["hotspot_id"] = df["cluster_label"].apply(lambda x: f"H{int(x)}" if int(x) >= 0 else "NOISE")

    dbscan.mean_latitude_ = mean_lat
    dbscan.grid_size_meters_ = grid_size_meters
    dbscan.eps_meters_ = eps_meters
    dbscan.cluster_grid_ = grid

    return df, dbscan


def calculate_hotspot_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = (
        df.groupby("hotspot_id", as_index=False)
        .agg(
            latitude_center=("latitude", "mean"),
            longitude_center=("longitude", "mean"),
            total_violations=("hotspot_id", "size"),
            junction_name=("junction_name", mode_or_unknown),
            police_station=("police_station", mode_or_unknown),
            dominant_vehicle_type=("vehicle_type", mode_or_unknown),
            dominant_violation_type=("violation_type", mode_or_unknown),
            dominant_offence_code=("offence_code", mode_or_unknown),
        )
        .sort_values("total_violations", ascending=False)
        .reset_index(drop=True)
    )
    profile["junction_name"] = profile["junction_name"].apply(normalize_junction_name)
    return profile


def create_zero_filled_panel(hourly_counts: pd.DataFrame, hotspot_profile: pd.DataFrame) -> pd.DataFrame:
    """
    Create hotspot-date-hour panel and fill missing rows with zero violations.

    The original CSV only contains rows where violations happened. This panel
    also teaches the model about 0-count periods.
    """
    min_date = pd.to_datetime(hourly_counts["date"]).min().normalize()
    max_date = pd.to_datetime(hourly_counts["date"]).max().normalize()
    all_dates = pd.date_range(min_date, max_date, freq="D")
    all_hours = list(range(24))
    all_hotspots = hotspot_profile["hotspot_id"].unique().tolist()

    index = pd.MultiIndex.from_product(
        [all_hotspots, all_dates, all_hours],
        names=["hotspot_id", "date", "hour"],
    )
    panel = index.to_frame(index=False)

    hourly_counts = hourly_counts.copy()
    hourly_counts["date"] = pd.to_datetime(hourly_counts["date"]).dt.normalize()
    hourly_counts["hour"] = hourly_counts["hour"].astype(int)

    panel = panel.merge(hourly_counts, on=["hotspot_id", "date", "hour"], how="left")
    panel["violation_count"] = panel["violation_count"].fillna(0).astype(float)

    panel = panel.merge(hotspot_profile, on="hotspot_id", how="left")
    panel["day_of_week"] = panel["date"].dt.day_name()
    panel["month"] = panel["date"].dt.month.astype(int)
    panel["is_weekend"] = panel["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return panel


def add_lag_and_average_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add time-aware historical features without future leakage."""
    panel = panel.copy()

    panel = panel.sort_values(["hotspot_id", "hour", "date"]).reset_index(drop=True)
    hotspot_hour_group = panel.groupby(["hotspot_id", "hour"], sort=False)["violation_count"]
    panel["lag_1_day_count"] = hotspot_hour_group.shift(1)
    panel["lag_7_day_count"] = hotspot_hour_group.shift(7)
    panel["rolling_7_day_avg"] = hotspot_hour_group.transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
    )

    panel = panel.sort_values(["hotspot_id", "date", "hour"]).reset_index(drop=True)
    panel["hotspot_avg_count"] = panel.groupby("hotspot_id", sort=False)["violation_count"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    panel = panel.sort_values(["police_station", "hour", "date", "hotspot_id"]).reset_index(drop=True)
    panel["station_hour_avg_count"] = panel.groupby(["police_station", "hour"], sort=False)["violation_count"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    feature_cols = [
        "lag_1_day_count",
        "lag_7_day_count",
        "rolling_7_day_avg",
        "hotspot_avg_count",
        "station_hour_avg_count",
    ]
    panel[feature_cols] = panel[feature_cols].fillna(0)
    panel = panel.sort_values(["date", "hour", "hotspot_id"]).reset_index(drop=True)
    return panel


def preprocess_raw_data(
    csv_path: Path = RAW_DATA_PATH,
    eps_meters: float = 150.0,
    min_samples: int = 15,
    min_hotspot_violations: int = 20,
    drop_noise: bool = True,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, DBSCAN]:
    ensure_directories()
    df = load_raw_data(csv_path)

    required_cols = ["latitude", "longitude", "created_datetime"]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset missing required columns: {missing_required}")

    df = df.dropna(subset=["latitude", "longitude", "created_datetime"]).copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))].copy()

    df = add_time_features(df, "created_datetime")
    df = df.dropna(subset=["created_datetime", "date", "hour", "month"]).copy()

    for col in ["violation_type", "offence_code"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_stringified_list)

    categorical_cols = ["junction_name", "police_station", "vehicle_type", "violation_type", "offence_code"]
    df = fill_missing_categoricals(df, categorical_cols)
    if "junction_name" in df.columns:
        df["junction_name"] = df["junction_name"].apply(normalize_junction_name)

    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns], errors="ignore")

    print("Applying DBSCAN clustering...")
    df, dbscan_model = apply_dbscan_clustering(df, eps_meters=eps_meters, min_samples=min_samples)

    if drop_noise:
        before = len(df)
        df = df[df["hotspot_id"] != "NOISE"].copy()
        print(f"Dropped DBSCAN noise records: {before - len(df):,}")

    cluster_sizes = df["hotspot_id"].value_counts()
    valid_hotspots = cluster_sizes[cluster_sizes >= min_hotspot_violations].index
    df = df[df["hotspot_id"].isin(valid_hotspots)].copy()
    if df.empty:
        raise ValueError("No hotspots left. Increase eps_meters or lower min_samples/min_hotspot_violations.")

    hotspot_profile = calculate_hotspot_profile(df)

    hourly_counts = (
        df.groupby(["hotspot_id", "date", "hour"], as_index=False)
        .size()
        .rename(columns={"size": "violation_count"})
    )

    panel = create_zero_filled_panel(hourly_counts, hotspot_profile)
    panel = add_lag_and_average_features(panel)

    if save_outputs:
        panel.to_csv(AGGREGATED_DATA_PATH, index=False)
        hotspot_profile.to_csv(HOTSPOT_SUMMARY_PATH, index=False)
        save_joblib(dbscan_model, DBSCAN_MODEL_PATH)

    return panel, hotspot_profile, dbscan_model


if __name__ == "__main__":
    print("Starting preprocessing...")
    aggregated_df, hotspot_summary_df, _ = preprocess_raw_data()
    print(f"Aggregated training data saved to: {AGGREGATED_DATA_PATH}")
    print(f"Hotspot summary saved to: {HOTSPOT_SUMMARY_PATH}")
    print(f"Rows in training dataset: {len(aggregated_df):,}")
    print(f"Total hotspots used: {hotspot_summary_df['hotspot_id'].nunique():,}")
    print("Target: total violation_count per hotspot_id + date + hour")
