from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATA_FILE_NAME = "jan to may police violation_anonymized791b166.csv"
RAW_DATA_PATH = DATA_DIR / DATA_FILE_NAME

AGGREGATED_DATA_PATH = OUTPUTS_DIR / "aggregated_hotspot_training_data.csv"
PREDICTIONS_PATH = OUTPUTS_DIR / "hotspot_predictions.csv"
HOTSPOT_SUMMARY_PATH = OUTPUTS_DIR / "hotspot_summary.csv"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.txt"

CATBOOST_MODEL_PATH = MODELS_DIR / "catboost_parking_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "random_forest_parking_model.pkl"
DBSCAN_MODEL_PATH = MODELS_DIR / "dbscan_cluster_model.pkl"
METADATA_PATH = MODELS_DIR / "encoders_or_metadata.pkl"

DROP_COLUMNS = [
    "id",
    "vehicle_number",
    "updated_vehicle_number",
    "description",
    "closed_datetime",
    "action_taken_timestamp",
    "validation_timestamp",
    "validation_status",
]

CATEGORICAL_FEATURES = [
    "hotspot_id",
    "junction_name",
    "police_station",
    "dominant_vehicle_type",
    "dominant_violation_type",
    "dominant_offence_code",
    "day_of_week",
]

NUMERIC_FEATURES = [
    "latitude_center",
    "longitude_center",
    "hour",
    "month",
    "is_weekend",
    "hotspot_avg_count",
    "station_hour_avg_count",
    "lag_1_day_count",
    "lag_7_day_count",
    "rolling_7_day_avg",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "violation_count"


def ensure_directories() -> None:
    for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def check_dataset_exists(path: Path = RAW_DATA_PATH) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n\n"
            "Place the CSV inside backend/data/ with this exact name:\n"
            f"{DATA_FILE_NAME}"
        )


def clean_stringified_list(value: Any) -> str:
    """Clean values like [\"WRONG PARKING\"] into WRONG PARKING."""
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            items = [str(item).strip() for item in parsed if str(item).strip()]
            return " | ".join(items) if items else "Unknown"
    except (ValueError, SyntaxError):
        pass
    return text.replace("[", "").replace("]", "").replace('"', "").strip() or "Unknown"


def safe_datetime(series):
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")


def add_time_features(df: pd.DataFrame, datetime_col: str = "created_datetime") -> pd.DataFrame:
    df = df.copy()

    df[datetime_col] = safe_datetime(df[datetime_col])

    df["date"] = df[datetime_col].dt.date
    df["hour"] = df[datetime_col].dt.hour.astype("Int64")
    df["day_of_week"] = df[datetime_col].dt.day_name()
    df["month"] = df[datetime_col].dt.month.astype("Int64")
    df["is_weekend"] = df[datetime_col].dt.dayofweek.isin([5, 6]).astype(int)

    return df


def fill_missing_categoricals(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    return df


def mode_or_unknown(series: pd.Series) -> str:
    series = series.dropna().astype(str).str.strip()
    if series.empty:
        return "Unknown"
    mode_values = series.mode()
    if mode_values.empty:
        return "Unknown"
    return str(mode_values.iloc[0])


def normalize_junction_name(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"no junction", "unknown", "nan", "none", ""}:
        return "Unmapped Location"
    return text


def make_risk_levels(predicted_values: pd.Series) -> pd.Series:
    values = predicted_values.fillna(0).astype(float)
    if len(values) == 0:
        return pd.Series([], dtype=str)
    if values.nunique() <= 1:
        return pd.Series(["Low"] * len(values), index=values.index)
    q50 = values.quantile(0.50)
    q80 = values.quantile(0.80)
    return pd.cut(
        values,
        bins=[-np.inf, q50, q80, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)


def recommended_time_from_hour(hour: int) -> str:
    try:
        hour = int(hour)
    except (ValueError, TypeError):
        return "Unknown"

    def fmt(h: int) -> str:
        h = h % 24
        if h == 0:
            return "12 AM"
        if h < 12:
            return f"{h} AM"
        if h == 12:
            return "12 PM"
        return f"{h - 12} PM"

    return f"{fmt(hour)} - {fmt(hour + 2)}"


def save_joblib(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return joblib.load(path)


def save_metrics_text(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")
