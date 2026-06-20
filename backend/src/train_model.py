from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .preprocess import preprocess_raw_data
    from .utils import (
        AGGREGATED_DATA_PATH,
        CATEGORICAL_FEATURES,
        CATBOOST_MODEL_PATH,
        FEATURE_COLUMNS,
        METADATA_PATH,
        METRICS_PATH,
        NUMERIC_FEATURES,
        RF_MODEL_PATH,
        TARGET_COLUMN,
        ensure_directories,
        fill_missing_categoricals,
        save_joblib,
        save_metrics_text,
    )
except ImportError:
    from preprocess import preprocess_raw_data
    from utils import (
        AGGREGATED_DATA_PATH,
        CATEGORICAL_FEATURES,
        CATBOOST_MODEL_PATH,
        FEATURE_COLUMNS,
        METADATA_PATH,
        METRICS_PATH,
        NUMERIC_FEATURES,
        RF_MODEL_PATH,
        TARGET_COLUMN,
        ensure_directories,
        fill_missing_categoricals,
        save_joblib,
        save_metrics_text,
    )


def load_or_create_training_data(force_preprocess: bool = False) -> pd.DataFrame:
    required_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN, "date"])

    if not force_preprocess and AGGREGATED_DATA_PATH.exists():
        df = pd.read_csv(AGGREGATED_DATA_PATH)
        if required_cols.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df.dropna(subset=["date"]).copy()
        print("Old or incomplete aggregated file detected. Regenerating...")

    df, _, _ = preprocess_raw_data()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).copy()


def time_based_split(df: pd.DataFrame, validation_fraction: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(df["date"].dropna().unique())
    if len(unique_dates) < 5:
        raise ValueError("Need at least 5 unique dates for time-based validation split.")

    split_index = int(len(unique_dates) * (1 - validation_fraction))
    split_index = max(1, min(split_index, len(unique_dates) - 1))
    cutoff_date = unique_dates[split_index]

    train_df = df[df["date"] < cutoff_date].copy()
    valid_df = df[df["date"] >= cutoff_date].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("Time-based split produced empty train/validation data.")
    return train_df, valid_df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")

    df = fill_missing_categoricals(df, CATEGORICAL_FEATURES)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0)
    return X, y


def train_catboost_model(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
    from catboost import CatBoostRegressor

    cat_feature_indices = [
        X_train.columns.get_loc(col)
        for col in CATEGORICAL_FEATURES
        if col in X_train.columns
    ]

    model = CatBoostRegressor(
        iterations=700,
        learning_rate=0.06,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )
    return model


def encode_for_random_forest(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]]]:
    X_train_encoded = X_train.copy()
    X_valid_encoded = X_valid.copy()
    encoders: Dict[str, Dict[str, int]] = {}

    for col in CATEGORICAL_FEATURES:
        mapping = {
            category: idx
            for idx, category in enumerate(sorted(X_train_encoded[col].fillna("Unknown").astype(str).unique()))
        }
        encoders[col] = mapping
        X_train_encoded[col] = X_train_encoded[col].fillna("Unknown").astype(str).map(mapping).fillna(-1).astype(int)
        X_valid_encoded[col] = X_valid_encoded[col].fillna("Unknown").astype(str).map(mapping).fillna(-1).astype(int)

    return X_train_encoded, X_valid_encoded, encoders


def train_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series):
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=24,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred = np.maximum(y_pred, 0)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2 Score": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    ensure_directories()
    print("Loading/preparing training data...")
    df = load_or_create_training_data()
    print(f"Training rows before split: {len(df):,}")
    print("Target: total violation_count per hotspot_id + date + hour")

    train_df, valid_df = time_based_split(df, validation_fraction=0.20)
    X_train, y_train = prepare_features(train_df)
    X_valid, y_valid = prepare_features(valid_df)

    metadata: Dict = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET_COLUMN,
        "target_definition": "total violation_count grouped by hotspot_id + date + hour",
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "train_start_date": str(train_df["date"].min().date()),
        "train_end_date": str(train_df["date"].max().date()),
        "valid_start_date": str(valid_df["date"].min().date()),
        "valid_end_date": str(valid_df["date"].max().date()),
    }

    try:
        print("Training CatBoostRegressor...")
        model = train_catboost_model(X_train, y_train, X_valid, y_valid)
        y_pred = model.predict(X_valid)
        save_joblib(model, CATBOOST_MODEL_PATH)
        metadata["model_type"] = "CatBoostRegressor"
        metadata["model_path"] = str(CATBOOST_MODEL_PATH)
    except Exception as exc:
        print("\nCatBoost failed or is unavailable. Using RandomForestRegressor fallback.")
        print(f"Reason: {exc}\n")
        X_train_rf, X_valid_rf, encoders = encode_for_random_forest(X_train, X_valid)
        model = train_random_forest_model(X_train_rf, y_train)
        y_pred = model.predict(X_valid_rf)
        save_joblib(model, RF_MODEL_PATH)
        metadata["model_type"] = "RandomForestRegressor"
        metadata["model_path"] = str(RF_MODEL_PATH)
        metadata["fallback_encoders"] = encoders

    metrics = evaluate_model(y_valid, y_pred)
    metadata.update(metrics)
    save_joblib(metadata, METADATA_PATH)

    metrics_for_text = {
        "Model Type": metadata["model_type"],
        "Target": metadata["target_definition"],
        "Train Rows": metadata["train_rows"],
        "Validation Rows": metadata["valid_rows"],
        "Train Date Range": f"{metadata['train_start_date']} to {metadata['train_end_date']}",
        "Validation Date Range": f"{metadata['valid_start_date']} to {metadata['valid_end_date']}",
        **metrics,
    }
    save_metrics_text(metrics_for_text, METRICS_PATH)

    print("\nTraining complete.")
    print(f"Model saved at: {metadata['model_path']}")
    print(f"Metadata saved at: {METADATA_PATH}")
    print(f"Metrics saved at: {METRICS_PATH}")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
