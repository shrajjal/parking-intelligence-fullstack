from pathlib import Path
import pandas as pd

from utils import RAW_DATA_PATH, clean_stringified_list, normalize_junction_name, safe_datetime

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RAW_DATA_PATH)
df = df.dropna(subset=["latitude", "longitude", "created_datetime"]).copy()

df["created_datetime"] = safe_datetime(df["created_datetime"])
df = df.dropna(subset=["created_datetime"]).copy()

df["hour"] = df["created_datetime"].dt.hour

if "violation_type" in df.columns:
    df["violation_type"] = df["violation_type"].apply(clean_stringified_list)

for col in ["police_station", "junction_name", "vehicle_type", "violation_type"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

if "junction_name" in df.columns:
    df["junction_name"] = df["junction_name"].apply(normalize_junction_name)

summary = {
    "total_violations": int(len(df)),
    "police_stations": int(df["police_station"].nunique()),
    "junctions": int(df["junction_name"].nunique()),
    "vehicle_types": int(df["vehicle_type"].nunique()),
}

pd.Series(summary).to_json(OUTPUT_DIR / "dashboard_summary.json")

df["police_station"].value_counts().head(15).reset_index().rename(
    columns={"police_station": "name", "count": "violations"}
).to_csv(OUTPUT_DIR / "dashboard_top_stations.csv", index=False)

df[df["junction_name"].astype(str) != "Unmapped Location"]["junction_name"].value_counts().head(15).reset_index().rename(
    columns={"junction_name": "name", "count": "violations"}
).to_csv(OUTPUT_DIR / "dashboard_top_junctions.csv", index=False)

df.groupby("hour", as_index=False).size().rename(
    columns={"size": "violations"}
).to_csv(OUTPUT_DIR / "dashboard_hourly_trend.csv", index=False)

df["vehicle_type"].value_counts().head(12).reset_index().rename(
    columns={"vehicle_type": "name", "count": "violations"}
).to_csv(OUTPUT_DIR / "dashboard_vehicle_types.csv", index=False)

print("Dashboard summary files exported successfully.")