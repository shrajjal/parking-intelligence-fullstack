from __future__ import annotations

import os
from typing import Dict

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

COMMUTE_PEAK_HOURS = {7, 8, 9, 17, 18, 19, 20}
NEAR_PEAK_HOURS = {6, 10, 16, 21, 22}


def calculate_proxy_congestion_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback congestion impact calculation.

    Works without live traffic API.
    """
    df = df.copy()

    if df.empty:
        return df

    df["parking_intensity_score"] = (
        df["predicted_violation_count"].rank(method="dense", pct=True) * 100
    ).round(2)

    if "historical_total" in df.columns:
        df["historical_hotspot_score"] = (
            df["historical_total"].rank(method="dense", pct=True) * 100
        ).round(2)
    else:
        df["historical_hotspot_score"] = df["parking_intensity_score"]

    def peak_hour_score(hour: int) -> int:
        hour = int(hour)
        if hour in COMMUTE_PEAK_HOURS:
            return 100
        if hour in NEAR_PEAK_HOURS:
            return 70
        return 40

    df["peak_hour_score"] = df["hour"].apply(peak_hour_score)

    def junction_confidence_score(name: str) -> int:
        name = str(name).strip().lower()
        if name in {"no junction", "unmapped location", "unknown", "nan", ""}:
            return 65
        return 100

    df["junction_confidence_score"] = df["junction_name"].apply(
        junction_confidence_score
    )

    df["congestion_impact_score"] = (
        0.45 * df["parking_intensity_score"]
        + 0.25 * df["historical_hotspot_score"]
        + 0.20 * df["peak_hour_score"]
        + 0.10 * df["junction_confidence_score"]
    ).round(2)

    if "priority_score" not in df.columns:
        df["priority_score"] = df["parking_intensity_score"]

    df["final_enforcement_score"] = (
        0.60 * df["priority_score"]
        + 0.40 * df["congestion_impact_score"]
    ).round(2)

    default_live_cols = {
        "live_traffic_score": 0.0,
        "traffic_delay_ratio": 1.0,
        "normal_duration_seconds": 0.0,
        "traffic_duration_seconds": 0.0,
        "current_speed_kmph": 0.0,
        "free_flow_speed_kmph": 0.0,
        "traffic_confidence": 0.0,
        "road_closure": False,
        "road_latitude": 0.0,
        "road_longitude": 0.0,
    }

    for col, default_value in default_live_cols.items():
        if col not in df.columns:
            df[col] = default_value

    df["traffic_data_source"] = "proxy"

    return df


class TomTomTrafficClient:
    """
    TomTom live traffic client.

    Uses TomTom Flow Segment Data API.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("TOMTOM_API_KEY", "").strip()
        self.base_url = (
            "https://api.tomtom.com/traffic/services/4/"
            "flowSegmentData/absolute/10/json"
        )

    def enabled(self) -> bool:
        return bool(self.api_key)

    def get_live_traffic_impact(self, lat: float, lon: float) -> Dict:
        """
        Fetch live traffic flow around a hotspot coordinate.

        TomTom expects:
        point=latitude,longitude
        """
        if not self.enabled():
            return self._fallback("TomTom API key not found")

        params = {
            "key": self.api_key,
            "point": f"{lat},{lon}",
            "unit": "kmph",
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=8)

            if response.status_code != 200:
                safe_text = response.text.replace(self.api_key, "***")
                return self._fallback(
                    f"TomTom API error {response.status_code}: {safe_text}"
                )

            data = response.json()
            flow = data.get("flowSegmentData", {})

            current_speed = float(flow.get("currentSpeed", 0) or 0)
            free_flow_speed = float(flow.get("freeFlowSpeed", 0) or 0)
            current_travel_time = float(flow.get("currentTravelTime", 0) or 0)
            free_flow_travel_time = float(flow.get("freeFlowTravelTime", 0) or 0)
            confidence = float(flow.get("confidence", 0) or 0)
            road_closure = bool(flow.get("roadClosure", False))

            coordinates = flow.get("coordinates", {}).get("coordinate", [])

            road_latitude = 0.0
            road_longitude = 0.0

            if coordinates:
                mid = coordinates[len(coordinates) // 2]
                road_latitude = float(mid.get("latitude", 0) or 0)
                road_longitude = float(mid.get("longitude", 0) or 0)

            if free_flow_travel_time > 0:
                delay_ratio = current_travel_time / free_flow_travel_time
            else:
                delay_ratio = 1.0

            if road_closure:
                live_traffic_score = 100.0
            else:
                live_traffic_score = min(
                    100.0,
                    max(0.0, (delay_ratio - 1.0) * 100),
                )

            return {
                "live_traffic_available": True,
                "traffic_data_source": "tomtom_live",
                "traffic_delay_ratio": round(delay_ratio, 3),
                "live_traffic_score": round(live_traffic_score, 2),
                "traffic_duration_seconds": round(current_travel_time, 2),
                "normal_duration_seconds": round(free_flow_travel_time, 2),
                "current_speed_kmph": round(current_speed, 2),
                "free_flow_speed_kmph": round(free_flow_speed, 2),
                "traffic_confidence": round(confidence, 2),
                "road_closure": road_closure,
                "road_latitude": round(road_latitude, 6),
                "road_longitude": round(road_longitude, 6),
            }

        except Exception as exc:
            safe_error = str(exc).replace(self.api_key, "***")
            return self._fallback(f"TomTom request failed: {safe_error}")

    @staticmethod
    def _fallback(reason: str) -> Dict:
        return {
            "live_traffic_available": False,
            "traffic_data_source": "proxy",
            "traffic_delay_ratio": 1.0,
            "live_traffic_score": 0.0,
            "traffic_duration_seconds": 0.0,
            "normal_duration_seconds": 0.0,
            "current_speed_kmph": 0.0,
            "free_flow_speed_kmph": 0.0,
            "traffic_confidence": 0.0,
            "road_closure": False,
            "road_latitude": 0.0,
            "road_longitude": 0.0,
            "debug_reason": reason,
        }


def enrich_with_live_traffic(
    df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Add TomTom live traffic impact for top N hotspots only.
    """
    df = calculate_proxy_congestion_scores(df)

    client = TomTomTrafficClient()

    if not client.enabled():
        return df

    df = df.sort_values("final_enforcement_score", ascending=False).copy()

    for idx, row in df.head(top_n).iterrows():
        live = client.get_live_traffic_impact(
            lat=float(row["latitude_center"]),
            lon=float(row["longitude_center"]),
        )

        if live.get("live_traffic_available"):
            df.loc[idx, "traffic_data_source"] = "tomtom_live"
            df.loc[idx, "live_traffic_score"] = live["live_traffic_score"]
            df.loc[idx, "traffic_delay_ratio"] = live["traffic_delay_ratio"]
            df.loc[idx, "traffic_duration_seconds"] = live[
                "traffic_duration_seconds"
            ]
            df.loc[idx, "normal_duration_seconds"] = live[
                "normal_duration_seconds"
            ]
            df.loc[idx, "current_speed_kmph"] = live["current_speed_kmph"]
            df.loc[idx, "free_flow_speed_kmph"] = live["free_flow_speed_kmph"]
            df.loc[idx, "traffic_confidence"] = live["traffic_confidence"]
            df.loc[idx, "road_closure"] = live["road_closure"]

            df.loc[idx, "road_latitude"] = live.get("road_latitude", 0.0)
            df.loc[idx, "road_longitude"] = live.get("road_longitude", 0.0)

            df.loc[idx, "congestion_impact_score"] = round(
                0.60 * df.loc[idx, "congestion_impact_score"]
                + 0.40 * live["live_traffic_score"],
                2,
            )

            df.loc[idx, "final_enforcement_score"] = round(
                0.60 * df.loc[idx, "priority_score"]
                + 0.40 * df.loc[idx, "congestion_impact_score"],
                2,
            )
        else:
            df.loc[idx, "traffic_data_source"] = "proxy"

    return df.sort_values(
        "final_enforcement_score",
        ascending=False,
    ).reset_index(drop=True)