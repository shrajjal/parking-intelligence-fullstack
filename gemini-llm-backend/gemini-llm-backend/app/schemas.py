from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HotspotExplanationRequest(BaseModel):
    hotspot_id: int | str = Field(..., examples=[4])
    location: Optional[str] = Field(default="Unknown location", examples=["Sakchi Junction"])
    police_station: Optional[str] = Field(default="Unknown police station", examples=["Sakchi"])
    junction_name: Optional[str] = Field(default=None, examples=["Sakchi Roundabout"])

    hour: Optional[int] = Field(default=None, ge=0, le=23, examples=[18])
    day_of_week: Optional[str] = Field(default=None, examples=["Monday"])
    month: Optional[int] = Field(default=None, ge=1, le=12, examples=[6])

    predicted_violation_count: Optional[float] = Field(default=None, ge=0, examples=[22])
    risk_score: Optional[float] = Field(default=None, ge=0, examples=[0.86])
    risk_level: Optional[str] = Field(default=None, examples=["High"])
    recommended_time: Optional[str] = Field(default=None, examples=["6 PM - 8 PM"])

    current_speed: Optional[float] = Field(default=None, ge=0, examples=[14])
    free_flow_speed: Optional[float] = Field(default=None, ge=0, examples=[48])
    congestion_score: Optional[float] = Field(default=None, ge=0, examples=[0.72])
    traffic_delay: Optional[str] = Field(default=None, examples=["High"])
    road_closure: Optional[bool] = Field(default=None, examples=[False])

    common_vehicle_type: Optional[str] = Field(default=None, examples=["Two-wheeler"])
    common_violation_type: Optional[str] = Field(default=None, examples=["No Parking"])


class HotspotExplanationResponse(BaseModel):
    hotspot_id: int | str
    explanation: str


class EnforcementReportRequest(BaseModel):
    report_title: str = Field(default="Daily Parking Enforcement Summary")
    city_or_area: str = Field(default="Selected Area")
    total_hotspots: Optional[int] = Field(default=None, ge=0)
    high_risk_count: Optional[int] = Field(default=None, ge=0)
    top_hotspots: List[HotspotExplanationRequest] = Field(default_factory=list)


class EnforcementReportResponse(BaseModel):
    report: str
