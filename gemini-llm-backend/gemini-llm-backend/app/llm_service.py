from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException
from google import genai

from app.config import Settings
from app.schemas import EnforcementReportRequest, HotspotExplanationRequest


SENSITIVE_KEYS = {
    "vehicle_number",
    "device_id",
    "created_by_id",
    "id",
    "validation_timestamp",
}


def _safe_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Remove fields that should never be sent to the LLM."""
    return {key: value for key, value in data.items() if key not in SENSITIVE_KEYS and value is not None}


def _as_pretty_json(data: dict[str, Any]) -> str:
    return json.dumps(_safe_dict(data), indent=2, ensure_ascii=False)


def build_hotspot_prompt(hotspot: HotspotExplanationRequest) -> str:
    hotspot_json = _as_pretty_json(hotspot.model_dump())

    return f"""
You are an AI enforcement assistant for a traffic police parking intelligence system.

Your task:
Explain the selected illegal-parking hotspot in simple, practical language for traffic police officers.

Rules:
- Do not mention model internals like CatBoost, DBSCAN, APIs, JSON, or prompt.
- Do not invent personal data, vehicle numbers, challan IDs, or officer names.
- Base your answer only on the hotspot details given below.
- Keep the response concise and useful for a dashboard.
- Use this exact format:

Risk Explanation:
<2-3 lines explaining why this hotspot is risky>

Main Reasons:
- <reason 1>
- <reason 2>
- <reason 3>

Recommended Action:
<2-3 lines giving practical enforcement action>

Best Time to Act:
<one line>

Hotspot details:
{hotspot_json}
""".strip()


def build_report_prompt(report: EnforcementReportRequest) -> str:
    safe_top_hotspots = [_safe_dict(h.model_dump()) for h in report.top_hotspots]
    report_json = json.dumps(
        {
            "report_title": report.report_title,
            "city_or_area": report.city_or_area,
            "total_hotspots": report.total_hotspots,
            "high_risk_count": report.high_risk_count,
            "top_hotspots": safe_top_hotspots,
        },
        indent=2,
        ensure_ascii=False,
    )

    return f"""
You are an AI enforcement assistant for a traffic police parking intelligence system.

Generate a short daily enforcement report from the given hotspot data.

Rules:
- Do not mention APIs, ML model internals, JSON, or prompt.
- Do not invent personal data, vehicle numbers, challan IDs, or officer names.
- Use simple language for police/administration users.
- Keep it ready to show in a dashboard.
- Use this exact format:

Daily Enforcement Summary
<brief summary paragraph>

Priority Hotspots
1. <hotspot/location> — <why it needs priority>
2. <hotspot/location> — <why it needs priority>
3. <hotspot/location> — <why it needs priority>

Recommended Deployment Plan
- <deployment point 1>
- <deployment point 2>
- <deployment point 3>

Expected Impact
<short paragraph>

Report data:
{report_json}
""".strip()


def _mock_hotspot_response(hotspot: HotspotExplanationRequest) -> str:
    location = hotspot.junction_name or hotspot.location or f"Hotspot {hotspot.hotspot_id}"
    risk = hotspot.risk_level or "High"
    time_text = hotspot.recommended_time or "the predicted peak hour"

    return f"""
Risk Explanation:
{location} is marked as {risk} priority because the system predicts repeated illegal parking activity here. The location also appears to have traffic pressure, so unmanaged parking may slow movement further.

Main Reasons:
- Predicted violation count is high for this time window.
- Historical records show repeated parking violations around this area.
- Current traffic conditions suggest enforcement can reduce congestion.

Recommended Action:
Deploy an enforcement team near {location}. Focus on quick removal of illegally parked vehicles and repeated violation types.

Best Time to Act:
Take action during {time_text}.
""".strip()


def _mock_report_response(report: EnforcementReportRequest) -> str:
    return f"""
Daily Enforcement Summary
{report.city_or_area} has {report.high_risk_count or 'multiple'} high-priority parking hotspots today. Enforcement should focus on areas where predicted violations and congestion are both high.

Priority Hotspots
1. Hotspot 1 — High predicted illegal parking activity.
2. Hotspot 2 — Repeated violations during peak hours.
3. Hotspot 3 — Traffic movement may be affected by roadside parking.

Recommended Deployment Plan
- Deploy teams during the predicted peak violation window.
- Prioritize high-risk junctions and market-side roads.
- Focus on quick clearing and repeat-offender monitoring.

Expected Impact
Targeted enforcement can reduce roadside obstruction, improve vehicle flow, and help police use limited resources more efficiently.
""".strip()


class GeminiLLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None

        if not settings.mock_llm:
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is missing. Add it in your .env file or set MOCK_LLM=true.")
            self.client = genai.Client(api_key=settings.gemini_api_key)

    def _generate_text(self, prompt: str) -> str:
        if self.settings.mock_llm:
            return ""

        try:
            assert self.client is not None
            response = self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if not text:
                raise HTTPException(status_code=502, detail="Gemini returned an empty response.")
            return text.strip()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc

    def explain_hotspot(self, hotspot: HotspotExplanationRequest) -> str:
        if self.settings.mock_llm:
            return _mock_hotspot_response(hotspot)

        prompt = build_hotspot_prompt(hotspot)
        return self._generate_text(prompt)

    def generate_report(self, report: EnforcementReportRequest) -> str:
        if self.settings.mock_llm:
            return _mock_report_response(report)

        prompt = build_report_prompt(report)
        return self._generate_text(prompt)
