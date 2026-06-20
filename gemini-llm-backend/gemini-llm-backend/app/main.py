from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.llm_service import GeminiLLMService
from app.schemas import (
    EnforcementReportRequest,
    EnforcementReportResponse,
    HotspotExplanationRequest,
    HotspotExplanationResponse,
)


settings = get_settings()
llm_service = GeminiLLMService(settings)

app = FastAPI(
    title="Parking Intelligence Gemini LLM Backend",
    description="Separate FastAPI backend for Gemini-based hotspot explanation and enforcement reports.",
    version="1.0.0",
)

cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Gemini LLM backend is running",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model": settings.gemini_model,
        "mock_llm": settings.mock_llm,
    }


@app.post("/api/explain-hotspot", response_model=HotspotExplanationResponse)
def explain_hotspot(payload: HotspotExplanationRequest):
    explanation = llm_service.explain_hotspot(payload)
    return HotspotExplanationResponse(
        hotspot_id=payload.hotspot_id,
        explanation=explanation,
    )


@app.post("/api/generate-report", response_model=EnforcementReportResponse)
def generate_report(payload: EnforcementReportRequest):
    report = llm_service.generate_report(payload)
    return EnforcementReportResponse(report=report)
