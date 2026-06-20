from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    allowed_origins: List[str]
    mock_llm: bool


def get_settings() -> Settings:
    origins_raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    )
    allowed_origins = [origin.strip() for origin in origins_raw.split(",") if origin.strip()]

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        allowed_origins=allowed_origins,
        mock_llm=os.getenv("MOCK_LLM", "false").lower() in {"1", "true", "yes", "y"},
    )
