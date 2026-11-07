import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=str(BASE_DIR / ".env"),
        extra="allow",
    )

    PROJECT_NAME: str = "AI Data Readiness Platform"
    API_V1_STR: str = "/api/v1"

    # Database (set DATABASE_URL in backend/.env; defaults to a local SQLite file)
    DATABASE_URL: str = f"sqlite:///{(BASE_DIR / 'datareadiness.db').as_posix()}"

    # Agents / LLM access via OpenRouter (OpenAI-compatible API). Without a key every agent
    # falls back to its deterministic rules, so the app keeps working.
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    # "openrouter/free" routes each request to an available free model.
    OPENROUTER_MODEL: str = "openrouter/free"
    OPENROUTER_FALLBACK_MODELS: List[str] = ["nvidia/nemotron-3.5-super-vl:free", "google/gemma-4-31b-it:free"]
    LLM_TIMEOUT_SEC: float = 90.0
    LLM_MAX_RETRIES: int = 2
    # Analyst agent: max tool calls per question; every agent run: wall-clock budget
    AGENT_MAX_STEPS: int = 8
    AGENT_RUN_BUDGET_SEC: float = 300.0
    # What agents may send to the LLM: "schema" (names/types only), "aggregates" (+ stats), "samples" (+ redacted values)
    AI_PRIVACY_MODE: str = "samples"
    APP_PUBLIC_URL: str = "http://localhost:3000"

    # Storage Paths
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    RAW_DATA_DIR: str = str(BASE_DIR / "uploads" / "raw")
    CLEANED_DATA_DIR: str = str(BASE_DIR / "uploads" / "cleaned")
    ARTIFACTS_DIR: str = str(BASE_DIR / "uploads" / "artifacts")

    # Upload limits & retention
    MAX_UPLOAD_MB: int = 200
    MIN_ROWS: int = 10
    FILE_RETENTION_DAYS: int = 7

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
os.makedirs(settings.CLEANED_DATA_DIR, exist_ok=True)
os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)
