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

    # Gemini AI
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.8-flash"
    # Tried in order after GEMINI_MODEL when it is overloaded, rate-limited or unavailable
    GEMINI_FALLBACK_MODELS: List[str] = ["gemini-3.7-flash", "gemini-3.8-flash"]
    GEMINI_TIMEOUT_SEC: float = 30.0
    GEMINI_MAX_RETRIES: int = 2
    # Upper bound on total time spent waiting for Gemini before using statistical explanations
    GEMINI_TOTAL_BUDGET_SEC: float = 45.0

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
