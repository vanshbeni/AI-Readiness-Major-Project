import os
from pathlib import Path
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Data Readiness Platform"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = (
        "postgresql://neondb_owner:npg_Jig5y4mTavZS@ep-little-bar-ax1d7myc-pooler.c-4.us-east-2.aws.neon.tech/neondb?sslmode=require"
    )
    
    # Gemini AI
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    # Storage Paths
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    RAW_DATA_DIR: str = str(BASE_DIR / "uploads" / "raw")
    CLEANED_DATA_DIR: str = str(BASE_DIR / "uploads" / "cleaned")
    ARTIFACTS_DIR: str = str(BASE_DIR / "uploads" / "artifacts")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "allow"


settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
os.makedirs(settings.CLEANED_DATA_DIR, exist_ok=True)
os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)
