import sys
from pathlib import Path

# Add project root directory to sys.path so 'app' module imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import init_db
from app.core.storage import cleanup_expired_files
from app.api.v1.router import api_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up AI Data Readiness Platform Backend...")
    init_db()
    removed = cleanup_expired_files()
    if removed:
        logger.info(f"Removed {removed} stored file(s) older than {settings.FILE_RETENTION_DAYS} days.")
    yield
    logger.info("Shutting down AI Data Readiness Platform Backend...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Explainable Pre-ML Data Diagnosis, Cleaning & Model Recommendation Engine",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {
        "message": "AI Data Readiness Platform API is Running",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
