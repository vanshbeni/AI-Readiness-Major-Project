from fastapi import APIRouter
from app.api.v1.endpoints import (
    health,
    datasets,
    objectives,
    diagnostics,
    recommendations,
    pipeline,
    benchmarks,
    exports,
)

api_router = APIRouter()

api_router.include_router(health.router, prefix="", tags=["Health"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(objectives.router, prefix="/datasets", tags=["Objectives"])
api_router.include_router(diagnostics.router, prefix="/datasets", tags=["Diagnostics"])
api_router.include_router(recommendations.router, prefix="/datasets", tags=["Recommendations"])
api_router.include_router(pipeline.router, prefix="/datasets", tags=["Pipeline Execution"])
api_router.include_router(benchmarks.router, prefix="/datasets", tags=["Model Benchmarks"])
api_router.include_router(exports.router, prefix="/datasets", tags=["Exports & Reports"])
