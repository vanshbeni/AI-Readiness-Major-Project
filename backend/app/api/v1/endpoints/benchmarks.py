from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset
from app.core.database import get_db
from app.models.dataset import Dataset
from app.schemas.dataset import ModelBenchmarkLeaderboardResponse
from app.services.pipeline import ServiceError, benchmark

router = APIRouter()


@router.post("/{dataset_id}/benchmark", response_model=ModelBenchmarkLeaderboardResponse, status_code=status.HTTP_200_OK)
def run_model_benchmarks(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """Cross-validates candidate models on the cleaned dataset and returns a ranked leaderboard."""
    try:
        return benchmark(db, db_dataset)
    except ServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
