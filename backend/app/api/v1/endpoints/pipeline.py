from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset
from app.core.database import get_db
from app.models.dataset import Dataset
from app.schemas.dataset import ExecutionResponse
from app.services.pipeline import ServiceError, execute

router = APIRouter()


@router.post("/{dataset_id}/execute", response_model=ExecutionResponse, status_code=status.HTTP_200_OK)
def run_pipeline_execution(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """
    Executes the approved cleaning steps, stores the cleaned dataset and pipeline script,
    re-scores the cleaned data and reports before/after deltas.
    """
    try:
        return execute(db, db_dataset)
    except ServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
