from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset
from app.core.database import get_db
from app.models.dataset import Dataset
from app.schemas.dataset import FullDiagnosticResponse
from app.services.pipeline import ServiceError, run_diagnostics as run_diagnostics_service

router = APIRouter()


@router.post("/{dataset_id}/diagnose", response_model=FullDiagnosticResponse, status_code=status.HTTP_200_OK)
def run_diagnostics(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """
    Profiling -> Issue Detection -> Health Score -> Rule Matrix -> Explanations.
    Requires an objective. Re-running clears previous recommendations and any executed pipeline results.
    """
    try:
        return run_diagnostics_service(db, db_dataset)
    except ServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
