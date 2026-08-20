import os
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.dataset import Dataset, Objective
from app.schemas.dataset import ObjectiveCreate, ObjectiveResponse

router = APIRouter()


@router.post("/{dataset_id}/objective", response_model=ObjectiveResponse, status_code=status.HTTP_200_OK)
async def set_objective(
    dataset_id: str,
    payload: ObjectiveCreate,
    db: Session = Depends(get_db),
):
    """Sets or updates the ML objective and target column for a dataset."""
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not os.path.exists(db_dataset.raw_file_path):
        raise HTTPException(status_code=404, detail="Dataset file missing on server.")

    # Validate column exists
    ext = os.path.splitext(db_dataset.raw_file_path)[1].lower()
    df_head = pd.read_csv(db_dataset.raw_file_path, nrows=5) if ext == ".csv" else pd.read_excel(db_dataset.raw_file_path, nrows=5)
    
    if payload.target_column not in df_head.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{payload.target_column}' does not exist in dataset. Available: {list(df_head.columns)}"
        )

    prob_type = payload.problem_type.lower()
    if prob_type not in ["classification", "regression"]:
        raise HTTPException(status_code=400, detail="Problem type must be either 'classification' or 'regression'.")

    # Update or Create Objective
    existing_obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    if existing_obj:
        existing_obj.problem_type = prob_type
        existing_obj.target_column = payload.target_column
        db.commit()
        db.refresh(existing_obj)
        return existing_obj
    else:
        new_obj = Objective(
            dataset_id=dataset_id,
            problem_type=prob_type,
            target_column=payload.target_column,
        )
        db.add(new_obj)
        db.commit()
        db.refresh(new_obj)
        return new_obj


@router.get("/{dataset_id}/objective", response_model=ObjectiveResponse)
async def get_objective(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """Gets the current ML objective and target column."""
    obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Objective not set for this dataset.")
    return obj
