import os

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset
from app.core.database import get_db
from app.core.storage import json_safe, read_dataset
from app.engine.visualizer import build_explore_payload
from app.models.dataset import Dataset, Objective

router = APIRouter()


@router.get("/{dataset_id}/explore")
def explore_dataset(
    source: str = Query("raw", pattern="^(raw|cleaned)$"),
    max_rows: int = Query(2000, ge=100, le=10000),
    max_cols: int = Query(60, ge=2, le=200),
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """Returns a sampled, typed, column-oriented view of the raw or cleaned dataset for interactive charts."""
    if source == "cleaned":
        path = db_dataset.cleaned_file_path
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="No cleaned dataset yet. Apply the fixes first.")
    else:
        path = db_dataset.raw_file_path

    obj = db.query(Objective).filter(Objective.dataset_id == db_dataset.id).first()
    df = read_dataset(path)
    target = obj.target_column if obj and obj.target_column in df.columns else None
    payload = build_explore_payload(df, target=target, max_rows=max_rows, max_cols=max_cols)
    payload["source"] = source
    payload["filename"] = db_dataset.filename
    payload["has_cleaned"] = bool(db_dataset.cleaned_file_path and os.path.exists(db_dataset.cleaned_file_path))
    return json_safe(payload)
