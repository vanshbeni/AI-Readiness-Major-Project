import os

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset
from app.core.database import get_db
from app.agents.policy import get_policy, load_prepared
from app.core.storage import json_safe, read_dataset
from app.engine.profiler import type_overrides
from app.engine.summary import build_summary
from app.engine.tasks import target_or_none
from app.engine.visualizer import build_explore_payload
from app.models.dataset import Dataset, DatasetSnapshot, Objective
from app.services.pipeline import assignments_file

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
    df, _, types = load_prepared(db, db_dataset, path)
    target = obj.target_column if obj and obj.target_column in df.columns else None
    with type_overrides(types):
        payload = build_explore_payload(df, target=target, max_rows=max_rows, max_cols=max_cols)
    payload["source"] = source
    payload["filename"] = db_dataset.filename
    payload["has_cleaned"] = bool(db_dataset.cleaned_file_path and os.path.exists(db_dataset.cleaned_file_path))
    return json_safe(payload)


@router.get("/{dataset_id}/summary")
def dataset_summary(
    source: str = Query("raw", pattern="^(raw|cleaned)$"),
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """Everything the summary page shows: full-data charts, target drivers, pipeline results and written insights."""
    has_cleaned = bool(db_dataset.cleaned_file_path and os.path.exists(db_dataset.cleaned_file_path))
    if source == "cleaned" and not has_cleaned:
        raise HTTPException(status_code=404, detail="No cleaned dataset yet. Apply the fixes first.")
    obj = db.query(Objective).filter(Objective.dataset_id == db_dataset.id).first()
    df, _, types = load_prepared(db, db_dataset, db_dataset.cleaned_file_path if source == "cleaned" else None)
    problem_type = obj.problem_type if obj else None
    target = target_or_none(problem_type, obj.target_column if obj else None)
    target = target if target in df.columns else None
    snap = db.query(DatasetSnapshot).filter(DatasetSnapshot.dataset_id == db_dataset.id).first()
    pol = get_policy(db, db_dataset.id)
    assignments = None
    path = assignments_file(db_dataset.id)
    if source == "cleaned" and os.path.exists(path):
        assignments = read_dataset(path)
    with type_overrides(types):
        payload = build_summary(
            df, target, problem_type,
            diagnostics=snap.diagnostics if snap else None,
            execution=snap.execution if snap else None,
            leaderboard=snap.leaderboard if snap else None,
            report=pol.report if pol else None,
            assignments=assignments,
        )
    payload.update(source=source, filename=db_dataset.filename, has_cleaned=has_cleaned, dataset_id=db_dataset.id)
    return json_safe(payload)
