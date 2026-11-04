from typing import Dict, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset, get_snapshot, invalidate_execution
from app.core.database import get_db
from app.agents.policy import column_overrides, load_prepared, prepare, type_map
from app.core.storage import read_dataset
from app.engine.tasks import ALL_TASKS, LABELS, needs_target
from app.engine.benchmark import BenchmarkError, MIN_BENCHMARK_ROWS, prepare_target
from app.engine.profiler import infer_column_type, type_overrides
from app.models.dataset import Dataset, HealthScore, IssueDetection, Objective, ProfileReport, Recommendation
from app.schemas.dataset import ObjectiveCreate, ObjectiveResponse

router = APIRouter()


def validate_target(df: pd.DataFrame, target_column: str, problem_type: str) -> None:
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' does not exist in dataset. Available: {list(df.columns)}",
        )
    y = df[target_column].dropna()
    if len(y) < MIN_BENCHMARK_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Target '{target_column}' has only {len(y)} non-missing values; at least {MIN_BENCHMARK_ROWS} are required.",
        )
    if infer_column_type(df[target_column], target_column) == "id":
        raise HTTPException(status_code=400, detail=f"'{target_column}' is a unique identifier and cannot be a prediction target.")
    try:
        prepare_target(y, problem_type)
    except BenchmarkError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if problem_type == "classification":
        min_class = int(y.astype(str).value_counts().min())
        if min_class < 2:
            raise HTTPException(
                status_code=400,
                detail=f"At least one class of '{target_column}' has a single row; every class needs 2 or more rows.",
            )


def suggest_problem_type(series: pd.Series, col_name: str) -> Dict[str, Any]:
    col_type = infer_column_type(series, col_name)
    clean = series.dropna()
    if col_type == "id" or clean.empty:
        return {"suitable": False, "problem_type": None, "reason": "identifier or empty column"}
    if pd.api.types.is_numeric_dtype(clean):
        n_unique = clean.nunique()
        non_integer = bool((clean % 1 != 0).any())
        # Small integer codes (0/1, 1-5 ratings, 1-10 levels) are classes; a few distinct *amounts*
        # (salary bands 50000..70000) are still quantities on a numeric scale.
        span = float(clean.max() - clean.min())
        code_like = span <= max(20.0, 3.0 * n_unique)
        if n_unique <= 2 or (not non_integer and n_unique <= 10 and code_like):
            return {"suitable": True, "problem_type": "classification", "reason": f"{n_unique} discrete values"}
        if not non_integer and n_unique <= 10:
            return {"suitable": True, "problem_type": "regression",
                    "reason": f"numeric amounts with only {n_unique} distinct levels; classification of those bands also works"}
        return {"suitable": True, "problem_type": "regression", "reason": f"continuous numeric ({n_unique} values)"}
    n_unique = clean.nunique()
    if n_unique > max(20, 0.5 * len(clean)):
        return {"suitable": False, "problem_type": None, "reason": "free text with too many distinct values"}
    return {"suitable": True, "problem_type": "classification", "reason": f"{n_unique} categories"}


@router.get("/{dataset_id}/target-suggestions")
def get_target_suggestions(db_dataset: Dataset = Depends(get_owned_dataset), db: Session = Depends(get_db)):
    """Suggests a problem type for each column so the UI can pre-select a sensible objective."""
    overrides = column_overrides(db, db_dataset.id)
    df, _, auto = prepare(read_dataset(db_dataset.raw_file_path), overrides)
    notes: Dict[str, list] = {}
    for st in auto:
        if st["kind"] == "disguised_missing":
            notes.setdefault(st["column"], []).append(
                f"{st['count']} placeholder value(s) ({', '.join(repr(e) for e in st['examples'][:2])}) count as missing and "
                f"those rows cannot be used for training")
        else:
            notes.setdefault(st["column"], []).append(
                f"{st['count']} value(s) written as text (e.g. {st['examples'][0]!r}) are read as numbers")
    out = {}
    with type_overrides(type_map(overrides)):
        for c in df.columns:
            s = suggest_problem_type(df[c], str(c))
            if notes.get(str(c)):
                s["warning"] = "; ".join(notes[str(c)])
            missing = float(df[c].isna().mean())
            if s["suitable"] and missing > 0.3:
                s["suitable"] = False
                s["reason"] = f"{round(missing * 100)}% of its values are missing"
            out[str(c)] = s
    return out


@router.post("/{dataset_id}/objective", response_model=ObjectiveResponse, status_code=status.HTTP_200_OK)
def set_objective(
    payload: ObjectiveCreate,
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """Sets or updates the ML objective. Changing it invalidates all downstream results."""
    prob_type = payload.problem_type.lower().strip().replace(" ", "_")
    if prob_type not in ALL_TASKS:
        raise HTTPException(status_code=400, detail=f"Problem type must be one of: {', '.join(ALL_TASKS)}.")

    df, _, _ = load_prepared(db, db_dataset)
    target = (payload.target_column or "").strip()
    if needs_target(prob_type):
        validate_target(df, target, prob_type)
    else:
        # Unsupervised: no target. Every column is a feature, so the data needs rows and at least two columns.
        target = ""
        if len(df) < 20:
            raise HTTPException(status_code=400, detail=f"{LABELS[prob_type].capitalize()} needs at least 20 rows; this dataset has {len(df)}.")
        if df.shape[1] < 2:
            raise HTTPException(status_code=400, detail=f"{LABELS[prob_type].capitalize()} needs at least two columns.")

    dataset_id = db_dataset.id
    existing = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    changed = not existing or existing.problem_type != prob_type or existing.target_column != target

    if changed:
        db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).delete()
        db.query(IssueDetection).filter(IssueDetection.dataset_id == dataset_id).delete()
        db.query(ProfileReport).filter(ProfileReport.dataset_id == dataset_id).delete()
        db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "before").delete()
        invalidate_execution(db, db_dataset)
        get_snapshot(db, dataset_id).diagnostics = None

    if existing:
        existing.problem_type = prob_type
        existing.target_column = target
        obj = existing
    else:
        obj = Objective(dataset_id=dataset_id, problem_type=prob_type, target_column=target)
        db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.get("/{dataset_id}/objective", response_model=ObjectiveResponse)
def get_objective(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    obj = db.query(Objective).filter(Objective.dataset_id == db_dataset.id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Objective not set for this dataset.")
    return obj
