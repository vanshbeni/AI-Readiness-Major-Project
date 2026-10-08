import os
import uuid
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_client_id, get_owned_dataset
from app.core.config import settings
from app.core.database import get_db
from app.core.storage import (
    SUPPORTED_EXTENSIONS,
    DatasetParseError,
    json_safe,
    parse_uploaded_file,
    read_dataset,
    remove_files,
    sanitize_filename,
)
from app.models.dataset import Dataset, DatasetSnapshot, HealthScore, Objective, ProcessingJob, ProfileReport, Recommendation
from app.schemas.dataset import (
    DatasetResponse,
    DatasetSampleResponse,
    DatasetSessionResponse,
    ExecutionResponse,
    FullDiagnosticResponse,
    ModelBenchmarkLeaderboardResponse,
    ObjectiveResponse,
    RecommendationItem,
)

router = APIRouter()


def _titanic(rng: np.random.Generator) -> pd.DataFrame:
    n = 300
    pclass = rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], n, p=[0.64, 0.36])
    age = np.round(rng.normal(29, 14, n).clip(0.5, 80), 1)
    age[rng.choice(n, 4, replace=False)] = -rng.integers(1, 10, 4)          # invalid negatives
    age[rng.choice(n, 55, replace=False)] = np.nan                          # ~18% missing
    fare = np.round(rng.exponential(14, n) + np.where(pclass == 1, 60, np.where(pclass == 2, 15, 5)), 2)
    fare[rng.choice(n, 6, replace=False)] = rng.uniform(250, 520, 6)        # outliers
    survive_p = 0.15 + 0.5 * (sex == "female") + 0.15 * (pclass == 1) - 0.05 * (pclass == 3)
    cabin = np.where(rng.random(n) < 0.25, [f"C{rng.integers(1, 150)}" for _ in range(n)], None)
    embarked = rng.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]).astype(object)
    embarked[rng.choice(n, 3, replace=False)] = None
    df = pd.DataFrame({
        "PassengerId": np.arange(1, n + 1),
        "Survived": (rng.random(n) < survive_p.clip(0.02, 0.95)).astype(int),
        "Pclass": pclass,
        "Name": [f"Passenger {i}" for i in range(1, n + 1)],
        "Sex": sex,
        "Age": age,
        "SibSp": rng.choice([0, 1, 2, 3, 4], n, p=[0.68, 0.23, 0.04, 0.03, 0.02]),
        "Parch": rng.choice([0, 1, 2], n, p=[0.76, 0.14, 0.10]),
        "Fare": fare,
        "Cabin": cabin,
        "Embarked": embarked,
    })
    return pd.concat([df, df.iloc[:4].assign(PassengerId=np.arange(n + 1, n + 5))], ignore_index=True)


def _churn(rng: np.random.Generator) -> pd.DataFrame:
    n = 300
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18, 118, n), 2)
    total = np.round(tenure * monthly * rng.uniform(0.95, 1.05, n), 2)
    total[rng.choice(n, 12, replace=False)] = np.nan
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    score = -1.2 + 1.3 * (contract == "Month-to-month") - 0.035 * tenure + 0.01 * (monthly - 65)
    churn_p = 1 / (1 + np.exp(-score))
    churn = (rng.random(n) < churn_p * 0.6).astype(int)  # roughly 80:20
    return pd.DataFrame({
        "CustomerID": [f"CUST-{1000 + i}" for i in range(n)],
        "Gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "Churn": churn,
    })


def _housing(rng: np.random.Generator) -> pd.DataFrame:
    n = 300
    med_inc = np.round(rng.lognormal(1.3, 0.45, n), 4)
    rooms = np.round(rng.normal(5.4, 1.1, n).clip(2, 12), 2)
    rooms[rng.choice(n, 6, replace=False)] = rng.uniform(20, 40, 6)          # outliers
    bedrooms = np.round(rooms * rng.normal(0.2, 0.01, n), 2)                 # collinear with rooms
    bedrooms[rng.choice(n, 15, replace=False)] = np.nan
    lat = np.round(rng.uniform(32.5, 41.9, n), 2)
    value = np.round((med_inc * 42000 + rng.normal(0, 30000, n) + (lat < 36) * 25000).clip(15000, 500000), 0)
    df = pd.DataFrame({
        "MedInc": med_inc,
        "HouseAge": rng.integers(1, 53, n).astype(float),
        "AveRooms": rooms,
        "AveBedrms": bedrooms,
        "Population": rng.integers(100, 5000, n).astype(float),
        "AveOccup": np.round(rng.normal(2.9, 0.7, n).clip(1, 8), 2),
        "Latitude": lat,
        "Longitude": np.round(rng.uniform(-124.3, -114.3, n), 2),
        "MedianHouseValue": value,
    })
    return pd.concat([df, df.iloc[:5]], ignore_index=True)


DEMO_DATASETS = {
    "titanic": {"filename": "titanic_passenger_survival.csv", "generator": lambda: _titanic(np.random.default_rng(7))},
    "churn": {"filename": "telco_customer_churn.csv", "generator": lambda: _churn(np.random.default_rng(11))},
    "housing": {"filename": "california_housing_prices.csv", "generator": lambda: _housing(np.random.default_rng(23))},
}


def build_dataset_response(db: Session, ds: Dataset, warnings: List[str] = None) -> DatasetResponse:
    return DatasetResponse(
        id=ds.id,
        filename=ds.filename,
        file_size_bytes=ds.file_size_bytes,
        row_count=ds.row_count,
        col_count=ds.col_count,
        uploaded_at=ds.uploaded_at,
        has_objective=db.query(Objective).filter(Objective.dataset_id == ds.id).count() > 0,
        has_profile=db.query(ProfileReport).filter(ProfileReport.dataset_id == ds.id).count() > 0,
        has_health_score=db.query(HealthScore).filter(HealthScore.dataset_id == ds.id, HealthScore.stage == "before").count() > 0,
        has_cleaned_data=bool(ds.cleaned_file_path and os.path.exists(ds.cleaned_file_path)),
        warnings=warnings or [],
    )


def build_sample(ds: Dataset) -> DatasetSampleResponse:
    df = read_dataset(ds.raw_file_path, nrows=15)
    records = json_safe(df.astype(object).where(pd.notnull(df), None).to_dict(orient="records"))
    return DatasetSampleResponse(
        id=ds.id,
        filename=ds.filename,
        total_rows=ds.row_count,
        total_cols=ds.col_count,
        columns=[str(c) for c in df.columns],
        sample_data=records,
    )


def _store_dataset(db: Session, df: pd.DataFrame, filename: str, file_size: int, client_id: str, warnings: List[str]) -> DatasetResponse:
    dataset_id = str(uuid.uuid4())
    raw_path = os.path.join(settings.RAW_DATA_DIR, f"{dataset_id}.csv")
    df.to_csv(raw_path, index=False)
    try:
        db_dataset = Dataset(
            id=dataset_id,
            owner_token=client_id,
            filename=filename,
            file_size_bytes=file_size,
            raw_file_path=raw_path,
            row_count=int(df.shape[0]),
            col_count=int(df.shape[1]),
        )
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
    except Exception:
        db.rollback()
        remove_files(raw_path)
        raise HTTPException(status_code=500, detail="Failed to save the dataset record. Please try again.")
    return build_dataset_response(db, db_dataset, warnings)


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def upload_dataset(
    file: UploadFile = File(...),
    client_id: str = Depends(get_client_id),
    db: Session = Depends(get_db),
):
    """Uploads a CSV or Excel file, validates it and stores a normalized copy."""
    filename = sanitize_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only CSV and Excel (.xlsx, .xls) files are supported.")

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    content = file.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File is larger than the {settings.MAX_UPLOAD_MB} MB limit.")

    try:
        df, warnings = parse_uploaded_file(content, ext)
    except DatasetParseError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _store_dataset(db, df, filename, len(content), client_id, warnings)


@router.post("/demo/{demo_key}", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def load_demo_dataset(
    demo_key: str,
    client_id: str = Depends(get_client_id),
    db: Session = Depends(get_db),
):
    """Loads a curated sample dataset (titanic, churn, housing)."""
    demo_key = demo_key.lower()
    if demo_key not in DEMO_DATASETS:
        raise HTTPException(status_code=404, detail=f"Demo dataset '{demo_key}' not found. Available: {list(DEMO_DATASETS.keys())}")
    config = DEMO_DATASETS[demo_key]
    df = config["generator"]()
    size = len(df.to_csv(index=False).encode("utf-8"))
    return _store_dataset(db, df, config["filename"], size, client_id, [])


@router.get("/{dataset_id}/sample", response_model=DatasetSampleResponse)
def get_dataset_sample(db_dataset: Dataset = Depends(get_owned_dataset)):
    """Retrieves preview sample rows from the raw dataset."""
    return build_sample(db_dataset)


@router.get("/{dataset_id}/session", response_model=DatasetSessionResponse)
def get_dataset_session(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """Returns everything needed to restore the UI for a dataset after a page refresh."""
    obj = db.query(Objective).filter(Objective.dataset_id == db_dataset.id).first()
    snap = db.query(DatasetSnapshot).filter(DatasetSnapshot.dataset_id == db_dataset.id).first()

    diagnostics = execution = leaderboard = None
    if snap and snap.diagnostics:
        diagnostics = FullDiagnosticResponse.model_validate(snap.diagnostics)
        recs = db.query(Recommendation).filter(Recommendation.dataset_id == db_dataset.id).all()
        diagnostics.recommendations = [RecommendationItem.model_validate(r) for r in recs]
        diagnostics.dataset = build_dataset_response(db, db_dataset)
    if snap and snap.execution:
        execution = ExecutionResponse.model_validate(snap.execution)
    if snap and snap.leaderboard:
        leaderboard = ModelBenchmarkLeaderboardResponse.model_validate(snap.leaderboard)

    return DatasetSessionResponse(
        dataset=build_dataset_response(db, db_dataset),
        sample=build_sample(db_dataset),
        objective=ObjectiveResponse.model_validate(obj) if obj else None,
        diagnostics=diagnostics,
        execution=execution,
        leaderboard=leaderboard,
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: str,
    client_id: str = Depends(get_client_id),
    db: Session = Depends(get_db),
):
    """Deletes a dataset, all derived records and all stored files."""
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset or (db_dataset.owner_token and db_dataset.owner_token != client_id):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    jobs = db.query(ProcessingJob).filter(ProcessingJob.dataset_id == dataset_id).all()
    files = [db_dataset.raw_file_path, db_dataset.cleaned_file_path]
    files += [p for j in jobs for p in (j.script_file_path, j.report_file_path)]
    db.delete(db_dataset)
    db.commit()
    remove_files(*files)
    remove_files(os.path.join(settings.ARTIFACTS_DIR, f"{dataset_id}_audit_report.pdf"))
