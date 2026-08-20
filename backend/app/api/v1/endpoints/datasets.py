import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.config import settings
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetResponse, DatasetSampleResponse

router = APIRouter()


# Demo Dataset Generators
DEMO_DATASETS = {
    "titanic": {
        "filename": "titanic_passenger_survival.csv",
        "default_target": "Survived",
        "default_problem_type": "classification",
        "generator": lambda: pd.DataFrame({
            "PassengerId": range(1, 101),
            "Survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1] * 10,
            "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2] * 10,
            "Name": [f"Passenger {i}" for i in range(1, 101)],
            "Sex": ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female"] * 10,
            "Age": [22.0, 38.0, 26.0, 35.0, -5.0, None, 54.0, 2.0, 27.0, 14.0] * 10,  # includes negative & missing
            "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1] * 10,
            "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 2, 0] * 10,
            "Fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.45, 512.3, 21.07, 11.13, 30.07] * 10,  # outlier 512.3
            "Cabin": [None, "C85", None, "C123", None, None, "E46", None, None, None] * 10,  # high missing
            "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C"] * 10,
        })
    },
    "churn": {
        "filename": "telco_customer_churn.csv",
        "default_target": "Churn",
        "default_problem_type": "classification",
        "generator": lambda: pd.DataFrame({
            "CustomerID": [f"CUST-{1000+i}" for i in range(100)],
            "Gender": ["Male", "Female"] * 50,
            "SeniorCitizen": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0] * 10,
            "Tenure": [1, 34, 2, 45, 8, 22, 10, 28, 62, 13] * 10,
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 29.75, 104.80, 56.15] * 10,
            "TotalCharges": [29.85, 1889.5, 108.15, 1840.75, 151.65, None, 1949.4, 346.45, 3046.05, None] * 10,  # missing
            "Contract": ["Month-to-month", "One year", "Month-to-month", "One year", "Month-to-month", "Two year", "Month-to-month", "Month-to-month", "Month-to-month", "Month-to-month"] * 10,
            "PaymentMethod": ["Electronic check", "Mailed check", "Mailed check", "Bank transfer", "Electronic check", "Electronic check", "Credit card", "Mailed check", "Electronic check", "Mailed check"] * 10,
            "Churn": [0, 0, 1, 0, 1, 1, 0, 0, 0, 1] * 10,  # class ratio
        })
    },
    "housing": {
        "filename": "california_housing_prices.csv",
        "default_target": "MedianHouseValue",
        "default_problem_type": "regression",
        "generator": lambda: pd.DataFrame({
            "MedInc": [8.3252, 8.3014, 7.2574, 5.6431, 3.8462, 4.0368, 3.6591, 3.1200, 2.0804, 3.6912] * 10,
            "HouseAge": [41.0, 21.0, 52.0, 52.0, 52.0, 52.0, 52.0, 52.0, 42.0, 52.0] * 10,
            "AveRooms": [6.98, 6.23, 8.28, 5.81, 6.28, 4.76, 4.93, 4.79, 4.29, 4.97] * 10,
            "AveBedrms": [1.02, 0.97, 1.07, 1.07, 1.08, 1.10, 0.95, 1.06, 1.11, 1.03] * 10,
            "Population": [322.0, 2401.0, 496.0, 558.0, 565.0, 413.0, 1094.0, 1157.0, 1206.0, 1551.0] * 10,
            "AveOccup": [2.55, 2.10, 2.80, 2.54, 2.18, 2.13, 3.64, 1.78, 2.02, 2.17] * 10,
            "Latitude": [37.88, 37.86, 37.85, 37.85, 37.85, 37.85, 37.84, 37.84, 37.84, 37.84] * 10,
            "Longitude": [-122.23, -122.22, -122.24, -122.25, -122.25, -122.25, -122.25, -122.25, -122.26, -122.26] * 10,
            "MedianHouseValue": [452600.0, 358500.0, 352100.0, 341300.0, 342200.0, 269700.0, 299200.0, 241400.0, 226700.0, 261100.0] * 10,
        })
    }
}


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Uploads a CSV or Excel file and parses initial schema."""
    filename = file.filename or "dataset.csv"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(status_code=400, detail="Only CSV and Excel (.xlsx, .xls) files are supported.")

    dataset_id = str(uuid.uuid4())
    raw_path = os.path.join(settings.RAW_DATA_DIR, f"{dataset_id}_{filename}")

    # Save uploaded file
    try:
        content = await file.read()
        file_size = len(content)
        with open(raw_path, "wb") as f:
            f.write(content)

        # Parse DataFrame
        if ext == ".csv":
            df = pd.read_csv(raw_path)
        else:
            df = pd.read_excel(raw_path)

    except Exception as e:
        if os.path.exists(raw_path):
            os.remove(raw_path)
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {str(e)}")

    row_cnt, col_cnt = df.shape

    # Create DB Record
    db_dataset = Dataset(
        id=dataset_id,
        filename=filename,
        file_size_bytes=file_size,
        raw_file_path=raw_path,
        row_count=row_cnt,
        col_count=col_cnt,
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)

    return DatasetResponse(
        id=db_dataset.id,
        filename=db_dataset.filename,
        file_size_bytes=db_dataset.file_size_bytes,
        row_count=db_dataset.row_count,
        col_count=db_dataset.col_count,
        uploaded_at=db_dataset.uploaded_at,
        has_objective=False,
        has_profile=False,
        has_health_score=False,
        has_cleaned_data=False,
    )


@router.post("/demo/{demo_key}", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def load_demo_dataset(
    demo_key: str,
    db: Session = Depends(get_db),
):
    """Instantly loads a standard curated sample dataset (titanic, churn, housing)."""
    demo_key = demo_key.lower()
    if demo_key not in DEMO_DATASETS:
        raise HTTPException(status_code=404, detail=f"Demo dataset '{demo_key}' not found. Available: {list(DEMO_DATASETS.keys())}")

    config = DEMO_DATASETS[demo_key]
    df = config["generator"]()
    dataset_id = str(uuid.uuid4())
    filename = config["filename"]
    raw_path = os.path.join(settings.RAW_DATA_DIR, f"{dataset_id}_{filename}")
    
    df.to_csv(raw_path, index=False)
    file_size = os.path.getsize(raw_path)

    db_dataset = Dataset(
        id=dataset_id,
        filename=filename,
        file_size_bytes=file_size,
        raw_file_path=raw_path,
        row_count=len(df),
        col_count=len(df.columns),
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)

    return DatasetResponse(
        id=db_dataset.id,
        filename=db_dataset.filename,
        file_size_bytes=db_dataset.file_size_bytes,
        row_count=db_dataset.row_count,
        col_count=db_dataset.col_count,
        uploaded_at=db_dataset.uploaded_at,
        has_objective=False,
        has_profile=False,
        has_health_score=False,
        has_cleaned_data=False,
    )


@router.get("/{dataset_id}/sample", response_model=DatasetSampleResponse)
async def get_dataset_sample(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """Retrieves preview sample rows from the raw dataset."""
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not os.path.exists(db_dataset.raw_file_path):
        raise HTTPException(status_code=404, detail="Raw dataset file not found on disk.")

    ext = os.path.splitext(db_dataset.raw_file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(db_dataset.raw_file_path, nrows=15)
    else:
        df = pd.read_excel(db_dataset.raw_file_path, nrows=15)

    # Sanitize NaN values for clean JSON serialization
    sample_records = df.where(pd.notnull(df), None).to_dict(orient="records")

    return DatasetSampleResponse(
        id=db_dataset.id,
        filename=db_dataset.filename,
        total_rows=db_dataset.row_count,
        total_cols=db_dataset.col_count,
        columns=list(df.columns),
        sample_data=sample_records,
    )
