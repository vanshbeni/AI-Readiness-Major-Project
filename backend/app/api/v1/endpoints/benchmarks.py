import os
import uuid
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.dataset import Dataset, Objective, ModelBenchmark
from app.schemas.dataset import ModelBenchmarkLeaderboardResponse, ModelBenchmarkItem
from app.engine.benchmark import benchmark_models

router = APIRouter()


@router.post("/{dataset_id}/benchmark", response_model=ModelBenchmarkLeaderboardResponse, status_code=status.HTTP_200_OK)
async def run_model_benchmarks(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """
    Fits and cross-validates candidate ML models on the cleaned dataset.
    Returns ranked benchmark leaderboard.
    """
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    if not obj:
        raise HTTPException(status_code=400, detail="Objective must be set before benchmarking.")

    # Read cleaned data (or fallback to raw if not cleaned yet)
    data_path = db_dataset.cleaned_file_path if db_dataset.cleaned_file_path and os.path.exists(db_dataset.cleaned_file_path) else db_dataset.raw_file_path
    if not data_path or not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data file not found.")

    ext = os.path.splitext(data_path)[1].lower()
    df = pd.read_csv(data_path) if ext == ".csv" else pd.read_excel(data_path)

    try:
        results, primary_metric, summary_text = benchmark_models(
            df_cleaned=df,
            target_col=obj.target_column,
            problem_type=obj.problem_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model benchmarking failed: {str(e)}")

    # Clear previous benchmarks
    db.query(ModelBenchmark).filter(ModelBenchmark.dataset_id == dataset_id).delete()

    benchmark_items = []
    for r in results:
        bench_id = str(uuid.uuid4())
        db_bench = ModelBenchmark(
            id=bench_id,
            dataset_id=dataset_id,
            model_name=r["model_name"],
            problem_type=r["problem_type"],
            metric_name=r["metric_name"],
            metric_value=r["metric_value"],
            training_time_sec=r["training_time_sec"],
            rank=r["rank"],
            is_recommended=r["is_recommended"],
            details={"description": r["description"], "suitability": r["suitability"]},
        )
        db.add(db_bench)
        benchmark_items.append(ModelBenchmarkItem(
            id=bench_id,
            model_name=r["model_name"],
            problem_type=r["problem_type"],
            metric_name=r["metric_name"],
            metric_value=r["metric_value"],
            metric_display=r["metric_display"],
            training_time_sec=r["training_time_sec"],
            rank=r["rank"],
            is_recommended=r["is_recommended"],
            description=r["description"],
            suitability=r["suitability"],
        ))

    db.commit()

    return ModelBenchmarkLeaderboardResponse(
        dataset_id=dataset_id,
        problem_type=obj.problem_type,
        primary_metric=primary_metric,
        models=benchmark_items,
        best_model_summary=summary_text,
    )
