import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset, get_snapshot
from app.core.database import get_db
from app.core.storage import dataset_lock, json_safe, read_dataset
from app.engine.benchmark import BenchmarkError, benchmark_models
from app.models.dataset import Dataset, ModelBenchmark, Objective, Recommendation
from app.schemas.dataset import ModelBenchmarkItem, ModelBenchmarkLeaderboardResponse

router = APIRouter()


def _imbalance_strategy(db: Session, dataset_id: str):
    recs = db.query(Recommendation).filter(
        Recommendation.dataset_id == dataset_id,
        Recommendation.issue_type == "class_imbalance",
        Recommendation.is_approved == True,  # noqa: E712
    ).all()
    for r in recs:
        key = (r.params or {}).get("method_key")
        if key in ("class_weights", "smote_class_weights"):
            return key
    return None


@router.post("/{dataset_id}/benchmark", response_model=ModelBenchmarkLeaderboardResponse, status_code=status.HTTP_200_OK)
def run_model_benchmarks(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """Cross-validates candidate models on the cleaned dataset and returns a ranked leaderboard."""
    dataset_id = db_dataset.id
    obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    if not obj:
        raise HTTPException(status_code=400, detail="Objective must be set before benchmarking.")
    if not db_dataset.cleaned_file_path or not os.path.exists(db_dataset.cleaned_file_path):
        raise HTTPException(status_code=400, detail="Execute the cleaning pipeline before benchmarking models.")

    with dataset_lock(dataset_id):
        df = read_dataset(db_dataset.cleaned_file_path)
        try:
            results, primary_metric, summary_text, cv_folds = benchmark_models(
                df_cleaned=df,
                target_col=obj.target_column,
                problem_type=obj.problem_type,
                imbalance_strategy=_imbalance_strategy(db, dataset_id),
            )
        except BenchmarkError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model benchmarking failed: {e}")

        db.query(ModelBenchmark).filter(ModelBenchmark.dataset_id == dataset_id).delete()
        items = []
        for r in results:
            bench_id = str(uuid.uuid4())
            db.add(ModelBenchmark(
                id=bench_id,
                dataset_id=dataset_id,
                model_name=r["model_name"],
                problem_type=r["problem_type"],
                metric_name=r["metric_name"],
                metric_value=r["metric_value"],
                training_time_sec=r["training_time_sec"],
                rank=r["rank"],
                is_recommended=r["is_recommended"],
                details={"description": r["description"], "suitability": r["suitability"], "metric_display": r["metric_display"]},
            ))
            items.append(ModelBenchmarkItem(id=bench_id, **{k: r[k] for k in (
                "model_name", "problem_type", "metric_name", "metric_value", "metric_display",
                "training_time_sec", "rank", "is_recommended", "description", "suitability",
            )}))

        response = ModelBenchmarkLeaderboardResponse(
            dataset_id=dataset_id,
            problem_type=obj.problem_type,
            primary_metric=primary_metric,
            cv_folds=cv_folds,
            models=items,
            best_model_summary=summary_text,
        )
        get_snapshot(db, dataset_id).leaderboard = json_safe(response.model_dump(mode="json"))
        db.commit()

    return response
