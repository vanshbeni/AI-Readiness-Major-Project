import os
import uuid
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.config import settings
from app.models.dataset import Dataset, Objective, Recommendation, HealthScore, ProcessingJob
from app.schemas.dataset import ExecutionResponse, MetricDelta
from app.engine.executor import execute_pipeline
from app.engine.profiler import profile_dataset
from app.engine.detector import detect_issues
from app.engine.scorer import calculate_health_score

router = APIRouter()


@router.post("/{dataset_id}/execute", response_model=ExecutionResponse, status_code=status.HTTP_200_OK)
async def run_pipeline_execution(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """
    Executes approved data cleaning transformations, generates cleaned dataset,
    re-computes Health Score (After stage), and quantifies quality improvement.
    """
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not os.path.exists(db_dataset.raw_file_path):
        raise HTTPException(status_code=404, detail="Raw data file missing.")

    obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    target_col = obj.target_column if obj else None
    prob_type = obj.problem_type if obj else "classification"

    # Get approved recommendations
    db_recs = db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).all()
    rec_dicts = [
        {
            "id": r.id,
            "issue_type": r.issue_type,
            "column": r.column,
            "method": r.method,
            "severity": r.severity,
            "is_approved": r.is_approved,
        }
        for r in db_recs
    ]

    # Get before health score
    before_health = db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "before").first()
    before_score = before_health.composite_score if before_health else 50.0

    # Read raw data
    ext = os.path.splitext(db_dataset.raw_file_path)[1].lower()
    df_raw = pd.read_csv(db_dataset.raw_file_path) if ext == ".csv" else pd.read_excel(db_dataset.raw_file_path)
    raw_missing = int(df_raw.isna().sum().sum())
    raw_dups = int(df_raw.duplicated().sum())

    # Execute Transformation Pipeline
    cleaned_filename = f"{dataset_id}_cleaned.csv"
    cleaned_path = os.path.join(settings.CLEANED_DATA_DIR, cleaned_filename)
    script_filename = f"{dataset_id}_pipeline.py.txt"
    script_path = os.path.join(settings.ARTIFACTS_DIR, script_filename)

    try:
        df_cleaned, applied_steps, script_code = execute_pipeline(
            df_raw=df_raw,
            recommendations=rec_dicts,
            target_col=target_col,
            problem_type=prob_type,
        )

        # Save cleaned data and python script
        df_cleaned.to_csv(cleaned_path, index=False)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_code)

        # Update dataset record
        db_dataset.cleaned_file_path = cleaned_path

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline transformation failed: {str(e)}")

    # ---------------- Re-Profile & Re-Score (After Stage) ----------------
    summary_stats_after, col_profiles_after = profile_dataset(df_cleaned, target_col=target_col)
    issues_after = detect_issues(df_cleaned, target_col=target_col, problem_type=prob_type)
    after_score, sub_scores_after, grade_after, summary_after, drivers_after = calculate_health_score(
        summary_stats_after, col_profiles_after, issues_after, problem_type=prob_type
    )

    # Save After Health Score
    db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "after").delete()
    after_health_record = HealthScore(
        dataset_id=dataset_id,
        stage="after",
        composite_score=after_score,
        sub_scores=sub_scores_after,
        summary_text=summary_after,
    )
    db.add(after_health_record)

    # Save Processing Job
    job_id = str(uuid.uuid4())
    job = ProcessingJob(
        id=job_id,
        dataset_id=dataset_id,
        status="completed",
        pipeline_file_path=cleaned_path,
        script_file_path=script_path,
        completed_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()

    # Metric Deltas
    cleaned_missing = int(df_cleaned.isna().sum().sum())
    cleaned_dups = int(df_cleaned.duplicated().sum())
    delta_score = round(after_score - before_score, 1)

    metric_deltas = [
        MetricDelta(
            metric_name="Data Health Score",
            before_value=f"{before_score} / 100",
            after_value=f"{after_score} / 100",
            improvement=f"+{delta_score} pts",
        ),
        MetricDelta(
            metric_name="Missing Values",
            before_value=f"{raw_missing} cells",
            after_value=f"{cleaned_missing} cells",
            improvement=f"-{raw_missing - cleaned_missing} missing cells",
        ),
        MetricDelta(
            metric_name="Duplicate Rows",
            before_value=f"{raw_dups} rows",
            after_value=f"{cleaned_dups} rows",
            improvement=f"-{raw_dups - cleaned_dups} duplicates",
        ),
        MetricDelta(
            metric_name="Dataset Dimensions",
            before_value=f"{df_raw.shape[0]} rows × {df_raw.shape[1]} cols",
            after_value=f"{df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} cols",
            improvement="Ready for ML",
        ),
    ]

    return ExecutionResponse(
        job_id=job_id,
        status="completed",
        message=f"Pipeline executed successfully. Health Score improved by +{delta_score} points.",
        before_health_score=before_score,
        after_health_score=after_score,
        health_score_delta=delta_score,
        transformations_applied=applied_steps,
        metric_deltas=metric_deltas,
        cleaned_rows=df_cleaned.shape[0],
        cleaned_cols=df_cleaned.shape[1],
    )
