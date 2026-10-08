import os
import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset, get_snapshot, invalidate_execution
from app.core.config import settings
from app.core.database import get_db
from app.core.storage import dataset_lock, json_safe, read_dataset, remove_files
from app.engine.detector import detect_issues
from app.engine.executor import execute_pipeline
from app.engine.profiler import count_duplicate_records, identifier_columns, profile_dataset
from app.engine.scorer import calculate_health_score
from app.models.dataset import Dataset, HealthScore, Objective, ProcessingJob, Recommendation
from app.schemas.dataset import ExecutionResponse, MetricDelta

logger = logging.getLogger(__name__)
router = APIRouter()


def _signed(value, suffix: str) -> str:
    return f"+{value}{suffix}" if value > 0 else f"{value}{suffix}"


@router.post("/{dataset_id}/execute", response_model=ExecutionResponse, status_code=status.HTTP_200_OK)
def run_pipeline_execution(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """
    Executes the approved cleaning steps, stores the cleaned dataset and pipeline script,
    re-scores the cleaned data and reports before/after deltas.
    """
    dataset_id = db_dataset.id
    obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    before_health = db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "before").first()
    if not obj or not before_health:
        raise HTTPException(status_code=400, detail="Run diagnostics before executing the cleaning pipeline.")

    with dataset_lock(dataset_id):
        target_col, prob_type = obj.target_column, obj.problem_type
        db_recs = db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).all()
        rec_dicts = [
            {"id": r.id, "issue_type": r.issue_type, "column": r.column, "method": r.method,
             "params": r.params or {}, "is_approved": bool(r.is_approved)}
            for r in db_recs
        ]
        before_score = before_health.composite_score

        df_raw = read_dataset(db_dataset.raw_file_path)
        raw_missing = int(df_raw.isna().sum().sum())
        raw_dups = count_duplicate_records(df_raw)

        invalidate_execution(db, db_dataset)
        job = ProcessingJob(id=str(uuid.uuid4()), dataset_id=dataset_id, status="running")
        db.add(job)
        db.commit()

        cleaned_path = os.path.join(settings.CLEANED_DATA_DIR, f"{dataset_id}_cleaned.csv")
        script_path = os.path.join(settings.ARTIFACTS_DIR, f"{dataset_id}_pipeline.py")
        try:
            result = execute_pipeline(df_raw=df_raw, recommendations=rec_dicts, target_col=target_col, problem_type=prob_type)
            df_cleaned, df_final = result.df_cleaned, result.df_model_ready

            # Re-score the cleaned (pre-encoding) data so the comparison is like-for-like with the raw data.
            summary_after, profiles_after = profile_dataset(df_cleaned, target_col=target_col)
            issues_after = detect_issues(df_cleaned, target_col=target_col, problem_type=prob_type)
            raw_ids = set(identifier_columns(df_raw))
            informative_features = [c for c in df_raw.columns if c != target_col and c not in raw_ids]
            dropped_ratio = len(result.dropped_informative_columns) / max(len(informative_features), 1)
            after_score, sub_scores_after, _, summary_text_after, _ = calculate_health_score(
                summary_after, profiles_after, issues_after, problem_type=prob_type,
                dropped_informative_ratio=dropped_ratio,
            )

            df_final.to_csv(cleaned_path, index=False)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(result.script_code)
        except Exception as e:
            logger.exception("Pipeline execution failed")
            remove_files(cleaned_path, script_path)
            job.status = "failed"
            job.error_message = str(e)[:2000]
            job.completed_at = datetime.utcnow()
            db.commit()
            raise HTTPException(status_code=500, detail=f"Pipeline transformation failed: {e}")

        db_dataset.cleaned_file_path = cleaned_path
        db.add(HealthScore(
            dataset_id=dataset_id, stage="after", composite_score=after_score,
            sub_scores=sub_scores_after, summary_text=summary_text_after,
        ))
        job.status = "completed"
        job.pipeline_file_path = cleaned_path
        job.script_file_path = script_path
        job.applied_steps = result.applied_steps
        job.completed_at = datetime.utcnow()

        cleaned_missing = int(df_final.isna().sum().sum())
        cleaned_dups = count_duplicate_records(df_cleaned)
        delta_score = round(after_score - before_score, 1)

        if delta_score > 0:
            message = f"Pipeline executed successfully. Health Score improved by {delta_score} points."
        elif delta_score == 0:
            message = "Pipeline executed successfully. Health Score is unchanged."
        else:
            message = f"Pipeline executed, but the Health Score dropped by {abs(delta_score)} points. Review the approved fixes."

        response = ExecutionResponse(
            job_id=job.id,
            status="completed",
            message=message,
            before_health_score=before_score,
            after_health_score=after_score,
            health_score_delta=delta_score,
            transformations_applied=result.applied_steps,
            metric_deltas=[
                MetricDelta(metric_name="Data Health Score", before_value=f"{before_score} / 100",
                            after_value=f"{after_score} / 100", improvement=_signed(delta_score, " pts")),
                MetricDelta(metric_name="Missing Values", before_value=f"{raw_missing} cells",
                            after_value=f"{cleaned_missing} cells", improvement=_signed(cleaned_missing - raw_missing, " cells")),
                MetricDelta(metric_name="Duplicate Rows", before_value=f"{raw_dups} rows",
                            after_value=f"{cleaned_dups} rows", improvement=_signed(cleaned_dups - raw_dups, " rows")),
                MetricDelta(metric_name="Dataset Dimensions",
                            before_value=f"{df_raw.shape[0]} rows × {df_raw.shape[1]} cols",
                            after_value=f"{df_final.shape[0]} rows × {df_final.shape[1]} cols",
                            improvement=f"{_signed(df_final.shape[0] - df_raw.shape[0], ' rows')}, {_signed(df_final.shape[1] - df_raw.shape[1], ' cols')}"),
            ],
            cleaned_rows=int(df_final.shape[0]),
            cleaned_cols=int(df_final.shape[1]),
        )
        get_snapshot(db, dataset_id).execution = json_safe(response.model_dump(mode="json"))
        db.commit()

    return response
