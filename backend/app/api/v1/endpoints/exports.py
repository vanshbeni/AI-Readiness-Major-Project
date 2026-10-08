import json
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset
from app.core.config import settings
from app.core.database import get_db
from app.core.storage import json_safe
from app.engine.reporter import generate_pdf_report
from app.engine.scorer import grade_for_score
from app.models.dataset import Dataset, HealthScore, ModelBenchmark, ProcessingJob, ProfileReport, Recommendation

router = APIRouter()


def _base_name(ds: Dataset) -> str:
    return os.path.splitext(ds.filename)[0] or "dataset"


def _latest_job(db: Session, dataset_id: str):
    return (
        db.query(ProcessingJob)
        .filter(ProcessingJob.dataset_id == dataset_id, ProcessingJob.status == "completed")
        .order_by(ProcessingJob.completed_at.desc())
        .first()
    )


@router.get("/{dataset_id}/export/cleaned-csv")
def download_cleaned_csv(db_dataset: Dataset = Depends(get_owned_dataset)):
    if not db_dataset.cleaned_file_path or not os.path.exists(db_dataset.cleaned_file_path):
        raise HTTPException(status_code=404, detail="Cleaned dataset not found. Execute the pipeline first.")
    return FileResponse(path=db_dataset.cleaned_file_path, filename=f"cleaned_{_base_name(db_dataset)}.csv", media_type="text/csv")


@router.get("/{dataset_id}/export/pipeline-script")
def download_pipeline_script(db_dataset: Dataset = Depends(get_owned_dataset), db: Session = Depends(get_db)):
    job = _latest_job(db, db_dataset.id)
    if not job or not job.script_file_path or not os.path.exists(job.script_file_path):
        raise HTTPException(status_code=404, detail="Pipeline script not found. Execute the pipeline first.")
    return FileResponse(path=job.script_file_path, filename=f"pipeline_{_base_name(db_dataset)}.py", media_type="text/x-python")


@router.get("/{dataset_id}/export/pdf-report")
def download_pdf_report(db_dataset: Dataset = Depends(get_owned_dataset), db: Session = Depends(get_db)):
    dataset_id = db_dataset.id
    before = db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "before").first()
    if not before:
        raise HTTPException(status_code=400, detail="Run diagnostics before generating the report.")
    after = db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "after").first()
    prof = db.query(ProfileReport).filter(ProfileReport.dataset_id == dataset_id).first()
    job = _latest_job(db, dataset_id)
    benchmarks = db.query(ModelBenchmark).filter(ModelBenchmark.dataset_id == dataset_id).order_by(ModelBenchmark.rank).all()

    before_dict = {"composite_score": before.composite_score, "sub_scores": before.sub_scores, "grade": grade_for_score(before.composite_score)}
    after_dict = (
        {"composite_score": after.composite_score, "sub_scores": after.sub_scores, "grade": grade_for_score(after.composite_score)}
        if after else None
    )
    bench_dicts = [
        {
            "rank": b.rank,
            "model_name": b.model_name,
            "metric_name": b.metric_name,
            "metric_display": (b.details or {}).get("metric_display", f"{b.metric_value}"),
            "training_time_sec": b.training_time_sec,
            "suitability": (b.details or {}).get("suitability", "-"),
        }
        for b in benchmarks
    ]

    pdf_path = os.path.join(settings.ARTIFACTS_DIR, f"{dataset_id}_audit_report.pdf")
    try:
        generate_pdf_report(
            dataset_name=db_dataset.filename,
            summary_stats=prof.summary_stats if prof else {},
            before_health=before_dict,
            after_health=after_dict,
            transformations=(job.applied_steps or []) if job else [],
            benchmarks=bench_dicts,
            output_pdf_path=pdf_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF report generation failed: {e}")

    return FileResponse(path=pdf_path, filename=f"Data_Readiness_Report_{_base_name(db_dataset)}.pdf", media_type="application/pdf")


@router.get("/{dataset_id}/export/manifest-json")
def download_manifest_json(db_dataset: Dataset = Depends(get_owned_dataset), db: Session = Depends(get_db)):
    dataset_id = db_dataset.id
    before = db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "before").first()
    after = db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "after").first()
    recs = db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).all()
    benchmarks = db.query(ModelBenchmark).filter(ModelBenchmark.dataset_id == dataset_id).order_by(ModelBenchmark.rank).all()
    job = _latest_job(db, dataset_id)

    manifest = {
        "dataset": {"id": db_dataset.id, "filename": db_dataset.filename, "rows": db_dataset.row_count, "cols": db_dataset.col_count},
        "health_score": {
            "before": before.composite_score if before else None,
            "after": after.composite_score if after else None,
            "grade_before": grade_for_score(before.composite_score) if before else None,
            "grade_after": grade_for_score(after.composite_score) if after else None,
            "sub_scores_before": before.sub_scores if before else None,
            "sub_scores_after": after.sub_scores if after else None,
        },
        "recommendations": [
            {
                "issue_type": r.issue_type,
                "column": r.column,
                "method": r.method,
                "parameters": r.params,
                "explanation": r.explanation_text,
                "explanation_source": r.explanation_source,
                "is_approved": r.is_approved,
            }
            for r in recs
        ],
        "applied_steps": (job.applied_steps or []) if job else [],
        "model_benchmarks": [
            {"rank": b.rank, "model": b.model_name, "metric_name": b.metric_name,
             "metric_value": b.metric_value, "training_time_sec": b.training_time_sec}
            for b in benchmarks
        ],
    }
    return Response(
        content=json.dumps(json_safe(manifest), indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="manifest_{_base_name(db_dataset)}.json"'},
    )
