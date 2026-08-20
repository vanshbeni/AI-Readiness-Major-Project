import os
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.config import settings
from app.models.dataset import Dataset, ProfileReport, HealthScore, ProcessingJob, ModelBenchmark, Recommendation
from app.engine.reporter import generate_pdf_report

router = APIRouter()


@router.get("/{dataset_id}/export/cleaned-csv")
async def download_cleaned_csv(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """Downloads the cleaned, ML-ready CSV dataset."""
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset or not db_dataset.cleaned_file_path:
        raise HTTPException(status_code=404, detail="Cleaned dataset not found. Execute the pipeline first.")

    if not os.path.exists(db_dataset.cleaned_file_path):
        raise HTTPException(status_code=404, detail="Cleaned CSV file missing on server.")

    clean_name = f"cleaned_{db_dataset.filename}"
    return FileResponse(
        path=db_dataset.cleaned_file_path,
        filename=clean_name,
        media_type="text/csv",
    )


@router.get("/{dataset_id}/export/pipeline-script")
async def download_pipeline_script(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """Downloads the standalone Python preprocessing pipeline script."""
    job = db.query(ProcessingJob).filter(ProcessingJob.dataset_id == dataset_id, ProcessingJob.status == "completed").first()
    if not job or not job.script_file_path or not os.path.exists(job.script_file_path):
        raise HTTPException(status_code=404, detail="Pipeline script not found. Execute the pipeline first.")

    return FileResponse(
        path=job.script_file_path,
        filename=f"pipeline_{dataset_id}.py",
        media_type="text/x-python",
    )


@router.get("/{dataset_id}/export/pdf-report")
async def download_pdf_report(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """Compiles and downloads the executive PDF Data Quality & Readiness Audit Report."""
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    prof = db.query(ProfileReport).filter(ProfileReport.dataset_id == dataset_id).first()
    summary_stats = prof.summary_stats if prof else {}

    before_health = db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "before").first()
    after_health = db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "after").first()

    before_dict = {
        "composite_score": before_health.composite_score if before_health else 50.0,
        "sub_scores": before_health.sub_scores if before_health else {},
        "grade": "Fair",
    }
    after_dict = {
        "composite_score": after_health.composite_score if after_health else before_dict["composite_score"],
        "sub_scores": after_health.sub_scores if after_health else before_dict["sub_scores"],
        "grade": "Good",
    }

    # Fetch applied transformations from recommendations
    recs = db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id, Recommendation.is_approved == True).all()
    transforms = [f"{r.method} ({r.reason_title})" for r in recs]

    benchmarks = db.query(ModelBenchmark).filter(ModelBenchmark.dataset_id == dataset_id).order_by(ModelBenchmark.rank).all()
    bench_dicts = [
        {
            "rank": b.rank,
            "model_name": b.model_name,
            "metric_name": b.metric_name,
            "metric_display": f"{b.metric_value}",
            "training_time_sec": b.training_time_sec,
            "suitability": b.details.get("suitability", "High") if b.details else "High",
        }
        for b in benchmarks
    ]

    pdf_filename = f"{dataset_id}_audit_report.pdf"
    pdf_path = os.path.join(settings.ARTIFACTS_DIR, pdf_filename)

    try:
        generate_pdf_report(
            dataset_name=db_dataset.filename,
            summary_stats=summary_stats,
            before_health=before_dict,
            after_health=after_dict,
            transformations=transforms,
            benchmarks=bench_dicts,
            output_pdf_path=pdf_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF report generation failed: {str(e)}")

    return FileResponse(
        path=pdf_path,
        filename=f"Data_Readiness_Report_{db_dataset.filename.replace('.csv', '')}.pdf",
        media_type="application/pdf",
    )


@router.get("/{dataset_id}/export/manifest-json")
async def download_manifest_json(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """Downloads the full machine-readable diagnostic and recommendation manifest as JSON."""
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    prof = db.query(ProfileReport).filter(ProfileReport.dataset_id == dataset_id).first()
    before_health = db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "before").first()
    after_health = db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "after").first()
    recs = db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).all()
    benchmarks = db.query(ModelBenchmark).filter(ModelBenchmark.dataset_id == dataset_id).order_by(ModelBenchmark.rank).all()

    manifest = {
        "dataset": {
            "id": db_dataset.id,
            "filename": db_dataset.filename,
            "rows": db_dataset.row_count,
            "cols": db_dataset.col_count,
        },
        "health_score": {
            "before": before_health.composite_score if before_health else None,
            "after": after_health.composite_score if after_health else None,
            "sub_scores_before": before_health.sub_scores if before_health else None,
            "sub_scores_after": after_health.sub_scores if after_health else None,
        },
        "recommendations": [
            {
                "issue_type": r.issue_type,
                "column": r.column,
                "method": r.method,
                "explanation": r.explanation_text,
                "is_approved": r.is_approved,
            }
            for r in recs
        ],
        "model_benchmarks": [
            {
                "rank": b.rank,
                "model": b.model_name,
                "metric_name": b.metric_name,
                "metric_value": b.metric_value,
                "training_time_sec": b.training_time_sec,
            }
            for b in benchmarks
        ]
    }

    return Response(
        content=json.dumps(manifest, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=manifest_{dataset_id}.json"}
    )
