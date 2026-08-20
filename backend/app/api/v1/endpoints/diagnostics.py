import os
import uuid
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.dataset import Dataset, Objective, ProfileReport, IssueDetection, HealthScore, Recommendation
from app.schemas.dataset import (
    FullDiagnosticResponse,
    DatasetResponse,
    ObjectiveResponse,
    ProfileReportResponse,
    HealthScoreResponse,
    IssueItem,
    RecommendationItem,
)
from app.engine.profiler import profile_dataset
from app.engine.detector import detect_issues
from app.engine.scorer import calculate_health_score
from app.engine.rules import select_remediation_methods
from app.engine.explainer import explain_recommendations_with_gemini

router = APIRouter()


@router.post("/{dataset_id}/diagnose", response_model=FullDiagnosticResponse, status_code=status.HTTP_200_OK)
async def run_diagnostics(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """
    Executes the full Pre-ML Data Quality Diagnostic Pipeline:
    Profiling -> Issue Detection -> Health Score -> Stage 2 Rules -> Stage 3 Gemini Explainer.
    """
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not os.path.exists(db_dataset.raw_file_path):
        raise HTTPException(status_code=404, detail="Raw data file missing on server.")

    # Load objective
    obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    target_col = obj.target_column if obj else None
    prob_type = obj.problem_type if obj else "classification"

    # Read dataset
    ext = os.path.splitext(db_dataset.raw_file_path)[1].lower()
    df = pd.read_csv(db_dataset.raw_file_path) if ext == ".csv" else pd.read_excel(db_dataset.raw_file_path)

    # 1. Profiling
    summary_stats, col_profiles = profile_dataset(df, target_col=target_col)

    # 2. Issue Detection
    detected_issues = detect_issues(df, target_col=target_col, problem_type=prob_type)

    # 3. Health Score (Before Stage)
    comp_score, sub_scores, grade, summary_text, drivers = calculate_health_score(
        summary_stats, col_profiles, detected_issues, problem_type=prob_type
    )

    # 4. Stage 2 Rule Matrix
    raw_recommendations = select_remediation_methods(
        detected_issues, col_profiles, problem_type=prob_type
    )

    # 5. Stage 3 Grounded LLM Explanation Layer
    explained_recs = explain_recommendations_with_gemini(raw_recommendations)

    # ---------------- Persist / Update in DB ----------------
    # Clear old profiling and issues for clean re-runs
    db.query(ProfileReport).filter(ProfileReport.dataset_id == dataset_id).delete()
    db.query(IssueDetection).filter(IssueDetection.dataset_id == dataset_id).delete()
    db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "before").delete()
    db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).delete()

    # Save Profile Report
    profile_record = ProfileReport(
        dataset_id=dataset_id,
        column_profiles=col_profiles,
        summary_stats=summary_stats,
    )
    db.add(profile_record)

    # Save Issues
    issue_items = []
    for issue in detected_issues:
        issue_id = str(uuid.uuid4())
        db_issue = IssueDetection(
            id=issue_id,
            dataset_id=dataset_id,
            issue_type=issue["issue_type"],
            column=issue.get("column"),
            severity=issue["severity"],
            details=issue["details"],
        )
        db.add(db_issue)
        issue_items.append(IssueItem(
            id=issue_id,
            issue_type=issue["issue_type"],
            column=issue.get("column"),
            severity=issue["severity"],
            title=issue["title"],
            details=issue["details"],
        ))

    # Save Health Score
    health_record = HealthScore(
        dataset_id=dataset_id,
        stage="before",
        composite_score=comp_score,
        sub_scores=sub_scores,
        summary_text=summary_text,
    )
    db.add(health_record)

    # Save Recommendations
    rec_items = []
    for rec in explained_recs:
        rec_id = str(uuid.uuid4())
        db_rec = Recommendation(
            id=rec_id,
            dataset_id=dataset_id,
            issue_type=rec["issue_type"],
            column=rec.get("column"),
            method=rec["method"],
            reason_title=rec["reason_title"],
            explanation_text=rec["explanation_text"],
            severity=rec["severity"],
            is_destructive=rec["is_destructive"],
            is_approved=rec["is_approved"],
        )
        db.add(db_rec)
        rec_items.append(RecommendationItem(
            id=rec_id,
            issue_type=rec["issue_type"],
            column=rec.get("column"),
            method=rec["method"],
            reason_title=rec["reason_title"],
            explanation_text=rec["explanation_text"],
            severity=rec["severity"],
            is_destructive=rec["is_destructive"],
            is_approved=rec["is_approved"],
        ))

    db.commit()

    return FullDiagnosticResponse(
        dataset=DatasetResponse(
            id=db_dataset.id,
            filename=db_dataset.filename,
            file_size_bytes=db_dataset.file_size_bytes,
            row_count=db_dataset.row_count,
            col_count=db_dataset.col_count,
            uploaded_at=db_dataset.uploaded_at,
            has_objective=bool(obj),
            has_profile=True,
            has_health_score=True,
            has_cleaned_data=bool(db_dataset.cleaned_file_path),
        ),
        objective=ObjectiveResponse.model_validate(obj) if obj else None,
        profile=ProfileReportResponse(
            dataset_id=dataset_id,
            summary=summary_stats,
            columns=col_profiles,
        ),
        health_score=HealthScoreResponse(
            stage="before",
            composite_score=comp_score,
            grade=grade,
            sub_scores=sub_scores,
            summary_text=summary_text,
            top_negative_drivers=drivers,
        ),
        issues=issue_items,
        recommendations=rec_items,
    )
