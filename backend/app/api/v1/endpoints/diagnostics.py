import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset, get_snapshot, invalidate_execution
from app.api.v1.endpoints.datasets import build_dataset_response
from app.core.database import get_db
from app.core.storage import dataset_lock, json_safe, read_dataset
from app.engine.detector import detect_issues
from app.engine.explainer import explain_recommendations_with_gemini
from app.engine.profiler import profile_dataset
from app.engine.rules import select_remediation_methods
from app.engine.scorer import calculate_health_score
from app.models.dataset import Dataset, HealthScore, IssueDetection, Objective, ProfileReport, Recommendation
from app.schemas.dataset import (
    FullDiagnosticResponse,
    HealthScoreResponse,
    IssueItem,
    ObjectiveResponse,
    ProfileReportResponse,
    RecommendationItem,
)

router = APIRouter()


@router.post("/{dataset_id}/diagnose", response_model=FullDiagnosticResponse, status_code=status.HTTP_200_OK)
def run_diagnostics(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """
    Profiling -> Issue Detection -> Health Score -> Rule Matrix -> Explanations.
    Requires an objective. Re-running clears previous recommendations and any executed pipeline results.
    """
    dataset_id = db_dataset.id
    obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    if not obj:
        raise HTTPException(status_code=400, detail="Set the ML objective (target column and problem type) before running diagnostics.")

    with dataset_lock(dataset_id):
        target_col, prob_type = obj.target_column, obj.problem_type
        df = read_dataset(db_dataset.raw_file_path)

        summary_stats, col_profiles = profile_dataset(df, target_col=target_col)
        detected_issues = detect_issues(df, target_col=target_col, problem_type=prob_type)
        comp_score, sub_scores, grade, summary_text, drivers = calculate_health_score(
            summary_stats, col_profiles, detected_issues, problem_type=prob_type
        )
        raw_recommendations = select_remediation_methods(detected_issues, col_profiles, problem_type=prob_type)
        explained_recs = explain_recommendations_with_gemini(raw_recommendations)

        summary_stats = json_safe(summary_stats)
        col_profiles = json_safe(col_profiles)

        try:
            db.query(ProfileReport).filter(ProfileReport.dataset_id == dataset_id).delete()
            db.query(IssueDetection).filter(IssueDetection.dataset_id == dataset_id).delete()
            db.query(HealthScore).filter(HealthScore.dataset_id == dataset_id, HealthScore.stage == "before").delete()
            db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).delete()
            invalidate_execution(db, db_dataset)

            db.add(ProfileReport(dataset_id=dataset_id, column_profiles=col_profiles, summary_stats=summary_stats))

            issue_items = []
            for issue in detected_issues:
                issue_id = str(uuid.uuid4())
                details = json_safe(issue["details"])
                db.add(IssueDetection(
                    id=issue_id, dataset_id=dataset_id, issue_type=issue["issue_type"],
                    column=issue.get("column"), severity=issue["severity"], details=details,
                ))
                issue_items.append(IssueItem(
                    id=issue_id, issue_type=issue["issue_type"], column=issue.get("column"),
                    severity=issue["severity"], title=issue["title"], details=details,
                ))

            db.add(HealthScore(
                dataset_id=dataset_id, stage="before", composite_score=comp_score,
                sub_scores=sub_scores, summary_text=summary_text,
            ))

            rec_items = []
            for rec in explained_recs:
                rec_id = str(uuid.uuid4())
                db.add(Recommendation(
                    id=rec_id,
                    dataset_id=dataset_id,
                    issue_type=rec["issue_type"],
                    column=rec.get("column"),
                    method=rec["method"],
                    reason_title=rec["reason_title"],
                    explanation_text=rec["explanation_text"],
                    explanation_source=rec.get("explanation_source", "statistical"),
                    params=json_safe(rec.get("stats_context", {})),
                    severity=rec["severity"],
                    is_destructive=rec["is_destructive"],
                    is_approved=rec["is_approved"],
                ))
                rec_items.append(RecommendationItem(
                    id=rec_id,
                    issue_type=rec["issue_type"],
                    column=rec.get("column"),
                    method=rec["method"],
                    reason_title=rec["reason_title"],
                    explanation_text=rec["explanation_text"],
                    explanation_source=rec.get("explanation_source", "statistical"),
                    severity=rec["severity"],
                    is_destructive=rec["is_destructive"],
                    is_approved=rec["is_approved"],
                ))
            db.flush()

            response = FullDiagnosticResponse(
                dataset=build_dataset_response(db, db_dataset),
                objective=ObjectiveResponse.model_validate(obj),
                profile=ProfileReportResponse(dataset_id=dataset_id, summary=summary_stats, columns=col_profiles),
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
            get_snapshot(db, dataset_id).diagnostics = json_safe(response.model_dump(mode="json"))
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=409, detail="Diagnostics were updated concurrently. Please retry.")

    return response
