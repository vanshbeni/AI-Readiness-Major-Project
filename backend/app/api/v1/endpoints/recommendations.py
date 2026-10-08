from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset, invalidate_execution
from app.core.config import settings
from app.core.database import get_db
from app.core.storage import dataset_lock
from app.engine.explainer import explain_recommendations_with_gemini
from app.models.dataset import Dataset, Recommendation
from app.schemas.dataset import RecommendationItem, RecommendationApprovalRequest

router = APIRouter()


@router.get("/{dataset_id}/recommendations", response_model=List[RecommendationItem])
def get_recommendations(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    return db.query(Recommendation).filter(Recommendation.dataset_id == db_dataset.id).all()


@router.post("/{dataset_id}/recommendations/approve", response_model=List[RecommendationItem])
def update_recommendation_approvals(
    payload: RecommendationApprovalRequest,
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """Updates is_approved flags. Changing any approval invalidates previously executed results."""
    recs = db.query(Recommendation).filter(Recommendation.dataset_id == db_dataset.id).all()
    rec_by_id = {r.id: r for r in recs}

    unknown = [rid for rid in payload.approvals if rid not in rec_by_id]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"{len(unknown)} recommendation id(s) do not belong to this dataset. Re-run diagnostics and try again.",
        )

    changed = False
    for rec_id, approved in payload.approvals.items():
        rec = rec_by_id[rec_id]
        if rec.is_approved != approved:
            rec.is_approved = approved
            changed = True

    if changed:
        invalidate_execution(db, db_dataset)
    db.commit()
    return db.query(Recommendation).filter(Recommendation.dataset_id == db_dataset.id).all()


@router.post("/{dataset_id}/recommendations/explain", response_model=List[RecommendationItem])
def regenerate_explanations(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    """Re-requests AI explanations (e.g. after Gemini was overloaded). Does not change approvals or results."""
    if not settings.GEMINI_API_KEY.strip():
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not configured on the backend.")

    with dataset_lock(db_dataset.id):
        recs = db.query(Recommendation).filter(Recommendation.dataset_id == db_dataset.id).all()
        if not recs:
            raise HTTPException(status_code=400, detail="Run diagnostics first.")

        rec_dicts = [
            {
                "issue_type": r.issue_type,
                "column": r.column,
                "method": r.method,
                "stats_context": dict(r.params or {}),
            }
            for r in recs
        ]
        explained = explain_recommendations_with_gemini(rec_dicts)
        if not any(e.get("explanation_source") == "ai" for e in explained):
            raise HTTPException(
                status_code=503,
                detail="Gemini is currently unavailable (high demand or timeout). Statistical explanations were kept; try again in a minute.",
            )
        for rec, e in zip(recs, explained):
            if e.get("explanation_source") == "ai":
                rec.explanation_text = e["explanation_text"]
                rec.explanation_source = "ai"
        db.commit()
    return db.query(Recommendation).filter(Recommendation.dataset_id == db_dataset.id).all()
