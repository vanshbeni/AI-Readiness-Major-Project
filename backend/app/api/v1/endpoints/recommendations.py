from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.dataset import Recommendation
from app.schemas.dataset import RecommendationItem, RecommendationApprovalRequest

router = APIRouter()


@router.get("/{dataset_id}/recommendations", response_model=List[RecommendationItem])
async def get_recommendations(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """Retrieves all generated recommendations for a dataset."""
    recs = db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).all()
    return recs


@router.post("/{dataset_id}/recommendations/approve", response_model=List[RecommendationItem])
async def update_recommendation_approvals(
    dataset_id: str,
    payload: RecommendationApprovalRequest,
    db: Session = Depends(get_db),
):
    """Updates the is_approved flag for specific recommendations."""
    recs = db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).all()
    if not recs:
        raise HTTPException(status_code=404, detail="No recommendations found for this dataset.")

    for rec in recs:
        if rec.id in payload.approvals:
            rec.is_approved = payload.approvals[rec.id]

    db.commit()
    # Refresh and return
    updated_recs = db.query(Recommendation).filter(Recommendation.dataset_id == dataset_id).all()
    return updated_recs
