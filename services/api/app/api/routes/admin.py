from __future__ import annotations

from fastapi import APIRouter

from app.schemas.analysis import EvaluationSummary
from app.services.evaluation_data import sample_evaluation_summary


router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/evaluations/summary", response_model=EvaluationSummary)
async def get_evaluation_summary() -> EvaluationSummary:
    return sample_evaluation_summary()
