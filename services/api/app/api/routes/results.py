from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.analysis import AnalysisResult
from app.services.store import result_store


router = APIRouter(prefix="/results", tags=["results"])


@router.get("/{document_id}", response_model=AnalysisResult)
async def get_result(document_id: str) -> AnalysisResult:
    result = result_store.get(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis result not found.")
    return result
