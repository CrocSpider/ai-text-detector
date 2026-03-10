from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.schemas.analysis import AnalysisResult, BatchAnalysisResponse, TextAnalyzeRequest
from app.services.orchestrator import analyze_text_input, analyze_uploaded_file


router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post("/text", response_model=AnalysisResult)
async def analyze_text(payload: TextAnalyzeRequest) -> AnalysisResult:
    source_name = payload.source_name or payload.title or "Pasted text"
    return await analyze_text_input(payload.text, source_name=source_name)


@router.post("/file", response_model=AnalysisResult)
async def analyze_file(file: UploadFile = File(...)) -> AnalysisResult:
    return await analyze_uploaded_file(file)


@router.post("/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(files: list[UploadFile] = File(...)) -> BatchAnalysisResponse:
    settings = get_settings()
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")
    if len(files) > settings.batch_limit:
        raise HTTPException(status_code=400, detail=f"Batch limit is {settings.batch_limit} files.")

    results = []
    for upload in files:
        results.append(await analyze_uploaded_file(upload))
    return BatchAnalysisResponse(count=len(results), results=results)
