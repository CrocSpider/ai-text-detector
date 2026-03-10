from __future__ import annotations

from fastapi import UploadFile

from app.core.config import get_settings
from app.schemas.analysis import AnalysisResult
from app.services.extraction import build_text_document, extract_upload
from app.services.inference import analyze_document
from app.services.store import result_store


async def analyze_text_input(text: str, source_name: str | None = None) -> AnalysisResult:
    extracted = build_text_document(text=text, source_name=source_name or "Pasted text")
    result = analyze_document(extracted)
    result_store.save(result)
    return result


async def analyze_uploaded_file(upload: UploadFile) -> AnalysisResult:
    settings = get_settings()
    extracted = await extract_upload(upload, max_bytes=settings.max_upload_size_mb * 1024 * 1024)
    result = analyze_document(extracted)
    result_store.save(result)
    return result
