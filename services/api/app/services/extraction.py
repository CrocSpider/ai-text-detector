from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.services.normalization import normalize_text, split_paragraphs


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".html",
    ".htm",
    ".rtf",
    ".docx",
    ".pdf",
}


@dataclass(slots=True)
class ExtractedDocument:
    source_type: str
    source_name: str
    text: str
    paragraphs: list[str]
    extraction_quality: str
    warnings: list[str] = field(default_factory=list)
    content_type: str | None = None


def build_text_document(text: str, source_name: str = "Pasted text") -> ExtractedDocument:
    normalized = normalize_text(text)
    quality, warnings = assess_extraction_quality(normalized)
    return ExtractedDocument(
        source_type="text",
        source_name=source_name,
        text=normalized,
        paragraphs=split_paragraphs(normalized),
        extraction_quality=quality,
        warnings=warnings,
    )


async def extract_upload(upload: UploadFile, max_bytes: int) -> ExtractedDocument:
    suffix = Path(upload.filename or "upload.txt").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {suffix or 'unknown'}",
        )

    payload = await upload.read()
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the configured upload size limit.",
        )

    text = extract_text_from_bytes(payload, suffix)
    normalized = normalize_text(text)
    quality, warnings = assess_extraction_quality(normalized)
    return ExtractedDocument(
        source_type="file",
        source_name=upload.filename or "Uploaded file",
        text=normalized,
        paragraphs=split_paragraphs(normalized),
        extraction_quality=quality,
        warnings=warnings,
        content_type=upload.content_type,
    )


def extract_text_from_bytes(payload: bytes, suffix: str) -> str:
    if suffix in {".txt", ".md"}:
        return _decode_bytes(payload)
    if suffix == ".csv":
        return _extract_csv(payload)
    if suffix == ".json":
        return _extract_json(payload)
    if suffix in {".html", ".htm"}:
        return _extract_html(payload)
    if suffix == ".rtf":
        return _extract_rtf(payload)
    if suffix == ".docx":
        return _extract_docx(payload)
    if suffix == ".pdf":
        return _extract_pdf(payload)
    return _decode_bytes(payload)


def assess_extraction_quality(text: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if not text.strip():
        return "poor", ["No readable text could be extracted from the document."]

    weird_char_count = len(re.findall(r"[^\w\s\.,;:!?()\[\]{}'\"\-/%&@#]", text))
    weird_ratio = weird_char_count / max(len(text), 1)
    avg_line_length = sum(len(line) for line in text.splitlines()) / max(len(text.splitlines()), 1)

    if len(text) < 120:
        warnings.append("The extracted text is short, which reduces reliability.")
    if weird_ratio > 0.07:
        warnings.append("The text contains elevated symbol noise, which may indicate weak extraction or OCR artifacts.")
    if avg_line_length < 12:
        warnings.append("The extracted text is highly fragmented across short lines.")

    if len(text) < 80 or weird_ratio > 0.12:
        return "poor", warnings
    if len(text) < 300 or weird_ratio > 0.05 or avg_line_length < 24:
        return "fair", warnings
    return "good", warnings


def _decode_bytes(payload: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="ignore")


def _extract_csv(payload: bytes) -> str:
    text = _decode_bytes(payload)
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return ""

    lines: list[str] = []
    for row in rows:
        textual_cells = [cell.strip() for cell in row if _looks_textual(cell)]
        if textual_cells:
            lines.append(" | ".join(textual_cells))
    return "\n\n".join(lines)


def _extract_json(payload: bytes) -> str:
    parsed = json.loads(_decode_bytes(payload))
    lines: list[str] = []

    def walk(value: object, path: str = "root") -> None:
        if isinstance(value, str) and value.strip():
            lines.append(f"{path}: {value.strip()}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")

    walk(parsed)
    return "\n\n".join(lines)


def _extract_html(payload: bytes) -> str:
    html = _decode_bytes(payload)
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text("\n")
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)


def _extract_rtf(payload: bytes) -> str:
    rtf = _decode_bytes(payload)
    try:
        from striprtf.striprtf import rtf_to_text

        return rtf_to_text(rtf)
    except Exception:
        fallback = re.sub(r"\\[a-z]+[0-9-]* ?", " ", rtf)
        return fallback.replace("{", " ").replace("}", " ")


def _extract_docx(payload: bytes) -> str:
    try:
        from docx import Document

        document = Document(io.BytesIO(payload))
        return "\n\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"DOCX extraction failed: {exc}") from exc


def _extract_pdf(payload: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(payload))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        return "\n\n".join(page for page in pages if page)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {exc}") from exc


def _looks_textual(cell: str) -> bool:
    stripped = cell.strip()
    if not stripped:
        return False
    digit_ratio = sum(char.isdigit() for char in stripped) / max(len(stripped), 1)
    return digit_ratio < 0.7
