from __future__ import annotations

import math
import re
from dataclasses import dataclass


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class TextChunk:
    label: str
    text: str
    paragraph_index: int
    start_paragraph: int
    end_paragraph: int


def normalize_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def split_paragraphs(text: str) -> list[str]:
    normalized = normalize_text(text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    if paragraphs:
        return paragraphs
    single_line_parts = [line.strip() for line in normalized.split("\n") if line.strip()]
    return single_line_parts or ([normalized] if normalized else [])


def chunk_paragraphs(
    paragraphs: list[str],
    target_chars: int = 1200,
    max_chars: int = 1800,
) -> list[TextChunk]:
    if not paragraphs:
        return []

    chunks: list[TextChunk] = []
    bucket: list[str] = []
    start = 0
    current_chars = 0

    for index, paragraph in enumerate(paragraphs):
        candidate_size = current_chars + len(paragraph)
        should_flush = bucket and (candidate_size > max_chars or current_chars >= target_chars)

        if should_flush:
            chunk_text = "\n\n".join(bucket)
            chunk_index = len(chunks)
            chunks.append(
                TextChunk(
                    label=f"Section {chunk_index + 1}",
                    text=chunk_text,
                    paragraph_index=start,
                    start_paragraph=start,
                    end_paragraph=index - 1,
                )
            )
            bucket = []
            current_chars = 0
            start = index

        bucket.append(paragraph)
        current_chars += len(paragraph)

    if bucket:
        chunk_index = len(chunks)
        chunks.append(
            TextChunk(
                label=f"Section {chunk_index + 1}",
                text="\n\n".join(bucket),
                paragraph_index=start,
                start_paragraph=start,
                end_paragraph=len(paragraphs) - 1,
            )
        )

    return chunks


def tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    pieces = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(stripped) if segment.strip()]
    return pieces or [stripped]


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
