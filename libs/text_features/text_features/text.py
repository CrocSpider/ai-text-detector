"""Core text processing utilities shared between training and inference.

This is the single source of truth for tokenisation, sentence splitting,
text normalisation, chunking, and numeric helpers.  Both ``services/ml`` and
``services/api`` must import from here so that training and serving stay in
sync.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class TextChunk:
    """A contiguous portion of a document produced by paragraph-boundary chunking."""

    text: str
    start_paragraph: int
    end_paragraph: int


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, target_chars: int, max_chars: int) -> list[TextChunk]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[TextChunk] = []
    bucket: list[str] = []
    current_chars = 0
    start = 0

    for index, paragraph in enumerate(paragraphs):
        candidate_size = current_chars + len(paragraph)
        should_flush = bucket and (candidate_size > max_chars or current_chars >= target_chars)
        if should_flush:
            chunks.append(TextChunk(text="\n\n".join(bucket), start_paragraph=start, end_paragraph=index - 1))
            bucket = []
            current_chars = 0
            start = index

        bucket.append(paragraph)
        current_chars += len(paragraph)

    if bucket:
        chunks.append(TextChunk(text="\n\n".join(bucket), start_paragraph=start, end_paragraph=len(paragraphs) - 1))
    return chunks


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(stripped) if segment.strip()]
    return parts or [stripped]


def text_length_bucket(text: str) -> str:
    token_count = len(tokenize_words(text))
    if token_count < 150:
        return "short"
    if token_count < 500:
        return "medium"
    return "long"


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def trimmed_mean(values: list[float], trim_frac: float = 0.10) -> float:
    """Mean after dropping the top *trim_frac* fraction of values.

    For short lists (<= 3 items) the plain mean is returned to avoid
    removing meaningful signal on short documents.
    """
    if not values:
        return 0.0
    n = len(values)
    if n <= 3:
        return mean(values)
    n_drop = max(1, int(n * trim_frac))
    trimmed = sorted(values)[: n - n_drop]  # drop the highest n_drop values
    return mean(trimmed)


def safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))
