"""Text normalisation — thin re-export of the shared text_features library.

All core logic lives in ``libs/text_features`` so that training and API
stay in sync.  This module re-exports the public API and provides the
API-specific ``TextChunk`` variant (which carries an extra ``label`` field
for UI display).
"""

from __future__ import annotations

from dataclasses import dataclass

# Re-export shared utilities so existing ``from app.services.normalization import …``
# statements continue to work without changes.
from text_features.text import (  # noqa: F401
    SENTENCE_SPLIT_RE,
    WORD_RE,
    clamp,
    mean,
    normalize_text,
    safe_std,
    split_paragraphs,
    split_sentences,
    tokenize_words,
)


@dataclass(slots=True)
class TextChunk:
    """API-specific chunk with a display label and paragraph index."""

    label: str
    text: str
    paragraph_index: int
    start_paragraph: int
    end_paragraph: int


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
