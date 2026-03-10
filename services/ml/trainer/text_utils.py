"""Text processing utilities — thin re-export of the shared text_features library.

All core logic lives in ``libs/text_features`` so that training and API
stay in sync.  This module re-exports the public API for backward
compatibility with existing ``trainer.*`` imports.
"""

from __future__ import annotations

# Re-export everything from the shared library
from text_features.text import (  # noqa: F401
    SENTENCE_SPLIT_RE,
    WORD_RE,
    TextChunk,
    chunk_text,
    clamp,
    mean,
    normalize_text,
    safe_std,
    split_paragraphs,
    split_sentences,
    text_length_bucket,
    tokenize_words,
)
