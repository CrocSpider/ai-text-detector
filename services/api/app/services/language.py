from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_LANGUAGES = {"en"}


@dataclass(slots=True)
class LanguageResult:
    code: str
    supported: bool
    confidence: float


def detect_language(text: str) -> LanguageResult:
    sample = text[:5000].strip()
    if len(sample) < 40:
        return LanguageResult(code="unknown", supported=False, confidence=0.2)

    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0
        candidates = detect_langs(sample)
        if not candidates:
            return LanguageResult(code="unknown", supported=False, confidence=0.2)
        best = candidates[0]
        code = getattr(best, "lang", "unknown")
        confidence = float(getattr(best, "prob", 0.0))
        return LanguageResult(
            code=code,
            supported=code in SUPPORTED_LANGUAGES and confidence >= 0.60,
            confidence=confidence,
        )
    except Exception:
        ascii_ratio = sum(1 for char in sample if ord(char) < 128) / max(len(sample), 1)
        if ascii_ratio > 0.9:
            return LanguageResult(code="en", supported=True, confidence=0.5)
        return LanguageResult(code="unknown", supported=False, confidence=0.2)
