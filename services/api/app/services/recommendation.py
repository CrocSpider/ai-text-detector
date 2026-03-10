from __future__ import annotations

from app.schemas.analysis import ConfidenceLevel, RiskBand


def confidence_label(score: float) -> ConfidenceLevel:
    if score >= 0.72:
        return "high"
    if score >= 0.48:
        return "medium"
    return "low"


def risk_band(score: int) -> RiskBand:
    if score <= 24:
        return "low"
    if score <= 49:
        return "moderate"
    if score <= 74:
        return "elevated"
    return "high"


def recommendation_for(score: int, confidence: ConfidenceLevel, extraction_quality: str) -> str:
    if extraction_quality == "poor" or confidence == "low":
        return "Inconclusive due to low confidence or poor extraction"
    if score <= 24:
        return "No strong signal"
    if score <= 49:
        return "Borderline - human review recommended"
    return "Elevated risk - verify with contextual evidence"


def summary_for(score: int, confidence: ConfidenceLevel, agreement: float, mixed: bool) -> str:
    if confidence == "low":
        return "The result is cautious because the text is short, mixed, poorly extracted, or the models disagree. Human review is recommended."
    if score <= 24:
        return "No strong machine-origin signal stands out. Variation across style and structure looks closer to ordinary human drafting."
    if score <= 49:
        return "Some sections show signals that can align with machine-generated or machine-edited writing, but the evidence is mixed."
    if mixed or agreement < 0.55:
        return "Several sections show elevated machine-like signals, but model disagreement keeps the conclusion cautious."
    return "Several sections show unusually uniform style patterns that are consistent with machine-generated or heavily machine-edited text."


def agreement_summary(agreement: float, strongest_signal: str, weakest_signal: str) -> str:
    if agreement >= 0.72:
        return f"Most modules align. The strongest signal is {strongest_signal}, while {weakest_signal} is weaker but not contradictory."
    if agreement >= 0.50:
        return f"The modules agree moderately. {strongest_signal} is the clearest signal, but {weakest_signal} is softer."
    return f"The modules disagree materially. {strongest_signal} is elevated, but {weakest_signal} points in a different direction."


def default_limitations(language_supported: bool) -> list[str]:
    limitations = [
        "This detector is advisory and estimates similarity to benchmark patterns; it is not proof of origin.",
        "False positives remain possible, especially for short, polished, formulaic, or highly edited writing.",
        "Mixed-origin documents and heavily revised drafts can reduce reliability.",
    ]
    if not language_supported:
        limitations.append("This language is outside the current high-confidence support set, so calibration is weaker.")
    return limitations


def default_warnings() -> list[str]:
    return [
        "Do not use this result as sole evidence for disciplinary, legal, employment, or academic penalties.",
        "Review the highlighted sections together with contextual evidence such as drafts, revision history, and source material.",
    ]
