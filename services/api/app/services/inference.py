from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.analysis import AnalysisResult, ModelSignalSummary, SegmentResult, SegmentSignals
from app.services.extraction import ExtractedDocument
from app.services.features import SegmentFeatureSet, extract_segment_features
from app.services.language import detect_language
from app.services.normalization import clamp, chunk_paragraphs, mean, safe_std, tokenize_words
from app.services.recommendation import (
    agreement_summary,
    confidence_label,
    default_limitations,
    default_warnings,
    recommendation_for,
    risk_band,
    summary_for,
)


MODEL_VERSION = "heuristic-ensemble-v0"
CALIBRATION_VERSION = "temperature-lite-v0"


def analyze_document(extracted: ExtractedDocument) -> AnalysisResult:
    language = detect_language(extracted.text)
    chunks = chunk_paragraphs(extracted.paragraphs)

    if not chunks:
        chunks = chunk_paragraphs([extracted.text])

    segment_features = [extract_segment_features(chunk.text) for chunk in chunks]
    consistency = compute_document_consistency(segment_features)

    segments: list[SegmentResult] = []
    segment_scores: list[float] = []
    all_reasons: list[str] = []

    for chunk, feature_set in zip(chunks, segment_features, strict=False):
        score = score_segment(feature_set, consistency)
        segment_scores.append(score)
        segment_confidence = segment_confidence_score(feature_set, extracted.extraction_quality, language.supported)
        all_reasons.extend(feature_set.reasons)

        segments.append(
            SegmentResult(
                segment_id=f"seg_{uuid4().hex[:10]}",
                paragraph_index=chunk.paragraph_index,
                start_paragraph=chunk.start_paragraph,
                end_paragraph=chunk.end_paragraph,
                label=chunk.label,
                excerpt=excerpt_for(chunk.text),
                risk_score=round(score * 100),
                confidence_level=confidence_label(segment_confidence),
                key_signals=feature_set.reasons[:3],
                signals=SegmentSignals(
                    classifier_probability=round(feature_set.classifier_probability, 4),
                    stylometric_anomaly_score=round(feature_set.stylometric_anomaly_score, 4),
                    surprisal_signal=round(feature_set.surprisal_signal, 4),
                    consistency_signal=round(consistency, 4),
                ),
            )
        )

    classifier_values = [feature.classifier_probability for feature in segment_features]
    stylometric_values = [feature.stylometric_anomaly_score for feature in segment_features]
    surprisal_values = [feature.surprisal_signal for feature in segment_features]
    classifier_mean = mean(classifier_values)
    stylometric_mean = mean(stylometric_values)
    surprisal_mean = mean(surprisal_values)
    quality_penalty = quality_penalty_for(extracted.extraction_quality, language.supported, extracted.text)
    agreement = model_agreement(classifier_mean, stylometric_mean, surprisal_mean, consistency)
    uncertainty_penalty = clamp(1.0 - agreement)

    raw_score = clamp(
        0.40 * classifier_mean
        + 0.22 * stylometric_mean
        + 0.15 * surprisal_mean
        + 0.13 * consistency
        - 0.06 * quality_penalty
        - 0.04 * uncertainty_penalty
    )
    calibrated = calibrate_score(raw_score)
    token_count = len(tokenize_words(extracted.text))
    confidence_score = document_confidence_score(
        token_count=token_count,
        extraction_quality=extracted.extraction_quality,
        language_supported=language.supported,
        agreement=agreement,
        segment_count=len(segments),
    )
    confidence = confidence_label(confidence_score)
    overall_score = round(calibrated * 100)
    band = risk_band(overall_score)
    recommendation = recommendation_for(overall_score, confidence, extracted.extraction_quality)
    most_common_reasons = [item for item, _count in Counter(all_reasons).most_common(5)]
    strongest_signal_name, weakest_signal_name = strongest_and_weakest(
        classifier_mean,
        stylometric_mean,
        surprisal_mean,
        consistency,
    )
    mixed_signals = safe_std([classifier_mean, stylometric_mean, surprisal_mean, consistency]) > 0.15

    warnings = default_warnings() + extracted.warnings
    if not language.supported:
        warnings.append("Language support is limited for this document, so the result is more cautious.")

    return AnalysisResult(
        document_id=f"doc_{uuid4().hex[:12]}",
        created_at=datetime.now(timezone.utc),
        source_type=extracted.source_type,  # type: ignore[arg-type]
        source_name=extracted.source_name,
        language=language.code,
        language_supported=language.supported,
        extraction_quality=extracted.extraction_quality,  # type: ignore[arg-type]
        extraction_warnings=extracted.warnings,
        character_count=len(extracted.text),
        token_count=token_count,
        overall_risk_score=overall_score,
        confidence_level=confidence,
        risk_band=band,
        short_summary=summary_for(overall_score, confidence, agreement, mixed_signals),
        recommendation=recommendation,
        key_signals=most_common_reasons[:4],
        model_agreement_summary=agreement_summary(agreement, strongest_signal_name, weakest_signal_name),
        calibration_disclaimer=(
            "This score is advisory and calibrated against benchmark-like patterns. "
            "It does not prove authorship or intent."
        ),
        limitations=default_limitations(language.supported),
        warnings=warnings,
        signals=ModelSignalSummary(
            classifier_probability=round(classifier_mean, 4),
            stylometric_anomaly_score=round(stylometric_mean, 4),
            surprisal_signal=round(surprisal_mean, 4),
            cross_segment_consistency=round(consistency, 4),
            quality_penalty=round(quality_penalty, 4),
            uncertainty_penalty=round(uncertainty_penalty, 4),
            model_agreement=round(agreement, 4),
        ),
        segments=segments,
        model_version=MODEL_VERSION,
        calibration_version=CALIBRATION_VERSION,
    )


def score_segment(feature_set: SegmentFeatureSet, consistency: float) -> float:
    return clamp(
        0.44 * feature_set.classifier_probability
        + 0.26 * feature_set.stylometric_anomaly_score
        + 0.16 * feature_set.surprisal_signal
        + 0.14 * consistency
    )


def compute_document_consistency(features: list[SegmentFeatureSet]) -> float:
    if len(features) <= 1:
        return 0.35

    avg_sentence_lengths = [feature.avg_sentence_length for feature in features]
    ttrs = [feature.type_token_ratio for feature in features]
    entropy_values = [feature.entropy for feature in features]

    sentence_uniformity = 1.0 - clamp(safe_std(avg_sentence_lengths) / 10.0)
    ttr_uniformity = 1.0 - clamp(safe_std(ttrs) / 0.15)
    entropy_uniformity = 1.0 - clamp(safe_std(entropy_values) / 1.2)
    return clamp(0.40 * sentence_uniformity + 0.30 * ttr_uniformity + 0.30 * entropy_uniformity)


def model_agreement(classifier: float, stylometric: float, surprisal: float, consistency: float) -> float:
    disagreement = safe_std([classifier, stylometric, surprisal, consistency])
    return clamp(1.0 - disagreement / 0.35)


def document_confidence_score(
    *,
    token_count: int,
    extraction_quality: str,
    language_supported: bool,
    agreement: float,
    segment_count: int,
) -> float:
    length_support = clamp((token_count - 60) / 440)
    quality_support = {"good": 1.0, "fair": 0.72, "poor": 0.35}[extraction_quality]
    language_support = 1.0 if language_supported else 0.45
    segment_support = clamp(segment_count / 3.0)
    return clamp(
        0.35 * agreement
        + 0.25 * length_support
        + 0.20 * quality_support
        + 0.10 * language_support
        + 0.10 * segment_support
    )


def segment_confidence_score(feature_set: SegmentFeatureSet, extraction_quality: str, language_supported: bool) -> float:
    length_support = clamp((feature_set.token_count - 50) / 250)
    quality_support = {"good": 1.0, "fair": 0.72, "poor": 0.35}[extraction_quality]
    language_support = 1.0 if language_supported else 0.45
    internal_agreement = model_agreement(
        feature_set.classifier_probability,
        feature_set.stylometric_anomaly_score,
        feature_set.surprisal_signal,
        mean(
            [
                feature_set.classifier_probability,
                feature_set.stylometric_anomaly_score,
                feature_set.surprisal_signal,
            ]
        ),
    )
    return clamp(0.45 * internal_agreement + 0.30 * length_support + 0.15 * quality_support + 0.10 * language_support)


def quality_penalty_for(extraction_quality: str, language_supported: bool, text: str) -> float:
    penalty = {"good": 0.05, "fair": 0.16, "poor": 0.34}[extraction_quality]
    if not language_supported:
        penalty += 0.12
    if len(text) < 300:
        penalty += 0.10
    return clamp(penalty)


def calibrate_score(raw_score: float) -> float:
    compressed = 0.50 + ((raw_score - 0.50) * 0.88)
    return clamp(compressed)


def excerpt_for(text: str, max_chars: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def strongest_and_weakest(classifier: float, stylometric: float, surprisal: float, consistency: float) -> tuple[str, str]:
    signals = {
        "classifier probability": classifier,
        "stylometric anomaly": stylometric,
        "surprisal proxy": surprisal,
        "cross-segment consistency": consistency,
    }
    sorted_items = sorted(signals.items(), key=lambda item: item[1], reverse=True)
    return sorted_items[0][0], sorted_items[-1][0]
