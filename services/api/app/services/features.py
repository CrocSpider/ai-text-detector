"""API-side feature extraction — delegates to the shared text_features library.

The ``SegmentFeatureSet`` dataclass is kept here because it includes
API-specific fields (``reasons``) that aren't needed during training.
All numerical computation is delegated to
``text_features.features.compute_stylometric_features``.
"""

from __future__ import annotations

from dataclasses import dataclass

from text_features.features import (  # noqa: F401
    COMMON_TRANSITIONS,
    STOPWORDS,
    STYLOMETRY_FEATURE_NAMES,
    compute_stylometric_features,
    shannon_entropy,
    sigmoid,
    stopword_ratio,
    stylometric_feature_map as _shared_stylometric_feature_map,
    stylometric_feature_vector as _shared_stylometric_feature_vector,
)


@dataclass(slots=True)
class SegmentFeatureSet:
    token_count: int
    sentence_count: int
    avg_sentence_length: float
    sentence_length_std: float
    type_token_ratio: float
    repeated_bigram_ratio: float
    repeated_sentence_ratio: float
    punctuation_density: float
    punctuation_regularization: float
    transition_density: float
    entropy: float
    stopword_ratio: float
    uniform_sentence_signal: float
    low_diversity_signal: float
    repetition_signal: float
    punctuation_signal: float
    transition_signal: float
    entropy_signal: float
    classifier_probability: float
    stylometric_anomaly_score: float
    surprisal_signal: float
    reasons: list[str]


def extract_segment_features(text: str) -> SegmentFeatureSet:
    """Compute all segment-level features from raw text.

    Delegates numerical computation to the shared library and adds
    API-specific interpretive reasons.
    """
    raw = compute_stylometric_features(text)

    reasons: list[str] = []
    if raw["uniform_sentence_signal"] > 0.62 and raw["token_count"] > 120:
        reasons.append("Sentence lengths are unusually uniform across the segment.")
    if raw["low_diversity_signal"] > 0.58:
        reasons.append("Lexical diversity is lower than expected for the segment length.")
    if raw["repetition_signal"] > 0.35:
        reasons.append("Repeated phrasing patterns appear more often than usual.")
    if raw["entropy_signal"] > 0.55:
        reasons.append("Token variation is compressed, which can resemble low-surprisal drafting.")
    if raw["transition_signal"] > 0.55:
        reasons.append("Transition markers appear at a highly regular cadence.")
    if not reasons:
        reasons.append("The segment shows mixed stylistic signals rather than one strong pattern.")

    return SegmentFeatureSet(
        token_count=raw["token_count"],
        sentence_count=raw["sentence_count"],
        avg_sentence_length=raw["avg_sentence_length"],
        sentence_length_std=raw["sentence_length_std"],
        type_token_ratio=raw["type_token_ratio"],
        repeated_bigram_ratio=raw["repeated_bigram_ratio"],
        repeated_sentence_ratio=raw["repeated_sentence_ratio"],
        punctuation_density=raw["punctuation_density"],
        punctuation_regularization=raw["punctuation_regularization"],
        transition_density=raw["transition_density"],
        entropy=raw["entropy"],
        stopword_ratio=raw["stopword_ratio"],
        uniform_sentence_signal=raw["uniform_sentence_signal"],
        low_diversity_signal=raw["low_diversity_signal"],
        repetition_signal=raw["repetition_signal"],
        punctuation_signal=raw["punctuation_signal"],
        transition_signal=raw["transition_signal"],
        entropy_signal=raw["entropy_signal"],
        classifier_probability=raw["heuristic_classifier_probability"],
        stylometric_anomaly_score=raw["heuristic_stylometric_score"],
        surprisal_signal=raw["surprisal_signal"],
        reasons=reasons,
    )


def stylometric_feature_map(feature_set: SegmentFeatureSet) -> dict[str, float]:
    return _shared_stylometric_feature_map({
        "token_count": float(feature_set.token_count),
        "sentence_count": float(feature_set.sentence_count),
        "avg_sentence_length": feature_set.avg_sentence_length,
        "sentence_length_std": feature_set.sentence_length_std,
        "type_token_ratio": feature_set.type_token_ratio,
        "repeated_bigram_ratio": feature_set.repeated_bigram_ratio,
        "repeated_sentence_ratio": feature_set.repeated_sentence_ratio,
        "punctuation_density": feature_set.punctuation_density,
        "punctuation_regularization": feature_set.punctuation_regularization,
        "transition_density": feature_set.transition_density,
        "entropy": feature_set.entropy,
        "stopword_ratio": feature_set.stopword_ratio,
        "uniform_sentence_signal": feature_set.uniform_sentence_signal,
        "low_diversity_signal": feature_set.low_diversity_signal,
        "repetition_signal": feature_set.repetition_signal,
        "punctuation_signal": feature_set.punctuation_signal,
        "transition_signal": feature_set.transition_signal,
        "entropy_signal": feature_set.entropy_signal,
    })


def stylometric_feature_vector(feature_set: SegmentFeatureSet, feature_names: list[str] | None = None) -> list[float]:
    return _shared_stylometric_feature_vector(stylometric_feature_map(feature_set), feature_names)
