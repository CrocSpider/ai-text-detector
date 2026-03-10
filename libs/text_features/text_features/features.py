"""Stylometric and document-level feature computation.

This is the single source of truth for all feature formulas.  Both the
training pipeline (``services/ml``) and the API service (``services/api``)
import from here so that feature computation is guaranteed to be identical
at training time and at inference time.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

from text_features.text import clamp, mean, safe_std, split_sentences, tokenize_words


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMON_TRANSITIONS = {
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "overall",
    "in addition",
    "for example",
    "in conclusion",
    "as a result",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

STYLOMETRY_FEATURE_NAMES = [
    "token_count",
    "sentence_count",
    "avg_sentence_length",
    "sentence_length_std",
    "type_token_ratio",
    "repeated_bigram_ratio",
    "repeated_sentence_ratio",
    "punctuation_density",
    "punctuation_regularization",
    "transition_density",
    "entropy",
    "stopword_ratio",
    "uniform_sentence_signal",
    "low_diversity_signal",
    "repetition_signal",
    "punctuation_signal",
    "transition_signal",
    "entropy_signal",
]

META_FEATURE_NAMES = [
    "classifier_mean",
    "classifier_std",
    "stylometry_mean",
    "stylometry_std",
    "surprisal_mean",
    "surprisal_std",
    "consistency",
    "quality_penalty",
    "uncertainty_penalty",
    "token_count",
    "segment_count",
    "language_supported",
    "mean_segment_tokens",
    "max_segment_tokens",
]


# ---------------------------------------------------------------------------
# Primitive math helpers
# ---------------------------------------------------------------------------

def shannon_entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def stopword_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in STOPWORDS) / len(tokens)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


# ---------------------------------------------------------------------------
# Stylometric feature computation (segment-level)
# ---------------------------------------------------------------------------

def compute_stylometric_features(text: str) -> dict[str, Any]:
    """Compute all segment-level stylometric features from raw text.

    Returns a flat dict whose keys are a superset of
    ``STYLOMETRY_FEATURE_NAMES`` plus the derived heuristic scores.

    This function is the **single source of truth** for feature computation —
    both training and inference must call it.
    """
    tokens = tokenize_words(text)
    sentences = split_sentences(text)
    token_count = len(tokens)
    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences if sentence.strip()]
    sentence_lengths = [length for length in sentence_lengths if length > 0]

    avg_sentence_length = mean([float(length) for length in sentence_lengths])
    sentence_length_std = safe_std([float(length) for length in sentence_lengths])
    type_token_ratio = len(set(tokens)) / max(token_count, 1)
    token_stopword_ratio = stopword_ratio(tokens)

    bigrams = list(zip(tokens, tokens[1:]))
    repeated_bigram_ratio = 0.0
    if bigrams:
        repeated_bigram_ratio = (len(bigrams) - len(set(bigrams))) / len(bigrams)

    normalized_sentences = [" ".join(tokenize_words(sentence)) for sentence in sentences if sentence.strip()]
    repeated_sentence_ratio = 0.0
    if normalized_sentences:
        repeated_sentence_ratio = (
            len(normalized_sentences) - len(set(normalized_sentences))
        ) / len(normalized_sentences)

    punctuation_chars = [char for char in text if char in ".,;:!?-"]
    punctuation_density = len(punctuation_chars) / max(len(text), 1)
    punctuation_counts = [sum(1 for char in sentence if char in ".,;:!?-") for sentence in sentences if sentence]
    punctuation_std = safe_std([float(count) for count in punctuation_counts])
    punctuation_regularization = 1.0 - clamp(punctuation_std / 5.0)

    lowered = text.lower()
    transition_hits = sum(lowered.count(marker) for marker in COMMON_TRANSITIONS)
    transition_density = transition_hits / max(len(sentences), 1)

    entropy = shannon_entropy(tokens)
    entropy_signal = 1.0 - clamp((entropy - 3.5) / 2.2)
    uniform_sentence_signal = 1.0 - clamp(sentence_length_std / max(avg_sentence_length, 1.0))
    low_diversity_signal = clamp((0.48 - type_token_ratio) / 0.24)
    repetition_signal = clamp((repeated_bigram_ratio * 1.5) + (repeated_sentence_ratio * 2.0))
    punctuation_signal = clamp((punctuation_regularization - 0.40) / 0.50)
    transition_signal = clamp((transition_density - 0.4) / 1.0)

    heuristic_stylometric_score = clamp(
        0.35 * uniform_sentence_signal
        + 0.25 * low_diversity_signal
        + 0.20 * punctuation_signal
        + 0.20 * transition_signal
    )
    surprisal_signal = clamp(0.55 * entropy_signal + 0.45 * repetition_signal)
    heuristic_classifier_probability = sigmoid(
        -1.1
        + 1.7 * uniform_sentence_signal
        + 1.3 * low_diversity_signal
        + 1.2 * repetition_signal
        + 0.8 * punctuation_signal
        + 0.4 * transition_signal
    )

    return {
        "token_count": token_count,
        "sentence_count": len(sentences),
        "avg_sentence_length": avg_sentence_length,
        "sentence_length_std": sentence_length_std,
        "type_token_ratio": type_token_ratio,
        "repeated_bigram_ratio": repeated_bigram_ratio,
        "repeated_sentence_ratio": repeated_sentence_ratio,
        "punctuation_density": punctuation_density,
        "punctuation_regularization": punctuation_regularization,
        "transition_density": transition_density,
        "entropy": entropy,
        "stopword_ratio": token_stopword_ratio,
        "uniform_sentence_signal": uniform_sentence_signal,
        "low_diversity_signal": low_diversity_signal,
        "repetition_signal": repetition_signal,
        "punctuation_signal": punctuation_signal,
        "transition_signal": transition_signal,
        "entropy_signal": entropy_signal,
        "heuristic_classifier_probability": heuristic_classifier_probability,
        "heuristic_stylometric_score": heuristic_stylometric_score,
        "surprisal_signal": surprisal_signal,
    }


# ---------------------------------------------------------------------------
# Stylometric feature helpers (dict → vector conversion)
# ---------------------------------------------------------------------------

def stylometric_feature_map(features: dict[str, Any]) -> dict[str, float]:
    """Extract the stylometry-model input features from a full feature dict."""
    return {name: float(features.get(name, 0.0)) for name in STYLOMETRY_FEATURE_NAMES}


def stylometric_feature_vector(features: Any, feature_names: list[str] | None = None) -> list[float]:
    """Convert a feature dict (or dataclass with matching attributes) to an ordered vector.

    Accepts either a plain dict or any object whose attributes match the
    feature names (for backward compatibility with existing dataclass-based
    callers).
    """
    ordered_names = feature_names or STYLOMETRY_FEATURE_NAMES
    if isinstance(features, dict):
        return [float(features.get(name, 0.0)) for name in ordered_names]
    # Dataclass / object fallback
    feature_map_dict = {
        name: float(getattr(features, name, 0.0))
        for name in ordered_names
    }
    return [feature_map_dict.get(name, 0.0) for name in ordered_names]


# ---------------------------------------------------------------------------
# Document-level helpers
# ---------------------------------------------------------------------------

def compute_document_consistency(feature_dicts: list[dict[str, Any]]) -> float:
    """Compute cross-segment consistency from a list of per-segment feature dicts.

    Also accepts dataclass instances with matching attribute names.
    """
    if len(feature_dicts) <= 1:
        return 0.35

    def _get(obj: Any, key: str) -> float:
        if isinstance(obj, dict):
            return float(obj.get(key, 0.0))
        return float(getattr(obj, key, 0.0))

    avg_sentence_lengths = [_get(f, "avg_sentence_length") for f in feature_dicts]
    ttrs = [_get(f, "type_token_ratio") for f in feature_dicts]
    entropy_values = [_get(f, "entropy") for f in feature_dicts]

    sentence_uniformity = 1.0 - clamp(safe_std(avg_sentence_lengths) / 10.0)
    ttr_uniformity = 1.0 - clamp(safe_std(ttrs) / 0.15)
    entropy_uniformity = 1.0 - clamp(safe_std(entropy_values) / 1.2)
    return clamp(0.40 * sentence_uniformity + 0.30 * ttr_uniformity + 0.30 * entropy_uniformity)


def model_agreement(classifier: float, stylometric: float, surprisal: float, consistency: float) -> float:
    disagreement = safe_std([classifier, stylometric, surprisal, consistency])
    return clamp(1.0 - disagreement / 0.35)


def quality_penalty_for(extraction_quality: str, language_supported: bool, text: str) -> float:
    penalty = {"good": 0.05, "fair": 0.16, "poor": 0.34}.get(extraction_quality, 0.16)
    if not language_supported:
        penalty += 0.12
    if len(text) < 300:
        penalty += 0.10
    return clamp(penalty)
