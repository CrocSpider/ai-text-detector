from __future__ import annotations

import math
from dataclasses import dataclass

from trainer.text_utils import clamp, mean, safe_std, split_sentences, tokenize_words


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


@dataclass(slots=True)
class TextFeatureRow:
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
    heuristic_classifier_probability: float
    heuristic_stylometric_score: float
    surprisal_signal: float


def extract_text_features(text: str) -> TextFeatureRow:
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

    return TextFeatureRow(
        token_count=token_count,
        sentence_count=len(sentences),
        avg_sentence_length=avg_sentence_length,
        sentence_length_std=sentence_length_std,
        type_token_ratio=type_token_ratio,
        repeated_bigram_ratio=repeated_bigram_ratio,
        repeated_sentence_ratio=repeated_sentence_ratio,
        punctuation_density=punctuation_density,
        punctuation_regularization=punctuation_regularization,
        transition_density=transition_density,
        entropy=entropy,
        stopword_ratio=token_stopword_ratio,
        uniform_sentence_signal=uniform_sentence_signal,
        low_diversity_signal=low_diversity_signal,
        repetition_signal=repetition_signal,
        punctuation_signal=punctuation_signal,
        transition_signal=transition_signal,
        entropy_signal=entropy_signal,
        heuristic_classifier_probability=heuristic_classifier_probability,
        heuristic_stylometric_score=heuristic_stylometric_score,
        surprisal_signal=surprisal_signal,
    )


def stylometric_feature_map(feature_row: TextFeatureRow) -> dict[str, float]:
    return {
        "token_count": float(feature_row.token_count),
        "sentence_count": float(feature_row.sentence_count),
        "avg_sentence_length": feature_row.avg_sentence_length,
        "sentence_length_std": feature_row.sentence_length_std,
        "type_token_ratio": feature_row.type_token_ratio,
        "repeated_bigram_ratio": feature_row.repeated_bigram_ratio,
        "repeated_sentence_ratio": feature_row.repeated_sentence_ratio,
        "punctuation_density": feature_row.punctuation_density,
        "punctuation_regularization": feature_row.punctuation_regularization,
        "transition_density": feature_row.transition_density,
        "entropy": feature_row.entropy,
        "stopword_ratio": feature_row.stopword_ratio,
        "uniform_sentence_signal": feature_row.uniform_sentence_signal,
        "low_diversity_signal": feature_row.low_diversity_signal,
        "repetition_signal": feature_row.repetition_signal,
        "punctuation_signal": feature_row.punctuation_signal,
        "transition_signal": feature_row.transition_signal,
        "entropy_signal": feature_row.entropy_signal,
    }


def stylometric_feature_vector(feature_row: TextFeatureRow, feature_names: list[str] | None = None) -> list[float]:
    feature_map = stylometric_feature_map(feature_row)
    ordered_names = feature_names or STYLOMETRY_FEATURE_NAMES
    return [feature_map.get(name, 0.0) for name in ordered_names]


def compute_document_consistency(feature_rows: list[TextFeatureRow]) -> float:
    if len(feature_rows) <= 1:
        return 0.35

    avg_sentence_lengths = [feature.avg_sentence_length for feature in feature_rows]
    ttrs = [feature.type_token_ratio for feature in feature_rows]
    entropy_values = [feature.entropy for feature in feature_rows]

    sentence_uniformity = 1.0 - clamp(safe_std(avg_sentence_lengths) / 10.0)
    ttr_uniformity = 1.0 - clamp(safe_std(ttrs) / 0.15)
    entropy_uniformity = 1.0 - clamp(safe_std(entropy_values) / 1.2)
    return clamp(0.40 * sentence_uniformity + 0.30 * ttr_uniformity + 0.30 * entropy_uniformity)


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
