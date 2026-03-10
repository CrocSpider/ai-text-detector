from __future__ import annotations

import math
from dataclasses import dataclass

from app.services.normalization import clamp, mean, safe_std, split_sentences, tokenize_words


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
    classifier_probability: float
    stylometric_anomaly_score: float
    surprisal_signal: float
    reasons: list[str]


def extract_segment_features(text: str) -> SegmentFeatureSet:
    tokens = tokenize_words(text)
    sentences = split_sentences(text)
    token_count = len(tokens)
    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences if sentence.strip()]
    sentence_lengths = [length for length in sentence_lengths if length > 0]

    avg_sentence_length = mean([float(length) for length in sentence_lengths])
    sentence_length_std = safe_std([float(length) for length in sentence_lengths])
    ttr = len(set(tokens)) / max(token_count, 1)

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
    low_diversity_signal = clamp((0.48 - ttr) / 0.24)
    repetition_signal = clamp((repeated_bigram_ratio * 1.5) + (repeated_sentence_ratio * 2.0))
    punctuation_signal = clamp((punctuation_regularization - 0.40) / 0.50)
    transition_signal = clamp((transition_density - 0.4) / 1.0)

    stylometric = clamp(
        0.35 * uniform_sentence_signal
        + 0.25 * low_diversity_signal
        + 0.20 * punctuation_signal
        + 0.20 * transition_signal
    )
    surprisal = clamp(0.55 * entropy_signal + 0.45 * repetition_signal)
    classifier_probability = sigmoid(
        -1.1
        + 1.7 * uniform_sentence_signal
        + 1.3 * low_diversity_signal
        + 1.2 * repetition_signal
        + 0.8 * punctuation_signal
        + 0.4 * transition_signal
    )

    reasons: list[str] = []
    if uniform_sentence_signal > 0.62 and token_count > 120:
        reasons.append("Sentence lengths are unusually uniform across the segment.")
    if low_diversity_signal > 0.58:
        reasons.append("Lexical diversity is lower than expected for the segment length.")
    if repetition_signal > 0.35:
        reasons.append("Repeated phrasing patterns appear more often than usual.")
    if entropy_signal > 0.55:
        reasons.append("Token variation is compressed, which can resemble low-surprisal drafting.")
    if transition_signal > 0.55:
        reasons.append("Transition markers appear at a highly regular cadence.")
    if not reasons:
        reasons.append("The segment shows mixed stylistic signals rather than one strong pattern.")

    return SegmentFeatureSet(
        token_count=token_count,
        sentence_count=len(sentences),
        avg_sentence_length=avg_sentence_length,
        sentence_length_std=sentence_length_std,
        type_token_ratio=ttr,
        repeated_bigram_ratio=repeated_bigram_ratio,
        repeated_sentence_ratio=repeated_sentence_ratio,
        punctuation_density=punctuation_density,
        punctuation_regularization=punctuation_regularization,
        transition_density=transition_density,
        entropy=entropy,
        classifier_probability=classifier_probability,
        stylometric_anomaly_score=stylometric,
        surprisal_signal=surprisal,
        reasons=reasons,
    )


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
