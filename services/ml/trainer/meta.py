from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from trainer.datasets import PreparedSplit
from trainer.features import TextFeatureRow
from text_features.features import (
    META_FEATURE_NAMES,
    compute_document_consistency,
    model_agreement,
    quality_penalty_for,
)
from text_features.text import clamp, mean, safe_std

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MetaTrainingOutputs:
    model_path: Path
    calibrator_path: Path | None
    feature_names_path: Path
    train_probabilities: dict[str, float]
    validation_probabilities: dict[str, float]
    test_probabilities: dict[str, float]
    threshold_document_ids: list[str] | None = None


def build_document_feature_rows(
    *,
    split: PreparedSplit,
    feature_rows: dict[str, TextFeatureRow],
    classifier_probabilities: dict[str, float],
    stylometry_probabilities: dict[str, float],
) -> list[dict[str, Any]]:
    segments_by_document: dict[str, list[Any]] = {}
    for segment in split.segments:
        segments_by_document.setdefault(segment.document_id, []).append(segment)

    rows: list[dict[str, Any]] = []
    for document in split.documents:
        segments = segments_by_document.get(document.document_id, [])
        if not segments:
            continue

        document_feature_rows = [feature_rows[segment.segment_id] for segment in segments]
        classifier_values = [
            classifier_probabilities.get(segment.segment_id, feature_rows[segment.segment_id].heuristic_classifier_probability)
            for segment in segments
        ]
        stylometry_values = [
            stylometry_probabilities.get(segment.segment_id, feature_rows[segment.segment_id].heuristic_stylometric_score)
            for segment in segments
        ]
        surprisal_values = [feature_rows[segment.segment_id].surprisal_signal for segment in segments]
        segment_tokens = [float(feature_rows[segment.segment_id].token_count) for segment in segments]

        consistency = compute_document_consistency(document_feature_rows)
        quality_penalty = quality_penalty_for(
            extraction_quality=document.metadata.get("extraction_quality", "good"),
            language_supported=document.metadata.get("language", "unknown") == "en",
            text=document.text,
        )
        agreement = model_agreement(
            mean(classifier_values),
            mean(stylometry_values),
            mean(surprisal_values),
            consistency,
        )
        uncertainty_penalty = clamp(1.0 - agreement)

        row = {
            "document_id": document.document_id,
            "label": document.label,
            "classifier_mean": mean(classifier_values),
            "classifier_std": safe_std(classifier_values),
            "stylometry_mean": mean(stylometry_values),
            "stylometry_std": safe_std(stylometry_values),
            "surprisal_mean": mean(surprisal_values),
            "surprisal_std": safe_std(surprisal_values),
            "consistency": consistency,
            "quality_penalty": quality_penalty,
            "uncertainty_penalty": uncertainty_penalty,
            "token_count": float(sum(segment_tokens)),
            "segment_count": float(len(segments)),
            "language_supported": 1.0 if document.metadata.get("language", "unknown") == "en" else 0.0,
            "mean_segment_tokens": mean(segment_tokens),
            "max_segment_tokens": max(segment_tokens) if segment_tokens else 0.0,
            **document.metadata,
        }
        rows.append(row)
    return rows


def train_meta_model(
    *,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    output_dir: Path,
) -> MetaTrainingOutputs:
    linear_model_module = import_module("sklearn.linear_model")
    isotonic_module = import_module("sklearn.isotonic")
    pipeline_module = import_module("sklearn.pipeline")
    preprocessing_module = import_module("sklearn.preprocessing")
    dump = getattr(import_module("joblib"), "dump")
    LogisticRegression = getattr(linear_model_module, "LogisticRegression")
    IsotonicRegression = getattr(isotonic_module, "IsotonicRegression")
    Pipeline = getattr(pipeline_module, "Pipeline")
    StandardScaler = getattr(preprocessing_module, "StandardScaler")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a Pipeline with StandardScaler so that L2 regularisation treats
    # all meta-features equally regardless of their native scale.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    train_matrix = _build_matrix(train_rows)
    validation_matrix = _build_matrix(validation_rows)
    test_matrix = _build_matrix(test_rows)
    train_labels = [int(row["label"]) for row in train_rows]
    validation_labels = [int(row["label"]) for row in validation_rows]

    model.fit(train_matrix, train_labels)

    # Split validation into two stratified halves:
    #   - calibration_half: fit the isotonic calibrator
    #   - threshold_half:   choose operating thresholds (never seen by calibrator)
    # This prevents over-optimistic threshold selection.
    calibration_indices, threshold_indices = _stratified_half_split(validation_labels)

    calibrator = None
    validation_raw = model.predict_proba(validation_matrix)[:, 1]
    if len(set(validation_labels)) > 1 and len(calibration_indices) >= 4:
        calibration_raw = [validation_raw[i] for i in calibration_indices]
        calibration_labels = [validation_labels[i] for i in calibration_indices]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(calibration_raw, calibration_labels)
        logger.info(
            "Isotonic calibrator fitted on %d/%d validation samples; "
            "%d held out for threshold selection",
            len(calibration_indices), len(validation_labels), len(threshold_indices),
        )

    model_path = output_dir / "model.joblib"
    feature_names_path = output_dir / "feature_names.json"
    calibrator_path = output_dir / "calibrator.joblib"
    dump(model, model_path)
    feature_names_path.write_text(json.dumps(META_FEATURE_NAMES, indent=2), encoding="utf-8")
    if calibrator is not None:
        dump(calibrator, calibrator_path)
    else:
        calibrator_path = None

    train_probabilities = _predict_document_probabilities(model, calibrator, train_rows)
    validation_probabilities = _predict_document_probabilities(model, calibrator, validation_rows)
    test_probabilities = _predict_document_probabilities(model, calibrator, test_rows)

    # Expose the threshold-half indices so the caller uses only the
    # uncontaminated portion for threshold selection.
    threshold_document_ids = [validation_rows[i]["document_id"] for i in threshold_indices]

    return MetaTrainingOutputs(
        model_path=model_path,
        calibrator_path=calibrator_path,
        feature_names_path=feature_names_path,
        train_probabilities=train_probabilities,
        validation_probabilities=validation_probabilities,
        test_probabilities=test_probabilities,
        threshold_document_ids=threshold_document_ids,
    )


def heuristic_document_probability(row: dict[str, Any]) -> float:
    raw = clamp(
        0.40 * float(row["classifier_mean"])
        + 0.22 * float(row["stylometry_mean"])
        + 0.15 * float(row["surprisal_mean"])
        + 0.13 * float(row["consistency"])
        - 0.06 * float(row["quality_penalty"])
        - 0.04 * float(row["uncertainty_penalty"])
    )
    return clamp(0.50 + ((raw - 0.50) * 0.88))


def _predict_document_probabilities(model: Any, calibrator: Any | None, rows: list[dict[str, Any]]) -> dict[str, float]:
    matrix = _build_matrix(rows)
    raw_probabilities = model.predict_proba(matrix)[:, 1]
    if calibrator is not None:
        calibrated = calibrator.predict(raw_probabilities)
    else:
        calibrated = raw_probabilities
    return {
        row["document_id"]: float(calibrated[index])
        for index, row in enumerate(rows)
    }


def _build_matrix(rows: list[dict[str, Any]]) -> list[list[float]]:
    return [[float(row.get(name, 0.0)) for name in META_FEATURE_NAMES] for row in rows]


def _stratified_half_split(labels: list[int], seed: int = 99) -> tuple[list[int], list[int]]:
    """Split indices into two halves preserving class proportions.

    Returns (calibration_indices, threshold_indices).
    """
    import random
    rng = random.Random(seed)

    class_indices: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        class_indices.setdefault(label, []).append(i)

    calibration: list[int] = []
    threshold: list[int] = []
    for _cls, indices in class_indices.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        calibration.extend(shuffled[:mid])
        threshold.extend(shuffled[mid:])

    return calibration, threshold
