from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from trainer.config import StylometryConfig
from trainer.datasets import SegmentRecord
from trainer.evaluate import binary_classification_metrics
from trainer.features import STYLOMETRY_FEATURE_NAMES, TextFeatureRow, stylometric_feature_vector


@dataclass(slots=True)
class StylometryTrainingOutputs:
    model_path: Path
    feature_names_path: Path
    train_probabilities: dict[str, float]
    validation_probabilities: dict[str, float]
    test_probabilities: dict[str, float]
    validation_metrics: dict[str, float]


def train_stylometry_model(
    *,
    train_segments: list[SegmentRecord],
    validation_segments: list[SegmentRecord],
    test_segments: list[SegmentRecord],
    feature_rows: dict[str, TextFeatureRow],
    stylometry_config: StylometryConfig,
    output_dir: Path,
) -> StylometryTrainingOutputs:
    np = import_module("numpy")
    dump = getattr(import_module("joblib"), "dump")
    LGBMClassifier = getattr(import_module("lightgbm"), "LGBMClassifier")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=stylometry_config.n_estimators,
        learning_rate=stylometry_config.learning_rate,
        num_leaves=stylometry_config.num_leaves,
        class_weight="balanced",
        random_state=42,
    )

    train_matrix = _build_matrix(train_segments, feature_rows)
    validation_matrix = _build_matrix(validation_segments, feature_rows)
    test_matrix = _build_matrix(test_segments, feature_rows)
    train_labels = np.asarray([segment.label for segment in train_segments], dtype=np.int32)

    model.fit(train_matrix, train_labels)

    model_path = output_dir / "model.joblib"
    feature_names_path = output_dir / "feature_names.json"
    dump(model, model_path)
    feature_names_path.write_text(json.dumps(STYLOMETRY_FEATURE_NAMES, indent=2), encoding="utf-8")

    train_probabilities = _predict_probabilities(model, train_segments, train_matrix)
    validation_probabilities = _predict_probabilities(model, validation_segments, validation_matrix)
    test_probabilities = _predict_probabilities(model, test_segments, test_matrix)

    validation_metrics = binary_classification_metrics(
        labels=[segment.label for segment in validation_segments],
        probabilities=[validation_probabilities[segment.segment_id] for segment in validation_segments],
        threshold=0.5,
    )

    return StylometryTrainingOutputs(
        model_path=model_path,
        feature_names_path=feature_names_path,
        train_probabilities=train_probabilities,
        validation_probabilities=validation_probabilities,
        test_probabilities=test_probabilities,
        validation_metrics=validation_metrics,
    )


def _build_matrix(segments: list[SegmentRecord], feature_rows: dict[str, TextFeatureRow]) -> Any:
    np = import_module("numpy")
    return np.asarray(
        [stylometric_feature_vector(feature_rows[segment.segment_id]) for segment in segments],
        dtype=np.float64,
    )


def _predict_probabilities(model: Any, segments: list[SegmentRecord], matrix: Any) -> dict[str, float]:
    probabilities = model.predict_proba(matrix)[:, 1]
    return {
        segment.segment_id: float(probabilities[index])
        for index, segment in enumerate(segments)
    }
