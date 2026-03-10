from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from trainer.artifacts import write_config_snapshot, write_evaluation_report, write_manifest
from trainer.classifier import train_transformer_classifier, generate_oof_probabilities as classifier_oof_probabilities
from trainer.config import TrainingConfig, dump_config, load_training_config
from trainer.datasets import PreparedSplit, prepare_datasets
from trainer.evaluate import (
    binary_classification_metrics,
    build_slice_reports,
    choose_threshold_with_target_fpr,
    probability_band_thresholds,
)
from trainer.features import TextFeatureRow, extract_text_features
from trainer.meta import build_document_feature_rows, heuristic_document_probability, train_meta_model
from trainer.public_datasets import SplitRatio, prepare_hc3_dataset
from trainer.stylometry import train_stylometry_model, generate_oof_probabilities as stylometry_oof_probabilities
from trainer.text_utils import clamp


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AI text origin risk ensemble")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train classifier, stylometry, and meta artifacts")
    train_parser.add_argument("--config", required=True, help="Path to a YAML training config")

    prepare_hc3_parser = subparsers.add_parser("prepare-hc3", help="Prepare HC3 JSONL train/validation/test splits")
    prepare_hc3_parser.add_argument("--output-dir", required=True, help="Directory where train.jsonl, validation.jsonl, and test.jsonl will be written")
    prepare_hc3_parser.add_argument("--subset", default="all", help="HC3 subset to use, for example all, finance, medicine, open_qa, reddit_eli5, wiki_csai")
    prepare_hc3_parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    prepare_hc3_parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap on human and AI examples per class")
    prepare_hc3_parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    prepare_hc3_parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio")
    prepare_hc3_parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")

    describe_parser = subparsers.add_parser("describe-config", help="Print a parsed training config")
    describe_parser.add_argument("--config", required=True, help="Path to a YAML training config")

    args = parser.parse_args()

    if args.command == "train":
        run_training(load_training_config(args.config))
        return
    if args.command == "prepare-hc3":
        ratios = SplitRatio(train=args.train_ratio, validation=args.validation_ratio, test=args.test_ratio)
        if round(ratios.train + ratios.validation + ratios.test, 6) != 1.0:
            raise ValueError("train-ratio + validation-ratio + test-ratio must equal 1.0")
        summary = prepare_hc3_dataset(
            output_dir=args.output_dir,
            seed=args.seed,
            subset=args.subset,
            max_per_class=args.max_per_class,
            split_ratio=ratios,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.command == "describe-config":
        config = load_training_config(args.config)
        print(json.dumps(dump_config(config), indent=2))


def run_training(config: TrainingConfig) -> None:
    output_dir = Path(config.run.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing training artifacts to %s", output_dir)

    config_snapshot_path = write_config_snapshot(output_dir, config)
    splits = prepare_datasets(config.dataset, seed=config.run.seed)
    for split_name, split in splits.items():
        if not split.documents or not split.segments:
            raise ValueError(f"Split '{split_name}' is empty after preprocessing. Check dataset paths and minimum text length settings.")
    feature_rows = {name: _build_feature_lookup(split) for name, split in splits.items()}

    classifier_outputs = train_transformer_classifier(
        train_segments=splits["train"].segments,
        validation_segments=splits["validation"].segments,
        test_segments=splits["test"].segments,
        model_config=config.model,
        run_config=config.run,
        output_dir=output_dir / "classifier",
    )

    if config.stylometry.enabled:
        stylometry_outputs = train_stylometry_model(
            train_segments=splits["train"].segments,
            validation_segments=splits["validation"].segments,
            test_segments=splits["test"].segments,
            feature_rows={**feature_rows["train"], **feature_rows["validation"], **feature_rows["test"]},
            stylometry_config=config.stylometry,
            output_dir=output_dir / "stylometry",
        )
        stylometry_train = stylometry_outputs.train_probabilities
        stylometry_validation = stylometry_outputs.validation_probabilities
        stylometry_test = stylometry_outputs.test_probabilities
    else:
        stylometry_outputs = None
        stylometry_train = _heuristic_stylometry_probabilities(splits["train"], feature_rows["train"])
        stylometry_validation = _heuristic_stylometry_probabilities(splits["validation"], feature_rows["validation"])
        stylometry_test = _heuristic_stylometry_probabilities(splits["test"], feature_rows["test"])

    # --- Out-of-fold predictions for meta-model training ---
    # Use OOF predictions for training set to avoid data leakage in the meta-model.
    # Validation/test predictions from the full models are already out-of-sample.
    oof_folds = config.meta.oof_folds if config.meta.enabled else 0
    if oof_folds >= 2 and config.stylometry.enabled:
        logger.info("Generating %d-fold OOF stylometry predictions for meta-model training", oof_folds)
        stylometry_train_for_meta = stylometry_oof_probabilities(
            train_segments=splits["train"].segments,
            feature_rows=feature_rows["train"],
            stylometry_config=config.stylometry,
            n_folds=oof_folds,
            seed=config.run.seed,
        )
    else:
        stylometry_train_for_meta = stylometry_train

    if oof_folds >= 2 and config.meta.classifier_oof:
        logger.info("Generating %d-fold OOF classifier predictions for meta-model training (this is expensive)", oof_folds)
        classifier_train_for_meta = classifier_oof_probabilities(
            train_segments=splits["train"].segments,
            model_config=config.model,
            run_config=config.run,
            n_folds=oof_folds,
            output_dir=output_dir / "classifier",
        )
    else:
        classifier_train_for_meta = classifier_outputs.train_probabilities
        if oof_folds >= 2:
            logger.warning(
                "Classifier OOF is disabled (meta.classifier_oof=false). "
                "Meta-model will use in-sample classifier predictions for training — "
                "this leaks signal. Set meta.classifier_oof=true (the default) for "
                "fully leak-free meta training."
            )

    train_document_rows = build_document_feature_rows(
        split=splits["train"],
        feature_rows=feature_rows["train"],
        classifier_probabilities=classifier_train_for_meta,
        stylometry_probabilities=stylometry_train_for_meta,
    )
    validation_document_rows = build_document_feature_rows(
        split=splits["validation"],
        feature_rows=feature_rows["validation"],
        classifier_probabilities=classifier_outputs.validation_probabilities,
        stylometry_probabilities=stylometry_validation,
    )
    test_document_rows = build_document_feature_rows(
        split=splits["test"],
        feature_rows=feature_rows["test"],
        classifier_probabilities=classifier_outputs.test_probabilities,
        stylometry_probabilities=stylometry_test,
    )

    if config.meta.enabled:
        meta_outputs = train_meta_model(
            train_rows=train_document_rows,
            validation_rows=validation_document_rows,
            test_rows=test_document_rows,
            output_dir=output_dir / "meta",
        )
        validation_document_probabilities = meta_outputs.validation_probabilities
        test_document_probabilities = meta_outputs.test_probabilities
        # Use only the threshold-half of validation (not seen by the calibrator)
        # for threshold selection to avoid over-optimistic operating points.
        threshold_ids = set(meta_outputs.threshold_document_ids or [])
    else:
        meta_outputs = None
        validation_document_probabilities = {
            row["document_id"]: heuristic_document_probability(row) for row in validation_document_rows
        }
        test_document_probabilities = {
            row["document_id"]: heuristic_document_probability(row) for row in test_document_rows
        }
        threshold_ids = None

    # When the meta-model provides a clean split, only use the threshold half
    # for threshold selection; otherwise fall back to the full validation set.
    if threshold_ids:
        threshold_rows = [row for row in validation_document_rows if row["document_id"] in threshold_ids]
    else:
        threshold_rows = validation_document_rows

    elevated_threshold = choose_threshold_with_target_fpr(
        labels=[int(row["label"]) for row in threshold_rows],
        probabilities=[validation_document_probabilities[row["document_id"]] for row in threshold_rows],
        max_false_positive_rate=config.meta.target_max_false_positive_rate,
    )
    thresholds = probability_band_thresholds(elevated_threshold)

    # Compute validation metrics on the threshold-half only so they are not
    # contaminated by calibrator-seen samples.  Falls back to full validation
    # when no clean split exists (e.g. meta-model disabled).
    validation_metric_rows = threshold_rows if threshold_ids else validation_document_rows
    validation_document_metrics = binary_classification_metrics(
        labels=[int(row["label"]) for row in validation_metric_rows],
        probabilities=[validation_document_probabilities[row["document_id"]] for row in validation_metric_rows],
        threshold=elevated_threshold,
    )
    test_document_metrics = binary_classification_metrics(
        labels=[int(row["label"]) for row in test_document_rows],
        probabilities=[test_document_probabilities[row["document_id"]] for row in test_document_rows],
        threshold=elevated_threshold,
    )

    segment_probabilities = _segment_probabilities(
        split=splits["test"],
        feature_rows=feature_rows["test"],
        classifier_probabilities=classifier_outputs.test_probabilities,
        stylometry_probabilities=stylometry_test,
    )
    test_segment_metrics = binary_classification_metrics(
        labels=[segment.label for segment in splits["test"].segments],
        probabilities=[segment_probabilities[segment.segment_id] for segment in splits["test"].segments],
        threshold=thresholds["review_threshold"],
    )

    slice_rows = []
    for row in test_document_rows:
        enriched = dict(row)
        enriched["probability"] = test_document_probabilities[row["document_id"]]
        slice_rows.append(enriched)

    evaluation_report = {
        "document_metrics": test_document_metrics,
        "validation_document_metrics": validation_document_metrics,
        "segment_metrics": test_segment_metrics,
        "thresholds": thresholds,
        "slices": build_slice_reports(
            slice_rows,
            slice_keys=config.dataset.slice_keys,
            probability_key="probability",
            threshold=elevated_threshold,
            min_examples=config.evaluation.min_slice_examples,
        ),
        "notes": _meta_training_notes(config),
    }

    evaluation_report_path = write_evaluation_report(output_dir, evaluation_report)
    manifest_path = write_manifest(
        output_dir=output_dir,
        config=config,
        thresholds=thresholds,
        evaluation_report_path=evaluation_report_path,
        config_snapshot_path=config_snapshot_path,
        include_stylometry=config.stylometry.enabled,
        include_meta=config.meta.enabled,
    )

    summary = {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "document_auroc": round(float(test_document_metrics["auroc"]), 4),
        "document_fpr": round(float(test_document_metrics["false_positive_rate"]), 4),
        "segment_auroc": round(float(test_segment_metrics["auroc"]), 4),
        "review_threshold": thresholds["review_threshold"],
        "elevated_threshold": thresholds["elevated_threshold"],
    }
    print(json.dumps(summary, indent=2))


def _build_feature_lookup(split: PreparedSplit) -> dict[str, TextFeatureRow]:
    return {
        segment.segment_id: extract_text_features(segment.text)
        for segment in split.segments
    }


def _heuristic_stylometry_probabilities(split: PreparedSplit, feature_rows: dict[str, TextFeatureRow]) -> dict[str, float]:
    return {
        segment.segment_id: feature_rows[segment.segment_id].heuristic_stylometric_score
        for segment in split.segments
    }


def _segment_probabilities(
    *,
    split: PreparedSplit,
    feature_rows: dict[str, TextFeatureRow],
    classifier_probabilities: dict[str, float],
    stylometry_probabilities: dict[str, float],
) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for segment in split.segments:
        feature_row = feature_rows[segment.segment_id]
        classifier_score = classifier_probabilities.get(segment.segment_id, feature_row.heuristic_classifier_probability)
        stylometry_score = stylometry_probabilities.get(segment.segment_id, feature_row.heuristic_stylometric_score)
        probabilities[segment.segment_id] = clamp(
            0.65 * classifier_score + 0.25 * stylometry_score + 0.10 * feature_row.surprisal_signal
        )
    return probabilities


def _meta_training_notes(config: TrainingConfig) -> list[str]:
    notes: list[str] = []
    oof_folds = config.meta.oof_folds if config.meta.enabled else 0
    if oof_folds >= 2:
        if config.meta.classifier_oof:
            notes.append(f"Meta-model trained using {oof_folds}-fold OOF predictions from both base models (leak-free).")
        else:
            notes.append(
                f"Meta-model trained using {oof_folds}-fold OOF stylometry predictions. "
                "Classifier predictions are in-sample (set meta.classifier_oof=true for full OOF)."
            )
    else:
        notes.append(
            "The meta-model is trained on train documents using in-sample base-model predictions. "
            "Set meta.oof_folds >= 2 for out-of-fold meta training."
        )
    return notes


if __name__ == "__main__":
    main()
