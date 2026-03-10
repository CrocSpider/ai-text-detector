from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.config import TrainingConfig, dump_config


ARTIFACT_FORMAT_VERSION = "1.0"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_config_snapshot(output_dir: Path, config: TrainingConfig) -> Path:
    path = output_dir / "reports" / "config.json"
    write_json(path, dump_config(config))
    return path


def write_evaluation_report(output_dir: Path, report: dict[str, Any]) -> Path:
    path = output_dir / "reports" / "evaluation.json"
    write_json(path, report)
    return path


def write_manifest(
    *,
    output_dir: Path,
    config: TrainingConfig,
    thresholds: dict[str, float],
    evaluation_report_path: Path,
    config_snapshot_path: Path,
    include_stylometry: bool,
    include_meta: bool,
) -> Path:
    manifest: dict[str, Any] = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_version": config.run.run_name,
        "calibration_version": "isotonic-v1" if include_meta else "temperature-lite-v0",
        "feature_version": "stylometry-v1",
        "experiment_name": config.run.experiment_name,
        "base_model_name": config.model.base_model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds,
        "reports": {
            "evaluation": str(evaluation_report_path.relative_to(output_dir)),
            "config": str(config_snapshot_path.relative_to(output_dir)),
        },
        "classifier": {
            "path": "classifier",
            "positive_label": 1,
            "max_length": config.model.max_length,
        },
    }

    if include_stylometry:
        manifest["stylometry"] = {
            "path": "stylometry/model.joblib",
            "feature_names_path": "stylometry/feature_names.json",
        }
    if include_meta:
        meta_section: dict[str, Any] = {
            "path": "meta/model.joblib",
            "feature_names_path": "meta/feature_names.json",
        }
        calibrator_path = output_dir / "meta" / "calibrator.joblib"
        if calibrator_path.exists():
            meta_section["calibrator_path"] = "meta/calibrator.joblib"
        manifest["meta"] = meta_section

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path
