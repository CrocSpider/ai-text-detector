from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RunConfig:
    output_dir: str
    experiment_name: str = "ai-text-origin-risk"
    run_name: str = "baseline"
    seed: int = 42


@dataclass(slots=True)
class DatasetConfig:
    train_sources: list[Any]
    validation_sources: list[Any]
    test_sources: list[Any]
    text_key: str = "text"
    label_key: str = "label"
    document_id_key: str = "document_id"
    language_key: str = "language"
    domain_key: str = "domain"
    writer_profile_key: str = "writer_profile"
    attack_type_key: str = "attack_type"
    source_dataset_key: str = "source_dataset"
    extraction_quality_key: str = "extraction_quality"
    min_text_chars: int = 80
    target_chunk_chars: int = 900
    max_chunk_chars: int = 1500
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    slice_keys: list[str] = field(
        default_factory=lambda: [
            "language",
            "domain",
            "writer_profile",
            "attack_type",
            "source_dataset",
            "text_length_bucket",
        ]
    )


@dataclass(slots=True)
class ModelConfig:
    base_model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    epochs: float = 2.0
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 25
    save_total_limit: int = 2


@dataclass(slots=True)
class StylometryConfig:
    enabled: bool = True
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31


@dataclass(slots=True)
class MetaConfig:
    enabled: bool = True
    target_max_false_positive_rate: float = 0.05


@dataclass(slots=True)
class EvaluationConfig:
    positive_threshold: float = 0.5
    min_slice_examples: int = 20


@dataclass(slots=True)
class TrainingConfig:
    run: RunConfig
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    stylometry: StylometryConfig = field(default_factory=StylometryConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_training_config(config_path: str | Path) -> TrainingConfig:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return TrainingConfig(
        run=RunConfig(**_section(payload, "run", required=True)),
        dataset=DatasetConfig(**_section(payload, "dataset", required=True)),
        model=ModelConfig(**_section(payload, "model")),
        stylometry=StylometryConfig(**_section(payload, "stylometry")),
        meta=MetaConfig(**_section(payload, "meta")),
        evaluation=EvaluationConfig(**_section(payload, "evaluation")),
    )


def dump_config(config: TrainingConfig) -> dict[str, Any]:
    return asdict(config)


def _section(payload: dict[str, Any], key: str, required: bool = False) -> dict[str, Any]:
    section = payload.get(key, {})
    if required and not section:
        raise ValueError(f"Missing required config section: {key}")
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return section
