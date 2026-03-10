from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import inspect
from pathlib import Path
from typing import Any

from trainer.config import ModelConfig, RunConfig
from trainer.datasets import SegmentRecord
from trainer.evaluate import binary_classification_metrics


@dataclass(slots=True)
class ClassifierTrainingOutputs:
    model_dir: Path
    train_probabilities: dict[str, float]
    validation_probabilities: dict[str, float]
    test_probabilities: dict[str, float]
    validation_metrics: dict[str, float]


def train_transformer_classifier(
    *,
    train_segments: list[SegmentRecord],
    validation_segments: list[SegmentRecord],
    test_segments: list[SegmentRecord],
    model_config: ModelConfig,
    run_config: RunConfig,
    output_dir: Path,
) -> ClassifierTrainingOutputs:
    datasets_module = import_module("datasets")
    transformers_module = import_module("transformers")
    Dataset = getattr(datasets_module, "Dataset")
    AutoModelForSequenceClassification = getattr(transformers_module, "AutoModelForSequenceClassification")
    AutoTokenizer = getattr(transformers_module, "AutoTokenizer")
    DataCollatorWithPadding = getattr(transformers_module, "DataCollatorWithPadding")
    Trainer = getattr(transformers_module, "Trainer")
    TrainingArguments = getattr(transformers_module, "TrainingArguments")
    set_seed = getattr(transformers_module, "set_seed")

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(run_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_config.base_model_name, num_labels=2)

    train_dataset = Dataset.from_list(_records_to_examples(train_segments))
    validation_dataset = Dataset.from_list(_records_to_examples(validation_segments))
    test_dataset = Dataset.from_list(_records_to_examples(test_segments))

    tokenized_train = train_dataset.map(lambda batch: _tokenize_batch(batch, tokenizer, model_config.max_length), batched=True)
    tokenized_validation = validation_dataset.map(lambda batch: _tokenize_batch(batch, tokenizer, model_config.max_length), batched=True)
    tokenized_test = test_dataset.map(lambda batch: _tokenize_batch(batch, tokenizer, model_config.max_length), batched=True)

    training_argument_params = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "overwrite_output_dir": True,
        "num_train_epochs": model_config.epochs,
        "per_device_train_batch_size": model_config.train_batch_size,
        "per_device_eval_batch_size": model_config.eval_batch_size,
        "learning_rate": model_config.learning_rate,
        "weight_decay": model_config.weight_decay,
        "warmup_ratio": model_config.warmup_ratio,
        "gradient_accumulation_steps": model_config.gradient_accumulation_steps,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": model_config.logging_steps,
        "save_total_limit": model_config.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_auroc",
        "greater_is_better": True,
        "report_to": [],
        "fp16": model_config.fp16,
        "bf16": model_config.bf16,
        "seed": run_config.seed,
    }
    if "evaluation_strategy" in training_argument_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in training_argument_params:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        raise RuntimeError("TrainingArguments no longer exposes an evaluation strategy parameter")

    training_args = TrainingArguments(**training_kwargs)

    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_validation,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_classifier_metrics,
    }
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    train_probabilities = _predict_probabilities(trainer, tokenized_train, train_segments)
    validation_probabilities = _predict_probabilities(trainer, tokenized_validation, validation_segments)
    test_probabilities = _predict_probabilities(trainer, tokenized_test, test_segments)

    validation_metrics = binary_classification_metrics(
        labels=[segment.label for segment in validation_segments],
        probabilities=[validation_probabilities[segment.segment_id] for segment in validation_segments],
        threshold=0.5,
    )

    return ClassifierTrainingOutputs(
        model_dir=output_dir,
        train_probabilities=train_probabilities,
        validation_probabilities=validation_probabilities,
        test_probabilities=test_probabilities,
        validation_metrics=validation_metrics,
    )


def compute_classifier_metrics(eval_prediction: Any) -> dict[str, float]:
    logits = eval_prediction.predictions if hasattr(eval_prediction, "predictions") else eval_prediction[0]
    labels = eval_prediction.label_ids if hasattr(eval_prediction, "label_ids") else eval_prediction[1]
    probabilities = _positive_class_probabilities(logits)
    metrics = binary_classification_metrics(labels.tolist(), probabilities.tolist(), threshold=0.5)
    return {
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "f1": metrics["f1"],
    }


def _predict_probabilities(trainer: Any, dataset: Any, segments: list[SegmentRecord]) -> dict[str, float]:
    predictions = trainer.predict(dataset)
    probabilities = _positive_class_probabilities(predictions.predictions)
    return {
        segment.segment_id: float(probabilities[index])
        for index, segment in enumerate(segments)
    }


def _positive_class_probabilities(logits: Any) -> Any:
    np = import_module("numpy")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def _records_to_examples(segments: list[SegmentRecord]) -> list[dict[str, int | str]]:
    return [{"text": segment.text, "label": segment.label} for segment in segments]


def _tokenize_batch(batch: dict[str, list[str]], tokenizer: Any, max_length: int) -> dict[str, Any]:
    return tokenizer(batch["text"], truncation=True, max_length=max_length)
