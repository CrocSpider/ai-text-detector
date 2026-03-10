# Training Service

This package prepares and trains the real ML stack for the AI Text Origin Risk Analyzer on a larger machine or GPU-enabled Kubernetes job.

## What it trains

- a transformer chunk classifier
- a stylometric classifier
- a document-level meta-model with calibration
- evaluation reports and an artifact manifest that the API can load

## Expected input schema

Each record should contain at least:

```json
{"document_id":"doc-1","text":"...","label":0}
```

Recommended metadata fields:

- `language`
- `domain`
- `writer_profile`
- `attack_type`
- `source_dataset`
- `extraction_quality`

## Example config

See `configs/train.example.yaml`.

For heterogeneous sources with per-dataset field mappings, see `configs/train.multisource.example.yaml`.

Each source entry can be either a plain path/URI string or a mapping with dataset-specific overrides:

```yaml
- uri: hf://liamdugan/raid?subset=raid&split=train
  text_key: generation
  label_key: model
  label_strategy: non_human_is_ai
  domain_key: domain
  attack_type_key: attack
  writer_profile_key: model
  metadata_defaults:
    language: en
    source_dataset: RAID
```

Supported source mapping fields:

- `uri`
- `text_key`
- `label_key`
- `document_id_key`
- `language_key`
- `domain_key`
- `writer_profile_key`
- `attack_type_key`
- `source_dataset_key`
- `extraction_quality_key`
- `label_strategy`
- `label_map`
- `metadata_defaults`
- `include`
- `exclude`

## Run locally on a strong machine

```bash
cd services/ml
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m trainer.cli train --config configs/train.example.yaml
```

## Build the training image

```bash
docker build -t eu.gcr.io/halfpipe-io/we-asia/ai-text-detector-trainer:1.3 .
```

## Prepare public seed data

The trainer can materialize a starter dataset directly from HC3 and write the three JSONL files the training job expects.

```bash
.venv/bin/python -m trainer.cli prepare-hc3 --output-dir /tmp/ai-text-data --subset all --max-per-class 12000
```

This writes:

- `/tmp/ai-text-data/train.jsonl`
- `/tmp/ai-text-data/validation.jsonl`
- `/tmp/ai-text-data/test.jsonl`

Each row includes at least `document_id`, `text`, and `label`, plus metadata like `domain` and `source_dataset`.

## Artifact layout

The training run writes:

- `classifier/` transformer weights and tokenizer
- `stylometry/model.joblib`
- `stylometry/feature_names.json`
- `meta/model.joblib`
- `meta/calibrator.joblib`
- `meta/feature_names.json`
- `reports/evaluation.json`
- `manifest.json`

The API can consume these artifacts by setting `API_ENABLE_ARTIFACT_MODELS=true` and `API_ML_ARTIFACT_DIR=/path/to/run`.
