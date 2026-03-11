# AI Text Origin Risk — Technical Reference

This document describes the full model pipeline: how it was trained, what each component computes, how inference works, and why specific engineering decisions were made.

---

## 1. System Overview

The detector operates as a **three-layer ensemble**:

```
Input text
    │
    ├─ [1] Transformer classifier   ←  ModernBERT-base, fine-tuned
    │       segment-level probability (0–1)
    │
    ├─ [2] Stylometry model         ←  LightGBM on hand-crafted features
    │       segment-level anomaly score (0–1)
    │
    └─ [3] Meta-model               ←  sklearn Pipeline + LightGBM
            document-level aggregation → final probability
                    │
            IsotonicRegression calibrator
                    │
                 Final score (0–100)
```

Layers 1 and 2 operate at the **segment level** (chunks of ~1200–1800 characters). Layer 3 aggregates segment signals into a single **document-level** decision. A heuristic fallback formula is used when no trained artifacts are available.

---

## 2. Training Pipeline

### 2.1 Entry point

```bash
python -m trainer train --config configs/train.yaml
```

The full orchestration lives in `services/ml/trainer/cli.py → run_training()`. It executes the following steps in order:

1. Load YAML config → `TrainingConfig`
2. Prepare dataset splits (train / validation / test)
3. Extract text features for every segment
4. Train the transformer classifier
5. Train the stylometry model
6. Generate out-of-fold (OOF) predictions for both models on the training set
7. Build document-level feature rows using OOF predictions
8. Train the meta-model on document-level features
9. Select operating thresholds targeting a 5% FPR budget
10. Write evaluation reports, config snapshot, and `manifest.json`

### 2.2 Configuration (`TrainingConfig`)

Key fields in `services/ml/trainer/config.py`:

| Section | Key | Default | Description |
|---|---|---|---|
| `run` | `output_dir` | — | Where all artifacts are written |
| `run` | `seed` | 42 | Global random seed |
| `model` | `base_model_name` | `answerdotai/ModernBERT-base` | HuggingFace model ID |
| `model` | `epochs` | 2.0 | Fine-tuning epochs |
| `model` | `train_batch_size` | 8 | Per-device batch size |
| `model` | `gradient_accumulation_steps` | 1 | Effective batch = batch × accum |
| `model` | `bf16` | false | Use bfloat16 (set true for A100) |
| `stylometry` | `n_estimators` | 300 | LightGBM trees |
| `meta` | `oof_folds` | 5 | Cross-validation folds for OOF predictions |
| `meta` | `target_max_false_positive_rate` | 0.05 | FPR budget for threshold selection |

The production run used:
- `bf16: true`, `gradient_accumulation_steps: 2`, `train_batch_size: 16`, `epochs: 2`
- `oof_folds: 5`, `classifier_oof: true` (fully leak-free meta training)
- Multi-source dataset covering several domains and AI generators

### 2.3 Dataset preparation

`services/ml/trainer/datasets.py` reads JSONL files and produces three splits. Each JSONL record provides a document with fields `text`, `label` (0 = human, 1 = AI), and optional metadata (`domain`, `language`, `writer_profile`, `attack_type`). Documents are chunked into segments using the shared `chunk_text()` utility (target 900 chars, max 1500 chars for training; 1200/1800 for inference).

Minimum text threshold: 80 characters. Records shorter than this are discarded.

### 2.4 Data leakage prevention (OOF)

A naive setup would train the meta-model using probabilities produced by the classifier and stylometry models on the **same training documents they were fitted on** — this leaks signal because the base models overfit to their training examples. The production run prevents this with **5-fold out-of-fold cross-validation**:

- Training data is split into 5 folds.
- For each fold, a fresh base model is trained on the other 4 folds and generates predictions on the held-out fold.
- The meta-model only ever sees predictions that were produced on data the base model had not seen.
- Validation and test set predictions use the final full-data base models (already out-of-sample by definition).

This is logged in `evaluation.json` as:
```json
"notes": ["Meta-model trained using 5-fold OOF predictions from both base models (leak-free)."]
```

---

## 3. Layer 1 — Transformer Classifier

**Code:** `services/ml/trainer/classifier.py`  
**Artifact:** `classifier/` directory (safetensors weights + tokenizer)

### Architecture

- Base: `answerdotai/ModernBERT-base` (encoder-only transformer)
- Head: 2-class classification (`AutoModelForSequenceClassification`, `num_labels=2`)
- Input: raw segment text, tokenized to max 512 tokens
- Output: `softmax(logits)[:, 1]` — probability the segment is AI-generated

### Training

Fine-tuned with HuggingFace `Trainer`. Best checkpoint selected by `eval_auroc`. The model treats the problem as a binary sequence classification task: given a text segment of up to 512 tokens, assign a probability that it was produced by an LLM.

### Inference

`model_bundle.py → predict_classifier_probabilities(texts: list[str])`:
1. Batch-tokenize all chunks (padding, truncation at 512 tokens)
2. Move to device (CPU or CUDA)
3. Forward pass under `torch.inference_mode()`
4. Return `softmax(logits)[:, 1]` as a list of floats

This is the **most reliable signal** in the ensemble. When `classifier_mean` (trimmed mean over segments) is near 0, the text is very likely human-authored.

---

## 4. Layer 2 — Stylometry Model

**Code:** `services/ml/trainer/stylometry.py`, `libs/text_features/text_features/features.py`  
**Artifact:** `stylometry/model.joblib`, `stylometry/feature_names.json`

### Features (18 total)

All features are computed per segment by `compute_stylometric_features()` in `libs/text_features/text_features/features.py`:

| Feature | Description |
|---|---|
| `token_count` | Word count |
| `sentence_count` | Number of sentences (split on `.!?`) |
| `avg_sentence_length` | Mean words per sentence |
| `sentence_length_std` | Standard deviation of sentence lengths |
| `type_token_ratio` | Unique tokens / total tokens (lexical diversity) |
| `repeated_bigram_ratio` | Bigrams appearing more than once / total bigrams |
| `repeated_sentence_ratio` | Sentences reused verbatim / total sentences |
| `punctuation_density` | Punctuation characters / total characters |
| `punctuation_regularization` | Uniformity of punctuation placement across sentences |
| `transition_density` | Frequency of connective phrases ("however", "therefore", etc.) |
| `entropy` | Shannon entropy over word tokens |
| `stopword_ratio` | Stopwords / total tokens |
| `uniform_sentence_signal` | Normalized score penalizing uniform sentence lengths |
| `low_diversity_signal` | Normalized score penalizing low type-token ratio |
| `repetition_signal` | Normalized score for repeated n-gram patterns |
| `punctuation_signal` | Normalized score for atypical punctuation frequency |
| `transition_signal` | Normalized score for overuse of transition phrases |
| `entropy_signal` | Normalized score for abnormally high lexical entropy |

### Model

LightGBM binary classifier (`objective="binary"`, `class_weight="balanced"`):
- 300 trees, learning rate 0.05, 31 leaves
- Trained on the 18-feature matrix with segment-level binary labels
- Outputs `predict_proba(matrix)[:, 1]`

### Why stylometry?

AI-generated text exhibits characteristic statistical regularities that the transformer may miss on individual segments — unusually high transition-phrase frequency, artificially uniform sentence lengths, and anomalously high lexical entropy. The stylometry model provides an orthogonal signal that is computationally cheap and interpretable.

---

## 5. Layer 3 — Meta-model

**Code:** `services/ml/trainer/meta.py`  
**Artifact:** `meta/model.joblib`, `meta/calibrator.joblib`, `meta/feature_names.json`

### Feature inputs (14 features)

The meta-model receives **document-level aggregates** over all segments, not individual segment text. Feature names are defined in `META_FEATURE_NAMES` in `libs/text_features/text_features/features.py`:

| Feature | Description |
|---|---|
| `classifier_mean` | Trimmed mean (top 10% dropped) of classifier probabilities across segments |
| `classifier_std` | Standard deviation of classifier probabilities |
| `stylometry_mean` | Trimmed mean of stylometry scores |
| `stylometry_std` | Std of stylometry scores |
| `surprisal_mean` | Trimmed mean of surprisal proxy signals |
| `surprisal_std` | Std of surprisal signals |
| `consistency` | Cross-segment feature consistency score (0–1) |
| `quality_penalty` | Extraction quality downweight (0 = good, 1 = poor) |
| `uncertainty_penalty` | `1 - model_agreement` (how much the three signals disagree) |
| `token_count` | Total word tokens in the document |
| `segment_count` | Number of segments the document was split into |
| `language_supported` | 1.0 if English, 0.0 otherwise |
| `mean_segment_tokens` | Mean token count per segment |
| `max_segment_tokens` | Token count of the longest segment |

### Model

sklearn `Pipeline` wrapping LightGBM (or similar), fitted on document-level rows using OOF predictions from the base models. Calibrated with `IsotonicRegression` post-hoc.

### Out-of-distribution (OOD) protection

The meta-model was trained on typical documents of 200–2000 tokens with 2–8 segments. Very long out-of-distribution inputs (e.g. full scientific journal articles, 10k+ tokens, 14 segments) can cause extrapolation errors. Before calling the meta-model, four features are clamped to training-typical ranges:

```python
_OOD_CLAMPS = {
    "token_count":          (0.0, 5000.0),
    "mean_segment_tokens":  (0.0,  600.0),
    "max_segment_tokens":   (0.0,  800.0),
    "segment_count":        (0.0,   20.0),
}
```

This prevents pathologically high meta-model outputs on documents the model was never designed to handle.

---

## 6. Inference Pipeline

**Code:** `services/api/app/services/inference.py → analyze_document()`

### Step-by-step

```
ExtractedDocument
    │
    ├─ detect_language()                          → language code + supported flag
    │
    ├─ chunk_paragraphs()                         → list[TextChunk] (1200/1800 char target/max)
    │
    ├─ extract_segment_features(chunk.text)       → SegmentFeatureSet × N
    │
    ├─ compute_document_consistency()             → consistency (scalar)
    │
    ├─ apply_artifact_predictions()
    │   ├─ predict_classifier_probabilities()     → updates feature_set.classifier_probability
    │   └─ predict_stylometry_probabilities()     → updates feature_set.stylometric_anomaly_score
    │
    ├─ [per segment] score_segment()
    │       = 0.44 × classifier_probability
    │       + 0.26 × stylometric_anomaly_score
    │       + 0.16 × surprisal_signal
    │       + 0.14 × consistency
    │
    ├─ aggregate signals (trimmed mean, top 10% dropped)
    │       classifier_mean, stylometric_mean, surprisal_mean
    │
    ├─ compute raw_score
    │       = 0.40 × classifier_mean
    │       + 0.22 × stylometric_mean
    │       + 0.15 × surprisal_mean
    │       + 0.13 × consistency
    │       - 0.06 × quality_penalty
    │       - 0.04 × uncertainty_penalty
    │
    ├─ calibrate_score(raw_score)   → calibrated_raw  (heuristic path)
    │
    ├─ predict_document_probability(feature_map)  → artifact_probability  (meta-model path)
    │
    └─ BLEND:
            blend_weight = agreement × min(1, classifier_mean / 0.10)
            final = blend_weight × artifact_probability
                  + (1 - blend_weight) × calibrated_raw
```

### Blending logic

The blend weight controls how much the meta-model's output is trusted versus the simpler heuristic calibrated score. It has two factors:

1. **Model agreement** (`agreement = model_agreement(classifier, stylometry, surprisal, consistency)`): how consistently all three signals point in the same direction. When signals strongly disagree, the meta-model is trusted less.

2. **Classifier certainty**: when the transformer (the strongest signal) assigns a segment probability < 10%, the blend weight is reduced proportionally to `classifier_mean / 0.10`. At `classifier_mean = 0.001`, the blend weight is effectively zero — the final score comes almost entirely from the calibrated heuristic.

This prevents the meta-model from producing a high risk score when the primary evidence (transformer output) clearly says "human."

### Heuristic calibration

When no artifact bundle is available, or when the meta-model is largely bypassed:
```python
calibrate_score(raw_score) = 0.50 + (raw_score - 0.50) × 0.88
```
A small compression toward 0.5 that acknowledges model uncertainty.

---

## 7. Document Scoring

### Per-segment risk score

Each `SegmentResult.risk_score` is the segment-level weighted formula (above) scaled to 0–100.

### Document `overall_risk_score`

The blended final probability, scaled to 0–100 and rounded to the nearest integer.

### Risk bands

| Band | Score range |
|---|---|
| `low` | 0–30 |
| `moderate` | 31–60 |
| `high` | 61–100 |

### Confidence level

Derived from `document_confidence_score()`, a weighted combination of:
- **Model agreement** (35%): do the three signals agree?
- **Segment length support** (25%): more tokens = more reliable
- **Extraction quality** (20%): good > fair > poor
- **Language support** (10%): English only for now
- **Segment count** (10%): more segments = stricter evidence

Mapped to `low` / `medium` / `high`.

---

## 8. Production Evaluation (rerun model)

Model: `modernbert-en-v1-rerun-20260310`  
Evaluated on 72,914 documents (24,242 train + 24,242 validation + 24,242 test approximately after split), English only.

| Metric | Document level | Segment level |
|---|---|---|
| AUROC | 0.839 | 0.759 |
| AUPRC | 0.996 | 0.995 |
| Precision | 1.000 | 1.000 |
| Recall | 0.444 | 0.435 |
| FPR @ threshold | **0.0025** | 0.0033 |
| Operating threshold | 0.854 | 0.704 |

FPR of 0.25% means roughly 1 in 400 human documents will be incorrectly flagged at the operating threshold. Recall of ~44% means the model misses roughly half of AI documents. This tradeoff was chosen deliberately — a false accusation of AI authorship is more harmful than a missed detection.

Threshold pair reported in `manifest.json`:
- `review_threshold: 0.7037` — flag for human review
- `elevated_threshold: 0.8537` — high-confidence AI signal

---

## 9. PDF Handling

**Code:** `services/api/app/services/extraction.py`

PDF text is extracted with `pypdf` (plain mode, not layout mode). Layout mode was evaluated but produces 3× more text due to whitespace padding in two-column formats, breaking paragraph detection and the references-section heuristic.

Post-extraction cleaning:

1. **Per-page cleaning** (`_clean_pdf_page`): strips DOI lines, ISSN, copyright notices, bare page numbers, short all-uppercase running headers (journal name, "ARTICLE"), and lines where isolated-letter spacing indicates two-column seam artifacts (`I z u m i S h i m a d a` → dropped).

2. **Document-level stripping** (`_strip_pdf_boilerplate`):
   - Locates the last occurrence of a References/Bibliography/Works Cited heading in the **final 50% of the document** and truncates there. The references section produces highly repetitive citation text that inflates stylometric anomaly scores.
   - Identifies lines repeated verbatim across 3+ pages (running headers/footers) and removes them.

Without these steps, a clean scientific paper can score 97/100 due to reference-list repetition and formatting artifacts.

---

## 10. Artifact Layout

After training, artifacts are written to the configured `output_dir`:

```
<output_dir>/
│
├─ manifest.json                 ← paths to all artifacts + thresholds + version strings
│
├─ classifier/
│   ├─ config.json               ← HuggingFace model config
│   ├─ model.safetensors         ← fine-tuned weights
│   ├─ tokenizer.json
│   ├─ tokenizer_config.json
│   ├─ special_tokens_map.json
│   └─ checkpoints/              ← intermediate checkpoints (training only)
│
├─ stylometry/
│   ├─ model.joblib              ← serialised LightGBM classifier
│   └─ feature_names.json       ← ordered list of 18 feature names
│
├─ meta/
│   ├─ model.joblib              ← serialised sklearn Pipeline (meta-model)
│   ├─ calibrator.joblib        ← serialised IsotonicRegression
│   └─ feature_names.json       ← ordered list of 14 document-level feature names
│
└─ reports/
    ├─ evaluation.json           ← AUROC, FPR, precision/recall, slice breakdowns
    └─ config_snapshot.json      ← exact config used for this run
```

The API loads all components lazily on first request via `get_artifact_bundle()` (LRU-cached). If any component fails to load, it logs a warning and the pipeline falls back to the heuristic path for that component.

---

## 11. Known Limitations

- **Recall is ~44% at the operating threshold.** The model is calibrated for precision (low FPR), not recall. Many AI-generated documents will score below the threshold.
- **Non-English text.** The transformer was fine-tuned on English data. Other languages fall back to the heuristic path with reduced confidence.
- **Adversarial paraphrasing.** Text that has been lightly edited by a human after AI generation is harder to detect. "Attack type" slices are tracked in evaluation but performance varies.
- **Very short text.** Fewer than ~150 tokens gives low confidence scores.
- **Long scientific documents.** OOD clamping and blend weight reduction mitigate but do not eliminate meta-model extrapolation errors on 10k+ token documents.
- **PDF extraction quality.** Multi-column PDFs, scanned documents, and papers with heavy mathematical notation may produce noisy extracted text that reduces reliability.
