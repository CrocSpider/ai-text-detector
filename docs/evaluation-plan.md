# Evaluation Plan

## Core metrics

- AUROC
- AUPRC
- F1
- precision and recall
- false positive rate and false negative rate
- expected calibration error

## Required slices

- short, medium, and long texts
- domain slices such as academic, business, technical, and informal
- native and non-native English where ethically and legally supportable
- clean AI, human, AI-edited human, human-edited AI, and paraphrased AI
- poor-extraction and OCR-noise cases

## Operational guidance

- Keep low-risk scores advisory only.
- Treat moderate results as a human-review band.
- Require medium or high confidence before surfacing elevated-risk guidance.
- Prefer inconclusive outcomes over aggressive calls when extraction quality or model agreement is weak.
