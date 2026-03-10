# Dataset Strategy

## Seed datasets

- `HC3` for human vs ChatGPT responses across multiple domains.
- `TuringBench` for broad supervised English detection baselines across multiple generators.
- `GPT-wiki-intro` for simple paired human and generated text.
- `xsum_hallucination_annotations` or similar clean human news/article corpora for additional human-only anchors.

## Stress-test datasets

- `RAID` for adversarial and robustness testing.
- `PADBen` for paraphrase laundering and rewrite attacks.
- `BEEMO` for human-edited AI and AI-edited human scenarios.
- `SemEval 2024 Task 8` for multilingual and change-point evaluation.
- `AuthorAwareDetectionBench` for fairness slices involving non-native English writing.

## Recommended mix for the first serious training run

- `HC3` as a compact seed set and sanity-check baseline.
- `TuringBench` as the main broad supervised training source.
- `RAID` primarily as validation/test and robustness stress data, not as the only training source.
- human-only anchor corpora from your target domains to control false positives.
- internally generated edit pipelines for AI-edited human, human-edited AI, typo noise, and paraphrase attacks.

## Notes on the sources you shared

- `vohai2003/TuringBench-tokenized` is usable because it still exposes a text field named `Generation`; do not train from `input_ids` alone.
- `liamdugan/raid` is excellent for adversarial evaluation because it exposes `generation`, `model`, `attack`, and `domain` metadata.
- `RAID` should influence thresholds and robustness reporting heavily because it captures jailbreaks, rewrites, and prompt-based evasions.
- `TuringBench` and `HC3` can leak model-family artifacts if over-weighted, so they should be balanced with harder edited and out-of-domain slices.

## Data policy

- Keep train, validation, and final deployment evaluation fully separated.
- Hold out entire domains, prompts, and model families to reduce leakage.
- Label mixed-provenance text explicitly instead of forcing clean binary labels.
- Do not train on sensitive customer uploads without explicit policy support and consent.
