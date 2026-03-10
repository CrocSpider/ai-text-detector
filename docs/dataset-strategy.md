# Dataset Strategy

## Seed datasets

- `HC3` for human vs ChatGPT responses across multiple domains.
- `TuringBench` for broad supervised English detection baselines.
- `GPT-wiki-intro` for simple paired human and generated text.

## Stress-test datasets

- `RAID` for adversarial and robustness testing.
- `PADBen` for paraphrase laundering and rewrite attacks.
- `BEEMO` for human-edited AI and AI-edited human scenarios.
- `SemEval 2024 Task 8` for multilingual and change-point evaluation.
- `AuthorAwareDetectionBench` for fairness slices involving non-native English writing.

## Data policy

- Keep train, validation, and final deployment evaluation fully separated.
- Hold out entire domains, prompts, and model families to reduce leakage.
- Label mixed-provenance text explicitly instead of forcing clean binary labels.
- Do not train on sensitive customer uploads without explicit policy support and consent.
