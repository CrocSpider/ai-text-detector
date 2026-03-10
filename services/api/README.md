# API Service

FastAPI backend for the AI Text Origin Risk Analyzer.

- extracts text from supported file types
- runs a cautious heuristic ensemble
- returns explainable document and segment results

Optional artifact-backed inference is available by installing `.[ml]` and setting:

```bash
../../.venv/bin/python -m pip install -e '.[ml]'
```

- `API_ENABLE_ARTIFACT_MODELS=true`
- `API_ML_ARTIFACT_DIR=/path/to/artifacts`
- `API_ARTIFACT_DEVICE=auto`

If no trained artifacts are available, the API falls back to the conservative heuristic ensemble.
