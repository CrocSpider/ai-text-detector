# AI Text Origin Risk Analyzer

A production-minded scaffold for a cautious AI-text risk analyzer. The app estimates whether text shows signals consistent with machine-generated or heavily machine-edited writing, while surfacing uncertainty, segment-level highlights, and explicit limitations.

## What is included

- `apps/web`: Next.js frontend with paste, upload, batch analysis, result cards, and an admin evaluation page.
- `services/api`: FastAPI backend with safe text extraction, heuristic ensemble scoring, confidence estimation, and explainable result formatting.
- `docs`: product, privacy, and model notes for the initial release.

Additional delivery docs include `docs/system-architecture.md`, `docs/dataset-strategy.md`, `docs/evaluation-plan.md`, `docs/ui-design.md`, `docs/implementation-roadmap.md`, `docs/sample-outputs.md`, and `docs/guardrails.md`.

## Current state

This first implementation is a working scaffold with:

- text paste and file upload support
- extraction for `TXT`, `MD`, `CSV`, `JSON`, `HTML`, `RTF`, `DOCX`, and `PDF`
- paragraph chunking and segment-level scoring
- a cautious placeholder ensemble for transformer/stylometric/surprisal/consistency signals
- calibrated-style confidence and abstention behavior
- an admin-facing evaluation summary placeholder

The current scoring stack is intentionally conservative and heuristic-backed until the trained ensemble is added.

## Local development

### Backend

```bash
python3 -m venv .venv
cd services/api
../../.venv/bin/python -m pip install -e '.[dev]'
../../.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd apps/web
npm install
npm run dev
```

### Tests

```bash
cd services/api
../../.venv/bin/pytest
```

## API endpoints

- `POST /v1/analyze/text`
- `POST /v1/analyze/file`
- `POST /v1/analyze/batch`
- `GET /v1/results/{document_id}`
- `GET /v1/admin/evaluations/summary`
- `GET /health`

## Guardrails

- advisory output only; never proof of authorship
- low-confidence and poor-extraction cases fall back to inconclusive guidance
- warnings clearly state the detector should not be used as sole evidence

## Next implementation targets

1. Replace heuristic classifier placeholder with a fine-tuned encoder.
2. Add persistent storage and async job orchestration.
3. Add benchmark runners, fairness slices, and model version registry.
