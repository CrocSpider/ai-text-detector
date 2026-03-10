# System Architecture

## Components

- `apps/web`: Next.js frontend for paste, upload, results, and admin evaluation views.
- `services/api`: FastAPI API for uploads, text extraction, orchestration, and scoring.
- `services/api/app/services/extraction.py`: format-specific text extraction and extraction-quality assessment.
- `services/api/app/services/inference.py`: segment scoring, ensemble aggregation, calibration-style compression, and recommendation generation.
- `services/api/app/services/store.py`: in-memory result storage placeholder that can later be replaced with Postgres and object storage.

## Data flow

1. User submits text or files from the web app.
2. API validates file type and size.
3. Extraction normalizes text and preserves paragraph boundaries.
4. Language detection and extraction-quality checks run before scoring.
5. Paragraphs are chunked into analyzable sections.
6. Segment-level features are scored and aggregated into a document result.
7. Result, confidence, disclaimers, and recommendations are returned to the UI.

## Planned production extensions

- Redis-backed queue for long jobs
- Postgres metadata store and object storage for temporary uploads
- trained transformer service and model registry
- benchmark runner and fairness-slice reporting service
